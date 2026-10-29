"""High-quality renders: second-order (chord) geodesic steps, ~10k steps per ray,
jittered supersampling, work split over processes in strips.

usage: python3 render_hq.py scan <scene> [res spp n_dir n_steps]   (CURV=1 for the curvature view)
       python3 render_hq.py view <name>  [W H spp n_steps]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time
import numpy as np
import multiprocessing as mp
from PIL import Image

WORKERS = int(os.environ.get("WORKERS", "32"))

# ============================================================== 2D scans

def _scan_setup(name):
    from scenes import SCENES
    from co2sdf import jac, recip, tangent, normalize, dot
    sc = SCENES[name]
    J = getattr(sc, "jac", None) or (lambda p: jac(sc.F, p))
    cam = sc.cam[None, :].astype(float)
    for _ in range(10):
        val = sc.F(cam); a, b = J(cam); ra, rb = recip(a, b)
        cam = cam - val.real[..., None] * ra - val.imag[..., None] * rb
    a, b = J(cam)
    e1 = normalize(tangent(a, b, sc.hint1[None, :]))
    t2 = tangent(a, b, sc.hint2[None, :])
    e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    return sc, J, cam, e1, e2


# curvature view (as in portal_gravity's curved_space_2d): Gaussian curvature by the Gauss
# equation, from how the projector onto the normal plane turns along the world
K_NEG, K_ZERO, K_POS = np.array([96, 166, 225]) / 255, np.array([218, 218, 218]) / 255, np.array([251, 158, 35]) / 255


def gauss_curvature(J, p, h=1e-3):
    from co2sdf import recip, dot
    a, b = J(p)
    t = np.linalg.svd(np.stack([a, b], -2))[2][:, 2:]            # orthonormal tangent basis (n, 2, 4)
    t1, t2 = t[:, 0], t[:, 1]
    def normal_part(q, w):
        a, b = J(q); ra, rb = recip(a, b)
        return lambda v: dot(v, a)[:, None] * ra + dot(v, b)[:, None] * rb
    P1p, P1m, P2p, P2m = (normal_part(p + s * h * tt, None) for s, tt in [(1, t1), (-1, t1), (1, t2), (-1, t2)])
    s11 = P1p(t1) - P1m(t1)
    s22 = P2p(t2) - P2m(t2)
    s12 = 0.5 * (P1p(t2) - P1m(t2) + P2p(t1) - P2m(t1))
    return (dot(s11, s22) - dot(s12, s12)) / (4 * h * h)


def curvature_color(K, base, scale):
    """portal_gravity's colours for curvature K, over a faint copy of the scene's pattern."""
    t = np.clip(np.abs(np.nan_to_num(K) * scale), 0, 1)[:, None]
    tint = K_ZERO + (np.where((K < 0)[:, None], K_NEG, K_POS) - K_ZERO) * t
    lum = np.clip((base @ np.array([0.3, 0.59, 0.11])) / 0.6, 0, 1)[:, None]
    return tint * (0.75 + 0.25 * lum)


def creature_coords(pos, cam, e1, e2):
    """Coordinates of world points in the camera's frame, for points near the camera (by 4D distance;
    scenes whose world comes back close to itself in 4D give their own, intrinsic, version)."""
    d = pos - cam
    a1, a2 = d @ e1, d @ e2
    off = np.linalg.norm(d - a1[:, None] * e1 - a2[:, None] * e2, axis=1)
    return a1, a2, off < 0.2


def paint_creature(col, a1, a2, ok, size=0.36):
    """A chiral creature at the camera, so that its copies show whether they are mirrored:
    a red disc with a white nose forward (e2, up in the scan) and a white ear on its left (-e1)."""
    body = ok & (np.hypot(a1, a2) < size)
    nose = body & (np.hypot(a1, a2 - 0.62 * size) < 0.28 * size)
    ear = body & (np.hypot(a1 + 0.6 * size, a2 + 0.06 * size) < 0.22 * size)
    col = col.copy()
    col[body] = np.array([0.9, 0.15, 0.15])
    col[nose | ear] = 1.0
    return col


def _scan_worker(args):
    name, j0, j1, n_dir, n_steps, r_max, th_s, r_s, ids, curv = args
    from co2sdf import recip, tangent, normalize
    sc, J, cam, e1, e2 = _scan_setup(name)
    J = getattr(sc, "jac", None) or J
    dt = r_max / n_steps
    th = 2 * np.pi * np.arange(j0, j1) / n_dir
    p = np.repeat(cam, j1 - j0, 0)
    v = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    P = np.empty((n_steps + 1, j1 - j0, 4))
    P[0] = p
    a, b = J(p)
    for i in range(n_steps):
        q = p + v * dt
        val = sc.F(q); ra, rb = recip(a, b)             # along the normals at the start
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
        a, b = J(q); val = sc.F(q); ra, rb = recip(a, b)  # with fresh normals
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
        a, b = J(q)                                     # normals at the new point, reused next step
        v = normalize(tangent(a, b, q - p))             # chord update: second order
        p = q
        P[i + 1] = p
    # colour the samples that belong to these rays
    k = np.round(th_s / (2 * np.pi) * n_dir).astype(int) % n_dir - j0
    x = np.clip(r_s / dt, 0, n_steps)
    i0 = np.minimum(np.floor(x).astype(int), n_steps - 1)
    f = (x - i0)[:, None]
    pos = P[i0, k] * (1 - f) + P[i0 + 1, k] * f
    col = sc.color(pos)
    if getattr(sc, "creature", False):
        coords = getattr(sc, "creature_coords", None) or creature_coords
        col = paint_creature(col, *coords(pos, cam[0], e1[0], e2[0]))
    if curv:
        col = curvature_color(gauss_curvature(J, pos), col, curv)
    resid = np.abs(sc.F(P.reshape(-1, 4))).max()
    return ids, col, resid


def scan(name, res=640, spp=10, n_dir=8192, n_steps=10000, out=None, curv=0.0):
    from scenes import SCENES
    sc = SCENES[name]
    r_max = getattr(sc, "r_max", 6.0)
    rng = np.random.default_rng(7)
    py, px = np.mgrid[0:res, 0:res]
    px = np.repeat(px.reshape(-1), spp) + rng.random(res * res * spp)
    py = np.repeat(py.reshape(-1), spp) + rng.random(res * res * spp)
    x = px / res * 2 - 1
    y = py / res * 2 - 1
    r = np.hypot(x, y) * r_max
    th = np.mod(np.arctan2(-y, x), 2 * np.pi)
    inside = r <= r_max
    ray = np.round(th / (2 * np.pi) * n_dir).astype(int) % n_dir
    bounds = np.linspace(0, n_dir, WORKERS + 1).astype(int)
    jobs = []
    for w in range(WORKERS):
        m = inside & (ray >= bounds[w]) & (ray < bounds[w + 1])
        ids = np.nonzero(m)[0]
        jobs.append((name, bounds[w], bounds[w + 1], n_dir, n_steps, r_max, th[ids], r[ids], ids, curv))
    col = np.zeros((res * res * spp, 3))
    t0 = time.time()
    with mp.Pool(WORKERS) as pool:
        resid = 0.0
        for ids, c, rs in pool.imap_unordered(_scan_worker, jobs):
            col[ids] = c
            resid = max(resid, rs)
    if not getattr(sc, "creature", False):       # camera marker (a creature marks it otherwise)
        col[r < 0.02 * r_max] = 0.05
    rgb = col.reshape(res * res, spp, 3)
    a = inside.reshape(res * res, spp).astype(float)
    alpha = a.mean(1)
    rgb = (rgb * a[..., None]).sum(1) / np.maximum(a.sum(1), 1)[:, None]
    img = np.concatenate([rgb, alpha[:, None]], 1).reshape(res, res, 4)
    out = out or ("hq_scan_%s%s.png" % (name, "_curv" if curv else ""))
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(out)
    print("%s: %.0f s, max |F| on rays %.1e -> %s" % (name, time.time() - t0, resid, out), flush=True)

# ============================================================== 3D views

def make_view(name):
    import scenes5d as S5
    import render6d as R6
    if name == "gate_N2":
        sc = S5.Stargate(N=2)
    elif name == "gate_N2_bare":
        sc = S5.Stargate(N=2, frame=False)
    elif name == "gate_N3":
        sc = S5.Stargate(N=3)
    elif name == "gate_N3_back":
        sc = S5.Stargate(N=3)
        sc.cam = sc.on_world(np.array([[0.0, -0.4, 5.0]]), 0)[0]
        sc.look, sc.right = [0, 0, -1, 0, 0], [-1, 0, 0, 0, 0]
    elif name == "tubes3":
        sc = S5.TubeWorlds()
    elif name == "throat2":
        sc = S5.FlatThroat(a=1.2, b=0.7, L=7.0, L1=4.0, L2=5.6, k=0.4)
        sc.cam = sc.cl(0.0, 0.0, 0.0)
        sc.look, sc.up, sc.right = [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
        cols = [np.array([0.9, 0.2, 0.2]), np.array([0.2, 0.4, 0.95]), np.array([0.95, 0.85, 0.2])]
        sc.spheres = [(2, sc.cl(t, f, s), 0.16, cols[i]) for i, (t, f, s) in enumerate([(1.6, 0.0, 0.0), (1.6, 0.0, 0.7), (1.6, 0.0, -0.7)])]
        sc.spheres += [(2, sc.cl(1.6, 0.9, 0.0), 0.12, np.array([0.9, 0.9, 0.9]))]
    elif name == "neck6d":
        sc = R6.InversionNeck()
    else:
        raise KeyError(name)
    cfg = dict(gate_N2=(75, 18.0), gate_N2_bare=(75, 18.0), gate_N3=(75, 18.0), gate_N3_back=(75, 18.0),
               tubes3=(75, 16.0), throat2=(90, 22.0), neck6d=(75, 16.0))[name]
    return sc, cfg


class Jet2:
    """codim-2 helpers on top of scene.F_J"""
    def __init__(self, sc):
        self.sc = sc

    def project(self, p, iters=2):
        from render3d import recip
        for _ in range(iters):
            F, a, b = self.sc.F_J(p)
            ra, rb = recip(a, b)
            p = p - F.real[..., None] * ra - F.imag[..., None] * rb
        return p, np.stack([a, b], 1)

    def tangent(self, J, v):
        from render3d import tangent
        return tangent(J[:, 0], J[:, 1], v)

    def jacobian(self, p):
        F, a, b = self.sc.F_J(p)
        return np.stack([a, b], 1)

    def correct(self, q, J):
        from render3d import recip
        F, _, _ = self.sc.F_J(q)
        ra, rb = recip(J[:, 0], J[:, 1])
        return q - F.real[..., None] * ra - F.imag[..., None] * rb


class JetK:
    """codim-k helpers for scenes exposing F(p) -> (..., k)"""
    def __init__(self, sc):
        self.sc = sc

    def project(self, p, iters=2):
        import render6d as R6
        for _ in range(iters):
            f, J = R6.num_jet(self.sc.F_locked(p) if hasattr(self.sc, "F_locked") else self.sc.F, p)
            p = p - R6.gram_apply(J, f)
        return p, J

    def tangent(self, J, v):
        import render6d as R6
        return R6.tangent(J, v)

    def jacobian(self, p):
        import render6d as R6
        return R6.num_jet(self.sc.F_locked(p), p)[1] if hasattr(self.sc, "F_locked") else R6.num_jet(self.sc.F, p)[1]

    def correct(self, q, J):
        import render6d as R6
        return q - R6.gram_apply(J, self.sc.F(q))


def _jet_for(sc):
    return Jet2(sc) if hasattr(sc, "F_J") else JetK(sc)


def _camera(name):
    from render3d import normalize
    sc, (fov, max_len) = make_view(name)
    jet = _jet_for(sc)
    cam, J = jet.project(np.asarray(sc.cam, float)[None], 8)
    t = lambda h: jet.tangent(J, np.asarray(h, float)[None])
    fwd = normalize(t(sc.look))
    up = t(sc.up); up = normalize(up - (up * fwd).sum(-1)[..., None] * fwd)
    rt = t(sc.right)
    rt = normalize(rt - (rt * fwd).sum(-1)[..., None] * fwd - (rt * up).sum(-1)[..., None] * up)
    return sc, jet, cam, fwd, up, rt, fov, max_len


def _hits(sc, p, q, dn, dist):
    if hasattr(sc, "hits"):
        return sc.hits(p, q, dn, dist)
    # codim-3 neck scene (render6d): spheres + floor in each world's own coordinates
    from render5d import floor_color
    from render3d import WORLD, normalize
    x0, _ = sc.coords(p); x1, w = sc.coords(q)
    hit = np.zeros(len(q), bool); col = np.zeros((len(q), 3))
    for (k, c0, r0, cc) in sc.spheres:
        m = (w == k) & (np.linalg.norm(x1 - c0, axis=-1) < r0) & ~hit
        if m.any():
            nrm = normalize(x1[m] - c0)
            lam = 0.35 + 0.65 * np.clip(nrm @ normalize(np.array([0.4, 1.0, -0.5])), 0, 1)
            col[m] = cc * lam[:, None]; hit |= m
    fl = (x0[:, 1] > sc.floor) & (x1[:, 1] <= sc.floor) & ~hit
    if fl.any():
        col[fl] = floor_color(np.stack([WORLD[k] for k in w[fl]]), x1[fl], dist[fl]); hit |= fl
    return hit, col


def _sky(sc, p, d):
    if hasattr(sc, "sky"):
        return sc.sky(p, d)
    from render3d import WORLD
    _, w = sc.coords(p)
    base = np.stack([WORLD[k] for k in w])
    return (0.55 + 0.45 * base) * 0.85


def _view_worker(args):
    name, U, V, n_steps = args
    from render3d import normalize
    sc, jet, cam, fwd, up, rt, fov, max_len = _camera(name)
    d = normalize(fwd + U[:, None] * rt + V[:, None] * up)
    n = len(d)
    p = np.repeat(cam, n, 0)
    Js = np.repeat(jet.jacobian(cam), n, 0)
    col = np.zeros((n, 3)); alive = np.ones(n, bool); dist = np.zeros(n)
    dt0 = max_len / n_steps
    dt_fn = getattr(sc, "dt_fn", None)
    guard = 0
    while alive.any() and guard < 6 * n_steps:
        guard += 1
        idx = np.nonzero(alive)[0]
        h = np.full(len(idx), dt0)
        if dt_fn is not None:   # finer near features, never coarser than dt0
            h = np.minimum(h, dt_fn(p[idx]) * dt0 / 0.03)
        q = p[idx] + d[idx] * h[:, None]
        q = jet.correct(q, Js[idx])                     # along the normals at the start
        q, _ = jet.project(q, 1)                        # with fresh normals
        J = jet.jacobian(q)                             # normals at the new point
        dn = normalize(jet.tangent(J, q - p[idx]))
        Js[idx] = J
        dist[idx] += h
        hit, c = _hits(sc, p[idx], q, dn, dist[idx])
        col[idx[hit]] = c[hit]
        alive[idx[hit]] = False
        p[idx] = q; d[idx] = dn
        out = alive & (dist > max_len)
        if out.any():
            k = np.nonzero(out)[0]
            col[k] = _sky(sc, p[k], d[k])
            alive[k] = False
    if alive.any():
        k = np.nonzero(alive)[0]
        col[k] = _sky(sc, p[k], d[k])
    return col


def view(name, W=600, H=400, spp=10, n_steps=10000, out=None):
    _, (fov, _) = make_view(name)
    t = np.tan(np.radians(fov) / 2)
    rng = np.random.default_rng(11)
    py, px = np.mgrid[0:H, 0:W]
    px = np.repeat(px.reshape(-1), spp) + rng.random(W * H * spp)
    py = np.repeat(py.reshape(-1), spp) + rng.random(W * H * spp)
    U = (px / W * 2 - 1) * t
    V = -(py / H * 2 - 1) * t * H / W
    # strips: interleave samples so every worker gets a similar mix of cheap and expensive rays
    order = rng.permutation(len(U))
    chunks = np.array_split(order, WORKERS * 4)
    t0 = time.time()
    col = np.zeros((len(U), 3))
    with mp.Pool(WORKERS) as pool:
        for ids, c in zip(chunks, pool.imap(_view_worker, [(name, U[c], V[c], n_steps) for c in chunks])):
            col[ids] = c
    img = col.reshape(H * W, spp, 3).mean(1).reshape(H, W, 3)
    out = out or "hq_%s.png" % name
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(out)
    print("%s: %.0f s -> %s" % (name, time.time() - t0, out), flush=True)


if __name__ == "__main__":
    kind, name = sys.argv[1], sys.argv[2]
    nums = [int(x) for x in sys.argv[3:]]
    if kind == "scan":                       # CURV=<scale> colours the scan by curvature
        scan(name, *nums, curv=float(os.environ.get("CURV", "0")))
    else:
        view(name, *nums)
