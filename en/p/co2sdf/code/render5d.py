"""Generic perspective renderer for 3D worlds in R^5 given a scene with
F_J(p) -> (F complex, a = grad Re F, b = grad Im F), world_of(p), floor, spheres."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render3d import recip, tangent, normalize, WORLD


def num_FJ(F, p, h=1e-5):
    a = np.empty(p.shape); b = np.empty(p.shape)
    for k in range(p.shape[-1]):
        e = np.zeros(p.shape[-1]); e[k] = h
        d = (F(p + e) - F(p - e)) / (2 * h)
        a[..., k] = d.real; b[..., k] = d.imag
    return F(p), a, b


def project(sc, p, iters=2):
    for _ in range(iters):
        F, a, b = sc.F_J(p)
        ra, rb = recip(a, b)
        p = p - F.real[..., None] * ra - F.imag[..., None] * rb
    return p, a, b


def render(sc, res=(480, 320), fov=75, dt=0.02, max_len=20.0, sky_by_world=True, dt_fn=None):
    W_, H_ = res
    cam = np.asarray(sc.cam, float)[None]
    cam, a, b = project(sc, cam, 6)
    def tan_of(h):
        return tangent(a, b, np.asarray(h, float)[None])
    fwd = normalize(tan_of(sc.look))
    up = tan_of(sc.up); up = normalize(up - (up * fwd).sum(-1)[..., None] * fwd)
    right = tan_of(sc.right)
    right = normalize(right - (right * fwd).sum(-1)[..., None] * fwd - (right * up).sum(-1)[..., None] * up)
    t = np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.linspace(-t, t, W_), np.linspace(t * H_ / W_, -t * H_ / W_, H_))
    d = normalize(fwd + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up)
    n = d.shape[0]
    p = np.repeat(cam, n, 0)
    col = np.zeros((n, 3)); alive = np.ones(n, bool); dist = np.zeros(n)
    while alive.any():
        idx = np.nonzero(alive)[0]
        h = dt if dt_fn is None else dt_fn(p[idx])
        hh = h if np.ndim(h) == 0 else h[:, None]
        q = p[idx] + d[idx] * hh
        q, a, b = project(sc, q)
        dn = normalize(tangent(a, b, d[idx]))
        dist[idx] += h
        hit, c = sc.hits(p[idx], q, dn, dist[idx])
        col[idx[hit]] = c[hit]
        alive[idx[hit]] = False
        p[idx] = q; d[idx] = dn
        out = alive & (dist > max_len)
        if out.any():
            k = np.nonzero(out)[0]
            col[k] = sc.sky(p[k], d[k])
            alive[k] = False
    return col.reshape(H_, W_, 3)


def floor_color(base, x, dist):
    ch = ((np.floor(x[:, 0]) + np.floor(x[:, 2])) % 2)[:, None]
    c = base * (0.62 + 0.38 * ch)
    fog = np.exp(-0.03 * dist)[:, None]
    return c * fog + (1 - fog) * 0.85


def sphere_hits(spheres, world_of, p, q, dn, dist):
    hit = np.zeros(len(q), bool); col = np.zeros((len(q), 3))
    w = world_of(q)
    for (k, c0, r0, cc) in spheres:
        qq = q[:, :len(c0)]
        m = (np.linalg.norm(qq - c0, axis=-1) < r0) & (w == k) & ~hit
        if m.any():
            nrm = normalize(qq[m] - c0)[:, :3]
            lam = 0.35 + 0.65 * np.clip(nrm @ normalize(np.array([0.4, 1.0, -0.5])), 0, 1)
            col[m] = cc * lam[:, None]; hit |= m
    return hit, col


def save(img, name):
    plt.imsave(name, np.clip(img, 0, 1))
