"""Checks of the composable constructions of the page (turning torus, flat throat,
Y-junction, handles).  Run: python3 checks_composable.py   (a few minutes)
The flat throat is the page's flat_throat() exactly: smoothstep(4.0, 5.6, |v|), a = 1.2, b = 0.7."""
"""Composable codim-2 worlds (numpy): world = solid + 1j * carrier."""
import sys
import numpy as np
from co2sdf import jac, recip, dot, smoothstep, smootherstep, smin_h, fold_union, union, cutoff

def ball(*xs_r):
    *xs, r = xs_r
    return np.sqrt(sum(x * x for x in xs)) - r

def extrude(p, d, h):          # portal_gravity's extrude: d extruded along the coordinate p, from -h to h
    wx, wy = d, np.abs(p) - h
    return np.minimum(np.maximum(wx, wy), 0.0) + np.hypot(np.maximum(wx, 0), np.maximum(wy, 0))

def world(solid, carrier): return solid + 1j * carrier
def turn(a, b, angle): return np.cos(angle) * a + np.sin(angle) * b
def smooth_union(A, B, k): return fold_union(A, B, k)
def wextrude(A, coord, h): return world(extrude(coord, A.real, h), A.imag)

def capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip(dot(pa, ba) / dot(ba, ba), 0, 1)
    return np.linalg.norm(pa - ba * h[..., None], axis=-1) - r

def project(F, p, iters=40, J=None):
    J = J or (lambda x: jac(F, x))
    for _ in range(iters):
        v = F(p)
        a, b = J(p)
        ra, rb = recip(a, b)
        p = p - v.real[..., None] * ra - v.imag[..., None] * rb
    return p

def sing_min(F, p, J=None):
    a, b = (J or (lambda x: jac(F, x)))(p)
    M = np.stack([a, b], -2)
    return np.linalg.svd(M, compute_uv=False)[..., -1]

def normal_projector(F, p, J=None):
    a, b = (J or (lambda x: jac(F, x)))(p)
    ra, rb = recip(a, b)
    return a[..., :, None] * ra[..., None, :] + b[..., :, None] * rb[..., None, :]   # P w = (w.a) ra + (w.b) rb, as matrix P[i,j] w[i]

def tangent_basis(F, p, J=None):
    """Orthonormal tangent basis (..., n-2, n) at p."""
    a, b = (J or (lambda x: jac(F, x)))(p)
    n = p.shape[-1]
    N = np.stack([a, b], -2)                                    # (..., 2, n)
    # null space via SVD
    u, s, vt = np.linalg.svd(N)
    return vt[..., 2:, :]

def sectional(F, p, X, Y, h=1e-3, J=None):
    """Sectional curvature of the world at p (on it) for orthonormal tangent X, Y (Gauss equation)."""
    def PN(q):
        P = normal_projector(F, q, J)
        return P
    def dP(Z, W):   # (d_Z P_N) W
        Pp, Pm = PN(p + h * Z), PN(p - h * Z)
        return (np.einsum('...ij,...i->...j', Pp, W) - np.einsum('...ij,...i->...j', Pm, W)) / (2 * h)
    IIxx, IIyy = dP(X, X), dP(Y, Y)
    IIxy = 0.5 * (dP(X, Y) + dP(Y, X))
    return dot(IIxx, IIyy) - dot(IIxy, IIxy)


# ======================================================================
# t_torus.py
rng = np.random.default_rng(0)
a, b = 1.0, 0.5
for deg in [0, 30, 60, 89, 90]:
    th = np.radians(deg)
    F = lambda p: world(ball(ball(p[..., 0], p[..., 1], a), p[..., 2], p[..., 3], b), turn(p[..., 3], ball(p[..., 0], p[..., 1], a), th))
    p0 = rng.uniform(-2, 2, (20000, 4))
    p = project(F, p0)
    ok = np.abs(F(p)) < 1e-10
    q = p[ok]
    # expected: r = a + b cos(th) cos(psi), z = b sin(psi), w = -b cos(psi) sin(th)
    r = np.hypot(q[:, 0], q[:, 1]); z, w = q[:, 2], q[:, 3]
    cpsi = (r - a) * np.cos(th) - w * np.sin(th)        # in-plane coordinate e = cos th rhat - sin th what
    psi = np.arctan2(z, cpsi)
    resid = np.abs(np.hypot(cpsi, z) - b).max(), np.abs(np.sin(th) * (r - a) + np.cos(th) * w).max()
    sv = sing_min(F, q[:2000])
    # curvature at 300 points vs formula
    m = 300
    T = tangent_basis(F, q[:m])
    K = sectional(F, q[:m], T[:, 0], T[:, 1])
    Kf = np.cos(th) * np.cos(psi[:m]) / (b * (a + b * np.cos(th) * np.cos(psi[:m])))
    print(f"angle {deg:3d}: converged {ok.mean():.3f}  on-torus residual {max(resid):.1e}  min sing.value {sv.min():.4f}  "
          f"|K - formula| max {np.abs(K - Kf).max():.1e}  K range [{K.min():+.3f}, {K.max():+.3f}]")


# ======================================================================
# t_throat.py
rng = np.random.default_rng(1)
a, b, h, k = 1.2, 0.7, 7.0, 0.4
def torus(p4, angle):   # p4 = (..., 4): one 4D donut solid; carrier turned from w (donut) to ball(p.xy, a) (flat torus)
    x, y, z, w = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    return world(ball(ball(x, y, a), z, w, b), turn(w, ball(x, y, a), angle))
def flat_throat(p):
    v = p[..., 4]
    flat = 1 - smoothstep(h - 3.0, h - 1.4, np.abs(v))       # 1 in the middle, 0 near the worlds
    throat = wextrude(torus(p[..., :4], np.pi / 2 * flat), v, h)
    worlds = world(h - np.abs(v), p[..., 3])
    return smooth_union(worlds, throat, k)
F = flat_throat
lo = np.array([-3.5, -3.5, -2.5, -1.5, -8.0]); hi = -lo
p0 = rng.uniform(lo, hi, (60000, 5))
p = project(F, p0, iters=60)
ok = (np.abs(F(p)) < 1e-10) & np.all((p > lo) & (p < hi), axis=1)
q = p[ok]
x, y, z, w, v = q.T
r = np.hypot(x, y)
flat = 1 - smoothstep(h - 3.0, h - 1.4, np.abs(v)); th = np.pi / 2 * flat
on_torus = (np.abs(np.sqrt((r - a) ** 2 + z ** 2 + w ** 2) - b) < 1e-6) & (np.abs(turn(w, r - a, th)) < 1e-6) & (np.abs(v) < h)
on_world = (np.abs(w) < 1e-6) & (np.abs(v) > h - k - 0.05)
print("converged in box:", ok.sum(), " on throat torus:", on_torus.sum(), " on worlds / mouth rounding:", on_world.sum(),
      " unexplained:", (~on_torus & ~on_world).sum())
print("min singular value of the Jacobian: %.4f" % sing_min(F, q).min())
mid = on_torus & (np.abs(v) < h - 3.2)
qm = q[mid][:400]
T = tangent_basis(F, qm)
Kmax = 0
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    Kmax = max(Kmax, np.abs(sectional(F, qm, T[:, i], T[:, j])).max())
c = rng.normal(size=(len(qm), 3)); c /= np.linalg.norm(c, axis=1, keepdims=True)
X = np.einsum('ni,nij->nj', c, T); d = rng.normal(size=(len(qm), 3)); Y = np.einsum('ni,nij->nj', d, T)
Y -= dot(Y, X)[:, None] * X; Y /= np.linalg.norm(Y, axis=1, keepdims=True)
Kmax = max(Kmax, np.abs(sectional(F, qm, X, Y)).max())
print("middle of the throat (|v| < %.1f): %d points, max |sectional curvature| = %.1e, r - a max %.1e, |zw| - b max %.1e"
      % (h - 3.2, len(qm), Kmax, np.abs(np.hypot(qm[:, 0], qm[:, 1]) - a).max(), np.abs(np.hypot(qm[:, 2], qm[:, 3]) - b).max()))
tr = on_torus & (np.abs(v) > h - 3.0) & (np.abs(v) < h - 1.4)
qt = q[tr][:400]; T = tangent_basis(F, qt)
Ks = np.concatenate([sectional(F, qt, T[:, i], T[:, j]) for (i, j) in [(0, 1), (0, 2), (1, 2)]])
print("turning zone: %d points, sectional curvature range [%.3f, %.3f]" % (len(qt), Ks.min(), Ks.max()))
dz = on_torus & (np.abs(v) > h - 1.4) & (np.abs(v) < h - k - 0.1)
qd = q[dz][:400]; T = tangent_basis(F, qd)
Kd = np.concatenate([sectional(F, qd, T[:, i], T[:, j]) for (i, j) in [(0, 1), (0, 2), (1, 2)]])
print("donut stretch between the turn and the mouth: %d points, sectional curvature range [%.3f, %.3f] (a donut is curved)" % (len(qd), Kd.min(), Kd.max()))
wd = on_world & (np.hypot(np.hypot(x, y) - a, z) > b + 1.0)
qw = q[wd][:400]; T = tangent_basis(F, qw)
print("worlds away from the mouth: %d points, max |K| = %.1e" % (len(qw), max(np.abs(sectional(F, qw, T[:, i], T[:, j])).max() for (i, j) in [(0, 1), (0, 2), (1, 2)])))


# ======================================================================
# t_yjunction.py
rng = np.random.default_rng(2)
def make(dim, rho, r, kj, km, m=0.5, half_space=False):
    """dim = 4: 2D worlds (x, y | n = (z, w)), carrier at the centre: y.  dim = 5: 3D worlds (x, y, z | n = (w, v)), carrier at the centre: z."""
    ds = [np.array([np.cos(t), np.sin(t)]) for t in np.pi / 2 + 2 * np.pi * np.arange(3) / 3]
    ic = dim - 3                                    # index of the centre carrier coordinate (y in 4D, z in 5D)
    def arm(p, d):
        n = p[..., dim - 2:]
        along, side = n @ d, n @ np.array([-d[1], d[0]])
        end = np.zeros(dim); end[dim - 2:] = rho * d
        tube = capsule(p, np.zeros(dim), end, r)
        angle = np.pi / 2 * smootherstep(0.3, 0.7, along / rho)
        return world(tube, turn(p[..., ic], side, angle))
    def space(p, d):
        n = p[..., dim - 2:]
        along, side = n @ d, n @ np.array([-d[1], d[0]])
        return world(rho - along if half_space else extrude(side, rho - along, m), side)
    def F(p):
        f = smooth_union(smooth_union(arm(p, ds[0]), arm(p, ds[1]), kj), arm(p, ds[2]), kj)
        for d in ds:
            f = smooth_union(f, space(p, d), km)
        return f
    return F, ds
for dim, rho, r, kj, km in [(4, 3.0, 0.7, 0.6, 0.4), (5, 2.4, 0.8, 0.6, 0.45)]:
    F, ds = make(dim, rho, r, kj, km)
    lo = np.full(dim, -4.0); lo[dim - 2:] = -(rho + 1.5); hi = -lo
    p0 = rng.uniform(lo, hi, (60000, dim))
    p = project(F, p0, iters=60)
    ok = (np.abs(F(p)) < 1e-10) & np.all((p > lo) & (p < hi), axis=1)
    q = p[ok]
    n = q[:, dim - 2:]; x = q[:, :dim - 2]
    # expected: on a world (n = rho d_k), or within r (+ blend) of an arm axis
    dw = np.min([np.linalg.norm(n - rho * d, axis=1) for d in ds], axis=0)
    def seg_dist(d):
        t = np.clip(n @ d / rho, 0, 1)
        return np.sqrt((x ** 2).sum(1) + ((n - rho * t[:, None] * d) ** 2).sum(1))
    da = np.min([seg_dist(d) for d in ds], axis=0)
    on_world = dw < 1e-6
    near_arm = da < r + 0.05
    blend = (~on_world) & (~near_arm)
    # points in the blends must be near a mouth (|x| small, n near a world) or near the junction
    mouthish = blend & (dw < 0.8) & (np.linalg.norm(x, axis=1) < r + 1.2)
    junction = blend & (np.linalg.norm(n, axis=1) < 1.5) & (np.linalg.norm(x, axis=1) < r + 1.0)
    print(f"{dim - 2}D worlds in {dim}D: converged {ok.sum()}, on worlds {on_world.sum()}, on arms {near_arm.sum()}, "
          f"in blends near mouths {mouthish.sum()}, near the junction {junction.sum()}, unexplained {(blend & ~mouthish & ~junction).sum()}")
    sv = sing_min(F, q)
    print(f"   min singular value {sv.min():.3f}  (1st percentile {np.percentile(sv, 1):.3f})")


# ======================================================================
# t_handle.py
import render                       # the reference second-order step
from co2sdf import normalize, tangent, jac
R, r, k = 1.5, 0.5, 0.4
def handle(p, twist):
    x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    along = np.arctan2(np.maximum(z, 0.0), x)                       # 0 at the mouth x = +R, pi at x = -R
    solid = ball(ball(x, z, R), y, w, r)                            # a 4D donut around the circle |xz| = R
    angle = twist * smootherstep(0.25 * np.pi, 0.75 * np.pi, along)
    return world(solid, turn(w, y, angle))
def plane_with_handle(twist):
    def F(p):
        x, z, w = p[..., 0], p[..., 2], p[..., 3]
        plane = world(z, np.where((twist > 0) & (x < 0), -w, w))
        return smooth_union(plane, handle(p, twist), k)
    return F
for name, twist in [("handle", 0.0), ("mirror", np.pi)]:
    F = plane_with_handle(twist)
    n = 400
    cam = project(F, np.array([[R, -1.6, 0.0, 0.0]]), 8)            # in front of the mouth x = +R
    a, b = jac(F, cam)
    e1 = normalize(tangent(a, b, np.array([[0.0, 1.0, 0, 0]])))
    t2 = tangent(a, b, np.array([[1.0, 0, 0, 0]])); e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    th = np.linspace(-0.25, 0.25, n)
    p = np.repeat(cam, n, 0)
    v = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    wv = -np.sin(th)[:, None] * e1 + np.cos(th)[:, None] * e2
    orient = lambda v, w: np.sign(v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0])
    o0 = orient(v, wv); maxz = np.zeros(n)
    for i in range(1500):
        p, v = render.step(F, p, v, 0.01)
        a, b = jac(F, p)
        wv = tangent(a, b, wv); wv = wv - dot(wv, v)[..., None] * v; wv = normalize(wv)
        maxz = np.maximum(maxz, p[:, 2])
    back = (np.abs(p[:, 2]) < 1e-6) & (np.abs(p[:, 3]) < 1e-6) & (maxz > R * 0.8) & (p[:, 0] < 0)
    print("%-7s rays through the handle and back onto the plane: %d / %d; orientation kept (+1) or mirrored (-1): %s"
          % (name, back.sum(), n, np.unique((o0 * orient(v, wv))[back])))
    # spurious points / conditioning
    rng = np.random.default_rng(3)
    q = project(F, rng.uniform([-4, -4, -2.5, -1.5], [4, 4, 3.0, 1.5], (30000, 4)), 50)
    ok = np.abs(F(q)) < 1e-10; q = q[ok]
    x, y, z, w = q.T
    on_plane = (np.abs(z) < 1e-9) & (np.abs(w) < 1e-9)
    on_tube = np.abs(ball(ball(x, z, R), y, w, r)) < 1e-6
    rest = ~on_plane & ~on_tube
    near_mouth = rest & (np.minimum(np.hypot(x - R, y), np.hypot(x + R, y)) < r + k + 0.3) & (np.abs(z) < k + 0.3)
    print("        converged %d: plane %d, tube %d, mouth blends %d, unexplained %d; min singular value %.3f"
          % (ok.sum(), on_plane.sum(), on_tube.sum(), near_mouth.sum(), (rest & ~near_mouth).sum(), np.nanmin(sing_min(F, q))))

# ======================================================================
# Y-junction: no stray worlds far out in the extra plane; the plain half-space version has one
def far_strays(F, ds, dim, rho, seed=5):
    rng = np.random.default_rng(seed)
    lo = np.full(dim, -4.0); lo[dim - 2:] = -3 * rho; hi = -lo
    p = project(F, rng.uniform(lo, hi, (60000, dim)), iters=60)
    ok = (np.abs(F(p)) < 1e-10) & np.all((p > lo) & (p < hi), axis=1)
    n = p[ok][:, dim - 2:]
    dw = np.min([np.linalg.norm(n - rho * d, axis=1) for d in ds], axis=0)
    return ((dw > 1e-6) & (np.linalg.norm(n, axis=1) > rho + 1.0)).sum()

for dim, rho, r, kj, km in [(4, 3.0, 0.7, 0.6, 0.4), (5, 2.4, 0.8, 0.6, 0.45)]:
    F, ds = make(dim, rho, r, kj, km)
    print(f"Y-junction {dim}D, points off the worlds beyond |n| > rho + 1 (out to 3 rho): {far_strays(F, ds, dim, rho)}")
F, ds = make(4, 3.0, 0.7, 0.6, 0.4, half_space=True)
print(f"Y-junction 4D with plain half-space world solids: stray points far out: {far_strays(F, ds, 4, 3.0)} (the ghost world)")
