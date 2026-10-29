import numpy as np
from render3d import PortalScene, WORLD, normalize
from render5d import num_FJ, floor_color, sphere_hits


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


class Stargate(PortalScene):
    def __init__(self, N=2, rho=0.7, R=1.6, floor=-2.3, frame=True):
        super().__init__(N=N, rho=rho, R=R, floor=floor)
        self.frame = frame
        self.cam = self.on_world(np.array([[0.0, -0.4, -5.0]]), 0)[0]
        self.look, self.up, self.right = [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]
        self.spheres = [(0, np.array([2.6, -1.6, 2.0]), 0.7, np.array([0.9, 0.2, 0.2])),
                        (1, np.array([-0.5, -1.5, 4.0]), 0.8, np.array([0.95, 0.9, 0.2])),
                        (2, np.array([0.7, -1.5, 3.5]), 0.8, np.array([0.9, 0.3, 0.9]))]

    def world_of(self, p):
        return self.sheet(p)

    def hits(self, p, q, dn, dist):
        hit, col = sphere_hits(self.spheres, self.world_of, p, q, dn, dist)
        f = (p[:, 1] > self.floor) & (q[:, 1] <= self.floor) & ~hit
        if f.any():
            w = self.world_of(q[f])
            col[f] = floor_color(np.stack([WORLD[k] for k in w]), q[f, :3], dist[f]); hit |= f
        if self.frame:
            fr = (self.ring_dist(q[:, :3]) < 0.05) & ~hit
            col[fr] = np.array([0.85, 0.7, 0.25]); hit |= fr
        return hit, col

    def sky(self, p, d):
        w = self.world_of(p)
        base = np.stack([WORLD[k] for k in w])
        hgt = np.clip(d[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
        return (0.55 + 0.45 * base) * (0.6 + 0.4 * hgt)

    def dt_fn(self, p):  # finer steps near the ring core
        return np.clip(0.25 * self.ring_dist(p[:, :3]), 0.004, 0.03)


class TubeWorlds:
    """Three parallel 3D worlds R^3 x {c_k} in R^5, every pair joined by an
    S^2 x segment tube lying in the 4-space spanned by R^3 and the segment."""
    def __init__(self, L=1.6, r=0.75, k=0.45, floor=-1.6):
        self.L, self.r, self.k, self.floor = L, r, k, floor
        rho = L / np.sqrt(3)
        ang = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
        self.c = [rho * np.array([np.cos(a), np.sin(a)]) for a in ang]
        self.q = {(0, 1): np.array([-1.6, 0.0, 0.5]), (2, 0): np.array([1.6, 0.0, 0.5]),
                  (1, 2): np.array([0.0, 0.0, 4.5])}
        self.pairs = [(0, 1), (1, 2), (2, 0)]
        self.cam = np.array([0.0, -0.2, -3.5, *self.c[0]])
        self.look, self.up, self.right = [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]
        self.spheres = [(1, np.array([-0.8, -1.0, 1.5]), 0.6, np.array([0.95, 0.9, 0.2])),
                        (2, np.array([1.2, -1.0, 2.0]), 0.6, np.array([0.9, 0.3, 0.9])),
                        (0, np.array([0.0, -1.1, 3.0]), 0.5, np.array([0.9, 0.2, 0.2]))]

    def pair(self, p, i, j):
        ci, cj = self.c[i], self.c[j]
        e = (cj - ci) / self.L; n = np.array([-e[1], e[0]])
        zx, zy = p[..., 3] - ci[0], p[..., 4] - ci[1]
        t = zx * e[0] + zy * e[1]; s = zx * n[0] + zy * n[1]
        x = p[..., :3] - self.q[(i, j)]
        cap = np.sqrt((x * x).sum(-1) + s * s) - self.r
        return smin(np.minimum(t, self.L - t), cap, self.k) + 1j * s

    def F_J(self, p):
        # hard union: differentiate only the winning piece of each ray
        vals = [self.pair(p, i, j) for (i, j) in self.pairs]
        win = np.argmin(np.stack([v.real for v in vals]), 0)
        F = np.choose(win, vals)
        a = np.zeros(p.shape); b = np.zeros(p.shape)
        for m, (i, j) in enumerate(self.pairs):
            sel = np.nonzero(win == m)[0]
            if len(sel):
                _, am, bm = num_FJ(lambda x: self.pair(x, i, j), p[sel])
                a[sel] = am; b[sel] = bm
        return F, a, b

    def world_of(self, p):
        z = p[..., 3:5]
        return np.argmin(np.stack([np.linalg.norm(z - ck, axis=-1) for ck in self.c], -1), -1)

    def hits(self, p, q, dn, dist):
        hit, col = sphere_hits(self.spheres, self.world_of, p, q, dn, dist)
        f = (p[:, 1] > self.floor) & (q[:, 1] <= self.floor) & ~hit
        if f.any():
            w = self.world_of(q[f])
            col[f] = floor_color(np.stack([WORLD[k] for k in w]), q[f, :3], dist[f]); hit |= f
        return hit, col

    def sky(self, p, d):
        w = self.world_of(p)
        base = np.stack([WORLD[k] for k in w])
        hgt = np.clip(d[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
        return (0.55 + 0.45 * base) * (0.6 + 0.4 * hgt)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class FlatThroat:
    """Wormhole whose throat is T^2 x R (flat): a Clifford torus
    {|x12| = a, |x34| = b} extruded along x5, morphed near the ends into a
    torus of revolution lying in x4 = 0, fold-unioned with worlds x5 = +-L."""
    def __init__(self, a=1.0, b=0.45, L=3.0, L1=0.9, L2=1.9, k=0.35, floor=-1.8):
        self.a, self.b, self.L, self.L1, self.L2, self.k, self.floor = a, b, L, L1, L2, k, floor
        self.cam = np.array([a, 0.0, b, 0.0, 0.0])
        self.look, self.up, self.right = [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
        self.spheres = [  # objects inside the throat, given on the Clifford torus
            (2, self.cl(np.pi, 0.0, 0.0), 0.28, np.array([0.9, 0.2, 0.2])),
            (2, self.cl(np.pi / 2, np.pi, 0.4), 0.25, np.array([0.2, 0.4, 0.95])),
            (2, self.cl(-np.pi / 2, np.pi / 2, -0.3), 0.22, np.array([0.95, 0.85, 0.2])),
        ]

    def cl(self, th, ph, s):
        return np.array([self.a * np.cos(th), self.a * np.sin(th), self.b * np.cos(ph), self.b * np.sin(ph), s])

    def throat(self, p):
        x1, x2, x3, x4, x5 = [p[..., i] for i in range(5)]
        rho = np.hypot(x1, x2)
        al = 0.5 * np.pi * smoothstep(self.L1, self.L2, np.abs(x5))
        # profile circle: centre (a,0,0) in (rho,x3,x4); plane spanned by e3 and cos(al) e4 + sin(al) e_rho
        q0, q3, q4 = rho - self.a, x3, x4
        h = -np.sin(al) * q4 + np.cos(al) * q0          # out of the circle's plane
        w = np.cos(al) * q4 + np.sin(al) * q0           # in-plane, second axis
        return (np.hypot(q3, w) - self.b) + 1j * h

    def worlds(self, p):
        return (self.L - np.abs(p[..., 4])) - 1j * p[..., 3]

    def F(self, p):
        A, B = self.worlds(p), self.throat(p)
        hk = np.clip(0.5 + 0.5 * (B.real - A.real) / self.k, 0, 1)
        d = B.real + (A.real - B.real) * hk - self.k * hk * (1 - hk)
        wgt = hk * hk * (3 - 2 * hk)
        return d + 1j * (B.imag + (A.imag - B.imag) * wgt)

    def F_J(self, p):
        return num_FJ(self.F, p)

    def world_of(self, p):  # 0: world x5=-L, 1: world x5=+L, 2: throat
        x5 = p[..., 4]
        return np.where(x5 < -self.L + 0.6, 0, np.where(x5 > self.L - 0.6, 1, 2))

    def hits(self, p, q, dn, dist):
        hit, col = sphere_hits(self.spheres, self.world_of, p, q, dn, dist)
        inw = np.abs(q[:, 4]) > self.L - 0.05
        f = inw & (p[:, 1] > self.floor) & (q[:, 1] <= self.floor) & ~hit
        if f.any():
            w = self.world_of(q[f])
            col[f] = floor_color(np.stack([WORLD[k] for k in w]), q[f, :3], dist[f]); hit |= f
        return hit, col

    def sky(self, p, d):
        w = self.world_of(p)
        base = np.stack([np.array([0.6, 0.6, 0.65]) if k == 2 else WORLD[k] for k in w])
        return (0.5 + 0.5 * base) * 0.85


def smootherstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def floor_L(base, x, dist, size=1.4):
    """Floor checker with a chiral L glyph per tile (bar along +x3, foot along +x1)."""
    fx = x[:, 0] / size - np.floor(x[:, 0] / size)
    fz = x[:, 2] / size - np.floor(x[:, 2] / size)
    ch = ((np.floor(x[:, 0] / size) + np.floor(x[:, 2] / size)) % 2)[:, None]
    c = base * (0.62 + 0.38 * ch)
    L = ((fx > 0.25) & (fx < 0.37) & (fz > 0.2) & (fz < 0.8)) | ((fx > 0.25) & (fx < 0.7) & (fz > 0.2) & (fz < 0.32))
    c[L] = 0.15 * c[L] + 0.85
    fog = np.exp(-0.03 * dist)[:, None]
    return c * fog + (1 - fog) * 0.85


class Handle3:
    """3D world R^3 = {x4 = x5 = 0} with a handle leaving through x4.
    mirror=True turns the S^2 cross-section through x5 (x3 -> x5 -> -x3)."""
    def __init__(self, mirror=False, R=1.5, r=0.6, k=0.4, floor=-1.4):
        self.mirror, self.R, self.r, self.k, self.floor = mirror, R, r, k, floor
        self.cam = np.array([-R, -0.1, -2.6, 0.0, 0.0])
        self.look, self.up, self.right = [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]
        self.spheres = []

    def tube(self, p):
        x1, x2, x3, x4, x5 = [p[..., i] for i in range(5)]
        t = np.arctan2(x4, -x1)
        t = np.where(t < 0, np.where(x1 > 0, np.pi, 0.0), t)
        ct, st = np.cos(t), np.sin(t)
        tau = np.pi * smootherstep(0.25 * np.pi, 0.75 * np.pi, t) if self.mirror else 0 * t
        cs, ss = np.cos(tau), np.sin(tau)
        d1, d4 = x1 + self.R * ct, x4 - self.R * st
        xn = -ct * d1 + st * d4
        xT = st * d1 + ct * d4
        xe = cs * x3 + ss * x5
        xq = -ss * x3 + cs * x5
        return (np.sqrt(xn * xn + x2 * x2 + xe * xe + xT * xT) - self.r) + 1j * xq

    def plane(self, p):
        s = np.where(p[..., 0] > 0, -1.0, 1.0) if self.mirror else 1.0
        return p[..., 3] + 1j * p[..., 4] * s

    def F(self, p):
        A, B = self.plane(p), self.tube(p)
        hk = np.clip(0.5 + 0.5 * (B.real - A.real) / self.k, 0, 1)
        d = B.real + (A.real - B.real) * hk - self.k * hk * (1 - hk)
        w = hk * hk * (3 - 2 * hk)
        return d + 1j * (B.imag + (A.imag - B.imag) * w)

    def F_J(self, p):
        return num_FJ(self.F, p)

    def world_of(self, p):
        return np.zeros(p.shape[:-1], int)

    def hits(self, p, q, dn, dist):
        hit = np.zeros(len(q), bool); col = np.zeros((len(q), 3))
        onw = np.abs(q[:, 3]) < 0.05
        f = onw & (p[:, 1] > self.floor) & (q[:, 1] <= self.floor)
        if f.any():
            col[f] = floor_L(WORLD[0][None, :] + 0 * q[f, :3], q[f, :3], dist[f]); hit |= f
        return hit, col

    def sky(self, p, d):
        hgt = np.clip(d[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
        return (0.55 + 0.45 * WORLD[0]) * (0.6 + 0.4 * hgt) + 0 * d[:, :3]
