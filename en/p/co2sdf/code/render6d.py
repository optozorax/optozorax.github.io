"""Codimension-k renderer (here k=3: 3D worlds in R^6). Jet: F (n,k), J (n,k,D)."""
import numpy as np
from render5d import floor_color, save
from render3d import WORLD


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def gram_apply(J, r):   # J^T (J J^T)^{-1} r
    G = J @ np.swapaxes(J, -1, -2)
    return (np.swapaxes(J, -1, -2) @ np.linalg.solve(G, r[..., None]))[..., 0]


def tangent(J, v):
    return v - gram_apply(J, (J @ v[..., None])[..., 0])


def num_jet(F, p, h=1e-5):
    D = p.shape[-1]; f0 = F(p); J = np.empty(p.shape[:-1] + (f0.shape[-1], D))
    for i in range(D):
        e = np.zeros(D); e[i] = h
        J[..., i] = (F(p + e) - F(p - e)) / (2 * h)
    return f0, J


def cutoff(t):
    t = np.clip((t - 0.5) / 0.5, 0, 1)
    return 1 - t * t * t * (t * (6 * t - 15) + 10)


class InversionNeck:
    """World A = u-space {v = 0}, world B = v-space {u = 0}, u, v in R^3.
    Neck union by inversion: A-branch u-side U = v - eps*phi*u/|u|^2."""
    def __init__(self, eps=0.5, k=2.6, floor=-1.4):
        self.eps, self.k, self.floor = eps, k, floor
        self.cam = np.array([0.3, 0.0, -4.0, 0, 0, 0])
        self.look, self.up, self.right = [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]
        self.spheres = [(0, np.array([1.8, -0.9, 0.5]), 0.5, np.array([0.9, 0.2, 0.2])),
                        (1, np.array([-0.6, -0.9, 2.5]), 0.6, np.array([0.95, 0.9, 0.2]))]

    def F(self, p):
        u, v = p[..., :3], p[..., 3:]
        ru, rv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        A = v - (self.eps * cutoff(ru / self.k) / np.maximum(ru, 1e-12) ** 2)[..., None] * u
        B = u - (self.eps * cutoff(rv / self.k) / np.maximum(rv, 1e-12) ** 2)[..., None] * v
        return np.where((rv < ru)[..., None], A, B)

    def F_locked(self, p0):
        """F with the branch frozen to the one selected at p0 (for Jacobians)."""
        u0, v0 = p0[..., :3], p0[..., 3:]
        selA = np.linalg.norm(v0, axis=-1) < np.linalg.norm(u0, axis=-1)
        def F(p):
            u, v = p[..., :3], p[..., 3:]
            ru, rv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
            A = v - (self.eps * cutoff(ru / self.k) / np.maximum(ru, 1e-12) ** 2)[..., None] * u
            B = u - (self.eps * cutoff(rv / self.k) / np.maximum(rv, 1e-12) ** 2)[..., None] * v
            return np.where(selA[..., None], A, B)
        return F

    def world_of(self, p):
        return (np.linalg.norm(p[..., 3:], axis=-1) > np.linalg.norm(p[..., :3], axis=-1)).astype(int)

    def coords(self, p):  # in-world coordinates (x, y-up, z) for each world
        w = self.world_of(p)
        return np.where(w[:, None] == 0, p[:, :3], p[:, 3:]), w


def render(sc, res=(480, 320), fov=75, dt=0.02, max_len=16.0):
    W_, H_ = res
    p = np.asarray(sc.cam, float)[None]
    for _ in range(8):
        f, J = num_jet(sc.F, p); p = p - gram_apply(J, f)
    f, J = num_jet(sc.F, p)
    fwd = normalize(tangent(J, np.asarray(sc.look, float)[None]))
    up = tangent(J, np.asarray(sc.up, float)[None]); up = normalize(up - (up * fwd).sum(-1)[..., None] * fwd)
    rt = tangent(J, np.asarray(sc.right, float)[None])
    rt = normalize(rt - (rt * fwd).sum(-1)[..., None] * fwd - (rt * up).sum(-1)[..., None] * up)
    t = np.tan(np.radians(fov) / 2)
    uu, vv = np.meshgrid(np.linspace(-t, t, W_), np.linspace(t * H_ / W_, -t * H_ / W_, H_))
    d = normalize(fwd + uu.reshape(-1, 1) * rt + vv.reshape(-1, 1) * up)
    n = len(d); P = np.repeat(p, n, 0); col = np.zeros((n, 3)); alive = np.ones(n, bool); dist = 0.0
    while alive.any() and dist < max_len:
        idx = np.nonzero(alive)[0]
        q = P[idx] + d[idx] * dt
        for _ in range(2):
            f, J = num_jet(sc.F, q); q = q - gram_apply(J, f)
        dn = normalize(tangent(J, d[idx]))
        x0, _ = sc.coords(P[idx]); x1, w = sc.coords(q)
        hit = np.zeros(len(idx), bool)
        for (k, c0, r0, cc) in sc.spheres:
            m = (w == k) & (np.linalg.norm(x1 - c0, axis=-1) < r0) & ~hit
            if m.any():
                nrm = normalize(x1[m] - c0)
                lam = 0.35 + 0.65 * np.clip(nrm @ normalize(np.array([0.4, 1.0, -0.5])), 0, 1)
                col[idx[m]] = cc * lam[:, None]; hit |= m
        fl = (x0[:, 1] > sc.floor) & (x1[:, 1] <= sc.floor) & ~hit
        if fl.any():
            col[idx[fl]] = floor_color(np.stack([WORLD[k] for k in w[fl]]), x1[fl], np.full(fl.sum(), dist)); hit |= fl
        alive[idx[hit]] = False
        P[idx] = q; d[idx] = dn; dist += dt
    idx = np.nonzero(alive)[0]
    if len(idx):
        _, w = sc.coords(P[idx]); base = np.stack([WORLD[k] for k in w])
        x1, _ = sc.coords(d[idx] + 0 * P[idx])
        col[idx] = (0.55 + 0.45 * base) * 0.85
    return col.reshape(H_, W_, 3)


if __name__ == "__main__":
    sc = InversionNeck()
    save(render(sc), "neck6d.png"); print("ok neck6d")
