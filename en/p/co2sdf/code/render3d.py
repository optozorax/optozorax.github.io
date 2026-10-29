"""Perspective renderer for 3D worlds embedded in R^5 (codimension 2).
p = (x1, x2, x3, z_re, z_im); x2 is "up" inside the worlds."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORLD = [np.array(c) for c in ([0.30, 0.75, 0.25], [0.95, 0.55, 0.15], [0.25, 0.50, 0.95], [0.75, 0.35, 0.85])]


def recip(a, b):
    aa, ab, bb = (a * a).sum(-1), (a * b).sum(-1), (b * b).sum(-1)
    det = aa * bb - ab * ab
    return ((bb[..., None] * a - ab[..., None] * b) / det[..., None],
            (aa[..., None] * b - ab[..., None] * a) / det[..., None])


def tangent(a, b, v):
    ra, rb = recip(a, b)
    return v - (v * a).sum(-1)[..., None] * ra - (v * b).sum(-1)[..., None] * rb


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class PortalScene:
    """N parallel 3D worlds at zeta = rho * omega^k, joined by one ring-shaped
    branch locus: zeta^N = rho^N W(x), W = (|x|^2 - R^2 + 2 i R x3) / (|x|^2 + R^2)."""

    def __init__(self, N=2, rho=1.0, R=1.6, floor=-1.9):
        self.N, self.rho, self.R, self.floor = N, rho, R, floor
        self.spheres = [  # (world, centre, radius, colour)
            (0, np.array([2.4, -1.2, 2.0]), 0.7, np.array([0.9, 0.2, 0.2])),
            (1, np.array([-0.6, -1.0, 4.0]), 0.9, np.array([0.95, 0.9, 0.2])),
            (2 % N, np.array([0.8, -1.1, 3.5]), 0.8, np.array([0.9, 0.3, 0.9])),
        ]

    def W(self, x):
        r2 = (x * x).sum(-1)
        A = r2 - self.R ** 2 + 2j * self.R * x[..., 2]
        B = r2 + self.R ** 2
        dA = 2 * x + 0j
        dA[..., 2] += 2j * self.R
        dB = 2 * x
        return A / B, (dA * B[..., None] - A[..., None] * dB) / (B * B)[..., None]

    def F_J(self, p):
        x, z = p[..., :3], p[..., 3] + 1j * p[..., 4]
        N, rho = self.N, self.rho
        W, dW = self.W(x)
        s = N * rho ** (N - 1)
        F = (z ** N - rho ** N * W) / s
        dz = (z / rho) ** (N - 1)                         # dF/dzeta
        g = np.concatenate([-rho ** N * dW / s, dz[..., None], 1j * dz[..., None]], -1)
        return F, g.real, g.imag

    def sheet(self, p):
        x, z = p[..., :3], p[..., 3] + 1j * p[..., 4]
        W, _ = self.W(x)
        base = self.rho * np.abs(W) ** (1 / self.N) * np.exp(1j * np.angle(W) / self.N)
        k = np.arange(self.N)
        cand = base[..., None] * np.exp(2j * np.pi * k / self.N)
        return np.argmin(np.abs(cand - z[..., None]), -1)

    def on_world(self, x, k):
        """Point of R^5 on sheet k above x."""
        W, _ = self.W(x)
        z = self.rho * np.abs(W) ** (1 / self.N) * np.exp(1j * (np.angle(W) + 2 * np.pi * k) / self.N)
        return np.concatenate([x, z.real[..., None], z.imag[..., None]], -1)

    def ring_dist(self, x):
        return np.hypot(np.hypot(x[..., 0], x[..., 1]) - self.R, x[..., 2])


def project(scene, p, iters=2):
    for _ in range(iters):
        F, a, b = scene.F_J(p)
        ra, rb = recip(a, b)
        p = p - F.real[..., None] * ra - F.imag[..., None] * rb
    return p, a, b


def render(scene, cam_x, look, res=(480, 320), fov=75, dt=0.02, max_len=18.0, world=0, frame=True):
    W_, H_ = res
    cam = scene.on_world(np.array(cam_x, float)[None], world)
    cam, a, b = project(scene, cam, 4)
    fwd = normalize(tangent(a, b, np.array([[look[0], look[1], look[2], 0, 0]], float)))
    up = tangent(a, b, np.array([[0, 1.0, 0, 0, 0]]))
    up = normalize(up - (up * fwd).sum(-1)[..., None] * fwd)
    # right = third tangent direction orthogonal to fwd, up
    right = tangent(a, b, np.array([[1.0, 0, 0, 0, 0]]))
    right = right - (right * fwd).sum(-1)[..., None] * fwd - (right * up).sum(-1)[..., None] * up
    right = normalize(right)
    if np.cross(fwd[0, :3], up[0, :3]) @ right[0, :3] > 0:
        right = -right
    t = np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.linspace(-t, t, W_), np.linspace(t * H_ / W_, -t * H_ / W_, H_))
    d = normalize(fwd + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up)
    n = d.shape[0]
    p = np.repeat(cam, n, 0)
    col = np.zeros((n, 3))
    alive = np.ones(n, bool)
    for i in range(int(max_len / dt)):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        q = p[idx] + d[idx] * dt
        q, a, b = project(scene, q)
        d_new = normalize(tangent(a, b, d[idx]))
        x0, x1 = p[idx, :3], q[:, :3]
        # floor hit
        hit = (x0[:, 1] > scene.floor) & (x1[:, 1] <= scene.floor)
        if hit.any():
            s = scene.sheet(q[hit])
            xs = x1[hit]
            ch = ((np.floor(xs[:, 0]) + np.floor(xs[:, 2])) % 2)[:, None]
            base = np.stack([WORLD[k] for k in s])
            c = base * (0.65 + 0.35 * ch)
            fog = np.exp(-0.02 * (i * dt))
            col[idx[hit]] = c * fog + (1 - fog) * 0.8
            alive[idx[hit]] = False
        # spheres
        for (k, c0, r0, cc) in scene.spheres:
            if k >= scene.N:
                continue
            inside = (np.linalg.norm(x1 - c0, axis=-1) < r0) & alive[idx]
            if inside.any():
                s = scene.sheet(q[inside])
                ok = s == k
                if ok.any():
                    nrm = normalize(x1[inside][ok] - c0)
                    lam = 0.35 + 0.65 * np.clip(nrm @ normalize(np.array([0.4, 1.0, -0.5])), 0, 1)
                    col[idx[inside][ok]] = cc * lam[:, None]
                    alive[idx[inside][ok]] = False
        # the branch ring itself (drawn as a thin golden frame)
        if frame:
            fr = (scene.ring_dist(x1) < 0.045) & alive[idx]
            if fr.any():
                col[idx[fr]] = np.array([0.85, 0.7, 0.25])
                alive[idx[fr]] = False
        p[idx] = q
        d[idx] = d_new
    # sky for rays that never hit anything: tint by the world they ended in
    idx = np.nonzero(alive)[0]
    if len(idx):
        s = scene.sheet(p[idx])
        base = np.stack([WORLD[k] for k in s])
        h = np.clip(d[idx, 1] * 0.5 + 0.5, 0, 1)[:, None]
        col[idx] = (0.55 + 0.45 * base) * (0.6 + 0.4 * h)
    return col.reshape(H_, W_, 3)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sc = PortalScene(N=N)
    img = render(sc, cam_x=[0.0, -0.2, -5.0], look=[0, 0, 1], res=(480, 320))
    plt.imsave("portal5d_N%d.png" % N, np.clip(img, 0, 1))
    print("done")
