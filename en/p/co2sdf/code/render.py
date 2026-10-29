"""Exponential-map ("scan") renderer for 2D worlds embedded in R^4."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from co2sdf import *
from scenes import SCENES


def step(F, p, v, dt, iters=2, J=None):
    """Second-order geodesic step (see co2sdf.geodesic_step); recomputes the
    start jet each call, which is simpler but costs one extra Jacobian."""
    q, v, _ = geodesic_step(F, p, v, dt, J=J)
    return q, v


def trace_fan(scene, n_dir=1440, r_max=None, dt=0.01, heading=0.0):
    r_max = r_max or getattr(scene, "r_max", 6.0)
    F = scene.F
    J = getattr(scene, "jac", None)
    cam = scene.cam[None, :].astype(float)
    for _ in range(8):
        val = F(cam); a, b = J(cam) if J else jac(F, cam); ra, rb = recip(a, b)
        cam = cam - val.real[..., None] * ra - val.imag[..., None] * rb
    a, b = J(cam) if J else jac(F, cam)
    e1 = normalize(tangent(a, b, scene.hint1[None, :]))
    t2 = tangent(a, b, scene.hint2[None, :])
    e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    th = np.linspace(0, 2 * np.pi, n_dir, endpoint=False) + heading
    p = np.repeat(cam, n_dir, 0)
    v = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    n_steps = int(np.ceil(r_max / dt))
    cols = np.zeros((n_dir, n_steps + 1, 3))
    pos = np.zeros((n_dir, n_steps + 1, 4))
    cols[:, 0] = scene.color(p)
    pos[:, 0] = p
    for i in range(n_steps):
        p, v = step(F, p, v, dt, J=J)
        cols[:, i + 1] = scene.color(p)
        pos[:, i + 1] = p
    return cols, pos, dt


def scan_image(cols, dt, res=512, r_max=None):
    r_max = r_max or (cols.shape[1] - 1) * dt
    n_dir, n_r, _ = cols.shape
    y, x = np.mgrid[-1:1:res * 1j, -1:1:res * 1j]
    r = np.hypot(x, y) * r_max
    th = np.mod(np.arctan2(-y, x), 2 * np.pi)  # image y axis points down
    j = np.round(th / (2 * np.pi) * n_dir).astype(int) % n_dir
    i = np.clip(np.round(r / dt).astype(int), 0, n_r - 1)
    img = cols[j, i]
    img[r > r_max] = 1.0
    # camera marker
    img[(r < 0.02 * r_max)] = 0.05
    return img


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENES)
    for name in names:
        sc = SCENES[name]
        cols, pos, dt = trace_fan(sc)
        img = scan_image(cols, dt)
        plt.imsave("scan_%s.png" % name, np.clip(img, 0, 1))
        res = np.abs(sc.F(pos.reshape(-1, 4))).max()
        print(name, "max |F| along geodesics: %.2e" % res)
