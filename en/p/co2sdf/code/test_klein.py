import numpy as np
from co2sdf import *
from scenes import SCENES
from render import step

for name in ["handle", "klein"]:
    sc = SCENES[name]; F = sc.F
    # fan of rays from a point in front of mouth 1, heading roughly into the mouth
    n = 400
    cam = project(F, np.array([[-sc.R, -1.6, 0.0, 0.0]]), 8)
    a, b = jac(F, cam)
    e1 = normalize(tangent(a, b, np.array([[0.0, 1.0, 0, 0]])))
    t2 = tangent(a, b, np.array([[1.0, 0, 0, 0]])); e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    th = np.linspace(-0.25, 0.25, n)
    p = np.repeat(cam, n, 0)
    v = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2          # velocity
    w = -np.sin(th)[:, None] * e1 + np.cos(th)[:, None] * e2         # transported side vector
    def orient(v, w):  # orientation of (v, w) measured in the base plane (x1, x2)
        return np.sign(v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0])
    o0 = orient(v, w)
    maxz = np.zeros(n)
    for i in range(1500):
        p, v = step(F, p, v, 0.01)
        a, b = jac(F, p)
        w = tangent(a, b, w); w = w - dot(w, v)[..., None] * v; w = normalize(w)
        maxz = np.maximum(maxz, p[:, 2])
    back = (np.abs(p[:, 2]) < 1e-6) & (np.abs(p[:, 3]) < 1e-6) & (maxz > sc.R * 0.8) & (p[:, 0] > 0)
    o1 = orient(v, w)
    print("%-6s rays through handle and back on the plane: %d / %d ; orientation preserved: %s" %
          (name, back.sum(), n, np.unique((o0 * o1)[back])))
