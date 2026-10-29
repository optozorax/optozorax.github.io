import numpy as np
from co2sdf import jac, recip, dot, normalize, tangent

def stepA(F, p, v, dt):           # current loop: project v, drift, Newton to the surface
    a, b = jac(F, p); v = normalize(tangent(a, b, v))
    q = p + v * dt
    for _ in range(3):
        val = F(q); a2, b2 = jac(F, q); ra, rb = recip(a2, b2)
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
    return q, v

def stepB(F, p, v, dt):           # RATTLE: correct along the normal plane at p, then project chord velocity
    a, b = jac(F, p)
    q = p + v * dt
    mu = np.zeros(p.shape[:-1] + (2,))
    for _ in range(4):
        val = F(q); a2, b2 = jac(F, q)
        # J(q) [a b] dmu = -F(q)
        M = np.stack([np.stack([dot(a2, a), dot(a2, b)], -1), np.stack([dot(b2, a), dot(b2, b)], -1)], -2)
        rhs = -np.stack([val.real, val.imag], -1)
        d = np.linalg.solve(M, rhs[..., None])[..., 0]
        q = q + d[..., :1] * a + d[..., 1:] * b
    a2, b2 = jac(F, q)
    v2 = normalize(tangent(a2, b2, (q - p) / dt))
    return q, v2

def run(F, p0, v0, T, n, step):
    p, v = p0.copy(), v0.copy(); dt = T / n
    for _ in range(n):
        p, v = step(F, p, v, dt)
    return p

# Clifford torus T^2(a,b) in R^4: geodesics are straight in (a*theta, b*phi)
A_, B_ = 1.0, 0.6
F = lambda p: (np.hypot(p[..., 0], p[..., 1]) - A_) + 1j * (np.hypot(p[..., 2], p[..., 3]) - B_)
p0 = np.array([[A_, 0, B_, 0.0]]); ang = 0.7
v0 = np.array([[0, np.cos(ang), 0, np.sin(ang)]]); T = 4.0
th, ph = T * np.cos(ang) / A_, T * np.sin(ang) / B_
exact = np.array([A_ * np.cos(th), A_ * np.sin(th), B_ * np.cos(ph), B_ * np.sin(ph)])
# unit sphere in R^3 written as a co2sdf with a flat 4th direction: F = (|x|-1) + i x4
Fs = lambda p: (np.linalg.norm(p[..., :3], axis=-1) - 1.0) + 1j * p[..., 3]
ps0 = np.array([[1.0, 0, 0, 0]]); vs0 = np.array([[0, 1.0, 0, 0]]); Ts = 4.0
exact_s = np.array([np.cos(Ts), np.sin(Ts), 0, 0])
for name, step in [("A current", stepA), ("B rattle ", stepB)]:
    e1 = [np.linalg.norm(run(F, p0, v0, T, n, step)[0] - exact) for n in (50, 100, 200, 400)]
    e2 = [np.linalg.norm(run(Fs, ps0, vs0, Ts, n, step)[0] - exact_s) for n in (50, 100, 200, 400)]
    print(name, "torus err", ["%.2e" % e for e in e1], "ratios", ["%.2f" % (e1[i] / e1[i + 1]) for i in range(3)])
    print(name, "sphere err", ["%.2e" % e for e in e2], "ratios", ["%.2f" % (e2[i] / e2[i + 1]) for i in range(3)])

def stepC(F, p, v, dt):   # current projection, but new direction = projected chord
    q = p + v * dt
    a, b = jac(F, p)
    for it in range(3):
        if it > 0:
            a, b = jac(F, q)
        val = F(q); ra, rb = recip(a, b)
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
    a2, b2 = jac(F, q)
    return q, normalize(tangent(a2, b2, q - p))

def stepD(F, p, v, dt):   # C but the chord is reflected: v' = 2*chord/dt - v (then projected)
    q = p + v * dt
    a, b = jac(F, p)
    for it in range(3):
        if it > 0:
            a, b = jac(F, q)
        val = F(q); ra, rb = recip(a, b)
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
    a2, b2 = jac(F, q)
    return q, normalize(tangent(a2, b2, 2 * (q - p) / dt - v))

for name, step in [("C chord   ", stepC), ("D reflect ", stepD)]:
    e1 = [np.linalg.norm(run(F, p0, v0, T, n, step)[0] - exact) for n in (50, 100, 200, 400)]
    e2 = [np.linalg.norm(run(Fs, ps0, vs0, Ts, n, step)[0] - exact_s) for n in (50, 100, 200, 400)]
    print(name, "torus err", ["%.2e" % e for e in e1], "ratios", ["%.2f" % (e1[i] / e1[i + 1]) for i in range(3)])
    print(name, "sphere err", ["%.2e" % e for e in e2], "ratios", ["%.2f" % (e2[i] / e2[i + 1]) for i in range(3)])
