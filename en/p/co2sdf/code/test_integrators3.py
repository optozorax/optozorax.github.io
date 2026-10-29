import numpy as np
exec(open('test_integrators.py').read().split("# Clifford torus")[0])
exec("def stepC" + open('test_integrators.py').read().split("def stepC")[1].split("for name, step")[0])
def stepE(F, p, v, dt, iters=2):   # library / render_hq form: Newton from the trial point's normals, chord update
    q = p + v * dt
    for _ in range(iters):
        val = F(q); a, b = jac(F, q); ra, rb = recip(a, b)
        q = q - val.real[..., None] * ra - val.imag[..., None] * rb
    return q, normalize(tangent(a, b, q - p))
def stepE3(F, p, v, dt):
    return stepE(F, p, v, dt, 3)
A_, B_ = 1.0, 0.6
F = lambda p: (np.hypot(p[..., 0], p[..., 1]) - A_) + 1j * (np.hypot(p[..., 2], p[..., 3]) - B_)
p0 = np.array([[A_, 0, B_, 0.0]]); ang = 0.7
v0 = np.array([[0, np.cos(ang), 0, np.sin(ang)]]); T = 4.0
th, ph = T * np.cos(ang) / A_, T * np.sin(ang) / B_
exact = np.array([A_ * np.cos(th), A_ * np.sin(th), B_ * np.cos(ph), B_ * np.sin(ph)])
Ft = lambda p: (np.hypot(np.hypot(p[..., 0], p[..., 1]) - 1.0, p[..., 2]) - 0.4) + 1j * p[..., 3]
pt0 = np.array([[1.4, 0, 0, 0.0]]); vt0 = normalize(np.array([[0, 0.8, 0.6, 0.0]])); Tt = 3.0
ref = run(Ft, pt0, vt0, Tt, 40000, stepB)[0]
for name, step in [("C  old-normal first, chord", stepC), ("E  trial-normal Newton x2, chord", stepE), ("E3 trial-normal Newton x3, chord", stepE3)]:
    e1 = [np.linalg.norm(run(F, p0, v0, T, n, step)[0] - exact) for n in (50, 100, 200, 400)]
    e2 = [np.linalg.norm(run(Ft, pt0, vt0, Tt, n, step)[0] - ref) for n in (50, 100, 200, 400)]
    print("%-34s Clifford %s ratios %s" % (name, ["%.2e" % x for x in e1], ["%.2f" % (e1[i] / e1[i + 1]) for i in range(3)]))
    print("%-34s R3 torus %s ratios %s" % ("", ["%.2e" % x for x in e2], ["%.2f" % (e2[i] / e2[i + 1]) for i in range(3)]))
