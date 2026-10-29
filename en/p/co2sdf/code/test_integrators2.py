import numpy as np
exec(open('test_integrators.py').read().split("# Clifford torus")[0])
exec("def stepC" + open('test_integrators.py').read().split("def stepC")[1].split("for name, step")[0])
# codim-1 torus of revolution in R^3 (R=1, r=0.4), padded with a flat 4th direction
Ft = lambda p: (np.hypot(np.hypot(p[..., 0], p[..., 1]) - 1.0, p[..., 2]) - 0.4) + 1j * p[..., 3]
p0 = np.array([[1.4, 0, 0, 0.0]]); v0 = normalize(np.array([[0, 0.8, 0.6, 0.0]])); T = 3.0
ref = run(Ft, p0, v0, T, 40000, stepB)[0]
for name, step in [("current", stepA), ("chord  ", stepC)]:
    e = [np.linalg.norm(run(Ft, p0, v0, T, n, step)[0] - ref) for n in (50, 100, 200, 400)]
    print(name, "R^3 torus err", ["%.2e" % x for x in e], "ratios", ["%.2f" % (e[i] / e[i + 1]) for i in range(3)])
