import numpy as np
exec(open('test_integrators3.py').read().split("A_, B_ = 1.0, 0.6")[0])
def make_user(iters, chord=False):
    # the user's sdf_geodesic_step: project dir with the normal at p, step, then
    # sdf_project_from_normal: first correction along the OLD normal, later ones along fresh normals
    def step(F, p, v, dt):
        a0, b0 = jac(F, p)
        v = normalize(tangent(a0, b0, v))
        q = p + v * dt
        a, b = a0, b0
        for i in range(iters):
            val = F(q); ra, rb = recip(a, b)
            q = q - val.real[..., None] * ra - val.imag[..., None] * rb
            if i + 1 < iters:
                a, b = jac(F, q)
        if chord:
            a2, b2 = jac(F, q)
            v = normalize(tangent(a2, b2, q - p))
        return q, v
    return step
exec("A_, B_ = 1.0, 0.6" + open('test_integrators3.py').read().split("A_, B_ = 1.0, 0.6")[1].split("for name, step")[0])
cases = [("your loop, 1 iteration (inside view)", make_user(1)), ("your loop, 2 iterations", make_user(2)), ("your loop, 4 iterations", make_user(4)),
         ("your loop + chord, 2 iterations", make_user(2, True)), ("your loop + chord, 4 iterations", make_user(4, True))]
for name, step in cases:
    e1 = [np.linalg.norm(run(F, p0, v0, T, n, step)[0] - exact) for n in (50, 100, 200, 400)]
    e2 = [np.linalg.norm(run(Ft, pt0, vt0, Tt, n, step)[0] - ref) for n in (50, 100, 200, 400)]
    print("%-38s Clifford %s ratios %s" % (name, ["%.2e" % x for x in e1], ["%.2f" % (e1[i] / e1[i + 1]) for i in range(3)]))
    print("%-38s R3 torus %s ratios %s" % ("", ["%.2e" % x for x in e2], ["%.2f" % (e2[i] / e2[i + 1]) for i in range(3)]))
