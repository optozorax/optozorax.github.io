import numpy as np
from pga import GA
rng = np.random.default_rng(1)
for n in (3, 4, 5):
    G = GA(n)
    for trial in range(3):
        # random point p, random "SDF values and gradients" (f, a), (g, b) at p
        p = rng.normal(size=n); a = rng.normal(size=n); b = rng.normal(size=n); f, g = rng.normal(size=2)
        # tangent hyperplanes: f + a.(x-p) = 0  ->  a.x + (f - a.p) = 0
        A = G.vec(f - a @ p, a); B = G.vec(g - b @ p, b)
        L = G.wedge(A, B)                       # codim-2 flat (bivector)
        P = G.point(p)
        # Gauss-Newton foot point
        J = np.stack([a, b]); gn = p - J.T @ np.linalg.solve(J @ J.T, np.array([f, g]))
        Linv = L / G.gp(L, L)[0]                # L^2 is a scalar
        cand = {
          "(P.L) L^-1": G.gp(G.inner(P, L), Linv),
          "(P.L) ^ L": G.wedge(G.inner(P, L), L),
        }
        out = []
        for name, X in cand.items():
            Xn = G.gsel(X, n)
            other = np.abs(X - Xn).max()
            q = G.point_coords(Xn)
            out.append("%s: err=%.2e non-point-part=%.1e" % (name, np.abs(q - gn).max(), other))
        # distance from p to the flat via the join (P v L), normalised
        H = G.vee(P, L)   # hyperplane through P and L
        hn = np.sqrt(sum(H[1 << (i+1)]**2 for i in range(n)))
        Pw = G.point(np.zeros(n)); Pw_n = abs(P[np.argmax(np.abs(Pw))] / Pw[np.argmax(np.abs(Pw))])
        Lnorm = np.sqrt(abs(G.gp(L, L)[0]))
        dist_pga = hn / (Lnorm * Pw_n)
        print("n=%d  %s | dist: pga=%.6f exact=%.6f" % (n, " | ".join(out), dist_pga, np.linalg.norm(gn - p)))
