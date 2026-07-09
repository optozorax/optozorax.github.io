"""
Phase diagram of the head-on self-collision over the space of teleport rules.

Question ("what if inertial mass changed differently between teleports?"):
does ANY inertial-mass rule save the object from being crushed?

Axes:  gamma = inertial-mass exponent  (copy n has mass k^{gamma n};
               0 = engine, 2 = scale-consistent 2D, -2 = the rule that would
               conserve kinetic energy through a crossing)
       s     = contact-stiffness exponent (kap ~ k^{s n};
               0 = engine, 2 = same material scaled, s < -1 = "magic")

Hand-derived prediction: the verdict is decided by STATICS alone -- in the
self-similar contact state the net force on every copy points inward unless
kap_{n+1}/kap_n = k^s > 1/k, i.e. bounce iff s < -1 -- a vertical boundary,
totally independent of the inertial-mass rule gamma.
"""
import numpy as np
import time
import chainlib

K = 0.8          # softer scale step than the main sim: keeps the stiffest
W = 0.08         # corner of the sweep integrable in reasonable time;
                 # the predicted boundary s = -1 does not depend on K
GAMMAS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
SS = np.linspace(-2.5, 2.0, 19)

if __name__ == "__main__":
    sq = np.full((len(GAMMAS), len(SS)), np.nan)
    verd = np.empty((len(GAMMAS), len(SS)), dtype=object)
    tcol = np.full((len(GAMMAS), len(SS)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, s in enumerate(SS):
            t0 = time.time()
            r = chainlib.run(gamma=g, s=s, K=K, N=8, NG=4, depth=3,
                             W=W, T=8.0, nsave=400)
            sq[i, j] = r['max_squash']
            verd[i, j] = r['verdict']
            if r['verdict'] == 'collapse':
                tcol[i, j] = r['ts'][-1]
            print(f"gamma={g:+.0f} s={s:+.2f}  verdict={r['verdict']:8s} "
                  f"max_squash={100*r['max_squash']:6.1f}%  "
                  f"t_end={r['ts'][-1]:5.2f}  [{time.time()-t0:5.1f}s]",
                  flush=True)
    np.savez('phase.npz', sq=sq, tcol=tcol,
             verd=np.array([[{'collapse': 0, 'bounce': 1, 'ongoing': 2}[v]
                             for v in row] for row in verd]),
             gammas=GAMMAS, ss=SS, K=K)
    print('saved phase.npz')
