"""
Generalized 1D self-collision chain (same setup as ../sims/collision.py:
head-on collision of a body with its own scaled copies, portal-periodic
scale-wrap boundary), but with the teleport rules as free exponents:

    mass of copy n      :  m_n   = k^(gamma * n)
    contact stiffness   :  kap   = K0 * k^(s * min_level)

so (gamma, s) = (0, 0)  is the engine model      (mass unchanged by teleport)
   (gamma, s) = (2, 2)  is scale-consistent 2D   (mass ~ area, same material)
   (gamma, s) = (3, 2)  would be a 3D-mass variant
   (gamma, s) = (0,-2)  is the "magic stiffness" material that bounces.

Hand-derived claim to test: whether the object survives (bounces) depends
ONLY on s (bounce iff kap ratio per level k^s > 1/k, i.e. s < -1) and not at
all on the inertial-mass rule gamma.  Also tracks the energy ledger.
"""
import numpy as np


def run(gamma=0.0, s=0.0, K=0.7, N=10, NG=5, depth=4,
        C0=1.0, W=0.12, V0=0.3, K0=200.0, T=12.0, nsave=2000):
    n = np.arange(N)
    scale = K ** n
    x = C0 * scale.copy()
    hw = W * scale
    v = -V0 * scale
    m = K ** (gamma * n)

    idx_full = np.arange(-NG, N + NG)
    hw_full = W * K ** idx_full.astype(float)
    ii, jj = np.triu_indices(len(idx_full), 1)
    keep = (jj - ii) <= depth
    ii, jj = ii[keep], jj[keep]
    smin = np.minimum(idx_full[ii], idx_full[jj]).astype(float)
    kap_pair = K0 * K ** (s * smin)
    # energy convention: each physical contact counted once per unit cell ->
    # pairs whose OUTER member is a real copy
    epair = (ii >= NG) & (ii < NG + N)

    def forces(x, want_pe=False):
        x_ext = np.concatenate([x[N - NG:] * K ** (-N), x, x[:NG] * K ** N])
        ov = (x_ext[jj] + hw_full[jj]) - (x_ext[ii] - hw_full[ii])
        act = ov > 0
        F_ext = np.zeros(len(x_ext))
        if np.any(act):
            f = kap_pair[act] * ov[act]
            np.add.at(F_ext, ii[act], +f)
            np.add.at(F_ext, jj[act], -f)
        pe = 0.5 * np.sum(kap_pair[act & epair] * ov[act & epair] ** 2) \
            if want_pe else 0.0
        return F_ext[NG:NG + N], pe, np.any(act)

    wmax = np.sqrt(np.max(kap_pair) * 2.0 / np.min(m))
    dt = 0.02 / wmax
    steps = int(T / dt)
    save_every = max(1, steps // nsave)
    mid = N // 2
    ts, xis, sq, ke, pe = [], [], [], [], []
    verdict = 'ongoing'
    F, _, _ = forces(x)
    for st in range(steps):
        v += 0.5 * dt * F / m
        x += dt * v
        F, _, _ = forces(x)
        v += 0.5 * dt * F / m
        if st % save_every == 0:
            _, p, contact = forces(x, want_pe=True)
            ts.append(st * dt)
            xis.append(x / scale)
            o = max((x[mid + 1] + hw[mid + 1]) - (x[mid] - hw[mid]), 0.0)
            sq.append(o / (2 * hw[mid + 1]))
            ke.append(0.5 * np.sum(m * v ** 2))
            pe.append(p)
            t = st * dt
            if x[mid] / scale[mid] < 0.02:
                verdict = 'collapse'
                break
            if t > 1.5 and not contact and np.all(v > 0):
                verdict = 'bounce'
                break
    return dict(ts=np.array(ts), xis=np.array(xis), sq=np.array(sq),
                ke=np.array(ke), pe=np.array(pe), K=K, N=N,
                gamma=gamma, s=s, verdict=verdict,
                max_squash=float(np.max(sq)) if sq else 0.0)
