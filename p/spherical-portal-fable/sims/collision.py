"""
Head-on collision of a body with its own scaled copy (spherical scaling portal).

Unrolled cover: an infinite 1D chain of copies, copy n at x_n = xi_n * k^n,
half-width w*k^n, initial velocity -v*k^n: the object flying straight at the
fixed point x = 0, i.e. head-on into its own smaller image.

BOUNDARY CONDITION: the true portal identification (scale-periodic wrap).
We integrate N copies; ghosts below are the top copies scaled down by k^N,
ghosts above are the bottom copies scaled up by k^-N. No truncation ends,
no artificial vacuum on the inner side.

Contact: linear springs on overlap, all pairs (piles can go several deep).
Models (m_n = copy mass, kap(i,j) = stiffness of contact, s = min(i,j)):

  engine     : m_n = 1,      kap = K0            (the script's engine model)
  consistent : m_n = k^{2n}, kap = K0*k^{2s}     (fully scale-consistent 2D physics)
  magic      : m_n = 1,      kap = K0*k^{-2s}    (counterfactual: smaller copies
                                                  stiffer by more than 1/k)

Hand-derived criterion: in a self-similar contact state the net force on every
copy points INWARD (deeper squash) unless kap_{n}/kap_{n-1} > 1/k, i.e. unless
smaller copies are stiffer than their bigger neighbour by more than the scale
factor. 'engine' and 'consistent' violate it -> squash, no bounce.
'magic' satisfies it -> should bounce. Testing that here.
"""
import numpy as np

K = 0.7
N = 10           # simulated levels
NG = 5           # ghost levels on each side
C0 = 1.0
W = 0.12         # < C0*(1-K)/(1+K) = 0.176 -> no initial overlap
V0 = 0.3
K0 = 200.0


def stiff(model, smin):
    if model == 'engine':
        return K0 * np.ones_like(smin, dtype=float)
    if model == 'consistent':
        return K0 * K ** (2.0 * smin)
    if model == 'magic':
        return K0 * K ** (-2.0 * smin)


def masses(model):
    n = np.arange(N)
    return np.ones(N) if model in ('engine', 'magic') else K ** (2.0 * n)


def forces(x, model, hw_full, idx_full, pair_i, pair_j, kap_pair):
    # build extended state: ghosts are wrapped, scaled copies of the real ones
    x_ext = np.concatenate([x[N - NG:] * K ** (-N), x, x[:NG] * K ** N])
    ov = (x_ext[pair_j] + hw_full[pair_j]) - (x_ext[pair_i] - hw_full[pair_i])
    act = ov > 0
    F_ext = np.zeros(len(x_ext))
    if np.any(act):
        f = kap_pair[act] * ov[act]
        np.add.at(F_ext, pair_i[act], +f)   # outer copy pushed outward
        np.add.at(F_ext, pair_j[act], -f)   # inner copy pushed inward
    return F_ext[NG:NG + N]


def run(model, T=12.0):
    n = np.arange(N)
    scale = K ** n
    x = C0 * scale
    hw = W * scale
    v = -V0 * scale
    m = masses(model)

    idx_full = np.arange(-NG, N + NG)                    # absolute level index
    hw_full = W * K ** idx_full.astype(float)
    ii, jj = np.triu_indices(len(idx_full), 1)           # i outer, j inner
    keep = (jj - ii) <= 4                                # contacts up to 4 levels deep
    ii, jj = ii[keep], jj[keep]
    smin = np.minimum(idx_full[ii], idx_full[jj]).astype(float)
    kap_pair = stiff(model, smin)

    wmax = np.sqrt(np.max(kap_pair) * 2.0 / np.min(m))
    dt = 0.02 / wmax
    steps = int(T / dt)
    save_every = max(1, steps // 2000)
    ts, xis, squash = [], [], []
    mid = N // 2
    F = forces(x, model, hw_full, idx_full, ii, jj, kap_pair)
    for s in range(steps):
        v += 0.5 * dt * F / m
        x += dt * v
        F = forces(x, model, hw_full, idx_full, ii, jj, kap_pair)
        v += 0.5 * dt * F / m
        if s % save_every == 0:
            ts.append(s * dt)
            xis.append(x / scale)
            o = max((x[mid + 1] + hw[mid + 1]) - (x[mid] - hw[mid]), 0.0)
            squash.append(o / (2 * hw[mid + 1]))
        if x[mid] / scale[mid] < 0.02:                  # hit the fixed point
            break
    return np.array(ts), np.array(xis), np.array(squash), x / scale, v / scale


if __name__ == "__main__":
    for model in ('engine', 'consistent', 'magic'):
        ts, xis, sq, xi_end, vi_end = run(model)
        verdict = 'BOUNCED (moving outward)' if np.all(vi_end > 0) else \
                  ('COLLAPSED to fixed point' if xi_end.min() < 0.05 else 'still going')
        print(f"{model:11s} t_end={ts[-1]:7.3f}  xi_mid={xi_end[N//2]:+.4f} "
              f"vi_mid={vi_end[N//2]:+.4f}  max_squash={sq.max():.3f}  -> {verdict}")
        np.savez(f"collision_{model}.npz", ts=ts, xis=xis, sq=sq, K=K, N=N)
