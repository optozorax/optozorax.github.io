"""
The energy ledger of a scaling portal under different teleport rules.

(a) A force-free ball coasting toward the fixed point.  In the unrolled plane
    this is nothing at all: uniform straight-line motion that reaches the
    fixed point at t_c = r0/v and continues.  But the pocket-space
    bookkeeping (the state your engine actually stores for "the copy at your
    scale") multiplies its kinetic energy by a fixed factor every crossing,
    and the crossings pile up like Zeno's paradox: infinitely many before
    t_c.  Under the engine rule (mass unchanged) the ledger KE REACHES
    INFINITY IN FINITE TIME.  Under the scale-consistent rule it is worse.
    Only m' = m/k^2 keeps the ledger flat -- and that rule breaks momentum
    and gravity instead.

(b) Where the phantom energy becomes real: the self-collision.  Total kinetic
    energy of the simulated copies during the head-on crush (chain of
    ../sims/collision.py, portal-periodic boundary).  The engine-model and
    scale-consistent collapses are POWERED by this books-don't-balance
    energy; the magic-stiffness bounce returns the initial energy honestly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V
import chainlib

K = 0.7
V0, R0 = 0.25, 1.0


def staircase(factor, n_max=40):
    """KE(t) of the coasting ball's pocket representative."""
    ts, ks = [0.0], [1.0]
    for n in range(1, n_max):
        t = (R0 - K ** n * R0) / V0          # n-th inward crossing
        ts += [t, t]
        ks += [ks[-1], ks[-1] * factor]
    return np.array(ts), np.array(ks)


if __name__ == "__main__":
    V.apply_style()
    tc = R0 / V0
    box = dict(fc=V.SURFACE, ec='none', alpha=0.88, pad=1.5)

    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6))
    fig.suptitle('Free energy is not a glitch, it is the engine model\'s '
                 f'bookkeeping (k = {K})', fontsize=12, y=0.99)

    ax = axs[0]
    for lbl, f, c in [('engine:  m´ = m   (KE ×1/k² per lap)', 1 / K ** 2, V.BLUE),
                      ('scale-consistent:  m´ = k² m   (KE ×1/k⁴)', 1 / K ** 4, V.AQUA),
                      ('energy rule:  m´ = m/k²   (flat ledger)', 1.0, V.YELLOW)]:
        ts, ks = staircase(f)
        ax.semilogy(ts, ks, color=c, lw=1.7, label=lbl)
    ax.axvline(tc, color=V.ORANGE, lw=1.4)
    ax.annotate('ball reaches the\nfixed point at $t_c$\n(after ∞ crossings)',
                (tc, 1e2), xytext=(tc - 1.45, 3e3), fontsize=8,
                color=V.ORANGE, bbox=box)
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0.5, 1e9)
    ax.set_xlabel('time')
    ax.set_ylabel('kinetic energy in the ledger (KE / KE₀, log)')
    ax.set_title('(a) coasting ball, no forces at all:\n'
                 'the ledger of its pocket representative')
    ax.legend(loc='upper left', fontsize=8)

    ax = axs[1]
    runs = [('engine model', dict(gamma=0, s=0), V.BLUE),
            ('scale-consistent', dict(gamma=2, s=2), V.AQUA),
            ('magic stiffness (bounces)', dict(gamma=0, s=-2), V.YELLOW)]
    for lbl, kw, c in runs:
        r = chainlib.run(K=K, N=10, NG=5, depth=4, W=0.12, V0=0.3,
                         T=12.0, nsave=1500, **kw)
        E = r['ke'] + r['pe']
        ax.semilogy(r['ts'], E / E[0], color=c, lw=1.7, label=lbl)
        print(f"{lbl:28s} verdict={r['verdict']:9s} t_end={r['ts'][-1]:6.2f} "
              f"E_end/E_0={E[-1]/E[0]:9.2f}  max_squash={100*r['max_squash']:.0f}%")
    ax.axhline(1.0, color=V.BASE, lw=0.9, ls=(0, (3, 3)))
    ax.annotate('collapse: total energy of the copies\ngrows without bound '
                'as they dive in', (0.30, 0.80), xycoords='axes fraction',
                fontsize=8, color=V.INK2, bbox=box)
    ax.annotate('bounce: energy honestly returned', (0.30, 0.10),
                xycoords='axes fraction', fontsize=8, color='#9a6a00', bbox=box)
    ax.set_xlabel('time')
    ax.set_ylabel('total energy of simulated copies, E/E₀ (log)')
    ax.set_title('(b) the self-collision: the crush is powered\n'
                 'by exactly this phantom energy')
    ax.legend(loc='center right', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('fig_ledger.png', bbox_inches='tight')
    print('saved fig_ledger.png')
    for ax, suf in zip(axs, 'ab'):
        V.save_axes(fig, ax, f'fig_ledger_{suf}.png')
