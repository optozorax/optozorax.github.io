"""Figure for the head-on self-collision experiment (loads collision_*.npz)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V

V.apply_style()

MODELS = [
    ('engine', V.BLUE, 'engine model\n(mass unchanged by teleport)'),
    ('consistent', V.AQUA, 'scale-consistent model\n(mass ∝ area, stiffness scaled)'),
    ('magic', V.YELLOW, '"magic stiffness"\n(smaller copies stiffer ×1/k²)'),
]

data = {m: np.load(f'collision_{m}.npz') for m, _, _ in MODELS}
K = float(data['engine']['K'])
N = int(data['engine']['N'])

fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))
fig.suptitle('Head-on collision with your own smaller self: the stack always wins '
             '(1D chain, portal-periodic boundary)', fontsize=12, y=0.99)

# (a) worldlines of every copy, engine model
ax = axs[0]
d = data['engine']
ts, xis = d['ts'], d['xis']
scale = K ** np.arange(N)
cmap = V.CMAP_BLUE
for n in range(N):
    c = cmap(0.25 + 0.55 * n / (N - 1))
    ax.plot(ts, xis[:, n] * scale[n], color=c, lw=1.4)
ax.set_yscale('log')
ax.set_xlabel('time')
ax.set_ylabel('distance from fixed point (log)')
ax.set_title('engine model: all copies dive into\nthe fixed point in finite time')
ax.annotate('copy 0 (the object)', (ts[len(ts)//3], xis[len(ts)//3, 0] * 1.25),
            fontsize=8, color=V.INK)
ax.annotate('copy 9 (deepest simulated image)',
            (ts[len(ts)//4], xis[len(ts)//4, N-1] * scale[N-1] * 0.6),
            fontsize=8, color=V.INK2, va='top')
ax.annotate('contact\nbegins', (1.07, 2.2e-3), fontsize=8, color=V.INK2,
            ha='center')
ax.axvline(1.07, color=V.BASE, lw=0.9, ls=(0, (3, 3)))

# (b) squash of the mid copy
ax = axs[1]
for m, c, lbl in MODELS:
    d = data[m]
    ax.plot(d['ts'], 100 * d['sq'], color=c, lw=1.8, label=lbl.split('\n')[0])
ax.axhline(100, color=V.BASE, lw=0.9, ls=(0, (3, 3)))
ax.annotate('100% = fully crushed', (0.03, 0.92), xycoords='axes fraction',
            fontsize=8, color=V.INK2)
ax.set_xlim(0, 3.0)
ax.set_xlabel('time')
ax.set_ylabel('overlap ÷ copy diameter (%)')
ax.set_title('how squashed the object gets')
ax.legend(loc='center right')

# (c) rescaled position of the middle copy
ax = axs[2]
for m, c, lbl in MODELS:
    d = data[m]
    ax.plot(d['ts'], d['xis'][:, N // 2], color=c, lw=1.8)
ax.annotate('"magic stiffness":\nthe only bounce', (7.5, 1.85), fontsize=8,
            color='#9a6a00', ha='center')
ax.annotate('engine & scale-consistent:\ncollapse to the fixed point',
            (2.4, 0.30), fontsize=8, color=V.INK2, ha='left')
ax.axhline(0, color=V.BASE, lw=0.9)
ax.annotate('fixed point', (0.985, 0.045), xycoords='axes fraction',
            fontsize=8, color=V.INK2, ha='right')
ax.set_xlabel('time')
ax.set_ylabel('position (portal-rescaled units)')
ax.set_title('collapse vs bounce:\nonly unphysical stiffness bounces')

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('fig_collision.png', bbox_inches='tight')
print('saved fig_collision.png')
for ax, suf in zip(axs, 'abc'):
    V.save_axes(fig, ax, f'fig_collision_{suf}.png')
