"""Figure for the teleport-rule phase diagram (loads phase.npz)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import vizstyle as V

V.apply_style()
d = np.load('phase.npz')
sq, gammas, ss, K = 100 * d['sq'], d['gammas'], d['ss'], float(d['K'])

fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6),
                        gridspec_kw=dict(width_ratios=[1.3, 1]))
fig.suptitle('Does ANY inertial-mass rule save you from your own collision? '
             'No — survival depends only on the stiffness rule '
             f'(1D chain, k = {K})', fontsize=12, y=0.99)

ax = axs[0]
ds = ss[1] - ss[0]
dg = gammas[1] - gammas[0]
ext = [ss[0] - ds / 2, ss[-1] + ds / 2, gammas[0] - dg / 2, gammas[-1] + dg / 2]
im = ax.imshow(np.clip(sq, 0, 130), origin='lower', aspect='auto', extent=ext,
               cmap=V.CMAP_BLUE, vmin=0, vmax=130)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label('worst squash of the object (%)')
ax.axvline(-1.0, color=V.ORANGE, lw=2.0)
ax.annotate('predicted boundary s = −1\n(stiffness must grow inward\nfaster than 1/k '
            'to bounce)', (-1.0, gammas[-1]), xytext=(-2.45, 1.35), fontsize=8,
            color=V.ORANGE,
            bbox=dict(fc=V.SURFACE, ec='none', alpha=0.9, pad=1.5))
marks = [(0, 0, 'engine model', (0.35, -0.55)),
         (2, 2, '“same material”,\nscale-consistent', (0.55, 1.0)),
         (0, -2, 'kinetic-energy-conserving\nmass rule', (0.35, -1.8)),
         (-2, 0, 'the “magic” material', (-2.45, -0.75))]
for s_, g_, lbl, xy in marks:
    ax.plot(s_, g_, 'o', ms=7, mfc=V.YELLOW, mec=V.INK, mew=1.0, zorder=5)
    ax.annotate(lbl, (s_, g_), xytext=xy, fontsize=8, color=V.INK,
                bbox=dict(fc=V.SURFACE, ec='none', alpha=0.85, pad=1.5),
                arrowprops=dict(arrowstyle='-', color=V.MUTED, lw=0.8))
ax.set_xlabel('stiffness rule  s   (contact stiffness of copy n  ∝ $k^{sn}$)')
ax.set_ylabel('inertial-mass rule  γ   (mass of copy n  ∝ $k^{γn}$)')
ax.set_title('outcome map: the collapse/bounce boundary is vertical —\n'
             'the inertial-mass rule γ is irrelevant to the verdict')

ax = axs[1]
cols = V.CMAP_BLUE(np.linspace(0.35, 0.9, len(gammas)))
for i, g in enumerate(gammas):
    ax.plot(ss, np.clip(sq[i], 0, 108), color=cols[i], lw=1.6,
            label=f'γ = {g:+.0f}')
ax.axvline(-1.0, color=V.ORANGE, lw=1.6)
ax.axhline(100, color=V.BASE, lw=0.9, ls=(0, (3, 3)))
ax.annotate('above 100% = fully crushed (clipped)', (0.03, 0.9),
            xycoords='axes fraction', fontsize=8, color=V.INK2)
ax.set_xlabel('stiffness rule  s')
ax.set_ylabel('worst squash (%)')
ax.set_title('all five mass rules give the same curve shape:\n'
             'squash blows up crossing s = −1')
ax.legend(loc='center right', fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('fig_phase.png', bbox_inches='tight')
print('saved fig_phase.png')
V.save_axes(fig, [axs[0], cb.ax], 'fig_phase_a.png')
V.save_axes(fig, axs[1], 'fig_phase_b.png')
