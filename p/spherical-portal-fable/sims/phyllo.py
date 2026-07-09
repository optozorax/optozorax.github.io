"""
The golden-angle scene from the script, rebuilt from its stated numbers:
outer portal sphere R = 141 km, inner r = 140 km, gap between the spheres 4 m,
relative rotation = golden angle. Where is the fixed point, what lattice do
the images form, and which Fibonacci spiral families does the eye see?
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V

V.apply_style()

R_BIG = 141_000.0
R_SMALL = 140_000.0
ALPHA = np.deg2rad(137.50776)
K = R_SMALL / R_BIG
lam = K * np.exp(1j * ALPHA)

print('teleport map: shrink x{:.6f}, rotate {:.3f} deg'.format(K, np.rad2deg(ALPHA)))
for name, d in (('centers 4 m apart', 4.0),
                ('surface gap 4 m (centers 996 m apart)', R_BIG - R_SMALL - 4.0)):
    zf = d / (1 - lam)
    print(f'  if "{name}": fixed point at {abs(zf):9.3f} m from the big '
          f'sphere center  (z* = {zf.real:+.1f}{zf.imag:+.1f}i)')

# use the surface-gap interpretation for the render
d = R_BIG - R_SMALL - 4.0
zfix = d / (1 - lam)
z0 = complex(d + R_SMALL + 2.0, 0.0)         # object mid-gap, +x side
w0 = z0 - zfix
rho0 = abs(w0)
n_1m = np.log(rho0 / 1.0) / np.log(1 / K)
n_1mm = np.log(rho0 / 1e-3) / np.log(1 / K)
print(f'object sits {rho0/1000:.2f} km from the fixed point')
print(f'copies until images shrink to 1 m: {n_1m:.0f}; to 1 mm: {n_1mm:.0f}')

# which lattice neighbours are closest (log-polar metric)? -> parastichies
du_step = np.log(K)
Fs = np.arange(1, 90)
ang = np.angle(np.exp(1j * Fs * ALPHA))
dist = np.hypot(Fs * du_step, ang)
best = Fs[np.argsort(dist)[:4]]
print(f'closest lattice neighbours are n ± {sorted(best.tolist())} steps '
      f'-> the visible spiral (parastichy) families; Fibonacci: 1 2 3 5 8 13 21 34 55 89')

NPTS = 2600
n = np.arange(NPTS)
zz = w0 * lam ** n                            # image positions around the fixed point
rho = np.abs(zz)

fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.6),
                        gridspec_kw=dict(width_ratios=[1.25, 1]))
fig.suptitle('The golden-angle scene, rebuilt from the script\'s numbers '
             '(R = 141 km, r = 140 km, 4 m gap)', fontsize=12, y=0.99)

ax = axs[0]
ax.grid(False)
ax.set_aspect('equal')
lim = 1.12 * rho0 / 1000
smax = 216.0
cols = V.CMAP_BLUE(0.85 - 0.6 * n / NPTS)
ax.scatter(zz.real / 1000, zz.imag / 1000, s=smax * (rho / rho0) ** 2,
           c=cols, linewidths=0)
ax.plot(0, 0, marker='+', ms=10, mew=1.6, color=V.INK)
box = dict(fc=V.SURFACE, ec='none', alpha=0.85, pad=1.5)
ax.annotate('fixed point\n(536 m from sphere centre)', (0, 0),
            xytext=(0.02, 0.90), textcoords='axes fraction',
            fontsize=8, color=V.INK2, bbox=box)
ax.annotate('the object\n(in the 4 m gap, 140.5 km out)',
            (zz.real[0] / 1000, zz.imag[0] / 1000),
            xytext=(0.62, 0.06), textcoords='axes fraction',
            fontsize=8, color=V.INK, bbox=box,
            arrowprops=dict(arrowstyle='-', color=V.MUTED, lw=0.8))
ax.set_xlabel('km from fixed point')
ax.set_ylabel('km')
ax.set_title(f'all {NPTS} images: each next one ×{K:.5f} smaller, '
             'rotated 137.5°\n(marker size ∝ image scale, exaggerated '
             f'~10⁴×; {n_1m:.0f} copies down to 1 m)')

ax = axs[1]
NSHOW = 620
u = np.log(rho[:NSHOW])
th = np.mod(np.angle(zz[:NSHOW]), 2 * np.pi)
ax.scatter(np.rad2deg(th), u, s=7, c=V.CMAP_BLUE(0.85 - 0.6 * n[:NSHOW] / NSHOW),
           linewidths=0)
ax.set_xlabel('angle around fixed point (degrees)')
ax.set_ylabel('ln(distance to fixed point)')
ax.set_title('same images in log-polar coordinates:\n'
             'the geometric spiral becomes a perfectly even lattice')
ax.set_xlim(0, 360)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig_phyllotaxis.png', bbox_inches='tight')
print('saved fig_phyllotaxis.png')
for ax, suf in zip(axs, 'ab'):
    V.save_axes(fig, ax, f'fig_phyllotaxis_{suf}.png')
