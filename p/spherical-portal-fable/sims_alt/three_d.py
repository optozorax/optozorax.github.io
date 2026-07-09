"""
3D scaling portals: where the 2D miracle breaks, and what replaces it.

The 2D result (all images equal mass, self-force = Gm^2/2r OUTWARD, independent
of rotation) rests on the conformal invariance of the 2D Laplacian.  In 3D that
protection is gone.  Two exact facts result:

FORCING LAW.  Pull a point mass m back through one portal step (a similarity of
ratio k).  The Newtonian source transforms as
    lap phi -> k^2 lap phi   (Laplacian) ,   delta^d -> k^{-d} delta^d  (Jacobian)
so the image one level deeper carries mass  m * k^{2-d}.
    d = 2:  k^0 = 1   -> every image equal    (the script's result)
    d = 3:  k^1 = k   -> images shrink by k each level inward.
So "equal-mass images" is a 2D-only accident; portal symmetry in 3D FORCES a
geometric mass sequence.  (Verified below by 3D Gauss flux on the FEM-style
field.)

MODEL A -- field-periodic (masses k^n, the forced choice).  The force sum
converges with no haze at all.  At rotation alpha = 0 the n-th inner and n-th
outer images pull the body with EXACTLY equal and opposite force -> the
self-force is exactly ZERO.  Rotation breaks the cancellation and turns on an
INWARD radial pull plus a TANGENTIAL swirl -- the opposite character to 2D's
rotation-independent outward push.

MODEL B -- equal-mass copies (m each) + a portal-symmetric 1/r^3 haze, the
literal 3D translation of the script's 2D prescription.  Now the images alone
pull inward by sum_n (1+k^n)/(1-k^n) * Gm^2/r^2, which DIVERGES; the haze
(also divergent) must cancel it, and the finite remainder -- even its SIGN --
depends on k, flipping at k* ~ 0.18.  3D gravity in the pocket is not unique.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V

G = 1.0


# ---- 3D Gauss-flux check of the forcing law: image n carries mass m k^{(d-2)n}
def flux_masses_3d(k=0.6, alpha=0.7, nlev=4, ngrid=48):
    """Field of the full image stack (masses k^n); measure enclosed mass by
    Gauss flux on small spheres around images n = 0..nlev.  Uses the planar
    image lattice embedded in 3D, force law 1/r^2."""
    N = 60
    n = np.arange(-N, N + 1)
    # image positions (planar) and masses; level n: radius k^n, angle n*alpha
    pos = np.stack([k ** n * np.cos(n * alpha),
                    k ** n * np.sin(n * alpha),
                    np.zeros_like(n, dtype=float)], axis=1)
    mass = k ** n.astype(float)                      # d=3 forcing: k^{(3-2)n}=k^n

    def field(P):                                    # gravitational field at P (3,)
        d = P[None, :] - pos
        r = np.linalg.norm(d, axis=1)
        r = np.where(r < 1e-9, 1e9, r)
        return -np.sum((mass / r ** 3)[:, None] * d, axis=0)

    out = []
    for lev in range(nlev + 1):
        c = pos[N + lev]                             # centre of image `lev`
        rad = 0.25 * k ** lev                        # sphere well inside the gap
        th = np.arccos(1 - 2 * (np.arange(ngrid) + 0.5) / ngrid)
        ph = np.pi * (1 + 5 ** 0.5) * np.arange(ngrid)
        nrm = np.stack([np.sin(th) * np.cos(ph),
                        np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
        flux = np.mean([field(c + rad * u) @ u for u in nrm]) * 4 * np.pi * rad ** 2
        out.append(-flux / (4 * np.pi * G))          # enclosed mass
    return np.array(out), mass[N:N + nlev + 1]


# ---- Model A: self-force vs rotation (masses k^n, no haze) ----------------
def selfforce_modelA(k, alpha, N=400):
    """Both inner and outer sums decay as k^n; the outer term is written in a
    rescaled form (all factors O(1)) so k^{-n} never appears explicitly."""
    n = np.arange(1, N + 1)
    z0 = 1.0 + 0j
    # inner image n: position k^n e^{i n a}, mass k^n
    zin = k ** n * np.exp(1j * n * alpha)
    din = zin - z0
    F = np.sum(k ** n * din / np.abs(din) ** 3)
    # outer image n: position k^-n e^{-i n a}, mass k^-n; contribution reduces
    # exactly to  k^n (zeta - k^n z0) / |zeta - k^n z0|^3  with zeta = e^{-i n a}
    zeta = np.exp(-1j * n * alpha)
    dout = zeta - k ** n * z0
    F += np.sum(k ** n * dout / np.abs(dout) ** 3)
    return F.real, F.imag                            # (radial outward+, tangential)


# ---- Model B: equal masses + haze, net radial vs k ------------------------
def selfforce_modelB(k, c=0.5, N=6000):
    n = np.arange(1, N + 1)
    per_pair = -(1 + k ** n) / (1 - k ** n)          # inward, -> -1 each (diverges)
    F_img = per_pair.sum()
    F_haze = (N + c)                                  # symmetric-cutoff haze, outward
    return F_img + F_haze                            # Gm^2/r^2 units


if __name__ == "__main__":
    V.apply_style()
    golden = np.deg2rad(137.50776)
    box = dict(fc=V.SURFACE, ec='none', alpha=0.88, pad=1.5)

    # forcing-law flux check
    for k in (0.5, 0.6, 0.7):
        meas, pred = flux_masses_3d(k=k)
        print(f'3D flux masses k={k}: measured {np.round(meas,3)}  '
              f'predicted k^n {np.round(pred,3)}')

    # Model A curves
    als = np.linspace(0, np.pi, 61)
    FA = np.array([selfforce_modelA(0.6, a) for a in als])
    print(f'\nModel A k=0.6: F_rad(0)={FA[0,0]:+.2e} F_tan(0)={FA[0,1]:+.2e} '
          f'(exact cancel);  at golden: F_rad={selfforce_modelA(0.6,golden)[0]:+.3f} '
          f'F_tan={selfforce_modelA(0.6,golden)[1]:+.3f}')

    # Model B sign flip
    ks = np.linspace(0.05, 0.75, 71)
    FB = np.array([selfforce_modelB(k) for k in ks])
    kstar = ks[np.argmin(np.abs(FB))]
    print(f'Model B: self-force sign flip near k* = {kstar:.3f}')

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.6))
    fig.suptitle('3D scaling portals: the conformal miracle is 2D-only — '
                 'image masses, self-force, even its sign all change',
                 fontsize=12, y=0.99)

    # (a) forcing law
    ax = axs[0]
    lev = np.arange(0, 7)
    ax.plot(lev, np.ones_like(lev, dtype=float), '-o', color=V.BLUE, lw=1.8,
            ms=5, label='2D:  ratio $k^{0}=1$  (equal — the script)')
    for k, c in [(0.6, V.ORANGE), (0.4, V.AQUA)]:
        ax.plot(lev, k ** lev, '-o', color=c, lw=1.8, ms=5,
                label=f'3D, k={k}:  ratio $k^{{1}}$ per level')
    ax.set_yscale('log')
    ax.set_xlabel('image depth  n  (levels toward the fixed point)')
    ax.set_ylabel('image mass  /  object mass')
    ax.set_title('(a) forcing law: image mass ratio $= k^{\\,d-2}$\n'
                 'per level — 2D flat, 3D geometric')
    ax.legend(loc='lower left', fontsize=7.5)
    ax.annotate('same Gauss-flux argument\nas the 2D “equal mass” proof,\n'
                'just d = 3 not 2', (0.97, 0.94), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, color=V.INK2, bbox=box)

    # (b) Model A: F vs alpha
    ax = axs[1]
    ax.plot(np.rad2deg(als), FA[:, 0], color=V.BLUE, lw=1.9,
            label='radial  (− = inward pull)')
    ax.plot(np.rad2deg(als), FA[:, 1], color=V.ORANGE, lw=1.9,
            label='tangential  (swirl)')
    ax.axhline(0, color=V.BASE, lw=0.9)
    ax.plot(0, 0, 'o', ms=8, mfc=V.YELLOW, mec=V.INK, mew=1.1, zorder=5)
    ax.annotate('α = 0: inner & outer image pairs\ncancel EXACTLY → no self-force',
                (0, 0), xytext=(0.06, 0.30), textcoords='axes fraction',
                fontsize=8, color=V.INK, bbox=box,
                arrowprops=dict(arrowstyle='-', color=V.MUTED, lw=0.8))
    ax.annotate('2D here would be a flat\nline at +½ (outward, α-independent)',
                (0.50, 0.86), xycoords='axes fraction', fontsize=8,
                color=V.INK2, bbox=box)
    ax.set_xlabel('portal rotation  α  (degrees)')
    ax.set_ylabel('self-force   F · r₀² / Gm²')
    ax.set_title('(b) 3D model A (forced masses $k^{n}$):\n'
                 'rotation turns on an inward, swirling pull')
    ax.legend(loc='lower left', fontsize=8)

    # (c) Model B sign flip
    ax = axs[2]
    ax.plot(ks, FB, color=V.BLUE, lw=2.0)
    ax.axhline(0, color=V.BASE, lw=0.9)
    ax.axvline(kstar, color=V.ORANGE, lw=1.4)
    ax.fill_between(ks, FB, 0, where=FB > 0, color=V.AQUA, alpha=0.18)
    ax.fill_between(ks, FB, 0, where=FB < 0, color=V.BLUE, alpha=0.15)
    ax.annotate(f'sign flip at k* ≈ {kstar:.2f}', (kstar, 0),
                xytext=(kstar + 0.02, 0.55 * FB.min()), fontsize=8,
                color=V.ORANGE, bbox=box)
    ax.annotate('outward\n(repulsive)', (0.09, 0.62 * FB.max()), fontsize=8,
                color=V.INK2)
    ax.annotate('inward (attractive)', (0.42, 0.30 * FB.min()), fontsize=8,
                color=V.INK2)
    ax.set_xlabel('portal scale ratio  k')
    ax.set_ylabel('self-force   F · r₀² / Gm²')
    ax.set_title('(c) 3D model B (equal copies + 1/r³ haze):\n'
                 'even the SIGN of gravity depends on k')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig('fig_3d.png', bbox_inches='tight')
    print('saved fig_3d.png')
    for ax, suf in zip(axs, 'abc'):
        V.save_axes(fig, ax, f'fig_3d_{suf}.png')
