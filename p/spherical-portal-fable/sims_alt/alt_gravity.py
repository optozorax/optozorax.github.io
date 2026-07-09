"""
Is the script's FEM gravity the only way to do gravity in the pocket space?

Three answers in one experiment:

(a) THE SOURCE-WEIGHT KNOB.  On the torus the Poisson source is
        S = 2 pi G sigma e^{w u},
    and only w = 2 is "Cartesian matter" (a blob means the same Cartesian
    mass at every depth -> all images equal).  Every other w is a
    self-consistent alternative gravity in which an image n levels deeper
    carries source k^{n(w-2)}.  We sweep w and measure the self-force:
    the script's law F = G m^2/(2 r) is just the w = 2 point of a whole
    family - and torus-native matter (w = 0... measured here) says otherwise.

(b) THE BATTLE OF INFINITIES - a 4-line exact derivation of F = Gm^2/2r.
    Force of ALL images on the body (2D pair force G m1 m2 / d): the n-th
    inner + n-th outer image pull the body INWARD by exactly G m^2/r0 per
    pair, for EVERY n and EVERY rotation alpha (verified numerically below).
    The negative haze below the body repels it outward by G m^2 (N + c)/r0
    when you cut both sums at the same log-window (c = the fraction of a lap
    of haze beyond the last image).  Net: (N + c) - N = c.  The symmetric,
    zero-monopole choice - the one the FEM makes silently - is c = 1/2:
        F = G m^2 / (2 r0),  outward.  QED.

(c) THE HIDDEN DIALS.  The vacuum freedom on the torus is exactly two
    harmonic 1-forms: a radial 1/r field (= invisible point mass sitting AT
    the fixed point, i.e. outside the pocket space itself) and an azimuthal
    1/r vortex (a "spin" of the fixed point) - portal-symmetric,
    source-free, undetectable by any local mass measurement.  Same matter,
    different orbits.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V

G = 1.0


def solve_case_w(k=0.5, alpha=0.0, w=2.0, u0_frac=0.5, a_frac=0.05, m=1.0,
                 Nu=256, theta0=np.pi):
    """FFT Poisson solve on the twisted torus with source weight w
    (w = 2 reproduces ../sims/selfforce.py exactly)."""
    L = np.log(1.0 / k)
    beta = alpha / L
    Nt = int(np.ceil(Nu * 2 * np.pi / L / 2) * 2)
    du_, dth = L / Nu, 2 * np.pi / Nt
    u = (np.arange(Nu) + 0.5) * du_
    tp = np.arange(Nt) * dth
    U, TP = np.meshgrid(u, tp, indexing='ij')
    TH = TP + beta * U
    R = np.exp(U)
    r0 = np.exp(u0_frac * L)
    a = a_frac * r0
    d = np.hypot(R * np.cos(TH) - r0 * np.cos(theta0),
                 R * np.sin(TH) - r0 * np.sin(theta0))
    sigma = np.zeros_like(d)
    sigma[d <= 0.8 * a] = 1.0
    ring = (d > 0.8 * a) & (d < a)
    sigma[ring] = 0.5 * (1 + np.cos(np.pi * (d[ring] - 0.8 * a) / (0.2 * a)))
    sigma *= m / np.sum(sigma * np.exp(2 * U) * du_ * dth)

    S = 2 * np.pi * G * sigma * np.exp(w * U)
    S -= S.mean()
    kap = 2 * np.pi * np.fft.fftfreq(Nu, d=du_)
    mth = 2 * np.pi * np.fft.fftfreq(Nt, d=dth)
    KAP, MTH = np.meshgrid(kap, mth, indexing='ij')
    denom = (KAP - beta * MTH) ** 2 + MTH ** 2
    denom[0, 0] = 1.0
    Ph = -np.fft.fft2(S) / denom
    Ph[0, 0] = 0.0
    phi_u = np.real(np.fft.ifft2(1j * KAP * Ph)) - beta * np.real(np.fft.ifft2(1j * MTH * Ph))
    phi_t = np.real(np.fft.ifft2(1j * MTH * Ph))
    wgt = sigma * np.exp(U) * du_ * dth
    ct, st = np.cos(TH), np.sin(TH)
    Fx = -np.sum(wgt * (phi_u * ct - phi_t * st))
    Fy = -np.sum(wgt * (phi_u * st + phi_t * ct))
    er = np.array([np.cos(theta0), np.sin(theta0)])
    return (Fx * er[0] + Fy * er[1]) * r0 / (G * m ** 2)   # F_rad in units Gm^2/r0


def battle(N, c, k, alpha):
    """Images summed pair by pair + haze cut c laps past the last image."""
    n = np.arange(1, N + 1)
    zi = k ** n * np.exp(1j * n * alpha)       # inner images (units of r0)
    zo = k ** -n * np.exp(-1j * n * alpha)     # outer images
    F_img = np.sum(((zi - 1) / np.abs(zi - 1) ** 2).real
                   + ((zo - 1) / np.abs(zo - 1) ** 2).real)
    F_haze = N + c                             # enclosed haze mass, outward
    return F_img, F_haze, F_img + F_haze       # units G m^2 / r0


def orbit(c1, c2, z0, v0, T=60.0, dt=0.002, rmin=0.03):
    """Newtonian orbit in pure vacuum-hair field g = (-c1 + i c2) z/|z|^2.
    The vortex part is curl-free but its potential is multivalued (-c2*theta):
    every lap around the fixed point does net work 2 pi c2 on the orbit."""
    n = int(T / dt)
    zs = [complex(z0)]
    y = np.array([z0, v0], dtype=complex)

    def f(y):
        z, v = y
        return np.array([v, (-c1 + 1j * c2) * z / np.abs(z) ** 2])
    for i in range(n):
        k1 = f(y); k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if np.abs(y[0]) < rmin:
            break
        zs.append(complex(y[0]))
    return np.array(zs)


if __name__ == "__main__":
    V.apply_style()
    box = dict(fc=V.SURFACE, ec='none', alpha=0.88, pad=1.5)
    golden = np.deg2rad(137.50776)

    # (a) weight sweep
    ws = np.linspace(0.0, 3.0, 13)
    Fw = np.array([solve_case_w(w=w) for w in ws])
    print('source weight w -> F * r0 / Gm^2 :')
    for w, f in zip(ws, Fw):
        print(f'  w = {w:4.2f}   F = {f:+.4f}')

    # (b) battle of infinities
    Ns = np.arange(1, 26)
    tot0 = [battle(N, 0.5, 0.5, 0.0) for N in Ns]
    totg = [battle(N, 0.5, 0.5, golden) for N in Ns]
    img0 = np.array([t[0] for t in tot0])
    hz0 = np.array([t[1] for t in tot0])
    net0 = np.array([t[2] for t in tot0])
    netg = np.array([t[2] for t in totg])
    print(f'\nbattle of infinities, k=0.5: net force with c=1/2:'
          f'  alpha=0: {net0[-1]:.12f}   alpha=golden: {netg[-1]:.12f}')

    # (c) hair orbits
    zs_mass = orbit(1.0, 0.0, 1.0 + 0j, 0.75j)
    zs_vort = orbit(1.0, -0.15, 1.0 + 0j, 0.75j)

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.6),
                            gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    fig.suptitle('Is the FEM the only possible gravity here? The unique choices it makes, '
                 'and the freedom it hides', fontsize=12, y=0.99)

    ax = axs[0]
    ax.plot(ws, Fw, '-', color=V.BLUE, lw=1.8, zorder=3)
    ax.plot(ws, Fw, 'o', ms=4.5, mfc=V.BLUE, mec='none', zorder=4)
    ax.plot([2], [0.5], 'o', ms=9, mfc=V.YELLOW, mec=V.INK, mew=1.1, zorder=5)
    ax.annotate('your FEM: w = 2, the one\nchoice where all images weigh\n'
                'the same → F = Gm²/2r', (2, 0.5), xytext=(0.6, 0.62),
                fontsize=8, color=V.INK, bbox=box,
                arrowprops=dict(arrowstyle='-', color=V.MUTED, lw=0.8))
    ax.axhline(0, color=V.BASE, lw=0.9)
    ax.set_xlabel('matter coupling  w   (torus source ∝ $e^{wu}$)')
    ax.set_ylabel('self-force   F · r₀ / Gm²')
    ax.set_title('(a) a whole family of consistent gravities:\n'
                 'the self-force strength is a knob')

    ax = axs[1]
    ax.plot(Ns, img0, color=V.ORANGE, lw=1.7, label='all images: pull inward, −N·Gm²/r')
    ax.plot(Ns, hz0, color=V.AQUA, lw=1.7, label='negative haze: push outward, +(N+½)·Gm²/r')
    ax.plot(Ns, net0, color=V.BLUE, lw=2.2, label='sum: exactly ½ · Gm²/r  for every N')
    ax.plot(Ns[::3], netg[::3], 's', ms=5, mfc='none', mec=V.INK, mew=1.0,
            label='same with rotation α = 137.5°')
    ax.axhline(0, color=V.BASE, lw=0.9)
    ax.annotate('each image pair (n-th inner + n-th outer)\npulls inward by '
                'EXACTLY Gm²/r, any n, any α', (0.05, 0.30),
                xycoords='axes fraction', fontsize=8, color=V.INK2, bbox=box)
    ax.set_xlabel('image pairs included, N')
    ax.set_ylabel('force on the body   (Gm²/r₀)')
    ax.set_title('(b) the battle of infinities:\nyour law F = Gm²/2r in four lines')
    ax.legend(loc='upper left', fontsize=7.5)

    ax = axs[2]
    ax.set_aspect('equal')
    ax.plot(zs_mass.real, zs_mass.imag, color=V.BLUE, lw=0.9,
            label='invisible mass at the fixed point')
    ax.plot(zs_vort.real, zs_vort.imag, color=V.ORANGE, lw=0.9,
            label='+ vortex hair: each lap does work,\nthe orbit spirals into the fixed point')
    ax.plot(0, 0, '+', color=V.INK, ms=10, mew=1.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(c) the two hidden vacuum dials:\nsame matter, different orbits')
    ax.legend(loc='upper left', fontsize=7.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('fig_altgravity.png', bbox_inches='tight')
    print('saved fig_altgravity.png')
    for ax, suf in zip(axs, 'abc'):
        V.save_axes(fig, ax, f'fig_altgravity_{suf}.png')
