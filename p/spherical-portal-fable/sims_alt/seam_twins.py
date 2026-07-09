"""
THE TWIN TEST: is the portal seam physically detectable?

A trajectory z(t) and its portal image lam*z(t) are the SAME physical motion
(lam = k e^{-i alpha}).  If z(t) obeys the laws of physics but lam*z(t) does
not, the laws can tell which copy is "the real one" -- the seam is detectable
and the quotient dynamics is ill-defined at crossings.

Take any force field that respects the portal symmetry (every physically
sensible field must:  g(lam z) = g(z)/conj(lam), which is what the FEM gravity
solution satisfies -- field magnitudes grow as 1/r toward the fixed point).

* NEWTONIAN dynamics  z'' = g(z):  map a solution through the portal and it
  feels a force wrong by the factor k^2.  Test particles carry no mass, so NO
  inertial-mass rule can repair this.  Twins must diverge.

* SCALE-RELATIVE dynamics (uniform Newtonian physics on the log-polar torus,
  velocity measured relative to your own scale, eta = z'/z):  the portal map
  commutes with the equations exactly.  Twins must coincide forever.

Same force field, same initial throw, both integrated in raw Cartesian
coordinates with the same RK4.  Only the definition of inertia differs.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizstyle as V

K = 0.5
ALPHA = 0.9
L = np.log(1.0 / K)
BETA = ALPHA / L
LAM = K * np.exp(-1j * ALPHA)   # inward teleport: u -> u - L, theta -> theta - alpha
A, EPS = 2.0, 0.15

T, DT = 50.0, 0.002


def pattern(z):
    """Torus force (f_u, f_theta) as complex f_u + i f_th, exactly invariant
    under (u,theta) -> (u - L, theta - alpha): the most general kind of
    'matter distribution' the pocket space allows."""
    u = np.log(np.abs(z))
    th = np.angle(z)
    chi = 2 * np.pi * u / L + th - BETA * u
    f_u = -A * np.sin(2 * np.pi * u / L) + EPS * (2 * np.pi / L - BETA) * np.sin(chi)
    f_t = EPS * np.sin(chi)
    return f_u + 1j * f_t


def g_cart(z):
    """Physical (Cartesian) force field: g = e^{i theta} (f_u + i f_th)/r.
    Satisfies g(lam z) = g(z)/conj(lam) EXACTLY."""
    return (z / np.abs(z) ** 2) * pattern(z)


def rk4(deriv, y, t_end, dt):
    n = int(t_end / dt)
    out = np.empty((n + 1,) + y.shape, dtype=complex)
    out[0] = y
    for i in range(n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    return out


def newton_deriv(y):        # y = [z, v] for both twins stacked: shape (2,2)
    z, v = y
    return np.array([v, g_cart(z)])


def torus_deriv(y):         # y = [z, eta],  z' = z eta,  eta' = f(u,theta)
    z, eta = y
    return np.array([z * eta, pattern(z)])


if __name__ == "__main__":
    V.apply_style()
    z0 = np.exp(L / 4) * np.exp(0j)      # r = e^{L/4}, theta = 0
    eta0 = 0.45j                         # thetadot = 0.45, udot = 0
    v0 = z0 * eta0                       # same physical initial velocity

    # --- Newtonian twins: (z0, v0) and (lam z0, lam v0)
    yN = np.array([[z0, LAM * z0], [v0, LAM * v0]])
    trN = rk4(newton_deriv, yN, T, DT)
    zA_n, zB_n = trN[:, 0, 0], trN[:, 0, 1]

    # --- scale-relative twins: (z0, eta0) and (lam z0, eta0)
    yT = np.array([[z0, LAM * z0], [eta0, eta0]])
    trT = rk4(torus_deriv, yT, T, DT)
    zA_t, zB_t = trT[:, 0, 0], trT[:, 0, 1]

    tt = np.linspace(0, T, len(zA_n))
    D_n = np.abs(zB_n - LAM * zA_n) / (K * np.abs(zA_n))
    D_t = np.abs(zB_t - LAM * zA_t) / (K * np.abs(zA_t))
    print(f"Newtonian twins:      relative split after t={T}:  {D_n[-1]:.3f} "
          f"(first exceeds 10% at t={tt[np.argmax(D_n > 0.1)]:.1f})")
    print(f"scale-relative twins: relative split after t={T}:  {D_t.max():.2e} "
          f"(machine precision)")

    # --- conservation bookkeeping table for mass rules (analytic) ----------
    print("\nWhat one inward crossing (come out k times smaller, v -> k v "
          "forced by continuity) does to a ball of mass m:")
    rows = [("engine (his): m' = m", 1.0),
            ("2D scale-consistent: m' = k^2 m", K ** 2),
            ("momentum-conserving: m' = m/k", 1 / K),
            ("energy-conserving: m' = m/k^2", 1 / K ** 2)]
    print(f"{'rule':38s}{'KE factor':>12s}{'p factor':>12s}{'grav mass':>12s}")
    for name, chi in rows:
        print(f"{name:38s}{chi * K**2:12.3f}{chi * K:12.3f}{chi:12.3f}")
    print("-> conserving BOTH needs chi k^2 = chi k = 1, impossible for k<1;")
    print("   and 2D gravity needs chi = 1 (equal-mass images), a third demand.")
    print("   Scale-relative dynamics keeps v/r instead: E, p (local) and")
    print("   gravity all consistent at once - at the price of the self-force.")

    # ---------------------------- figure ----------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.5),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    fig.suptitle('The twin test: throw a ball, and watch the same throw through the portal '
                 f'(k = {K}, rotation {ALPHA} rad; identical portal-symmetric force field)',
                 fontsize=12, y=0.99)
    box = dict(fc=V.SURFACE, ec='none', alpha=0.85, pad=1.5)

    nsh = int(0.55 * len(zA_n))          # show the first ~27 time units
    ax = axs[0]
    ax.set_aspect('equal')
    ax.plot((LAM * zA_n)[:nsh].real, (LAM * zA_n)[:nsh].imag, color=V.INK, lw=1.1,
            label='original throw, mapped through portal')
    ax.plot(zB_n[:nsh].real, zB_n[:nsh].imag, color=V.BLUE, lw=1.1,
            label='the twin throw, evolved by physics')
    ax.plot(0, 0, '+', color=V.INK, ms=9, mew=1.5)
    ax.annotate('fixed point', (0, 0), xytext=(4, 4), textcoords='offset points',
                fontsize=8, color=V.INK2)
    ax.set_title('Newtonian inertia ($m\\ddot{z} = F$):\nthe curves split '
                 '— the seam is detectable')
    ax.legend(loc='upper left', fontsize=7.5)

    ax = axs[1]
    ax.set_aspect('equal')
    ax.plot((LAM * zA_t)[:nsh].real, (LAM * zA_t)[:nsh].imag, color=V.INK, lw=1.6,
            label='original, mapped')
    ax.plot(zB_t[:nsh].real, zB_t[:nsh].imag, color=V.AQUA, lw=0.9, ls=(0, (4, 3)),
            label='twin, evolved')
    ax.plot(0, 0, '+', color=V.INK, ms=9, mew=1.5)
    ax.set_title('scale-relative inertia (speed measured\nin units of your '
                 'own size): exact overlap')
    ax.legend(loc='upper left', fontsize=7.5)

    ax = axs[2]
    ax.semilogy(tt, np.maximum(D_n, 1e-18), color=V.BLUE, lw=1.8,
                label='Newtonian dynamics')
    ax.semilogy(tt, np.maximum(D_t, 1e-18), color=V.AQUA, lw=1.8,
                label='scale-relative dynamics')
    ax.axhline(1.0, color=V.BASE, lw=0.9, ls=(0, (3, 3)))
    ax.annotate('twins fully separated', (0.02, 0.90), xycoords='axes fraction',
                fontsize=8, color=V.INK2, bbox=box)
    ax.annotate('floating-point noise', (0.55, 0.13), xycoords='axes fraction',
                fontsize=8, color=V.INK2, bbox=box)
    ax.set_xlabel('time')
    ax.set_ylabel('relative separation of the twins')
    ax.set_ylim(1e-17, 30)
    ax.set_title('divergence of the twins\n(test particles: no mass rule can fix this)')
    ax.legend(loc='center right', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig('fig_twins.png', bbox_inches='tight')
    print('saved fig_twins.png')
    for ax, suf in zip(axs, 'abc'):
        V.save_axes(fig, ax, f'fig_twins_{suf}.png')
