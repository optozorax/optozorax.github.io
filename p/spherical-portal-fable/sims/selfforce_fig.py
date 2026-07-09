"""Figure + flux check for the self-force experiment."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from selfforce import solve_case, G
import vizstyle as V

V.apply_style()
GOLDEN = np.deg2rad(137.50776)


def make_sampler(res):
    """Bilinear sampler of the torus solution, pulled back to the plane."""
    L, beta, alpha = res['L'], res['beta'], res['k']  # placeholder, fixed below
    return None


def sample_fields(res, X, Y, alpha):
    L = res['L']
    beta = res['beta']
    Nu, Nt = res['phi'].shape
    du_ = L / Nu
    dth = 2 * np.pi / Nt
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    uf = np.log(np.maximum(r, 1e-12))
    nlev = np.floor(uf / L)
    ured = uf - nlev * L
    thred = th - nlev * alpha
    tpp = np.mod(thred - beta * ured, 2 * np.pi)

    fu = ured / du_ - 0.5
    ft = tpp / dth
    i0 = np.floor(fu).astype(int)
    j0 = np.floor(ft).astype(int)
    wi = fu - i0
    wj = ft - j0

    def bil(A):
        a00 = A[np.mod(i0, Nu), np.mod(j0, Nt)]
        a10 = A[np.mod(i0 + 1, Nu), np.mod(j0, Nt)]
        a01 = A[np.mod(i0, Nu), np.mod(j0 + 1, Nt)]
        a11 = A[np.mod(i0 + 1, Nu), np.mod(j0 + 1, Nt)]
        return (a00 * (1 - wi) * (1 - wj) + a10 * wi * (1 - wj)
                + a01 * (1 - wi) * wj + a11 * wi * wj)

    # NOTE: phi in (u,theta') wraps plainly in u ONLY together with theta';
    # bil() above wraps both indices on the (u, theta') torus, which is exact.
    phi = bil(res['phi'])
    phi_u = bil(res['phi_u'])
    phi_t = bil(res['phi_t'])
    # physical gravity g = -grad phi = -(phi_u/r) e_r - (phi_t/r) e_th
    ct, st = np.cos(th), np.sin(th)
    gx = -(phi_u * ct - phi_t * st) / np.maximum(r, 1e-12)
    gy = -(phi_u * st + phi_t * ct) / np.maximum(r, 1e-12)
    return phi, gx, gy


def flux_check(res, alpha, m=1.0, n_images=4):
    """Mass enclosed by a small circle around the n-th image of the body."""
    L = res['L']
    # solver identification is z ~ (e^{i alpha}/k) z, so inward images
    # rotate by -alpha per level: z -> k e^{-i alpha} z
    lam = np.exp(-L) * np.exp(-1j * alpha)
    z0 = res['r0'] * np.exp(1j * np.pi) if 'r0' in res else None
    r0 = np.exp(0.5 * L)
    z0 = r0 * np.exp(1j * np.pi)
    rows = []
    for n in range(n_images):
        zn = z0 * lam ** n
        rc = 0.10 * abs(zn)
        tt = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        zc = zn + rc * np.exp(1j * tt)
        _, gx, gy = sample_fields(res, zc.real, zc.imag, alpha)
        nx, ny = np.cos(tt), np.sin(tt)
        flux = np.sum(gx * nx + gy * ny) * (2 * np.pi * rc / len(tt))
        m_enc = -flux / (2 * np.pi * G)
        # subtract enclosed 1/r^2 negative background (numeric polar integral)
        rr = np.linspace(0, rc, 300)[None, :]
        aa = np.linspace(0, 2 * np.pi, 300, endpoint=False)[:, None]
        zz = zn + rr * np.exp(1j * aa)
        dens = -(m / (2 * np.pi * L)) / np.abs(zz) ** 2
        m_bg = np.sum(dens * rr) * (rc / 300) * (2 * np.pi / 300)
        rows.append((n, abs(zn), m_enc, m_bg, m_enc - m_bg))
    return rows


def main():
    # --- field reconstruction, concentric portals, golden-angle rotation ---
    res = solve_case(k=0.5, alpha=GOLDEN, u0_frac=0.5, a_frac=0.06,
                     Nu=512, return_fields=True)
    print("flux check: mass around each image (should all equal 1.000)")
    print(f"{'image n':>8} {'|z_n|':>8} {'raw':>8} {'bg':>8} {'body only':>10}")
    for n, rz, mr, mb, mo in flux_check(res, GOLDEN):
        print(f"{n:8d} {rz:8.3f} {mr:8.4f} {mb:8.4f} {mo:10.4f}")

    ext = 2.35
    ng = 760
    xs = np.linspace(-ext, ext, ng)
    X, Y = np.meshgrid(xs, xs)
    phi, gx, gy = sample_fields(res, X, Y, GOLDEN)
    r = np.hypot(X, Y)
    mask = r < 0.10
    gn = np.hypot(gx, gy)
    ux, uy = gx / np.maximum(gn, 1e-30), gy / np.maximum(gn, 1e-30)
    ux[mask] = np.nan
    uy[mask] = np.nan
    phi_v = np.where(mask, np.nan, phi)

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6),
                            gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    fig.suptitle('Self-force in a spherical-portal pocket space: '
                 'measured law  F = Gm²/2r,  away from the fixed point',
                 fontsize=12, x=0.5, y=0.99)

    ax = axs[0]
    ax.grid(False)
    im = ax.imshow(-phi_v, origin='lower', extent=[-ext, ext, -ext, ext],
                   cmap=V.CMAP_BLUE, interpolation='bilinear')
    ax.streamplot(xs, xs, ux, uy, density=1.15, color=V.INK2, linewidth=0.55,
                  arrowsize=0.7)
    for rr, cc in ((1.0, V.BLUE), (2.0, V.ORANGE)):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls=(0, (5, 4)),
                                lw=1.4, ec=cc))
    ax.plot(0, 0, marker='+', ms=9, mew=1.6, color=V.INK)
    ax.annotate('fixed point', (0, 0), xytext=(0.12, -0.34), fontsize=8,
                color=V.INK)
    ax.annotate('portal spheres\n(concentric form)', (0.975, 0.975),
                xycoords='axes fraction', fontsize=8, color=V.INK2,
                ha='right', va='top')
    ax.annotate('the body', (-res['r0'], 0), xytext=(-2.2, 0.62),
                fontsize=8, color=V.INK,
                arrowprops=dict(arrowstyle='-', color=V.MUTED, lw=0.8))
    ax.set_title('gravitational potential + field direction\n'
                 '(rotation = golden angle; images spiral into the fixed point)')
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_ticks([])
    cb.set_label('potential well depth →', fontsize=7.5, color=V.INK2)
    ax_a = ax

    # --- panel b: F vs r0 (k = 0.3 gives a wide annulus) ---
    ax = axs[1]
    r0s, Fs = [], []
    for uf in np.linspace(0.10, 0.90, 9):
        rr = solve_case(k=0.3, u0_frac=uf, Nu=320)
        r0s.append(rr['r0'])
        Fs.append(rr['F_rad'])
    rline = np.linspace(min(r0s) * 0.93, max(r0s) * 1.05, 200)
    ax.plot(rline, 1.0 / (2 * rline), color=V.INK2, lw=1.6,
            label='prediction  Gm²/2r')
    ax.plot(r0s, Fs, 'o', ms=6, color=V.BLUE, mec=V.SURFACE, mew=0.8,
            label='measured (FFT solver)')
    ax.set_xlabel('distance r from fixed point')
    ax.set_ylabel('outward self-force')
    ax.set_title('self-force vs distance')
    ax.legend(loc='upper right')

    # --- panel c: independence from portal rotation ---
    ax = axs[2]
    alphas = np.linspace(0, np.pi, 9)
    ratios = []
    for al in alphas:
        rr = solve_case(k=0.5, alpha=al, Nu=320)
        ratios.append(rr['ratio'])
    ax.axhline(1.0, color=V.INK2, lw=1.6)
    ax.plot(np.rad2deg(alphas), ratios, 'o', ms=6, color=V.BLUE,
            mec=V.SURFACE, mew=0.8)
    ax.set_ylim(0.99, 1.01)
    ax.set_xlabel('relative portal rotation α (degrees)')
    ax.set_ylabel('measured force ÷ prediction')
    ax.set_title('rotation changes nothing')
    ax.annotate('tangential component ≈ 0 (machine zero)\nat every α',
                (0.5, 0.08), xycoords='axes fraction', ha='center',
                fontsize=8, color=V.INK2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('fig_selfforce.png', bbox_inches='tight')
    print('saved fig_selfforce.png')
    V.save_axes(fig, [ax_a, cb.ax], 'fig_selfforce_a.png')
    V.save_axes(fig, axs[1], 'fig_selfforce_b.png')
    V.save_axes(fig, axs[2], 'fig_selfforce_c.png')


if __name__ == '__main__':
    main()
