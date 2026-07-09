"""
Self-force of a body in a recursive pocket space made of two concentric
spherical (circular, in 2D) scaling portals.

Space: the annulus quotient  z ~ lam * z,  lam = k * e^{i*alpha},  0 < k < 1.
The fixed point of the teleportation is the origin.

Coordinates: u = ln r, theta.  The quotient is the torus
    (u, theta) ~ (u + L, theta + alpha),  theta ~ theta + 2pi,   L = ln(1/k).
2D gravity:  lap phi = 2 pi G sigma  (Cartesian).  In (u, theta):
    (d_u^2 + d_th^2) phi = 2 pi G sigma * e^{2u}
so a Cartesian point mass m is a *unit-strength* point source of magnitude m
on the torus no matter how deep it sits (conformal invariance) -- this is the
script's "all images have exactly the same mass" observation.

Solvability on the torus requires zero total source  =>  the compensating
negative mass must be uniform in (u,theta)  =>  Cartesian density ~ 1/r^2,
exactly the script's empirically found prescription.

Prediction derived by hand before running this (from the conformal factor:
|z-z0| = r0*|dw|*(1 + Re(dw)/2 + ...), dw = du + i*dth):
    F_self = G m^2 / (2 r0),  pointing AWAY from the fixed point,
independent of alpha, of k, and of body size (to leading order).

This program solves the Poisson problem spectrally on the (possibly twisted)
torus and integrates the physical Cartesian force over the body.
"""
import numpy as np

G = 1.0


def solve_case(k=0.5, alpha=0.0, u0_frac=0.5, a_frac=0.05, m=1.0,
               Nu=384, theta0=np.pi, return_fields=False):
    L = np.log(1.0 / k)
    beta = alpha / L
    # grid: (u, theta') with theta' = theta - beta*u  -> plain torus (L, 2pi)
    Nt = int(np.ceil(Nu * 2 * np.pi / L / 2) * 2)
    du_ = L / Nu
    dth = 2 * np.pi / Nt
    u = (np.arange(Nu) + 0.5) * du_
    tp = (np.arange(Nt)) * dth
    U, TP = np.meshgrid(u, tp, indexing='ij')
    TH = TP + beta * U          # true polar angle theta
    R = np.exp(U)

    # body: smooth disk of radius a at Cartesian (r0, theta0)
    r0 = np.exp(u0_frac * L)
    a = a_frac * r0
    x0, y0 = r0 * np.cos(theta0), r0 * np.sin(theta0)
    X, Y = R * np.cos(TH), R * np.sin(TH)
    d = np.hypot(X - x0, Y - y0)
    sigma = np.zeros_like(d)
    core = d <= 0.8 * a
    ring = (d > 0.8 * a) & (d < a)
    sigma[core] = 1.0
    sigma[ring] = 0.5 * (1 + np.cos(np.pi * (d[ring] - 0.8 * a) / (0.2 * a)))
    cell_area = np.exp(2 * U) * du_ * dth          # Cartesian area of a cell
    Mtot = np.sum(sigma * cell_area)
    sigma *= m / Mtot                              # exact total mass m

    S = 2 * np.pi * G * sigma * np.exp(2 * U)
    S -= S.mean()                                  # uniform-on-torus negative mass
                                                   # == Cartesian density ~ 1/r^2

    # spectral solve:  ((d_u - beta d_th')^2 + d_th'^2) Phi = S
    kap = 2 * np.pi * np.fft.fftfreq(Nu, d=du_)
    mth = 2 * np.pi * np.fft.fftfreq(Nt, d=dth)
    KAP, MTH = np.meshgrid(kap, mth, indexing='ij')
    denom = (KAP - beta * MTH) ** 2 + MTH ** 2
    denom[0, 0] = 1.0
    Sh = np.fft.fft2(S)
    Ph = -Sh / denom
    Ph[0, 0] = 0.0
    Phi_u = np.real(np.fft.ifft2(1j * KAP * Ph))   # d/du at fixed theta'
    Phi_t = np.real(np.fft.ifft2(1j * MTH * Ph))   # d/dtheta'

    phi_u = Phi_u - beta * Phi_t                   # d/du at fixed theta
    phi_t = Phi_t                                  # d/dtheta

    # physical force on the body: F = -int sigma grad_cart(phi) dA
    # grad phi = (phi_u/r) e_r + (phi_t/r) e_th ;  dA = e^{2u} du dth
    w = sigma * np.exp(U) * du_ * dth
    ct, st = np.cos(TH), np.sin(TH)
    Fx = -np.sum(w * (phi_u * ct - phi_t * st))
    Fy = -np.sum(w * (phi_u * st + phi_t * ct))
    # components along/perp the outward direction at the body
    er = np.array([np.cos(theta0), np.sin(theta0)])
    et = np.array([-np.sin(theta0), np.cos(theta0)])
    F_rad = Fx * er[0] + Fy * er[1]
    F_tan = Fx * et[0] + Fy * et[1]
    pred = G * m ** 2 / (2 * r0)
    out = dict(k=k, alpha=alpha, r0=r0, a=a, Nu=Nu, Nt=Nt,
               F_rad=F_rad, F_tan=F_tan, pred=pred, ratio=F_rad / pred)
    if return_fields:
        phi = np.real(np.fft.ifft2(Ph))
        out.update(u=u, tp=tp, L=L, beta=beta, phi=phi,
                   phi_u=phi_u, phi_t=phi_t, sigma=sigma)
    return out


def report(tag, res):
    print(f"{tag:38s} k={res['k']:.2f} alpha={res['alpha']:7.4f} "
          f"r0={res['r0']:.3f} a/r0={res['a']/res['r0']:.3f} "
          f"F_rad={res['F_rad']:+.5f} F_tan={res['F_tan']:+.2e} "
          f"pred={res['pred']:.5f} ratio={res['ratio']:.4f}")


if __name__ == "__main__":
    golden = np.deg2rad(137.50776)
    print("== prediction: F = G m^2/(2 r0), outward, independent of alpha,k,a ==")
    report("baseline", solve_case())
    report("resolution/2", solve_case(Nu=192))
    report("resolution x1.5", solve_case(Nu=576))
    print("-- rotation of one portal (alpha) --")
    report("alpha=1.0 rad", solve_case(alpha=1.0))
    report("alpha=golden 137.5deg", solve_case(alpha=golden))
    report("alpha=pi", solve_case(alpha=np.pi))
    print("-- body size --")
    report("a=0.03 r0", solve_case(a_frac=0.03))
    report("a=0.08 r0", solve_case(a_frac=0.08))
    print("-- position (r0) --")
    report("u0=0.25L", solve_case(u0_frac=0.25))
    report("u0=0.75L", solve_case(u0_frac=0.75))
    print("-- portal scale ratio k --")
    report("k=0.3", solve_case(k=0.3))
    report("k=0.7", solve_case(k=0.7))
    print("-- mass scaling (F ~ m^2 ?) --")
    report("m=2", solve_case(m=2.0))
