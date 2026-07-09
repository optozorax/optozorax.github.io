# Simulations for the spherical-portal video script

Checks of three claims from `../script.txt`, run on 2026-07-08. Needs
`numpy` + `matplotlib`.

- `selfforce.py` — spectral (FFT) Poisson solver for 2D gravity on the pocket
  space (the quotient torus in log-polar coordinates, with portal rotation as
  a twist). Measures the self-force on a body and compares with the
  closed-form law **F = Gm²/(2r)**, outward from the fixed point.
  `selfforce_fig.py` renders `fig_selfforce.png` and also verifies by Gauss
  flux that every image of the body carries exactly the same mass.
- `collision.py` — head-on collision of a body with its own scaled image:
  1D chain of copies with the true portal-periodic (scale-wrap) boundary.
  Three material models; only stiffness growing inward faster than 1/k can
  bounce, so the head-on squash is universal, not an engine artifact.
  `collision_fig.py` renders `fig_collision.png`.
- `phyllo.py` — rebuilds the golden-angle scene from the script's numbers
  (R = 141 km, r = 140 km, 4 m gap): fixed-point location, image lattice,
  Fibonacci parastichies. Renders `fig_phyllotaxis.png`.
