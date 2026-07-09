# Alternative models for the spherical-portal video

Follow-up to `../sims/`. The first round verified the script's own claims;
this round asks the counterfactuals — *what if the physics were defined
differently, and is the script's choice the only consistent one?* Run on
2026-07-09. Needs `numpy` + `matplotlib`.

The one-line answer: **the mechanical results are robust (no inertial-mass rule
saves you from the crush), while the gravity results are one point in a family
(the self-force strength is a free coupling, and 3D changes everything).**

### Inertia / collisions — the rule you pick doesn't matter

- `chainlib.py` — the `../sims/collision.py` chain generalized to free
  exponents: copy *n* has mass `k^{γn}` and contact stiffness `k^{sn}`.
  `γ = 0` engine, `γ = 2` scale-consistent, `γ = -2` kinetic-energy-conserving.
- `phase_diagram.py` → `phase.npz`, `phase_fig.py` → **`fig_phase.png`**.
  Sweeps the whole `(γ, s)` plane. The collapse/bounce boundary is a **vertical
  line at `s = -1`**: survival depends only on the stiffness rule, never on the
  inertial-mass rule. No way of rescaling mass on teleport saves the object.
- `seam_twins.py` → **`fig_twins.png`**. The deeper reason. A throw and the
  same throw seen through the portal are the *same motion*; under Newtonian
  inertia they diverge (the seam becomes physically detectable) for **any** mass
  rule, because test particles carry no mass to rescale. They coincide to
  machine precision only under **scale-relative** dynamics (speed measured in
  units of your own size, `η = ż/z`) — which is exactly the rule that has no
  self-force. Also prints the conservation table: no mass rule conserves energy
  *and* momentum *and* matches gravity across a crossing.
- `energy_ledger.py` → **`fig_ledger.png`**. Where the crush energy comes from:
  a force-free coasting ball's pocket-representative gains a fixed KE factor per
  portal lap, and the laps pile up Zeno-style — the ledger diverges in finite
  time. Panel (b) shows the self-collision is powered by exactly this: total
  copy energy runs away ×10³–10⁴ in the collapsing models, stays flat in the
  bouncing one.

### Gravity — the FEM is one choice in a family

- `alt_gravity.py` → **`fig_altgravity.png`**.
  - (a) The torus source is `2πG σ e^{wu}`; only `w = 2` is Cartesian matter
    (equal-mass images). Sweeping `w` gives a family of self-consistent
    gravities — the self-force strength `F·r/Gm²` is a **dial**, and the
    script's `½` is just the `w = 2` value.
  - (b) A four-line exact derivation of `F = Gm²/2r`: each image pair pulls
    inward by exactly `Gm²/r` (any *n*, any α), the haze pushes out by
    `(N + ½)Gm²/r`; the symmetric zero-monopole cut leaves `½` — the value the
    FEM finds silently. Verified: net = `0.5000000000` at α = 0 and α = golden.
  - (c) Two hidden vacuum dials: a radial `1/r` field (an invisible point mass
    sitting *at* the fixed point) and an azimuthal `1/r` vortex — both
    portal-symmetric and source-free, so undetectable by any local mass
    measurement, yet they bend orbits. Same matter, different motion.
- `three_d.py` → **`fig_3d.png`**. The 2D story is a **2D-only accident**.
  - (a) Forcing law: pulling a point mass through one portal step multiplies its
    image mass by `k^{d-2}` (Laplacian gives `k²`, the delta-function Jacobian
    gives `k^{-d}`). `d = 2 → k⁰ = 1` (equal, the script); `d = 3 → k¹`
    (geometric). Confirmed by 3D Gauss flux: measured image masses = `kⁿ` exactly.
  - (b) 3D model A (the forced `kⁿ` masses): at α = 0 the inner/outer image pairs
    cancel **exactly** → zero self-force; rotation turns on an *inward*,
    swirling pull — the opposite character to 2D's rotation-independent outward push.
  - (c) 3D model B (equal copies + a `1/r³` haze, the literal translation of the
    script's recipe): images and haze both diverge, and the finite remainder
    flips **sign** near `k* ≈ 0.17`. In 3D even the sign of the self-force is
    not universal.
