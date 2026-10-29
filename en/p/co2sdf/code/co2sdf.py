"""Reference implementation of complex-valued codimension-2 SDFs ("co2sdf").

A co2sdf is a function F: R^n -> C (numpy complex array, vectorised over the
leading axes of p with p.shape = (..., n)).  The surface is Z(F) = {F = 0}.
Re F is the "solid" SDF inside the carrier, Im F is the SDF of the carrier.
"""
import numpy as np

# ----------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------

def jac(F, p, h=1e-5):
    """Rows a = grad Re F, b = grad Im F (central differences)."""
    n = p.shape[-1]
    a = np.empty(p.shape)
    b = np.empty(p.shape)
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        d = (F(p + e) - F(p - e)) / (2 * h)
        a[..., k] = d.real
        b[..., k] = d.imag
    return a, b


def dot(x, y):
    return (x * y).sum(-1)


def recip(a, b):
    """Reciprocal frame of (a, b): ra.a = 1, ra.b = 0, rb.a = 0, rb.b = 1."""
    aa, ab, bb = dot(a, a), dot(a, b), dot(b, b)
    det = aa * bb - ab * ab
    ra = (bb[..., None] * a - ab[..., None] * b) / det[..., None]
    rb = (aa[..., None] * b - ab[..., None] * a) / det[..., None]
    return ra, rb


def project(F, p, iters=3):
    """Gauss-Newton projection onto Z(F): p <- p - J^+ F."""
    for _ in range(iters):
        v = F(p)
        a, b = jac(F, p)
        ra, rb = recip(a, b)
        p = p - v.real[..., None] * ra - v.imag[..., None] * rb
    return p


def tangent(a, b, v):
    """Remove the normal-plane part of v (normal plane = span(a, b))."""
    ra, rb = recip(a, b)
    return v - dot(v, a)[..., None] * ra - dot(v, b)[..., None] * rb


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def correct(F, q, a, b):
    """One Newton correction of q with the gradients (a, b) of another point."""
    val = F(q)
    ra, rb = recip(a, b)
    return q - val.real[..., None] * ra - val.imag[..., None] * rb


def geodesic_step(F, p, v, dt, J=None, jet_p=None):
    """Second-order step: first correction along the normal plane at p, one
    correction with fresh normals, new direction = chord projected onto the new
    tangent plane.  Returns (q, v, jet at q); pass that jet back as jet_p."""
    J = J or (lambda x: jac(F, x))
    a, b = jet_p if jet_p is not None else J(p)
    q = correct(F, p + v * dt, a, b)
    q = correct(F, q, *J(q))
    a, b = J(q)
    return q, normalize(tangent(a, b, q - p)), (a, b)


def tangent_frame(F, p, hint1, hint2):
    """Orthonormal tangent frame at p from two hint vectors."""
    a, b = jac(F, p)
    e1 = normalize(tangent(a, b, hint1))
    t2 = tangent(a, b, hint2)
    e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    return e1, e2

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def C(x, y):
    return x + 1j * y


def u_of(p):  # complex coordinate of the (x1, x2) plane
    return p[..., 0] + 1j * p[..., 1]


def v_of(p):  # complex coordinate of the (x3, x4) plane
    return p[..., 2] + 1j * p[..., 3]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def smin_h(a, b, k):
    """Polynomial smooth min and its blend weight h (h = 1 -> a)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1 - h), h

# ----------------------------------------------------------------------------
# Booleans on co2sdfs
# ----------------------------------------------------------------------------

def union(F1, F2):
    """Hard union: keep the value whose solid (Re) is smaller."""
    return np.where(F1.real <= F2.real, F1, F2)


def intersect(F1, F2):
    return np.where(F1.real >= F2.real, F1, F2)


def subtract(F1, F2):  # F1 minus solid of F2
    G2 = -F2.real + 1j * F2.imag
    return np.where(F1.real >= G2.real, F1, G2)


def fold_union(F1, F2, k):
    """Smooth union of the solids; the carriers are mixed with the same weight."""
    d, h = smin_h(F1.real, F2.real, k)
    w = smoothstep(0.0, 1.0, h)  # C1 weight for the carrier
    return d + 1j * (F2.imag + (F1.imag - F2.imag) * w)


def cutoff(t):
    """1 for t <= 1/2, 0 for t >= 1, C2 in between."""
    return 1.0 - smootherstep(0.5, 1.0, t)


def neck_branch(F1, F2):
    """Which branch neck_union uses at a point (True: the F1 branch)."""
    return np.abs(F1) < np.abs(F2)


def neck_union_on(F1, F2, eps, k, sel):
    """neck_union with a frozen branch choice (for consistent Jacobians: the
    selected field is not smooth across |F1| = |F2| off the surface)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        A = F1 - eps * cutoff(np.abs(F2) / k) / F2
        B = F2 - eps * cutoff(np.abs(F1) / k) / F1
    return np.where(sel, A, B)


def neck_union(F1, F2, eps, k):
    """Union whose transverse crossings are replaced by necks {F1 F2 = eps}.
    Outside |F| > k the pieces are untouched.  Waist radius ~ sqrt(|eps|)."""
    r1, r2 = np.abs(F1), np.abs(F2)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = F1 - eps * cutoff(r2 / k) / F2
        B = F2 - eps * cutoff(r1 / k) / F1
    return np.where(r1 < r2, A, B)

# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------

def flat(p, n1, n2, o):
    """Codim-2 flat through o with orthonormal normals n1, n2."""
    d = p - o
    return dot(d, n1) + 1j * dot(d, n2)


def clifford_torus(p, a, b):
    return (np.hypot(p[..., 0], p[..., 1]) - a) + 1j * (np.hypot(p[..., 2], p[..., 3]) - b)


def cone(p, c, m, delta):
    """Graph v = c * u^m / |u|^(m-1) (smoothed). Total angle
    2 pi sqrt((1 + m^2 c^2) / (1 + c^2)).  m = 0 gives an ordinary cone (in R^3)."""
    u, v = u_of(p), v_of(p)
    s = np.sqrt(np.abs(u) ** 2 + delta ** 2)
    if m == 0:
        return v - c * s
    return v - c * u ** m / s ** (m - 1)


def cone_angle(c, m):
    return 2 * np.pi * np.sqrt((1 + m * m * c * c) / (1 + c * c))

# ----------------------------------------------------------------------------
# composable worlds: the names of the page's pseudocode
# ----------------------------------------------------------------------------
# A world is two ordinary SDFs, written here as one complex number
# solid + 1j * carrier: the world is the boundary of the solid, inside the
# carrier (the zero set of the carrier).  The booleans above act on the solid
# and carry the carrier along.

def ball(*xs_and_r):
    """portal_gravity's ball, in any dimension: ball(x, y, ..., r) = |(x, y, ...)| - r."""
    *xs, r = xs_and_r
    return np.sqrt(sum(x * x for x in xs)) - r


def extrude(p, d, h):
    """portal_gravity's extrude: the SDF value d extruded along the coordinate p, from -h to h."""
    wx, wy = d, np.abs(p) - h
    return np.minimum(np.maximum(wx, wy), 0.0) + np.hypot(np.maximum(wx, 0.0), np.maximum(wy, 0.0))


def smin(a, b, k):
    return smin_h(a, b, k)[0]


def capsule(p, a, b, r):
    """Distance to the segment a-b minus r, in any dimension (p: (..., n))."""
    pa, ba = p - a, b - a
    h = np.clip(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba * h[..., None], axis=-1) - r


def world(solid, carrier):
    return solid + 1j * carrier


smooth_union = fold_union


def extrude_world(A, p, h):
    """extrude(world, p, h) of the page: the solid extruded along p, the carrier unchanged."""
    return world(extrude(p, A.real, h), A.imag)


def turn(a, b, angle):
    """A carrier turned by `angle` from the carrier a towards the carrier b."""
    return np.cos(angle) * a + np.sin(angle) * b


def roots(n, W, rho, N):
    """N worlds at n = rho * (the N-th roots of W); n and W complex."""
    return rho * ((n / rho) ** N - W) / N
