"""Demo scenes: 2D worlds embedded in R^4 as zero sets of complex SDFs."""
import numpy as np
from co2sdf import *

# ---------------------------------------------------------------- colours
GREEN = np.array([0.30, 0.75, 0.25])
ORANGE = np.array([0.95, 0.55, 0.15])
BLUE = np.array([0.25, 0.50, 0.95])
PURPLE = np.array([0.70, 0.35, 0.85])
GREY = np.array([0.75, 0.75, 0.75])
WORLD_COLORS = [GREEN, ORANGE, BLUE, PURPLE]


def checker(x, y, size=1.0):
    return ((np.floor(x / size) + np.floor(y / size)) % 2).astype(float)


def shade(base, x, y, size=1.0, lines=True):
    """Checkerboard + thin grid lines + chiral 'L' marks so mirroring is visible."""
    c = checker(x, y, size)
    col = base[None, :] * (0.78 + 0.22 * c[:, None])
    fx = x / size - np.floor(x / size)
    fy = y / size - np.floor(y / size)
    if lines:
        g = (np.minimum(fx, 1 - fx) < 0.03) | (np.minimum(fy, 1 - fy) < 0.03)
        col[g] *= 0.55
    # chiral L glyph in every cell: vertical bar at left + foot at the bottom
    L = ((fx > 0.25) & (fx < 0.35) & (fy > 0.25) & (fy < 0.75)) | \
        ((fx > 0.25) & (fx < 0.6) & (fy > 0.25) & (fy < 0.35))
    col[L] = col[L] * 0.25 + 0.75 * np.array([1.0, 1.0, 1.0])
    return col


class Scene:
    name = "scene"
    cam = np.array([0.0, 0.0, 0.0, 0.0])
    hint1 = np.array([1.0, 0.0, 0.0, 0.0])
    hint2 = np.array([0.0, 1.0, 0.0, 0.0])

    def F(self, p):
        raise NotImplementedError

    def color(self, p):
        raise NotImplementedError

# ---------------------------------------------------------------- scenes

class Plane(Scene):
    name = "plane"
    cam = np.array([0.2, 0.3, 0.0, 0.0])

    def F(self, p):
        return v_of(p)

    def color(self, p):
        return shade(GREEN, p[:, 0], p[:, 1])


class Neck(Scene):
    """Two perpendicular planes touching at the origin; neck union turns the
    touching point into a wormhole: {u v = eps} near the origin."""
    name = "neck"
    eps, k = 0.36, 2.4
    cam = np.array([-3.0, 0.0, 0.0, 0.0])

    def F(self, p):
        return neck_union(v_of(p), u_of(p), self.eps, self.k)

    def jac(self, p):
        sel = neck_branch(v_of(p), u_of(p))
        return jac(lambda q: neck_union_on(v_of(q), u_of(q), self.eps, self.k, sel), p)

    def color(self, p):
        u, v = u_of(p), v_of(p)
        inA = np.abs(v) <= np.abs(u)
        cA = shade(GREEN, p[:, 0], p[:, 1])
        cB = shade(ORANGE, p[:, 2], p[:, 3])
        return np.where(inA[:, None], cA, cB)


class ThreeNecks(Scene):
    """Three planes that pairwise touch at single points (three complex lines
    v=0, u=0, u+v=c).  Neck unions at the three touching points."""
    name = "three_necks"
    c, eps, k = 3.0, 0.3, 1.4
    cam = np.array([1.5, -1.8, 0.0, 0.0])

    def parts(self, p):
        u, v = u_of(p), v_of(p)
        return v, u, (u + v - self.c) / np.sqrt(2)

    def F(self, p):
        F1, F2, F3 = self.parts(p)
        return neck_union(neck_union(F1, F2, self.eps, self.k), F3, self.eps, self.k)

    def jac(self, p):
        F1, F2, F3 = self.parts(p)
        s1 = neck_branch(F1, F2)
        s2 = neck_branch(neck_union_on(F1, F2, self.eps, self.k, s1), F3)
        def locked(q):
            G1, G2, G3 = self.parts(q)
            return neck_union_on(neck_union_on(G1, G2, self.eps, self.k, s1), G3, self.eps, self.k, s2)
        return jac(locked, p)

    def color(self, p):
        F1, F2, F3 = self.parts(p)
        d = np.stack([np.abs(F1), np.abs(F2), np.abs(F3)], -1)
        w = np.argmin(d, -1)
        u, v = u_of(p), v_of(p)
        c1 = shade(GREEN, u.real, u.imag)
        c2 = shade(ORANGE, v.real, v.imag)
        # world 3 coordinates: along the line u+v=c, use (u - v)/sqrt2
        s = (u - v) / np.sqrt(2)
        c3 = shade(BLUE, s.real, s.imag)
        return np.where((w == 0)[:, None], c1, np.where((w == 1)[:, None], c2, c3))


class Cone(Scene):
    """Graph v = c u^m / |u|^(m-1), smoothed tip."""
    name = "cone"
    c, m, delta = 0.845, 2, 0.35
    cam = np.array([-3.0, 0.0, 0.0, 0.0])

    def F(self, p):
        return cone(p, self.c, self.m, self.delta)

    def color(self, p):
        return shade(GREEN, p[:, 0], p[:, 1])


class Clifford(Scene):
    name = "clifford"
    a, b = 1.0, 0.7
    cam = np.array([1.0, 0.0, 0.7, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([0.0, 0.0, 0.0, 1.0])

    def F(self, p):
        return clifford_torus(p, self.a, self.b)

    def color(self, p):
        s = self.a * np.arctan2(p[:, 1], p[:, 0])
        t = self.b * np.arctan2(p[:, 3], p[:, 2])
        # tiles that exactly fit the two circumferences
        col = shade(GREEN, s * 8 / (2 * np.pi * self.a), t * 6 / (2 * np.pi * self.b), 1.0)
        th, ph = np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 3], p[:, 2])
        dth = np.angle(np.exp(1j * (th - 2.0))) * self.a
        dph = np.angle(np.exp(1j * (ph - 1.0))) * self.b
        col[np.hypot(dth, dph) < 0.35] = np.array([0.9, 0.15, 0.15])
        return col


class Portal(Scene):
    """N parallel worlds at zeta = rho * omega^k (zeta = x3 + i x4).
    Surface: zeta^N = rho^N W(x), W = (|x|^2 - R^2 + 2iR x2)/(|x|^2 + R^2).
    The segment x2 = 0, |x1| < R is a branch cut = a portal."""
    name = "portal"
    N, rho, R = 2, 1.0, 2.0
    cam = np.array([0.0, -2.5, 1.0, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([-1.0, 0.0, 0.0, 0.0])

    def W(self, p):
        x1, x2 = p[..., 0], p[..., 1]
        r2 = x1 * x1 + x2 * x2
        return (r2 - self.R ** 2 + 2j * self.R * x2) / (r2 + self.R ** 2)

    def F(self, p):
        return roots(v_of(p), self.W(p), self.rho, self.N)

    def world_index(self, p):
        # sheet label: which root of W the point is on (principal branch)
        z = v_of(p)
        base = self.rho * np.exp(1j * np.angle(self.W(p)) / self.N) * np.abs(self.W(p)) ** (1 / self.N)
        ks = np.arange(self.N)
        cand = base[..., None] * np.exp(2j * np.pi * ks / self.N)
        return np.argmin(np.abs(cand - z[..., None]), -1)

    def color(self, p):
        w = self.world_index(p)
        out = np.zeros(p.shape[:-1] + (3,))
        for k in range(self.N):
            ck = shade(WORLD_COLORS[k], p[:, 0], p[:, 1])
            out = np.where((w == k)[:, None], ck, out)
        # draw the portal frame (the branch points) as dark dots
        x1, x2 = p[:, 0], p[:, 1]
        near = np.minimum(np.hypot(x1 - self.R, x2), np.hypot(x1 + self.R, x2)) < 0.08
        out[near] = 0.1
        return out


class Portal3(Portal):
    name = "portal3"
    N, rho, R = 3, 1.0, 2.0


class Handle(Scene):
    """Plane z = w = 0 with a tube from the mouth x = +R to the mouth x = -R: the
    upper half of a 4D donut around the circle |xz| = R (its lower half is
    hidden in the plane's solid).  twist = pi turns the tube's circle over
    through w on the way: the mirror handle."""
    name = "handle"
    R, r, k = 1.5, 0.5, 0.4
    twist = 0.0
    r_max = 10.0
    cam = np.array([-1.5, -2.0, 0.0, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([-1.0, 0.0, 0.0, 0.0])

    def handle(self, p):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        solid = ball(ball(x, z, self.R), y, w, self.r)              # a donut around the circle |xz| = R, in 4D
        along = np.arctan2(np.maximum(z, 0.0), x)                     # 0 at the mouth x = +R, pi at x = -R
        angle = self.twist * smootherstep(0.25 * np.pi, 0.75 * np.pi, along)
        return world(solid, turn(w, y, angle))

    def plane(self, p, flip=None):
        """The plane; with a twist its carrier is turned over (-w) where the second mouth arrives."""
        x, z, w = p[..., 0], p[..., 2], p[..., 3]
        if flip is None:
            flip = x < 0
        return world(z, np.where(flip & (self.twist != 0), -w, w))

    def F(self, p):
        return smooth_union(self.plane(p), self.handle(p), self.k)

    def jac(self, p):
        # the plane's carrier jumps at x = 0 (mirror handle): differentiate the side p is on
        flip = p[..., 0] < 0
        return jac(lambda q: smooth_union(self.plane(q, flip), self.handle(q), self.k), p)

    def color(self, p):
        left = shade(GREEN, p[:, 0], p[:, 1], 0.6)
        right = shade(np.array([0.2, 0.7, 0.7]), p[:, 0], p[:, 1], 0.6)
        return np.where((p[:, 0] < 0)[:, None], left, right)


class Klein(Handle):
    name = "klein"
    twist = np.pi


class ThreeParallel(Scene):
    """Three parallel planes R^2 x {c_k}; every pair joined by a tube in the
    3-space that contains both planes (an ordinary codim-1 wormhole there);
    the three pair-constructions are combined with the hard (Re-min) union."""
    name = "three_parallel"
    L, r, k = 2.4, 0.55, 0.35
    q = {(0, 1): np.array([-1.6, 0.6]), (1, 2): np.array([1.6, 0.6]), (2, 0): np.array([0.0, -1.6])}
    cam = np.array([0.0, 0.0, 0.0, 0.0])

    def __init__(self):
        rho = self.L / np.sqrt(3)
        ang = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
        self.c = [rho * np.array([np.cos(a), np.sin(a)]) for a in ang]
        self.cam = np.array([0.0, 0.0, self.c[0][0], self.c[0][1]])

    def pair(self, p, i, j):
        ci, cj = self.c[i], self.c[j]
        e = (cj - ci) / self.L
        n = np.array([-e[1], e[0]])
        zx, zy = p[..., 2] - ci[0], p[..., 3] - ci[1]
        t = zx * e[0] + zy * e[1]
        s = zx * n[0] + zy * n[1]
        qq = self.q[(i, j)]
        rx, ry = p[..., 0] - qq[0], p[..., 1] - qq[1]
        cap = np.sqrt(rx * rx + ry * ry + s * s) - self.r
        d, _ = smin_h(np.minimum(t, self.L - t), cap, self.k)
        return d + 1j * s

    def pieces(self):
        return [lambda p, i=i, j=j: self.pair(p, i, j) for (i, j) in [(0, 1), (1, 2), (2, 0)]]

    def F(self, p):
        return union(self.pair(p, 0, 1), union(self.pair(p, 1, 2), self.pair(p, 2, 0)))

    def jac(self, p):
        # hard union: the winner changes phase discontinuously, so differentiate
        # the winning piece only (what analytic gradients would do automatically)
        vals = np.stack([f(p) for f in self.pieces()], 0)
        win = np.argmin(vals.real, 0)
        a = np.zeros(p.shape); b = np.zeros(p.shape)
        for k, f in enumerate(self.pieces()):
            sel = np.nonzero(win == k)[0]
            if len(sel):
                ak, bk = jac(f, p[sel]); a[sel] = ak; b[sel] = bk
        return a, b

    def color(self, p):
        z = p[:, 2:4]
        d = np.stack([np.linalg.norm(z - ck, axis=-1) for ck in self.c], -1)
        w = np.argmin(d, -1)
        out = np.zeros((len(p), 3))
        for k in range(3):
            out = np.where((w == k)[:, None], shade(WORLD_COLORS[k], p[:, 0], p[:, 1], 0.8), out)
        return out


SCENES = {s.name: s for s in [Plane(), Neck(), ThreeNecks(), Cone(), Clifford(), Portal(), Portal3(), Handle(), Klein(), ThreeParallel()]}


# ---------------------------------------------------------------- new scenes (v2)

class FlatPortal(Scene):
    """N parallel worlds at n = rho * omega^k (n = x3 + i x4), joined through a
    slit |x1| < R on the line x2 = 0.  Inside the slit strip and outside a box
    around each slit end, the sheets depend on one coordinate only, so the
    space is exactly flat there: only the two ends (the 'frame') are curved."""
    name = "flat_portal"
    N, rho, R, a, T = 2, 1.0, 2.0, 0.6, 1.2
    cam = np.array([0.0, -2.5, 1.0, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([-1.0, 0.0, 0.0, 0.0])

    def W(self, p):
        x1, x2 = p[..., 0], p[..., 1]
        frame = 1 - smootherstep(self.R - self.a, self.R + self.a, np.abs(x1))   # 1 inside the frame, 0 outside
        full_turn = np.exp(-2j * np.pi * smootherstep(-self.T, self.T, x2))      # once around 0 while going through
        return 1 + frame * (full_turn - 1)

    def F(self, p):
        return roots(v_of(p), self.W(p), self.rho, self.N)

    def world_index(self, p):
        z = v_of(p)
        Wp = self.W(p)
        base = self.rho * np.abs(Wp) ** (1 / self.N) * np.exp(1j * np.angle(Wp) / self.N)
        cand = base[..., None] * np.exp(2j * np.pi * np.arange(self.N) / self.N)
        return np.argmin(np.abs(cand - z[..., None]), -1)

    def color(self, p):
        w = self.world_index(p)
        out = np.zeros(p.shape[:-1] + (3,))
        for k in range(self.N):
            out = np.where((w == k)[:, None], shade(WORLD_COLORS[k], p[:, 0], p[:, 1]), out)
        return out


class FlatPortal3(FlatPortal):
    name = "flat_portal3"
    N = 3


class YJunction(Scene):
    """Three parallel worlds R^2 x {rho d_k} joined by ONE three-way wormhole:
    three 4D capsules from the centre to the worlds whose carriers turn from
    y = 0 (at the centre, where the arms share a 3D slice) to the world's own
    slice; each world's solid is a half-space extruded only a little sideways."""
    name = "y_junction"
    rho, r, kj, km, m = 3.0, 0.7, 0.6, 0.4, 0.5
    r_max = 12.0
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([-1.0, 0.0, 0.0, 0.0])

    def __init__(self):
        th = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
        self.d = [np.array([np.cos(t), np.sin(t)]) for t in th]
        self.cam = np.array([0.0, -2.2, *(self.rho * self.d[0])])

    def frame(self, p, k):
        d = self.d[k]
        along = p[..., 2] * d[0] + p[..., 3] * d[1]
        side = -p[..., 2] * d[1] + p[..., 3] * d[0]
        return along, side

    def arm(self, p, k):
        along, side = self.frame(p, k)
        tube = capsule(p, np.zeros(4), np.array([0.0, 0.0, *(self.rho * self.d[k])]), self.r)
        angle = 0.5 * np.pi * smootherstep(0.3, 0.7, along / self.rho)     # the carrier turns along the arm
        return world(tube, turn(p[..., 1], side, angle))

    def parallel_world(self, p, k):
        along, side = self.frame(p, k)
        return world(extrude(side, self.rho - along, self.m), side)

    def F(self, p):
        out = smooth_union(smooth_union(self.arm(p, 0), self.arm(p, 1), self.kj), self.arm(p, 2), self.kj)
        for k in range(3):
            out = smooth_union(out, self.parallel_world(p, k), self.km)
        return out

    def color(self, p):
        z = p[:, 2:4]
        dist = np.stack([np.linalg.norm(z - self.rho * dk, axis=-1) for dk in self.d], -1)
        w = np.argmin(dist, -1)
        out = np.zeros((len(p), 3))
        for k in range(3):
            out = np.where((w == k)[:, None], shade(WORLD_COLORS[k], p[:, 0], p[:, 1], 0.8), out)
        tube = np.min(dist, -1) > 0.35
        out[tube] = out[tube] * 0.55 + 0.45 * GREY
        return out


class TriplePoint(Scene):
    """Three planes through one point (complex lines v=0, u=0, v=u), triple point smoothed."""
    name = "triple_point"
    eps, k = 0.25, 2.5
    cam = np.array([-3.0, 0.0, 0.0, 0.0])

    def F(self, p):
        u, v = u_of(p), v_of(p)
        r2 = np.abs(u) ** 2 + np.abs(v) ** 2
        return (u * v * (u - v) - self.eps * cutoff(np.sqrt(r2) / self.k)) / (r2 + 0.05)

    def color(self, p):
        u, v = u_of(p), v_of(p)
        d = np.stack([np.abs(v), np.abs(u), np.abs(u - v) / np.sqrt(2)], -1)
        w = np.argmin(d, -1)
        s = (u + v) / np.sqrt(2)
        return np.where((w == 0)[:, None], shade(GREEN, u.real, u.imag),
               np.where((w == 1)[:, None], shade(ORANGE, v.real, v.imag), shade(BLUE, s.real, s.imag)))


class Sphere(Scene):
    """A round sphere in the 3D slice w = 0."""
    name = "sphere"
    r = 1.5
    cam = np.array([1.5, 0.0, 0.0, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([0.0, 0.0, 1.0, 0.0])

    def F(self, p):
        return world(ball(p[..., 0], p[..., 1], p[..., 2], self.r), p[..., 3])

    def color(self, p):
        lon = np.arctan2(p[:, 1], p[:, 0]); lat = np.arcsin(np.clip(p[:, 2] / self.r, -1, 1))
        return shade(GREEN, lon * 8 / (2 * np.pi), lat * 6 / np.pi)


class Torus(Scene):
    """One 4D donut solid, cut by a turning carrier: angle 0 gives the ordinary
    donut (carrier w), 90 degrees the flat torus (carrier |xy| - a)."""
    name = "torus"
    a, b, angle = 1.0, 0.5, 0.0
    cam = np.array([1.0, 0.0, 0.5, 0.0])       # on top of the tube: on the torus for every angle
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([1.0, 0.0, 0.0, -1.0])

    def F(self, p):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        solid = ball(ball(x, y, self.a), z, w, self.b)
        return world(solid, turn(w, ball(x, y, self.a), np.radians(self.angle)))

    def color(self, p):
        th = np.radians(self.angle)
        r = np.hypot(p[:, 0], p[:, 1])
        e = np.cos(th) * (r - self.a) - np.sin(th) * p[:, 3]           # in-plane coordinate of the tube's circle
        phi, psi = np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 2], e)
        return shade(GREEN, phi * 8 / (2 * np.pi), psi * 6 / (2 * np.pi))


class TwoPlanes(Scene):
    """portal_gravity's wormhole_between_two_planes, unchanged, in the slice w = 0."""
    name = "two_planes"
    height, wormhole_size, smoothing = 1.0, 0.15, 0.1
    r_max = 1.5
    cam = np.array([0.26, 0.0, 0.30, 0.0])
    hint1 = np.array([1.0, 0.0, 0.0, 0.0])
    hint2 = np.array([0.0, 0.0, 1.0, 0.0])

    def F(self, p):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        d1 = y
        d2 = y - self.height
        d3 = ball(x, z, self.wormhole_size)
        d3 = extrude(y - self.height / 2.0, d3, self.height)
        d4 = smin(-d2, d3, self.smoothing)
        return world(smin(d1, d4, self.smoothing), w)

    def color(self, p):
        y = p[:, 1]
        base = np.where((y < self.smoothing)[:, None], np.array([0.2, 0.8, 0.1]),
               np.where((y > self.height - self.smoothing)[:, None], np.array([0.9, 0.6, 0.1]), np.array([0.1, 0.6, 0.9])))
        return shade(np.ones(3), p[:, 0], p[:, 2], 0.1) * base


for _s in [FlatPortal(), FlatPortal3(), YJunction(), TriplePoint(), Sphere(), Torus(), TwoPlanes()]:
    SCENES[_s.name] = _s


# ---------------------------------------------------------------- one-mouth wormholes

TEAL = np.array([0.2, 0.7, 0.7])
RED = np.array([0.9, 0.15, 0.15])


class MonoWormhole(Scene):
    """A plane whose mouth leads back into itself (the page's mono_wormhole).  The throat is the
    flat Moebius band {|u|^2 + 4|v|^2 = R^2, arg v = 2 arg u + spin(|u|)}, u = x + iy, v = z + iw:
    a flat tube of circumference 2 pi R whose far end is glued to itself, point to opposite point.
    Two charts: away from the centre the normal plane turned by 2 arg u + spin (there the world is
    a plane with a dome); near the centre the band's rulings, lines through the centre at half
    the angle of v (that half angle jumps once around: a sign cut, like the mirror handle's flip)."""
    name = "mono"
    R, k, twist = 1.0, 0.15, 16.0
    r_max = 12.0
    creature = True                      # scans paint a chiral creature at the camera
    cam = np.array([-2.5, 0.0, 0.0, 0.0])
    hint1 = np.array([0.0, -1.0, 0.0, 0.0])
    hint2 = np.array([1.0, 0.0, 0.0, 0.0])

    def spin(self, r):
        t = np.clip(1 - (r / self.R) ** 2, 0, None)
        return self.twist * t * t

    def turned(self, p):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        angle = 2 * np.arctan2(y, x) + self.spin(np.hypot(x, y))
        s, c = turn(z, w, angle), turn(w, -z, angle)
        return smooth_union(world(s, c), world(ball(x, y, 2 * s, self.R), c), self.k)

    def central(self, p, psi0=None):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        psi = np.arctan2(w, z)
        if psi0 is not None:                                   # continue the branch of psi0
            psi = psi0 + np.arctan2(np.sin(psi - psi0), np.cos(psi - psi0))
        half = 0.5 * (psi - self.spin(np.hypot(x, y)))
        return world(ball(x, y, 2 * np.hypot(z, w), self.R), turn(y, -x, half))

    def field(self, p, inner, psi0=None):
        return np.where(inner, self.central(p, psi0), self.turned(p))

    def F(self, p):
        return self.field(p, np.hypot(p[..., 0], p[..., 1]) < 0.5 * self.R)

    def jac(self, p):
        # the chart and the branch of atan(w, z) are frozen at p: differentiate the side p is on
        inner = np.hypot(p[..., 0], p[..., 1]) < 0.5 * self.R
        return jac(lambda q: self.field(q, inner, np.arctan2(p[..., 3], p[..., 2])), p)

    def throat_tables(self):
        """Flat coordinates of the throat as functions of r = |u|: depth d from the mouth and the
        shift of the around coordinate (sigma = R arg u + shift), from its metric."""
        if getattr(self, "_tab_key", None) != (self.R, self.twist):
            R = self.R
            al = np.linspace(0, np.pi / 2, 4001)                   # r = R cos(al): 0 at the mouth, pi/2 at the far end
            r = R * np.cos(al)
            sp_a = np.gradient(self.spin(r), al)
            e = R * np.sqrt(np.sin(al) ** 2 + 0.25 * np.cos(al) ** 2 + 0.25 * (np.sin(al) * np.cos(al) * sp_a) ** 2)
            f = 0.5 * R * np.sin(al) ** 2 * sp_a
            cum = lambda g: np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(al))])
            self._tab = (r[::-1], cum(e)[::-1], cum(f)[::-1])
            self._tab_key = (self.R, self.twist)
        return self._tab

    @property
    def length(self):
        return self.throat_tables()[1][0]

    def color(self, p):
        rr, dd, ss = self.throat_tables()
        r = np.hypot(p[:, 0], p[:, 1])
        sigma = self.R * np.arctan2(p[:, 1], p[:, 0]) + np.interp(r, rr, ss)
        throat = shade(TEAL, np.mod(sigma, 2 * np.pi * self.R), np.interp(r, rr, dd), 2 * np.pi * self.R / 12)
        return np.where((r < self.R)[:, None], throat, shade(GREEN, p[:, 0], p[:, 1], 0.5))

    def creature_coords(self, pos, cam, e1, e2):
        """The camera frame in the throat's flat coordinates, unfolded across the far end: a point
        (sigma, d) is also (sigma + pi R, 2 L - d) on the other side of the glued end."""
        R, L, C = self.R, self.length, 2 * np.pi * self.R
        rr, dd, ss = self.throat_tables()
        def flat(q):
            r = np.hypot(q[..., 0], q[..., 1])
            return R * np.arctan2(q[..., 1], q[..., 0]) + np.interp(r, rr, ss), np.interp(r, rr, dd), r
        sc, dc, rc = flat(cam)
        if rc >= R:                                          # camera on the plane: the plane is flat too
            a1, a2 = (pos[:, :2] - cam[:2]) @ e1[:2], (pos[:, :2] - cam[:2]) @ e2[:2]
            return a1, a2, np.hypot(pos[:, 0], pos[:, 1]) > R + 0.2
        s, d, r = flat(pos)
        wrapc = lambda x: (x + 0.5 * C) % C - 0.5 * C
        ds1, dd1 = wrapc(s - sc), d - dc
        ds2, dd2 = wrapc(s + 0.5 * C - sc), (2 * L - d) - dc
        use2 = np.hypot(ds2, dd2) < np.hypot(ds1, dd1)
        ds, dd_ = np.where(use2, ds2, ds1), np.where(use2, dd2, dd1)
        # e1 points along -theta (decreasing sigma), e2 towards the far end (increasing d)
        return -ds, dd_, r < R

    def at_depth(self, frac, theta=np.pi):
        """A point of the throat at a fraction of its depth, with the frame (around, towards the far end)."""
        rr, dd, _ = self.throat_tables()
        r = np.interp(frac * dd[0], dd[::-1], rr[::-1])
        def P(al, th):
            r = self.R * np.cos(al); rho = 0.5 * self.R * np.sin(al); ang = 2 * th + self.spin(r)
            return np.array([r * np.cos(th), r * np.sin(th), rho * np.cos(ang), rho * np.sin(ang)])
        al, h = np.arccos(r / self.R), 1e-4
        return P(al, theta), -(P(al, theta + h) - P(al, theta - h)) / (2 * h), (P(al + h, theta) - P(al - h, theta)) / (2 * h)


class MonoWormholeInside(MonoWormhole):
    """The same, seen from inside the throat, near its far end."""
    name = "mono_inside"
    r_max = 7.5

    def __init__(self):
        self.cam, self.hint1, self.hint2 = self.at_depth(0.9)


class Pocket(Scene):
    """The unmirrored alternative, for the checks: a tube whose end is squeezed flat into a fold.
    Its cross-section is a stadium, half-width b -> 0 and straight half-length a -> pi R / 2 over
    the last hf of its height H.  The walls and the fold are flat; the two ends of the fold carry
    the curvature 2 pi that closing a tube costs.  Lives in the slice w = 0."""
    name = "pocket"
    R, k, H, hf = 1.0, 0.15, 3.5, 1.5
    r_max = 8.0
    cam = np.array([1.0, 0.0, 1.7, 0.0])
    hint1 = np.array([0.0, 1.0, 0.0, 0.0])
    hint2 = np.array([0.0, 0.0, 1.0, 0.0])

    def ab(self, z):
        t = np.clip((z - (self.H - self.hf)) / self.hf, 0, None)
        s = np.clip(t, 0, 1)
        return 0.5 * np.pi * self.R * s * s * (3 - 2 * s), self.R ** 2 * (1 - t * t)   # a, b^2 (< 0 above: closed)

    def F(self, p):
        x, y, z, w = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        a, b2 = self.ab(z)
        pocket = world((x * x + np.maximum(np.abs(y) - a, 0) ** 2 - b2) / (2 * self.R), w)
        return smooth_union(world(z, w), pocket, self.k)

    def color(self, p):
        return np.where((p[:, 2] > 0.2)[:, None], shade(TEAL, p[:, 1], p[:, 2], 0.5), shade(GREEN, p[:, 0], p[:, 1], 0.5))


for _s in [MonoWormhole(), MonoWormholeInside(), Pocket()]:
    SCENES[_s.name] = _s
