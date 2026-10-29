// High-quality perspective renderer for 3D worlds of codimension 2 (R^5) and 3 (R^6).
// Port of render_hq.py / scenes5d.py / render6d.py. Double precision, OpenMP.
// usage: trace3d <scene> <W> <H> <spp> <n_steps> <out.rgbf>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cd = std::complex<double>;
constexpr int DMAX = 6, KMAX = 3;
using V = std::array<double, DMAX>;
struct RGB { double r, g, b; };

static const RGB WORLD[4] = {{0.30, 0.75, 0.25}, {0.95, 0.55, 0.15}, {0.25, 0.50, 0.95}, {0.75, 0.35, 0.85}};
static double clampd(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
static double smoothstep(double e0, double e1, double x) { double t = clampd((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
static double smootherstep(double e0, double e1, double x) { double t = clampd((x - e0) / (e1 - e0), 0, 1); return t * t * t * (t * (6 * t - 15) + 10); }

struct Scene {
    int D = 5, K = 2;
    double fov = 75, max_len = 18;
    double fov_v = 0;     // when set, the vertical view angle (fov is then ignored)
    double self_r = 0.18; // the camera is drawn as a ball of this radius just behind the eye (0: not drawn)
    V cam{}, look{}, up{}, right{};
    virtual ~Scene() {}
    virtual void value(const V& p, double* f) const = 0;
    virtual void jet(const V& p, double* f, double J[KMAX][DMAX]) const { num_jet(p, f, J, [this](const V& q, double* g) { value(q, g); }); }
    template <class Fn> void num_jet(const V& p, double* f, double J[KMAX][DMAX], Fn fn) const {
        const double h = 1e-5;
        fn(p, f);
        for (int i = 0; i < D; i++) {
            V a = p, b = p; a[i] += h; b[i] -= h;
            double fa[KMAX], fb[KMAX];
            fn(a, fa); fn(b, fb);
            for (int k = 0; k < K; k++) J[k][i] = (fa[k] - fb[k]) / (2 * h);
        }
    }
    virtual bool hit(const V& p, const V& q, double dist, RGB& c) const = 0;
    virtual RGB sky(const V& p, const V& d) const = 0;
    virtual double dt_scale(const V&) const { return 1.0; }
    // absolute step size; adaptive scenes override this (big steps where space is flat)
    virtual double h_at(const V& p, double dt0) const { return dt0 * std::min(1.0, dt_scale(p)); }
    // distance to the nearest object (balls), so steps never jump over one
    virtual double obj_dist(const V&) const { return 1e9; }
};

// ---------------------------------------------------------------- shared shading
static double FOG = 0.03;
// coordinate i of the point where the segment p->q crosses the floor y = f (y is coordinate 1)
template <class A, class B> static double lerp_hit(const A& p, const B& q, double f, int i) {
    double t = (p[1] - f) / (p[1] - q[1]);
    return p[i] + t * (q[i] - p[i]);
}
static RGB floor_color(RGB base, double x0, double x2, double dist) {
    int ch = ((int)std::floor(x0) + (int)std::floor(x2)) & 1;
    double s = 0.62 + 0.38 * ch, fog = std::exp(-FOG * dist);
    return {base.r * s * fog + (1 - fog) * 0.85, base.g * s * fog + (1 - fog) * 0.85, base.b * s * fog + (1 - fog) * 0.85};
}
static double lambert(double nx, double ny, double nz) {
    double lx = 0.4, ly = 1.0, lz = -0.5, ln = std::sqrt(lx * lx + ly * ly + lz * lz);
    return 0.35 + 0.65 * clampd((nx * lx + ny * ly + nz * lz) / ln, 0, 1);
}
static RGB scale(RGB c, double s) { return {c.r * s, c.g * s, c.b * s}; }
static RGB sky_mix(RGB base, double dy) {
    double h = clampd(dy * 0.5 + 0.5, 0, 1), s = 0.6 + 0.4 * h;
    return {(0.55 + 0.45 * base.r) * s, (0.55 + 0.45 * base.g) * s, (0.55 + 0.45 * base.b) * s};
}
struct Ball { int world; double c[6]; int nc; double r; RGB col; };
static double balls_dist(const std::vector<Ball>& bs, const double* x) {
    double d = 1e9;
    for (auto& b : bs) d = std::min(d, std::sqrt((x[0] - b.c[0]) * (x[0] - b.c[0]) + (x[1] - b.c[1]) * (x[1] - b.c[1]) + (x[2] - b.c[2]) * (x[2] - b.c[2])) - b.r);
    return d;
}

// ---------------------------------------------------------------- the page's library
// Numbers (portal_gravity's ball / extrude / smin), then worlds: a world is two
// ordinary SDFs, the solid and the carrier; it is the boundary of the solid,
// inside the carrier.
static double ball2(double x, double y, double r) { return std::sqrt(x * x + y * y) - r; }
static double ball3(double x, double y, double z, double r) { return std::sqrt(x * x + y * y + z * z) - r; }
static double extrude(double p, double d, double h) {
    double wx = d, wy = std::fabs(p) - h, mx = std::max(wx, 0.0), my = std::max(wy, 0.0);
    return std::min(std::max(wx, wy), 0.0) + std::sqrt(mx * mx + my * my);
}
static double turn(double a, double b, double angle) { return std::cos(angle) * a + std::sin(angle) * b; }
struct World { double solid, carrier; };
static World world(double solid, double carrier) { return {solid, carrier}; }
static World extrude(World A, double p, double h) { return {extrude(p, A.solid, h), A.carrier}; }
static World smooth_union(World A, World B, double k) {
    double h = clampd(0.5 + 0.5 * (B.solid - A.solid) / k, 0, 1);
    return {B.solid + (A.solid - B.solid) * h - k * h * (1 - h), B.carrier + (A.carrier - B.carrier) * smoothstep(0, 1, h)};
}

// ---------------------------------------------------------------- Stargate: zeta^N = rho^N W(x)
struct Stargate : Scene {
    int N; double rho, R, floorh; bool frame;
    std::vector<Ball> balls;
    Stargate(int N_, bool frame_, bool back) : N(N_), rho(0.7), R(1.6), floorh(-2.3), frame(frame_) {
        D = 5; K = 2; fov = 75; max_len = 18;
        balls = {{0, {2.6, -1.6, 2.0}, 3, 0.7, {0.9, 0.2, 0.2}}, {1, {-0.5, -1.5, 4.0}, 3, 0.8, {0.95, 0.9, 0.2}}, {2, {0.7, -1.5, 3.5}, 3, 0.8, {0.9, 0.3, 0.9}}};
        double z0 = back ? 5.0 : -5.0;
        on_world(0.0, -0.4, z0, 0, cam);
        look = {0, 0, back ? -1.0 : 1.0, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {back ? -1.0 : 1.0, 0, 0, 0, 0, 0};
    }
    cd W(double x0, double x1, double x2) const {
        double r2 = x0 * x0 + x1 * x1 + x2 * x2;
        return cd(r2 - R * R, 2 * R * x2) / (r2 + R * R);
    }
    void on_world(double x0, double x1, double x2, int k, V& p) const {
        cd w = W(x0, x1, x2);
        cd z = rho * std::pow(std::abs(w), 1.0 / N) * std::exp(cd(0, (std::arg(w) + 2 * M_PI * k) / N));
        p = {x0, x1, x2, z.real(), z.imag(), 0};
    }
    int sheet(const V& p) const {
        cd w = W(p[0], p[1], p[2]), z(p[3], p[4]);
        cd base = rho * std::pow(std::abs(w), 1.0 / N) * std::exp(cd(0, std::arg(w) / N));
        int best = 0; double bd = 1e300;
        for (int k = 0; k < N; k++) {
            double d = std::abs(base * std::exp(cd(0, 2 * M_PI * k / N)) - z);
            if (d < bd) { bd = d; best = k; }
        }
        return best;
    }
    double ring_dist(const V& p) const { return std::hypot(std::hypot(p[0], p[1]) - R, p[2]); }
    void value(const V& p, double* f) const override {
        cd z(p[3], p[4]);
        cd F = (std::pow(z, N) - std::pow(rho, N) * W(p[0], p[1], p[2])) / (N * std::pow(rho, N - 1));
        f[0] = F.real(); f[1] = F.imag();
    }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {
        double x[3] = {p[0], p[1], p[2]};
        double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        cd A(r2 - R * R, 2 * R * x[2]); double B = r2 + R * R;
        cd w = A / B, z(p[3], p[4]);
        double s = N * std::pow(rho, N - 1), rN = std::pow(rho, N);
        cd F = (std::pow(z, N) - rN * w) / s;
        f[0] = F.real(); f[1] = F.imag();
        for (int k = 0; k < 3; k++) {
            cd dA(2 * x[k], k == 2 ? 2 * R : 0.0); double dB = 2 * x[k];
            cd g = -rN * ((dA * B - A * dB) / (B * B)) / s;
            J[0][k] = g.real(); J[1][k] = g.imag();
        }
        cd dz = std::pow(z / rho, N - 1);
        J[0][3] = dz.real(); J[1][3] = dz.imag();
        cd dzi = cd(0, 1) * dz;
        J[0][4] = dzi.real(); J[1][4] = dzi.imag();
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = sheet(q);
        for (auto& b : balls) {
            if (b.world >= N || b.world != w) continue;
            double dx = q[0] - b.c[0], dy = q[1] - b.c[1], dz = q[2] - b.c[2], l = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (l < b.r) { c = scale(b.col, lambert(dx / l, dy / l, dz / l)); return true; }
        }
        if (p[1] > floorh && q[1] <= floorh) { c = floor_color(WORLD[w], lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), dist); return true; }
        if (frame && ring_dist(q) < 0.05) { c = {0.85, 0.7, 0.25}; return true; }
        return false;
    }
    RGB sky(const V& p, const V& d) const override { return sky_mix(WORLD[sheet(p)], d[1]); }
    double dt_scale(const V& p) const override { return clampd(0.25 * ring_dist(p), 0.004, 0.03) / 0.03; }
};

// ---------------------------------------------------------------- three parallel worlds, pairwise tubes
struct TubeWorlds : Scene {
    double L = 1.6, r = 0.75, k = 0.45, floorh = -1.6;
    double c[3][2];
    double q[3][3] = {{-1.6, 0.0, 0.5}, {0.0, 0.0, 4.5}, {1.6, 0.0, 0.5}};   // pairs (0,1), (1,2), (2,0)
    int pi_[3] = {0, 1, 2}, pj_[3] = {1, 2, 0};
    std::vector<Ball> balls;
    TubeWorlds() {
        D = 5; K = 2; fov = 75; max_len = 150;
        double rho = L / std::sqrt(3.0);
        for (int i = 0; i < 3; i++) { double a = M_PI / 2 + 2 * M_PI * i / 3; c[i][0] = rho * std::cos(a); c[i][1] = rho * std::sin(a); }
        cam = {0.0, -0.2, -3.5, c[0][0], c[0][1], 0};
        look = {0, 0, 1, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        balls = {{1, {-0.8, -1.0, 1.5}, 3, 0.6, {0.95, 0.9, 0.2}}, {2, {1.2, -1.0, 2.0}, 3, 0.6, {0.9, 0.3, 0.9}}, {0, {0.0, -1.1, 3.0}, 3, 0.5, {0.9, 0.2, 0.2}}};
    }
    static double smin(double a, double b, double kk) { double h = clampd(0.5 + 0.5 * (b - a) / kk, 0, 1); return b + (a - b) * h - kk * h * (1 - h); }
    void pair(const V& p, int m, double* f) const {
        int i = pi_[m], j = pj_[m];
        double ex = (c[j][0] - c[i][0]) / L, ey = (c[j][1] - c[i][1]) / L, nx = -ey, ny = ex;
        double zx = p[3] - c[i][0], zy = p[4] - c[i][1];
        double t = zx * ex + zy * ey, s = zx * nx + zy * ny;
        double x0 = p[0] - q[m][0], x1 = p[1] - q[m][1], x2 = p[2] - q[m][2];
        double cap = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2 + s * s) - r;
        f[0] = smin(std::min(t, L - t), cap, k); f[1] = s;
    }
    int winner(const V& p) const {
        int best = 0; double bv = 1e300;
        for (int m = 0; m < 3; m++) { double f[2]; pair(p, m, f); if (f[0] < bv) { bv = f[0]; best = m; } }
        return best;
    }
    void value(const V& p, double* f) const override { pair(p, winner(p), f); }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {
        int m = winner(p);   // differentiate the winning piece only
        num_jet(p, f, J, [this, m](const V& x, double* g) { pair(x, m, g); });
    }
    int world_of(const V& p) const {
        int best = 0; double bd = 1e300;
        for (int i = 0; i < 3; i++) { double d = std::hypot(p[3] - c[i][0], p[4] - c[i][1]); if (d < bd) { bd = d; best = i; } }
        return best;
    }
    bool hit(const V& p, const V& qq, double dist, RGB& col) const override {
        int w = world_of(qq);
        for (auto& b : balls) {
            if (b.world != w) continue;
            double dx = qq[0] - b.c[0], dy = qq[1] - b.c[1], dz = qq[2] - b.c[2], l = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (l < b.r) { col = scale(b.col, lambert(dx / l, dy / l, dz / l)); return true; }
        }
        if (p[1] > floorh && qq[1] <= floorh) { col = floor_color(WORLD[w], lerp_hit(p, qq, floorh, 0), lerp_hit(p, qq, floorh, 2), dist); return true; }
        return false;
    }
    RGB sky(const V& p, const V& d) const override { return sky_mix(WORLD[world_of(p)], d[1]); }
    double obj_dist(const V& p) const override { return balls_dist(balls, &p[0]); }
    double h_at(const V& p, double) const override {
        double best = 1e300;
        for (int m = 0; m < 3; m++) best = std::min(best, std::sqrt(std::pow(p[0] - q[m][0], 2) + std::pow(p[1] - q[m][1], 2) + std::pow(p[2] - q[m][2], 2)));
        double fd = best - (r + k + 0.6);
        double zc = std::hypot(p[3] - c[world_of(p)][0], p[4] - c[world_of(p)][1]);
        if (fd <= 0 || zc > 0.05) return 0.006;
        return clampd(0.3 * fd, 0.006, 0.5);
    }
};

// ---------------------------------------------------------------- flat T^2 x R throat
struct FlatThroat : Scene {
    // proportions from portal_gravity's base_5d scene: Clifford radii a = b = 0.5, flat half-length 2.5,
    // a collar of 3 (worlds at v = +-5.5), mouth donut 3 x 1.5, floor 4.9 below the throat's centre
    double a = 0.5, b = 0.5, ma = 3.0, mb = 1.5, L = 5.5, L1 = 2.5, L2 = 5.5, k = 0.4, floorh = -4.9, tile = 2.0;
    std::vector<Ball> balls;
    void cl(double th, double ph, double s, double* out) const { out[0] = a * std::cos(th); out[1] = a * std::sin(th); out[2] = b * std::cos(ph); out[3] = b * std::sin(ph); out[4] = s; }
    bool outside = false, far = false;
    FlatThroat(bool outside_ = false, bool far_ = false) : outside(outside_), far(far_) {
        D = 5; K = 2; fov_v = 90; max_len = 22;
        double c0[5]; cl(0, 0, 0, c0);
        cam = {c0[0], c0[1], c0[2], c0[3], c0[4], 0};
        look = {0, 1, 0, 0, 0, 0}; up = {0, 0, 0, 1, 0, 0}; right = {0, 0, 0, 0, 1, 0};
        // four balls around the line of sight, which is left free: straight ahead, around the torus, you see yourself
        RGB cols[3] = {{0.62, 0.3, 0.88}, {0.2, 0.4, 0.95}, {0.95, 0.85, 0.2}};
        double ph[3] = {-1.6, 0.0, 0.0}, ss[3] = {0.0, 0.7, -0.7};
        for (int i = 0; i < 3; i++) { Ball bl{2, {}, 5, 0.13, cols[i]}; cl(2.6, ph[i], ss[i], bl.c); balls.push_back(bl); }
        Ball w{2, {}, 5, 0.10, {0.9, 0.9, 0.9}}; cl(2.6, 1.6, 0.0, w.c); balls.push_back(w);
        if (far) max_len = 60;  // the inside view, seen much further
        if (outside) {          // base_5d's orbit camera: in the world v = +L, 6.98 from the mouth's centre, looking at it
            max_len = 150;
            cam = {3.8246, -0.3799, -5.8267, 0.0, L, 0};
            look = {-0.5479, 0.0544, 0.8347, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {-0.8360, 0, -0.5487, 0, 0, 0};
        }
    }
    double h_at(const V& p, double dt0) const override {
        double fd = std::hypot(std::hypot(p[0], p[1]) - ma, p[2]) - (mb + k + 0.6);     // distance from the mouth
        if (far) {                                        // fine steps in the throat, bigger ones far out in the worlds
            if (std::fabs(p[4]) < L - 0.02) return 0.003;
            return fd <= 0 ? 0.003 : clampd(0.3 * fd, 0.003, 0.5);
        }
        if (!outside) return dt0;
        if (std::fabs(p[4]) < L - 0.02) return 0.006;
        return fd <= 0 ? 0.006 : clampd(0.3 * fd, 0.006, 0.5);
    }
    // the page's torus(): one 4D donut, cut by a carrier that turns from w (donut) to |xy| - a (flat torus)
    World torus(const V& p, double ta, double tb, double angle) const {
        double ring = ball2(p[0], p[1], ta);
        return world(ball3(ring, p[2], p[3], tb), turn(p[3], ring, angle));
    }
    // the page's flat_throat(): the torus, extruded along v, smooth-unioned with the worlds v = +-L.
    // Flat in |v| < L1; between L1 and L2 it grows into the mouth donut and turns into the worlds' carrier w,
    // like base_5d's collar, which is slow where it leaves the flat part and fastest at the mouth.
    void value(const V& p, double* f) const override {
        double v = p[4];
        double t = clampd((std::fabs(v) - L1) / (L2 - L1), 0, 1), s = t * t;    // the collar: slow at first, fastest at the mouth
        World throat = extrude(torus(p, a + (ma - a) * s, b + (mb - b) * s, 0.5 * M_PI * (1.0 - s)), v, L);
        World worlds = world(L - std::fabs(v), p[3]);
        World F = smooth_union(worlds, throat, k);
        f[0] = F.solid; f[1] = F.carrier;
    }
    int world_of(const V& p) const { return p[4] < -L + 0.6 ? 0 : (p[4] > L - 0.6 ? 1 : 2); }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = world_of(q);
        for (auto& bl : balls) {
            if (bl.world != w) continue;
            double d[5], l2 = 0;
            for (int i = 0; i < 5; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; }
            double l = std::sqrt(l2);
            if (l < bl.r) { c = scale(bl.col, lambert(d[0] / l, d[1] / l, d[2] / l)); return true; }
        }
        if (std::fabs(q[4]) > L - 0.05 && p[1] > floorh && q[1] <= floorh) { c = floor_color(WORLD[w], lerp_hit(p, q, floorh, 0) / tile, lerp_hit(p, q, floorh, 2) / tile, dist); return true; }
        return false;
    }
    RGB sky(const V& p, const V&) const override {
        int w = world_of(p);
        RGB base = w == 2 ? RGB{0.6, 0.6, 0.65} : WORLD[w];
        return {(0.5 + 0.5 * base.r) * 0.85, (0.5 + 0.5 * base.g) * 0.85, (0.5 + 0.5 * base.b) * 0.85};
    }
};

// ---------------------------------------------------------------- a wormhole into itself, for a 3D world: the mirrored one
// The page's 2D mono_wormhole (a plane whose mouth leads into a flat Moebius tube), revolved around an axis:
// the mouth becomes a donut, and the far end of the throat glues every point to the opposite side of the
// tube's small circle.  A trip through it mirrors you (the end is the twisted I-bundle over a torus).
struct MonoDonut : Scene {
    double R = 1.0, k = 0.15, twist = 16.0, c = 2.6, yc = 0.0, floorh = -4.2;
    std::vector<Ball> balls;
    double spin(double r) const { double t = std::max(1 - r * r / (R * R), 0.0); return twist * t * t; }
    // the 2D field at (u1, u2) in the world plane and (n1, n2) in the extra plane; chart: -1 auto, 0 turned, 1 centre
    void field2(double u1, double u2, double n1, double n2, double* f, int chart, double psi0) const {
        double r = std::hypot(u1, u2);
        bool inner = chart < 0 ? r < 0.5 * R : chart == 1;
        if (!inner) {
            double angle = 2 * std::atan2(u2, u1) + spin(r);
            double sv = turn(n1, n2, angle), cv = turn(n2, -n1, angle);
            World F = smooth_union(world(sv, cv), world(ball3(u1, u2, 2 * sv, R), cv), k);
            f[0] = F.solid; f[1] = F.carrier;
        } else {
            double psi = std::atan2(n2, n1);
            if (chart >= 0) psi = psi0 + std::atan2(std::sin(psi - psi0), std::cos(psi - psi0));   // continue psi0's branch
            f[0] = ball3(u1, u2, 2 * std::hypot(n1, n2), R);
            f[1] = turn(u2, -u1, 0.5 * (psi - spin(r)));
        }
    }
    // the page's revolve: the 2D slice at angle phi around the axis, with u1 = distance from the axis - c, u2 = z
    void slice(const V& p, double& u1, double& u2) const { u1 = std::hypot(p[0], p[1] - yc) - c; u2 = p[2]; }
    void value(const V& p, double* f) const override { double u1, u2; slice(p, u1, u2); field2(u1, u2, p[3], p[4], f, -1, 0); }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {   // chart and branch frozen at p
        double u1, u2; slice(p, u1, u2);
        int chart = std::hypot(u1, u2) < 0.5 * R ? 1 : 0; double psi0 = std::atan2(p[4], p[3]);
        num_jet(p, f, J, [&](const V& q, double* g) { double a, b; slice(q, a, b); field2(a, b, q[3], q[4], g, chart, psi0); });
    }
    MonoDonut(int view) {
        D = 5; K = 2; fov_v = 64; max_len = 60;
        const Ball bs[] = {{0, {-4.6, floorh + 0.7, -1.0}, 3, 0.7, {0.95, 0.85, 0.2}}, {0, {4.4, floorh + 0.6, 0.8}, 3, 0.6, {0.2, 0.4, 0.95}},
                           {0, {0.6, floorh + 0.8, -6.0}, 3, 0.8, {0.62, 0.3, 0.88}}};
        for (const auto& b : bs) balls.push_back(b);
        self_r = 0.25;
        if (view == 0) {            // in front of the mouth, on its axis
            cam = {0.0, yc, 7.0, 0, 0, 0}; look = {0, 0, -1, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        } else if (view == 3) {     // inside the throat near its far end, looking around the tube's small circle
            // the 2D throat point at alpha, theta (r = R cos alpha), in the slice phi = 0 (the +x side of the ring)
            auto P = [&](double al, double th) {
                double r = R * std::cos(al), rho = 0.5 * R * std::sin(al), ang = 2 * th + spin(r);
                return V{c + r * std::cos(th), yc, r * std::sin(th), rho * std::cos(ang), rho * std::sin(ang), 0};
            };
            const double al = 1.2, th = M_PI, e = 1e-5;
            cam = P(al, th);
            V a1 = P(al, th + e), a0 = P(al, th - e), b1 = P(al + e, th), b0 = P(al - e, th);
            for (int i = 0; i < 5; i++) { look[i] = -(a1[i] - a0[i]); up[i] = b1[i] - b0[i]; }
            right = {0, 1, 0, 0, 0, 0};      // around the ring
            max_len = 30;
        } else if (view == 2) {     // in the plane of the ring, outside it, looking at the tube
            cam = {c + 2.5, yc, 0.0, 0, 0, 0}; look = {-1, 0, 0, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {0, 0, -1, 0, 0, 0};
        } else {                    // from the side and a little above
            double a = 0.62, d = 8.2;
            cam = {d * std::sin(a), yc + 1.2, d * std::cos(a), 0, 0, 0};
            double l = std::hypot(std::hypot(d * std::sin(a), 1.2), d * std::cos(a));
            look = {-d * std::sin(a) / l, -1.2 / l, -d * std::cos(a) / l, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {std::cos(a), 0, -std::sin(a), 0, 0, 0};
        }
    }
    double h_at(const V& p, double) const override {
        double u1, u2; slice(p, u1, u2);
        double fd = std::hypot(u1, u2) - (R + 0.6);                     // distance from the mouth
        return fd <= 0 ? 0.002 : clampd(0.3 * fd, 0.002, 0.5);
    }
    double obj_dist(const V& p) const override { return balls_dist(balls, p.data()); }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        for (const auto& bl : balls) {
            double d[3], l2 = 0;
            for (int i = 0; i < 3; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; }
            l2 += q[3] * q[3] + q[4] * q[4];
            double l = std::sqrt(l2);
            if (l < bl.r) { c = scale(bl.col, lambert(d[0] / l, d[1] / l, d[2] / l)); return true; }
        }
        if (p[1] > floorh && q[1] <= floorh && std::hypot(q[3], q[4]) < 0.05) { c = floor_color(WORLD[0], lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), dist); return true; }
        return false;
    }
    RGB sky(const V&, const V& d) const override { return sky_mix(WORLD[0], d[1]); }
};

// ---------------------------------------------------------------- a wormhole into itself, for a 3D world: the unmirrored one
// R^3 # RP^3: a ball-shaped mouth whose throat ends glued to itself, every point to the opposite one.
// World coordinates (x, y, z) with u = x + i y, h = z; extra plane v = (w, v5).  The far end is the
// I-bundle X(t, d) = ((t - i a d3) w, t d3, depth + b w^2), w = d1 + i d2, over the projective plane
// (t, d) ~ (-t, -d); a collar untwists it and brings it down to the flat world at the mouth.
// Construction and checks: codex (gpt-6-astra), tmp/codex_rp3; four charts, two of them with a scalar root.
struct MonoBall : Scene {
    double a = 1.0, b = 1.0, depth = 6.0, width = 1.0, floorh;
    std::vector<Ball> balls;
    double mouth() const { return 2 * a + width; }
    static double cutoff(double r, double s0, double s1) {       // C-infinity: 1 below s0, 0 above s1
        double s = (r - s0) / (s1 - s0);
        if (s <= 0) return 1;
        if (s >= 1) return 0;
        double lr = 1 / s - 1 / (1 - s);
        return 1 / (1 + std::exp(-lr));
    }
    double twist(double r) const { return a * cutoff(r, a, 2 * a); }
    double height(double r) const { return cutoff(r, 2 * a, mouth()); }
    int chart(const V& p) const {                               // 0 flat, 1 collar, 2 middle, 3 pole, 4 empty
        double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        cd v(p[3], p[4]), eta = (v - depth) / b;
        if (r >= mouth() || std::abs(v) > depth + b + a) return 0;
        if (r >= a) return 1;
        double u2 = p[0] * p[0] + p[1] * p[1], h = p[2];
        if (std::abs(eta) < 0.25 && u2 < h * h / 2 + 2 * a * a / 9) return 3;
        return std::abs(eta) > 0 ? 2 : 4;
    }
    double outer_root(double r2, double h, double A) const {    // T + A^2 h^2/T - A^2 h^4/T^2 = r^2
        double lo = std::max(h * h, r2 - A * A / 4), T = r2;
        for (int i = 0; i < 8; i++) {
            double q = h * h / T, f = T - r2 + A * A * q * (1 - q), df = 1 + (A * A / T) * (2 * q * q - q);
            T = clampd(T - f / df, lo, r2);
        }
        return T;
    }
    double pole_root(double U, double h) const {                // the q in (2/3, 1] with U = (h^2/q + a^2 q)(1 - q)
        double q = 1;
        for (int i = 0; i < 12; i++) {
            double f = U - (h * h / q + a * a * q) * (1 - q), df = h * h / (q * q) + a * a * (2 * q - 1);
            q = clampd(q - f / df, 2.0 / 3.0, 1);
        }
        return q;
    }
    void field(const V& p, double* f, int c) const {
        cd u(p[0], p[1]), v(p[3], p[4]); double h = p[2];
        if (c == 0) { f[0] = v.real(); f[1] = v.imag(); return; }
        if (c == 1) {                                           // the collar: a graph over the world
            double r2 = p[0] * p[0] + p[1] * p[1] + h * h, r = std::sqrt(r2), A = twist(r), T = outer_root(r2, h, A);
            cd w2 = T * u * u / ((T - cd(0, 1) * A * h) * (T - cd(0, 1) * A * h));
            cd val = v - height(r) * (depth + b * w2);
            f[0] = val.real(); f[1] = val.imag(); return;
        }
        if (c == 2) {                                           // the middle: closed form
            cd eta = (v - depth) / b, Q = u * u / eta; double s = std::abs(eta);
            f[0] = h + Q.imag() / (2 * a);
            f[1] = a * (s - 1) + (std::norm(u) / s - Q.real()) / (2 * a);
            return;
        }
        if (c == 3) {                                           // around the poles: a graph again
            double q = pole_root(std::norm(u), h);
            cd w2 = q * u * u / ((h - cd(0, 1) * a * q) * (h - cd(0, 1) * a * q));
            cd val = v - depth - b * w2;
            f[0] = val.real(); f[1] = val.imag(); return;
        }
        f[0] = a; f[1] = 0;
    }
    void value(const V& p, double* f) const override { field(p, f, chart(p)); }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {   // chart frozen at p
        int c = chart(p);
        num_jet(p, f, J, [&](const V& q, double* g) { field(q, g, c); });
    }
    MonoBall(int view) {
        D = 5; K = 2; fov_v = 64; max_len = 80; self_r = 0.3;
        if (const char* e = std::getenv("MONO_W")) width = std::atof(e);
        floorh = -(mouth() + 0.8);
        const Ball bs[] = {{0, {-5.0, floorh + 0.7, 1.5}, 3, 0.7, {0.95, 0.85, 0.2}}, {0, {4.8, floorh + 0.6, -0.5}, 3, 0.6, {0.2, 0.4, 0.95}},
                           {0, {0.8, floorh + 0.8, -7.0}, 3, 0.8, {0.62, 0.3, 0.88}}};
        for (const auto& bl : bs) balls.push_back(bl);
        if (view == 0) {            // in front of the mouth
            cam = {0.0, 0.0, 8.0, 0, 0, 0}; look = {0, 0, -1, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        } else {                    // inside the throat, looking up it towards the glued end
            auto X = [&](double r, double d1, double d2, double d3) {   // the collar's parametrization
                double l = std::sqrt(d1 * d1 + d2 * d2 + d3 * d3); d1 /= l; d2 /= l; d3 /= l;
                double A = twist(r), t = std::sqrt(r * r - A * A * d3 * d3 * (1 - d3 * d3));
                cd w(d1, d2), uu = (t - cd(0, 1) * A * d3) * w, vv = height(r) * (depth + b * w * w);
                return V{uu.real(), uu.imag(), t * d3, vv.real(), vv.imag(), 0};
            };
            const double r0 = view == 1 ? 2 * a + 0.35 * width : view == 2 ? 2 * a + 0.15 * width : 1.4, e = 1e-5;
            cam = X(r0, 1, 0, 0);
            V r1 = X(r0 - e, 1, 0, 0), r2 = X(r0 + e, 1, 0, 0), z1 = X(r0, 1, 0, e), z2 = X(r0, 1, 0, -e), y1 = X(r0, 1, e, 0), y2 = X(r0, 1, -e, 0);
            for (int i = 0; i < 5; i++) { look[i] = r1[i] - r2[i]; up[i] = y1[i] - y2[i]; right[i] = z1[i] - z2[i]; }
            max_len = 50;
        }
    }
    double h_at(const V& p, double) const override {
        double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        double fd = r - (mouth() + 0.4);
        return fd <= 0 ? 0.003 : clampd(0.3 * fd, 0.003, 0.5);
    }
    double obj_dist(const V& p) const override { return balls_dist(balls, p.data()); }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        for (const auto& bl : balls) {
            double d[3], l2 = 0;
            for (int i = 0; i < 3; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; }
            l2 += q[3] * q[3] + q[4] * q[4];
            double l = std::sqrt(l2);
            if (l < bl.r) { c = scale(bl.col, lambert(d[0] / l, d[1] / l, d[2] / l)); return true; }
        }
        if (p[1] > floorh && q[1] <= floorh && std::hypot(q[3], q[4]) < 0.05) { c = floor_color(WORLD[1], lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), dist); return true; }
        return false;
    }
    RGB sky(const V&, const V& d) const override { return sky_mix(WORLD[1], d[1]); }
};

// ---------------------------------------------------------------- a long flat throat, seen from its middle
// The same flat_throat() with the worlds far away (v = +-40): rays stay in the flat torus x line and see
// the few spheres repeated on its lattice; rays that find nothing end black.
struct LongThroat : FlatThroat {
    LongThroat() : FlatThroat(false) {
        a = ma = 1.2; b = mb = 0.7; L = 40.0; L1 = 37.0; L2 = 38.6;
        fov_v = 0; fov = 90; max_len = 70;
        balls.clear();
        struct Spec { double th, ph, s, r; RGB c; };
        const Spec specs[] = {
            {1.2, 0.0, 0.0, 0.22, {0.90, 0.20, 0.20}}, {2.6, 1.4, 1.6, 0.22, {0.20, 0.45, 0.95}}, {4.0, 3.0, -1.6, 0.22, {0.95, 0.85, 0.20}},
            {5.2, 4.6, 3.3, 0.20, {0.30, 0.80, 0.35}}, {0.6, 2.2, -3.4, 0.20, {0.85, 0.40, 0.90}}, {3.3, 5.4, 5.2, 0.20, {0.95, 0.55, 0.15}},
            {4.7, 1.0, -5.6, 0.20, {0.92, 0.92, 0.92}}};
        for (const auto& q : specs) { Ball bl{2, {}, 5, q.r, q.c}; cl(q.th, q.ph, q.s, bl.c); balls.push_back(bl); }
    }
    double h_at(const V& p, double) const override { return std::fabs(p[4]) < L1 - 0.5 ? 0.03 : 0.006; }
    double obj_dist(const V& p) const override {
        double d = 1e9;
        for (const auto& bl : balls) { double l2 = 0; for (int i = 0; i < 5; i++) l2 += (p[i] - bl.c[i]) * (p[i] - bl.c[i]); d = std::min(d, std::sqrt(l2) - bl.r); }
        return d;
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        for (const auto& bl : balls) {
            double d[5], l2 = 0, t2 = 0, dn = 0;
            for (int i = 0; i < 5; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; t2 += (q[i] - p[i]) * (q[i] - p[i]); }
            double l = std::sqrt(l2);
            if (l >= bl.r) continue;
            for (int i = 0; i < 5; i++) dn += d[i] / l * (p[i] - q[i]);
            double shade = 0.3 + 0.7 * clampd(dn / std::sqrt(t2), 0, 1);       // lit from the camera
            c = scale(bl.col, shade * std::exp(-dist / 28.0));                 // far copies fade to black
            return true;
        }
        return false;
    }
    RGB sky(const V&, const V&) const override { return {0, 0, 0}; }
};

// ---------------------------------------------------------------- three 3D worlds in 5D meeting along one line
// a = x + i z, b = w + i v: the worlds b = 0 (xyz), a = 0 (y w v) and a = b share only the y-axis.
// The page's triple_point(), which does not depend on y: the 2D junction, extruded along the shared line.
struct LineJunction : Scene {
    double eps = 0.4, k = 2.5, floorh = -1.6;
    std::vector<Ball> balls;
    static void in_world(int w, double s, double y, double t, double* out) {   // a point (s, y, t) of world w
        const double h = std::sqrt(0.5);
        if (w == 0) { out[0] = s; out[1] = y; out[2] = t; out[3] = 0; out[4] = 0; }
        else if (w == 1) { out[0] = 0; out[1] = y; out[2] = 0; out[3] = s; out[4] = t; }
        else { out[0] = s * h; out[1] = y; out[2] = t * h; out[3] = s * h; out[4] = t * h; }
    }
    LineJunction(double angle_deg, bool top) {
        D = 5; K = 2; fov = 70; max_len = 150;
        struct Spec { int w; double s, y, t, r; RGB c; };
        const Spec specs[] = {
            {0, 2.4, -1.05, 2.2, 0.55, {0.90, 0.20, 0.20}}, {0, -2.6, -1.15, -1.8, 0.45, {0.92, 0.92, 0.92}},
            {1, 2.0, -1.05, 2.6, 0.55, {0.95, 0.90, 0.20}}, {1, -2.4, -1.15, 1.6, 0.45, {0.90, 0.30, 0.90}},
            {2, 2.6, -1.05, -2.0, 0.55, {0.90, 0.30, 0.90}}, {2, -1.8, -1.15, 2.6, 0.45, {0.95, 0.90, 0.20}}};
        for (const auto& q : specs) { Ball bl{q.w, {}, 5, q.r, q.c}; in_world(q.w, q.s, q.y, q.t, bl.c); balls.push_back(bl); }
        double th = angle_deg * M_PI / 180, R = 5.0;
        if (!top) {      // in world 1 (xyz), on a circle around the shared line, looking at it
            cam = {-R * std::sin(th), -0.3, -R * std::cos(th), 0, 0, 0};
            look = {std::sin(th), 0, std::cos(th), 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {std::cos(th), 0, -std::sin(th), 0, 0, 0};
        } else {         // above world 1's floor, looking down at the shared line
            cam = {0.0, 5.0, -4.2, 0, 0, 0};
            double l = std::hypot(4.2, 6.6);
            look = {0, -6.6 / l, 4.2 / l, 0, 0, 0}; up = {0, 4.2 / l, 6.6 / l, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        }
    }
    static double cutoff(double t) { return 1.0 - smootherstep(0.5, 1.0, t); }
    void value(const V& p, double* f) const override {
        cd a(p[0], p[2]), b(p[3], p[4]);
        double r2 = std::norm(a) + std::norm(b);
        cd open = a * b * (a - b) - eps * cutoff(std::sqrt(r2) / k);      // the three worlds, crossing opened up
        cd F = open / (r2 + 0.05);
        f[0] = F.real(); f[1] = F.imag();
    }
    int world_of(const V& p) const {
        cd a(p[0], p[2]), b(p[3], p[4]);
        double d0 = std::abs(b), d1 = std::abs(a), d2 = std::abs(a - b) / std::sqrt(2.0);
        return d0 <= d1 && d0 <= d2 ? 0 : (d1 <= d2 ? 1 : 2);
    }
    void floor_coords(int w, double x, double z, double wc, double v, double& s, double& t) const {
        const double h = std::sqrt(0.5);
        if (w == 0) { s = x; t = z; } else if (w == 1) { s = wc; t = v; } else { s = (x + wc) * h; t = (z + v) * h; }
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = world_of(q);
        for (const auto& bl : balls) {
            if (bl.world != w) continue;
            double d[5], l2 = 0;
            for (int i = 0; i < 5; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; }
            double l = std::sqrt(l2);
            if (l < bl.r) {      // shade with the ball's normal in its own world's 3D coordinates
                double ns, nt; floor_coords(w, d[0], d[2], d[3], d[4], ns, nt);
                c = scale(bl.col, lambert(ns / l, d[1] / l, nt / l)); return true;
            }
        }
        if (p[1] > floorh && q[1] <= floorh) {
            double s, t;
            floor_coords(w, lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), lerp_hit(p, q, floorh, 3), lerp_hit(p, q, floorh, 4), s, t);
            c = floor_color(WORLD[w], s, t, dist); return true;
        }
        return false;
    }
    RGB sky(const V& p, const V& d) const override { return sky_mix(WORLD[world_of(p)], d[1]); }
    double obj_dist(const V& p) const override {
        double dd = 1e9;
        for (const auto& bl : balls) { double l2 = 0; for (int i = 0; i < 5; i++) l2 += (p[i] - bl.c[i]) * (p[i] - bl.c[i]); dd = std::min(dd, std::sqrt(l2) - bl.r); }
        return dd;
    }
    double h_at(const V& p, double) const override {
        double r = std::sqrt(p[0] * p[0] + p[2] * p[2] + p[3] * p[3] + p[4] * p[4]);
        double fd = r - (k + 0.3);                     // outside the junction the worlds are exactly flat
        return fd <= 0 ? 0.006 : clampd(0.3 * fd, 0.006, 0.5);
    }
};

// ---------------------------------------------------------------- three 3D worlds in 6D meeting at one point
// u = (p0, p1, p2), v = (p3, p4, p5).  The worlds v = 0, u = 0 and u = v meet only at the origin.  Each world is
// three numbers (A = v, B = u, C = u - v); the vector part of the quaternion product A B C,
// (A.C) B - (A.B) C - (B.C) A, is zero exactly on the three worlds; a small push opens the crossing.
struct PointJunction6 : Scene {
    double eps = 1.0, k = 2.6, floorh = -1.6;
    std::vector<Ball> balls;
    static void in_world(int w, double s, double y, double t, double* out) {     // the point (s, y, t) of world w
        const double h = std::sqrt(0.5);
        double a[3] = {s, y, t};
        for (int i = 0; i < 3; i++) {
            out[i] = w == 0 ? a[i] : (w == 1 ? 0.0 : a[i] * h);
            out[3 + i] = w == 0 ? 0.0 : (w == 1 ? a[i] : a[i] * h);
        }
    }
    PointJunction6(double angle_deg, bool top) {
        D = 6; K = 3; fov = 70; max_len = 150;
        struct Spec { int w; double s, y, t, r; RGB c; };
        const Spec specs[] = {
            {0, 2.4, -1.05, 2.2, 0.55, {0.90, 0.20, 0.20}}, {0, -2.6, -1.15, -1.8, 0.45, {0.92, 0.92, 0.92}},
            {1, 2.0, -1.05, 2.6, 0.55, {0.95, 0.90, 0.20}}, {1, -2.4, -1.15, 1.6, 0.45, {0.90, 0.30, 0.90}},
            {2, 2.6, -1.05, -2.0, 0.55, {0.90, 0.30, 0.90}}, {2, -1.8, -1.15, 2.6, 0.45, {0.95, 0.90, 0.20}}};
        for (const auto& q : specs) { Ball bl{q.w, {}, 6, q.r, q.c}; in_world(q.w, q.s, q.y, q.t, bl.c); balls.push_back(bl); }
        double th = angle_deg * M_PI / 180, R = 5.0;
        if (!top) {
            cam = {-R * std::sin(th), -0.3, -R * std::cos(th), 0, 0, 0};
            look = {std::sin(th), 0, std::cos(th), 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {std::cos(th), 0, -std::sin(th), 0, 0, 0};
        } else {
            cam = {0.0, 5.0, -4.2, 0, 0, 0};
            double l = std::hypot(4.2, 6.6);
            look = {0, -6.6 / l, 4.2 / l, 0, 0, 0}; up = {0, 4.2 / l, 6.6 / l, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        }
    }
    static double cutoff(double t) { return 1.0 - smootherstep(0.5, 1.0, t); }
    void value(const V& p, double* f) const override {
        const double* u = &p[0]; const double* v = &p[3];
        double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2], vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2], uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        double r2 = uu + vv, c = eps * cutoff(std::sqrt(r2) / k), s = 1.0 / (r2 + 0.05);
        for (int i = 0; i < 3; i++) f[i] = (-vv * u[i] + (2 * uv - uu) * v[i] - (i == 2 ? c : 0.0)) * s;
    }
    int world_of(const V& p) const {
        double dA = 0, dB = 0, dC = 0;
        for (int i = 0; i < 3; i++) { dA += p[3 + i] * p[3 + i]; dB += p[i] * p[i]; dC += 0.5 * (p[i] - p[3 + i]) * (p[i] - p[3 + i]); }
        return dA <= dB && dA <= dC ? 0 : (dB <= dC ? 1 : 2);
    }
    static void own(int w, const V& p, double& s, double& y, double& t) {   // a point's coordinates in world w's own frame
        const double h = std::sqrt(0.5);
        if (w == 0) { s = p[0]; y = p[1]; t = p[2]; }
        else if (w == 1) { s = p[3]; y = p[4]; t = p[5]; }
        else { s = (p[0] + p[3]) * h; y = (p[1] + p[4]) * h; t = (p[2] + p[5]) * h; }
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = world_of(q);
        for (const auto& bl : balls) {
            if (bl.world != w) continue;
            double d[6], l2 = 0;
            for (int i = 0; i < 6; i++) { d[i] = q[i] - bl.c[i]; l2 += d[i] * d[i]; }
            double l = std::sqrt(l2);
            if (l < bl.r) {
                V dv{}; for (int i = 0; i < 6; i++) dv[i] = d[i];
                double ns, ny, nt; own(w, dv, ns, ny, nt);
                c = scale(bl.col, lambert(ns / l, ny / l, nt / l)); return true;
            }
        }
        double ps, py, pt, qs, qy, qt;
        own(w, p, ps, py, pt); own(w, q, qs, qy, qt);
        if (py > floorh && qy <= floorh) {
            double t = (py - floorh) / (py - qy);
            c = floor_color(WORLD[w], ps + t * (qs - ps), pt + t * (qt - pt), dist); return true;
        }
        return false;
    }
    RGB sky(const V& p, const V& d) const override {
        int w = world_of(p); double s, y, t; own(w, d, s, y, t);
        return sky_mix(WORLD[w], y);
    }
    double obj_dist(const V& p) const override {
        double dd = 1e9;
        for (const auto& bl : balls) { double l2 = 0; for (int i = 0; i < 6; i++) l2 += (p[i] - bl.c[i]) * (p[i] - bl.c[i]); dd = std::min(dd, std::sqrt(l2) - bl.r); }
        return dd;
    }
    double h_at(const V& p, double) const override {
        double r = 0; for (int i = 0; i < 6; i++) r += p[i] * p[i];
        double fd = std::sqrt(r) - (k + 0.3);           // outside the cutoff the worlds are exactly flat
        return fd <= 0 ? 0.006 : clampd(0.3 * fd, 0.006, 0.5);
    }
};

// ---------------------------------------------------------------- codim 3: inversion neck in R^6
struct InversionNeck : Scene {
    double eps = 0.5, k = 2.6, floorh = -1.4;
    std::vector<Ball> balls;
    InversionNeck() {
        D = 6; K = 3; fov = 75; max_len = 150;
        cam = {0.3, 0.0, -4.0, 0, 0, 0};
        look = {0, 0, 1, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        balls = {{0, {1.8, -0.9, 0.5}, 3, 0.5, {0.9, 0.2, 0.2}}, {1, {-0.6, -0.9, 2.5}, 3, 0.6, {0.95, 0.9, 0.2}}};
    }
    static double cutoff(double t) { t = clampd((t - 0.5) / 0.5, 0, 1); return 1 - t * t * t * (t * (6 * t - 15) + 10); }
    static bool branchA(const V& p) {
        return p[3] * p[3] + p[4] * p[4] + p[5] * p[5] < p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    }
    void value(const V& p, double* f) const override { value_on(p, branchA(p), f); }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {
        bool br = branchA(p);   // differentiate the branch selected at p
        num_jet(p, f, J, [this, br](const V& q, double* g) { value_on(q, br, g); });
    }
    void value_on(const V& p, bool brA, double* f) const {
        double ru = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), rv = std::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]);
        if (brA) {
            double s = eps * cutoff(ru / k) / std::pow(std::max(ru, 1e-12), 2);
            for (int i = 0; i < 3; i++) f[i] = p[3 + i] - s * p[i];
        } else {
            double s = eps * cutoff(rv / k) / std::pow(std::max(rv, 1e-12), 2);
            for (int i = 0; i < 3; i++) f[i] = p[i] - s * p[3 + i];
        }
    }
    int world_of(const V& p) const {
        double ru = p[0] * p[0] + p[1] * p[1] + p[2] * p[2], rv = p[3] * p[3] + p[4] * p[4] + p[5] * p[5];
        return rv > ru ? 1 : 0;
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w0 = world_of(p), w = world_of(q);
        const double* x1 = &q[w ? 3 : 0];
        const double* x0 = &p[w0 ? 3 : 0];
        for (auto& bl : balls) {
            if (bl.world != w) continue;
            double dx = x1[0] - bl.c[0], dy = x1[1] - bl.c[1], dz = x1[2] - bl.c[2], l = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (l < bl.r) { c = scale(bl.col, lambert(dx / l, dy / l, dz / l)); return true; }
        }
        if (x0[1] > floorh && x1[1] <= floorh) { c = floor_color(WORLD[w], lerp_hit(x0, x1, floorh, 0), lerp_hit(x0, x1, floorh, 2), dist); return true; }
        return false;
    }
    double h_at(const V& p, double) const override {
        double ru = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), rv = std::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]);
        double fd = std::max(ru, rv) - 3.0;
        return fd <= 0 ? 0.006 : clampd(0.3 * fd, 0.006, 0.5);
    }
    RGB sky(const V& p, const V&) const override {
        RGB b = WORLD[world_of(p)];
        return {(0.55 + 0.45 * b.r) * 0.85, (0.55 + 0.45 * b.g) * 0.85, (0.55 + 0.45 * b.b) * 0.85};
    }
};


static double dsmootherstep(double e0, double e1, double x) {
    double t = (x - e0) / (e1 - e0); if (t <= 0 || t >= 1) return 0;
    return 30 * t * t * (1 - t) * (1 - t) / (e1 - e0);
}

// ---------------------------------------------------------------- flat portal: n^N = rho^N W(x), flat except at the rim
struct FlatGate : Scene {
    int N; double rho = 0.7, R = 1.6, a = 0.3, T = 0.8, floorh = -2.3; bool frame;
    std::vector<Ball> balls;
    FlatGate(int N_, bool frame_, bool back) : N(N_), frame(frame_) {
        D = 5; K = 2; fov = 70; max_len = 150;
        balls = {{0, {2.6, -1.6, 2.0}, 3, 0.7, {0.9, 0.2, 0.2}}, {1, {-0.5, -1.5, 4.0}, 3, 0.8, {0.95, 0.9, 0.2}}, {2, {0.7, -1.5, 3.5}, 3, 0.8, {0.9, 0.3, 0.9}}};
        double z0 = back ? 6.0 : -6.0;
        on_world(0.0, -0.4, z0, 0, cam);
        look = {0, 0, back ? -1.0 : 1.0, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {back ? -1.0 : 1.0, 0, 0, 0, 0, 0};
    }
    // W and its gradient: inside the cylinder r < R-a it depends on x3 only, outside r > R+a it is 1
    cd W(const V& p, cd* g) const {
        double r = std::hypot(p[0], p[1]), x3 = p[2];
        double q = 2 * smootherstep(-T, T, x3) - 1, dq = 2 * dsmootherstep(-T, T, x3);
        cd e = std::exp(cd(0, -M_PI * q)), Wc = -e, dWc = cd(0, M_PI * dq) * e;
        double chi = 1 - smootherstep(R - a, R + a, r), dchi = -dsmootherstep(R - a, R + a, r);
        if (g) {
            double rr = std::max(r, 1e-12);
            g[0] = dchi * (p[0] / rr) * (Wc - 1.0); g[1] = dchi * (p[1] / rr) * (Wc - 1.0); g[2] = chi * dWc;
        }
        return 1.0 + chi * (Wc - 1.0);
    }
    void on_world(double x0, double x1, double x2, int k, V& p) const {
        V t = {x0, x1, x2, 0, 0, 0}; cd w = W(t, nullptr);
        cd z = rho * std::pow(std::abs(w), 1.0 / N) * std::exp(cd(0, (std::arg(w) + 2 * M_PI * k) / N));
        p = {x0, x1, x2, z.real(), z.imag(), 0};
    }
    int sheet(const V& p) const {
        cd w = W(p, nullptr), z(p[3], p[4]);
        cd base = rho * std::pow(std::abs(w), 1.0 / N) * std::exp(cd(0, std::arg(w) / N));
        int best = 0; double bd = 1e300;
        for (int k = 0; k < N; k++) { double d = std::abs(base * std::exp(cd(0, 2 * M_PI * k / N)) - z); if (d < bd) { bd = d; best = k; } }
        return best;
    }
    double ring_dist(const V& p) const { return std::hypot(std::hypot(p[0], p[1]) - R, p[2]); }
    void value(const V& p, double* f) const override { double J[KMAX][DMAX]; jet(p, f, J); }
    void jet(const V& p, double* f, double J[KMAX][DMAX]) const override {
        cd g[3]; cd w = W(p, g), z(p[3], p[4]);
        double s = N * std::pow(rho, N - 1), rN = std::pow(rho, N);
        cd F = (std::pow(z, N) - rN * w) / s;
        f[0] = F.real(); f[1] = F.imag();
        for (int k = 0; k < 3; k++) { cd d = -rN * g[k] / s; J[0][k] = d.real(); J[1][k] = d.imag(); }
        cd dz = std::pow(z / rho, N - 1), dzi = cd(0, 1) * dz;
        J[0][3] = dz.real(); J[1][3] = dz.imag(); J[0][4] = dzi.real(); J[1][4] = dzi.imag();
    }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = sheet(q);
        for (auto& b : balls) {
            if (b.world >= N || b.world != w) continue;
            double dx = q[0] - b.c[0], dy = q[1] - b.c[1], dz = q[2] - b.c[2], l = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (l < b.r) { c = scale(b.col, lambert(dx / l, dy / l, dz / l)); return true; }
        }
        if (p[1] > floorh && q[1] <= floorh) { c = floor_color(WORLD[w], lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), dist); return true; }
        if (frame && ring_dist(q) < 0.06) { c = {0.85, 0.7, 0.25}; return true; }
        return false;
    }
    double obj_dist(const V& p) const override { return balls_dist(balls, &p[0]); }
    RGB sky(const V& p, const V& d) const override { return sky_mix(WORLD[sheet(p)], d[1]); }
    double h_at(const V& p, double) const override {
        double r = std::hypot(p[0], p[1]);
        double dr = std::max(0.0, r - (R + a)), dz = std::max(0.0, std::fabs(p[2]) - T);
        double d = std::hypot(dr, dz);                              // distance to the region where sheets bend
        if (d == 0) return clampd(0.15 * ring_dist(p), 0.003, 0.02);
        return clampd(0.3 * d, 0.02, 0.5);
    }
};

// ---------------------------------------------------------------- three parallel 3D worlds, ONE three-way wormhole
struct YJunction5 : Scene {
    double rho = 2.4, r = 0.8, kj = 0.6, km = 0.45, m = 0.5, floorh = -1.7;
    double d[3][2];
    std::vector<Ball> balls;
    YJunction5() {
        D = 5; K = 2; fov = 70; max_len = 150;
        for (int k = 0; k < 3; k++) { double t = M_PI / 2 + 2 * M_PI * k / 3; d[k][0] = std::cos(t); d[k][1] = std::sin(t); }
        cam = {0.0, -0.3, -4.0, rho * d[0][0], rho * d[0][1], 0};
        look = {0, 0, 1, 0, 0, 0}; up = {0, 1, 0, 0, 0, 0}; right = {1, 0, 0, 0, 0, 0};
        balls = {{0, {2.2, -1.2, 1.5}, 3, 0.5, {0.9, 0.2, 0.2}}, {1, {-0.9, -1.2, 2.5}, 3, 0.55, {0.95, 0.9, 0.2}}, {2, {0.9, -1.2, 2.5}, 3, 0.55, {0.9, 0.3, 0.9}}};
    }
    // the page's arm(): a 5D capsule from the centre to the world, carrier turning from z = 0 to the world's slice
    World arm(const V& p, int k) const {
        double along = p[3] * d[k][0] + p[4] * d[k][1], side = -p[3] * d[k][1] + p[4] * d[k][0];
        double t = clampd(along / rho, 0, 1), extra = along - t * rho;
        double tube = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + side * side + extra * extra) - r;
        double angle = 0.5 * M_PI * smootherstep(0.3, 0.7, along / rho);
        return world(tube, turn(p[2], side, angle));
    }
    // the page's parallel_world(): solid just beyond the world, and only near its own slice
    World parallel_world(const V& p, int k) const {
        double along = p[3] * d[k][0] + p[4] * d[k][1], side = -p[3] * d[k][1] + p[4] * d[k][0];
        return world(extrude(side, rho - along, m), side);
    }
    void value(const V& p, double* f) const override {
        World F = smooth_union(smooth_union(arm(p, 0), arm(p, 1), kj), arm(p, 2), kj);
        for (int k = 0; k < 3; k++) F = smooth_union(F, parallel_world(p, k), km);
        f[0] = F.solid; f[1] = F.carrier;
    }
    int world_of(const V& p) const {
        int best = 0; double bd = 1e300;
        for (int k = 0; k < 3; k++) { double dd = std::hypot(p[3] - rho * d[k][0], p[4] - rho * d[k][1]); if (dd < bd) { bd = dd; best = k; } }
        return best;
    }
    bool in_tube(const V& p) const { return std::hypot(p[3], p[4]) < rho - 0.4; }
    bool hit(const V& p, const V& q, double dist, RGB& c) const override {
        int w = world_of(q);
        if (!in_tube(q)) for (auto& b : balls) {
            if (b.world != w) continue;
            double dx = q[0] - b.c[0], dy = q[1] - b.c[1], dz = q[2] - b.c[2], l = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (l < b.r) { c = scale(b.col, lambert(dx / l, dy / l, dz / l)); return true; }
        }
        if (p[1] > floorh && q[1] <= floorh) { c = floor_color(WORLD[w], lerp_hit(p, q, floorh, 0), lerp_hit(p, q, floorh, 2), dist); return true; }
        return false;
    }
    double obj_dist(const V& p) const override { return balls_dist(balls, &p[0]); }
    RGB sky(const V& p, const V& dd) const override { return sky_mix(WORLD[world_of(p)], dd[1]); }
    double h_at(const V& p, double) const override {
        double x = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        double fd = x - (r + 1.2);                                   // distance from the mouth on a world
        if (in_tube(p) || fd <= 0) return 0.006;
        return clampd(0.3 * fd, 0.006, 0.5);
    }
};

// ---------------------------------------------------------------- kernel (codim K, Gram solve)
struct Kernel {
    const Scene& sc;
    double J[KMAX][DMAX];
    explicit Kernel(const Scene& s) : sc(s) {}
    void gram_solve(const double* r, double* y) const { gram_solve_J(J, r, y); }
    // y = G^{-1} r with G = J J^T
    void gram_solve_J(const double J[KMAX][DMAX], const double* r, double* y) const {
        int K = sc.K, D = sc.D;
        double G[KMAX][KMAX];
        for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) { double s = 0; for (int d = 0; d < D; d++) s += J[i][d] * J[j][d]; G[i][j] = s; }
        if (K == 2) {
            double det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
            y[0] = (G[1][1] * r[0] - G[0][1] * r[1]) / det;
            y[1] = (G[0][0] * r[1] - G[1][0] * r[0]) / det;
        } else {   // K == 3, Cramer
            double a = G[0][0], b = G[0][1], c = G[0][2], d = G[1][0], e = G[1][1], f = G[1][2], g = G[2][0], h = G[2][1], i = G[2][2];
            double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
            double det = a * A + b * B + c * C;
            double inv[3][3] = {{A, -(b * i - c * h), b * f - c * e}, {B, a * i - c * g, -(a * f - c * d)}, {C, -(a * h - b * g), a * e - b * d}};
            for (int m = 0; m < 3; m++) y[m] = (inv[m][0] * r[0] + inv[m][1] * r[1] + inv[m][2] * r[2]) / det;
        }
    }
    void project(V& q, int iters) {
        double f[KMAX], y[KMAX];
        for (int it = 0; it < iters; it++) {
            sc.jet(q, f, J);
            gram_solve(f, y);
            for (int d = 0; d < sc.D; d++) { double s = 0; for (int k = 0; k < sc.K; k++) s += J[k][d] * y[k]; q[d] -= s; }
        }
    }
    // one Newton correction of q using the Jacobian Js (value taken at q)
    void correct(V& q, const double Js[KMAX][DMAX]) const {
        double f[KMAX], y[KMAX];
        sc.value(q, f);
        gram_solve_J(Js, f, y);
        for (int d = 0; d < sc.D; d++) { double s = 0; for (int k = 0; k < sc.K; k++) s += Js[k][d] * y[k]; q[d] -= s; }
    }
    V tangent(const V& v) const {
        double r[KMAX], y[KMAX];
        for (int k = 0; k < sc.K; k++) { double s = 0; for (int d = 0; d < sc.D; d++) s += J[k][d] * v[d]; r[k] = s; }
        gram_solve(r, y);
        V t = v;
        for (int d = 0; d < sc.D; d++) { double s = 0; for (int k = 0; k < sc.K; k++) s += J[k][d] * y[k]; t[d] -= s; }
        return t;
    }
};
static double dotv(const V& a, const V& b, int D) { double s = 0; for (int i = 0; i < D; i++) s += a[i] * b[i]; return s; }
static V normv(V a, int D) { double l = std::sqrt(dotv(a, a, D)); for (int i = 0; i < D; i++) a[i] /= l; return a; }

int main(int argc, char** argv) {
    if (argc < 7 && !(argc == 3 && std::string(argv[2]) == "eval")) { std::fprintf(stderr, "usage: trace3d scene W H spp n_steps out | trace3d scene eval < points\n"); return 1; }
    std::string name = argv[1];
    int W = argc > 3 ? std::atoi(argv[2]) : 0, H = argc > 3 ? std::atoi(argv[3]) : 0, spp = argc > 4 ? std::atoi(argv[4]) : 0, n_steps = argc > 5 ? std::atoi(argv[5]) : 0;
    Scene* sc = nullptr;
    if (name == "gate_N2") sc = new Stargate(2, true, false);
    else if (name == "gate_N2_bare") sc = new Stargate(2, false, false);
    else if (name == "gate_N3") sc = new Stargate(3, true, false);
    else if (name == "gate_N3_back") sc = new Stargate(3, true, true);
    else if (name == "tubes3") sc = new TubeWorlds();
    else if (name == "throat2") sc = new FlatThroat();
    else if (name == "neck6d") sc = new InversionNeck();
    else if (name == "flat_gate2") sc = new FlatGate(2, true, false);
    else if (name == "flat_gate2_bare") sc = new FlatGate(2, false, false);
    else if (name == "flat_gate3") sc = new FlatGate(3, true, false);
    else if (name == "flat_gate3_back") sc = new FlatGate(3, true, true);
    else if (name == "y_junction") sc = new YJunction5();
    else if (name == "throat_mouth") sc = new FlatThroat(true);
    else if (name == "throat_long") sc = new LongThroat();
    else if (name == "throat_far") sc = new FlatThroat(false, true);
    else if (name == "mono_donut_front") sc = new MonoDonut(0);
    else if (name == "mono_donut_side") sc = new MonoDonut(1);
    else if (name == "mono_donut_plane") sc = new MonoDonut(2);
    else if (name == "mono_donut_inside") sc = new MonoDonut(3);
    else if (name == "mono_ball_front") sc = new MonoBall(0);
    else if (name == "mono_ball_inside") sc = new MonoBall(1);
    else if (name == "mono_ball_inside2") sc = new MonoBall(2);
    else if (name == "mono_ball_end") sc = new MonoBall(3);
    else if (name == "line_junction_0") sc = new LineJunction(0, false);
    else if (name == "line_junction_45") sc = new LineJunction(45, false);
    else if (name == "line_junction_90") sc = new LineJunction(90, false);
    else if (name == "line_junction_top") sc = new LineJunction(0, true);
    else if (name == "point_junction_0") sc = new PointJunction6(0, false);
    else if (name == "point_junction_45") sc = new PointJunction6(45, false);
    else if (name == "point_junction_90") sc = new PointJunction6(90, false);
    else if (name == "point_junction_top") sc = new PointJunction6(0, true);
    else { std::fprintf(stderr, "unknown scene\n"); return 1; }
    const int D = sc->D;
    if (std::string(argv[2]) == "eval") {           // test mode: read points (D numbers per line), print the field
        V p{}; double f[KMAX];
        while (true) {
            for (int i = 0; i < D; i++) if (std::scanf("%lf", &p[i]) != 1) return 0;
            sc->value(p, f);
            std::printf("%.17g %.17g\n", f[0], f[1]);
        }
    }
    // camera frame
    Kernel k0(*sc);
    V cam = sc->cam;
    k0.project(cam, 8);
    { double f[KMAX]; sc->jet(cam, f, k0.J); }
    V fwd = normv(k0.tangent(sc->look), D);
    V up = k0.tangent(sc->up); { double s = dotv(up, fwd, D); for (int i = 0; i < D; i++) up[i] -= s * fwd[i]; } up = normv(up, D);
    V rt = k0.tangent(sc->right);
    { double s1 = dotv(rt, fwd, D), s2 = dotv(rt, up, D); for (int i = 0; i < D; i++) rt[i] -= s1 * fwd[i] + s2 * up[i]; } rt = normv(rt, D);
    const double t = sc->fov_v > 0 ? std::tan(sc->fov_v * M_PI / 360.0) * W / H : std::tan(sc->fov * M_PI / 360.0);
    // the camera itself, a ball just behind the eye: walk back along the world from the eye
    const double self_r = std::getenv("NOSELF") ? 0.0 : sc->self_r;
    V self_c = cam;
    if (self_r > 0) {
        Kernel kb(*sc);
        V dir; for (int i = 0; i < D; i++) dir[i] = -fwd[i];
        { double f[KMAX]; sc->jet(self_c, f, kb.J); }
        const int n = 200; const double hs = (self_r + 0.06) / n;
        for (int i = 0; i < n; i++) {
            V q; for (int k = 0; k < D; k++) q[k] = self_c[k] + dir[k] * hs;
            kb.correct(q, kb.J); { double f[KMAX]; sc->jet(q, f, kb.J); } kb.correct(q, kb.J); { double f[KMAX]; sc->jet(q, f, kb.J); }
            V ch; for (int k = 0; k < D; k++) ch[k] = q[k] - self_c[k];
            dir = normv(kb.tangent(ch), D); self_c = q;
        }
    }
    // its texture, in the eye's frame: red, with a white nose in front, a white ear on the left, a yellow top
    auto self_color = [&](const V& q) -> RGB {
        V n; double l = 0; for (int i = 0; i < D; i++) { n[i] = q[i] - self_c[i]; l += n[i] * n[i]; }
        l = std::sqrt(l); for (int i = 0; i < D; i++) n[i] /= l;
        double nf = dotv(n, fwd, D), nu = dotv(n, up, D), nr = dotv(n, rt, D);
        RGB c = {0.85, 0.12, 0.12};
        if (nf > 0.8 || -nr > 0.8) c = {0.97, 0.97, 0.97};
        else if (nu > 0.8) c = {0.95, 0.85, 0.2};
        return scale(c, 0.62 + 0.38 * clampd(0.5 + 0.3 * nf + 0.5 * nu - 0.3 * nr, 0, 1));
    };
    const double dt0 = sc->max_len / n_steps;
    if (const char* fg = std::getenv("FOG")) FOG = std::atof(fg);
    std::vector<float> img((size_t)W * H * 3, 0.f);
    long long total_steps = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total_steps)
    for (int py = 0; py < H; py++) {
        std::mt19937_64 rng(1234567ULL + py);
        std::uniform_real_distribution<double> U01(0, 1);
        Kernel kn(*sc);
        for (int px = 0; px < W; px++) {
            RGB acc{0, 0, 0};
            for (int s = 0; s < spp; s++) {
                double u = ((px + U01(rng)) / W * 2 - 1) * t;
                double v = -((py + U01(rng)) / H * 2 - 1) * t * H / W;
                V d; for (int i = 0; i < D; i++) d[i] = fwd[i] + u * rt[i] + v * up[i];
                d = normv(d, D);
                V p = cam;
                double Js[KMAX][DMAX];
                std::memcpy(Js, k0.J, sizeof(Js));            // jet at the camera
                double dist = 0; RGB c{0, 0, 0}; bool done = false;
                for (int it = 0; it < 8 * n_steps; it++) {
                    double h = std::min(sc->h_at(p, dt0), std::max(0.004, 0.8 * sc->obj_dist(p)));
                    if (self_r > 0) {
                        double l2 = 0; for (int i = 0; i < D; i++) l2 += (p[i] - self_c[i]) * (p[i] - self_c[i]);
                        h = std::min(h, std::max(0.004, 0.8 * (std::sqrt(l2) - self_r)));
                    }
                    V q; for (int i = 0; i < D; i++) q[i] = p[i] + d[i] * h;
                    kn.correct(q, Js);                          // along the normals at the start
                    { double f[KMAX]; sc->jet(q, f, kn.J); }
                    kn.correct(q, kn.J);                        // with fresh normals
                    { double f[KMAX]; sc->jet(q, f, kn.J); }    // normals at the new point
                    V ch; for (int i = 0; i < D; i++) ch[i] = q[i] - p[i];
                    V dn = normv(kn.tangent(ch), D);          // chord update
                    std::memcpy(Js, kn.J, sizeof(Js));        // reused as the next start normals
                    dist += h; total_steps++;
                    if (self_r > 0) {
                        double l2 = 0; for (int i = 0; i < D; i++) l2 += (q[i] - self_c[i]) * (q[i] - self_c[i]);
                        if (l2 < self_r * self_r) { c = self_color(q); double fog = std::exp(-FOG * dist); c = {c.r * fog + (1 - fog) * 0.85, c.g * fog + (1 - fog) * 0.85, c.b * fog + (1 - fog) * 0.85}; done = true; break; }
                    }
                    if (sc->hit(p, q, dist, c)) { done = true; break; }
                    p = q; d = dn;
                    if (dist > sc->max_len) break;
                }
                if (!done) c = sc->sky(p, d);
                acc.r += c.r; acc.g += c.g; acc.b += c.b;
            }
            size_t o = ((size_t)py * W + px) * 3;
            img[o] = acc.r / spp; img[o + 1] = acc.g / spp; img[o + 2] = acc.b / spp;
        }
    }
    FILE* fo = std::fopen(argv[6], "wb");
    std::fwrite(img.data(), sizeof(float), img.size(), fo);
    std::fclose(fo);
    std::fprintf(stderr, "%s: %dx%d spp=%d steps=%d, %.3g ray-steps\n", name.c_str(), W, H, spp, n_steps, (double)total_steps);
    return 0;
}
