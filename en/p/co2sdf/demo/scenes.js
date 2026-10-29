/* scenes.js — the demo worlds of the page: 2D surfaces in R^4, each built from
 * the page's small library (world, smooth_union, extrude, turn, neck_union,
 * roots, graph) on top of portal_gravity's ball / extrude / smin.
 * A scene has F(p, out), color(p, rgb), optionally J(p, a, b), the camera
 * position cam with hint vectors hint1, hint2 for its frame, and rMax.  For
 * the demo also: params (slider specs), update() (after a param change), box()
 * (region sampled for the outside view; centre() is its midpoint), view
 * (initial 4D view rotation) and code (its pseudocode, as on the page).
 * co2scenes(K) takes the kernel K = co2sdf() and returns fresh instances. */
function co2scenes(K) {
  "use strict";
  const { cx, cset, cabs, uOf, vOf, cmul, cdiv, cpow, jac, smin, union, neckUnion, neckUnionOn,
          cutoff, cone, coneAngle, smootherstep, ball2, ball3, extrude, turn, capsule4, world,
          smoothUnion, roots } = K;
  const DEG = Math.PI / 180;

  // ---------------------------------------------------------------- colours
  const GREEN = [0.30, 0.75, 0.25], ORANGE = [0.95, 0.55, 0.15];
  const BLUE = [0.25, 0.50, 0.95], PURPLE = [0.70, 0.35, 0.85];
  const TEAL = [0.2, 0.7, 0.7], RED = [0.9, 0.15, 0.15], GREY = [0.75, 0.75, 0.75];
  const WORLD_COLORS = [GREEN, ORANGE, BLUE, PURPLE];
  const PLANE_COLORS = [[0.2, 0.8, 0.1], [0.9, 0.6, 0.1], [0.1, 0.6, 0.9]];   // wormhole_between_two_planes' colours

  /** Checkerboard + thin grid lines + chiral 'L' marks so mirroring is visible. */
  function shade(rgb, base, x, y, size = 1) {
    const X = Math.floor(x / size), Y = Math.floor(y / size);
    const fx = x / size - X, fy = y / size - Y;
    const c = 0.78 + 0.22 * Math.abs((X + Y) % 2);
    const grid = Math.min(fx, 1 - fx) < 0.03 || Math.min(fy, 1 - fy) < 0.03;
    // chiral L glyph in every cell: vertical bar at left + foot at the bottom
    const L = fx > 0.25 && fy > 0.25 && ((fx < 0.35 && fy < 0.75) || (fx < 0.6 && fy < 0.35));
    for (let k = 0; k < 3; k++) {
      let v = base[k] * c;
      if (grid) v *= 0.55;
      if (L) v = v * 0.25 + 0.75;
      rgb[k] = v;
    }
    return rgb;
  }

  const wrap = (x) => Math.atan2(Math.sin(x), Math.cos(x));   // angle(exp(i x))
  const param = (key, label, min, max, step) => ({ key, label, min, max, step });

  class Scene {
    cam = [0, 0, 0, 0];
    hint1 = [1, 0, 0, 0];
    hint2 = [0, 1, 0, 0];
    rMax = 6;
    params = [];
    view = [[1, 2, -60]];       // [axis i, axis j, degrees] rotations of the view
    code = "";
    update() {}                 // recompute derived constants after a param change
    box() { return [[-4, 4], [-4, 4], [-4, 4], [-4, 4]]; }
    centre() { return this.box().map(([lo, hi]) => (lo + hi) / 2); }
  }

  // ================================================================ primitives

  class Plane extends Scene {
    name = "plane"; title = "Plane";
    cam = [0.2, 0.3, 0, 0];
    view = [[1, 2, -60], [0, 2, 15]];
    code = `vec2 plane(vec4 p) { return world(p.z, p.w); }   // the xy-plane: solid z < 0, carrier w = 0`;
    F(p, o) { return world(o, p[2], p[3]); }
    color(p, rgb) { return shade(rgb, GREEN, p[0], p[1]); }
    box() { return [[-5, 5], [-5, 5], [-1, 1], [-1, 1]]; }
  }

  class Sphere extends Scene {
    name = "sphere"; title = "Sphere";
    r = 1.5;
    cam = [1.5, 0, 0, 0];
    hint1 = [0, 1, 0, 0];
    hint2 = [0, 0, 1, 0];
    params = [param("r", "r", 0.5, 3, 0.01)];
    view = [[1, 2, -70], [0, 2, 20]];
    code = `vec2 sphere(vec4 p, float r) { return world(ball(p.xyz, r), p.w); }   // a ball, inside the slice w = 0`;
    update() { this.cam = [this.r, 0, 0, 0]; }
    F(p, o) { return world(o, ball3(p[0], p[1], p[2], this.r), p[3]); }
    color(p, rgb) {
      const lon = Math.atan2(p[1], p[0]), lat = Math.asin(Math.max(-1, Math.min(1, p[2] / this.r)));
      return shade(rgb, GREEN, lon * 8 / (2 * Math.PI), lat * 6 / Math.PI);
    }
    box() { const s = this.r + 0.3; return [[-s, s], [-s, s], [-s, s], [-0.5, 0.5]]; }
  }

  /** One 4D donut solid, cut by a turning carrier: angle 0 gives the ordinary
   *  donut (carrier w), 90 degrees the flat torus (carrier |xy| - a). */
  class Torus extends Scene {
    name = "torus"; title = "Torus: donut to flat";
    a = 1.0; b = 0.5; angle = 0;
    cam = [1.0, 0, 0.5, 0];     // on top of the tube: on the torus for every angle
    hint1 = [0, 1, 0, 0];
    hint2 = [1, 0, 0, -1];
    params = [param("angle", "angle °", 0, 90, 1), param("a", "a", 0.8, 2, 0.01), param("b", "b", 0.2, 0.7, 0.01)];
    view = [[1, 2, -60], [0, 3, 20]];
    code = `// One 4D donut, cut by a carrier that turns.
// angle 0:  the slice w = 0 cuts out an ordinary donut.
// angle 90: the cylinder |xy| = a cuts out the flat torus.
vec2 torus(vec4 p, float a, float b, float angle) {
    float solid = ball(vec3(ball(p.xy, a), p.z, p.w), b);   // the donut, with w inside the ball
    return world(solid, turn(p.w, ball(p.xy, a), angle));   // 0: a donut; 90 deg: the flat torus
}`;
    update() { this.cam = [this.a, 0, this.b, 0]; }
    F(p, o) {
      const ring = ball2(p[0], p[1], this.a);                 // ball(p.xy, a)
      return world(o, ball3(ring, p[2], p[3], this.b), turn(p[3], ring, this.angle * DEG));
    }
    color(p, rgb) {
      const th = this.angle * DEG, ring = ball2(p[0], p[1], this.a);
      const e = Math.cos(th) * ring - Math.sin(th) * p[3];   // in-plane coordinate of the tube's circle
      const phi = Math.atan2(p[1], p[0]), psi = Math.atan2(p[2], e);
      shade(rgb, GREEN, phi * 8 / (2 * Math.PI), psi * 6 / (2 * Math.PI));
      if (Math.hypot(wrap(phi - 2.0), wrap(psi - 1.0) * 0.6) < 0.25) rgb.set(PURPLE);   // a spot on the torus (red is the camera)
      return rgb;
    }
    box() { const s = this.a + this.b + 0.3, t = this.b + 0.3; return [[-s, s], [-s, s], [-t, t], [-t, t]]; }
  }

  class Clifford extends Scene {
    name = "clifford"; title = "Flat torus";
    a = 1.0; b = 0.7;
    cam = [1.0, 0.0, 0.7, 0.0];
    hint1 = [0, 1, 0, 0];
    hint2 = [0, 0, 0, 1];
    params = [param("a", "a", 0.3, 2, 0.01), param("b", "b", 0.3, 2, 0.01)];
    view = [[0, 3, 30], [1, 2, -50]];
    code = `vec2 flat_torus(vec4 p, float a, float b) { return world(ball(p.xy, a), ball(p.zw, b)); }   // a circle in xy and a circle in zw`;
    F(p, o) { return world(o, ball2(p[0], p[1], this.a), ball2(p[2], p[3], this.b)); }
    color(p, rgb) {
      const a = this.a, b = this.b;
      const th = Math.atan2(p[1], p[0]), ph = Math.atan2(p[3], p[2]);
      // tiles that exactly fit the two circumferences
      shade(rgb, GREEN, a * th * 8 / (2 * Math.PI * a), b * ph * 6 / (2 * Math.PI * b));
      if (Math.hypot(wrap(th - 2.0) * a, wrap(ph - 1.0) * b) < 0.35) rgb.set(PURPLE);   // a spot on the torus (red is the camera)
      return rgb;
    }
    box() { const a = this.a + 0.3, b = this.b + 0.3; return [[-a, a], [-a, a], [-b, b], [-b, b]]; }
  }

  // ================================================================ joining on the solid

  /** portal_gravity's wormhole_between_two_planes, unchanged, in the slice w = 0. */
  class TwoPlanes extends Scene {
    name = "two_planes"; title = "Wormhole between two planes";
    height = 1.0; wormhole_size = 0.15; smoothing = 0.1;
    rMax = 1.5;
    cam = [0.26, 0, 0.30, 0];
    hint1 = [1, 0, 0, 0];
    hint2 = [0, 0, 1, 0];
    params = [param("height", "height", 0.3, 2, 0.01), param("wormhole_size", "size", 0.05, 0.4, 0.005),
              param("smoothing", "smoothing", 0.02, 0.3, 0.005)];
    view = [[0, 2, 35], [1, 2, -20]];
    code = `// a 2D wormhole between two planes, as a 2D world in the slice w = 0
vec2 two_planes_wormhole(vec4 p) {
    float d1 = p.y;
    float d2 = p.y - height;
    float d3 = ball(p.xz, wormhole_size);
    d3 = extrude(p.y - height / 2.0, d3, height);
    float d4 = smin(-d2, d3, smoothing);
    return world(smin(d1, d4, smoothing), p.w);
}`;
    F(p, o) {
      const y = p[1], d1 = y, d2 = y - this.height;
      let d3 = ball2(p[0], p[2], this.wormhole_size);
      d3 = extrude(y - this.height / 2, d3, this.height);
      const d4 = smin(-d2, d3, this.smoothing);
      return world(o, smin(d1, d4, this.smoothing), p[3]);
    }
    color(p, rgb) {
      const y = p[1], base = y < this.smoothing ? PLANE_COLORS[0] : y > this.height - this.smoothing ? PLANE_COLORS[1] : PLANE_COLORS[2];
      return shade(rgb, base, p[0], p[2], 0.1);
    }
    box() { return [[-1.5, 1.5], [-0.3, this.height + 0.3], [-1.5, 1.5], [-0.3, 0.3]]; }
  }

  /** One pair of ThreeParallel: worlds i and j lie in one 3D slice; there they
   *  are two parallel planes joined by a tube (the two-planes wormhole). */
  class Pair {
    constructor(sc, i, j, q) {
      const ci = sc.c[i], cj = sc.c[j];
      this.sc = sc; this.ci = ci; this.q = q;
      this.e = [(cj[0] - ci[0]) / sc.L, (cj[1] - ci[1]) / sc.L];
    }
    F(p, o) {
      const { sc, ci, e, q } = this;
      const t = (p[2] - ci[0]) * e[0] + (p[3] - ci[1]) * e[1];      // 0 on world i, L on world j
      const s = -(p[2] - ci[0]) * e[1] + (p[3] - ci[1]) * e[0];     // 0 on the slice that holds both worlds
      const worlds = Math.min(t, sc.L - t);                         // everything beyond the two worlds
      const tube = ball3(p[0] - q[0], p[1] - q[1], s, sc.r);        // its ends are inside the worlds' solid
      return world(o, smin(worlds, tube, sc.k), s);
    }
  }

  /** Three parallel planes R^2 x {c_k}; every pair joined by a tube in the
   *  3-space that contains both planes; the three pairs combined with union. */
  class ThreeParallel extends Scene {
    name = "three_parallel"; title = "Three parallel worlds, pairwise";
    L = 2.4; r = 0.55; k = 0.35;
    q = [[-1.6, 0.6], [1.6, 0.6], [0.0, -1.6]];       // tubes of pairs (0,1), (1,2), (2,0)
    params = [param("L", "L", 1.5, 4, 0.01), param("r", "r", 0.2, 1, 0.01), param("k", "k", 0.05, 0.8, 0.01)];
    view = [[1, 2, -75], [0, 2, 25]];
    code = `// worlds i and j lie in one 3D slice; there they are two planes, joined by a tube
vec2 wormhole_between(vec4 p, vec2 ci, vec2 cj, vec2 mouth) {
    vec2  e = normalize(cj - ci);
    float t = dot(p.zw - ci, e);                   // 0 on world i, L on world j
    float s = dot(p.zw - ci, vec2(-e.y, e.x));     // 0 on the slice that holds both worlds: the carrier
    float worlds = min(t, L - t);                  // d1 and -d2: everything beyond the two worlds
    float tube = ball(vec3(p.xy - mouth, s), r);   // d3; its ends are inside the worlds' solid
    return world(smin(worlds, tube, k), s);
}
vec2 three_worlds(vec4 p) {
    return union(wormhole_between(p, c0, c1, mouth01),
           union(wormhole_between(p, c1, c2, mouth12),
                 wormhole_between(p, c2, c0, mouth20)));
}`;
    v = [cx(), cx(), cx()];
    constructor() { super(); this.update(); }
    update() {
      const rho = this.L / Math.sqrt(3);
      this.c = [0, 1, 2].map((k) => {
        const a = Math.PI / 2 + 2 * Math.PI * k / 3;
        return [rho * Math.cos(a), rho * Math.sin(a)];
      });
      this.cam = [0, 0, this.c[0][0], this.c[0][1]];
      this.pieces = [[0, 1], [1, 2], [2, 0]].map(([i, j], n) => new Pair(this, i, j, this.q[n]));
    }
    F(p, o) {
      const P = this.pieces, v = this.v;
      return union(o, P[0].F(p, v[0]), union(v[1], P[1].F(p, v[1]), P[2].F(p, v[2])));
    }
    J(p, a, b) {
      // hard union: the winner changes phase discontinuously, so differentiate
      // the winning piece only (what analytic gradients would do automatically)
      let win = 0, best = Infinity;
      for (let n = 0; n < 3; n++) {
        const re = this.pieces[n].F(p, this.v[n])[0];
        if (re < best) { best = re; win = n; }
      }
      jac(this.pieces[win], p, a, b);
    }
    color(p, rgb) {
      let w = 0, dmin = Infinity;
      for (let k = 0; k < 3; k++) {
        const dx = p[2] - this.c[k][0], dy = p[3] - this.c[k][1], d = Math.sqrt(dx * dx + dy * dy);
        if (d < dmin) { dmin = d; w = k; }
      }
      return shade(rgb, WORLD_COLORS[w], p[0], p[1], 0.8);
    }
    box() { const z = this.L / Math.sqrt(3) + 0.8; return [[-3.5, 3.5], [-3.5, 3.5], [-z, z], [-z, z]]; }
  }

  // ================================================================ turning the carrier

  /** Three parallel worlds R^2 x {rho d_k} joined by ONE three-way wormhole:
   *  three 4D capsules from the origin to the worlds, whose carriers turn from
   *  y = 0 (at the centre, where the arms share a 3D slice) to the world's own
   *  slice; each world's solid is a half-space extruded only a little sideways. */
  class YJunction extends Scene {
    name = "y_junction"; title = "Three worlds, one wormhole";
    rho = 3.0; r = 0.7; kj = 0.6; km = 0.4; m = 0.5;
    rMax = 12;
    hint1 = [0, 1, 0, 0];
    hint2 = [-1, 0, 0, 0];
    params = [param("rho", "ρ", 1.5, 5, 0.01), param("r", "r", 0.3, 1.1, 0.01), param("kj", "kj", 0.05, 1.2, 0.01)];
    view = [[1, 2, -75], [0, 2, 25]];
    code = `// three worlds at n = rho * d[i] in the extra plane n = p.zw, joined by one wormhole
vec2 arm(vec4 p, vec2 d) {                               // a tube from the centre to the world in the direction d
    float along = dot(p.zw, d), side = dot(p.zw, vec2(-d.y, d.x));
    float tube  = capsule(p, vec4(0.0), vec4(0.0, 0.0, rho * d), r);
    float angle = radians(90.0) * smoothstep(0.3, 0.7, along / rho);
    return world(tube, turn(p.y, side, angle));          // carrier: y = 0 at the centre, the world's own slice at the world
}
vec2 parallel_world(vec4 p, vec2 d) {                    // the world n = rho * d
    float along = dot(p.zw, d), side = dot(p.zw, vec2(-d.y, d.x));
    return world(extrude(side, rho - along, 0.5), side); // solid: just beyond the world, and only near its own slice
}
vec2 three_worlds_one_wormhole(vec4 p) {
    vec2 f = smooth_union(smooth_union(arm(p, d[0]), arm(p, d[1]), kj), arm(p, d[2]), kj);
    for (int i = 0; i < 3; i++) f = smooth_union(f, parallel_world(p, d[i]), km);
    return f;
}`;
    d = [0, 1, 2].map((k) => { const t = Math.PI / 2 + 2 * Math.PI * k / 3; return [Math.cos(t), Math.sin(t)]; });
    g = cx();
    constructor() { super(); this.update(); }
    update() { this.cam = [0, -2.2, this.rho * this.d[0][0], this.rho * this.d[0][1]]; }
    arm(p, k, o) {
      const d = this.d[k], along = p[2] * d[0] + p[3] * d[1], side = -p[2] * d[1] + p[3] * d[0];
      const tube = capsule4(p[0], p[1], p[2], p[3], this.rho * d[0], this.rho * d[1], this.r);
      const angle = 90 * DEG * smootherstep(0.3, 0.7, along / this.rho);
      return world(o, tube, turn(p[1], side, angle));
    }
    parallelWorld(p, k, o) {
      const d = this.d[k], along = p[2] * d[0] + p[3] * d[1], side = -p[2] * d[1] + p[3] * d[0];
      return world(o, extrude(side, this.rho - along, this.m), side);
    }
    F(p, o) {
      const g = this.g;
      smoothUnion(o, this.arm(p, 0, o), this.arm(p, 1, g), this.kj);
      smoothUnion(o, o, this.arm(p, 2, g), this.kj);
      for (let k = 0; k < 3; k++) smoothUnion(o, o, this.parallelWorld(p, k, g), this.km);
      return o;
    }
    color(p, rgb) {
      let w = 0, dmin = Infinity;
      for (let k = 0; k < 3; k++) {
        const dx = p[2] - this.rho * this.d[k][0], dy = p[3] - this.rho * this.d[k][1], d = Math.sqrt(dx * dx + dy * dy);
        if (d < dmin) { dmin = d; w = k; }
      }
      shade(rgb, WORLD_COLORS[w], p[0], p[1], 0.8);
      if (dmin > 0.35) for (let k = 0; k < 3; k++) rgb[k] = rgb[k] * 0.55 + 0.45 * GREY[k];   // the tubes
      return rgb;
    }
    box() { const z = this.rho + 0.8; return [[-4, 4], [-4, 4], [-z, z], [-z, z]]; }
  }

  /** Plane z = w = 0 with a tube from the mouth x = +R to the mouth x = -R: the
   *  upper half of a 4D donut around the circle |xz| = R (its lower half is
   *  hidden in the plane's solid).  twist = pi turns the tube's circle over
   *  through w on the way: the mirror handle. */
  class Handle extends Scene {
    name = "handle"; title = "Handle";
    R = 1.5; r = 0.5; k = 0.4; twist = 0;
    rMax = 10;
    cam = [-1.5, -2.0, 0, 0];
    hint1 = [0, 1, 0, 0];
    hint2 = [-1, 0, 0, 0];
    params = [param("r", "r", 0.2, 0.9, 0.01)];
    view = [[1, 2, -65], [0, 2, -30]];
    code = `// a tube from the mouth x = +R over the plane to the mouth x = -R
vec2 handle(vec4 p, float twist) {
    float solid = ball(vec3(ball(p.xz, R), p.y, p.w), r);   // a 4D donut around the circle |xz| = R
    float along = atan(max(p.z, 0.0), p.x);                 // 0 at x = +R, PI at x = -R
    float angle = twist * smoothstep(0.25 * PI, 0.75 * PI, along);
    return world(solid, turn(p.w, p.y, angle));
}
// twist = 0: an ordinary handle; twist = PI: a mirror handle
vec2 plane_with_handle(vec4 p, float twist) {
    float flip = twist > 0.0 && p.x < 0.0 ? -1.0 : 1.0;     // the plane, turned over where
    return smooth_union(world(p.z, flip * p.w), handle(p, twist), k);   // the turned mouth arrives
}`;
    f1 = cx(); f2 = cx();
    handle(p, o) {
      const solid = ball3(ball2(p[0], p[2], this.R), p[1], p[3], this.r);
      const along = Math.atan2(Math.max(p[2], 0), p[0]);
      const angle = this.twist * smootherstep(0.25 * Math.PI, 0.75 * Math.PI, along);
      return world(o, solid, turn(p[3], p[1], angle));
    }
    field(p, o, flipped) {
      const flip = this.twist > 0 && flipped ? -1 : 1;
      return smoothUnion(o, world(this.f1, p[2], flip * p[3]), this.handle(p, this.f2), this.k);
    }
    F(p, o) { return this.field(p, o, p[0] < 0); }
    // the plane's carrier jumps at x = 0 (mirror handle): differentiate the side the point is on
    J(p, a, b) { this.side = p[0] < 0; jac(this.locked, p, a, b); }
    locked = { F: (p, o) => this.field(p, o, this.side) };
    color(p, rgb) { return shade(rgb, p[0] < 0 ? GREEN : TEAL, p[0], p[1], 0.6); }
    box() { return [[-3.5, 3.5], [-3.5, 3.5], [-0.7, 2.3], [-0.9, 0.9]]; }
  }

  class Klein extends Handle {
    name = "klein"; title = "Mirror handle";
    twist = Math.PI;
  }

  // ================================================================ complex operations

  /** Two perpendicular planes touching at the origin; neck union turns the
   *  touching point into a wormhole: {u v = eps} near the origin. */
  class Neck extends Scene {
    name = "neck"; title = "Neck";
    eps = 0.36; k = 2.4;
    cam = [-3, 0, 0, 0];
    params = [param("eps", "ε", 0.05, 0.6, 0.01), param("k", "k", 1.6, 4, 0.05)];   // sqrt(eps) < k/2: no crease
    view = [[1, 2, -90], [2, 3, -45], [1, 2, 25], [0, 2, 30]];      // plane x1x2 as the floor, x3x4 as a wall
    code = `vec2 two_worlds_one_neck(vec4 p) {
    vec2 A = world(p.z, p.w);       // the xy-plane
    vec2 B = world(p.x, p.y);       // the zw-plane: touches A only at the origin
    return neck_union(A, B, eps, k);
}`;
    A = cx(); B = cx();
    F(p, o) { return neckUnion(o, world(this.A, p[2], p[3]), world(this.B, p[0], p[1]), this.eps, this.k); }
    // Jacobian of the branch selected at p (the selected field is not smooth across |A| = |B|)
    J(p, a, b) { this.near = Math.hypot(p[2], p[3]) < Math.hypot(p[0], p[1]); jac(this.locked, p, a, b); }
    locked = { F: (p, o) => neckUnionOn(o, world(this.A, p[2], p[3]), world(this.B, p[0], p[1]), this.eps, this.k, this.near) };
    color(p, rgb) {
      const inA = Math.hypot(p[2], p[3]) <= Math.hypot(p[0], p[1]);
      return inA ? shade(rgb, GREEN, p[0], p[1]) : shade(rgb, ORANGE, p[2], p[3]);
    }
    box() { return [[-3.5, 3.5], [-3.5, 3.5], [-3.5, 3.5], [-3.5, 3.5]]; }
  }

  /** Three planes that pairwise touch at single points (three complex lines
   *  v=0, u=0, u+v=c).  Neck unions at the three touching points. */
  class ThreeNecks extends Scene {
    name = "three_necks"; title = "Three necks";
    c = 3.0; eps = 0.3; k = 1.4;
    cam = [1.5, -1.8, 0, 0];
    params = [param("eps", "ε", 0.05, 0.45, 0.01)];   // sqrt(eps) < k/2
    view = [[1, 2, -90], [2, 3, -45], [1, 2, 25], [0, 2, 30]];
    code = `// three planes; every two of them touch at one point
vec2 three_necks(vec4 p) {
    vec2 A = world(p.z, p.w);
    vec2 B = world(p.x, p.y);
    vec2 C = world(p.x + p.z - c, p.y + p.w) / sqrt(2.0);
    return neck_union(neck_union(A, B, eps, k), C, eps, k);
}`;
    f = [cx(), cx(), cx()];
    parts(p) {                  // the worlds A, B, C
      const f = this.f, s = 1 / Math.sqrt(2);
      world(f[0], p[2], p[3]);
      world(f[1], p[0], p[1]);
      world(f[2], (p[0] + p[2] - this.c) * s, (p[1] + p[3]) * s);
      return f;
    }
    F(p, o) {
      const f = this.parts(p);
      return neckUnion(o, neckUnion(o, f[0], f[1], this.eps, this.k), f[2], this.eps, this.k);
    }
    // Jacobian with both branch choices frozen at p
    J(p, a, b) {
      const f = this.parts(p);
      this.s1 = cabs(f[0]) < cabs(f[1]);
      neckUnionOn(this.t, f[0], f[1], this.eps, this.k, this.s1);
      this.s2 = cabs(this.t) < cabs(f[2]);
      jac(this.locked, p, a, b);
    }
    t = cx();
    locked = { F: (p, o) => {
      const f = this.parts(p);
      neckUnionOn(this.t, f[0], f[1], this.eps, this.k, this.s1);
      return neckUnionOn(o, this.t, f[2], this.eps, this.k, this.s2);
    } };
    color(p, rgb) {
      const f = this.parts(p), d0 = cabs(f[0]), d1 = cabs(f[1]), d2 = cabs(f[2]);
      if (d0 <= d1 && d0 <= d2) return shade(rgb, GREEN, p[0], p[1]);
      if (d1 <= d2) return shade(rgb, ORANGE, p[2], p[3]);
      // world 3 coordinates: along the line u+v=c, use (u - v)/sqrt2
      const s = 1 / Math.sqrt(2);
      return shade(rgb, BLUE, (p[0] - p[2]) * s, (p[1] - p[3]) * s);
    }
    box() { return [[-2.5, 5], [-3.5, 3.5], [-2.5, 5], [-3.5, 3.5]]; }
  }

  /** Three planes through one point (complex lines v = 0, u = 0, v = u), the
   *  triple point smoothed: F = (u v (u - v) - eps·cutoff(r/k)) / (r² + 0.05). */
  class TriplePoint extends Scene {
    name = "triple_point"; title = "Triple point";
    eps = 0.25; k = 2.5;
    cam = [-3, 0, 0, 0];
    params = [param("eps", "ε", 0.02, 1, 0.01)];
    view = [[1, 2, -90], [2, 3, -45], [1, 2, 25], [0, 2, 30]];
    code = `// the planes u = 0, v = 0 and u = v all pass through the origin
vec2 triple_point(vec4 p) {
    vec2 u = p.xy, v = p.zw;
    vec2 all = cmul(cmul(u, v), u - v);     // zero on all three: their union, crossing at the origin
    float r2 = dot(u, u) + dot(v, v);
    vec2 open = all - vec2(eps * cutoff(sqrt(r2) / k), 0.0);   // open the crossing
    return open / (r2 + 0.05);                                  // divide to stay distance-like
}`;
    u = cx(); v = cx(); w = cx();
    F(p, o) {
      const u = uOf(this.u, p), v = vOf(this.v, p), w = this.w, r2 = cabs(u) ** 2 + cabs(v) ** 2;
      cmul(o, cmul(o, u, v), cset(w, u[0] - v[0], u[1] - v[1]));
      o[0] -= this.eps * cutoff(Math.sqrt(r2) / this.k);
      return cdiv(o, o, cset(w, r2 + 0.05, 0));
    }
    color(p, rgb) {
      const u = uOf(this.u, p), v = vOf(this.v, p), d0 = cabs(v), d1 = cabs(u);
      const d2 = cabs(cset(this.w, u[0] - v[0], u[1] - v[1])) / Math.sqrt(2), s = 1 / Math.sqrt(2);
      if (d0 <= d1 && d0 <= d2) return shade(rgb, GREEN, p[0], p[1]);
      if (d1 <= d2) return shade(rgb, ORANGE, p[2], p[3]);
      return shade(rgb, BLUE, (p[0] + p[2]) * s, (p[1] + p[3]) * s);   // along v = u: (u + v)/sqrt2
    }
    box() { return [[-3.5, 3.5], [-3.5, 3.5], [-3.5, 3.5], [-3.5, 3.5]]; }
  }

  /** N parallel worlds n = rho * (N-th roots of W(x)), n = (z, w).  W = 1 far
   *  away; walking through the slit |x| < R on the line y = 0, W goes once
   *  around 0, so n goes 1/N of the way around: into the next world. */
  class FlatPortal extends Scene {
    name = "flat_portal"; title = "Flat portal, 2 worlds";
    N = 2; rho = 1.0; R = 2.0; a = 0.6; T = 1.2;
    cam = [0.0, -2.5, 1.0, 0.0];
    hint1 = [0, 1, 0, 0];
    hint2 = [-1, 0, 0, 0];
    params = [param("rho", "ρ", 0.25, 2.5, 0.01), param("R", "R", 1, 4, 0.01),
              param("a", "a", 0.1, 1, 0.01), param("T", "T", 0.2, 3, 0.01)];
    view = [[1, 2, -75], [0, 2, 25], [2, 3, 35]];
    code = `// N worlds, joined through the slit |x| < R on the line y = 0
vec2 flat_portal(vec4 p, int N) {
    float frame = 1.0 - smoothstep(R - a, R + a, abs(p.x));    // 1 inside the frame, 0 outside
    vec2  full_turn = cexp(-TAU * smoothstep(-T, T, p.y));     // once around 0 as you cross the slit
    vec2  W = mix(vec2(1.0, 0.0), full_turn, frame);
    return roots(p.zw, W, rho, N);
}`;
    w = cx(); n = cx();
    W(p, o) {
      const frame = 1 - smootherstep(this.R - this.a, this.R + this.a, Math.abs(p[0]));
      const t = -2 * Math.PI * smootherstep(-this.T, this.T, p[1]);
      return cset(o, 1 + frame * (Math.cos(t) - 1), frame * Math.sin(t));
    }
    F(p, o) { return roots(o, vOf(this.n, p), this.W(p, this.w), this.rho, this.N); }
    worldIndex(p) {             // sheet label: which root of W p is on (principal branch)
      const W = this.W(p, this.w), N = this.N;
      const ang = Math.atan2(W[1], W[0]) / N, mag = this.rho * cabs(W) ** (1 / N);
      let best = 0, dmin = Infinity;
      for (let k = 0; k < N; k++) {
        const a = ang + 2 * Math.PI * k / N;
        const d = Math.hypot(mag * Math.cos(a) - p[2], mag * Math.sin(a) - p[3]);
        if (d < dmin) { dmin = d; best = k; }
      }
      return best;
    }
    color(p, rgb) { return shade(rgb, WORLD_COLORS[this.worldIndex(p)], p[0], p[1]); }
    box() { const z = 1.3 * this.rho; return [[-4.5, 4.5], [-4.5, 4.5], [-z, z], [-z, z]]; }
  }

  class FlatPortal3 extends FlatPortal {
    name = "flat_portal3"; title = "Flat portal, 3 worlds";
    N = 3;
  }

  /** The first portal: W = (|x|^2 - R^2 + 2iR y)/(|x|^2 + R^2) goes once around
   *  0 along every loop around a frame point (±R, 0); curved everywhere. */
  class Portal extends FlatPortal {
    name = "portal"; title = "Curved portal, 2 worlds (first version)";
    params = [param("rho", "ρ", 0.25, 2.5, 0.01), param("R", "R", 0.5, 4, 0.01)];
    code = `vec2 curved_portal(vec4 p, int N) {            // the first version: curved everywhere, not only at the frame
    float r2 = dot(p.xy, p.xy);
    vec2  W  = vec2(r2 - R * R, 2.0 * R * p.y) / (r2 + R * R);
    return roots(p.zw, W, rho, N);
}`;
    W(p, o) {
      const r2 = p[0] * p[0] + p[1] * p[1], s = 1 / (r2 + this.R ** 2);
      return cset(o, (r2 - this.R ** 2) * s, 2 * this.R * p[1] * s);
    }
    color(p, rgb) {
      shade(rgb, WORLD_COLORS[this.worldIndex(p)], p[0], p[1]);
      // draw the portal frame (the branch points) as dark dots
      if (Math.min(Math.hypot(p[0] - this.R, p[1]), Math.hypot(p[0] + this.R, p[1])) < 0.08) rgb.fill(0.1);
      return rgb;
    }
  }

  class Portal3 extends Portal {
    name = "portal3"; title = "Curved portal, 3 worlds (first version)";
    N = 3;
  }

  /** Graph v = c u^2 / |u| (smoothed tip): more than 360° around the tip. */
  class Cone extends Scene {
    name = "cone"; title = "Cone with extra angle";
    c = 0.845; m = 2; delta = 0.35;
    cam = [-3, 0, 0, 0];
    params = [param("c", "c", 0, 4, 0.005)];
    view = [[1, 2, -65], [0, 2, 30]];
    code = `vec2 cone(vec4 p, float c, float delta) {     // delta rounds the tip
    vec2 u = p.xy;
    vec2 h = c * cmul(u, u) / sqrt(dot(u, u) + delta * delta);    // squaring doubles the angle
    return graph(p.zw, h);
}`;
    F(p, o) { return cone(o, p, this.c, this.m, this.delta); }
    color(p, rgb) { return shade(rgb, GREEN, p[0], p[1]); }
    info() {
      const t = coneAngle(this.c, this.m) / (2 * Math.PI);
      return `angle around the tip, on large circles: 2π·${t.toFixed(3)} = ${(360 * t).toFixed(1)}°`;
    }
    box() { const h = 4 * this.c + 0.5; return [[-4, 4], [-4, 4], [-h, h], [-h, h]]; }
  }

  /** A plane whose mouth leads back into itself.  The throat is the flat Möbius band
   *  {|u|² + 4|v|² = R², arg v = 2 arg u + spin(|u|)}: a flat tube of circumference 2πR whose
   *  far end is glued to itself, point to opposite point.  Away from the centre the normal plane
   *  is turned by 2 arg u + spin (there the world is a plane with a dome); near the centre the
   *  band's rulings, lines through the centre at half the angle of v (a sign cut, like the
   *  mirror handle's flip). */
  class MonoWormhole extends Scene {
    name = "mono"; title = "One-mouth wormhole";
    R = 1; k = 0.15; twist = 16;
    rMax = 12;
    cam = [-2.5, 0, 0, 0];
    hint1 = [0, -1, 0, 0];
    hint2 = [1, 0, 0, 0];
    params = [param("twist", "twist", 0, 30, 0.1), param("R", "R", 0.5, 2, 0.01)];
    view = [[1, 2, -55], [0, 2, 25], [2, 3, 40]];
    code = `// a plane whose mouth leads back into itself: a flat tube of circumference 2 PI R, whose far end
// is glued to itself, every point to the opposite one (a Möbius strip)
vec2 mono_wormhole(vec4 p) {
    float r = length(p.xy);
    float spin = twist * pow(max(1.0 - r * r / (R * R), 0.0), 2.0);   // keeps turning inside: a longer throat
    if (r > 0.5 * R) {
        float angle = 2.0 * atan(p.y, p.x) + spin;     // the normal plane, turned twice as you go around
        float s = turn(p.z, p.w, angle), c = turn(p.w, -p.z, angle);
        return smooth_union(world(s, c), world(ball(vec3(p.xy, 2.0 * s), R), c), k);   // a plane with a dome
    }
    // at the top of the dome the turning breaks down; there: lines through the centre,
    // at half the angle of p.zw (once around, the half angle jumps: a harmless sign flip)
    float half = 0.5 * (atan(p.w, p.z) - spin);
    return world(ball(vec3(p.xy, 2.0 * length(p.zw)), R), turn(p.y, -p.x, half));
}`;
    f1 = cx(); f2 = cx();
    spin(r) { const t = Math.max(1 - (r * r) / (this.R * this.R), 0); return this.twist * t * t; }
    field(p, o, inner, psi0) {
      const [x, y, z, w] = p, r = Math.hypot(x, y);
      if (!inner) {
        const angle = 2 * Math.atan2(y, x) + this.spin(r);
        const s = turn(z, w, angle), c = turn(w, -z, angle);
        return smoothUnion(o, world(this.f1, s, c), world(this.f2, ball3(x, y, 2 * s, this.R), c), this.k);
      }
      let psi = Math.atan2(w, z);
      if (psi0 !== undefined) psi = psi0 + wrap(psi - psi0);       // continue the branch of psi0
      return world(o, ball3(x, y, 2 * Math.hypot(z, w), this.R), turn(y, -x, 0.5 * (psi - this.spin(r))));
    }
    F(p, o) { return this.field(p, o, Math.hypot(p[0], p[1]) < 0.5 * this.R); }
    // the chart and the branch of atan(w, z) are frozen at p: differentiate the side p is on
    J(p, a, b) { this.inner = Math.hypot(p[0], p[1]) < 0.5 * this.R; this.psi0 = Math.atan2(p[3], p[2]); jac(this.locked, p, a, b); }
    locked = { F: (p, o) => this.field(p, o, this.inner, this.psi0) };
    /** Flat coordinates of the throat as functions of r = |u|: depth from the mouth, around-shift. */
    tables() {
      if (this.tabKey === this.R + "," + this.twist) return this.tab;
      const n = 2001, R = this.R, r = new Float64Array(n), d = new Float64Array(n), sh = new Float64Array(n);
      let pe = 0, pf = 0, pa = 0, prevSp = this.spin(R);
      for (let i = 0; i < n; i++) {
        const al = (i / (n - 1)) * (Math.PI / 2), h = (Math.PI / 2) / (n - 1);
        r[i] = R * Math.cos(al);
        const spA = (this.spin(R * Math.cos(al + 1e-5)) - this.spin(R * Math.cos(al - 1e-5))) / 2e-5;
        const sa = Math.sin(al), ca = Math.cos(al);
        const e = R * Math.sqrt(sa * sa + 0.25 * ca * ca + 0.25 * (sa * ca * spA) ** 2), f = 0.5 * R * sa * sa * spA;
        if (i > 0) { d[i] = d[i - 1] + 0.5 * (e + pe) * h; sh[i] = sh[i - 1] + 0.5 * (f + pf) * h / R; }
        pe = e; pf = f;
      }
      this.tab = { r, d, sh, n }; this.tabKey = R + "," + this.twist;
      return this.tab;
    }
    color(p, rgb) {
      const r = Math.hypot(p[0], p[1]);
      if (r >= this.R) return shade(rgb, GREEN, p[0], p[1], 0.5);
      const T = this.tables(), t = Math.acos(Math.min(r / this.R, 1)) / (Math.PI / 2) * (T.n - 1);
      const i = Math.min(Math.floor(t), T.n - 2), f = t - i;
      const d = T.d[i] * (1 - f) + T.d[i + 1] * f, sh = T.sh[i] * (1 - f) + T.sh[i + 1] * f;
      const C = 2 * Math.PI * this.R, sigma = this.R * Math.atan2(p[1], p[0]) + sh;
      return shade(rgb, TEAL, ((sigma % C) + C) % C, d, C / 12);
    }
    box() { const b = 1.8 * this.R; return [[-b, b], [-b, b], [-0.6 * this.R, 0.6 * this.R], [-0.6 * this.R, 0.6 * this.R]]; }
  }

  const scenes = {};
  for (const C of [Plane, Sphere, Torus, Clifford, TwoPlanes, ThreeParallel, YJunction, Handle, Klein,
                   Neck, ThreeNecks, TriplePoint, FlatPortal, FlatPortal3, Cone, Portal, Portal3, MonoWormhole]) {
    const s = new C();
    scenes[s.name] = s;
  }
  return scenes;
}

if (typeof module === "object" && module.exports) module.exports = co2scenes;
