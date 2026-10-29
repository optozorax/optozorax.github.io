/* co2sdf.js — complex codimension-2 SDFs, a port of co2sdf.py (+ render.py).
 *
 * A surface S ⊂ R^4 is the zero set of a complex function F(p): Re F is the
 * "solid" SDF, Im F the "carrier" SDF.  Everything here works on one point at
 * a time: points and vectors are Float64Array(4), complex values are
 * Float64Array(2) = [re, im], and functions write into arguments they are
 * given, so the hot loops do not allocate.
 *
 * A surface is an object S with a method S.F(p, out) (writes F(p) into out and
 * returns it) and optionally S.J(p, a, b), which replaces the numerical
 * Jacobian (rows a = ∇Re F, b = ∇Im F).
 */
function co2sdf() {
  "use strict";

  // ------------------------------------------------------------- vectors
  const vec = (n = 4) => new Float64Array(n);
  const dot = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
  const hypot = (x, y) => Math.sqrt(x * x + y * y);

  function normalize(v) {
    const n = Math.sqrt(dot(v, v));
    for (let k = 0; k < 4; k++) v[k] /= n;
    return v;
  }

  // ------------------------------------------------------------- complex
  // Every op writes into its first argument `o` and returns it; `o` may alias
  // an input.
  const cx = (re = 0, im = 0) => Float64Array.of(re, im);
  const cset = (o, re, im) => { o[0] = re; o[1] = im; return o; };
  const cabs = (z) => hypot(z[0], z[1]);
  const uOf = (o, p) => cset(o, p[0], p[1]);   // complex coordinate of the (x1, x2) plane
  const vOf = (o, p) => cset(o, p[2], p[3]);   // complex coordinate of the (x3, x4) plane

  const cmul = (o, x, y) => cset(o, x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]);

  /** o = x / y by Smith's algorithm (what numpy does). */
  function cdiv(o, x, y) {
    if (Math.abs(y[0]) >= Math.abs(y[1])) {
      const t = y[1] / y[0], s = 1 / (y[0] + y[1] * t);
      return cset(o, (x[0] + x[1] * t) * s, (x[1] - x[0] * t) * s);
    }
    const t = y[0] / y[1], s = 1 / (y[1] + y[0] * t);
    return cset(o, (x[0] * t + x[1]) * s, (x[1] * t - x[0]) * s);
  }

  /** o = z^n for an integer n >= 1 (repeated multiplication). */
  function cpow(o, z, n) {
    let re = z[0], im = z[1];
    for (let i = 1; i < n; i++) {
      const t = re * z[0] - im * z[1];
      im = re * z[1] + im * z[0];
      re = t;
    }
    return cset(o, re, im);
  }

  // ------------------------------------------------------------- kernel
  const fp = cx(), fm = cx(), val = cx();

  /** Rows a = ∇Re F, b = ∇Im F at p by central differences (p is restored). */
  function jac(S, p, a, b, h = 1e-5) {
    const s = 1 / (2 * h);
    for (let k = 0; k < 4; k++) {
      const x = p[k];
      p[k] = x + h; S.F(p, fp);
      p[k] = x - h; S.F(p, fm);
      p[k] = x;
      a[k] = (fp[0] - fm[0]) * s;
      b[k] = (fp[1] - fm[1]) * s;
    }
  }

  /** Reciprocal frame of (a, b): ra·a = 1, ra·b = 0, rb·a = 0, rb·b = 1. */
  function recip(a, b, ra, rb) {
    const aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
    const det = aa * bb - ab * ab;
    for (let k = 0; k < 4; k++) {
      ra[k] = (bb * a[k] - ab * b[k]) / det;
      rb[k] = (aa * b[k] - ab * a[k]) / det;
    }
  }

  /** Normal frame N = {a, b, ra, rb}: Jacobian rows at a point + reciprocals. */
  const normals = () => ({ a: vec(), b: vec(), ra: vec(), rb: vec() });

  /** Jacobian rows a, b at p: the scene's own S.J if it has one. */
  function grads(S, p, a, b) {
    if (S.J) S.J(p, a, b); else jac(S, p, a, b);
  }

  function normalsAt(S, p, N) {
    grads(S, p, N.a, N.b);
    recip(N.a, N.b, N.ra, N.rb);
    return N;
  }

  /** Remove the normal-plane part of v (normal plane = span(a, b)), in place. */
  function tangent(N, v) {
    const va = dot(v, N.a), vb = dot(v, N.b);
    for (let k = 0; k < 4; k++) v[k] = v[k] - va * N.ra[k] - vb * N.rb[k];
    return v;
  }

  /** One Gauss-Newton update p <- p - Re F·ra - Im F·rb.  Leaves in N the
   *  normal frame used for the update (at the old p); returns |F(old p)|. */
  function newton(S, p, N) {
    S.F(p, val);
    normalsAt(S, p, N);
    for (let k = 0; k < 4; k++) p[k] = p[k] - val[0] * N.ra[k] - val[1] * N.rb[k];
    return cabs(val);
  }

  /** Gauss-Newton projection onto Z(F). */
  function project(S, p, N, iters = 3) {
    for (let i = 0; i < iters; i++) newton(S, p, N);
    return p;
  }

  /** Gauss-Newton projection onto the curve {F = 0, n·p = c} (a slice of the
   *  surface by a hyperplane; codimension 3 in R^4): the step is the
   *  minimum-norm solution for the rows a, b, n, i.e. the 3x3 Gram system
   *  solved by cofactors.  Returns the residual |(F, n·p - c)| at the final p
   *  (NaN if the Gram matrix degenerates). */
  const ga = vec(), gb = vec();
  function projectSlice(S, p, n, c, iters = 30, tol = 1e-13) {
    for (let i = 0; ; i++) {
      S.F(p, val);
      const r0 = val[0], r1 = val[1], r2 = dot(n, p) - c, res = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2);
      if (!(res > tol) || i === iters) return res;
      grads(S, p, ga, gb);
      const aa = dot(ga, ga), ab = dot(ga, gb), an = dot(ga, n), bb = dot(gb, gb), bn = dot(gb, n), nn = dot(n, n);
      const c00 = bb * nn - bn * bn, c01 = an * bn - ab * nn, c02 = ab * bn - an * bb;
      const c11 = aa * nn - an * an, c12 = ab * an - aa * bn, c22 = aa * bb - ab * ab;
      const det = aa * c00 + ab * c01 + an * c02;
      const y0 = (c00 * r0 + c01 * r1 + c02 * r2) / det;
      const y1 = (c01 * r0 + c11 * r1 + c12 * r2) / det;
      const y2 = (c02 * r0 + c12 * r1 + c22 * r2) / det;
      for (let k = 0; k < 4; k++) p[k] -= y0 * ga[k] + y1 * gb[k] + y2 * n[k];
    }
  }

  /** Second-order geodesic step.  N holds the normal frame at p on entry:
   *  the first correction uses it, the second uses fresh normals, and the new
   *  direction is the chord projected onto the new tangent plane.  On return
   *  N holds the normals at the new p, ready for the next step. */
  const start = vec();
  function step(S, p, v, dt, N) {
    for (let k = 0; k < 4; k++) { start[k] = p[k]; p[k] += v[k] * dt; }
    S.F(p, val);
    for (let k = 0; k < 4; k++) p[k] = p[k] - val[0] * N.ra[k] - val[1] * N.rb[k];
    newton(S, p, N);
    normalsAt(S, p, N);
    for (let k = 0; k < 4; k++) v[k] = p[k] - start[k];
    return normalize(tangent(N, v));
  }

  /** Parallel transport of an orthonormal frame over one step: project both
   *  vectors onto the new tangent plane, then re-orthonormalise. */
  function transport(N, e1, e2) {
    normalize(tangent(N, e1));
    tangent(N, e2);
    const d = dot(e2, e1);
    for (let k = 0; k < 4; k++) e2[k] = e2[k] - d * e1[k];
    normalize(e2);
  }

  /** Orthonormal tangent frame (e1, e2) at p from two hint vectors. */
  function tangentFrame(S, p, hint1, hint2, e1, e2, N) {
    e1.set(hint1);
    e2.set(hint2);
    transport(normalsAt(S, p, N), e1, e2);
  }

  /** Walk `dist` along the geodesic that leaves p in the unit tangent
   *  direction d, parallel-transporting the frame (e1, e2) with it. */
  function walk(S, p, d, e1, e2, dist, N, dt = 0.01) {
    const n = Math.ceil(dist / dt);
    normalsAt(S, p, N);
    for (let i = 0; i < n; i++) {
      step(S, p, d, dist / n, N);
      transport(N, e1, e2);
    }
  }

  // ------------------------------------------------------------- curvature
  /** Orthonormal tangent frame (t1, t2) at a point with normals N, from the
   *  tangent parts of the coordinate axes (the longest ones). */
  const axis = vec();
  function tangentBasis(N, t1, t2) {
    for (const t of [t1, t2]) {
      let best = -1;
      for (let k = 0; k < 4; k++) {
        axis.fill(0);
        axis[k] = 1;
        tangent(N, axis);
        if (t === t2) {
          const d = dot(axis, t1);
          for (let i = 0; i < 4; i++) axis[i] -= d * t1[i];
        }
        const n = dot(axis, axis);
        if (n > best) { best = n; t.set(axis); }
      }
      normalize(t);
    }
  }

  /** Gaussian curvature of the surface at p (on it; N: the normals at p).
   *  P = a ra^T + b rb^T projects onto the normal plane; how it turns along
   *  the surface is the second fundamental form II(x, y) = -(∂_x P) y, a
   *  vector in the normal plane (central differences of step h along a
   *  tangent frame t1, t2).  The Gauss equation then gives
   *  K = II(t1, t1)·II(t2, t2) - |II(t1, t2)|².  P depends only on the plane,
   *  not on the pair (a, b), so fields that switch branches or flip the sign
   *  of Im F off the surface give the right K. */
  const kT = [vec(), vec()], kD = [vec(), vec(), vec(), vec()], kq = vec(), kN = normals();
  function curvature(S, p, N, h = 1e-3) {
    tangentBasis(N, kT[0], kT[1]);
    for (const D of kD) D.fill(0);
    for (let i = 0; i < 2; i++) {
      for (let sgn = 1; sgn >= -1; sgn -= 2) {
        for (let k = 0; k < 4; k++) kq[k] = p[k] + sgn * h * kT[i][k];
        normalsAt(S, kq, kN);
        for (let j = 0; j < 2; j++) {               // kD[2i + j] = (∂_i P) t_j = -II(t_i, t_j)
          const s = sgn / (2 * h), wa = dot(kT[j], kN.a) * s, wb = dot(kT[j], kN.b) * s, D = kD[2 * i + j];
          for (let k = 0; k < 4; k++) D[k] += wa * kN.ra[k] + wb * kN.rb[k];
        }
      }
    }
    let m = 0;
    for (let k = 0; k < 4; k++) { const x = 0.5 * (kD[1][k] + kD[2][k]); m += x * x; }
    return dot(kD[0], kD[3]) - m;
  }

  /** The colours of portal_gravity's curvature view (display RGB, 0..1). */
  const K_NEG = [96 / 255, 166 / 255, 225 / 255], K_ZERO = [218 / 255, 218 / 255, 218 / 255], K_POS = [251 / 255, 158 / 255, 35 / 255];

  /** rgb = the colour of curvature k: K_ZERO at 0, mixed towards K_NEG or
   *  K_POS by |k·scale| up to 1 (portal_gravity's sdf_curvature_color). */
  function curvatureColor(k, scale, rgb) {
    const t = Math.min(Math.abs(k * scale), 1) || 0, c = k < 0 ? K_NEG : K_POS;
    for (let i = 0; i < 3; i++) rgb[i] = K_ZERO[i] + (c[i] - K_ZERO[i]) * t;
    return rgb;
  }

  // ------------------------------------------------------------- helpers
  const clamp01 = (t) => Math.min(Math.max(t, 0), 1);

  function smoothstep(e0, e1, x) {
    const t = clamp01((x - e0) / (e1 - e0));
    return t * t * (3 - 2 * t);
  }

  function smootherstep(e0, e1, x) {
    const t = clamp01((x - e0) / (e1 - e0));
    return t * t * t * (t * (6 * t - 15) + 10);
  }

  /** Polynomial smooth min of a, b: blend weight h (h = 1 -> a) and value. */
  const sminWeight = (a, b, k) => clamp01(0.5 + 0.5 * (b - a) / k);
  const smin = (a, b, k, h = sminWeight(a, b, k)) => b + (a - b) * h - k * h * (1 - h);

  // ------------------------------------------------------------- Booleans
  /** Hard union: keep the value whose solid (Re) is smaller. */
  function union(o, f1, f2) {
    const f = f1[0] <= f2[0] ? f1 : f2;
    return cset(o, f[0], f[1]);
  }

  /** Smooth union of the solids; the carriers are mixed with the same weight. */
  function foldUnion(o, f1, f2, k) {
    const h = sminWeight(f1[0], f2[0], k);
    const w = smoothstep(0, 1, h);                  // C1 weight for the carrier
    return cset(o, smin(f1[0], f2[0], k, h), f2[1] + (f1[1] - f2[1]) * w);
  }

  /** 1 for t <= 1/2, 0 for t >= 1, C2 in between. */
  const cutoff = (t) => 1 - smootherstep(0.5, 1, t);

  /** Union whose transverse crossings are replaced by necks {F1 F2 = eps}.
   *  Outside |F| > k the pieces are untouched.  Waist radius ~ sqrt(|eps|).
   *  Near piece f (the smaller |F|) the value is f - eps·cutoff(|g|/k) / g. */
  const quot = cx();
  /** neckUnion with a frozen branch choice (near1: the f1 branch).  Off the
   *  surface the two branches differ, so Jacobians must use one branch. */
  function neckUnionOn(o, f1, f2, eps, k, near1) {
    const f = near1 ? f1 : f2, g = near1 ? f2 : f1;
    cdiv(quot, cset(quot, eps * cutoff(cabs(g) / k), 0), g);
    return cset(o, f[0] - quot[0], f[1] - quot[1]);
  }
  function neckUnion(o, f1, f2, eps, k) {
    const r1 = cabs(f1), r2 = cabs(f2), near1 = r1 < r2;
    const f = near1 ? f1 : f2, g = near1 ? f2 : f1;
    cdiv(quot, cset(quot, eps * cutoff((near1 ? r2 : r1) / k), 0), g);
    return cset(o, f[0] - quot[0], f[1] - quot[1]);
  }

  // ------------------------------------------------------------- the page's library
  // The names of the page's pseudocode.  Numbers first (portal_gravity's
  // ball / extrude / smin), then worlds: a world is [solid, carrier], two
  // ordinary SDFs; it is the boundary of the solid, inside the carrier.
  const ball2 = (x, y, r) => Math.sqrt(x * x + y * y) - r;
  const ball3 = (x, y, z, r) => Math.sqrt(x * x + y * y + z * z) - r;
  const ball4 = (x, y, z, w, r) => Math.sqrt(x * x + y * y + z * z + w * w) - r;

  /** portal_gravity's extrude: the SDF value d extruded along the coordinate p, from -h to h. */
  function extrude(p, d, h) {
    const wx = d, wy = Math.abs(p) - h, mx = Math.max(wx, 0), my = Math.max(wy, 0);
    return Math.min(Math.max(wx, wy), 0) + Math.sqrt(mx * mx + my * my);
  }

  /** A carrier turned by `angle` from the carrier a towards the carrier b. */
  const turn = (a, b, angle) => Math.cos(angle) * a + Math.sin(angle) * b;

  /** Distance from (x, y, z, w) to the segment from the origin to (0, 0, ez, ew), minus r. */
  function capsule4(x, y, z, w, ez, ew, r) {
    const h = clamp01((z * ez + w * ew) / (ez * ez + ew * ew));
    return ball4(x, y, z - h * ez, w - h * ew, r);
  }

  const world = (o, solid, carrier) => cset(o, solid, carrier);
  const smoothUnion = (o, A, B, k) => foldUnion(o, A, B, k);
  /** extrude(world, p, h) of the page: the solid extruded along p, the carrier unchanged. */
  const extrudeWorld = (o, A, p, h) => cset(o, extrude(p, A[0], h), A[1]);

  /** N worlds at n = rho * (the N-th roots of W): (rho (n/rho)^N - rho W) / N. */
  function roots(o, n, W, rho, N) {
    cpow(o, cset(o, n[0] / rho, n[1] / rho), N);
    return cset(o, rho * (o[0] - W[0]) / N, rho * (o[1] - W[1]) / N);
  }

  // ------------------------------------------------------------- primitives
  /** Clifford torus |u| = a, |v| = b. */
  const cliffordTorus = (o, p, a, b) => cset(o, hypot(p[0], p[1]) - a, hypot(p[2], p[3]) - b);

  /** Graph v = c u^m / |u|^(m-1) (smoothed).  Total angle
   *  2π sqrt((1 + m²c²) / (1 + c²)).  m = 0 gives an ordinary cone (in R^3). */
  function cone(o, p, c, m, delta) {
    const au = hypot(p[0], p[1]), s = Math.sqrt(au * au + delta * delta);
    if (m === 0) return cset(o, p[2] - c * s, p[3]);
    cpow(o, uOf(o, p), m);
    const t = 1 / s ** (m - 1);
    return cset(o, p[2] - c * o[0] * t, p[3] - c * o[1] * t);
  }

  const coneAngle = (c, m) => 2 * Math.PI * Math.sqrt((1 + m * m * c * c) / (1 + c * c));

  // ------------------------------------------------------------- renderer
  /** Rays j0..j1-1 of render.py's `trace_fan`: nDir geodesics leave cam at
   *  angles θ_j = 2πj/nDir (from e1 towards e2) and take nSteps steps of dt.
   *  The colour at step i is stored as bytes floor(255·rgb), like matplotlib,
   *  at cols[3 ((j - j0) (nSteps + 1) + i)]; track(j, i, p) sees every point.
   *  paint(p, N, i, rgb) (N: the normals at p) replaces S.color if given. */
  function traceFan(S, cam, e1, e2, nDir, nSteps, dt, cols, j0 = 0, j1 = nDir, track = null, paint = null) {
    const p = vec(), v = vec(), rgb = vec(3), N = normals();
    for (let j = j0; j < j1; j++) {
      const th = j * (2 * Math.PI / nDir), c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < 4; k++) { p[k] = cam[k]; v[k] = c * e1[k] + s * e2[k]; }
      normalsAt(S, p, N);
      for (let i = 0, o = 3 * (j - j0) * (nSteps + 1); ; i++, o += 3) {
        if (paint) paint(p, N, i, rgb); else S.color(p, rgb);
        cols[o] = rgb[0] * 255; cols[o + 1] = rgb[1] * 255; cols[o + 2] = rgb[2] * 255;
        if (track) track(j, i, p);
        if (i === nSteps) break;
        step(S, p, v, dt, N);
      }
    }
    return cols;
  }

  /** numpy's round(): half to even. */
  function rint(x) {
    const r = Math.round(x);
    return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  /** Pixel lookup of render.py's `scan_image` for a res x res image (right =
   *  e1, up = e2): j·nR + i (ray j, sample i), -1 outside the disk of radius
   *  r_max = (nR - 1) dt, -2 for the camera marker. */
  function scanIndex(res, nDir, nR, dt, marker = 0.02) {
    const idx = new Int32Array(res * res), rMax = (nR - 1) * dt, s = 2 / (res - 1);
    for (let row = 0; row < res; row++) {
      for (let col = 0; col < res; col++) {
        const x = col * s - 1, y = row * s - 1, r = Math.hypot(x, y) * rMax;
        let th = Math.atan2(-y, x);                 // image y axis points down
        if (th < 0) th += 2 * Math.PI;
        const j = rint(th / (2 * Math.PI) * nDir) % nDir;
        const i = Math.min(Math.max(rint(r / dt), 0), nR - 1);
        idx[row * res + col] = r > rMax ? -1 : r < marker * rMax ? -2 : j * nR + i;
      }
    }
    return idx;
  }

  /** Fill RGBA `out` from colours `cols` through a scanIndex (defaults: the
   *  white background and floor(0.05·255) grey marker of render.py). */
  function scanImage(cols, idx, out, outside = [255, 255, 255, 255], marker = [12, 12, 12, 255]) {
    for (let q = 0; q < idx.length; q++) {
      const m = idx[q], o = 4 * q, c = m === -1 ? outside : m === -2 ? marker : null;
      if (c) { out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = c[3]; }
      else { out[o] = cols[3 * m]; out[o + 1] = cols[3 * m + 1]; out[o + 2] = cols[3 * m + 2]; out[o + 3] = 255; }
    }
    return out;
  }

  return {
    vec, dot, normalize, cx, cset, cabs, uOf, vOf, cmul, cdiv, cpow,
    jac, grads, recip, normals, normalsAt, tangent, newton, project, projectSlice, step, transport, tangentFrame, walk,
    tangentBasis, curvature, curvatureColor,
    clamp01, smoothstep, smootherstep, sminWeight, smin,
    union, foldUnion, cutoff, neckUnion, neckUnionOn, cliffordTorus, cone, coneAngle,
    ball2, ball3, ball4, extrude, turn, capsule4, world, smoothUnion, extrudeWorld, roots,
    traceFan, scanIndex, scanImage,
  };
}

if (typeof module === "object" && module.exports) module.exports = co2sdf;
