/* demo.js — interactive co2sdf widget.  Load demo.css, co2sdf.js and
 * scenes.js first; every <div class="co2-demo"> on the page becomes a widget.
 * Attributes:  data-scene="neck"     start scene
 *              data-compact="1"      small inline widget: no scene menu, the two
 *                                    views side by side, sliders in one row
 *              data-params="0"       (compact) hide the sliders
 *              data-outside="slice"  start the outside view in slice mode
 *              data-slice="w"        the coordinate the slice fixes (x, y, z or
 *                                    w; default y)
 *              data-curvature="1"    start in the curvature view (colours by
 *                                    Gaussian curvature, as in portal_gravity)
 *   Inside view:  exponential-map scan around a camera that walks on the
 *                 surface (computed in Web Workers made from a Blob, so the
 *                 page also works from file://; one pool serves all widgets).
 *   Outside view: the surface as a point cloud, projected R^4 -> R^3 -> screen
 *                 (any 4D rotation), or ("slice") its intersection with the 3D
 *                 space where one coordinate is fixed, e.g. y = 0, drawn in
 *                 the other three (x, z, w).
 * Test flags in the page URL: ?sync=1 renders synchronously on load (no
 * worker; for headless screenshots), ?scene=<name> picks the start scene,
 * ?outside=slice starts in slice mode, ?slice=w fixes w, ?curvature=1 starts
 * in the curvature view. */
(function () {
  "use strict";

  // ====================================================== worker side
  // co2jobs + co2scheduler run in the worker (their source is stringified
  // into it) and, with ?sync=1 or when workers are unavailable, on the page.

  /** Jobs are generators: every `yield` is a point where a newer request may
   *  cut in; yielding or returning [message, transfer] posts a message. */
  function co2jobs(K, scenes) {
    /** A fresh scene instance per job: jobs of several widgets interleave. */
    function sceneFor(m) {
      const S = Object.assign(new scenes[m.scene].constructor(), m.params);
      S.update();
      return S;
    }

    /** The camera drawn on the world: a red disc with a white nose in front (e2, up in
     *  the scan) and a white ear on its left (-e1), so that copies of it show whether
     *  they are mirrored.  A world point is part of it if it is near the camera in 4D
     *  and the world there has the camera's tangent plane (a different sheet of the
     *  world that only passes close by in 4D is not). */
    function creaturePaint(cam, e1, e2, base, size = 0.3) {
      const d = K.vec(), t1 = K.vec(), t2 = K.vec();
      return (p, N, i, rgb) => {
        base(p, N, i, rgb);
        let d2 = 0;
        for (let k = 0; k < 4; k++) { d[k] = p[k] - cam[k]; d2 += d[k] * d[k]; }
        if (d2 > size * size) return rgb;
        const a1 = K.dot(d, e1), a2 = K.dot(d, e2);
        if (d2 - a1 * a1 - a2 * a2 > 0.04 * size * size) return rgb;
        t1.set(e1); t2.set(e2); K.tangent(N, t1); K.tangent(N, t2);
        if (K.dot(t1, t1) < 0.8 || K.dot(t2, t2) < 0.8) return rgb;
        const white = Math.hypot(a1, a2 - 0.62 * size) < 0.28 * size || Math.hypot(a1 + 0.6 * size, a2 + 0.06 * size) < 0.22 * size;
        rgb[0] = white ? 1 : 0.9; rgb[1] = rgb[2] = white ? 1 : 0.15;
        return rgb;
      };
    }

    /** The curvature view, paint(p, N, i, rgb) for traceFan: Gaussian
     *  curvature in portal_gravity's colours (saturating at |K| = 1/scale)
     *  over a faint copy of the scene's own pattern.  The curvature is computed
     *  at samples i = 0, every, 2·every, ... of a ray and held in between. */
    function curvaturePaint(S, scale, every = 1) {
      const tint = K.vec(3);
      let k = 0;
      return (p, N, i, rgb) => {
        if (i % every === 0) k = K.curvature(S, p, N);
        S.color(p, rgb);
        const l = Math.min(1, (0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]) / 0.6);
        K.curvatureColor(k, scale, tint);
        for (let c = 0; c < 3; c++) rgb[c] = tint[c] * (0.75 + 0.25 * l);
        return rgb;
      };
    }

    /** Rays j0..j1-1 of the fan of geodesics around the camera -> colour
     *  samples, plus positions along the tracked rays (for the outside view). */
    function* scan(m) {
      const S = sceneFor(m), nSteps = Math.ceil(m.rMax / m.dt - 1e-9), nR = nSteps + 1;
      // the camera's size follows the scene's own scale (r_max 1.5 for the two-planes wormhole, 6 to 12 for most)
      const paint = creaturePaint(m.cam, m.e1, m.e2, m.kScale ? curvaturePaint(S, m.kScale, m.kEvery) : (p, N, i, rgb) => S.color(p, rgb), Math.min(0.3, 0.04 * S.rMax));
      const cols = new Uint8Array((m.j1 - m.j0) * nR * 3);
      const every = Math.max(1, Math.round(nSteps / 120)), nS = Math.floor(nSteps / every) + 1;
      const stride = m.nDir / m.nTrack, paths = new Float32Array(m.nTrack * nS * 4);
      const track = (j, i, p) => {
        if (j % stride === 0 && i % every === 0) paths.set(p, ((j / stride) * nS + i / every) * 4);
      };
      let ms = 0;
      for (let j = m.j0; j < m.j1; j += m.chunk) {
        const t = performance.now();
        K.traceFan(S, m.cam, m.e1, m.e2, m.nDir, nSteps, m.dt, cols.subarray(3 * (j - m.j0) * nR), j, Math.min(m.j1, j + m.chunk), track, paint);
        ms += performance.now() - t;
        if (j + m.chunk < m.j1) yield null;
      }
      const { owner, id, req, parts, quality, j0, j1, nDir, dt } = m;
      return [{ kind: "scan", owner, id, req, parts, quality, j0, j1, nDir, nR, dt, cols, paths, nS, ms }, [cols.buffer, paths.buffer]];
    }

    /** Random points of the scene box moved by project(p) (true: converged);
     *  keeps those that end inside the box, with their colours (of the
     *  curvature view if m.kScale).  Posts one preview once m.preview points
     *  exist or 60 ms have passed. */
    function* sample(m, S, project) {
      const box = S.box(), p = K.vec(), rgb = K.vec(3), t0 = performance.now();
      const paint = m.kScale ? curvaturePaint(S, m.kScale) : null, N = K.normals();
      const pos = new Float32Array(4 * m.count), col = new Uint8Array(3 * m.count);
      let seed = 12345, n = 0, previewSent = false;
      const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const inBox = () => p.every((x, k) => x >= box[k][0] && x <= box[k][1]);
      const message = (partial) => {
        const P = pos.slice(0, 4 * n), C = col.slice(0, 3 * n);
        return [{ kind: m.kind, owner: m.owner, id: m.id, n, pos: P, col: C, partial, ms: performance.now() - t0 }, [P.buffer, C.buffer]];
      };
      for (let tries = 1; n < m.count && tries <= m.tries * m.count; tries++) {
        for (let k = 0; k < 4; k++) p[k] = box[k][0] + (box[k][1] - box[k][0]) * rand();
        if (project(p) && inBox()) {
          if (paint) paint(p, K.normalsAt(S, p, N), 0, rgb); else S.color(p, rgb);
          pos.set(p, 4 * n);
          col[3 * n] = 255 * rgb[0]; col[3 * n + 1] = 255 * rgb[1]; col[3 * n + 2] = 255 * rgb[2];
          n++;
        }
        if (tries % 200 === 0) {
          const preview = !previewSent && (n >= m.preview || performance.now() - t0 > 60);
          previewSent = previewSent || preview;
          yield preview ? message(true) : null;
        }
      }
      return message(false);
    }

    /** The surface: Newton projection, converged = |F| < 1e-9. */
    function* cloud(m) {
      const S = sceneFor(m), N = K.normals(), f = K.cx();
      return yield* sample(m, S, (p) => {
        for (let it = 0; it < 30; it++) if (!(K.newton(S, p, N) > 1e-13)) break;   // converged or NaN
        return K.cabs(S.F(p, f)) < 1e-9;
      });
    }

    /** The slice {F = 0, p[axis] = c}: seeds put into that 3D space, then
     *  Gauss-Newton with the rows a, b, e_axis.  Where the surface lies in the
     *  space itself (e.g. a world in w = 0 cut at w = 0) the Gram matrix
     *  degenerates (NaN) and the slice is that piece of the surface: plain
     *  Newton from the seed, kept if it stays in the space.  16 iterations:
     *  seeds that converge do so within ~12, and an empty slice is done sooner. */
    function* slice(m) {
      const S = sceneFor(m), k = m.axis, c = m.c, n = K.vec(), seed = K.vec(), N = K.normals(), f = K.cx();
      n[k] = 1;
      return yield* sample(m, S, (p) => {
        p[k] = c;
        seed.set(p);
        const res = K.projectSlice(S, p, n, c, 16);
        if (!Number.isNaN(res)) return res < 1e-9;
        p.set(seed);
        for (let it = 0; it < 30; it++) if (!(K.newton(S, p, N) > 1e-13)) break;   // converged or NaN
        return Math.abs(p[k] - c) < 1e-9 && K.cabs(S.F(p, f)) < 1e-9;
      });
    }

    return { scan, cloud, slice };
  }

  /** Runs the newest request per (widget, kind) in slices, one per tick:
   *  "scan" before "slice" before "cloud", widgets taking turns.  A newer
   *  request replaces (aborts) the old one; {kind: "cancel", owner} drops all
   *  jobs of a widget.  Returns the request handler; results go to post(). */
  function co2scheduler(K, scenes, post, sync) {
    const jobs = co2jobs(K, scenes), running = new Map(), ORDER = ["scan", "slice", "cloud"];
    const ch = sync ? null : new MessageChannel();
    let queued = false;
    const wake = () => {
      if (ch && !queued && running.size) { queued = true; ch.port2.postMessage(0); }
    };
    function tick() {
      queued = false;
      let key = null;
      for (const kind of ORDER) {
        for (const [k, job] of running) if (job.kind === kind) { key = k; break; }
        if (key !== null) break;
      }
      if (key === null) return;
      const job = running.get(key), r = job.gen.next();
      running.delete(key);
      if (!r.done) running.set(key, job);            // to the back: other widgets go first
      if (r.value) post(r.value[0], r.value[1]);
      wake();
    }
    if (ch) ch.port1.onmessage = tick;
    return (m) => {
      if (m.kind === "cancel") {
        for (const k of [...running.keys()]) if (k.startsWith(m.owner + " ")) running.delete(k);
        return;
      }
      const key = m.owner + " " + m.kind;
      running.delete(key);
      running.set(key, { kind: m.kind, gen: jobs[m.kind](m) });
      if (sync) while (running.size) tick();
      else wake();
    };
  }

  function co2worker() {
    const K = co2sdf();
    const handle = co2scheduler(K, co2scenes(K), (msg, transfer) => postMessage(msg, transfer), false);
    self.onmessage = (e) => handle(e.data);
  }

  let workerURL = null;
  function makeWorker() {
    if (!workerURL) {
      const src = '"use strict";\n' + [co2sdf, co2scenes, co2jobs, co2scheduler, co2worker].join("\n\n") + "\nco2worker();\n";
      workerURL = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    }
    return new Worker(workerURL);
  }

  // ====================================================== page side
  const QUERY = new URLSearchParams(location.search);
  const SYNC = QUERY.get("sync") === "1";
  const FAST = { nDir: 240, steps: 150 };                         // while interacting
  const FULL = { nDir: 1024, dt: 0.01, steps: 1200, chunk: 32 };  // when idle for IDLE_MS (dt grows beyond r_max 12)
  const K_EVERY = 2;      // full pass: curvature at every 2nd sample of a ray (still finer than the pixels)
  const CLOUD = { count: 20000, preview: 3000, tries: 10 }, SLICE = { count: 12000, preview: 1000, tries: 8 };
  const N_TRACK = 16, IDLE_MS = 150;
  const POOL = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));   // workers, shared by all widgets
  const VIEW_D = 4, VIEW_F = 0.36;                   // outside view: default camera distance, focal factor
  const R_RANGE = [0.5, 40], D_RANGE = [1.3, 30];    // zoom limits (r_max; camera distance in cloud radii)
  const K_LOG = [-2, 2];                             // curvature view: log10 range of the saturating |K|
  const AXES = "xyzw", DEG = Math.PI / 180;
  // slice mode: the world axes [right, up, into the screen] for fixed x, y, z, w (w up where it is seen),
  // then the view turned by SLICE_TURN about the vertical and tilted by SLICE_TILT to look down a little
  const SLICE_VIEW = [[1, 3, 2], [0, 3, 2], [0, 3, 1], [0, 1, 2]], SLICE_TURN = 30, SLICE_TILT = 20;
  // axis gizmo: x red, y green, z blue, w amber (as in the videos), for light and for dark backgrounds
  const AXIS_COLORS = [["#d42a2a", "#2b8a3e", "#1864c8", "#b07400"], ["#ff6b6b", "#5bd46f", "#5cb1f7", "#fcc419"]];
  const TOUCH = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  const HINTS = TOUCH
    ? ["drag: walk · two fingers: turn, pinch: zoom", "drag: orbit · two fingers: 4D, pinch: zoom", "drag: orbit · two fingers: turn, pinch: zoom"]
    : ["drag or W/S: walk · right-drag or Q/E: turn · wheel: zoom", "drag: orbit · shift-drag: rotate in 4D · wheel: zoom",
       "drag: orbit · wheel: zoom"];   // [inside, outside: projection, outside: slice]
  const HELP = [
    "Walk: drag, W/S or ↑/↓.  Turn: right-drag or shift-drag around the centre, two-finger rotate, Q/E, A/D or ←/→ " +
    "(click the view first for keys).  Zoom: wheel or pinch changes r_max, how far the scan reaches.  Double-click: back to the start.",
    "Projection: the whole surface with one 4D direction dropped; drag orbits the 3D picture, shift-drag, right-drag or two " +
    "fingers rotate in 4D (change the dropped direction).  Slice: the surface where one coordinate (x, y, z or w) has the " +
    "chosen value, drawn in the other three; drag orbits.  Zoom: wheel or pinch.  Double-click: reset the view.",
  ];

  /** The worker pool shared by all widgets of the page; replies are routed by
   *  m.owner.  Without workers (or if they fail to start, e.g. in a sandbox
   *  that forbids them) the same scheduler runs on the page. */
  const pool = {
    demos: [],
    workers: null,
    get() {
      if (this.workers) return this.workers;
      if (!SYNC) {
        try {
          return (this.workers = Array.from({ length: POOL }, () => {
            const w = makeWorker();
            w.onmessage = (e) => this.receive(e.data);
            w.onerror = (e) => this.fail(e);
            return w;
          }));
        } catch (err) {
          console.warn("co2-demo: no workers, computing on the page", err);
        }
      }
      return (this.workers = [this.local()]);
    },
    local() {
      const K = co2sdf();
      return { postMessage: co2scheduler(K, co2scenes(K), (m) => this.receive(m), SYNC), terminate() {} };
    },
    receive(m) {
      const d = this.demos[m.owner];
      if (d) d.receive(m);
    },
    fail(err) {
      if (this.failed) return;
      this.failed = true;
      console.warn("co2-demo: worker failed, computing on the page", err);
      for (const w of this.workers) w.terminate();
      this.workers = [this.local()];
      for (const d of this.demos) { d.drop(); d.resume(); }
    },
  };

  function el(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    if (parent) parent.append(e);
    return e;
  }

  const decimals = (step) => Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  const clamp = (x, [lo, hi]) => Math.min(Math.max(x, lo), hi);
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  const done = (r, id) => r && r.id === id && !r.partial;

  /** [luma 0-1, alpha] of a computed CSS colour "rgb(r, g, b)" / "rgba(r, g, b, a)". */
  function lumaAlpha(s) {
    const v = (s.match(/[\d.]+/g) || [0, 0, 0]).map(Number);
    return [(0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]) / 255, v.length > 3 ? v[3] : 1];
  }

  class Demo {
    constructor(root) {
      const K = (this.K = co2sdf());
      this.root = root;
      this.compact = "compact" in root.dataset && root.dataset.compact !== "0";
      this.owner = pool.demos.push(this) - 1;
      this.scenes = co2scenes(K);
      this.defaults = {};
      for (const S of Object.values(this.scenes)) {
        this.defaults[S.name] = Object.fromEntries(S.params.map((P) => [P.key, S[P.key]]));
      }
      this.N = K.normals();
      this.cam = K.vec(); this.e1 = K.vec(); this.e2 = K.vec();
      this.Rp = new Float64Array(16);                // 4D view rotation, rows = view axes (row 3 hidden): projection,
      this.Rs = new Float64Array(16);                // ... slice (row 3 = e_fix),
      this.R = this.Rp;                              // ... the one in use
      this.version = 0;                              // bumps on every change of camera/scene/params/r_max
      this.req = 0;                                  // scan requests (to merge the per-worker parts)
      this.cloudId = this.sliceId = this.sliceFloor = 0;   // wanted cloud / slice; older slices are of another surface
      this.cloudSent = this.sliceSent = -1;
      this.scan = this.cloud = this.slice = null;
      this.scr = new Float64Array(3);
      this.parts = {};                               // scan parts being merged, per quality
      this.fastBusy = this.sliceBusy = false;
      this.visible = SYNC;
      this.keys = new Set();
      this.idxCache = new Map();
      this.mode = (QUERY.get("outside") || root.dataset.outside) === "slice" ? "slice" : "projection";
      const fix = [...AXES].indexOf(QUERY.get("slice") || root.dataset.slice);
      this.fix = fix >= 0 ? fix : 1;                 // slice: the fixed coordinate p[fix] = c (default y) ...
      this.cAt = [0, 0, 0, 0];                       // ... and the values the other choices had
      this.kOn = (QUERY.get("curvature") || root.dataset.curvature) === "1";   // the curvature view
      this.kMax = 1;                                 // ... saturates at |K| = kMax
      this.build();
      this.bind();
      this.showCurvature();
      this.resize();
      const name = QUERY.get("scene") || root.dataset.scene;
      this.setScene(this.scenes[name] ? name : "neck");
    }

    // ------------------------------------------------------------ DOM
    build() {
      const root = this.root, compact = this.compact;
      root.textContent = "";
      root.classList.toggle("co2-compact", compact);
      const bar = el("div", "co2-bar");
      if (!compact) {
        const field = el("label", "co2-field", bar);
        el("span", "co2-label", field, "Scene");
        this.sel = el("select", "co2-select", field);
        for (const S of Object.values(this.scenes)) this.sel.add(new Option(S.title, S.name));
      }
      this.paramsEl = el("div", "co2-params", bar);
      const kField = el("label", "co2-field", bar);
      this.kBox = el("input", "", kField);
      this.kBox.type = "checkbox";
      el("span", "co2-label", kField, "curvature");
      kField.title = "Colour by Gaussian curvature: grey = flat, blue = negative (saddle), orange = positive (dome)";
      this.kRange = el("label", "co2-field", bar);
      el("span", "co2-label", this.kRange, "K range");
      this.kInput = el("input", "", this.kRange);
      Object.assign(this.kInput, { type: "range", min: K_LOG[0], max: K_LOG[1], step: 0.05, value: Math.log10(this.kMax) });
      this.kShow = el("output", "", this.kRange);
      this.kRange.title = "the |K| at which the colours are fully blue or orange";
      this.resetEl = el("button", "co2-btn", bar, compact ? "↺" : "Reset");
      Object.assign(this.resetEl, { type: "button", title: "Reset the scene, the camera and the view" });
      this.infoEl = el("div", "co2-info", compact ? bar : null);
      if (!compact) {
        root.append(bar, this.infoEl);
        const details = el("details", "co2-code", root);       // the scene's pseudocode, as on the page
        el("summary", "", details, "code");
        this.codeEl = el("pre", "", details);
      }

      const views = el("div", "co2-views", root);
      const view = (title, hint, help) => {
        const fig = el("figure", "co2-view", views), frame = el("div", "co2-frame", fig);
        const canvas = el("canvas", "co2-canvas", frame), overlay = el("div", "co2-overlay", frame);
        const cap = el("figcaption", "co2-cap", fig);
        el("b", "", cap, title);
        const status = el("span", "co2-status", cap), hintEl = el("div", "co2-hint", cap, hint);
        hintEl.title = help;
        return [canvas, status, overlay, hintEl];
      };
      let overlay;
      const hint = (h) => compact ? h.replace("wheel: zoom", "ctrl+wheel: zoom") : h;
      this.hint = hint;                                             // the outside hint changes with the mode
      [this.cIn, this.statusIn, overlay] = view(compact ? "Inside" : "Inside · exponential map", hint(HINTS[0]), HELP[0]);
      this.rEl = el("span", "co2-badge", overlay);                  // r_max readout
      this.kLegend = el("span", "co2-badge co2-legend", overlay);   // curvature colours
      this.kLegend.title = "Gaussian curvature K: blue < 0 (saddle), grey = 0 (flat), orange > 0 (dome)";
      this.kLo = el("span", "", this.kLegend);
      el("i", "co2-kbar", this.kLegend);
      this.kHi = el("span", "", this.kLegend);
      [this.cOut, this.statusOut, overlay, this.hintOut] = view(compact ? "Outside" : "Outside · R⁴ → R³", hint(HINTS[1]), HELP[1]);
      const seg = el("div", "co2-seg", overlay);
      this.modeEls = ["projection", "slice"].map((mode) => {
        const b = el("button", "", seg, mode);
        b.type = "button";
        b.addEventListener("click", () => this.setMode(mode));
        return b;
      });
      // slice mode: which coordinate is fixed (x | y | z | w) and its value
      this.sliceEl = el("div", "co2-slice", overlay);
      const fixSeg = el("div", "co2-seg", this.sliceEl);
      fixSeg.setAttribute("role", "group");
      fixSeg.setAttribute("aria-label", "fixed coordinate");
      this.fixEls = [...AXES].map((a, k) => {
        const b = el("button", "", fixSeg, a);
        b.type = "button";
        b.title = `slice at a fixed ${a}: the surface in ${[...AXES].filter((_, j) => j !== k).join(", ")}`;
        b.addEventListener("click", () => this.setFix(k));
        return b;
      });
      this.cField = el("label", "co2-field co2-badge co2-live", this.sliceEl);
      this.cLabel = el("span", "co2-label", this.cField);
      this.cInput = el("input", "", this.cField);
      this.cInput.type = "range";
      this.cShow = el("output", "", this.cField);
      this.cField.title = "the value of the fixed coordinate (the slider spans the scene)";
      if (compact && root.dataset.params !== "0") root.append(bar);

      this.cIn.tabIndex = 0;
      this.cIn.setAttribute("aria-label", "Inside view: the surface around the camera. " + HELP[0]);
      this.cOut.setAttribute("aria-label", "Outside view: the surface seen from R4. " + HELP[1]);
      this.cOut.classList.add("co2-outside");
      this.ctxIn = this.cIn.getContext("2d");
      this.ctxOut = this.cOut.getContext("2d");
    }

    buildParams() {
      const S = this.S;
      this.paramsEl.textContent = "";
      for (const P of S.params) {
        const field = el("label", "co2-field", this.paramsEl);
        el("span", "co2-label", field, P.label);
        const input = el("input", "", field), out = el("output", "", field);
        Object.assign(input, { type: "range", min: P.min, max: P.max, step: P.step, value: S[P.key] });
        const show = () => { out.textContent = S[P.key].toFixed(decimals(P.step)); };
        show();
        input.addEventListener("input", () => { S[P.key] = +input.value; show(); this.paramsChanged(); });
      }
      this.infoEl.textContent = S.info ? S.info() : "";
      if (this.codeEl) this.codeEl.textContent = S.code || "";
    }

    bind() {
      if (this.sel) this.sel.addEventListener("change", () => this.setScene(this.sel.value));
      this.resetEl.addEventListener("click", () => this.setScene(this.S.name, true));

      // inside: keyboard (while focused), drag = walk, right/shift-drag or two fingers = turn, wheel/pinch = r_max
      const c = this.cIn;
      const KEYS = { KeyW: "f", ArrowUp: "f", KeyS: "b", ArrowDown: "b", KeyA: "l", ArrowLeft: "l", KeyD: "r", ArrowRight: "r", KeyQ: "l", KeyE: "r" };
      c.addEventListener("keydown", (e) => {
        const k = KEYS[e.code];
        if (!k || e.altKey || e.ctrlKey || e.metaKey) return;
        e.preventDefault();
        this.keys.add(k);
        if (!this.raf) { this.lastT = performance.now(); this.raf = requestAnimationFrame((t) => this.animate(t)); }
      });
      c.addEventListener("keyup", (e) => { if (KEYS[e.code]) this.keys.delete(KEYS[e.code]); });
      c.addEventListener("blur", () => this.keys.clear());
      this.gestures(c, {
        drag: (dx, dy, alt, e) => {
          if (alt) {                                 // the picture follows the pointer around the centre
            const r = c.getBoundingClientRect(), x = e.clientX - r.left - r.width / 2, y = e.clientY - r.top - r.height / 2;
            return this.turn(wrap(Math.atan2(y, x) - Math.atan2(y - dy, x - dx)));
          }
          const dist = (Math.hypot(dx, dy) / (c.clientWidth / 2)) * this.rMax;
          if (dist > 0) this.walkScreen(dx, -dy, dist);
        },
        pinch: (dx, dy, scale, angle) => { this.turn(angle); this.setRMax(this.rMax / scale); },
        wheel: (f) => this.setRMax(this.rMax * f),
      });
      c.addEventListener("dblclick", () => { this.resetCamera(); this.setRMax(this.S.rMax); this.changed(); });

      // outside: drag = 3D orbit, shift/right-drag or two fingers = 4D rotation (orbit in slice mode), wheel/pinch = distance
      this.gestures(this.cOut, {
        drag: (dx, dy, alt) => (alt ? this.rotate4(dx, dy) : this.orbit(dx, dy)),
        pinch: (dx, dy, scale, angle) => { this.rotate(0, 1, -angle); this.dist = clamp(this.dist / scale, D_RANGE); this.rotate4(dx, dy); },
        wheel: (f) => { this.dist = clamp(this.dist * f, D_RANGE); this.drawOutside(); },
      });
      this.cOut.addEventListener("dblclick", () => {
        if (this.mode === "slice") this.sliceView(); else this.resetView();
        this.dist = VIEW_D;
        this.drawOutside();
      });
      this.cInput.addEventListener("input", () => { this.c = +this.cInput.value; this.showC(); this.sliceMoved(); });
      this.kBox.addEventListener("change", () => this.setCurvature(this.kBox.checked, this.kMax));
      this.kInput.addEventListener("input", () => this.setCurvature(true, 10 ** +this.kInput.value));

      const ro = new ResizeObserver(() => this.resize());
      ro.observe(this.cIn);
      ro.observe(this.cOut);
      if (!SYNC) {
        new IntersectionObserver((entries) => this.setVisible(entries[entries.length - 1].isIntersecting)).observe(this.root);
      }
    }

    /** Pointer gestures on a canvas, in CSS pixels: h.drag(dx, dy, alt, e) for
     *  one pointer (alt: right button or shift), h.pinch(dx, dy, scale, angle)
     *  for two (centroid motion, distance ratio, turn of the line through them,
     *  y down), h.wheel(factor) for the wheel (and trackpad pinch). */
    gestures(canvas, h) {
      const ptr = new Map();
      let two = null;
      const pair = () => {
        const [p, q] = ptr.values();
        return { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2, d: Math.hypot(q.x - p.x, q.y - p.y) || 1, a: Math.atan2(q.y - p.y, q.x - p.x) };
      };
      canvas.addEventListener("pointerdown", (e) => {
        if (canvas === this.cIn) canvas.focus({ preventScroll: true });
        try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
        ptr.set(e.pointerId, { x: e.clientX, y: e.clientY, alt: e.button === 2 });
        two = ptr.size === 2 ? pair() : null;
        e.preventDefault();
      });
      canvas.addEventListener("pointermove", (e) => {
        const p = ptr.get(e.pointerId);
        if (!p) return;
        const dx = e.clientX - p.x, dy = e.clientY - p.y;
        p.x = e.clientX; p.y = e.clientY;
        if (ptr.size === 1) h.drag(dx, dy, p.alt || e.shiftKey, e);
        else if (two) {
          const q = pair();
          h.pinch(q.x - two.x, q.y - two.y, q.d / two.d, wrap(q.a - two.a));
          two = q;
        }
      });
      const up = (e) => { ptr.delete(e.pointerId); two = ptr.size === 2 ? pair() : null; };
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointercancel", up);
      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      canvas.addEventListener("wheel", (e) => {
        if (this.compact && !(e.ctrlKey || e.metaKey)) return;   // inline widgets: let the page scroll
        e.preventDefault();
        const dy = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);   // lines / pages -> pixels
        h.wheel(Math.exp(clamp(dy * 0.0015, [-0.5, 0.5])));
      }, { passive: false });
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      let any = false;
      for (const c of [this.cIn, this.cOut]) {
        const w = Math.round(Math.min(1024, Math.max(160, (c.clientWidth || 360) * dpr)));
        if (c.width !== w) { c.width = c.height = w; any = true; }
      }
      if (any) { this.drawInside(); this.drawOutside(); }
    }

    setVisible(v) {
      if (v === this.visible) return;
      this.visible = v;
      if (v) return this.resume();
      for (const w of pool.get()) w.postMessage({ kind: "cancel", owner: this.owner });
      this.drop();
    }

    // ------------------------------------------------------------ requests
    /** Forget the requests in flight (cancelled or lost); resume() re-sends what is missing. */
    drop() {
      clearTimeout(this.idle);
      this.parts = {};
      this.fastBusy = this.sliceBusy = false;
      if (!done(this.cloud, this.cloudId)) this.cloudSent = -1;
      if (!done(this.slice, this.sliceId)) this.sliceSent = -1;
    }

    resume() { this.schedule(); this.sendCloud(); this.sendSlice(); }

    receive(m) {
      if (m.kind === "scan") this.onScanPart(m);
      else if (m.kind === "cloud") this.onCloud(m);
      else this.onSlice(m);
    }

    params() { return Object.fromEntries(this.S.params.map((P) => [P.key, this.S[P.key]])); }

    /** A scan, split into contiguous ray ranges over the workers. */
    sendScan(quality) {
      const full = quality === "full", nDir = full ? FULL.nDir : FAST.nDir, W = pool.get(), n = W.length;
      const m = {
        kind: "scan", owner: this.owner, id: this.version, req: ++this.req, parts: n, quality, scene: this.S.name, params: this.params(),
        cam: Array.from(this.cam), e1: Array.from(this.e1), e2: Array.from(this.e2), nDir, rMax: this.rMax, nTrack: N_TRACK,
        dt: full ? Math.max(FULL.dt, this.rMax / FULL.steps) : this.rMax / FAST.steps, chunk: full ? FULL.chunk : nDir,
        kScale: this.kScale(), kEvery: full ? K_EVERY : 1,
      };
      W.forEach((w, i) => w.postMessage({ ...m, j0: Math.round((i * nDir) / n), j1: Math.round(((i + 1) * nDir) / n) }));
    }

    /** Merge the per-worker ray ranges of one scan; complete -> onScan. */
    onScanPart(m) {
      let acc = this.parts[m.quality];
      if (acc && m.req < acc.req) return;             // part of an aborted scan
      if (!acc || m.req > acc.req) {
        acc = this.parts[m.quality] = Object.assign({}, m, {
          cols: new Uint8Array(m.nDir * m.nR * 3), paths: new Float32Array(m.paths.length), left: m.parts, ms: 0,
        });
      }
      acc.cols.set(m.cols, 3 * m.j0 * m.nR);
      const s = m.nDir / N_TRACK, k0 = Math.ceil(m.j0 / s), k1 = Math.ceil(m.j1 / s), L = 4 * m.nS;
      acc.paths.set(m.paths.subarray(k0 * L, k1 * L), k0 * L);
      acc.ms = Math.max(acc.ms, m.ms);
      if (--acc.left === 0) { delete this.parts[m.quality]; this.onScan(acc); }
    }

    /** At most one fast pass in flight; a full pass follows after IDLE_MS of
     *  quiet.  A fast request also aborts a running full pass in the worker. */
    schedule() {
      if (!this.visible) return;
      clearTimeout(this.idle);
      if (SYNC && !this.scan) return this.sendScan("full");   // first render of ?sync=1
      if (this.fastBusy) return;
      if (!this.scan || this.scan.id !== this.version) { this.fastBusy = true; return this.sendScan("fast"); }
      if (this.scan.quality !== "fast") return;
      const id = this.version;
      this.idle = setTimeout(() => {
        if (this.version !== id) return;
        this.statusIn.textContent += " · refining…";
        this.sendScan("full");
      }, IDLE_MS);
    }

    onScan(r) {
      if (r.quality === "fast") this.fastBusy = false;
      const cur = this.scan;
      if (!cur || r.id > cur.id || (r.id === cur.id && r.quality === "full")) {
        this.scan = r;
        this.drawInside();
        this.drawOutside();
      }
      if (r.quality === "fast") this.schedule();
    }

    sendCloud() {
      if (!this.visible || this.cloudSent === this.cloudId) return;
      this.cloudSent = this.cloudId;
      const W = pool.get();
      W[W.length - 1 - (this.owner % W.length)].postMessage({
        kind: "cloud", owner: this.owner, id: this.cloudId, scene: this.S.name, params: this.params(), kScale: this.kScale(), ...CLOUD,
      });
    }

    onCloud(r) {
      if (r.id !== this.cloudId) return;
      const c = this.centre;
      let rad = 1e-9;
      for (let i = 0; i < r.n; i++) {
        const o = 4 * i;
        rad = Math.max(rad, Math.hypot(r.pos[o] - c[0], r.pos[o + 1] - c[1], r.pos[o + 2] - c[2], r.pos[o + 3] - c[3]));
      }
      this.cloud = r;
      if (r.n) this.radius = rad;
      this.drawOutside();
    }

    /** Slices go one at a time: the first message of a job (its preview or
     *  its result) frees the slot, and a newer request then aborts the rest. */
    sendSlice() {
      if (this.mode !== "slice" || !this.visible || this.sliceBusy || this.sliceSent === this.sliceId) return;
      this.sliceBusy = true;
      this.sliceSent = this.sliceId;
      const W = pool.get();
      W[this.owner % W.length].postMessage({
        kind: "slice", owner: this.owner, id: this.sliceId, scene: this.S.name, params: this.params(),
        axis: this.fix, c: this.c, kScale: this.kScale(), ...SLICE,
      });
    }

    onSlice(r) {
      if (r.id === this.sliceSent) this.sliceBusy = false;
      const cur = this.slice;
      if (r.id >= this.sliceFloor && (!cur || r.id > cur.id || (r.id === cur.id && !r.partial))) {
        this.slice = r;
        if (this.mode === "slice") this.drawOutside();
      }
      this.sendSlice();
    }

    /** The value c of the fixed coordinate moved. */
    sliceMoved() {
      this.sliceId++;
      this.sendSlice();
    }

    // ------------------------------------------------------------ state
    setScene(name, reset = false) {
      const S = (this.S = this.scenes[name]);
      if (reset) { Object.assign(S, this.defaults[name]); S.update(); }
      if (this.sel) this.sel.value = name;
      this.buildParams();
      this.resetCamera();
      this.resetView();
      this.sliceView();
      this.rMax = S.rMax;
      this.dist = VIEW_D;
      this.cAt = S.box().map(([lo, hi]) => (lo <= 0 && hi >= 0 ? 0 : (lo + hi) / 2));   // slice: 0, or the box centre
      this.c = this.cAt[this.fix];
      this.cloud = this.slice = null;
      this.radius = 0;
      this.showMode();
      this.surfaceChanged();
    }

    paramsChanged() {
      const S = this.S;
      S.update();
      this.infoEl.textContent = S.info ? S.info() : "";
      this.keepOnSurface();
      this.surfaceChanged();
    }

    /** New scene or parameters: new cloud, slice and scan.  Until the cloud
     *  gives the radius of the outside view, guess it from the box. */
    surfaceChanged() {
      this.centre = this.S.centre();
      if (!this.radius) this.radius = 0.7 * Math.hypot(...this.S.box().map(([lo, hi]) => (hi - lo) / 2));
      this.sliceRange();
      this.cloudId++;
      this.sliceFloor = ++this.sliceId;
      this.sendCloud();
      this.sendSlice();
      this.changed();
    }

    /** Something changed: redraw the camera marker now, re-trace soon. */
    changed() {
      this.version++;
      this.rEl.textContent = `r_max = ${this.rMax.toFixed(this.rMax < 10 ? 2 : 1)}`;
      this.drawOutside();
      this.schedule();
    }

    setRMax(r) {
      r = clamp(r, R_RANGE);
      if (r !== this.rMax) { this.rMax = r; this.changed(); }
    }

    /** Projection or slice; entering slice mode starts from its default view. */
    setMode(mode) {
      if (mode === this.mode) return;
      this.mode = mode;
      if (mode === "slice") this.sliceView();
      this.showMode();
      this.sendSlice();
      this.drawOutside();
    }

    showMode() {
      const slice = this.mode === "slice";
      this.R = slice ? this.Rs : this.Rp;
      this.modeEls.forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === this.mode)));
      this.fixEls.forEach((b, k) => b.setAttribute("aria-pressed", String(k === this.fix)));
      this.sliceEl.hidden = !slice;
      this.hintOut.textContent = this.hint(HINTS[slice ? 2 : 1]);
    }

    /** Slice mode: fix coordinate k instead (at the value it had before, if any). */
    setFix(k) {
      if (k === this.fix) return;
      this.cAt[this.fix] = this.c;
      this.fix = k;
      this.c = this.cAt[k];
      this.sliceRange();
      this.sliceView();
      this.showMode();
      this.slice = null;                             // the old curves are in another 3D space
      this.sliceFloor = ++this.sliceId;
      this.sendSlice();
      this.drawOutside();
    }

    /** The slider for the fixed coordinate spans the scene's box; c is kept inside it. */
    sliceRange() {
      const [lo, hi] = this.S.box()[this.fix];
      this.c = clamp(this.c, [lo, hi]);
      Object.assign(this.cInput, { min: lo, max: hi, step: (hi - lo) / 200 });
      this.cInput.value = this.c;
      this.cLabel.textContent = AXES[this.fix] + " =";
      this.showC();
    }

    showC() { this.cShow.textContent = (Math.abs(this.c) < 0.005 ? 0 : this.c).toFixed(2); }   // no "-0.00"

    /** Curvature view: the scale of the colours (0: off, the scene's own colours). */
    kScale() { return this.kOn ? 1 / this.kMax : 0; }

    /** The curvature view on or off, saturating at |K| = kMax: new colours in
     *  both views (the camera stays). */
    setCurvature(on, kMax) {
      this.kOn = on;
      this.kMax = kMax;
      this.showCurvature();
      this.cloudId++;
      this.sliceFloor = ++this.sliceId;
      this.sendCloud();
      this.sendSlice();
      this.changed();
    }

    showCurvature() {
      const k = this.kMax, t = k >= 10 ? k.toFixed(0) : k >= 1 ? k.toFixed(1) : k.toFixed(2);
      this.kBox.checked = this.kOn;
      this.kRange.hidden = this.kLegend.hidden = !this.kOn;
      this.kShow.textContent = "±" + t;
      this.kLo.textContent = "−" + t;
      this.kHi.textContent = "+" + t;
    }

    resetCamera() {
      const { K, S, N } = this;
      this.cam.set(S.cam);
      K.project(S, this.cam, N, 8);
      K.tangentFrame(S, this.cam, S.hint1, S.hint2, this.e1, this.e2, N);
    }

    /** After a parameter change: project the camera onto the new surface and
     *  its frame onto the new tangent plane (or start over if that fails). */
    keepOnSurface() {
      const { K, S, N, cam, e1, e2 } = this;
      K.project(S, cam, N, 8);
      K.transport(K.normalsAt(S, cam, N), e1, e2);
      const ok = K.cabs(S.F(cam, K.cx())) < 1e-8 && [...cam, ...e1, ...e2].every(Number.isFinite);
      if (!ok) this.resetCamera();
    }

    /** Walk `dist` along the geodesic in screen direction (x right, y up). */
    walkScreen(x, y, dist) {
      const { K, S, N, cam, e1, e2 } = this, d = K.vec();
      const saved = [cam.slice(), e1.slice(), e2.slice()];
      for (let k = 0; k < 4; k++) d[k] = x * e1[k] + y * e2[k];
      K.walk(S, cam, K.normalize(d), e1, e2, dist, N);
      if (![...cam, ...e1, ...e2].every(Number.isFinite)) { cam.set(saved[0]); e1.set(saved[1]); e2.set(saved[2]); return; }
      this.changed();
    }

    /** Turn the camera frame by angle a (a > 0: to the left; the picture turns clockwise). */
    turn(a) {
      const { e1, e2 } = this, c = Math.cos(a), s = Math.sin(a);
      for (let k = 0; k < 4; k++) {
        const x = e1[k], y = e2[k];
        e1[k] = c * x + s * y;
        e2[k] = -s * x + c * y;
      }
      this.changed();
    }

    animate(t) {
      const dt = Math.min(0.05, (t - this.lastT) / 1000), ks = this.keys;
      this.lastT = t;
      const fwd = ks.has("f") - ks.has("b"), turn = ks.has("l") - ks.has("r");
      if (turn) this.turn(turn * 1.6 * dt);
      if (fwd) this.walkScreen(0, fwd, 0.3 * this.rMax * dt);
      this.raf = ks.size ? requestAnimationFrame((t2) => this.animate(t2)) : 0;
    }

    /** The projection's view: the scene's own 4D rotation. */
    resetView() {
      const R = this.Rp;
      R.fill(0);
      for (let i = 0; i < 4; i++) R[5 * i] = 1;
      for (const [i, j, deg] of this.S.view) this.rotate(i, j, deg * DEG, R);
    }

    /** The slice's view: the hidden view axis is exactly e_fix, the other
     *  three world axes face right, up and into the screen (view axis 2
     *  points at the viewer), turned and tilted a little.  Orbits keep row 3. */
    sliceView() {
      const R = this.Rs, [right, up, into] = SLICE_VIEW[this.fix];
      R.fill(0);
      R[right] = 1; R[4 + up] = 1; R[8 + into] = -1; R[12 + this.fix] = 1;
      this.rotate(0, 2, SLICE_TURN * DEG, R);
      this.rotate(1, 2, SLICE_TILT * DEG, R);
    }

    /** Outside view: orbit the projected 3D picture (view axes 0-2 only) ... */
    orbit(dx, dy) {
      this.rotate(0, 2, -0.01 * dx);
      this.rotate(1, 2, 0.01 * dy);
      this.drawOutside();
    }

    /** ... or rotate in 4D, turning the screen axes into the hidden axis 3
     *  (in slice mode the hidden axis is the fixed coordinate: orbit instead). */
    rotate4(dx, dy) {
      if (this.mode === "slice") return this.orbit(dx, dy);
      this.rotate(0, 3, -0.01 * dx);
      this.rotate(1, 3, 0.01 * dy);
      this.drawOutside();
    }

    /** Rotate the view R in the plane of view axes i, j (left-multiplication). */
    rotate(i, j, a, R = this.R) {
      const c = Math.cos(a), s = Math.sin(a);
      for (let k = 0; k < 4; k++) {
        const x = R[4 * i + k], y = R[4 * j + k];
        R[4 * i + k] = c * x - s * y;
        R[4 * j + k] = s * x + c * y;
      }
      for (let i2 = 0; i2 < 4; i2++) {               // re-orthonormalise the rows
        for (let j2 = 0; j2 < i2; j2++) {
          let d = 0;
          for (let k = 0; k < 4; k++) d += R[4 * i2 + k] * R[4 * j2 + k];
          for (let k = 0; k < 4; k++) R[4 * i2 + k] -= d * R[4 * j2 + k];
        }
        let n = 0;
        for (let k = 0; k < 4; k++) n += R[4 * i2 + k] ** 2;
        for (let k = 0; k < 4; k++) R[4 * i2 + k] /= Math.sqrt(n);
      }
    }

    // ------------------------------------------------------------ drawing
    drawInside() {
      const r = this.scan, res = this.cIn.width;
      if (!r) return;
      const key = `${res} ${r.nDir} ${r.nR} ${r.dt}`;
      let idx = this.idxCache.get(key);
      if (!idx) {
        if (this.idxCache.size > 3) this.idxCache.clear();
        this.idxCache.set(key, (idx = this.K.scanIndex(res, r.nDir, r.nR, r.dt, 0)));   // no marker: the camera is drawn
      }
      if (!this.imgIn || this.imgIn.width !== res) this.imgIn = this.ctxIn.createImageData(res, res);
      this.K.scanImage(r.cols, idx, this.imgIn.data, [0, 0, 0, 0]);
      this.ctxIn.putImageData(this.imgIn, 0, 0);
      const n = pool.get().length, threads = n > 1 ? ` (${n} workers)` : "";
      this.statusIn.textContent = `${r.quality} · ${r.nDir} rays · ${Math.round(r.ms)} ms${threads}`;
    }

    /** World point P[o..o+3] -> [screen x, screen y, depth] (in this.scr), or
     *  null behind the camera: rotate by R, drop view axis 3 (the hidden
     *  one), perspective from distance this.dist along view axis 2. */
    toScreen(P, o, W) {
      const R = this.R, c = this.centre, s = 1 / this.radius, out = this.scr;
      const d0 = (P[o] - c[0]) * s, d1 = (P[o + 1] - c[1]) * s, d2 = (P[o + 2] - c[2]) * s, d3 = (P[o + 3] - c[3]) * s;
      const z = R[8] * d0 + R[9] * d1 + R[10] * d2 + R[11] * d3;
      if (this.dist - z < 0.05) return null;
      const f = (VIEW_F * W * VIEW_D) / (this.dist - z);
      out[0] = W / 2 + f * (R[0] * d0 + R[1] * d1 + R[2] * d2 + R[3] * d3);
      out[1] = W / 2 - f * (R[4] * d0 + R[5] * d1 + R[6] * d2 + R[7] * d3);
      out[2] = z;
      return out;
    }

    drawOutside() {
      const ctx = this.ctxOut, W = this.cOut.width, slice = this.mode === "slice", pts = slice ? this.slice : this.cloud;
      if (!this.centre) return;
      if (!this.imgOut || this.imgOut.width !== W) {
        this.imgOut = ctx.createImageData(W, W);
        this.zbuf = new Float32Array(W * W);
      }
      const img = this.imgOut.data;
      img.fill(0);
      this.zbuf.fill(-Infinity);
      // slice mode: the slice curves, then the projected surface faintly behind them
      if (pts) this.splat(pts, W, slice ? W / 110 : W / 220, slice ? 0.25 : 0.5, 255);
      if (slice && this.cloud) this.splat(this.cloud, W, 2, 0.5, 40);
      ctx.putImageData(this.imgOut, 0, 0);

      const css = getComputedStyle(this.root), accent = css.getPropertyValue("--co2-accent").trim() || "#d6336c";
      const fg = css.color, u = W / 360;               // ~1 CSS px at 360 px width
      ctx.lineJoin = ctx.lineCap = "round";
      const r = this.scan;
      if (r && !slice) {                              // the tracked inside-view rays
        ctx.strokeStyle = accent;
        for (let k = 0; k < N_TRACK; k++) {
          ctx.globalAlpha = k === N_TRACK / 4 ? 1 : 0.75;   // ray k = N/4 points "forward" (up)
          ctx.lineWidth = (k === N_TRACK / 4 ? 2.2 : 1.1) * u;
          ctx.beginPath();
          for (let i = 0, pen = false; i < r.nS; i++) {
            const q = this.toScreen(r.paths, 4 * (k * r.nS + i), W);
            if (q) ctx[pen ? "lineTo" : "moveTo"](q[0], q[1]);
            pen = !!q;
          }
          ctx.stroke();
        }
      }
      // the camera (in slice mode faint when it is off the slice)
      const cam = this.cam, radius = this.radius, fix = this.fix;
      const off = slice && Math.abs(cam[fix] - this.c) > 0.03 * radius;
      const ahead = cam.map((x, k) => x + 0.12 * radius * this.e2[k]), q0 = this.toScreen(cam, 0, W);
      if (q0) {
        const [cx, cy] = q0, q1 = this.toScreen(ahead, 0, W);
        ctx.globalAlpha = off ? 0.35 : 1;
        ctx.strokeStyle = fg;
        ctx.lineWidth = 2 * u;
        if (q1) { ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(q1[0], q1[1]); ctx.stroke(); }   // forward direction
        ctx.fillStyle = accent;
        ctx.beginPath(); ctx.arc(cx, cy, 4.5 * u, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }

      // axis gizmo: projections of the world axes x, y, z, w (in slice mode the
      // three that span the slice), coloured, with a halo of the background
      const g = 22 * u, gx = 30 * u, gy = W - 30 * u, R = this.R;
      const dark = lumaAlpha(fg)[0] > 0.5, colors = AXIS_COLORS[dark ? 1 : 0], bgc = css.backgroundColor;
      const bg = lumaAlpha(bgc)[1] > 0 ? bgc : dark ? "#000" : "#fff";
      ctx.font = `bold ${11 * u}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let k = 0; k < 4; k++) {
        if (slice && k === fix) continue;
        const ax = R[k], ay = R[4 + k], lx = gx + (g + 8 * u) * ax, ly = gy - (g + 8 * u) * ay;
        ctx.strokeStyle = colors[k];
        ctx.lineWidth = 1.6 * u;
        ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gx + g * ax, gy - g * ay); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = bg;
        ctx.lineWidth = 3 * u;
        ctx.strokeText(AXES[k], lx, ly);
        ctx.fillStyle = colors[k];
        ctx.fillText(AXES[k], lx, ly);
      }

      const axis = AXES[fix], value = this.cShow.textContent, rest = [...AXES].filter((_, k) => k !== fix).join(", ");
      ctx.fillStyle = fg;
      if (slice) {                                    // which 3D space this is
        ctx.globalAlpha = 0.7;
        ctx.font = `${10 * u}px sans-serif`;
        ctx.textAlign = "right";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(`${rest} at ${axis} = ${value}`, W - 8 * u, W - 8 * u);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
      }
      if (slice && pts && !pts.partial && !pts.n) {
        ctx.globalAlpha = 0.7;
        ctx.font = `${13 * u}px sans-serif`;
        ctx.fillText(`empty slice: move ${axis}`, W / 2, W / 2);
      }
      ctx.globalAlpha = 1;
      let t = pts ? `${pts.n} points${pts.partial ? "…" : ` · ${Math.round(pts.ms)} ms`}` : "sampling…";
      if (slice) t += ` · slice ${axis} = ${value}, axes ${rest}`;
      this.statusOut.textContent = t;
    }

    /** Z-buffered square splats of points {n, pos, col} into imgOut: size ~
     *  size0 px at the default distance (nearer = bigger), depth cue from
     *  1 - dark (far) to 1 (near).  alpha < 255: only into empty pixels. */
    splat(pts, W, size0, dark, alpha) {
      const img = this.imgOut.data, zb = this.zbuf;
      for (let i = 0; i < pts.n; i++) {
        const q = this.toScreen(pts.pos, 4 * i, W);
        if (!q) continue;
        const z = q[2], ps = Math.min(16, Math.max(2, Math.round((size0 * VIEW_D) / (this.dist - z))));
        const x0 = Math.round(q[0] - ps / 2), y0 = Math.round(q[1] - ps / 2);
        if (x0 < 0 || y0 < 0 || x0 + ps > W || y0 + ps > W) continue;
        const shade = 1 - dark + 0.5 * dark * Math.min(Math.max(z + 1, 0), 2);   // far = darker
        for (let y = y0; y < y0 + ps; y++) {
          for (let x = x0; x < x0 + ps; x++) {
            const px = y * W + x;
            if (alpha < 255 ? img[4 * px + 3] !== 0 : z <= zb[px]) continue;
            zb[px] = z;
            img[4 * px] = pts.col[3 * i] * shade;
            img[4 * px + 1] = pts.col[3 * i + 1] * shade;
            img[4 * px + 2] = pts.col[3 * i + 2] * shade;
            img[4 * px + 3] = alpha;
          }
        }
      }
    }
  }

  function mountAll() {
    for (const root of document.querySelectorAll(".co2-demo")) if (!root.co2demo) root.co2demo = new Demo(root);
  }
  window.CO2Demo = { mount: (root) => root.co2demo || (root.co2demo = new Demo(root)), mountAll, pool };   // pool: for tests
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll);
  else mountAll();
})();
