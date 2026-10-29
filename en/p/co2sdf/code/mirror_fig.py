import numpy as np
from co2sdf import *
from scenes import SCENES
from render import step

def trace(name, th0):
    sc = SCENES[name]; F = sc.F
    cam = project(F, np.array([[-sc.R - 0.15, -1.9, 0.0, 0.0]]), 8)
    a, b = jac(F, cam)
    e1 = normalize(tangent(a, b, np.array([[0.0, 1.0, 0, 0]])))
    t2 = tangent(a, b, np.array([[1.0, 0, 0, 0]])); e2 = normalize(t2 - dot(t2, e1)[..., None] * e1)
    v = np.cos(th0) * e1 + np.sin(th0) * e2
    w = -np.sin(th0) * e1 + np.cos(th0) * e2
    p = cam.copy(); path = [p[0].copy()]; frames = [(p[0].copy(), v[0].copy(), w[0].copy())]
    been_up = False
    for i in range(2000):
        p, v = step(F, p, v, 0.01)
        a, b = jac(F, p); w = tangent(a, b, w); w = normalize(w - dot(w, v)[..., None] * v)
        path.append(p[0].copy())
        been_up |= p[0, 2] > 0.5
        back = abs(p[0, 2]) < 1e-3 and abs(p[0, 3]) < 1e-3
        if been_up and back and np.hypot(p[0, 0] - sc.R, p[0, 1]) > sc.r + sc.k + 0.75:
            break
    frames.append((p[0].copy(), v[0].copy(), w[0].copy()))
    return np.array(path), frames, sc

def svg_panel(name, th0, ox, title):
    path, frames, sc = trace(name, th0)
    S = 46.0  # px per unit
    def X(q): return ox + 150 + S * q[0]
    def Y(q): return 175 - S * q[1]
    out = []
    out.append('<text x="%.1f" y="28" font-size="15" font-weight="700" fill="var(--ink)">%s</text>' % (ox + 20, title))
    for m in (-sc.R, sc.R):
        out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="var(--paper)" stroke="var(--ink-2)" stroke-width="1.5"/>' % (ox + 150 + S * m, 175, S * sc.r))
    out.append('<text x="%.1f" y="%.1f" font-size="12" fill="var(--ink-2)" text-anchor="middle">mouth 1</text>' % (ox + 150 - S * sc.R, 175 + S * sc.r + 16))
    out.append('<text x="%.1f" y="%.1f" font-size="12" fill="var(--ink-2)" text-anchor="middle">mouth 2</text>' % (ox + 150 + S * sc.R, 175 + S * sc.r + 16))
    onp = (np.abs(path[:, 2]) < 1e-3) & (np.abs(path[:, 3]) < 1e-3)
    segs, cur, kind = [], [], None
    for q, o in zip(path, onp):
        if kind is None or o == kind:
            cur.append(q)
        else:
            segs.append((kind, cur)); cur = [cur[-1], q]
        kind = o
    segs.append((kind, cur))
    for k, s in segs:
        pts = " ".join("%.1f,%.1f" % (X(q), Y(q)) for q in s[::3] + [s[-1]])
        dash = '' if k else ' stroke-dasharray="4 4"'
        out.append('<polyline points="%s" fill="none" stroke="var(--orange)" stroke-width="2.2"%s/>' % (pts, dash))
    for (q, v, w), lab in zip(frames, ("start", "after the handle")):
        vv = v[:2] / np.linalg.norm(v[:2]); ww = w[:2] / np.linalg.norm(w[:2])
        L1, L2 = 0.62, 0.34
        a0 = (X(q), Y(q)); a1 = (X(q + L1 * np.r_[vv, 0, 0]), Y(q + L1 * np.r_[vv, 0, 0]))
        a2 = (X(q + L2 * np.r_[ww, 0, 0]), Y(q + L2 * np.r_[ww, 0, 0]))
        out.append('<polyline points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" stroke="var(--blue)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>' % (a1 + a0 + a2))
        out.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="var(--blue)"/>' % a0)
    return "\n".join(out), path

p1, path1 = svg_panel("handle", 0.12, 0, "Ordinary handle")
p2, path2 = svg_panel("klein", 0.12, 330, "Mirror handle")
svg = ('<svg class="diagram" viewBox="0 0 660 300" role="img" aria-label="Top view of the world plane with two mouths. An orange geodesic enters mouth 1, travels through the handle (dashed) and leaves mouth 2. A blue L-shaped frame is drawn at the start and at the end; through the ordinary handle the L comes back unmirrored, through the mirror handle it comes back mirrored.">\n'
       '<rect x="0" y="0" width="660" height="300" rx="12" fill="var(--paper-2)"/>\n' + p1 + "\n" + p2 + "\n"
       '<text x="20" y="288" font-size="12" fill="var(--ink-2)">Top view of the world plane. Solid: on the plane; dashed: inside the handle (projected). Blue L: bar = direction of travel, foot = transported side vector.</text>\n</svg>')
open("mirror_fig.svg", "w").write(svg)
for pth, nm in ((path1, "handle"), (path2, "klein")):
    print(nm, "end point", np.round(pth[-1], 3), "max x3 along path %.2f" % pth[:, 2].max())
