"""A tiny numpy ray tracer for explanatory 3D figures: translucent quads and discs
with grids, rods (cylinders), cones, balls; exact front-to-back blending of all
hits, Lambert + ambient shading, supersampling.  Labels are not drawn here: the
figure scripts project 3D anchor points with the same camera and write SVG text
over the image, so the text stays sharp and uses the page's font."""
import numpy as np
from PIL import Image

INF = 1e30


def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


LIGHT = norm([0.25, -0.65, 0.72])


def shade(n, rd, col, amb=0.55, diff=0.45):
    n = np.where((np.einsum('ij,ij->i', n, rd) > 0)[:, None], -n, n)      # face the viewer
    lam = np.clip(n @ LIGHT, 0, 1)
    return col * (amb + diff * lam)[:, None]


class Prim:
    def hits(self, ro, rd):
        raise NotImplementedError


class Patch(Prim):
    """A flat patch: centre c, half-axes u, v (vectors).  shape 'quad' or 'disc'.
    grid: spacing of grid lines in units of the patch's own length (None: no grid).
    back: colour of the other face (None: same).  tex(s, t) -> (mask, rgb) optional
    extra marking in local coordinates s, t in [-1, 1]."""
    def __init__(self, c, u, v, col, alpha=1.0, shape='quad', grid=None, edge=0.012, back=None, tex=None, line=0.55):
        self.c, self.u, self.v = map(lambda a: np.asarray(a, float), (c, u, v))
        self.col, self.alpha, self.shape, self.grid, self.edge = np.asarray(col, float), alpha, shape, grid, edge
        self.back = None if back is None else np.asarray(back, float)
        self.tex, self.line = tex, line
        self.n = norm(np.cross(self.u, self.v))
        self.lu, self.lv = np.linalg.norm(self.u), np.linalg.norm(self.v)

    def hits(self, ro, rd):
        den = rd @ self.n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((self.c - ro) @ self.n) / den
        q = ro + rd * t[:, None] - self.c
        s = q @ self.u / self.lu ** 2
        w = q @ self.v / self.lv ** 2
        inside = (s * s + w * w <= 1) if self.shape == 'disc' else (np.abs(s) <= 1) & (np.abs(w) <= 1)
        ok = inside & (t > 1e-6) & np.isfinite(t)
        col = np.repeat(self.col[None], len(t), 0)
        if self.back is not None:
            col = np.where((den > 0)[:, None], self.back[None], col)
        dark = np.zeros(len(t), bool)
        if self.grid:
            gs, gw = s * self.lu / self.grid, w * self.lv / self.grid
            fs, fw = gs - np.round(gs), gw - np.round(gw)
            px = 0.035
            dark |= (np.abs(fs) < px) | (np.abs(fw) < px)
        if self.edge:
            if self.shape == 'disc':
                dark |= np.sqrt(s * s + w * w) > 1 - self.edge * 2
            else:
                dark |= (np.abs(s) > 1 - self.edge * self.lv / self.lu * 2) | (np.abs(w) > 1 - self.edge * 2)
        col = np.where(dark[:, None], col * self.line, col)
        if self.tex is not None:
            m, tc = self.tex(s, w)
            col = np.where(m[:, None], tc, col)
        nn = np.repeat(self.n[None], len(t), 0)
        return [(np.where(ok, t, INF), nn, col, self.alpha)]


class Rod(Prim):
    """A cylinder from a to b with radius r (side surface; closed by caps)."""
    def __init__(self, a, b, r, col, alpha=1.0):
        self.a, self.b, self.r, self.col, self.alpha = np.asarray(a, float), np.asarray(b, float), r, np.asarray(col, float), alpha

    def hits(self, ro, rd):
        ax = self.b - self.a
        L = np.linalg.norm(ax); d = ax / L
        oc = ro - self.a
        rd_p = rd - (rd @ d)[:, None] * d
        oc_p = oc - (oc @ d)[:, None] * d
        A = np.einsum('ij,ij->i', rd_p, rd_p); B = 2 * np.einsum('ij,ij->i', rd_p, oc_p); C = np.einsum('ij,ij->i', oc_p, oc_p) - self.r ** 2
        disc = B * B - 4 * A * C
        sq = np.sqrt(np.maximum(disc, 0))
        out = []
        for sgn in (-1, 1):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (-B + sgn * sq) / (2 * A)
            h = (oc + rd * t[:, None]) @ d
            ok = (disc > 0) & (t > 1e-6) & (h >= 0) & (h <= L)
            q = oc + rd * t[:, None]
            n = norm(q - h[:, None] * d + 1e-12)
            out.append((np.where(ok, t, INF), n, np.repeat(self.col[None], len(t), 0), self.alpha))
        for cap, sgn in ((self.a, -1), (self.b, 1)):   # caps
            den = rd @ d
            with np.errstate(divide='ignore', invalid='ignore'):
                t = ((cap - ro) @ d) / den
            q = ro + rd * t[:, None] - cap
            ok = (np.einsum('ij,ij->i', q, q) <= self.r ** 2) & (t > 1e-6)
            out.append((np.where(ok, t, INF), np.repeat((sgn * d)[None], len(t), 0), np.repeat(self.col[None], len(t), 0), self.alpha))
        return out


class Ball(Prim):
    def __init__(self, c, r, col, alpha=1.0):
        self.c, self.r, self.col, self.alpha = np.asarray(c, float), r, np.asarray(col, float), alpha

    def hits(self, ro, rd):
        oc = ro - self.c
        b = np.einsum('ij,ij->i', oc, rd); c = np.einsum('ij,ij->i', oc, oc) - self.r ** 2
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0))
        out = []
        for sgn in (-1, 1):
            t = -b + sgn * sq
            ok = (disc > 0) & (t > 1e-6)
            n = norm(oc + rd * t[:, None])
            out.append((np.where(ok, t, INF), n, np.repeat(self.col[None], len(t), 0), self.alpha))
            if self.alpha >= 1: break
        return out


class Cone(Prim):
    """A solid cone with base centre `base`, radius r, apex `tip`."""
    def __init__(self, base, tip, r, col, alpha=1.0):
        self.base, self.tip, self.r, self.col, self.alpha = np.asarray(base, float), np.asarray(tip, float), r, np.asarray(col, float), alpha

    def hits(self, ro, rd):
        ax = self.base - self.tip; H = np.linalg.norm(ax); d = ax / H
        k = (self.r / H) ** 2
        co = ro - self.tip
        dv, cv = rd @ d, co @ d
        A = 1 - (1 + k) * dv * dv
        B = 2 * (np.einsum('ij,ij->i', rd, co) - (1 + k) * dv * cv)
        C = np.einsum('ij,ij->i', co, co) - (1 + k) * cv * cv
        disc = B * B - 4 * A * C
        sq = np.sqrt(np.maximum(disc, 0))
        out = []
        for sgn in (-1, 1):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (-B + sgn * sq) / (2 * A)
            h = cv + t * dv
            ok = (disc > 0) & (t > 1e-6) & (h >= 0) & (h <= H)
            q = co + rd * t[:, None]
            n = norm(q - ((1 + k) * h)[:, None] * d + 1e-12)
            out.append((np.where(ok, t, INF), n, np.repeat(self.col[None], len(t), 0), self.alpha))
        den = rd @ d
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((self.base - ro) @ d) / den
        q = ro + rd * t[:, None] - self.base
        ok = (np.einsum('ij,ij->i', q, q) <= self.r ** 2) & (t > 1e-6)
        out.append((np.where(ok, t, INF), np.repeat(d[None], len(t), 0), np.repeat(self.col[None], len(t), 0), self.alpha))
        return out


def arrow(a, b, r, col, head=None, alpha=1.0):
    """Rod + cone from a to b; the cone is `head` long (default 6 r)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a); d = (b - a) / L
    hl = head or 6 * r
    m = b - d * hl
    return [Rod(a, m, r, col, alpha), Cone(m, b, 2.4 * r, col, alpha)]


def dashed(a, b, r, col, n=9, alpha=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = []
    for i in range(n):
        t0, t1 = i / n, (i + 0.55) / n
        out.append(Rod(a + (b - a) * t0, a + (b - a) * t1, r, col, alpha))
    return out


class Camera:
    def __init__(self, pos, target, up=(0, 0, 1), fov=40, W=1200, H=800):
        self.pos = np.asarray(pos, float); self.W, self.H = W, H
        self.f = norm(np.asarray(target, float) - self.pos)
        self.r = norm(np.cross(self.f, up)); self.u = np.cross(self.r, self.f)
        self.t = np.tan(np.radians(fov) / 2)

    def project(self, p):
        """3D point -> pixel (x, y) in the W x H image."""
        d = np.asarray(p, float) - self.pos
        z = d @ self.f
        x = (d @ self.r) / (z * self.t); y = (d @ self.u) / (z * self.t)
        return (self.W / 2 + x * self.W / 2, self.H / 2 - y * self.W / 2)

    def rays(self, px, py):
        x = (px / self.W * 2 - 1) * self.t
        y = -(py / self.H * 2 - 1) * self.t * self.H / self.W
        d = self.f[None] + x[:, None] * self.r[None] + y[:, None] * self.u[None]
        return np.repeat(self.pos[None], len(px), 0), norm(d)


def render(prims, cam, ss=3, bg_top=rgb('#f4f5f7'), bg_bottom=rgb('#e4e7ec'), fog=0.0, path=None):
    W, H = cam.W, cam.H
    img = np.zeros((H, W, 3))
    offs = (np.arange(ss) + 0.5) / ss
    rows = 40
    for y0 in range(0, H, rows):
        ys = np.arange(y0, min(H, y0 + rows))
        py, px = np.meshgrid(ys, np.arange(W), indexing='ij')
        acc = np.zeros((len(ys) * W, 3))
        for oy in offs:
            for ox in offs:
                ro, rd = cam.rays(px.ravel() + ox, py.ravel() + oy)
                hits = []
                for pr in prims:
                    hits += pr.hits(ro, rd)
                T = np.stack([h[0] for h in hits], 1)                     # (N, K)
                order = np.argsort(T, 1)
                col = np.zeros((len(ro), 3)); a_acc = np.zeros(len(ro))
                for j in range(T.shape[1]):
                    idx = order[:, j]
                    tj = T[np.arange(len(ro)), idx]
                    valid = tj < INF / 2
                    if not valid.any():
                        break
                    for k in np.unique(idx[valid]):
                        sel = valid & (idx == k)
                        t, n, c, a = hits[k]
                        cc = shade(n[sel], rd[sel], c[sel])
                        if fog:
                            ff = np.exp(-fog * t[sel])[:, None]
                            cc = cc * ff + (1 - ff) * bg_bottom
                        w = (1 - a_acc[sel]) * a
                        col[sel] += w[:, None] * cc
                        a_acc[sel] += w
                frac = (py.ravel() + oy) / H
                bg = bg_top[None] * (1 - frac[:, None]) + bg_bottom[None] * frac[:, None]
                acc += col + (1 - a_acc)[:, None] * bg
        img[ys] = (acc / ss ** 2).reshape(len(ys), W, 3)
    im = Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.0) * 255 + 0.5).astype(np.uint8))
    if path:
        im.save(path)
    return im


def svg_figure(png_path, W, H, labels, aria, out_path, scale=0.5):
    """Inline SVG: the render plus text labels.  labels: list of dicts with x, y,
    text, and optional anchor ('start'|'middle'|'end'), color, size, weight, and
    leader=(x2, y2) for a line from the text to a point."""
    w, h = W * scale, H * scale
    parts = [f'<svg class="diagram fig3d" viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{aria}">',
             f'<image href="{{{{img {png_path}}}}}" x="0" y="0" width="{w:.0f}" height="{h:.0f}"/>']
    for L in labels:
        x, y = L['x'] * scale, L['y'] * scale
        col = L.get('color', '#1f2328')
        if 'leader' in L:
            x2, y2 = L['leader'][0] * scale, L['leader'][1] * scale
            lx, ly = L.get('lx', L['x']) * scale, L.get('ly', L['y']) * scale
            parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="1.4"/>')
            parts.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="2.6" fill="{col}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{L.get("size", 14)}" font-weight="{L.get("weight", 600)}" fill="{col}" '
                     f'text-anchor="{L.get("anchor", "start")}" paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round">{L["text"]}</text>')
    parts.append('</svg>')
    open(out_path, 'w').write('\n'.join(parts) + '\n')
