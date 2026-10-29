from fig3d import *
GREEN, INK, TEAL, GOLD, BLUE = rgb('#4caa47'), rgb('#26292e'), rgb('#2a9d9d'), rgb('#e0a526'), rgb('#3a6fd0')
R = 1.5
STATIONS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]          # along / PI at the seven stations
def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0); return t * t * (3 - 2 * t)
def turn_angle(along):                                    # the page's handle(): PI * smoothstep(PI/4, 3PI/4, along)
    return np.pi * smoothstep(0.25 * np.pi, 0.75 * np.pi, along)
def centre(along):                                        # the middle of the tube: the circle |xz| = R, upper half
    return np.array([R * np.cos(along), 0.0, R * np.sin(along)])

W, H = 1200, 560
cam = Camera(pos=(2.6, -7.8, 4.6), target=(0.0, 0.2, 0.75), up=(0, 0, 1), fov=36, W=W, H=H)
plane = Patch((0, 0, 0), (3.4, 0, 0), (0, 2.3, 0), GREEN, alpha=0.85, grid=0.6, line=0.8)
prims = [plane]
for x in (R, -R):                                         # the two mouths: holes with a dark rim
    prims.append(Patch((x, 0, 0.005), (0.5, 0, 0), (0, 0.5, 0), INK * 1.6, alpha=0.4, shape='disc', edge=0.0))
    for i in range(40):
        a0, a1 = 2 * np.pi * i / 40, 2 * np.pi * (i + 1) / 40
        prims.append(Rod((x + 0.5 * np.cos(a0), 0.5 * np.sin(a0), 0.01), (x + 0.5 * np.cos(a1), 0.5 * np.sin(a1), 0.01), 0.022, INK))
N = 48
for i in range(N):                                        # the path of the tube
    a0, a1 = np.pi * i / N, np.pi * (i + 1) / N
    prims.append(Rod(centre(a0), centre(a1), 0.045, TEAL))
for k, s in enumerate(STATIONS):
    prims.append(Ball(centre(s * np.pi), 0.12, rgb('#f4f4f4')))
render(prims, cam, ss=3, path='mirror3d.png')
P = cam.project

# ---- SVG: the render, numbered stations, then the filmstrip of cross-sections
sc = 0.5
w, h3 = W * sc, H * sc
rowh, top = 118, h3 + 16
hh = top + 2 * rowh + 50
out = [f'<svg class="diagram fig3d" viewBox="0 0 {w:.0f} {hh:.0f}" role="img" aria-label="A plane with a handle: a tube from one mouth over the plane to the other. Seven numbered stations along it. Below, the tube\'s circle at each station with an L drawn on it. For the ordinary handle the circle and its L stay the same. For the mirror handle the circle turns through the fourth direction w: in 3D it looks squashed, at station 4 it is only a line, and afterwards it shows its other face with the L mirrored.">',
       f'<rect x="0" y="0" width="{w:.0f}" height="{hh:.0f}" rx="10" fill="#f0f2f5"/>',
       f'<image href="{{{{img figs/mirror3d.png}}}}" x="0" y="0" width="{w:.0f}" height="{h3:.0f}"/>']
def text(x, y, s, col='#1f2328', size=13, anchor='middle', weight=600):
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{col}" text-anchor="{anchor}" paint-order="stroke" stroke="{"#1f2328" if col == "#ffffff" else "#ffffff"}" stroke-width="3.5" stroke-linejoin="round">{s}</text>')
for k, s in enumerate(STATIONS):
    x, y = P(centre(s * np.pi) + np.array([0, 0, 0.32]))
    text(x * sc, y * sc, str(k + 1), size=14)
x1, y1 = P((R + 0.55, 0, 0)); x2, y2 = P((-R - 0.55, 0, 0))
text(x1 * sc + 6, y1 * sc + 4, 'mouth 1', '#1f2328', 12, 'start'); text(x2 * sc - 6, y2 * sc + 4, 'mouth 2', '#1f2328', 12, 'end')
text(w - 14, 22, 'drawn: the centre line of the tube, with 7 stations', '#1e7c7c', 12, 'end', 500)
# filmstrip
x0, dx, rho = 58, (w - 116) / 6, 33
def L_path(cx, cy, sx):
    """an L: the bar along the tube's outward direction (up on the page), the foot along the
    circle's second axis, scaled by sx (negative: mirrored)"""
    pts = [(-0.45, -0.6), (-0.45, 0.6), (-0.2, 0.6), (-0.2, -0.35), (0.5, -0.35), (0.5, -0.6)]
    return ' '.join(f'{cx + px * rho * sx:.1f},{cy - py * rho:.1f}' for px, py in pts)
for row, (title, mirror) in enumerate([('ordinary handle', False), ('mirror handle', True)]):
    cy = top + rowh * row + 62
    text(12, top + rowh * row + 18, title, '#1f2328', 13, 'start', 700)
    for k, s in enumerate(STATIONS):
        cx = x0 + dx * k
        c = np.cos(turn_angle(s * np.pi)) if mirror else 1.0
        sx = c if abs(c) > 0.04 else 0.04
        face, edge = ('#f1c45a', '#a8781a') if sx > 0 else ('#8fb1ee', '#2f5aa8')
        out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rho * abs(sx):.1f}" ry="{rho}" fill="{face}" stroke="{edge}" stroke-width="1.6"/>')
        if abs(sx) > 0.1:
            out.append(f'<polygon points="{L_path(cx, cy, sx)}" fill="#1f2328"/>')
        text(cx, cy + rho + 17, str(k + 1), '#3a3f46', 12)
    if mirror:
        cx = x0 + dx * 3
        text(cx, cy - rho - 8, 'edge-on: the circle lies along w', '#a06a00', 12)
text(w / 2, hh - 26, 'Each circle: the tube seen end-on at that station. Gold: the face it started with; blue: its other face.', '#3a3f46', 11.5, 'middle', 500)
text(w / 2, hh - 10, 'In 3D the circle must stay square to the tube, so it can only spin, and spinning never mirrors an L.', '#3a3f46', 11.5, 'middle', 500)
out.append('</svg>')
open('mirror.svg.html', 'w').write('\n'.join(out) + '\n')
print('ok')
