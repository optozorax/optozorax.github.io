from fig3d import *
GREEN, ORANGE, INK, BLUE, PURPLE, GREY = rgb('#3f9e3a'), rgb('#e8862a'), rgb('#26292e'), rgb('#2f6fd6'), rgb('#8a55c9'), rgb('#8a8f98')
W, H = 1200, 800
cam = Camera(pos=(10.5, -4.6, 4.2), target=(0.3, 0.3, 0.35), up=(0, 0, 1), fov=33, W=W, H=H)
o = np.array([0.4, 0, 0]); p = np.array([0.4, 1.25, 0.95])
rho = np.hypot(1.25, 0.95)
front = Rod((0.4, 0, 0), (3.8, 0, 0), 0.06, INK)                 # the part of the world in front of the disc
back = Rod((-3.4, 0, 0), (0.4, 0, 0), 0.06, rgb('#9aa0aa'))        # behind it: lighter
disc = Patch(o, (0, 2.3, 0), (0, 0, 2.3), PURPLE, alpha=0.26, shape='disc', edge=0.008, line=0.5)
prims = [front, back, disc, Ball(o, 0.08, INK), Ball(p, 0.11, BLUE)]
prims += arrow(o, o + (0, 0, 2.0), 0.04, rgb('#2d7a2a')) + arrow(o, o + (0, 2.0, 0), 0.04, rgb('#c46a1e'))
prims += arrow(o, p - (p - o) * 0.07, 0.035, BLUE)
prims += dashed(p, (0.4, 1.25, 0), 0.016, rgb('#2d7a2a'), n=7) + dashed(p, (0.4, 0, 0.95), 0.016, rgb('#c46a1e'), n=8)
N = 64                                                            # the circle through p: same distance, every angle
for i in range(0, N, 2):
    a0, a1 = 2 * np.pi * i / N, 2 * np.pi * (i + 1) / N
    prims.append(Rod(o + rho * np.array([0, np.cos(a0), np.sin(a0)]), o + rho * np.array([0, np.cos(a1), np.sin(a1)]), 0.02, rgb('#6b3fa8')))
th = np.arctan2(0.95, 1.25)                                       # the angle, from b towards p
for i in range(12):
    a0, a1 = th * i / 12, th * (i + 1) / 12
    prims.append(Rod(o + 0.55 * np.array([0, np.cos(a0), np.sin(a0)]), o + 0.55 * np.array([0, np.cos(a1), np.sin(a1)]), 0.018, BLUE))
s = 0.25                                                          # right angle between a and b
prims += [Rod(o + np.array([0, s, 0]), o + np.array([0, s, s]), 0.014, INK), Rod(o + np.array([0, 0, s]), o + np.array([0, s, s]), 0.014, INK)]
render(prims, cam, ss=3, path='meaning.png')
P = cam.project
mid = o + 0.5 * (p - o)
lab = [
    dict(x=P(p)[0] + 16, y=P(p)[1] - 10, text='p', color='#2f6fd6', size=16),
    dict(x=P(o + (0, 0, 2.05))[0] + 10, y=P(o + (0, 0, 2.05))[1], text='a: f grows', color='#2d7a2a'),
    dict(x=P(o + (0, 2.05, 0))[0] + 10, y=P(o + (0, 2.05, 0))[1] + 6, text='b: g grows', color='#a4561a'),
    dict(x=P((0.4, 0, 0.95))[0] - 12, y=P((0.4, 0, 0.95))[1] + 6, text='f', color='#2d7a2a', anchor='end', size=16),
    dict(x=P((0.4, 1.25, 0))[0], y=P((0.4, 1.25, 0))[1] + 28, text='g', color='#a4561a', anchor='middle', size=16),
    dict(x=P(mid)[0] - 14, y=P(mid)[1] - 8, text='distance', color='#2f6fd6', anchor='end'),
    dict(x=P(o + 0.75 * np.array([0, np.cos(th / 2), np.sin(th / 2)]))[0] + 6, y=P(o + 0.75 * np.array([0, np.cos(th / 2), np.sin(th / 2)]))[1] + 6, text='angle', color='#2f6fd6', size=12),
    dict(x=24, y=H - 26, text='same distance, every angle around the world', color='#6b3fa8', anchor='start', leader=P(o + rho * np.array([0, np.cos(-2.0), np.sin(-2.0)])), lx=200, ly=H - 48),
    dict(x=24, y=44, text='directions that leave the world', color='#6b3fa8', anchor='start'),
    dict(x=P((3.5, 0, 0))[0] - 6, y=P((3.5, 0, 0))[1] + 30, text='the world', color='#1f2328', anchor='middle'),
]
svg_figure('figs/meaning.png', W, H, lab, 'The world is a black line piercing a purple disc, the directions that leave it. In the disc, a green arrow a and an orange arrow b at a right angle show where the first and the second number grow. A point p in the disc has coordinates f and g along them; the blue arrow to p is the distance, its angle from b says which way around the world p lies, and a dashed circle through p shows all points at the same distance.', 'meaning.svg.html')
print('ok')
