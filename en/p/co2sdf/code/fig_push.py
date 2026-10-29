from fig3d import *
INK, BLUE, PURPLE, GREY = rgb('#26292e'), rgb('#2f6fd6'), rgb('#8a55c9'), rgb('#7d838c')
W, H = 1200, 700
cam = Camera(pos=(4.8, -10.8, 3.8), target=(0.4, 0.0, 0.2), up=(0, 0, 1), fov=42, W=W, H=H)
line = Rod((-4.2, 0, 0), (4.2, 0, 0), 0.06, INK)
p = np.array([0.6, -1.25, 1.55]); foot = np.array([0.6, 0, 0])
disc = Patch(foot, (0, 1.75, 0), (0, 0, 1.75), PURPLE, alpha=0.30, shape='disc', edge=0.012, line=0.45)
prims = [line, disc, Ball(p, 0.11, BLUE), Ball(foot, 0.09, INK)]
prims += arrow(p, foot + (p - foot) * 0.06, 0.045, BLUE)
others = [np.array([-2.6, 0, 0]), np.array([3.4, 0, 0])]
for q in others:
    d = (q - p) / np.linalg.norm(q - p)
    prims += dashed(p, q - d * 0.32, 0.02, GREY, n=10)
    prims += [Cone(q - d * 0.32, q - d * 0.05, 0.065, GREY), Ball(q, 0.07, GREY)]
u = (p - foot) / np.linalg.norm(p - foot); e = np.array([1.0, 0, 0]); s = 0.36     # right-angle mark at the foot
prims += [Rod(foot + u * s, foot + u * s + e * s, 0.02, INK), Rod(foot + e * s, foot + u * s + e * s, 0.02, INK)]
render(prims, cam, ss=3, path='push.png')
P = cam.project
F = P(foot)
lab = [
    dict(x=P(p)[0] + 20, y=P(p)[1] - 14, text='p, just off the world', color='#2f6fd6'),
    dict(x=F[0] + 64, y=F[1] + 96, text='shortest move: straight across,', color='#2f6fd6', leader=P(foot + e * s * 0.5 + u * 0.02), lx=F[0] + 60, ly=F[1] + 74),
    dict(x=F[0] + 64, y=F[1] + 124, text='at a right angle to the world', color='#2f6fd6'),
    dict(x=P(others[0])[0] + 10, y=P(others[0])[1] + 62, text='longer moves also reach the world,', color='#4e535b', anchor='middle'),
    dict(x=P(others[0])[0] + 10, y=P(others[0])[1] + 90, text='but slide along it', color='#4e535b', anchor='middle'),
    dict(x=P((0.6, 0, 1.75))[0] + 24, y=P((0.6, 0, 1.75))[1] - 6, text='directions that leave the world', color='#6b3fa8'),
    dict(x=P((-3.6, 0, 0))[0], y=P((-3.6, 0, 0))[1] - 26, text='the world', color='#1f2328', anchor='middle'),
]
svg_figure('figs/push.png', W, H, lab, 'The world is a black line; a blue point p is just off it. A bold blue arrow goes straight across to the nearest point of the world, at a right angle, inside a purple disc of the directions that leave the world. Grey dashed arrows go to other points of the world: they are longer and slide along it.', 'push.svg.html')
print('ok')
