from fig3d import *
GREEN, ORANGE, INK, BLUE = rgb('#3f9e3a'), rgb('#e8862a'), rgb('#26292e'), rgb('#3f74d4')
W, H = 1200, 560
cam = Camera(pos=(6.4, -12.0, 6.0), target=(0.0, 0.2, 0.05), up=(0, 0, 1), fov=29, W=W, H=H)
plane = Patch((0, 0, 0), (3.0, 0, 0), (0, 2.2, 0), BLUE * 1.15, alpha=0.35, grid=0.5, line=0.8)
ball = Ball((0, 0, 0.3), 1.25, GREEN * 1.1, alpha=0.42)
rr = np.sqrt(1.25 ** 2 - 0.3 ** 2)
inside = Patch((0, 0, 0.004), (rr, 0, 0), (0, rr, 0), ORANGE * 1.1, alpha=0.22, shape='disc', edge=0.0)
ring = []
N = 90
for i in range(N):
    a0, a1 = 2 * np.pi * i / N, 2 * np.pi * (i + 1) / N
    ring.append(Rod((rr * np.cos(a0), rr * np.sin(a0), 0), (rr * np.cos(a1), rr * np.sin(a1), 0), 0.045, ORANGE * 0.95))
render([plane, inside, ball] + ring, cam, ss=3, path='circle.png')
P = cam.project
lab = [
    dict(x=P((-2.9, -2.2, 0))[0] + 10, y=P((-2.9, -2.2, 0))[1] - 14, text='carrier: a plane', color='#2a55a8'),
    dict(x=P((0, 0, 1.55))[0], y=P((0, 0, 1.55))[1] - 14, text='solid: a ball', color='#2d7a2a', anchor='middle'),
    dict(x=P((1.9, -2.1, 0))[0] + 30, y=P((1.9, -2.1, 0))[1] + 34, text='world: where the ball’s surface meets the plane', color='#b55a12', anchor='middle',
         leader=P((rr * np.cos(-0.9), rr * np.sin(-0.9), 0)), lx=P((1.9, -2.1, 0))[0] + 30, ly=P((1.9, -2.1, 0))[1] + 12),
]
svg_figure('figs/circle.png', W, H, lab, 'One dimension down: a see-through green ball crosses a blue plane. The plane inside the ball is a shaded disc, and its orange edge, where the ball’s surface meets the plane, is the world.', 'circle.svg.html')
print('ok')
