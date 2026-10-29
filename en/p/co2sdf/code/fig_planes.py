import sys, json
from fig3d import *
GREEN, ORANGE, INK, BLUE, PURPLE = rgb('#3f9e3a'), rgb('#e8862a'), rgb('#26292e'), rgb('#2f6fd6'), rgb('#8a55c9')
W, H = 1200, 760
cam = Camera(pos=(6.2, -11.8, 6.4), target=(0.2, 0.4, -0.35), up=(0, 0, 1), fov=40, W=W, H=H)
floor = Patch((0, 0, 0), (3.8, 0, 0), (0, 3.0, 0), GREEN * 1.05, alpha=0.72, grid=1.0, line=0.72)
wall = Patch((0, 0, 0), (3.8, 0, 0), (0, 0, 2.3), ORANGE, alpha=0.42, grid=1.0, line=0.72)
line = Rod((-3.8, 0, 0), (3.8, 0, 0), 0.06, INK)
render([floor, wall, line], cam, ss=3, path='planes.png')
P = cam.project
lab = [
    dict(x=P((-2.6, -3.0, 0))[0] + 20, y=P((-2.6, -3.0, 0))[1] + 40, text='floor: the first number f = 0', color='#2d7a2a'),
    dict(x=P((3.8, 0, 2.3))[0] - 10, y=P((3.8, 0, 2.3))[1] - 16, text='wall: the second number g = 0', color='#a4561a', anchor='end'),
    dict(x=1170, y=P((3.8, 0, 0))[1] + 78, text='the world: both are 0', color='#1f2328', anchor='end', leader=P((3.3, 0, 0)), lx=1060, ly=P((3.8, 0, 0))[1] + 58),
]
svg_figure('figs/planes.png', W, H, lab, 'A green floor and a see-through orange wall crossing it. Each is where one number is zero; the black line where the wall meets the floor is where both are zero: the world.', 'planes.svg.html')
print('ok')
