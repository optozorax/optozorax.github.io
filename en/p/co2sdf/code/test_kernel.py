import numpy as np
from co2sdf import *
rng = np.random.default_rng(0)

# 1. Clifford torus: projection accuracy and geodesics are straight in (theta,phi)
a, b = 1.0, 0.7
F = lambda p: clifford_torus(p, a, b)
p = rng.normal(size=(1000, 4)); p = project(F, p, 6)
print("clifford |F| after projection:", np.abs(F(p)).max())
# geodesic from (theta,phi)=(0,0) with direction angle 30deg in arclength coords
p0 = np.array([[a, 0, b, 0.0]])
d = np.array([[0, np.cos(np.pi/6), 0, np.sin(np.pi/6)]])
P, V = p0.copy(), d.copy()
dt = 1e-3
for i in range(3000):
    P, V = geodesic_step(F, P, V, dt)
th = np.arctan2(P[0,1], P[0,0]); ph = np.arctan2(P[0,3], P[0,2])
print("clifford geodesic end: s_theta=a*theta=%.5f expect %.5f ; b*phi=%.5f expect %.5f" % (a*th, 3*np.cos(np.pi/6), b*ph, 3*np.sin(np.pi/6)))

# 2. neck union of two orthogonal planes: smoothness (rank-2 Jacobian on surface)
FA = lambda p: v_of(p)          # world A: the u-plane (x1,x2)
FB = lambda p: u_of(p)          # world B: the v-plane (x3,x4)
eps, k = 0.25, 2.0
N = lambda p: neck_union(FA(p), FB(p), eps, k)
p = rng.normal(size=(20000, 4)) * 1.5
p = project(N, p, 12)
res = np.abs(N(p)); ok = res < 1e-9
aa, bb = jac(N, p[ok])
G = np.stack([np.stack([dot(aa,aa), dot(aa,bb)],-1), np.stack([dot(aa,bb), dot(bb,bb)],-1)],-2)
sv = np.linalg.svd(G, compute_uv=False)
print("neck: converged %d/%d, min sqrt(smallest Gram eig)=%.4f, max=%.4f" % (ok.sum(), len(p), np.sqrt(sv[:,1].min()), np.sqrt(sv[:,0].max())))
u, v = u_of(p[ok]), v_of(p[ok])
print("neck: min max(|u|,|v|) on surface = %.4f (waist radius sqrt(eps)=%.4f)" % (np.maximum(abs(u),abs(v)).min(), np.sqrt(eps)))
# points far out: exactly on a plane?
far = np.maximum(abs(u), abs(v)) > k
print("neck: far points max min(|u|,|v|) =", np.minimum(abs(u),abs(v))[far].max())

# 3. cone angle check: circumference of a big loop around the tip, measured on the surface
for (c, m) in [(0.845, 2), (3.0, 2), (1.0, 0)]:
    Fc = lambda p: cone(p, c, m, 0.3)
    R = 6.0
    th = np.linspace(0, 2*np.pi, 20001)
    u = R*np.exp(1j*th)
    pts = np.stack([u.real, u.imag, np.zeros_like(th), np.zeros_like(th)], -1)
    # the cone is a graph over u: put v = G(u) exactly by projecting along v only
    vv = -(Fc(pts) - v_of(pts))  # G(u)
    pts[:,2], pts[:,3] = vv.real, vv.imag
    L = np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum()
    # radial distance from tip along the surface (u from 0 to R along real axis)
    r = np.linspace(0, R, 20001)
    q = np.stack([r, 0*r, 0*r, 0*r], -1); g = -(Fc(q) - v_of(q)); q[:,2], q[:,3] = g.real, g.imag
    rho = np.linalg.norm(np.diff(q, axis=0), axis=-1).sum()
    print("cone c=%.3f m=%d: measured angle/2pi (far) ~ %.4f, formula %.4f" % (c, m, L/rho/(2*np.pi), cone_angle(c, m)/(2*np.pi)))
