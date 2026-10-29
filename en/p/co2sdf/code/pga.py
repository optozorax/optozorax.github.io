"""Tiny dense geometric algebra R(n,0,1): basis e0 (e0^2=0), e1..en (ei^2=1).
Blades are bitmasks: bit 0 = e0, bit i = ei."""
import numpy as np
from functools import lru_cache

class GA:
    def __init__(self, n):
        self.n = n; self.N = n + 1; self.size = 1 << self.N
        self.metric = [0.0] + [1.0] * n
        self.grade = np.array([bin(i).count("1") for i in range(self.size)])
        T = np.zeros((self.size, self.size)); I = np.zeros((self.size, self.size), dtype=int)
        for a in range(self.size):
            for b in range(self.size):
                s = self._sign(a, b)
                m = 1.0
                for i in range(self.N):
                    if (a & b) >> i & 1: m *= self.metric[i]
                T[a, b] = s * m; I[a, b] = a ^ b
        self.T, self.I = T, I
    def _sign(self, a, b):
        a >>= 1; s = 0
        while a:
            s += bin(a & b).count("1"); a >>= 1
        return -1.0 if s & 1 else 1.0
    def vec(self, d, a):  # hyperplane a.x + d = 0  ->  d e0 + sum a_i e_i
        x = np.zeros(self.size); x[1] = d
        for i, ai in enumerate(a): x[1 << (i + 1)] = ai
        return x
    def gp(self, x, y, keep=None):
        out = np.zeros(self.size)
        nz_x, nz_y = np.nonzero(x)[0], np.nonzero(y)[0]
        for a in nz_x:
            for b in nz_y:
                c = self.I[a, b]
                if keep is not None and not keep(self.grade[a], self.grade[b], self.grade[c]): continue
                out[c] += x[a] * y[b] * self.T[a, b]
        return out
    def wedge(self, x, y): return self.gp(x, y, lambda ga, gb, gc: gc == ga + gb)
    def inner(self, x, y): return self.gp(x, y, lambda ga, gb, gc: gc == abs(ga - gb))
    def lc(self, x, y): return self.gp(x, y, lambda ga, gb, gc: gc == gb - ga)  # left contraction x _| y
    def rev(self, x):
        g = self.grade; return x * np.where((g * (g - 1) // 2) % 2 == 1, -1.0, 1.0)
    def gsel(self, x, k): return np.where(self.grade == k, x, 0.0)
    def dual(self, x):  # Hodge-like dual via complement bitmask (Poincare duality, metric-free)
        out = np.zeros(self.size); full = self.size - 1
        for a in np.nonzero(x)[0]:
            b = full ^ a
            out[b] += x[a] * self._sign(a, b)
        return out
    def undual(self, x):
        out = np.zeros(self.size); full = self.size - 1
        for b in np.nonzero(x)[0]:
            a = full ^ b
            out[a] += x[b] / self._sign(a, b)
        return out
    def vee(self, x, y): return self.undual(self.wedge(self.dual(x), self.dual(y)))
    def point(self, c):  # meet of the n hyperplanes x_i - c_i = 0
        P = None
        for i, ci in enumerate(c):
            H = self.vec(-ci, np.eye(self.n)[i])
            P = H if P is None else self.wedge(P, H)
        return P
    def point_coords(self, P):
        # read back a point by meeting with... simpler: solve linear map numerically
        cs = []
        base = self.point(np.zeros(self.n))
        w = P[np.argmax(np.abs(base))] / base[np.argmax(np.abs(base))]
        for i in range(self.n):
            e = np.zeros(self.n); e[i] = 1.0
            Pi = self.point(e) - base
            k = np.argmax(np.abs(Pi)); cs.append(P[k] / Pi[k])
        return np.array(cs) / w if w != 0 else None
