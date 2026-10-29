"""Regular chartwise field for a punctured RP^3 in R^5.  Only numpy is needed.

Coordinates are (Re u, Im u, h, Re v, Im v).  Select a chart once at each
Newton evaluation, and keep that selection fixed for numerical gradients.
"""
from dataclasses import dataclass
import numpy as np


def cutoff(r, start, end):
    """C-infinity: one for r <= start, zero for r >= end."""
    s = (np.asarray(r, dtype=float) - start) / (end - start)
    out = np.ones_like(s)
    out[s >= 1] = 0
    mask = (s > 0) & (s < 1)
    q = s[mask]
    # chi = sigmoid(1/q - 1/(1-q)); stable even very near an endpoint.
    log_ratio = 1 / q - 1 / (1 - q)
    out[mask] = np.exp(-np.logaddexp(0, -log_ratio))
    return out


@dataclass
class Wormhole:
    a: float = 1.0
    b: float = 1.0
    depth: float = 6.0
    width: float = 1.0

    def __post_init__(self):
        assert self.a > 0 and self.b > 0 and self.width > 0
        assert self.depth > self.b

    @property
    def mouth_radius(self):
        return 2 * self.a + self.width

    def twist(self, r):
        return self.a * cutoff(r, self.a, 2 * self.a)

    def height(self, r):
        return cutoff(r, 2 * self.a, self.mouth_radius)

    def charts(self, points):
        """0=flat/far, 1=outer graph, 2=middle, 3=pole, 4=empty."""
        p = np.atleast_2d(points)
        u = p[:, 0] + 1j * p[:, 1]
        h = p[:, 2]
        v = p[:, 3] + 1j * p[:, 4]
        r = np.linalg.norm(p[:, :3], axis=1)
        eta = (v - self.depth) / self.b
        far = (r >= self.mouth_radius) | (
            np.abs(v) > self.depth + self.b + self.a)
        outer = (~far) & (r >= self.a)
        pole = (~far) & (~outer) & (np.abs(eta) < 0.25) & (
            np.abs(u)**2 < h*h/2 + 2*self.a*self.a/9)
        # The middle formula is also valid below its usual threshold.  Use
        # it when an off-world point leaves the pole graph's domain.
        middle = (~far) & (~outer) & (~pole) & (np.abs(eta) > 0)
        result = np.full(len(p), 4, dtype=int)
        result[far] = 0
        result[outer] = 1
        result[middle] = 2
        result[pole] = 3
        return result

    def outer_root(self, r2, h, A):
        """T + A^2 h^2/T - A^2 h^4/T^2 = r^2, physical root.

        For r >= a and 0 <= A <= a, derivative >= 5/6 on the
        physical bracket.  Eight clipped Newton iterations suffice here.
        """
        r2, h, A = np.broadcast_arrays(r2, h, A)
        lo = np.maximum(h*h, r2 - A*A/4)
        T = r2.copy()
        if np.all(A == 0):
            return T
        for _ in range(8):
            q = h*h/T
            f = T - r2 + A*A*q*(1-q)
            df = 1 + (A*A/T)*(2*q*q-q)
            T = np.clip(T-f/df, lo, r2)
        return T

    def pole_root(self, U, h):
        """Unique q in (2/3,1] with U=(h^2/q+a^2*q)*(1-q)."""
        U, h = np.broadcast_arrays(U, h)
        q = np.ones_like(U)
        for _ in range(12):
            f = U - (h*h/q + self.a*self.a*q)*(1-q)
            df = h*h/(q*q) + self.a*self.a*(2*q-1)
            q = np.clip(q-f/df, 2/3, 1)
        return q

    def field(self, points, frozen=None):
        single = np.asarray(points).ndim == 1
        p = np.atleast_2d(np.asarray(points, dtype=float))
        chart = self.charts(p) if frozen is None else np.broadcast_to(frozen, (len(p),))
        result = np.zeros((len(p), 2))
        for code in range(5):
            mask = chart == code
            if not np.any(mask):
                continue
            part = p[mask]
            u = part[:, 0] + 1j * part[:, 1]
            h = part[:, 2]
            v = part[:, 3] + 1j * part[:, 4]
            if code == 0:
                value = v
            elif code == 1:
                r2 = np.sum(part[:, :3]**2, axis=1)
                r = np.sqrt(r2)
                A = self.twist(r)
                T = self.outer_root(r2, h, A)
                w2 = T*u*u/(T - 1j*A*h)**2
                value = v - self.height(r)*(self.depth + self.b*w2)
            elif code == 2:
                eta = (v - self.depth)/self.b
                s = np.abs(eta)
                Q = u*u/eta
                result[mask, 0] = h + Q.imag/(2*self.a)
                result[mask, 1] = self.a*(s-1) + (
                    np.abs(u)**2/s - Q.real)/(2*self.a)
                continue
            elif code == 3:
                q = self.pole_root(np.abs(u)**2, h)
                w2 = q*u*u/(h - 1j*self.a*q)**2
                value = v - self.depth - self.b*w2
            else:
                value = np.full(len(part), self.a, dtype=complex)
            result[mask] = np.column_stack((value.real, value.imag))
        return result[0] if single else result

    def jacobian(self, points, step=1e-5, frozen=None, order=4):
        """Frozen-chart finite difference Jacobian, order 2 or 4."""
        single = np.asarray(points).ndim == 1
        p = np.atleast_2d(np.asarray(points, dtype=float))
        chart = self.charts(p) if frozen is None else frozen
        result = np.empty((len(p), 2, 5))
        for j in range(5):
            offset = np.zeros(5)
            offset[j] = step
            plus = self.field(p + offset, chart)
            minus = self.field(p - offset, chart)
            if order == 2:
                result[:, :, j] = (plus-minus)/(2*step)
            else:
                plus2 = self.field(p + 2*offset, chart)
                minus2 = self.field(p - 2*offset, chart)
                result[:, :, j] = (8*(plus-minus)-(plus2-minus2))/(12*step)
        return result[0] if single else result

    @staticmethod
    def pack(u, h, v):
        return np.stack((u.real, u.imag, h, v.real, v.imag), axis=-1)

    def core(self, t, d):
        d = np.asarray(d)
        w = d[..., 0] + 1j*d[..., 1]
        z = d[..., 2]
        return self.pack((t-1j*self.a*z)*w, t*z,
                         self.depth+self.b*w*w)

    def core_limit(self, d):
        q = np.asarray(d)[..., 2]**2
        return self.a*np.sqrt(1-q*(1-q))

    def collar(self, r, d):
        d = np.asarray(d)
        r = np.asarray(r)
        w = d[..., 0] + 1j*d[..., 1]
        z = d[..., 2]
        A = self.twist(r)
        t = np.sqrt(r*r-A*A*z*z*(1-z*z))
        return self.pack((t-1j*A*z)*w, t*z,
                         self.height(r)*(self.depth+self.b*w*w))

    def project(self, points, tolerance=1e-12, max_steps=16, step=1e-5):
        p = np.atleast_2d(np.asarray(points, dtype=float)).copy()
        count = np.zeros(len(p), dtype=int)
        initial_chart = self.charts(p)
        changes = np.zeros(len(p), dtype=bool)
        valid = initial_chart != 4
        for _ in range(max_steps):
            chart = self.charts(p)
            changes |= chart != initial_chart
            f = self.field(p, chart)
            active = valid & (np.linalg.norm(f, axis=1) > tolerance)
            if not np.any(active):
                break
            J = self.jacobian(p[active], step=step, frozen=chart[active], order=2)
            gram = J @ np.swapaxes(J, 1, 2)
            try:
                multipliers = np.linalg.solve(gram, f[active, :, None])[..., 0]
            except np.linalg.LinAlgError:
                raise RuntimeError("Newton reached a singular off-world chart")
            p[active] -= np.einsum('nki,nk->ni', J, multipliers)
            count[active] += 1
            valid &= self.charts(p) != 4
        residual = np.linalg.norm(self.field(p), axis=1)
        return p, count, residual, changes, valid
