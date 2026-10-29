"""Reproducible numerical checks; writes results.json in this directory."""
import json
from pathlib import Path
import numpy as np
from wormhole import Wormhole


def directions(rng, count):
    d = rng.normal(size=(count, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def sample_world(wh, rng):
    d = directions(rng, 100_000)
    core = wh.core(rng.uniform(-1, 1, len(d))*wh.core_limit(d), d)
    d = directions(rng, 100_000)
    collar = wh.collar(rng.uniform(wh.a, wh.mouth_radius+wh.a, len(d)), d)
    # Deliberate poles, near-poles, equator, chart interfaces, and cutoff seams.
    edge = []
    for s in [0, 1e-24, 1e-16, 1e-10, .25-1e-8, .25, .25+1e-8, .5, 1]:
        for theta in np.linspace(0, 2*np.pi, 17)[:-1]:
            d = np.array([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta),
                          np.sqrt(1-s)])
            tmax = wh.core_limit(d)
            for t in [-tmax, -.7*tmax, -1e-10, 0, 1e-10, .7*tmax, tmax]:
                edge.append(wh.core(t, d))
            for r0 in [wh.a, 2*wh.a, wh.mouth_radius]:
                for delta in [-1e-8, 0, 1e-8]:
                    r = r0 + delta*wh.a
                    if r >= wh.a:
                        edge.append(wh.collar(r, d))
    return np.vstack([core, collar, edge])


def reconstruction_error(wh, points):
    """Recover actual parametrized world points independently of the residual."""
    chart = wh.charts(points)
    reconstructed = np.empty_like(points)
    for code in range(4):
        mask = chart == code
        p = points[mask]
        if not len(p):
            continue
        u = p[:, 0] + 1j*p[:, 1]
        h = p[:, 2]
        if code == 0:
            p = p.copy()
            p[:, 3:] = 0
            reconstructed[mask] = p
            continue
        if code == 1:
            r = np.linalg.norm(p[:, :3], axis=1)
            A = wh.twist(r)
            T = wh.outer_root(r*r, h, A)
            t = np.sqrt(T)
            z = h/t
            w = u/(t-1j*A*z)
        elif code == 2:
            eta = (p[:, 3] + 1j*p[:, 4] - wh.depth)/wh.b
            w = np.sqrt(eta)
            ratio = u/w
            t = ratio.real
            z = -ratio.imag/wh.a
        else:
            q = wh.pole_root(np.abs(u)**2, h)
            z = np.sqrt(q)
            t = h/z
            w = u/(t-1j*wh.a*z)
        d = np.column_stack((w.real, w.imag, z))
        d /= np.linalg.norm(d, axis=1)[:, None]
        reconstructed[mask] = wh.collar(r, d) if code == 1 else wh.core(t, d)
    if np.any(chart == 4):
        return float('inf')
    return float(np.max(np.linalg.norm(points-reconstructed, axis=1)))


def make_loop(wh, d, count):
    """One fibre crossing, then a flat exterior semicircle back to the start."""
    d = np.asarray(d, dtype=float)
    d /= np.linalg.norm(d)
    r = np.linspace(wh.mouth_radius, wh.a, count)
    incoming = wh.collar(r, np.broadcast_to(d, (count, 3)))
    tmax = wh.core_limit(d)
    core = wh.core(np.linspace(tmax, -tmax, count),
                   np.broadcast_to(d, (count, 3)))
    outgoing = wh.collar(r[::-1], np.broadcast_to(-d, (count, 3)))
    e = np.cross(d, [1., 0., 0.])
    e /= np.linalg.norm(e)
    theta = np.linspace(0, np.pi, count)
    outside = np.zeros((count, 5))
    outside[:, :3] = wh.mouth_radius*(
        -np.cos(theta)[:, None]*d + np.sin(theta)[:, None]*e)
    outside[-1] = incoming[0]
    return np.vstack((incoming, core[1:], outgoing[1:], outside[1:]))


def transport(wh, path):
    """Projection plus polar orthonormalization approximates parallel transport."""
    J = wh.jacobian(path)
    trans = np.swapaxes(J, 1, 2)
    normals = trans @ np.linalg.solve(J @ trans, J)
    P = np.eye(5)[None, :, :] - normals
    frame = np.eye(5)[:, :3]
    minimum_overlap = 1.
    for projector in P[1:]:
        candidate = projector @ frame
        gram = candidate.T @ candidate
        eig, vec = np.linalg.eigh(gram)
        minimum_overlap = min(minimum_overlap, float(np.sqrt(eig[0])))
        frame = candidate @ ((vec/np.sqrt(eig)) @ vec.T)
    rotation = frame[:3]
    return {
        'points': len(path),
        'determinant': float(np.linalg.det(rotation)),
        'orthogonality_error': float(np.linalg.norm(rotation.T@rotation-np.eye(3))),
        'normal_leakage': float(np.linalg.norm(frame[3:])),
        'minimum_step_overlap': minimum_overlap,
        'rotation': rotation.tolist(),
        'length': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        'field_residual': float(np.max(np.linalg.norm(wh.field(path), axis=1))),
    }


def main():
    rng = np.random.default_rng(20261028)
    wh = Wormhole()
    points = sample_world(wh, rng)
    f = wh.field(points)
    chart = wh.charts(points)
    assert not np.any(chart == 4)
    J = wh.jacobian(points)
    sv = np.linalg.svd(J, compute_uv=False)
    results = {
        'parameters': vars(wh),
        'seed': 20261028,
        'world_samples': len(points),
        'world_residual_max': float(np.max(np.linalg.norm(f, axis=1))),
        'jacobian_difference_step': 1e-5,
        'singular_value_min': float(sv[:, -1].min()),
        'singular_value_max': float(sv[:, 0].max()),
        'by_chart': {},
    }
    for code in range(4):
        m = chart == code
        results['by_chart'][str(code)] = {
            'count': int(m.sum()),
            'residual_max': float(np.linalg.norm(f[m], axis=1).max()),
            'singular_min': float(sv[m, -1].min()),
            'singular_max': float(sv[m, 0].max()),
        }
    results['projection'] = []
    for sigma, count in [(1e-3, 50_000), (2e-2, 20_000)]:
        base = points[rng.integers(len(points), size=count)]
        noisy = base+rng.normal(scale=sigma, size=base.shape)
        projected, steps, residual, changes, valid = wh.project(noisy)
        success = valid & (residual <= 1e-12)
        results['projection'].append({
            'noise_standard_deviation': sigma,
            'count': count,
            'successes': int(success.sum()),
            'iterations_max': int(steps.max()),
            'iterations_median': float(np.median(steps)),
            'residual_max': float(residual.max()),
            'chart_switches': int(changes.sum()),
            'parametrization_distance_max': reconstruction_error(wh, projected),
        })
        assert np.all(success), results['projection'][-1]

    # Exact support and mouth geometry.
    d = directions(rng, 10_000)
    external = wh.collar(rng.uniform(wh.mouth_radius, 20, len(d)), d)
    mouth = wh.collar(np.full(len(d), wh.mouth_radius), d)
    opposite = wh.collar(np.full(len(d), wh.mouth_radius), -d)
    results['flat_exterior_normal_max'] = float(np.abs(external[:, 3:]).max())
    results['mouth_antipodal_error'] = float(np.abs(mouth[:, :3]+opposite[:, :3]).max())
    results['mouth_radius_error'] = float(np.abs(
        np.linalg.norm(mouth[:, :3], axis=1)-wh.mouth_radius).max())

    # Frame holonomy around two representatives of the nontrivial Z/2 loop.
    results['transport'] = {}
    for name, d in [('middle', [.48, .64, .6]),
                    ('pole', [.18, .24, np.sqrt(.91)])]:
        coarse = transport(wh, make_loop(wh, d, 1200))
        fine = transport(wh, make_loop(wh, d, 2400))
        fine['coarse_fine_rotation_difference'] = float(np.linalg.norm(
            np.array(coarse['rotation'])-np.array(fine['rotation'])))
        results['transport'][name] = fine
        assert fine['determinant'] > .99999999

    assert results['world_residual_max'] < 1e-12
    assert results['singular_value_min'] > 1-1e-7
    assert results['flat_exterior_normal_max'] == 0
    Path('results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
