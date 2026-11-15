"""Small sufficient-statistic objects; no raw image/feature persistence."""
import numpy as np
import scipy.linalg as la

LAYERS = np.array([0, 3, 7, 10, 13, 16, 20, 23])


def moments(z, y):
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if z.ndim != 2 or y.shape != (len(z),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary-labelled projected rows required')
    return {'g': z.T @ z, 'h': z.T @ y, 'sz': z.sum(0),
            'sy': np.asarray(y.sum()), 'n': np.asarray(len(y), dtype=np.int64)}


def add(a, b):
    if a['g'].shape != b['g'].shape:
        raise ValueError('Cannot combine statistics under different feature dimensions')
    return {k: a[k] + b[k] for k in ['g', 'h', 'sz', 'sy', 'n']}


def solve(state, rank=128, eta=.15):
    g = state['g']
    n, sy = int(state['n']), float(state['sy'])
    if n <= 0 or not 0 < sy < n or g.shape[0] % rank:
        raise ValueError('Invalid sample counts or projected block width')
    q = state['h'] - sy / n * state['sz']
    energy = np.diag(g).reshape(-1, rank).sum(1) / rank
    penalty = eta * np.repeat(energy, rank)
    if not np.isfinite(penalty).all() or not (penalty > 0).all():
        raise ValueError('Nonpositive block trace penalty')
    system = g.copy()
    system.flat[::len(system) + 1] += penalty
    w = la.cho_solve(la.cho_factor(system, lower=True, check_finite=False), q, check_finite=False)
    residual = float(la.norm(g @ w + penalty * w - q) / max(la.norm(q), 1e-30))
    if residual >= 1e-8 or not np.isfinite(w).all():
        raise ValueError('Analytic ridge normal-equation check failed')
    return w, {'relative_normal_equation_residual': residual, 'n': n,
               'n_fake': int(sy), 'label_mean': sy / n, 'eta': eta,
               'block_penalties': (eta * energy).tolist()}


def project(x, mapping):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != mapping['mean'].shape:
        raise ValueError('Raw readouts differ from declared mapping')
    normalized = (x - mapping['mean']) / mapping['scale']
    return np.einsum('nld,ldr->nlr', normalized, mapping['basis'], optimize=True).reshape(len(x), -1)


def fit_mapping(x, y, rank=128):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 3 or not (y == 0).any() or len(x) < rank:
        raise ValueError('Insufficient source rows or genuine references')
    real = x[y == 0]
    mu = real.mean(0)
    scale = np.sqrt(real.var(0) + 1e-6)
    bases, errors = [], []
    for layer in range(x.shape[1]):
        z = (x[:, layer] - mu[layer]) / scale[layer]
        g = z.T @ z
        ev, basis = la.eigh(g, subset_by_index=(g.shape[0] - rank, g.shape[0] - 1), check_finite=False)
        ev, basis = ev[::-1], basis[:, ::-1].copy()
        signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)])
        signs[signs == 0] = 1
        basis *= signs
        error = float(la.norm(g @ basis - basis * ev) / max(la.norm(g), 1e-30))
        if error >= 1e-8:
            raise ValueError('Spectral mapping residual failed')
        bases.append(basis)
        errors.append(error)
    return {'mean': mu, 'scale': scale, 'basis': np.stack(bases)}, {'spectral_residuals': errors}


def fold(mapping, weights):
    rank = mapping['basis'].shape[-1]
    w = np.asarray(weights).reshape(mapping['mean'].shape[0], rank)
    beta = np.einsum('ldr,lr->ld', mapping['basis'], w) / mapping['scale']
    bias = -float(np.sum(beta * mapping['mean']))
    return beta, bias
