"""Simulation designs of Section 6 and Appendix H.

generate() draws one dataset. Random numbers come from four independent streams per repetition
(seed_plan): the data stream (covariates, then control outcomes), the assignment stream, the stream of
the independent calibration sample used by the outcome adjustment, and the Monte Carlo stream (shared
randomization draws). The zero-effect set consumes no randomness.

Zero-effect sets (`zero`):
  'centre10'     the floor(0.1 n) units nearest the centre, one candidate region (Sections 6.2-6.5);
                 in one dimension this is the centred block of floor(0.1 n) units;
  'two_regions'  the 20% of units nearest a left centre and the 10% nearest a right centre (Appendix H.2);
  'strip20'      two covariates, first-coordinate rank at most 0.2 n (Section 6.3);
  'circle'       the disc of area 0.10 around (0.5, 0.5) in covariate space, and the centred block in one
                 dimension (the design of the timing comparison, Section 6.4);
  'none'         no zero-effect unit (dispersed and cancelling effects, Appendix H.5).
Effects outside the zero-effect set (`effect`): 'const' (delta), 'grad1' (delta times the normalized
distance to the zero-effect set), 'neg' (-delta), 'asym' (+delta below the zero block, -delta/2 above),
'dispersed5' and 'cancel5' (groups of five units in rank order). Scenarios (`scenario`, Appendix H.1):
'S0' normal errors, 'S1' t3 errors, 'S2' heteroskedastic errors, 'S3' binary outcome, 'S4' opposite-sign
effects, 'S5' stratified design.

"Nearest" uses the metric of the method (Section 4): integer coordinate ranks with ties by unit index,
squared Euclidean rank distance, the unit itself first, distance ties by the smaller first-coordinate rank.
"""
from __future__ import annotations

import hashlib

import numpy as np
from numba import njit

from .grids import paper_grid
from .randomization import sample_crd, sample_half_within_strata

BASE_CALIBRATION_SEED = 20260619
CIRCLE_AREA = 0.10
CIRCLE_RADIUS = float(np.sqrt(CIRCLE_AREA / np.pi))
# size index of each sample size of the paper (0 for other sizes); it enters only the seed of the independent
# calibration sample
SIZE_INDEX = {1000: 0, 4000: 1, 7000: 2, 10000: 3, 13000: 4, 200: 5, 400: 6, 600: 7, 800: 8,
              500: 9, 2000: 10, 8000: 11, 16000: 12, 20000: 13}
CENTRE_FRACTION = 0.10
FRACTIONS = {'left': 0.20, 'right': 0.10}
CENTRES_1D = {'left': 0.25, 'right': 0.75}
CENTRES_2D = {'left': (0.28, 0.32), 'right': (0.74, 0.70)}


def seed_plan(rep, size_index, calibration_base=BASE_CALIBRATION_SEED):
    if min(rep, size_index) < 0:
        raise ValueError('rep and size_index must be nonnegative')
    return dict(data=1 + int(rep), assignment=10001 + int(rep),
                monte_carlo=20001 + 1000 * int(rep),
                calibration=int(calibration_base) + 1000000 * int(size_index) + 600000 + int(rep))


def baseline_mean(W):
    w = np.asarray(W)
    if w.ndim == 1:
        return 0.5 + 2.0 * w
    if w.ndim == 2 and w.shape[1] == 2:
        return 0.5 + 2.0 * w[:, 0] + 2.0 * w[:, 1]
    raise ValueError('W must be a vector or an n by 2 array')


def design_matrix(W):
    w = np.asarray(W)
    if w.ndim == 1:
        w = w[:, None]
    return np.column_stack((np.ones(len(w)), w))


def predict_baseline(W, beta):
    """b0 + b1 W (+ b2 W2), evaluated term by term."""
    w, beta = np.asarray(W), np.asarray(beta)
    expected = 2 if w.ndim == 1 else 3
    if beta.shape != (expected,) or not np.isfinite(beta).all():
        raise ValueError('coefficients must be finite and match the dimension')
    with np.errstate(all='raise'):
        fitted = beta[0] + beta[1] * (w if w.ndim == 1 else w[:, 0])
        if w.ndim == 2:
            fitted = fitted + beta[2] * w[:, 1]
    return fitted


def independent_baseline(dimension, n_cal, sigma, seed):
    """Least-squares fit of the control outcome on the covariates in an independent calibration sample of
    size n_cal (it never sees the experimental outcomes or assignments)."""
    rng = np.random.default_rng(seed)
    shape = n_cal if dimension == 1 else (n_cal, 2)
    W_cal = rng.uniform(0, 1, shape)
    Y0_cal = baseline_mean(W_cal) + rng.normal(0, sigma, n_cal)
    with np.errstate(all='raise'):
        beta, _, _, _ = np.linalg.lstsq(design_matrix(W_cal), Y0_cal, rcond=None)
    return beta, W_cal, Y0_cal


# ------------------------------------------------------------------------------------- zero-effect sets
def ranks_of(W):
    """Integer coordinate ranks 1..n with ties by unit index."""
    w = np.asarray(W, float)
    w = w[:, None] if w.ndim == 1 else w
    n, d = w.shape
    ids = np.arange(n)
    ranks = np.empty((n, d), dtype=np.int64)
    for j in range(d):
        ix = np.lexsort((ids, w[:, j]))
        ranks[ix, j] = np.arange(1, n + 1)
    return ranks


def nearest(ranks, centre, k):
    """The k units nearest to `centre` (the candidate region of size k around it)."""
    delta = ranks - ranks[centre]
    distance = np.sum(delta * delta, axis=1, dtype=np.int64)
    distance[centre] = -1
    return np.lexsort((ranks[:, 0], distance))[:k]


def centre_region(W, dimension):
    """The floor(0.1 n) units nearest to the centre: one candidate region."""
    ranks = ranks_of(W)
    n = len(ranks)
    k = int(np.floor(CENTRE_FRACTION * n))
    if dimension == 1:
        start = (n - k) // 2
        centre = int(np.flatnonzero(ranks[:, 0] == start + k // 2 + 1)[0])
    else:
        gap = np.sum((ranks - np.asarray((.5, .5)) * n) ** 2, axis=1)
        centre = int(np.lexsort((ranks[:, 0], gap))[0])
    members = nearest(ranks, centre, k)
    zero = np.zeros(n, bool)
    zero[members] = True
    if dimension == 1:
        block = np.zeros(n, bool)
        block[np.argsort(ranks[:, 0])[start:start + k]] = True
        if not np.array_equal(zero, block):
            raise ValueError('1D centre region differs from the centred block')
    return zero, {'centre': dict(centre=centre, fraction=CENTRE_FRACTION, members=np.sort(members))}


def two_regions(W, dimension):
    """20% of units nearest a left centre and 10% nearest a right centre: two separate candidate regions."""
    ranks = ranks_of(W)
    n = len(ranks)
    out = {}
    for side, frac in FRACTIONS.items():
        if dimension == 1:
            target = np.array([round(CENTRES_1D[side] * n)], float)
        else:
            target = np.asarray(CENTRES_2D[side], float) * n
        gap = np.sum((ranks - target) ** 2, axis=1)
        centre = int(np.lexsort((ranks[:, 0], gap))[0])
        members = nearest(ranks, centre, int(np.floor(frac * n)))
        out[side] = dict(centre=centre, fraction=frac, members=np.sort(members))
    if np.intersect1d(out['left']['members'], out['right']['members']).size:
        raise ValueError('the two zero-effect regions overlap')
    zero = np.zeros(n, bool)
    for r in out.values():
        zero[r['members']] = True
    return zero, out


def zero_set(W, dimension, zero):
    """Zero-effect indicator, coordinate ranks and the named regions (None if not a union of regions)."""
    n = len(W)
    ranks = ranks_of(W)
    regions = None
    if zero == 'centre10':
        zmask, regions = centre_region(W, dimension)
    elif zero == 'two_regions':
        zmask, regions = two_regions(W, dimension)
    elif zero == 'strip20':
        zmask = ranks[:, 0] <= int(np.floor(0.2 * n))
    elif zero == 'circle':
        if dimension == 1:
            count = int(np.floor(.10 * n))
            start = (n - count) // 2
            zmask = np.zeros(n, dtype=bool)
            zmask[start:start + count] = True
        else:
            zmask = np.sum((W - .5) ** 2, axis=1) <= CIRCLE_RADIUS ** 2
    elif zero == 'none':
        zmask = np.zeros(n, bool)
    else:
        raise ValueError(zero)
    return zmask, ranks, regions


def _dist_to_zero(ranks, zmask):
    """Euclidean distance in normalized ranks from every unit to the nearest zero-effect unit."""
    from scipy.spatial import cKDTree
    v = ranks / float(len(ranks))
    d, _ = cKDTree(v[zmask]).query(v, k=1)
    d[zmask] = 0.0
    return d


def effects(zmask, ranks, effect, delta):
    r0 = ranks[:, 0]
    if effect == 'const':
        return np.where(zmask, 0.0, delta)
    if effect == 'grad1':
        d = _dist_to_zero(ranks, zmask)
        return np.where(zmask, 0.0, delta * d / d.max())
    if effect == 'neg':
        return np.where(zmask, 0.0, -delta)
    if effect == 'asym':
        lo = r0[zmask].min()
        return np.where(zmask, 0.0, np.where(r0 < lo, delta, -0.5 * delta))
    if effect == 'mixed':
        lo = r0[zmask].min()
        return np.where(zmask, 0.0, np.where(r0 < lo, delta, -delta))
    if effect == 'dispersed5':
        return np.where(((r0 - 1) // 5) % 2 == 0, 0.0, delta)
    if effect == 'cancel5':
        return np.where(((r0 - 1) // 5) % 2 == 0, delta, -delta)
    raise ValueError(effect)


def strata_of(ranks, n_strata=10):
    n = len(ranks)
    return ((ranks[:, 0] - 1) * n_strata) // n


# ------------------------------------------------------------------------------------------- datasets
def generate(dim, n, rep, zero='centre10', effect='const', scenario='S0', M=500, delta=1.0, sigma=0.5, pi=0.5,
             size_index=None):
    """One simulated dataset. Returns covariates W, ranks, the zero-effect indicator, effects tau, the
    assignment z, the raw outcome y, the outcome adjusted with the independent calibration sample
    (adj_indep), the outcome adjusted in sample (adj_insample, Appendix H.1 only), the M shared draws and
    the 5% grid."""
    if dim not in (1, 2) or n < 3 or M < 1:
        raise ValueError('dimension must be 1 or 2, n >= 3 and M >= 1')
    seeds = seed_plan(rep, SIZE_INDEX.get(n, 0) if size_index is None else size_index)
    rng = np.random.default_rng(seeds['data'])
    W = rng.uniform(0, 1, n if dim == 1 else (n, 2))
    if dim == 1:
        W.sort()
    zmask, ranks, regions = zero_set(W, dim, zero)
    baseline = baseline_mean(W)
    w1 = W if dim == 1 else W[:, 0]
    y1_binary = None
    if scenario in ('S0', 'S4', 'S5'):
        Y0 = baseline + rng.normal(0, sigma, n)
    elif scenario == 'S1':
        Y0 = baseline + sigma * rng.standard_t(3, n) / np.sqrt(3.0)
    elif scenario == 'S2':
        Y0 = baseline + rng.normal(0, 1.0, n) * 0.25 * (1.0 + 2.0 * w1)
    elif scenario == 'S3':
        u = rng.uniform(0, 1, n)
        p0 = 0.2 + 0.4 * w1
        Y0 = (u < p0).astype(np.float64)
        y1_binary = (u < np.where(zmask, p0, p0 + 0.2)).astype(np.float64)
    else:
        raise ValueError(scenario)
    if scenario == 'S3':
        tau = y1_binary - Y0
    else:
        tau = effects(zmask, ranks, 'mixed' if scenario == 'S4' else effect, delta)
    strata = None
    if scenario == 'S5':
        strata = strata_of(ranks)
        z = sample_half_within_strata(strata, 1, np.random.default_rng(seeds['assignment']))[0]
    else:
        z = sample_crd(n, int(np.floor(pi * n)), 1, np.random.default_rng(seeds['assignment']))[0]
    y = np.where(z == 1, Y0 + tau, Y0)
    # independent calibration sample for the adjustment (control outcomes of the same scenario)
    if scenario in ('S0', 'S4', 'S5'):
        beta, _, _ = independent_baseline(dim, n, sigma, seeds['calibration'])
    else:
        crng = np.random.default_rng(seeds['calibration'])
        Wc = crng.uniform(0, 1, n)
        bc = baseline_mean(Wc)
        if scenario == 'S1':
            Yc = bc + sigma * crng.standard_t(3, n) / np.sqrt(3.0)
        elif scenario == 'S2':
            Yc = bc + crng.normal(0, 1.0, n) * 0.25 * (1.0 + 2.0 * Wc)
        else:
            Yc = (crng.uniform(0, 1, n) < 0.2 + 0.4 * Wc).astype(np.float64)
        X = np.column_stack((np.ones(n), Wc))
        beta = np.linalg.lstsq(X, Yc, rcond=None)[0]
    adj_indep = y - predict_baseline(W, beta)
    Xs = np.column_stack((np.ones(n), W if dim == 2 else W[:, None]))
    bs = np.linalg.lstsq(Xs, y, rcond=None)[0]
    adj_insample = y - Xs @ bs
    if scenario == 'S5':
        draws = sample_half_within_strata(strata, M, np.random.default_rng(seeds['monte_carlo']))
    else:
        draws = sample_crd(n, int(z.sum()), M, np.random.default_rng(seeds['monte_carlo']))
    return dict(dim=dim, n=n, rep=rep, seeds=seeds, W=W, ranks=ranks, zmask=zmask, regions=regions, tau=tau,
                z=z.astype(np.int8), y=y, adj_indep=adj_indep, adj_insample=adj_insample, beta=beta,
                draws=draws.astype(np.int8), strata=strata, grid=paper_grid(n))


# ------------------------------------------------------------------------------- true zero-effect size
@njit(cache=True)
def _largest_zero_region(orders, zero):
    n = orders.shape[0]
    best = 0
    for c in range(n):
        k = 0
        while k < n and zero[orders[c, k]]:
            k += 1
        if k > best:
            best = k
    return best


def true_size(design, orders=None):
    """K0 (L0 in one dimension): the size of the largest candidate region in which every effect is zero."""
    zero = design['tau'] == 0
    if design['dim'] == 1:                          # W is sorted: the longest run of zero effects
        best = run = 0
        for v in zero:
            run = run + 1 if v else 0
            best = max(best, run)
        return int(best)
    if orders is None:
        from .search import knn_orders
        orders = knn_orders(design['W'])
    return int(_largest_zero_region(orders, zero))


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()
