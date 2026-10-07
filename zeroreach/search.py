"""Evaluators and the four searches of Table 1 (Sections 3 and 4, Appendix A).

Each evaluator exposes max_p(k), the largest plus-one Monte Carlo p-value over all candidate regions of
size k, and gaussian_max(k), the Gaussian proposal used by the warm start. The searches are
  descending(ev, n)              sizes n, n-1, ...; the first non-rejected size is the bound;
  bisection(ev, n, warm=False)   requires the monotonicity assumption of Section 3.3; warm=True first
                                 brackets the bound with the Gaussian proposal (Appendix A).
numpy_grid_pvalues() is a plain numpy evaluator of one-dimensional blocks on a grid, used for the
two-sided statistic |T_B| (Section 6.6 and Appendix H).
"""
from __future__ import annotations

import numpy as np

from . import kernels as K

ALPHA = 0.05


def _ranks(x):
    order = np.lexsort((np.arange(len(x)), x))       # ties by stable unit id
    r = np.empty(len(x), np.int64)
    r[order] = np.arange(1, len(x) + 1)
    return r


def knn_orders(W):
    return K.knn_orders(_ranks(W[:, 0]), _ranks(W[:, 1]))


class _Stats:
    def __init__(self):
        self.cache = {}
        self.gauss_cache = {}
        self.unresolved = 0
        self.structural = 0
        self.mc_queries = []
        self.gauss_queries = []


# ---------------------------------------------------------------- shared 1D
class Shared1D(_Stats):
    def __init__(self, y, z, draws):
        super().__init__()
        self.n, self.M = len(y), len(draws)
        self.y = np.ascontiguousarray(y, np.float64)
        self.z = np.ascontiguousarray(z, np.int8)
        self.draws = np.ascontiguousarray(draws.astype(np.int8))
        self.Fz, self.Fhi, self.Flo = K.shared1d_prefix_dm(self.y, self.draws)
        import numba
        self.nchunk = max(1, min(self.M, 2 * numba.get_num_threads()))
        self.Ts = 256
        self.Yhi, self.Ylo = K.comp_prefix(self.y)
        self.Ohi, self.Olo = K.comp_prefix(self.y * self.z)
        self.Zc = np.concatenate(([0], np.cumsum(self.z, dtype=np.int64)))
        self.changes = K.change_prefix(self.y)
        self.radius_unit = K.TIE_FACTOR * K.U * float(np.abs(self.y).sum())
        self._g = None

    def _chunk(self, klist):
        klist = np.asarray(klist, np.int64)
        cnt, unres, struct = K.shared1d_chunk_counts(klist, self.z, self.draws, self.Fz, self.Fhi, self.Flo,
                                                     self.Yhi, self.Ylo, self.Zc, self.Ohi, self.Olo,
                                                     self.changes, self.radius_unit, self.Ts)
        self.unresolved += int(unres)
        self.structural += int(struct)
        return cnt

    def max_p(self, k):
        if k in self.cache:
            return self.cache[k]
        cnt = self._chunk([k])
        p = (cnt[0, :self.n - k + 1].max() + 1.0) / (self.M + 1.0)
        self.cache[k] = p
        self.mc_queries.append(k)
        return p

    def descending(self, alpha=ALPHA, chunk=32):
        k_hi = self.n
        while k_hi >= 1:
            klist = np.arange(k_hi, max(0, k_hi - chunk), -1)
            cnt = self._chunk(klist)
            for j, k in enumerate(klist):
                p = (cnt[j, :self.n - k + 1].max() + 1.0) / (self.M + 1.0)
                self.cache[int(k)] = p
                self.mc_queries.append(int(k))
                if p > alpha:
                    return int(k)
            k_hi = int(klist[-1]) - 1
        return 0

    def local_p(self, k):
        cnt = self._chunk([k])
        return (cnt[0, :self.n - k + 1] + 1.0) / (self.M + 1.0)

    def gaussian_max(self, k):
        if k in self.gauss_cache:
            return self.gauss_cache[k]
        if self._g is None:
            yc = self.y - self.y.mean()
            self._g = (np.concatenate(([0.], np.cumsum(yc))), np.concatenate(([0.], np.cumsum(yc * yc))),
                       np.concatenate(([0.], np.cumsum(self.z.astype(np.float64)))),
                       np.concatenate(([0.], np.cumsum(yc * self.z))), float(self.z.sum()))
        p = K.gauss1d_max(k, *self._g)
        self.gauss_cache[k] = p
        self.gauss_queries.append(k)
        return p


# ---------------------------------------------------------------- shared 2D
STAGGER = 136      # float64 elements = 1088 bytes (1 KB + one cache line)


def _staggered(n, M, count=3):
    """count (n, M) float64 arrays carved from one buffer, starting 0, 1088, 2176, ... bytes past a
    4 KB boundary. Entries with equal (c, m) then never share address bits 0-11, which avoids 4K-aliasing
    stalls when the move kernel stores to one state array and loads the next. Engineering only: values
    and results are unchanged. Measured on an Intel Xeon 8358 with 50 threads: 2.3 ns per update when the
    allocator happened to place the arrays at equal offsets, 0.8 ns otherwise, 0.70-0.75 ns at every n
    with this layout."""
    size = n * M
    total = count * size + (count + 2) * 512
    buf = np.empty(total)
    base = (-(buf.ctypes.data // 8)) % 512
    views, end = [], base
    for j in range(count):
        start = end + (base + j * STAGGER - end) % 512
        end = start + size
        views.append(buf[start:end].reshape(n, M))
    return views


class Shared2D(_Stats):
    def __init__(self, y, z, draws, orders, layout='staggered'):
        super().__init__()
        n, M = len(y), len(draws)
        self.n, self.M = n, M
        self.y = np.ascontiguousarray(y, np.float64)
        self.z = np.ascontiguousarray(z, np.int8)
        self.orders = orders
        self.drawsT = np.ascontiguousarray(draws.T.astype(np.int8))
        self.constant_until = K.constant_until_2d(self.y, orders)
        # state at k = n: every centre's region is the whole sample
        tot_hi, tot_lo = K.comp_prefix(self.y)
        dz = draws.sum(axis=1).astype(np.int32)
        dhi = np.empty(M)
        dlo = np.empty(M)
        for m in range(M):
            h, l = K.comp_prefix(self.y * draws[m])
            dhi[m], dlo[m] = h[-1], l[-1]
        ohi, olo = K.comp_prefix(self.y * self.z)
        if layout == 'legacy':           # plain allocation (same results; slower at some n)
            self.Qz = np.repeat(dz[None, :], n, axis=0)
            self.Qhi = np.repeat(dhi[None, :], n, axis=0)
            self.Qlo = np.repeat(dlo[None, :], n, axis=0)
        else:                            # treated counts held exactly as float64 (integers <= n)
            self.Qz, self.Qhi, self.Qlo = _staggered(n, M)
            self.Qz[...] = dz[None, :]
            self.Qhi[...] = dhi[None, :]
            self.Qlo[...] = dlo[None, :]
        self.Qy = np.full(n, tot_hi[-1])
        self.Qylo = np.full(n, tot_lo[-1])
        self.Qzo = np.full(n, int(self.z.sum()), np.int64)
        self.Qzyo = np.full(n, ohi[-1])
        self.Qzyolo = np.full(n, olo[-1])
        self.current = n
        self.radius_unit = K.TIE_FACTOR * K.U * float(np.abs(self.y).sum())
        self._g = None

    def _counts(self, klist):
        klist = np.asarray(klist, np.int64)
        out = np.zeros((len(klist), self.n), np.int64)
        unres = np.zeros((len(klist), self.n), np.int64)
        struct = np.zeros((len(klist), self.n), np.int64)
        K.shared2d_move_and_count(self.current, klist, self.y, self.z, self.drawsT, self.orders,
                                  self.constant_until, self.Qz, self.Qhi, self.Qlo, self.Qy,
                                  self.Qylo, self.Qzo, self.Qzyo, self.Qzyolo, self.radius_unit,
                                  out, unres, struct)
        self.current = int(klist[-1])
        self.unresolved += int(unres.sum())
        self.structural += int(struct.sum())
        return out

    def max_p(self, k):
        if k in self.cache:
            return self.cache[k]
        out = self._counts([k])
        p = (out[0].max() + 1.0) / (self.M + 1.0)
        self.cache[k] = p
        self.mc_queries.append(k)
        return p

    def descending(self, alpha=ALPHA, chunk=64):
        """Sizes n, n-1, ... in chunks; the first non-rejection is the bound."""
        k_hi = self.n
        while k_hi >= 1:
            klist = np.arange(k_hi, max(0, k_hi - chunk), -1)
            out = self._counts(klist)
            for j, k in enumerate(klist):
                p = (out[j].max() + 1.0) / (self.M + 1.0)
                self.cache[int(k)] = p
                self.mc_queries.append(int(k))
                if p > alpha:
                    return int(k)
            k_hi = int(klist[-1]) - 1
        return 0

    def local_p(self, k):
        out = self._counts([k])
        return (out[0] + 1.0) / (self.M + 1.0)

    def gaussian_max(self, k):
        if k in self.gauss_cache:
            return self.gauss_cache[k]
        if self._g is None:
            yc = self.y - self.y.mean()
            self._g = dict(yc=yc, Sy=np.full(self.n, yc.sum()), Sy2=np.full(self.n, (yc * yc).sum()),
                           Sz=np.full(self.n, float(self.z.sum())), Szy=np.full(self.n, (yc * self.z).sum()),
                           cur=self.n)
        g = self._g
        p = K.gauss2d_move_max(g['cur'], k, g['yc'], self.z, self.orders, g['Sy'], g['Sy2'], g['Sz'],
                               g['Szy'], float(self.z.sum()))
        g['cur'] = k
        self.gauss_cache[k] = p
        self.gauss_queries.append(k)
        return p


# ---------------------------------------------------------------- conditional
class Cond1D(_Stats):
    def __init__(self, y, z, M, seed):
        super().__init__()
        self.n, self.M, self.seed = len(y), M, seed
        self.y = np.ascontiguousarray(y, np.float64)
        self.z = np.ascontiguousarray(z, np.int8)

    def max_p(self, k):
        if k in self.cache:
            return self.cache[k]
        counts, unres, struct = K.cond1d_counts(k, self.y, self.z, self.M, self.seed)
        self.unresolved += int(unres.sum())
        self.structural += int(struct.sum())
        p = (counts.max() + 1.0) / (self.M + 1.0)
        self.cache[k] = p
        self.mc_queries.append(k)
        return p


class Cond2D(_Stats):
    def __init__(self, y, z, orders, M, seed):
        super().__init__()
        n = len(y)
        self.n, self.M, self.seed = n, M, seed
        self.y = np.ascontiguousarray(y, np.float64)
        self.z = np.ascontiguousarray(z, np.int8)
        self.orders = orders
        rng = np.random.default_rng(0x5EED)          # fixed unit keys; not experiment RNG
        self.h1 = rng.integers(0, 2**63, n, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
        self.h2 = rng.integers(0, 2**63, n, dtype=np.int64).astype(np.uint64)
        full1 = np.bitwise_xor.reduce(self.h1)
        full2 = np.bitwise_xor.reduce(self.h2)
        self.fp1 = np.full(n, full1, np.uint64)
        self.fp2 = np.full(n, full2, np.uint64)
        self.current = n
        self.requested = 0
        self.computed = 0

    def _move(self, k):
        lo, hi = sorted((k, self.current))
        if lo != hi:
            K.fp_remove(self.fp1, self.fp2, self.h1, self.h2, self.orders, hi, lo)
        self.current = k

    def max_p(self, k):
        if k in self.cache:
            return self.cache[k]
        self._move(k)
        reps, _ = K.group_regions(self.fp1, self.fp2, self.orders, k)
        counts, unres, struct = K.cond2d_counts(k, reps, self.fp1, self.y, self.z, self.orders,
                                                self.M, self.seed)
        self.requested += self.n
        self.computed += len(reps)
        self.unresolved += int(unres.sum())
        self.structural += int(struct.sum())
        p = (counts.max() + 1.0) / (self.M + 1.0)
        self.cache[k] = p
        self.mc_queries.append(k)
        return p


# ---------------------------------------------------------------- searches
def descending(ev, n, alpha=ALPHA):
    """Largest length with P_k > alpha, checking n, n-1, ... (no monotonicity)."""
    if hasattr(ev, 'descending'):
        return ev.descending(alpha)
    for k in range(n, 0, -1):
        if ev.max_p(k) > alpha:
            return k
    return 0


def bisection(ev, n, alpha=ALPHA, warm=False):
    """Section 3.3 and Appendix A; requires the monotonicity assumption."""
    bracket, accepted = None, False
    if warm:
        a = 1
        if ev.gaussian_max(a) > alpha:
            while a < n:
                b = min(n, 2 * a)
                if ev.gaussian_max(b) <= alpha:
                    while b - a > 1:
                        c = (a + b) // 2
                        if ev.gaussian_max(c) > alpha:
                            a = c
                        else:
                            b = c
                    bracket = (a, b)
                    break
                a = b
        if bracket is not None:
            a, b = bracket
            pa, pb = ev.max_p(a), ev.max_p(b)
            if pa > alpha and pb <= alpha:
                return a, dict(bracket=bracket, accepted=True)
    if ev.max_p(1) <= alpha:
        return 0, dict(bracket=bracket, accepted=False)
    if ev.max_p(n) > alpha:
        return n, dict(bracket=bracket, accepted=False)
    a, b = 1, n
    while b - a > 1:
        c = (a + b) // 2
        if ev.max_p(c) > alpha:
            a = c
        else:
            b = c
    return a, dict(bracket=bracket, accepted=False)


def monotonicity_violations(cache):
    ks = sorted(cache)
    return [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if cache[b] > cache[a]]


# ---------------------------------------------------------------- numpy evaluator (one dimension, grid)
def numpy_grid_pvalues(y, z, draws, grid, two_sided):
    """Largest plus-one p-value over all blocks of each grid size, with the centred statistic T_B
    (two_sided=False) or with |T_B| (two_sided=True). Returns {size: P}."""
    n, M = len(y), len(draws)
    zf = z.astype(np.float64)
    Dy = draws.astype(np.float64)
    Y = np.concatenate(([0.0], np.cumsum(y)))
    Zo = np.concatenate(([0.0], np.cumsum(zf)))
    ZYo = np.concatenate(([0.0], np.cumsum(zf * y)))
    Zd = np.concatenate((np.zeros((M, 1)), np.cumsum(Dy, axis=1)), axis=1)
    ZYd = np.concatenate((np.zeros((M, 1)), np.cumsum(Dy * y[None, :], axis=1)), axis=1)
    tol = 1e-9 * (np.abs(y).sum() + 1.0)
    P = {}
    for k in grid:
        s = np.arange(0, n - k + 1)
        sy = Y[s + k] - Y[s]
        tobs = (ZYo[s + k] - ZYo[s]) - sy / k * (Zo[s + k] - Zo[s])
        td = (ZYd[:, s + k] - ZYd[:, s]) - sy[None, :] / k * (Zd[:, s + k] - Zd[:, s])
        if two_sided:
            cnt = (np.abs(td) >= np.abs(tobs)[None, :] - tol).sum(axis=0)
        else:
            cnt = (td >= tobs[None, :] - tol).sum(axis=0)
        P[int(k)] = float(((cnt + 1.0) / (M + 1.0)).max())
    return P
