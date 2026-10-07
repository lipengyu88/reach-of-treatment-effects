"""Compiled kernels for the searches of Table 1 (one-dimensional blocks, two-dimensional rank neighbourhoods).

Choices that follow the paper:
  * the centred linear statistic, written as k*sum(zY) - sum(Y)*sum(z) (Section 3.2);
  * the plus-one Monte Carlo p-value with ties counted, maximized over candidate regions;
  * shared tests: M global completely randomized draws reused for every region (Section 3.2);
  * conditional tests: every region redrawn from scratch with its observed treated count fixed
    (Section 3.1; Table 1, "computed from scratch");
  * two-dimensional candidates: all centres, Euclidean nearest neighbours in normalized ranks, the unit
    itself first, distance ties broken by the first-coordinate rank (Section 4).
Engineering choices that do not change any result: numba threads, compensated (TwoSum) cumulative sums,
and a near-tie band whose members are resolved by exact set comparison. A near tie that is not a
structural tie is counted as 'unresolved'; a run with unresolved == 0 has every comparison certified exact.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

U = 2.0 ** -53
TIE_FACTOR = 40.0          # near-tie radius = TIE_FACTOR * k * U * sum|y|


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
@njit(inline='always')
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@njit(inline='always')
def _splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x, z ^ (z >> np.uint64(31))


@njit(inline='always')
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(inline='always')
def _next(s):
    result = _rotl(s[1] * np.uint64(5), 7) * np.uint64(9)
    t = s[1] << np.uint64(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return result


@njit(inline='always')
def _seed(s, a, b, c):
    x = np.uint64(a) * np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(b) ^ (np.uint64(c) * np.uint64(0xD1B54A32D192ED03))
    for q in range(4):
        x, v = _splitmix(x)
        s[q] = v


@njit(inline='always')
def _bounded(s, x32, bound):
    """Lemire's unbiased map of a 32-bit value into [0, bound)."""
    b = np.uint64(bound)
    m = x32 * b
    low = m & np.uint64(0xFFFFFFFF)
    if low < b:
        t = (np.uint64(0x100000000) - b) % b
        while low < t:
            x32 = _next(s) >> np.uint64(32)
            m = x32 * b
            low = m & np.uint64(0xFFFFFFFF)
    return np.int64(m >> np.uint64(32))


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def knn_orders(r1, r2):
    """orders[i] = units sorted by squared rank distance to i, ties by rank r1."""
    n = r1.size
    orders = np.empty((n, n), np.int32)
    for i in prange(n):
        key = np.empty(n, np.int64)
        for j in range(n):
            d1 = r1[j] - r1[i]
            d2 = r2[j] - r2[i]
            key[j] = (d1 * d1 + d2 * d2) * (n + 1) + r1[j]
        orders[i] = np.argsort(key).astype(np.int32)
    return orders


# ----------------------------------------------------------------------------
# shared 1D: compensated prefix sums of each draw, blocks via differences
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def shared1d_prefix(y, drawsT):
    """drawsT: n x M int8 (unit-major). Returns (Fz int32, Fhi, Flo) of shape (n+1) x M."""
    n, M = drawsT.shape
    Fz = np.zeros((n + 1, M), np.int32)
    Fhi = np.zeros((n + 1, M), np.float64)
    Flo = np.zeros((n + 1, M), np.float64)
    W = 32
    nblk = (M + W - 1) // W
    for b in prange(nblk):
        m0 = b * W
        m1 = min(M, m0 + W)
        for j in range(n):
            yj = y[j]
            for m in range(m0, m1):
                d = drawsT[j, m]
                Fz[j + 1, m] = Fz[j, m] + d
                h, e = _two_sum(Fhi[j, m], yj * d)
                Fhi[j + 1, m] = h
                Flo[j + 1, m] = Flo[j, m] + e
    return Fz, Fhi, Flo


@njit(cache=True)
def comp_prefix(x):
    n = x.size
    hi = np.zeros(n + 1)
    lo = np.zeros(n + 1)
    h = 0.0
    l = 0.0
    for j in range(n):
        h, e = _two_sum(h, x[j])
        l += e
        hi[j + 1] = h
        lo[j + 1] = l
    return hi, lo


@njit(parallel=True, cache=True)
def shared1d_counts(k, y, z, drawsT, Fz, Fhi, Flo, Yhi, Ylo, Zc, Ohi, Olo, changes, radius_unit):
    """Exceedance counts for all n-k+1 blocks of length k (sorted order).

    Returns counts, unresolved near ties (per block), structural near ties.
    """
    n, M = drawsT.shape
    nb = n - k + 1
    counts = np.zeros(nb, np.int64)
    unresolved = np.zeros(nb, np.int64)
    structural = np.zeros(nb, np.int64)
    R = radius_unit * k
    for s in prange(nb):
        e = s + k
        qy = (Yhi[e] - Yhi[s]) + (Ylo[e] - Ylo[s])
        qzo = Zc[e] - Zc[s]
        qzyo = (Ohi[e] - Ohi[s]) + (Olo[e] - Olo[s])
        if changes[e - 1] == changes[s]:      # constant block: every statistic is 0
            counts[s] = M
            continue
        tobs = 0.0 if (qzo == 0 or qzo == k) else k * qzyo - qy * qzo
        c = 0
        for m in range(M):
            qz = Fz[e, m] - Fz[s, m]
            if qz == 0 or qz == k:
                t = 0.0
            else:
                qzy = (Fhi[e, m] - Fhi[s, m]) + (Flo[e, m] - Flo[s, m])
                t = k * qzy - qy * qz
            d = t - tobs
            if abs(d) <= R:
                # structural tie: identical within-block assignment or both degenerate
                same = qz == qzo
                if same and not (qz == 0 or qz == k):
                    for i in range(s, e):
                        if drawsT[i, m] != z[i]:
                            same = False
                            break
                if same:
                    structural[s] += 1
                    c += 1
                    continue
                unresolved[s] += 1
            if d >= 0.0:
                c += 1
        counts[s] = c
    return counts, unresolved, structural


# ----------------------------------------------------------------------------
# shared 2D: per-centre compensated running sums, moved along neighbour orders
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def shared2d_move_and_count(k_from, k_to_list, y, z, drawsT, orders, constant_until,
                            Qz, Qhi, Qlo, Qy, Qylo, Qzo, Qzyo, Qzyolo, radius_unit, count_out,
                            unres_out, struct_out):
    """Move every centre's state from size k_from through the sizes in
    k_to_list (monotone sequence), recording exceedance counts at each listed
    size into count_out[j, centre]. State arrays are updated in place.
    """
    n, M = drawsT.shape
    nk = k_to_list.size
    for c in prange(n):
        kcur = k_from
        for j in range(nk):
            kt = k_to_list[j]
            # move
            while kcur > kt:
                u = orders[c, kcur - 1]
                yu = y[u]
                h, e = _two_sum(Qy[c], -yu)
                Qy[c] = h
                Qylo[c] += e
                if z[u]:
                    Qzo[c] -= 1
                    h, e = _two_sum(Qzyo[c], -yu)
                    Qzyo[c] = h
                    Qzyolo[c] += e
                for m in range(M):
                    d = drawsT[u, m]
                    Qz[c, m] -= d
                    h, e = _two_sum(Qhi[c, m], -yu * d)
                    Qhi[c, m] = h
                    Qlo[c, m] += e
                kcur -= 1
            while kcur < kt:
                u = orders[c, kcur]
                yu = y[u]
                h, e = _two_sum(Qy[c], yu)
                Qy[c] = h
                Qylo[c] += e
                if z[u]:
                    Qzo[c] += 1
                    h, e = _two_sum(Qzyo[c], yu)
                    Qzyo[c] = h
                    Qzyolo[c] += e
                for m in range(M):
                    d = drawsT[u, m]
                    Qz[c, m] += d
                    h, e = _two_sum(Qhi[c, m], yu * d)
                    Qhi[c, m] = h
                    Qlo[c, m] += e
                kcur += 1
            k = kcur
            if k <= constant_until[c]:          # constant region (includes k=1)
                count_out[j, c] = M
                continue
            qy = Qy[c] + Qylo[c]
            qzo = Qzo[c]
            tobs = 0.0 if (qzo == 0 or qzo == k) else k * (Qzyo[c] + Qzyolo[c]) - qy * qzo
            R = radius_unit * k
            cnt = 0
            for m in range(M):
                qz = Qz[c, m]
                if qz == 0 or qz == k:
                    t = 0.0
                else:
                    t = k * (Qhi[c, m] + Qlo[c, m]) - qy * qz
                d = t - tobs
                if abs(d) <= R:
                    same = qz == qzo
                    if same and not (qz == 0 or qz == k):
                        for p in range(k):
                            uu = orders[c, p]
                            if drawsT[uu, m] != z[uu]:
                                same = False
                                break
                    if same:
                        struct_out[j, c] += 1
                        cnt += 1
                        continue
                    unres_out[j, c] += 1
                if d >= 0.0:
                    cnt += 1
            count_out[j, c] = cnt


# ----------------------------------------------------------------------------
# conditional (1D blocks and 2D regions): fresh fixed-count draws per region
# ----------------------------------------------------------------------------
@njit(cache=True)
def _cond_region(yloc, zloc, M, s):
    """Exceedance count of a fixed-count conditional test on one region.

    yloc/zloc are the region's outcomes/assignments (contiguous copies).
    Draws: partial Fisher-Yates of min(c, k-c) labels, two indices per
    64-bit RNG output. Returns (count, unresolved, structural).
    """
    k = yloc.size
    c = 0
    sabs = 0.0
    y0 = yloc[0]
    constant = True
    for i in range(k):
        if zloc[i]:
            c += 1
        sabs += abs(yloc[i])
        if yloc[i] != y0:
            constant = False
    if c == 0 or c == k or constant:
        return M, 0, 0
    comp = c > k - c
    cc = k - c if comp else c
    obs = 0.0
    for i in range(k):
        if (zloc[i] != 0) != comp:
            obs += yloc[i]
    R = 4.0 * k * U * sabs + 1e-300
    idx = np.arange(k).astype(np.int32)
    cnt = 0
    unres = 0
    struct = 0
    for m in range(M):
        acc = 0.0
        j = 0
        while j < cc:
            r64 = _next(s)
            r = j + _bounded(s, r64 >> np.uint64(32), k - j)
            t = idx[j]
            idx[j] = idx[r]
            idx[r] = t
            acc += yloc[idx[j]]
            j += 1
            if j < cc:
                r = j + _bounded(s, r64 & np.uint64(0xFFFFFFFF), k - j)
                t = idx[j]
                idx[j] = idx[r]
                idx[r] = t
                acc += yloc[idx[j]]
                j += 1
        d = acc - obs
        if abs(d) <= R:
            same = True
            for q in range(cc):
                if (zloc[idx[q]] != 0) == comp:   # drawn unit not in observed subset
                    same = False
                    break
            if same:
                struct += 1
                cnt += 1
                continue
            unres += 1
        # not comp: drawn treated sum >= observed treated sum
        # comp:     drawn control sum <= observed control sum
        if (d >= 0.0) if not comp else (d <= 0.0):
            cnt += 1
    return cnt, unres, struct


@njit(parallel=True, cache=True)
def cond1d_counts(k, y, z, M, seed):
    n = y.size
    nb = n - k + 1
    counts = np.zeros(nb, np.int64)
    unres = np.zeros(nb, np.int64)
    struct = np.zeros(nb, np.int64)
    for s0 in prange(nb):
        st = np.empty(4, np.uint64)
        _seed(st, seed, k, s0)
        a, b, c = _cond_region(y[s0:s0 + k], z[s0:s0 + k], M, st)
        counts[s0] = a
        unres[s0] = b
        struct[s0] = c
    return counts, unres, struct


@njit(parallel=True, cache=True)
def cond2d_counts(k, reps, fp1, y, z, orders, M, seed):
    """Conditional counts for representative centres of distinct regions."""
    nr = reps.size
    counts = np.zeros(nr, np.int64)
    unres = np.zeros(nr, np.int64)
    struct = np.zeros(nr, np.int64)
    for r in prange(nr):
        c = reps[r]
        yloc = np.empty(k)
        zloc = np.empty(k, np.int8)
        for p in range(k):
            u = orders[c, p]
            yloc[p] = y[u]
            zloc[p] = z[u]
        st = np.empty(4, np.uint64)
        _seed(st, seed, k, fp1[c])
        a, b, cc = _cond_region(yloc, zloc, M, st)
        counts[r] = a
        unres[r] = b
        struct[r] = cc
    return counts, unres, struct


@njit(parallel=True, cache=True)
def fp_remove(fp1, fp2, h1, h2, orders, k_from, k_to):
    """Update XOR fingerprints of every centre's region from size k_from down to k_to."""
    n = fp1.size
    for c in prange(n):
        for p in range(k_to, k_from):
            u = orders[c, p]
            fp1[c] ^= h1[u]
            fp2[c] ^= h2[u]


@njit(cache=True)
def _same_region(orders, a, b, k):
    n = orders.shape[1]
    if k <= n - k:
        x = np.sort(orders[a, :k])
        y = np.sort(orders[b, :k])
    else:
        x = np.sort(orders[a, k:])
        y = np.sort(orders[b, k:])
    for i in range(x.size):
        if x[i] != y[i]:
            return False
    return True


@njit(cache=True)
def group_regions(fp1, fp2, orders, k):
    """Exact grouping of identical regions: fingerprints propose, sets verify."""
    n = fp1.size
    order = np.argsort(fp1)
    group = np.empty(n, np.int64)
    reps = np.empty(n, np.int64)
    nrep = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and fp1[order[j + 1]] == fp1[order[i]]:
            j += 1
        # verify members i..j against representatives created in this bucket
        first_rep = nrep
        for t in range(i, j + 1):
            c = order[t]
            found = -1
            for q in range(first_rep, nrep):
                if fp2[reps[q]] == fp2[c] and _same_region(orders, reps[q], c, k):
                    found = q
                    break
            if found < 0:
                reps[nrep] = c
                found = nrep
                nrep += 1
            group[c] = found
        i = j + 1
    return reps[:nrep].copy(), group


# ----------------------------------------------------------------------------
# Gaussian warm-start proposals (Appendix A); float64 on centred outcomes
# ----------------------------------------------------------------------------
@njit(cache=True)
def _upper_tail(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


@njit(parallel=True, cache=True)
def gauss1d_max(k, Py, Py2, Pz, Pzy, n1):
    n = Py.size - 1
    nb = n - k + 1
    fac = n1 * (n - n1) / (n * (n - 1.0))
    best = np.zeros(nb)
    for s in prange(nb):
        e = s + k
        sy = Py[e] - Py[s]
        sy2 = Py2[e] - Py2[s]
        sz = Pz[e] - Pz[s]
        szy = Pzy[e] - Pzy[s]
        sq = k * sy2 - sy * sy
        if sq <= 0.0:
            best[s] = 1.0
            continue
        t = k * szy - sy * sz
        best[s] = _upper_tail(t / math.sqrt(k * sq * fac))
    return best.max()


@njit(parallel=True, cache=True)
def gauss2d_move_max(k_from, k_to, yc, z, orders, Sy, Sy2, Sz, Szy, n1):
    n = yc.size
    fac = n1 * (n - n1) / (n * (n - 1.0))
    best = np.zeros(n)
    for c in prange(n):
        kc = k_from
        while kc > k_to:
            u = orders[c, kc - 1]
            Sy[c] -= yc[u]
            Sy2[c] -= yc[u] * yc[u]
            if z[u]:
                Sz[c] -= 1
                Szy[c] -= yc[u]
            kc -= 1
        while kc < k_to:
            u = orders[c, kc]
            Sy[c] += yc[u]
            Sy2[c] += yc[u] * yc[u]
            if z[u]:
                Sz[c] += 1
                Szy[c] += yc[u]
            kc += 1
        k = k_to
        sq = k * Sy2[c] - Sy[c] * Sy[c]
        if sq <= 1e-12 * k * Sy2[c]:
            best[c] = 1.0
            continue
        t = k * Szy[c] - Sy[c] * Sz[c]
        best[c] = _upper_tail(t / math.sqrt(k * sq * fac))
    return best.max()


@njit(parallel=True, cache=True)
def constant_until_2d(y, orders):
    """Largest k such that the first k neighbours of each centre share one y value."""
    n = y.size
    out = np.empty(n, np.int64)
    for c in prange(n):
        y0 = y[orders[c, 0]]
        p = 1
        while p < n and y[orders[c, p]] == y0:
            p += 1
        out[c] = p
    return out


@njit(cache=True)
def change_prefix(y):
    n = y.size
    ch = np.zeros(n, np.int64)
    for i in range(1, n):
        ch[i] = ch[i - 1] + (1 if y[i] != y[i - 1] else 0)
    return ch


# ----------------------------------------------------------------------------
# shared 1D, draw-major layout (cache-resident per draw; replaces the
# length-major sweep, which was memory-bandwidth bound)
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def shared1d_prefix_dm(y, draws):
    """draws: M x n int8. Returns Fz (M x n+1 int32), Fhi, Flo (M x n+1 float64)."""
    M, n = draws.shape
    Fz = np.zeros((M, n + 1), np.int32)
    Fhi = np.zeros((M, n + 1), np.float64)
    Flo = np.zeros((M, n + 1), np.float64)
    for m in prange(M):
        cz = 0
        hi = 0.0
        lo = 0.0
        for j in range(n):
            d = draws[m, j]
            cz += d
            hi, e = _two_sum(hi, y[j] * d)
            lo += e
            Fz[m, j + 1] = cz
            Fhi[m, j + 1] = hi
            Flo[m, j + 1] = lo
    return Fz, Fhi, Flo


@njit(parallel=True, cache=True)
def shared1d_counts_dm(k, y, z, draws, Fz, Fhi, Flo, Yhi, Ylo, Zc, Ohi, Olo, changes,
                       radius_unit, nchunk):
    M, n = draws.shape
    nb = n - k + 1
    qy = np.empty(nb)
    tobs = np.empty(nb)
    qzo = np.empty(nb, np.int64)
    const = np.zeros(nb, np.bool_)
    for s in prange(nb):
        e = s + k
        qy[s] = (Yhi[e] - Yhi[s]) + (Ylo[e] - Ylo[s])
        qzo[s] = Zc[e] - Zc[s]
        const[s] = changes[e - 1] == changes[s]
        if qzo[s] == 0 or qzo[s] == k:
            tobs[s] = 0.0
        else:
            tobs[s] = k * ((Ohi[e] - Ohi[s]) + (Olo[e] - Olo[s])) - qy[s] * qzo[s]
    R = radius_unit * k
    cnt = np.zeros((nchunk, nb), np.int32)
    unres = np.zeros(nchunk, np.int64)
    struct = np.zeros(nchunk, np.int64)
    for ch in prange(nchunk):
        m0 = ch * M // nchunk
        m1 = (ch + 1) * M // nchunk
        for m in range(m0, m1):
            for s in range(nb):
                e = s + k
                qz = Fz[m, e] - Fz[m, s]
                if qz == 0 or qz == k:
                    t = 0.0
                else:
                    t = k * ((Fhi[m, e] - Fhi[m, s]) + (Flo[m, e] - Flo[m, s])) - qy[s] * qz
                d = t - tobs[s]
                ge = d >= 0.0
                if abs(d) <= R and not const[s]:
                    same = qz == qzo[s]
                    if same and not (qz == 0 or qz == k):
                        for i in range(s, e):
                            if draws[m, i] != z[i]:
                                same = False
                                break
                    if same:
                        struct[ch] += 1
                        ge = True
                    else:
                        unres[ch] += 1
                if ge:
                    cnt[ch, s] += 1
    counts = np.zeros(nb, np.int64)
    for s in prange(nb):
        if const[s]:
            counts[s] = M
        else:
            tot = 0
            for ch in range(nchunk):
                tot += cnt[ch, s]
            counts[s] = tot
    return counts, unres.sum(), struct.sum()


# ----------------------------------------------------------------------------
# shared 1D, tiled: a chunk of lengths x a tile of block starts x all draws.
# Per tile the observed-block arrays stay in cache and each draw's prefix
# segment is read once per chunk; the inner loop is branch-free.
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def shared1d_chunk_counts(klist, z, draws, Fz, Fhi, Flo, Yhi, Ylo, Zc, Ohi, Olo, changes,
                          radius_unit, Ts):
    M, n = draws.shape
    b = klist.size
    kmin = klist.min()
    nbmax = n - kmin + 1
    qy = np.zeros((b, nbmax))
    tobs = np.zeros((b, nbmax))
    qzo = np.zeros((b, nbmax), np.int64)
    const = np.zeros((b, nbmax), np.bool_)
    for j in prange(b):
        k = klist[j]
        for s in range(n - k + 1):
            e = s + k
            qy[j, s] = (Yhi[e] - Yhi[s]) + (Ylo[e] - Ylo[s])
            q = Zc[e] - Zc[s]
            qzo[j, s] = q
            const[j, s] = changes[e - 1] == changes[s]
            if q == 0 or q == k:
                tobs[j, s] = 0.0
            else:
                tobs[j, s] = k * ((Ohi[e] - Ohi[s]) + (Olo[e] - Olo[s])) - qy[j, s] * q
    cnt = np.zeros((b, nbmax), np.int64)
    ntile = (nbmax + Ts - 1) // Ts
    unres = np.zeros(ntile, np.int64)
    struct = np.zeros(ntile, np.int64)
    for t in prange(ntile):
        s0 = t * Ts
        s1 = min(nbmax, s0 + Ts)
        for m in range(M):
            for j in range(b):
                k = klist[j]
                send = min(s1, n - k + 1)
                if send <= s0:
                    continue
                R = radius_unit * k
                nnear = 0
                for s in range(s0, send):
                    e = s + k
                    qz = Fz[m, e] - Fz[m, s]
                    tt = k * ((Fhi[m, e] - Fhi[m, s]) + (Flo[m, e] - Flo[m, s])) - qy[j, s] * qz
                    tt = 0.0 if (qz == 0 or qz == k) else tt
                    d = tt - tobs[j, s]
                    cnt[j, s] += d >= 0.0
                    nnear += abs(d) <= R
                if nnear > 0:
                    for s in range(s0, send):
                        e = s + k
                        qz = Fz[m, e] - Fz[m, s]
                        tt = k * ((Fhi[m, e] - Fhi[m, s]) + (Flo[m, e] - Flo[m, s])) - qy[j, s] * qz
                        tt = 0.0 if (qz == 0 or qz == k) else tt
                        d = tt - tobs[j, s]
                        if abs(d) > R or const[j, s]:
                            continue
                        same = qz == qzo[j, s]
                        if same and not (qz == 0 or qz == k):
                            for i in range(s, e):
                                if draws[m, i] != z[i]:
                                    same = False
                                    break
                        if same:
                            struct[t] += 1
                            if d < 0.0:
                                cnt[j, s] += 1
                        else:
                            unres[t] += 1
    for j in prange(b):
        k = klist[j]
        for s in range(n - k + 1):
            if const[j, s]:
                cnt[j, s] = M
    return cnt, unres.sum(), struct.sum()
