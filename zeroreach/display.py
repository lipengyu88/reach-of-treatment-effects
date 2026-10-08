"""Aggregates behind the region displays (Sections 1, 6.5, 6.7 and 6.8).

The displays show which units are covered by candidate regions of the reported size that are not rejected.
They are descriptive, not confidence sets for the location of a zero-effect region. For the real data only
these aggregates are stored, never unit-level values.
"""
from __future__ import annotations

import numpy as np


def hist_pieces(W, mask, edges):
    """Single-coloured histogram pieces. W is sorted in the order of the test (covariate ascending, ties
    by unit identifier); inside each column the units are laid out from left to right in that order, so a
    column that contains a region boundary is split side by side, never stacked. Returns rows
    (left, width, height, covered): every piece of a column has the column's count as height and a width
    proportional to its number of units."""
    W, mask, edges = np.asarray(W, float), np.asarray(mask, bool), np.asarray(edges, float)
    if np.any(np.diff(W) < 0):
        raise ValueError('W must be sorted in the order of the test')
    if W[0] < edges[0] or W[-1] >= edges[-1]:
        raise ValueError('edges must cover the data')
    cut = np.searchsorted(W, edges, side='left')
    pieces = []
    for j in range(len(edges) - 1):
        a, b = cut[j], cut[j + 1]
        if b == a:
            continue
        width, seg = edges[j + 1] - edges[j], mask[a:b]
        change = np.flatnonzero(np.diff(seg.astype(np.int8))) + 1
        for s, e in zip(np.r_[0, change], np.r_[change, b - a]):
            pieces.append((float(edges[j] + width * s / (b - a)), float(width * (e - s) / (b - a)), int(b - a),
                           bool(seg[s])))
    return pieces


FILL_SHARE = 0.75


def fill_columns(pieces, share=FILL_SHARE):
    """Draw a split column as one covered piece when at least `share` of its units are covered.

    `pieces` are rows (left, width, height, covered) as returned by hist_pieces: the pieces of a column are
    adjacent, have the column's count as height and together span one bin. Columns with a smaller covered
    share, and columns that are not split, are returned unchanged."""
    rows = sorted(tuple(r) for r in pieces)
    if not rows:
        return rows
    width = max(r[1] for r in rows)          # a column that is not split spans one whole bin
    out, i = [], 0
    while i < len(rows):
        group, span = [rows[i]], rows[i][1]
        while span < width * (1 - 1e-9) and i + len(group) < len(rows):
            group.append(rows[i + len(group)])
            span += group[-1][1]
        if abs(span - width) > 1e-6 * width or len({r[2] for r in group}) != 1:
            raise ValueError('pieces do not group into columns of equal width')
        covered = sum(r[1] for r in group if r[3])
        if len(group) > 1 and covered >= share * span:
            out.append((group[0][0], span, group[0][2], True))
        else:
            out.extend(group)
        i += len(group)
    return out


def mask_runs(mask):
    """Maximal runs of covered units, as (start, end) positions (end exclusive) in the order of the test."""
    m = np.asarray(mask, bool).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m, [0]))))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]
