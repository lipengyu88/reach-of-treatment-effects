"""Histogram pieces behind the region displays."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroreach.display import fill_columns, hist_pieces  # noqa: E402


def test_fill_columns_only_merges_mostly_covered_columns():
    W = (np.arange(40) + 0.5) / 40
    mask = np.zeros(40, bool)
    mask[8:20] = True          # column [0, 0.25): 2 of 10 covered; column [0.25, 0.5): all covered
    mask[22:30] = True         # column [0.5, 0.75): first 8 of 10 covered
    pieces = hist_pieces(W, mask, np.linspace(0, 1, 5))
    out = fill_columns(pieces, 0.75)
    assert out == fill_columns(out, 0.75)
    # the 80% column becomes one covered piece; the 20% column keeps both pieces
    assert (0.5, 0.25, 10, True) in [(round(a, 12), round(b, 12), c, d) for a, b, c, d in out]
    assert sum(1 for p in out if 0.0 <= p[0] < 0.25) == 2
    # widths still add up to the whole range and heights are unchanged
    assert sum(p[1] for p in out) == pytest.approx(1.0)
    assert {p[2] for p in out} == {10}


def test_fill_columns_threshold_is_inclusive_and_can_be_disabled():
    pieces = [(0.0, 0.75, 4, True), (0.75, 0.25, 4, False), (1.0, 1.0, 2, False)]
    assert fill_columns(pieces, 0.75) == [(0.0, 1.0, 4, True), (1.0, 1.0, 2, False)]
    assert fill_columns(pieces, 0.8) == pieces
    with pytest.raises(ValueError):
        fill_columns([(0.0, 0.5, 4, True), (0.5, 0.3, 3, False), (1.0, 1.0, 2, False)])
