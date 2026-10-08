"""Figure style shared by plot_figures.py: palette, fonts and small drawing helpers."""
from __future__ import annotations

import numpy as np

from zeroreach.display import FILL_SHARE, fill_columns

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e6e5e1', '#c9c8c2'
RC = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.6, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'pdf.fonttype': 42,
}
SYM_L, SYM_K = r'\widehat{L}_{n,0}^{\,U}', r'\widehat{K}_{0}^{\,U}'


def annotate(ax, lines, loc='upper left'):
    x, ha = (0.02, 'left') if loc == 'upper left' else (0.98, 'right')
    ax.text(x, 0.97, '\n'.join(lines), transform=ax.transAxes, va='top', ha=ha, fontsize=8.5,
            color=INK, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.85))


def bound_lines(bound, n, cert=None, sym=SYM_L):
    lines = [rf'${sym}/n$ = {bound:,}/{n:,} = {100 * bound / n:.1f}%']
    if cert is not None:
        lines.append(f'integer bound: {cert:,}/{n:,} = {100 * cert / n:.1f}%')
    return lines


def draw_pieces(ax, pieces, fill_share=FILL_SHARE):
    """Draw histogram pieces (left, width, height, covered) from zeroreach.display.hist_pieces. A column in
    which at least `fill_share` of the units are covered is drawn entirely blue (display.fill_columns)."""
    if fill_share is not None:
        pieces = fill_columns(pieces, fill_share)
    x, w, h, blue = (np.asarray(v) for v in zip(*pieces))
    blue = blue.astype(bool)
    for sel, colour in ((~blue, NEUTRAL), (blue, BLUE)):
        ax.bar(x[sel], h[sel], width=w[sel], align='edge', color=colour, edgecolor='white', linewidth=0.4)
    ax.set_ylabel('number of units')
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
