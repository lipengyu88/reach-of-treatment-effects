"""Randomization inference for zero-effect regions.

Code for "The reach of treatment effects: randomization inference for zero-effect regions" by Pengyu Li
and Yao Zhang. Local Fisher randomization tests over candidate regions of a pre-treatment covariate
(blocks of consecutive units in one dimension, nearest-neighbour regions in two) are inverted into an
upper confidence bound for the size of the largest region in which treatment has no effect.

Modules: kernels (compiled kernels), search (evaluators and searches of Table 1), grids (searched grids
and inversion), randomization (draws), designs (simulation designs), realdata (readers for the three
experiments), api (upper_bound, the whole procedure for one dataset).
"""
from .utils import limit_threads as _limit_threads

_limit_threads()        # containers often show more CPUs than their quota; NUMBA_NUM_THREADS overrides this

from .api import upper_bound  # noqa: E402

__version__ = '1.0.0'
__all__ = ['upper_bound']
