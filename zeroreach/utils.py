"""Small helpers shared by the runners and the figure and table scripts."""
from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.959963984540054):
    """Coverage estimate and its 95% Wilson interval."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def check_blas():
    """Check float64 matrix products and least squares before long runs.

    A failing check requires a compatible BLAS build or CPU dispatch setting. Forcing a kernel
    that requires instructions absent from the host can terminate the process.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 200))
    gemm_err = float(np.abs(A @ A - (A[:, :, None] * A[None, :, :]).sum(axis=1)).max())
    x = rng.uniform(size=500)
    X = np.column_stack((np.ones(500), x))
    y = 0.5 + 2.0 * x + 0.01 * rng.standard_normal(500)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    sxx, sx, sy, sxy = (x * x).sum(), x.sum(), y.sum(), (x * y).sum()
    det = 500 * sxx - sx * sx
    ref = np.array([(sxx * sy - sx * sxy) / det, (500 * sxy - sx * sy) / det])
    ls_err = float(np.abs(b - ref).max())
    if not (gemm_err < 1e-9 and ls_err < 1e-9):
        raise RuntimeError(f'BLAS check failed (matrix product error {gemm_err:.1e}, least squares error '
                           f'{ls_err:.1e}); use a BLAS build or dispatch setting compatible with this CPU')


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment():
    import numba
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                numba=numba.__version__, numba_threads=numba.get_num_threads(), cpu=cpu_model(),
                omp_threads=os.environ.get('OMP_NUM_THREADS'), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_default) + '\n')
    os.replace(tmp, path)


def _default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def single_thread_blas():
    """One BLAS thread per process: results of the small least-squares fits do not depend on threading."""
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')


def cpu_quota():
    """CPUs granted by a cgroup quota (containers often show more CPUs than they may use), else None."""
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()[:2]
        if quota != 'max':
            return max(1, int(int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def limit_threads():
    """Unless NUMBA_NUM_THREADS is set, use no more numba threads than the CPU quota of the container."""
    import numba
    q = cpu_quota()
    if 'NUMBA_NUM_THREADS' not in os.environ and q is not None and q < numba.get_num_threads():
        numba.set_num_threads(q)
    return numba.get_num_threads()
