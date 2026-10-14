#!/usr/bin/env python3
"""Exact all-integer GGL search, with shared draws and restartable size shards.

Prepare the original 5,000 within-cell draws once; scan disjoint inclusive size
ranges in separate processes; then merge. A shard searches in descending order
and may stop at its first nonrejection. Merge verifies the requested intervals
cover every size 1,...,n without gaps or overlap. Thus no monotonicity of P_k is
assumed and no sparse-grid certificate is used to limit the search.

Example (NUMBA_NUM_THREADS=1 and single-thread BLAS for each scan process):
  run_out="results/reproductions/run_01/real_data/ggl"
  python runners/run_ggl_integer.py prepare --data-dir data/raw --shared "$run_out/shared"
  python runners/run_ggl_integer.py scan --shared "$run_out/shared" --lo 1 --hi 60000 --out "$run_out/parts/part1"
  python runners/run_ggl_integer.py scan --shared "$run_out/shared" --lo 60001 --hi 119999 --out "$run_out/parts/part2"
  python runners/run_ggl_integer.py merge --shared "$run_out/shared" --parts "$run_out/parts/part1" "$run_out/parts/part2" --out "$run_out/summary"

Draw generation, integer turnout scaling, age order and p-value ties are
identical to run_real_data.py. Scan reuses realdata.int_window_counts in small
batches, avoiding the quadratic-size count arrays of a single full-grid call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from zeroreach import realdata as RD
from zeroreach.display import hist_pieces
from zeroreach.randomization import permutations_within_cells
from zeroreach.utils import environment, write_json

M, SEED, ALPHA = 5000, 20260526, 0.05
TIE_RULES = ('A_primary', 'B_reverse', 'C_household_size', 'D_hash1', 'E_hash2')
SEEDS = (20260526, 20260527, 20260528)


def _ggl_order(sub, tie_rule):
    """The five ordering rules already used in run_real_data.py's robustness table."""
    cell, hh, age = (sub[c].to_numpy() for c in ('cell_id', 'hh_id', 'age_key'))
    def md5_key(k):
        return np.array([int.from_bytes(hashlib.md5(f'{int(c)}|{int(h)}|ggl_tie_{k}'.encode()).digest()[:8], 'big')
                         for c, h in zip(cell, hh)], dtype=np.uint64)
    keys = {'A_primary': lambda: (hh, cell, age),
            'B_reverse': lambda: (-hh, -cell, age),
            'C_household_size': lambda: (hh, cell, sub.members.to_numpy(), age),
            'D_hash1': lambda: (md5_key(1), age),
            'E_hash2': lambda: (md5_key(2), age)}
    if tie_rule not in keys:
        raise ValueError('unknown tie rule: ' + tie_rule)
    return np.lexsort(keys[tie_rule]())


def _reorder_draws(draws, source_order, target_order):
    """Map already-generated rows between orders; preserve every assignment exactly."""
    inverse = np.empty(len(source_order), dtype=np.int64)
    inverse[source_order] = np.arange(len(source_order))
    return np.ascontiguousarray(draws[:, inverse[target_order]])


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _save(path, array):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def prepare(data_dir, shared, Mloc=M, seed=SEED, tie_rule='A_primary', reuse_shared=None):
    shared = Path(shared)
    source = Path(data_dir) / 'ggl' / 'ggl_individuals.dta'
    ties = 'age, then cell id, then household id' if tie_rule == 'A_primary' else tie_rule
    identity = dict(source_sha256=_digest(source), M=int(Mloc), seed=int(seed), alpha=ALPHA, ties=ties)
    manifest_file = shared / 'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError('existing prepared data have different inputs or parameters')
        for name, sha in manifest['sha256'].items():
            if _digest(shared / name) != sha:
                raise ValueError('prepared file checksum mismatch: ' + name)
        print(json.dumps(dict(event='prepared_data_reused', **identity)), flush=True)
        return manifest
    t0 = time.monotonic()
    sub, L = RD.load_ggl(source)
    cell, hh, age = (sub[c].to_numpy() for c in ('cell_id', 'hh_id', 'age_key'))
    order = _ggl_order(sub, tie_rule)
    shared.mkdir(parents=True, exist_ok=True)
    if reuse_shared is None:
        print(json.dumps(dict(event='generating_original_draws', n=len(sub), M=Mloc, seed=seed)), flush=True)
        draws = permutations_within_cells(cell, sub.z.to_numpy(), Mloc, seed)
        if not np.array_equal(draws[0], sub.z.to_numpy()):
            raise ValueError('observed assignment mismatch')
        ordered_draws = np.ascontiguousarray(draws[1:, order])
    else:
        reuse_shared = Path(reuse_shared)
        original = json.loads((reuse_shared / 'manifest.json').read_text())
        for k in ('source_sha256', 'M', 'seed'):
            if original[k] != identity[k]:
                raise ValueError('cannot reuse draws with different source, M, or seed')
        for name in ('draws.npy', 'order.npy', 'z.npy'):
            if _digest(reuse_shared / name) != original['sha256'][name]:
                raise ValueError('reused array checksum mismatch: ' + name)
        original_order = np.load(reuse_shared / 'order.npy', allow_pickle=False)
        original_z = np.load(reuse_shared / 'z.npy', allow_pickle=False)
        if not np.array_equal(original_z, sub.z.to_numpy()[original_order]):
            raise ValueError('reused draws have different observed assignments')
        draws = np.load(reuse_shared / 'draws.npy', mmap_mode='r', allow_pickle=False)
        ordered_draws = _reorder_draws(draws, original_order, order)
        print(json.dumps(dict(event='reordered_existing_draws', from_shared=str(reuse_shared), tie_rule=tie_rule)), flush=True)
    arrays = dict(y=sub.yint.to_numpy()[order], z=sub.z.to_numpy()[order],
                  age=age[order] / L, rate=(sub.vsum / sub.members).to_numpy()[order],
                  order=order)
    for name, value in arrays.items():
        _save(shared / (name + '.npy'), value)
    _save(shared / 'draws.npy', ordered_draws)
    names = [name + '.npy' for name in arrays] + ['draws.npy']
    manifest = dict(**identity, n=len(sub), n1=int(sub.z.sum()), cells=int(sub.cell_id.nunique()),
                    outcome_integer_scale=L, source=str(source), tie_rule=tie_rule,
                    draws_reused_from=str(reuse_shared) if reuse_shared is not None else None,
                    environment=environment(),
                    sha256={name: _digest(shared / name) for name in names},
                    prepare_seconds=time.monotonic() - t0)
    write_json(manifest_file, manifest)
    print(json.dumps(dict(event='prepared', n=len(sub), seconds=manifest['prepare_seconds'])), flush=True)
    return manifest


def _load(shared):
    shared = Path(shared)
    manifest = json.loads((shared / 'manifest.json').read_text())
    arrays = {name: np.load(shared / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('y', 'z', 'draws')}
    n, Mloc = manifest['n'], manifest['M']
    if arrays['draws'].shape != (Mloc, n) or any(arrays[name].shape != (n,) for name in ('y', 'z')):
        raise ValueError('prepared array dimensions do not match manifest')
    return manifest, arrays


def _identity(manifest):
    return {k: manifest[k] for k in ('n', 'M', 'seed', 'alpha', 'ties', 'source_sha256', 'sha256')}


def scan(shared, out, lo, hi, batch=32, nchunk=1):
    manifest, arr = _load(shared)
    n, Mloc = manifest['n'], manifest['M']
    if not (1 <= lo <= hi <= n) or batch < 1 or nchunk < 1:
        raise ValueError('invalid range, batch size, or nchunk')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / 'checkpoint.json'
    identity = dict(**_identity(manifest), lo=int(lo), hi=int(hi))
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if any(state.get(k) != v for k, v in identity.items()):
            raise ValueError('checkpoint belongs to different data or range')
    else:
        state = dict(**identity, next_size=hi, bound=0, complete=False, evaluated=[], seconds=0.)
    if state['complete']:
        print(json.dumps(dict(event='shard_already_complete', lo=lo, hi=hi, bound=state['bound'])), flush=True)
        return state
    while state['next_size'] >= lo:
        t0 = time.monotonic()
        start = state['next_size']
        grid = np.arange(start, max(lo - 1, start - batch), -1, dtype=np.int64)
        counts = RD.int_window_counts(arr['y'], arr['z'], arr['draws'], grid, nchunk)
        accepted = False
        for j, r0 in enumerate(grid):
            r = int(r0)
            max_count = int(counts[j, :n - r + 1].max())
            p = (max_count + 1.) / (Mloc + 1.)
            state['evaluated'].append(dict(r=r, max_count=max_count, P=p))
            state['next_size'] = r - 1
            if p > manifest['alpha']:
                state['bound'] = r
                state['complete'] = True
                _save(out / 'bound_counts.npy', counts[j, :n - r + 1].copy())
                accepted = True
                break
        state['seconds'] += time.monotonic() - t0
        if state['next_size'] < lo:
            state['complete'] = True
        # The counts file is written before the atomic checkpoint records completion.
        write_json(state_file, state)
        print(json.dumps(dict(event='batch_complete', lo=lo, hi=hi, next_size=state['next_size'],
                              bound=state['bound'], seconds=state['seconds'])), flush=True)
        if accepted:
            break
    return state


def merge(shared, parts, out):
    shared, out = Path(shared), Path(out)
    manifest, _ = _load(shared)
    states = [(Path(p), json.loads((Path(p) / 'checkpoint.json').read_text())) for p in parts]
    states.sort(key=lambda x: x[1]['lo'])
    expected = 1
    for path, state in states:
        if not state['complete'] or any(state.get(k) != v for k, v in _identity(manifest).items()):
            raise ValueError('incomplete or mismatched shard: ' + str(path))
        if state['lo'] != expected:
            raise ValueError('requested integer ranges have a gap or overlap at ' + str(expected))
        expected = state['hi'] + 1
        visited = [row['r'] for row in state['evaluated']]
        last = state['bound'] or state['lo']
        if visited != list(range(state['hi'], last - 1, -1)):
            raise ValueError('evaluated sizes are not a contiguous descending path')
        for row in state['evaluated']:
            if row['P'] != (row['max_count'] + 1.) / (manifest['M'] + 1.):
                raise ValueError('invalid p-value in checkpoint')
            if (row['P'] > manifest['alpha']) != (row['r'] == state['bound']):
                raise ValueError('invalid stopping decision in checkpoint')
    if expected != manifest['n'] + 1:
        raise ValueError('requested integer ranges do not reach n')
    bound = max((s['bound'] for _, s in states), default=0)
    n = manifest['n']
    mask = np.zeros(n, dtype=bool)
    windows = 0
    if bound:
        owner, _ = next((p, s) for p, s in states if s['bound'] == bound)
        counts = np.load(owner / 'bound_counts.npy', allow_pickle=False)
        if counts.shape != (n - bound + 1,):
            raise ValueError('invalid bound counts')
        starts = np.flatnonzero((counts + 1.) / (manifest['M'] + 1.) > manifest['alpha'])
        for s in starts:
            mask[s:s + bound] = True
        windows = len(starts)
    W, rate, z = (np.load(shared / (name + '.npy'), allow_pickle=False) for name in ('age', 'rate', 'z'))
    differences = {}
    for name, sel in (('inside', mask), ('outside', ~mask), ('all', np.ones(n, bool))):
        treated, control = sel & (z == 1), sel & (z == 0)
        differences[name] = dict(n=int(sel.sum()), diff=float(rate[treated].mean() - rate[control].mean())
                                if treated.any() and control.any() else None)
    summary = dict(dataset='GGL', covariate='household mean age in 2006 (years)', **_identity(manifest),
                   search='all integer sizes 1,...,n; descending exhaustive within disjoint shards',
                   n1=manifest['n1'], cells=manifest['cells'], bound=bound, bound_fraction=bound / n,
                   first_rejected=bound + 1 if bound < n else None, certificate=bound,
                   nonrejected_windows=windows, covered_units=int(mask.sum()),
                   covered_age_range=[float(W[mask].min()), float(W[mask].max())] if mask.any() else None,
                   turnout_difference=differences,
                   shard_intervals=[[s['lo'], s['hi']] for _, s in states],
                   aggregate_scan_seconds=sum(s['seconds'] for _, s in states))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)
    pd.DataFrame([row for _, s in states for row in s['evaluated']]).sort_values('r').to_csv(
        out / 'pvalue_path.csv', index=False)
    pieces = hist_pieces(W, mask, np.arange(np.floor(W.min()), np.floor(W.max()) + 2))
    pd.DataFrame(pieces, columns=['left', 'width', 'height', 'covered']).to_csv(out / 'display_pieces.csv', index=False)
    (out / 'local').mkdir(exist_ok=True)
    _save(out / 'local/covered_mask.npy', mask)
    print(json.dumps(dict(event='merged', bound=bound, covered_units=int(mask.sum()))), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--shared', required=True)
    p.add_argument('--M', type=int, default=M, help='change only for explicit sensitivity/benchmark runs')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--tie-rule', choices=TIE_RULES, default='A_primary')
    p.add_argument('--reuse-shared', help='prepared same-seed assignments to reorder without resampling')
    p = sub.add_parser('scan')
    p.add_argument('--shared', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lo', type=int, required=True)
    p.add_argument('--hi', type=int, required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--nchunk', type=int, default=1)
    p = sub.add_parser('merge')
    p.add_argument('--shared', required=True)
    p.add_argument('--parts', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.data_dir, a.shared, a.M, a.seed, a.tie_rule, a.reuse_shared)
    elif a.command == 'scan':
        scan(a.shared, a.out, a.lo, a.hi, a.batch, a.nchunk)
    else:
        merge(a.shared, a.parts, a.out)


if __name__ == '__main__':
    main()
