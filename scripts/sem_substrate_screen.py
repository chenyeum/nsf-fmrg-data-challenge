#!/usr/bin/env python
"""SEM substrate-morphology within-track screen (step 1 of SEM-as-input).

Tests the organizers' implied hypothesis (README: SEM characterizes
surrounding substrate morphology as an input) at the cheapest possible
level: do per-tile substrate roughness features co-vary with the local
track width *within* each dev track?

Granularity is TILE-LEVEL (n ~= 13 per track): the within-tile column-to-x
orientation has never been physically validated, so no sub-tile alignment
is assumed. Features come only from the anti-leakage substrate crops
(track band + margin excluded, Task-2 machinery).

Same decision protocol as the thermal screen (2026-07-16): a transferable
signal needs BOTH non-trivial magnitude AND a consistent sign across the
three dev tracks. If this table is flat, SEM does not enter the residual
GP and the multimodal question is closed at the feature level.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from nsf_fmrg_data import load_sem_tile
from preprocessing import (COMMON_X_END_MM, COMMON_X_START_MM, DEV_TRACKS,
                           _resolve_sem_band, _row_roughness, crop_sem_context,
                           sem_tile_paths)

SEM_DIR = REPO / 'data' / 'raw' / 'sem'
CACHE = REPO / 'processed_data' / 'features'
MIN_WINDOWS_PER_TILE = 5

FEATURES = ('rough_mean', 'rough_spread', 'rough_asym', 'int_mean', 'int_std')


def substrate_features(img, row_start, row_stop):
    """Roughness/intensity stats of the substrate (track band excluded)."""
    ctx = crop_sem_context(img, row_start, row_stop)
    top, bottom = ctx['top'].astype(np.float32), ctx['bottom'].astype(np.float32)
    r_top = _row_roughness(top) if top.shape[0] > 1 else np.array([np.nan])
    r_bot = _row_roughness(bottom) if bottom.shape[0] > 1 else np.array([np.nan])
    r_all = np.concatenate([r_top, r_bot])
    sub = np.concatenate([top.ravel(), bottom.ravel()])
    return {
        'rough_mean': float(np.nanmean(r_all)),
        'rough_spread': float(np.nanstd(r_all)),
        'rough_asym': float(abs(np.nanmean(r_top) - np.nanmean(r_bot))),
        'int_mean': float(sub.mean()),
        'int_std': float(sub.std()),
    }


def tile_index_for_x(x_mm, n_tiles):
    tw = (COMMON_X_END_MM - COMMON_X_START_MM) / n_tiles
    return int(min(max((COMMON_X_END_MM - x_mm) // tw + 1, 1), n_tiles))


def main():
    runs = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    dataset_run = sys.argv[1] if len(sys.argv) > 1 else runs[-1].name
    d = np.load(CACHE / f'{dataset_run}_thermal_v1.npz', allow_pickle=False)
    y, valid, track, x_mm = (d['width_mean_mm'], d['valid'].astype(bool),
                             d['track_id'], d['x_mm'])

    per_track = {}   # track -> (feature_matrix, tile_median_width, tile_ids)
    for t in DEV_TRACKS:
        paths = sem_tile_paths(SEM_DIR, t)
        n_tiles = len(paths)
        rows, widths, kept, borrowed = [], [], [], 0
        m_t = valid & (track == t) & np.isfinite(y)
        win_tiles = np.array([tile_index_for_x(x, n_tiles) for x in x_mm[m_t]])
        for ti in range(1, n_tiles + 1):
            w_tile = y[m_t][win_tiles == ti]
            if len(w_tile) < MIN_WINDOWS_PER_TILE:
                continue
            row_start, row_stop, src = _resolve_sem_band(t, ti, SEM_DIR, paths)
            if row_start is None:
                continue
            borrowed += int(src != ti)
            feats = substrate_features(load_sem_tile(paths[ti - 1]), row_start, row_stop)
            rows.append([feats[k] for k in FEATURES])
            widths.append(float(np.median(w_tile)))
            kept.append(ti)
        per_track[t] = (np.array(rows), np.array(widths), kept)
        print(f'track {t}: {len(kept)}/{n_tiles} tiles usable, '
              f'{borrowed} borrowed band(s)')

    print()
    print(f"{'feature':<14} " + "  ".join(f"t{t}: P / S" for t in DEV_TRACKS)
          + "   sign-consistent  pooled-demeaned-P (n)")
    for j, name in enumerate(FEATURES):
        per, pooled_f, pooled_w = {}, [], []
        for t in DEV_TRACKS:
            F, W, _ = per_track[t]
            per[t] = (stats.pearsonr(F[:, j], W)[0], stats.spearmanr(F[:, j], W)[0])
            pooled_f.append(F[:, j] - F[:, j].mean())
            pooled_w.append(W - W.mean())
        ps = [per[t][0] for t in DEV_TRACKS]
        consistent = (not np.any(np.isnan(ps))) and np.all(np.sign(ps) == np.sign(ps[0]))
        fp, wp = np.concatenate(pooled_f), np.concatenate(pooled_w)
        r_pool, p_pool = stats.pearsonr(fp, wp)
        cells = "  ".join(f"{per[t][0]:+.2f}/{per[t][1]:+.2f}" for t in DEV_TRACKS)
        print(f"{name:<14} {cells}   {'YES' if consistent else 'no':<15} "
              f"{r_pool:+.2f} p={p_pool:.2f} ({len(fp)})")


if __name__ == '__main__':
    main()
