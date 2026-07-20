#!/usr/bin/env python
"""SEM substrate screen v2: STRIP-LEVEL, using the official tile alignment.

Supersedes the tile-level screen's granularity limit. Official alignment
rule (user-relayed 2026-07-17):
  1. stitch tiles 01..N left-to-right using their real overlapping regions
     (~5%; estimated here per pair by normalized cross-correlation),
  2. no per-tile flip/rotation,
  3. fliplr the finished mosaic -> left-to-right runs 20 mm .. 100 mm,
     same direction as the height map (tile N at 20 mm, tile 01 at 100 mm).

Per dev track: stitch -> flip -> detect the track row band on the mosaic ->
for every valid width window, cut a +-STRIP_HALF_MM strip at its x, compute
substrate roughness/intensity features from the rows outside the band
(+margin), then the same within-track correlation protocol as the thermal
and tile-level screens (per-track Pearson/Spearman, cross-track sign
consistency, pooled demeaned correlation). n ~= 370-380 per track instead
of 13.

The mosaic's column->x mapping is linear from COMMON_X_START_MM to
COMMON_X_END_MM across the full stitched width (end tiles may physically
extend slightly beyond; accepted as approximation and noted here).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from nsf_fmrg_data import load_sem_tile
from preprocessing import (COMMON_X_END_MM, COMMON_X_START_MM, DEV_TRACKS,
                           SEM_BAND_MARGIN_PX, _row_roughness,
                           detect_track_row_band, sem_tile_paths)

SEM_DIR = REPO / 'data' / 'raw' / 'sem'
CACHE = REPO / 'processed_data' / 'features'
OVERLAP_RANGE_PX = (10, 123)      # ~1%..12% of a 1024-px tile
STRIP_HALF_MM = 0.25              # strip half-width around each window's x
FEATURES = ('rough_mean', 'rough_spread', 'rough_asym', 'int_mean', 'int_std')


NOMINAL_OVERLAP_PX = 51           # the official "~5%" of a 1024-px tile
MIN_MATCH_CORR = 0.3              # below this the estimate is noise -> nominal


def estimate_overlap(left_img, right_img):
    """Best overlap (px) between left tile's right edge and right tile's left
    edge, by Pearson correlation of the overlapping pixel blocks. Falls back
    to the official nominal ~5% when no candidate matches convincingly."""
    best_o, best_r = OVERLAP_RANGE_PX[0], -np.inf
    for o in range(*OVERLAP_RANGE_PX):
        a = left_img[:, -o:].astype(np.float32).ravel()
        b = right_img[:, :o].astype(np.float32).ravel()
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_o, best_r = o, r
    if best_r < MIN_MATCH_CORR:
        return NOMINAL_OVERLAP_PX, float(best_r)
    return best_o, float(best_r)


def build_mosaic(track_id):
    """Stitch tiles 01..N (no per-tile flip), then fliplr the whole mosaic."""
    paths = sem_tile_paths(SEM_DIR, track_id)
    tiles = [load_sem_tile(p) for p in paths]
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    parts, overlaps = [tiles[0]], []
    for left, right in zip(tiles[:-1], tiles[1:]):
        o, r = estimate_overlap(left, right)
        overlaps.append((o, float(r)))
        parts.append(right[:, o:])
    mosaic = np.hstack(parts)
    # Band rows on the full mosaic can fail when low-contrast tiles dilute the
    # smooth-row density, so detect per tile and take the median of the valid
    # detections (rows are untouched by horizontal stitching and fliplr).
    starts, stops = [], []
    for tile in tiles:
        row_start, row_stop, ok = detect_track_row_band(tile)
        if ok:
            starts.append(row_start)
            stops.append(row_stop)
    band = ((int(np.median(starts)), int(np.median(stops)))
            if starts else (None, None))
    return np.fliplr(mosaic), overlaps, band


def strip_features(mosaic, band, col_lo, col_hi):
    """Substrate features of mosaic[:, col_lo:col_hi], band rows excluded."""
    row_start, row_stop = band
    top = mosaic[:max(0, row_start - SEM_BAND_MARGIN_PX), col_lo:col_hi]
    bottom = mosaic[min(mosaic.shape[0], row_stop + SEM_BAND_MARGIN_PX):, col_lo:col_hi]
    r_top = _row_roughness(top) if top.shape[0] > 1 else np.array([np.nan])
    r_bot = _row_roughness(bottom) if bottom.shape[0] > 1 else np.array([np.nan])
    r_all = np.concatenate([r_top, r_bot])
    sub = np.concatenate([top.astype(np.float32).ravel(),
                          bottom.astype(np.float32).ravel()])
    return [float(np.nanmean(r_all)), float(np.nanstd(r_all)),
            float(abs(np.nanmean(r_top) - np.nanmean(r_bot))),
            float(sub.mean()), float(sub.std())]


def main():
    runs = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    dataset_run = sys.argv[1] if len(sys.argv) > 1 else runs[-1].name
    d = np.load(CACHE / f'{dataset_run}_thermal_v1.npz', allow_pickle=False)
    y, valid, track, x_mm = (d['width_mean_mm'], d['valid'].astype(bool),
                             d['track_id'], d['x_mm'])

    span_mm = COMMON_X_END_MM - COMMON_X_START_MM
    per_track = {}
    for t in DEV_TRACKS:
        mosaic, overlaps, band = build_mosaic(t)
        o_px = [o for o, _ in overlaps]
        o_r = [r for _, r in overlaps]
        if band[0] is None:
            print(f'track {t}: no tile with a valid band, skipping')
            continue
        px_per_mm = mosaic.shape[1] / span_mm
        m_t = valid & (track == t) & np.isfinite(y)
        feats, widths = [], []
        for xi, yi in zip(x_mm[m_t], y[m_t]):
            c = (xi - COMMON_X_START_MM) * px_per_mm
            lo = int(max(0, c - STRIP_HALF_MM * px_per_mm))
            hi = int(min(mosaic.shape[1], c + STRIP_HALF_MM * px_per_mm))
            if hi - lo < 5:
                continue
            feats.append(strip_features(mosaic, band, lo, hi))
            widths.append(yi)
        per_track[t] = (np.array(feats), np.array(widths))
        print(f'track {t}: mosaic {mosaic.shape[1]}px ({px_per_mm:.1f}px/mm), '
              f'band rows {band[0]}-{band[1]}, {len(widths)} strips, '
              f'overlap px median {int(np.median(o_px))} '
              f'(corr {min(o_r):.2f}-{max(o_r):.2f})')

    print()
    print(f"{'feature':<14} " + "  ".join(f"t{t}: P / S" for t in per_track)
          + "   sign-consistent  pooled-demeaned-P (n)")
    for j, name in enumerate(FEATURES):
        per, pooled_f, pooled_w = {}, [], []
        for t, (F, W) in per_track.items():
            per[t] = (stats.pearsonr(F[:, j], W)[0], stats.spearmanr(F[:, j], W)[0])
            pooled_f.append(F[:, j] - F[:, j].mean())
            pooled_w.append(W - W.mean())
        ps = [per[t][0] for t in per_track]
        consistent = (not np.any(np.isnan(ps))) and np.all(np.sign(ps) == np.sign(ps[0]))
        fp, wp = np.concatenate(pooled_f), np.concatenate(pooled_w)
        r_pool, p_pool = stats.pearsonr(fp, wp)
        cells = "  ".join(f"{per[t][0]:+.2f}/{per[t][1]:+.2f}" for t in per_track)
        print(f"{name:<14} {cells}   {'YES' if consistent else 'no':<15} "
              f"{r_pool:+.2f} p={p_pool:.3f} ({len(fp)})")


if __name__ == '__main__':
    main()
