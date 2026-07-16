#!/usr/bin/env python3
"""Cross-modality alignment self-check (never run before — see PREPROCESSING_PLAN.md).

Within-code consistency (thermal frame-center convention <-> height-map window)
was already verified by construction in build_track_samples. What that check
CANNOT catch: an independent physical-zero offset between the thermal camera's
and the profilometer's own calibration, since each instrument's x=20mm..100mm
is set by its own hardware, not a shared ruler.

Method: build one width-like signal per modality as a function of x, resample
both onto the same 0.2mm grid, and cross-correlate. If the two instruments
agree on where x=0 is, the peak should sit at lag ~0 (well under one frame,
0.2mm). A peak at nonzero lag means the modalities are shifted relative to
each other by that many frames.

Thermal-side proxy: per-frame hot-pixel fraction above (median + 3*MAD) of
that frame -- the laser-heated track should occupy more of the frame where
the track is wider, so this should track width_mean_mm even though it's not
a metric width itself. Same median/MAD-threshold idiom already used in
_column_track_boundary and the SEM band detector, not a new convention.

**Scale fix (v2)**: a full 400x400 frame is a 5.6mm x 5.6mm field of view
(14 um/pixel, per README/paper), but the frame only advances 0.2mm between
captures -- adjacent frames overlap ~96%. Computing hot-fraction over the
whole frame smears the signal over ~5.6mm of physical extent, a completely
different scale than the height-map side's 0.2mm window, and the first
version of this script (lag0 corr 0.02-0.13, best-lag sign/magnitude
inconsistent across tracks: -0.6/-1.8/-0.4/+1.6mm) was consistent with that
mismatch, not with a real registration offset. Fix: crop to a central
patch (CROP_FRAC of each side) before thresholding -- the paper states the
melt pool sits "in a smaller central region" of the frame, so a central
crop is a reasonable stand-in for "the local region at this frame's x",
without needing to know which pixel axis is the scan direction (not
documented, so an isotropic center crop is used rather than a directional
strip along a guessed axis).
"""
CROP_FRAC = 0.15  # central 15% per side (~60x60px, ~0.84mm) of the 400x400 frame
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from nsf_fmrg_data import extract_final_thermal_frames
from preprocessing import DEV_TRACKS, HELD_OUT_TEST_TRACK

DATASET_RUN = REPO / 'processed_data' / 'datasets' / '20260715_145326'
MAX_LAG_FRAMES = 10  # +/- 2mm search window


def thermal_hot_fraction(frames, crop_frac=CROP_FRAC):
    frames = frames.astype(np.float64)
    h, w = frames.shape[1:]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    frames = frames[:, ch:h - ch, cw:w - cw]
    med = np.median(frames, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(frames - med), axis=(1, 2), keepdims=True)
    thresh = med + 3.0 * 1.4826 * np.maximum(mad, 1e-9)
    return (frames > thresh).mean(axis=(1, 2))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - np.nanmean(a)) / np.nanstd(a)


def best_lag(thermal_sig, height_sig, max_lag):
    """Return (lag_in_frames, correlation) maximizing corr(thermal, height shifted by lag)."""
    n = len(thermal_sig)
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = thermal_sig[lag:], height_sig[:n - lag]
        else:
            a, b = thermal_sig[:n + lag], height_sig[-lag:]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 20:
            continue
        corr = np.corrcoef(a[mask], b[mask])[0, 1]
        if corr > best[1]:
            best = (lag, corr)
    return best


def check_track(track_id, thermal_dir, height_dir):
    thermal = extract_final_thermal_frames(thermal_dir, track_id)
    hot_frac = thermal_hot_fraction(thermal['frames'])

    with open(DATASET_RUN / f'track_{track_id}_samples.pkl', 'rb') as f:
        rows = pickle.load(f)
    row_by_frame = {r['frame_index']: r for r in rows}

    # align both signals onto the sample table's frame_index (already the
    # common index into thermal['frames'], per build_track_samples)
    idx = sorted(row_by_frame)
    height_sig = np.array([row_by_frame[t]['width_mean_mm'] for t in idx])
    thermal_sig = hot_frac[idx]

    finite = np.isfinite(height_sig)
    corr_lag0 = np.corrcoef(zscore(thermal_sig)[finite], zscore(height_sig)[finite])[0, 1]
    lag, corr = best_lag(zscore(thermal_sig), zscore(height_sig), MAX_LAG_FRAMES)
    print(f'track {track_id:>2}: lag0 corr={corr_lag0:+.3f}  '
          f'best lag={lag:+d} frames ({lag * 0.2:+.2f}mm) corr={corr:+.3f}')


if __name__ == '__main__':
    thermal_dir = REPO / 'data' / 'raw' / 'thermal'
    height_dir = REPO / 'data' / 'raw' / 'height_maps'
    for track_id in (*DEV_TRACKS, HELD_OUT_TEST_TRACK):
        check_track(track_id, thermal_dir, height_dir)
