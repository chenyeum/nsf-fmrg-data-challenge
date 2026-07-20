#!/usr/bin/env python
"""Probe: can thermal features predict the WAVINESS (smoothed) component of
local width, better (relatively) than they predict raw width (LOTO)?

Waviness here = width_mean_mm smoothed along x within each track with a
5 mm centered physical-distance window -- the same smoothing used in the
2026-07-17 EDA note ("within-track width variance is 80-90% high-frequency;
5mm smoothing keeps 9-18%"). It is a function of width alone (a difference
of two boundary estimates), so unlike boundary/centerline it is
registration-safe.

Quick probe only: same linear3 features + BayesianRidge + LOTO as
run_linear_baseline.py, no track-21 prediction, no provenance file. Checks
whether the model/const MAE ratio improves once the unlearnable
high-frequency noise is smoothed out of the target -- i.e. whether there is
genuine extra low-frequency signal here, not just a smaller target variance.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from preprocessing import DEV_TRACKS, loto_cv_splits
from thermal_features import FEATURE_NAMES, window_features

DATASET_RUN = sorted((REPO / 'processed_data' / 'datasets').iterdir())[-1].name
RUN_DIR = REPO / 'processed_data' / 'datasets' / DATASET_RUN
SMOOTH_WINDOW_MM = 5.0

_N = FEATURE_NAMES[:18]
_J_AREA, _J_PEAK, _J_TAIL = (_N.index('mean_area_1500'), _N.index('mean_peak'),
                             _N.index('mean_tail_len'))


def linear3(A):
    return np.column_stack([np.sqrt(A[:, _J_AREA]), A[:, _J_PEAK], A[:, _J_TAIL]])


def smooth_by_x(x, w, window_mm):
    """Centered physical-distance moving average of w(x) (irregular spacing)."""
    out = np.empty_like(w)
    for i, xi in enumerate(x):
        m = np.abs(x - xi) <= window_mm / 2
        out[i] = w[m].mean()
    return out


def main():
    feats, width, x_mm, track = [], [], [], []
    for track_id in DEV_TRACKS:
        with open(RUN_DIR / f'track_{track_id}_samples.pkl', 'rb') as f:
            rows = pickle.load(f)
        for r in rows:
            if not r['valid']:
                continue
            feats.append(window_features(r['thermal_window']))
            width.append(r['width_mean_mm'])
            x_mm.append(r['x_mm'])
            track.append(track_id)
        del rows

    X = np.array(feats)
    width = np.array(width)
    x_mm = np.array(x_mm)
    track = np.array(track)

    wavy = np.empty_like(width)
    for track_id in DEV_TRACKS:
        m = track == track_id
        wavy[m] = smooth_by_x(x_mm[m], width[m], SMOOTH_WINDOW_MM)
        print(f'track {track_id}: n={m.sum()}, raw var={width[m].var():.5f}, '
              f'smoothed var={wavy[m].var():.5f}, '
              f'variance retained={wavy[m].var() / width[m].var():.1%}')

    F = linear3(X)
    print(f"\n{'target':<10} {'val_track':<10} {'n':>5} {'const_MAE':>10} "
          f"{'model_MAE':>10} {'ratio':>7}")
    for name, y in (('width', width), ('waviness', wavy)):
        ratios = []
        for train_tracks, val_track in loto_cv_splits():
            tr = np.isin(track, train_tracks)
            va = track == val_track
            scaler = StandardScaler().fit(F[tr])
            model = BayesianRidge().fit(scaler.transform(F[tr]), y[tr])
            pred = model.predict(scaler.transform(F[va]))
            const_mae = np.abs(y[va] - y[tr].mean()).mean()
            model_mae = np.abs(y[va] - pred).mean()
            ratios.append(model_mae / const_mae)
            print(f'{name:<10} {val_track:<10} {int(va.sum()):>5} {const_mae:>10.4f} '
                  f'{model_mae:>10.4f} {model_mae / const_mae:>7.2f}')
        print(f'{name:<10} pooled mean model/const ratio: {np.mean(ratios):.3f}\n')


if __name__ == '__main__':
    main()
