#!/usr/bin/env python
"""Probe: can thermal features predict edge_roughness_mm (LOTO)?

edge_roughness_mm is already extracted by local_width_stats_at_window
(average of left/right boundary std across VALLEY_N_SUBBLOCKS sub-blocks per
window) but has never been used as a modeling target -- only as an input to
GP noise variance. It is registration-safe (a magnitude, not an absolute
position), so unlike boundary/centerline it is a legitimate candidate
richer-output.

Quick probe only: same linear3 features + BayesianRidge + LOTO as
run_linear_baseline.py, no track-21 prediction, no provenance file. Just
answers "is there signal worth building out" before investing in a full
pipeline.
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

_N = FEATURE_NAMES[:18]
_J_AREA, _J_PEAK, _J_TAIL = (_N.index('mean_area_1500'), _N.index('mean_peak'),
                             _N.index('mean_tail_len'))


def linear3(A):
    return np.column_stack([np.sqrt(A[:, _J_AREA]), A[:, _J_PEAK], A[:, _J_TAIL]])


def main():
    feats, rough, track = [], [], []
    for track_id in DEV_TRACKS:
        with open(RUN_DIR / f'track_{track_id}_samples.pkl', 'rb') as f:
            rows = pickle.load(f)
        for r in rows:
            if not r['valid'] or not np.isfinite(r['edge_roughness_mm']):
                continue
            feats.append(window_features(r['thermal_window']))
            rough.append(r['edge_roughness_mm'])
            track.append(track_id)
        del rows

    X = np.array(feats)
    y = np.array(rough)
    track = np.array(track)
    print(f'n={len(y)} valid+finite-roughness windows across {DEV_TRACKS}')
    print(f'roughness mm: mean={y.mean():.4f} std={y.std():.4f} '
          f'range=[{y.min():.4f}, {y.max():.4f}]')

    F = linear3(X)
    print(f"\n{'val_track':<10} {'n':>5} {'const_MAE':>10} {'model_MAE':>10}")
    maes_const, maes_model = [], []
    for train_tracks, val_track in loto_cv_splits():
        tr = np.isin(track, train_tracks)
        va = track == val_track
        scaler = StandardScaler().fit(F[tr])
        model = BayesianRidge().fit(scaler.transform(F[tr]), y[tr])
        pred = model.predict(scaler.transform(F[va]))
        const_mae = np.abs(y[va] - y[tr].mean()).mean()
        model_mae = np.abs(y[va] - pred).mean()
        maes_const.append(const_mae)
        maes_model.append(model_mae)
        print(f'{val_track:<10} {int(va.sum()):>5} {const_mae:>10.4f} {model_mae:>10.4f}')
    print(f"\npooled mean {'const_MAE':>10}: {np.mean(maes_const):.4f}")
    print(f"pooled mean {'model_MAE':>10}: {np.mean(maes_model):.4f}")


if __name__ == '__main__':
    main()
