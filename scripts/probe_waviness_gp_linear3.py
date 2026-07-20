#!/usr/bin/env python
"""Fair cross-check: GP on the SAME 3 linear3 features BayesianRidge used
(probe_waviness_signal.py), not the 18-dim set (probe_waviness_gp.py).

probe_waviness_gp.py's 18-feat+x_mm GP failed on width itself (ratio~1.0)
even though width is known to have real signal (linear3 BayesianRidge gets
0.663) -- a known GP limitation (RBF mean-reversion under track-to-track
extrapolation, see run_gp_baseline.py), not evidence against the waviness
finding. This version uses the identical 3-feature input BayesianRidge
used, so a difference in outcome is attributable to the model, not the
feature set.

Quick probe only: no track-21 prediction, no provenance file.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from preprocessing import DEV_TRACKS, loto_cv_splits
from thermal_features import FEATURE_NAMES, window_features

DATASET_RUN = sorted((REPO / 'processed_data' / 'datasets').iterdir())[-1].name
RUN_DIR = REPO / 'processed_data' / 'datasets' / DATASET_RUN
SEED = 0
SMOOTH_WINDOW_MM = 5.0

_N = FEATURE_NAMES[:18]
_J_AREA, _J_PEAK, _J_TAIL = (_N.index('mean_area_1500'), _N.index('mean_peak'),
                             _N.index('mean_tail_len'))


def linear3(A):
    return np.column_stack([np.sqrt(A[:, _J_AREA]), A[:, _J_PEAK], A[:, _J_TAIL]])


def smooth_by_x(x, w, window_mm):
    out = np.empty_like(w)
    for i, xi in enumerate(x):
        m = np.abs(x - xi) <= window_mm / 2
        out[i] = w[m].mean()
    return out


def fit_predict_gp(Xtr, ytr, Xva):
    y_mean, y_std = ytr.mean(), ytr.std()
    ytr_z = (ytr - y_mean) / y_std
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=np.ones(Xtr.shape[1]), length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(1e-2, (1e-8, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                  random_state=SEED, normalize_y=False)
    gp.fit(Xtr, ytr_z)
    return gp.predict(Xva) * y_std + y_mean


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

    X18 = np.array(feats)
    width = np.array(width)
    x_mm = np.array(x_mm)
    track = np.array(track)
    F = linear3(X18)  # 3 dims, same as BayesianRidge probe -- no x_mm

    wavy = np.empty_like(width)
    for track_id in DEV_TRACKS:
        m = track == track_id
        wavy[m] = smooth_by_x(x_mm[m], width[m], SMOOTH_WINDOW_MM)

    print(f'input dims: {F.shape[1]} (sqrt(mean_area_1500), mean_peak, mean_tail_len)\n')
    print(f"{'target':<10} {'val_track':<10} {'n':>5} {'const_MAE':>10} "
          f"{'gp_MAE':>10} {'ratio':>7}")
    for name, y in (('width', width), ('waviness', wavy)):
        ratios = []
        for train_tracks, val_track in loto_cv_splits():
            tr = np.isin(track, train_tracks)
            va = track == val_track
            xscaler = StandardScaler().fit(F[tr])
            Ftr, Fva = xscaler.transform(F[tr]), xscaler.transform(F[va])
            pred = fit_predict_gp(Ftr, y[tr], Fva)
            const_mae = np.abs(y[va] - y[tr].mean()).mean()
            gp_mae = np.abs(y[va] - pred).mean()
            ratios.append(gp_mae / const_mae)
            print(f'{name:<10} {val_track:<10} {int(va.sum()):>5} {const_mae:>10.4f} '
                  f'{gp_mae:>10.4f} {gp_mae / const_mae:>7.2f}')
        print(f'{name:<10} pooled mean model/const ratio: {np.mean(ratios):.3f}\n')
    print('reference (BayesianRidge, same 3 feats): width 0.663 | waviness 0.593')


if __name__ == '__main__':
    main()
