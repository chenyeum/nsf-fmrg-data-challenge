#!/usr/bin/env python
"""Cross-check: does a GP (ARD, ~run_gp_baseline.py recipe) confirm the
BayesianRidge finding that the waviness (5mm-smoothed width) target is
relatively more predictable than raw width (probe_waviness_signal.py)?

Motivation: probe_roughness_gp.py just showed BayesianRidge can produce a
misleadingly bad result (looked like real overfitting collapse on track14)
where a flexible GP honestly found no signal (ratio exactly 1.00, all
length scales pinned at the upper bound). Before trusting the waviness
"improvement" found with BayesianRidge, check it holds up with a model
that doesn't have BayesianRidge's fixed-form bias -- same 18 thermal feats
+ x_mm input set as run_gp_baseline.py / probe_roughness_gp.py.

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
from thermal_features import window_features

DATASET_RUN = sorted((REPO / 'processed_data' / 'datasets').iterdir())[-1].name
RUN_DIR = REPO / 'processed_data' / 'datasets' / DATASET_RUN
SEED = 0
SMOOTH_WINDOW_MM = 5.0


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
    X = np.column_stack([X18, x_mm])  # 18 thermal feats + x_mm

    wavy = np.empty_like(width)
    for track_id in DEV_TRACKS:
        m = track == track_id
        wavy[m] = smooth_by_x(x_mm[m], width[m], SMOOTH_WINDOW_MM)

    print(f"{'target':<10} {'val_track':<10} {'n':>5} {'const_MAE':>10} "
          f"{'gp_MAE':>10} {'ratio':>7}")
    for name, y in (('width', width), ('waviness', wavy)):
        ratios = []
        for train_tracks, val_track in loto_cv_splits():
            tr = np.isin(track, train_tracks)
            va = track == val_track
            xscaler = StandardScaler().fit(X[tr])
            Xtr, Xva = xscaler.transform(X[tr]), xscaler.transform(X[va])
            pred = fit_predict_gp(Xtr, y[tr], Xva)
            const_mae = np.abs(y[va] - y[tr].mean()).mean()
            gp_mae = np.abs(y[va] - pred).mean()
            ratios.append(gp_mae / const_mae)
            print(f'{name:<10} {val_track:<10} {int(va.sum()):>5} {const_mae:>10.4f} '
                  f'{gp_mae:>10.4f} {gp_mae / const_mae:>7.2f}')
        print(f'{name:<10} pooled mean model/const ratio: {np.mean(ratios):.3f}\n')


if __name__ == '__main__':
    main()
