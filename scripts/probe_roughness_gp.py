#!/usr/bin/env python
"""Probe: does a GP (flexible, ARD, ~same recipe as run_gp_baseline.py) do
any better than BayesianRidge at predicting edge_roughness_mm?

probe_roughness_signal.py already found BayesianRidge+linear3 fails badly
(track14 model_MAE ~5x worse than the constant baseline) -- the working
hypothesis is that edge_roughness_mm is dominated by its own measurement
noise (a std computed from <=5 sub-block boundary estimates per window), so
no model, however flexible, should find a stable relationship. This probe
tests that hypothesis directly with a more flexible model + all 18 hand
features + x_mm (same input set run_gp_baseline.py uses for width).

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


def main():
    feats, rough, x_mm, track = [], [], [], []
    for track_id in DEV_TRACKS:
        with open(RUN_DIR / f'track_{track_id}_samples.pkl', 'rb') as f:
            rows = pickle.load(f)
        for r in rows:
            if not r['valid'] or not np.isfinite(r['edge_roughness_mm']):
                continue
            feats.append(window_features(r['thermal_window']))
            rough.append(r['edge_roughness_mm'])
            x_mm.append(r['x_mm'])
            track.append(track_id)
        del rows

    X18 = np.array(feats)
    X = np.column_stack([X18, np.array(x_mm)])  # 18 thermal feats + x_mm, like run_gp_baseline.py
    y = np.array(rough)
    track = np.array(track)
    print(f'n={len(y)}, {X.shape[1]} input dims (18 thermal feats + x_mm)')

    print(f"\n{'val_track':<10} {'n':>5} {'const_MAE':>10} {'gp_MAE':>10} {'ratio':>7}")
    ratios = []
    for train_tracks, val_track in loto_cv_splits():
        tr = np.isin(track, train_tracks)
        va = track == val_track
        xscaler = StandardScaler().fit(X[tr])
        Xtr, Xva = xscaler.transform(X[tr]), xscaler.transform(X[va])
        y_mean, y_std = y[tr].mean(), y[tr].std()
        ytr = (y[tr] - y_mean) / y_std

        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=np.ones(Xtr.shape[1]), length_scale_bounds=(1e-2, 1e3))
                  + WhiteKernel(1e-2, (1e-8, 1e1)))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                      random_state=SEED, normalize_y=False)
        gp.fit(Xtr, ytr)
        pred = gp.predict(Xva) * y_std + y_mean

        const_mae = np.abs(y[va] - y[tr].mean()).mean()
        gp_mae = np.abs(y[va] - pred).mean()
        ratios.append(gp_mae / const_mae)
        print(f'{val_track:<10} {int(va.sum()):>5} {const_mae:>10.4f} {gp_mae:>10.4f} '
              f'{gp_mae / const_mae:>7.2f}')
    print(f'\npooled mean model/const ratio: {np.mean(ratios):.3f}  '
          f'(BayesianRidge+linear3 reference: 2.13)')


if __name__ == '__main__':
    main()
