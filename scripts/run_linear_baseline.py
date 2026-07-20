#!/usr/bin/env python
"""Melt-pool linear model (BayesianRidge) — primary candidate for track 21.

Discovery chain (2026-07-17): the CVAE run's fold-3 anomaly revealed that
thermal features carry BETWEEN-track level information which the anchor
RBF-GP could not use (RBF extrapolation reverts to the train mean). A linear
map extrapolates along the feature axis instead, and needs NO power mapping.
(The reversed power map was organizer-confirmed later on 2026-07-17, so this
independence is no longer a risk argument — but it remains a genuine second
evidence route: thermal-only and power-law predictions corroborate each
other on track 21.)

Variants (all BayesianRidge, train-fold-only standardization):
  linear3 [PRIMARY] - sqrt(mean_area_1500), mean_peak, mean_tail_len.
      Physics-motivated monotone features: melt-pool linear scale, peak
      intensity, cooling-tail length. Best LOTO MAE of the screened set and
      passes the label-free SEM external check (ratio 1.20, at the dev edge).
  area1d            - raw mean_area_1500 only. Simplest passer, most robust
      external check (ratio 1.04).
  full18            - all 18 aggregates. NEGATIVE CONTROL: good LOTO but
      collinear mixed-sign coefficients explode under far extrapolation
      (t21 ratio 3.04) — kept to document why feature restriction matters.

Selection honesty note (recorded in provenance): five feature sets were
screened post-hoc on 2026-07-17 against two independent criteria (LOTO MAE,
label-free SEM check). The screen is physics-constrained but IS a post-hoc
selection; stated in the report.

Per variant: LOTO with gaussian metrics + cross-fold empirical variance
inflation (rule of run_gp_calib_inflation.py), then a final 3-dev-track fit
predicting every track-21 window from its THERMAL INPUTS ONLY, compared to
the track-21 SEM band width (0.379 mm; dev width/SEM ratios ran 0.94-1.20).

Track-21 access: `thermal_features.load_samples` refuses track 21 by design
(.CLAUDE.md section 3). Stage 2 reads track_21_samples.pkl directly,
USER-SANCTIONED 2026-07-17, input side only (thermal_window/x_mm/
frame_index); width/validity columns are never read — predictions are made
for every window. Per-window primary predictions are saved for the future
predict_track21 submission script.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from run_gp_baseline import X_RANGE_T21, gaussian_metrics
from temporal_cnn import mae_mm
from thermal_features import FEATURE_NAMES, window_features
from train_temporal_cnn import config_hash, git_provenance

SEED = 0
SEM_BAND_T21_MM = 0.379             # square-pixel assumption, ~6 um/px
DEV_WIDTH_SEM_RATIO = (0.94, 1.20)  # dev-track width/SEM-band ratio range
PRIMARY = 'linear3'

_N = FEATURE_NAMES[:18]
_J_AREA, _J_PEAK, _J_TAIL = (_N.index('mean_area_1500'), _N.index('mean_peak'),
                             _N.index('mean_tail_len'))
VARIANTS = {
    'linear3': lambda A: np.column_stack(
        [np.sqrt(A[:, _J_AREA]), A[:, _J_PEAK], A[:, _J_TAIL]]),
    'area1d': lambda A: A[:, [_J_AREA]],
    'full18': lambda A: A[:, :18],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name)
    p.add_argument('--out-root', default=str(REPO / 'results' / 'linear_baseline'))
    return p.parse_args()


def t21_input_features(run_dir, cache_path):
    """Thermal features for every track-21 window, INPUT SIDE ONLY (see header)."""
    if cache_path.exists():
        d = np.load(cache_path)
        return d['features'], d['x_mm']
    with open(run_dir / f'track_{HELD_OUT_TEST_TRACK}_samples.pkl', 'rb') as f:
        rows = pickle.load(f)
    feats = np.array([window_features(r['thermal_window']) for r in rows])
    x_mm = np.array([r['x_mm'] for r in rows])
    frame_index = np.array([r['frame_index'] for r in rows])
    del rows
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feats, x_mm=x_mm,
                        frame_index=frame_index)
    return feats, x_mm


def run_variant(name, featurize, X, y, track, valid, x_all, X21, x21, out_dir):
    folds, biases = [], {}
    F = featurize(X)
    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits(), start=1):
        assert val_track != HELD_OUT_TEST_TRACK
        tr = np.isin(track, train_tracks) & valid
        va = (track == val_track) & valid
        scaler = StandardScaler().fit(F[tr])
        model = BayesianRidge().fit(scaler.transform(F[tr]), y[tr])
        mu, sd = model.predict(scaler.transform(F[va]), return_std=True)
        biases[fold_i] = float(np.median(mu - y[va]))
        folds.append({'fold': fold_i, 'val_track': int(val_track),
                      'y': y[va], 'x': x_all[va], 'mu': mu, 'sigma': sd,
                      'const_mae': mae_mm(y[va], np.full(int(va.sum()), y[tr].mean()))})

    result = {'per_fold': [], 'pooled': {}, 'fold_median_bias_mm': biases}
    pooled = {'y': [], 'mu': [], 'sig0': [], 'sig1': []}
    for f in folds:
        v_k = float(np.mean([b ** 2 for j, b in biases.items() if j != f['fold']]))
        sig1 = np.sqrt(f['sigma'] ** 2 + v_k)
        in_rng = (f['x'] >= X_RANGE_T21[0]) & (f['x'] <= X_RANGE_T21[1])
        result['per_fold'].append({
            'fold': f['fold'], 'val_track': f['val_track'],
            'const_mae': f['const_mae'], 'inflation_sd_mm': float(np.sqrt(v_k)),
            'full': {'before': gaussian_metrics(f['y'], f['mu'], f['sigma']),
                     'after': gaussian_metrics(f['y'], f['mu'], sig1)},
            'x29_99': {'before': gaussian_metrics(
                *(a[in_rng] for a in (f['y'], f['mu'], f['sigma']))),
                'after': gaussian_metrics(f['y'][in_rng], f['mu'][in_rng], sig1[in_rng])},
        })
        for k, arr in zip(('y', 'mu', 'sig0', 'sig1'), (f['y'], f['mu'], f['sigma'], sig1)):
            pooled[k].append(arr)
    yc, muc = np.concatenate(pooled['y']), np.concatenate(pooled['mu'])
    result['pooled'] = {
        'before': gaussian_metrics(yc, muc, np.concatenate(pooled['sig0'])),
        'after': gaussian_metrics(yc, muc, np.concatenate(pooled['sig1']))}

    # final 3-dev-track fit -> track-21 input-side prediction
    scaler = StandardScaler().fit(F[valid])
    final = BayesianRidge().fit(scaler.transform(F[valid]), y[valid])
    F21 = featurize(X21)
    mu21, sd21 = final.predict(scaler.transform(F21), return_std=True)
    v_final = float(np.mean([b ** 2 for b in biases.values()]))
    sig21 = np.sqrt(sd21 ** 2 + v_final)
    med = float(np.median(mu21))
    result['t21_external_check'] = {
        'n_windows': int(len(mu21)), 'median_pred_mm': med,
        'iqr_mm': [float(np.percentile(mu21, 25)), float(np.percentile(mu21, 75))],
        'inflation_sd_mm': float(np.sqrt(v_final)),
        'sem_band_mm': SEM_BAND_T21_MM, 'median_over_sem': med / SEM_BAND_T21_MM,
        'dev_width_sem_ratio_range': DEV_WIDTH_SEM_RATIO,
    }
    if name == PRIMARY:
        np.savez_compressed(out_dir / 't21_predictions_primary.npz',
                            x_mm=x21, mu_mm=mu21, sigma_mm=sig21,
                            sigma_no_inflation_mm=sd21)
    return result


def main():
    args = parse_args()
    np.random.seed(SEED)
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    cache = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1.npz'
    d = np.load(cache, allow_pickle=False)
    X, y = d['features'], d['width_mean_mm']
    track, valid = d['track_id'], d['valid']
    x_all = d['x_mm']
    assert HELD_OUT_TEST_TRACK not in set(track.tolist())

    run_dir = REPO / 'processed_data' / 'datasets' / args.dataset_run
    t21_cache = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1_t21inputs.npz'
    X21, x21 = t21_input_features(run_dir, t21_cache)

    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {'primary': PRIMARY, 'variants': {}}
    print(f"\n{'variant':<10} {'MAE':>7} {'NLL':>7} {'cov90':>6} {'cov90+':>7} "
          f"{'NLL+':>7} {'t21 med':>8} {'/SEM':>6}  fold MAEs")
    for name, featurize in VARIANTS.items():
        r = run_variant(name, featurize, X, y, track, valid, x_all, X21, x21, out_dir)
        metrics['variants'][name] = r
        b, a = r['pooled']['before'], r['pooled']['after']
        t21 = r['t21_external_check']
        fold_maes = '/'.join(f"{f['full']['before']['mae']:.4f}" for f in r['per_fold'])
        # three-state check: the dev ratio range is an empirical 3-point band,
        # not a spec — a sub-2% excursion is far below the SEM band's own
        # square-pixel scale uncertainty, so it is 'borderline', not a fail.
        lo, hi = DEV_WIDTH_SEM_RATIO
        ratio = t21['median_over_sem']
        if lo <= ratio <= hi:
            flag = ' '
        elif lo * 0.98 <= ratio <= hi * 1.02:
            flag = '~'
        else:
            flag = 'X'
        r['t21_external_check']['check'] = {' ': 'pass', '~': 'borderline', 'X': 'fail'}[flag]
        star = '*' if name == PRIMARY else ' '
        print(f"{star}{name:<9} {b['mae']:7.4f} {b['nll']:7.3f} {b['coverage_90']:6.2f} "
              f"{a['coverage_90']:7.2f} {a['nll']:7.3f} {t21['median_pred_mm']:8.3f} "
              f"{t21['median_over_sem']:5.2f}{flag}  {fold_maes}")
    print('* = primary; ~ = borderline (<=2% outside the empirical dev ratio '
          f'range {DEV_WIDTH_SEM_RATIO[0]}-{DEV_WIDTH_SEM_RATIO[1]}); X = fail')
    print('reference: const 0.2043 | phys_only 0.1384 (t21/SEM 0.98) | '
          'phys_gp 0.1419 (cov90 0.95 after inflation) | NO power mapping used here')

    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    cfg = {'seed': SEED, 'model': 'BayesianRidge', 'primary': PRIMARY,
           'variants': {n: 'see module docstring' for n in VARIANTS},
           'excluded': ['x_mm', 'nan_frac', 'n_cols_used'],
           'inflation_rule': 'v_k = mean of other folds median-bias^2'}
    (out_dir / 'provenance.json').write_text(json.dumps(
        {'commit': commit, 'dirty': dirty, 'dataset_run': args.dataset_run,
         'config_hash': config_hash(cfg), 'config': cfg,
         'selection_note': ('five feature sets screened post-hoc 2026-07-17 on '
                            'LOTO MAE + label-free SEM check; physics-constrained '
                            'but post-hoc — stated in the report'),
         't21_access_note': ('track_21_samples.pkl read directly, bypassing the '
                             'guarded loader: USER-SANCTIONED 2026-07-17, input '
                             'side only (thermal_window/x_mm/frame_index); '
                             'width/validity columns never read.')},
        indent=2, sort_keys=True))
    print(f'outputs -> {out_dir}')


if __name__ == '__main__':
    main()
