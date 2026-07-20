#!/usr/bin/env python
"""Cross-fold empirical variance inflation for phys_gp (calibration fix).

The A-phys run showed phys_gp's extrapolation folds are overconfident
(fold1 cov90 0.51) because the power-law mean's model-form error
(|m(P_val) - val median| = 0.13-0.17mm) dwarfs both the GP's predictive sd
and the delta-method parameter uncertainty (0.0006-0.013mm, falsified
2026-07-16). The only data-driven estimate of that error scale available is
the other folds' realized biases.

Rule: for fold k, inflate variance by v_k = mean over the other two folds j
of (m_j(P_val_j) - median(y_val_j))^2, i.e. sigma_new = sqrt(sigma^2 + v_k).
Point predictions are untouched.

Honesty note (recorded in provenance): within LOTO, fold j's power-law fit
was trained on tracks that include fold k's val track, so v_k is not fully
independent of fold k's val track — an unavoidable overlap with only 3 dev
tracks. For the actual track-21 submission the same rule IS fully clean:
the inflation comes from the 3 LOTO fold biases and track-21 labels are
never involved.

Refits phys_gp exactly as scripts/run_gp_physmean.py (same seed, same
standardization, same kernel) since per-sample predictions were not saved.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from run_gp_baseline import (ALPHA_FLOOR, SEED, X_RANGE_T21, gaussian_metrics)
from run_gp_physmean import (POWER_W, fit_power_law, fit_residual_gp,
                             sample_noise_var, parse_args)
from sklearn.preprocessing import StandardScaler
from train_temporal_cnn import config_hash, git_provenance


def main():
    args = parse_args()
    np.random.seed(SEED)
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    cache_path = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1.npz'
    d = np.load(cache_path, allow_pickle=False)
    feats = d['features']
    track, valid = d['track_id'], d['valid']
    assert HELD_OUT_TEST_TRACK not in set(track.tolist())
    y_all, x_all = d['width_mean_mm'], d['x_mm']
    wstd_all, ncols_all = d['width_std_mm'], d['n_cols_used'].astype(np.float64)

    out_dir = Path(str(REPO / 'results' / 'gp_calib_inflation')) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits(), start=1):
        assert val_track != HELD_OUT_TEST_TRACK and HELD_OUT_TEST_TRACK not in train_tracks
        tr = np.isin(track, train_tracks) & valid
        va = (track == val_track) & valid
        y_tr, y_va, x_va = y_all[tr], y_all[va], x_all[va]

        wstd_fill = float(np.nanmedian(wstd_all[tr]))
        nv_tr = sample_noise_var(wstd_all[tr], ncols_all[tr], wstd_fill)
        nv_va = sample_noise_var(wstd_all[va], ncols_all[va], wstd_fill)

        P_tr = np.array([POWER_W[t] for t in track[tr]])
        a, b = fit_power_law(P_tr, y_tr, nv_tr)
        m_val = a * POWER_W[val_track] ** b
        r_tr = y_tr - a * P_tr ** b

        scaler = StandardScaler().fit(feats[tr])
        r_mean, r_std = float(r_tr.mean()), float(r_tr.std())
        gp = fit_residual_gp(scaler.transform(feats[tr]),
                             (r_tr - r_mean) / r_std,
                             np.maximum(nv_tr / r_std ** 2, ALPHA_FLOOR))
        mu_s, sd_s = gp.predict(scaler.transform(feats[va]), return_std=True)
        mu = m_val + (mu_s * r_std + r_mean)
        sigma = np.sqrt(sd_s ** 2 + np.maximum(nv_va / r_std ** 2, ALPHA_FLOOR)) * r_std

        folds.append({
            'fold': fold_i, 'val_track': int(val_track),
            'bias': float(m_val - np.median(y_va)),
            'y': y_va, 'x': x_va, 'mu': mu, 'sigma': sigma,
        })

    metrics = {'stage1_bias_mm': {f['fold']: f['bias'] for f in folds},
               'per_fold': [], 'pooled': {}}
    pooled = {'y': [], 'mu': [], 'sig0': [], 'sig1': []}
    print(f"{'fold':<12} {'bias':>7} {'infl_sd':>8} | "
          f"{'cov90':>5} {'->':>3} {'cov90+':>6} {'cov50':>6} {'->':>3} {'cov50+':>6} "
          f"{'NLL':>7} {'->':>3} {'NLL+':>7}")
    for k, f in enumerate(folds):
        v_k = float(np.mean([g['bias'] ** 2 for j, g in enumerate(folds) if j != k]))
        sig1 = np.sqrt(f['sigma'] ** 2 + v_k)
        before = gaussian_metrics(f['y'], f['mu'], f['sigma'])
        after = gaussian_metrics(f['y'], f['mu'], sig1)
        in_rng = (f['x'] >= X_RANGE_T21[0]) & (f['x'] <= X_RANGE_T21[1])
        metrics['per_fold'].append({
            'fold': f['fold'], 'val_track': f['val_track'],
            'inflation_sd_mm': float(np.sqrt(v_k)),
            'full': {'before': before, 'after': after},
            'x29_99': {'before': gaussian_metrics(*(arr[in_rng] for arr in (f['y'], f['mu'], f['sigma']))),
                       'after': gaussian_metrics(f['y'][in_rng], f['mu'][in_rng], sig1[in_rng])},
        })
        pooled['y'].append(f['y']); pooled['mu'].append(f['mu'])
        pooled['sig0'].append(f['sigma']); pooled['sig1'].append(sig1)
        print(f"fold{f['fold']} v{f['val_track']:<3}   {f['bias']:+.3f} {np.sqrt(v_k):8.3f} | "
              f"{before['coverage_90']:5.2f} -> {after['coverage_90']:6.2f} "
              f"{before['coverage_50']:6.2f} -> {after['coverage_50']:6.2f} "
              f"{before['nll']:7.3f} -> {after['nll']:7.3f}")

    y, mu = np.concatenate(pooled['y']), np.concatenate(pooled['mu'])
    before = gaussian_metrics(y, mu, np.concatenate(pooled['sig0']))
    after = gaussian_metrics(y, mu, np.concatenate(pooled['sig1']))
    metrics['pooled'] = {'before': before, 'after': after}
    print(f"{'pooled':<12} {'':>7} {'':>8} | "
          f"{before['coverage_90']:5.2f} -> {after['coverage_90']:6.2f} "
          f"{before['coverage_50']:6.2f} -> {after['coverage_50']:6.2f} "
          f"{before['nll']:7.3f} -> {after['nll']:7.3f}")
    print(f"CRPS pooled: {before['crps']:.4f} -> {after['crps']:.4f}")

    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    provenance = {
        'commit': commit, 'dirty': dirty,
        'dataset_run': args.dataset_run,
        'config_hash': config_hash({'seed': SEED, 'rule': 'v_k = mean of other folds bias^2',
                                    'power_w': POWER_W, 'alpha_floor': ALPHA_FLOOR}),
        'honesty_note': ('LOTO evaluation of the inflation has unavoidable overlap: '
                         'fold j fits include fold k\'s val track in training. The rule is '
                         'fully clean for the track-21 submission (uses only dev-track biases).'),
    }
    (out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(f'outputs -> {out_dir}')


if __name__ == '__main__':
    main()
