#!/usr/bin/env python
"""Fit phys_gp_linear3 (CORROBORATION model) and cache its track-21 prediction.

Two-stage model, refit on all 3 dev tracks, applied to every track-21 window:

  stage 1  power-law mean m(P) = a*P**b, power only (run_gp_physmean.py's
           phys_gp stage 1, unchanged).
  stage 2  residual GP over the SAME 3 linear3 features PRIMARY uses
           (run_linear_baseline.py's linear3 featurizer), x_mm dropped.
           Uses a DotProduct (linear) kernel, not run_gp_physmean.py's
           RBF kernel: an RBF residual GP was found (2026-07-20) to
           collapse to a FLAT line on track 21 -- predicted-width std
           5.5e-17mm, zero spatial signal -- because track 21's P=200W is
           further outside the dev tracks' 300-400W range than any dev-
           track LOTO fold (the project's known RBF mean-reversion-under-
           extrapolation failure mode, see probe_waviness_gp*.py). Swapped
           to DotProduct -- validated in probe_waviness_gp_linear_kernel.py
           to recover BayesianRidge-level performance -- and restricted to
           3 features specifically (not all 18: run_linear_baseline.py's
           full18 variant shows collinear 18-feature linear coefficients
           explode under extrapolation, t21 ratio 3.04, and a DotProduct
           kernel IS linear regression, so the same collinearity risk
           applies at 18 features).

This model is NOT submitted -- it exists to check that two independently-
structured models (PRIMARY's one-stage linear regression vs. this power-
law-mean + GP-residual model) agree on track 21, since there is no ground
truth to validate against directly. See predict_track21.py for the actual
submission assembly and notebooks/07_report_figures.ipynb for the PRIMARY
vs. CORROBORATION comparison figure.

Track-21 access: as in run_linear_baseline.py, track_21_samples.pkl is read
INPUT SIDE ONLY (thermal_window/x_mm/frame_index/features) -- width/
width_std/n_cols_used are never read for track 21 (NaN/0 in the raw pickle
since width was never computed for the held-out track). This model's per-
sample label-noise term (sample_noise_var, which normally comes from the
width measurement's own column-to-column std) has no such measurement at
track 21 by construction, so every track-21 window uses the pooled dev-
track noise floor (median wstd / median n_cols across all valid dev
samples) instead of a per-window value. This is an approximation, stated
here for the report.

Variance inflation (v_final) is recomputed inline from the 3 LOTO stage-1
biases, following this codebase's existing convention
(run_gp_calib_inflation.py refits rather than depending on another
script's cached per-sample output).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from run_gp_baseline import ALPHA_FLOOR, SEED
from run_gp_physmean import POWER_W, fit_power_law, sample_noise_var
from run_linear_baseline import VARIANTS as LINEAR_VARIANTS
from submission_checks import sanity_check
from train_temporal_cnn import config_hash, git_provenance

FEATURIZE = LINEAR_VARIANTS['linear3']  # same 3 physics features as PRIMARY


def fit_residual_gp_dotproduct(Xtr, rtr, alpha_std):
    """DotProduct (linear) kernel residual GP -- extrapolates like Bayesian
    linear regression, unlike run_gp_physmean.fit_residual_gp's RBF kernel.
    Validated against BayesianRidge in probe_waviness_gp_linear_kernel.py."""
    kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)) + WhiteKernel(1e-2, (1e-8, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_std, n_restarts_optimizer=5,
                                  random_state=SEED, normalize_y=False)
    gp.fit(Xtr, rtr)
    return gp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name)
    p.add_argument('--out-root', default=str(REPO / 'results' / 'phys_gp_linear3'))
    return p.parse_args()


def fit_phys_gp_linear3(feats, y_all, track, valid, wstd_all, ncols_all, X21):
    """Final phys_gp_linear3 fit on all 3 dev tracks -> track-21 prediction."""
    biases = []
    for train_tracks, val_track in loto_cv_splits():
        tr = np.isin(track, train_tracks) & valid
        va = (track == val_track) & valid
        wstd_fill_f = float(np.nanmedian(wstd_all[tr]))
        nv_tr_f = sample_noise_var(wstd_all[tr], ncols_all[tr], wstd_fill_f)
        P_tr_f = np.array([POWER_W[t] for t in track[tr]])
        a_f, b_f = fit_power_law(P_tr_f, y_all[tr], nv_tr_f)
        m_val = a_f * POWER_W[val_track] ** b_f
        biases.append(float(m_val - np.median(y_all[va])))
    v_final = float(np.mean(np.square(biases)))

    wstd_fill = float(np.nanmedian(wstd_all[valid]))
    ncols_fill = float(np.nanmedian(ncols_all[valid]))
    nv_tr = sample_noise_var(wstd_all[valid], ncols_all[valid], wstd_fill)
    nv_t21 = np.full(len(X21), max((wstd_fill / np.sqrt(ncols_fill)) ** 2, ALPHA_FLOOR))

    P_tr = np.array([POWER_W[t] for t in track[valid]])
    a, b = fit_power_law(P_tr, y_all[valid], nv_tr)
    m_t21 = a * POWER_W[HELD_OUT_TEST_TRACK] ** b

    F, F21 = FEATURIZE(feats[valid]), FEATURIZE(X21)
    scaler = StandardScaler().fit(F)
    r_tr = y_all[valid] - a * P_tr ** b
    r_mean, r_std = float(r_tr.mean()), float(r_tr.std())
    gp = fit_residual_gp_dotproduct(scaler.transform(F), (r_tr - r_mean) / r_std,
                                    np.maximum(nv_tr / r_std ** 2, ALPHA_FLOOR))
    mu_s, sd_s = gp.predict(scaler.transform(F21), return_std=True)
    mu = m_t21 + (mu_s * r_std + r_mean)
    sigma = np.sqrt(sd_s ** 2 + nv_t21 / r_std ** 2) * r_std
    sigma_inflated = np.sqrt(sigma ** 2 + v_final)

    return {'a': a, 'b': b, 'm_t21_mm': float(m_t21), 'mu': mu,
            'sigma': sigma_inflated, 'sigma_no_inflation': sigma,
            'v_final_mm2': v_final, 'loto_biases_mm': biases,
            'kernel': str(gp.kernel_),
            'noise_floor_mm': float(np.sqrt(nv_t21[0]))}


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
    feats, track, valid = d['features'], d['track_id'], d['valid']
    y_all = d['width_mean_mm']
    wstd_all, ncols_all = d['width_std_mm'], d['n_cols_used'].astype(np.float64)
    assert HELD_OUT_TEST_TRACK not in set(track.tolist())

    t21_cache = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1_t21inputs.npz'
    if not t21_cache.exists():
        sys.exit(f'missing {t21_cache} -- run scripts/run_linear_baseline.py first '
                 '(it builds this cache from track_21_samples.pkl)')
    dt21 = np.load(t21_cache)
    X21, x21 = dt21['features'], dt21['x_mm']

    corr = fit_phys_gp_linear3(feats, y_all, track, valid, wstd_all, ncols_all, X21)
    sanity = sanity_check('phys_gp_linear3', x21, corr['mu'], corr['sigma'])

    print(f"phys_gp_linear3: median {np.median(corr['mu']):.4f} mm, "
          f"pooled mean sigma {corr['sigma'].mean():.4f} mm "
          f"(m(P=200W)={corr['m_t21_mm']:.4f} mm, b={corr['b']:.3f})")
    print(f"inflation: LOTO biases {corr['loto_biases_mm']} -> "
          f"sigma_infl={np.sqrt(corr['v_final_mm2']):.4f} mm; "
          f"per-window label-noise floor {corr['noise_floor_mm']:.4f} mm (dev-pooled fallback)")

    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / 'track21_corroboration.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_mm', 'width_mm', 'width_sigma_mm'])
        for xi, mi, si in zip(x21, corr['mu'], corr['sigma']):
            w.writerow([f'{xi:.4f}', f'{mi:.6f}', f'{si:.6f}'])
    print(f'corroboration predictions -> {out_path}  ({len(x21)} windows)')

    metrics = {
        'model': 'phys_gp_linear3',
        'power_law': {'a': corr['a'], 'b': corr['b'], 'm_t21_mm': corr['m_t21_mm']},
        'inflation': {'loto_biases_mm': corr['loto_biases_mm'], 'v_final_mm2': corr['v_final_mm2']},
        'noise_floor_mm': corr['noise_floor_mm'],
        'gp_kernel': corr['kernel'],
        'sanity_check': sanity,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))

    cfg = {'seed': SEED,
           'model': 'phys_gp_linear3 (power-law mean + DotProduct-kernel residual GP on linear3 features, x_mm dropped)',
           'featurizer': 'run_linear_baseline.VARIANTS[linear3]'}
    (out_dir / 'provenance.json').write_text(json.dumps(
        {'commit': commit, 'dirty': dirty, 'dataset_run': args.dataset_run,
         'config_hash': config_hash(cfg), 'config': cfg,
         't21_access_note': ('track_21_samples.pkl / t21inputs.npz read input-side only, '
                             'as in run_linear_baseline.py; noise floor uses the pooled '
                             'dev-track median wstd/n_cols since width was never computed '
                             'for track 21.')},
        indent=2, sort_keys=True))
    print(f'outputs -> {out_dir}')


if __name__ == '__main__':
    main()
