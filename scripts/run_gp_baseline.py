#!/usr/bin/env python
"""Stage-2 probabilistic anchor: GP regression on hand-crafted melt-pool features.

3-fold LOTO CV (loto_cv_splits, Track 21 never loaded — enforced in
thermal_features.load_samples). Anchor numbers only: no kernel/PCA variants.

Model: anisotropic-RBF (ARD) GP with heteroscedastic per-sample noise
alpha_i = (width_std_mm_i / sqrt(n_cols_used_i))^2, transformed to
standardized-target units. Feature and target scalers fit on train fold only.

References reported alongside: constant (train-fold mean width) and interp
(per-x mean of train-track widths, linearly interpolated) — the fair
"no-thermal-information" baseline.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from temporal_cnn import mae_mm  # single metric implementation (.CLAUDE.md §3)
from thermal_features import FEATURE_NAMES, build_feature_cache
from preprocessing import DEV_TRACKS, HELD_OUT_TEST_TRACK, loto_cv_splits
from train_temporal_cnn import config_hash, git_provenance

SEED = 0
ALPHA_FLOOR = 1e-6
X_RANGE_T21 = (29.0, 99.0)  # track-21-comparable coverage
Z_50 = norm.ppf(0.75)
Z_90 = norm.ppf(0.95)

COLOR_GP = '#3B82F6'
COLOR_INTERP = '#F59E0B'
COLOR_INK = '#1F2937'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name,
                   help='dataset run tag (default: latest)')
    p.add_argument('--out-root', default=str(REPO / 'results' / 'gp_baseline'))
    return p.parse_args()


def gaussian_metrics(y, mu, sigma):
    z = (y - mu) / sigma
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2))
    crps = float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
    return {
        'mae': mae_mm(y, mu),
        'rmse': float(np.sqrt(np.mean((y - mu) ** 2))),
        'nll': nll,
        'crps': crps,
        'coverage_50': float(np.mean(np.abs(z) <= Z_50)),
        'coverage_90': float(np.mean(np.abs(z) <= Z_90)),
        'n': int(len(y)),
    }


def point_metrics(y, pred):
    return {'mae': mae_mm(y, pred),
            'rmse': float(np.sqrt(np.mean((y - pred) ** 2))),
            'n': int(len(y))}


def metrics_for_range(y, x, mu, sigma, const_pred, interp_pred, x_range=None):
    m = np.ones(len(y), dtype=bool) if x_range is None else \
        (x >= x_range[0]) & (x <= x_range[1])
    return {
        'gp': gaussian_metrics(y[m], mu[m], sigma[m]),
        'constant': point_metrics(y[m], const_pred[m]),
        'interp': point_metrics(y[m], interp_pred[m]),
    }


def main():
    args = parse_args()
    np.random.seed(SEED)

    dataset_dir = REPO / 'processed_data' / 'datasets' / args.dataset_run
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    cache_path = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1.npz'
    if not cache_path.exists():
        print(f'feature cache missing, building -> {cache_path}')
        build_feature_cache(dataset_dir, cache_path)
    d = np.load(cache_path, allow_pickle=False)

    # model input = 18 window aggregates + x_mm (matches FEATURE_NAMES order)
    X_all = np.hstack([d['features'], d['x_mm'][:, None]])
    assert X_all.shape[1] == len(FEATURE_NAMES)
    track = d['track_id']
    assert HELD_OUT_TEST_TRACK not in set(track.tolist())
    valid = d['valid']
    y_all = d['width_mean_mm']
    x_all = d['x_mm']
    wstd_all = d['width_std_mm']
    ncols_all = d['n_cols_used'].astype(np.float64)

    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    pooled = {'y': [], 'x': [], 'mu': [], 'sigma': [], 'const': [], 'interp': []}
    kernel_summaries = []

    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits(), start=1):
        assert val_track != HELD_OUT_TEST_TRACK and HELD_OUT_TEST_TRACK not in train_tracks
        tr = np.isin(track, train_tracks) & valid
        va = (track == val_track) & valid

        x_scaler = StandardScaler().fit(X_all[tr])
        Xtr, Xva = x_scaler.transform(X_all[tr]), x_scaler.transform(X_all[va])
        y_mean, y_std = float(y_all[tr].mean()), float(y_all[tr].std())
        ytr = (y_all[tr] - y_mean) / y_std

        # heteroscedastic per-sample noise in standardized-target units;
        # non-finite width_std (sub-block detection < 2) -> train-fold median
        wstd_fill = float(np.nanmedian(wstd_all[tr]))
        def alpha_for(mask):
            w = np.where(np.isfinite(wstd_all[mask]), wstd_all[mask], wstd_fill)
            a = (w / np.sqrt(ncols_all[mask])) ** 2 / y_std ** 2
            return np.maximum(a, ALPHA_FLOOR)

        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=np.ones(Xtr.shape[1]),
                        length_scale_bounds=(1e-2, 1e3))
                  + WhiteKernel(1e-2, (1e-8, 1e1)))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_for(tr),
                                      n_restarts_optimizer=5, random_state=SEED,
                                      normalize_y=False)
        gp.fit(Xtr, ytr)
        kernel_summaries.append(str(gp.kernel_))

        mu_s, sd_s = gp.predict(Xva, return_std=True)
        mu = mu_s * y_std + y_mean
        sigma = np.sqrt(sd_s ** 2 + alpha_for(va)) * y_std  # total incl. sample noise

        const_pred = np.full(va.sum(), y_all[tr].mean())
        interp_pred = np.zeros(va.sum())
        for t in train_tracks:
            tm = (track == t) & valid
            order = np.argsort(x_all[tm])
            interp_pred += np.interp(x_all[va], x_all[tm][order], y_all[tm][order])
        interp_pred /= len(train_tracks)

        y_va, x_va = y_all[va], x_all[va]
        fold = {
            'fold': fold_i, 'train_tracks': list(train_tracks), 'val_track': int(val_track),
            'n_train': int(tr.sum()), 'n_val': int(va.sum()),
            'full': metrics_for_range(y_va, x_va, mu, sigma, const_pred, interp_pred),
            'x29_99': metrics_for_range(y_va, x_va, mu, sigma, const_pred, interp_pred,
                                        X_RANGE_T21),
        }
        per_fold.append(fold)
        for k, v in zip(pooled, (y_va, x_va, mu, sigma, const_pred, interp_pred)):
            pooled[k].append(v)

        # prediction plot
        fig, ax = plt.subplots(figsize=(11, 4))
        o = np.argsort(x_va)
        ax.fill_between(x_va[o], (mu - 2 * sigma)[o], (mu + 2 * sigma)[o],
                        color=COLOR_GP, alpha=0.18, linewidth=0, label='GP ±2σ')
        ax.plot(x_va[o], mu[o], color=COLOR_GP, lw=2, label='GP mean')
        ax.plot(x_va[o], interp_pred[o], color=COLOR_INTERP, lw=2, ls='--',
                label='interp baseline')
        ax.plot(x_va, y_va, '.', color=COLOR_INK, ms=5, label=f'track {val_track} true')
        inv = (track == val_track) & ~valid
        for xi in x_all[inv]:
            ax.axvspan(xi - 0.1, xi + 0.1, color='0.85', zorder=0, lw=0)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('width (mm)')
        ax.set_title(f'fold {fold_i}: val track {val_track} '
                     f'(train {train_tracks}); shaded = invalid windows')
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc='best', frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f'fold{fold_i}_val{val_track}_predictions.png', dpi=150)
        plt.close(fig)

    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    results = {
        'per_fold': per_fold,
        'pooled': {
            'full': metrics_for_range(pooled['y'], pooled['x'], pooled['mu'],
                                      pooled['sigma'], pooled['const'], pooled['interp']),
            'x29_99': metrics_for_range(pooled['y'], pooled['x'], pooled['mu'],
                                        pooled['sigma'], pooled['const'], pooled['interp'],
                                        X_RANGE_T21),
        },
    }

    # PIT histograms: per fold + pooled
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    start = 0
    for ax, fold in zip(axes[:3], per_fold):
        n = fold['n_val']
        sl = slice(start, start + n)
        pit = norm.cdf((pooled['y'][sl] - pooled['mu'][sl]) / pooled['sigma'][sl])
        ax.hist(pit, bins=10, range=(0, 1), color=COLOR_GP, edgecolor='white')
        ax.axhline(n / 10, color=COLOR_INK, lw=1, ls=':')
        ax.set_title(f"fold {fold['fold']} (val {fold['val_track']})")
        ax.set_xlabel('PIT')
        start += n
    pit = norm.cdf((pooled['y'] - pooled['mu']) / pooled['sigma'])
    axes[3].hist(pit, bins=10, range=(0, 1), color=COLOR_GP, edgecolor='white')
    axes[3].axhline(len(pit) / 10, color=COLOR_INK, lw=1, ls=':')
    axes[3].set_title('pooled')
    axes[3].set_xlabel('PIT')
    axes[0].set_ylabel('count')
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle('PIT calibration (dotted = uniform reference)', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'pit_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    config = {
        'model': 'GaussianProcessRegressor',
        'kernel': 'C*RBF(ARD)+White', 'n_restarts_optimizer': 5,
        'alpha': 'per-sample (width_std/sqrt(n_cols))^2 standardized, floor 1e-6',
        'features': FEATURE_NAMES, 'seed': SEED,
        'dataset_run': args.dataset_run, 'x_range_alt': list(X_RANGE_T21),
        'train_subsampled': False,
    }
    provenance = {
        'commit_hash': commit, 'git_dirty': dirty,
        'dataset_run': args.dataset_run,
        'feature_file': str(cache_path.relative_to(REPO)),
        'feature_file_md5': hashlib.md5(cache_path.read_bytes()).hexdigest(),
        'seed': SEED, 'config_hash': config_hash(config), 'config': config,
        'fitted_kernels': kernel_summaries,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    with open(out_dir / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2, sort_keys=True)

    print(f"\n{'':14s}{'MAE':>8s}{'RMSE':>8s}{'NLL':>8s}{'CRPS':>8s}{'cov50':>7s}{'cov90':>7s}")
    for fold in per_fold:
        for rng in ('full', 'x29_99'):
            g = fold[rng]['gp']
            print(f"fold{fold['fold']} v{fold['val_track']:>2} {rng:<7s}"
                  f"{g['mae']:8.4f}{g['rmse']:8.4f}{g['nll']:8.3f}{g['crps']:8.4f}"
                  f"{g['coverage_50']:7.2f}{g['coverage_90']:7.2f}"
                  f"   | const {fold[rng]['constant']['mae']:.4f}"
                  f" interp {fold[rng]['interp']['mae']:.4f}")
    for rng in ('full', 'x29_99'):
        g = results['pooled'][rng]['gp']
        print(f"pooled   {rng:<7s}{g['mae']:8.4f}{g['rmse']:8.4f}{g['nll']:8.3f}"
              f"{g['crps']:8.4f}{g['coverage_50']:7.2f}{g['coverage_90']:7.2f}"
              f"   | const {results['pooled'][rng]['constant']['mae']:.4f}"
              f" interp {results['pooled'][rng]['interp']['mae']:.4f}")
    print(f'\noutputs -> {out_dir}')


if __name__ == '__main__':
    main()
