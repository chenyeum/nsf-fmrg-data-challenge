#!/usr/bin/env python
"""Variant A-phys: power-law physical mean function + GP residual, vs. anchor.

Two-stage model: m(P) = a * P**b fitted on the training fold's (power, width)
pairs (2 distinct powers per fold -> exactly identified; a trend anchor, NOT a
validated law), then the anchor-config GP fits the residuals from melt-pool
features. Power enters ONLY through the mean function, never as a GP feature.

Ablation (same folds, directly comparable):
  anchor_ref   - anchor metrics copied from results/gp_baseline (not refit)
  phys_only    - m(P_val) alone (pure power law; point metrics only)
  phys_gp      - m(P) + residual GP, x_mm DROPPED from features (main variant)
  phys_gp_xmm  - m(P) + residual GP with x_mm kept (isolates the x_mm decision)
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
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from run_gp_baseline import (ALPHA_FLOOR, COLOR_GP, COLOR_INK, SEED,
                             X_RANGE_T21, gaussian_metrics, metrics_for_range,
                             point_metrics)
from thermal_features import FEATURE_NAMES
from train_temporal_cnn import config_hash, git_provenance

# Laser power per track, W. ORGANIZER-CONFIRMED 2026-07-17 (relayed by the
# user: "400W - Track 8, 350W - Track 10, 300W - Track 14, 200W - Track 21").
# Historical note: the original task-spec direction ({8:200,...,21:400}) was a
# known point of participant confusion; this mapping was first inferred from
# the data on 2026-07-16 (width monotonicity, SEM band ordering, all four
# (P, median width) points on one power law, b ~= 1.15) and confirmed after.
POWER_W = {8: 400.0, 10: 350.0, 14: 300.0, 21: 200.0}
B_BOUNDS = (0.1, 6.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name)
    p.add_argument('--out-root', default=str(REPO / 'results' / 'gp_physmean'))
    return p.parse_args()


def sample_noise_var(wstd, ncols, fill):
    w = np.where(np.isfinite(wstd), wstd, fill)
    return np.maximum((w / np.sqrt(ncols)) ** 2, ALPHA_FLOOR)


def fit_power_law(P, y, noise_var):
    """Weighted LS fit of a*P**b; falls back to weighted log-log line."""
    try:
        (a, b), _ = curve_fit(
            lambda p, a, b: a * p ** b, P, y,
            p0=[float(y.mean()) / 300.0, 1.0],
            sigma=np.sqrt(noise_var), absolute_sigma=True,
            bounds=([1e-8, B_BOUNDS[0]], [np.inf, B_BOUNDS[1]]), maxfev=10000)
    except RuntimeError:
        wts = 1.0 / noise_var
        b, loga = np.polyfit(np.log(P), np.log(y), 1, w=np.sqrt(wts))
        b = float(np.clip(b, *B_BOUNDS))
        a = float(np.exp(loga))
    return float(a), float(b)


def fit_residual_gp(Xtr, rtr, alpha_std):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=np.ones(Xtr.shape[1]),
                    length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(1e-2, (1e-8, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_std,
                                  n_restarts_optimizer=5, random_state=SEED,
                                  normalize_y=False)
    gp.fit(Xtr, rtr)
    return gp


def main():
    args = parse_args()
    np.random.seed(SEED)
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    anchor_path = REPO / 'results' / 'gp_baseline' / args.dataset_run / 'metrics.json'
    anchor = json.loads(anchor_path.read_text())

    cache_path = REPO / 'processed_data' / 'features' / f'{args.dataset_run}_thermal_v1.npz'
    d = np.load(cache_path, allow_pickle=False)
    feats = d['features']                      # 18 aggregates, no x_mm
    x_col = d['x_mm'][:, None]
    track, valid = d['track_id'], d['valid']
    assert HELD_OUT_TEST_TRACK not in set(track.tolist())
    y_all, x_all = d['width_mean_mm'], d['x_mm']
    wstd_all, ncols_all = d['width_std_mm'], d['n_cols_used'].astype(np.float64)

    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {'phys_only': {}, 'phys_gp': {}, 'phys_gp_xmm': {}}
    per_fold = {k: [] for k in variants}
    pooled = {k: {q: [] for q in ('y', 'x', 'mu', 'sigma')} for k in variants}
    stage1 = []
    fitted_kernels = {'phys_gp': [], 'phys_gp_xmm': []}

    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits(), start=1):
        assert val_track != HELD_OUT_TEST_TRACK and HELD_OUT_TEST_TRACK not in train_tracks
        tr = np.isin(track, train_tracks) & valid
        va = (track == val_track) & valid
        y_tr, y_va, x_va = y_all[tr], y_all[va], x_all[va]

        wstd_fill = float(np.nanmedian(wstd_all[tr]))
        nv_tr = sample_noise_var(wstd_all[tr], ncols_all[tr], wstd_fill)
        nv_va = sample_noise_var(wstd_all[va], ncols_all[va], wstd_fill)

        # stage 1: physical mean (2 distinct powers -> exactly identified)
        P_tr = np.array([POWER_W[t] for t in track[tr]])
        a, b = fit_power_law(P_tr, y_tr, nv_tr)
        m_val = a * POWER_W[val_track] ** b
        m_tr = a * P_tr ** b
        stage1.append({
            'fold': fold_i, 'val_track': int(val_track), 'a': a, 'b': b,
            'm_P_val': float(m_val),
            'val_median_width': float(np.median(y_va)),
        })

        # variant: phys_only (point predictor)
        mu_phys = np.full(int(va.sum()), m_val)
        per_fold['phys_only'].append({
            'fold': fold_i, 'val_track': int(val_track),
            'full': {'phys_only': point_metrics(y_va, mu_phys)},
            'x29_99': {'phys_only': point_metrics(
                y_va[(x_va >= X_RANGE_T21[0]) & (x_va <= X_RANGE_T21[1])],
                mu_phys[(x_va >= X_RANGE_T21[0]) & (x_va <= X_RANGE_T21[1])])},
        })
        pooled['phys_only']['y'].append(y_va)
        pooled['phys_only']['x'].append(x_va)
        pooled['phys_only']['mu'].append(mu_phys)
        pooled['phys_only']['sigma'].append(np.full_like(mu_phys, np.nan))

        # variants: residual GPs, without / with x_mm
        r_tr = y_tr - m_tr
        for name, X_full in (('phys_gp', feats), ('phys_gp_xmm', np.hstack([feats, x_col]))):
            scaler = StandardScaler().fit(X_full[tr])
            Xtr, Xva = scaler.transform(X_full[tr]), scaler.transform(X_full[va])
            r_mean, r_std = float(r_tr.mean()), float(r_tr.std())
            gp = fit_residual_gp(Xtr, (r_tr - r_mean) / r_std,
                                 np.maximum(nv_tr / r_std ** 2, ALPHA_FLOOR))
            fitted_kernels[name].append(str(gp.kernel_))
            mu_s, sd_s = gp.predict(Xva, return_std=True)
            mu = m_val + (mu_s * r_std + r_mean)
            sigma = np.sqrt(sd_s ** 2 + np.maximum(nv_va / r_std ** 2, ALPHA_FLOOR)) * r_std

            fold_metrics = {
                'fold': fold_i, 'val_track': int(val_track),
                'full': {'gp': gaussian_metrics(y_va, mu, sigma)},
                'x29_99': {'gp': gaussian_metrics(
                    *(arr[(x_va >= X_RANGE_T21[0]) & (x_va <= X_RANGE_T21[1])]
                      for arr in (y_va, mu, sigma)))},
            }
            per_fold[name].append(fold_metrics)
            for q, arr in zip(('y', 'x', 'mu', 'sigma'), (y_va, x_va, mu, sigma)):
                pooled[name][q].append(arr)

        # prediction plot for the main variant
        mu = pooled['phys_gp']['mu'][-1]
        sigma = pooled['phys_gp']['sigma'][-1]
        fig, ax = plt.subplots(figsize=(11, 4))
        o = np.argsort(x_va)
        ax.fill_between(x_va[o], (mu - 2 * sigma)[o], (mu + 2 * sigma)[o],
                        color=COLOR_GP, alpha=0.18, linewidth=0, label='phys+GP ±2σ')
        ax.plot(x_va[o], mu[o], color=COLOR_GP, lw=2, label='phys+GP mean')
        ax.axhline(m_val, color=COLOR_INK, lw=1.2, ls='--',
                   label=f'm(P)={m_val:.3f}mm')
        ax.plot(x_va, y_va, '.', color=COLOR_INK, ms=5, label=f'track {val_track} true')
        inv = (track == val_track) & ~valid
        for xi in x_all[inv]:
            ax.axvspan(xi - 0.1, xi + 0.1, color='0.85', zorder=0, lw=0)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('width (mm)')
        ax.set_title(f'A-phys fold {fold_i}: val track {val_track} '
                     f'(P={POWER_W[val_track]:.0f}W, a={a:.3g}, b={b:.2f})')
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc='best', frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f'fold{fold_i}_val{val_track}_phys_gp.png', dpi=150)
        plt.close(fig)

    # pooled metrics
    results = {'stage1': stage1, 'variants': {'anchor_ref': anchor}}
    from scipy.stats import norm
    for name in variants:
        pool = {q: np.concatenate(v) for q, v in pooled[name].items()}
        if name == 'phys_only':
            m = {'full': {'phys_only': point_metrics(pool['y'], pool['mu'])},
                 'x29_99': {'phys_only': point_metrics(
                     *(arr[(pool['x'] >= X_RANGE_T21[0]) & (pool['x'] <= X_RANGE_T21[1])]
                       for arr in (pool['y'], pool['mu'])))}}
        else:
            m = {'full': {'gp': gaussian_metrics(pool['y'], pool['mu'], pool['sigma'])},
                 'x29_99': {'gp': gaussian_metrics(
                     *(arr[(pool['x'] >= X_RANGE_T21[0]) & (pool['x'] <= X_RANGE_T21[1])]
                       for arr in (pool['y'], pool['mu'], pool['sigma'])))}}
        results['variants'][name] = {'per_fold': per_fold[name], 'pooled': m}

    # PIT figure for the main variant
    pool = {q: np.concatenate(v) for q, v in pooled['phys_gp'].items()}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    start = 0
    for ax, fold in zip(axes[:3], per_fold['phys_gp']):
        n = fold['full']['gp']['n']
        sl = slice(start, start + n)
        pit = norm.cdf((pool['y'][sl] - pool['mu'][sl]) / pool['sigma'][sl])
        ax.hist(pit, bins=10, range=(0, 1), color=COLOR_GP, edgecolor='white')
        ax.axhline(n / 10, color=COLOR_INK, lw=1, ls=':')
        ax.set_title(f"fold {fold['fold']} (val {fold['val_track']})")
        ax.set_xlabel('PIT')
        start += n
    pit = norm.cdf((pool['y'] - pool['mu']) / pool['sigma'])
    axes[3].hist(pit, bins=10, range=(0, 1), color=COLOR_GP, edgecolor='white')
    axes[3].axhline(len(pit) / 10, color=COLOR_INK, lw=1, ls=':')
    axes[3].set_title('pooled')
    axes[3].set_xlabel('PIT')
    axes[0].set_ylabel('count')
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle('A-phys (phys_gp) PIT calibration', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'pit_hist_phys_gp.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    config = {
        'variant': 'A-phys', 'power_w': {str(k): v for k, v in POWER_W.items()},
        'power_mapping_note': 'reversed vs task spec; inferred from width/SEM '
                              'monotonicity + power-law consistency 2026-07-16, '
                              'ORGANIZER-CONFIRMED 2026-07-17',
        'mean_function': 'a*P**b, weighted LS on train fold (2 powers: exactly '
                         'identified trend anchor, not a validated law)',
        'b_bounds': list(B_BOUNDS),
        'gp': 'anchor config; x_mm dropped in phys_gp (anchor ARD: x_mm memorizes '
              'per-track profiles), kept in phys_gp_xmm',
        'seed': SEED, 'dataset_run': args.dataset_run,
    }
    provenance = {
        'commit_hash': commit, 'git_dirty': dirty, 'dataset_run': args.dataset_run,
        'feature_file_md5': hashlib.md5(cache_path.read_bytes()).hexdigest(),
        'anchor_metrics': str(anchor_path.relative_to(REPO)),
        'seed': SEED, 'config_hash': config_hash(config), 'config': config,
        'fitted_kernels': fitted_kernels,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(results, indent=2, sort_keys=True))
    (out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True))

    # stdout summary
    print('\nstage 1 per fold:')
    for s in stage1:
        print(f"  fold{s['fold']} val{s['val_track']:>2}: a={s['a']:.4g} b={s['b']:.3f}"
              f"  m(P_val)={s['m_P_val']:.4f}mm  val_median={s['val_median_width']:.4f}mm")

    def pooled_row(name):
        v = results['variants'][name]
        if name == 'anchor_ref':
            g = v['pooled']['full']['gp']
            fm = [f['full']['gp']['mae'] for f in v['per_fold']]
        elif name == 'phys_only':
            g = {**v['pooled']['full']['phys_only'], 'nll': float('nan'),
                 'crps': float('nan'), 'coverage_90': float('nan')}
            fm = [f['full']['phys_only']['mae'] for f in v['per_fold']]
        else:
            g = v['pooled']['full']['gp']
            fm = [f['full']['gp']['mae'] for f in v['per_fold']]
        return g, fm

    print(f"\n{'variant':<14s}{'MAE':>8s}{'NLL':>8s}{'CRPS':>8s}{'cov90':>7s}"
          f"{'f1 MAE':>9s}{'f2 MAE':>9s}{'f3 MAE':>9s}")
    for name in ('anchor_ref', 'phys_only', 'phys_gp', 'phys_gp_xmm'):
        g, fm = pooled_row(name)
        print(f"{name:<14s}{g['mae']:8.4f}{g['nll']:8.3f}{g['crps']:8.4f}"
              f"{g['coverage_90']:7.2f}{fm[0]:9.4f}{fm[1]:9.4f}{fm[2]:9.4f}")

    print('\nInterpretation targets (for the human):')
    print(' - Does phys_only already beat anchor on folds 1/3? (yes -> trend >> features)')
    print(' - Does phys_gp fix cov90 on folds 1/3, or stay overconfident?')
    print(' - Fold 2 sanity: phys_gp should not be much worse than anchor there.')
    print(f'\noutputs -> {out_dir}')


if __name__ == '__main__':
    main()
