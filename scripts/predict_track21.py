#!/usr/bin/env python
"""Assemble the track-21 width submission.

Reads the PRIMARY model's cached track-21 prediction (linear3 BayesianRidge,
scripts/run_linear_baseline.py -- t21_predictions_primary.npz), sanity-checks
it, and writes the actual submission CSV (x_mm, width_mm, width_sigma_mm).

Uncertainty format: per-x Gaussian mean +/- sigma, sigma already inflated by
the cross-fold empirical rule (see run_linear_baseline.py). This is the
self-decided default (see memory output-scope-decision-2026-07-19) pending
an organizer reply on uncertainty submission format; only w(x) is submitted,
not boundary/centerline (see the same memory for why that's a firm skip).

This script does NOT train anything and does NOT compare against the
CORROBORATION model -- see scripts/fit_phys_gp_linear3.py for that model's
training/caching, and notebooks/07_report_figures.ipynb (section 3) for the
PRIMARY vs. CORROBORATION comparison figure.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from submission_checks import sanity_check
from train_temporal_cnn import config_hash, git_provenance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name)
    p.add_argument('--out-root', default=str(REPO / 'results' / 'predict_track21'))
    return p.parse_args()


def main():
    args = parse_args()
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    primary_path = REPO / 'results' / 'linear_baseline' / args.dataset_run / 't21_predictions_primary.npz'
    if not primary_path.exists():
        sys.exit(f'missing {primary_path} -- run scripts/run_linear_baseline.py first')
    dprim = np.load(primary_path)
    x_p, mu_p, sig_p = dprim['x_mm'], dprim['mu_mm'], dprim['sigma_mm']

    sanity = sanity_check('PRIMARY', x_p, mu_p, sig_p)
    print(f"PRIMARY (linear3): median {np.median(mu_p):.4f} mm, "
          f"pooled mean sigma {sig_p.mean():.4f} mm")

    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_path = out_dir / 'track21_submission.csv'
    with open(submission_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_mm', 'width_mm', 'width_sigma_mm'])
        for xi, mi, si in zip(x_p, mu_p, sig_p):
            w.writerow([f'{xi:.4f}', f'{mi:.6f}', f'{si:.6f}'])
    print(f'submission -> {submission_path}  ({len(x_p)} windows)')

    metrics = {
        'primary': {'x_mm': x_p.tolist(), 'width_mm': mu_p.tolist(), 'width_sigma_mm': sig_p.tolist()},
        'sanity_check': sanity,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))

    cfg = {'primary_model': 'linear3 BayesianRidge',
           'uncertainty_format': 'per-x Gaussian mean +/- sigma, cross-fold inflated',
           'submitted_columns': ['x_mm', 'width_mm', 'width_sigma_mm']}
    (out_dir / 'provenance.json').write_text(json.dumps(
        {'commit': commit, 'dirty': dirty, 'dataset_run': args.dataset_run,
         'config_hash': config_hash(cfg), 'config': cfg},
        indent=2, sort_keys=True))
    print(f'outputs -> {out_dir}')


if __name__ == '__main__':
    main()
