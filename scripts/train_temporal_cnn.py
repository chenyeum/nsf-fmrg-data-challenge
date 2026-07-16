#!/usr/bin/env python3
"""Train the Step-2 temporal CNN across the 3 leave-one-track-out CV folds.

Examples:
    uv run python scripts/train_temporal_cnn.py --temporal conv
    uv run python scripts/train_temporal_cnn.py --temporal bilstm
    uv run python scripts/train_temporal_cnn.py --smoke   # tiny pipeline check

Writes processed_data/model_runs/<timestamp>_<temporal>/results.json plus the
best per-fold weights, with pinned provenance {commit_hash, config_hash,
data_hash} per .CLAUDE.md §6 (data_hash inherited from the processed dataset
run's own provenance.json). Track 21 never enters any fold (§3) — asserted at
runtime, not just assumed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# must be set before cuBLAS init for deterministic LSTM matmuls on CUDA (§5)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from temporal_cnn import TEMPORAL_MODULES, TemporalCNN, TrackSampleDataset, mae_mm

DATASETS_DIR = REPO / 'processed_data' / 'datasets'
MODEL_RUNS_DIR = REPO / 'processed_data' / 'model_runs'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--temporal', choices=sorted(TEMPORAL_MODULES), default='conv')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--patience', type=int, default=15,
                   help='stop a fold early after this many epochs without a new best val MAE')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset-run', default=None,
                   help='processed_data/datasets/<tag> to use (default: latest)')
    p.add_argument('--smoke', action='store_true',
                   help='2 epochs on 64/32-sample subsets, nothing written')
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def setup_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_provenance():
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
    commit = run('git', 'rev-parse', 'HEAD')
    dirty = bool(run('git', 'status', '--porcelain'))
    return commit, dirty


def config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run_epoch(model, loader, device, mean, std, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    all_true, all_pred = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for thermal, target in loader:
            thermal = ((thermal - mean) / std).to(device)
            target = target.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(thermal)
            loss = loss_fn(pred, target)
            if train:
                loss.backward()
                optimizer.step()
            all_true.append(target.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    return mae_mm(np.concatenate(all_true), np.concatenate(all_pred))


def train_one_fold(fold_i, train_tracks, val_track, run_dir, args, device):
    assert HELD_OUT_TEST_TRACK not in (*train_tracks, val_track), \
        f'held-out track {HELD_OUT_TEST_TRACK} leaked into fold {fold_i + 1} (§3)'

    train_ds = TrackSampleDataset(run_dir, train_tracks)
    val_ds = TrackSampleDataset(run_dir, [val_track])
    if args.smoke:
        train_ds.rows = train_ds.rows[:64]
        val_ds.rows = val_ds.rows[:32]

    # scaler fit on train only (§3) — val tracks never touch these stats
    mean, std = train_ds.thermal_mean_std()

    # constant baseline (predict the train-set target mean), the reference floor
    baseline_val_mae = mae_mm(val_ds.targets(),
                              np.full(len(val_ds), train_ds.targets().mean()))

    loader_kwargs = dict(pin_memory=True, num_workers=4) if device.type == 'cuda' else dict()
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold_i)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    model = TemporalCNN(temporal=args.temporal, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.L1Loss()

    history = {'train_mae': [], 'val_mae': []}
    best_val_mae, best_epoch, best_state = float('inf'), -1, None
    t_start = time.time()
    for epoch in range(args.epochs):
        train_mae = run_epoch(model, train_loader, device, mean, std, loss_fn, optimizer)
        val_mae = run_epoch(model, val_loader, device, mean, std, loss_fn)
        scheduler.step()
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        if val_mae < best_val_mae:
            best_val_mae, best_epoch = val_mae, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  epoch {epoch + 1:2d}/{args.epochs}: '
              f'train_MAE={train_mae:.4f}mm  val_MAE={val_mae:.4f}mm')
        if epoch - best_epoch >= args.patience:
            print(f'  early stop: no val improvement for {args.patience} epochs')
            break

    result = dict(
        fold=fold_i + 1, train_tracks=list(train_tracks), val_track=val_track,
        n_train=len(train_ds), n_val=len(val_ds),
        thermal_mean=mean, thermal_std=std,
        baseline_val_mae=baseline_val_mae,
        best_val_mae=best_val_mae, best_epoch=best_epoch + 1,
        final_val_mae=history['val_mae'][-1],
        history=history, fold_time_sec=time.time() - t_start,
    )
    return result, best_state


def main():
    args = parse_args()
    device = pick_device()
    setup_determinism(args.seed)

    run_dir = (DATASETS_DIR / args.dataset_run if args.dataset_run
               else sorted(d for d in DATASETS_DIR.iterdir() if d.is_dir())[-1])
    dataset_provenance = json.loads((run_dir / 'provenance.json').read_text())

    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty — this run is exploratory, '
              'not decision-grade (.CLAUDE.md §0)', file=sys.stderr)

    config = dict(
        temporal=args.temporal, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        patience=args.patience, seed=args.seed, dataset_run=run_dir.name,
        device=device.type, smoke=args.smoke,
    )
    print(f'dataset run: {run_dir.name}   device: {device.type}   '
          f'temporal: {args.temporal}{"   [SMOKE]" if args.smoke else ""}')

    fold_results, fold_states = [], []
    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits()):
        print(f'\n=== fold {fold_i + 1}/3: train={train_tracks} val={val_track} ===')
        result, state = train_one_fold(fold_i, train_tracks, val_track,
                                       run_dir, args, device)
        print(f'  best val_MAE={result["best_val_mae"]:.4f}mm at epoch '
              f'{result["best_epoch"]}, baseline={result["baseline_val_mae"]:.4f}mm, '
              f'{result["fold_time_sec"]:.0f}s')
        fold_results.append(result)
        fold_states.append(state)

    print(f'\n{"fold":>4} {"val":>4} {"baseline":>9} {"best":>8} {"epoch":>6} {"beats?":>7}')
    wins = 0
    for r in fold_results:
        beats = r['best_val_mae'] < r['baseline_val_mae']
        wins += beats
        print(f'{r["fold"]:>4} {r["val_track"]:>4} {r["baseline_val_mae"]:>9.4f} '
              f'{r["best_val_mae"]:>8.4f} {r["best_epoch"]:>6} {str(beats):>7}')
    mean_baseline = np.mean([r['baseline_val_mae'] for r in fold_results])
    mean_best = np.mean([r['best_val_mae'] for r in fold_results])
    print(f'mean baseline={mean_baseline:.4f}mm  mean best={mean_best:.4f}mm  '
          f'improvement={100 * (mean_baseline - mean_best) / mean_baseline:.1f}%  '
          f'beats baseline in {wins}/3 folds')

    if args.smoke:
        print('\nSMOKE PASS — nothing written')
        return

    out_dir = MODEL_RUNS_DIR / f'{datetime.now():%Y%m%d_%H%M%S}_{args.temporal}'
    out_dir.mkdir(parents=True)
    for r, state in zip(fold_results, fold_states):
        torch.save(state, out_dir / f'fold{r["fold"]}_val{r["val_track"]}_best.pt')
    (out_dir / 'results.json').write_text(json.dumps(dict(
        config=config,
        config_hash=config_hash(config),
        commit_hash=commit,
        git_dirty=dirty,
        data_hash=dataset_provenance['data_hash'],
        dataset_provenance_commit=dataset_provenance['commit_hash'],
        fold_results=fold_results,
    ), indent=2))
    print(f'\nresults + weights written to {out_dir.relative_to(REPO)}')


if __name__ == '__main__':
    main()
