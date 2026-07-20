#!/usr/bin/env python
"""Lightweight Conditional VAE for probabilistic local width prediction.

Architecture (per user spec 2026-07-17, with two evidence-based deviations):
  - Temporal encoder: shared FrameEncoder (reused from src/temporal_cnn.py)
    + a single-layer GRU over the 11 frame embeddings.
  - Spatial encoder: shallow 3-conv CNN + AdaptiveAvgPool2d over the two
    anti-leakage SEM substrate crops (top/bottom, resized to 64x256 each).
  - Condition: laser power as a CONTINUOUS standardized log(P) scalar through
    a small linear branch — NOT nn.Embedding. An embedding is a lookup table
    over discrete ids: the held-out track's key would be untrained, which
    re-creates the track-identity cold-start problem. A continuous power axis
    is the only representation that can extrapolate to track 21 (200 W).
  - Deviation 2: the reconstruction target is the scalar width_mean_mm, not
    the height map. The raw Wyko height signal is ~15x below its noise floor,
    so a height-map decoder would spend all capacity reconstructing noise.
  - CVAE core: encoder MLP([condition, y] -> mu/logvar), reparameterization,
    decoder MLP([condition, z] -> y_hat). Beta-VAE loss: MSE + beta * KL.

Evaluation: 3-fold LOTO, same protocol/metrics as the GP scripts. Predictive
distribution at val time = decoder pushforward of z ~ N(0, I) (MC samples);
note this covers latent variability only, not the decoder's residual noise,
so calibration is expected to under-cover — reported, not patched.

Track 21 never enters any fold (asserted). Standardization (thermal pixels,
power, target) is fit on the training fold only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from preprocessing import HELD_OUT_TEST_TRACK, loto_cv_splits
from run_gp_baseline import X_RANGE_T21, gaussian_metrics
from run_gp_physmean import POWER_W
from temporal_cnn import FrameEncoder, mae_mm
from thermal_features import load_samples
from train_temporal_cnn import (config_hash, git_provenance, pick_device,
                                setup_determinism)

SEED = 0
SEM_SIZE = (64, 256)          # (H, W) each crop is resized to
EMBED_DIM = 64
SEM_DIM = 32
COND_DIM = 32
LATENT_DIM = 8
MC_SAMPLES = 200


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    datasets = sorted((REPO / 'processed_data' / 'datasets').iterdir())
    p.add_argument('--dataset-run', default=datasets[-1].name)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-root', default=str(REPO / 'results' / 'cvae'))
    return p.parse_args()


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CVAETrackDataset(Dataset):
    """Valid rows of one or more dev tracks: thermal window, SEM crops, power, y.

    SEM crops are resized once at init and cached per (track, tile) — there
    are only ~13 distinct tiles per track, so per-row storage would waste RAM.
    Thermal windows stay raw; per-fold standardization happens in the loop.
    """

    def __init__(self, run_dir, track_ids):
        self.rows, self.sem_cache = [], {}
        for track_id in track_ids:
            for r in load_samples(run_dir, track_id):
                if not r['valid']:
                    continue
                key = (track_id, r['sem_tile_index'])
                if key not in self.sem_cache:
                    self.sem_cache[key] = torch.stack([
                        self._resize(r['sem_context_top']),
                        self._resize(r['sem_context_bottom']),
                    ])                                    # (2, H, W) in [-0.5, 0.5]
                self.rows.append({
                    'thermal': r['thermal_window'],
                    'sem_key': key,
                    'log_p': float(np.log(POWER_W[track_id])),
                    'y': float(r['width_mean_mm']),
                    'x_mm': float(r['x_mm']),
                })

    @staticmethod
    def _resize(img):
        t = torch.from_numpy(img.astype(np.float32) / 255.0 - 0.5)
        return F.interpolate(t[None, None], size=SEM_SIZE,
                             mode='bilinear', align_corners=False)[0, 0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (torch.from_numpy(r['thermal'].copy()), self.sem_cache[r['sem_key']],
                torch.tensor([r['log_p']]), torch.tensor([r['y']]))

    def targets(self):
        return np.array([r['y'] for r in self.rows])

    def x_mm(self):
        return np.array([r['x_mm'] for r in self.rows])

    def thermal_mean_std(self):
        total, total_sq, n = 0.0, 0.0, 0
        for r in self.rows:
            w = r['thermal'].astype(np.float64)
            total += w.sum()
            total_sq += (w * w).sum()
            n += w.size
        mean = total / n
        return float(mean), float(np.sqrt(total_sq / n - mean * mean))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ThermalGRUEncoder(nn.Module):
    """Shared FrameEncoder per frame, then a single-layer GRU over the 11
    embeddings. Mean over GRU outputs (the window is symmetric around the
    prediction point, so no single directional read-out is privileged)."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.frames = FrameEncoder(embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=1, batch_first=True)

    def forward(self, x):                       # (B, T, H, W)
        b, t, h, w = x.shape
        emb = self.frames(x.reshape(b * t, 1, h, w)).reshape(b, t, -1)
        out, _ = self.gru(emb)
        return out.mean(dim=1)                  # (B, embed_dim)


class SEMEncoder(nn.Module):
    """Shallow 3-conv CNN over the 2-channel substrate crops."""

    def __init__(self, out_dim=SEM_DIM):
        super().__init__()
        channels = (2, 16, 32, out_dim)
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, c_out), c_out),
                       nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                       # (B, 2, H, W)
        return self.pool(self.net(x)).flatten(1)


class ConditionNet(nn.Module):
    """Fuse thermal, SEM and power branches into one condition vector."""

    def __init__(self):
        super().__init__()
        self.thermal = ThermalGRUEncoder()
        self.sem = SEMEncoder()
        self.power = nn.Sequential(nn.Linear(1, 8), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Linear(EMBED_DIM + SEM_DIM + 8, 64), nn.ReLU(inplace=True),
            nn.Linear(64, COND_DIM))

    def forward(self, thermal, sem, power_std):
        parts = [self.thermal(thermal), self.sem(sem), self.power(power_std)]
        return self.fuse(torch.cat(parts, dim=1))


class CVAE(nn.Module):
    """Standard CVAE core over a scalar target, conditioned on ConditionNet."""

    def __init__(self):
        super().__init__()
        self.condition = ConditionNet()
        self.encoder = nn.Sequential(nn.Linear(COND_DIM + 1, 32), nn.ReLU(inplace=True))
        self.mu_head = nn.Linear(32, LATENT_DIM)
        self.logvar_head = nn.Linear(32, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(COND_DIM + LATENT_DIM, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1))

    def forward(self, thermal, sem, power_std, y_std):
        cond = self.condition(thermal, sem, power_std)
        h = self.encoder(torch.cat([cond, y_std], dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterize
        recon = self.decoder(torch.cat([cond, z], dim=1))
        return recon, mu, logvar

    def decode_samples(self, cond, n_samples):
        """Predictive samples from the prior: decoder(cond, z), z ~ N(0, I)."""
        b = cond.shape[0]
        cond_rep = cond.repeat_interleave(n_samples, dim=0)
        z = torch.randn(b * n_samples, LATENT_DIM, device=cond.device)
        out = self.decoder(torch.cat([cond_rep, z], dim=1))
        return out.reshape(b, n_samples)


def beta_vae_loss(recon, y_std, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, y_std)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon_loss + beta * kl, recon_loss, kl


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def fold_loop(fold_i, train_tracks, val_track, run_dir, args, device):
    train_ds = CVAETrackDataset(run_dir, train_tracks)
    val_ds = CVAETrackDataset(run_dir, [val_track])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)          # forks OOM this 15GB box
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    th_mean, th_std = train_ds.thermal_mean_std()
    y_tr = train_ds.targets()
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    logp = np.array([r['log_p'] for r in train_ds.rows])
    p_mean, p_std = float(logp.mean()), float(logp.std())

    model = CVAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def prep(batch):
        thermal, sem, power, y = (t.to(device) for t in batch)
        return ((thermal - th_mean) / th_std, sem,
                (power - p_mean) / p_std, (y - y_mean) / y_std)

    best = {'val_mae': np.inf, 'state': None, 'epoch': -1}
    for epoch in range(args.epochs):
        model.train()
        tot, tot_rec, tot_kl, nb = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            thermal, sem, power, y = prep(batch)
            recon, mu, logvar = model(thermal, sem, power, y)
            loss, rec, kl = beta_vae_loss(recon, y, mu, logvar, args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item(); tot_rec += rec.item(); tot_kl += kl.item(); nb += 1

        # early stopping on the deterministic point prediction (z = 0)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                thermal, sem, power, _ = prep(batch)
                cond = model.condition(thermal, sem, power)
                z0 = torch.zeros(cond.shape[0], LATENT_DIM, device=device)
                preds.append(model.decoder(torch.cat([cond, z0], dim=1)).cpu().numpy())
        val_mae = mae_mm(val_ds.targets(),
                         np.concatenate(preds).ravel() * y_std + y_mean)
        print(f'  fold{fold_i} epoch {epoch:02d} loss {tot/nb:.4f} '
              f'(rec {tot_rec/nb:.4f} kl {tot_kl/nb:.4f}) val MAE {val_mae:.4f}',
              flush=True)
        if val_mae < best['val_mae']:
            best = {'val_mae': val_mae, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best['epoch'] >= args.patience:
            break

    # MC predictive distribution with the best weights
    model.load_state_dict(best['state'])
    model.eval()
    mus, sds = [], []
    with torch.no_grad():
        for batch in val_loader:
            thermal, sem, power, _ = prep(batch)
            cond = model.condition(thermal, sem, power)
            samples = model.decode_samples(cond, MC_SAMPLES) * y_std + y_mean
            mus.append(samples.mean(dim=1).cpu().numpy())
            sds.append(samples.std(dim=1).cpu().numpy())
    y_va, x_va = val_ds.targets(), val_ds.x_mm()
    mu, sd = np.concatenate(mus), np.concatenate(sds)
    const_mae = mae_mm(y_va, np.full_like(y_va, y_tr.mean()))
    return {'fold': fold_i, 'val_track': int(val_track),
            'best_epoch': best['epoch'], 'const_mae': const_mae,
            'y': y_va, 'x': x_va, 'mu': mu, 'sigma': sd}


def main():
    args = parse_args()
    setup_determinism(SEED)
    device = pick_device()
    commit, dirty = git_provenance()
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    run_dir = REPO / 'processed_data' / 'datasets' / args.dataset_run
    out_dir = Path(args.out_root) / args.dataset_run
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for fold_i, (train_tracks, val_track) in enumerate(loto_cv_splits(), start=1):
        assert val_track != HELD_OUT_TEST_TRACK
        assert HELD_OUT_TEST_TRACK not in train_tracks
        folds.append(fold_loop(fold_i, train_tracks, val_track, run_dir, args, device))

    metrics = {'per_fold': [], 'pooled': {}}
    print(f"\n{'fold':<12} {'MAE':>7} {'const':>7} {'NLL':>7} {'CRPS':>7} "
          f"{'cov50':>6} {'cov90':>6} {'best_ep':>8}")
    pooled = {k: [] for k in ('y', 'mu', 'sigma')}
    for f in folds:
        g = gaussian_metrics(f['y'], f['mu'], f['sigma'])
        in_rng = (f['x'] >= X_RANGE_T21[0]) & (f['x'] <= X_RANGE_T21[1])
        metrics['per_fold'].append({
            'fold': f['fold'], 'val_track': f['val_track'],
            'best_epoch': f['best_epoch'], 'const_mae': f['const_mae'],
            'full': g,
            'x29_99': gaussian_metrics(*(a[in_rng] for a in (f['y'], f['mu'], f['sigma']))),
        })
        for k in pooled:
            pooled[k].append(f[k])
        print(f"fold{f['fold']} v{f['val_track']:<3} {g['mae']:7.4f} {f['const_mae']:7.4f} "
              f"{g['nll']:7.3f} {g['crps']:7.4f} {g['coverage_50']:6.2f} "
              f"{g['coverage_90']:6.2f} {f['best_epoch']:8d}")
    gp = gaussian_metrics(*(np.concatenate(pooled[k]) for k in ('y', 'mu', 'sigma')))
    metrics['pooled'] = gp
    print(f"{'pooled':<12} {gp['mae']:7.4f} {'':>7} {gp['nll']:7.3f} {gp['crps']:7.4f} "
          f"{gp['coverage_50']:6.2f} {gp['coverage_90']:6.2f}")
    print('\nreference (same folds): const 0.2043 | anchor GP 0.2066 | '
          'phys_only 0.1384 | phys_gp 0.1419 (cov90 0.95 after inflation)')

    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    cfg = {'seed': SEED, 'beta': args.beta, 'epochs': args.epochs,
           'patience': args.patience, 'batch_size': args.batch_size, 'lr': args.lr,
           'latent_dim': LATENT_DIM, 'cond_dim': COND_DIM, 'embed_dim': EMBED_DIM,
           'sem_dim': SEM_DIM, 'sem_size': SEM_SIZE, 'mc_samples': MC_SAMPLES,
           'power_w': POWER_W, 'condition': 'continuous standardized log(P)'}
    (out_dir / 'provenance.json').write_text(json.dumps(
        {'commit': commit, 'dirty': dirty, 'dataset_run': args.dataset_run,
         'config_hash': config_hash(cfg), 'config': cfg,
         'deviations_from_spec': [
             'power as continuous scalar instead of nn.Embedding (cold-start)',
             'target = width_mean_mm scalar instead of height map (noise floor)'],
         }, indent=2, sort_keys=True))
    print(f'outputs -> {out_dir}')


if __name__ == '__main__':
    main()
