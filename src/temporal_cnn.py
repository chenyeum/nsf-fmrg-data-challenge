"""Step-2 model (PREPROCESSING_PLAN.md Phase 2): shared per-frame CNN encoder
+ swappable temporal module over the 11-frame thermal window.

Why factorized: one 2D encoder weight-shared across frames trains on ~8k
individual frames per fold instead of ~700 channel-stacked samples, and frame
order becomes an explicit, ablatable choice via the temporal module:

  'mean'   — order-free mean pool (control: if this matches 'conv', temporal
             structure isn't where the signal is)
  'conv'   — two 1D convs over the time axis (default bet: direction-neutral,
             matching the symmetric T_{t-k:t+k} window, and deterministic)
  'bilstm' — bidirectional LSTM readout (recurrent alternative to compare)

`mae_mm` here is the canonical metric implementation (.CLAUDE.md §3 "metric has
a single implementation"). Notebooks 05/06 predate this file and still carry
inline copies — migrate them to this import when they are next touched.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

TARGET_KEY = 'width_mean_mm'


def mae_mm(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


class TrackSampleDataset(Dataset):
    """Valid-target rows from one or more tracks of a processed dataset run.

    Returns raw (unnormalized) thermal windows: standardization belongs to the
    training loop, because mean/std must be fit on the fold's train tracks only
    (.CLAUDE.md §3) and the dataset can't know which fold it's serving.
    """

    def __init__(self, run_dir, track_ids, target_key=TARGET_KEY):
        self.rows = []
        for track_id in track_ids:
            with open(Path(run_dir) / f'track_{track_id}_samples.pkl', 'rb') as f:
                track_rows = pickle.load(f)
            self.rows.extend(r for r in track_rows if r['valid'])
        self.target_key = target_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        thermal = torch.from_numpy(row['thermal_window'].copy())
        target = torch.tensor(row[self.target_key], dtype=torch.float32)
        return thermal, target

    def targets(self):
        return np.array([r[self.target_key] for r in self.rows], dtype=np.float64)

    def thermal_mean_std(self):
        """Global mean/std over every pixel of every thermal window.

        One global pair, not per-frame-index stats: per-frame stats would
        contradict the shared encoder's premise that every frame is an
        exchangeable input to one appearance model. float64 accumulation.
        """
        total, total_sq, n = 0.0, 0.0, 0
        for row in self.rows:
            w = row['thermal_window'].astype(np.float64)
            total += w.sum()
            total_sq += (w * w).sum()
            n += w.size
        mean = total / n
        var = total_sq / n - mean * mean
        return float(mean), float(np.sqrt(var))


class FrameEncoder(nn.Module):
    """Shared per-frame 2D encoder: (N, 1, H, W) -> (N, embed_dim).

    GroupNorm rather than BatchNorm: the flattened (B*11) batch holds 11
    highly correlated frames per sample, which biases batch statistics.
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        channels = (1, 16, 32, 64, embed_dim)
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.blocks(x)).flatten(1)


class MeanPoolTemporal(nn.Module):
    """Order-free control: mean over the frame axis. (B, T, D) -> (B, D)."""

    def __init__(self, embed_dim=64):
        super().__init__()

    def forward(self, seq):
        return seq.mean(dim=1)


class ConvTemporal(nn.Module):
    """Two 1D convs over the time axis, then mean pool. (B, T, D) -> (B, D)."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, seq):
        return self.net(seq.transpose(1, 2)).mean(dim=2)


class BiLSTMTemporal(nn.Module):
    """Bidirectional LSTM readout, mean over per-step outputs. (B, T, D) -> (B, D).

    Bidirectional because the input window T_{t-k:t+k} is symmetric around the
    prediction point — a one-directional read would impose an arbitrary time
    direction. Hidden size embed_dim//2 per direction keeps output embed_dim.
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2,
                            batch_first=True, bidirectional=True)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return out.mean(dim=1)


TEMPORAL_MODULES = {
    'mean': MeanPoolTemporal,
    'conv': ConvTemporal,
    'bilstm': BiLSTMTemporal,
}


class TemporalCNN(nn.Module):
    """Shared frame encoder + temporal module + linear head.

    Input: (B, T, H, W) thermal window stack. Output: (B,) when n_targets == 1
    (matching SimpleCNN's contract, so training loops are interchangeable),
    else (B, n_targets) — head is already sized for the Step-3 multi-task
    variant (n_targets=5).
    """

    def __init__(self, temporal='conv', embed_dim=64, n_targets=1, dropout=0.3):
        super().__init__()
        self.encoder = FrameEncoder(embed_dim)
        self.temporal = TEMPORAL_MODULES[temporal](embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, n_targets))
        self.n_targets = n_targets

    def forward(self, x):
        b, t, h, w = x.shape
        feat = self.encoder(x.reshape(b * t, 1, h, w))
        seq = feat.reshape(b, t, -1)
        out = self.head(self.temporal(seq))
        return out.squeeze(-1) if self.n_targets == 1 else out
