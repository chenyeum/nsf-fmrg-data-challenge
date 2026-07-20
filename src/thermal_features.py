"""Hand-crafted melt-pool features from thermal windows (GP / tree baselines).

Per frame (raw camera counts, 400x400):
  - pool area above each fixed absolute threshold in AREA_THRESHOLDS
  - peak intensity
  - mean intensity of pixels above POOL_THRESHOLD
  - pool bounding-box aspect ratio (x extent / y extent)
  - pool centroid offset from frame center, x and y (pixels)
  - cooling-tail length: extent of pixels above TAIL_THRESHOLD behind the pool
    centroid along the scan axis. Empirically the tail trails in -x frame
    direction (pool centroid sits at ~(223, 253), warm region extends left);
    note the tail can clip at the frame border, which caps this feature.
Aggregated over the 11 frames of a window: mean and std of each -> 18 features,
plus the sample's physical x_mm -> 19 model inputs.

nan_frac and n_cols_used are deliberately NOT features: they proxy laser power
/ track identity through profilometer behavior (label-side leakage).

Track 21 is the held-out test set and is refused at loader level
(.CLAUDE.md section 3): `load_samples` raises on anything outside DEV_TRACKS.
"""
import pickle
from pathlib import Path

import numpy as np

from preprocessing import DEV_TRACKS

AREA_THRESHOLDS = (1500.0, 1800.0, 2100.0)
POOL_THRESHOLD = 1500.0
TAIL_THRESHOLD = 1200.0
FRAME_CENTER_XY = (200.0, 200.0)

_PER_FRAME_NAMES = [
    *(f'area_{int(t)}' for t in AREA_THRESHOLDS),
    'peak', 'mean_above_pool',
    'bbox_aspect', 'centroid_dx', 'centroid_dy', 'tail_len',
]
FEATURE_NAMES = [f'{stat}_{n}' for stat in ('mean', 'std') for n in _PER_FRAME_NAMES] + ['x_mm']


def frame_features(frame):
    frame = np.asarray(frame, dtype=np.float32)
    out = [float((frame > t).sum()) for t in AREA_THRESHOLDS]
    out.append(float(frame.max()))

    pool = frame > POOL_THRESHOLD
    if pool.any():
        ys, xs = np.nonzero(pool)
        out.append(float(frame[pool].mean()))
        out.append(float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)))
        cx, cy = float(xs.mean()), float(ys.mean())
        out.append(cx - FRAME_CENTER_XY[0])
        out.append(cy - FRAME_CENTER_XY[1])
        tail = frame > TAIL_THRESHOLD
        tys, txs = np.nonzero(tail)
        behind = txs[txs < cx]
        out.append(float(cx - behind.min()) if behind.size else 0.0)
    else:
        out.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return out


def window_features(thermal_window):
    per_frame = np.array([frame_features(f) for f in thermal_window])
    return np.concatenate([per_frame.mean(axis=0), per_frame.std(axis=0)])


def load_samples(dataset_dir, track_id):
    """Load one dev track's sample rows. Refuses non-dev tracks (Track 21)."""
    if track_id not in DEV_TRACKS:
        raise ValueError(
            f'track {track_id} is not a dev track (held-out test or unknown); '
            f'this loader only serves {DEV_TRACKS}')
    with open(Path(dataset_dir) / f'track_{track_id}_samples.pkl', 'rb') as f:
        return pickle.load(f)


def build_feature_cache(dataset_dir, out_path):
    """Extract features for all dev-track rows -> one npz, keyed by
    (track_id, frame_index). Also carries the target-side columns the model
    scripts need (width/width_std/n_cols_used/valid), so they never have to
    reload the multi-GB thermal pickles."""
    cols = {k: [] for k in ('track_id', 'frame_index', 'features', 'x_mm',
                            'width_mean_mm', 'width_std_mm', 'n_cols_used', 'valid')}
    for track_id in DEV_TRACKS:
        rows = load_samples(dataset_dir, track_id)
        for r in rows:
            cols['track_id'].append(r['track_id'])
            cols['frame_index'].append(r['frame_index'])
            cols['features'].append(window_features(r['thermal_window']))
            cols['x_mm'].append(r['x_mm'])
            cols['width_mean_mm'].append(r['width_mean_mm'])
            cols['width_std_mm'].append(r['width_std_mm'])
            cols['n_cols_used'].append(r['n_cols_used'])
            cols['valid'].append(r['valid'])
        del rows

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feature_names=np.array(FEATURE_NAMES),
        track_id=np.array(cols['track_id'], dtype=np.int64),
        frame_index=np.array(cols['frame_index'], dtype=np.int64),
        features=np.array(cols['features'], dtype=np.float64),
        x_mm=np.array(cols['x_mm'], dtype=np.float64),
        width_mean_mm=np.array(cols['width_mean_mm'], dtype=np.float64),
        width_std_mm=np.array(cols['width_std_mm'], dtype=np.float64),
        n_cols_used=np.array(cols['n_cols_used'], dtype=np.int64),
        valid=np.array(cols['valid'], dtype=bool),
    )
    return out_path
