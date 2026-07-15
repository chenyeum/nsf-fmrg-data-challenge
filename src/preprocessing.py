"""Our preprocessing code, built on top of the official `nsf_fmrg_data` starter module.

`nsf_fmrg_data.py` (lines 1-242) is official starter code (organizer upload d82b222)
and is never edited here, only imported from, so organizer patches stay merge-clean.
Everything in this file is ours.
"""
import re

import numpy as np

from nsf_fmrg_data import (
    COMMON_X_END_MM,
    COMMON_X_START_MM,
    THERMAL_MM_PER_FRAME,
    extract_final_thermal_frames,
    get_sem_tile_paths,
    largest_true_run,
    load_sem_tile,
    load_wyko_asc,
    robust_plane_detrend,
)

_WIDTH_STATS_DTYPE = [
    ('x_mm', 'f8'), ('valid', '?'), ('nan_frac', 'f8'),
    ('width_mean_mm', 'f8'), ('width_std_mm', 'f8'),
    ('boundary_left_mean_mm', 'f8'), ('boundary_right_mean_mm', 'f8'),
    ('edge_roughness_mm', 'f8'), ('n_cols_used', 'i8'),
]


def _column_track_boundary(col, y_mm, mad_k=3.0, min_valid_points=5):
    finite = np.isfinite(col)
    if finite.sum() < min_valid_points:
        return None, None, False

    med = np.median(col[finite])
    mad = 1.4826 * np.median(np.abs(col[finite] - med))
    thr = mad_k * max(mad, 1e-9)

    deviation = med - col
    mask = deviation > thr

    if not mask.any():
        return None, None, False

    track_y = y_mm[mask]
    left, right = float(track_y.min()), float(track_y.max())
    return left, right, True


def local_width_stats_at_window(Z_mm, x_mm, y_mm, x_center_mm, window_mm,
                                 nan_frac_max=0.6, min_valid_cols=5, mad_k=3.0):
    # 1. locate columns whose physical x falls inside this window
    col_mask = (x_mm >= x_center_mm - window_mm / 2) & (x_mm <= x_center_mm + window_mm / 2)
    col_indices = np.flatnonzero(col_mask)

    # 2. window falls outside the height map's covered x range
    if col_indices.size == 0:
        return {'x_mm': x_center_mm, 'valid': False, 'nan_frac': 1.0,
                'width_mean_mm': np.nan, 'width_std_mm': np.nan,
                'boundary_left_mean_mm': np.nan, 'boundary_right_mean_mm': np.nan,
                'edge_roughness_mm': np.nan, 'n_cols_used': 0}

    # 3. first gate: overall NaN fraction across the selected block
    block = Z_mm[:, col_indices]
    nan_frac = float(np.isnan(block).sum() / block.size)
    if nan_frac > nan_frac_max:
        return {'x_mm': x_center_mm, 'valid': False, 'nan_frac': nan_frac,
                'width_mean_mm': np.nan, 'width_std_mm': np.nan,
                'boundary_left_mean_mm': np.nan, 'boundary_right_mean_mm': np.nan,
                'edge_roughness_mm': np.nan, 'n_cols_used': 0}

    # 4. per-column boundary detection, keep only successful columns
    lefts, rights, widths = [], [], []
    for idx in col_indices:
        left, right, ok = _column_track_boundary(Z_mm[:, idx], y_mm, mad_k=mad_k)
        if ok:
            lefts.append(left)
            rights.append(right)
            widths.append(right - left)
    n_valid_cols = len(widths)

    # 5. second gate: enough columns must have succeeded to trust the average
    if n_valid_cols < min_valid_cols:
        return {'x_mm': x_center_mm, 'valid': False, 'nan_frac': nan_frac,
                'width_mean_mm': np.nan, 'width_std_mm': np.nan,
                'boundary_left_mean_mm': np.nan, 'boundary_right_mean_mm': np.nan,
                'edge_roughness_mm': np.nan, 'n_cols_used': n_valid_cols}

    # 6. aggregate per-column results into this window's summary stats
    lefts = np.array(lefts)
    rights = np.array(rights)
    widths = np.array(widths)
    return {
        'x_mm': x_center_mm,
        'valid': True,
        'nan_frac': nan_frac,
        'width_mean_mm': float(widths.mean()),
        'width_std_mm': float(widths.std()),
        'boundary_left_mean_mm': float(lefts.mean()),
        'boundary_right_mean_mm': float(rights.mean()),
        'edge_roughness_mm': float((lefts.std() + rights.std()) / 2),
        'n_cols_used': n_valid_cols,
    }


def extract_local_width_stats(Z_mm, x_mm, y_mm, x_centers_mm, window_mm,
                               nan_frac_max=0.6, min_valid_cols=5, mad_k=3.0):
    x_centers_mm = np.asarray(x_centers_mm)
    out = np.empty(len(x_centers_mm), dtype=_WIDTH_STATS_DTYPE)
    for i, xc in enumerate(x_centers_mm):
        r = local_width_stats_at_window(
            Z_mm, x_mm, y_mm, xc, window_mm,
            nan_frac_max=nan_frac_max, min_valid_cols=min_valid_cols, mad_k=mad_k,
        )
        out[i] = tuple(r[name] for name, _ in _WIDTH_STATS_DTYPE)
    return out


# --- Task 2: SEM anti-leakage crop + tile mapping -------------------------
SEM_BAND_REL_THRESH = 0.5
SEM_BAND_MIN_ROWS = 30
SEM_BAND_MARGIN_PX = 15
SEM_BAND_DENSITY_WINDOW = 21
SEM_BAND_DENSITY_THRESH = 0.5
SEM_BAND_MIN_PEAK_DENSITY = 0.9


def sem_tile_paths(sem_dir, track_id):
    """Official `get_sem_tile_paths`, re-sorted by trailing tile number.

    The official sort is lexicographic on filename and silently mis-orders
    SEM_14 (files 02-13 are misnamed `Scale_SEM_14_*.tif` but are actually
    plain images — see PREPROCESSING_PLAN.md). Sorting by the trailing
    integer in the stem is prefix-agnostic and fixes that without touching
    the official file.
    """
    paths = get_sem_tile_paths(sem_dir, track_id)
    return sorted(paths, key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))


def _row_roughness(img):
    """Per-row std of adjacent-pixel differences — low inside the smooth track band."""
    return np.diff(img.astype(np.float32), axis=1).std(axis=1)


def _smooth_row_density(mask, window):
    """Fraction of smooth rows in a sliding window centered on each row."""
    kernel = np.ones(window) / window
    return np.convolve(mask.astype(np.float64), kernel, mode='same')


def detect_track_row_band(img, rel_thresh=SEM_BAND_REL_THRESH, min_rows=SEM_BAND_MIN_ROWS,
                           density_window=SEM_BAND_DENSITY_WINDOW,
                           density_thresh=SEM_BAND_DENSITY_THRESH,
                           min_peak_density=SEM_BAND_MIN_PEAK_DENSITY):
    """Find the track band as the longest run of locally-dense-smooth rows.

    Relative threshold (row roughness < rel_thresh * median roughness), not
    `median - k*MAD`: the absolute rule failed on 20/53 tiles because
    substrate roughness spread varies too much tile to tile.

    A real band can have a handful of individually-rough rows inside it
    (debris, local imaging noise), which fragments a plain smoothness mask
    into several short runs — `largest_true_run` on the raw mask then grabs
    only the biggest fragment and often fails `min_rows`. To fix that
    without reintroducing the old min/max-extent bug (scattered noise
    stretching the band to near-full-image), we don't merge on gap size
    alone: we require local *density* of smooth rows to stay high. A
    genuine band keeps >~density_thresh of rows smooth within any
    density_window-sized neighborhood; scattered substrate noise doesn't.

    That density gate alone is still foolable: a cluster of short, noisy
    smooth streaks can nudge the windowed average over density_thresh
    without ever being a real band (empirically, on this dataset, that
    happens up to a peak density of ~0.86). A genuine band always contains
    a near-fully-smooth core (peak density >= ~0.95 in every confirmed real
    case here). So a second gate requires the candidate band to contain a
    peak density of at least min_peak_density, confirming a solid core
    exists rather than just an averaged-over cluster of noise.
    """
    r = _row_roughness(img)
    smooth_mask = r < (np.median(r) * rel_thresh)
    density = _smooth_row_density(smooth_mask, density_window)
    band_mask = density >= density_thresh
    row_start, row_stop = largest_true_run(band_mask)
    if row_start is None or (row_stop - row_start) < min_rows:
        return None, None, False
    if density[row_start:row_stop].max() < min_peak_density:
        return None, None, False
    return row_start, row_stop, True


def sem_tile_for_x(x_mm, track_id, sem_dir):
    """Map a physical x position to the SEM tile that covers it.

    Tile 01 sits at the COMMON_X_END_MM end; tile width is derived from the
    actual tile count for this track (13 for tracks 8/10/14, 14 for track
    21), not hardcoded, so it stays correct if a track's tile count differs.
    """
    paths = sem_tile_paths(sem_dir, track_id)
    n_tiles = len(paths)
    tile_width_mm = (COMMON_X_END_MM - COMMON_X_START_MM) / n_tiles
    index = int((COMMON_X_END_MM - x_mm) // tile_width_mm) + 1
    index = min(max(index, 1), n_tiles)  # x == COMMON_X_START_MM lands one past the end
    return index, paths[index - 1]


def crop_sem_context(img, row_start, row_stop, margin_px=SEM_BAND_MARGIN_PX):
    """Return SEM context rows with the track band (+ safety margin) excluded.

    Pure function: no neighbor-tile fallback here for invalid bands — that's
    Task 4's problem, since it needs to know about *other* tiles.
    """
    h = img.shape[0]
    top_edge = max(0, row_start - margin_px)
    bottom_edge = min(h, row_stop + margin_px)
    return {
        'top': img[:top_edge],
        'bottom': img[bottom_edge:],
        'band': (row_start, row_stop),
    }


# --- Task 4: build_track_samples -------------------------------------------
def _resolve_sem_band(track_id, tile_index, sem_dir, tile_paths=None):
    """Get a usable (row_start, row_stop) for a tile, borrowing from a neighbor if invalid.

    Tries the tile's own detection first. If invalid (e.g. a low-contrast
    end-tile — see PREPROCESSING_PLAN.md), searches outward by tile-index
    distance for the nearest same-track tile with a valid band and borrows
    its row range. The image actually cropped is still this tile's own —
    only the row coordinates are borrowed, on the assumption that the track's
    cross-track (row) position is roughly consistent tile to tile.

    Returns (row_start, row_stop, source_tile_index). source_tile_index
    equals tile_index when the tile's own detection was used, or the
    neighbor's index when borrowed. Returns (None, None, None) if no tile in
    the whole track has a valid band (shouldn't happen in practice).
    """
    if tile_paths is None:
        tile_paths = sem_tile_paths(sem_dir, track_id)
    n_tiles = len(tile_paths)

    own_img = load_sem_tile(tile_paths[tile_index - 1])
    row_start, row_stop, valid = detect_track_row_band(own_img)
    if valid:
        return row_start, row_stop, tile_index

    for radius in range(1, n_tiles):
        for neighbor in (tile_index - radius, tile_index + radius):
            if neighbor < 1 or neighbor > n_tiles:
                continue
            neighbor_img = load_sem_tile(tile_paths[neighbor - 1])
            row_start, row_stop, valid = detect_track_row_band(neighbor_img)
            if valid:
                return row_start, row_stop, neighbor
    return None, None, None


def build_track_samples(track_id, thermal_dir, sem_dir, height_dir, k=5):
    """Assemble one track's full sample table: one row per usable thermal frame.

    Each row pairs the frame's thermal window T_{t-k:t+k}, the resolved
    (anti-leakage-cropped) SEM context for that frame's physical x, and the
    Task-1 height-map target stats at that same x. Frames at either end of
    the track without a full +/-k window are dropped.

    Z_mm is plane-detrended before target-stat extraction: tracks have a
    real physical mounting tilt (confirmed empirically — e.g. track 8's
    untracked tilt shifts its median width estimate by >2x), which biases
    the per-column MAD threshold in `_column_track_boundary` if left in.
    """
    thermal = extract_final_thermal_frames(thermal_dir, track_id)
    frames = thermal['frames']
    x_mm_center = thermal['x_mm_center']
    n_frames = len(frames)

    height = load_wyko_asc(height_dir, track_id, crop_to_common=True)
    Z_mm, _ = robust_plane_detrend(height['Z_mm'], height['x_actual_mm'], height['y_mm'])
    x_actual_mm = height['x_actual_mm']
    y_mm = height['y_mm']

    tile_paths = sem_tile_paths(sem_dir, track_id)
    sem_img_cache = {}

    rows = []
    for t in range(k, n_frames - k):
        x_mm = float(x_mm_center[t])
        thermal_window = frames[t - k: t + k + 1]

        tile_index, tile_path = sem_tile_for_x(x_mm, track_id, sem_dir)
        row_start, row_stop, band_source = _resolve_sem_band(track_id, tile_index, sem_dir, tile_paths)
        if row_start is None:
            continue  # no valid band anywhere in this track (shouldn't happen)

        if tile_index not in sem_img_cache:
            sem_img_cache[tile_index] = load_sem_tile(tile_path)
        sem_context = crop_sem_context(sem_img_cache[tile_index], row_start, row_stop)

        target = local_width_stats_at_window(
            Z_mm, x_actual_mm, y_mm, x_mm, window_mm=THERMAL_MM_PER_FRAME,
        )

        rows.append({
            'track_id': track_id,
            'frame_index': t,
            'thermal_window': thermal_window,
            'sem_tile_index': tile_index,
            'sem_band_source_tile': band_source,
            'sem_context_top': sem_context['top'],
            'sem_context_bottom': sem_context['bottom'],
            'sem_band': sem_context['band'],
            **target,  # includes x_mm (== x_mm above, by construction), valid, nan_frac, width/boundary/roughness stats
        })

    return rows


# --- Task 3: loto_cv_splits --------------------------------------------------
HELD_OUT_TEST_TRACK = 21
DEV_TRACKS = (8, 10, 14)


def loto_cv_splits(dev_tracks=DEV_TRACKS):
    """Leave-one-track-out CV folds over the dev tracks (excludes the held-out test track).

    Yields (train_tracks, val_track) for each fold, e.g. with the default
    dev_tracks=(8, 10, 14): ((10, 14), 8), ((8, 14), 10), ((8, 10), 14).
    """
    for val_track in dev_tracks:
        train_tracks = tuple(t for t in dev_tracks if t != val_track)
        yield train_tracks, val_track
