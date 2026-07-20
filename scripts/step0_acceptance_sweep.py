"""Step-0 acceptance sweep for the NaN-valley target redefinition.

Recomputes Task-1 target stats at every sample x of the reference dataset run
(20260715_232715) using the *current working-tree* preprocessing code, and
checks them against the stored targets plus external physics anchors. Only the
height-map path is exercised — thermal and SEM are untouched by Step 0.

Acceptance criteria (target REDEFINITION, so width values are expected to
change wholesale; validity may flip in both directions):
  1. per-window nan_frac byte-identical to the reference run wherever both
     runs computed it — nan_frac is still recorded the same way (paper
     Table 2 anchor). Windows the old code short-circuited before computing
     nan_frac (0-column case) are exempt.
  2. physics anchors: per-track median width ordering must be 8>10>14>21
     (SEM band widths / laser-power monotonicity), and the ratio of each
     track's median width to its SEM band width must land in [0.7, 1.4].
  3. no near-zero garbage: report every valid window with width < 25% of its
     track's median width (expected 0).
  4. blast radius visible: valid-count changes per track, both directions.

Run twice and diff stdout to confirm determinism before trusting a verdict.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from nsf_fmrg_data import THERMAL_MM_PER_FRAME, load_wyko_asc, robust_plane_detrend
from preprocessing import local_width_stats_at_window

REFERENCE_RUN = REPO / 'processed_data/datasets/20260715_232715'
HEIGHT_DIR = REPO / 'data/raw/height_maps'
TRACKS = [8, 10, 14, 21]

# SEM anti-leakage band widths in mm (independent modality; square-pixel
# assumption, tile span 80mm/n_tiles over 1024 px). Track+spatter zone, so a
# true melt-track width should land near but not far above/below it.
SEM_BAND_MM = {8: 0.895, 10: 0.781, 14: 0.721, 21: 0.379}
SEM_RATIO_RANGE = (0.7, 1.4)


def main():
    failures = []
    medians = {}
    residual_near_zero = []

    for tid in TRACKS:
        rows = pickle.load(open(REFERENCE_RUN / f'track_{tid}_samples.pkl', 'rb'))
        h = load_wyko_asc(HEIGHT_DIR, tid, crop_to_common=True)
        Z_mm, _ = robust_plane_detrend(h['Z_mm'], h['x_actual_mm'], h['y_mm'])

        new = [local_width_stats_at_window(
                   Z_mm, h['x_actual_mm'], h['y_mm'], r['x_mm'],
                   window_mm=THERMAL_MM_PER_FRAME)
               for r in rows]

        # 1. nan_frac recording must be untouched (where old code computed it)
        nan_drift = [i for i, (r, n) in enumerate(zip(rows, new))
                     if r['nan_frac'] < 1.0
                     and not np.isclose(r['nan_frac'], n['nan_frac'], rtol=0, atol=0)]
        if nan_drift:
            failures.append(f'track {tid}: nan_frac drifted on {len(nan_drift)} '
                            f'windows (first at idx {nan_drift[0]})')

        # 4. validity flips, both directions (allowed, reported)
        old_valid = np.array([r['valid'] for r in rows])
        new_valid = np.array([n['valid'] for n in new])
        to_invalid = int(np.sum(old_valid & ~new_valid))
        to_valid = int(np.sum(~old_valid & new_valid))

        new_w = np.array([n['width_mean_mm'] for n in new if n['valid']])
        med = float(np.median(new_w))
        medians[tid] = med

        # 3. residual near-zero widths
        near_zero = [(tid, n['x_mm'], n['width_mean_mm'])
                     for n in new if n['valid'] and n['width_mean_mm'] < 0.25 * med]
        residual_near_zero.extend(near_zero)

        # 2b. magnitude anchor vs SEM band
        ratio = med / SEM_BAND_MM[tid]
        if not (SEM_RATIO_RANGE[0] <= ratio <= SEM_RATIO_RANGE[1]):
            failures.append(f'track {tid}: median/SEM ratio {ratio:.2f} outside '
                            f'{SEM_RATIO_RANGE}')

        print(f'track {tid:>2}: n={len(rows):3d}  valid {int(old_valid.sum())} -> '
              f'{int(new_valid.sum())}  (->invalid: {to_invalid}, ->valid: {to_valid})')
        print(f'          nan_frac drift: {len(nan_drift)} windows'
              + ('' if not nan_drift else '  FAIL'))
        print(f'          width median={med:.4f}  p5={np.percentile(new_w, 5):.4f}  '
              f'p95={np.percentile(new_w, 95):.4f}  min={new_w.min():.4f}mm')
        print(f'          vs SEM band {SEM_BAND_MM[tid]:.3f}mm: ratio={ratio:.2f}')
        print(f'          residual valid windows < 25% of track median: {len(near_zero)}')

    # 2a. ordering anchor
    order = sorted(medians, key=medians.get, reverse=True)
    print(f'\nwidth ordering by median: {" > ".join(map(str, order))}'
          f'  (SEM/power expects 8 > 10 > 14 > 21)')
    if order != [8, 10, 14, 21]:
        failures.append('track width ordering does not match SEM/power monotonicity')
    if residual_near_zero:
        failures.append(f'{len(residual_near_zero)} residual near-zero-width windows')

    print('VERDICT: ' + ('; '.join(failures) if failures else 'clean'))
    for tid, x, w in residual_near_zero[:20]:
        print(f'  track {tid} x={x:.2f}mm width={w:.4f}mm')


if __name__ == '__main__':
    main()
