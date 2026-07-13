# Preprocessing plan — status as of 2026-07-13

## Environment
- Managed with `uv` (`pyproject.toml` + `uv.lock`, both tracked in git).
- `uv sync` recreates `.venv` with: numpy, scipy, matplotlib, pillow, pandas, h5py, ipykernel.
- In VS Code: open a notebook → "Select Kernel" → "Python Environments" → `.venv (Python 3.12)`.
- Raw data already downloaded locally into `data/raw/{thermal,sem,height_maps}/` (matches
  README layout). `data/*.zip` and `data/raw/**` are gitignored — not meant to be committed.

## Known bug already found
`notebooks/01_starter_code_loading_and_visualization.ipynb`, cell 2 (and the equivalent cell
in notebook 02) hardcodes `MANUAL_PROJECT_DIR = 'D:/Abhi Hanchate/NSF FMRG Data Challenge/'`
(the original author's Windows path). Must be changed to this repo's actual local path before
`from nsf_fmrg_data import ...` will resolve. Also: notebook 01 cell 10 references
`Z_detrended`/`height` before cell 11 defines them — cell order bug, not yet fixed.

## Design decisions made (see `.CLAUDE.md` §3 for the resulting invariants)
Read `paper/2607.07965v1.pdf` (only 4 pages) for the formal task definition — it is more
precise than the README:
- Task: model `p(g_i(x) | T_{i,t-k:t+k}, S_i, x)` — a local geometry descriptor at physical
  scan coordinate `x`, conditioned on a **window** of thermal frames (not a single frame),
  the SEM context image, and `x` itself.
- **Primary target = local width variation only** (boundary/contour/roughness are secondary,
  candidate alternative definitions — no need to predict all of them).
- **Output granularity: low-dimensional distributional statistics per sample** (e.g. width
  mean/std, boundary mean, edge-roughness stat) — NOT a dense/fine-resolution reconstruction
  of the full height-map cross-section. High-dim image input → low-dim probabilistic output.
- **Alignment grid = thermal frame** (~400 samples/track, one every 0.2 mm). Height-map data
  is ~50x finer → aggregate down per window. SEM tiles are ~32x coarser (6.41 mm/tile) →
  reused across ~32 consecutive frame-samples, with the track region cropped out (anti-leakage).
- **Split: Track 21 is a fixed held-out test set** (paper's own recommendation — Track 21 has
  the worst height-map NaN coverage, 55.5%, so it tests robustness rather than being tuned on).
  Cross-validation for model development runs over Tracks 8, 10, 14 only (3-fold leave-one-out).
- **NaN handling**: all 4 tracks have substantial height-map NaN fraction (37–56%, paper Table
  2). Any sample window whose NaN fraction exceeds a cutoff must be marked invalid and excluded,
  not silently zero-filled.

## Implementation tasks (tracked, not yet started — paused to configure environment)
1. `extract_local_width_stats` in `src/nsf_fmrg_data.py` — per-window boundary detection on
   the detrended height map, reusing the existing `largest_true_run` helper (same "longest
   run above a robust threshold" logic already used for thermal laser-on detection, applied
   in the cross-track/y direction instead of time). Must return NaN-fraction-aware validity.
2. `crop_sem_context` + `sem_tile_for_x` — mask out the assumed track band in a SEM tile
   (axis/fraction are a placeholder guess — **must be visually verified against a real SEM
   tile image once inspected**, no ground truth on track position within a tile yet) and map
   a physical x (mm) to a tile index (tile 01 = 100 mm side, highest tile = 20 mm side).
3. `loto_cv_splits` — Track 21 fixed held out; 3-fold CV generator over (8, 10, 14).
4. `build_track_samples` — assembles one track's full sample table: for each thermal frame,
   pairs a `T_{t-k:t+k}` window, the resolved SEM tile crop, and the aggregated height-map
   target stats.
5. `scripts/build_processed_dataset.py` — CLI that runs all 4 tracks, writes per-track sample
   tables + arrays into `processed_data/`, plus a provenance file (raw-file checksums, commit
   hash) per the `data_hash` invariant in `.CLAUDE.md`.

None of steps 1–5 have been written yet — session paused right before step 1 to set up `uv`
and fix the notebook path bug instead. Resume there.
