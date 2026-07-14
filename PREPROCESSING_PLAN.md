# Preprocessing plan — status as of 2026-07-14

## Environment
- Managed with `uv` (`pyproject.toml` + `uv.lock`, both tracked in git).
- `uv sync` recreates `.venv` with: numpy, scipy, matplotlib, pillow, pandas, h5py, ipykernel.
- In VS Code: open a notebook → "Select Kernel" → "Python Environments" → `.venv (Python 3.12)`.
- Raw data already downloaded locally into `data/raw/{thermal,sem,height_maps}/` (matches
  README layout). `data/*.zip` and `data/raw/**` are gitignored — not meant to be committed.

## Known bug already found (status: unresolved, low priority)
`notebooks/01_starter_code_loading_and_visualization.ipynb` cell 10 references
`Z_detrended`/`height` before cell 11 defines them — cell order bug. Not yet fixed. (The earlier
`MANUAL_PROJECT_DIR` hardcoded-Windows-path bug was already fixed by the user.)

## Design decisions made (see `.CLAUDE.md` §3 for the resulting invariants)
From `paper/2607.07965v1.pdf`:
- Task: model `p(g_i(x) | T_{i,t-k:t+k}, S_i, x)` — thermal frame **window** (not single frame) +
  SEM context (no time axis) + physical x → local geometry descriptor at x.
- Primary target = local width variation only (from the height map).
- Output granularity: low-dimensional distributional statistics per sample (width mean/std,
  boundary means, edge roughness) — not a dense cross-section reconstruction.
- Alignment grid = thermal frame (~400 samples/track, 0.2 mm/frame). Height-map columns are
  ~50x finer per frame-window; SEM tiles are ~32x coarser (6.41 mm/tile) and get reused across
  ~32 consecutive frame samples, with the track region masked out (anti-leakage).
- Split: Track 21 fixed held-out test set (paper's own recommendation — worst height-map NaN
  coverage, 55.5%). CV for development runs over Tracks 8, 10, 14 only (3-fold leave-one-out).
- NaN handling: per-window NaN fraction gate, invalid windows excluded, never zero-filled.
- Confirmed (verbally reported, not paper-sourced) via `robust_plane_detrend`'d height-map data:
  the laser track is a **depression** (lower than surrounding substrate baseline) in the
  detrended height map — this fixed the sign convention used in `_column_track_boundary` below.
- Note: an "official pipeline" summary from an organizer YouTube video was cross-checked against
  the paper. Steps 1–5 matched existing design 1:1. Step 6 said "accuracy" as an evaluation
  metric, which was reconciled as informal phrasing for prediction error (MAE), not literal
  classification accuracy — F1/precision/recall are not computable for a continuous regression
  target, so this does not change the probabilistic-regression framing already established.

## Task 1 — height-map local width/boundary/roughness extraction: DONE
Implemented in `src/nsf_fmrg_data.py` (verified present on disk, lines ~245–330):

- `_column_track_boundary(col, y_mm, mad_k=3.0, min_valid_points=5)` — private helper. For one
  height-map column (fixed x, ~480 y-values): gates on minimum finite-point count; computes a
  robust baseline (median) and spread (1.4826 * MAD) from finite values only; builds a
  **depression-direction** mask `(med - col) > thr` directly over the full column (NaN entries
  evaluate to False in the comparison, so no separate NaN-index bookkeeping is needed); takes
  the **min/max extent** of the masked y-values as left/right boundary (deliberately not a
  contiguous-run definition, since NaN gaps inside the track would fragment a strict run).
  Returns `(left_mm, right_mm, valid_bool)`.

- `local_width_stats_at_window(Z_mm, x_mm, y_mm, x_center_mm, window_mm, nan_frac_max=0.6,
  min_valid_cols=5, mad_k=3.0)` — selects height-map columns whose *physical* x_mm falls in
  the window (never raw array index); gates on empty selection, then on overall NaN fraction
  of the selected block, then (after running `_column_track_boundary` per column) on the count
  of columns that actually produced a valid boundary; aggregates width mean/std, boundary
  left/right means, and edge roughness (`(std(lefts) + std(rights)) / 2`) across valid columns.
  Returns a dict with fixed keys in both the valid and invalid case (invalid numeric fields are
  `np.nan`, not `None` — required for the numpy structured-array batch output below).

- `extract_local_width_stats(Z_mm, x_mm, y_mm, x_centers_mm, window_mm, ...)` — batch wrapper,
  loops over one track's thermal-frame x-centers (from
  `extract_final_thermal_frames(...)['x_mm_center']`), writes into a preallocated numpy
  structured array (`_WIDTH_STATS_DTYPE`, defined near the top of the file next to the other
  module constants) rather than a pandas DataFrame — chosen deliberately over pandas since the
  row count per track (~400) makes performance irrelevant and the goal was to avoid the extra
  dependency; field access afterward works the same way (`result['width_mean_mm']`,
  boolean-index with `result['valid']`).

Not yet done within task 1: no track has actually been run end-to-end through
`extract_local_width_stats` yet to sanity-check the `valid` ratio against the paper's Table 2
NaN fractions (8: 0.369, 10: 0.516, 14: 0.511, 21: 0.555) — worth doing as a first smoke test
before moving on to model input assembly.

## Task 2 — SEM anti-leakage crop + tile mapping: IN PROGRESS
Two independent functions planned: `sem_tile_for_x(x_mm, track_id)` (pure lookup, not started)
and `crop_sem_context(...)` (image-content-dependent, design in progress, not coded yet).

Visually confirmed against a real tile (`data/raw/sem/SEM_8/PlainImages/Plain_SEM_8_07.tif`,
1024×768 px, width=scan direction/x, height=cross-track/y): the laser track renders as a
**smooth horizontal band roughly across the vertical middle of the image**, in visible contrast
to a rough/speckled substrate texture above and below it. This directly informed the intended
design (agreed with the user, not yet implemented):

- Per-row "roughness" score (e.g. std of adjacent-pixel differences within the row) — track
  rows should score low (smooth), substrate rows high (speckled/textured).
- Same median+MAD robust-threshold skeleton as `_column_track_boundary`, but on rows instead
  of columns, and masking on "roughness below baseline" instead of "height below baseline".
- Track row band = min/max extent of masked rows (same extent-not-contiguous-run reasoning
  as the height-map case, for robustness).
- `crop_sem_context` then excludes/masks that row band, keeping only the top+bottom context
  regions as the anti-leakage SEM input feature.

Open question before coding, flagged to the user but not yet resolved: whether the
"track is smoother than substrate" contrast holds across other tiles/tracks (only tile 07 of
track 8 has been visually checked so far) — worth spot-checking 1–2 more tiles (e.g. a tile
from track 21, and one from near either end of a track) before finalizing the roughness metric,
in case surface oxidation/cracking on some tracks makes the track region look rougher than
usual, not smoother.

`sem_tile_for_x` design note: tile 01 = 100 mm side, highest-numbered tile = 20 mm side (index
direction flipped, same as the height-map ASC x remap). Track 8/10/14 have 13 tiles; Track 21
has 14. Because 13 × 6.41 mm ≈ 83.3 mm doesn't line up exactly with the 80 mm common window,
plan is to derive each tile's x-boundaries from the track's *actual* tile count (via
`get_sem_tile_paths`), not from the fixed 6.41 mm nominal width.

## Tasks 3–5: NOT STARTED
3. `loto_cv_splits` — Track 21 fixed held out; 3-fold CV generator over (8, 10, 14).
4. `build_track_samples` — assembles one track's full sample table: pairs each thermal frame's
   `T_{t-k:t+k}` window, the resolved (masked) SEM tile crop, and the height-map target stats
   from task 1.
5. `scripts/build_processed_dataset.py` — CLI that runs all 4 tracks, writes per-track sample
   tables + arrays into `processed_data/`, plus a provenance file (raw-file checksums, commit
   hash) per the `data_hash` invariant in `.CLAUDE.md`.

## Resume point
Next concrete step: either (a) run `extract_local_width_stats` on a real track as a smoke test,
or (b) spot-check 1–2 more SEM tiles to confirm the roughness-contrast assumption, then
implement `_row_roughness`/track-row detection + `crop_sem_context` following the same layered
pattern as task 1. Either is a reasonable next session starting point.
