# Preprocessing plan — status as of 2026-07-14 (evening)

## Environment
- Managed with `uv` (`pyproject.toml` + `uv.lock`, both tracked in git).
- `uv sync` recreates `.venv` with: numpy, scipy, matplotlib, pillow, pandas, h5py, ipykernel.
- In VS Code: open a notebook → "Select Kernel" → "Python Environments" → `.venv (Python 3.12)`.
- **This machine is a fresh clone (2026-07-14).** Raw data was re-downloaded from Zenodo
  record 21285367 (~638 MB: thermal.zip 374M, height_maps.zip 199M, sem.zip 64M) and
  extracted into `data/raw/{thermal,sem,height_maps}/` matching the README layout.
  The zips are still in `data/` (gitignored, safe to delete). `data/*.zip` and
  `data/raw/**` are gitignored — not meant to be committed.

## Code ownership boundary (decided 2026-07-14)
- `src/nsf_fmrg_data.py` lines 1–242 are **official starter code** (organizer upload
  d82b222); only the Task-1 functions at lines ~245–330 are ours (commit 265df79).
- **Decision: stop adding to the official file.** All new code goes into a new module
  `src/preprocessing.py` that imports from `nsf_fmrg_data`. Official functions are
  wrapped, never edited — keeps the official diff minimal and organizer patches
  (they have shipped `PATCH_README.md` before) merge-clean.
- Optional later cleanup: migrate the Task-1 functions into `preprocessing.py` too
  (notebook imports would need updating). Not urgent.
- Division of labor for Task 2: **the user writes `src/preprocessing.py` himself**;
  Claude then runs the 53-tile acceptance sweep (criteria below).

## Known bugs / dataset quirks
- `notebooks/01_starter_code_loading_and_visualization.ipynb` cell 10 references
  `Z_detrended`/`height` before cell 11 defines them — cell order bug. Still unresolved,
  low priority. (The earlier `MANUAL_PROJECT_DIR` hardcoded-Windows-path bug was fixed.)
- **`SEM_14/PlainImages/` files 02–13 are misnamed `Scale_SEM_14_*.tif` but ARE plain
  images** (no annotation banner; verified visually and by md5 vs the real Scale files
  in `SEM_14/`). Official `get_sem_tile_paths` returns them in correct order only
  because `'Plain_' < 'Scale_'` and the single Plain file happens to be tile 01 —
  fragile. Fix lives in our wrapper (see Task 2), not in the official file.
- The outer `SEM_*/Scale_*.tif` files are annotated duplicates (1 mm scale bar, 47×,
  Texas A&M banner) — not model inputs; useful for pixel-size sanity checks.

## Design decisions made (see `.CLAUDE.md` §3 for the resulting invariants)
From `paper/2607.07965v1.pdf`:
- Task: model `p(g_i(x) | T_{i,t-k:t+k}, S_i, x)` — thermal frame **window** (not single
  frame) + SEM context (no time axis) + physical x → local geometry descriptor at x.
- Primary target = local width variation only (from the height map).
- Output granularity: low-dimensional distributional statistics per sample (width
  mean/std, boundary means, edge roughness) — not a dense cross-section reconstruction.
- Alignment grid = thermal frame (~400 samples/track, 0.2 mm/frame). Height-map columns
  are ~50x finer per frame-window; SEM tiles are ~32x coarser (6.41 mm/tile) and get
  reused across ~32 consecutive frame samples, with the track region masked out
  (anti-leakage).
- Split: Track 21 fixed held-out test set (paper's own recommendation — worst height-map
  NaN coverage, 55.5%). CV for development runs over Tracks 8, 10, 14 only (3-fold
  leave-one-out).
- NaN handling: per-window NaN fraction gate, invalid windows excluded, never zero-filled.
- Confirmed via detrended height maps: the laser track is a **depression** in the
  detrended height map — fixed the sign convention in `_column_track_boundary`.
- Organizer YouTube "official pipeline" cross-checked against the paper earlier; steps
  1–5 matched 1:1, "accuracy" reconciled as informal phrasing for prediction error (MAE).

## Task 1 — height-map width/boundary/roughness extraction: DONE + smoke-tested ✅
Implemented in `src/nsf_fmrg_data.py` (~245–330): `_column_track_boundary`,
`local_width_stats_at_window`, `extract_local_width_stats` (numpy structured array
output, `_WIDTH_STATS_DTYPE`). Design details unchanged from previous plan version.

**Smoke test ran 2026-07-14 on all four tracks** (400 windows each, window = 0.2 mm,
defaults `nan_frac_max=0.6, min_valid_cols=5, mad_k=3.0`):

| Track | valid windows | overall NaN (paper Table 2) | median width_mean |
|-------|--------------|------------------------------|-------------------|
| 8     | 378 (94.5%)  | 0.369 (0.369) ✅             | 0.41 mm |
| 10    | 351 (87.8%)  | 0.516 (0.516) ✅             | 0.47 mm |
| 14    | 364 (91.0%)  | 0.511 (0.511) ✅             | 0.57 mm |
| 21    | 337 (84.2%)  | 0.555 (0.555) ✅             | 1.30 mm |

- NaN fractions match paper Table 2 **exactly** → ASC loading, x-remap, and
  common-window crop confirmed correct.
- Track 21 (held-out test) is ~2–3× wider than the dev tracks — distributionally
  different, as the paper implies.
- **Open caveats (not yet acted on):** a few "valid" windows have near-zero widths
  (min ~0.001 mm ≈ a single noise pixel passing the MAD gate), and edge-roughness
  medians (0.19–0.56 mm) are large relative to width — the extent-based boundary can
  latch onto isolated below-threshold pixels. Candidate hardening: require a minimum
  number of connected mask points per column in `_column_track_boundary`. Decide before
  treating targets as final (at latest, before Task 4 assembly).

## Task 2 — SEM anti-leakage crop + tile mapping: design FINALIZED, implementation next
**Assumption validated 2026-07-14 on all 53 tiles (4 tracks):** the smoothest rows
(lowest per-row std of adjacent-pixel differences) always fall inside the visible track
band, contrast 2–4.8× vs substrate median — holds even on heavily spattered tiles.

**But the originally planned `med − r > 3·MAD` rule failed 20/53 tiles**, two modes:
1. *Zero detection* (8_01–04, 8_09–11, 10_10, 14_01, 14_12, 21_01): substrate roughness
   spread inflates MAD until the threshold drops below the band's actual roughness
   (e.g. 8_09: thr 1.71 < band r≈3.5 despite 4× contrast).
2. *Extent blowout* (8_13, 21_04, 21_11, 14_02, 14_09, 10_01): isolated smooth rows far
   from the band stretch the min/max extent to near-full image.

**Revised detection rule (agreed):**
- Threshold: relative, `r < median(r) / 2` (single source: module constant).
- Band: **largest contiguous run** of smooth rows via existing `largest_true_run`
  (`nsf_fmrg_data.py:93`) — opposite of the height-map extent choice, justified because
  clean tiles show fill = 1.00 (no NaN-style fragmentation on rows).
- Gate: run shorter than `SEM_BAND_MIN_ROWS` → invalid.

**To implement in new `src/preprocessing.py` (user writes; spec agreed):**
- Constants: `SEM_BAND_REL_THRESH = 0.5`, `SEM_BAND_MIN_ROWS = 30`,
  `SEM_BAND_MARGIN_PX = 15`.
- `sem_tile_paths(sem_dir, track_id)` — wraps official `get_sem_tile_paths`, re-sorts
  by trailing tile number (`int(re.search(r'(\d+)$', p.stem).group(1))`) — prefix-
  agnostic, fixes the SEM_14 misnaming fragility without touching official code.
- `_row_roughness(img)` — `np.diff(img.astype(np.float32), axis=1).std(axis=1)`.
- `detect_track_row_band(img, ...)` → `(row_start, row_stop, valid)`; handle
  `largest_true_run` returning `(None, None)`.
- `sem_tile_for_x(x_mm, track_id, sem_dir)` → `(index, path)`; tile 01 at the 100 mm
  end, width = 80 mm / actual tile count (13 for tracks 8/10/14, **14 for track 21**);
  clip index (x = 20.0 exactly lands one past the end).
- `crop_sem_context(img, row_start, row_stop, margin_px)` → expand band by margin, clip
  to image; return `{'top': ..., 'bottom': ..., 'band': (s, e)}`. Pure function — **no
  neighbor-tile fallback here**; invalid bands are the caller's problem (Task 4).

**Acceptance sweep (Claude runs after implementation), all 53 tiles must show:**
- Previous zero-detection tiles recover (8_09 band ≈ 283..445, 10_10 ≈ 332..439).
- Extent blowouts gone: 21_04/21_11 back to ≈262..350 (consistent with the rest of
  track 21); 8_13, 14_02, 14_09 back to normal band heights.
- Every valid band height in ~60–200 rows (≈0.5–1.6 mm, consistent with Task-1 widths).
- End tiles (8_01, 10_01, 14_01, 21_01 — rounded end-cap, band covers partial width,
  contrast diluted to ~1.8×) are **allowed to fail**; ≤1–2 invalid tiles per track.

## Task 3 — `loto_cv_splits`: NOT STARTED
Track 21 fixed held out; 3-fold leave-one-track-out CV generator over (8, 10, 14).
Goes in `src/preprocessing.py`.

## Task 4 — `build_track_samples`: NOT STARTED
Assembles one track's full sample table: pairs each thermal frame's `T_{t-k:t+k}`
window, the resolved (masked) SEM tile crop, and the Task-1 height-map target stats.
**Now also owns the end-tile fallback**: when `detect_track_row_band` is invalid for a
tile, borrow the band rows from the nearest valid neighboring tile of the same track.

## Task 5 — `scripts/build_processed_dataset.py`: NOT STARTED
CLI that runs all 4 tracks, writes per-track sample tables + arrays into
`processed_data/`, plus a provenance file (raw-file checksums, commit hash) per the
`data_hash` invariant in `.CLAUDE.md`. Note: the Zenodo zips' md5s are available from
the Zenodo API — candidate anchor for `data_hash`.

## Resume point
1. **User writes `src/preprocessing.py`** per the Task-2 spec above.
2. Claude runs the 53-tile acceptance sweep and reports against the criteria.
3. Then: decide the Task-1 hardening caveat (near-zero widths), then Tasks 3–5 in order.
