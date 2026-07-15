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

## Code ownership boundary (decided 2026-07-14, migration done 2026-07-15)
- `src/nsf_fmrg_data.py` is now **official starter code only** (organizer upload
  d82b222) — the Task-1 functions have been moved out.
- `src/preprocessing.py` holds all of our own code: `_WIDTH_STATS_DTYPE`,
  `_column_track_boundary`, `local_width_stats_at_window`, `extract_local_width_stats`
  (moved 2026-07-15, unchanged logic — verified functionally identical), plus all
  future Task 2–5 code. It imports from `nsf_fmrg_data` rather than editing it, so
  organizer patches (they have shipped `PATCH_README.md` before) stay merge-clean.
- No notebook imported the Task-1 functions directly (confirmed by repo-wide grep),
  so the migration required no notebook updates.
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
Implemented in `src/preprocessing.py` (moved from `nsf_fmrg_data.py` 2026-07-15):
`_column_track_boundary`, `local_width_stats_at_window`, `extract_local_width_stats`
(numpy structured array output, `_WIDTH_STATS_DTYPE`). Design details unchanged from
previous plan version.

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

## Task 2 — DONE 2026-07-15 (revised beyond the original spec above)
First acceptance sweep with the plain `largest_true_run` rule above found a *third*
failure mode not caught during design: a real band can internally **fragment** into
several short runs (a few individually-rough rows inside an otherwise-real band — e.g.
`21_05`, `8_10`), which `largest_true_run` alone either shrinks or invalidates. Fixed
with two additional layers, both implemented in `src/preprocessing.py`:
- `_smooth_row_density(mask, window)` — a sliding-window moving average of the raw
  smoothness mask (`SEM_BAND_DENSITY_WINDOW = 21`), thresholded at
  `SEM_BAND_DENSITY_THRESH = 0.5` before taking `largest_true_run`. Bridges small
  internal gaps in a real band.
- Density-threshold alone is foolable (scattered noise can nudge a windowed average
  over 0.5 without ever being a real band — found on `10_02`, a genuine end-tile that
  got a false-positive ~34-row "band"). Fixed with a second gate,
  `SEM_BAND_MIN_PEAK_DENSITY = 0.9`: the candidate band must contain a peak density
  near 1.0 (a genuinely solid smooth core) somewhere inside it, not just an averaged-up
  cluster. Threshold chosen from real data — every confirmed real band peaks >= 0.952,
  the worst noise false-positive peaks at 0.857, a clean gap.

Full investigation (roughness/density plots, the fragmentation and false-positive
examples, the peak-density scatter used to pick 0.9) is in
`notebooks/03_task2_sem_band_detection_investigation.ipynb`.

**Final 53-tile sweep: 45/53 valid.** Invalid tiles: `8_01/02/03`, `10_01/02/03`,
`14_01`, `21_01`. Tracks 14/21 lose only their true end-cap (matches the original
expectation); tracks 8/10 lose their end-cap **plus 2 more tiles each** — checked, not a
detection bug: those extra tiles' best candidate bands have peak density 0.52–0.71,
clearly noise, not borderline. Reading: contrast dilution near the rounded end-cap
extends ~3 tiles deep (~19mm) on tracks 8/10, vs. ~1 tile on 14/21 — a real per-track
physical difference. All invalid tiles are handled by Task 4's neighbor-tile fallback.

## Task 3 — `loto_cv_splits`: DONE 2026-07-15
Implemented in `src/preprocessing.py`. `HELD_OUT_TEST_TRACK = 21`, `DEV_TRACKS = (8,
10, 14)`. `loto_cv_splits()` yields `(train_tracks, val_track)` per fold: `((10,14),
8)`, `((8,14), 10)`, `((8,10), 14)`. Verified output matches.

## Task 4 — `build_track_samples`: implemented 2026-07-15, smoke-tested on track 8
Assembles one track's full sample table: pairs each thermal frame's `T_{t-k:t+k}`
window, the resolved (masked) SEM tile crop, and the Task-1 height-map target stats.
Implemented in `src/preprocessing.py`. Decisions made:
- `k=5` (11 thermal frames, ≈2.2mm window) for the thermal input; frames at either end
  of a track without a full window are dropped (~390 samples/track instead of ~400).
- Target stats keep Task-1's narrower `window_mm=THERMAL_MM_PER_FRAME` (0.2mm, one
  frame's width) — deliberately much narrower than the thermal input window, matching
  the paper's `p(g_i(x) | T_{t-k:t+k}, S_i, x)`: wide context in, precise local target
  out.
- Samples store full arrays inline (thermal window stack, SEM top/bottom crops), not
  just references — one row per sample, so the table is a list of dicts (SEM crop
  shapes vary tile to tile, doesn't fit a fixed-dtype structured array like Task 1's).
- **`Z_mm` is plane-detrended (`robust_plane_detrend`) before Task-1 stats are
  computed.** Confirmed empirically (compared detrended vs. raw Task-1 output against
  the already-recorded Task-1 smoke-test medians — detrended matches closely on all 4
  tracks, e.g. track 8: raw gives 0.21mm median width, detrended gives 0.44mm, recorded
  smoke test was 0.41mm) **and confirmed from provenance**: `robust_plane_detrend` is
  official code (commit `d82b222`, the organizer upload), and the official starter
  notebook detrends before *every* height-map plot it makes — never plots raw `Z_mm`.
  Note: don't confuse this with `display_shear_grid` (also official) — that one is
  explicitly commented "display-only... does not resample the height values," a purely
  cosmetic plotting-grid shear, not a data correction; not used in our pipeline.
  Track 8 turned out to have the *largest* cross-track tilt of all four tracks
  (`slope_y = 0.0083`), despite not being in the notebook's `SELECTED_SLOPE_EFF` list
  (which only tracked 10/14/21) — that list was incomplete for our purposes.
- **Now owns the end-tile fallback**: `_resolve_sem_band(track_id, tile_index, ...)` —
  when a tile's own `detect_track_row_band` is invalid, searches outward by tile-index
  distance for the nearest same-track tile with a valid band and borrows its row range
  (still crops the original tile's own image, just with borrowed row coordinates).

**Smoke test, all 4 tracks (2026-07-15): DONE.**

| Track | samples | valid target stats | used SEM fallback |
|-------|---------|---------------------|---------------------|
| 8     | 390     | 375 (96.2%)         | 87                  |
| 10    | 390     | 350 (89.7%)          | 87                  |
| 14    | 390     | 363 (93.1%)          | 26                  |
| 21    | 390     | 336 (86.2%)          | 24                  |

Valid-target-stats rates track closely with Task 1's earlier overall NaN-fraction
smoke test (8: 94.5%, 10: 87.8%, 14: 91.0%, 21: 84.2% valid windows) — consistent.
Fallback counts scale with each track's known invalid-tile count (tracks 8/10 have 3
invalid SEM tiles each -> ~87 borrowed samples; tracks 14/21 have 1 each -> ~24-26).

## Task 5 — `scripts/build_processed_dataset.py`: DONE 2026-07-15
CLI that runs all 4 tracks via `build_track_samples`, writes one
`track_<id>_samples.pkl` per track to `processed_data/datasets/<run_tag>/`, plus
`provenance.json` with `commit_hash`, `git_dirty`, and `data_hash` (md5 of each raw
Zenodo zip in `data/`) — satisfies the `.CLAUDE.md` §3 data_hash invariant and §6
provenance requirement. Warns to stderr and sets `git_dirty: true` if the working tree
isn't clean (per §0, such a run is exploratory, not decision-grade).

**Full run 2026-07-15 (exploratory — working tree was dirty):** all 4 tracks, 390
samples each, valid counts matched the earlier per-track smoke tests exactly (8: 375,
10: 350, 14: 363, 21: 336). Total output 11GB (thermal windows are `(11,400,400)`
float32 ≈ 6.7MB/sample, dominates size — expected, per the "full arrays inline"
storage-strategy decision in Task 4). Pickle round-trip verified.

**All 5 preprocessing tasks now implemented.** Not yet done: a *clean* (non-dirty)
decision-grade run once this code is committed; the Task-1 near-zero-width hardening
caveat; the track-21 SEM-band-width open question (see Task 2).

## Preprocessing: complete (2026-07-15)
All 5 tasks implemented and smoke-tested on all 4 tracks. `scripts/build_processed_dataset.py`
produced a full run (`processed_data/datasets/20260715_145326/`, ~11GB, one pickle per
track + `provenance.json` with `commit_hash`/`data_hash`/`git_dirty`). That run was
**exploratory** — working tree was dirty at the time — not decision-grade.

Still-open items from preprocessing, not yet acted on: the Task-1 near-zero-width
hardening caveat, and why track 21's SEM band heights run narrower in pixels than
tracks 8/10/14 despite having the widest height-map track (see Task 2's sweep results
above).

## Phase 2: model training (started 2026-07-15)

**Verified against the paper directly** (`paper/2607.07965v1.pdf` §5): Track-21-held-out
+ cross-track validation is the paper's own recommendation, and MAE on local width is
explicitly listed as a candidate metric — our approach matches on both counts. The
specific 3-fold leave-one-track-out mechanism (`loto_cv_splits`, Task 3) is our own
reasonable implementation of "cross-track validation," not a literal paper requirement.
Also cross-checked Table 2's NaN fractions against our own Task-1 smoke test — exact
match, another independent confirmation the height-map loading/alignment is correct.

**`.CLAUDE.md` §2/§4 protocol now followed strictly from this phase onward** (it was
missed for Tasks 1-4 earlier in the session — the operating contract wasn't in context
for that part of the conversation; caught mid-session via a direct grep of the file).

Notebooks added (all in `notebooks/`, all auto-detect the repo root by searching
upward for `pyproject.toml` — portable across machines, not just this Mac):
- **05_baseline_pipeline_test.ipynb**: constant-baseline harness (`ConstantBaseline`
  predicts the train-set target mean, no gradient descent) — validates the dataset
  loading / CV-split routing / DataLoader batching / metric computation end to end,
  independent of whether any real model can learn. All 3 CV folds ran cleanly.
  Baseline val MAE per fold: `val=8: 0.2376mm, val=10: 0.1501mm, val=14: 0.2572mm`.
- **06_cnn_full_training.ipynb**: first real (gradient-trained) model —
  `SimpleCNN` (35,809 params, 3 strided conv layers + global avg pool + linear head),
  treats the 11 thermal frames in `T_{t-k:t+k}` as input channels (not temporally
  aware), predicts `width_mean_mm` only. Adam lr=1e-3, L1 loss, 20 epochs, batch 16.
  SEM context and the other 4 target stats (boundary positions, edge roughness) are
  **not yet used** by this model.

**Full training results (3 folds x 20 epochs), reproduced independently by the user
on their own run — numbers matched to 4 decimal places:**

| fold | val_track | n_train | n_val | baseline val MAE | CNN best val MAE | best epoch | CNN final val MAE | improvement |
|------|-----------|---------|-------|-------------------|-------------------|------------|---------------------|-------------|
| 1 | 8  | 713 | 375 | 0.2376mm | 0.2187mm | 19 | 0.2227mm | 7.9% |
| 2 | 10 | 738 | 350 | 0.1501mm | 0.1458mm | 10 | 0.1504mm | 2.8% |
| 3 | 14 | 725 | 363 | 0.2572mm | 0.2463mm | 13 | 0.2529mm | 4.2% |

Reading: CNN beats the baseline in all 3 folds (consistent direction, real signal) but
by a modest margin (~5% average). `best_epoch` varies a lot per fold (19/10/13, no
stable "sweet spot") and `final_val_mae` stays close to `best_val_mae` in every fold —
val loss isn't diverging/overfitting by epoch 20, it's just plateauing early. This
small thermal-only model has found a low ceiling, not a bug.

**Speed** (this Mac, Apple MPS backend): ~100ms/batch, ~160 samples/sec, ~9.1s/epoch
(45 batches, 713 train samples). Device selection checks CUDA first, then MPS, then
CPU — same code should pick up CUDA automatically on a Linux GPU machine.

**Linux/CUDA portability fixes made 2026-07-15** (after finding hardcoded
`/Users/C.Y./nsf-fmrg-data-challenge` paths in the pre-existing notebooks 01/02):
- All 6 notebooks: replaced the fragile `Path.cwd().name == 'notebooks'` heuristic with
  a real upward search for `pyproject.toml` — robust to whatever directory a Jupyter
  server was actually launched from (matters on a remote Linux box).
- `uv.lock` already resolves proper Linux+CUDA wheels for torch (confirmed by
  inspection — `nvidia-cudnn`, `nvidia-nccl`, `cuda-toolkit` etc. are pinned as
  `sys_platform == 'linux'`-conditional dependencies), so `uv sync` on a Linux machine
  needs no extra index/config.
- Notebooks 05/06: `DataLoader` now gets `pin_memory=True, num_workers=4` when
  `device.type == 'cuda'` (no-op elsewhere); notebook 06 also seeds
  `torch.cuda.manual_seed_all(0)` alongside `torch.manual_seed(0)` for CUDA
  reproducibility.
- **Bug caught while doing this**: `ConstantBaseline().to(device)` followed by
  `.fit()` would have silently reset the parameter back to CPU, since `.fit()` created
  a fresh `torch.tensor(...)` with no device argument. Fixed by pinning it to
  `self.value.device`. Re-ran notebook 05 end-to-end after the fix — identical results
  to before the fix, confirming correctness on MPS; not yet run on an actual Linux CUDA
  machine.

## Resume point (updated 2026-07-15)
1. **Open decision, not yet made**: where to push next on the CNN — options discussed
   but not chosen: (a) bring in the SEM modality (built in Task 4, unused by the model
   so far), (b) give the thermal input real temporal structure instead of stacking 11
   frames as plain channels, (c) just tune training (more epochs, LR schedule) on the
   current thermal-only setup to see if the ~5% ceiling moves. User was heading out to
   review progress and decide next time.
2. `notebooks/06_cnn_full_training.ipynb` is a **clean, unexecuted** notebook (source
   only, no baked outputs) — intended to be run by the user themselves, possibly on a
   different (Linux/CUDA) machine, to watch it train live.
3. Nothing has been committed this session — working tree is still dirty. A clean,
   decision-grade training run (per `.CLAUDE.md` §0) would need a commit first.
