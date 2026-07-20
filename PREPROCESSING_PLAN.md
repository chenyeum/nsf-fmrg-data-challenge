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

## Phase 2 — architecture comparison (2026-07-16): DONE, architecture is NOT the bottleneck

The open (a)/(b)/(c) decision above was resolved by testing (b) directly. Built and
committed (`7241303`):
- `src/temporal_cnn.py` — Step-2 factorized model: one small 2D `FrameEncoder`
  (4x stride-2 conv blocks, GroupNorm, ~60k params) weight-shared across the 11
  thermal frames, then a **swappable temporal module** over the 11 embeddings:
  `mean` (order-free control, 60.6k total) / `conv` (2x Conv1d over time, 85.3k) /
  `bilstm` (bidirectional LSTM, 85.7k). Head already sized for a future 5-target
  multi-task variant. Also now the canonical home of `mae_mm` (§3 single metric
  implementation — notebook 06 still carries an inline copy, migrate when touched).
- `scripts/train_temporal_cnn.py` — 3-fold LOTO CV harness: normalization fit on
  train tracks only (§3), runtime assert Track 21 never enters a fold, in-fold
  constant baseline, AdamW + cosine LR + early stopping (patience 15), CUDA
  determinism settings, writes `processed_data/model_runs/<tag>/results.json` with
  pinned `{config_hash, commit_hash, data_hash}` (§6).
- Notebooks 04 (pipeline walkthrough) and 05 (baseline) deleted — superseded by
  `build_processed_dataset.py` and the in-script baseline. 03 kept as the SEM-band
  research record; 06 kept as the SimpleCNN reference implementation.

**Results, all on dataset run `20260715_232715`, this Linux box (RTX 5060), best
val MAE per fold in mm:**

| fold (val) | baseline | SimpleCNN | mean | bilstm | conv |
|-----------|----------|-----------|--------|--------|--------|
| 1 (8)  | 0.2376 | 0.2187 | 0.2188 | 0.2195 | 0.2183 |
| 2 (10) | 0.1501 | 0.1458 | 0.1446 | 0.1446 | 0.1459 |
| 3 (14) | 0.2572 | 0.2463 | 0.2467 | 0.2497 | 0.2500 |
| **mean** | 0.2150 | 0.2036 | **0.2034** | 0.2046 | 0.2047 |

- `mean`/`bilstm` runs are **decision-grade** (clean tree at `7241303`), in
  `model_runs/20260716_001751_mean` and `20260716_002846_bilstm`.
- `conv` first attempt was **killed by the Linux OOM killer** (fold-1 epoch 6):
  `DataLoader(num_workers=4)` worker forks turn the ~8GB in-RAM dataset into real
  copies via refcount-triggered COW on this 15GB machine. Fixed with
  `num_workers=0` (loading isn't the bottleneck; rows are already in RAM and the
  workload is GPU-bound). The rerun (`model_runs/20260716_004754_conv`) completed
  clean but is formally **exploratory** — tree was dirty with the uncommitted
  one-liner. Seeded, so a post-commit rerun reproduces exactly.

**Reading (the important part):** four architectures — channel-stacked SimpleCNN,
order-free mean-pool, temporal conv, BiLSTM — all land on the same ~0.204mm plateau
within ±0.001mm. Order-aware models do NOT beat the order-free control, so temporal
structure is not where the missing signal is. The ceiling is in the data, not the
architecture. Corroborating detail: `mean`/`bilstm` hit best val at epochs 1–5;
`conv` trains more gradually (best at 22/11/13) but reaches the same endpoint.

## Plan for future (ordered by expected value; decided 2026-07-16)

> **Superseded same day** — Step 0 escalated into a full target redefinition and
> Step 4 was executed as a GP (not boosted trees); Step 5 was redesigned. See the
> two sections below. The skips (item 6) were re-confirmed empirically.

1. **Commit the `num_workers=0` one-liner** (only dirty change) so the conv run can
   be reproduced decision-grade.
2. **Step 0 — harden Task-1 targets. Now the top priority, and it blocks everything
   else.** The near-zero-width caveat (single noise pixel passing the MAD gate →
   width ~0.001mm in a ~0.4mm-median track) is label noise that puts a floor under
   val MAE larger than the differences we're trying to measure. Fix = minimum
   connected-run length per column in `_column_track_boundary` (flagged since Task 1).
   Core-chain change: §2/§4 probe + full dataset rebuild + rerun of at least the
   constant baseline and one model variant to re-anchor the numbers.
3. **Step 4 — gradient-boosted trees on hand-crafted melt-pool features** (max temp,
   area above threshold, centroid, elongation per frame x 11 ≈ ~50 features).
   Strong small-data baseline, seconds to train, and diagnostic: if it matches the
   CNNs' plateau, the extractable thermal signal is mostly pool statistics — which
   reframes all remaining model effort.
4. **Step 3 — multi-task head** (predict all 5 Task-1 stats, report width MAE only):
   cheap regularization test, the `TemporalCNN` head already supports `n_targets=5`.
5. **Step 5 — SEM branch, last, low expectations:** only ~13 distinct tiles/track
   (~26 distinct images in a 2-track fold) → an encoder will likely memorize
   per-tile offsets. Helps-in-train-but-not-LOTO = memorization signature.
6. **Skip:** more thermal-only architecture variants (settled by the comparison
   above), big pretrained backbones, 3D conv / transformers at this data size.

## Task-1 target redefinition (2026-07-16 afternoon): NaN-valley labels

Step 0 escalated. Investigating the 8-vs-10 median flip showed the **old width
target measured texture noise**: the track height signal (+4–8 µm) is ~15× below
the Wyko noise floor (per-column MAD 43–115 µm), so any below-median-depression
extraction returns noise. User-approved fix (root-cause, not patch):

- **New definition** (`src/preprocessing.py`, `_nan_valley_band`): the track is the
  **low-NaN valley** (4–15% NaN density) between rough high-NaN shoulders (60–85%).
  Per-row NaN density → coverage-normalized moving average (`VALLEY_SMOOTH_PX=25`)
  → shoulder level = median of outer quarters, floor = central minimum → require
  valley contrast ≥ `VALLEY_MIN_CONTRAST=0.25` (else invalid) → width = contiguous
  below-half-threshold run containing the floor. `width_std`/`edge_roughness` from
  `VALLEY_N_SUBBLOCKS=5` column sub-blocks; validity = valley detection succeeds
  (nan_frac recorded but no longer a gate). Old `_column_track_boundary` deleted.
- **Acceptance sweep v2** (`scripts/step0_acceptance_sweep.py`) all green:
  nan_frac byte-identical to old code, per-track medians 1.00/0.81/0.68/0.45 mm
  (ordering 8>10>14>21 now correct), width/SEM-band ratios 0.94–1.20, valid counts
  383/363/374/358 (+8/+13/+11/+22), deterministic.
- **5 residual narrow windows**: 3 weak-contrast marginal (t10 x=26.5/26.7/98.3),
  2 plausible early-scan transients (t8 x=22.3, t14 x=24.3). Decision pending:
  accept vs raise `VALLEY_MIN_CONTRAST` to ~0.45 (needs revalidation).
- New dataset build: `processed_data/datasets/20260716_151021` (exploratory).

## Phase 3 — probabilistic GP modeling on new labels (2026-07-16): the power law IS the model

All on dataset `20260716_151021`, all runs seeded + determinism-verified
(two runs → identical metrics.json), all **exploratory** (dirty tree).

- `src/thermal_features.py`: 18 hand melt-pool features (fixed thresholds
  1500/1800/2100, peak, mean-above-pool, bbox aspect, centroid offsets, tail
  length; mean+std over 11 frames) + `x_mm`; feature cache npz; track-21 blocked
  at loader level. `nan_frac`/`n_cols_used` deliberately excluded (leakage).
- **GP anchor** (`scripts/run_gp_baseline.py` → `results/gp_baseline/20260716_151021`):
  ARD-RBF + White kernel, heteroscedastic per-sample alpha, train-only
  standardization. Pooled MAE 0.2066 vs constant 0.2043 vs interp 0.2097 — the GP
  does **not** beat the constant baseline. Fold 2 (val 10 = interpolation) is well
  calibrated (cov90 0.94); folds 1/3 (extrapolation) collapse to the train mean
  and are overconfident (cov90 0.04/0.80). ARD length scales show x_mm
  memorization. Runtime 9:17.
- **A-phys** (`scripts/run_gp_physmean.py` → `results/gp_physmean/20260716_151021`):
  two-stage m(P)=a·P^b (weighted curve_fit, b∈[0.1,6]) + residual GP (anchor
  config minus x_mm). Power map **reversed vs task spec**, inferred from data
  {8:400, 10:350, 14:300, 21:200} W — inferred 2026-07-16, then
  ORGANIZER-CONFIRMED 2026-07-17 (a known participant confusion point; our
  data-driven inference preceded the confirmation — report material).
  Pooled MAE: anchor 0.2066 / **phys_only 0.1384** / phys_gp 0.1419 /
  phys_gp_xmm 0.1425. phys_gp cov90 0.71 pooled (fold1 0.51, fold3 0.85).
  Runtime 23:54 (over the 15-min budget; recorded). **Reading: the power trend
  carries all transferable signal; thermal features add ≈0 (slightly negative on
  extrapolation folds); x_mm adds nothing once the trend is explicit.**
  Per-fold b swings 0.65–2.07 (2 power points → exactly identified: trend anchor,
  not a validated law).
- **Within-track correlation screen**: all 18 features have |r| ≤ 0.2 vs width
  inside each track, no cross-track sign-consistent signal; within-track width std
  0.09–0.14 mm ≈ label noise. **Kills CNN-on-residuals, transformer, ResNet
  transfer as width predictors** — there is no within-track signal to learn.
  (Optional closure experiment if ever needed: frozen resnet18 embeddings +
  ridge probe on residuals, expected negative.)
- **Calibration dead end**: delta-method sd of m(P_val) from curve_fit pcov is
  0.0006–0.013 mm vs actual extrapolation bias 0.13–0.17 mm — parameter
  uncertainty is invisible next to model-form error, which 2 power points cannot
  estimate. Only honest fix identified: cross-fold empirical variance inflation
  (fold k inflated by the other folds' m-bias RMS). User decision pending.
- **3-track fit (final-model configuration)**: b=1.416, per-power residuals
  ≤ 0.023 mm (power-law shape actually holds across 300–400 W with 1 dof);
  extrapolated m(200 W)=0.3705 mm vs track-21 SEM band 0.379 mm (**−2.2%**) —
  independent external validation of the reversed power map and the final model.
- **README scope check** (user asked "is power-only too simple?"): the challenge
  is *probabilistic* local geometric variation from *multimodal* data; SEM is
  explicitly invited as a substrate-morphology **input** (track region masked).
  Old Step 5 redesigned: not an SEM encoder branch — extract substrate roughness
  features along x from SEM tiles, run the same within-track correlation screen,
  and only wire into the residual GP if signal exists. Richer targets
  (edge roughness, boundary positions) are cheap extensions already computed by
  preprocessing.

## Phase 3 closure experiments (2026-07-17): SEM negative, variance inflation works

Both run on dataset `20260716_151021`, exploratory (dirty tree),
determinism-verified.

- **SEM substrate screen** (`scripts/sem_substrate_screen.py`): tile-level
  (n≈13/track — within-tile column-to-x orientation never validated, so no
  sub-tile alignment assumed) substrate roughness/intensity features from the
  Task-2 anti-leakage crops, vs per-tile median width, within track. 38/39
  tiles usable (7 borrowed bands). **Negative across the board**: no feature
  sign-consistent across the 3 dev tracks (best single-track |r|=0.43 at n=13
  flips sign on the other tracks); pooled demeaned |r| ≤ 0.14, p ≥ 0.42.
  **Evidence chain closed: within-track width fluctuation is explained by
  neither process signal (thermal) nor substrate state (SEM) — it is
  measurement noise. SEM does not enter the residual GP; the README's
  SEM-as-input invitation was tested and declined on evidence.**
- **SEM screen v2, strip-level with official alignment**
  (`scripts/sem_substrate_screen_aligned.py`, 2026-07-17 evening): organizers'
  alignment rule (user-relayed): stitch tiles 01..N via real overlaps (~5%),
  no per-tile flip, then fliplr the whole mosaic → left-to-right = 20→100 mm,
  tile N at 20 mm. Implemented with per-pair cross-correlation overlap
  estimation (fallback to nominal 51 px when match corr < 0.3 — many pairs
  are weak, so sub-tile alignment carries ~±0.3 mm uncertainty), per-tile
  band detection medianed for the mosaic. 0.5 mm strips at every valid width
  window (n=1120 pooled). **Still negative**: no sign-consistent feature,
  pooled demeaned |r| ≤ 0.05, p ≥ 0.083. The tile-level negative was not an
  alignment artifact; SEM stays closed as an input.
- **Cross-fold variance inflation** (`scripts/run_gp_calib_inflation.py` →
  `results/gp_calib_inflation/20260716_151021`): fold k's predictive variance
  inflated by the other folds' mean squared stage-1 bias (sigma_new =
  sqrt(sigma² + v_k), points untouched). Refit reproduced A-phys stage-1
  biases exactly (−0.170/+0.029/−0.129 mm). Results: pooled cov90 0.71→0.95
  (fold1 0.51→0.91), NLL better on every fold (pooled −0.116→−0.349), CRPS
  0.0993→0.0958 — a genuine calibration improvement, not just wider bars.
  **cov50 stays low (pooled 0.35, fold1 0.03)**: inflation widens but cannot
  re-center; the center offset is power-law model-form error, unfixable with
  3 power levels — goes in the report as a stated limitation. Honesty note in
  provenance: within LOTO the inflation has unavoidable train/val overlap;
  for the track-21 submission the rule is fully clean (dev-fold biases only,
  track-21 labels never involved).

## Phase 4 (2026-07-17): CVAE probe → melt-pool LINEAR model becomes primary

User-requested CVAE (`scripts/run_cvae.py`, 112k params: FrameEncoder+GRU
thermal branch, shallow CNN on SEM crops, continuous standardized log(P)
condition — NOT nn.Embedding, cold-start; target = width scalar, not height
map — noise floor; beta-VAE loss). Results (`results/cvae/`): pooled MAE
0.1425 ≈ phys_gp but wildly unstable folds (0.209/0.153/0.064), calibration
unusable (fold2 NLL 34, best epoch 0). NOT for submission.

**But the fold-3 anomaly (MAE 0.064 ≈ noise floor on an extrapolation fold)
exposed a wrong earlier conclusion**: thermal features DO carry between-track
level signal (melt-pool area/peak/tail are monotone 8>10>14>21, t21 lowest —
yet another independent confirmation of the reversed power map). The anchor
RBF-GP couldn't use it because **RBF extrapolation reverts to the train
mean** — a kernel-structure limitation, not absent signal. The within-track
= 0 conclusion still stands; only the between-track half is revised.

`scripts/run_linear_baseline.py` (BayesianRidge, determinism-verified,
`results/linear_baseline/`): three variants + label-free t21 external check
(predict every t21 window from THERMAL INPUTS ONLY — user-sanctioned direct
pkl read, labels never touched — compare median vs SEM band 0.379 mm):

| variant | LOTO MAE | cov90+infl | t21 med | /SEM (dev 0.94–1.20) |
|---------|----------|-----------|---------|----------------------|
| *linear3 (sqrt(area), peak, tail) | **0.1189** | 0.92 | 0.456 | 1.20 borderline |
| area1d (raw area_1500) | 0.1354 | 0.97 | 0.393 | 1.04 pass |
| full18 (negative control) | 0.1374 | 0.98 | 1.154 | **3.04 FAIL** |

full18's failure is the lesson: collinear mixed-sign coefficients explode
under far extrapolation (t21 area_2100 is 27x out of range); LOTO cannot see
it because dev folds extrapolate mildly. Physics-constrained low-dim
features fix it. Selection honesty: 5 feature sets screened post-hoc against
LOTO + SEM check (noted in provenance and to be stated in the report).
Primary t21 per-window predictions saved to
`results/linear_baseline/<run>/t21_predictions_primary.npz`.

## Evaluation criteria (organizer clarification, user-relayed 2026-07-17)

Final ranking = QUANTITATIVE evaluation against the measured track-21 height
map (report/presentation assess justification, reproducibility,
interpretation, novelty — but do not replace the numbers). Minimum output:
local width w(x) = y_upper(x) − y_lower(x) as a SPATIAL sequence along the
scan direction (not a time series); boundary functions / centerline /
waviness are optional richer outputs. Scored on: (1) error vs measured local
width or boundaries; (2) fidelity of spatial variations along the track;
(3) overall geometry agreement; (4) calibration and usefulness of predicted
uncertainty (probabilistic models).

Power mapping ORGANIZER-CONFIRMED 2026-07-17: 400 W→8, 350 W→10, 300 W→14,
200 W→21 (a known participant confusion; matches our 2026-07-16 inference).

Same-evening strategy probes (exploratory):
- Within-track width variance is 80–90% high-frequency (5 mm smoothing keeps
  9–18%); smoothed feature-width correlations stay weak/inconsistent →
  criterion (2) ceiling is low; submit the SMOOTH component, do not emit
  uncorrelated per-window wiggles (smoothing predictions along x should
  improve error — verify in LOTO).
- One real, sign-consistent spatial structure: width drifts UP along scan on
  all three dev tracks (r(x, 5mm-smoothed width) = +0.42/+0.10/+0.34, thermal
  buildup signature) → test adding a linear x term to linear3 in LOTO.
- Centerline drifts 0.4–0.8 mm, smooth and strongly x-correlated per track
  but with per-track random sign (+0.71 vs −0.77) and an unknowable
  registration constant for t21 → boundary output needs the organizers'
  registration convention first; w(x) is the safe minimum deliverable.
- Open questions sent to organizers (pending): ground-truth extraction
  method + sampling Δx; metric for criterion (2); boundary registration
  convention; uncertainty submission format.

## Session 2026-07-18: code review + shift toward report writing

**Code review (user, line-by-line, one section at a time)**: fully reviewed
`src/thermal_features.py` and `scripts/run_linear_baseline.py`. No
correctness issues found. Two minor non-blocking notes surfaced:
- `frame_features()`'s no-pool-pixels branch zero-fills 5 features; possible
  edge case on t21's weakest frames (200 W, median area_1500 = 945, lowest of
  all tracks). Empirical check on cached dev features found **zero**
  anomalous rows (99th-pct std_peak = 202, no row > 2x that) — the same class
  of risk, checked, and clean on dev data. Optional follow-up: log the
  t21-else-branch trigger count in `predict_track21.py` as a diagnostic
  (cheap, non-blocking, not yet done).
- No connected-component filtering on pool/area pixels and `peak` uses raw
  `frame.max()` (both spatter-sensitive in principle) — accepted as
  documented simplifications, not changed (would require a full rerun for
  unverified benefit).
- Confirmed end-to-end: track-21 access in `t21_input_features()` reads only
  `thermal_window`/`x_mm`/`frame_index` from the raw pickle — no width/valid
  columns touched anywhere in that function.

**Literature check (web search, melt-pool thermal imaging + DED/LPBF ML)**:
ROI-cropping + CNN/ViT is the field's standard approach, but those studies
train across many process-parameter combinations and interpolate; our
3-power-level setup is pure extrapolation, where the linear feature route
wins regardless of architecture. Also: our threshold-blob feature extraction
(`frame_features`) is already an aggressive form of ROI reduction (176万
pixels → a handful of physical scalars), just without a trailing CNN —
folded into the "why not raw-pixel deep learning" report narrative.

**"Wave smoothing" discussion**: user's collaborator observed linear3's
output is nearly flat, smoothing over real-looking width waviness. Confirmed
this is expected given within-track signal = 0. Three lightweight, not-yet-run
diagnostics proposed to upgrade "measurement noise" from inference to
evidence (candidates for report figures, low cost, no model training):
1. Compare `width_std_mm` (within-window profilometer column spread) against
   within-track wave amplitude — if same order, wave ≈ repeat-measurement
   noise.
2. Lag-1 spatial autocorrelation of width residuals after removing per-track
   mean + x-trend — near-zero → white noise, confirms non-predictable.
3. Cross-instrument check: correlate demeaned SEM-band width wiggle against
   demeaned height-map width wiggle per dev track (SEM band read for
   diagnostics only, never as model input — output-leakage rule still
   applies to model inputs, not to noise-characterization analysis). Diluted
   by ±0.3 mm SEM/height-map alignment uncertainty, so a null result here is
   weak evidence, a positive correlation would be strong evidence.

**Strategic decision (user, 2026-07-18)**: linear3 is considered likely
near-final; further feature engineering is de-prioritized except the
already-queued x-trend LOTO test. Rationale agreed: within-track feature
engineering has been exhausted across two independent modalities (18 thermal
features + 2 rounds of SEM) with zero signal both times; MAE 0.1189 mm is
already at the within-track noise floor (0.09–0.14 mm); the generalization
axis has only 3 points, so new engineered features can't be validated beyond
curve-fitting 3 dev tracks. **Effort shifts to report writing next week** —
user emphasized report quality matters more than usual for this competition
specifically because there is no strict/precise ground truth (organizer
criteria explicitly weight justification/reproducibility/novelty alongside
the quantitative width-error number).

## Resume point (updated 2026-07-18)
1. **Tree dirty with two+ days' work**: `src/preprocessing.py` NaN-valley
   rewrite, `scripts/step0_acceptance_sweep.py`, `src/thermal_features.py`,
   `scripts/run_gp_baseline.py`, `scripts/run_gp_physmean.py`,
   `scripts/run_gp_calib_inflation.py`, `scripts/sem_substrate_screen.py`,
   `scripts/sem_substrate_screen_aligned.py` (official-alignment v2),
   `scripts/run_cvae.py`, `scripts/run_linear_baseline.py`, this plan update,
   `pyproject.toml`/`uv.lock` (scikit-learn), `results/`. Everything is
   exploratory until user reviews + commits; all runs are seeded and
   determinism-verified (CVAE: single seed, not re-verified — GPU run).
2. **Final track-21 recipe (user-approved 2026-07-17)**:
   PRIMARY = linear3 melt-pool BayesianRidge + cross-fold variance inflation
   (no power mapping dependency); CORROBORATION = phys_gp + inflation
   (independent evidence route via power law; both routes put t21 at
   0.37–0.46 mm). CVAE goes in the report as the discovery trigger + deep
   negative control, not in the submission.
3. **Review status (2026-07-18)**: `thermal_features.py` and
   `run_linear_baseline.py` fully reviewed, no blocking issues. Still to
   review by the user: `run_gp_physmean.py`, `run_gp_calib_inflation.py`,
   `sem_substrate_screen.py` / `_aligned.py`, `run_cvae.py`,
   `step0_acceptance_sweep.py`, `preprocessing.py` diff. Continuing
   2026-07-19, `run_gp_calib_inflation.py` next.
4. **Open user decision**: the 5 marginal narrow windows (accept vs raise
   `VALLEY_MIN_CONTRAST` to ~0.45 + revalidate).
5. **Remaining build**: `scripts/predict_track21.py` assembling the primary +
   corroboration outputs into the submission format (t21 per-window
   predictions already cached for the primary).
6. **De-prioritized**: further feature engineering (see 2026-07-18 decision
   above) — only the x-trend LOTO test remains queued; the 3 noise-vs-signal
   diagnostics above are optional report-figure material, not required.
7. **Report writing targeted for next week** — user flagged this as
   unusually high-value for this competition given the lack of strict
   ground truth; start drafting once review + commit + marginal-window
   decision are settled.
8. Machine note: this 15GB-RAM box cannot afford DataLoader worker forks with
   the in-RAM dataset — keep `num_workers=0` in any future training script.
