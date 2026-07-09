# NSF Future Manufacturing Data Challenge

This repository contains the starter code, notebooks, paper source, and documentation for the **NSF Future Manufacturing Data Challenge**.

The challenge focuses on predicting probabilistic local geometric variation of single laser tracks in directed energy deposition (DED) using multimodal data:

- in-situ thermal image sequences,
- SEM images of surrounding substrate morphology,
- Bruker/Wyko full-field height maps.

This competition and associated material are based upon work supported by the National Science Foundation under Grant Number **FMRG-2328395**.

## Repository structure

```text
nsf-fmrg-data-challenge/
├── README.md
├── DATA_USE_LICENSE.md
├── CITATION.cff
├── requirements.txt
├── data/
│   └── raw/
│       ├── thermal/
│       ├── sem/
│       └── height_maps/
├── notebooks/
│   ├── 00_colab_processing_visualization_videos_final_v4_STANDALONE.ipynb
│   └── 01_participant_guide_data_loading_visualization_v3.ipynb
├── src/
│   └── nsf_fmrg_data.py
├── scripts/
│   └── run_thermal_video_export.py
└── paper/
    ├── nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.tex
    ├── nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.pdf
    ├── nsf_fmrg_overleaf_v8_SETUP_REFS_source.zip
    └── figures/
```

## Data layout

Place the raw data using the following structure:

```text
data/raw/thermal/
  Thermal_8.mat
  Thermal_10.mat
  Thermal_14.mat
  Thermal_21.mat

data/raw/sem/
  SEM_8/PlainImages/
  SEM_10/PlainImages/
  SEM_14/PlainImages/
  SEM_21/PlainImages/

data/raw/height_maps/
  Heightmap_8.ASC
  Heightmap_10.ASC
  Heightmap_14.ASC
  Heightmap_21.ASC
```

The raw data may be distributed separately from GitHub if file sizes are too large. In that case, keep the same folder structure after downloading or receiving access.

## Physical coordinate conventions

### Thermal data

- File type: `.mat`
- Native frame size: `400 × 400`
- Pixel size: approximately `14 µm/pixel`
- Field of view: approximately `5.6 mm × 5.6 mm`
- Frame rate: `50 fps`
- Scan speed: `10 mm/s`
- Travel per thermal frame: `0.2 mm/frame`
- The 20–100 mm analysis window contains approximately `400` thermal frames.
- Thermal files include frames before laser turn-on and after laser shutoff. The processing notebook detects laser shutoff and extracts the previous 400 frames.

### SEM data

- File type: `.tif`
- Images are stored as per-track tiles in `PlainImages`.
- Tile 01 corresponds to the physical 100 mm side.
- The highest-numbered tile corresponds to the physical 20 mm side.
- The participant starter notebook reads SEM tiles but does not stitch them.
- SEM images should be used to characterize surrounding substrate morphology. Avoid using the processed track region directly as an input feature to prevent output leakage.

### Bruker/Wyko height maps

- File type: Wyko ASCII `.ASC`
- `x` and `y` values are stored in millimeters.
- `z` values are stored in nanometers and converted to millimeters or micrometers in the code.
- Raw ASC local `x = 0` corresponds to the physical 100 mm side.
- The loader sorts columns so returned height maps increase from 20 mm to 100 mm in actual part coordinates.

## Notebooks

### Organizer/testing notebook

Use this notebook when checking data, generating paper figures, extracting thermal frames, and exporting thermal videos:

```text
notebooks/00_colab_processing_visualization_videos_final_v4_STANDALONE.ipynb
```

This notebook is fully standalone and does **not** depend on `src/`.

It saves results to:

```text
processed_data/run_outputs/<YYYYMMDD_HHMMSS>/
```

### Participant starter notebook

Use this notebook as the clean starting point for participants:

```text
notebooks/01_participant_guide_data_loading_visualization_v3.ipynb
```

This notebook demonstrates:
- thermal loading and 20–100 mm extraction,
- SEM tile loading,
- Bruker/Wyko height-map loading,
- basic physical-coordinate visualization,
- optional display-only tilt inspection.

## Paper

The companion arXiv/Overleaf source is in:

```text
paper/nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.tex
```

Compiled preview:

```text
paper/nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.pdf
```

The Overleaf-ready source bundle is:

```text
paper/nsf_fmrg_overleaf_v8_SETUP_REFS_source.zip
```

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The notebooks are designed to run in Google Colab. For local use, a standard scientific Python environment with NumPy, SciPy, Matplotlib, and Pillow is sufficient.

## Citation

If you use this dataset or code outside the NSF Future Manufacturing Data Challenge, cite the dataset paper and this repository.

```bibtex
@misc{hanchate2026nsffmrgdedchallenge,
  title        = {NSF Future Manufacturing Data Challenge: A Multimodal DED Dataset for Probabilistic Local Geometry Prediction in Laser Tracks},
  author       = {Hanchate, Abhishek and Balhara, Himanshu and Bukkapatnam, Satish T. S.},
  year         = {2026},
  note         = {NSF Future Manufacturing Data Challenge dataset and code repository},
  howpublished = {\url{https://github.com/abhishekhanchate/nsf-fmrg-data-challenge}}
}
```

## License and data-use terms

See [`DATA_USE_LICENSE.md`](DATA_USE_LICENSE.md).

Challenge use is permitted for registered participants. Any use outside the NSF Future Manufacturing Data Challenge must cite the dataset paper and repository.
