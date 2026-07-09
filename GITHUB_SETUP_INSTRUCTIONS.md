# Creating the GitHub Repository

These steps create the repository on your GitHub profile and push the files.

## Option A: GitHub website + command line

1. Go to GitHub and sign in to the account where you want the repository.
2. Click **New repository**.
3. Repository name:

   ```text
   nsf-fmrg-data-challenge
   ```

4. Choose **Public** or **Private**.
5. Do **not** initialize with README, `.gitignore`, or license because those files are already included here.
6. Click **Create repository**.
7. On your computer, unzip the repository folder.
8. Open a terminal in the unzipped folder and run:

   ```bash
   git init
   git add .
   git commit -m "Initial NSF FMRG Data Challenge release"
   git branch -M main
   git remote add origin https://github.com/abhishekhanchate/nsf-fmrg-data-challenge.git
   git push -u origin main
   ```

## Option B: GitHub CLI

If you have the GitHub CLI installed and authenticated:

```bash
cd nsf-fmrg-data-challenge_github_release_v8
git init
git add .
git commit -m "Initial NSF FMRG Data Challenge release"
gh repo create abhishekhanchate/nsf-fmrg-data-challenge --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if you want the repository private.

## After pushing

1. Open the repository on GitHub.
2. Confirm the README renders correctly.
3. Confirm these files are visible:
   - `notebooks/01_participant_guide_data_loading_visualization_v3.ipynb`
   - `notebooks/00_colab_processing_visualization_videos_final_v4_STANDALONE.ipynb`
   - `src/nsf_fmrg_data.py`
   - `paper/nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.pdf`
   - `paper/nsf_fmrg_data_challenge_dataset_arxiv_v8_SETUP_REFS.tex`
4. If raw data files are too large for GitHub, upload them to a separate data-access location and add the access link to the README.
5. After arXiv submission, update the README and `CITATION.cff` with the arXiv identifier.

## Suggested first release

After the first push:

```bash
git tag -a v0.1.0 -m "Initial NSF FMRG Data Challenge release"
git push origin v0.1.0
```

Then on GitHub, create a release from tag `v0.1.0`.
