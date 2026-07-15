#!/usr/bin/env python
"""Build the full processed dataset (all 4 tracks) from raw data.

Writes one sample-table pickle per track plus a provenance.json recording
data_hash (raw Zenodo zip md5s) and commit_hash, per the data_hash invariant
in .CLAUDE.md section 3 and the provenance requirement in section 6.
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from preprocessing import DEV_TRACKS, HELD_OUT_TEST_TRACK, build_track_samples

ALL_TRACKS = (*DEV_TRACKS, HELD_OUT_TEST_TRACK)
RAW_ZIP_NAMES = ('thermal.zip', 'sem.zip', 'height_maps.zip')


def _md5(path, chunk_size=2 ** 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO_ROOT).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(REPO_ROOT / 'data'))
    parser.add_argument('--out-dir', default=str(REPO_ROOT / 'processed_data' / 'datasets'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--tracks', type=int, nargs='+', default=list(ALL_TRACKS))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    thermal_dir = data_dir / 'raw' / 'thermal'
    sem_dir = data_dir / 'raw' / 'sem'
    height_dir = data_dir / 'raw' / 'height_maps'

    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.out_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    dirty = bool(_git('status', '--porcelain'))
    if dirty:
        print('WARNING: working tree is dirty. This run is exploratory only, '
              'not decision-grade (.CLAUDE.md section 0). Commit before a real run.',
              file=sys.stderr)

    data_hash = {}
    for name in RAW_ZIP_NAMES:
        zip_path = data_dir / name
        if zip_path.exists():
            data_hash[name] = _md5(zip_path)
        else:
            data_hash[name] = None
            print(f'WARNING: {zip_path} not found, cannot checksum.', file=sys.stderr)

    provenance = {
        'run_tag': run_tag,
        'commit_hash': _git('rev-parse', 'HEAD'),
        'git_dirty': dirty,
        'data_hash': data_hash,
        'k': args.k,
        'tracks': {},
    }

    for track_id in args.tracks:
        print(f'building track {track_id}...')
        rows = build_track_samples(track_id, thermal_dir, sem_dir, height_dir, k=args.k)
        n_valid = sum(1 for r in rows if r['valid'])
        out_path = run_dir / f'track_{track_id}_samples.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  {len(rows)} samples ({n_valid} valid target stats) -> {out_path}')
        provenance['tracks'][track_id] = {'n_samples': len(rows), 'n_valid': n_valid}

    provenance_path = run_dir / 'provenance.json'
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    print(f'provenance -> {provenance_path}')


if __name__ == '__main__':
    main()
