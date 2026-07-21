"""Shared sanity check for any script writing a final track-21 prediction CSV.

Guards against a silent extreme/degenerate value slipping into a submitted
result unnoticed -- the failure mode that hurt a past competition submission.
"""
import sys

import numpy as np

WIDTH_BOUNDS_MM = (0.0, 5.0)   # generous physical cap; dev-track widths are ~0.3-1.0mm
SIGMA_BOUNDS_MM = (0.0, 2.0)   # generous cap; inflated sigma is ~0.1-0.2mm in practice


def sanity_check(name, x, mu, sigma, width_bounds=WIDTH_BOUNDS_MM, sigma_bounds=SIGMA_BOUNDS_MM):
    """Hard-fails (raises) on NaN/Inf/non-monotonic x; soft-fails (prints a
    WARNING, still returns) on out-of-physical-range values, since those may
    be legitimate at the model-selection stage.
    """
    problems = []
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        raise SystemExit(f'{name}: NaN/Inf in predictions -- aborting, not writing output')
    if not np.all(np.diff(x) > 0):
        raise SystemExit(f'{name}: x_mm is not strictly increasing -- aborting')
    if np.any(mu < width_bounds[0]) or np.any(mu > width_bounds[1]):
        bad = mu[(mu < width_bounds[0]) | (mu > width_bounds[1])]
        problems.append(f'width_mm out of [{width_bounds[0]}, {width_bounds[1]}]: '
                        f'{len(bad)} values, range [{bad.min():.4f}, {bad.max():.4f}]')
    if np.any(sigma <= sigma_bounds[0]) or np.any(sigma > sigma_bounds[1]):
        bad = sigma[(sigma <= sigma_bounds[0]) | (sigma > sigma_bounds[1])]
        problems.append(f'width_sigma_mm out of ({sigma_bounds[0]}, {sigma_bounds[1]}]: '
                        f'{len(bad)} values, range [{bad.min():.4f}, {bad.max():.4f}]')
    if problems:
        for p in problems:
            print(f'WARNING [{name} sanity check]: {p}', file=sys.stderr)
    else:
        print(f'sanity check [{name}]: OK (width [{mu.min():.4f}, {mu.max():.4f}] mm, '
              f'sigma [{sigma.min():.4f}, {sigma.max():.4f}] mm)')
    return {'ok': not problems, 'problems': problems}
