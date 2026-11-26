"""Bounded-memory binned Gaussian risk and Kish effective support.

Squared IPCW contributions are accumulated per patient before smoothing; using
the squared sum in each bin would incorrectly treat co-located people as one.
"""
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter


def support_coverage(effective, density_mask, prior_risk_mask, ix, iy,
                     thresholds=(0, 5, 10, 20, 25, 50, 100), minimum_count=5):
    """Aggregate spatial coverage AND actual patient coverage; no IDs exported."""
    cells = int(density_mask.sum())
    n = len(ix)
    result = {}
    for threshold in thresholds:
        visible = density_mask & prior_risk_mask & (effective >= threshold)
        count = int(visible[ix, iy].sum())
        # Suppress count/fraction if either count or its complement is small.
        reportable = all(value == 0 or value >= minimum_count for value in (count, n-count))
        prefix = f'ess_{threshold}'
        result[prefix+'_grid_cells'] = int(visible.sum())
        result[prefix+'_grid_fraction'] = float(visible.sum()/cells) if cells else np.nan
        result[prefix+'_patients'] = count if reportable else np.nan
        result[prefix+'_patient_fraction'] = float(count/n) if n and reportable else np.nan
    return result


def binned_ipcw_surface(ix, iy, targets, weights, *, grid, sigma):
    ix, iy = np.asarray(ix), np.asarray(iy)
    targets, weights = np.asarray(targets, float), np.asarray(weights, float)
    if not (ix.shape == iy.shape == targets.shape == weights.shape):
        raise ValueError('Risk surface inputs must have identical shapes')
    if sigma <= 0 or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Invalid kernel or IPCW weights')
    numerator, denominator, squared = (np.zeros((grid, grid)) for _ in range(3))
    np.add.at(numerator, (ix, iy), weights * targets)
    np.add.at(denominator, (ix, iy), weights)
    np.add.at(squared, (ix, iy), weights ** 2)
    numerator = gaussian_filter(numerator, sigma, mode='constant')
    denominator = gaussian_filter(denominator, sigma, mode='constant')
    # Match scipy's default truncated Gaussian kernel exactly, then square it.
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    squared = convolve1d(squared, kernel ** 2, axis=0, mode='constant')
    squared = convolve1d(squared, kernel ** 2, axis=1, mode='constant')
    risk = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    effective = np.divide(denominator ** 2, squared, out=np.zeros_like(squared), where=squared > 0)
    return risk, effective
