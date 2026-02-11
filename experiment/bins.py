"""Module for producing and working with histogram bins."""

import warnings

import numpy as np
import tensorflow as tf


def get_bins(n_bins, pad_ratio, sig_ratio, low=0.0, high=1.0):
    """Return the histogram bins given the HL parameters.
    Produces n_bins bins with pad_ratio * sig_ratio * bin_width padding on each side.

    Produces bins with width according to the following equation:
        w = (high - low) / (n_bins - 2 * pad_ratio * sig_ratio)

    Note: The low/high params must be broadcastable to the target shape to use the
    bins directly for HL-Gaussian. If the dimensions are a subset of the target shape,
    you must add the appropriate axes with tf.expand_dims before using.

    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side
        sig_ratio - the ratio of sigma to bin width
        low - the lower bounds of the histogram support; shape (x1, ..., xn)
        high - the upper boudns of the histogram support; same shape as low

    Returns:
        borders - a Tensor of shape (n_bins + 1, x1, ..., xn) of bin borders
        sigma - the sigma to use for HL-Gaussian of shape (x1, ..., xn)
    """
    bin_width = (high - low) / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(low - pad_width, high + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma


def _make_strictly_increasing(values, min_step=1e-6):
    """Ensure a 1-D array is strictly increasing by nudging ties upward."""
    out = np.asarray(values, dtype=np.float32).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + min_step
    return out


def get_quantile_bins(y_values, n_bins, min_step=1e-6, padding=0.0, low=0.0, high=1.0):
    """Return quantile-based bin borders and centers for 1-D targets.

    Params:
        y_values - array-like of target values (assumed scaled to [0, 1])
        n_bins - number of bins
        min_step - minimum spacing to enforce strictly increasing borders
        padding - extra support on each side as a fraction of (high - low)
        low - minimum of the target range (default 0.0)
        high - maximum of the target range (default 1.0)

    Returns:
        borders - np.ndarray of shape (n_bins + 1,)
        centers - np.ndarray of shape (n_bins,)
    """
    y = np.asarray(y_values, dtype=np.float32).reshape(-1)
    y = np.clip(y, low, high)
    if len(y) < n_bins:
        warnings.warn(
            f"get_quantile_bins: n_bins={n_bins} exceeds number of samples={len(y)}; "
            "quantile borders may be duplicated.",
            RuntimeWarning,
        )
    uniq = np.unique(y)
    if len(uniq) < n_bins + 1:
        warnings.warn(
            f"get_quantile_bins: only {len(uniq)} unique targets for n_bins={n_bins}; "
            "quantile borders will repeat and be nudged by min_step.",
            RuntimeWarning,
        )

    span = high - low
    if padding <= 0.0 or span <= 0.0:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        borders = np.quantile(y, qs)
        borders = _make_strictly_increasing(borders, min_step=min_step)
        centers = 0.5 * (borders[:-1] + borders[1:])
        return borders, centers

    # Estimate tail bin widths from full-quantile bins
    qs_full = np.linspace(0.0, 1.0, n_bins + 1)
    borders_full = np.quantile(y, qs_full)
    borders_full = _make_strictly_increasing(borders_full, min_step=min_step)
    w_left = max(float(borders_full[1] - borders_full[0]), min_step)
    w_right = max(float(borders_full[-1] - borders_full[-2]), min_step)

    pad_width = padding * span
    k_left = max(1, int(np.ceil(pad_width / w_left)))
    k_right = max(1, int(np.ceil(pad_width / w_right)))

    interior_bins = n_bins - k_left - k_right
    if interior_bins < 1:
        # Fallback: keep interior quantiles and just extend outer borders.
        borders = np.quantile(y, qs_full)
        borders[0] = low - pad_width
        borders[-1] = high + pad_width
        borders = _make_strictly_increasing(borders, min_step=min_step)
        centers = 0.5 * (borders[:-1] + borders[1:])
        return borders, centers

    qs = np.linspace(0.0, 1.0, interior_bins + 1)
    inner = np.quantile(y, qs)
    inner = _make_strictly_increasing(inner, min_step=min_step)

    if interior_bins > 1:
        w_left = max(float(inner[1] - inner[0]), min_step)
        w_right = max(float(inner[-1] - inner[-2]), min_step)

    left = inner[0] - w_left * np.arange(k_left, 0, -1)
    right = inner[-1] + w_right * np.arange(1, k_right + 1)
    borders = np.concatenate([left, inner, right])
    borders = _make_strictly_increasing(borders, min_step=min_step)
    centers = 0.5 * (borders[:-1] + borders[1:])
    return borders, centers
