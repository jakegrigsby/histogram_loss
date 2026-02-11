#!/usr/bin/env python3
"""Print uniform vs quantile bin locations for a dataset and bin count.

Uses the TRAIN split to compute quantiles, matching HLP behavior.
Bins are computed in scaled [0, 1] space, then also shown in original units.
"""

import argparse
import numpy as np
import tensorflow as tf

from replication.csvdataset import CSVDataset
from experiment.preprocessing import Scaler
from experiment.bins import get_quantile_bins


TEST_RATIO = 0.2
DEFAULT_PADDING = 0.125


def get_dataset_by_name(base_dir, name, batch_size=256):
    data_dir = f"{base_dir}/data"
    if name == "ctscan":
        ds = CSVDataset(
            f"{data_dir}/slice_localization_data.csv",
            "reference",
            batch_size=batch_size,
        )
        ds.bounds = (0.0, 100.0)
        ds.name = "ctscan"
    elif name == "bike":
        ds = CSVDataset(
            f"{data_dir}/hour.csv",
            "cnt",
            drop="dteday",
            batch_size=batch_size,
        )
        ds.bounds = (0.0, 1000.0)
        ds.name = "bike"
    elif name == "songyear":
        ds = CSVDataset(
            f"{data_dir}/YearPredictionMSD.txt",
            0,
            header=None,
            batch_size=batch_size,
        )
        ds.bounds = (1922.0, 2011.0)
        ds.name = "songyear"
    elif name == "pole":
        ds = CSVDataset(f"{data_dir}/pole.csv", "target", batch_size=batch_size)
        ds.bounds = (0.0, 100.0)
        ds.name = "pole"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds


def _collect_targets(ds):
    ys = []
    for _, y in ds:
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0).reshape(-1)


def _print_array(label, arr, full=False, max_show=10):
    arr = np.asarray(arr)
    if full or len(arr) <= 2 * max_show:
        print(f"{label} ({len(arr)}): {arr.tolist()}")
        return
    head = arr[:max_show].tolist()
    tail = arr[-max_show:].tolist()
    print(f"{label} ({len(arr)}): {head} ... {tail}")


def _width_stats(borders):
    widths = np.diff(borders)
    return float(np.min(widths)), float(np.median(widths)), float(np.max(widths))


def main():
    p = argparse.ArgumentParser(
        description="Show uniform vs quantile bin locations for a dataset."
    )
    p.add_argument("--data_dir", default=".", help="Directory containing data/")
    p.add_argument(
        "--dataset",
        default="ctscan",
        choices=["ctscan", "bike", "pole", "songyear"],
    )
    p.add_argument("--n_bins", type=int, default=100)
    p.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    p.add_argument("--full", action="store_true", help="Print all borders/centers")
    args = p.parse_args()

    ds = get_dataset_by_name(args.data_dir, args.dataset)

    # Train/test split (train used for quantiles)
    train, _ = ds.get_split(TEST_RATIO, shuffle=True)
    sc = Scaler(*ds.bounds)
    train = sc.transform(train)

    y_scaled = _collect_targets(train)
    y_scaled = np.clip(y_scaled, 0.0, 1.0)

    # Quantile bins (train-based)
    low = float(np.min(y_scaled)) if len(y_scaled) else 0.0
    high = float(np.max(y_scaled)) if len(y_scaled) else 1.0
    q_borders, q_centers = get_quantile_bins(
        y_scaled, args.n_bins, padding=args.padding, low=low, high=high
    )

    # Uniform bins (standard HL spacing in scaled space)
    y_min, y_max = 0.0, 1.0
    y_range = y_max - y_min
    new_min = y_min - args.padding * y_range
    new_max = y_max + args.padding * y_range
    u_borders = np.linspace(new_min, new_max, args.n_bins + 1)
    u_centers = 0.5 * (u_borders[:-1] + u_borders[1:])

    # Report
    print(f"Dataset: {args.dataset}")
    print(f"n_bins: {args.n_bins}")
    print(f"train samples used for quantiles: {len(y_scaled)}")
    print(f"padding (uniform): {args.padding}")
    print("")

    u_stats = _width_stats(u_borders)
    q_stats = _width_stats(q_borders)
    print("Width stats (scaled):")
    print(
        f"  uniform  min/median/max = {u_stats[0]:.6f} / {u_stats[1]:.6f} / {u_stats[2]:.6f}"
    )
    print(
        f"  quantile min/median/max = {q_stats[0]:.6f} / {q_stats[1]:.6f} / {q_stats[2]:.6f}"
    )
    print("")

    print("Scaled borders:")
    _print_array("  uniform", u_borders, full=args.full)
    _print_array("  quantile", q_borders, full=args.full)
    print("")

    print("Scaled centers:")
    _print_array("  uniform", u_centers, full=args.full)
    _print_array("  quantile", q_centers, full=args.full)
    print("")

    # Convert to original units
    orig_min, orig_max = ds.bounds
    orig_range = orig_max - orig_min
    u_borders_orig = orig_min + u_borders * orig_range
    q_borders_orig = orig_min + q_borders * orig_range
    u_centers_orig = orig_min + u_centers * orig_range
    q_centers_orig = orig_min + q_centers * orig_range

    print("Original-unit borders:")
    _print_array("  uniform", u_borders_orig, full=args.full)
    _print_array("  quantile", q_borders_orig, full=args.full)
    print("")

    print("Original-unit centers:")
    _print_array("  uniform", u_centers_orig, full=args.full)
    _print_array("  quantile", q_centers_orig, full=args.full)


if __name__ == "__main__":
    main()
