#!/usr/bin/env python3
"""Analyze sweep results and generate figures.

Loads JSON results from sweep_results/{dataset}/, aggregates across seeds,
and produces scatter plots or heatmaps for comparing methods.

Scatter plots:
  - X-axis: a sweep parameter (n_bins, sig_ratio, int_dim, lr)
  - Color: method (HL-Gaussian vs HL-Projected)
  - Subplot grid: up to two facet dimensions (--facet_row, --facet_col)
  - Remaining parameters: fixed via --filter_*
  - Each dot = one config averaged over seeds; error bars = min/max across seeds.

Heatmaps:
  - Single-method view (default HL-Gaussian) with two parameters as axes.

Usage examples:
    # Scatter: test_rmse vs n_bins, subplots by int_dim (rows) × lr (cols)
    python -m replication.analyze --dataset ctscan --metric test_rmse \\
        --x n_bins --facet_row int_dim --facet_col lr --filter_sig_ratio 1.0

    # Scatter: test_rmse vs sig_ratio, subplots by lr
    python -m replication.analyze --dataset ctscan --metric test_rmse \\
        --x sig_ratio --facet_col lr --filter_int_dim 0.5

    # Heatmap: HL-Gaussian test_rmse by sig_ratio × n_bins
    python -m replication.analyze --dataset ctscan --metric test_rmse \\
        --plot heatmap --rows sig_ratio --cols n_bins --filter_lr 0.001

    # Summary table only (top configs per method)
    python -m replication.analyze --dataset ctscan --metric test_rmse --plot none --summary
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import ticker


# ── Global style ────────────────────────────────────────────────────────────

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
}


def _apply_style():
    matplotlib.rcParams.update(STYLE)


# ── Pretty label names ──────────────────────────────────────────────────────

PRETTY_LABELS = {
    "n_bins": "Number of Bins",
    "sig_ratio": "σ ratio",
    "int_dim": "Hidden-Width Multiplier",
    "lr": "Learning Rate",
    "test_rmse": "Test RMSE",
    "train_rmse": "Train RMSE",
    "test_mae": "Test MAE",
    "train_mae": "Train MAE",
    "test_rmse_scaled": "Test RMSE (scaled)",
    "train_rmse_scaled": "Train RMSE (scaled)",
    "test_mae_scaled": "Test MAE (scaled)",
    "train_mae_scaled": "Train MAE (scaled)",
    "best_epoch": "Best Epoch",
}


def _pretty(name):
    return PRETTY_LABELS.get(name, name)


# ── Data loading ────────────────────────────────────────────────────────────

CONFIG_COLS = ["method", "n_bins", "sig_ratio", "int_dim", "lr"]
ALL_METRICS = [
    "train_rmse",
    "test_rmse",
    "train_mae",
    "test_mae",
    "train_rmse_scaled",
    "test_rmse_scaled",
    "train_mae_scaled",
    "test_mae_scaled",
    "best_epoch",
]


def load_results(results_dirs, dataset):
    """Load all JSON result files for a dataset into a DataFrame.

    ``results_dirs`` may be a single path (str) or a list of paths.
    Results from all directories are merged into one DataFrame.
    """
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]

    all_files = []
    for rdir in results_dirs:
        pattern = os.path.join(rdir, dataset, "*.json")
        all_files.extend(sorted(glob.glob(pattern)))

    if not all_files:
        searched = [os.path.join(d, dataset, "*.json") for d in results_dirs]
        raise FileNotFoundError(f"No results found at {searched}")

    records = []
    for f in all_files:
        with open(f) as fh:
            records.append(json.load(fh))

    df = pd.DataFrame(records)
    if "lr" not in df.columns and "learning_rate" in df.columns:
        df["lr"] = df["learning_rate"]
    elif "lr" in df.columns and "learning_rate" not in df.columns:
        df["learning_rate"] = df["lr"]
    print(
        f"Loaded {len(df)} results for '{dataset}' from {len(results_dirs)} dir(s) "
        f"({df['method'].nunique()} methods, "
        f"{df.groupby(CONFIG_COLS).ngroups} configs × seeds)"
    )
    return df


def aggregate_seeds(df, metric):
    """Group by config (everything except seed) and compute mean / min / max."""
    agg = (
        df.groupby(CONFIG_COLS, dropna=False)[metric]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = CONFIG_COLS + [
        f"{metric}_mean",
        f"{metric}_min",
        f"{metric}_max",
        "n_seeds",
    ]
    return agg


def apply_filters(df, args):
    """Apply CLI filters to the DataFrame."""
    if args.filter_method:
        df = df[df["method"].isin(args.filter_method)]
    if args.filter_n_bins:
        df = df[df["n_bins"].isin(args.filter_n_bins)]
    if args.filter_sig_ratio:
        df = df[df["sig_ratio"].isin(args.filter_sig_ratio)]
    if args.filter_int_dim:
        df = df[df["int_dim"].isin(args.filter_int_dim)]
    if args.filter_lr:
        df = df[df["lr"].isin(args.filter_lr)]
    if len(df) == 0:
        raise ValueError(
            "No data remaining after filtering! Check your --filter_* args."
        )
    filt_info = []
    for col, vals in [
        ("method", args.filter_method),
        ("n_bins", args.filter_n_bins),
        ("sig_ratio", args.filter_sig_ratio),
        ("int_dim", args.filter_int_dim),
        ("lr", args.filter_lr),
    ]:
        if vals:
            filt_info.append(f"{col}={vals}")
    if filt_info:
        print(f"  Filters: {', '.join(filt_info)} → {len(df)} rows")
    return df


# ── Method visual identity ──────────────────────────────────────────────────

METHOD_COLORS = {
    "HL-Gaussian": "#2274A5",  # steel blue
    "HL-Projected": "#E76F51",  # burnt orange
    "HL-Gibbs": "#2CA02C",  # green
    "HL-MCGaussian": "#9467BD",  # purple
    "HLP-Gaussian": "#4F9DD8",  # light blue
    "HLP-Projected": "#F4A261",  # light orange
    "HLP-Gibbs": "#6CCB5F",  # light green
    "HLP-GaussianLocal": "#3C7FB1",  # blue-gray
    "HLP-GibbsWidth": "#3D9C3D",  # green
}
METHOD_MARKERS = {
    "HL-Gaussian": "o",
    "HL-Projected": "D",
    "HL-Gibbs": "s",
    "HL-MCGaussian": "P",
    "HLP-Gaussian": "^",
    "HLP-Projected": "v",
    "HLP-Gibbs": "P",
    "HLP-GaussianLocal": "X",
    "HLP-GibbsWidth": "*",
}
_FALLBACK_COLORS = ["#2CA02C", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"]
_FALLBACK_MARKERS = ["s", "^", "P", "X", "v", "<", ">"]


def _method_color(method, idx=0):
    return METHOD_COLORS.get(method, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _method_marker(method, idx=0):
    return METHOD_MARKERS.get(method, _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)])


# ── Facet title banner ──────────────────────────────────────────────────────


def _facet_title(ax, text):
    """Draw a subtle gray banner at the top of a subplot for the facet label."""
    ax.set_title(
        text,
        fontsize=9.5,
        fontweight="semibold",
        color="#333333",
        pad=8,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#e8e8e8",
            edgecolor="#cccccc",
            linewidth=0.6,
        ),
    )


# ── Scatter plot ────────────────────────────────────────────────────────────


def scatter_plot(
    df,
    metric,
    x_col,
    facet_row=None,
    facet_col=None,
    outfile=None,
    title=None,
    show_envelope=False,
):
    """Grouped scatter: methods as color, 2-D facet grid, seed error bars."""
    _apply_style()

    agg = aggregate_seeds(df, metric)
    methods = sorted(agg["method"].unique())

    # Facet values
    row_vals = sorted(agg[facet_row].unique()) if facet_row else [None]
    col_vals = sorted(agg[facet_col].unique()) if facet_col else [None]
    n_rows, n_cols = len(row_vals), len(col_vals)

    # X positions
    x_vals = sorted(agg[x_col].unique())
    x_pos = {v: i for i, v in enumerate(x_vals)}
    n_methods = len(methods)
    span = min(0.45, 0.18 * n_methods)
    offsets = {}
    for mi, method in enumerate(methods):
        offsets[method] = -span / 2 + (mi + 0.5) * span / n_methods

    # Figure sizing
    cell_w = max(3.5, 0.75 * len(x_vals) + 1.8)
    cell_h = 3.6
    fig_w = cell_w * n_cols + 1.6
    fig_h = cell_h * n_rows + 1.2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharey=True, sharex=True
    )

    for ri, rval in enumerate(row_vals):
        for ci, cval in enumerate(col_vals):
            ax = axes[ri, ci]

            # Subset for this facet cell
            sub = agg
            if facet_row and rval is not None:
                sub = sub[sub[facet_row] == rval]
            if facet_col and cval is not None:
                sub = sub[sub[facet_col] == cval]

            for mi, method in enumerate(methods):
                msub = sub[sub["method"] == method]
                color = _method_color(method, mi)
                marker = _method_marker(method, mi)
                off = offsets[method]

                xs, ys, yerr_lo, yerr_hi = [], [], [], []
                for _, row in msub.iterrows():
                    xp = x_pos[row[x_col]] + off
                    mean_v = row[f"{metric}_mean"]
                    xs.append(xp)
                    ys.append(mean_v)
                    yerr_lo.append(mean_v - row[f"{metric}_min"])
                    yerr_hi.append(row[f"{metric}_max"] - mean_v)

                if not xs:
                    continue

                # Combined markers + error bars in one call
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    fmt=marker,
                    color=color,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=3.5,
                    capthick=0.9,
                    alpha=0.85,
                    zorder=3,
                )

            # Best-envelope overlay
            if show_envelope:
                # Build the raw (non-aggregated) subset for this facet cell
                raw_sub = df
                if facet_row and rval is not None:
                    raw_sub = raw_sub[raw_sub[facet_row] == rval]
                if facet_col and cval is not None:
                    raw_sub = raw_sub[raw_sub[facet_col] == cval]
                _draw_best_envelope(ax, raw_sub, metric, x_col, x_pos, methods, offsets)

            # X ticks
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels([str(v) for v in x_vals], fontsize=9)
            ax.tick_params(axis="x", length=3, pad=3)
            ax.tick_params(axis="y", length=3, pad=3)

            # Only label edges
            if ri == n_rows - 1:
                ax.set_xlabel(_pretty(x_col), fontsize=10.5, labelpad=5)
            if ci == 0:
                ax.set_ylabel(_pretty(metric), fontsize=10.5, labelpad=5)

            # X margins
            ax.set_xlim(-0.5, len(x_vals) - 0.5)

            # Facet banner
            parts = []
            if facet_row and rval is not None:
                parts.append(f"{_pretty(facet_row)} = {rval}")
            if facet_col and cval is not None:
                parts.append(f"{_pretty(facet_col)} = {cval}")
            if parts:
                _facet_title(ax, "   ".join(parts))

    # ── Legend ──
    legend_handles = []
    for mi, method in enumerate(methods):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=_method_marker(method, mi),
                color=_method_color(method, mi),
                linestyle="None",
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=method,
            )
        )

    leg = fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9.5,
        handletextpad=0.5,
        bbox_to_anchor=(0.995, 0.995),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=False,
    )
    leg.get_frame().set_linewidth(0.6)

    # Suptitle
    sup = title or f"{_pretty(metric)}  vs  {_pretty(x_col)}"
    fig.suptitle(sup, fontsize=14, fontweight="bold", color="#222222", y=1.015)

    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5, w_pad=2.0)

    if outfile:
        fig.savefig(outfile, bbox_inches="tight", facecolor="white")
        print(f"Saved → {outfile}")
    plt.close(fig)


# ── Heatmap plot ────────────────────────────────────────────────────────────


def heatmap_plot(
    df,
    metric,
    row_col,
    col_col,
    method="HL-Gaussian",
    facet_row=None,
    facet_col=None,
    outfile=None,
):
    """Heatmap for a single method across two parameters, optionally faceted."""
    _apply_style()

    sub = df[df["method"] == method]
    if len(sub) == 0:
        raise ValueError(f"No data for method='{method}' after filtering.")

    agg = aggregate_seeds(sub, metric)

    frow_vals = sorted(agg[facet_row].unique()) if facet_row else [None]
    fcol_vals = sorted(agg[facet_col].unique()) if facet_col else [None]
    n_fr, n_fc = len(frow_vals), len(fcol_vals)

    all_means = agg[f"{metric}_mean"].values
    vmin, vmax = np.nanmin(all_means), np.nanmax(all_means)

    cell_w = max(6.5, 1.2 * agg[col_col].nunique() + 2)
    cell_h = max(3.5, 0.55 * agg[row_col].nunique() + 1.5)
    fig, axes = plt.subplots(
        n_fr, n_fc, figsize=(cell_w * n_fc + 2, cell_h * n_fr + 1), squeeze=False
    )

    # Good perceptual colormap: low=green, high=red
    cmap = plt.get_cmap("RdYlGn_r")

    for fri, frval in enumerate(frow_vals):
        for fci, fcval in enumerate(fcol_vals):
            ax = axes[fri, fci]
            cell = agg
            if facet_row and frval is not None:
                cell = cell[cell[facet_row] == frval]
            if facet_col and fcval is not None:
                cell = cell[cell[facet_col] == fcval]

            pivot_mean = cell.pivot_table(
                values=f"{metric}_mean", index=row_col, columns=col_col
            )
            pivot_min = cell.pivot_table(
                values=f"{metric}_min", index=row_col, columns=col_col
            )
            pivot_max = cell.pivot_table(
                values=f"{metric}_max", index=row_col, columns=col_col
            )

            pivot_mean = pivot_mean.sort_index(ascending=False)
            pivot_min = pivot_min.reindex(pivot_mean.index)[pivot_mean.columns]
            pivot_max = pivot_max.reindex(pivot_mean.index)[pivot_mean.columns]

            nr, nc = pivot_mean.shape
            vals = pivot_mean.values
            im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_xticks(range(nc))
            ax.set_xticklabels(pivot_mean.columns, fontsize=9)
            ax.set_yticks(range(nr))
            ax.set_yticklabels(pivot_mean.index, fontsize=9)

            if fri == n_fr - 1:
                ax.set_xlabel(_pretty(col_col), fontsize=10.5, labelpad=5)
            if fci == 0:
                ax.set_ylabel(_pretty(row_col), fontsize=10.5, labelpad=5)

            # Annotate cells with outline for legibility
            mid = (vmin + vmax) / 2
            for i in range(nr):
                for j in range(nc):
                    v = vals[i, j]
                    if np.isnan(v):
                        continue
                    spread = (pivot_max.values[i, j] - pivot_min.values[i, j]) / 2
                    txt = f"{v:.3f}\n±{spread:.3f}"
                    fg = "white" if v > mid else "#222222"
                    outline_c = "#222222" if v > mid else "white"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=fg,
                        fontweight="medium",
                        path_effects=[pe.withStroke(linewidth=2, foreground=outline_c)],
                    )

            # Facet title
            parts = []
            if facet_row and frval is not None:
                parts.append(f"{_pretty(facet_row)} = {frval}")
            if facet_col and fcval is not None:
                parts.append(f"{_pretty(facet_col)} = {fcval}")
            if parts:
                _facet_title(ax, "   ".join(parts))

    # Shared colorbar
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), label=_pretty(metric), shrink=0.75, pad=0.03
    )
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"{method}  —  {_pretty(metric)}",
        fontsize=14,
        fontweight="bold",
        color="#222222",
    )

    if outfile:
        fig.savefig(outfile, bbox_inches="tight", facecolor="white")
        print(f"Saved → {outfile}")
    plt.close(fig)


# ── Summary table ───────────────────────────────────────────────────────────


def print_summary(df, metric, top_n=10):
    """Print best configs per method, ranked by metric (ascending)."""
    agg = aggregate_seeds(df, metric)

    for method in sorted(agg["method"].unique()):
        msub = agg[agg["method"] == method].sort_values(f"{metric}_mean")
        print(f"\n{'=' * 74}")
        print(
            f"  {method}  —  Top {min(top_n, len(msub))} by {metric}  (lower is better)"
        )
        print(f"{'=' * 74}")
        for i, (_, row) in enumerate(msub.head(top_n).iterrows()):
            params = (
                f"bins={int(row['n_bins']):>3d}  "
                f"sig={row['sig_ratio']:<5.2f}  "
                f"idim={row['int_dim']:<4.1f}  "
                f"lr={row['lr']}"
            )
            val = (
                f"{row[f'{metric}_mean']:.4f}  "
                f"[{row[f'{metric}_min']:.4f}, {row[f'{metric}_max']:.4f}]"
            )
            seeds = int(row["n_seeds"])
            print(f"  {i+1:>2d}. {params}  →  {val}  (n={seeds})")


# ── Gap analysis ────────────────────────────────────────────────────────────


def _best_per_x(df, metric, x_col):
    """For each (method, x_value), find the config with the best mean metric.

    Returns a DataFrame with one row per (method, x_value), containing the
    best config's full info.
    """
    agg = aggregate_seeds(df, metric)
    idx = agg.groupby(["method", x_col])[f"{metric}_mean"].idxmin()
    return agg.loc[idx].reset_index(drop=True)


def print_gap_analysis(df, metric, x_col):
    """Print a table comparing best config per method per x value.

    Supports 2+ methods.  The reference (baseline) is the alphabetically-first
    method (typically HL-Gaussian).  Gap columns show each other method vs the
    reference.
    """
    best = _best_per_x(df, metric, x_col)
    methods = sorted(best["method"].unique())

    if len(methods) < 2:
        print(f"\n  Gap analysis requires ≥2 methods, found: {methods}")
        return

    x_vals = sorted(best[x_col].unique())
    ref = methods[0]  # reference method (alphabetical first, e.g. HL-Gaussian)
    others = methods[1:]

    # ── Build column widths ──
    mw = 14  # method column width
    sep = "  │  "
    # Header
    width = 100 + mw * len(others)
    print(f"\n{'─' * width}")
    print(f"  Gap analysis vs {ref}  across {_pretty(x_col)}")
    print(f"  Metric: {_pretty(metric)}  (lower is better)")
    print(f"{'─' * width}")

    # Column headers
    hdr = f"  {'x':>8s}{sep}{ref:>{mw}s}"
    for m in others:
        hdr += f"{sep}{m:>{mw}s}{sep}{'Gap':>8s}{sep}{'Gap%':>6s}"
    hdr += f"{sep}Winner"
    print(hdr)
    rule = f"  {'─'*8}──┼──{'─'*mw}"
    for _ in others:
        rule += f"──┼──{'─'*mw}──┼──{'─'*8}──┼──{'─'*6}"
    rule += f"──┼──{'─'*14}"
    print(rule)

    win_counts = {m: 0 for m in methods}

    for xv in x_vals:
        r_ref = best[(best["method"] == ref) & (best[x_col] == xv)]
        v_ref = r_ref[f"{metric}_mean"].values[0] if len(r_ref) else float("nan")

        line = f"  {str(xv):>8s}{sep}{v_ref:>{mw}.4f}"
        row_best_method = ref
        row_best_val = v_ref

        for m in others:
            r_m = best[(best["method"] == m) & (best[x_col] == xv)]
            v_m = r_m[f"{metric}_mean"].values[0] if len(r_m) else float("nan")

            if np.isnan(v_ref) or np.isnan(v_m):
                line += f"{sep}{v_m:>{mw}.4f}{sep}{'—':>8s}{sep}{'—':>6s}"
            else:
                gap = v_m - v_ref
                pct = gap / min(v_ref, v_m) * 100
                line += f"{sep}{v_m:>{mw}.4f}{sep}{gap:>+8.4f}{sep}{pct:>+5.1f}%"

            if not np.isnan(v_m) and (np.isnan(row_best_val) or v_m < row_best_val):
                row_best_val = v_m
                row_best_method = m

        if np.isnan(row_best_val):
            line += f"{sep}—"
        else:
            win_counts[row_best_method] += 1
            if row_best_method == ref:
                line += f"{sep}← {ref}"
            else:
                line += f"{sep}→ {row_best_method}"

        print(line)

    # Overall best row
    print(rule)
    overall_vals = {}
    for m in methods:
        msub = best[best["method"] == m][f"{metric}_mean"]
        overall_vals[m] = msub.min() if len(msub) else float("nan")

    v_ref_best = overall_vals[ref]
    line = f"  {'BEST':>8s}{sep}{v_ref_best:>{mw}.4f}"
    overall_winner = ref
    overall_best = v_ref_best
    for m in others:
        v_m = overall_vals[m]
        if not np.isnan(v_ref_best) and not np.isnan(v_m):
            gap = v_m - v_ref_best
            pct = gap / min(v_ref_best, v_m) * 100
            line += f"{sep}{v_m:>{mw}.4f}{sep}{gap:>+8.4f}{sep}{pct:>+5.1f}%"
        else:
            line += f"{sep}{v_m:>{mw}.4f}{sep}{'—':>8s}{sep}{'—':>6s}"
        if not np.isnan(v_m) and v_m < overall_best:
            overall_best = v_m
            overall_winner = m
    line += f"{sep}★ {overall_winner}"
    print(line)

    win_strs = [f"{m} {win_counts[m]}/{len(x_vals)}" for m in methods]
    print(f"  Wins:  {',  '.join(win_strs)}")
    print()


def _draw_best_envelope(ax, df, metric, x_col, x_pos, methods, offsets):
    """Overlay a 'best envelope' line on a scatter axis.

    For each method, connects the best config at each x value with a line.
    """
    best = _best_per_x(df, metric, x_col)

    for mi, method in enumerate(methods):
        mbest = best[best["method"] == method].sort_values(x_col)
        if mbest.empty:
            continue
        color = _method_color(method, mi)
        xs = [x_pos[row[x_col]] + offsets[method] for _, row in mbest.iterrows()]
        ys = [row[f"{metric}_mean"] for _, row in mbest.iterrows()]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8,
            alpha=0.6,
            zorder=5,
            linestyle="--",
            marker="",
            dash_capstyle="round",
        )


# ── Auto filename helper ───────────────────────────────────────────────────


def _auto_filename(args):
    """Build a descriptive output filename from CLI args."""
    parts = [args.dataset, args.metric]
    if args.plot == "scatter":
        parts.append(f"x-{args.x}")
        if args.facet_row:
            parts.append(f"fr-{args.facet_row}")
        if args.facet_col:
            parts.append(f"fc-{args.facet_col}")
    elif args.plot == "heatmap":
        parts += ["heatmap", args.heatmap_method, f"r-{args.rows}", f"c-{args.cols}"]
        if args.facet_row:
            parts.append(f"fr-{args.facet_row}")
        if args.facet_col:
            parts.append(f"fc-{args.facet_col}")

    for tag, vals in [
        ("meth", args.filter_method),
        ("bins", args.filter_n_bins),
        ("sig", args.filter_sig_ratio),
        ("idim", args.filter_int_dim),
        ("lr", args.filter_lr),
    ]:
        if vals:
            parts.append(f"{tag}={'_'.join(str(v) for v in vals)}")

    return "_".join(parts) + ".png"


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Analyze sweep results and generate figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--dataset", default="ctscan", choices=["ctscan", "bike", "pole", "songyear"]
    )
    p.add_argument(
        "--results_dir",
        nargs="+",
        default=["sweep_results"],
        help="One or more directories containing sweep results",
    )
    p.add_argument(
        "--metric",
        default="test_rmse",
        help=f"Metric to plot. Choices include: {', '.join(ALL_METRICS)}",
    )

    p.add_argument(
        "--plot",
        default="scatter",
        choices=["scatter", "heatmap", "none"],
        help="Plot type. 'none' prints summary only.",
    )

    PARAM_CHOICES = ["n_bins", "sig_ratio", "int_dim", "lr"]
    p.add_argument(
        "--x", default="n_bins", choices=PARAM_CHOICES, help="X-axis parameter"
    )

    p.add_argument(
        "--facet_row",
        default=None,
        choices=PARAM_CHOICES,
        help="Parameter for subplot rows",
    )
    p.add_argument(
        "--facet_col",
        default=None,
        choices=PARAM_CHOICES,
        help="Parameter for subplot columns",
    )

    p.add_argument("--rows", default="sig_ratio", help="Heatmap row parameter")
    p.add_argument("--cols", default="n_bins", help="Heatmap column parameter")
    p.add_argument("--heatmap_method", default="HL-Gaussian", help="Method for heatmap")

    p.add_argument("--filter_method", nargs="+", default=None)
    p.add_argument("--filter_n_bins", type=int, nargs="+", default=None)
    p.add_argument("--filter_sig_ratio", type=float, nargs="+", default=None)
    p.add_argument("--filter_int_dim", type=float, nargs="+", default=None)
    p.add_argument("--filter_lr", type=float, nargs="+", default=None)

    p.add_argument(
        "--summary", action="store_true", help="Print top configs per method"
    )
    p.add_argument(
        "--top_n", type=int, default=10, help="Number of top configs to show"
    )
    p.add_argument(
        "--gap",
        action="store_true",
        help="Print best-config gap table (per x value) and draw envelope on scatter",
    )

    p.add_argument(
        "--out", default=None, help="Output file path (auto-generated if omitted)"
    )
    p.add_argument(
        "--out_dir", default="figures", help="Directory for auto-named output files"
    )

    args = p.parse_args()

    df = load_results(args.results_dir, args.dataset)
    df = apply_filters(df, args)

    if args.summary or args.plot == "none":
        print_summary(df, args.metric, args.top_n)
        if args.plot == "none":
            if args.gap:
                print_gap_analysis(df, args.metric, args.x)
            return

    outfile = args.out
    if outfile is None:
        os.makedirs(args.out_dir, exist_ok=True)
        outfile = os.path.join(args.out_dir, _auto_filename(args))

    if args.plot == "scatter":
        scatter_plot(
            df,
            args.metric,
            args.x,
            facet_row=args.facet_row,
            facet_col=args.facet_col,
            outfile=outfile,
            show_envelope=args.gap,
        )
    elif args.plot == "heatmap":
        heatmap_plot(
            df,
            args.metric,
            args.rows,
            args.cols,
            method=args.heatmap_method,
            facet_row=args.facet_row,
            facet_col=args.facet_col,
            outfile=outfile,
        )

    if args.gap:
        print_gap_analysis(df, args.metric, args.x)
    if args.summary:
        print_summary(df, args.metric, args.top_n)


if __name__ == "__main__":
    main()
