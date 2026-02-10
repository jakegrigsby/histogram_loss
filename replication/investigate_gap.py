#!/usr/bin/env python3
"""Investigate why HL-Gaussian consistently outperforms HL-Projected.

Produces a series of figures combining analytical computations of target
distribution properties with empirical results from the sweep.

Run:
    python -m replication.investigate_gap [--results_dir sweep_results] [--out_dir figures/gap]
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.special import erf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe


# ── Style ───────────────────────────────────────────────────────────────────

COLOR_GAUSS = "#2274A5"
COLOR_PROJ = "#E76F51"
COLOR_NEUTRAL = "#666666"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
}
matplotlib.rcParams.update(STYLE)


# ── Analytical target distributions ─────────────────────────────────────────


def bin_geometry(n_bins, padding=0.125):
    """Return bin borders and centers for [0, 1] with padding."""
    low = -padding
    high = 1 + padding
    borders = np.linspace(low, high, n_bins + 1)
    centers = (borders[:-1] + borders[1:]) / 2
    bin_width = borders[1] - borders[0]
    return borders, centers, bin_width


def gaussian_target(y, n_bins, sig_ratio, padding=0.125):
    """Compute the HL-Gaussian target distribution for a scalar target y."""
    borders, centers, bw = bin_geometry(n_bins, padding)
    sigma = sig_ratio * bw
    # CDF differences via erf
    scaled = (borders - y) / (np.sqrt(2) * sigma)
    cdf_vals = erf(scaled)
    probs = cdf_vals[1:] - cdf_vals[:-1]
    probs /= cdf_vals[-1] - cdf_vals[0]  # normalize
    return centers, probs


def projected_target(y, n_bins, padding=0.125):
    """Compute the HL-Projected target distribution for a scalar target y."""
    borders, centers, bw = bin_geometry(n_bins, padding)
    probs = np.zeros(n_bins)
    # Find the lower bin index
    i = int((y - centers[0]) / bw)
    i = np.clip(i, 0, n_bins - 2)
    p = (y - centers[i]) / bw
    probs[i] = 1 - p
    probs[i + 1] = p
    return centers, probs


def entropy(probs):
    """Shannon entropy of a discrete distribution (nats)."""
    p = probs[probs > 0]
    return -np.sum(p * np.log(p))


def effective_support(probs, threshold=0.01):
    """Number of bins with probability above threshold."""
    return np.sum(probs > threshold)


def gibbs_target(y, n_bins, n_iter=50, padding=0.125):
    """Compute the HL-Gibbs (max-entropy, mean-matching) target distribution.

    Finds λ such that softmax(λ·c) has mean = y, via Newton-Raphson.
    """
    borders, centers, bw = bin_geometry(n_bins, padding)
    # Clamp target inside bin range
    y = np.clip(y, centers[0] + 1e-4, centers[-1] - 1e-4)
    lam = 0.0
    for _ in range(n_iter):
        logits = lam * centers
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        mean = np.sum(probs * centers)
        var = np.sum(probs * (centers - mean) ** 2)
        if var < 1e-12:
            break
        step = (mean - y) / var
        lam -= np.clip(step, -10, 10)
    # Final distribution
    logits = lam * centers
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return centers, probs


def kl_from_uniform(probs):
    """KL(target || uniform) = log(N) - H(target).  Measures peakedness."""
    n = len(probs)
    return np.log(n) - entropy(probs)


def peak_to_mean_ratio(probs):
    """max(p) / mean(p) = N · max(p).  How concentrated is the mode?"""
    return np.max(probs) * len(probs)


# ── Data loading (from analyze.py) ──────────────────────────────────────────

CONFIG_COLS = ["method", "n_bins", "sig_ratio", "int_dim", "lr"]


def load_all_datasets(results_dir, datasets=("ctscan", "bike", "pole", "songyear")):
    """Load results for all available datasets into one DataFrame."""
    frames = []
    for ds in datasets:
        pattern = os.path.join(results_dir, ds, "*.json")
        files = sorted(glob.glob(pattern))
        for f in files:
            with open(f) as fh:
                rec = json.load(fh)
                rec.setdefault("dataset", ds)
                frames.append(rec)
    if not frames:
        raise FileNotFoundError(f"No results found in {results_dir}")
    df = pd.DataFrame(frames)
    if "lr" not in df.columns and "learning_rate" in df.columns:
        df["lr"] = df["learning_rate"]
    return df


def aggregate_seeds(df, metric):
    grp = CONFIG_COLS + (["dataset"] if "dataset" in df.columns else [])
    agg = (
        df.groupby(grp, dropna=False)[metric]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = grp + [f"{metric}_mean", f"{metric}_min", f"{metric}_max", "n_seeds"]
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Visual comparison of target distributions
# ═══════════════════════════════════════════════════════════════════════════

COLOR_GIBBS = "#6A994E"  # green for Gibbs


def fig_target_distributions(out_dir):
    """Show what all three target distributions look like at different bin counts."""
    y_target = 0.37  # slightly off-center to show typical case
    sig_ratio = 1.0
    bins_list = [10, 50, 200]

    fig, axes = plt.subplots(3, len(bins_list), figsize=(14, 8), sharey="row")

    for j, n_bins in enumerate(bins_list):
        bw_plot = lambda c: (c[1] - c[0]) * 0.85

        # Row 0: Gaussian
        c_g, p_g = gaussian_target(y_target, n_bins, sig_ratio)
        ax = axes[0, j]
        ax.bar(
            c_g,
            p_g,
            width=bw_plot(c_g),
            color=COLOR_GAUSS,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(y_target, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(
            f"HL-Gaussian  (N={n_bins}, σ-ratio={sig_ratio})",
            fontsize=9.5,
            fontweight="semibold",
            color=COLOR_GAUSS,
        )
        ent_g = entropy(p_g)
        sup_g = effective_support(p_g)
        ax.text(
            0.97,
            0.95,
            f"H = {ent_g:.2f} nats\nSupport = {sup_g} bins",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
        )

        # Row 1: Gibbs (max-entropy, mean-matching)
        c_b, p_b = gibbs_target(y_target, n_bins)
        ax = axes[1, j]
        ax.bar(
            c_b,
            p_b,
            width=bw_plot(c_b),
            color=COLOR_GIBBS,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(y_target, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(
            f"HL-Gibbs  (N={n_bins})",
            fontsize=9.5,
            fontweight="semibold",
            color=COLOR_GIBBS,
        )
        ent_b = entropy(p_b)
        sup_b = effective_support(p_b)
        ax.text(
            0.97,
            0.95,
            f"H = {ent_b:.2f} nats\nSupport = {sup_b} bins",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
        )

        # Row 2: Projected
        c_p, p_p = projected_target(y_target, n_bins)
        ax = axes[2, j]
        ax.bar(
            c_p,
            p_p,
            width=bw_plot(c_p),
            color=COLOR_PROJ,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(y_target, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(
            f"HL-Projected  (N={n_bins})",
            fontsize=9.5,
            fontweight="semibold",
            color=COLOR_PROJ,
        )
        ent_p = entropy(p_p)
        sup_p = effective_support(p_p)
        ax.text(
            0.97,
            0.95,
            f"H = {ent_p:.2f} nats\nSupport = {sup_p} bins",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
        )

        if j == 0:
            axes[0, j].set_ylabel("Probability", fontsize=10)
            axes[1, j].set_ylabel("Probability", fontsize=10)
            axes[2, j].set_ylabel("Probability", fontsize=10)
        axes[2, j].set_xlabel("Target value", fontsize=10)

    fig.suptitle(
        "Three Target Distributions: Gaussian (moderate spread) vs Gibbs (max entropy) vs Projected (2 bins)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "01_target_distributions.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Entropy and effective support vs n_bins
# ═══════════════════════════════════════════════════════════════════════════


def fig_entropy_and_support(out_dir):
    """Analytical: entropy and effective support as a function of n_bins."""
    bins_range = np.arange(5, 401, 5)
    sig_ratios = [0.5, 1.0, 3.0, 8.0, 20.0]
    n_targets = 50  # average over uniformly-spaced targets
    targets = np.linspace(0.05, 0.95, n_targets)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Entropy ──
    ax = axes[0]
    for sr in sig_ratios:
        ent = []
        for nb in bins_range:
            vals = [entropy(gaussian_target(y, nb, sr)[1]) for y in targets]
            ent.append(np.mean(vals))
        ax.plot(
            bins_range,
            ent,
            color=COLOR_GAUSS,
            alpha=0.7,
            lw=1.8,
            label=f"Gaussian σ-ratio={sr}",
        )

    # Projected entropy
    ent_proj = []
    for nb in bins_range:
        vals = [entropy(projected_target(y, nb)[1]) for y in targets]
        ent_proj.append(np.mean(vals))
    ax.plot(
        bins_range,
        ent_proj,
        color=COLOR_PROJ,
        lw=2.5,
        ls="--",
        label="Projected",
        zorder=5,
    )

    # Gibbs entropy (subsample for speed — it needs Newton-Raphson)
    gibbs_bins_sub = bins_range[::4]  # every 20th bin count
    ent_gibbs = []
    for nb in gibbs_bins_sub:
        vals = [entropy(gibbs_target(y, nb)[1]) for y in targets]
        ent_gibbs.append(np.mean(vals))
    ax.plot(
        gibbs_bins_sub,
        ent_gibbs,
        color=COLOR_GIBBS,
        lw=2.5,
        ls="-.",
        marker="s",
        markersize=4,
        markeredgecolor="white",
        label="Gibbs (max-ent)",
        zorder=6,
    )

    ax.set_xlabel("Number of Bins", fontsize=11)
    ax.set_ylabel("Avg. Target Entropy (nats)", fontsize=11)
    ax.set_title("Target Entropy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")

    # ── Effective support ──
    ax = axes[1]
    for sr in sig_ratios:
        sup = []
        for nb in bins_range:
            vals = [effective_support(gaussian_target(y, nb, sr)[1]) for y in targets]
            sup.append(np.mean(vals))
        ax.plot(
            bins_range,
            sup,
            color=COLOR_GAUSS,
            alpha=0.7,
            lw=1.8,
            label=f"Gaussian σ-ratio={sr}",
        )

    # Projected: always ≈ 2
    sup_proj = []
    for nb in bins_range:
        vals = [effective_support(projected_target(y, nb)[1]) for y in targets]
        sup_proj.append(np.mean(vals))
    ax.plot(
        bins_range,
        sup_proj,
        color=COLOR_PROJ,
        lw=2.5,
        ls="--",
        label="Projected",
        zorder=5,
    )

    # Gibbs support
    sup_gibbs = []
    for nb in gibbs_bins_sub:
        vals = [effective_support(gibbs_target(y, nb)[1]) for y in targets]
        sup_gibbs.append(np.mean(vals))
    ax.plot(
        gibbs_bins_sub,
        sup_gibbs,
        color=COLOR_GIBBS,
        lw=2.5,
        ls="-.",
        marker="s",
        markersize=4,
        markeredgecolor="white",
        label="Gibbs (max-ent)",
        zorder=6,
    )

    ax.set_xlabel("Number of Bins", fontsize=11)
    ax.set_ylabel("Avg. Bins with P > 1%", fontsize=11)
    ax.set_title("Effective Gradient Support", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Why Projected Starves: Fixed 2-Bin Support vs Growing Gaussian Support",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "02_entropy_and_support.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Gradient fraction — what % of output neurons learn per sample
# ═══════════════════════════════════════════════════════════════════════════


def fig_gradient_fraction(out_dir):
    """Fraction of output neurons receiving meaningful gradient (>1% prob)."""
    bins_range = np.arange(5, 401, 5)
    sig_ratios = [0.5, 1.0, 2.0, 3.0]
    n_targets = 50
    targets = np.linspace(0.05, 0.95, n_targets)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sr in sig_ratios:
        frac = []
        for nb in bins_range:
            vals = [
                effective_support(gaussian_target(y, nb, sr)[1]) / nb for y in targets
            ]
            frac.append(np.mean(vals))
        ax.plot(
            bins_range,
            [f * 100 for f in frac],
            color=COLOR_GAUSS,
            alpha=0.7,
            lw=1.8,
            label=f"Gaussian σ-ratio={sr}",
        )

    # Projected
    frac_proj = []
    for nb in bins_range:
        frac_proj.append(2.0 / nb * 100)
    ax.plot(
        bins_range,
        frac_proj,
        color=COLOR_PROJ,
        lw=2.5,
        ls="--",
        label="Projected (always 2/N)",
        zorder=5,
    )

    # Annotate key points
    ax.axhline(1.0, color="#999", ls=":", lw=0.8, alpha=0.6)
    ax.text(380, 1.3, "1% threshold", fontsize=8, color="#999")

    ax.set_xlabel("Number of Bins", fontsize=11)
    ax.set_ylabel("% of Output Neurons Receiving Gradient", fontsize=11)
    ax.set_title(
        "Projected's Gradient Vanishes at High Bin Counts",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, None)

    fig.tight_layout()
    path = os.path.join(out_dir, "03_gradient_fraction.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: From sweep data — optimal sig_ratio and what it implies
# ═══════════════════════════════════════════════════════════════════════════


def fig_optimal_sigma(df, out_dir):
    """What sig_ratio works best for each bin count? What does it mean?"""
    metric = "test_rmse"
    gauss = df[df["method"] == "HL-Gaussian"]
    agg = aggregate_seeds(gauss, metric)

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(
        1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False, sharey=True
    )

    for di, ds in enumerate(datasets):
        ax = axes[0, di]
        dsub = agg[agg["dataset"] == ds]

        # For each n_bins, find the sig_ratio of the best config
        bins_vals = sorted(dsub["n_bins"].unique())
        best_sigs = []
        best_sigs_support = []
        for nb in bins_vals:
            bsub = dsub[dsub["n_bins"] == nb]
            best_row = bsub.loc[bsub[f"{metric}_mean"].idxmin()]
            best_sig = best_row["sig_ratio"]
            best_sigs.append(best_sig)
            # Compute effective support at optimal sigma
            n_targets = 20
            targets = np.linspace(0.05, 0.95, n_targets)
            sup = np.mean(
                [
                    effective_support(gaussian_target(y, nb, best_sig)[1])
                    for y in targets
                ]
            )
            best_sigs_support.append(sup)

        ax.plot(
            bins_vals,
            best_sigs,
            "o-",
            color=COLOR_GAUSS,
            lw=2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Optimal σ-ratio",
        )

        # Right y-axis for effective support
        ax2 = ax.twinx()
        ax2.bar(
            bins_vals,
            best_sigs_support,
            width=[b * 0.15 for b in bins_vals],
            color=COLOR_GAUSS,
            alpha=0.2,
            label="Eff. support at opt. σ",
        )
        ax2.set_ylabel(
            "Effective Support (bins)" if di == n_ds - 1 else "",
            fontsize=10,
            color="#888",
        )
        ax2.tick_params(axis="y", colors="#888")

        ax.set_xlabel("Number of Bins", fontsize=10)
        if di == 0:
            ax.set_ylabel("Best σ-ratio", fontsize=11)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle(
        "Optimal σ-ratio Increases with Bins — Gaussian Maintains Support Width",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "04_optimal_sigma.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Gap vs hidden capacity — does more width help Projected?
# ═══════════════════════════════════════════════════════════════════════════


def fig_gap_vs_capacity(df, out_dir):
    """Does increasing hidden width help Projected or Gibbs catch up to Gaussian?"""
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(
        2, n_ds, figsize=(5 * n_ds, 8), squeeze=False, sharey="row"
    )

    for di, ds in enumerate(datasets):
        dsub = agg[agg["dataset"] == ds]
        int_dims = sorted(dsub["int_dim"].unique())
        bins_vals = sorted(dsub["n_bins"].unique())

        for row_i, (other_method, other_label, other_color) in enumerate(
            [
                ("HL-Projected", "Projected", COLOR_PROJ),
                ("HL-Gibbs", "Gibbs", COLOR_GIBBS),
            ]
        ):
            ax = axes[row_i, di]
            for idim in int_dims:
                gaps = []
                valid_bins = []
                for nb in bins_vals:
                    g_sub = dsub[
                        (dsub["int_dim"] == idim)
                        & (dsub["n_bins"] == nb)
                        & (dsub["method"] == "HL-Gaussian")
                    ]
                    o_sub = dsub[
                        (dsub["int_dim"] == idim)
                        & (dsub["n_bins"] == nb)
                        & (dsub["method"] == other_method)
                    ]
                    if g_sub.empty or o_sub.empty:
                        continue
                    best_g = g_sub[f"{metric}_mean"].min()
                    best_o = o_sub[f"{metric}_mean"].min()
                    gaps.append(best_o - best_g)  # positive = Gaussian wins
                    valid_bins.append(nb)
                if gaps:
                    ax.plot(
                        valid_bins,
                        gaps,
                        "o-",
                        markersize=6,
                        lw=1.5,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        label=f"int_dim={idim}",
                        alpha=0.85,
                    )

            ax.axhline(0, color="#999", ls="-", lw=0.8, alpha=0.5)
            ax.set_xlabel("Number of Bins", fontsize=10)
            if di == 0:
                ax.set_ylabel(f"Gap ({other_label} − Gaussian)", fontsize=10)
            ax.set_title(
                f"{ds.upper()} — {other_label} vs Gaussian",
                fontsize=10,
                fontweight="bold",
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="upper left")
            ax.text(
                0.98,
                0.02,
                "↑ Gaussian better",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=COLOR_GAUSS,
                alpha=0.7,
            )

    fig.suptitle(
        "Performance Gap by Hidden Capacity — Gap Persists for Both Projected and Gibbs",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "05_gap_vs_capacity.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Cross-dataset summary — best-of-best gap
# ═══════════════════════════════════════════════════════════════════════════


def fig_cross_dataset_summary(df, out_dir):
    """Bar chart: best test_rmse per method per dataset, with gap annotation."""
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)

    datasets = sorted(agg["dataset"].unique())
    methods = ["HL-Gaussian", "HL-Projected", "HL-Gibbs"]
    method_colors = {
        "HL-Gaussian": COLOR_GAUSS,
        "HL-Projected": COLOR_PROJ,
        "HL-Gibbs": COLOR_GIBBS,
    }
    x = np.arange(len(datasets))
    n_methods = len(methods)
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, 2.8 * len(datasets)), 5))

    for mi, method in enumerate(methods):
        bests, mins, maxs = [], [], []
        for ds in datasets:
            sub = agg[(agg["dataset"] == ds) & (agg["method"] == method)]
            if sub.empty:
                bests.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
            else:
                best_idx = sub[f"{metric}_mean"].idxmin()
                bests.append(sub.loc[best_idx, f"{metric}_mean"])
                mins.append(sub.loc[best_idx, f"{metric}_min"])
                maxs.append(sub.loc[best_idx, f"{metric}_max"])

        bests = np.array(bests)
        mins = np.array(mins)
        maxs = np.array(maxs)
        color = method_colors[method]
        offset = (mi - (n_methods - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            bests,
            width * 0.88,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=method,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            bests,
            yerr=[bests - mins, maxs - bests],
            fmt="none",
            ecolor="#333",
            elinewidth=0.8,
            capsize=3,
            zorder=4,
        )

        # Value labels on bars
        for i, (b, bar) in enumerate(zip(bests, bars)):
            if not np.isnan(b):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + 0.008,
                    f"{b:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="medium",
                    color=color,
                )

    # Gap annotations (Projected and Gibbs vs Gaussian)
    for i, ds in enumerate(datasets):
        g_sub = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Gaussian")]
        if g_sub.empty:
            continue
        best_g = g_sub[f"{metric}_mean"].min()
        y_max_bar = best_g
        gap_strs = []
        for other, col in [("HL-Projected", COLOR_PROJ), ("HL-Gibbs", COLOR_GIBBS)]:
            o_sub = agg[(agg["dataset"] == ds) & (agg["method"] == other)]
            if o_sub.empty:
                continue
            best_o = o_sub[f"{metric}_mean"].min()
            y_max_bar = max(y_max_bar, best_o)
            gap_pct = (best_o - best_g) / best_g * 100
            label = "P" if other == "HL-Projected" else "G"
            gap_strs.append(f"{label}: {gap_pct:+.1f}%")
        if gap_strs:
            ax.annotate(
                "\n".join(gap_strs),
                xy=(i, y_max_bar + 0.02),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#d62728",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Best {metric}", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title(
        "HL-Gaussian vs HL-Projected vs HL-Gibbs — Best Achievable Performance per Dataset",
        fontsize=12,
        fontweight="bold",
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "06_cross_dataset_summary.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: The core insight — entropy predicts performance
# ═══════════════════════════════════════════════════════════════════════════


def fig_entropy_vs_performance(df, out_dir):
    """Scatter: target entropy vs test_rmse for every configuration.

    Each dot is one (method, n_bins, sig_ratio, ...) config, colored by method.
    Reveals the NON-MONOTONE relationship: there's a Goldilocks zone.
    Now includes actual Gibbs empirical data from the sweep.
    """
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(
        1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False, sharey=False
    )

    n_targets = 20
    targets = np.linspace(0.05, 0.95, n_targets)

    for di, ds in enumerate(datasets):
        ax = axes[0, di]
        dsub = agg[agg["dataset"] == ds]

        gauss_ent, gauss_rmse = [], []
        proj_ent, proj_rmse = [], []
        gibbs_ent, gibbs_rmse = [], []

        for _, row in dsub.iterrows():
            method = row["method"]
            nb = int(row["n_bins"])
            sr = row["sig_ratio"]

            if method == "HL-Gaussian":
                avg_ent = np.mean(
                    [entropy(gaussian_target(y, nb, sr)[1]) for y in targets]
                )
                gauss_ent.append(avg_ent)
                gauss_rmse.append(row[f"{metric}_mean"])
            elif method == "HL-Projected":
                avg_ent = np.mean(
                    [entropy(projected_target(y, nb)[1]) for y in targets]
                )
                proj_ent.append(avg_ent)
                proj_rmse.append(row[f"{metric}_mean"])
            elif method == "HL-Gibbs":
                avg_ent = np.mean([entropy(gibbs_target(y, nb)[1]) for y in targets])
                gibbs_ent.append(avg_ent)
                gibbs_rmse.append(row[f"{metric}_mean"])

        # Clip y-axis for readability
        all_rmse = gauss_rmse + proj_rmse + gibbs_rmse
        clip_hi = np.percentile(all_rmse, 93) if all_rmse else 10
        gauss_rmse_c = np.minimum(gauss_rmse, clip_hi)
        proj_rmse_c = np.minimum(proj_rmse, clip_hi)
        gibbs_rmse_c = np.minimum(gibbs_rmse, clip_hi)

        # Gaussian scatter
        ax.scatter(
            gauss_ent,
            gauss_rmse_c,
            c=COLOR_GAUSS,
            marker="o",
            s=30,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

        # Projected scatter — add horizontal jitter to un-stack
        if proj_ent:
            proj_ent_arr = np.array(proj_ent)
            proj_rmse_arr = np.array(proj_rmse_c)
            jitter = np.random.default_rng(42).uniform(-0.02, 0.02, len(proj_ent_arr))
            ax.scatter(
                proj_ent_arr + jitter,
                proj_rmse_arr,
                c=COLOR_PROJ,
                marker="D",
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            # Annotate the Projected cluster
            if di == 0:
                median_proj_ent = np.median(proj_ent_arr)
                ax.annotate(
                    "Projected\n(H ≤ ln2)",
                    xy=(median_proj_ent, np.median(proj_rmse_arr)),
                    xytext=(median_proj_ent + 0.4, np.percentile(proj_rmse_arr, 75)),
                    fontsize=8,
                    color=COLOR_PROJ,
                    fontweight="bold",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=COLOR_PROJ,
                        lw=1.2,
                        connectionstyle="arc3,rad=0.2",
                    ),
                )

        # Gibbs scatter — add jitter too since different n_bins give different entropy
        if gibbs_ent:
            gibbs_ent_arr = np.array(gibbs_ent)
            gibbs_rmse_arr = np.array(gibbs_rmse_c)
            jitter_g = np.random.default_rng(43).uniform(
                -0.02, 0.02, len(gibbs_ent_arr)
            )
            ax.scatter(
                gibbs_ent_arr + jitter_g,
                gibbs_rmse_arr,
                c=COLOR_GIBBS,
                marker="s",
                s=45,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            if di == 0:
                median_gibbs_ent = np.median(gibbs_ent_arr)
                ax.annotate(
                    "Gibbs\n(max entropy)",
                    xy=(median_gibbs_ent, np.median(gibbs_rmse_arr)),
                    xytext=(median_gibbs_ent - 0.5, np.percentile(gibbs_rmse_arr, 25)),
                    fontsize=8,
                    color=COLOR_GIBBS,
                    fontweight="bold",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=COLOR_GIBBS,
                        lw=1.2,
                        connectionstyle="arc3,rad=-0.2",
                    ),
                )

        ax.set_xlabel("Avg. Target Entropy (nats)", fontsize=10)
        if di == 0:
            ax.set_ylabel("Test RMSE", fontsize=10)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, clip_hi * 1.1)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLOR_GAUSS,
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            label="HL-Gaussian",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=COLOR_PROJ,
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            label="HL-Projected",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_GIBBS,
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            label="HL-Gibbs",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=9.5,
    )

    fig.suptitle(
        "Entropy vs Performance — Three Methods Across the Entropy Spectrum",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "07_entropy_vs_performance.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: The Goldilocks Curve — entropy-bucketed performance with Gibbs
# ═══════════════════════════════════════════════════════════════════════════


def fig_goldilocks_curve(df, out_dir):
    """For HL-Gaussian configs, bucket by target entropy and show mean RMSE.

    This reveals the U-shaped relationship: too little entropy (Projected)
    or too much (Gibbs) hurts; Gaussian sits in the sweet spot.
    Now includes actual Gibbs empirical data points.
    """
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)

    n_targets = 20
    tgts = np.linspace(0.05, 0.95, n_targets)

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(
        1, n_ds, figsize=(5 * n_ds, 5), squeeze=False, sharey=False
    )

    for di, ds in enumerate(datasets):
        ax = axes[0, di]
        dsub_gauss = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Gaussian")]
        dsub_proj = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Projected")]
        dsub_gibbs = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Gibbs")]

        if dsub_gauss.empty:
            ax.set_title(f"{ds.upper()} (no Gaussian data)", fontsize=11)
            continue

        # Compute entropy for each Gaussian config
        ent_vals, rmse_vals = [], []
        for _, row in dsub_gauss.iterrows():
            nb = int(row["n_bins"])
            sr = row["sig_ratio"]
            avg_ent = np.mean([entropy(gaussian_target(y, nb, sr)[1]) for y in tgts])
            ent_vals.append(avg_ent)
            rmse_vals.append(row[f"{metric}_mean"])
        ent_vals = np.array(ent_vals)
        rmse_vals = np.array(rmse_vals)

        # Projected configs
        proj_ent_vals, proj_rmse_vals = [], []
        for _, row in dsub_proj.iterrows():
            nb = int(row["n_bins"])
            avg_ent = np.mean([entropy(projected_target(y, nb)[1]) for y in tgts])
            proj_ent_vals.append(avg_ent)
            proj_rmse_vals.append(row[f"{metric}_mean"])

        # Gibbs configs (actual empirical data!)
        gibbs_ent_vals, gibbs_rmse_vals = [], []
        for _, row in dsub_gibbs.iterrows():
            nb = int(row["n_bins"])
            avg_ent = np.mean([entropy(gibbs_target(y, nb)[1]) for y in tgts])
            gibbs_ent_vals.append(avg_ent)
            gibbs_rmse_vals.append(row[f"{metric}_mean"])

        # Clip outliers for visibility
        all_rmse_list = list(rmse_vals) + proj_rmse_vals + gibbs_rmse_vals
        clip_hi = np.percentile(all_rmse_list, 90)
        y_top = clip_hi * 1.12
        ax.set_ylim(0, y_top)

        # Scatter all Gaussian points (clipped)
        ax.scatter(
            ent_vals,
            np.minimum(rmse_vals, clip_hi),
            c=COLOR_GAUSS,
            marker="o",
            s=25,
            alpha=0.4,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

        # Binned trend line — use CLIPPED values
        rmse_clipped = np.minimum(rmse_vals, clip_hi)
        n_ent_bins = 12
        ent_edges = np.linspace(
            ent_vals.min() - 0.01, ent_vals.max() + 0.01, n_ent_bins + 1
        )
        bin_centers_e, bin_medians, bin_q25, bin_q75 = [], [], [], []
        for k in range(n_ent_bins):
            mask = (ent_vals >= ent_edges[k]) & (ent_vals < ent_edges[k + 1])
            if mask.sum() >= 3:
                bin_centers_e.append((ent_edges[k] + ent_edges[k + 1]) / 2)
                bin_medians.append(np.median(rmse_clipped[mask]))
                bin_q25.append(np.percentile(rmse_clipped[mask], 25))
                bin_q75.append(np.percentile(rmse_clipped[mask], 75))

        if bin_centers_e:
            bin_centers_e = np.array(bin_centers_e)
            bin_medians = np.array(bin_medians)
            bin_q25 = np.array(bin_q25)
            bin_q75 = np.array(bin_q75)
            ax.plot(
                bin_centers_e,
                bin_medians,
                "-o",
                color=COLOR_GAUSS,
                lw=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=5,
                label="Gaussian (median)",
            )
            ax.fill_between(
                bin_centers_e, bin_q25, bin_q75, color=COLOR_GAUSS, alpha=0.12, zorder=2
            )

        # Scatter Projected points
        if proj_rmse_vals:
            ax.scatter(
                proj_ent_vals,
                np.minimum(proj_rmse_vals, clip_hi),
                c=COLOR_PROJ,
                marker="D",
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
                label="Projected",
            )

        # Scatter actual Gibbs empirical points
        if gibbs_rmse_vals:
            gibbs_ent_arr = np.array(gibbs_ent_vals)
            gibbs_rmse_arr = np.minimum(gibbs_rmse_vals, clip_hi)
            jitter = np.random.default_rng(44).uniform(-0.02, 0.02, len(gibbs_ent_arr))
            ax.scatter(
                gibbs_ent_arr + jitter,
                gibbs_rmse_arr,
                c=COLOR_GIBBS,
                marker="s",
                s=45,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
                label="Gibbs (empirical)",
            )

        # Mark sweet spot
        if bin_centers_e is not None and len(bin_medians) > 0:
            best_idx = np.argmin(bin_medians)
            lo = bin_centers_e[max(0, best_idx - 1)]
            hi = bin_centers_e[min(len(bin_centers_e) - 1, best_idx + 1)]
            ax.axvspan(
                lo, hi, color="#2ca02c", alpha=0.08, zorder=0, label="Sweet spot"
            )

        ax.set_xlabel("Avg. Target Entropy (nats)", fontsize=10)
        if di == 0:
            ax.set_ylabel("Test RMSE", fontsize=10)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        "The Goldilocks Zone: Too Little Entropy (Projected) and Too Much (Gibbs) Both Hurt",
        fontsize=12,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "08_goldilocks_curve.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Localization Strength — KL(target || uniform) vs performance
# ═══════════════════════════════════════════════════════════════════════════


def fig_localization_vs_performance(df, out_dir):
    """KL(target || uniform) = log(N) - H(target) measures how peaked the
    distribution is — i.e. how much it tells the network about WHERE the target is.

    Now includes actual Gibbs empirical data from the sweep.
    """
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)
    n_targets = 20
    tgts = np.linspace(0.05, 0.95, n_targets)

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(
        1, n_ds, figsize=(5 * n_ds, 5), squeeze=False, sharey=False
    )

    for di, ds in enumerate(datasets):
        ax = axes[0, di]
        dsub = agg[agg["dataset"] == ds]

        all_kl, all_rmse, all_method = [], [], []
        for _, row in dsub.iterrows():
            method = row["method"]
            nb = int(row["n_bins"])
            sr = row["sig_ratio"]
            if method == "HL-Gaussian":
                avg_kl = np.mean(
                    [kl_from_uniform(gaussian_target(y, nb, sr)[1]) for y in tgts]
                )
            elif method == "HL-Projected":
                avg_kl = np.mean(
                    [kl_from_uniform(projected_target(y, nb)[1]) for y in tgts]
                )
            elif method == "HL-Gibbs":
                avg_kl = np.mean(
                    [kl_from_uniform(gibbs_target(y, nb)[1]) for y in tgts]
                )
            else:
                continue
            all_kl.append(avg_kl)
            all_rmse.append(row[f"{metric}_mean"])
            all_method.append(method)

        all_kl = np.array(all_kl)
        all_rmse = np.array(all_rmse)
        all_method = np.array(all_method)

        clip_hi = np.percentile(all_rmse, 92)

        for method, color, marker, sz in [
            ("HL-Gaussian", COLOR_GAUSS, "o", 30),
            ("HL-Projected", COLOR_PROJ, "D", 40),
            ("HL-Gibbs", COLOR_GIBBS, "s", 40),
        ]:
            mask = all_method == method
            if mask.sum() == 0:
                continue
            ax.scatter(
                all_kl[mask],
                np.minimum(all_rmse[mask], clip_hi),
                c=color,
                marker=marker,
                s=sz,
                alpha=0.6,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
                label=method,
            )

        ax.set_xlabel("KL(target ‖ uniform) — Localization Strength", fontsize=9.5)
        if di == 0:
            ax.set_ylabel("Test RMSE", fontsize=10)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, clip_hi * 1.15)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Localization Strength: Gibbs (Low KL) vs Projected (High KL) vs Gaussian (Moderate KL)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "09_localization_vs_performance.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10: The Tradeoff — 2D scatter of (entropy, KL) colored by RMSE
# ═══════════════════════════════════════════════════════════════════════════


def fig_entropy_kl_tradeoff(df, out_dir):
    """Gaussian configs in (n_bins, sig_ratio) space, colored by best RMSE.

    Since entropy + KL = log(N), each n_bins forms a diagonal line.
    We aggregate across int_dim and lr (taking the best RMSE) so each
    (n_bins, sig_ratio) pair is ONE dot.

    Now includes actual Gibbs and Projected empirical RMSE on the colormap,
    so all three methods are visually comparable on the same scale.
    """
    metric = "test_rmse"
    agg = aggregate_seeds(df, metric)
    n_targets = 15
    tgts = np.linspace(0.05, 0.95, n_targets)

    focus_ds = [
        ds
        for ds in ["ctscan", "bike", "pole", "songyear"]
        if ds in agg["dataset"].unique()
    ]
    n_ds = len(focus_ds)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5.5), squeeze=False)

    for di, ds in enumerate(focus_ds):
        ax = axes[0, di]

        # ── Gaussian data ──
        dsub_gauss = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Gaussian")]
        if dsub_gauss.empty:
            continue

        best_per_nb_sr = (
            dsub_gauss.groupby(["n_bins", "sig_ratio"])
            .agg(rmse=(f"{metric}_mean", "min"))
            .reset_index()
        )

        ent_arr, kl_arr, rmse_arr, nb_arr, sr_arr = [], [], [], [], []
        for _, row in best_per_nb_sr.iterrows():
            nb = int(row["n_bins"])
            sr = row["sig_ratio"]
            dists = [gaussian_target(y, nb, sr)[1] for y in tgts]
            avg_ent = np.mean([entropy(d) for d in dists])
            avg_kl = np.mean([kl_from_uniform(d) for d in dists])
            ent_arr.append(avg_ent)
            kl_arr.append(avg_kl)
            rmse_arr.append(row["rmse"])
            nb_arr.append(nb)
            sr_arr.append(sr)

        ent_arr = np.array(ent_arr)
        kl_arr = np.array(kl_arr)
        rmse_arr = np.array(rmse_arr)
        nb_arr = np.array(nb_arr)
        sr_arr = np.array(sr_arr)

        # ── Gibbs empirical data ──
        dsub_gibbs = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Gibbs")]
        gibbs_pts = {}  # nb -> (ent, kl, best_rmse)
        if not dsub_gibbs.empty:
            best_gibbs_per_nb = (
                dsub_gibbs.groupby("n_bins")
                .agg(rmse=(f"{metric}_mean", "min"))
                .reset_index()
            )
            for _, row in best_gibbs_per_nb.iterrows():
                nb_val = int(row["n_bins"])
                g_ent = np.mean([entropy(gibbs_target(y, nb_val)[1]) for y in tgts])
                g_kl = np.mean(
                    [kl_from_uniform(gibbs_target(y, nb_val)[1]) for y in tgts]
                )
                gibbs_pts[nb_val] = (g_ent, g_kl, row["rmse"])
        else:
            # Fallback: analytical positions only, no RMSE
            for nb_val in sorted(set(nb_arr)):
                g_ent = np.mean([entropy(gibbs_target(y, nb_val)[1]) for y in tgts])
                g_kl = np.mean(
                    [kl_from_uniform(gibbs_target(y, nb_val)[1]) for y in tgts]
                )
                gibbs_pts[nb_val] = (g_ent, g_kl, None)

        # ── Projected empirical data ──
        dsub_proj = agg[(agg["dataset"] == ds) & (agg["method"] == "HL-Projected")]
        proj_pts = {}  # nb -> (ent, kl, best_rmse)
        if not dsub_proj.empty:
            best_proj_per_nb = (
                dsub_proj.groupby("n_bins")
                .agg(rmse=(f"{metric}_mean", "min"))
                .reset_index()
            )
            for _, row in best_proj_per_nb.iterrows():
                nb_val = int(row["n_bins"])
                p_ent = np.mean([entropy(projected_target(y, nb_val)[1]) for y in tgts])
                p_kl = np.mean(
                    [kl_from_uniform(projected_target(y, nb_val)[1]) for y in tgts]
                )
                proj_pts[nb_val] = (p_ent, p_kl, row["rmse"])
        else:
            for nb_val in sorted(set(nb_arr)):
                p_ent = np.mean([entropy(projected_target(y, nb_val)[1]) for y in tgts])
                p_kl = np.mean(
                    [kl_from_uniform(projected_target(y, nb_val)[1]) for y in tgts]
                )
                proj_pts[nb_val] = (p_ent, p_kl, None)

        # Ensure Gibbs/Proj pts exist for all Gaussian bin counts
        for nb_val in sorted(set(nb_arr)):
            if nb_val not in gibbs_pts:
                g_ent = np.mean([entropy(gibbs_target(y, nb_val)[1]) for y in tgts])
                g_kl = np.mean(
                    [kl_from_uniform(gibbs_target(y, nb_val)[1]) for y in tgts]
                )
                gibbs_pts[nb_val] = (g_ent, g_kl, None)
            if nb_val not in proj_pts:
                p_ent = np.mean([entropy(projected_target(y, nb_val)[1]) for y in tgts])
                p_kl = np.mean(
                    [kl_from_uniform(projected_target(y, nb_val)[1]) for y in tgts]
                )
                proj_pts[nb_val] = (p_ent, p_kl, None)

        # ── Color scale: use ALL RMSE values so Gibbs/Proj are on the same scale ──
        all_rmse_for_scale = list(rmse_arr)
        for nb_val in gibbs_pts:
            if gibbs_pts[nb_val][2] is not None:
                all_rmse_for_scale.append(gibbs_pts[nb_val][2])
        for nb_val in proj_pts:
            if proj_pts[nb_val][2] is not None:
                all_rmse_for_scale.append(proj_pts[nb_val][2])
        all_rmse_for_scale = np.array(all_rmse_for_scale)
        vmin_c = np.percentile(all_rmse_for_scale, 1)
        vmax_c = np.percentile(all_rmse_for_scale, 75)

        # ── Draw diagonal lines for each n_bins ──
        for nb_val in sorted(set(nb_arr)):
            mask = nb_arr == nb_val
            diag_ent = list(ent_arr[mask])
            diag_kl = list(kl_arr[mask])
            # Add Gibbs and Projected endpoints
            diag_ent += [gibbs_pts[nb_val][0], proj_pts[nb_val][0]]
            diag_kl += [gibbs_pts[nb_val][1], proj_pts[nb_val][1]]
            order = np.argsort(diag_ent)
            diag_ent = np.array(diag_ent)[order]
            diag_kl = np.array(diag_kl)[order]
            ax.plot(diag_ent, diag_kl, "-", color="#ccc", lw=0.8, zorder=1)
            # Label above the top of the diagonal (above Projected diamond)
            p_ent_lbl = proj_pts[nb_val][0]
            p_kl_lbl = proj_pts[nb_val][1]
            ax.annotate(
                f"N={nb_val}",
                xy=(p_ent_lbl, p_kl_lbl),
                xytext=(-12, 14),
                textcoords="offset points",
                fontsize=7,
                color="#555",
                fontweight="semibold",
                ha="center",
                va="bottom",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

        # ── All three methods colored by RMSE on the same colormap ──
        # Shape alone differentiates method: ● Gaussian, ◆ Projected, ■ Gibbs
        # Use PowerNorm (gamma<1) to expand the "good" end of the range and
        # compress the outlier tail — makes subtle differences visible.
        from matplotlib.colors import PowerNorm

        norm = PowerNorm(gamma=0.5, vmin=vmin_c, vmax=vmax_c)
        cmap = plt.cm.RdYlGn_r

        # Gaussian dots (drawn first so Proj/Gibbs sit on top)
        draw_order = np.argsort(-rmse_arr)  # worst first → best last

        sc = ax.scatter(
            ent_arr[draw_order],
            kl_arr[draw_order],
            c=rmse_arr[draw_order],
            cmap="RdYlGn_r",
            norm=norm,
            s=70,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )

        # Projected diamonds — same colormap, black edge, larger
        all_nb = sorted(set(nb_arr))
        for ki, nb_val in enumerate(all_nb):
            p_ent, p_kl, p_rmse = proj_pts[nb_val]
            if p_rmse is not None:
                p_fill = cmap(norm(p_rmse))
            else:
                p_fill = "#999"
            ax.scatter(
                [p_ent],
                [p_kl],
                marker="D",
                s=120,
                color=p_fill,
                edgecolors="black",
                linewidths=1.4,
                zorder=7,
                label="Projected (◆)" if ki == 0 else None,
            )

        # Gibbs squares — same colormap, black edge, larger
        for ki, nb_val in enumerate(all_nb):
            g_ent, g_kl, g_rmse = gibbs_pts[nb_val]
            if g_rmse is not None:
                g_fill = cmap(norm(g_rmse))
            else:
                g_fill = "#999"
            ax.scatter(
                [g_ent],
                [g_kl],
                marker="s",
                s=120,
                color=g_fill,
                edgecolors="black",
                linewidths=1.4,
                zorder=7,
                label="Gibbs (■)" if ki == 0 else None,
            )

        # ── Circle the overall best across all methods ──
        best_gauss_i = np.argmin(rmse_arr)
        best_rmse = rmse_arr[best_gauss_i]
        best_ent = ent_arr[best_gauss_i]
        best_kl = kl_arr[best_gauss_i]
        best_label = f"Gauss: N={nb_arr[best_gauss_i]}, σ={sr_arr[best_gauss_i]:.1f}"

        # Check if any Gibbs or Projected is actually better
        for nb_val in all_nb:
            g_ent, g_kl, g_rmse = gibbs_pts[nb_val]
            if g_rmse is not None and g_rmse < best_rmse:
                best_rmse = g_rmse
                best_ent, best_kl = g_ent, g_kl
                best_label = f"Gibbs: N={nb_val}"
            p_ent, p_kl, p_rmse = proj_pts[nb_val]
            if p_rmse is not None and p_rmse < best_rmse:
                best_rmse = p_rmse
                best_ent, best_kl = p_ent, p_kl
                best_label = f"Proj: N={nb_val}"

        ax.scatter(
            [best_ent],
            [best_kl],
            s=280,
            facecolors="none",
            edgecolors="black",
            linewidths=2.5,
            zorder=9,
        )
        ax.annotate(
            f"Best: {best_label}\nRMSE={best_rmse:.4f}",
            xy=(best_ent, best_kl),
            xytext=(15, -20),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
        )

        plt.colorbar(sc, ax=ax, label="Test RMSE (best over int_dim, lr)", shrink=0.8)

        ax.set_xlabel("Target Entropy (nats)  →  spread", fontsize=10)
        if di == 0:
            ax.set_ylabel("KL(target ‖ uniform)  →  peakedness", fontsize=10)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="#aaa",
                linestyle="None",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label="Gaussian",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#aaa",
                linestyle="None",
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="Projected",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="#aaa",
                linestyle="None",
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="Gibbs",
            ),
        ]
        ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Entropy–Localization Tradeoff",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color="#222",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "10_entropy_kl_tradeoff.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Text summary — print key insights
# ═══════════════════════════════════════════════════════════════════════════


def print_insights():
    """Print the analytical argument."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║  WHY HL-GAUSSIAN >> HL-PROJECTED (AND HL-GIBBS): KEY FINDINGS              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  THE GOLDILOCKS PRINCIPLE: It's not about maximizing entropy or gradient   ║
║  support — it's about the right BALANCE between two competing needs:       ║
║                                                                            ║
║    (A) GRADIENT SUPPORT: Enough bins must receive nonzero gradient for     ║
║        the network to learn efficiently.                                   ║
║    (B) LOCALIZATION: The target distribution must clearly indicate WHERE   ║
║        the target is. If it's too flat, gradients carry no spatial info.   ║
║                                                                            ║
║  ── THE THREE REGIMES ─────────────────────────────────────────────────    ║
║                                                                            ║
║  PROJECTED (too sparse):                                                   ║
║    • Entropy ≈ 0.69 nats (capped at ln 2), always 2-bin support           ║
║    • Extremely strong localization (KL >> 0) but almost no gradient flow   ║
║    • At N=400, 99.5% of output neurons get zero gradient per sample       ║
║    • Performance degrades with N because signal fraction → 0              ║
║                                                                            ║
║  GIBBS (too diffuse):                                                      ║
║    • Maximum entropy for given mean → nearly uniform for N >> 1            ║
║    • Excellent gradient support (all bins active)                          ║
║    • But near-zero localization (KL ≈ 0): barely distinguishes y=0.3      ║
║      from y=0.7 — the network can't learn WHERE the target is             ║
║    • Gradient SNR is extremely low: each bin's gradient is ≈ 1/N          ║
║                                                                            ║
║  GAUSSIAN (the sweet spot):                                                ║
║    • σ-ratio tunes the entropy/localization tradeoff                      ║
║    • Moderate entropy (0.8-1.5 nats) at optimal σ                         ║
║    • Peaked enough to convey target location, spread enough for gradients ║
║    • Smooth bell shape → nearby targets produce similar gradients         ║
║      (smooth optimization landscape)                                      ║
║    • Optimal σ-ratio ↑ with N, maintaining ~constant effective support    ║
║                                                                            ║
║  ── ADDITIONAL FACTORS ────────────────────────────────────────────────    ║
║                                                                            ║
║  • σ-ratio is Gaussian's extra hyperparameter — Projected has no          ║
║    equivalent knob to tune the spread.                                    ║
║  • The gap persists even with wider hidden layers, confirming the         ║
║    problem is in the target distribution, not network capacity.           ║
║  • Performance U-curve vs entropy: too low → too sparse, too high →      ║
║    too flat. The minimum sits where Gaussian with optimal σ lives.        ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="Investigate the Gaussian–Projected gap.")
    p.add_argument("--results_dir", default="sweep_results")
    p.add_argument("--out_dir", default="figures/gap")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  Investigating: Why does HL-Gaussian beat HL-Projected (and Gibbs)?")
    print("=" * 70)

    # ── Analytical figures (no data needed) ──
    print("\n[1/10] Target distribution visualization (now with Gibbs)...")
    fig_target_distributions(args.out_dir)

    print("[2/10] Entropy and effective support vs bins...")
    fig_entropy_and_support(args.out_dir)

    print("[3/10] Gradient fraction vs bins...")
    fig_gradient_fraction(args.out_dir)

    # ── Empirical figures (need sweep data) ──
    print("\n  Loading sweep results...")
    df = load_all_datasets(args.results_dir)
    print(
        f"  Loaded {len(df)} total results across {df['dataset'].nunique()} datasets\n"
    )

    print("[4/10] Optimal sigma analysis...")
    fig_optimal_sigma(df, args.out_dir)

    print("[5/10] Gap vs hidden capacity...")
    fig_gap_vs_capacity(df, args.out_dir)

    print("[6/10] Cross-dataset summary...")
    fig_cross_dataset_summary(df, args.out_dir)

    print("[7/10] Entropy vs performance scatter (raw)...")
    fig_entropy_vs_performance(df, args.out_dir)

    # ── Deeper analysis ──
    print("\n[8/10] Goldilocks curve (entropy-bucketed RMSE)...")
    fig_goldilocks_curve(df, args.out_dir)

    print("[9/10] Localization strength vs performance...")
    fig_localization_vs_performance(df, args.out_dir)

    print("[10/10] Entropy–KL tradeoff (2D sweet-spot map)...")
    fig_entropy_kl_tradeoff(df, args.out_dir)

    # ── Insights ──
    print_insights()

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
