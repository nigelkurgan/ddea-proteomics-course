"""
Intensity distribution plots.

WHAT TO LOOK FOR
----------------
- sample_boxplot    : After normalisation, all boxes should be roughly aligned.
                      Outlier samples appear as shifted boxes.
- density_per_sample: All curves should overlap closely.  A sample with a very
                      different peak location or shape is suspicious.
- intensity_histogram: The global distribution should be roughly unimodal and
                       symmetric in log2 space.
- protein_rank_abundance: A steep left shoulder means many low-abundance proteins;
                           the "knee" of the curve is where detection becomes unreliable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE


def plot_sample_boxplot(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    color_by: str = "plate",
    outlier_samples: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Box plot of per-sample intensity distributions.

    Each box represents one sample.  Samples are coloured by batch (plate) by
    default to make batch-driven shifts visible.  Confirmed outliers are
    annotated with a red marker.

    WHAT TO LOOK FOR:
      - All boxes at similar heights → normalisation worked
      - Systematic shifts within a plate group → residual batch effect
      - A box much higher or lower than all others → possible outlier
    """
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.12 + 2), 5))

    if metadata is not None and color_by in metadata.columns:
        labels = metadata[color_by].reindex(df.index).fillna("Unknown")
        unique_labels = labels.unique()
        color_map = {lb: PALETTE[i % len(PALETTE)] for i, lb in enumerate(unique_labels)}
        colors = [color_map[lb] for lb in labels]
    else:
        colors = [PALETTE[0]] * len(df)

    # Subsample for legibility if many samples
    if len(df) > 300:
        sample_idx = np.linspace(0, len(df) - 1, 300, dtype=int)
        df_plot = df.iloc[sample_idx]
        colors_plot = [colors[i] for i in sample_idx]
    else:
        df_plot = df
        colors_plot = colors

    positions = np.arange(len(df_plot))
    bplot = ax.boxplot(
        [df_plot.iloc[i].dropna().values for i in range(len(df_plot))],
        positions=positions,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1},
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        widths=0.7,
    )
    for patch, color in zip(bplot["boxes"], colors_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Mark confirmed outliers with a red X
    if outlier_samples:
        for i, idx in enumerate(df_plot.index):
            if idx in outlier_samples:
                ax.axvline(i, color="red", alpha=0.4, linewidth=1)

    ax.set_xticks([])
    ax.set_xlabel("Samples")
    ax.set_ylabel("log₂ intensity")
    ax.set_title("Per-sample intensity distribution")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_density_per_sample(
    df: pd.DataFrame,
    outlier_samples: list[str] | None = None,
    output_path: Path | None = None,
    max_samples: int = 100,
) -> None:
    """
    Kernel density estimate of intensity distribution for each sample.

    Draws all samples as light grey lines and highlights outlier samples
    in red.  Subsample to max_samples for legibility in large cohorts.
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose x grid from global min/max
    x_min = float(df.min().min())
    x_max = float(df.max().max())
    grid = np.linspace(x_min, x_max, 300)

    plot_samples = df.index.tolist()
    if len(plot_samples) > max_samples:
        rng = np.random.default_rng(42)
        plot_samples = rng.choice(plot_samples, max_samples, replace=False).tolist()

    for sample in plot_samples:
        vals = df.loc[sample].dropna().values
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals)
        color = "red" if (outlier_samples and sample in outlier_samples) else "#888888"
        alpha = 0.9 if (outlier_samples and sample in outlier_samples) else 0.15
        lw = 1.5 if (outlier_samples and sample in outlier_samples) else 0.6
        ax.plot(grid, kde(grid), color=color, alpha=alpha, linewidth=lw)

    ax.set_xlabel("log₂ intensity")
    ax.set_ylabel("Density")
    ax.set_title("Per-sample intensity density\n(red = flagged outlier)")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_protein_rank_abundance(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Rank-abundance (dynamic range) plot: proteins ranked by mean intensity.

    Shows how many proteins are reliably detected and their abundance range.
    The 'knee' of the curve is where proteins transition from confidently
    detected to borderline-detected.
    """
    means = df.mean(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(means)) + 1, means.values, lw=1.5, color=PALETTE[0])
    ax.set_xlabel("Protein rank (by mean intensity)")
    ax.set_ylabel("Mean log₂ intensity")
    ax.set_title(f"Protein rank-abundance curve  (n = {len(means)} proteins)")
    ax.set_xlim(1, len(means))
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
