"""
Missing-value visualisation plots.

WHAT TO LOOK FOR
----------------
- missing_heatmap     : Structured patterns (e.g. all proteins missing for one sample)
                        suggest a failed run.  Protein-level stripes suggest a
                        systematically low-abundance protein; not necessarily problematic.
- missing_by_threshold: Use the threshold barplot to choose your completeness cutoff
                        (see filters.py).  Pick the "knee" of the curve.
- missing_per_group   : If one group (plate, timepoint, condition) has consistently
                        higher missing rates, check whether that batch had technical issues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE


def plot_missing_heatmap(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    sort_by: str | None = "plate",
    output_path: Path | None = None,
    max_proteins: int = 500,
) -> None:
    """
    Heatmap of missing values: samples × proteins, black = missing.

    Samples are sorted by the `sort_by` column in metadata (e.g. plate) so
    batch-structured missingness is visually obvious.
    """
    # Subsample proteins for legibility
    df_plot = df.sample(min(max_proteins, df.shape[1]), axis=1, random_state=42) \
        if df.shape[1] > max_proteins else df

    if metadata is not None and sort_by and sort_by in metadata.columns:
        order = metadata[sort_by].reindex(df_plot.index).sort_values().index
        df_plot = df_plot.loc[order]

    fig, ax = plt.subplots(figsize=(12, max(4, len(df_plot) * 0.04 + 1)))
    # Present: white (1), missing: black (0)
    ax.imshow(df_plot.notna().values.T, aspect="auto", cmap="Greys_r",
              interpolation="none", vmin=0, vmax=1)
    ax.set_xlabel("Samples")
    ax.set_ylabel(f"Proteins (n={df_plot.shape[1]})")
    ax.set_title("Missing value pattern  (black = missing)")
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_missing_by_threshold(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Bar chart: number of proteins retained at each detection-rate threshold.

    Use this to choose the completeness_threshold parameter in the config.
    The vertical drop between consecutive thresholds shows how many proteins
    are lost by tightening the criterion.
    """
    from proteomics_qc.proteomics.filters import completeness_at_thresholds
    counts = completeness_at_thresholds(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"{t:.0%}" for t in counts.index], counts.values,
                  color=PALETTE[0], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(int(val)), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Detection rate threshold")
    ax.set_ylabel("Proteins retained")
    ax.set_title("Proteins retained vs. completeness threshold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_missing_per_group(
    df: pd.DataFrame,
    groups: pd.Series,
    output_path: Path | None = None,
) -> None:
    """
    Bar chart of mean missing-value rate per group (plate, condition, etc.).

    If one group has a systematically higher missing rate, it may indicate a
    technical problem with that batch (e.g. injection failure, short gradient).
    """
    miss = df.isna().mean(axis=1)   # per-sample missing rate
    grp = groups.reindex(df.index)
    group_miss = miss.groupby(grp).mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(group_miss) * 0.5 + 2), 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(group_miss))]
    ax.bar(group_miss.index.astype(str), group_miss.values * 100,
           color=colors, edgecolor="white")
    ax.set_xlabel("Group")
    ax.set_ylabel("Mean missing rate (%)")
    ax.set_title("Missing-value rate per group")
    ax.set_xticklabels(group_miss.index.astype(str), rotation=30, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
