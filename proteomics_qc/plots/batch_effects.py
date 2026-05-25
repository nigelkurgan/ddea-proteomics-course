"""
Batch-effect and PCA visualisation.

WHAT TO LOOK FOR
----------------
- plot_pca (colour=plate)  : If samples cluster tightly by plate colour, a batch
                             effect is present and correction is needed.
- plot_pca (colour=group)  : After correction, biological grouping should be the
                             main source of variance — ideally visible in PC1/PC2.
- plot_plate_distances     : Within < between = batch effect detected.
- plot_pc_factor_heatmap   : High -log10(p) for plate in PC1/2 = batch dominates;
                             high for biology in PC1/2 = signal preserved.
- plot_correlation_heatmap : Off-diagonal block structure by plate = batch effect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .style import PALETTE


def plot_pca(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    color_by: str = "plate",
    output_path: Path | None = None,
    title: str = "PCA",
    outlier_samples: list[str] | None = None,
) -> None:
    """
    2-component PCA scatter plot coloured by a metadata variable.

    The percentage of variance explained by each PC is shown on the axis
    label — PC1 typically explains 20–50 % in well-behaved proteomics data.

    Parameters
    ----------
    df             : fully-imputed (samples × proteins) DataFrame
    metadata       : DataFrame with sample-level variables
    color_by       : metadata column to use for colouring
    outlier_samples: these samples are drawn as ×, others as •
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df.values)
    pca_df = pd.DataFrame(coords, index=df.index, columns=["PC1", "PC2"])

    labels = metadata[color_by].reindex(df.index).fillna("Unknown") \
        if color_by in metadata.columns else pd.Series("All", index=df.index)
    unique_labels = sorted(labels.unique())
    color_map = {lb: PALETTE[i % len(PALETTE)] for i, lb in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for lb in unique_labels:
        mask = labels == lb
        ax.scatter(
            pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
            c=color_map[lb], label=str(lb), alpha=0.7, s=30, linewidths=0
        )

    # Mark outlier samples with ×
    if outlier_samples:
        for s in outlier_samples:
            if s in pca_df.index:
                ax.scatter(
                    pca_df.loc[s, "PC1"], pca_df.loc[s, "PC2"],
                    c="red", marker="x", s=80, linewidths=1.5, zorder=5
                )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, markerscale=1.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_plate_distances(
    within: list[float],
    between: list[float],
    ks_pvalue: float,
    output_path: Path | None = None,
) -> None:
    """
    Violin plot comparing within-plate vs. between-plate PCA distances.

    If within-plate distances are much smaller, samples are more similar
    within a plate than across plates — a sign of a batch effect.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [within, between]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [PALETTE[0], PALETTE[1]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Within plate", "Between plate"])
    ax.set_ylabel("Euclidean distance in PCA space")
    ax.set_title(f"Plate distance distribution\n(KS p = {ks_pvalue:.3g})")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_pc_factor_heatmap(
    pc_assoc: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Heatmap of PC × factor associations (-log10 p-value from ANOVA).

    Rows = PCA components; columns = covariates (plate, group, etc.).
    Bright cells indicate the covariate strongly explains that component.

    IDEAL RESULT AFTER QC:
      - Batch/plate: low values (not driving any PC)
      - Biology: high values in PC1/PC2 (biology is the main source of variance)
    """
    if pc_assoc.empty:
        return

    fig, ax = plt.subplots(figsize=(max(4, len(pc_assoc.columns) + 2),
                                    max(3, len(pc_assoc) * 0.5 + 1)))
    im = ax.imshow(pc_assoc.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pc_assoc.columns)))
    ax.set_xticklabels(pc_assoc.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pc_assoc.index)))
    ax.set_yticklabels(pc_assoc.index)
    for i in range(pc_assoc.shape[0]):
        for j in range(pc_assoc.shape[1]):
            v = pc_assoc.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 3 else "black")
    plt.colorbar(im, ax=ax, label="-log₁₀(p-value)")
    ax.set_title("PC × factor associations\n(ANOVA, -log₁₀ p)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    color_by: str = "plate",
    output_path: Path | None = None,
    max_samples: int = 200,
) -> None:
    """
    Sample × sample Pearson correlation heatmap.

    Samples are sorted by their batch label so within-batch blocks appear
    on the diagonal.  High within-batch correlation vs. cross-batch correlation
    = strong batch effect.
    """
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    if metadata is not None and color_by in metadata.columns:
        order = metadata[color_by].reindex(df.index).sort_values().index
        df = df.loc[order]

    corr = df.T.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Samples")
    ax.set_ylabel("Samples")
    ax.set_title(f"Sample correlation heatmap (sorted by {color_by})")
    plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
