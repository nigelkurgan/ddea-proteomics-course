"""
Coefficient of variation (CV) analysis.

KEY CONCEPT — what is CV and why does it matter?
-------------------------------------------------
Coefficient of variation = (standard deviation / mean) × 100 %

In proteomics, CV measures the technical reproducibility of protein
quantification.  Lower CV = more reproducible.

Three types of CV are computed:
  - Overall CV   : variability of each protein across ALL samples.
                   This combines technical AND biological variation.
  - Intra-individual CV: variability of a protein within the same subject
                   across repeated measurements (e.g. QC samples, pooled reference).
                   Reflects TECHNICAL reproducibility.
  - Inter-individual CV: variability of a protein between different subjects
                   (measured once each).  Reflects BIOLOGICAL variability.

Typical benchmarks for good DIA data:
  - Intra-individual CV < 20 % for most proteins
  - Inter-individual CV > intra-individual CV for biologically variable proteins
    (otherwise the assay can't distinguish individuals)

High intra-CV after normalisation suggests: poor LC reproducibility, batch
effects not fully corrected, or a fundamentally noisy protein.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE


def compute_sample_cv(df_linear: pd.DataFrame) -> pd.Series:
    """
    Per-protein CV across all samples (combined technical + biological).

    Parameters
    ----------
    df_linear : (samples × proteins) DataFrame in LINEAR (not log2) scale.
                Convert: df_linear = 2 ** df_log2

    Returns
    -------
    Series of CV (%) per protein.
    """
    return (df_linear.std(axis=0) / df_linear.mean(axis=0) * 100).dropna()


def compute_intra_cv(
    df_linear: pd.DataFrame,
    subject_ids: pd.Series,
) -> pd.Series:
    """
    Per-protein intra-individual CV: variability within subjects.

    For each subject with multiple observations, compute the within-subject CV
    per protein.  Return the median CV across subjects.

    Parameters
    ----------
    df_linear   : (samples × proteins) linear-scale DataFrame
    subject_ids : Series mapping sample → subject ID (same index as df)
    """
    subjects = subject_ids.reindex(df_linear.index).dropna()
    multi_subject = subjects[subjects.duplicated(keep=False)].index

    if len(multi_subject) < 2:
        return pd.Series(dtype=float, name="intra_cv")

    df_multi = df_linear.loc[multi_subject]
    sub_multi = subjects.loc[multi_subject]

    cv_per_subject = {}
    for sid, grp_idx in sub_multi.groupby(sub_multi).groups.items():
        grp = df_multi.loc[grp_idx]
        if len(grp) >= 2:
            cv_per_subject[sid] = grp.std(axis=0) / grp.mean(axis=0) * 100

    if not cv_per_subject:
        return pd.Series(dtype=float, name="intra_cv")

    return pd.DataFrame(cv_per_subject).median(axis=1).dropna()


def compute_inter_cv(
    df_linear: pd.DataFrame,
    subject_ids: pd.Series,
) -> pd.Series:
    """
    Per-protein inter-individual CV: variability between subjects.

    For each subject, compute the mean protein intensity across their samples.
    Then compute the CV of those per-subject means across subjects.

    Parameters
    ----------
    df_linear   : (samples × proteins) linear-scale DataFrame
    subject_ids : Series mapping sample → subject ID (same index as df)
    """
    sub = subject_ids.reindex(df_linear.index).dropna()
    df_aligned = df_linear.loc[sub.index]
    subject_means = df_aligned.groupby(sub).mean()   # (subjects × proteins)

    if len(subject_means) < 2:
        return pd.Series(dtype=float, name="inter_cv")

    return (subject_means.std(axis=0) / subject_means.mean(axis=0) * 100).dropna()


def plot_cv_violin(
    overall_cv: pd.Series,
    output_path: Path | None = None,
) -> None:
    """
    Violin plot of per-protein CV distribution across all samples.

    A good result: median CV < 30 %, most proteins below 50 %.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([overall_cv.dropna().values], positions=[1],
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE[0])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    median = overall_cv.median()
    ax.annotate(f"Median: {median:.1f}%", xy=(1, median),
                xytext=(1.25, median), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "grey"})
    ax.set_xticks([1])
    ax.set_xticklabels(["All proteins"])
    ax.set_ylabel("CV (%)")
    ax.set_title("Protein-level CV distribution")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cv_comparison(
    intra_cv: pd.Series,
    inter_cv: pd.Series,
    output_path: Path | None = None,
) -> None:
    """
    Violin plot comparing intra- vs. inter-individual CV.

    WHAT TO LOOK FOR:
      - Inter CV >> Intra CV: the protein varies more between individuals than
        within — good, the assay can detect individual differences.
      - Intra CV ≈ Inter CV: technical noise is as large as biological variation
        — the assay is not precise enough to resolve inter-individual differences.
    """
    common = intra_cv.index.intersection(inter_cv.index)
    if len(common) < 5:
        print("  [skip] Not enough proteins for CV comparison")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [intra_cv.loc[common].dropna().values,
            inter_cv.loc[common].dropna().values]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [PALETTE[0], PALETTE[1]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Intra-individual", "Inter-individual"])
    ax.set_ylabel("CV (%)")
    ax.set_title(f"Intra- vs. inter-individual CV\n(n = {len(common)} proteins)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
