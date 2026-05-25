"""
Protein / peptide completeness filtering.

All functions operate on a (samples × features) DataFrame — callers should
transpose before passing if their data is stored as (features × samples).

KEY CONCEPT — missing values in DIA proteomics
----------------------------------------------
Spectronaut and DIA-NN report intensities only when a peptide is confidently
detected above the precursor and fragment score thresholds.  An absent value
is therefore *not* random: low-abundance proteins are systematically missing
across many samples (missing-not-at-random, MNAR).

The standard practice is to:
  1. Set a minimum detection rate (e.g. 20 % of samples) and drop proteins
     below this threshold — these are too noisy to interpret biologically.
  2. Impute the remaining missing values when a complete matrix is needed
     (e.g. for PCA, clustering) using KNN or minimum-value imputation.
  3. Keep missing values for differential expression / linear modelling,
     since most statistical frameworks handle them natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_by_completeness(
    df: pd.DataFrame,
    threshold: float = 0.20,
    axis: str = "features",
) -> pd.DataFrame:
    """
    Remove features (proteins/peptides) or samples below a detection threshold.

    Parameters
    ----------
    df        : (samples × features) DataFrame of log2 intensities.
                Missing values should be NaN (not zero).
    threshold : Minimum fraction of non-missing values required [0, 1].
                0.20 means "keep if detected in at least 20 % of samples".
    axis      : "features" — filter proteins/peptides (column-wise, most common)
                "samples"  — filter low-coverage samples (row-wise)

    Returns
    -------
    Filtered DataFrame with the same orientation.

    Notes
    -----
    Choosing the threshold is a trade-off:
      - Too low  (e.g. 0.05): retains noisy, rarely detected proteins that
        add noise without statistical power.
      - Too high (e.g. 0.70): discards low-abundance proteins that may carry
        important biological signal in one condition.
    20 % is a common default for longitudinal or multi-condition studies where
    a protein may only be reliably detected at certain timepoints or in certain
    groups.
    """
    if axis == "features":
        # Count the fraction of samples where each protein is NOT missing
        detection_rate = df.notna().mean(axis=0)   # shape: (n_proteins,)
        keep = df.columns[detection_rate >= threshold]
        n_before, n_after = df.shape[1], len(keep)
        print(f"[filter] Proteins: {n_before} → {n_after} "
              f"(removed {n_before - n_after} below {threshold:.0%} detection rate)")
        return df[keep]

    elif axis == "samples":
        # Count the fraction of proteins detected per sample
        detection_rate = df.notna().mean(axis=1)   # shape: (n_samples,)
        keep = df.index[detection_rate >= threshold]
        n_before, n_after = len(df), len(keep)
        print(f"[filter] Samples: {n_before} → {n_after} "
              f"(removed {n_before - n_after} below {threshold:.0%} detection rate)")
        return df.loc[keep]

    else:
        raise ValueError("axis must be 'features' or 'samples'")


def filter_by_group(
    df: pd.DataFrame,
    groups: pd.Series,
    threshold: float = 0.50,
    min_groups: int = 1,
) -> pd.DataFrame:
    """
    Keep features detected above `threshold` in at least `min_groups` groups.

    This is preferable to a global threshold when groups differ strongly in
    protein expression (e.g. disease vs. control, or early vs. late timepoints).
    A protein absent in the control but present in 70 % of cases would pass
    with min_groups=1 even if the global rate is low.

    Parameters
    ----------
    df         : (samples × features) DataFrame
    groups     : Series with sample labels, index matching df.index
    threshold  : minimum detection rate within a group to count it as "passing"
    min_groups : minimum number of groups that must exceed the threshold

    Returns
    -------
    Filtered DataFrame.
    """
    unique_groups = groups.unique()
    # Build a (features × groups) boolean table: True where group passes threshold
    group_presence = pd.DataFrame(index=df.columns, dtype=bool)

    for grp in unique_groups:
        # Subset to samples in this group
        mask = groups == grp
        sub = df.loc[mask]
        group_presence[grp] = sub.notna().mean(axis=0) >= threshold

    # Keep proteins that pass in at least min_groups groups
    n_passing = group_presence.sum(axis=1)
    keep = n_passing[n_passing >= min_groups].index

    n_before, n_after = df.shape[1], len(keep)
    print(f"[filter_by_group] Proteins: {n_before} → {n_after} "
          f"(≥{threshold:.0%} in ≥{min_groups} groups)")
    return df[keep]


def filter_by_sample_list(
    df: pd.DataFrame,
    exclude: list[str],
) -> pd.DataFrame:
    """Remove specific samples by name (e.g. confirmed outliers or failed runs)."""
    to_remove = [s for s in exclude if s in df.index]
    if to_remove:
        print(f"[filter] Removing {len(to_remove)} specified samples: {to_remove}")
    return df.drop(index=to_remove, errors="ignore")


def completeness_by_group_table(
    df: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    """
    Return a (features × groups) table of per-group detection rates.

    Useful for inspection before choosing a threshold — you can see which
    proteins are group-specific and whether a global threshold would be fair.
    """
    result = {}
    for grp in sorted(groups.unique()):
        mask = groups == grp
        result[grp] = df.loc[mask].notna().mean(axis=0)
    return pd.DataFrame(result)


def completeness_at_thresholds(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.Series:
    """
    Return the number of proteins retained at each threshold.

    Use this to choose a threshold before filtering — plot the output to see
    the trade-off between stringency and protein count.
    """
    if thresholds is None:
        thresholds = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
    presence = df.notna().mean(axis=0)
    return pd.Series(
        {t: int((presence >= t).sum()) for t in thresholds},
        name="n_proteins_retained",
    )
