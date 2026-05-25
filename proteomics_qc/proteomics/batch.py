"""
Batch-effect characterisation for multi-plate proteomics experiments.

KEY CONCEPT — batch effects in proteomics
-----------------------------------------
A "batch" in proteomics is any grouping of samples processed together that
introduces a systematic technical difference from other batches.  Common
sources:
  • MS plates / run dates (most common in DIA experiments)
  • Sample preparation batches (freeze-thaw cycles, digestion date)
  • Operator, reagent lot, or instrument differences

Batch effects manifest as:
  1. Samples from the same plate clustering together in PCA
  2. Within-plate distances being smaller than between-plate distances
  3. Plate being a stronger predictor of PC1/PC2 than the biology of interest

This module provides two diagnostic tools:
  - plate_distance_stats : quantifies how much plates separate in PCA space
  - pc_factor_associations : tests each PC for association with batch and
    biological factors — helps decide whether correction is needed and how
    much variance is explained by each factor

Correction is handled in normalise.py (plate_median_correction / combat_correction).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
from sklearn.decomposition import PCA


def plate_distance_stats(
    df_imputed: pd.DataFrame,
    plate_labels: pd.Series,
    n_pca_components: int = 10,
) -> dict:
    """
    Quantify batch separation by comparing within- vs. between-plate distances
    in PCA space.

    HOW TO INTERPRET:
    -----------------
    - If within-plate distances << between-plate distances: strong batch effect.
      Samples are more similar to their plate-mates than to the rest of the
      cohort — biological signal may be confounded with plate.
    - KS p-value < 0.05: the two distance distributions are significantly different
      (i.e. plate structure is detectable).  Does NOT tell you whether it is
      biologically meaningful — always check the PCA plot coloured by plate.

    Parameters
    ----------
    df_imputed       : (samples × proteins) fully-imputed DataFrame
    plate_labels     : Series mapping sample → plate label
    n_pca_components : PCA dimensionality to use (10 captures most variance)

    Returns
    -------
    dict with keys:
      "within"   — list of pairwise Euclidean distances within each plate
      "between"  — list of pairwise distances between different plates
      "ks_pvalue"— two-sample KS test p-value (within vs. between)
    """
    from scipy.stats import ks_2samp

    n = min(n_pca_components, df_imputed.shape[0] - 1, df_imputed.shape[1])
    pca = PCA(n_components=n)
    coords = pca.fit_transform(df_imputed.values)

    plates = plate_labels.loc[df_imputed.index].values
    unique_plates = np.unique(plates[~pd.isnull(plates)])

    # Full pairwise distance matrix in PCA space
    dmat = squareform(pdist(coords, metric="euclidean"))

    within: list[float] = []
    between: list[float] = []

    # Within-plate distances: all pairs of samples on the same plate
    for p in unique_plates:
        idx = np.where(plates == p)[0]
        if len(idx) >= 2:
            within.extend(pdist(coords[idx], metric="euclidean"))

    # Between-plate distances: all pairs of samples from different plates
    for p1, p2 in combinations(unique_plates, 2):
        idx1 = np.where(plates == p1)[0]
        idx2 = np.where(plates == p2)[0]
        for i in idx1:
            for j in idx2:
                between.append(dmat[i, j])

    _, ks_p = ks_2samp(within, between) if within and between else (np.nan, np.nan)
    return {"within": within, "between": between, "ks_pvalue": ks_p}


def variance_explained_by_factor(
    df_imputed: pd.DataFrame,
    factor: pd.Series,
    n_pca_components: int = 10,
) -> pd.DataFrame:
    """
    Test how much of each PCA component is explained by a given factor.

    Uses one-way ANOVA across factor levels for each PC score.

    HOW TO INTERPRET:
    -----------------
    - High F-statistic (low p-value) for PC1 associated with "plate":
      strong batch effect in the dominant axis of variation.
    - High F-statistic for PC1 associated with "group" but not "plate":
      biological signal dominates — good sign.
    - After batch correction: plate association should drop; biological
      association should be preserved or increase.

    Parameters
    ----------
    df_imputed       : (samples × proteins) fully-imputed DataFrame
    factor           : categorical Series with same index (e.g. plate, group)
    n_pca_components : number of PCA components to test

    Returns
    -------
    DataFrame with columns: F_stat, p_value, var_explained per PC.
    """
    n = min(n_pca_components, df_imputed.shape[0] - 1, df_imputed.shape[1])
    pca = PCA(n_components=n)
    coords = pca.fit_transform(df_imputed.values)
    pca_df = pd.DataFrame(
        coords,
        index=df_imputed.index,
        columns=[f"PC{i+1}" for i in range(n)],
    )

    # Align factor to available samples
    fac = factor.loc[df_imputed.index].dropna()
    pca_df = pca_df.loc[fac.index]

    results = []
    for pc in pca_df.columns:
        groups = [pca_df.loc[fac == g, pc].values for g in fac.unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            results.append({"PC": pc, "F_stat": np.nan, "p_value": np.nan,
                            "var_explained": pca.explained_variance_ratio_[int(pc[2:])-1]})
            continue
        try:
            F, p = f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        results.append({
            "PC": pc,
            "F_stat": F,
            "p_value": p,
            "var_explained": pca.explained_variance_ratio_[int(pc[2:])-1],
        })

    return pd.DataFrame(results).set_index("PC")


def pc_factor_associations(
    df_imputed: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    n_pca_components: int = 10,
) -> pd.DataFrame:
    """
    Run variance_explained_by_factor for multiple factors simultaneously.

    Returns a (PCs × factors) DataFrame of -log10(p-value), suitable for
    plotting as a heatmap to show which factors drive each PC.

    A value of 2 corresponds to p = 0.01; values > 4 indicate very strong
    association (p < 0.0001).

    Parameters
    ----------
    df_imputed : (samples × proteins) fully-imputed DataFrame
    metadata   : DataFrame with sample-level covariates (plate, group, etc.)
    factors    : list of column names in metadata to test
    """
    all_results = {}
    for factor in factors:
        if factor not in metadata.columns:
            continue
        fac = metadata[factor]
        res = variance_explained_by_factor(df_imputed, fac, n_pca_components)
        # -log10(p): higher = stronger association; clip to avoid -inf from p=0
        all_results[factor] = -np.log10(res["p_value"].clip(lower=1e-300))

    return pd.DataFrame(all_results)
