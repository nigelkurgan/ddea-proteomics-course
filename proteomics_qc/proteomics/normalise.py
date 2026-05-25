"""
Normalisation functions for proteomics intensity matrices.

All functions operate on (samples × features) DataFrames and return the
same shape.  Input is assumed to be log2-scale unless noted otherwise.

KEY CONCEPT — why normalise?
----------------------------
Raw intensities from DIA proteomics instruments vary between samples due to:
  - Differences in total protein injected (sample loading variation)
  - Instrument sensitivity drifting across a run
  - Pipetting and sample-preparation variability

Normalisation removes these *technical* sources of variation while preserving
the *biological* differences we care about.

Choosing a normalisation method:
  - Median scaling  : robust default for DIA data; handles missing values.
  - Quantile norm   : stronger alignment; use when distributions diverge a lot.
  - No normalisation: only if samples are already normalised upstream (rare).

All methods here operate in log2 space, where multiplicative fold-changes
become additive shifts — convenient for linear modelling downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def median_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full median scaling:
      1. Per-sample median centring   — align all sample medians to the global median
      2. Median-absolute-value (MAV) scaling — equalise spread across samples

    This is the recommended default for DIA proteomics because:
      - Step 1 corrects loading differences (systematic shifts)
      - Step 2 corrects for variation in dynamic range between samples
      - Both steps use the median, which is robust to a small number of
        differentially expressed proteins

    Parameters
    ----------
    df : (samples × features) log2 intensity DataFrame; may contain NaN.

    Returns
    -------
    Normalised DataFrame (same shape, index, columns).
    """
    mat = df.values.copy().astype(float)

    # Step 1: centre each sample so all medians equal the global median.
    # np.nanmedian ignores NaN — important for DIA data with missing values.
    col_medians = np.nanmedian(mat, axis=1)        # (n_samples,) — one median per sample
    global_median = np.nanmedian(col_medians)
    shift = (col_medians - global_median).reshape(-1, 1)
    mat_centered = mat - shift

    # Step 2: MAV scale so the geometric mean of per-sample spread equals 1
    mat_centered = _median_abs_value_scale(mat_centered)

    return pd.DataFrame(mat_centered, index=df.index, columns=df.columns)


def _median_abs_value_scale(mat: np.ndarray) -> np.ndarray:
    """
    Scale rows so the geometric mean of per-sample median-absolute-values = 1.

    MAV = median(|x_i|) for each sample i.
    We compute log(MAV) per sample, subtract the mean log(MAV), then
    exponentiate to get per-sample scale factors.  Dividing each sample by
    its scale factor equalises spread without affecting the median.
    """
    # log(MAV) per sample — nanmedian ignores missing values
    log_mav = np.log(np.nanmedian(np.abs(mat), axis=1))   # (n_samples,)
    # Per-sample scale factor: exp(log_MAV − mean_log_MAV)
    scale = np.exp(log_mav - np.mean(log_mav))             # (n_samples,)
    return mat / scale.reshape(-1, 1)


def median_centering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each sample so its median equals the global median across all samples.

    This is Step 1 of median_scaling without the MAV step — use when you only
    want to correct for loading differences, not spread.
    """
    col_medians = df.median(axis=1)           # per-sample median
    global_median = col_medians.median()
    shift = col_medians - global_median
    return df.sub(shift, axis=0)


def quantile_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force every sample to have the same intensity distribution (the mean
    distribution across all samples).

    Stronger than median scaling — use when sample distributions are clearly
    non-overlapping.  Not recommended when many proteins are expected to be
    differentially expressed, as it can suppress true biological differences.
    """
    mat = df.values.copy().astype(float)
    # Rank positions within each sample (row), average over ties
    ranks = np.argsort(np.argsort(mat, axis=1), axis=1)
    sorted_mat = np.sort(mat, axis=1)
    # Target distribution: mean intensity at each rank across all samples
    mean_per_rank = np.nanmean(sorted_mat, axis=0)
    normalised = mean_per_rank[ranks]
    return pd.DataFrame(normalised, index=df.index, columns=df.columns)


def log2_transform(
    df: pd.DataFrame,
    pseudocount: float = 0.0,
) -> pd.DataFrame:
    """
    Log2-transform raw intensities.  Values ≤ 0 become NaN.

    Spectronaut already outputs log2 values by default, so this step is only
    needed when working with raw (linear) intensity exports from MaxQuant or
    similar tools.

    Parameters
    ----------
    pseudocount : Small value added before the log transform to avoid log(0).
                  Use 1 for count data; leave at 0 for continuous intensities
                  where zeros represent absence rather than true zero counts.
    """
    shifted = df.add(pseudocount).where(df > 0)
    return np.log2(shifted)


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    K-nearest-neighbour imputation via sklearn.  Returns a fully-observed matrix.

    WHY KNN FOR PCA?
    ----------------
    PCA requires a complete matrix (no NaN).  KNN imputation estimates missing
    values from the `n_neighbors` most similar samples (based on shared observed
    proteins), which tends to produce more realistic imputed values than simple
    column-mean imputation.

    IMPORTANT: Use this imputed matrix ONLY for PCA, clustering, and
    visualisation — not as input to differential expression or linear modelling,
    which should retain the original missing values.

    Parameters
    ----------
    n_neighbors : Number of neighbouring samples used for imputation.
                  5 is a robust default for most proteomics cohorts.
    """
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)


def plate_median_correction(
    df: pd.DataFrame,
    plate_labels: pd.Series,
) -> pd.DataFrame:
    """
    Remove plate-level additive shifts while preserving within-plate variation.

    For each plate:
      corrected = (sample intensities) − (plate protein median) + (global protein median)

    Operating in log2 space makes this equivalent to dividing by the plate-wise
    fold-change relative to the global reference — a multiplicative correction
    on the linear scale.

    WHY PLATE CORRECTION?
    ---------------------
    In multi-plate DIA experiments, samples run on different plates (= different
    LC-MS runs on different days) show systematic intensity offsets.  These
    offsets are technical artefacts — not biological signal.  If uncorrected,
    plate effects dominate PCA and inflate variance estimates in statistical tests.

    Parameters
    ----------
    df           : (samples × proteins) log2 DataFrame (normalised, may have NaN)
    plate_labels : Series mapping sample → plate label (same index as df)

    Returns
    -------
    Corrected DataFrame (same shape, index, columns).
    """
    global_median = df.median(axis=0)   # protein-wise median across all samples
    corrected = df.copy()
    for plate in plate_labels.unique():
        idx = plate_labels[plate_labels == plate].index.intersection(df.index)
        if len(idx) == 0:
            continue
        plate_median = df.loc[idx].median(axis=0)   # protein-wise median within this plate
        # Shift plate samples by (global − plate) median: removes the plate offset
        corrected.loc[idx] = df.loc[idx].sub(plate_median, axis=1).add(global_median, axis=1)
    return corrected


def combat_correction(
    df: pd.DataFrame,
    plate_labels: pd.Series,
) -> pd.DataFrame:
    """
    Bayesian ComBat batch correction (inmoose.pycombat).

    ComBat is a parametric empirical Bayes method that models and removes
    additive and multiplicative batch effects simultaneously.  It is more
    powerful than plate-median correction when batch effects are complex,
    but requires:
      - A fully-observed (no NaN) matrix — impute with KNN first.
      - At least 2 samples per batch.

    Parameters
    ----------
    df           : (samples × proteins) FULLY OBSERVED log2 DataFrame
    plate_labels : Series mapping sample → plate (same index as df)

    Returns
    -------
    Corrected DataFrame (same shape, index, columns).
    """
    from inmoose.pycombat import pycombat_norm

    batch = plate_labels.reindex(df.index).values
    # pycombat_norm expects a (features × samples) matrix — transpose in/out
    data_T = df.T
    corrected_T = pycombat_norm(data_T.values, batch)
    return pd.DataFrame(corrected_T.T, index=df.index, columns=df.columns)


def impute_min_value(
    df: pd.DataFrame,
    fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Replace NaN with a fraction of the column minimum (feature-wise).

    This "left-shifted" imputation assumes missing values arise because a
    protein is below the detection limit (MNAR), not at random.  It places
    imputed values at the low end of the observed distribution, which is
    biologically plausible for absent / very low abundance proteins.

    Use KNN imputation instead when you believe missingness is partly random
    (e.g. instrument variability at moderate intensities).
    """
    col_min = df.min(axis=0) * fraction
    return df.fillna(col_min)
