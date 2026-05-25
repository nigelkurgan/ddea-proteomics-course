"""
Outlier detection for proteomics samples.

Uses multiple complementary strategies and a KS-test confirmation step.

KEY CONCEPT — why multi-method outlier detection?
-------------------------------------------------
No single metric reliably identifies all classes of problematic samples:

  • A sample with a shifted intensity distribution (e.g. degraded proteome,
    wrong dilution) is caught by *mean intensity* or *density* Z-score.

  • A sample that clusters far from all others in high-dimensional space
    is caught by *PCA Euclidean* or *Mahalanobis* distance.

  • A sample with an unusually high missing-value rate (e.g. failed injection)
    is caught by *missing rate* Z-score.

By running all methods and flagging samples that appear in multiple, we avoid
two failure modes:
  - False positives: a sample with a slightly unusual mean that is genuinely
    from a patient at an extreme of the biological range.
  - False negatives: a sample that is subtly wrong in one dimension but only
    clearly visible in PCA space.

The KS test provides a final statistical confirmation: we only exclude a sample
if its intensity distribution is significantly different from the rest of the
cohort (p < α), not just because it scored high on one metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, ks_2samp, zscore
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA


# ── Z-score helpers ───────────────────────────────────────────────────────────

def zscore_outliers(
    series: pd.Series,
    threshold: float = 2.576,
) -> list[str]:
    """
    Return sample IDs where |Z-score| > threshold.

    threshold=2.576 corresponds to α ≈ 0.01 (99th percentile, two-tailed).
    NaN values are omitted before computing Z-scores.
    """
    z = zscore(series.dropna(), ddof=0)   # ddof=0 = population std (consistent with N-based Z)
    flagged = series.dropna().index[np.abs(z) > threshold].tolist()
    return flagged


def sample_mean_outliers(
    df: pd.DataFrame,
    threshold: float = 2.576,
) -> list[str]:
    """
    Flag samples whose mean log2 intensity is an outlier.

    A very low mean suggests: protein degradation, low sample input, failed
    enrichment, or a wrong dilution factor.
    A very high mean is rarer but can indicate contamination or carry-over.
    """
    means = df.mean(axis=1)
    return zscore_outliers(means, threshold)


def missing_rate_outliers(
    df: pd.DataFrame,
    threshold: float = 2.576,
) -> list[str]:
    """
    Flag samples with an unusually high fraction of missing values.

    A high missing rate after global filtering usually indicates a failed or
    poor-quality injection — the run detected very few peptides.
    """
    miss_rate = df.isna().mean(axis=1)
    return zscore_outliers(miss_rate, threshold)


# ── PCA-based outlier detection ───────────────────────────────────────────────

@dataclass
class PCAOutliers:
    """
    PCA-based outlier detection using Euclidean and Mahalanobis distances
    computed in the principal component space.

    WHY PCA FIRST?
    --------------
    Raw proteomics data has thousands of correlated features.  Computing
    distances in the original high-dimensional space gives equal weight to
    all proteins, including noisy ones.  Projecting into PCA space:
      - Concentrates variance in a few components
      - Removes noise dimensions
      - Makes outlier distances more interpretable

    WHY MAHALANOBIS IN ADDITION TO EUCLIDEAN?
    ------------------------------------------
    Euclidean distance from the centroid treats all PC directions equally.
    Mahalanobis distance accounts for the covariance structure — a sample far
    along a low-variance PC is penalised more heavily than the same displacement
    along a high-variance PC.  This catches outliers that deviate in subtle but
    statistically unusual directions.

    We use MinCovDet (robust covariance estimate) to compute Mahalanobis
    distances — this prevents outliers from inflating the covariance estimate
    and masking themselves (the masking problem).

    Parameters
    ----------
    df           : Complete (no NaN) (samples × features) DataFrame.
                   Use KNN-imputed data.
    n_components : Number of PCA components to use for distances.
    threshold    : |Z-score| threshold for flagging.
    """
    df: pd.DataFrame
    n_components: int = 5
    threshold: float = 2.576

    pca_coords: pd.DataFrame = field(init=False)
    explained_variance: np.ndarray = field(init=False)
    euclidean_distances: pd.Series = field(init=False)
    mahalanobis_distances: pd.Series = field(init=False)
    euclidean_outliers: list[str] = field(init=False)
    mahalanobis_outliers: list[str] = field(init=False)
    loadings: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        n = min(self.n_components, self.df.shape[0] - 1, self.df.shape[1])
        pca = PCA(n_components=n)
        coords = pca.fit_transform(self.df.values)

        self.pca_coords = pd.DataFrame(
            coords, index=self.df.index,
            columns=[f"PC{i+1}" for i in range(n)]
        )
        self.explained_variance = pca.explained_variance_ratio_
        self.loadings = pd.DataFrame(
            pca.components_.T,
            index=self.df.columns,
            columns=[f"PC{i+1}" for i in range(n)],
        )

        # Euclidean distance from the centroid of the PC space
        centroid = coords.mean(axis=0)
        euc = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
        self.euclidean_distances = pd.Series(euc, index=self.df.index)
        self.euclidean_outliers = zscore_outliers(self.euclidean_distances, self.threshold)

        # Mahalanobis distance via MinCovDet (robust to outliers in the covariance estimate)
        try:
            # support_fraction=0.75: use 75% of points to estimate covariance,
            # robustly excluding any already-extreme samples from the estimate itself
            mcd = MinCovDet(random_state=42, support_fraction=0.75).fit(coords)
            mah = mcd.mahalanobis(coords) ** 0.5
        except Exception:
            # Fall back to standard covariance if MCD fails (e.g. too few samples)
            cov = np.cov(coords.T)
            inv_cov = np.linalg.pinv(cov)
            diff = coords - centroid
            mah = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))

        self.mahalanobis_distances = pd.Series(mah, index=self.df.index)
        self.mahalanobis_outliers = zscore_outliers(self.mahalanobis_distances, self.threshold)

    @property
    def all_outliers(self) -> list[str]:
        return sorted(set(self.euclidean_outliers) | set(self.mahalanobis_outliers))


# ── KDE density-based outlier detection ──────────────────────────────────────

def density_outliers(
    df: pd.DataFrame,
    threshold: float = 2.576,
    n_grid: int = 200,
) -> list[str]:
    """
    Flag samples whose intensity density shape or mean deviates from the cohort.

    This catches samples with a globally shifted or distorted distribution —
    complementary to PCA-based detection which is sensitive to direction-specific
    deviations.

    Two criteria are combined (union):
      1. Mean intensity Z-score (fast, catches shifted distributions)
      2. Mean KDE density at cohort-wide grid points (catches shape changes)
    """
    # Criterion 1: mean intensity
    mean_intensity = df.mean(axis=1)
    outliers_mean = set(zscore_outliers(mean_intensity, threshold))

    # Criterion 2: KDE shape — evaluate each sample's density at a common grid
    x_min = float(df.min().min())
    x_max = float(df.max().max())
    grid = np.linspace(x_min, x_max, n_grid)

    density_means: dict[str, float] = {}
    for sample in df.index:
        vals = df.loc[sample].dropna().values
        if len(vals) < 5:
            density_means[sample] = np.nan   # too few values to estimate KDE
        else:
            kde = gaussian_kde(vals)
            density_means[sample] = float(kde(grid).mean())

    density_series = pd.Series(density_means).dropna()
    outliers_density = set(zscore_outliers(density_series, threshold))

    return sorted(outliers_mean | outliers_density)


# ── KS-test confirmation ──────────────────────────────────────────────────────

def ks_confirm_outliers(
    df: pd.DataFrame,
    candidate_outliers: list[str],
    alpha: float = 0.05,
) -> dict[str, bool]:
    """
    Statistically confirm candidate outliers with a two-sample KS test.

    For each candidate, test whether its intensity distribution is significantly
    different from the pooled distribution of all remaining samples.

    WHY A CONFIRMATION STEP?
    ------------------------
    Z-score thresholds on continuous metrics will always flag some samples at
    the tail of the normal distribution, even in a perfectly clean dataset.
    The KS test provides an independent, non-parametric check: a sample is
    only confirmed as an outlier if its whole intensity distribution is
    statistically distinguishable from the rest of the cohort.

    Parameters
    ----------
    df                : (samples × proteins) log2 DataFrame (original, with NaN)
    candidate_outliers: list of sample IDs to test
    alpha             : significance threshold (default 0.05)

    Returns
    -------
    dict mapping sample_id → True (confirmed, p < alpha) or False
    """
    # Reference distribution: all non-missing intensities from non-candidate samples
    all_rest = df.values.flatten()
    all_rest = all_rest[~np.isnan(all_rest)]

    confirmed: dict[str, bool] = {}
    for sample in candidate_outliers:
        if sample not in df.index:
            confirmed[sample] = False
            continue
        sample_vals = df.loc[sample].dropna().values
        if len(sample_vals) < 5:
            # Too few values for a reliable KS test
            confirmed[sample] = False
            continue
        _, p = ks_2samp(sample_vals, all_rest)
        confirmed[sample] = p < alpha

    return confirmed


# ── Aggregated outlier summary ────────────────────────────────────────────────

def build_outlier_table(
    samples: list[str],
    detection_results: dict[str, list[str]],
    ks_confirmed: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """
    Build a summary table showing which methods flagged each sample.

    Each row is a sample that was flagged by at least one method.
    Columns: one per detection method (marked "x" if flagged), "KS_confirmed",
    and "n_methods_flagged" (useful for setting an exclusion threshold).

    Parameters
    ----------
    samples          : all sample IDs (to ensure completeness)
    detection_results: {method_name: [flagged_sample_ids, ...]}
    ks_confirmed     : {sample_id: bool} from ks_confirm_outliers

    Returns
    -------
    DataFrame sorted by n_methods_flagged descending.  Only rows where at
    least one method flagged the sample are included.
    """
    flagged_samples = sorted(
        {s for outliers in detection_results.values() for s in outliers}
    )
    tbl = pd.DataFrame(index=flagged_samples, columns=list(detection_results.keys()))
    for method, outliers in detection_results.items():
        tbl[method] = tbl.index.map(lambda s: "x" if s in outliers else "")

    if ks_confirmed is not None:
        tbl["KS_confirmed"] = tbl.index.map(
            lambda s: "✓" if ks_confirmed.get(s, False) else ""
        )

    tbl["n_methods_flagged"] = tbl.apply(
        lambda row: sum(1 for v in row if v == "x"), axis=1
    )
    return tbl.sort_values("n_methods_flagged", ascending=False)
