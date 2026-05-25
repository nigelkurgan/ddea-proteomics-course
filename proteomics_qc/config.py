"""
Generic configuration for the proteomics QC pipeline.

Edit this file (or pass a Config instance with overrides) to adapt the
pipeline to your dataset.  All paths default to the bundled demo data so
the pipeline runs out-of-the-box after cloning the repository.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root — two levels up from this file
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class QCConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    root: Path = ROOT
    output_dir: Path = ROOT / "results" / "qc"

    # Input: (proteins × samples) parquet with PG_* metadata columns.
    # The pipeline transposes this to (samples × proteins) internally.
    # Replace with your own file path.
    protein_matrix: Path = ROOT / "data" / "demo" / "demo_proteomics.parquet"

    # Optional: CSV with one row per sample and columns for batch/group labels.
    # If None, the pipeline infers plate labels from the parquet column names.
    sample_metadata: Path | None = None

    # ── Filtering ─────────────────────────────────────────────────────────────
    # Proteins detected in fewer than this fraction of samples are dropped.
    # 0.20 = keep proteins present in at least 20 % of samples.
    # Lower → more proteins retained but noisier; higher → fewer but cleaner.
    completeness_threshold: float = 0.20

    # ── Normalisation ─────────────────────────────────────────────────────────
    # Median scaling is the default (shift + MAV, works well for DIA data).
    # Change to "quantile" for older label-free datasets if distributions differ.
    normalisation_method: str = "median_scaling"

    # ── Outlier detection ─────────────────────────────────────────────────────
    # |Z-score| threshold for flagging. 2.576 corresponds to the 99th percentile
    # of a normal distribution (α ≈ 0.01, two-tailed).
    zscore_threshold: float = 2.576

    # Number of PCA components used to compute Euclidean / Mahalanobis distances.
    # 5 captures most meaningful variance in typical proteomics datasets (200–5000 proteins).
    pca_n_components: int = 5

    # KS test significance threshold for outlier confirmation.
    ks_alpha: float = 0.05

    # Minimum number of detection methods that must flag a sample before it is
    # excluded from the analysis-ready output.
    outlier_n_methods_threshold: int = 4

    # ── Batch correction ──────────────────────────────────────────────────────
    # "auto"         — try ComBat, fall back to plate-median, then none
    # "combat"       — Bayesian ComBat (requires inmoose package)
    # "plate-median" — simple ratio-based correction (works with missing values)
    # "none"         — skip batch correction
    batch_correction_method: str = "auto"

    # Column in the sample metadata that identifies the MS run/plate.
    batch_column: str = "plate"

    # ── Report ────────────────────────────────────────────────────────────────
    report_title: str = "Proteomics QC Report"
