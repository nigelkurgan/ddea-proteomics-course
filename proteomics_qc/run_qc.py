"""
Proteomics QC Pipeline — main entry point.

Usage
-----
From the repository root:
    python -m proteomics_qc.run_qc                        # uses demo data
    python -m proteomics_qc.run_qc --protein-matrix data/my_data.parquet
    python -m proteomics_qc.run_qc --completeness 0.30 --batch-correction combat

Pipeline stages
---------------
1.  Load data        — read protein matrix parquet → (samples × proteins) DataFrame
2.  Load metadata    — plate/batch labels per sample
3.  Filtering        — remove proteins below completeness threshold
4.  Normalisation    — median scaling (or quantile)
5.  Outlier detection — Z-score, PCA Euclidean, PCA Mahalanobis, KDE density
6.  Batch-effect QC  — plate distances, PC × factor associations
7.  CV analysis      — overall, intra- and inter-individual
8.  Plots            — all visualisations saved to results/qc/plots/
9.  Analysis-ready output — batch-corrected parquet + outlier decisions JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proteomics_qc.config import QCConfig
from proteomics_qc.proteomics.filters import filter_by_completeness, completeness_at_thresholds
from proteomics_qc.proteomics.normalise import (
    median_scaling, impute_knn, plate_median_correction, combat_correction,
)
from proteomics_qc.proteomics.outliers import (
    sample_mean_outliers, missing_rate_outliers, PCAOutliers,
    density_outliers, ks_confirm_outliers, build_outlier_table,
)
from proteomics_qc.proteomics.batch import plate_distance_stats, pc_factor_associations
from proteomics_qc.plots.style import apply_style
from proteomics_qc.plots.distribution import (
    plot_sample_boxplot, plot_density_per_sample, plot_protein_rank_abundance,
)
from proteomics_qc.plots.missing import (
    plot_missing_heatmap, plot_missing_by_threshold, plot_missing_per_group,
)
from proteomics_qc.plots.batch_effects import (
    plot_pca, plot_plate_distances, plot_pc_factor_heatmap,
    plot_correlation_heatmap,
)
from proteomics_qc.plots.cv import (
    compute_sample_cv, compute_intra_cv, compute_inter_cv,
    plot_cv_violin, plot_cv_comparison,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Print a formatted section header to the console."""
    print(f"\n{'='*65}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*65}", flush=True)


def load_protein_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a (proteins × samples) parquet file into a (samples × proteins) DataFrame.

    The parquet is expected to follow the Spectronaut / DIA-NN export convention:
      - Columns starting with "PG_" are protein metadata (accession, gene name, etc.)
      - All other columns are sample intensities (one column per sample)

    Returns
    -------
    quant_df : (samples × proteins) log2 intensity DataFrame, index = sample IDs
    meta_df  : protein metadata DataFrame, index = PG_ProteinAccessions
    """
    raw = pd.read_parquet(path)
    meta_cols = [c for c in raw.columns if c.startswith("PG_")]
    sample_cols = [c for c in raw.columns if not c.startswith("PG_")]

    if "PG_ProteinAccessions" not in raw.columns:
        # Fallback: use first column as protein ID if no PG_ columns
        raw = raw.rename(columns={raw.columns[0]: "PG_ProteinAccessions"})
        meta_cols = ["PG_ProteinAccessions"]
        sample_cols = [c for c in raw.columns if c != "PG_ProteinAccessions"]

    meta_df = raw.set_index("PG_ProteinAccessions")[meta_cols[1:]]
    quant_df = raw.set_index("PG_ProteinAccessions")[sample_cols].T   # transpose to samples × proteins
    return quant_df, meta_df


def load_metadata(
    metadata_path: Path | None,
    quant_df: pd.DataFrame,
    batch_column: str = "plate",
) -> pd.DataFrame:
    """
    Load or infer sample metadata.

    If a metadata CSV is provided, it is used as-is (must have 'sample' index
    or a column matching sample IDs).  Otherwise, plate labels are inferred
    from the sample names: the portion after the last underscore, or "plate_1"
    if none.

    Returns
    -------
    DataFrame with index = sample ID and at least a `batch_column` column.
    """
    if metadata_path is not None and metadata_path.exists():
        meta = pd.read_csv(metadata_path, index_col=0)
        # Align index to quantification matrix
        meta = meta.reindex(quant_df.index)
        print(f"  Loaded metadata from {metadata_path}  ({len(meta)} samples)")
        return meta

    # Infer plate from sample name: e.g. "SampleX_P01" → plate "P01"
    plates = {}
    for s in quant_df.index:
        parts = str(s).rsplit("_", 1)
        plates[s] = parts[-1] if len(parts) == 2 and len(parts[-1]) <= 6 else "plate_1"
    meta = pd.DataFrame({"plate": plates})
    print(f"  Metadata inferred from sample names.  "
          f"Plates found: {sorted(meta['plate'].unique())}")
    return meta


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: QCConfig) -> None:
    apply_style()

    plots_dir = cfg.output_dir / "plots"
    tables_dir = cfg.output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load protein matrix ─────────────────────────────────────────
    _section("Stage 1: Load Protein Matrix")
    quant_df, prot_meta = load_protein_matrix(cfg.protein_matrix)
    print(f"  Samples × Proteins: {quant_df.shape}")
    print(f"  Intensity range (log2): {quant_df.min().min():.1f} – {quant_df.max().max():.1f}")

    # ── Stage 2: Load sample metadata ────────────────────────────────────────
    _section("Stage 2: Load Sample Metadata")
    sample_meta = load_metadata(cfg.sample_metadata, quant_df, cfg.batch_column)
    sample_meta.to_csv(tables_dir / "sample_metadata.csv")

    # ── Stage 3: Filtering ────────────────────────────────────────────────────
    _section("Stage 3: Protein Completeness Filtering")
    # Show how many proteins are retained at each threshold — useful for choosing
    plot_missing_by_threshold(quant_df, output_path=plots_dir / "missing_threshold_barplot.png")
    quant_filt = filter_by_completeness(quant_df, cfg.completeness_threshold, "features")
    print(f"  After filtering: {quant_filt.shape}")

    # ── Stage 4: Normalisation ─────────────────────────────────────────────────
    _section("Stage 4: Normalisation")
    if cfg.normalisation_method == "median_scaling":
        quant_norm = median_scaling(quant_filt)
        print("  Median scaling applied (per-sample median centring + MAV scaling).")
    else:
        from proteomics_qc.proteomics.normalise import quantile_normalisation
        quant_norm = quantile_normalisation(quant_filt)
        print("  Quantile normalisation applied.")

    # KNN-imputed version for PCA / clustering (complete matrix required)
    try:
        quant_imputed = impute_knn(quant_norm, n_neighbors=5)
        print("  KNN imputation applied (for PCA/clustering only — not for stats).")
    except Exception as e:
        print(f"  KNN failed ({e}), using column-mean imputation.")
        quant_imputed = quant_norm.fillna(quant_norm.mean(axis=0))

    # ── Stage 5: Outlier Detection ────────────────────────────────────────────
    _section("Stage 5: Outlier Detection")
    outlier_mean    = sample_mean_outliers(quant_norm, cfg.zscore_threshold)
    outlier_missing = missing_rate_outliers(quant_filt, cfg.zscore_threshold)
    outlier_density = density_outliers(quant_norm, cfg.zscore_threshold)

    pca_out = PCAOutliers(quant_imputed, n_components=cfg.pca_n_components,
                          threshold=cfg.zscore_threshold)

    detection_results = {
        "Mean intensity":    outlier_mean,
        "Missing rate":      outlier_missing,
        "Density":           outlier_density,
        "PCA-Euclidean":     pca_out.euclidean_outliers,
        "PCA-Mahalanobis":   pca_out.mahalanobis_outliers,
    }

    all_candidates = sorted({s for lst in detection_results.values() for s in lst})
    print(f"  Candidate outliers (≥1 method): {len(all_candidates)}")

    ks_confirmed = ks_confirm_outliers(quant_norm, all_candidates, cfg.ks_alpha)
    outlier_table = build_outlier_table(
        quant_norm.index.tolist(), detection_results, ks_confirmed
    )
    outlier_table.to_csv(tables_dir / "outlier_summary.csv")

    n_confirmed = outlier_table["KS_confirmed"].eq("✓").sum()
    print(f"  KS-confirmed outliers: {n_confirmed}")
    if not outlier_table.empty:
        print(outlier_table[["n_methods_flagged", "KS_confirmed"]].head(10).to_string())

    confirmed_outliers = [s for s, v in ks_confirmed.items() if v]

    # ── Stage 6: Batch-effect QC ──────────────────────────────────────────────
    _section("Stage 6: Batch-Effect QC")
    plate_labels = sample_meta[cfg.batch_column]

    dist_stats = plate_distance_stats(quant_imputed, plate_labels, n_pca_components=10)
    print(f"  KS p-value (within vs. between plate): {dist_stats['ks_pvalue']:.4g}")
    if dist_stats["ks_pvalue"] < 0.05:
        print("  → Significant plate separation detected — batch correction recommended.")
    else:
        print("  → No significant plate separation — batch correction optional.")

    avail_factors = [f for f in [cfg.batch_column, "group", "condition"]
                     if f in sample_meta.columns]
    pc_assoc = pc_factor_associations(quant_imputed, sample_meta, avail_factors, 10)
    pc_assoc.to_csv(tables_dir / "pc_factor_associations.csv")

    # ── Stage 7: CV Analysis ──────────────────────────────────────────────────
    _section("Stage 7: CV Analysis")
    quant_linear = 2 ** quant_norm.fillna(np.nan)   # convert log2 → linear for CV
    overall_cv = compute_sample_cv(quant_linear)

    subject_col = "subject_id" if "subject_id" in sample_meta.columns else None
    if subject_col:
        subject_ids = sample_meta[subject_col].dropna().astype(str)
        common_idx = quant_linear.index.intersection(subject_ids.index)
        intra_cv = compute_intra_cv(quant_linear.loc[common_idx],
                                    subject_ids.reindex(common_idx))
        inter_cv = compute_inter_cv(quant_linear.loc[common_idx],
                                    subject_ids.reindex(common_idx))
    else:
        # Use plate as surrogate for repeated measures when no subject_id column
        intra_cv = compute_intra_cv(quant_linear, plate_labels.reindex(quant_linear.index))
        inter_cv = compute_inter_cv(quant_linear, plate_labels.reindex(quant_linear.index))

    pd.DataFrame({"overall_cv": overall_cv, "intra_cv": intra_cv,
                  "inter_cv": inter_cv}).to_csv(tables_dir / "cv_summary.csv")
    print(f"  Median overall CV : {overall_cv.median():.1f}%")
    print(f"  Median intra CV   : {intra_cv.median():.1f}%" if not intra_cv.empty else "  Intra CV: N/A")
    print(f"  Median inter CV   : {inter_cv.median():.1f}%" if not inter_cv.empty else "  Inter CV: N/A")

    # ── Stage 8: Generate Plots ───────────────────────────────────────────────
    _section("Stage 8: Plots")

    # Build outlier-removed clean matrix for downstream PCAs
    confirmed_set = set(confirmed_outliers)
    quant_imputed_clean = quant_imputed.loc[
        [s for s in quant_imputed.index if s not in confirmed_set]
    ]
    meta_clean = sample_meta.reindex(quant_imputed_clean.index)
    print(f"  Clean matrix for PCA: {len(quant_imputed_clean)} samples "
          f"({len(confirmed_outliers)} outliers removed)")

    # Missing values
    plot_missing_heatmap(quant_filt, sample_meta, sort_by=cfg.batch_column,
                         output_path=plots_dir / "missing_heatmap.png")
    plot_missing_per_group(quant_filt, plate_labels.reindex(quant_filt.index),
                           output_path=plots_dir / "missing_per_group.png")

    # Distributions
    plot_sample_boxplot(quant_norm, sample_meta, color_by=cfg.batch_column,
                        outlier_samples=confirmed_outliers,
                        output_path=plots_dir / "sample_distribution.png")
    plot_density_per_sample(quant_norm, outlier_samples=confirmed_outliers,
                            output_path=plots_dir / "density_per_sample.png")
    plot_protein_rank_abundance(quant_norm, output_path=plots_dir / "protein_rank_abundance.png")

    # PCA — coloured by plate (before correction)
    plot_pca(quant_imputed, sample_meta, color_by=cfg.batch_column,
             output_path=plots_dir / "pca_plate.png",
             title=f"PCA — {cfg.batch_column} (pre-correction, outliers marked ×)",
             outlier_samples=confirmed_outliers)

    # PCA — outliers removed; coloured by any "group" column if present
    group_col = "group" if "group" in sample_meta.columns else cfg.batch_column
    plot_pca(quant_imputed_clean, meta_clean, color_by=group_col,
             output_path=plots_dir / "pca_group.png",
             title=f"PCA — {group_col} (outliers removed)")

    # Batch-effect summaries
    plot_plate_distances(dist_stats["within"], dist_stats["between"],
                         dist_stats["ks_pvalue"],
                         output_path=plots_dir / "plate_distances.png")
    if not pc_assoc.empty:
        plot_pc_factor_heatmap(pc_assoc, output_path=plots_dir / "pc_factor_associations.png")

    plot_correlation_heatmap(quant_imputed_clean, meta_clean, color_by=cfg.batch_column,
                             output_path=plots_dir / "correlation_heatmap.png")

    # CV
    plot_cv_violin(overall_cv, output_path=plots_dir / "cv_violin.png")
    if not intra_cv.empty and not inter_cv.empty:
        plot_cv_comparison(intra_cv, inter_cv, output_path=plots_dir / "cv_comparison.png")

    # After correction PCA
    try:
        plate_lbl_aligned = sample_meta[cfg.batch_column].reindex(quant_imputed.index)
        if cfg.batch_correction_method in ("auto", "combat"):
            quant_corrected = combat_correction(quant_imputed, plate_lbl_aligned)
            method_used = "ComBat"
        elif cfg.batch_correction_method == "plate-median":
            quant_corrected = plate_median_correction(quant_imputed, plate_lbl_aligned)
            method_used = "plate-median"
        else:
            quant_corrected = quant_imputed
            method_used = "none"
        if method_used != "none":
            clean_corr = quant_corrected.loc[
                [s for s in quant_corrected.index if s not in confirmed_set]
            ]
            plot_pca(clean_corr, sample_meta.reindex(clean_corr.index),
                     color_by=cfg.batch_column,
                     output_path=plots_dir / "pca_plate_corrected.png",
                     title=f"PCA — {cfg.batch_column} ({method_used}-corrected)")
    except Exception as e:
        print(f"  [warn] Correction PCA failed: {e}")
        quant_corrected = quant_imputed

    print(f"  All plots saved to {plots_dir}")

    # ── Stage 9: Analysis-ready output ───────────────────────────────────────
    _section("Stage 9: Analysis-Ready Output")

    if not outlier_table.empty:
        remove_samples = outlier_table[
            outlier_table["n_methods_flagged"] >= cfg.outlier_n_methods_threshold
        ].index.tolist()
    else:
        remove_samples = []

    print(f"  Removing {len(remove_samples)} outliers "
          f"(n_methods_flagged >= {cfg.outlier_n_methods_threshold})")

    quant_clean = quant_norm.drop(index=remove_samples, errors="ignore")
    plate_lbl_clean = sample_meta[cfg.batch_column].reindex(quant_clean.index)

    # Apply batch correction to the clean, non-imputed matrix
    bc_applied = "none"
    try:
        if cfg.batch_correction_method in ("auto", "combat"):
            q_imp_clean = impute_knn(quant_clean, n_neighbors=5)
            quant_final = combat_correction(q_imp_clean, plate_lbl_clean)
            # Restore original missing values (don't impute in final output)
            quant_final[quant_clean.isna()] = float("nan")
            bc_applied = "combat"
            print("  ComBat batch correction applied to analysis-ready output.")
        elif cfg.batch_correction_method == "plate-median":
            quant_final = plate_median_correction(quant_clean, plate_lbl_clean)
            bc_applied = "plate-median"
            print("  Plate-median batch correction applied.")
        else:
            quant_final = quant_clean
            print("  No batch correction applied.")
    except Exception as e:
        print(f"  [warn] Batch correction failed ({e}), using uncorrected output.")
        quant_final = quant_clean

    print(f"  Final matrix: {quant_final.shape[0]} samples × {quant_final.shape[1]} proteins")

    # Save as proteins × samples parquet (matches input format)
    quant_out = quant_final.T
    out_df = prot_meta.join(quant_out, how="left").reset_index()
    analysis_ready_path = cfg.output_dir / "proteomics_analysis_ready.parquet"
    out_df.to_parquet(analysis_ready_path, index=False)
    print(f"  Saved: {analysis_ready_path}")

    qc_decisions = {
        "normalization": cfg.normalisation_method,
        "batch_correction": bc_applied,
        "completeness_threshold": cfg.completeness_threshold,
        "outlier_n_methods_threshold": cfg.outlier_n_methods_threshold,
        "n_outliers_removed": len(remove_samples),
        "outliers_removed": remove_samples,
        "n_samples": int(quant_final.shape[0]),
        "n_proteins": int(quant_final.shape[1]),
    }
    with open(tables_dir / "qc_decisions.json", "w") as f:
        json.dump(qc_decisions, f, indent=2)

    print(f"\n✓ QC pipeline complete.  Results in: {cfg.output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Proteomics QC Pipeline")
    parser.add_argument("--protein-matrix", type=str, default=None,
                        help="Path to (proteins × samples) parquet file")
    parser.add_argument("--sample-metadata", type=str, default=None,
                        help="Optional path to sample metadata CSV")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: results/qc/)")
    parser.add_argument("--completeness", type=float, default=None,
                        help="Protein completeness threshold [0–1] (default: 0.20)")
    parser.add_argument("--batch-correction",
                        choices=["auto", "combat", "plate-median", "none"],
                        default=None,
                        help="Batch correction method (default: auto)")
    parser.add_argument("--outlier-threshold", type=int, default=None,
                        help="Min n_methods_flagged to exclude a sample (default: 4)")
    args = parser.parse_args()

    cfg = QCConfig()
    if args.protein_matrix:
        cfg.protein_matrix = Path(args.protein_matrix)
    if args.sample_metadata:
        cfg.sample_metadata = Path(args.sample_metadata)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.completeness is not None:
        cfg.completeness_threshold = args.completeness
    if args.batch_correction is not None:
        cfg.batch_correction_method = args.batch_correction
    if args.outlier_threshold is not None:
        cfg.outlier_n_methods_threshold = args.outlier_threshold

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
