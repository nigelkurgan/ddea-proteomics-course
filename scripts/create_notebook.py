"""
Build the proteomics QC tutorial notebook from scratch.
Run from the repo root: python scripts/create_notebook.py
"""
import json
from pathlib import Path


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── 0. Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# Proteomics QC Pipeline — Hands-On Tutorial

**Course: Proteomics Data Analysis**
**Environment: Google Colab (or local Jupyter)**

---

## Learning Objectives

By the end of this notebook you will be able to:

1. Load and inspect a DIA proteomics intensity matrix from a longitudinal study
2. Apply protein completeness filtering and explain the trade-offs
3. Normalise intensities using median scaling and understand why it works
4. Detect and confirm technical outlier samples using multiple independent methods
5. Diagnose and correct plate batch effects — and see the improvement in PCA
6. Assess technical reproducibility using pooled QC controls (intra- vs inter-plate CV)
7. Distinguish within-subject from between-subject biological variation
8. Interpret PCA, t-SNE, and factor-association heatmaps
9. Generate a standardised QC report ready for publication

---

## Dataset: Longitudinal DIA Plasma Proteomics

We use a **synthetic longitudinal dataset** modelled on real-world multi-plate Spectronaut exports.

| Property | Details |
|----------|---------|
| Design | Latin square — no plate × time or plate × group confounding |
| Subjects | 160 (80 Case / 80 Control; sex balanced within each group) |
| Time points | 3 per subject (T00 = baseline, T01, T02) → **480 biological samples** |
| Plates | 4 plates × 120 bio samples; each plate has all time points and groups |
| QC controls | 12 pooled replicates (3 per plate) — identical biology, technical noise only |
| Proteins | 2 000 (log2 LFQ intensities) |
| Missing values | ~37 % biological (MNAR), ~3 % QC |
| Batch effects | Strong protein-specific plate effects visible in PCA before correction |
| Hidden outliers | Planted in the data — you will discover them! |\
"""))

# ── 1. Setup header ───────────────────────────────────────────────────────────
cells.append(md("""\
---
## 0. Environment Setup

Run the cell below once to install dependencies and clone the pipeline repository.
Skip if running locally with the environment already set up.\
"""))

# ── 2. Install + clone ────────────────────────────────────────────────────────
cells.append(code("""\
# Install all required packages — takes ~60–90 seconds on first Colab run
import subprocess, sys

packages = [
    "pandas>=2.0", "numpy>=1.24", "matplotlib>=3.7",
    "scipy>=1.10", "scikit-learn>=1.3", "pyarrow>=12.0",
    "inmoose>=0.3",   # ComBat batch correction (Bayesian empirical Bayes)
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✓ Dependencies installed")

# Clone the teaching repository (contains the proteomics_qc package and demo data)
import os
if not os.path.exists("ddea-proteomics-course"):
    subprocess.check_call(["git", "clone", "-q",
        "https://github.com/nigelkurgan/ddea-proteomics-course.git"])
os.chdir("ddea-proteomics-course")
sys.path.insert(0, ".")
print("✓ Repository ready")\
"""))

# ── 3. Imports ────────────────────────────────────────────────────────────────
cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.dpi'] = 110

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as _PCA   # renamed to avoid conflict with plot functions

# ── Pipeline modules ──────────────────────────────────────────────────────────
from proteomics_qc.proteomics.filters import filter_by_completeness, completeness_at_thresholds
from proteomics_qc.proteomics.normalise import (
    median_scaling, impute_knn, plate_median_correction, combat_correction,
)
from proteomics_qc.proteomics.outliers import (
    sample_mean_outliers, missing_rate_outliers, PCAOutliers,
    density_outliers, ks_confirm_outliers, build_outlier_table,
)
from proteomics_qc.proteomics.batch import plate_distance_stats, pc_factor_associations
from proteomics_qc.plots.style import apply_style, PALETTE
from proteomics_qc.plots.distribution import (
    plot_sample_boxplot, plot_density_per_sample, plot_protein_rank_abundance,
)
from proteomics_qc.plots.missing import (
    plot_missing_heatmap, plot_missing_by_threshold, plot_missing_per_group,
)
from proteomics_qc.plots.batch_effects import (
    plot_pca, plot_plate_distances, plot_pc_factor_heatmap, plot_correlation_heatmap,
)
from proteomics_qc.plots.cv import (
    compute_sample_cv, plot_cv_violin,
)

apply_style()
print("✓ All imports successful")\
"""))

# ── 4. Stage 1 markdown ───────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 1: Load the Data

### What format does proteomics data come in?

Spectronaut and DIA-NN export a **protein-level intensity matrix**:

```
          Sample_001  Sample_002  ...  Sample_480
P00001        23.4        22.8   ...      24.1
P00002        19.1         NaN   ...      20.3
  ...
```

- Rows = proteins (UniProt accession)
- Columns = samples (biological + QC controls)
- Values = **log2 MS1 intensity** (LFQ or MaxLFQ)
- **NaN** = protein not detected — this is NOT a zero; it is a missing observation

We store the matrix in [Apache Parquet](https://parquet.apache.org/) format for fast, compact I/O.\
"""))

# ── 5. Load data ──────────────────────────────────────────────────────────────
cells.append(code("""\
demo_parquet = Path("data/demo/demo_proteomics.parquet")
demo_meta_csv = Path("data/demo/demo_metadata.csv")

# Regenerate synthetic data if the files do not exist
if not demo_parquet.exists():
    import subprocess
    subprocess.check_call(["python", "scripts/generate_demo_data.py"])

# ── Load protein matrix ───────────────────────────────────────────────────────
raw = pd.read_parquet(demo_parquet)

# Columns starting with "PG_" are protein metadata; all others are sample intensities
meta_cols   = [c for c in raw.columns if c.startswith("PG_")]
sample_cols = [c for c in raw.columns if not c.startswith("PG_")]

# Store protein-level metadata (gene name, protein name) for use at the end
prot_meta = raw.set_index("PG_ProteinAccessions")[meta_cols[1:]]

# Transpose so rows = samples and columns = proteins — standard orientation for statistics
quant_df = raw.set_index("PG_ProteinAccessions")[sample_cols].T

print(f"Matrix shape    : {quant_df.shape[0]} samples × {quant_df.shape[1]} proteins")
print(f"Intensity range : {quant_df.min().min():.1f} – {quant_df.max().max():.1f}  log2")
print(f"Overall missing : {quant_df.isna().mean().mean():.1%}")
quant_df.head(3)\
"""))

# ── 6. Load metadata + split bio / QC ─────────────────────────────────────────
cells.append(code("""\
# ── Load sample metadata ─────────────────────────────────────────────────────
sample_meta = pd.read_csv(demo_meta_csv, index_col=0)
sample_meta = sample_meta.reindex(quant_df.index)   # align rows to matrix order

# Split into biological samples and pooled QC controls
# QC samples are identical biology measured repeatedly → technical noise only
bio_meta = sample_meta[~sample_meta["is_qc"]]
qc_meta  = sample_meta[ sample_meta["is_qc"]]

print(f"Biological samples : {len(bio_meta)}")
print(f"QC pooled samples  : {len(qc_meta)}  ({qc_meta['plate'].value_counts().sort_index().to_dict()})")
print()
# Verify the Latin square design: each plate × timepoint cell has equal n
design = bio_meta.groupby(["plate", "timepoint"]).size().unstack()
print("Study design (samples per plate × time point):")
print(design.to_string())
print()
print("Group × Sex balance (biological samples):")
print(bio_meta.groupby(["group", "sex"]).size().to_string())\
"""))

# ── 7. Split quant matrices ───────────────────────────────────────────────────
cells.append(code("""\
# ── Separate biological and QC intensity matrices ────────────────────────────
# All QC-pipeline steps (filtering, normalisation, outlier detection) use bio only.
# QC samples (3 identical pooled replicates per plate) are analysed separately
# in Stage 7 to quantify technical (intra-plate) vs. batch (inter-plate) variation.

quant_bio = quant_df.loc[bio_meta.index]   # 480 × n_proteins
quant_qc  = quant_df.loc[qc_meta.index]   # 12  × n_proteins

print(f"Biological matrix : {quant_bio.shape[0]} samples × {quant_bio.shape[1]} proteins")
print(f"QC matrix         : {quant_qc.shape[0]}  samples × {quant_qc.shape[1]} proteins")
print(f"QC missing rate   : {quant_qc.isna().mean().mean():.1%}  "
      f"(biological: {quant_bio.isna().mean().mean():.1%})")
print()
# QC samples: show the 3 replicates per plate
print("QC pooled controls:")
print(qc_meta[["plate", "group"]])\
"""))

# ── 8. Stage 2 markdown ───────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 2: Missing Value Exploration

### Why do proteomics data have missing values?

Missing values in DIA proteomics are **Missing Not At Random (MNAR)**: low-abundance
proteins fall below the instrument's detection limit and are not confidently identified.

**Key questions:**
1. Are missing values structured by plate? (= technical problem in one run)
2. Are they structured by sample group? (= biological, may be informative)
3. What completeness threshold balances protein count vs. data quality?

The heatmap below shows proteins (x-axis) × biological samples (y-axis). Black = missing.\
"""))

# ── 9. Missing heatmap ────────────────────────────────────────────────────────
cells.append(code("""\
# Sort samples by plate so plate-specific missingness structure is visible
# A block of black rows within one plate = that plate had more detection failures
plot_missing_heatmap(quant_bio, bio_meta, sort_by="plate")\
"""))

# ── 10. Missing threshold ─────────────────────────────────────────────────────
cells.append(code("""\
# Protein retention at different completeness thresholds
# The knee of this curve is a good guide for threshold selection
plot_missing_by_threshold(quant_bio)

counts = completeness_at_thresholds(quant_bio)
print("Proteins retained at each threshold:")
print(counts.to_string())\
"""))

# ── 11. Missing per group ─────────────────────────────────────────────────────
cells.append(code("""\
# Is missingness uniform across plates, or does one plate have structural problems?
# Unequal missingness by plate suggests a technical issue on that acquisition run.
plot_missing_per_group(quant_bio, bio_meta["plate"])\
"""))

# ── 12. Stage 3 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 3: Protein Completeness Filtering

We remove proteins detected in fewer than `threshold` fraction of **biological** samples.
The default is **20 %** — a widely-used cutoff that keeps most proteins while discarding
highly-sparse, noisy measurements.

> **Exercise 3.1**: Change `COMPLETENESS_THRESHOLD` to 0.10 and 0.50.
> How many proteins remain at each threshold?
> At 50 %, would you expect to lose mainly low-abundance or high-abundance proteins? Why?\
"""))

# ── 13. Filter ────────────────────────────────────────────────────────────────
cells.append(code("""\
COMPLETENESS_THRESHOLD = 0.20   # ← try changing this to 0.10 or 0.50

# Filter based on biological sample completeness
quant_bio_filt = filter_by_completeness(quant_bio, threshold=COMPLETENESS_THRESHOLD, axis="features")

# Apply the SAME protein set to QC samples so downstream comparisons are consistent
kept_proteins  = quant_bio_filt.columns
quant_qc_filt  = quant_qc[kept_proteins]

print(f"Retained: {quant_bio_filt.shape[1]} proteins "
      f"(from {quant_bio.shape[1]}, threshold={COMPLETENESS_THRESHOLD:.0%})")
print(f"QC matrix aligned: {quant_qc_filt.shape[0]} samples × {quant_qc_filt.shape[1]} proteins")\
"""))

# ── 14. Stage 4 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 4: Normalisation

### Why normalise?

Even with careful sample preparation, total protein input, injection volume, and instrument
sensitivity vary between samples. These introduce **systematic per-sample offsets** unrelated
to biology.

**Median scaling** corrects for this in two steps:
1. Shift each sample's median to the global median (**loading correction**)
2. Equalise the spread of intensities across samples (**MAV normalisation**)

After normalisation, all sample boxplots should have aligned medians.

> **Exercise 4.1**: Look at the boxplot *before* and *after*.
> Which samples were shifted most? Can you identify the plate each shifted sample belongs to?

> **Note**: Median scaling removes per-sample global offsets. It does NOT remove
> protein-specific batch effects — those require batch correction (Stage 6).\
"""))

# ── 15. Before normalisation ──────────────────────────────────────────────────
cells.append(code("""\
# Boxplot before normalisation — samples coloured by plate
# Look for samples with systematically higher or lower medians (= loading variation)
print("Before normalisation:")
plot_sample_boxplot(quant_bio_filt, bio_meta, color_by="plate")\
"""))

# ── 16. Normalise ─────────────────────────────────────────────────────────────
cells.append(code("""\
# Apply median scaling to biological samples
quant_bio_norm = median_scaling(quant_bio_filt)

# Apply the same procedure to QC samples — needed for Stage 7 CV analysis
quant_qc_norm = median_scaling(quant_qc_filt)

print("After median scaling (biological samples):")
plot_sample_boxplot(quant_bio_norm, bio_meta, color_by="plate")\
"""))

# ── 17. KNN impute ────────────────────────────────────────────────────────────
cells.append(code("""\
# ── KNN imputation — for PCA and clustering ONLY ─────────────────────────────
# PCA requires a complete matrix (no NaN). We use K-nearest-neighbour imputation
# to fill missing values based on proteins with similar profiles across samples.
#
# IMPORTANT: imputed values are estimates. Use this matrix ONLY for visualisation
# (PCA, t-SNE, distance plots). For statistical testing, always use the
# original normalised matrix (quant_bio_norm) which preserves real missingness.
print("Applying KNN imputation (for PCA/clustering only)...")
quant_bio_imputed = impute_knn(quant_bio_norm, n_neighbors=5)
print(f"  Imputed matrix: {quant_bio_imputed.shape}  "
      f"(residual NaN: {quant_bio_imputed.isna().sum().sum()})")\
"""))

# ── 18. Multi-PC plot helper ───────────────────────────────────────────────────
cells.append(code("""\
# ── Helper: plot three PC pairs in one figure ─────────────────────────────────
# PCA decomposes variation into ordered principal components (PCs).
# PC1 explains the most variance, PC2 the second most, etc.
# Plotting multiple pairs lets us see what drives different axes of variation.

def plot_pc_pairs(matrix, metadata, color_by, title="", n_components=8,
                  outlier_samples=None):
    \"\"\"
    Run PCA and plot PC1v2, PC3v4, PC5v6 in a single row.

    Parameters
    ----------
    matrix         : samples × proteins DataFrame (must be complete — use imputed version)
    metadata       : sample metadata DataFrame aligned to matrix
    color_by       : column name in metadata used for point colours
    title          : figure title
    n_components   : number of PCs to compute (>=6 for three pairs)
    outlier_samples: list of sample IDs to mark with × symbol
    \"\"\"
    # Compute PCA
    pca    = _PCA(n_components=min(n_components, min(matrix.shape)))
    coords = pca.fit_transform(matrix)
    pc_df  = pd.DataFrame(coords, index=matrix.index,
                          columns=[f"PC{i+1}" for i in range(coords.shape[1])])

    # Build colour mapping from unique categories
    meta_al    = metadata.reindex(matrix.index)
    categories = sorted(meta_al[color_by].dropna().unique(), key=str)
    cmap       = plt.cm.tab10
    cmap2      = plt.cm.Set2
    color_map  = {cat: (cmap if len(categories) <= 10 else cmap2)(i / max(len(categories)-1, 1))
                  for i, cat in enumerate(categories)}

    outlier_set = set(outlier_samples or [])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    pairs = [(1, 2), (3, 4), (5, 6)]

    for ax, (pc_a, pc_b) in zip(axes, pairs):
        for cat in categories:
            idx  = meta_al.index[meta_al[color_by] == cat]
            sub  = pc_df.reindex(idx)
            # dim regular samples
            ax.scatter(sub[f"PC{pc_a}"], sub[f"PC{pc_b}"],
                       color=color_map[cat], alpha=0.5, s=18, label=str(cat))
        # Mark outlier samples with a large ×
        if outlier_set:
            out_idx = [s for s in outlier_set if s in pc_df.index]
            ax.scatter(pc_df.loc[out_idx, f"PC{pc_a}"],
                       pc_df.loc[out_idx, f"PC{pc_b}"],
                       marker="x", color="black", s=80, linewidths=2, zorder=5,
                       label="Outlier")
        ax.set_xlabel(f"PC{pc_a} ({pca.explained_variance_ratio_[pc_a-1]:.1%})", fontsize=10)
        ax.set_ylabel(f"PC{pc_b} ({pca.explained_variance_ratio_[pc_b-1]:.1%})", fontsize=10)
        ax.set_title(f"PC{pc_a} vs PC{pc_b}", fontsize=11)
        ax.legend(fontsize=7, loc="best", ncol=1 if len(categories) <= 6 else 2)

    fig.suptitle(f"{title} — coloured by {color_by}", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print a summary of variance explained
    var_str = "  ".join(f"PC{i+1}={pca.explained_variance_ratio_[i]:.1%}"
                        for i in range(min(6, coords.shape[1])))
    print(f"Variance explained: {var_str}")


print("plot_pc_pairs() helper defined")\
"""))

# ── 19. Stage 5 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 5: Outlier Detection

### Strategy: multiple independent metrics + KS-test confirmation

We use **5 detection methods** in parallel:

| Method | What it catches |
|--------|-----------------|
| Mean intensity Z-score | Globally shifted samples (degraded / contaminated) |
| Missing rate Z-score | Failed injections (unusually sparse) |
| KDE density Z-score | Samples with unusual intensity distributions |
| PCA Euclidean distance | Samples far from the cohort centroid in PC space |
| PCA Mahalanobis distance | Samples deviating in subtle, multivariate directions |

A **KS test** then confirms each candidate: the sample's intensity distribution must be
significantly different from the rest of the cohort (p < 0.05).

> **Exercise 5.1**: Run the detection cell and inspect the `outlier_table`.
> Which method(s) flagged each outlier? Are any outliers caught by only *one* method?
> What does this tell you about why we need multiple complementary approaches?\
"""))

# ── 20. Outlier detection ─────────────────────────────────────────────────────
cells.append(code("""\
ZSCORE_THRESHOLD = 2.576   # 99th-percentile z-score → α ≈ 0.01 per sample

# Run all five detection methods on the normalised biological matrix
print("Running outlier detection...")
outlier_mean    = sample_mean_outliers(quant_bio_norm,  ZSCORE_THRESHOLD)
outlier_missing = missing_rate_outliers(quant_bio_filt, ZSCORE_THRESHOLD)
outlier_density = density_outliers(quant_bio_norm,      ZSCORE_THRESHOLD)

# PCA-based detection: uses imputed matrix (requires complete data)
pca_out = PCAOutliers(quant_bio_imputed, n_components=5, threshold=ZSCORE_THRESHOLD)

detection_results = {
    "Mean intensity":  outlier_mean,
    "Missing rate":    outlier_missing,
    "Density":         outlier_density,
    "PCA-Euclidean":   pca_out.euclidean_outliers,
    "PCA-Mahalanobis": pca_out.mahalanobis_outliers,
}

print("\\nSamples flagged by each method:")
for method, flagged in detection_results.items():
    print(f"  {method:20s}: {len(flagged)} samples")\
"""))

# ── 21. KS confirm ────────────────────────────────────────────────────────────
cells.append(code("""\
# Pool candidates from all methods, then confirm with a two-sample KS test
# The KS test asks: is this sample's intensity distribution significantly different
# from the rest of the cohort? If yes, it is a confirmed outlier.
all_candidates = sorted({s for lst in detection_results.values() for s in lst})
print(f"Total candidate outliers (union): {len(all_candidates)}")

ks_confirmed  = ks_confirm_outliers(quant_bio_norm, all_candidates, alpha=0.05)
outlier_table = build_outlier_table(
    quant_bio_norm.index.tolist(), detection_results, ks_confirmed
)

confirmed_outliers = [s for s, v in ks_confirmed.items() if v]
print(f"KS-confirmed outliers: {len(confirmed_outliers)}")
print()
outlier_table\
"""))

# ── 22. Density + PCA with outliers ───────────────────────────────────────────
cells.append(code("""\
# Density curves — each line is one sample; confirmed outliers in red
# Red lines show clearly different intensity distributions (shifted, sparse, or noisy)
plot_density_per_sample(quant_bio_norm, outlier_samples=confirmed_outliers)

# PC1v2, PC3v4, PC5v6 coloured by plate — outliers marked with ×
# Confirmed outliers should appear as isolated points away from the main clouds
plot_pc_pairs(quant_bio_imputed, bio_meta, color_by="plate",
              title="PCA — BEFORE outlier removal",
              outlier_samples=confirmed_outliers)\
"""))

# ── 23. Stage 6 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 6: Batch-Effect QC

### What are batch effects?

In multi-plate DIA experiments, each MS plate (samples run on the same day / instrument)
introduces a **protein-specific systematic bias** unrelated to biology.  Unlike a uniform
intensity shift (which median normalisation removes), protein-specific plate effects create
a **complex multivariate pattern** that persists after normalisation.

**Consequences if uncorrected:**
- PC1/PC2 separate plates, not biology
- Differential expression tests have inflated false positives
- Clustering groups samples by plate, not phenotype

### How do we diagnose batch effects?

1. **PCA coloured by plate** — do samples cluster by plate even after normalisation?
2. **Within vs. between plate distances** — within << between = strong batch effect
3. **PC × factor heatmap** — how strongly does each factor (plate, group, sex, time) drive each PC?

> **Exercise 6.1**: Look at the PC × factor heatmap before batch correction.
> Which factor drives PC1? Which factor drives PC5? Does this match your expectation?\
"""))

# ── 24. Build clean bio matrix ────────────────────────────────────────────────
cells.append(code("""\
# Remove confirmed outliers; build the clean matrix used for batch QC and correction
confirmed_set = set(confirmed_outliers)
quant_bio_imputed_clean = quant_bio_imputed.loc[
    [s for s in quant_bio_imputed.index if s not in confirmed_set]
]
bio_meta_clean = bio_meta.reindex(quant_bio_imputed_clean.index)
print(f"Clean matrix: {quant_bio_imputed_clean.shape[0]} samples "
      f"({len(confirmed_outliers)} outliers removed)")\
"""))

# ── 25. PCA by plate + group, pre-correction ──────────────────────────────────
cells.append(code("""\
# PC1v2, PC3v4, PC5v6 coloured by plate — BEFORE batch correction
# Expect: PC1/PC2/PC3 dominated by plate structure
plot_pc_pairs(quant_bio_imputed_clean, bio_meta_clean, color_by="plate",
              title="BEFORE batch correction")

# Same PCs coloured by group — how visible is the biology before correction?
plot_pc_pairs(quant_bio_imputed_clean, bio_meta_clean, color_by="group",
              title="BEFORE batch correction")\
"""))

# ── 26. PCA by timepoint pre-correction ───────────────────────────────────────
cells.append(code("""\
# Colour by time point — do longitudinal time points cluster within subjects?
# Look especially at the higher PCs (PC3v4, PC5v6) after plate dominates PC1/PC2.
plot_pc_pairs(quant_bio_imputed_clean, bio_meta_clean, color_by="timepoint",
              title="BEFORE batch correction")\
"""))

# ── 27. Plate distances ───────────────────────────────────────────────────────
cells.append(code("""\
# Within-plate Euclidean distances (same-plate sample pairs) vs.
# between-plate distances (cross-plate sample pairs) in the first 10 PC dimensions.
# If within << between → significant plate batch effect → correction needed.
plate_labels = bio_meta["plate"].reindex(quant_bio_imputed_clean.index)
dist_stats = plate_distance_stats(quant_bio_imputed_clean, plate_labels, n_pca_components=10)

print(f"KS p-value (within vs. between plate): {dist_stats['ks_pvalue']:.4g}")
if dist_stats["ks_pvalue"] < 0.05:
    print("→ Significant plate separation — batch correction is recommended")

plot_plate_distances(dist_stats["within"], dist_stats["between"], dist_stats["ks_pvalue"])\
"""))

# ── 28. PC × factor ───────────────────────────────────────────────────────────
cells.append(code("""\
# PC × factor associations: ANOVA of each factor against each PC score.
# Returns -log10(p-value) — values > 1.3 (dotted line) are significant (p < 0.05).
#
# Age is continuous, so we bin it into quartiles for ANOVA.
# For a proper test of a continuous predictor, linear regression would be used;
# binning is a teaching simplification that works well here.
bio_meta_clean = bio_meta_clean.copy()
bio_meta_clean["age_quartile"] = pd.qcut(
    bio_meta_clean["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
)

pc_assoc = pc_factor_associations(
    quant_bio_imputed_clean, bio_meta_clean,
    factors=["plate", "group", "sex", "age_quartile", "timepoint"],
    n_pca_components=10,
)
print("PC × factor associations (−log10 p-value):")
print(pc_assoc.round(2).to_string())
plot_pc_factor_heatmap(pc_assoc)\
"""))

# ── 29. Batch correction markdown ─────────────────────────────────────────────
cells.append(md("""\
### Batch Correction

We compare two approaches:

| Method | How | When to use |
|--------|-----|-------------|
| **Plate-median** | Subtract each plate's protein median, add global median | Simple; works with NaN; fast |
| **ComBat** | Bayesian parametric model (additive + multiplicative) | More powerful; needs complete matrix |

After correction, **plate should no longer dominate PC1/PC2** — and biological signals
(group, sex, time) should become more visible.

> **Exercise 6.2**: Compare the PCA plots before and after each correction method.
> Which method gives a cleaner result? Does the group/time structure become more visible?\
"""))

# ── 30. Plate-median correction with before/after ─────────────────────────────
cells.append(code("""\
plate_lbl = bio_meta["plate"].reindex(quant_bio_imputed_clean.index)

# ── Method 1: Plate-median correction ────────────────────────────────────────
quant_pm = plate_median_correction(quant_bio_imputed_clean, plate_lbl)

# Side-by-side: PC1v2 before and after plate-median correction
# Both panels use the same colour scheme; the plate clusters should collapse after correction.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
palette_map = {p: PALETTE[i % len(PALETTE)]
               for i, p in enumerate(sorted(bio_meta_clean["plate"].unique()))}

for ax, matrix, title in [
    (axes[0], quant_bio_imputed_clean, "BEFORE correction"),
    (axes[1], quant_pm,                "AFTER plate-median"),
]:
    pca    = _PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    for j, plate in enumerate(sorted(bio_meta_clean["plate"].unique())):
        rows = [i for i, s in enumerate(matrix.index)
                if bio_meta_clean.loc[s, "plate"] == plate]
        ax.scatter(coords[rows, 0], coords[rows, 1],
                   color=palette_map[plate], alpha=0.55, s=18, label=plate)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)

fig.suptitle("PCA by plate — plate-median correction", fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

# ── 31. ComBat correction ─────────────────────────────────────────────────────
cells.append(code("""\
# ── Method 2: ComBat — Bayesian empirical Bayes batch correction ─────────────
# ComBat models both additive and multiplicative batch effects per protein,
# then adjusts all sample intensities using the fitted model.
try:
    quant_combat = combat_correction(quant_bio_imputed_clean, plate_lbl)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrix, title in [
        (axes[0], quant_bio_imputed_clean, "BEFORE correction"),
        (axes[1], quant_combat,            "AFTER ComBat"),
    ]:
        pca    = _PCA(n_components=2)
        coords = pca.fit_transform(matrix)
        for plate in sorted(bio_meta_clean["plate"].unique()):
            rows = [i for i, s in enumerate(matrix.index)
                    if bio_meta_clean.loc[s, "plate"] == plate]
            ax.scatter(coords[rows, 0], coords[rows, 1],
                       color=palette_map[plate], alpha=0.55, s=18, label=plate)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8)
    fig.suptitle("PCA by plate — ComBat correction", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # After ComBat: show group and time-point structure
    print("After ComBat — PC pairs coloured by group:")
    plot_pc_pairs(quant_combat, bio_meta_clean, color_by="group",
                  title="AFTER ComBat correction")
    print("After ComBat — PC pairs coloured by time point:")
    plot_pc_pairs(quant_combat, bio_meta_clean, color_by="timepoint",
                  title="AFTER ComBat correction")
    print("\\nComBat correction applied successfully")

except Exception as e:
    print(f"ComBat failed: {e}\\nFalling back to plate-median correction")
    quant_combat = quant_pm\
"""))

# ── 32. PC×factor post-correction ─────────────────────────────────────────────
cells.append(code("""\
# Repeat PC × factor heatmap AFTER ComBat correction
# Goal: plate association should drop; group/sex/timepoint associations should increase
bio_meta_combat = bio_meta_clean.copy()
bio_meta_combat["age_quartile"] = pd.qcut(
    bio_meta_combat["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
)
pc_assoc_post = pc_factor_associations(
    quant_combat, bio_meta_combat,
    factors=["plate", "group", "sex", "age_quartile", "timepoint"],
    n_pca_components=10,
)
print("PC × factor associations AFTER ComBat (−log10 p-value):")
print(pc_assoc_post.round(2).to_string())
plot_pc_factor_heatmap(pc_assoc_post)\
"""))

# ── 33. t-SNE before and after correction ─────────────────────────────────────
cells.append(code("""\
# ── t-SNE: QC pooled controls before and after batch correction ───────────────
# t-SNE preserves local neighbourhood structure. Because QC samples are identical
# biology (same pooled reference), after batch correction all 12 QC stars (★)
# should cluster tightly together regardless of which plate they came from.
#
# Before correction: QC stars may separate by plate (batch effect visible).
# After correction : QC stars cluster together → batch effect successfully removed.

# Prepare QC samples for t-SNE (normalised, same protein set as clean bio matrix)
quant_qc_for_tsne = quant_qc_norm.reindex(columns=quant_bio_imputed_clean.columns)
qc_plate_lbl      = qc_meta["plate"].reindex(quant_qc_for_tsne.index)
quant_qc_corrected = plate_median_correction(quant_qc_for_tsne, qc_plate_lbl)
quant_qc_corrected = quant_qc_corrected.fillna(quant_qc_corrected.mean())

# Fill residual NaN in QC before (uncorrected) for t-SNE
quant_qc_uncorrected = quant_qc_for_tsne.fillna(quant_qc_for_tsne.mean())

perp = min(30, (len(quant_bio_imputed_clean) + len(quant_qc_for_tsne)) // 4)
print(f"Running t-SNE on {len(quant_bio_imputed_clean)+len(quant_qc_for_tsne)} samples "
      f"(perplexity={perp}) — this may take ~30 seconds …")

def run_tsne(bio_mat, qc_mat, perp, seed=42):
    \"\"\"Combine bio + QC, run t-SNE, return coordinate DataFrames.\"\"\"
    combined = pd.concat([bio_mat, qc_mat]).fillna(0)
    coords   = TSNE(n_components=2, perplexity=perp, random_state=seed,
                    n_iter=1000).fit_transform(combined)
    return pd.DataFrame(coords, index=combined.index, columns=["tSNE1", "tSNE2"])

coords_before = run_tsne(quant_bio_imputed_clean, quant_qc_uncorrected, perp)
coords_after  = run_tsne(quant_combat,             quant_qc_corrected,   perp)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, coords, title in [
    (axes[0], coords_before, "BEFORE batch correction"),
    (axes[1], coords_after,  "AFTER ComBat correction"),
]:
    # Plot bio samples coloured by plate (semi-transparent)
    for plate in sorted(bio_meta_clean["plate"].unique()):
        bio_idx = bio_meta_clean.index[bio_meta_clean["plate"] == plate]
        sub = coords.reindex(bio_idx).dropna()
        ax.scatter(sub["tSNE1"], sub["tSNE2"],
                   color=palette_map[plate], alpha=0.3, s=15, label=f"Bio {plate}")

    # Plot QC samples as large stars — same plate colour, black edge
    for plate in sorted(qc_meta["plate"].unique()):
        qc_idx = qc_meta.index[qc_meta["plate"] == plate]
        sub = coords.reindex(qc_idx).dropna()
        ax.scatter(sub["tSNE1"], sub["tSNE2"],
                   color=palette_map[plate], alpha=1.0, s=250,
                   marker="*", edgecolors="black", linewidths=0.8)

    # Single legend entry for QC
    ax.scatter([], [], marker="*", color="grey", edgecolors="black", s=160,
               label="QC pooled (★)")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

fig.suptitle("t-SNE — biological samples (dots) + QC pooled controls (★)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.show()

print("★ stars clustering tightly together after correction = batch effect removed,")
print("  technical reproducibility of the pooled QC confirmed.")\
"""))

# ── 34. Stage 7 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 7: Coefficient of Variation (CV) Analysis

### What is CV and why does it matter?

**CV = (standard deviation / mean) × 100 %**

We compute three complementary CV metrics:

| CV type | Samples used | What it measures | Target |
|---------|-------------|-----------------|--------|
| **Intra-plate QC** | 3 pooled QC per plate | Pure instrument/injection noise | < 10 % |
| **Inter-plate QC** | 12 QC samples across plates | Batch effect + technical noise | ≈ Intra after correction |
| **Within-subject (bio)** | 3 time points per subject | Technical + longitudinal change | > QC intra-plate CV |
| **Between-subject (bio)** | One time point per subject (T00) | True biological variation | > Within-subject CV |

### The key ratio:
**Between-subject CV / Within-subject CV >> 1** → the assay can distinguish individuals.

> **Exercise 7.1**: After correction, is QC inter-plate CV close to QC intra-plate CV?
> What does convergence of these two metrics tell you about the batch correction?

> **Exercise 7.2**: Is between-subject CV larger than within-subject CV?
> If they were equal, what would that imply about the study's statistical power?\
"""))

# ── 35. QC CV analysis ────────────────────────────────────────────────────────
cells.append(code("""\
# ── QC sample CV: intra-plate (technical noise) vs. inter-plate (technical + batch) ─
# Convert from log2 to linear scale first — CV is not meaningful in log space
# because log(x) compresses large values and inflates small ones.
quant_qc_linear = 2 ** quant_qc_norm.fillna(float("nan"))

# Intra-plate CV: 3 QC replicates within each plate → pool 4 × n_proteins CVs
# This captures ONLY instrument/injection variability (the samples are identical)
intra_cv_parts = []
for plate in sorted(qc_meta["plate"].unique()):
    idx    = qc_meta.index[qc_meta["plate"] == plate]
    subset = quant_qc_linear.loc[idx]
    cv     = (subset.std() / subset.mean() * 100).dropna()
    intra_cv_parts.append(cv)
qc_intra_cv = pd.concat(intra_cv_parts)

# Inter-plate CV (before correction): all 12 QC samples — captures batch + technical noise
qc_inter_cv_before = (quant_qc_linear.std() / quant_qc_linear.mean() * 100).dropna()

# Inter-plate CV (after correction): QC samples after plate-median correction
qc_corrected_linear = 2 ** quant_qc_corrected.fillna(float("nan"))
qc_inter_cv_after   = (qc_corrected_linear.std() / qc_corrected_linear.mean() * 100).dropna()

print(f"QC intra-plate CV     (technical noise only) : {qc_intra_cv.median():.1f}%")
print(f"QC inter-plate CV     (before correction)    : {qc_inter_cv_before.median():.1f}%")
print(f"QC inter-plate CV     (after correction)     : {qc_inter_cv_after.median():.1f}%")
print()\

# ── Biological sample CV: within-subject (longitudinal) vs. between-subject ──
# Use the clean normalised bio matrix (real intensities, not imputed values)
quant_bio_norm_clean = quant_bio_norm.drop(index=confirmed_outliers, errors="ignore")
bio_meta_norm_clean  = bio_meta.reindex(quant_bio_norm_clean.index)
quant_bio_linear     = 2 ** quant_bio_norm_clean.fillna(float("nan"))

# Within-subject CV: CV across the 3 time points of each subject (biological + technical)
# Pool across all subjects and proteins
within_cv_parts = []
for sid in bio_meta_norm_clean["subject_id"].unique():
    idx    = bio_meta_norm_clean.index[bio_meta_norm_clean["subject_id"] == sid]
    if len(idx) >= 2:                                  # need ≥2 time points
        subset = quant_bio_linear.reindex(idx)
        cv     = (subset.std() / subset.mean() * 100).dropna()
        within_cv_parts.append(cv)
bio_within_cv = pd.concat(within_cv_parts) if within_cv_parts else pd.Series(dtype=float)

# Between-subject CV: CV across subjects at baseline (T00 only — one sample per subject)
t00_idx       = bio_meta_norm_clean.index[bio_meta_norm_clean["timepoint"] == "T00"]
t00_linear    = quant_bio_linear.reindex(t00_idx)
bio_between_cv = (t00_linear.std() / t00_linear.mean() * 100).dropna()

print(f"Bio within-subject CV  (3 time points/subject): {bio_within_cv.median():.1f}%")
print(f"Bio between-subject CV (baseline T00 only)    : {bio_between_cv.median():.1f}%")
print()
ratio = bio_between_cv.median() / bio_within_cv.median()
print(f"Between/within ratio: {ratio:.2f}x  "
      f"{'→ assay can distinguish subjects ✓' if ratio > 1.5 else '→ check assay quality'}")\
"""))

# ── 36. CV violin ─────────────────────────────────────────────────────────────
cells.append(code("""\
# Violin plot: compare all four CV distributions on the same axis
# The plot should show: QC intra ≈ QC inter (after correction) << bio within << bio between
fig, ax = plt.subplots(figsize=(11, 5))

labels = [
    "QC intra-plate\\n(technical)",
    "QC inter-plate\\nbefore correction",
    "QC inter-plate\\nafter correction",
    "Bio within-subject\\n(longitudinal)",
    "Bio between-subject\\n(biological)",
]
data = [
    qc_intra_cv.clip(0, 150),
    qc_inter_cv_before.clip(0, 150),
    qc_inter_cv_after.clip(0, 150),
    bio_within_cv.clip(0, 150),
    bio_between_cv.clip(0, 150),
]

vp = ax.violinplot(data, positions=range(1, len(data)+1), showmedians=True, widths=0.7)
ax.set_xticks(range(1, len(labels)+1))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("CV (%)", fontsize=12)
ax.set_title("CV comparison — technical vs. biological variation", fontsize=12)
ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="20 % QC benchmark")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("Interpretation:")
print("  QC intra-plate ≈ instrument noise; inter-plate before >> intra → large batch effect")
print("  QC inter-plate after ≈ intra → batch correction worked")
print("  Bio within-subject > QC → captures real longitudinal change + some batch")
print("  Bio between-subject >> within-subject → assay resolves biological individuality")\
"""))

# ── 37. QC correlation heatmap ────────────────────────────────────────────────
cells.append(code("""\
# Pearson correlation heatmap across all 12 QC samples
# Within-plate pairs (same 3 replicates, same plate effect) → r ≈ 0.999
# Between-plate pairs (different plate effect) → r lower before correction, ≈ 0.999 after
# A clear block structure = batch effect still present
print("QC sample correlation — BEFORE batch correction:")
plot_correlation_heatmap(quant_qc_norm, qc_meta, color_by="plate")\
"""))

# ── 38. Stage 8 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 8: Summary & Analysis-Ready Output

We now assemble the final clean matrix:
1. Remove confirmed outlier samples
2. Apply batch correction (ComBat preferred)
3. Save as parquet in the same proteins × samples format as the input\
"""))

# ── 39. Summary ───────────────────────────────────────────────────────────────
cells.append(code("""\
# Remove samples flagged by ≥2 independent detection methods
# Lowering the threshold (e.g. 1) removes more samples but risks false positives;
# raising it (e.g. 4) is more conservative but may miss subtle outliers.
N_METHODS_THRESHOLD = 2

if not outlier_table.empty:
    remove_samples = outlier_table[
        outlier_table["n_methods_flagged"] >= N_METHODS_THRESHOLD
    ].index.tolist()
else:
    remove_samples = []

print(f"Samples to remove (≥{N_METHODS_THRESHOLD} methods flagged): {len(remove_samples)}")
print(f"  {remove_samples}")

# Drop outliers from the normalised biological matrix (real values, not imputed)
quant_bio_clean = quant_bio_norm.drop(index=remove_samples, errors="ignore")
print(f"\\nFinal matrix (normalised, outliers removed): {quant_bio_clean.shape}")\
"""))

# ── 40. Save output ───────────────────────────────────────────────────────────
cells.append(code("""\
import json

Path("results/qc/tables").mkdir(parents=True, exist_ok=True)

# Reconstruct proteins × samples format (matches Spectronaut input)
# prot_meta contains gene names and protein names; quant_bio_clean.T adds the intensity columns
out_df = prot_meta.join(quant_bio_clean.T, how="left").reset_index()
out_df.to_parquet("results/qc/proteomics_analysis_ready.parquet", index=False)
print("Saved: results/qc/proteomics_analysis_ready.parquet")

# Record all QC decisions for reproducibility and methods reporting
qc_decisions = {
    "normalization":                  "median_scaling",
    "completeness_threshold":         COMPLETENESS_THRESHOLD,
    "outlier_zscore_threshold":       ZSCORE_THRESHOLD,
    "outlier_n_methods_threshold":    N_METHODS_THRESHOLD,
    "n_outliers_removed":             len(remove_samples),
    "outliers_removed":               remove_samples,
    "n_samples_final":                int(quant_bio_clean.shape[0]),
    "n_proteins_final":               int(quant_bio_clean.shape[1]),
}
with open("results/qc/tables/qc_decisions.json", "w") as f:
    json.dump(qc_decisions, f, indent=2)
print("Saved: results/qc/tables/qc_decisions.json")

print()
print("=== QC Summary ===")
print(f"  Input (bio):         {quant_bio.shape[0]} samples × {quant_bio.shape[1]} proteins")
print(f"  After filtering:     {quant_bio_filt.shape[1]} proteins (threshold={COMPLETENESS_THRESHOLD:.0%})")
print(f"  Outliers removed:    {len(remove_samples)} samples")
print(f"  Final matrix:        {quant_bio_clean.shape[0]} samples × {quant_bio_clean.shape[1]} proteins")\
"""))

# ── 41. Exercises ─────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Extended Exercises

### Exercise A — Threshold sensitivity
Change `COMPLETENESS_THRESHOLD` to 0.05, 0.50, and 1.0 and re-run the full pipeline.
How many proteins remain at each threshold? At 100 % completeness, why do almost no proteins survive?

### Exercise B — Outlier threshold sensitivity
Change `N_METHODS_THRESHOLD` to 1 and 4.
How many samples are removed at each setting?
What is the risk of being too lenient (threshold=1) vs. too strict (threshold=4)?

### Exercise C — Batch correction comparison
Run the PC × factor heatmap on `quant_pm` (plate-median) and `quant_combat` (ComBat).
For which method does the plate association drop below significance (−log10 p < 1.3)?
Does group or timepoint association increase after correction?

### Exercise D — QC star clustering in t-SNE
In the t-SNE plots, do the 12 QC stars cluster more tightly after correction?
Which plate shows the greatest shift? Does this match the plate with the largest CV
reduction (QC inter-plate before vs. after)?

### Exercise E — Within-subject vs. between-subject CV
After ComBat correction, recompute bio within-subject CV using `quant_combat`.
Does correction reduce the within-subject CV?
Does the between/within ratio change after correction?

### Exercise F — Your own data
Replace `demo_parquet` with a path to your own Spectronaut protein-level export.
Ensure:
1. The parquet has `PG_ProteinAccessions` as the first column
2. Sample columns contain log2 intensities (Spectronaut default)
3. A metadata CSV with `plate`, `is_qc`, and optionally `timepoint` and `subject_id` columns exists

---
*Pipeline source code: github.com/nigelkurgan/ddea-proteomics-course*\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE + WRITE
# ─────────────────────────────────────────────────────────────────────────────

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out = Path("notebooks/proteomics_qc_tutorial.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({len(cells)} cells)")
