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


# ─────────────────────────────────────────────────────────────────────────────
# CELLS
# ─────────────────────────────────────────────────────────────────────────────

cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(md("""\
# Proteomics QC Pipeline — Hands-On Tutorial

**Course: Proteomics Data Analysis**
**Environment: Google Colab (or local Jupyter)**

---

## Learning Objectives

By the end of this notebook you will be able to:

1. Load and inspect a DIA proteomics intensity matrix
2. Apply protein completeness filtering and explain the trade-offs
3. Normalise intensities using median scaling and understand why it works
4. Detect and confirm technical outlier samples using multiple independent methods
5. Diagnose and correct plate batch effects using ComBat or plate-median correction
6. Assess technical reproducibility using pooled QC controls
7. Interpret coefficient of variation as a measure of assay reproducibility
8. Generate a standardised QC report ready for publication

---

## Dataset

We use a **synthetic dataset** with the following properties:
- 2 000 proteins, 160 biological samples across 4 plates (40 samples/plate)
- 80 Cases / 80 Controls (balanced); sex and age as covariates
- **12 pooled QC samples** (3 identical replicates per plate) for technical QC
- **Strong plate batch effects** — visible in PCA before correction
- **Hidden outlier samples** planted in the data — you will discover them!
- ~37 % missing values (realistic for DIA plasma proteomics — see Stage 2)\
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
# ── Install dependencies ─────────────────────────────────────────────────────
# This takes ~60–90 seconds on Colab's first run
import subprocess, sys

packages = [
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "pyarrow>=12.0",
    "inmoose>=0.3",   # ComBat batch correction
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✓ Dependencies installed")

# ── Clone pipeline repo ──────────────────────────────────────────────────────
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
from sklearn.decomposition import PCA as _PCA

# Pipeline modules
from proteomics_qc.proteomics.filters import filter_by_completeness, completeness_at_thresholds
from proteomics_qc.proteomics.normalise import (
    median_scaling, impute_knn, plate_median_correction, combat_correction
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
    compute_sample_cv, compute_intra_cv, compute_inter_cv,
    plot_cv_violin, plot_cv_comparison,
)

apply_style()
print("✓ All imports successful")\
"""))

# ── 4. Stage 1 markdown ───────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 1: Load the Data

### What format does proteomics data come in?

Spectronaut (and DIA-NN) export results as a **long-format report** or a **matrix**.
For QC, we use the **protein-level matrix**:

```
          Sample_001  Sample_002  ...  Sample_160
P00001        23.4        22.8   ...      24.1
P00002        19.1         NaN   ...      20.3
  ...          ...         ...   ...       ...
```

- Rows = proteins (identified by UniProt accession)
- Columns = samples (biological + pooled QC controls)
- Values = **log2 MS1 intensity** (LFQ or MaxLFQ)
- **NaN** = protein not detected in that sample

We store this in [Apache Parquet](https://parquet.apache.org/) format — columnar storage that is fast to read and compact.\
"""))

# ── 5. Load data ──────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Generate demo data (skip if already present) ─────────────────────────────
demo_parquet = Path("data/demo/demo_proteomics.parquet")
demo_meta_csv = Path("data/demo/demo_metadata.csv")

if not demo_parquet.exists():
    import subprocess
    subprocess.check_call(["python", "scripts/generate_demo_data.py"])

# ── Load protein matrix (proteins × samples parquet) ─────────────────────────
raw = pd.read_parquet(demo_parquet)

# Separate protein metadata columns (PG_*) from intensity columns
meta_cols   = [c for c in raw.columns if c.startswith("PG_")]
sample_cols = [c for c in raw.columns if not c.startswith("PG_")]

prot_meta = raw.set_index("PG_ProteinAccessions")[meta_cols[1:]]

# TRANSPOSE to (samples × proteins) — most analysis functions expect this orientation
quant_df = raw.set_index("PG_ProteinAccessions")[sample_cols].T

print(f"Matrix shape: {quant_df.shape[0]} samples × {quant_df.shape[1]} proteins")
print(f"Intensity range (log2): {quant_df.min().min():.1f} – {quant_df.max().max():.1f}")
print(f"Overall missing rate: {quant_df.isna().mean().mean():.1%}")
quant_df.head(3)\
"""))

# ── 6. Load metadata + split ──────────────────────────────────────────────────
cells.append(code("""\
# ── Load sample metadata ─────────────────────────────────────────────────────
sample_meta = pd.read_csv(demo_meta_csv, index_col=0)
sample_meta = sample_meta.reindex(quant_df.index)   # align to matrix row order

# Split into biological samples and pooled QC controls
bio_meta = sample_meta[~sample_meta["is_qc"]]
qc_meta  = sample_meta[sample_meta["is_qc"]]

print(f"Biological samples : {len(bio_meta)}")
print(f"QC pooled samples  : {len(qc_meta)} ({qc_meta['plate'].value_counts().to_dict()})")
print()
print("Plate × group counts (biological):")
print(bio_meta.value_counts(["plate", "group"]).sort_index().to_string())
bio_meta.head()\
"""))

# ── 7. Split quant matrices ───────────────────────────────────────────────────
cells.append(code("""\
# ── Separate biological and QC intensity matrices ────────────────────────────
# All filtering, normalisation and outlier detection use biological samples.
# QC pooled controls (identical biology, 3 per plate) are analysed separately
# in Stage 7 to quantify technical (intra-plate) vs. batch (inter-plate) variation.

quant_bio = quant_df.loc[bio_meta.index]
quant_qc  = quant_df.loc[qc_meta.index]

print(f"Biological matrix : {quant_bio.shape[0]} samples × {quant_bio.shape[1]} proteins")
print(f"QC matrix         : {quant_qc.shape[0]} samples × {quant_qc.shape[1]} proteins")
print(f"QC missing rate   : {quant_qc.isna().mean().mean():.1%}  "
      f"(biological: {quant_bio.isna().mean().mean():.1%})")
print()
print("QC samples (3 identical pooled controls per plate):")
print(qc_meta[["plate", "group"]])\
"""))

# ── 8. Stage 2 markdown ───────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 2: Missing Value Exploration

### Why do proteomics data have missing values?

Missing values in DIA proteomics are **not random** — they are **Missing Not At Random (MNAR)**:
low-abundance proteins fall below the instrument's detection limit.

**Key questions before filtering:**
1. Which proteins are most frequently missing?
2. Is missingness structured by plate (= technical) or by sample group (= biological)?
3. What completeness threshold balances protein count vs. data quality?

The heatmap below shows proteins (x-axis) × biological samples (y-axis). Black = missing.\
"""))

# ── 9. Missing heatmap ────────────────────────────────────────────────────────
cells.append(code("""\
# Missing value heatmap — biological samples sorted by plate to reveal structure
plot_missing_heatmap(quant_bio, bio_meta, sort_by="plate")\
"""))

# ── 10. Missing threshold ──────────────────────────────────────────────────────
cells.append(code("""\
# How many proteins survive different completeness thresholds?
plot_missing_by_threshold(quant_bio)

counts = completeness_at_thresholds(quant_bio)
print("\\nProteins retained at each threshold:")
print(counts.to_string())\
"""))

# ── 11. Missing per group ─────────────────────────────────────────────────────
cells.append(code("""\
# Is missingness uniform across plates, or does one plate have more missing?
plot_missing_per_group(quant_bio, bio_meta["plate"])\
"""))

# ── 12. Stage 3 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 3: Protein Completeness Filtering

We remove proteins detected in fewer than `threshold` fraction of **biological** samples.
The default is **20 %** — a commonly used cutoff for longitudinal DIA studies.

> **Exercise 3.1**: Change `COMPLETENESS_THRESHOLD` to 0.10 and 0.50 and note how many proteins are retained.
> Which threshold would you choose for a study with 2 groups where one group has systematically lower expression?\
"""))

# ── 13. Filter ────────────────────────────────────────────────────────────────
cells.append(code("""\
COMPLETENESS_THRESHOLD = 0.20  # ← try changing this

# Filter on biological samples; apply the same protein set to QC samples
quant_bio_filt = filter_by_completeness(quant_bio, threshold=COMPLETENESS_THRESHOLD, axis="features")

kept_proteins = quant_bio_filt.columns          # protein set to use throughout
quant_qc_filt = quant_qc[kept_proteins]         # same proteins for QC

print(f"Remaining: {quant_bio_filt.shape[1]} proteins in {quant_bio_filt.shape[0]} biological samples")
print(f"QC matrix aligned: {quant_qc_filt.shape[0]} QC samples × {quant_qc_filt.shape[1]} proteins")\
"""))

# ── 14. Stage 4 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 4: Normalisation

### Why normalise?

Even after careful sample preparation, total protein input, injection volume, and instrument sensitivity vary between samples.
This introduces **systematic offsets** unrelated to biology.

**Median scaling** corrects for this in two steps:
1. Shift each sample's median to the global median (**loading correction**)
2. Equalise the spread across samples (**MAV normalisation**)

After normalisation, the per-sample boxplot should show **aligned medians**.

> **Exercise 4.1**: Look at the boxplot *before* normalisation and *after*.
> Can you see which samples were shifted? Do you notice any persistent outliers?\
"""))

# ── 15. Before normalisation ──────────────────────────────────────────────────
cells.append(code("""\
# Before normalisation — biological samples coloured by plate
print("Before normalisation:")
plot_sample_boxplot(quant_bio_filt, bio_meta, color_by="plate")\
"""))

# ── 16. Normalise ─────────────────────────────────────────────────────────────
cells.append(code("""\
# Apply median scaling to biological samples
quant_bio_norm = median_scaling(quant_bio_filt)

# Apply the same procedure to QC samples (needed for Stage 7 CV analysis)
quant_qc_norm = median_scaling(quant_qc_filt)

print("After median scaling (biological samples):")
plot_sample_boxplot(quant_bio_norm, bio_meta, color_by="plate")\
"""))

# ── 17. KNN impute ────────────────────────────────────────────────────────────
cells.append(code("""\
# KNN imputation for PCA / clustering (complete matrix required)
# IMPORTANT: Use ONLY for visualisation — not for downstream statistics
print("Applying KNN imputation (for PCA/clustering only)...")
quant_bio_imputed = impute_knn(quant_bio_norm, n_neighbors=5)
print(f"  Imputed matrix: {quant_bio_imputed.shape}  "
      f"(fully observed: {quant_bio_imputed.isna().sum().sum()} NaN)")\
"""))

# ── 18. Stage 5 markdown ──────────────────────────────────────────────────────
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
| PCA Mahalanobis distance | Samples deviating in subtle, low-variance directions |

A **KS test** then confirms each candidate: the sample's intensity distribution must be
significantly different from the rest of the cohort (p < 0.05).

> **Exercise 5.1**: Run the detection cell below and inspect the `outlier_table`.
> Which methods flagged each outlier? Are any outliers caught by only *one* method?
> What does this tell you about why we use multiple methods rather than just one?\
"""))

# ── 19. Outlier detection ─────────────────────────────────────────────────────
cells.append(code("""\
ZSCORE_THRESHOLD = 2.576  # 99th percentile, α ≈ 0.01

# Run all detection methods on biological samples
print("Running outlier detection...")
outlier_mean    = sample_mean_outliers(quant_bio_norm, ZSCORE_THRESHOLD)
outlier_missing = missing_rate_outliers(quant_bio_filt, ZSCORE_THRESHOLD)
outlier_density = density_outliers(quant_bio_norm, ZSCORE_THRESHOLD)
pca_out         = PCAOutliers(quant_bio_imputed, n_components=5, threshold=ZSCORE_THRESHOLD)

detection_results = {
    "Mean intensity":  outlier_mean,
    "Missing rate":    outlier_missing,
    "Density":         outlier_density,
    "PCA-Euclidean":   pca_out.euclidean_outliers,
    "PCA-Mahalanobis": pca_out.mahalanobis_outliers,
}

print("\\nFlagged by each method:")
for method, flagged in detection_results.items():
    print(f"  {method:20s}: {len(flagged)} samples")\
"""))

# ── 20. KS confirm ────────────────────────────────────────────────────────────
cells.append(code("""\
# Pool all candidates and confirm with KS test
all_candidates = sorted({s for lst in detection_results.values() for s in lst})
print(f"Total candidate outliers: {len(all_candidates)}")

ks_confirmed   = ks_confirm_outliers(quant_bio_norm, all_candidates, alpha=0.05)
outlier_table  = build_outlier_table(
    quant_bio_norm.index.tolist(), detection_results, ks_confirmed
)

print(f"\\nKS-confirmed outliers: {outlier_table['KS_confirmed'].eq('✓').sum()}")
print()
outlier_table\
"""))

# ── 21. Density plot ──────────────────────────────────────────────────────────
cells.append(code("""\
# Density curves — outliers shown in red
confirmed_outliers = [s for s, v in ks_confirmed.items() if v]
plot_density_per_sample(quant_bio_norm, outlier_samples=confirmed_outliers)\
"""))

# ── 22. PCA with outliers ─────────────────────────────────────────────────────
cells.append(code("""\
# PCA coloured by plate — confirmed outliers marked with ×
plot_pca(quant_bio_imputed, bio_meta, color_by="plate",
         title="PCA — plate (pre-correction, outliers marked ×)",
         outlier_samples=confirmed_outliers)\
"""))

# ── 23. Stage 6 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 6: Batch-Effect QC

### What are batch effects?

In multi-plate DIA experiments each MS plate (samples run on the same day/instrument)
introduces a **systematic offset** unrelated to biology.  If uncorrected:
- PC1 separates plates, not biology
- Differential expression tests have inflated false positives
- Clustering groups by plate, not phenotype

### How do we diagnose batch effects?

1. **PCA coloured by plate** — do samples cluster by plate?
2. **Within vs. between plate distances** — within << between = strong batch effect
3. **PC × factor heatmap** — does plate explain more of PC1/PC2 than biology, sex, or age?

> **Exercise 6.1**: Look at the PC × factor heatmap.
> Which factor drives PC1: plate or group?
> Do sex and age also show significant associations with any PC?\
"""))

# ── 24. Build clean matrix ────────────────────────────────────────────────────
cells.append(code("""\
# Remove confirmed outliers; build clean bio matrix for batch QC
confirmed_set = set(confirmed_outliers)
quant_bio_imputed_clean = quant_bio_imputed.loc[
    [s for s in quant_bio_imputed.index if s not in confirmed_set]
]
bio_meta_clean = bio_meta.reindex(quant_bio_imputed_clean.index)
print(f"Clean matrix: {quant_bio_imputed_clean.shape[0]} samples ({len(confirmed_outliers)} outliers removed)")\
"""))

# ── 25. PCA by plate (pre-correction) ─────────────────────────────────────────
cells.append(code("""\
# PCA coloured by plate — before batch correction
plot_pca(quant_bio_imputed_clean, bio_meta_clean, color_by="plate",
         title="PCA — by plate (outliers removed, BEFORE batch correction)")\
"""))

# ── 26. PCA by group (pre-correction) ─────────────────────────────────────────
cells.append(code("""\
# PCA coloured by group — is biology visible before correction?
plot_pca(quant_bio_imputed_clean, bio_meta_clean, color_by="group",
         title="PCA — by group (BEFORE batch correction)")\
"""))

# ── 27. Plate distances ───────────────────────────────────────────────────────
cells.append(code("""\
# Within vs. between plate Euclidean distances in PC space
plate_labels = bio_meta["plate"].reindex(quant_bio_imputed_clean.index)
dist_stats = plate_distance_stats(quant_bio_imputed_clean, plate_labels, n_pca_components=10)
print(f"KS p-value (within vs. between plate): {dist_stats['ks_pvalue']:.4g}")
if dist_stats["ks_pvalue"] < 0.05:
    print("→ Significant plate separation detected — batch correction recommended")

plot_plate_distances(dist_stats["within"], dist_stats["between"], dist_stats["ks_pvalue"])\
"""))

# ── 28. PC × factor ───────────────────────────────────────────────────────────
cells.append(code("""\
# PC × factor associations: plate, group, sex and age (binned to quartiles for ANOVA)
bio_meta_clean = bio_meta_clean.copy()
bio_meta_clean["age_quartile"] = pd.qcut(
    bio_meta_clean["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
)

pc_assoc = pc_factor_associations(
    quant_bio_imputed_clean, bio_meta_clean,
    factors=["plate", "group", "sex", "age_quartile"],
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
| **Plate-median** | Subtract plate protein-median, add global median | Simple, works with NaN |
| **ComBat** | Bayesian parametric model (additive + multiplicative) | More powerful; needs complete matrix |

After correction, **plate should no longer drive PC1** — and group separation (biology) may improve.

> **Exercise 6.2**: Compare the PCA plots before and after each correction method.
> Does the plate structure disappear? Does group separation improve?
> Which method gives a cleaner result?\
"""))

# ── 30. Plate-median correction with before/after comparison ──────────────────
cells.append(code("""\
plate_lbl = bio_meta["plate"].reindex(quant_bio_imputed_clean.index)

# ── Method 1: Plate-median correction ────────────────────────────────────────
quant_pm = plate_median_correction(quant_bio_imputed_clean, plate_lbl)

# Side-by-side: PCA before vs. after plate-median correction
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
palette_map = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(sorted(bio_meta_clean["plate"].unique()))}

for ax, matrix, title in [
    (axes[0], quant_bio_imputed_clean, "BEFORE correction"),
    (axes[1], quant_pm,                "AFTER plate-median"),
]:
    pca = _PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    for j, plate in enumerate(sorted(bio_meta_clean["plate"].unique())):
        idx = bio_meta_clean.index[bio_meta_clean["plate"] == plate]
        rows = [i for i, s in enumerate(matrix.index) if s in set(idx)]
        ax.scatter(coords[rows, 0], coords[rows, 1],
                   color=palette_map[plate], alpha=0.65, s=30, label=plate)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)

fig.suptitle("PCA coloured by plate — batch correction comparison", fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

# ── 31. ComBat + before/after ─────────────────────────────────────────────────
cells.append(code("""\
# ── Method 2: ComBat (Bayesian, requires complete matrix) ────────────────────
try:
    quant_combat = combat_correction(quant_bio_imputed_clean, plate_lbl)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrix, title in [
        (axes[0], quant_bio_imputed_clean, "BEFORE correction"),
        (axes[1], quant_combat,            "AFTER ComBat"),
    ]:
        pca = _PCA(n_components=2)
        coords = pca.fit_transform(matrix)
        for plate in sorted(bio_meta_clean["plate"].unique()):
            idx = bio_meta_clean.index[bio_meta_clean["plate"] == plate]
            rows = [i for i, s in enumerate(matrix.index) if s in set(idx)]
            ax.scatter(coords[rows, 0], coords[rows, 1],
                       color=palette_map[plate], alpha=0.65, s=30, label=plate)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8)
    fig.suptitle("PCA coloured by plate — ComBat comparison", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # PCA by group after ComBat — biology should be more visible
    plot_pca(quant_combat, bio_meta_clean, color_by="group",
             title="PCA — by group (AFTER ComBat correction)")
    print("\\nComBat correction applied successfully")

except Exception as e:
    print(f"ComBat failed: {e}")
    print("Using plate-median correction instead")
    quant_combat = quant_pm\
"""))

# ── 32. t-SNE ─────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── t-SNE: QC pooled controls should cluster tightly after correction ─────────
# We include the 12 QC samples (3 per plate, all identical biology) alongside
# the batch-corrected biological samples.
# QC samples are corrected for plate effects using plate-median normalisation.

# Normalise and correct QC samples
quant_qc_for_tsne = quant_qc_norm.reindex(columns=quant_bio_imputed_clean.columns)
qc_plate_lbl      = qc_meta["plate"].reindex(quant_qc_for_tsne.index)
quant_qc_corrected = plate_median_correction(quant_qc_for_tsne, qc_plate_lbl)
quant_qc_corrected = quant_qc_corrected.fillna(quant_qc_corrected.mean())

# Combine bio (ComBat-corrected) + QC (plate-corrected)
combined = pd.concat([quant_combat, quant_qc_corrected])
combined = combined.fillna(combined.mean())   # fill any residual NaN

perplexity = min(30, len(combined) // 3)
print(f"Running t-SNE on {len(combined)} samples (perplexity={perplexity})…")
tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
coords = pd.DataFrame(
    tsne.fit_transform(combined),
    index=combined.index, columns=["tSNE1", "tSNE2"]
)

fig, ax = plt.subplots(figsize=(9, 7))

# Biological samples — semi-transparent, coloured by plate
for plate in sorted(bio_meta_clean["plate"].unique()):
    idx = bio_meta_clean.index[bio_meta_clean["plate"] == plate]
    sub = coords.loc[[s for s in idx if s in coords.index]]
    ax.scatter(sub["tSNE1"], sub["tSNE2"],
               color=palette_map[plate], alpha=0.45, s=28, label=f"Bio {plate}")

# QC samples — large stars, same colour as their plate, black edge
for plate in sorted(qc_meta["plate"].unique()):
    idx = qc_meta.index[qc_meta["plate"] == plate]
    sub = coords.loc[[s for s in idx if s in coords.index]]
    ax.scatter(sub["tSNE1"], sub["tSNE2"],
               color=palette_map[plate], alpha=1.0, s=200,
               marker="*", edgecolors="black", linewidths=0.7)

# Legend entry for QC
ax.scatter([], [], marker="*", color="grey", edgecolors="black", s=160,
           label="QC pooled (★)")
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title("t-SNE — batch-corrected samples\\n★ QC pooled controls should cluster tightly",
             fontsize=12)
ax.legend(fontsize=8, loc="upper right", ncol=2)
plt.tight_layout()
plt.show()

print("If QC stars (★) cluster near their plate's bio samples but also close to each other,")
print("batch correction removed most inter-plate bias while preserving biological variation.")\
"""))

# ── 33. Stage 7 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 7: Coefficient of Variation (CV) Analysis

### What is CV and why does it matter?

**CV = (standard deviation / mean) × 100 %**

We use three CV metrics to assess assay quality:

| CV type | Samples used | What it measures | Target |
|---------|-------------|-----------------|--------|
| **Intra-plate QC** | 3 pooled QC per plate | Pure instrument/injection noise | < 10 % |
| **Inter-plate QC** | 12 QC samples across plates | Batch effect + technical noise | ≈ Intra after correction |
| **Inter-individual (bio)** | 160 biological samples | True biological variation | > QC CV |

### Two key questions:
1. **Is intra-plate QC CV low?** (< 10 %) → The assay is technically reproducible
2. **Is inter-individual CV > QC CV?** → The assay can detect biological differences

> **Exercise 7.1**: Look at the intra-plate vs. inter-plate QC CV.
> Does batch correction make them converge? What does that tell you?

> **Exercise 7.2**: Is inter-individual bio CV larger than intra-plate QC CV?
> If they were equal, what would that imply about the study's power?\
"""))

# ── 34. QC CV analysis ────────────────────────────────────────────────────────
cells.append(code("""\
# ── QC sample CV: intra-plate (technical) vs. inter-plate (technical + batch) ─
quant_qc_linear = 2 ** quant_qc_norm.fillna(float("nan"))

# Intra-plate CV: 3 QC replicates within each plate → pool across 4 plates
intra_cv_parts = []
for plate in sorted(qc_meta["plate"].unique()):
    idx = qc_meta.index[qc_meta["plate"] == plate]
    subset = quant_qc_linear.loc[idx]
    cv = (subset.std() / subset.mean() * 100).dropna()
    intra_cv_parts.append(cv)
qc_intra_cv = pd.concat(intra_cv_parts)

# Inter-plate CV: all 12 QC samples — captures batch effects
qc_inter_cv_before = (quant_qc_linear.std() / quant_qc_linear.mean() * 100).dropna()

# Inter-plate CV after batch correction
qc_corrected_linear = 2 ** quant_qc_corrected.fillna(float("nan"))
qc_inter_cv_after   = (qc_corrected_linear.std() / qc_corrected_linear.mean() * 100).dropna()

print(f"QC intra-plate CV   (technical noise only) : {qc_intra_cv.median():.1f}%")
print(f"QC inter-plate CV   (before correction)    : {qc_inter_cv_before.median():.1f}%")
print(f"QC inter-plate CV   (after correction)     : {qc_inter_cv_after.median():.1f}%")

# Bio inter-individual CV
quant_bio_linear = 2 ** quant_bio_norm.fillna(float("nan"))
bio_inter_cv = (quant_bio_linear.std() / quant_bio_linear.mean() * 100).dropna()
print(f"Bio inter-individual CV (biological var.)  : {bio_inter_cv.median():.1f}%")\
"""))

# ── 35. QC CV violin ──────────────────────────────────────────────────────────
cells.append(code("""\
# Violin plot comparing all four CV distributions
fig, ax = plt.subplots(figsize=(11, 5))
labels = [
    "QC intra-plate\\n(technical)",
    "QC inter-plate\\n(before correction)",
    "QC inter-plate\\n(after correction)",
    "Bio inter-individual\\n(biological)",
]
data = [
    qc_intra_cv.clip(0, 150),
    qc_inter_cv_before.clip(0, 150),
    qc_inter_cv_after.clip(0, 150),
    bio_inter_cv.clip(0, 150),
]
vp = ax.violinplot(data, positions=range(1, 5), showmedians=True, widths=0.7)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("CV (%)", fontsize=12)
ax.set_title("CV comparison: technical vs. biological variation", fontsize=12)
ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="20 % benchmark")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("Interpretation:")
print("  • Intra-plate QC CV ≈ pure technical noise (injection reproducibility)")
print("  • Inter-plate QC before correction >> intra-plate  → large batch effect")
print("  • Inter-plate QC after correction  ≈ intra-plate  → batch removed")
print("  • Bio inter-individual CV >> QC CV → assay can detect biology")\
"""))

# ── 36. QC correlation heatmap ────────────────────────────────────────────────
cells.append(code("""\
# QC sample Pearson correlation heatmap
# Within-plate pairs (same 3 QC samples) should show r > 0.99
# Between-plate pairs may differ before correction; should converge after
plot_correlation_heatmap(quant_qc_norm, qc_meta, color_by="plate")\
"""))

# ── 37. Stage 8 markdown ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Stage 8: Summary & Analysis-Ready Output

We now produce a clean, analysis-ready protein matrix:
1. Remove confirmed outlier samples
2. Apply batch correction (ComBat preferred)
3. Save as parquet (same format as input, for downstream statistical tools)\
"""))

# ── 38. Summary ───────────────────────────────────────────────────────────────
cells.append(code("""\
# Exclude samples flagged by ≥ 2 methods (adjust as needed)
N_METHODS_THRESHOLD = 2

if not outlier_table.empty:
    remove_samples = outlier_table[
        outlier_table["n_methods_flagged"] >= N_METHODS_THRESHOLD
    ].index.tolist()
else:
    remove_samples = []

print(f"Samples to remove (≥{N_METHODS_THRESHOLD} methods): {len(remove_samples)}")
print(f"  {remove_samples}")

# Remove from normalised biological matrix (keeps NaN — real values only)
quant_bio_clean = quant_bio_norm.drop(index=remove_samples, errors="ignore")
print(f"\\nFinal matrix (normalised, outliers removed): {quant_bio_clean.shape}")\
"""))

# ── 39. Save output ───────────────────────────────────────────────────────────
cells.append(code("""\
import json

Path("results/qc/tables").mkdir(parents=True, exist_ok=True)

# Reconstruct proteins × samples format (matches Spectronaut input)
out_df = prot_meta.join(quant_bio_clean.T, how="left").reset_index()
out_df.to_parquet("results/qc/proteomics_analysis_ready.parquet", index=False)
print("Saved: results/qc/proteomics_analysis_ready.parquet")

qc_decisions = {
    "normalization": "median_scaling",
    "completeness_threshold": COMPLETENESS_THRESHOLD,
    "outlier_n_methods_threshold": N_METHODS_THRESHOLD,
    "n_outliers_removed": len(remove_samples),
    "outliers_removed": remove_samples,
    "n_samples_final": int(quant_bio_clean.shape[0]),
    "n_proteins_final": int(quant_bio_clean.shape[1]),
}
with open("results/qc/tables/qc_decisions.json", "w") as f:
    json.dump(qc_decisions, f, indent=2)
print("Saved: results/qc/tables/qc_decisions.json")

print()
print("=== QC Summary ===")
print(f"  Input:                 {quant_bio.shape[0]} bio samples × {quant_bio.shape[1]} proteins")
print(f"  After filtering:       {quant_bio_filt.shape[1]} proteins (threshold={COMPLETENESS_THRESHOLD:.0%})")
print(f"  After outlier removal: {quant_bio_clean.shape[0]} samples")
print(f"  Final:                 {quant_bio_clean.shape[0]} samples × {quant_bio_clean.shape[1]} proteins")\
"""))

# ── 40. Exercises ─────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Extended Exercises

### Exercise A — Threshold sensitivity
Change `COMPLETENESS_THRESHOLD` to 0.05, 0.50, and 1.0.
How many proteins remain at each threshold?
At 100 % completeness, how many proteins do you get? Why is this rarely used?

### Exercise B — Outlier threshold sensitivity
Change `N_METHODS_THRESHOLD` to 1 and 5.
How many samples are removed at each setting?
What is the risk of being too permissive (threshold=1) vs. too strict (threshold=5)?

### Exercise C — Batch correction comparison
Run the PC × factor heatmap on the `quant_combat` and `quant_pm` matrices.
Does the plate association drop below significance (−log10 p < 1.3) after ComBat?
Does group, sex, or age association increase relative to plate after correction?

### Exercise D — QC sample clustering
In the t-SNE plot, are all 12 QC stars (★) close together, or do they separate by plate?
What does tight clustering of QC samples across plates tell you about batch correction quality?

### Exercise E — Your own data
Replace `demo_parquet` with a path to your own Spectronaut export.
Ensure:
1. The parquet has `PG_ProteinAccessions` as the first column
2. Sample columns contain log2 intensities (Spectronaut default)
3. A metadata CSV with `plate` and `is_qc` columns exists

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
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out = Path("notebooks/proteomics_qc_tutorial.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({len(cells)} cells)")
