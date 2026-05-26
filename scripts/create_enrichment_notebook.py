"""
Build the proteomics enrichment analysis tutorial notebook.
Run from the repo root: python scripts/create_enrichment_notebook.py
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
# Proteomics Enrichment Analysis — Insulin Sensitivity

**Course: Proteomics Data Analysis**
**Environment: Google Colab (or local Jupyter)**

---

## Learning Objectives

By the end of this notebook you will be able to:

1. Locate and load curated protein annotation resources (HPA, GTEx, HAtlas, Proteome-Phenome Atlas)
2. Understand how to build an appropriate background universe for enrichment analysis
3. Define biologically meaningful protein lists from a multi-workflow correlation study
4. Interpret pre- vs post-clamp plasma proteomics differences biologically
5. Run Fisher's exact test enrichment and apply Benjamini–Hochberg FDR correction
6. Distinguish proteins positively vs negatively correlated with insulin sensitivity
7. Visualise enrichment results and compare them across conditions

---

## Study Background

This notebook uses data from a multi-workflow plasma proteomics study of insulin sensitivity
(Deshmukh *et al.,* 2025, preprint: Research Square rs-8585654).

### The M Value — Gold-Standard Insulin Sensitivity

Insulin sensitivity was measured using the **hyperinsulinemic-euglycemic clamp**, the clinical
gold standard. During the clamp, insulin is infused intravenously while glucose is kept constant;
the glucose infusion rate required to maintain euglycemia is the **M value**:

$$M = \\frac{\\text{glucose infused (mg/kg/min)}}{\\text{body weight (kg)}}$$

A **high M value** = high insulin sensitivity (glucose disposal).
A **low M value** = insulin resistance (cells do not respond to insulin).

The clamp has two phases relevant to plasma proteomics:
- **Pre-clamp (baseline)**: fasting plasma sample before insulin infusion
- **Post-clamp**: plasma sample collected after sustained insulin stimulation

### Five Proteomic Workflows

The study applied five complementary plasma proteomics workflows, each revealing different
layers of the plasma proteome:

| Workflow | Sample preparation | Focus |
|----------|-------------------|-------|
| **MagNet** | Magnetic bead depletion | Mid-abundance proteins |
| **Neat** | No depletion (direct) | High-abundance proteins |
| **PCA** | Protein corona assembly | Low-abundance proteins |
| **Depleted** | Immunodepletion of top-14 | Mid-to-low abundance |
| **Olink** | Proximity extension assay | Targeted panels (~1500 proteins) |

Together these workflows measured **~8 000 unique proteins** in the same 161 participants.

### Annotation Resources Used

| Resource | What it measures | Reference |
|----------|-----------------|-----------|
| **HPA** Human Protein Atlas | Secretome location, tissue specificity | [proteinatlas.org](https://proteinatlas.org) |
| **GTEx** Tissue Expression Atlas | RNA tissue enrichment across 20 organs | [gtexportal.org](https://gtexportal.org) |
| **HAtlas** Blood/Tissue Proteome | Mass-spec tissue label from Human Cell Atlas | Jiang *et al.* 2020, *Nat Commun* |
| **Proteome-Phenome Atlas** | Hazard ratios for incident ICD-10 diseases (UK Biobank, n~53 000) | [Deng *et al.* 2025, *Cell*](https://doi.org/10.1016/j.cell.2024.10.045) — [proteome-phenome-atlas.com](https://proteome-phenome-atlas.com/) |\
"""))

# ── 1. Setup ──────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 0. Environment Setup\
"""))

cells.append(code("""\
import subprocess, sys

packages = [
    "pandas>=2.0", "numpy>=1.24", "matplotlib>=3.7",
    "scipy>=1.10", "scikit-learn>=1.3", "pyarrow>=12.0",
    "statsmodels>=0.14", "openpyxl>=3.1",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✓ Dependencies installed")

import os, urllib.request
if not os.path.exists("ddea-proteomics-course"):
    subprocess.check_call(["git", "clone", "-q",
        "https://github.com/nigelkurgan/ddea-proteomics-course.git"])
os.chdir("ddea-proteomics-course")
sys.path.insert(0, ".")
print("✓ Repository ready")

ppa_path = "data/ppa_sumstats_incident.csv"
if not os.path.exists(ppa_path):
    print("Downloading Proteome-Phenome Atlas summary statistics (~113 MB)...")
    url = "https://github.com/nigelkurgan/ddea-proteomics-course/releases/download/v1.0/ppa_sumstats_incident.csv"
    urllib.request.urlretrieve(url, ppa_path)
    print("✓ PPA data downloaded")
else:
    print("✓ PPA data already present")\
"""))

# ── 2. Imports ────────────────────────────────────────────────────────────────
cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.dpi"] = 110

from pathlib import Path
from statsmodels.stats.multitest import multipletests

from proteomics_qc.enrichment import (
    run_enrichment,
    load_gtex, load_hatlas, load_hpa_secretome, load_hpa_tissue_specificity, load_ppa,
    plot_enrichment_barplot, plot_enrichment_dotplot,
)
from proteomics_qc.plots.style import apply_style, PALETTE

apply_style()

DATA = Path("data")
print("✓ All imports successful")\
"""))

# ── 3. Load annotation resources ──────────────────────────────────────────────
cells.append(md("""\
---
## Section 1: Load Annotation Resources

We load all four external annotation resources before building the protein lists.\
"""))

cells.append(code("""\
# ── Human Protein Atlas ───────────────────────────────────────────────────────
hpa = pd.read_csv(DATA / "hpa_032026.tsv", sep="\\t", low_memory=False)
print(f"HPA: {hpa.shape[0]:,} proteins")

# ── GTEx tissue atlas ─────────────────────────────────────────────────────────
gtex_raw = pd.read_excel(DATA / "gtex_tissue_enrichment.xlsx")
print(f"GTEx: {gtex_raw['organ'].nunique()} organs, "
      f"{gtex_raw['enriched'].sum():,} tissue-enriched proteins")

# ── HAtlas blood/tissue proteome ──────────────────────────────────────────────
hatlas_raw = pd.read_excel(DATA / "hatlas.xlsx")
print(f"HAtlas: {hatlas_raw['global_label'].str.split('.').str[0].nunique()} primary tissues")

# ── Proteome-Phenome Atlas — incident disease ─────────────────────────────────
# Deng et al. (2025) Cell 188(1):253-271. https://doi.org/10.1016/j.cell.2024.10.045
# Download from: https://proteome-phenome-atlas.com/
ppa_raw = pd.read_csv(DATA / "ppa_sumstats_incident.csv")
print(f"Proteome-Phenome Atlas: {ppa_raw['Protein'].nunique():,} proteins, "
      f"{ppa_raw['Disease'].nunique()} diseases")\
"""))

# ── 4. Load study data ────────────────────────────────────────────────────────
cells.append(md("""\
---
## Section 2: Load Study Data and Build Protein Lists

### 2a. Background Universe

The background must include **every protein that was tested** across all five workflows —
not just the significant ones. This is critical for Fisher's exact test to be calibrated correctly.

> If we used only the 488 significant proteins as background, every enrichment test would
> trivially return OR = 1 because the background IS the signal.
> The correct background is all 8 000+ measured proteins.\
"""))

cells.append(code("""\
SUPP = DATA / "Supplementary_Table_02.xlsx"
xl   = pd.ExcelFile(SUPP)

WORKFLOW_SHEETS = ["MagNet", "Neat", "PCA", "Depleted", "Olink"]

# ── Build background: union of all proteins from all five workflow sheets ──────
bg_uniprots = set()
bg_genes     = set()
workflow_sizes = {}

for sheet in WORKFLOW_SHEETS:
    df = pd.read_excel(xl, sheet_name=sheet)
    bg_uniprots.update(df["Accession_ID"].dropna())
    bg_genes.update(df["Gene_ID"].dropna())
    workflow_sizes[sheet] = len(df)
    print(f"  {sheet:10s}: {len(df):5,} proteins")

print(f"\\nBackground universe:")
print(f"  Total unique UniProt IDs : {len(bg_uniprots):,}")
print(f"  Total unique gene names  : {len(bg_genes):,}")\
"""))

cells.append(code("""\
# ── Load the unified significant results ──────────────────────────────────────
sig = pd.read_excel(xl, sheet_name="All_unique_significant")

print(f"All_unique_significant: {len(sig)} proteins")
print()
print("Significant category breakdown:")
print(sig["Significant"].value_counts().to_string())
print()
print("Workflow of origin (best p-value):")
print(sig["Workflow"].value_counts().to_string())
print()
print("Direction of M-value association (pre-clamp estimate):")
n_pos = (sig["M-value_pre_est"] > 0).sum()
n_neg = (sig["M-value_pre_est"] < 0).sum()
print(f"  Positive (higher in insulin sensitive): {n_pos}")
print(f"  Negative (lower in insulin sensitive) : {n_neg}")\
"""))

# ── 5. Overview plots ─────────────────────────────────────────────────────────
cells.append(md("""\
### 2b. Overview of the 488 M-Value-Associated Proteins

Before enrichment, we visualise the structure of the significant protein set.
Three sub-groups reflect how the clamp changes the proteomic insulin sensitivity signature.\
"""))

cells.append(code("""\
# ── Figure 1: Pre vs Post clamp correlation estimates ─────────────────────────
# Each dot is one protein. The x-axis is the standardised beta coefficient
# for the pre-clamp M value association; y-axis is the post-clamp estimate.
# Points near the diagonal = consistent pre and post.
# Points off the diagonal = the association strengthens or disappears with insulin stimulation.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cat_colors = {
    "Both":      "#2166ac",   # blue — consistent across conditions
    "Pre_only":  "#d6604d",   # red — only at baseline
    "Post_only": "#4dac26",   # green — only after clamp
}
cat_labels = {
    "Both":      f"Both (n={( sig['Significant']=='Both').sum()})",
    "Pre_only":  f"Pre only (n={(sig['Significant']=='Pre_only').sum()})",
    "Post_only": f"Post only (n={(sig['Significant']=='Post_only').sum()})",
}

ax = axes[0]
for cat, color in cat_colors.items():
    sub = sig[sig["Significant"] == cat]
    ax.scatter(sub["M-value_pre_est"], sub["M-value_post_est"],
               color=color, alpha=0.65, s=18, label=cat_labels[cat], zorder=3)

lim = max(abs(sig["M-value_pre_est"]).max(), abs(sig["M-value_post_est"]).max()) * 1.05
ax.axhline(0, color="grey", lw=0.6, linestyle="--")
ax.axvline(0, color="grey", lw=0.6, linestyle="--")
ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, linestyle=":", alpha=0.5,
        label="Identity (pre = post)")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("M-value association (pre-clamp β)", fontsize=11)
ax.set_ylabel("M-value association (post-clamp β)", fontsize=11)
ax.set_title("Pre vs Post clamp: M-value correlation coefficients", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")

# ── Workflow composition bar chart ────────────────────────────────────────────
ax2 = axes[1]
wf_cat = (sig.groupby(["Workflow", "Significant"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Pre_only", "Both", "Post_only"], fill_value=0))

wf_cat.plot(kind="bar", stacked=True, ax=ax2,
            color=["#d6604d", "#2166ac", "#4dac26"],
            edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Workflow", fontsize=11)
ax2.set_ylabel("Number of significant proteins", fontsize=11)
ax2.set_title("Significant proteins by workflow and clamp timing", fontsize=11, fontweight="bold")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
ax2.legend(title="Category", fontsize=8)

plt.tight_layout()
plt.show()\
"""))

cells.append(code("""\
# ── Figure 2: Rank of M-value associations — top proteins pre and post clamp ──
# Sort by absolute beta and show the top proteins in each direction.
# Negative β = lower protein level → higher M value (protein declines with insulin resistance)
# Positive β = higher protein level → higher M value (protein rises with insulin sensitivity)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, col, label in [
    (axes[0], "M-value_pre_est",  "Pre-clamp"),
    (axes[1], "M-value_post_est", "Post-clamp"),
]:
    df = sig[["ID", col, "Significant"]].copy().dropna(subset=[col])
    top_neg = df.nsmallest(15, col)     # most negative (low in insulin sensitive)
    top_pos = df.nlargest(15, col)      # most positive (high in insulin sensitive)
    top = pd.concat([top_neg, top_pos]).drop_duplicates("ID")
    top = top.sort_values(col)

    colors = [cat_colors.get(c, "grey") for c in top["Significant"]]
    ax.barh(top["ID"], top[col], color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"M-value β ({label})", fontsize=10)
    ax.set_title(f"Top M-value associations — {label}", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

# Shared legend
handles = [mpatches.Patch(color=c, label=l)
           for c, l in zip(cat_colors.values(), cat_labels.values())]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
> **Exercise 2.1**: In the scatter plot, which proteins have the largest shift from pre to post?
> Do these make biological sense given the physiology of the insulin clamp?
> (Hint: post-clamp blood samples reflect the direct effect of ~2 hours of insulin infusion.)
>
> **Exercise 2.2**: Why does the `Post_only` group (green) appear mostly in the lower right?
> What biological process might explain proteins that only associate with M value after insulin
> stimulation but not at baseline?\
"""))

# ── 6. Define protein sets ────────────────────────────────────────────────────
cells.append(md("""\
### 2c. Define Protein Sets for Enrichment

We define six protein sets based on clamp timing and direction of association:\
"""))

cells.append(code("""\
# ── Protein sets by clamp timing ──────────────────────────────────────────────
pre_proteins  = set(sig.loc[sig["Significant"].isin(["Pre_only", "Both"]),  "UniProt_ID"])
post_proteins = set(sig.loc[sig["Significant"].isin(["Post_only", "Both"]), "UniProt_ID"])
both_proteins = set(sig.loc[sig["Significant"] == "Both", "UniProt_ID"])
pre_only_prot = set(sig.loc[sig["Significant"] == "Pre_only",  "UniProt_ID"])
post_only_prot= set(sig.loc[sig["Significant"] == "Post_only", "UniProt_ID"])

print("Protein sets by clamp timing:")
print(f"  Pre-clamp significant (Pre_only + Both)  : {len(pre_proteins)}")
print(f"  Post-clamp significant (Post_only + Both): {len(post_proteins)}")
print(f"  Both (consistent)                        : {len(both_proteins)}")
print(f"  Pre only (lost after clamp)              : {len(pre_only_prot)}")
print(f"  Post only (emerged after clamp)          : {len(post_only_prot)}")
print()

# ── Protein sets by direction of M-value association ──────────────────────────
# Using pre-clamp estimate; for Both proteins this is the baseline effect
pos_proteins = set(sig.loc[sig["M-value_pre_est"] > 0, "UniProt_ID"])   # high in insulin sensitive
neg_proteins = set(sig.loc[sig["M-value_pre_est"] < 0, "UniProt_ID"])   # low in insulin sensitive

# Gene-name equivalents (for PPA which uses gene names)
pre_genes  = set(sig.loc[sig["Significant"].isin(["Pre_only", "Both"]),  "ID"])
post_genes = set(sig.loc[sig["Significant"].isin(["Post_only", "Both"]), "ID"])
both_genes = set(sig.loc[sig["Significant"] == "Both", "ID"])
pos_genes  = set(sig.loc[sig["M-value_pre_est"] > 0, "ID"])
neg_genes  = set(sig.loc[sig["M-value_pre_est"] < 0, "ID"])

print("Protein sets by direction:")
print(f"  Positive β (high in insulin sensitive) : {len(pos_proteins)}")
print(f"  Negative β (low in insulin sensitive)  : {len(neg_proteins)}")
print()
print(f"Background universe (all 5 workflows): {len(bg_uniprots):,} UniProt IDs")\
"""))

# ── 7. Fisher's test theory ───────────────────────────────────────────────────
cells.append(md("""\
---
## Section 3: Fisher's Exact Test — Setup

For each annotation term (e.g., "Liver-enriched in GTEx") we build a 2×2 table:

```
                       In annotation   Not in annotation
                    ┌───────────────┬──────────────────┐
  In protein set    │  a  (hits)    │  b               │  = your protein set
                    ├───────────────┼──────────────────┤
  Not in set        │  c            │  d               │  = rest of background
                    └───────────────┴──────────────────┘
                       detected in       not in term
                       this tissue
```

- **OR = (a × d) / (b × c)** > 1 → enrichment; < 1 → depletion
- The background here is **all 8 000+ proteins measured** across all workflows

The `run_enrichment()` function handles this automatically once you pass the protein sets.\
"""))

cells.append(code("""\
from proteomics_qc.enrichment.fisher import run_enrichment
import inspect
# Show the function signature to reinforce the pattern
sig_lines = inspect.getsource(run_enrichment).split("\\n")
print("\\n".join(sig_lines[:30]))\
"""))

# ── 8. GTEx tissue enrichment ─────────────────────────────────────────────────
cells.append(md("""\
---
## Section 4: GTEx Tissue Enrichment

We ask: do the M-value-associated proteins originate preferentially from a specific tissue?
GTEx labels proteins as tissue-enriched if RNA expression is ≥ 4× the median across all tissues.

We test three protein sets: **Pre_only** (baseline associations), **Post_only** (clamp-specific),
and **Both** (consistent across conditions).\
"""))

cells.append(code("""\
# Build annotation dict restricted to our background
gtex_annot = load_gtex(DATA / "gtex_tissue_enrichment.xlsx", background=bg_uniprots)

print("GTEx organs with enriched proteins in background:")
for organ, prots in sorted(gtex_annot.items(), key=lambda x: -len(x[1])):
    print(f"  {organ:15s}: {len(prots):4d} proteins")\
"""))

cells.append(code("""\
# Run GTEx enrichment for all three timing groups
gtex_pre_only  = run_enrichment(pre_only_prot,  bg_uniprots, gtex_annot, min_overlap=2)
gtex_post_only = run_enrichment(post_only_prot, bg_uniprots, gtex_annot, min_overlap=2)
gtex_both      = run_enrichment(both_proteins,  bg_uniprots, gtex_annot, min_overlap=2)

for label, df in [("Pre_only", gtex_pre_only),
                  ("Post_only", gtex_post_only),
                  ("Both",      gtex_both)]:
    n_sig = (df["fdr_bh"] < 0.05).sum() if not df.empty else 0
    print(f"{label:12s}: {len(df)} terms tested, {n_sig} significant (FDR < 0.05)")\
"""))

cells.append(code("""\
# ── Side-by-side GTEx bar charts ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

panel_colors = {"Pre_only": "#d6604d", "Post_only": "#4dac26", "Both": "#2166ac"}

for ax, (label, df) in zip(axes, [("Pre_only",  gtex_pre_only),
                                   ("Post_only", gtex_post_only),
                                   ("Both",      gtex_both)]):
    if df.empty:
        ax.text(0.5, 0.5, "No terms tested", ha="center", transform=ax.transAxes)
        ax.set_title(label)
        continue
    plot_df = df.head(12).sort_values("odds_ratio", ascending=True)
    bar_colors = [panel_colors[label] if f < 0.05 else "lightgrey"
                  for f in plot_df["fdr_bh"]]
    ax.barh(plot_df["term"], plot_df["odds_ratio"], color=bar_colors, edgecolor="white")
    ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.5)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["odds_ratio"] + 0.05, i,
                f"n={int(row['n_overlap'])} FDR={row['fdr_bh']:.2g}",
                va="center", ha="left", fontsize=6.5)
    n_set = len({"Pre_only": pre_only_prot, "Post_only": post_only_prot,
                  "Both": both_proteins}.get(label, set()))
    ax.set_xlabel("Odds Ratio", fontsize=10)
    ax.set_title(f"GTEx — {label} (n={n_set})",
                 fontsize=10, fontweight="bold", color=panel_colors[label])

plt.suptitle("GTEx tissue enrichment by clamp timing category", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

cells.append(code("""\
# ── Direction-aware GTEx enrichment ──────────────────────────────────────────
# Positive β (high in insulin sensitive) vs negative β (insulin resistance marker)
gtex_pos = run_enrichment(pos_proteins, bg_uniprots, gtex_annot, min_overlap=2)
gtex_neg = run_enrichment(neg_proteins, bg_uniprots, gtex_annot, min_overlap=2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, df, label, color in [
    (axes[0], gtex_pos, f"High in insulin sensitive (positive β, n={len(pos_proteins)})", "#d73027"),
    (axes[1], gtex_neg, f"Low in insulin sensitive (negative β, n={len(neg_proteins)})",  "#4575b4"),
]:
    if df.empty:
        ax.text(0.5, 0.5, "No terms tested", ha="center", transform=ax.transAxes)
        continue
    plot_df = df.head(12).sort_values("odds_ratio", ascending=True)
    bar_colors = [color if f < 0.05 else "lightgrey" for f in plot_df["fdr_bh"]]
    ax.barh(plot_df["term"], plot_df["odds_ratio"], color=bar_colors, edgecolor="white")
    ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Odds Ratio", fontsize=10)
    ax.set_title(label, fontsize=10, fontweight="bold")

plt.suptitle("GTEx tissue enrichment by direction of M-value association", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("GTEx positive β:", gtex_pos[["term","n_overlap","odds_ratio","fdr_bh"]].head(8).to_string(index=False))
print()
print("GTEx negative β:", gtex_neg[["term","n_overlap","odds_ratio","fdr_bh"]].head(8).to_string(index=False))\
"""))

cells.append(md("""\
> **Exercise 4.1**: Do the Pre_only and Post_only groups enrich for different tissues?
> If so, what does this tell you about which organs respond to acute insulin stimulation?
>
> **Exercise 4.2**: Are proteins positively correlated with M value enriched in the same tissue
> as proteins negatively correlated? If they differ, what does this suggest about the tissue
> origins of insulin sensitivity vs insulin resistance markers?\
"""))

# ── 9. HAtlas enrichment ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Section 5: HAtlas Blood/Tissue Proteome Enrichment

HAtlas provides an independent, protein-level (not RNA-level) tissue assignment for plasma
proteins using mass-spectrometry proteome atlases. This validates or contrasts the GTEx findings.\
"""))

cells.append(code("""\
hatlas_annot = load_hatlas(DATA / "hatlas.xlsx", background=bg_uniprots, primary_only=True)

# Run for all three timing groups
hatlas_pre  = run_enrichment(pre_only_prot,  bg_uniprots, hatlas_annot, min_overlap=2)
hatlas_post = run_enrichment(post_only_prot, bg_uniprots, hatlas_annot, min_overlap=2)
hatlas_both = run_enrichment(both_proteins,  bg_uniprots, hatlas_annot, min_overlap=2)

# Direction-aware
hatlas_pos = run_enrichment(pos_proteins, bg_uniprots, hatlas_annot, min_overlap=2)
hatlas_neg = run_enrichment(neg_proteins, bg_uniprots, hatlas_annot, min_overlap=2)

print("HAtlas significant terms (FDR < 0.05):")
for label, df in [("Pre_only", hatlas_pre), ("Post_only", hatlas_post), ("Both", hatlas_both),
                  ("Positive β", hatlas_pos), ("Negative β", hatlas_neg)]:
    n_sig = (df["fdr_bh"] < 0.05).sum() if not df.empty else 0
    print(f"  {label:15s}: {n_sig} significant terms")\
"""))

cells.append(code("""\
# ── HAtlas comparison: timing vs direction ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

panels = [
    (axes[0, 0], hatlas_pre,  "Pre_only",   "#d6604d"),
    (axes[0, 1], hatlas_post, "Post_only",  "#4dac26"),
    (axes[0, 2], hatlas_both, "Both",        "#2166ac"),
    (axes[1, 0], hatlas_pos,  "High in insulin sensitive (pos β)", "#d73027"),
    (axes[1, 1], hatlas_neg,  "Low in insulin sensitive (neg β)",  "#4575b4"),
]

for ax, df, label, color in panels:
    if df.empty:
        ax.text(0.5, 0.5, "No results", ha="center", transform=ax.transAxes)
        ax.set_title(label)
        continue
    plot_df = df.head(12).sort_values("odds_ratio", ascending=True)
    bar_colors = [color if f < 0.05 else "lightgrey" for f in plot_df["fdr_bh"]]
    ax.barh(plot_df["term"], plot_df["odds_ratio"], color=bar_colors, edgecolor="white")
    ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Odds Ratio", fontsize=9)
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)
    ax.tick_params(axis="y", labelsize=7)

# Hide unused subplot
axes[1, 2].axis("off")

plt.suptitle("HAtlas tissue enrichment — clamp timing (top) and direction (bottom)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

# ── 10. HPA enrichment ────────────────────────────────────────────────────────
cells.append(md("""\
---
## Section 6: HPA Secretome Enrichment

The secretome location tells us how each protein reaches the plasma. For the **Both** set
(consistent pre and post), are these canonical blood proteins or do they originate from
specific secretory compartments?\
"""))

cells.append(code("""\
hpa_sec_annot  = load_hpa_secretome(DATA / "hpa_032026.tsv", background=bg_uniprots)
hpa_spec_annot = load_hpa_tissue_specificity(DATA / "hpa_032026.tsv", background=bg_uniprots)

# Run secretome enrichment for all groups
groups = {
    "Pre_only":  (pre_only_prot, "#d6604d"),
    "Post_only": (post_only_prot,"#4dac26"),
    "Both":      (both_proteins, "#2166ac"),
    "Pos β":     (pos_proteins,  "#d73027"),
    "Neg β":     (neg_proteins,  "#4575b4"),
}

sec_results  = {k: run_enrichment(v, bg_uniprots, hpa_sec_annot,  min_overlap=2) for k, (v, _) in groups.items()}
spec_results = {k: run_enrichment(v, bg_uniprots, hpa_spec_annot, min_overlap=2) for k, (v, _) in groups.items()}

print("HPA secretome location enrichment (significant terms FDR < 0.05):")
for k, df in sec_results.items():
    n_sig = (df["fdr_bh"] < 0.05).sum() if not df.empty else 0
    print(f"  {k:15s}: {n_sig} significant terms")

print()
print("HPA RNA tissue specificity enrichment:")
for k, df in spec_results.items():
    n_sig = (df["fdr_bh"] < 0.05).sum() if not df.empty else 0
    print(f"  {k:15s}: {n_sig} significant terms")\
"""))

cells.append(code("""\
# ── Secretome stacked bar comparing all five groups ───────────────────────────
# For each group: proportion of overlap proteins in each secretome category
# This shows which secretory compartment is most represented in each protein set.

sec_overlap = {}
for group_name, (protein_set, _) in groups.items():
    row = {}
    for term, annotated in hpa_sec_annot.items():
        overlap = protein_set & annotated
        row[term] = len(overlap)
    total = sum(row.values())
    sec_overlap[group_name] = {k: v / total * 100 if total > 0 else 0 for k, v in row.items()}

sec_pct = pd.DataFrame(sec_overlap).T

# Simplify long category names for display
rename_sec = {
    "Secreted to blood": "→ Blood",
    "Intracellular and membrane": "Intracell./membrane",
    "Secreted in other tissues": "→ Other tissues",
    "Secreted to extracellular matrix": "→ ECM",
    "Secreted - unknown location": "→ Unknown loc.",
    "Secreted in male reproductive system": "→ Male repro.",
    "Secreted in female reproductive system": "→ Female repro.",
    "Secreted to digestive system": "→ Digestive",
    "Secreted in brain": "→ Brain",
    "Immunoglobulin genes": "Immunoglobulin",
}
sec_pct.columns = [rename_sec.get(c, c) for c in sec_pct.columns]

# Only keep categories with any content
sec_pct = sec_pct.loc[:, (sec_pct > 0).any()]

fig, ax = plt.subplots(figsize=(12, 4))
sec_pct.plot(kind="bar", stacked=True, ax=ax,
             colormap="tab10", edgecolor="white", linewidth=0.4)
ax.set_ylabel("% of annotated overlap proteins", fontsize=10)
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.set_title("HPA secretome location — proportion by protein group", fontsize=11, fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()\
"""))

# ── 11. Proteome-Phenome Atlas enrichment ─────────────────────────────────────
cells.append(md("""\
---
## Section 7: Proteome-Phenome Atlas — Incident Disease Enrichment

The **Proteome-Phenome Atlas** (Sun et al., *Cell* 2024) maps ~3 000 plasma proteins to
hundreds of incident ICD-10 coded diseases in the UK Biobank (~50 000 participants) using
Cox proportional hazard models. Available at [proteome-phenome-atlas.com](https://proteome-phenome-atlas.com/).

We ask: do proteins associated with insulin sensitivity also carry known associations with
specific metabolic, cardiovascular, or other diseases?

> **Note**: The Atlas uses gene names as protein identifiers; our study uses UniProt IDs.
> The `ID` column in the significant sheet provides gene names for matching.\
"""))

cells.append(code("""\
# Build PPA annotation dicts restricted to background gene names
ppa_risk_annot = load_ppa(
    DATA / "ppa_sumstats_incident.csv",
    p_threshold=0.05,
    hr_direction="risk",
    min_cases=50,
    background_genes=bg_genes,
)
ppa_prot_annot = load_ppa(
    DATA / "ppa_sumstats_incident.csv",
    p_threshold=0.05,
    hr_direction="protective",
    min_cases=50,
    background_genes=bg_genes,
)

print(f"PPA risk associations: {len(ppa_risk_annot)} diseases with ≥1 annotated gene")
print(f"PPA protective associations: {len(ppa_prot_annot)} diseases with ≥1 annotated gene")

# Run enrichment for all groups on risk associations
ppa_results = {
    "Pre_only":  run_enrichment(pre_genes,    bg_genes, ppa_risk_annot, min_overlap=2),
    "Post_only": run_enrichment(post_genes,   bg_genes, ppa_risk_annot, min_overlap=2),
    "Both":      run_enrichment(both_genes,   bg_genes, ppa_risk_annot, min_overlap=2),
    "Pos β":     run_enrichment(pos_genes,    bg_genes, ppa_risk_annot, min_overlap=2),
    "Neg β":     run_enrichment(neg_genes,    bg_genes, ppa_risk_annot, min_overlap=2),
}

print("\\nPPA risk enrichment (significant FDR < 0.05):")
for k, df in ppa_results.items():
    n_sig = (df["fdr_bh"] < 0.05).sum() if not df.empty else 0
    print(f"  {k:15s}: {n_sig} significant diseases")\
"""))

cells.append(code("""\
# ── Disease category enrichment ───────────────────────────────────────────────
# Group diseases by ICD-10 chapter for higher-level enrichment
ppa_cat_df = pd.read_csv(DATA / "ppa_sumstats_incident.csv")
ppa_cat_df = ppa_cat_df[
    (ppa_cat_df["P_value"] < 0.05) &
    (ppa_cat_df["NB_case"] >= 50) &
    (ppa_cat_df["Protein"].isin(bg_genes))
].copy()
ppa_cat_df["HR"] = ppa_cat_df["HR[95%CI]"].str.extract(r"^([\\d.]+)").astype(float)
ppa_cat_df_risk = ppa_cat_df[ppa_cat_df["HR"] > 1]

def shorten_icd(s):
    s = s.replace("Chapter ", "")
    for kw in ["Diseases of the ", "Diseases of blood", "Certain "]:
        s = s.replace(kw, "")
    return s[:45].strip()

ppa_cat_annot = {
    shorten_icd(cat): set(grp["Protein"])
    for cat, grp in ppa_cat_df_risk.groupby("Disease_category")
}

cat_results = {
    "Pre_only":  run_enrichment(pre_genes,  bg_genes, ppa_cat_annot, min_overlap=2),
    "Post_only": run_enrichment(post_genes, bg_genes, ppa_cat_annot, min_overlap=2),
    "Both":      run_enrichment(both_genes, bg_genes, ppa_cat_annot, min_overlap=2),
    "Pos β":     run_enrichment(pos_genes,  bg_genes, ppa_cat_annot, min_overlap=2),
    "Neg β":     run_enrichment(neg_genes,  bg_genes, ppa_cat_annot, min_overlap=2),
}

# Plot the Both and direction groups side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (k, color) in zip(axes, [("Both", "#2166ac"), ("Pos β", "#d73027"), ("Neg β", "#4575b4")]):
    df = cat_results[k]
    if df.empty:
        ax.text(0.5, 0.5, "No results", ha="center", transform=ax.transAxes)
        continue
    plot_df = df.head(10).sort_values("odds_ratio", ascending=True)
    bar_colors = [color if f < 0.05 else "lightgrey" for f in plot_df["fdr_bh"]]
    ax.barh(plot_df["term"], plot_df["odds_ratio"], color=bar_colors, edgecolor="white")
    ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Odds Ratio", fontsize=9)
    ax.set_title(f"PPA disease category — {k}", fontsize=10, fontweight="bold", color=color)
    ax.tick_params(axis="y", labelsize=7)

plt.suptitle("Proteome-Phenome Atlas incident disease enrichment (risk associations)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
> **Exercise 7.1**: Do proteins in the **Both** group (consistent pre and post) enrich for
> metabolic diseases (endocrine/nutritional/metabolic chapter)?
> This would confirm that these proteins are truly connected to insulin resistance pathophysiology.
>
> **Exercise 7.2**: Do **Pos β** proteins (high in insulin sensitive) associate with protective
> disease outcomes? Compare `ppa_prot_annot` enrichment for pos vs neg proteins.
> (Replace `ppa_risk_annot` with `ppa_prot_annot` and re-run.)\
"""))

# ── 12. Cross-resource summary ────────────────────────────────────────────────
cells.append(md("""\
---
## Section 8: Cross-Resource Summary

A dot plot lets us compare enrichment across resources simultaneously.
Terms appearing in multiple resources with high significance give the strongest
biological evidence.\
"""))

cells.append(code("""\
# ── Summary dot plot: Both proteins across all resources ──────────────────────
both_results = {
    "GTEx tissue":      gtex_both,
    "HAtlas tissue":    hatlas_both,
    "HPA secretome":    sec_results["Both"],
    "HPA specificity":  spec_results["Both"],
    "PPA disease cat.": cat_results["Both"],
}

print("Enrichment summary — Both (pre+post consistent) proteins:")
for resource, df in both_results.items():
    if df.empty:
        print(f"  {resource}: no terms tested")
    else:
        n_sig = (df["fdr_bh"] < 0.05).sum()
        print(f"  {resource}: {len(df)} terms, {n_sig} significant (FDR < 0.05)")

plot_enrichment_dotplot(both_results, fdr_threshold=0.05, top_n=8)\
"""))

cells.append(code("""\
# ── Comparison dot plot: Pre_only vs Post_only across GTEx and HAtlas ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, (res_pre, res_post, atlas_label) in zip(axes, [
    (gtex_pre_only,  gtex_post_only,  "GTEx"),
    (hatlas_pre,     hatlas_post,     "HAtlas"),
]):
    # Collect significant terms from either group
    sig_terms = set()
    for df in [res_pre, res_post]:
        if not df.empty:
            sig_terms.update(df.loc[df["fdr_bh"] < 0.05, "term"])
    if not sig_terms:
        ax.text(0.5, 0.5, "No significant terms", ha="center", transform=ax.transAxes)
        ax.set_title(atlas_label)
        continue

    terms = sorted(sig_terms)
    pre_or  = {r["term"]: r["odds_ratio"] for _, r in res_pre.iterrows()}  if not res_pre.empty  else {}
    post_or = {r["term"]: r["odds_ratio"] for _, r in res_post.iterrows()} if not res_post.empty else {}
    pre_fdr  = {r["term"]: r["fdr_bh"]    for _, r in res_pre.iterrows()}  if not res_pre.empty  else {}
    post_fdr = {r["term"]: r["fdr_bh"]    for _, r in res_post.iterrows()} if not res_post.empty else {}

    y = np.arange(len(terms))
    pre_vals  = [pre_or.get(t,  1.0) for t in terms]
    post_vals = [post_or.get(t, 1.0) for t in terms]

    ax.barh(y - 0.2, pre_vals,  height=0.35, color="#d6604d", alpha=0.8, label="Pre_only")
    ax.barh(y + 0.2, post_vals, height=0.35, color="#4dac26", alpha=0.8, label="Post_only")
    ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlabel("Odds Ratio", fontsize=10)
    ax.set_title(f"{atlas_label}: Pre_only vs Post_only enrichment", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

plt.suptitle("Tissue enrichment comparison: Pre_only vs Post_only proteins",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()\
"""))

# ── 13. Exercises ──────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Extended Exercises

### Exercise A — Explore protective disease associations

Re-run the PPA enrichment using `ppa_prot_annot` (HR < 1, protective associations):
```python
ppa_prot_results = {
    "Both":  run_enrichment(both_genes, bg_genes, ppa_prot_annot, min_overlap=2),
    "Pos β": run_enrichment(pos_genes,  bg_genes, ppa_prot_annot, min_overlap=2),
    "Neg β": run_enrichment(neg_genes,  bg_genes, ppa_prot_annot, min_overlap=2),
}
```
Do proteins positively correlated with M value (insulin sensitive) associate with
*protection* from metabolic disease? Does this make biological sense?

### Exercise B — Both pre and post vs either condition

Compare enrichment for:
1. `both_proteins` (n=255) — robust, condition-independent associations
2. `pre_proteins` (n=365, Pre_only + Both) — all baseline associations
3. `post_proteins` (n=378, Post_only + Both) — all post-clamp associations

Does the enrichment change substantially when you add Pre_only or Post_only proteins to
the Both set?

### Exercise C — Workflow-specific enrichment

The `Workflow` column in the significant sheet tells which proteomic method found each
protein. Do proteins found uniquely by Olink enrich for different tissues than those
found by mass-spectrometry workflows (MagNet, Neat, PCA, Depleted)?

```python
for workflow in sig["Workflow"].unique():
    prot_set = set(sig.loc[sig["Workflow"] == workflow, "UniProt_ID"])
    gtex_wf  = run_enrichment(prot_set, bg_uniprots, gtex_annot, min_overlap=2)
    print(f"{workflow}: top term = "
          f"{gtex_wf.iloc[0]['term'] if not gtex_wf.empty else 'none'}")
```

### Exercise D — Concordance between GTEx and HAtlas

Proteins identified as liver-enriched in GTEx should ideally also have a liver label in HAtlas.
Count how many proteins in each of the three timing groups are:
- GTEx liver-enriched AND HAtlas liver-labelled
- Only GTEx liver-enriched
- Only HAtlas liver-labelled

Use `overlap_ids` from the enrichment results to extract these proteins and then map
back to gene names using the study data (`sig[["UniProt_ID", "ID"]]`).

### Exercise E — False discovery rate sensitivity

Change the FDR threshold from 0.05 to 0.1 and 0.01 for the GTEx enrichment.
How does the number of significant tissues change? Discuss the trade-off between
sensitivity (catching real tissue signals) and specificity (avoiding noise).

---
*Pipeline source code: github.com/nigelkurgan/ddea-proteomics-course*
*Study preprint: Deshmukh et al. (2025), Research Square rs-8585654*\
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

out = Path("notebooks/proteomics_enrichment_tutorial.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({len(cells)} cells)")
