# Plasma Proteomics — QC and Enrichment Analysis Course

A hands-on teaching resource for plasma proteomics data analysis, covering quality control
of DIA mass-spectrometry data through to biological interpretation via enrichment analysis.

---

## Notebooks

### 1. Proteomics QC Pipeline

Step-by-step quality control of a multi-plate DIA plasma proteomics experiment —
from raw intensities to a batch-corrected, analysis-ready matrix.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nigelkurgan/ddea-proteomics-course/blob/main/notebooks/proteomics_qc_tutorial.ipynb)

| Stage | What | Why |
|-------|------|-----|
| Blood contamination | Erythrocyte & platelet marker scores | Remove samples with cell lysis |
| Missed cleavages | Per-sample MC rate from peptide data | Flag poor trypsin digestion |
| Missing values | Heatmap + group-level completeness | Spot failed injections |
| Filtering | Protein completeness threshold | Discard noisy low-abundance proteins |
| Normalisation | Median scaling (shift + MAV) | Remove per-sample loading variation |
| Outlier detection | Z-score, PCA, KDE + KS confirmation | Identify technical outlier samples |
| Batch QC | PCA, within/between distances, PC×factor | Diagnose plate batch effects |
| Batch correction | Plate-median and ComBat | Remove systematic plate biases |
| CV analysis | Intra-plate, inter-plate, within/between subject | Assess assay reproducibility |
| Output | Batch-corrected parquet + QC report | Analysis-ready data |

---

### 2. Enrichment Analysis

Biological interpretation of plasma proteomic associations with insulin sensitivity (M value)
from a multi-workflow study. Covers Fisher's exact test enrichment using four curated
annotation resources.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nigelkurgan/ddea-proteomics-course/blob/main/notebooks/proteomics_enrichment_tutorial.ipynb)

| Resource | What it answers |
|----------|----------------|
| **GTEx** tissue atlas | Which organ expresses these proteins? (RNA-level) |
| **HAtlas** blood proteome | Which tissue do these proteins originate from? (protein-level) |
| **HPA** secretome | How do these proteins reach the plasma? |
| **Proteome-Phenome Atlas** | Are these proteins known disease risk/protective factors? |

The notebook uses data from a multi-workflow plasma proteomics study of insulin sensitivity
(hyperinsulinemic-euglycemic clamp, n = 161) and demonstrates enrichment across three
protein groups: consistent pre- and post-clamp associations, baseline-only, and
clamp-stimulated-only.

All data, including the Proteome-Phenome Atlas summary statistics, are downloaded
automatically when running in Colab. *Citation: Deng YT et al. (2025) Atlas of the plasma
proteome in health and disease in 53,026 adults. Cell 188(1):253–271.
https://doi.org/10.1016/j.cell.2024.10.045*

---

## Quick Start — Local

```bash
git clone https://github.com/nigelkurgan/ddea-proteomics-course.git
cd ddea-proteomics-course

# conda
conda env create -f environment.yml
conda activate proteomics_qc

# or pip
pip install -r requirements.txt
```

Regenerate the notebooks from source:

```bash
python scripts/generate_demo_data.py          # synthetic QC demo dataset
python scripts/create_notebook.py             # QC tutorial notebook
python scripts/create_enrichment_notebook.py  # enrichment tutorial notebook
```

Run notebooks locally:

```bash
jupyter notebook notebooks/proteomics_qc_tutorial.ipynb
jupyter notebook notebooks/proteomics_enrichment_tutorial.ipynb
```

---

## Repository Structure

```
ddea-proteomics-course/
├── notebooks/
│   ├── proteomics_qc_tutorial.ipynb          # QC pipeline (Colab-ready)
│   └── proteomics_enrichment_tutorial.ipynb  # Enrichment analysis (Colab-ready)
├── proteomics_qc/                            # Python package
│   ├── proteomics/
│   │   ├── filters.py          # Completeness filtering
│   │   ├── normalise.py        # Median scaling, imputation, batch correction
│   │   ├── outliers.py         # Multi-method outlier detection + KS confirmation
│   │   ├── batch.py            # Plate distance stats, PC×factor associations
│   │   ├── blood_contamination.py  # Erythrocyte / platelet marker scoring
│   │   └── missed_cleavages.py     # Per-sample MC rate from peptide data
│   ├── enrichment/
│   │   ├── fisher.py           # Fisher's exact test enrichment (BH FDR)
│   │   ├── resources.py        # Loaders for GTEx, HAtlas, HPA, Olink PPA
│   │   └── plots.py            # Enrichment bar charts and dot plots
│   └── plots/
│       ├── distribution.py     # Boxplot, density, rank-abundance
│       ├── missing.py          # Missing value heatmap, threshold plot
│       ├── batch_effects.py    # PCA, correlation heatmap, distance violin
│       └── cv.py               # CV violin, intra/inter comparison
├── scripts/
│   ├── generate_demo_data.py          # Synthetic QC demo data generator
│   ├── create_notebook.py             # Builds proteomics_qc_tutorial.ipynb
│   └── create_enrichment_notebook.py  # Builds proteomics_enrichment_tutorial.ipynb
├── data/
│   ├── demo/                          # Generated demo dataset (parquet)
│   ├── hpa_032026.tsv                 # Human Protein Atlas (March 2026)
│   ├── gtex_tissue_enrichment.xlsx    # GTEx tissue enrichment labels
│   ├── hatlas.xlsx                    # Human Cell Atlas blood proteome atlas
│   └── ppa_sumstats_incident.csv      # Olink PPA incident disease associations
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| matplotlib, scipy | Plotting and statistics |
| scikit-learn | PCA, KNN imputation |
| pyarrow | Parquet file I/O |
| statsmodels | BH FDR correction |
| openpyxl | Excel file I/O |
| inmoose | ComBat batch correction |

---

*Developed for the CBMR Proteomics Course. Contact: nigel.kurgan@sund.ku.dk*
