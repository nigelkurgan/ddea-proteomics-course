# Proteomics QC Teaching Pipeline

A portable, well-documented quality-control pipeline for DIA (data-independent acquisition) proteomics data, designed as a teaching resource for proteomics courses.

## Quick Start — Google Colab

Click the badge to open the tutorial notebook directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nigelkurgan/dea-proteomics-course/blob/main/notebooks/proteomics_qc_tutorial.ipynb)

No installation required — all dependencies are installed in the first cell.

---

## What does this pipeline do?

| Stage | What | Why |
|-------|------|-----|
| 1. Load | Read (proteins × samples) parquet | Spectronaut / DIA-NN output format |
| 2. Missing QC | Heatmap + per-group missing rates | Spot failed injections and structured missingness |
| 3. Filtering | Remove proteins below detection threshold | Discard noisy low-abundance proteins |
| 4. Normalisation | Median scaling (shift + MAV) | Remove loading and dynamic range variation |
| 5. Outlier detection | Z-score, PCA, KDE + KS confirmation | Identify and exclude technical outlier samples |
| 6. Batch QC | PCA, within/between distances, PC×factor | Diagnose and correct plate batch effects |
| 7. CV analysis | Overall, intra-, inter-individual CV | Assess assay reproducibility |
| 8. Output | Batch-corrected parquet + QC JSON | Analysis-ready data for downstream statistics |

---

## Installation (local)

```bash
git clone https://github.com/nigelkurgan/dea-proteomics-course.git
cd dea-proteomics-course

# Option A: conda
conda env create -f environment.yml
conda activate proteomics_qc

# Option B: pip
pip install -r requirements.txt
```

## Generate demo data and run the pipeline

```bash
# Generate synthetic demo dataset (2000 proteins, 160 samples, 4 plates)
python scripts/generate_demo_data.py

# Run the full QC pipeline on the demo data
python -m proteomics_qc.run_qc

# Run on your own data
python -m proteomics_qc.run_qc \
    --protein-matrix /path/to/your/proteomics_protein.parquet \
    --sample-metadata /path/to/metadata.csv \
    --completeness 0.20 \
    --batch-correction auto
```

Results are written to `results/qc/`.

## Run the tutorial notebook locally

```bash
jupyter notebook notebooks/proteomics_qc_tutorial.ipynb
```

---

## Input data format

The pipeline expects a **proteins × samples parquet** file:

- Columns starting with `PG_` are protein metadata (`PG_ProteinAccessions`, `PG_Genes`, etc.)
- All other columns are sample intensities (log2 scale)
- Missing values are `NaN` (not zero)

This matches the default Spectronaut protein-level export format.

An optional **sample metadata CSV** should have samples as the row index and at least a `plate` column.

---

## Applying to a new project (fpm_soup example)

```python
from proteomics_qc.config import QCConfig
from proteomics_qc.run_qc import run_pipeline

cfg = QCConfig()
cfg.protein_matrix = "/path/to/fpm_soup/neat.parquet"
cfg.sample_metadata = "/path/to/fpm_soup/metadata.csv"
cfg.output_dir = "/path/to/results/fpm_soup_qc"
cfg.batch_correction_method = "auto"

run_pipeline(cfg)
```

---

## Repository structure

```
dea-proteomics-course/
├── proteomics_qc/              # Python package
│   ├── config.py               # Configurable parameters
│   ├── run_qc.py               # Main pipeline entry point
│   ├── proteomics/
│   │   ├── filters.py          # Completeness filtering
│   │   ├── normalise.py        # Median scaling, quantile norm, imputation
│   │   ├── outliers.py         # Multi-method outlier detection + KS confirmation
│   │   └── batch.py            # Plate distance stats, PC×factor associations
│   └── plots/
│       ├── distribution.py     # Boxplot, density, rank-abundance
│       ├── missing.py          # Missing value heatmap, threshold plot
│       ├── batch_effects.py    # PCA, correlation heatmap, distance violin
│       └── cv.py               # CV violin, intra/inter comparison
├── notebooks/
│   └── proteomics_qc_tutorial.ipynb   # Colab-ready teaching notebook
├── scripts/
│   └── generate_demo_data.py          # Synthetic data generator
├── data/
│   └── demo/                          # Generated demo dataset
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
| inmoose | ComBat batch correction |

---

*Developed for the CBMR Proteomics Course. Contact: nigel.kurgan@sund.ku.dk*
