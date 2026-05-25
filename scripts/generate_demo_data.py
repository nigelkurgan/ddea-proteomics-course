"""
Generate synthetic proteomics demo data for the QC teaching pipeline.

Dataset properties:
  - 160 biological samples across 4 plates (40/plate)
  - 12 pooled QC samples (3 per plate) — identical biological reference
  - 2 groups (Case / Control, balanced per plate)
  - Sex (Male / Female, balanced) and Age (40–65 years) as covariates
  - Strong plate batch effects for teaching batch-correction
  - 4 outlier samples — each primarily caught by a different detection method
  - MNAR missing values (low-abundance proteins more likely absent)

Outlier design (pedagogical):
  Outlier A — failed injection: 85% proteins missing → caught mainly by Missing rate
  Outlier B — degraded sample: global −5 log2 shift → caught mainly by Mean intensity + Density
  Outlier C — mislabelled/wrong sample: random Gaussian → caught mainly by PCA methods
  Outlier D — contamination: global +4.5 log2 shift → caught mainly by Mean intensity + Density

Note: The script does NOT print which samples are outliers — students discover them.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(
    n_proteins: int = 2000,
    n_bio_per_plate: int = 40,
    n_plates: int = 4,
    n_qc_per_plate: int = 3,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bio = n_bio_per_plate * n_plates          # 160
    n_qc  = n_qc_per_plate  * n_plates          # 12
    n_total = n_bio + n_qc                       # 172

    # ── Biological sample metadata ─────────────────────────────────────────────
    bio_ids, bio_plates, bio_groups = [], [], []
    bio_subjects, bio_sex, bio_age  = [], [], []

    subject_counter = 1
    for plate_i in range(1, n_plates + 1):
        for s_i in range(1, n_bio_per_plate + 1):
            bio_ids.append(f"BIO_{subject_counter:04d}_P{plate_i:02d}")
            bio_plates.append(f"Plate_{plate_i:02d}")
            bio_groups.append("Case" if subject_counter % 2 == 1 else "Control")
            bio_subjects.append(subject_counter)
            bio_sex.append("Male" if subject_counter % 2 == 1 else "Female")
            bio_age.append(float(rng.integers(40, 66)))
            subject_counter += 1

    # ── QC sample metadata ─────────────────────────────────────────────────────
    qc_ids, qc_plates = [], []
    for plate_i in range(1, n_plates + 1):
        for q_i in range(1, n_qc_per_plate + 1):
            qc_ids.append(f"QC_{plate_i:02d}_{q_i:02d}")
            qc_plates.append(f"Plate_{plate_i:02d}")

    # ── Protein metadata ───────────────────────────────────────────────────────
    protein_accs  = [f"P{i:05d}" for i in range(1, n_proteins + 1)]
    gene_names    = [f"GENE{i}"   for i in range(1, n_proteins + 1)]
    protein_names = [f"Protein {i}" for i in range(1, n_proteins + 1)]

    # Dynamic range: log2 intensities uniformly distributed 18–30
    protein_means = rng.uniform(18, 30, size=n_proteins)
    protein_sd    = rng.uniform(0.2, 0.7, size=n_proteins)

    # ── Biological intensity matrix ────────────────────────────────────────────
    bio_mat = (
        protein_means[:, np.newaxis]
        + rng.normal(0, protein_sd[:, np.newaxis], size=(n_proteins, n_bio))
    )

    # Strong plate batch effects — clearly visible before correction
    plate_shifts = {1: 0.0, 2: 1.3, 3: -1.0, 4: 1.6}
    bio_plate_num = [int(p.split("_")[1]) for p in bio_plates]
    bio_mat += np.array([plate_shifts[p] for p in bio_plate_num])[np.newaxis, :]

    # Differential expression: ~20 % of proteins (Case vs Control)
    de_idx = rng.choice(n_proteins, int(n_proteins * 0.20), replace=False)
    de_fc  = rng.choice([-1, 1], size=len(de_idx)) * rng.uniform(0.3, 0.8, size=len(de_idx))
    case_mask = np.array([1 if g == "Case" else 0 for g in bio_groups])
    for idx, fc in zip(de_idx, de_fc):
        bio_mat[idx] += fc * case_mask

    # Sex-biased proteins: ~5 % (some higher in females, some in males)
    sex_idx = rng.choice(n_proteins, int(n_proteins * 0.05), replace=False)
    sex_fc  = rng.choice([-1, 1], size=len(sex_idx)) * rng.uniform(0.5, 1.0, size=len(sex_idx))
    female_mask = np.array([1 if s == "Female" else 0 for s in bio_sex])
    for idx, fc in zip(sex_idx, sex_fc):
        bio_mat[idx] += fc * female_mask

    # Age-correlated proteins: ~5 %
    age_idx  = rng.choice(n_proteins, int(n_proteins * 0.05), replace=False)
    age_coef = rng.choice([-1, 1], size=len(age_idx)) * rng.uniform(0.025, 0.045, size=len(age_idx))
    ages_c   = np.array(bio_age) - np.mean(bio_age)       # centred
    for idx, coef in zip(age_idx, age_coef):
        bio_mat[idx] += coef * ages_c

    # MNAR missing values: low-abundance proteins missing more often
    detect_prob  = (protein_means - protein_means.min()) / (protein_means.max() - protein_means.min())
    missing_prob = np.clip(0.05 + (1 - detect_prob) * 0.65, 0.01, 0.80)
    bio_mat[rng.random(size=(n_proteins, n_bio)) < missing_prob[:, np.newaxis]] = np.nan

    # ── Inject 4 outliers (one per plate, silently) ────────────────────────────
    outlier_positions = {}    # plate → (local_sample_index, global_index, type)
    for plate_i, (otype, shift) in enumerate(
        [("missing", 0), ("shift_down", -5.0), ("random", 0), ("shift_up", 4.5)]
    ):
        local_idx = rng.integers(8, n_bio_per_plate - 8)   # avoid edges
        global_idx = plate_i * n_bio_per_plate + local_idx
        outlier_positions[plate_i + 1] = (local_idx, global_idx, otype)

        if otype == "missing":
            # Outlier A: 85 % of proteins randomly set to NaN
            # Mean of remaining 15 % is ~normal → NOT caught by mean intensity alone
            keep = rng.random(n_proteins) < 0.15
            bio_mat[:, global_idx] = np.where(keep,
                protein_means + rng.normal(0, protein_sd), np.nan)

        elif otype == "shift_down":
            # Outlier B: all intensities shifted down — degraded proteome
            bio_mat[:, global_idx] += shift   # shift is negative

        elif otype == "random":
            # Outlier C: random Gaussian, proteins uncorrelated with cohort
            # Mean ≈ global mean so NOT caught by mean intensity; caught by PCA
            bio_mat[:, global_idx] = rng.normal(np.nanmean(protein_means), 3.5, size=n_proteins)

        elif otype == "shift_up":
            # Outlier D: global upward shift — contamination
            bio_mat[:, global_idx] += shift

    # ── QC intensity matrix ────────────────────────────────────────────────────
    # Same pooled reference, technical noise only (SD=0.05 in log2 ≈ 3–4% CV)
    qc_ref = protein_means.copy()
    qc_mat = qc_ref[:, np.newaxis] + rng.normal(0, 0.05, size=(n_proteins, n_qc))

    # Apply same plate batch effects to QC samples
    qc_plate_num = [int(p.split("_")[1]) for p in qc_plates]
    qc_mat += np.array([plate_shifts[p] for p in qc_plate_num])[np.newaxis, :]

    # Very low missing rate for QC (pooled = enriched for all proteins)
    qc_mat[rng.random(size=(n_proteins, n_qc)) < (missing_prob[:, np.newaxis] * 0.08)] = np.nan

    # ── Assemble combined matrix ───────────────────────────────────────────────
    all_ids = bio_ids + qc_ids
    all_mat = np.concatenate([bio_mat, qc_mat], axis=1)   # proteins × (bio + qc)

    prot_df = pd.DataFrame(all_mat, index=protein_accs, columns=all_ids)
    prot_df.index.name = "PG_ProteinAccessions"
    prot_df.insert(0, "PG_Genes", gene_names)
    prot_df.insert(1, "PG_ProteinNames", protein_names)
    prot_df = prot_df.reset_index()

    # ── Sample metadata ────────────────────────────────────────────────────────
    bio_meta_df = pd.DataFrame({
        "sample_id":  bio_ids,
        "plate":      bio_plates,
        "group":      bio_groups,
        "subject_id": bio_subjects,
        "sex":        bio_sex,
        "age":        bio_age,
        "is_qc":      False,
    }).set_index("sample_id")

    qc_meta_df = pd.DataFrame({
        "sample_id":  qc_ids,
        "plate":      qc_plates,
        "group":      "QC",
        "subject_id": [np.nan] * n_qc,
        "sex":        [np.nan] * n_qc,
        "age":        [np.nan] * n_qc,
        "is_qc":      True,
    }).set_index("sample_id")

    meta_df = pd.concat([bio_meta_df, qc_meta_df])

    # ── Save ───────────────────────────────────────────────────────────────────
    parquet_path = output_dir / "demo_proteomics.parquet"
    meta_path    = output_dir / "demo_metadata.csv"
    prot_df.to_parquet(parquet_path, index=False)
    meta_df.to_csv(meta_path)

    print(f"  Proteins:           {n_proteins}")
    print(f"  Biological samples: {n_bio} ({n_bio_per_plate}/plate × {n_plates} plates)")
    print(f"  QC pooled samples:  {n_qc} ({n_qc_per_plate}/plate)")
    print(f"  Groups:             {bio_meta_df['group'].value_counts().to_dict()}")
    print(f"  Sex:                {bio_meta_df['sex'].value_counts().to_dict()}")
    print(f"  Age:                {bio_meta_df['age'].mean():.1f} ± {bio_meta_df['age'].std():.1f} yr")
    print(f"  Plate effects:      {plate_shifts}")
    print(f"  Biological missing: {np.isnan(bio_mat).mean():.1%}")
    print(f"  QC missing:         {np.isnan(qc_mat).mean():.1%}")
    print(f"  Saved: {parquet_path}")
    print(f"  Saved: {meta_path}")

    return prot_df, meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic proteomics demo data")
    parser.add_argument("--n-proteins",      type=int, default=2000)
    parser.add_argument("--n-bio-per-plate", type=int, default=40)
    parser.add_argument("--n-plates",        type=int, default=4)
    parser.add_argument("--n-qc-per-plate",  type=int, default=3)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output",          type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    generate_demo(
        n_proteins=args.n_proteins,
        n_bio_per_plate=args.n_bio_per_plate,
        n_plates=args.n_plates,
        n_qc_per_plate=args.n_qc_per_plate,
        seed=args.seed,
        output_dir=out,
    )
