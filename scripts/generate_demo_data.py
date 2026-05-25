"""
Generate synthetic proteomics demo data for the QC teaching pipeline.

Dataset properties:
  - 160 subjects across 4 plates (Latin square design — no plate×time or plate×subject confounding)
  - 3 time points per subject (T00, T01, T02) → 480 biological samples total
  - 12 pooled QC samples (3 per plate) — identical biological reference
  - 2 groups (Case / Control) with sex balanced WITHIN each group (no sex-group confound)
  - Age (40–65 years) as a continuous covariate
  - Strong within-subject protein correlation (~50 %) — time points cluster per subject in PCA
  - Protein-specific plate batch effects (survive median normalisation → visible in PC1/PC2)
  - 4 outlier samples — each primarily caught by a different detection method
  - MNAR missing values (low-abundance proteins more likely absent)

Latin square plate design (each plate has all time points and balanced groups/sex):
  Subject group A (40 subj): T00→Plate1, T01→Plate2, T02→Plate3
  Subject group B (40 subj): T00→Plate2, T01→Plate3, T02→Plate4
  Subject group C (40 subj): T00→Plate3, T01→Plate4, T02→Plate1
  Subject group D (40 subj): T00→Plate4, T01→Plate1, T02→Plate2
  Each plate receives: 40×T00 + 40×T01 + 40×T02 = 120 bio samples + 3 QC = 123 samples/plate.

Outlier design (pedagogical — NOT printed so students discover them):
  Outlier A — failed injection: 85 % proteins missing → caught mainly by Missing rate
  Outlier B — degraded sample: global −5 log2 shift → caught mainly by Mean intensity + Density
  Outlier C — mislabelled/wrong sample: random Gaussian → caught mainly by PCA methods
  Outlier D — contamination: global +4.5 log2 shift → caught mainly by Mean intensity + Density
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(
    n_proteins: int = 2000,
    n_subjects: int = 160,
    n_timepoints: int = 3,
    n_plates: int = 4,
    n_qc_per_plate: int = 3,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert n_subjects % n_plates == 0, "n_subjects must be divisible by n_plates"
    n_subj_per_group = n_subjects // n_plates   # 40 subjects per Latin square group
    n_bio = n_subjects * n_timepoints            # 480
    n_qc  = n_qc_per_plate * n_plates           # 12

    # ── Latin square plate assignment ─────────────────────────────────────────
    # Each row = one subject group; each column = one time point.
    # Plate numbers are 1-indexed.
    plate_ls = [
        [1, 2, 3],   # Group A: T00→P1, T01→P2, T02→P3
        [2, 3, 4],   # Group B: T00→P2, T01→P3, T02→P4
        [3, 4, 1],   # Group C: T00→P3, T01→P4, T02→P1
        [4, 1, 2],   # Group D: T00→P4, T01→P1, T02→P2
    ]

    # ── Subject metadata ──────────────────────────────────────────────────────
    # Sex is balanced INDEPENDENTLY from group within every subject group:
    #   each of the 4 Latin square groups has 10 CaseMale, 10 CaseFemale,
    #   10 ControlMale, 10 ControlFemale.
    subjects = []
    for grp_i in range(n_plates):
        for w in range(n_subj_per_group):
            q = n_subj_per_group // 4      # 10
            if   w < q:     condition, sex = "Case",    "Male"
            elif w < 2*q:   condition, sex = "Case",    "Female"
            elif w < 3*q:   condition, sex = "Control", "Male"
            else:           condition, sex = "Control", "Female"
            subjects.append({
                "subject_id":   grp_i * n_subj_per_group + w + 1,
                "plate_group":  grp_i,
                "group":        condition,
                "sex":          sex,
                "age":          float(rng.integers(40, 66)),
            })

    # ── Sample records (subject × time point) ────────────────────────────────
    bio_records = []
    for subj in subjects:
        for tp in range(n_timepoints):
            plate_num = plate_ls[subj["plate_group"]][tp]
            bio_records.append({
                "sample_id":  f"BIO_{subj['subject_id']:04d}_T{tp:02d}_P{plate_num:02d}",
                "subject_id": subj["subject_id"],
                "timepoint":  f"T{tp:02d}",
                "plate":      f"Plate_{plate_num:02d}",
                "group":      subj["group"],
                "sex":        subj["sex"],
                "age":        subj["age"],
                "is_qc":      False,
            })
    # Sort by plate so plate samples are contiguous in the matrix
    bio_records.sort(key=lambda r: (r["plate"], r["subject_id"], r["timepoint"]))

    bio_ids      = [r["sample_id"]                          for r in bio_records]
    bio_plate_num = [int(r["plate"].split("_")[1])          for r in bio_records]

    # ── QC sample records ─────────────────────────────────────────────────────
    qc_records = []
    for plate_i in range(1, n_plates + 1):
        for q_i in range(1, n_qc_per_plate + 1):
            qc_records.append({
                "sample_id":  f"QC_{plate_i:02d}_{q_i:02d}",
                "subject_id": np.nan,
                "timepoint":  np.nan,
                "plate":      f"Plate_{plate_i:02d}",
                "group":      "QC",
                "sex":        np.nan,
                "age":        np.nan,
                "is_qc":      True,
            })
    qc_ids       = [r["sample_id"]                for r in qc_records]
    qc_plate_num = [int(r["plate"].split("_")[1]) for r in qc_records]

    # ── Protein metadata ───────────────────────────────────────────────────────
    protein_accs  = [f"P{i:05d}" for i in range(1, n_proteins + 1)]
    gene_names    = [f"GENE{i}"   for i in range(1, n_proteins + 1)]
    protein_names = [f"Protein {i}" for i in range(1, n_proteins + 1)]

    protein_means = rng.uniform(18, 30, size=n_proteins)
    protein_sd    = rng.uniform(0.2, 0.7, size=n_proteins)  # measurement noise

    # ── Subject-specific protein profiles ────────────────────────────────────
    # Each subject has a personal "fingerprint" (fixed offset per protein).
    # This shared component is preserved across time points, creating the
    # within-subject correlation that makes time points cluster in PCA.
    sigma_subject = 0.8   # SD of subject fingerprint in log2 units
    subject_effects = rng.normal(0, sigma_subject, size=(n_proteins, n_subjects))

    # ── Biological intensity matrix (vectorised) ───────────────────────────────
    # For each sample s: intensity = global_mean + subject_fingerprint
    #                                + time_noise + measurement_noise
    # Within-subject correlation across proteins ≈ 0.8² / (0.8² + 0.6² + ~0.22) ≈ 0.52
    subject_idx_arr = np.array([r["subject_id"] - 1 for r in bio_records])  # 0-indexed

    bio_mat = (
        protein_means[:, np.newaxis]
        + subject_effects[:, subject_idx_arr]                           # (n_prot, n_bio)
        + rng.normal(0, 0.6, size=(n_proteins, n_bio))                 # time-specific noise
        + rng.normal(0, protein_sd[:, np.newaxis], size=(n_proteins, n_bio))  # measurement noise
    )

    # ── Protein-specific plate batch effects ──────────────────────────────────
    # IMPORTANT: a uniform shift per plate is removed by median normalisation.
    # To create a batch effect visible in PCA even after normalisation, we add a
    # PROTEIN-SPECIFIC component: each plate shifts proteins by different amounts.
    # This differential pattern cannot be removed by per-sample median centring.
    plate_global = {1: 0.0, 2: 0.3, 3: -0.5, 4: 0.7}   # small global component
    sigma_plate_protein = 1.5                              # large protein-specific component

    plate_effects = {}  # store for applying to QC samples
    for p in range(1, n_plates + 1):
        pe = plate_global[p] + rng.normal(0, sigma_plate_protein, size=n_proteins)
        plate_effects[p] = pe
        mask = np.array(bio_plate_num) == p
        bio_mat[:, mask] += pe[:, np.newaxis]

    # ── Biological variation: group, sex, age, longitudinal trend ─────────────
    case_mask   = np.array([1 if r["group"] == "Case"   else 0 for r in bio_records])
    female_mask = np.array([1 if r["sex"]   == "Female" else 0 for r in bio_records])
    ages        = np.array([r["age"]                            for r in bio_records])
    ages_c      = ages - np.nanmean(ages)   # centred
    tp_arr      = np.array([int(r["timepoint"][1:])             for r in bio_records])  # 0,1,2

    # Group differences: ~20 % of proteins (Case vs Control, log2FC 0.3–0.8)
    de_idx = rng.choice(n_proteins, int(n_proteins * 0.20), replace=False)
    de_fc  = rng.choice([-1, 1], len(de_idx)) * rng.uniform(0.3, 0.8, len(de_idx))
    for idx, fc in zip(de_idx, de_fc):
        bio_mat[idx] += fc * case_mask

    # Sex differences: ~5 % of proteins (log2FC 0.5–1.0)
    sex_idx = rng.choice(n_proteins, int(n_proteins * 0.05), replace=False)
    sex_fc  = rng.choice([-1, 1], len(sex_idx)) * rng.uniform(0.5, 1.0, len(sex_idx))
    for idx, fc in zip(sex_idx, sex_fc):
        bio_mat[idx] += fc * female_mask

    # Age-correlated proteins: ~5 %
    age_idx  = rng.choice(n_proteins, int(n_proteins * 0.05), replace=False)
    age_coef = rng.choice([-1, 1], len(age_idx)) * rng.uniform(0.025, 0.045, len(age_idx))
    for idx, coef in zip(age_idx, age_coef):
        bio_mat[idx] += coef * ages_c

    # Longitudinal trend: ~5 % of proteins change linearly over T00→T02
    long_idx  = rng.choice(n_proteins, int(n_proteins * 0.05), replace=False)
    long_coef = rng.choice([-1, 1], len(long_idx)) * rng.uniform(0.1, 0.3, len(long_idx))
    for idx, coef in zip(long_idx, long_coef):
        bio_mat[idx] += coef * tp_arr   # 0, 1, or 2 — linear slope per time point

    # ── MNAR missing values ───────────────────────────────────────────────────
    detect_prob  = (protein_means - protein_means.min()) / (protein_means.max() - protein_means.min())
    missing_prob = np.clip(0.05 + (1 - detect_prob) * 0.65, 0.01, 0.80)
    bio_mat[rng.random(size=(n_proteins, n_bio)) < missing_prob[:, np.newaxis]] = np.nan

    # ── Inject 4 outliers (one per plate, silently) ───────────────────────────
    for plate_i, (otype, shift) in enumerate(
        [("missing", 0), ("shift_down", -5.0), ("random", 0), ("shift_up", 4.5)]
    ):
        plate_num = plate_i + 1
        plate_indices = [i for i, pn in enumerate(bio_plate_num) if pn == plate_num]
        # pick a sample away from plate edges
        candidates = plate_indices[10:-10]
        global_idx = candidates[int(rng.integers(0, len(candidates)))]

        if otype == "missing":
            # Outlier A: 85 % proteins NaN; remaining 15 % ≈ normal mean → not caught by intensity alone
            keep = rng.random(n_proteins) < 0.15
            bio_mat[:, global_idx] = np.where(keep,
                protein_means + rng.normal(0, protein_sd), np.nan)
        elif otype == "shift_down":
            # Outlier B: degraded proteome — all proteins shifted down
            bio_mat[:, global_idx] += shift
        elif otype == "random":
            # Outlier C: random Gaussian — mean ≈ cohort mean so NOT caught by intensity;
            # completely uncorrelated with cohort → caught by PCA only
            bio_mat[:, global_idx] = rng.normal(np.nanmean(protein_means), 3.5, size=n_proteins)
        elif otype == "shift_up":
            # Outlier D: contamination — all proteins shifted up
            bio_mat[:, global_idx] += shift

    # ── QC intensity matrix ────────────────────────────────────────────────────
    # Same pooled reference for all QC samples; technical noise only (SD = 0.05 log2 ≈ 3–4 % CV).
    # Plate effects are applied so that inter-plate QC CV captures the batch effect.
    qc_ref = protein_means.copy()
    qc_mat = qc_ref[:, np.newaxis] + rng.normal(0, 0.05, size=(n_proteins, n_qc))
    for p in range(1, n_plates + 1):
        mask = np.array(qc_plate_num) == p
        qc_mat[:, mask] += plate_effects[p][:, np.newaxis]
    # Very low missing rate for pooled QC (enriched for all proteins)
    qc_mat[rng.random(size=(n_proteins, n_qc)) < (missing_prob[:, np.newaxis] * 0.08)] = np.nan

    # ── Assemble combined intensity matrix ────────────────────────────────────
    all_ids = bio_ids + qc_ids
    all_mat = np.concatenate([bio_mat, qc_mat], axis=1)   # proteins × (bio + qc)

    prot_df = pd.DataFrame(all_mat, index=protein_accs, columns=all_ids)
    prot_df.index.name = "PG_ProteinAccessions"
    prot_df.insert(0, "PG_Genes", gene_names)
    prot_df.insert(1, "PG_ProteinNames", protein_names)
    prot_df = prot_df.reset_index()

    # ── Sample metadata ────────────────────────────────────────────────────────
    bio_meta_df = pd.DataFrame(bio_records).set_index("sample_id")
    qc_meta_df  = pd.DataFrame(qc_records).set_index("sample_id")
    meta_df = pd.concat([bio_meta_df, qc_meta_df])

    # ── Save ───────────────────────────────────────────────────────────────────
    parquet_path = output_dir / "demo_proteomics.parquet"
    meta_path    = output_dir / "demo_metadata.csv"
    prot_df.to_parquet(parquet_path, index=False)
    meta_df.to_csv(meta_path)

    print(f"  Proteins:            {n_proteins}")
    print(f"  Subjects:            {n_subjects} ({n_subj_per_group}/Latin-square group)")
    print(f"  Time points:         {n_timepoints} (T00, T01, T02)")
    print(f"  Biological samples:  {n_bio} ({n_bio // n_plates}/plate × {n_plates} plates)")
    print(f"  QC pooled samples:   {n_qc} ({n_qc_per_plate}/plate)")
    gv = bio_meta_df.groupby(["group", "sex"]).size()
    print(f"  Group × Sex balance: {gv.to_dict()}")
    print(f"  Age:                 {bio_meta_df['age'].mean():.1f} ± {bio_meta_df['age'].std():.1f} yr")
    print(f"  Plate effects (σ_protein = {sigma_plate_protein}): {plate_global}")
    print(f"  Biological missing:  {np.isnan(bio_mat).mean():.1%}")
    print(f"  QC missing:          {np.isnan(qc_mat).mean():.1%}")
    print(f"  Saved: {parquet_path}")
    print(f"  Saved: {meta_path}")

    return prot_df, meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic proteomics demo data")
    parser.add_argument("--n-proteins",      type=int, default=2000)
    parser.add_argument("--n-subjects",      type=int, default=160)
    parser.add_argument("--n-timepoints",    type=int, default=3)
    parser.add_argument("--n-plates",        type=int, default=4)
    parser.add_argument("--n-qc-per-plate",  type=int, default=3)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output",          type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    generate_demo(
        n_proteins=args.n_proteins,
        n_subjects=args.n_subjects,
        n_timepoints=args.n_timepoints,
        n_plates=args.n_plates,
        n_qc_per_plate=args.n_qc_per_plate,
        seed=args.seed,
        output_dir=out,
    )
