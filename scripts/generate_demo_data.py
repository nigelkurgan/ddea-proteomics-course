"""
Generate synthetic proteomics demo data for the QC teaching pipeline.

Dataset properties:
  - 160 subjects across 4 plates (Latin square design — no plate×time or plate×subject confounding)
  - 3 time points per subject (T00, T01, T02) → 480 biological samples total
  - 12 pooled QC samples (3 per plate) — identical biological reference
  - 2 groups (Case / Control) with sex balanced WITHIN each group
  - Age (40–65 years) as a continuous covariate
  - High within-subject protein correlation (~50 %) — time points cluster per subject in PCA
  - Protein-specific plate batch effects (survive median normalisation → visible in PC1/PC2/PC3)
  - 4 technical outlier samples — each primarily caught by a different detection method
  - 3 blood-contaminated samples — caught by the blood contamination panel only:
      2 with platelet contamination (elevated TLN1, MYH9, TPM4)
      1 with erythrocyte contamination / haemolysis (elevated HBA1, HBB, CA1, HBD, …)
  - MNAR missing values (low-abundance proteins more likely absent)
  - First 14 proteins are real blood contamination marker accessions (from Geyer et al. 2019)

Latin square plate design (each plate has all time points and balanced groups/sex):
  Subject group A (40 subj): T00→Plate1, T01→Plate2, T02→Plate3
  Subject group B (40 subj): T00→Plate2, T01→Plate3, T02→Plate4
  Subject group C (40 subj): T00→Plate3, T01→Plate4, T02→Plate1
  Subject group D (40 subj): T00→Plate4, T01→Plate1, T02→Plate2

Outlier design (pedagogical — NOT printed; students discover them):
  Technical outlier A — failed injection: 85 % proteins missing → Missing rate
  Technical outlier B — degraded sample:  global −5 log2 shift   → Mean + Density
  Technical outlier C — mislabelled:      random Gaussian         → PCA only
  Technical outlier D — contamination:    global +4.5 log2 shift  → Mean + Density
  Blood contamination 1 & 2 — platelet:   TLN1/MYH9/TPM4 elevated → contamination panel
  Blood contamination 3    — haemolysis:  HBA1/HBB/CA1/… elevated → contamination panel
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ── Blood contamination marker proteins (real UniProt accessions) ─────────────
# These proteins are the "best markers" from the Geyer et al. 2019 panel.
# They are added as the FIRST proteins in the matrix so the panel can match them.
CONTAM_MARKERS = [
    # (accession,  gene,    protein_name,                    marker_type)
    # Erythrocyte markers — indicate haemolysis (red blood cell lysis)
    ("P69905", "HBA1",  "Hemoglobin subunit alpha",         "erythrocyte"),
    ("P68871", "HBB",   "Hemoglobin subunit beta",          "erythrocyte"),
    ("P00915", "CA1",   "Carbonic anhydrase 1",             "erythrocyte"),
    ("P02042", "HBD",   "Hemoglobin subunit delta",         "erythrocyte"),
    ("P32119", "PRDX2", "Peroxiredoxin-2",                  "erythrocyte"),
    ("P00918", "CA2",   "Carbonic anhydrase 2",             "erythrocyte"),
    ("P04040", "CAT",   "Catalase",                         "erythrocyte"),
    ("P30043", "BLVRB", "Flavin reductase (NADPH)",         "erythrocyte"),
    # Platelet markers — indicate insufficient platelet depletion during plasma prep
    ("Q9Y490", "TLN1",  "Talin-1",                          "platelet"),
    ("P35579", "MYH9",  "Myosin-9",                         "platelet"),
    ("P67936", "TPM4",  "Tropomyosin alpha-4 chain",        "platelet"),
    # Coagulation factors — normally present in plasma; serve as internal reference
    ("P02675", "FGB",   "Fibrinogen beta chain",            "coagulation"),
    ("P02679", "FGG",   "Fibrinogen gamma chain",           "coagulation"),
    ("P02671", "FGA",   "Fibrinogen alpha chain",           "coagulation"),
]
N_CONTAM = len(CONTAM_MARKERS)   # 14


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
    n_subj_per_group = n_subjects // n_plates    # 40
    n_bio            = n_subjects * n_timepoints  # 480
    n_qc             = n_qc_per_plate * n_plates  # 12
    n_synthetic      = n_proteins - N_CONTAM      # 1986 synthetic proteins

    # ── Latin square plate assignment ─────────────────────────────────────────
    plate_ls = [
        [1, 2, 3],   # Group A: T00→P1, T01→P2, T02→P3
        [2, 3, 4],   # Group B: T00→P2, T01→P3, T02→P4
        [3, 4, 1],   # Group C: T00→P3, T01→P4, T02→P1
        [4, 1, 2],   # Group D: T00→P4, T01→P1, T02→P2
    ]

    # ── Subject metadata — sex balanced within each group ─────────────────────
    subjects = []
    for grp_i in range(n_plates):
        for w in range(n_subj_per_group):
            q = n_subj_per_group // 4
            if   w < q:     condition, sex = "Case",    "Male"
            elif w < 2 * q: condition, sex = "Case",    "Female"
            elif w < 3 * q: condition, sex = "Control", "Male"
            else:           condition, sex = "Control", "Female"
            subjects.append({
                "subject_id":  grp_i * n_subj_per_group + w + 1,
                "plate_group": grp_i,
                "group":       condition,
                "sex":         sex,
                "age":         float(rng.integers(40, 66)),
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
    bio_records.sort(key=lambda r: (r["plate"], r["subject_id"], r["timepoint"]))

    bio_ids       = [r["sample_id"]                  for r in bio_records]
    bio_plate_num = [int(r["plate"].split("_")[1])   for r in bio_records]

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
    # First N_CONTAM proteins: real blood contamination marker accessions.
    # Remaining proteins: synthetic accessions starting after the marker block.
    contam_accs  = [m[0] for m in CONTAM_MARKERS]
    contam_genes = [m[1] for m in CONTAM_MARKERS]
    contam_names = [m[2] for m in CONTAM_MARKERS]

    # Load real protein accessions measured in the GDM cohort.
    # The reference CSV was pre-filtered to exclude any accession that appears in
    # the blood contamination panel, so panel matching works cleanly.
    ref_csv  = Path(__file__).resolve().parents[1] / "data" / "gdm_protein_reference.csv"
    ref_df   = pd.read_csv(ref_csv)
    synth_accs  = ref_df["accession"].tolist()[:n_synthetic]
    synth_genes = ref_df["gene"].tolist()[:n_synthetic]
    synth_names = ref_df["name"].tolist()[:n_synthetic]

    protein_accs  = contam_accs  + synth_accs
    gene_names    = contam_genes + synth_genes
    protein_names = contam_names + synth_names

    # ── Synthetic protein properties ───────────────────────────────────────────
    protein_means = rng.uniform(18, 30, size=n_synthetic)
    protein_sd    = rng.uniform(0.2, 0.7, size=n_synthetic)

    # ── Subject-specific protein profiles ─────────────────────────────────────
    # Each subject has a stable "fingerprint" (same across time points).
    # This creates high within-subject correlation so time points cluster in PCA.
    sigma_subject   = 0.8
    subject_effects = rng.normal(0, sigma_subject, size=(n_synthetic, n_subjects))

    # ── Biological intensity matrix for synthetic proteins (vectorised) ───────
    subject_idx_arr = np.array([r["subject_id"] - 1 for r in bio_records])

    bio_mat = (
        protein_means[:, np.newaxis]
        + subject_effects[:, subject_idx_arr]
        + rng.normal(0, 0.6,    size=(n_synthetic, n_bio))   # time-specific noise
        + rng.normal(0, protein_sd[:, np.newaxis], size=(n_synthetic, n_bio))
    )

    # ── Protein-specific plate batch effects ──────────────────────────────────
    # A uniform shift per plate is removed by median normalisation.
    # Protein-SPECIFIC effects persist after normalisation and drive PC1-PC3.
    plate_global = {1: 0.0, 2: 0.3, 3: -0.5, 4: 0.7}
    sigma_plate_protein = 1.5

    plate_effects: dict[int, np.ndarray] = {}
    for p in range(1, n_plates + 1):
        pe = plate_global[p] + rng.normal(0, sigma_plate_protein, size=n_synthetic)
        plate_effects[p] = pe
        mask = np.array(bio_plate_num) == p
        bio_mat[:, mask] += pe[:, np.newaxis]

    # ── Biological variation: group, sex, age, longitudinal trend ─────────────
    case_mask   = np.array([1 if r["group"] == "Case"   else 0 for r in bio_records])
    female_mask = np.array([1 if r["sex"]   == "Female" else 0 for r in bio_records])
    ages        = np.array([r["age"]                            for r in bio_records])
    ages_c      = ages - np.nanmean(ages)
    tp_arr      = np.array([int(r["timepoint"][1:])             for r in bio_records])

    de_idx = rng.choice(n_synthetic, int(n_synthetic * 0.20), replace=False)
    de_fc  = rng.choice([-1, 1], len(de_idx)) * rng.uniform(0.3, 0.8, len(de_idx))
    for idx, fc in zip(de_idx, de_fc):
        bio_mat[idx] += fc * case_mask

    sex_idx = rng.choice(n_synthetic, int(n_synthetic * 0.05), replace=False)
    sex_fc  = rng.choice([-1, 1], len(sex_idx)) * rng.uniform(0.5, 1.0, len(sex_idx))
    for idx, fc in zip(sex_idx, sex_fc):
        bio_mat[idx] += fc * female_mask

    age_idx  = rng.choice(n_synthetic, int(n_synthetic * 0.05), replace=False)
    age_coef = rng.choice([-1, 1], len(age_idx)) * rng.uniform(0.025, 0.045, len(age_idx))
    for idx, coef in zip(age_idx, age_coef):
        bio_mat[idx] += coef * ages_c

    long_idx  = rng.choice(n_synthetic, int(n_synthetic * 0.05), replace=False)
    long_coef = rng.choice([-1, 1], len(long_idx)) * rng.uniform(0.1, 0.3, len(long_idx))
    for idx, coef in zip(long_idx, long_coef):
        bio_mat[idx] += coef * tp_arr

    # ── MNAR missing values for synthetic proteins ────────────────────────────
    detect_prob  = (protein_means - protein_means.min()) / (protein_means.max() - protein_means.min())
    missing_prob = np.clip(0.05 + (1 - detect_prob) * 0.65, 0.01, 0.80)
    bio_mat[rng.random(size=(n_synthetic, n_bio)) < missing_prob[:, np.newaxis]] = np.nan

    # ── Blood contamination marker matrix ─────────────────────────────────────
    # Contamination markers are normally ABSENT from plasma.
    # Erythrocyte + platelet markers: mostly NaN; 4 % of samples have trace amounts.
    # Coagulation factors: genuinely present in plasma at detectable levels.
    contam_mat = np.full((N_CONTAM, n_bio), np.nan)

    for i, (acc, gene, name, mtype) in enumerate(CONTAM_MARKERS):
        if mtype == "coagulation":
            # Fibrinogen and related proteins are present in all plasma samples
            contam_mat[i] = rng.uniform(20, 26, size=n_bio)
        else:
            # Trace contamination in a small fraction of normal samples
            trace_mask = rng.random(n_bio) < 0.04
            contam_mat[i, trace_mask] = rng.uniform(15, 18, size=int(trace_mask.sum()))

    # ── Pre-select blood contamination sample positions ───────────────────────
    # Choose samples far from plate edges; avoid positions used for technical outliers later.
    # Contamination spans different plates to demonstrate it is plate-independent.

    def pick_from_plate(plate_num: int, exclude: set) -> int:
        idxs = [i for i, pn in enumerate(bio_plate_num) if pn == plate_num]
        cands = [i for i in idxs[8:-8] if i not in exclude]
        return int(cands[rng.integers(0, len(cands))])

    reserved: set[int] = set()

    # Two platelet-contaminated samples (plates 1 and 3)
    plt_idx1 = pick_from_plate(1, reserved); reserved.add(plt_idx1)
    plt_idx2 = pick_from_plate(3, reserved); reserved.add(plt_idx2)
    # One erythrocyte-contaminated sample (plate 2)
    ery_idx  = pick_from_plate(2, reserved); reserved.add(ery_idx)

    # Elevate platelet markers (TLN1, MYH9, TPM4 = indices 8, 9, 10)
    platelet_row_idx = [8, 9, 10]
    for s_idx in [plt_idx1, plt_idx2]:
        for r_idx in platelet_row_idx:
            contam_mat[r_idx, s_idx] = rng.uniform(24, 27)

    # Elevate erythrocyte markers (HBA1–BLVRB = indices 0–7)
    eryth_row_idx = list(range(8))
    for r_idx in eryth_row_idx:
        contam_mat[r_idx, ery_idx] = rng.uniform(25, 28)

    # ── Inject 4 technical outliers (one per plate, silently) ─────────────────
    for plate_i, (otype, shift) in enumerate(
        [("missing", 0), ("shift_down", -5.0), ("random", 0), ("shift_up", 4.5)]
    ):
        plate_num   = plate_i + 1
        plate_idxs  = [i for i, pn in enumerate(bio_plate_num) if pn == plate_num]
        # Avoid plate edges and already-reserved contamination positions
        candidates  = [i for i in plate_idxs[10:-10] if i not in reserved]
        global_idx  = int(candidates[rng.integers(0, len(candidates))])
        reserved.add(global_idx)   # prevent accidental overlap with later injections

        if otype == "missing":
            # 85 % of proteins NaN; remaining 15 % ≈ normal mean
            keep = rng.random(n_synthetic) < 0.15
            bio_mat[:, global_idx] = np.where(keep,
                protein_means + rng.normal(0, protein_sd), np.nan)
        elif otype == "shift_down":
            bio_mat[:, global_idx] += shift
        elif otype == "random":
            bio_mat[:, global_idx] = rng.normal(np.nanmean(protein_means), 3.5,
                                                  size=n_synthetic)
        elif otype == "shift_up":
            bio_mat[:, global_idx] += shift

    # ── Full biological matrix: [contam markers; synthetic proteins] ───────────
    full_bio_mat = np.concatenate([contam_mat, bio_mat], axis=0)  # (n_proteins, n_bio)

    # ── QC intensity matrix ────────────────────────────────────────────────────
    # Same pooled reference for all QC samples; technical noise only (SD = 0.05 log2).
    # Apply synthetic-protein plate effects to QC samples.
    qc_ref   = protein_means.copy()
    qc_synth = qc_ref[:, np.newaxis] + rng.normal(0, 0.05, size=(n_synthetic, n_qc))
    for p in range(1, n_plates + 1):
        mask = np.array(qc_plate_num) == p
        qc_synth[:, mask] += plate_effects[p][:, np.newaxis]
    qc_synth[rng.random(size=(n_synthetic, n_qc)) < (missing_prob[:, np.newaxis] * 0.08)] = np.nan

    # QC contamination markers: same as normal plasma (coagulation present, others absent)
    qc_contam = np.full((N_CONTAM, n_qc), np.nan)
    for i, (acc, gene, name, mtype) in enumerate(CONTAM_MARKERS):
        if mtype == "coagulation":
            qc_contam[i] = rng.uniform(20, 26, size=n_qc)

    full_qc_mat = np.concatenate([qc_contam, qc_synth], axis=0)

    # ── Assemble combined matrix ───────────────────────────────────────────────
    all_ids = bio_ids + qc_ids
    all_mat = np.concatenate([full_bio_mat, full_qc_mat], axis=1)

    prot_df = pd.DataFrame(all_mat, index=protein_accs, columns=all_ids)
    prot_df.index.name = "PG_ProteinAccessions"
    prot_df.insert(0, "PG_Genes", gene_names)
    prot_df.insert(1, "PG_ProteinNames", protein_names)
    prot_df = prot_df.reset_index()

    # ── Sample metadata ────────────────────────────────────────────────────────
    bio_meta_df = pd.DataFrame(bio_records).set_index("sample_id")
    qc_meta_df  = pd.DataFrame(qc_records).set_index("sample_id")
    meta_df     = pd.concat([bio_meta_df, qc_meta_df])

    # ── Peptide-level matrix with missed cleavage data ────────────────────────
    # This is a separate dataset used only for missed cleavage QC (Stage 3).
    # Each row is a unique peptide; columns are sample intensities.
    # Metadata columns mirror the Spectronaut export format.
    peptide_df, pd_bio_idx, pd_poor_idx = _generate_peptide_data(
        rng=rng,
        bio_ids=bio_ids,
        bio_plate_num=bio_plate_num,
        qc_ids=qc_ids,
        qc_plate_num=qc_plate_num,
        n_plates=n_plates,
        plate_effects=plate_effects,
        reserved=reserved,
        output_dir=output_dir,
    )

    # ── Save protein matrix and metadata ──────────────────────────────────────
    parquet_path = output_dir / "demo_proteomics.parquet"
    meta_path    = output_dir / "demo_metadata.csv"
    prot_df.to_parquet(parquet_path, index=False)
    meta_df.to_csv(meta_path)

    gv = bio_meta_df.groupby(["group", "sex"]).size()
    print(f"  Proteins:              {n_proteins} ({N_CONTAM} contamination markers + {n_synthetic} synthetic)")
    print(f"  Subjects:              {n_subjects} ({n_subj_per_group}/Latin-square group)")
    print(f"  Time points:           {n_timepoints} (T00, T01, T02)")
    print(f"  Biological samples:    {n_bio} ({n_bio // n_plates}/plate × {n_plates} plates)")
    print(f"  QC pooled samples:     {n_qc} ({n_qc_per_plate}/plate)")
    print(f"  Group × Sex balance:   {gv.to_dict()}")
    print(f"  Age:                   {bio_meta_df['age'].mean():.1f} ± {bio_meta_df['age'].std():.1f} yr")
    print(f"  Plate effects (σ={sigma_plate_protein}): {plate_global}")
    print(f"  Biological missing:    {np.isnan(full_bio_mat).mean():.1%}")
    print(f"  QC missing:            {np.isnan(full_qc_mat).mean():.1%}")
    print(f"  Blood-contaminated:    3 samples planted (2 platelet, 1 erythrocyte) — not revealed")
    print(f"  Technical outliers:    4 samples planted — not revealed")
    print(f"  Poor digestion:        2 samples planted (MC rate ~45–50 %) — not revealed")
    print(f"  Saved: {parquet_path}")
    print(f"  Saved: {meta_path}")

    return prot_df, meta_df


def _generate_peptide_data(
    rng:            np.random.Generator,
    bio_ids:        list[str],
    bio_plate_num:  list[int],
    qc_ids:         list[str],
    qc_plate_num:   list[int],
    n_plates:       int,
    plate_effects:  dict[int, np.ndarray],
    reserved:       set[int],
    output_dir:     Path,
    n_peptides:     int = 3000,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """
    Generate a synthetic peptide-level dataset for missed cleavage QC.

    The MC distribution is drawn from real GDM plasma data:
      MC=0: ~79 %,  MC=1: ~20 %,  MC=2: ~1 %

    Two biological samples are chosen as 'poor digestion' samples.  Their MC
    distributions are shifted towards higher missed cleavage counts, simulating
    insufficient trypsin activity or incomplete sample denaturation.

    Returns
    -------
    peptide_df   : peptide × samples DataFrame (Spectronaut-like format)
    bio_pd_idxs  : indices (in bio_ids) of poor-digestion samples planted
    poor_idxs    : same, for reporting
    """
    n_bio = len(bio_ids)
    n_qc  = len(qc_ids)

    # ── Peptide properties ────────────────────────────────────────────────────
    # Assign MC count to each peptide based on real GDM proportions
    mc_probs = [0.79, 0.20, 0.01]
    mc_counts = rng.choice([0, 1, 2], size=n_peptides, p=mc_probs)

    # Per-peptide baseline detection probability (proxy for abundance)
    # Low-abundance peptides (low detect_p) are more likely to be missed
    detect_p = rng.beta(2, 2, size=n_peptides)  # broad beta distribution
    detect_p = np.clip(detect_p, 0.05, 0.92)

    # Per-peptide mean intensity
    pep_means = rng.uniform(12, 28, size=n_peptides)

    # ── Assign poor-digestion samples (two bio samples, different plates) ────
    # Avoid positions already used for blood contamination / technical outliers
    def pick_pd_sample(plate_num: int, already_used: set[int]) -> int:
        candidates = [i for i, pn in enumerate(bio_plate_num)
                      if pn == plate_num and i not in already_used]
        return int(candidates[rng.integers(4, len(candidates) - 4)])

    pd_reserved = set(reserved)
    pd_idx1 = pick_pd_sample(2, pd_reserved); pd_reserved.add(pd_idx1)
    pd_idx2 = pick_pd_sample(4, pd_reserved); pd_reserved.add(pd_idx2)
    poor_set = {pd_idx1, pd_idx2}

    # ── Biological sample intensities ─────────────────────────────────────────
    bio_mat = np.full((n_peptides, n_bio), np.nan)
    for j in range(n_bio):
        if j in poor_set:
            # Poor digestion: boost high-MC detection, suppress MC=0 detection
            p_adj = np.where(mc_counts >= 1, detect_p * 2.5, detect_p * 0.35)
            p_adj = np.clip(p_adj, 0.03, 0.95)
        else:
            p_adj = detect_p
        detected = rng.random(n_peptides) < p_adj
        bio_mat[detected, j] = pep_means[detected] + rng.normal(0, 1.2, detected.sum())

    # ── QC sample intensities (same pooled reference, low noise) ─────────────
    qc_mat = np.full((n_peptides, n_qc), np.nan)
    for j in range(n_qc):
        detected = rng.random(n_peptides) < detect_p
        qc_mat[detected, j] = pep_means[detected] + rng.normal(0, 0.15, detected.sum())

    # ── Simplified peptide sequences (tryptic structure) ─────────────────────
    # For teaching purposes we generate plausible sequences that reflect MC count:
    #   MC=0 → ends in K or R, no internal K/R
    #   MC=1 → one internal K/R, ends in K or R
    #   MC=2 → two internal K/R, ends in K or R
    amino_acids = list("ACDEFGHILMNPQSTVWY")  # non-K/R amino acids
    kr = ["K", "R"]

    def _make_sequence(n_mc: int, rng: np.random.Generator) -> str:
        def seg(length: int) -> str:
            return "".join(rng.choice(amino_acids, size=length))
        length = int(rng.integers(6, 14))
        if n_mc == 0:
            return seg(length) + rng.choice(kr)
        elif n_mc == 1:
            half = max(2, length // 2)
            return seg(half) + rng.choice(kr) + seg(length - half) + rng.choice(kr)
        else:
            q = max(2, length // 3)
            return (seg(q) + rng.choice(kr) + seg(q) + rng.choice(kr)
                    + seg(length - 2 * q) + rng.choice(kr))

    stripped = [_make_sequence(int(mc), rng) for mc in mc_counts]
    modified = [f"_{s}_" for s in stripped]

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    pep_df = pd.DataFrame(
        np.concatenate([bio_mat, qc_mat], axis=1),
        columns=bio_ids + qc_ids,
    )
    pep_df.insert(0, "EG_ModifiedSequence",     modified)
    pep_df.insert(1, "PEP_StrippedSequence",    stripped)
    pep_df.insert(2, "PEP_NrOfMissedCleavages", mc_counts.astype(str))
    pep_df.insert(3, "PG_ProteinAccessions",    [f"PROT_{i % 500:04d}" for i in range(n_peptides)])
    pep_df.insert(4, "PG_Genes",                [f"GENE{i % 500}" for i in range(n_peptides)])

    pep_path = output_dir / "demo_peptide.parquet"
    pep_df.to_parquet(pep_path, index=False)

    # Compute and print per-sample MC stats for planted samples
    bio_mc_rates = []
    for j, sid in enumerate(bio_ids):
        detected = ~np.isnan(bio_mat[:, j])
        if detected.sum() == 0:
            continue
        rate = (mc_counts[detected] >= 1).mean()
        bio_mc_rates.append(rate)

    poor1_rate = (mc_counts[~np.isnan(bio_mat[:, pd_idx1])] >= 1).mean()
    poor2_rate = (mc_counts[~np.isnan(bio_mat[:, pd_idx2])] >= 1).mean()
    normal_median = float(np.median([r for j, r in enumerate(bio_mc_rates)
                                     if j not in poor_set]))

    print(f"  Peptides:              {n_peptides}")
    print(f"  MC=0 fraction:         {(mc_counts==0).mean():.1%} of peptides")
    print(f"  Normal MC rate:        {normal_median:.1%}  (median across samples)")
    print(f"  Poor digestion sample 1 ({bio_ids[pd_idx1]}):  MC rate {poor1_rate:.1%}")
    print(f"  Poor digestion sample 2 ({bio_ids[pd_idx2]}):  MC rate {poor2_rate:.1%}")
    print(f"  Saved: {pep_path}")

    return pep_df, [pd_idx1, pd_idx2], list(poor_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic proteomics demo data")
    parser.add_argument("--n-proteins",     type=int, default=2000)
    parser.add_argument("--n-subjects",     type=int, default=160)
    parser.add_argument("--n-timepoints",   type=int, default=3)
    parser.add_argument("--n-plates",       type=int, default=4)
    parser.add_argument("--n-qc-per-plate", type=int, default=3)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output",         type=str, default=None)
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
