"""
Generate synthetic proteomics demo data for the QC teaching pipeline.

Creates a realistic parquet file with known properties:
  - n_proteins  : 2000 proteins, 30 % with high missing rate
  - n_samples   : 160 samples across 4 plates (40 samples/plate)
  - 2 groups    : 80 case / 80 control (balanced)
  - batch effects: +0.3 shift on plate 2, −0.2 on plate 3, +0.5 on plate 4
  - outliers    : 4 samples with artificially degraded profiles
  - proteotypic : ~20 % of proteins have group-differential expression (log2FC ~ 0.5)

Usage
-----
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --output data/demo/my_demo.parquet

The script writes:
  data/demo/demo_proteomics.parquet  — protein × samples intensity matrix
  data/demo/demo_metadata.csv        — sample metadata (plate, group, subject_id)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo(
    n_proteins: int = 2000,
    n_samples: int = 160,
    n_plates: int = 4,
    n_outliers: int = 4,
    missing_rate_global: float = 0.25,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic log2 intensity data with realistic properties.

    Returns
    -------
    quant_df  : (proteins × samples) log2 intensity DataFrame (+ PG_* columns)
    meta_df   : (samples) metadata DataFrame
    """
    rng = np.random.default_rng(seed)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_per_plate = n_samples // n_plates
    # ── Sample IDs ──────────────────────────────────────────────────────────
    sample_ids: list[str] = []
    plate_labels: list[str] = []
    group_labels: list[str] = []
    subject_ids: list[int] = []

    for plate_i in range(1, n_plates + 1):
        for s_i in range(1, n_per_plate + 1):
            sid = (plate_i - 1) * n_per_plate + s_i
            sample_ids.append(f"SAMPLE_{sid:04d}_P{plate_i:02d}")
            plate_labels.append(f"Plate_{plate_i:02d}")
            group_labels.append("Case" if sid % 2 == 1 else "Control")
            subject_ids.append(sid)

    # ── Protein metadata ──────────────────────────────────────────────────────
    protein_accs = [f"P{i:05d}" for i in range(1, n_proteins + 1)]
    gene_names   = [f"GENE{i}" for i in range(1, n_proteins + 1)]
    protein_names = [f"Protein {i}" for i in range(1, n_proteins + 1)]

    # ── Base intensities: proteins follow a dynamic range distribution ────────
    # Mean log2 intensities drawn from a realistic range (~18–30 in log2)
    protein_means = rng.uniform(18, 30, size=n_proteins)
    # Noise level varies by protein (some more reproducible than others)
    protein_sd = rng.uniform(0.2, 1.0, size=n_proteins)

    # ── Generate core intensity matrix (proteins × samples) ───────────────────
    mat = (
        protein_means[:, np.newaxis]
        + rng.normal(0, protein_sd[:, np.newaxis], size=(n_proteins, n_samples))
    )

    # ── Add plate batch effects ────────────────────────────────────────────────
    plate_shifts = {1: 0.0, 2: 0.35, 3: -0.25, 4: 0.50}
    sample_plate_num = [int(p.split("_")[1]) for p in plate_labels]
    shifts = np.array([plate_shifts[p] for p in sample_plate_num])
    mat += shifts[np.newaxis, :]   # broadcast over proteins

    # ── Add differential expression: ~20 % of proteins are DE ────────────────
    n_de = int(n_proteins * 0.20)
    de_indices = rng.choice(n_proteins, n_de, replace=False)
    de_fc = rng.choice([-1, 1], size=n_de) * rng.uniform(0.3, 0.8, size=n_de)
    case_mask = np.array([1 if g == "Case" else 0 for g in group_labels])
    for idx, fc in zip(de_indices, de_fc):
        mat[idx, :] += fc * case_mask   # add fold-change to case samples

    # ── Add structured missingness ────────────────────────────────────────────
    # Low-abundance proteins are more likely to be missing (MNAR)
    # Normalise means to [0,1] and use as detection probability
    detect_prob = (protein_means - protein_means.min()) / (protein_means.max() - protein_means.min())
    # Scale: lowest-abundance protein has ~70 % missing, highest has ~5 % missing
    missing_prob = 0.05 + (1 - detect_prob) * 0.65
    # Add a global floor
    missing_prob = np.clip(missing_prob, 0.01, 0.80)

    missing_mask = rng.random(size=(n_proteins, n_samples)) < missing_prob[:, np.newaxis]
    mat[missing_mask] = np.nan

    # ── Inject 4 outlier samples ──────────────────────────────────────────────
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    for i, oi in enumerate(outlier_indices):
        if i == 0:
            # Type A: global intensity shift down (degraded sample)
            mat[:, oi] -= 3.0
        elif i == 1:
            # Type B: very high missing rate (failed injection)
            extra_miss = rng.random(n_proteins) < 0.70
            mat[extra_miss, oi] = np.nan
        elif i == 2:
            # Type C: random noise (swap/mislabelled)
            mat[:, oi] = rng.normal(24, 3, size=n_proteins)
        else:
            # Type D: intensity shift up (contamination)
            mat[:, oi] += 2.5

    print(f"  Outlier samples injected at positions: {sorted(outlier_indices.tolist())}")
    print(f"  Outlier sample IDs: {[sample_ids[i] for i in sorted(outlier_indices)]}")

    # ── Assemble parquet DataFrame (proteins × samples, with PG_* columns) ───
    prot_df = pd.DataFrame(mat, index=protein_accs, columns=sample_ids)
    prot_df.index.name = "PG_ProteinAccessions"
    prot_df.insert(0, "PG_Genes", gene_names)
    prot_df.insert(1, "PG_ProteinNames", protein_names)
    prot_df = prot_df.reset_index()   # PG_ProteinAccessions becomes a column

    # ── Sample metadata ───────────────────────────────────────────────────────
    meta_df = pd.DataFrame({
        "sample_id":  sample_ids,
        "plate":      plate_labels,
        "group":      group_labels,
        "subject_id": subject_ids,
    }).set_index("sample_id")

    # ── Save ──────────────────────────────────────────────────────────────────
    parquet_path = output_dir / "demo_proteomics.parquet"
    meta_path    = output_dir / "demo_metadata.csv"

    prot_df.to_parquet(parquet_path, index=False)
    meta_df.to_csv(meta_path)

    print(f"  Saved: {parquet_path}  ({prot_df.shape[0]} proteins × {len(sample_ids)} samples)")
    print(f"  Saved: {meta_path}")
    print(f"  Groups: {meta_df['group'].value_counts().to_dict()}")
    print(f"  Plates: {meta_df['plate'].value_counts().sort_index().to_dict()}")
    print(f"  Overall missing rate: {np.isnan(mat).mean():.1%}")

    return prot_df, meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic proteomics demo data")
    parser.add_argument("--n-proteins", type=int, default=2000)
    parser.add_argument("--n-samples",  type=int, default=160)
    parser.add_argument("--n-plates",   type=int, default=4)
    parser.add_argument("--n-outliers", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     type=str, default=None,
                        help="Directory to save output files (default: data/demo/)")
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    generate_demo(
        n_proteins=args.n_proteins,
        n_samples=args.n_samples,
        n_plates=args.n_plates,
        n_outliers=args.n_outliers,
        seed=args.seed,
        output_dir=out,
    )
