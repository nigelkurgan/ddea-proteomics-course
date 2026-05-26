"""
Blood contamination QC for plasma proteomics.

Plasma is prepared by centrifuging whole blood to remove cells.  If the
preparation is imperfect — slow centrifugation, prolonged delay, or
haemolysis during freezing/thawing — cell-specific proteins enter the
plasma fraction and confound downstream analysis.

This module detects two common contamination types using known marker proteins:
  * Erythrocyte (haemolysis):  haemoglobin subunits, carbonic anhydrases, catalase, etc.
  * Platelet:                  talin-1, myosin-9, tropomyosin-4

Coagulation factors (fibrinogen) are genuinely present in plasma and serve as an
internal reference to confirm that the proteomic measurement is working.

Reference: Geyer et al. (2019) doi:10.1038/s41467-019-13382-0
           https://pmc.ncbi.nlm.nih.gov/articles/PMC6835559/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colours for each contamination type — consistent with the reference paper
_TYPE_COLORS = {
    "erythrocyte": "#d62728",   # red
    "platelet":    "#ff7f0e",   # orange
    "coagulation": "#1f77b4",   # blue
}
_TYPE_ORDER = ["erythrocyte", "platelet", "coagulation"]


# ── Panel loading ──────────────────────────────────────────────────────────────

def load_contamination_panel(
    panel_path: str | Path,
    protein_accessions: list[str],
) -> pd.DataFrame:
    """
    Load the blood contamination marker panel and match it to the protein matrix.

    Each row of the panel can have multiple accessions (semicolon-separated).
    We search through them in order and return the first accession found in the
    matrix.  If no accession matches, the row is dropped.

    Parameters
    ----------
    panel_path         : path to blood_contamination_panel.xlsx
    protein_accessions : list of protein accessions in the quant matrix (column names)

    Returns
    -------
    DataFrame with columns: protein_names, genes, protein_ids, quality_marker,
                            best_markers, matched_id
    """
    panel = pd.read_excel(panel_path)
    matrix_set = set(protein_accessions)

    matched = []
    for _, row in panel.iterrows():
        # Try each semicolon-separated accession in order
        found = None
        for acc in str(row["protein_ids"]).split(";"):
            acc = acc.strip()
            if acc in matrix_set:
                found = acc
                break
        matched.append(found)

    panel["matched_id"] = matched

    # Keep only rows with a match; if the same matrix protein is matched by
    # multiple panel rows, keep the one flagged as a best marker.
    result = (
        panel[panel["matched_id"].notna()]
        .sort_values("best_markers", ascending=False, na_position="last")
        .drop_duplicates(subset="matched_id", keep="first")
        .reset_index(drop=True)
    )

    n_matched = len(result)
    n_best    = (result["best_markers"] == 1.0).sum()
    print(f"Blood contamination panel: {n_matched} proteins matched "
          f"({n_best} best markers)")
    for mtype in _TYPE_ORDER:
        sub  = result[result["quality_marker"] == mtype]
        best = (sub["best_markers"] == 1.0).sum()
        print(f"  {mtype:12s}: {len(sub):2d} total, {best} best markers  "
              f"({', '.join(sub[sub['best_markers']==1]['genes'].tolist())})")
    return result


# ── Score computation ──────────────────────────────────────────────────────────

def compute_contamination_scores(
    quant_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-sample median log2 intensity for best markers of each type.

    Only "best marker" proteins are used for scoring (best_markers == 1).
    Samples with all best markers missing receive NaN for that type.

    Parameters
    ----------
    quant_df : (samples × proteins) log2 intensity DataFrame
    panel_df : output of :func:`load_contamination_panel`

    Returns
    -------
    DataFrame: index=sample IDs, columns=contamination types
               Values = median log2 intensity of detected best markers.
    """
    scores = {}
    for mtype in _TYPE_ORDER:
        best_ids = (
            panel_df.loc[
                (panel_df["quality_marker"] == mtype) & (panel_df["best_markers"] == 1.0),
                "matched_id",
            ]
            .tolist()
        )
        # Only keep proteins that are actually in the matrix
        avail = [p for p in best_ids if p in quant_df.columns]
        if not avail:
            continue
        # Median across available best markers per sample (ignore NaN)
        scores[mtype] = quant_df[avail].median(axis=1, skipna=True)

    return pd.DataFrame(scores)


# ── Flagging ───────────────────────────────────────────────────────────────────

def flag_contaminated_samples(
    scores_df: pd.DataFrame,
    n_sd: float = 3.0,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """
    Flag samples where a contamination score exceeds mean + n_sd * std.

    The threshold is computed separately for each marker type from the score
    distribution across all samples.  Because contaminated samples are rare,
    they have minimal influence on the mean and SD, so the data-driven cutoff
    sits in the gap between normal and contaminated samples.

    Parameters
    ----------
    scores_df : output of :func:`compute_contamination_scores`
    n_sd      : number of standard deviations above the mean (default 3.0)

    Returns
    -------
    flags      : dict  marker_type → sorted list of flagged sample IDs
    thresholds : dict  marker_type → threshold value used (log2)
    """
    flags:      dict[str, list[str]] = {}
    thresholds: dict[str, float]     = {}
    for col in ["erythrocyte", "platelet"]:   # coagulation is expected in plasma
        if col not in scores_df.columns:
            continue
        col_scores = scores_df[col].dropna()
        thresh     = float(col_scores.mean() + n_sd * col_scores.std())
        thresholds[col] = thresh
        flagged = scores_df.index[scores_df[col] > thresh].tolist()
        if flagged:
            flags[col] = sorted(flagged)
    return flags, thresholds


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_contamination_panel(
    quant_df: pd.DataFrame,
    bio_meta: pd.DataFrame,
    panel_df: pd.DataFrame,
    title: str = "Blood contamination panel",
    flagged_samples: list[str] | None = None,
) -> None:
    """
    Three-row plot of contamination marker intensities across biological samples.

    Each row = one contamination type (erythrocyte / platelet / coagulation).
    Samples are sorted by plate on the x-axis; plate boundaries are marked.
    Best-marker proteins are plotted in the type colour; secondary markers in grey.
    The per-sample median of best markers is drawn as a solid line — spikes reveal
    contaminated samples.

    Parameters
    ----------
    quant_df        : (samples × proteins) log2 intensity DataFrame
    bio_meta        : sample metadata (must contain 'plate')
    panel_df        : output of :func:`load_contamination_panel`
    flagged_samples : optional list of flagged sample IDs to highlight in red
    """
    flagged_set = set(flagged_samples or [])

    # Sort samples by plate so plate-level patterns are visible
    plate_order   = sorted(bio_meta["plate"].unique())
    bio_sorted    = bio_meta.copy()
    bio_sorted["_pcat"] = pd.Categorical(bio_sorted["plate"], plate_order, ordered=True)
    sample_order  = [s for s in bio_sorted.sort_values("_pcat").index if s in quant_df.index]
    sample_pos    = {s: i for i, s in enumerate(sample_order)}

    fig, axes = plt.subplots(len(_TYPE_ORDER), 1,
                             figsize=(14, 3.5 * len(_TYPE_ORDER)), sharex=True)

    for ax, mtype in zip(axes, _TYPE_ORDER):
        color     = _TYPE_COLORS[mtype]
        sub_panel = panel_df[panel_df["quality_marker"] == mtype]
        best_ids  = sub_panel[sub_panel["best_markers"] == 1.0]["matched_id"].tolist()
        other_ids = sub_panel[sub_panel["best_markers"] != 1.0]["matched_id"].tolist()

        # ── Secondary markers (grey, small) ───────────────────────────────
        for prot in [p for p in other_ids if p in quant_df.columns]:
            xs, ys = [], []
            for s in sample_order:
                v = quant_df.loc[s, prot]
                if not pd.isna(v):
                    xs.append(sample_pos[s])
                    ys.append(v)
            if xs:
                ax.scatter(xs, ys, s=1.5, alpha=0.15, color="grey",
                           edgecolors="none", rasterized=True, zorder=1)

        # ── Best markers (coloured, larger) ──────────────────────────────
        best_vals: dict[str, dict[str, float]] = {}   # prot → sample → value
        for prot in [p for p in best_ids if p in quant_df.columns]:
            gene = sub_panel.loc[sub_panel["matched_id"] == prot, "genes"].values[0]
            xs, ys = [], []
            prot_vals = {}
            for s in sample_order:
                v = quant_df.loc[s, prot]
                prot_vals[s] = float(v) if not pd.isna(v) else np.nan
                if not pd.isna(v):
                    xs.append(sample_pos[s])
                    ys.append(v)
            best_vals[prot] = prot_vals
            if xs:
                ax.scatter(xs, ys, s=4, alpha=0.55, color=color,
                           edgecolors="none", rasterized=True, zorder=2, label=gene)

        # ── Per-sample median of best markers ────────────────────────────
        if best_vals:
            med_xs, med_ys = [], []
            for s in sample_order:
                vals = [v for v in (best_vals[p].get(s, np.nan) for p in best_vals)
                        if not np.isnan(v)]
                if vals:
                    med_xs.append(sample_pos[s])
                    med_ys.append(float(np.median(vals)))
            if med_xs:
                ax.plot(med_xs, med_ys, color=color, lw=0.8, alpha=0.8,
                        zorder=3, label="Median (best markers)")

        # ── Highlight flagged samples ─────────────────────────────────────
        for s in flagged_set:
            if s in sample_pos:
                ax.axvline(sample_pos[s], color="red", lw=1.2,
                           alpha=0.7, linestyle="--", zorder=4)

        # ── Plate boundaries and labels ───────────────────────────────────
        for plate in plate_order:
            pmembers = [s for s in sample_order if bio_meta.loc[s, "plate"] == plate]
            if not pmembers:
                continue
            lo, hi = sample_pos[pmembers[0]], sample_pos[pmembers[-1]]
            mid = (lo + hi) / 2
            if lo > 0:
                ax.axvline(lo - 0.5, color="black", lw=0.4, alpha=0.25, zorder=0)
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > 14 else 14,
                    plate, ha="center", va="bottom", fontsize=6.5,
                    color="#555555", clip_on=True)

        n_best = len([p for p in best_ids if p in quant_df.columns])
        ax.set_ylabel("log₂ intensity", fontsize=10)
        ax.set_title(f"{mtype.capitalize()} markers  ({n_best} best markers)",
                     loc="left", fontsize=11)
        ax.legend(fontsize=7, loc="upper left", ncol=min(n_best + 1, 5),
                  framealpha=0.7)
        ax.set_xlim(-1, len(sample_order))
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel(
        f"Samples sorted by plate  (n = {len(sample_order)} biological samples)",
        fontsize=10,
    )
    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()
