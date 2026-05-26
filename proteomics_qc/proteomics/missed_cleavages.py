"""
Missed cleavage QC for plasma proteomics.

Trypsin cleaves proteins at the C-terminus of Lysine (K) and Arginine (R).
A "missed cleavage" is a K/R site where trypsin failed to cut.  The fraction
of peptides carrying ≥1 missed cleavage is a sensitive indicator of digestion
efficiency and sample preparation quality.

Benchmarks for well-prepared human plasma (iST or FASP):
  MC=0 (fully cleaved)  :  75–85 % of detected peptides
  MC=1 (one skip)       :  14–22 %
  MC=2 (two skips)      :  < 3 %
  MC rate (≥1)          :  < 25 %

Elevated MC rates indicate:
  - Insufficient trypsin activity (suboptimal enzyme:substrate ratio)
  - Incomplete denaturation or alkylation
  - Inhibitors carried over from sample preparation
  - Plate-level reagent failure

Reference:
  Sielaff et al. (2017) doi:10.1021/acs.jproteome.7b00183
  Geyer et al. (2019) doi:10.1038/s41467-019-13382-0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# Canonical meta-column names in a Spectronaut peptide export
_META_COLS = {
    "EG_ModifiedSequence",
    "PEP_StrippedSequence",
    "PEP_IsProteotypic",
    "PEP_NrOfMissedCleavages",
    "PG_ProteinAccessions",
    "PG_Genes",
}

_MC_COLORS = {
    0: "#2ca02c",   # green — fully cleaved
    1: "#ff7f0e",   # orange — one missed cleavage
    2: "#d62728",   # red — two or more missed cleavages
}


# ── Per-sample statistics ──────────────────────────────────────────────────────

def compute_mc_stats(
    peptide_df: pd.DataFrame,
    mc_col:     str = "PEP_NrOfMissedCleavages",
) -> pd.DataFrame:
    """
    Compute per-sample missed cleavage statistics.

    Parameters
    ----------
    peptide_df : (peptides × samples) DataFrame.
                 Must contain a column named ``mc_col`` with integer MC counts.
                 All other non-metadata columns are treated as sample intensity
                 columns; a peptide is "detected" in a sample if its value is
                 not NaN.
    mc_col     : name of the missed cleavage count column (default:
                 'PEP_NrOfMissedCleavages')

    Returns
    -------
    DataFrame with one row per sample and columns:
        n_total       – number of detected peptides
        n_mc0         – peptides with 0 missed cleavages
        n_mc1         – peptides with 1 missed cleavage
        n_mc2plus     – peptides with ≥2 missed cleavages
        mc_rate       – fraction of detected peptides with ≥1 MC (0–1)
        mc_frequency  – weighted-average MC count per detected peptide
    """
    sample_cols = [c for c in peptide_df.columns if c not in _META_COLS]
    mc = pd.to_numeric(peptide_df[mc_col], errors="coerce")

    records = []
    for s in sample_cols:
        detected = peptide_df[s].notna()
        n_total = int(detected.sum())
        if n_total == 0:
            records.append(
                {"sample": s, "n_total": 0, "n_mc0": 0, "n_mc1": 0,
                 "n_mc2plus": 0, "mc_rate": np.nan, "mc_frequency": np.nan}
            )
            continue

        mc_det = mc[detected]
        n_mc0    = int((mc_det == 0).sum())
        n_mc1    = int((mc_det == 1).sum())
        n_mc2p   = int((mc_det >= 2).sum())
        mc_rate  = (n_mc1 + n_mc2p) / n_total
        mc_freq  = float(mc_det.mean())

        records.append({
            "sample":       s,
            "n_total":      n_total,
            "n_mc0":        n_mc0,
            "n_mc1":        n_mc1,
            "n_mc2plus":    n_mc2p,
            "mc_rate":      mc_rate,
            "mc_frequency": mc_freq,
        })

    return pd.DataFrame(records).set_index("sample")


# ── Flagging ───────────────────────────────────────────────────────────────────

def flag_high_mc_samples(
    mc_stats: pd.DataFrame,
    n_sd:     float = 3.0,
) -> tuple[list[str], float]:
    """
    Flag samples with MC rate > mean + n_sd * std.

    Parameters
    ----------
    mc_stats : output of :func:`compute_mc_stats`
    n_sd     : number of standard deviations above the mean (default 3.0)

    Returns
    -------
    flagged   : sorted list of flagged sample IDs
    threshold : the data-driven cutoff used
    """
    rates = mc_stats["mc_rate"].dropna()
    threshold = float(rates.mean() + n_sd * rates.std())
    flagged = sorted(mc_stats.index[mc_stats["mc_rate"] > threshold].tolist())
    return flagged, threshold


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_mc_distribution(
    mc_stats:        pd.DataFrame,
    sample_meta:     pd.DataFrame,
    flagged_samples: list[str] | None = None,
    threshold:       float | None = None,
    title:           str = "Missed cleavage distribution per sample",
) -> None:
    """
    Two-panel plot: stacked MC fraction bars and per-plate MC rate violin.

    Top panel — stacked bar chart:
        One bar per sample (sorted by plate).  Bar segments show MC=0 (green),
        MC=1 (orange), MC=2+ (red) fractions.  Flagged samples are marked with
        a red dashed border.

    Bottom panel — box plot of MC rate per plate:
        Shows plate-level variation in digestion quality.

    Parameters
    ----------
    mc_stats        : output of :func:`compute_mc_stats`
    sample_meta     : sample metadata with a 'plate' column
    flagged_samples : sample IDs to highlight in the top panel
    threshold       : data-driven MC rate threshold drawn as a dashed line
    """
    flagged_set  = set(flagged_samples or [])
    plate_order  = sorted(sample_meta["plate"].dropna().unique())

    # Align and sort by plate
    meta_al  = sample_meta.reindex(mc_stats.index).dropna(subset=["plate"])
    meta_al["_pcat"] = pd.Categorical(meta_al["plate"], plate_order, ordered=True)
    meta_al  = meta_al.sort_values("_pcat")
    ordered  = [s for s in meta_al.index if s in mc_stats.index]
    stats    = mc_stats.reindex(ordered)

    fig, (ax_bar, ax_box) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    # ── Stacked bar chart ─────────────────────────────────────────────────
    xs    = np.arange(len(ordered))
    total = stats["n_total"].replace(0, np.nan)
    f0    = stats["n_mc0"]    / total
    f1    = stats["n_mc1"]    / total
    f2    = stats["n_mc2plus"] / total

    ax_bar.bar(xs, f0, color=_MC_COLORS[0], width=0.9, label="MC = 0 (fully cleaved)")
    ax_bar.bar(xs, f1, bottom=f0, color=_MC_COLORS[1], width=0.9, label="MC = 1")
    ax_bar.bar(xs, f2, bottom=f0 + f1, color=_MC_COLORS[2], width=0.9, label="MC ≥ 2")

    # Highlight flagged samples
    for i, s in enumerate(ordered):
        if s in flagged_set:
            ax_bar.axvspan(i - 0.45, i + 0.45, color="red",
                           alpha=0.15, zorder=0)
            ax_bar.text(i, 1.02, "!", ha="center", va="bottom",
                        color="red", fontsize=9, fontweight="bold")

    # Plate boundary lines and labels
    for plate in plate_order:
        members = [s for s in ordered if meta_al.loc[s, "plate"] == plate]
        if not members:
            continue
        lo = ordered.index(members[0])
        hi = ordered.index(members[-1])
        mid = (lo + hi) / 2
        if lo > 0:
            ax_bar.axvline(lo - 0.5, color="black", lw=0.5, alpha=0.3)
        ax_bar.text(mid, -0.06, plate, ha="center", va="top",
                    fontsize=6.5, color="#555555")

    ax_bar.set_xlim(-0.5, len(ordered) - 0.5)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Fraction of detected peptides", fontsize=11)
    ax_bar.set_title(title, fontsize=11, fontweight="bold")
    ax_bar.set_xticks([])
    ax_bar.legend(loc="upper left", fontsize=9, framealpha=0.8)

    # Benchmark line (≥1 MC threshold)
    if threshold is not None:
        ax_bar.axhline(1 - threshold, color="grey", lw=1.0, linestyle="--",
                       alpha=0.6, label=f"Threshold MC≥1 > {threshold:.1%}")
        ax_bar.text(len(ordered) - 1, 1 - threshold + 0.01,
                    f"threshold ({threshold:.0%} MC rate)",
                    ha="right", va="bottom", fontsize=7, color="grey")

    # ── Per-plate box plot ─────────────────────────────────────────────────
    plate_data = []
    for plate in plate_order:
        members = [s for s in ordered if meta_al.loc[s, "plate"] == plate]
        rates = mc_stats.loc[members, "mc_rate"].dropna().values * 100
        plate_data.append(rates)

    bplot = ax_box.boxplot(
        plate_data, positions=range(len(plate_order)),
        labels=[p for p in plate_order],
        patch_artist=True, widths=0.5,
        medianprops={"color": "black", "lw": 1.5},
    )
    for patch in bplot["boxes"]:
        patch.set_facecolor("#aec7e8")

    if threshold is not None:
        ax_box.axhline(threshold * 100, color="red", lw=1.0, linestyle="--",
                       alpha=0.7, label=f"Threshold {threshold:.0%}")

    ax_box.set_ylabel("MC rate (%)\nper sample", fontsize=9)
    ax_box.set_xlabel("Plate", fontsize=10)
    ax_box.grid(axis="y", linestyle="--", alpha=0.3)
    ax_box.tick_params(axis="x", labelsize=8)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
