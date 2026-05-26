import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_enrichment_barplot(enrichment_df, title="Enrichment Analysis", top_n=15,
                             fdr_threshold=0.05, figsize=None):
    """
    Horizontal bar chart of top enriched terms sorted by odds ratio.

    Bars are coloured red (FDR < threshold) or blue (not significant).
    Annotation shows overlap count and FDR value for each term.
    """
    if enrichment_df.empty:
        print("No enrichment results to plot.")
        return

    df = enrichment_df.head(top_n).copy().sort_values("odds_ratio", ascending=True)
    colors = [
        "#d73027" if fdr < fdr_threshold else "#4575b4"
        for fdr in df["fdr_bh"]
    ]

    if figsize is None:
        figsize = (9, max(3, 0.45 * len(df)))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["term"], df["odds_ratio"], color=colors, edgecolor="white",
                   linewidth=0.5)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="OR = 1 (no enrichment)")

    max_or = df["odds_ratio"].max()
    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"n={row['n_overlap']}  FDR={row['fdr_bh']:.2g}"
        ax.text(bar.get_width() + max_or * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=7)

    sig_patch = mpatches.Patch(color="#d73027", label=f"FDR < {fdr_threshold}")
    ns_patch = mpatches.Patch(color="#4575b4", label=f"FDR ≥ {fdr_threshold}")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=8, loc="lower right")

    ax.set_xlabel("Odds Ratio", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_or * 1.35)
    plt.tight_layout()
    plt.show()


def plot_enrichment_dotplot(results_dict, fdr_threshold=0.05, top_n=10, figsize=None):
    """
    Dot plot comparing enrichment across multiple resources.

    Dot size = number of overlapping proteins.
    Dot colour = -log10(FDR), warmer = more significant.

    Parameters
    ----------
    results_dict : dict {resource_name: pd.DataFrame}
        Keyed by resource label; values are run_enrichment() outputs.
    top_n : int
        Maximum significant terms shown per resource.
    """
    rows = []
    for resource, df in results_dict.items():
        if df.empty:
            continue
        sig = df[df["fdr_bh"] < fdr_threshold].head(top_n)
        for _, row in sig.iterrows():
            rows.append({
                "resource": resource,
                "term": row["term"],
                "neg_log10_fdr": -np.log10(max(row["fdr_bh"], 1e-300)),
                "odds_ratio": row["odds_ratio"],
                "n_overlap": row["n_overlap"],
            })

    if not rows:
        print("No significant enrichments (FDR < {}) to plot.".format(fdr_threshold))
        return

    plot_df = pd.DataFrame(rows)
    resources = list(results_dict.keys())
    terms = plot_df.groupby("term")["neg_log10_fdr"].max().sort_values(
        ascending=False
    ).index.tolist()

    if figsize is None:
        figsize = (max(6, len(resources) * 2.5), max(4, len(terms) * 0.55))

    fig, ax = plt.subplots(figsize=figsize)

    r_idx = {r: i for i, r in enumerate(resources)}
    t_idx = {t: i for i, t in enumerate(terms)}

    sc = ax.scatter(
        [r_idx[r] for r in plot_df["resource"]],
        [t_idx[t] for t in plot_df["term"]],
        s=plot_df["n_overlap"] * 6,
        c=plot_df["neg_log10_fdr"],
        cmap="YlOrRd",
        edgecolors="grey",
        linewidths=0.4,
        alpha=0.9,
        vmin=0,
    )

    plt.colorbar(sc, ax=ax, label="-log₁₀(FDR)")
    ax.set_xticks(range(len(resources)))
    ax.set_xticklabels(resources, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlim(-0.5, len(resources) - 0.5)
    ax.set_ylim(-0.5, len(terms) - 0.5)
    ax.set_title(
        f"Enrichment dot plot (size = n overlap, colour = -log₁₀ FDR)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
