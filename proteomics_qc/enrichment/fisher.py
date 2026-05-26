import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def run_enrichment(gene_set, background, annotation_dict, min_overlap=3):
    """
    Fisher's exact test enrichment for each term in annotation_dict.

    Parameters
    ----------
    gene_set : iterable
        Protein IDs of interest (subset of background).
    background : iterable
        All protein IDs in the universe (all detected proteins).
    annotation_dict : dict {term: iterable_of_ids}
        Mapping from annotation terms to protein sets.
    min_overlap : int
        Minimum hits required to include a term in results.

    Returns
    -------
    pd.DataFrame with columns:
        term, n_background, n_annotated, n_in_set, n_overlap,
        odds_ratio, p_value, fdr_bh, overlap_ids
    """
    background = set(background)
    gene_set = set(gene_set) & background

    rows = []
    for term, annotated in annotation_dict.items():
        annotated_in_bg = set(annotated) & background
        if not annotated_in_bg:
            continue

        overlap = gene_set & annotated_in_bg
        if len(overlap) < min_overlap:
            continue

        a = len(overlap)
        b = len(gene_set) - a
        c = len(annotated_in_bg) - a
        d = len(background) - len(gene_set) - c

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")

        rows.append({
            "term": term,
            "n_background": len(background),
            "n_annotated": len(annotated_in_bg),
            "n_in_set": len(gene_set),
            "n_overlap": a,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "overlap_ids": ",".join(sorted(overlap)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "term", "n_background", "n_annotated", "n_in_set",
            "n_overlap", "odds_ratio", "p_value", "fdr_bh", "overlap_ids",
        ])

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    _, fdr, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["fdr_bh"] = fdr
    return df
