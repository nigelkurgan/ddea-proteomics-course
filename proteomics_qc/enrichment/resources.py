import pandas as pd


def load_gtex(path, background=None):
    """
    Load GTEx tissue enrichment resource.

    Returns dict {organ: set_of_uniprot_ids} for proteins enriched in each organ.
    """
    df = pd.read_excel(path)
    df = df[df["enriched"] == True].dropna(subset=["uniprot_id", "organ"])
    if background is not None:
        df = df[df["uniprot_id"].isin(background)]
    return {
        organ: set(grp["uniprot_id"])
        for organ, grp in df.groupby("organ")
    }


# HAtlas primary labels that are blood/immune cell types, not anatomical tissues.
# These are excluded from tissue-origin enrichment because they describe the cell
# compartment that expressed the protein, not the organ it originates from.
HATLAS_CELL_TYPE_LABELS = frozenset({
    "macrophages", "erythrocytes", "bcell", "neutrophils",
    "monocytes", "platelets", "tcellcd4", "tcellcd8",
    "plasma prot", "common",
})


def load_hatlas(path, background=None, primary_only=True, min_gls=1,
                tissue_only=True):
    """
    Load HAtlas blood/tissue proteome atlas resource.

    Parameters
    ----------
    primary_only : bool
        If True use only the primary tissue label (first term before '.' in global_label).
    min_gls : float
        Proteins with GLS <= min_gls are excluded. GLS reflects how many independent
        atlas sources agree on the assignment; GLS > 1 means at least two concur.
    tissue_only : bool
        If True (default) exclude entries whose primary label is a blood/immune cell
        type (e.g. bcell, monocytes) rather than an anatomical tissue.

    Returns dict {tissue: set_of_uniprot_ids}
    """
    df = pd.read_excel(path).dropna(subset=["uniprot_id", "global_label"])
    df = df[df["Gobal label score (GLS)"] > min_gls].copy()
    df["tissue"] = (
        df["global_label"].str.split(".").str[0] if primary_only else df["global_label"]
    )
    if tissue_only:
        df = df[~df["tissue"].isin(HATLAS_CELL_TYPE_LABELS)]
    if background is not None:
        df = df[df["uniprot_id"].isin(background)]
    return {
        tissue: set(grp["uniprot_id"])
        for tissue, grp in df.groupby("tissue")
    }


# Mapping from GTEx organ names to HAtlas primary tissue labels.
# Used for cross-resource concordance comparisons.
GTEX_TO_HATLAS = {
    "Adipose":   "adiposetissue",
    "Adrenal":   "adrenalgland",
    "Artery":    "artery",
    "Bladder":   "bladder",
    "Brain":     "brain",
    "Esophagus": "esophagus",
    "Heart":     "heart",
    "Intestine": "colon",
    "Kidney":    "kidney",
    "Liver":     "liver",
    "Lung":      "lung",
    "Muscle":    "muscle",
    "Ovary":     "ovary",
    "Pancreas":  "pancreas",
    "Prostate":  "prostate",
    "Skin":      "skin",
    "Spleen":    "spleen",
    "Stomach":   "stomach",
    "Thyroid":   "thyroid",
}


def load_hpa_secretome(path, background=None):
    """
    Load HPA secretome location annotations.

    Returns dict {secretome_location: set_of_uniprot_ids}
    """
    df = pd.read_csv(path, sep="\t").dropna(subset=["Uniprot", "Secretome location"])
    if background is not None:
        df = df[df["Uniprot"].isin(background)]
    return {
        loc: set(grp["Uniprot"])
        for loc, grp in df.groupby("Secretome location")
    }


def load_hpa_tissue_specificity(path, background=None):
    """
    Load HPA RNA tissue specificity categories.

    Returns dict {category: set_of_uniprot_ids}
    """
    df = pd.read_csv(path, sep="\t").dropna(subset=["Uniprot", "RNA tissue specificity"])
    if background is not None:
        df = df[df["Uniprot"].isin(background)]
    return {
        spec: set(grp["Uniprot"])
        for spec, grp in df.groupby("RNA tissue specificity")
    }


# Variant tissue names in HPA nTPM column that should be collapsed to a canonical label.
HPA_TISSUE_NAME_MAP = {
    "skin 1": "skin",
    "skin 2": "skin",
    "stomach 1": "stomach",
    "stomach 2": "stomach",
    "endometrium 1": "endometrium",
    "endometrium 2": "endometrium",
    "choroid plexus": "brain",
}


def load_hpa_tissue_enrichment(path, background=None, min_ntpm=1.0,
                                normalise_names=True):
    """
    Build an HPA tissue enrichment annotation from the 'RNA tissue specific nTPM' column.

    This is a protein-level (nTPM-based) tissue mapping, equivalent to the approach
    used in the R script figure_4_hpa_tissue_enrichment_olink_cleaned.Rmd.
    Only proteins with tissue-specific nTPM >= min_ntpm in a given tissue are included.

    Parameters
    ----------
    min_ntpm : float
        Minimum normalised TPM to consider a protein expressed in a tissue (default 1.0).
    normalise_names : bool
        If True, collapse HPA sub-tissue variants (e.g. 'skin 1' → 'skin') using
        HPA_TISSUE_NAME_MAP.

    Returns dict {tissue: set_of_uniprot_ids}
    """
    df = pd.read_csv(
        path, sep="\t",
        usecols=["Gene", "Uniprot", "RNA tissue specific nTPM"],
    ).dropna(subset=["Uniprot", "RNA tissue specific nTPM"])

    # HPA uses comma-separated UniProt IDs when a row covers multiple isoforms.
    df = df.copy()
    df["Uniprot"] = df["Uniprot"].str.split(", ")
    df = df.explode("Uniprot")

    if background is not None:
        df = df[df["Uniprot"].isin(background)]

    tissue_proteins: dict[str, set] = {}
    for _, row in df.iterrows():
        uniprot = row["Uniprot"]
        for pair in str(row["RNA tissue specific nTPM"]).split(";"):
            parts = pair.strip().split(": ", 1)
            if len(parts) != 2:
                continue
            tissue, ntpm_str = parts[0].strip(), parts[1].strip()
            if normalise_names:
                tissue = HPA_TISSUE_NAME_MAP.get(tissue, tissue)
            try:
                if float(ntpm_str) >= min_ntpm:
                    tissue_proteins.setdefault(tissue, set()).add(uniprot)
            except ValueError:
                continue

    return tissue_proteins


def load_ppa(path, p_threshold=0.05, hr_direction="any", min_cases=50,
             background_genes=None):
    """
    Load Proteome-Phenome Atlas incident disease associations.

    Deng et al. (2025) Atlas of the plasma proteome in health and disease in 53,026 adults.
    Cell 188(1):253-271. https://doi.org/10.1016/j.cell.2024.10.045
    Download: https://proteome-phenome-atlas.com/

    Parameters
    ----------
    p_threshold : float
        Significance threshold for disease-protein associations.
    hr_direction : str
        'risk' = HR > 1 only, 'protective' = HR < 1 only, 'any' = both.
    min_cases : int
        Minimum number of incident cases required for a disease-protein pair.
    background_genes : iterable, optional
        Restrict to gene names present in the study.

    Returns dict {disease: set_of_gene_names}
    """
    df = pd.read_csv(path)
    df = df[df["P_value"] < p_threshold]
    df = df[df["NB_case"] >= min_cases]

    if hr_direction != "any":
        df = df.copy()
        df["HR"] = df["HR[95%CI]"].str.extract(r"^([\d.]+)").astype(float)
        if hr_direction == "risk":
            df = df[df["HR"] > 1]
        elif hr_direction == "protective":
            df = df[df["HR"] < 1]

    if background_genes is not None:
        df = df[df["Protein"].isin(background_genes)]

    return {
        disease: set(grp["Protein"])
        for disease, grp in df.groupby("Disease")
    }
