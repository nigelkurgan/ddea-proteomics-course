#!/usr/bin/env python3
"""
Generate lecture slide decks for the proteomics QC and enrichment analysis course.
Run from repo root: python scripts/create_slides.py
Output:
    slides/proteomics_qc_slides.pptx
    slides/proteomics_enrichment_slides.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── palette ───────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1A, 0x35, 0x5C)
TEAL   = RGBColor(0x07, 0x89, 0xAD)
LTBLUE = RGBColor(0xD6, 0xEB, 0xF2)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARK   = RGBColor(0x1E, 0x29, 0x3B)
GREY   = RGBColor(0x64, 0x74, 0x8B)

SW, SH = 13.33, 7.5
CX  = 0.30       # content left (after teal strip)
CY  = 1.22       # content top  (below title bar)
CW  = SW - CX - 0.25
CH  = SH - CY - 0.15

FOOTER = "CBMR Proteomics Course  |  nigel.kurgan@sund.ku.dk"


# ── low-level helpers ─────────────────────────────────────────────────────────
def new_prs():
    prs = Presentation()
    prs.slide_width  = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(s, x, y, w, h, fill):
    sh = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def txt(s, text, x, y, w, h, size=16, bold=False, italic=False,
        color=DARK, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left   = Inches(0.05)
    tf.margin_right  = Inches(0.05)
    tf.margin_top    = Inches(0)
    tf.margin_bottom = Inches(0)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name   = "Calibri"
    return tb


def blist(s, items, x, y, w, h, size=15, color=DARK,
          header=None, hsize=17, hcolor=NAVY):
    """Bullet list with optional bold header."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left   = Inches(0.05)
    tf.margin_right  = Inches(0.05)
    tf.margin_top    = Inches(0)
    tf.margin_bottom = Inches(0)
    first_para = tf.paragraphs[0]
    if header:
        p = first_para
        r = p.add_run()
        r.text = header
        r.font.size  = Pt(hsize)
        r.font.bold  = True
        r.font.color.rgb = hcolor
        r.font.name  = "Calibri"
        p.space_after = Pt(6)
    for i, item in enumerate(items):
        p = first_para if (not header and i == 0) else tf.add_paragraph()
        r = p.add_run()
        r.text = f"▸  {item}"
        r.font.size  = Pt(size)
        r.font.color.rgb = color
        r.font.name  = "Calibri"
        p.space_after = Pt(3)
    return tb


def table(s, data, x, y, w, h, col_widths=None,
          hdr_fill=NAVY, hdr_color=WHITE, hsize=13, bsize=11):
    """Simple styled table; data[0] is the header row."""
    rows, cols = len(data), len(data[0])
    tbl = s.shapes.add_table(rows, cols,
                              Inches(x), Inches(y),
                              Inches(w), Inches(h)).table
    if col_widths:
        for ci, cw in enumerate(col_widths):
            tbl.columns[ci].width = Inches(cw)
    alt = LTBLUE
    for ri, row in enumerate(data):
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = str(cell_text)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.runs[0] if p.runs else p.add_run()
            run.text = str(cell_text)
            run.font.size  = Pt(hsize if ri == 0 else bsize)
            run.font.bold  = (ri == 0)
            run.font.name  = "Calibri"
            run.font.color.rgb = hdr_color if ri == 0 else DARK
            fill = cell.fill
            fill.solid()
            fill.fore_color.rgb = (hdr_fill if ri == 0
                                   else (WHITE if ri % 2 == 1 else alt))
    return tbl


# ── slide templates ───────────────────────────────────────────────────────────
def title_slide(prs, title, subtitle=None):
    s = blank(prs)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    rect(s, 0, SH - 0.18, SW, 0.18, TEAL)
    txt(s, title, 0.8, 1.6, SW - 1.6, 2.4,
        size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if subtitle:
        txt(s, subtitle, 0.8, 3.85, SW - 1.6, 1.0,
            size=21, italic=True, color=LTBLUE, align=PP_ALIGN.CENTER)
    txt(s, FOOTER, 0.5, SH - 0.48, SW - 1.0, 0.35,
        size=10, color=GREY, align=PP_ALIGN.CENTER)
    return s


def content_slide(prs, title):
    """White slide with navy title bar and teal left accent."""
    s = blank(prs)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    rect(s, 0, 0,       SW,   1.10, NAVY)
    rect(s, 0, 1.10,  0.14, SH - 1.18, TEAL)
    rect(s, 0, SH - 0.08, SW, 0.08, NAVY)
    txt(s, title, 0.35, 0.20, SW - 0.55, 0.75,
        size=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    return s


def closing_slide(prs, heading, body, detail):
    s = blank(prs)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = TEAL
    rect(s, 0, SH - 0.18, SW, 0.18, NAVY)
    txt(s, heading, 0.8, 1.4, SW - 1.6, 1.5,
        size=50, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, body, 0.8, 3.0, SW - 1.6, 1.2,
        size=22, italic=True, color=NAVY, align=PP_ALIGN.CENTER)
    txt(s, detail, 0.8, 4.3, SW - 1.6, 0.7,
        size=16, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, FOOTER, 0.5, SH - 0.48, SW - 1.0, 0.35,
        size=10, color=NAVY, align=PP_ALIGN.CENTER)
    return s


def info_cards(s, cards, x, y, w, h, n_cols=2):
    """Render a list of (title, body) cards in n_cols columns."""
    n = len(cards)
    n_rows = (n + n_cols - 1) // n_cols
    gap = 0.12
    cw = (w - gap * (n_cols - 1)) / n_cols
    ch = (h - gap * (n_rows - 1)) / n_rows
    for i, (ct, cb) in enumerate(cards):
        col = i % n_cols
        row = i // n_cols
        cx = x + col * (cw + gap)
        cy = y + row * (ch + gap)
        rect(s, cx, cy, cw, ch, LTBLUE)
        rect(s, cx, cy, cw, 0.36, NAVY)
        txt(s, ct, cx + 0.1, cy + 0.05, cw - 0.2, 0.30,
            size=13, bold=True, color=WHITE)
        txt(s, cb, cx + 0.1, cy + 0.40, cw - 0.2, ch - 0.48,
            size=12, color=DARK)


# ── QC deck ───────────────────────────────────────────────────────────────────
def build_qc_deck():
    prs = new_prs()
    c1w = CW / 2 - 0.25   # left column width
    c2x = CX + CW / 2 + 0.1  # right column x
    c2w = CW / 2 - 0.15   # right column width

    # S1 title
    title_slide(prs,
                "Quality Control of DIA Plasma Proteomics",
                "From raw intensities to an analysis-ready protein matrix")

    # S2 learning objectives
    s = content_slide(prs, "Learning Objectives")
    blist(s, [
        "Score and flag samples with blood contamination (haemolysis, platelet activation)",
        "Assess per-sample missed cleavage rates as a proxy for digestion quality",
        "Visualise and interpret missing value patterns at sample and protein level",
        "Apply median scaling to remove per-sample loading variation",
        "Detect technical outlier samples using multi-method approaches (Z-score, PCA, KS test)",
        "Diagnose and correct systematic plate batch effects",
        "Evaluate assay reproducibility using coefficient of variation (CV) analysis",
    ], CX, CY, CW, CH, size=16)

    # S3 DIA proteomics context
    s = content_slide(prs, "DIA Plasma Proteomics — Context")
    blist(s, [
        "Data-Independent Acquisition (DIA) — all peptide fragments sampled simultaneously",
        "Enables deep, reproducible quantification of thousands of plasma proteins",
        "Typical panel: 1,500–5,000 proteins per participant per workflow",
        "Multi-plate experiments introduce systematic technical variation (batch effects)",
        "Missing values are intrinsic — proteins near the detection limit are stochastically observed",
    ], CX, CY, c1w, CH, size=14,
       header="What is DIA plasma proteomics?", hsize=16)
    info_cards(s, [
        ("Depth",       "Quantifies secreted, shed, and leaked plasma proteins"),
        ("Throughput",  "Hundreds to thousands of samples per study"),
        ("Sensitivity", "pg/mL range for abundant proteins"),
        ("Challenge",   "Plate-to-plate and run-to-run technical variation"),
    ], c2x, CY, c2w, CH, n_cols=1)

    # S4 pipeline overview
    s = content_slide(prs, "QC Pipeline — Eight Steps")
    table(s, [
        ["Step", "What we measure", "Why it matters"],
        ["1  Blood contamination",  "Erythrocyte & platelet marker scores",           "Cell lysis inflates non-plasma proteins"],
        ["2  Missed cleavages",     "Per-sample MC rate from peptide data",            "Poor digestion distorts quantification"],
        ["3  Missing values",       "Heatmap + group-level completeness",              "Spot failed injections and low-abundance proteins"],
        ["4  Filtering",            "Protein completeness threshold",                  "Discard unreliable low-abundance features"],
        ["5  Normalisation",        "Median scaling (shift + MAV)",                    "Remove per-sample loading variation"],
        ["6  Outlier detection",    "Z-score, PCA, KDE + KS confirmation",            "Flag technical outlier samples"],
        ["7  Batch correction",     "Plate-median and ComBat",                         "Remove systematic plate biases"],
        ["8  CV analysis",          "Intra-plate, inter-plate, within/between subject","Assess assay reproducibility"],
    ], CX, CY, CW, CH, col_widths=[2.8, 4.6, 5.1], hsize=13, bsize=11)

    # S5 blood contamination
    s = content_slide(prs, "Step 1 — Blood Contamination Scoring")
    blist(s, [
        "Haemolysis: red blood cell lysis releases haemoglobin and cytoplasmic proteins",
        "Platelet activation: releases alpha-granule and cytoplasmic contents",
        "Both inflate abundances of proteins not genuinely present in healthy plasma",
        "Approach: compute a composite marker score per sample (mean of marker z-scores)",
        "Flag samples > 2–3 SD above the group median score for review or removal",
    ], CX, CY, c1w, CH, size=14,
       header="The problem", hsize=16)
    blist(s, [
        "HBB, HBA1/2 — haemoglobin subunits (RBC lysis)",
        "AHSP — alpha-haemoglobin stabilising protein",
        "PPBP (CXCL7) — platelet basic protein",
        "PF4 — platelet factor 4",
        "GP9 — glycoprotein IX (platelet membrane)",
    ], c2x, CY, c2w, CH - 0.55, size=14,
       header="Key marker proteins", hsize=16)
    rect(s, c2x, CY + CH - 0.5, c2w, 0.44, LTBLUE)
    txt(s, "Score = mean z-score across all markers per sample",
        c2x + 0.1, CY + CH - 0.48, c2w - 0.2, 0.40,
        size=13, italic=True, color=TEAL)

    # S6 missed cleavages
    s = content_slide(prs, "Step 2 — Missed Cleavages")
    blist(s, [
        "Trypsin cleaves peptide bonds after Lys (K) and Arg (R) residues",
        "A missed cleavage (MC) = a peptide that retains an internal K or R",
        "High MC rate signals incomplete digestion — shifts peptide intensity profiles",
        "Calculated from peptide-level data: MC peptides / total peptides per sample",
        "Flag samples with MC rate > group mean + 2 SD",
    ], CX, CY, c1w, CH, size=14,
       header="Trypsin digestion quality", hsize=16)
    blist(s, [
        "Good MC rate: < 20–30%",
        "High MC (> 40%) may indicate insufficient enzyme, wrong pH, or inhibitors in the sample",
        "Correlated with sample preparation quality, not instrument performance",
        "Can be plate-correlated — always inspect across plates",
        "Useful quality gate before normalisation and downstream analysis",
    ], c2x, CY, c2w, CH, size=14,
       header="Interpretation", hsize=16)

    # S7 missing values & filtering
    s = content_slide(prs, "Step 3 — Missing Values & Filtering")
    blist(s, [
        "Missing at random: protein near detection limit — stochastically detected",
        "Missing not at random: protein systematically absent in one biological group",
        "Failed injections: all proteins missing for a single sample (column of NAs)",
        "Heatmap visualisation: rows = proteins, columns = samples — reveals patterns instantly",
        "Column completeness: flag samples with fewer than 50–70% of proteins detected",
    ], CX, CY, c1w, CH, size=14,
       header="Why values go missing", hsize=16)
    blist(s, [
        "Protein-level filter: keep proteins present in >= 50% of samples",
        "Group-level filter: present in >= 50% within each biological group",
        "After filtering: smaller but higher-quality protein set",
        "Imputation (KNN): for downstream methods requiring a complete matrix",
        "Missing values carry biological information — do not impute blindly",
    ], c2x, CY, c2w, CH, size=14,
       header="Filtering strategy", hsize=16)

    # S8 normalisation
    s = content_slide(prs, "Step 4 — Normalisation")
    blist(s, [
        "Goal: remove per-sample loading variation while preserving biological signal",
        "Sources of variation: total protein injected, LC-MS/MS sensitivity drift within a run",
        "Median scaling (shift): subtract each sample's median, add the global median — aligns centres",
        "MAV scaling: additionally scale by median absolute variation — equalises spread",
        "Check: box plots of all samples should overlap after normalisation",
        "Normalise AFTER outlier removal, not before — outliers distort the median reference",
        "Do not normalise out group differences — use global or within-plate references only",
    ], CX, CY, CW, CH, size=15)

    # S9 outlier detection
    s = content_slide(prs, "Step 5 — Outlier Detection")
    blist(s, [
        "Z-score: flag samples whose mean protein intensity is > 3 SD from the group mean",
        "PCA: principal component analysis — outliers appear as isolated, distant points",
        "KDE: kernel density estimate of per-sample intensity distribution",
        "KS test: Kolmogorov-Smirnov — quantifies distributional distance from the group median",
    ], CX, CY, c1w, CH, size=14,
       header="Three complementary methods", hsize=16)
    blist(s, [
        "Multi-method consensus: only remove samples flagged by 2 or more methods",
        "KS test provides statistical confirmation for visually suspicious samples",
        "Check metadata: is the outlier a known problematic sample? (wrong volume, late injection)",
        "Remove outliers before normalisation and batch correction",
        "Record all removed samples in the QC report with reason for removal",
    ], c2x, CY, c2w, CH, size=14,
       header="Strategy", hsize=16)

    # S10 batch effects
    s = content_slide(prs, "Step 6 — Diagnosing Batch Effects")
    blist(s, [
        "Batch effect: systematic technical variation correlated with plate assignment or run date",
        "Cause: differences in reagent lots, operator, instrument condition, or sample preparation day",
        "Impact: inflates between-plate variance and can confound biological comparisons",
        "Diagnose with PCA: colour points by plate — clusters by plate indicate a batch effect",
        "Within vs between plate distances: batch-corrected distances should be similar",
        "PC x factor correlation: quantify how much variance each PC explains by plate identity",
        "Critical question: is plate assignment confounded with biological group?",
    ], CX, CY, CW, CH, size=15)

    # S11 batch correction
    s = content_slide(prs, "Step 7 — Batch Correction")
    blist(s, [
        "Plate-median correction",
        "For each protein: subtract the plate median, add the global median",
        "Simple, fast, and non-parametric",
        "Removes additive plate bias",
        "Recommended for most plasma proteomics datasets",
        "Does not require balanced group assignment across plates",
    ], CX, CY, c1w, CH, size=13,
       header="Method 1 — Plate-median", hsize=15)
    blist(s, [
        "ComBat (Johnson et al. 2007, Biostatistics)",
        "Empirical Bayes model of batch effects",
        "Removes both additive and multiplicative components",
        "More powerful but requires >= 2 samples per plate per biological group",
        "Preserves within-group variation — protects biological signal",
        "Use when plate-median correction is insufficient",
    ], c2x, CY, c2w, CH, size=13,
       header="Method 2 — ComBat", hsize=15)

    # S12 CV analysis + summary
    s = content_slide(prs, "Step 8 — CV Analysis & Pipeline Summary")
    blist(s, [
        "Coefficient of Variation (CV) = SD / mean x 100%",
        "Intra-plate CV: technical replicates on the same plate (instrument precision)",
        "Inter-plate CV: same sample across plates (batch-corrected reproducibility)",
        "Within-subject CV: repeated measures from the same participant",
        "Between-subject CV: biological variability across participants",
        "Target for plasma proteomics: intra-plate CV < 15%",
    ], CX, CY, c1w, CH, size=13,
       header="CV analysis", hsize=15)
    blist(s, [
        "Batch-corrected protein matrix (parquet format)",
        "QC report: per-sample flags, outlier list, batch diagnostics",
        "Log-transformed, normalised, batch-corrected — analysis-ready",
        "Ready for: differential expression, correlation, enrichment analysis",
    ], c2x, CY, c2w, 3.2, size=13,
       header="Pipeline outputs", hsize=15)
    rect(s, c2x, CY + 3.4, c2w, 2.4, LTBLUE)
    txt(s, "Rule of thumb", c2x + 0.1, CY + 3.45, c2w - 0.2, 0.38,
        size=13, bold=True, color=NAVY)
    blist(s, [
        "Intra-plate CV  < 15%",
        "Inter-plate CV  < 25%",
        "Between / within CV ratio  > 2",
    ], c2x + 0.05, CY + 3.8, c2w - 0.1, 1.9, size=13, color=DARK)

    # S13 closing
    closing_slide(prs,
                  "Over to You",
                  "Work through the QC notebook on synthetic demo data",
                  "data/demo/  →  notebooks/proteomics_qc_tutorial.ipynb")
    return prs


# ── enrichment deck ───────────────────────────────────────────────────────────
def build_enrichment_deck():
    prs = new_prs()
    c1w = CW / 2 - 0.25
    c2x = CX + CW / 2 + 0.1
    c2w = CW / 2 - 0.15

    # S1 title
    title_slide(prs,
                "Enrichment Analysis of Plasma Proteomics Data",
                "From protein lists to biological interpretation")

    # S2 learning objectives
    s = content_slide(prs, "Learning Objectives")
    blist(s, [
        "Define the background universe and explain why its choice affects enrichment results",
        "Construct a 2x2 contingency table and apply Fisher's exact test for set enrichment",
        "Apply Benjamini-Hochberg FDR correction and correctly interpret adjusted p-values",
        "Interrogate protein lists with four complementary annotation resources",
        "Integrate results across resources to build a coherent biological narrative",
        "Distinguish constitutive from insulin-stimulated proteomic signals in a clamp study",
    ], CX, CY, CW, CH, size=16)

    # S3 why enrichment?
    s = content_slide(prs, "From Protein Lists to Biological Themes")
    blist(s, [
        "Downstream analysis yields lists of proteins associated with a trait or condition",
        "Individual proteins are difficult to interpret without broader context",
        "Enrichment analysis asks: are these proteins over-represented in a biological category?",
        "Test whether the overlap with an annotation set exceeds what chance predicts",
        "Turns a list of IDs into interpretable biology: tissues, secretion routes, diseases",
    ], CX, CY, c1w, CH, size=14,
       header="The challenge", hsize=16)
    blist(s, [
        "Gene Ontology (GO) — biological processes, molecular functions",
        "KEGG / Reactome — metabolic and signalling pathways",
        "Tissue expression atlases — GTEx (RNA), HAtlas (protein)",
        "Secretome / localisation — Human Protein Atlas",
        "Disease associations — Proteome-Phenome Atlas (population-scale)",
    ], c2x, CY, c2w, CH, size=14,
       header="Types of annotation resource", hsize=16)

    # S4 the study
    s = content_slide(prs, "The Study — Multi-Workflow Plasma Proteomics")
    blist(s, [
        "Phenotype: M-value (insulin sensitivity) measured by hyperinsulinemic-euglycemic clamp",
        "Insulin infused at fixed rate; glucose infusion rate = how well tissues take up glucose",
        "n = 161 participants with paired pre-clamp (fasting) and post-clamp (insulin-stimulated) plasma",
        "Proteins correlated with M-value identified in each workflow independently",
    ], CX, CY, c1w, CH, size=14,
       header="Clinical phenotype", hsize=16)
    info_cards(s, [
        ("5 Workflows",  "MagNet, Neat, PCA, Depleted, Olink"),
        ("8,022 proteins", "Unique UniProt IDs across all workflows"),
        ("488 significant", "Best p-value per protein for M-value"),
        ("Pre & post", "Fasting and insulin-stimulated associations"),
    ], c2x, CY, c2w, CH, n_cols=1)

    # S5 three protein groups
    s = content_slide(prs, "Three Protein Groups — Timing of Association")
    card_w = (CW - 0.4) / 3
    card_gap = 0.2
    groups = [
        ("Both  (n = 255)",
         "Significant pre- AND post-clamp",
         ["Constitutive markers of insulin sensitivity",
          "Stable across insulin stimulation",
          "Likely reflect chronic metabolic state",
          "Enriched in liver biology (GTEx, HAtlas)"]),
        ("Pre-clamp only  (n = 110)",
         "Significant at baseline only",
         ["Fasting-state markers",
          "Association disappears after insulin",
          "May reflect body composition or basal metabolic rate",
          "Adipose and immune-related biology"]),
        ("Post-clamp only  (n = 123)",
         "Emerge after insulin infusion",
         ["Insulin-stimulated proteins",
          "Not detectable at fasting — novel",
          "May be secreted or modified by insulin signalling",
          "Pancreatic and hepatic biology"]),
    ]
    for i, (ct, csub, citems) in enumerate(groups):
        cx_card = CX + i * (card_w + card_gap)
        rect(s, cx_card, CY, card_w, CH, LTBLUE)
        rect(s, cx_card, CY, card_w, 0.65, NAVY)
        txt(s, ct, cx_card + 0.1, CY + 0.05, card_w - 0.2, 0.36,
            size=14, bold=True, color=WHITE)
        txt(s, csub, cx_card + 0.1, CY + 0.40, card_w - 0.2, 0.26,
            size=11, italic=True, color=TEAL)
        blist(s, citems, cx_card + 0.1, CY + 0.73, card_w - 0.2, CH - 0.82,
              size=12, color=DARK)

    # S6 Fisher's exact test
    s = content_slide(prs, "Fisher's Exact Test — The Method")
    blist(s, [
        "For each annotation term, build a 2x2 contingency table",
        "Test whether the overlap between protein set and annotated proteins exceeds chance",
        "One-sided test (alternative = 'greater') — overrepresentation only",
        "Odds ratio = (a x d) / (b x c)  — OR > 1 means enriched in the gene set",
        "Multiple testing: Benjamini-Hochberg FDR across all terms tested",
        "Threshold: FDR < 0.05, minimum overlap >= 3 proteins",
    ], CX, CY, c1w, CH - 1.75, size=13,
       header="The approach", hsize=15)
    table(s, [
        ["",               "In protein set", "Not in set"],
        ["Annotated",      "a  (overlap)",   "c"],
        ["Not annotated",  "b",              "d"],
    ], CX, CY + CH - 1.65, c1w, 1.58,
       col_widths=[c1w * 0.38, c1w * 0.31, c1w * 0.31],
       hdr_fill=TEAL, hsize=12, bsize=12)
    blist(s, [
        "a  = proteins in gene set AND annotated to this term",
        "b  = proteins in gene set NOT annotated to this term",
        "c  = annotated proteins in background NOT in gene set",
        "d  = background minus gene set minus annotated",
        "Large OR + small FDR = strong enrichment signal",
    ], c2x, CY, c2w, CH, size=13,
       header="Reading the table", hsize=15)

    # S7 background universe
    s = content_slide(prs, "The Background Universe — A Critical Choice")
    blist(s, [
        "Background = all proteins 'at risk' of being in the gene set",
        "In this study: 8,022 unique UniProt IDs detected across all 5 workflows",
        "NOT all ~20,000 human proteins",
        "Why this matters:",
        "Plasma proteomics is pre-selected for secreted and abundant proteins",
        "Using all human proteins inflates enrichment for secreted / liver proteins",
        "Using tested plasma proteins asks: enriched vs other plasma proteins we could detect?",
        "This is the correct biological question for a plasma proteomics study",
        "Rule: background = proteins you could have detected, not all possible proteins",
    ], CX, CY, CW, CH, size=15)

    # S8 four resources
    s = content_slide(prs, "Four Annotation Resources")
    table(s, [
        ["Resource",                   "What it measures",                                              "IDs used",    "Key question"],
        ["GTEx  Tissue Atlas",          "RNA expression enrichment across 20 organs (tissue-specific genes)", "UniProt ID", "Which organ transcribes these proteins?"],
        ["HAtlas  Blood Proteome",      "Mass-spec tissue labels from Human Cell Atlas (protein-level)",      "UniProt ID", "Where does this protein physically originate?"],
        ["HPA  Secretome",              "How proteins reach plasma: secreted, shed, or leaked",                "UniProt ID", "What is the secretion mechanism?"],
        ["Proteome-Phenome Atlas",      "Incident disease hazard ratios from UK Biobank (n ~ 53,000)",        "Gene name",  "Disease risk or protective factor?"],
    ], CX, CY, CW, CH, col_widths=[2.5, 4.8, 1.7, 3.6], hsize=13, bsize=11)

    # S9 GTEx results
    s = content_slide(prs, "GTEx Tissue Enrichment — Results")
    blist(s, [
        "GTEx classifies genes as tissue-enriched when RNA is significantly higher than other organs",
        "Plasma proteins with liver-enriched RNA likely originate from hepatocytes",
        "The liver is the dominant factory for plasma proteins (albumin, coagulation factors, APPs)",
        "Enrichment asks: are our associated proteins more liver-derived than background plasma proteins?",
    ], CX, CY, c1w, 3.0, size=13,
       header="Interpreting GTEx enrichment", hsize=15)
    rect(s, CX, CY + 3.15, c1w, 2.7, LTBLUE)
    txt(s, "Biological highlight", CX + 0.1, CY + 3.20, c1w - 0.2, 0.35,
        size=13, bold=True, color=TEAL)
    txt(s, ("Post-clamp pancreas enrichment is biologically compelling — "
            "insulin infusion may stimulate pancreatic secretion of proteins "
            "not detectable at fasting."),
        CX + 0.1, CY + 3.55, c1w - 0.2, 2.2, size=12, italic=True, color=DARK)

    findings = [
        (NAVY, WHITE, "Both  (n = 255)",
         "Strong liver enrichment (OR ~ 6-8, FDR < 0.001). Consistent with liver-derived, metabolically active proteins."),
        (TEAL, WHITE, "Post-clamp only  (n = 123)",
         "Pancreas enrichment emerges after insulin stimulation — a novel, insulin-stimulated signal."),
        (LTBLUE, DARK, "Pre-clamp only  (n = 110)",
         "More diffuse: adipose, liver, and immune tissues."),
    ]
    fh = CH / len(findings)
    for i, (bg, tc, ft, fb) in enumerate(findings):
        fy = CY + i * fh
        rect(s, c2x, fy, c2w, fh - 0.08, bg)
        txt(s, ft, c2x + 0.1, fy + 0.06, c2w - 0.2, 0.35, size=13, bold=True, color=tc)
        txt(s, fb, c2x + 0.1, fy + 0.42, c2w - 0.2, fh - 0.55, size=12, color=tc)

    # S10 HAtlas + HPA secretome
    s = content_slide(prs, "HAtlas & HPA Secretome")
    blist(s, [
        "Protein-level atlas: direct mass-spec detection, not RNA inference",
        "Global label: primary tissue.cell type (e.g. liver.hepatocyte)",
        "We use the primary tissue label (first term before the dot)",
        "Both (n = 255) enriched for:",
        "  liver — consistent with GTEx tissue expression",
        "  plasma proteins — constitutively secreted fraction",
        "  macrophages — immune and inflammatory context",
    ], CX, CY, c1w, CH, size=13,
       header="HAtlas — protein-level tissue origin", hsize=15)
    blist(s, [
        "HPA classifies how proteins physically enter the bloodstream",
        "Secreted: classically via signal peptide (most liver proteins)",
        "Membrane-shed: extracellular domain released by proteases",
        "Intracellular / leaked: cytoplasmic proteins released by cell damage",
        "Blood cell-derived: from erythrocytes, platelets, or immune cells",
        "Both (n = 255): enriched for classically secreted proteins",
        "Consistent with liver-origin, signal-peptide-bearing proteins",
    ], c2x, CY, c2w, CH, size=13,
       header="HPA Secretome — mechanism of entry", hsize=15)

    # S11 PPA
    s = content_slide(prs, "Proteome-Phenome Atlas — Disease Enrichment")
    blist(s, [
        "53,026 UK Biobank participants — incident ICD-10 disease outcomes over follow-up",
        "Cox proportional hazards: HR per SD increase in plasma protein level",
        "Proteins measured by Olink proximity extension assay",
        "Identifiers: gene names (not UniProt) — matched to our gene set",
        "Risk associations (HR > 1) in Both (n = 255):",
        "  Obesity (OR ~ 5, FDR < 0.01) — expected for insulin resistance biology",
        "  Chronic pancreatitis — consistent with post-clamp pancreas enrichment",
        "  Angina pectoris, coronary artery disease — cardiometabolic cluster",
        "Test risk and protective (HR < 1) associations separately",
    ], CX, CY, CW, CH, size=14)

    # S12 cross-resource interpretation
    s = content_slide(prs, "Cross-Resource Interpretation")
    table(s, [
        ["Protein group",         "GTEx",               "HAtlas",                        "HPA secretome",        "PPA"],
        ["Both  (n = 255)",       "Liver  (OR ~ 7)",    "Liver, plasma, macrophage",     "Classically secreted", "Obesity, cardiometabolic"],
        ["Post-clamp  (n = 123)", "Pancreas  (OR ~ 4)", "Pancreas, liver",               "Secreted",             "Emerging signals"],
        ["Pre-clamp  (n = 110)",  "Adipose, liver",     "Diffuse",                       "Mixed",                "Fasting metabolic"],
    ], CX, CY, CW, 2.7, col_widths=[2.4, 2.4, 3.2, 2.7, 2.1], hsize=13, bsize=12)

    txt(s, "Biological narrative", CX, CY + 2.85, CW, 0.35,
        size=15, bold=True, color=NAVY)
    blist(s, [
        "Both: liver-derived, classically secreted, cardiometabolic disease proteins — constitutive insulin resistance biology",
        "Post-clamp: pancreatic response to insulin — a novel signal only visible after stimulation",
        "Pre-clamp: fasting-state markers in adipose and immune context — may reflect body composition",
    ], CX, CY + 3.25, CW, CH - 3.25, size=14, color=DARK)

    # S13 closing
    closing_slide(prs,
                  "Over to You",
                  "Work through the enrichment notebook using real insulin sensitivity data",
                  "data/Supplementary_Table_02.xlsx  →  notebooks/proteomics_enrichment_tutorial.ipynb")
    return prs


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path("slides")
    out_dir.mkdir(exist_ok=True)

    print("Building QC slides ...")
    prs_qc = build_qc_deck()
    out_qc  = out_dir / "proteomics_qc_slides.pptx"
    prs_qc.save(str(out_qc))
    print(f"  Written: {out_qc}  ({len(prs_qc.slides)} slides)")

    print("Building enrichment slides ...")
    prs_en = build_enrichment_deck()
    out_en  = out_dir / "proteomics_enrichment_slides.pptx"
    prs_en.save(str(out_en))
    print(f"  Written: {out_en}  ({len(prs_en.slides)} slides)")
