#!/usr/bin/env python3
"""
Freezer Box Manifest and 96-well Plate Randomization Pipeline
for the proteomics QC teaching cohort (160 participants, 3 timepoints each).

Pipeline:
  Step 1 — Assign 480 samples to 10×10 freezer boxes.
            Each participant's timepoints must be in DIFFERENT boxes.
  Step 2 — Randomize samples into 96-well plates.
            All timepoints per participant go on the SAME plate.
            Two pooled-QC controls placed at A1/A2 (start) and H11/H12 (end).
            Case/Control groups balanced across plates.

Output:
  data/demo/demo_sample_manifest.xlsx  (multi-sheet workbook)
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Freezer box (10 × 10)
BOX_ROWS = 10   # A–J
BOX_COLS = 10   # 1–10
POSITIONS_PER_BOX = BOX_ROWS * BOX_COLS  # 100

# 96-well plate (8 × 12)
PLATE_ROWS = 8   # A–H
PLATE_COLS = 12  # 1–12
WELLS_PER_PLATE = PLATE_ROWS * PLATE_COLS  # 96
CONTROL_POSITIONS = [(0, 0), (0, 1), (7, 10), (7, 11)]  # A1, A2, H11, H12
SAMPLES_PER_PLATE = WELLS_PER_PLATE - len(CONTROL_POSITIONS)  # 92

# Colour palette for Excel output
COLOURS = {
    "header":    "1F4E79",  # dark blue  – header row
    "ctrl":      "FF0000",  # red        – control wells
    "T00":       "BDD7EE",  # light blue
    "T01":       "C6EFCE",  # light green
    "T02":       "FFEB9C",  # light yellow
    "Case":      "FCE4D6",  # light orange – participant group tint
    "Control":   "EAF0FD",  # very light blue
    "empty":     "F2F2F2",  # light grey  – empty well
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def row_letter(idx: int) -> str:
    return chr(65 + idx)


def well_name(row_idx: int, col_idx: int) -> str:
    return f"{row_letter(row_idx)}{col_idx + 1}"


def all_positions(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


# ── Step 1: Freezer Box Assignment ────────────────────────────────────────────

def assign_freezer_boxes(samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each sample to a 10×10 freezer box position.

    Constraint: for every participant, their 3 timepoints (T00, T01, T02)
    must be stored in 3 DIFFERENT boxes.  This is the expected real-world
    situation — samples accumulate in boxes over time as visits occur.

    Strategy (cyclic rotation):
      - Participants are randomly ordered then split into 5 equal groups.
      - Group k gets:  T00 → Box(k+1),  T01 → Box(k+2 mod 5),
                       T02 → Box(k+3 mod 5)
      - This guarantees each box holds exactly 96 samples (32 per timepoint ×
        3 rotation slots), with 4 empty positions per box.
      - Positions within each box are then shuffled independently.
    """
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    timepoints = sorted(samples_df["timepoint"].unique())  # e.g. T00, T01, T02
    n_tp = len(timepoints)

    participants = sorted(samples_df["subject_id"].unique())
    rng.shuffle(participants)
    n_participants = len(participants)

    # Number of boxes = minimum to hold all samples with >0 empty slots
    total_samples = len(samples_df)
    n_boxes = int(np.ceil(total_samples / POSITIONS_PER_BOX))
    # With 480 samples and 100 per box → 5 boxes (96 samples each, 4 empty)

    group_size = int(np.ceil(n_participants / n_boxes))

    # Build per-box position pools (all 100 positions, shuffled)
    box_positions = {}
    for box in range(1, n_boxes + 1):
        pos = all_positions(BOX_ROWS, BOX_COLS)
        rng.shuffle(pos)
        box_positions[box] = pos

    box_cursors = {box: 0 for box in range(1, n_boxes + 1)}

    assignments = []
    for rank, pid in enumerate(participants):
        group_idx = rank // group_size  # 0-based group within 0..n_boxes-1
        for tp_idx, tp in enumerate(timepoints):
            box = (group_idx + tp_idx) % n_boxes + 1
            cursor = box_cursors[box]
            r, c = box_positions[box][cursor]
            box_cursors[box] += 1

            row = samples_df.loc[
                (samples_df["subject_id"] == pid) & (samples_df["timepoint"] == tp)
            ].iloc[0].to_dict()
            row["box_number"] = box
            row["box_row"] = row_letter(r)
            row["box_col"] = c + 1
            row["box_position"] = well_name(r, c)
            assignments.append(row)

    return pd.DataFrame(assignments)


# ── Step 2: Plate Randomization ───────────────────────────────────────────────

def assign_plates(box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomize samples into 96-well plates.

    Constraints:
      - All timepoints for a participant go on the SAME plate (avoids
        between-plate variation confounding longitudinal comparisons).
      - Cases and Controls are balanced across plates.
      - 4 wells reserved per plate for pooled-QC controls:
          A1, A2 (start of run) and H11, H12 (end of run).

    With 160 participants × 3 timepoints = 480 samples and 92 sample wells
    per plate → 6 plates needed, each holding ~26–27 participants.
    """
    rng = random.Random(RANDOM_SEED + 1)
    np.random.seed(RANDOM_SEED + 1)

    # Participant metadata
    participants = (
        box_df.groupby("subject_id")["group"]
        .first()
        .reset_index()
        .rename(columns={"group": "group"})
    )
    cases = participants.loc[participants["group"] == "Case", "subject_id"].tolist()
    controls = participants.loc[participants["group"] == "Control", "subject_id"].tolist()
    rng.shuffle(cases)
    rng.shuffle(controls)

    # Calculate how many plates we need
    n_participants = len(participants)
    max_participants_per_plate = SAMPLES_PER_PLATE // 3  # each has 3 timepoints → 30
    n_plates = int(np.ceil(n_participants / max_participants_per_plate))

    plates: list[list] = [[] for _ in range(n_plates)]
    plate_case_count = [0] * n_plates
    plate_ctrl_count = [0] * n_plates

    def best_plate_for(is_case: bool) -> int:
        """Return the plate index with fewest samples of the relevant group."""
        min_val = min(plate_case_count if is_case else plate_ctrl_count)
        candidates = [
            p for p in range(n_plates)
            if (plate_case_count[p] if is_case else plate_ctrl_count[p]) == min_val
            and len(plates[p]) + 3 <= SAMPLES_PER_PLATE
        ]
        if not candidates:
            # Fall back: find any plate with room
            candidates = [
                p for p in range(n_plates)
                if len(plates[p]) + 3 <= SAMPLES_PER_PLATE
            ]
        return candidates[0]

    for pid in cases:
        p = best_plate_for(is_case=True)
        plates[p].append(pid)
        plate_case_count[p] += 1

    for pid in controls:
        p = best_plate_for(is_case=False)
        plates[p].append(pid)
        plate_ctrl_count[p] += 1

    # Build sample-well positions for each plate (exclude control wells)
    control_set = set(CONTROL_POSITIONS)
    sample_wells_template = [
        (r, c)
        for r in range(PLATE_ROWS)
        for c in range(PLATE_COLS)
        if (r, c) not in control_set
    ]

    plate_assignments = []
    for plate_num, pids in enumerate(plates, start=1):
        # Shuffle the participant list for within-plate randomization
        rng.shuffle(pids)
        # Expand to individual samples (all 3 timepoints per participant)
        plate_samples = []
        for pid in pids:
            for _, row in box_df[box_df["subject_id"] == pid].iterrows():
                plate_samples.append(row.to_dict())
        # Shuffle within plate
        rng.shuffle(plate_samples)

        wells = sample_wells_template.copy()
        for i, sample in enumerate(plate_samples):
            r, c = wells[i]
            sample["dest_plate"] = plate_num
            sample["dest_well"] = well_name(r, c)
            sample["dest_row"] = row_letter(r)
            sample["dest_col"] = c + 1
            plate_assignments.append(sample)

    return pd.DataFrame(plate_assignments)


# ── Excel Writers ─────────────────────────────────────────────────────────────

def _fill(hex_colour: str) -> PatternFill:
    return PatternFill(fill_type="solid", fgColor=hex_colour)


def _header_style(cell, hex_colour: str = "1F4E79") -> None:
    cell.fill = _fill(hex_colour)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.alignment = Alignment(horizontal="center", vertical="center")


def write_box_layout_sheet(ws, box_df: pd.DataFrame, box_num: int) -> None:
    """Write a 10×10 grid showing which sample is in each position."""
    ws.title = f"Box_{box_num:02d}"

    # Legend header
    ws.cell(1, 1, f"Freezer Box {box_num:02d}  —  10×10 layout").font = Font(bold=True, size=12)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=11)

    # Column headers (1–10)
    for c in range(BOX_COLS):
        cell = ws.cell(2, c + 2, str(c + 1))
        _header_style(cell)
        ws.column_dimensions[get_column_letter(c + 2)].width = 16

    ws.column_dimensions["A"].width = 4

    box_samples = box_df[box_df["box_number"] == box_num]
    pos_map = {
        (row["box_row"], row["box_col"]): row
        for _, row in box_samples.iterrows()
    }

    tp_colours = {tp: COLOURS.get(tp, "FFFFFF") for tp in COLOURS if tp.startswith("T")}

    for r in range(BOX_ROWS):
        rl = row_letter(r)
        ws.cell(r + 3, 1, rl).font = Font(bold=True)
        ws.cell(r + 3, 1).alignment = Alignment(horizontal="center")
        for c in range(BOX_COLS):
            cell = ws.cell(r + 3, c + 2)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            key = (rl, c + 1)
            if key in pos_map:
                row = pos_map[key]
                sid = int(row["subject_id"])
                tp = row["timepoint"]
                grp = row["group"]
                cell.value = f"{sid}\n{tp}"
                colour = tp_colours.get(tp, COLOURS.get(grp, "FFFFFF"))
                cell.fill = _fill(colour)
            else:
                cell.value = ""
                cell.fill = _fill(COLOURS["empty"])
        ws.row_dimensions[r + 3].height = 28


def write_plate_layout_sheet(ws, plate_df: pd.DataFrame, plate_num: int,
                              n_cases: int, n_ctrls: int) -> None:
    """Write an 8×12 grid showing which sample is in each well."""
    ws.title = f"Plate_{plate_num:02d}"

    header = (
        f"Plate {plate_num:02d}  —  8×12 layout  "
        f"({n_cases} Cases, {n_ctrls} Controls, {len(CONTROL_POSITIONS)} QC wells)"
    )
    ws.cell(1, 1, header).font = Font(bold=True, size=12)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=13)

    for c in range(PLATE_COLS):
        cell = ws.cell(2, c + 2, str(c + 1))
        _header_style(cell)
        ws.column_dimensions[get_column_letter(c + 2)].width = 14

    ws.column_dimensions["A"].width = 4

    ctrl_set = {well_name(r, c) for r, c in CONTROL_POSITIONS}
    well_map = {row["dest_well"]: row for _, row in plate_df.iterrows()}

    tp_colours = {tp: COLOURS.get(tp, "FFFFFF") for tp in COLOURS if tp.startswith("T")}

    for r in range(PLATE_ROWS):
        rl = row_letter(r)
        ws.cell(r + 3, 1, rl).font = Font(bold=True)
        ws.cell(r + 3, 1).alignment = Alignment(horizontal="center")
        for c in range(PLATE_COLS):
            well = well_name(r, c)
            cell = ws.cell(r + 3, c + 2)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            if well in ctrl_set:
                cell.value = "QC\nCTRL"
                cell.fill = _fill(COLOURS["ctrl"])
                cell.font = Font(bold=True, color="FFFFFF")
            elif well in well_map:
                row = well_map[well]
                sid = int(row["subject_id"])
                tp = row["timepoint"]
                cell.value = f"{sid}\n{tp}"
                colour = tp_colours.get(tp, "FFFFFF")
                cell.fill = _fill(colour)
            else:
                cell.fill = _fill(COLOURS["empty"])
        ws.row_dimensions[r + 3].height = 28


def write_educational_sheet(ws) -> None:
    """Add an educational questions sheet."""
    ws.title = "Educational_Questions"
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 38
    ws.column_dimensions["C"].width = 80

    headers = ["#", "Question", "Answer / Guidance"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(1, col, h)
        _header_style(cell, "1F4E79")

    questions = [
        (
            "Q1",
            "Why do we randomize samples within plates?",
            (
                "Randomization distributes potential systematic biases (instrument drift, "
                "reagent batch effects, operator fatigue, well-position effects) evenly across "
                "experimental groups.  Without randomization, if Cases happened to occupy the "
                "first half of the plate and Controls the second half, any positional gradient "
                "would appear as a biological signal — a classic batch effect.  Randomization "
                "ensures that any technical variation is uncorrelated with group membership, "
                "making downstream statistical comparisons valid."
            ),
        ),
        (
            "Q2",
            "Why should all timepoints for a given participant be run on the same plate?",
            (
                "Proteomics platforms introduce plate-level technical variation: differences in "
                "reagent preparation, digestion efficiency, TMT labelling, or instrument "
                "sensitivity between plates.  If a participant's T00 sample is on Plate 1 and "
                "their T02 sample is on Plate 3, any observed change between those timepoints "
                "is confounded by between-plate variability — you cannot distinguish a true "
                "longitudinal change from a plate artefact.  By keeping all samples from the "
                "same participant on one plate, any plate-level technical variation cancels out "
                "when computing within-participant changes, preserving the integrity of "
                "longitudinal analyses."
            ),
        ),
        (
            "Q3",
            "What is the purpose of the controls placed at A1/A2 and H11/H12?  "
            "What type of controls are these?",
            (
                "These are POOLED QC SAMPLES — aliquots of a single pooled reference made by "
                "combining equal volumes of every participant's sample.  They serve multiple "
                "purposes:\n"
                "  1. Run monitoring: placing controls at the START (A1/A2) and END (H11/H12) "
                "of the plate detects instrument drift over the course of the run.  A large "
                "CV between start and end controls signals a technical problem.\n"
                "  2. Between-plate normalisation: the same pooled sample on every plate "
                "provides a common reference that allows correction for plate-to-plate "
                "differences in absolute signal.\n"
                "  3. Reproducibility benchmarking: the CV of the pooled QC across plates "
                "quantifies the overall technical noise floor of the experiment.\n"
                "They differ from BIOLOGICAL CONTROLS (e.g. a healthy reference cohort) and "
                "BLANK CONTROLS (buffer only); they are purely technical anchors."
            ),
        ),
        (
            "Q4",
            "What additional randomization considerations exist for a longitudinal study?",
            (
                "Beyond well position, several other factors should be balanced:\n"
                "  • Plate assignment: Cases and Controls should be roughly equally "
                "represented on every plate (not all Cases on plates 1–3 and all Controls on "
                "4–6).\n"
                "  • Sex/age balance: if demographic sub-groups differ in sample quality, "
                "stratified randomization within plates reduces confounding.\n"
                "  • Batch/operator: if multiple lab technicians process plates, randomize "
                "which technician handles which plate, and record this in the manifest.\n"
                "  • Run order: the order in which plates are injected into the mass "
                "spectrometer should also be randomized (or at minimum recorded) to allow "
                "post-hoc run-order correction."
            ),
        ),
    ]

    for row_idx, (num, question, answer) in enumerate(questions, start=2):
        ws.cell(row_idx, 1, num).font = Font(bold=True)
        ws.cell(row_idx, 1).alignment = Alignment(vertical="top")

        q_cell = ws.cell(row_idx, 2, question)
        q_cell.font = Font(bold=True)
        q_cell.alignment = Alignment(wrap_text=True, vertical="top")

        a_cell = ws.cell(row_idx, 3, answer)
        a_cell.alignment = Alignment(wrap_text=True, vertical="top")

        ws.row_dimensions[row_idx].height = max(80, answer.count("\n") * 15 + 30)


# ── Summary Sheets ────────────────────────────────────────────────────────────

def write_box_summary(ws, box_df: pd.DataFrame, n_boxes: int) -> None:
    ws.title = "Freezer_Box_Summary"
    headers = ["Box", "Total Samples", "T00", "T01", "T02",
               "Cases", "Controls", "Empty Positions"]
    for col, h in enumerate(headers, 1):
        _header_style(ws.cell(1, col, h))
        ws.column_dimensions[get_column_letter(col)].width = 16

    for box in range(1, n_boxes + 1):
        sub = box_df[box_df["box_number"] == box]
        tp_counts = sub["timepoint"].value_counts()
        row = [
            box,
            len(sub),
            tp_counts.get("T00", 0),
            tp_counts.get("T01", 0),
            tp_counts.get("T02", 0),
            (sub["group"] == "Case").sum(),
            (sub["group"] == "Control").sum(),
            POSITIONS_PER_BOX - len(sub),
        ]
        for col, val in enumerate(row, 1):
            ws.cell(box + 1, col, val).alignment = Alignment(horizontal="center")


def write_plate_overview(ws, plate_df: pd.DataFrame, n_plates: int) -> None:
    ws.title = "Plate_Overview"
    headers = ["Plate", "Total Samples", "Participants", "Cases", "Controls",
               "T00", "T01", "T02", "Control Wells", "Empty Wells"]
    for col, h in enumerate(headers, 1):
        _header_style(ws.cell(1, col, h))
        ws.column_dimensions[get_column_letter(col)].width = 16

    for plate in range(1, n_plates + 1):
        sub = plate_df[plate_df["dest_plate"] == plate]
        tp_counts = sub["timepoint"].value_counts()
        n_pids = sub["subject_id"].nunique()
        n_cases = sub[sub["group"] == "Case"]["subject_id"].nunique()
        n_ctrls = sub[sub["group"] == "Control"]["subject_id"].nunique()
        row = [
            plate,
            len(sub),
            n_pids,
            n_cases,
            n_ctrls,
            tp_counts.get("T00", 0),
            tp_counts.get("T01", 0),
            tp_counts.get("T02", 0),
            len(CONTROL_POSITIONS),
            WELLS_PER_PLATE - len(sub) - len(CONTROL_POSITIONS),
        ]
        for col, val in enumerate(row, 1):
            ws.cell(plate + 1, col, val).alignment = Alignment(horizontal="center")


def write_picking_guide(ws, plate_df: pd.DataFrame) -> None:
    """Flat table: for each sample, where it lives in the freezer and where it goes."""
    ws.title = "Picking_Guide"
    cols = [
        "subject_id", "group", "timepoint",
        "box_number", "box_position",
        "dest_plate", "dest_well",
    ]
    headers = [
        "Subject ID", "Group", "Timepoint",
        "Source Box", "Source Position",
        "Destination Plate", "Destination Well",
    ]
    for col, h in enumerate(headers, 1):
        _header_style(ws.cell(1, col, h))
        ws.column_dimensions[get_column_letter(col)].width = 18

    sorted_df = plate_df[cols].sort_values(["box_number", "box_position"])
    for row_idx, (_, row) in enumerate(sorted_df.iterrows(), start=2):
        for col_idx, col_name in enumerate(cols, 1):
            cell = ws.cell(row_idx, col_idx, row[col_name])
            cell.alignment = Alignment(horizontal="center")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent.parent
    metadata_path = base / "data" / "demo" / "demo_metadata.csv"
    output_path = base / "data" / "demo" / "demo_sample_manifest.xlsx"

    print("Loading demo metadata…")
    meta = pd.read_csv(metadata_path)

    # Keep only real participant samples (not QC rows)
    samples = meta[meta["is_qc"] == False].copy()
    samples["subject_id"] = samples["subject_id"].astype(int)
    print(f"  {len(samples)} samples, {samples['subject_id'].nunique()} participants, "
          f"timepoints: {sorted(samples['timepoint'].unique())}")

    # ── Step 1: Freezer box assignment ────────────────────────────────────────
    print("\nStep 1: Assigning samples to freezer boxes…")
    box_df = assign_freezer_boxes(samples)
    n_boxes = box_df["box_number"].nunique()
    print(f"  {n_boxes} boxes, {POSITIONS_PER_BOX} positions each")
    for box in range(1, n_boxes + 1):
        n = (box_df["box_number"] == box).sum()
        print(f"  Box {box:02d}: {n} samples ({POSITIONS_PER_BOX - n} empty)")

    # Verify constraint: no participant has two timepoints in same box
    conflicts = (
        box_df.groupby(["subject_id", "box_number"])
        .size()
        .reset_index(name="count")
    )
    assert (conflicts["count"] == 1).all(), \
        "CONSTRAINT VIOLATED: participant has >1 timepoint in the same box!"
    print("  ✓ Constraint satisfied: each participant's timepoints are in different boxes")

    # ── Step 2: Plate randomization ───────────────────────────────────────────
    print("\nStep 2: Randomizing samples into 96-well plates…")
    plate_df = assign_plates(box_df)
    n_plates = plate_df["dest_plate"].nunique()
    print(f"  {n_plates} plates")
    for plate in range(1, n_plates + 1):
        sub = plate_df[plate_df["dest_plate"] == plate]
        n_pids = sub["subject_id"].nunique()
        n_cases = sub[sub["group"] == "Case"]["subject_id"].nunique()
        n_ctrls = sub[sub["group"] == "Control"]["subject_id"].nunique()
        print(f"  Plate {plate:02d}: {len(sub)} samples, {n_pids} participants "
              f"({n_cases} Cases / {n_ctrls} Controls)")

    # Verify constraint: all timepoints per participant on same plate
    participant_plates = (
        plate_df.groupby("subject_id")["dest_plate"]
        .nunique()
        .reset_index(name="n_plates")
    )
    assert (participant_plates["n_plates"] == 1).all(), \
        "CONSTRAINT VIOLATED: participant has samples on multiple plates!"
    print("  ✓ Constraint satisfied: all timepoints per participant are on the same plate")

    # ── Write Excel workbook ──────────────────────────────────────────────────
    print(f"\nWriting workbook to {output_path}…")
    wb = Workbook()
    wb.remove(wb.active)  # remove default empty sheet

    write_box_summary(wb.create_sheet(), box_df, n_boxes)
    for box in range(1, n_boxes + 1):
        write_box_layout_sheet(wb.create_sheet(), box_df, box)

    write_plate_overview(wb.create_sheet(), plate_df, n_plates)
    for plate in range(1, n_plates + 1):
        sub = plate_df[plate_df["dest_plate"] == plate]
        n_cases = sub[sub["group"] == "Case"]["subject_id"].nunique()
        n_ctrls = sub[sub["group"] == "Control"]["subject_id"].nunique()
        write_plate_layout_sheet(wb.create_sheet(), sub, plate, n_cases, n_ctrls)

    write_picking_guide(wb.create_sheet(), plate_df)
    write_educational_sheet(wb.create_sheet())

    wb.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"\nSheets: {[ws.title for ws in wb.worksheets]}")


if __name__ == "__main__":
    main()
