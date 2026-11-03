"""Shared helpers for the human-annotation kit (sheet I/O, label handling).

Used by sample_for_annotation.py and analyze_annotations.py.
Requires: pandas, numpy. Optional: openpyxl (xlsx sheets), pyyaml (config).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Offline servers: use the bundled pure-Python openpyxl (vendor/) when it is not installed.
_VENDOR = Path(__file__).resolve().parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.append(str(_VENDOR))

import pandas as pd

LABELS = ["CORRECT", "INCORRECT", "UNCLEAR"]
ISSUE_TAGS = [
    "PARTIAL",              # only part of the required answer is given
    "EXTRA_WRONG_CONTENT",  # right answer plus a wrong claim about the question
    "GRANULARITY",          # more / less specific than the reference
    "NO_ANSWER",            # empty, refusal, "I don't know", off-topic
    "REFERENCE_PROBLEM",    # reference looks wrong, outdated or ambiguous
    "OTHER",
]

# Accepted shorthand when reading filled sheets.
LABEL_ALIASES = {
    "CORRECT": "CORRECT", "C": "CORRECT", "1": "CORRECT", "TRUE": "CORRECT", "YES": "CORRECT",
    "INCORRECT": "INCORRECT", "I": "INCORRECT", "0": "INCORRECT", "FALSE": "INCORRECT",
    "WRONG": "INCORRECT", "NO": "INCORRECT",
    "UNCLEAR": "UNCLEAR", "U": "UNCLEAR", "?": "UNCLEAR",
}

# Text columns shown to annotators, in display order (only non-empty ones are written).
TEXT_COLUMNS = [
    "context", "question", "reference_answer",
    "other_acceptable_answers", "known_incorrect_answers", "model_response",
]
COLUMN_WIDTHS = {
    "item_id": 10, "dataset": 12, "context": 60, "question": 40, "reference_answer": 28,
    "other_acceptable_answers": 30, "known_incorrect_answers": 30, "model_response": 50,
    "label": 13, "issue_tag": 22, "notes": 30, "label_A": 13, "label_B": 13,
    "issue_tag_A": 20, "issue_tag_B": 20, "notes_A": 26, "notes_B": 26, "final_label": 13,
    "final_notes": 30,
}
EXCEL_CELL_LIMIT = 32000  # Excel hard limit is 32767 characters per cell

_ILLEGAL = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


def clean_text(value) -> str:
    """Make a value safe for an Excel / CSV cell."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = _ILLEGAL.sub(" ", text)
    if len(text) > EXCEL_CELL_LIMIT:
        text = text[:EXCEL_CELL_LIMIT] + " [TRUNCATED FOR DISPLAY]"
    return text


def have_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


def write_sheet(path_stem: Path, df: pd.DataFrame, input_cols: dict, fmt: str = "xlsx",
                instructions: list[str] | None = None) -> Path:
    """Write an annotation sheet.

    df          : rows to show (all columns are written, in order).
    input_cols  : {column_name: list_of_choices or None}; appended as empty columns
                  the annotator fills (choices -> dropdown in xlsx).
    Returns the path written.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(clean_text)
    for col in input_cols:
        out[col] = ""

    if fmt == "csv" or not have_openpyxl():
        path = Path(str(path_stem) + ".csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    cols = list(out.columns)
    n = len(out)

    header_fill = PatternFill("solid", fgColor="DDDDDD")
    input_fill = PatternFill("solid", fgColor="FFF2CC")
    for j, col in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=j, value=col)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(j)].width = COLUMN_WIDTHS.get(col, 20)

    for i, row in enumerate(out.itertuples(index=False), start=2):
        for j, value in enumerate(row, start=1):
            cell = ws.cell(row=i, column=j, value=value)
            # A response starting with "=" must stay text, not become a formula.
            if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
                cell.data_type = "s"
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    last = max(n + 1, 2)
    first_input_letter = None
    for col, choices in input_cols.items():
        letter = get_column_letter(cols.index(col) + 1)
        first_input_letter = first_input_letter or letter
        if choices:
            dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"',
                                allow_blank=True, showErrorMessage=True,
                                errorTitle="Invalid value",
                                error="Choose one of: " + ", ".join(choices))
            ws.add_data_validation(dv)
            dv.add(f"{letter}2:{letter}{last}")
    # Highlight the first input column (the label) while it is still empty.
    if first_input_letter and n:
        ws.conditional_formatting.add(
            f"{first_input_letter}2:{first_input_letter}{last}",
            FormulaRule(formula=[f"ISBLANK({first_input_letter}2)"], fill=input_fill))
    ws.freeze_panes = "A2"

    if instructions:
        info = wb.create_sheet("instructions")
        info.column_dimensions["A"].width = 110
        for i, line in enumerate(instructions, start=1):
            c = info.cell(row=i, column=1, value=line)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        if first_input_letter and n:
            r = len(instructions) + 2
            info.cell(row=r, column=1,
                      value=f'="Progress: "&COUNTA(items!{first_input_letter}2:'
                            f'{first_input_letter}{last})&" / {n} labelled"')

    path = Path(str(path_stem) + ".xlsx")
    wb.save(path)
    return path


def read_sheet(path_stem: Path) -> pd.DataFrame | None:
    """Read a (filled) sheet written by write_sheet; xlsx preferred, csv fallback."""
    xlsx = Path(str(path_stem) + ".xlsx")
    csv = Path(str(path_stem) + ".csv")
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name="items", dtype=str, keep_default_na=False)
    elif csv.exists():
        df = pd.read_csv(csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    else:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


def normalise_labels(series: pd.Series, where: str) -> pd.Series:
    """Map free-typed labels to the canonical set; '' stays '' (not yet labelled)."""
    raw = series.astype(str).str.strip().str.upper()
    out = raw.map(lambda v: "" if v == "" else LABEL_ALIASES.get(v, "__BAD__"))
    bad = sorted(set(raw[out == "__BAD__"]))
    if bad:
        raise SystemExit(f"[error] {where}: unrecognised label value(s): {bad}. "
                         f"Allowed: {LABELS}")
    return out


def cohen_kappa(a, b) -> float:
    """Unweighted Cohen's kappa for two equal-length label sequences."""
    a = pd.Series(list(a)).reset_index(drop=True)
    b = pd.Series(list(b)).reset_index(drop=True)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)
