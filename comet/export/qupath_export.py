"""
qupath_export.py - COMET export step

Exports classified cell tables to comma-delimited files keyed by FOV and label.
These tables keep COMET/NIMBUS identifiers together with the exported Class,
Additional_Labels, is_Pos_* calls, and raw marker scores for downstream review
or integration with the QuPath mask-based import workflow.

Workflow:
    1. Run threshold_slide() -> nimbus_cell_table_classified.csv
    2. export_to_qupath(classified_csv) -> _qupath.csv
    3. Use the exported CSV as a fov/label-keyed measurement table
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def export_to_qupath(
    classified_csv: str,
    output_path: Optional[str] = None,
    fov_col: str = "fov",
    cell_id_col: str = "label",
    class_cols: Optional[List[str]] = None,
    phenotype_cols: Optional[List[str]] = None,
) -> str:
    """
    Export classified cell table to a fov/label-keyed CSV file.

    This export does not create ROI coordinates. It is intended for workflows
    that identify cells by FOV and label rather than per-cell x/y centroids.

    Parameters
    ----------
    classified_csv : str
        Path to classified cell table CSV (output of threshold_slide()).
    output_path : str, optional
        Path for output CSV. Defaults to classified_csv stem + "_qupath.csv".
    fov_col : str
        Column name for field-of-view identifier (default "fov").
    cell_id_col : str
        Column name for cell ID. NIMBUS outputs this column as "label" by default.
    class_cols : list of str, optional
        Columns to use for the exported Class label.
        Defaults to ["Cell_Type"] if present, else the first is_Pos_* column found.
    phenotype_cols : list of str, optional
        Deprecated alias for class_cols. Retained for backward compatibility.

    Returns
    -------
    str
        Path to the exported CSV file.
    """
    print(f"[qupath_export] Loading: {classified_csv}")
    df = pd.read_csv(classified_csv)

    for col in [fov_col, cell_id_col]:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found. "
                f"Available: {list(df.columns)}"
            )

    resolved_class_cols = _resolve_class_cols(class_cols, phenotype_cols)

    out = pd.DataFrame()
    out["fov"] = df[fov_col].astype(str)
    out["label"] = df[cell_id_col].astype(str)
    out["Class"] = _assign_class(df, resolved_class_cols)

    for col in [c for c in df.columns if c.startswith("is_Pos_")]:
        out[col] = df[col].astype(int)

    if "Mutex_Correction" in df.columns:
        out["Mutex_Correction"] = df["Mutex_Correction"]
    if "Additional_Labels" in df.columns:
        out["Additional_Labels"] = df["Additional_Labels"]

    marker_cols = []
    for pos_col in [c for c in df.columns if c.startswith("is_Pos_")]:
        marker = pos_col.replace("is_Pos_", "")
        if marker in df.columns and marker not in marker_cols:
            marker_cols.append(marker)

    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    for col in marker_cols + [c for c in prob_cols if c not in marker_cols]:
        out[col] = df[col]

    if output_path is None:
        p = Path(classified_csv)
        output_path = str(p.parent / f"{p.stem}_qupath.csv")

    out.to_csv(output_path, index=False)
    print(f"[qupath_export] {len(out)} cells -> {output_path}")
    print("[qupath_export] Exported a fov/label-keyed CSV without x/y coordinates.")
    return output_path


def _resolve_class_cols(
    class_cols: Optional[List[str]],
    phenotype_cols: Optional[List[str]],
) -> Optional[List[str]]:
    """Normalize legacy phenotype_cols input to the current class_cols name."""
    if class_cols is not None and phenotype_cols is not None and class_cols != phenotype_cols:
        raise ValueError("Pass only one of class_cols or phenotype_cols, not both.")
    return class_cols if class_cols is not None else phenotype_cols


def _assign_class(df: pd.DataFrame, class_cols: Optional[List[str]]) -> pd.Series:
    """Assign exported class string. Prefers Cell_Type column if available."""
    if class_cols is not None:
        for col in class_cols:
            if col in df.columns:
                return df[col].astype(str)

    if "Cell_Type" in df.columns:
        return df["Cell_Type"].astype(str)

    bool_cols = [c for c in df.columns if c.startswith("is_Pos_")]
    if bool_cols:
        classes = pd.Series("Other", index=df.index)
        for col in bool_cols:
            classes[df[col] == True] = col.replace("is_Pos_", "")
        return classes

    return pd.Series("Unknown", index=df.index)


def write_qupath_selection_script(output_dir: str, markers: List[str]) -> str:
    """Write a QuPath Groovy script to export selected cell IDs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_comments = "\n".join([f"//   {m}" for m in markers])
    script = f"""// COMET - QuPath Cell Export Script
// ====================================
// Exports selected cell IDs and their Cell_Type classifications
// to a CSV for manual inspection or cross-validation.
//
// Markers in this experiment:
{marker_comments}
//
// Usage:
//   1. Import COMET detections (_qupath.csv) into QuPath.
//   2. Select cells of interest using QuPath selection tools.
//   3. Run this script (Automate -> Script Editor -> Run).
//   4. Check the exported CSV in your output directory.

import qupath.lib.objects.PathDetectionObject

def OUTPUT_PATH = "{output_dir}/selected_cells.csv"

def selected = getSelectedObjects()
    .findAll {{ it instanceof PathDetectionObject }}

if (selected.isEmpty()) {{
    print "No cells selected."
    return
}}

def file = new File(OUTPUT_PATH)
def isNew = !file.exists()

file.withWriterAppend {{ writer ->
    if (isNew) {{
        writer.writeLine("cell_id,cell_type")
    }}
    selected.each {{ cell ->
        def cellId   = cell.getName()
        def cellType = cell.getPathClass()?.toString() ?: "Unknown"
        writer.writeLine("${{cellId}},${{cellType}}")
    }}
}}

print "Exported ${{selected.size()}} cells -> ${{OUTPUT_PATH}}"
"""

    script_path = output_dir / "export_selected_cells.groovy"
    script_path.write_text(script)
    print(f"[qupath_export] Script saved: {script_path}")
    return str(script_path)
