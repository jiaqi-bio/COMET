"""
qupath_export.py - COMET export step

Exports classified cell table to QuPath-compatible TSV for visualization.
Cell classifications (Cell_Type, is_Pos_* columns) are displayed as
QuPath detection measurements overlaid on the fluorescence image.

Workflow:
    1. Run threshold_slide() -> nimbus_cell_table_classified.csv
    2. export_to_qupath(classified_csv) -> _qupath.tsv
    3. In QuPath: File -> Import -> Import detections from TSV
    4. Each cell appears as a point annotation colored by Cell_Type
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional


def export_to_qupath(
    classified_csv: str,
    output_path: Optional[str] = None,
    x_col: str = "x",
    y_col: str = "y",
    cell_id_col: str = "label",
    phenotype_cols: Optional[List[str]] = None,
) -> str:
    """
    Export classified cell table to a QuPath-compatible TSV file.

    Import into QuPath via:
        File -> Import -> Import detections from TSV

    Parameters
    ----------
    classified_csv : str
        Path to classified cell table CSV (output of threshold_slide()).
    output_path : str, optional
        Path for output TSV. Defaults to classified_csv stem + '_qupath.tsv'.
    x_col : str
        Column name for cell centroid X coordinate (default 'x').
    y_col : str
        Column name for cell centroid Y coordinate (default 'y').
    cell_id_col : str
        Column name for cell ID.
        NIMBUS outputs this column as 'label' (default).
        Change if your cell table uses a different column name.
    phenotype_cols : list of str, optional
        Columns to use for QuPath Class label.
        Defaults to ['Cell_Type'] if present, else first boolean column found.

    Returns
    -------
    str
        Path to the exported TSV file.
    """
    print(f"[qupath_export] Loading: {classified_csv}")
    df = pd.read_csv(classified_csv)

    for col in [x_col, y_col, cell_id_col]:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found. "
                f"Available: {list(df.columns)}"
            )

    out = pd.DataFrame()
    out["Image"]   = ""
    out["Name"]    = df[cell_id_col].astype(str)
    out["Class"]   = _assign_class(df, phenotype_cols)
    out["ROI"]     = [f"Point ({xi}, {yi})" for xi, yi in zip(df[x_col], df[y_col])]
    out["cell_id"] = df[cell_id_col]

    # Include is_Pos_* columns as 0/1 measurements
    for col in [c for c in df.columns if c.startswith("is_Pos_")]:
        out[col] = df[col].astype(int)

    # Include Mutex_Correction if present
    if "Mutex_Correction" in df.columns:
        out["Mutex_Correction"] = df["Mutex_Correction"]

    # Include raw NIMBUS probability columns
    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    for col in prob_cols:
        out[col] = df[col]

    if output_path is None:
        p = Path(classified_csv)
        output_path = str(p.parent / (p.stem + "_qupath.tsv"))

    out.to_csv(output_path, sep="\t", index=False)
    print(f"[qupath_export] {len(out)} cells -> {output_path}")
    print(f"[qupath_export] Import: QuPath -> File -> Import -> Import detections from TSV")

    return output_path


def _assign_class(df: pd.DataFrame, phenotype_cols: Optional[List[str]]) -> pd.Series:
    """Assign QuPath class string. Prefers Cell_Type column if available."""
    if phenotype_cols is not None:
        for col in phenotype_cols:
            if col in df.columns:
                return df[col].astype(str)

    if "Cell_Type" in df.columns:
        return df["Cell_Type"].astype(str)

    # Fallback: first boolean-like column
    bool_cols = [c for c in df.columns if c.startswith("is_Pos_")]
    if bool_cols:
        classes = pd.Series("Other", index=df.index)
        for col in bool_cols:
            classes[df[col] == True] = col.replace("is_Pos_", "")
        return classes

    return pd.Series("Unknown", index=df.index)


def write_qupath_selection_script(
    output_dir: str,
    markers: List[str],
) -> str:
    """
    Write a QuPath Groovy script to export selected cell IDs.

    This script is for manual inspection purposes: select cells in QuPath
    and export their IDs to a CSV for ad-hoc review.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV and script will be saved.
    markers : list of str
        Marker names (used for comments in the script).

    Returns
    -------
    str
        Path to the saved Groovy script.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_comments = "\n".join([
        f"//   {m}" for m in markers
    ])

    script = f'''// COMET - QuPath Cell Export Script
// ====================================
// Exports selected cell IDs and their Cell_Type classifications
// to a CSV for manual inspection or cross-validation.
//
// Markers in this experiment:
{marker_comments}
//
// Usage:
//   1. Import COMET detections (_qupath.tsv) into QuPath.
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
        writer.writeLine("cell_id,cell_type,x,y")
    }}
    selected.each {{ cell ->
        def cellId   = cell.getName()
        def cellType = cell.getPathClass()?.toString() ?: "Unknown"
        def roi      = cell.getROI()
        def cx       = roi?.getCentroidX() ?: 0
        def cy       = roi?.getCentroidY() ?: 0
        writer.writeLine("${{cellId}},${{cellType}},${{cx}},${{cy}}")
    }}
}}

print "Exported ${{selected.size()}} cells -> ${{OUTPUT_PATH}}"
'''

    script_path = output_dir / "export_selected_cells.groovy"
    script_path.write_text(script)
    print(f"[qupath_export] Script saved: {script_path}")

    return str(script_path)
