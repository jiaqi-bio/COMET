"""
threshold.py - COMET classification step 1

Classifies each segmented cell as positive or negative for each marker
using OTSU * factor thresholding, then assigns cell type labels.

Algorithm (from Code6):
    1. For each marker, compute OTSU threshold on raw NIMBUS probabilities.
    2. Final threshold = otsu_val * factor
       (factor < 1.0 lowers the bar; factor > 1.0 raises it)
    3. CD4/CD8 mutual exclusion: if a cell is double-positive for both,
       flip the weaker signal to negative.
    4. Assign cell type labels based on combinatorial marker status.

No arcsinh transform is applied. NIMBUS outputs probabilities in [0, 1];
arcsinh was only used for visualization in earlier analyses and does not
improve thresholding accuracy on probability data.

Default factors (from Code6 grid search):
    CD3=1.0, TCR=0.6, EOMES=1.6, CD4=0.4, CD8a=0.5, CD45=1.0

NIMBUS column mapping (Prob_ prefix, e.g. CD8a -> Prob_CD8a, TCR -> Prob_TCR):
    column_map = {
        'CD3':   'Prob_CD3',
        'CD4':   'Prob_CD4',
        'CD8a':  'Prob_CD8a',
        'TCR':   'Prob_TCR',
        'EOMES': 'Prob_EOMES',
        'CD45':  'Prob_CD45',
    }
    Pass this as the col_map argument if your NIMBUS output uses this naming.
    If your columns are already named CD3, CD4, etc., leave col_map=None.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Default configuration matching Code6
# ---------------------------------------------------------------------------

DEFAULT_FACTORS: Dict[str, float] = {
    "CD3":   1.0,
    "TCR":   0.6,
    "EOMES": 1.6,
    "CD4":   0.4,
    "CD8a":  0.5,
    "CD45":  1.0,
}

# NIMBUS output column name for each marker
DEFAULT_COL_MAP: Dict[str, str] = {
    "CD3":   "Prob_CD3",
    "CD4":   "Prob_CD4",
    "CD8a":  "Prob_CD8a",
    "TCR":   "Prob_TCR",
    "EOMES": "Prob_EOMES",
    "CD45":  "Prob_CD45",
}


# ---------------------------------------------------------------------------
# Step 1: Compute OTSU * factor threshold per marker
# ---------------------------------------------------------------------------

def compute_otsu_threshold(
    values: np.ndarray,
    factor: float = 1.0,
) -> dict:
    """
    Compute threshold for one marker: otsu_val * factor.

    Parameters
    ----------
    values : np.ndarray
        Raw NIMBUS probability values for one marker.
    factor : float
        Multiplicative factor applied to the OTSU threshold.
        factor < 1.0 -> more cells positive (lower bar).
        factor > 1.0 -> fewer cells positive (higher bar).

    Returns
    -------
    dict with keys: otsu_t, factor, threshold
    """
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return {"otsu_t": 0.5, "factor": factor, "threshold": 0.5 * factor}

    try:
        otsu_t = float(threshold_otsu(vals))
    except Exception:
        otsu_t = float(np.mean(vals))

    threshold = float(np.clip(otsu_t * factor, 0.0, 1.0))

    return {"otsu_t": otsu_t, "factor": factor, "threshold": threshold}


def compute_thresholds(
    cell_table: pd.DataFrame,
    markers: List[str],
    factors: Optional[Dict[str, float]] = None,
    col_map: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """
    Compute OTSU * factor thresholds for all markers.

    Parameters
    ----------
    cell_table : pd.DataFrame
        NIMBUS cell table.
    markers : list of str
        Marker names to threshold (e.g. ['CD3', 'CD4', 'CD8a', 'TCR', 'EOMES']).
    factors : dict, optional
        Per-marker factor values. Defaults to DEFAULT_FACTORS.
        Only need to supply markers you want to override.
    col_map : dict, optional
        Mapping from marker name to NIMBUS column name.
        Example: {'CD8a': 'Prob_CD8'} if your NIMBUS output uses Prob_CD8 instead of Prob_CD8a.
        If None and DEFAULT_COL_MAP covers your markers, leave as None.

    Returns
    -------
    dict: marker -> {otsu_t, factor, threshold, col_name}
    """
    if factors is None:
        factors = {}
    if col_map is None:
        col_map = {}

    results = {}
    for marker in markers:
        col_name = col_map.get(marker, marker)
        factor = factors.get(marker, DEFAULT_FACTORS.get(marker, 1.0))

        if col_name not in cell_table.columns:
            print(f"  [threshold] Warning: column '{col_name}' not found, skipping {marker}.")
            continue

        vals = cell_table[col_name].values
        info = compute_otsu_threshold(vals, factor)
        info["col_name"] = col_name
        results[marker] = info

        print(
            f"  [threshold] {marker} (col={col_name}): "
            f"otsu={info['otsu_t']:.4f}, factor={factor}, "
            f"threshold={info['threshold']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 2: Plot distributions for verification
# ---------------------------------------------------------------------------

def plot_marker_thresholds(
    cell_table: pd.DataFrame,
    threshold_info: Dict[str, dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot NIMBUS probability distributions with threshold lines.

    Each panel shows:
    - Raw probability histogram
    - Orange dashed line: OTSU threshold
    - Red solid line: final threshold (OTSU * factor)
    - Title: marker name, positive count, percentage

    Parameters
    ----------
    cell_table : pd.DataFrame
        NIMBUS cell table.
    threshold_info : dict
        Output of compute_thresholds().
    save_path : str, optional
        Save figure here. If None, displays interactively.
    """
    markers = list(threshold_info.keys())
    n = len(markers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, marker in zip(axes, markers):
        info = threshold_info[marker]
        col_name = info["col_name"]

        if col_name not in cell_table.columns:
            ax.set_title(f"{marker} (not found)")
            ax.axis("off")
            continue

        values = cell_table[col_name].values
        threshold = info["threshold"]
        otsu_t = info["otsu_t"]

        ax.hist(values, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(otsu_t, color="orange", linewidth=1.5, linestyle="--",
                   label=f"OTSU = {otsu_t:.4f}")
        ax.axvline(threshold, color="red", linewidth=2,
                   label=f"x{info['factor']} = {threshold:.4f}")

        n_pos = (values > threshold).sum()
        pct = 100 * n_pos / len(values) if len(values) > 0 else 0
        ax.set_title(f"{marker}\n{n_pos}/{len(values)} positive ({pct:.2f}%)")
        ax.set_xlabel("NIMBUS probability")
        ax.set_ylabel("Cell count")
        ax.legend(fontsize=7)

    for ax in axes[len(markers):]:
        ax.axis("off")

    plt.suptitle(
        "Verify thresholds against raw fluorescence images before proceeding.",
        fontsize=10, color="darkred", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [threshold] Plot saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Step 3: Classify markers (with CD4/CD8 mutex correction)
# ---------------------------------------------------------------------------

def classify_all_markers(
    cell_table: pd.DataFrame,
    threshold_info: Dict[str, dict],
) -> pd.DataFrame:
    """
    Classify cells as positive/negative for each marker.

    Applies CD4/CD8 mutual exclusion correction:
    If a cell is double-positive for both CD4 and CD8a, the one with the
    lower raw probability is flipped to negative. This matches Code6 logic.

    Parameters
    ----------
    cell_table : pd.DataFrame
        NIMBUS cell table.
    threshold_info : dict
        Output of compute_thresholds().

    Returns
    -------
    pd.DataFrame
        Cell table with is_Pos_<marker> boolean columns appended.
        Also adds Mutex_Correction column if both CD4 and CD8a are present.
    """
    result = cell_table.copy()

    # Raw classification
    for marker, info in threshold_info.items():
        col_name = info["col_name"]
        if col_name not in result.columns:
            continue
        result[f"is_Pos_{marker}"] = result[col_name] > info["threshold"]
        n_pos = result[f"is_Pos_{marker}"].sum()
        pct = 100 * n_pos / len(result) if len(result) > 0 else 0
        print(
            f"  [threshold] {marker} (t={info['threshold']:.4f}): "
            f"{n_pos}/{len(result)} positive ({pct:.2f}%)"
        )

    # CD4/CD8 mutual exclusion correction
    has_cd4 = "is_Pos_CD4" in result.columns
    has_cd8 = "is_Pos_CD8a" in result.columns
    if has_cd4 and has_cd8:
        cd4_col = threshold_info["CD4"]["col_name"]
        cd8_col = threshold_info["CD8a"]["col_name"]

        is_dp = result["is_Pos_CD4"] & result["is_Pos_CD8a"]
        result["Mutex_Correction"] = is_dp.map({True: "Corrected_DP", False: "None"})

        # Flip the weaker one to negative
        cd4_weaker = is_dp & (result[cd4_col] < result[cd8_col])
        cd8_weaker = is_dp & (result[cd8_col] < result[cd4_col])
        result.loc[cd4_weaker, "is_Pos_CD4"]  = False
        result.loc[cd8_weaker, "is_Pos_CD8a"] = False

        n_corrected = int(is_dp.sum())
        print(f"  [threshold] CD4/CD8a mutex correction: {n_corrected} double-positive cells corrected")

    return result


# ---------------------------------------------------------------------------
# Step 4: Assign cell type labels
# ---------------------------------------------------------------------------

def assign_cell_types(cell_table: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Cell_Type label based on marker status columns.

    Classification hierarchy (from Code6):
        CD3+ & TCR-                                         -> gd+T
        CD3+ & TCR+ & CD4+                                  -> ab+CD4+T
        CD3+ & TCR+ & CD8a+                                 -> ab+CD8+T
        CD3- & CD45+                                        -> CD45+CD3-
        CD3+ & TCR+ & CD4- & CD8a- & EOMES-                -> ab+EOMES-DNT
        CD3+ & TCR+ & CD4- & CD8a- & EOMES+                -> ab+EOMES+DNT
        CD45+  (fallback)                                   -> CD45+ Cells

    Requires is_Pos_* columns from classify_all_markers().

    Parameters
    ----------
    cell_table : pd.DataFrame
        Cell table with is_Pos_* columns.

    Returns
    -------
    pd.DataFrame
        Cell table with Cell_Type column appended.
    """
    result = cell_table.copy()

    def col(marker: str) -> pd.Series:
        c = f"is_Pos_{marker}"
        if c not in result.columns:
            return pd.Series(False, index=result.index)
        return result[c]

    cd3   = col("CD3")
    tcr   = col("TCR")
    cd4   = col("CD4")
    cd8a  = col("CD8a")
    eomes = col("EOMES")
    cd45  = col("CD45")

    conditions = [
        (cd3 & ~tcr,                             "gd+T"),
        (cd3 & tcr & cd4,                        "ab+CD4+T"),
        (cd3 & tcr & cd8a,                       "ab+CD8+T"),
        (~cd3 & cd45,                            "CD45+CD3-"),
        (cd3 & tcr & ~cd4 & ~cd8a & ~eomes,      "ab+EOMES-DNT"),
        (cd3 & tcr & ~cd4 & ~cd8a & eomes,       "ab+EOMES+DNT"),
        (cd45,                                   "CD45+ Cells"),
    ]

    cell_types = pd.Series("Unknown", index=result.index)
    # Apply in reverse so higher-priority rules overwrite lower ones
    for mask, label in reversed(conditions):
        cell_types[mask] = label

    result["Cell_Type"] = cell_types

    # Summary
    print("\n  [threshold] Cell type summary:")
    counts = result["Cell_Type"].value_counts()
    total = len(result)
    for ct, n in counts.items():
        print(f"    {ct}: {n} ({100*n/total:.2f}%)")

    return result


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def threshold_slide(
    nimbus_csv: str,
    markers: List[str],
    factors: Optional[Dict[str, float]] = None,
    col_map: Optional[Dict[str, str]] = None,
    manual_thresholds: Optional[Dict[str, float]] = None,
    phenotypes: Optional[List[dict]] = None,
    output_csv: Optional[str] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Full thresholding pipeline for one slide.

    Workflow:
        1. Compute per-marker thresholds (OTSU * factor)
        2. Apply any manual overrides
        3. Plot distributions for visual verification
        4. Classify all markers (with CD4/CD8 mutex correction)
        5. Assign cell type labels
        6. Save results and thresholds used

    Parameters
    ----------
    nimbus_csv : str
        Path to NIMBUS cell table CSV.
    markers : list of str
        Markers to threshold (e.g. ['CD3', 'CD4', 'CD8a', 'TCR', 'EOMES', 'CD45']).
    factors : dict, optional
        Per-marker factor values. Unspecified markers use DEFAULT_FACTORS.
        Example: {'EOMES': 1.6, 'CD4': 0.4}
    col_map : dict, optional
        Marker -> NIMBUS column name mapping.
        Only needed if NIMBUS output column names differ from DEFAULT_COL_MAP.
        Example: {'CD8a': 'Prob_CD8'} if your NIMBUS output uses Prob_CD8 instead of Prob_CD8a.
        Defaults to DEFAULT_COL_MAP for known markers, else uses marker name directly.
    manual_thresholds : dict, optional
        Override computed thresholds for specific markers (post-visualization).
        Example: {'EOMES': 0.42}
    phenotypes : list of dict, optional
        Additional custom phenotype definitions beyond default cell types.
        Each: {"name": str, "positive": [markers], "negative": [markers]}
    output_csv : str, optional
        Save classified table here.
        Defaults to nimbus_csv stem + '_classified.csv'.
    plot : bool
        Save distribution plot for verification (default True).

    Returns
    -------
    pd.DataFrame
        Classified cell table with is_Pos_* and Cell_Type columns.
    """
    print(f"[threshold] Loading: {nimbus_csv}")
    cell_table = pd.read_csv(nimbus_csv)

    # Build effective col_map: fill from DEFAULT_COL_MAP for known markers
    effective_col_map = {m: DEFAULT_COL_MAP[m] for m in markers if m in DEFAULT_COL_MAP}
    if col_map:
        effective_col_map.update(col_map)

    # Step 1: Compute thresholds
    print("[threshold] Computing OTSU * factor thresholds ...")
    threshold_info = compute_thresholds(cell_table, markers, factors, effective_col_map)

    # Step 2: Manual overrides
    if manual_thresholds:
        for marker, t in manual_thresholds.items():
            if marker in threshold_info:
                print(f"  [threshold] Manual override: {marker} = {t:.4f}")
                threshold_info[marker]["threshold"] = float(t)
            else:
                print(f"  [threshold] Warning: manual override for unknown marker '{marker}'")

    # Step 3: Plot
    if plot:
        plot_path = str(Path(nimbus_csv).parent / "threshold_distributions.png")
        plot_marker_thresholds(cell_table, threshold_info, save_path=plot_path)

    # Step 4: Classify
    print("[threshold] Classifying markers ...")
    cell_table = classify_all_markers(cell_table, threshold_info)

    # Step 5: Cell types
    print("[threshold] Assigning cell types ...")
    cell_table = assign_cell_types(cell_table)

    # Step 5b: Additional custom phenotypes if provided
    if phenotypes:
        for ph in phenotypes:
            name = ph.get("name", "phenotype")
            pos_cols = [f"is_Pos_{m}" for m in ph.get("positive", [])]
            neg_cols = [f"is_Pos_{m}" for m in ph.get("negative", [])]
            mask = pd.Series(True, index=cell_table.index)
            for c in pos_cols:
                if c in cell_table.columns:
                    mask &= cell_table[c]
            for c in neg_cols:
                if c in cell_table.columns:
                    mask &= ~cell_table[c]
            cell_table[name] = mask
            n = mask.sum()
            pct = 100 * n / len(cell_table) if len(cell_table) > 0 else 0
            print(f"  [threshold] {name}: {n}/{len(cell_table)} ({pct:.3f}%)")

    # Step 6: Save
    if output_csv is None:
        p = Path(nimbus_csv)
        output_csv = str(p.parent / (p.stem + "_classified.csv"))

    # Save threshold details for reproducibility
    rows = []
    for marker, info in threshold_info.items():
        rows.append({
            "marker":    marker,
            "col_name":  info["col_name"],
            "otsu_t":    info["otsu_t"],
            "factor":    info["factor"],
            "threshold": info["threshold"],
        })
    threshold_csv = str(Path(output_csv).parent / "thresholds_used.csv")
    pd.DataFrame(rows).to_csv(threshold_csv, index=False)
    print(f"[threshold] Thresholds saved: {threshold_csv}")

    cell_table.to_csv(output_csv, index=False)
    print(f"[threshold] Results saved: {output_csv}")

    return cell_table
