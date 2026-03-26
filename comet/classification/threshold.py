"""
threshold.py - COMET classification utilities

Utilities for converting per-cell NIMBUS marker scores into downstream
classification outputs.

This module supports four stages:
    1. Compute per-marker thresholds from NIMBUS scores using OTSU * factor.
    2. Plot threshold overlays for manual review.
    3. Convert marker scores into is_Pos_<marker> calls, with optional
       mutual-exclusion correction for user-specified marker pairs.
    4. Apply ordered user-defined class rules to produce a primary Cell_Type
       plus any secondary matches stored in Additional_Labels.

Key behavior:
    - NIMBUS columns are looked up by marker name unless col_map overrides them.
    - Per-marker factors default to 1.0 unless explicitly provided.
    - For a mutex pair, only the weaker of the two raw marker scores is flipped.
    - For cell_type_rules, the first matched rule becomes Cell_Type and any
      later matches are concatenated into Additional_Labels. Place more
      specific rules before broader parent classes.
    - If no class rule matches, Cell_Type defaults to "Unknown".

No arcsinh transform is applied. NIMBUS outputs are expected to be probability-
like scores in [0, 1], and thresholding is performed directly on those values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from typing import Dict, List, Optional, Tuple

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
        Marker names to threshold.
    factors : dict, optional
        Per-marker factor values. Unspecified markers default to 1.0.
        Only need to supply markers you want to override.
    col_map : dict, optional
        Mapping from marker name to NIMBUS column name.
        Example: {'TCR': 'TCRbeta'} if your NIMBUS output uses TCRbeta instead of TCR.
        If None, markers are looked up by their own names.

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
        factor = factors.get(marker, 1.0)

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
# Step 3: Classify markers (with optional mutex correction)
# ---------------------------------------------------------------------------

def classify_all_markers(
    cell_table: pd.DataFrame,
    threshold_info: Dict[str, dict],
    mutex_pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Classify cells as positive/negative for each marker.

    Optionally applies mutual-exclusion correction to configured marker pairs.
    For each pair, if a cell is double-positive for both markers, the one with
    the lower raw probability is flipped to negative.

    Parameters
    ----------
    cell_table : pd.DataFrame
        NIMBUS cell table.
    threshold_info : dict
        Output of compute_thresholds().
    mutex_pairs : list of tuple(str, str), optional
        Marker pairs that should not remain double-positive.
        Example: [('CD4', 'CD8a'), ('CD3', 'CD68')]

    Returns
    -------
    pd.DataFrame
        Cell table with is_Pos_<marker> boolean columns appended.
        Adds Mutex_Correction if any configured pair is corrected.
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

    if mutex_pairs:
        result["Mutex_Correction"] = "None"

        for marker_a, marker_b in mutex_pairs:
            pos_a = f"is_Pos_{marker_a}"
            pos_b = f"is_Pos_{marker_b}"

            if marker_a not in threshold_info or marker_b not in threshold_info:
                print(
                    f"  [threshold] Warning: mutex pair ({marker_a}, {marker_b}) skipped "
                    f"because one or both markers were not thresholded."
                )
                continue

            if pos_a not in result.columns or pos_b not in result.columns:
                print(
                    f"  [threshold] Warning: mutex pair ({marker_a}, {marker_b}) skipped "
                    f"because one or both positivity columns are missing."
                )
                continue

            col_a = threshold_info[marker_a]["col_name"]
            col_b = threshold_info[marker_b]["col_name"]
            is_dp = result[pos_a] & result[pos_b]

            if not is_dp.any():
                print(f"  [threshold] {marker_a}/{marker_b} mutex correction: 0 double-positive cells corrected")
                continue

            a_weaker = is_dp & (result[col_a] < result[col_b])
            b_weaker = is_dp & (result[col_b] < result[col_a])
            result.loc[a_weaker, pos_a] = False
            result.loc[b_weaker, pos_b] = False

            pair_label = f"Corrected_{marker_a}_{marker_b}"
            corrected_mask = a_weaker | b_weaker
            result.loc[corrected_mask, "Mutex_Correction"] = result.loc[
                corrected_mask, "Mutex_Correction"
            ].map(lambda x: pair_label if x == "None" else f"{x};{pair_label}")

            n_corrected = int(corrected_mask.sum())
            print(
                f"  [threshold] {marker_a}/{marker_b} mutex correction: "
                f"{n_corrected} double-positive cells corrected"
            )

    return result


# ---------------------------------------------------------------------------
# Step 4: Assign Cell_Type and Additional_Labels
# ---------------------------------------------------------------------------

def assign_cell_types(
    cell_table: pd.DataFrame,
    cell_type_rules: Optional[List[dict]] = None,
    default_label: str = "Unknown",
    additional_labels_col: str = "Additional_Labels",
) -> pd.DataFrame:
    """
    Assign Cell_Type labels using user-provided class rules.

    Requires is_Pos_* columns from classify_all_markers().

    Parameters
    ----------
    cell_table : pd.DataFrame
        Cell table with is_Pos_* columns.
    cell_type_rules : list of dict, optional
        Ordered class rules. Each rule should define:
        {"name": str, "positive": [markers], "negative": [markers]}
        Because the first matched rule becomes Cell_Type, place more specific
        rules before broader parent classes.
        Example: [{"name": "MyClass", "positive": ["CD3"], "negative": ["CD68"]}]
    default_label : str, optional
        Label assigned to cells that do not match any class rule.
    additional_labels_col : str, optional
        Column name used to store secondary rule matches as a semicolon-delimited string.

    Returns
    -------
    pd.DataFrame
        Cell table with Cell_Type and Additional_Labels columns appended.
    """
    result = cell_table.copy()
    primary_labels = pd.Series(default_label, index=result.index, dtype="object")
    additional_labels = pd.Series("", index=result.index, dtype="object")

    if cell_type_rules:
        matched_labels = pd.Series([[] for _ in range(len(result))], index=result.index, dtype="object")

        for rule in cell_type_rules:
            name = rule.get("name", "CellType")
            pos_cols = [f"is_Pos_{m}" for m in rule.get("positive", [])]
            neg_cols = [f"is_Pos_{m}" for m in rule.get("negative", [])]

            mask = pd.Series(True, index=result.index)
            for c in pos_cols:
                if c in result.columns:
                    mask &= result[c]
                else:
                    mask &= False
            for c in neg_cols:
                if c in result.columns:
                    mask &= ~result[c]
                else:
                    mask &= False

            for idx in result.index[mask]:
                matched_labels.at[idx] = matched_labels.at[idx] + [name]

        matched_any = matched_labels.map(bool)
        primary_labels.loc[matched_any] = matched_labels.loc[matched_any].map(lambda labels: labels[0])
        additional_labels.loc[matched_any] = matched_labels.loc[matched_any].map(
            lambda labels: ";".join(labels[1:]) if len(labels) > 1 else ""
        )
    else:
        print("  [threshold] No cell_type_rules provided; assigning default Cell_Type only.")

    result["Cell_Type"] = primary_labels
    result[additional_labels_col] = additional_labels

    print("\n  [threshold] Cell type summary:")
    counts = result["Cell_Type"].value_counts()
    total = len(result)
    for ct, n in counts.items():
        print(f"    {ct}: {n} ({100*n/total:.2f}%)")

    if additional_labels.ne("").any():
        n_multi = int(additional_labels.ne("").sum())
        print(f"  [threshold] Cells with additional labels: {n_multi}/{len(result)}")

    return result

# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def threshold_slide(
    nimbus_csv: str,
    markers: List[str],
    factors: Optional[Dict[str, float]] = None,
    col_map: Optional[Dict[str, str]] = None,
    mutex_pairs: Optional[List[Tuple[str, str]]] = None,
    cell_type_rules: Optional[List[dict]] = None,
    manual_thresholds: Optional[Dict[str, float]] = None,
    output_csv: Optional[str] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Full thresholding pipeline for one slide.

    Workflow:
        1. Compute per-marker thresholds (OTSU * factor)
        2. Apply any manual overrides
        3. Plot distributions for visual verification
        4. Classify all markers (with optional mutex correction)
        5. Optionally assign Cell_Type labels using user-provided class rules
        6. Save results and thresholds used

    Parameters
    ----------
    nimbus_csv : str
        Path to NIMBUS cell table CSV.
    markers : list of str
        Markers to threshold.
    factors : dict, optional
        Per-marker factor values. Unspecified markers default to 1.0.
        Example: {'EOMES': 1.6, 'CD4': 0.4}
    col_map : dict, optional
        Marker -> NIMBUS column name mapping.
        Only needed if NIMBUS output column names differ from the marker names.
        Example: {'TCR': 'TCRbeta'} if your NIMBUS output uses TCRbeta instead of TCR.
        If None, markers are looked up by their own names.
    mutex_pairs : list of tuple(str, str), optional
        Marker pairs that should not remain double-positive.
        Example: [('CD4', 'CD8a'), ('CD3', 'CD68')]
    cell_type_rules : list of dict, optional
        Ordered Cell_Type rules supplied by the user/frontend.
        Each rule should define:
        {"name": str, "positive": [markers], "negative": [markers]}
        The first matched rule becomes Cell_Type; any later matches are stored
        in Additional_Labels. Place more specific rules before broader parent
        classes.
        Example: [{"name": "MyClass", "positive": ["CD3"], "negative": ["CD68"]}]
    manual_thresholds : dict, optional
        Override computed thresholds for specific markers (post-visualization).
        Example: {'EOMES': 0.42}
    output_csv : str, optional
        Save classified table here.
        Defaults to nimbus_csv stem + '_classified.csv'.
    plot : bool
        Save distribution plot for verification (default True).

    Returns
    -------
    pd.DataFrame
        Classified cell table with is_Pos_*, Cell_Type, and Additional_Labels columns.
    """
    print(f"[threshold] Loading: {nimbus_csv}")
    cell_table = pd.read_csv(nimbus_csv)

    # Step 1: Compute thresholds
    print("[threshold] Computing OTSU * factor thresholds ...")
    threshold_info = compute_thresholds(cell_table, markers, factors, col_map)

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
    cell_table = classify_all_markers(cell_table, threshold_info, mutex_pairs=mutex_pairs)

    # Step 5: Cell types
    print("[threshold] Assigning cell types ...")
    cell_table = assign_cell_types(cell_table, cell_type_rules=cell_type_rules)

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
