"""
COMET Validation Script — manuscript fig. S7

Compares three conditions:
  - Complete pipeline  : normalized signal fusion + OTSU × factor thresholds
  - Baseline A         : direct channel addition (no per-channel normalization)
  - Baseline B         : normalized fusion + raw OTSU (factor = 1.0 for all markers)

Outputs:
  {slide_dir}/validation_output/
    panel_b_segmentation.png      Layer 1: segmentation accuracy
    panel_c_classification.png    Layer 2: DNT end-to-end accuracy
    validation_metrics.csv        All numeric results

─────────────────────────────────────────────────────────────────────────────
ANNOTATION FILES REQUIRED (place in {slide_dir}/validation/)
─────────────────────────────────────────────────────────────────────────────

1. annotations_segmentation.csv   (Layer 1 — 10 FOVs)
   Columns: fov, centroid_x, centroid_y
   Source : QuPath → export detection centroids (manual cell boundary annotations)

2. annotations_classification.csv  (Layer 2 — 5 FOVs, ≥200 DNTs)
   Columns: fov, centroid_x, centroid_y, CD3, CD4, CD8a, EOMES, TCR
   Values : 1 = positive, 0 = negative
   IMPORTANT: Column names must EXACTLY match the --markers argument names.
              Use "TCR" (not "TCRbeta") to match the pipeline's marker naming.
   Source : QuPath → export detection measurements with manual marker classifications

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

python validation/validate.py \\
    --slide_dir path/to/Patient1 \\
    --nuclear_markers DAPI EOMES \\
    --membrane_markers CD3 CD4 CD8a TCRbeta \\
    --nimbus_channels CD3 CD4 CD8a TCRbeta EOMES CD45 \\
    --markers CD3 CD4 CD8a TCR EOMES CD45 \\
    --col_map TCR=Prob_TCRbeta

# Skip re-running the pipeline if it's already been run:
python validation/validate.py --slide_dir path/to/Patient1 --skip_pipeline
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage.measure import regionprops_table
import tifffile

# Allow running from repo root or from validation/
sys.path.insert(0, str(Path(__file__).parent.parent))
import comet


# ─── Annotation loading ───────────────────────────────────────────────────────

def load_segmentation_annotations(csv_path: str) -> dict:
    """
    Load Layer 1 annotations (cell centroids).
    Returns {fov_name: [(x, y), ...]}
    """
    df = pd.read_csv(csv_path)
    result = {}
    for fov, group in df.groupby("fov"):
        result[str(fov)] = list(zip(group["centroid_x"], group["centroid_y"]))
    return result


def load_classification_annotations(csv_path: str) -> dict:
    """
    Load Layer 2 annotations (per-cell marker classifications).
    Returns {fov_name: DataFrame}
    """
    df = pd.read_csv(csv_path)
    result = {}
    for fov, group in df.groupby("fov"):
        result[str(fov)] = group.reset_index(drop=True)
    return result


# ─── Cell centroid extraction ─────────────────────────────────────────────────

def extract_centroids_from_mask(mask_path: str) -> list:
    """Extract cell centroids (x, y) from a segmentation mask file."""
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[0]
    mask = mask.astype(np.int32)
    if mask.max() == 0:
        return []
    props = regionprops_table(mask, properties=["centroid"])
    # regionprops: centroid-0 = row (y), centroid-1 = col (x)
    return list(zip(props["centroid-1"], props["centroid-0"]))


# ─── Cell matching ────────────────────────────────────────────────────────────

def match_centroids(gt_centroids: list, pred_centroids: list, match_radius: float = 12.0):
    """
    Greedy nearest-neighbor matching between GT and predicted centroids.
    Each predicted cell can be matched to at most one GT cell.

    Returns
    -------
    matched_gt      : indices into gt_centroids that were matched
    matched_pred    : indices into pred_centroids that were matched (same order)
    unmatched_gt    : GT indices with no match  → False Negatives
    unmatched_pred  : predicted indices with no GT match → False Positives
    """
    if not gt_centroids or not pred_centroids:
        return [], [], list(range(len(gt_centroids))), list(range(len(pred_centroids)))

    pred_arr = np.array(pred_centroids)
    gt_arr = np.array(gt_centroids)
    tree = KDTree(pred_arr)
    distances, indices = tree.query(gt_arr, k=1)

    matched_gt, matched_pred, used_pred = [], [], set()
    for gt_idx, (dist, pred_idx) in enumerate(zip(distances, indices)):
        if dist <= match_radius and pred_idx not in used_pred:
            matched_gt.append(gt_idx)
            matched_pred.append(int(pred_idx))
            used_pred.add(int(pred_idx))

    unmatched_gt   = [i for i in range(len(gt_centroids))   if i not in set(matched_gt)]
    unmatched_pred = [i for i in range(len(pred_centroids)) if i not in used_pred]
    return matched_gt, matched_pred, unmatched_gt, unmatched_pred


def compute_prf(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ─── Layer 1: Segmentation metrics ───────────────────────────────────────────

def compute_segmentation_metrics(
    gt_annotations: dict,
    seg_dir: Path,
    match_radius: float = 12.0,
) -> dict:
    """
    Compute P/R/F1 for segmentation by comparing GT centroids to predicted masks.

    Parameters
    ----------
    gt_annotations : dict
        {fov_name: [(x, y), ...]}
    seg_dir : Path
        Directory containing FOV*_whole_cell.tif masks.
    match_radius : float
        Pixels within which a predicted cell counts as matching a GT cell.

    Returns
    -------
    dict : tp, fp, fn, precision, recall, f1, per_fov
    """
    total_tp = total_fp = total_fn = 0
    per_fov = {}

    for fov_name, gt_centroids in gt_annotations.items():
        mask_paths = list(seg_dir.glob(f"{fov_name}_whole_cell.tif"))
        if not mask_paths:
            print(f"  [validation] Warning: mask not found for {fov_name} in {seg_dir}")
            fn = len(gt_centroids)
            total_fn += fn
            per_fov[fov_name] = {"tp": 0, "fp": 0, "fn": fn, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        pred_centroids = extract_centroids_from_mask(str(mask_paths[0]))
        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_centroids(
            gt_centroids, pred_centroids, match_radius
        )
        tp, fp, fn = len(matched_gt), len(unmatched_pred), len(unmatched_gt)
        p, r, f1 = compute_prf(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_fov[fov_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "gt_count": len(gt_centroids),
            "pred_count": len(pred_centroids),
            "pred_centroids": pred_centroids,
        }

    p, r, f1 = compute_prf(total_tp, total_fp, total_fn)
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": p, "recall": r, "f1": f1, "per_fov": per_fov}


# ─── Layer 2: Classification metrics ─────────────────────────────────────────

def compute_classification_metrics(
    gt_annotations: dict,
    classified_csv: str,
    markers: list,
    col_map: dict = None,
    match_radius: float = 12.0,
) -> dict:
    """
    Compute per-marker and DNT P/R/F1 by matching GT annotations to classified cells.

    DNT definition: CD3+, CD4-, CD8a-, EOMES+, TCR+ (TCRbeta in antibody name)

    Parameters
    ----------
    gt_annotations : dict
        {fov_name: DataFrame} from load_classification_annotations()
    classified_csv : str
        Path to nimbus_cell_table_classified.csv
    markers : list
        Marker names (must exist as is_Pos_{marker} columns in classified CSV)
    col_map : dict, optional
        Map from marker to is_Pos column name override
    match_radius : float
        Matching radius in pixels

    Returns
    -------
    dict : per_marker, dnt, per_fov_counts
    """
    pred_df = pd.read_csv(classified_csv)

    # Detect coordinate column names
    x_col   = "centroid_x" if "centroid_x" in pred_df.columns else "x"
    y_col   = "centroid_y" if "centroid_y" in pred_df.columns else "y"
    fov_col = "fov"         if "fov"         in pred_df.columns else "SampleID"

    DNT_POS = ["CD3", "EOMES", "TCR"]   # TCR = TCRbeta in pipeline marker naming
    DNT_NEG = ["CD4", "CD8a"]

    per_marker_tp = {m: 0 for m in markers}
    per_marker_fp = {m: 0 for m in markers}
    per_marker_fn = {m: 0 for m in markers}
    dnt_tp = dnt_fp = dnt_fn = 0
    per_fov_counts = {}

    for fov_name, gt_df in gt_annotations.items():
        pred_fov = pred_df[pred_df[fov_col].astype(str) == str(fov_name)]
        if pred_fov.empty:
            # Fallback: match by numeric suffix
            fov_num = "".join(filter(str.isdigit, fov_name))
            pred_fov = pred_df[pred_df[fov_col].astype(str).str.endswith(fov_num)]

        gt_centroids   = list(zip(gt_df["centroid_x"], gt_df["centroid_y"]))
        pred_centroids = list(zip(pred_fov[x_col], pred_fov[y_col]))

        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_centroids(
            gt_centroids, pred_centroids, match_radius
        )
        pred_fov_reset = pred_fov.reset_index(drop=True)

        # Per-marker comparison on matched cell pairs
        for gt_idx, pred_idx in zip(matched_gt, matched_pred):
            gt_row   = gt_df.iloc[gt_idx]
            pred_row = pred_fov_reset.iloc[pred_idx]
            for marker in markers:
                gt_label   = int(gt_row.get(marker, 0))
                pos_col = f"is_Pos_{marker}"
                if pos_col not in pred_row.index:
                    
                    mapped_name = (col_map or {}).get(marker, marker).replace("Prob_", "")
                    pos_col = f"is_Pos_{mapped_name}"

                pred_label = int(pred_row.get(pos_col, 0))
                if   gt_label == 1 and pred_label == 1: per_marker_tp[marker] += 1
                elif gt_label == 0 and pred_label == 1: per_marker_fp[marker] += 1
                elif gt_label == 1 and pred_label == 0: per_marker_fn[marker] += 1

        # DNT: GT definition = CD3+, TCR+, CD4-, CD8a-, EOMES+ (manuscript target cell)
        # Prediction: use Cell_Type column (pipeline's authoritative output, includes
        # CD4/CD8 mutex correction). "ab+EOMES+DNT" = the EOMES+ DNT from the manuscript.
        def is_dnt_gt(row):
            return (all(int(row.get(m, 0)) == 1 for m in DNT_POS) and
                    all(int(row.get(m, 0)) == 0 for m in DNT_NEG))

        def is_dnt_pred(row):
            return str(row.get("Cell_Type", "")).strip() == "ab+EOMES+DNT"

        gt_dnt_mask   = [is_dnt_gt(r)   for _, r in gt_df.iterrows()]
        pred_dnt_mask = [is_dnt_pred(r) for _, r in pred_fov_reset.iterrows()]

        gt_dnt_centroids   = [c for c, m in zip(gt_centroids,   gt_dnt_mask)   if m]
        pred_dnt_centroids = [c for c, m in zip(pred_centroids, pred_dnt_mask) if m]

        m_gt, m_pred, um_gt, um_pred = match_centroids(
            gt_dnt_centroids, pred_dnt_centroids, match_radius
        )
        dnt_tp += len(m_gt)
        dnt_fp += len(um_pred)
        dnt_fn += len(um_gt)

        per_fov_counts[fov_name] = {
            "gt_dnt_count":   len(gt_dnt_centroids),
            "pred_dnt_count": len(pred_dnt_centroids),
        }

    per_marker = {}
    for m in markers:
        tp, fp, fn = per_marker_tp[m], per_marker_fp[m], per_marker_fn[m]
        p, r, f1 = compute_prf(tp, fp, fn)
        per_marker[m] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    dnt_p, dnt_r, dnt_f1 = compute_prf(dnt_tp, dnt_fp, dnt_fn)
    return {
        "per_marker": per_marker,
        "dnt": {"tp": dnt_tp, "fp": dnt_fp, "fn": dnt_fn,
                "precision": dnt_p, "recall": dnt_r, "f1": dnt_f1,
                "false_positive_count": dnt_fp},
        "per_fov_counts": per_fov_counts,
    }


# ─── Figures ──────────────────────────────────────────────────────────────────

def make_panel_b(
    gt_annotations: dict,
    seg_complete: dict,
    seg_baseline_a: dict,
    slide_dir: Path,
    output_dir: Path,
    vis_fov: str = None,
):
    """Panel B: side-by-side visual + P/R/F1 bar chart."""
    vis_fov = vis_fov or next(iter(gt_annotations.keys()))
    gt_cents = gt_annotations.get(vis_fov, [])

    # Load background image (DAPI or first available channel)
    dapi_path = slide_dir / "image_data" / vis_fov / "DAPI.tif"
    if not dapi_path.exists():
        ch_dir = slide_dir / "image_data" / vis_fov
        tif_files = sorted(ch_dir.glob("*.tif"))
        dapi_path = tif_files[0] if tif_files else None

    if dapi_path and dapi_path.exists():
        raw = tifffile.imread(str(dapi_path)).astype(float)
        p1, p99 = np.percentile(raw, [1, 99])
        bg = np.clip((raw - p1) / max(p99 - p1, 1), 0, 1)
    else:
        bg = np.zeros((1024, 1024))

    def load_mask(path):
        if not Path(path).exists():
            return np.zeros_like(bg, dtype=bool)
        m = tifffile.imread(str(path))
        return (m[0] if m.ndim > 2 else m) > 0

    mask_complete   = load_mask(slide_dir / "segmentation" / "deepcell_output"           / f"{vis_fov}_whole_cell.tif")
    mask_baseline_a = load_mask(slide_dir / "segmentation" / "deepcell_output_baseline_a" / f"{vis_fov}_whole_cell.tif")

    fig = plt.figure(figsize=(18, 8))

    # Visual panels (top row, 4 subplots)
    panels = [
        (bg, "Raw image",                         None),
        (bg, "Baseline A\n(direct addition)",     mask_baseline_a),
        (bg, "Complete pipeline\n(normalized)",   mask_complete),
        (bg, "Manual annotation\n(ground truth)", None),
    ]
    for i, (img, title, mask) in enumerate(panels):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            overlay = np.zeros((*img.shape, 4))
            overlay[mask] = [0.1, 0.5, 1.0, 0.4]
            ax.imshow(overlay)
        if gt_cents and i in (0, 3):
            gx, gy = zip(*gt_cents)
            ax.scatter(gx, gy, s=12, c="lime", marker="+", linewidths=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Bar chart (lower right)
    ax_bar = fig.add_subplot(1, 2, 2)
    conditions  = ["Complete pipeline", "Baseline A"]
    colors      = ["#2196F3", "#FF9800"]
    data_points = [
        [seg_complete["precision"],   seg_complete["recall"],   seg_complete["f1"]],
        [seg_baseline_a["precision"], seg_baseline_a["recall"], seg_baseline_a["f1"]],
    ]
    x = np.arange(3)
    w = 0.35
    for i, (cond, color, vals) in enumerate(zip(conditions, colors, data_points)):
        bars = ax_bar.bar(x + i * w, vals, w, label=cond, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + w / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax_bar.set_xticks(x + w / 2)
    ax_bar.set_xticklabels(["Precision", "Recall", "F1"])
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title(f"Segmentation accuracy\n(Layer 1, {len(gt_annotations)} FOVs)", fontsize=10)
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "panel_b_segmentation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [validation] Panel B → {out}")


def make_panel_c(
    cls_complete: dict,
    cls_baseline_a: dict,
    cls_baseline_b: dict,
    markers: list,
    output_dir: Path,
):
    """Panel C: per-marker heatmap + DNT scatter + Bland-Altman + F1 bar."""
    fig = plt.figure(figsize=(20, 10))

    conditions = ["Complete", "Baseline A", "Baseline B"]
    cls_list   = [cls_complete, cls_baseline_a, cls_baseline_b]
    colors     = ["#2196F3", "#FF9800", "#9C27B0"]

    prec_matrix = np.array([[c["per_marker"].get(m, {"precision": 0})["precision"] for m in markers] for c in cls_list])
    rec_matrix  = np.array([[c["per_marker"].get(m, {"recall":    0})["recall"]    for m in markers] for c in cls_list])

    # Heatmaps
    for subplot_idx, (matrix, title) in enumerate(
        [(prec_matrix, "Per-marker Precision"), (rec_matrix, "Per-marker Recall")]
    ):
        ax = fig.add_subplot(2, 3, subplot_idx * 3 + 1)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(markers)))
        ax.set_xticklabels(markers, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(len(conditions)):
            for j in range(len(markers)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Gather per-FOV DNT counts
    all_fovs  = sorted(cls_complete["per_fov_counts"].keys())
    gt_counts = [cls_complete["per_fov_counts"][f]["gt_dnt_count"] for f in all_fovs]

    # Scatter: manual count vs pipeline count
    ax_scatter = fig.add_subplot(1, 3, 2)
    for label, cls, color in zip(conditions, cls_list, colors):
        pred = [cls["per_fov_counts"].get(f, {"pred_dnt_count": 0})["pred_dnt_count"] for f in all_fovs]
        ax_scatter.scatter(gt_counts, pred, label=label, color=color, s=70, alpha=0.85)
    mx = max(max(gt_counts) if gt_counts else 1, 5) * 1.1
    ax_scatter.plot([0, mx], [0, mx], "k--", alpha=0.4, linewidth=1)
    ax_scatter.set_xlabel("Manual DNT count (per FOV)")
    ax_scatter.set_ylabel("Pipeline DNT count (per FOV)")
    ax_scatter.set_title("DNT count: manual vs pipeline", fontsize=10)
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # Bland-Altman
    ax_ba = fig.add_subplot(2, 3, 3)
    for label, cls, color in zip(conditions, cls_list, colors):
        pred = [cls["per_fov_counts"].get(f, {"pred_dnt_count": 0})["pred_dnt_count"] for f in all_fovs]
        means = [(g + p) / 2 for g, p in zip(gt_counts, pred)]
        diffs = [p - g       for g, p in zip(gt_counts, pred)]
        ax_ba.scatter(means, diffs, label=label, color=color, s=60, alpha=0.85)
    ax_ba.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_ba.set_xlabel("Mean (manual + pipeline)")
    ax_ba.set_ylabel("Bias (pipeline − manual)")
    ax_ba.set_title("Bland-Altman", fontsize=9)
    ax_ba.legend(fontsize=7)
    ax_ba.grid(alpha=0.3)

    # DNT F1 bar chart
    ax_f1 = fig.add_subplot(2, 3, 6)
    metric_labels = ["Precision", "Recall", "F1"]
    x = np.arange(3)
    w = 0.25
    for i, (label, cls, color) in enumerate(zip(conditions, cls_list, colors)):
        d = cls["dnt"]
        vals = [d["precision"], d["recall"], d["f1"]]
        bars = ax_f1.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax_f1.text(bar.get_x() + w / 2, bar.get_height() + 0.01,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax_f1.set_xticks(x + w)
    ax_f1.set_xticklabels(metric_labels)
    ax_f1.set_ylim(0, 1.2)
    ax_f1.set_title("DNT end-to-end accuracy", fontsize=9)
    ax_f1.legend(fontsize=7)
    ax_f1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "panel_c_classification.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [validation] Panel C → {out}")


# ─── Pipeline runners ─────────────────────────────────────────────────────────

def run_complete_pipeline(
    slide_dir, nuclear_markers, membrane_markers,
    nimbus_channels, markers, col_map, factors,
):
    print("\n[validation] ── Complete pipeline ──────────────────────────────")
    comet.prepare_cellpose_inputs(
        base_dir=str(slide_dir),
        nuclear_markers=nuclear_markers,
        membrane_markers=membrane_markers,
    )
    comet.run_cellpose_slide(str(slide_dir))
    comet.deduplicate_slide(str(slide_dir))
    comet.run_nimbus_slide(str(slide_dir), include_channels=nimbus_channels)
    comet.threshold_slide(
        nimbus_csv=str(slide_dir / "nimbus_output" / "nimbus_cell_table.csv"),
        markers=markers, col_map=col_map, factors=factors, plot=True,
    )


def run_baseline_a(
    slide_dir, nuclear_markers, membrane_markers,
    nimbus_channels, markers, col_map, factors,
):
    print("\n[validation] ── Baseline A (direct addition, no normalization) ──")
    comet.prepare_cellpose_inputs(
        base_dir=str(slide_dir),
        nuclear_markers=nuclear_markers,
        membrane_markers=membrane_markers,
        normalize=False,
        output_subdir="cellpose_input_baseline_a",
    )
    comet.run_cellpose_slide(
        str(slide_dir),
        input_subdir="cellpose_input_baseline_a",
        raw_subdir="cellpose_output_baseline_a",
        final_subdir="deepcell_output_baseline_a",
    )
    comet.deduplicate_slide(str(slide_dir), seg_subdir="deepcell_output_baseline_a")
    comet.run_nimbus_slide(
        str(slide_dir),
        include_channels=nimbus_channels,
        seg_subdir="deepcell_output_baseline_a",
        output_subdir="nimbus_output_baseline_a",
    )
    comet.threshold_slide(
        nimbus_csv=str(slide_dir / "nimbus_output_baseline_a" / "nimbus_cell_table.csv"),
        markers=markers, col_map=col_map, factors=factors, plot=False,
    )


def run_baseline_b(slide_dir, markers, col_map):
    print("\n[validation] ── Baseline B (raw OTSU, factor = 1.0) ────────────")
    nimbus_csv = slide_dir / "nimbus_output" / "nimbus_cell_table.csv"
    if not nimbus_csv.exists():
        raise FileNotFoundError(
            f"Baseline B needs complete pipeline NIMBUS output:\n  {nimbus_csv}\n"
            "Run complete pipeline first (or use --skip_pipeline if already done)."
        )
    output_csv = slide_dir / "nimbus_output" / "nimbus_cell_table_classified_baseline_b.csv"
    comet.threshold_slide(
        nimbus_csv=str(nimbus_csv),
        markers=markers, col_map=col_map,
        factors={m: 1.0 for m in markers},  # raw OTSU, no factor adjustment
        output_csv=str(output_csv),
        plot=False,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COMET validation — generates fig. S7 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slide_dir", required=True,
                        help="Path to slide directory (e.g. my_experiment/Patient1)")
    parser.add_argument("--nuclear_markers", nargs="+", default=["DAPI", "EOMES"],
                        help="Nuclear markers for signal_prep (default: DAPI EOMES)")
    parser.add_argument("--membrane_markers", nargs="+", default=["CD3", "CD4", "CD8a", "TCRbeta"],
                        help="Membrane markers for signal_prep")
    parser.add_argument("--nimbus_channels", nargs="+",
                        default=["CD3", "CD4", "CD8a", "TCRbeta", "EOMES", "CD45"],
                        help="Channels to include in NIMBUS")
    parser.add_argument("--markers", nargs="+",
                        default=["CD3", "CD4", "CD8a", "TCR", "EOMES", "CD45"],
                        help="Markers to threshold")
    parser.add_argument("--col_map", nargs="*", default=None,
                        help="Column remapping as KEY=VALUE pairs, e.g. TCR=Prob_TCRbeta")
    parser.add_argument("--match_radius", type=float, default=12.0,
                        help="Cell matching radius in pixels (default 12 = half cell diameter)")
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="Skip running the pipeline and go straight to metrics/figures")
    parser.add_argument("--vis_fov", default=None,
                        help="FOV to use for side-by-side visual in Panel B (default: first in annotation)")
    args = parser.parse_args()

    slide_dir  = Path(args.slide_dir)
    output_dir = slide_dir / "validation_output"
    output_dir.mkdir(exist_ok=True)

    # Parse column remapping
    col_map = None
    if args.col_map:
        col_map = {}
        for pair in args.col_map:
            k, v = pair.split("=", 1)
            col_map[k.strip()] = v.strip()

    # Default validated factors
    factors = {"CD3": 1.0, "TCR": 0.6, "EOMES": 1.6, "CD4": 0.4, "CD8a": 0.5, "CD45": 1.0}

    # ── Check annotation files ──────────────────────────────────────────────
    annot_dir       = slide_dir / "validation"
    seg_annot_path  = annot_dir / "annotations_segmentation.csv"
    cls_annot_path  = annot_dir / "annotations_classification.csv"

    missing = [p for p in [seg_annot_path, cls_annot_path] if not p.exists()]
    if missing:
        for p in missing:
            print(f"[validation] ERROR: annotation file not found: {p}")
        print("\nExpected formats:")
        print("  annotations_segmentation.csv  — columns: fov, centroid_x, centroid_y")
        print("  annotations_classification.csv — columns: fov, centroid_x, centroid_y, CD3, CD4, CD8a, EOMES, TCR")
        sys.exit(1)

    # ── Run pipeline conditions ─────────────────────────────────────────────
    if not args.skip_pipeline:
        run_complete_pipeline(slide_dir, args.nuclear_markers, args.membrane_markers,
                              args.nimbus_channels, args.markers, col_map, factors)
        run_baseline_a(slide_dir, args.nuclear_markers, args.membrane_markers,
                       args.nimbus_channels, args.markers, col_map, factors)
        run_baseline_b(slide_dir, args.markers, col_map)
    else:
        print("[validation] --skip_pipeline: using existing pipeline outputs.")

    # ── Load annotations ────────────────────────────────────────────────────
    print("\n[validation] Loading annotations ...")
    seg_gt = load_segmentation_annotations(str(seg_annot_path))
    cls_gt = load_classification_annotations(str(cls_annot_path))
    n_seg_cells = sum(len(v) for v in seg_gt.values())
    n_cls_cells = sum(len(v) for v in cls_gt.values())
    print(f"  Segmentation:   {n_seg_cells} cells  across {len(seg_gt)} FOVs")
    print(f"  Classification: {n_cls_cells} cells  across {len(cls_gt)} FOVs")

    # ── Layer 1: Segmentation metrics ───────────────────────────────────────
    print("\n[validation] Layer 1 — segmentation ...")
    seg_dir_complete   = slide_dir / "segmentation" / "deepcell_output"
    seg_dir_baseline_a = slide_dir / "segmentation" / "deepcell_output_baseline_a"

    seg_complete   = compute_segmentation_metrics(seg_gt, seg_dir_complete,   args.match_radius)
    seg_baseline_a = compute_segmentation_metrics(seg_gt, seg_dir_baseline_a, args.match_radius)

    for label, m in [("Complete  ", seg_complete), ("Baseline A", seg_baseline_a)]:
        print(f"  {label}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # ── Layer 2: Classification metrics ─────────────────────────────────────
    print("\n[validation] Layer 2 — classification ...")
    csv_complete   = slide_dir / "nimbus_output"           / "nimbus_cell_table_classified.csv"
    csv_baseline_a = slide_dir / "nimbus_output_baseline_a" / "nimbus_cell_table_classified.csv"
    csv_baseline_b = slide_dir / "nimbus_output"           / "nimbus_cell_table_classified_baseline_b.csv"

    cls_complete   = compute_classification_metrics(cls_gt, str(csv_complete),   args.markers, col_map, args.match_radius)
    cls_baseline_a = compute_classification_metrics(cls_gt, str(csv_baseline_a), args.markers, col_map, args.match_radius)
    cls_baseline_b = compute_classification_metrics(cls_gt, str(csv_baseline_b), args.markers, col_map, args.match_radius)

    print("\n  DNT results:")
    for label, cls in [("Complete  ", cls_complete), ("Baseline A", cls_baseline_a), ("Baseline B", cls_baseline_b)]:
        d = cls["dnt"]
        print(f"  {label}  P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}  FP={d['false_positive_count']}")

    # ── Save metrics table ──────────────────────────────────────────────────
    rows = []
    for cond, seg, cls in [
        ("Complete pipeline",       seg_complete,   cls_complete),
        ("Baseline A (no norm.)",   seg_baseline_a, cls_baseline_a),
        ("Baseline B (raw OTSU)",   seg_complete,   cls_baseline_b),  # same segmentation
    ]:
        rows.append({
            "Condition":      cond,
            "Seg_Precision":  seg["precision"], "Seg_Recall": seg["recall"], "Seg_F1": seg["f1"],
            "Seg_TP": seg["tp"], "Seg_FP": seg["fp"], "Seg_FN": seg["fn"],
            "DNT_Precision":  cls["dnt"]["precision"],
            "DNT_Recall":     cls["dnt"]["recall"],
            "DNT_F1":         cls["dnt"]["f1"],
            "DNT_FP":         cls["dnt"]["false_positive_count"],
        })

    metrics_path = output_dir / "validation_metrics.csv"
    pd.DataFrame(rows).to_csv(str(metrics_path), index=False)
    print(f"\n[validation] Metrics table → {metrics_path}")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n[validation] Generating figures ...")
    make_panel_b(seg_gt, seg_complete, seg_baseline_a, slide_dir, output_dir, vis_fov=args.vis_fov)
    make_panel_c(cls_complete, cls_baseline_a, cls_baseline_b, args.markers, output_dir)

    print(f"\n[validation] Done. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
