"""
deduplication.py - COMET segmentation step 3
Two-stage post-processing of CellposeSAM masks:

Stage 1 - Border clearing:
    Remove cells touching tile borders to avoid incomplete cells at edges.

Stage 2 - Overlap deduplication (COMET improvement):
    CellposeSAM has no built-in tile-stitching. After splitting a WSI into
    overlapping tiles, the same cell can be segmented in two adjacent tiles.
    This module builds a virtual global canvas, assigns globally unique cell
    IDs, and resolves duplicates in overlap regions using a configurable
    overlap threshold. The deduplicated masks are re-exported per-FOV for
    downstream NIMBUS classification.

Reads:  segmentation/deepcell_output/FOV*_whole_cell.tif
Writes: segmentation/deepcell_output/FOV*_whole_cell.tif  (in-place, original backed up)

CSV columns expected: FOV_Name, X, Y, Width, Height
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from tqdm import tqdm
from typing import Optional

warnings.filterwarnings("ignore", message=".*TiffPage.*")


# ---------------------------------------------------------------------------
# Stage 1: Border clearing
# ---------------------------------------------------------------------------

def clear_tile_borders(
    input_dir: str,
    output_dir: str,
    buffer_size: int = 2,
) -> None:
    """
    Remove segmented cells that touch tile borders.

    Parameters
    ----------
    input_dir : str
        Directory containing masks (FOV*_whole_cell.tif).
    output_dir : str
        Directory to save border-cleared masks.
    buffer_size : int
        Pixel buffer from edge to consider as border (default 2).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_files = sorted(input_dir.glob("FOV*_whole_cell.tiff"))
    if not cell_files:
        raise FileNotFoundError(f"No 'FOV*_whole_cell.tiff' found in {input_dir}")

    success = 0
    for file_path in cell_files:
        try:
            mask = tifffile.imread(str(file_path))
            cleared = clear_border(mask, buffer_size=buffer_size)
            tifffile.imwrite(str(output_dir / file_path.name), cleared)
            success += 1
        except Exception as e:
            print(f"  [dedup] Warning: failed to process {file_path.name}: {e}")

    print(f"[dedup] Border clearing: {success}/{len(cell_files)} masks -> {output_dir}")


# ---------------------------------------------------------------------------
# Stage 2: Overlap deduplication
# ---------------------------------------------------------------------------

def _build_global_canvas(
    csv_file: Path,
    tile_dir: Path,
    tiff_pattern: str,
    overlap_threshold: float,
) -> tuple:
    """
    Stitch all FOV masks onto a virtual global canvas and resolve duplicates.

    Returns (global_mask, tile_info list, global_id_ranges dict).
    """
    df = pd.read_csv(csv_file)

    # Support both old (W/H) and current (Width/Height) column names
    if "Width" in df.columns:
        df = df.rename(columns={"Width": "W", "Height": "H"})

    tile_info = []
    for _, row in df.iterrows():
        fov_name = str(row["FOV_Name"]).upper()
        file_path = tile_dir / tiff_pattern.format(FOV_Name=fov_name)
        tile_info.append({
            "fov_name": fov_name,
            "path": file_path,
            "x": int(row["X"]),
            "y": int(row["Y"]),
            "w": int(row["W"]),
            "h": int(row["H"]),
        })

    max_x = int((df["X"] + df["W"]).max())
    max_y = int((df["Y"] + df["H"]).max())

    try:
        global_mask = np.zeros((max_y, max_x), dtype=np.uint32)
    except MemoryError:
        raise MemoryError(
            f"Cannot allocate global canvas ({max_y}x{max_x}). "
            "Consider reducing tile count or tile size."
        )

    sorted_tiles = sorted(tile_info, key=lambda t: (t["y"], t["x"]))
    global_id_counter = 1
    global_id_ranges = {}

    for tile in tqdm(sorted_tiles, desc="  Stitching tiles", leave=False):
        if not tile["path"].exists():
            global_id_ranges[tile["path"]] = (0, 0)
            continue

        try:
            local_mask = tifffile.imread(str(tile["path"]))
            x, y = tile["x"], tile["y"]
            h_local, w_local = local_mask.shape
            canvas_slice = global_mask[y: y + h_local, x: x + w_local]

            local_ids = np.unique(local_mask[local_mask > 0])
            if len(local_ids) == 0:
                global_id_ranges[tile["path"]] = (0, 0)
                continue

            remapped = np.zeros_like(local_mask, dtype=np.uint32)
            id_start = global_id_counter
            current_id = id_start

            all_areas = np.bincount(local_mask.ravel())
            conflict_pixels_map = canvas_slice > 0
            conflict_counts = np.bincount(local_mask[conflict_pixels_map])

            for local_id in local_ids:
                cell_area = all_areas[local_id]
                conflict_n = (
                    conflict_counts[local_id]
                    if local_id < len(conflict_counts) else 0
                )
                conflict_ratio = conflict_n / cell_area if cell_area > 0 else 0
                if conflict_ratio < overlap_threshold:
                    remapped[local_mask == local_id] = current_id
                    current_id += 1

            global_id_ranges[tile["path"]] = (id_start, current_id)
            global_id_counter = current_id

            write_mask = (canvas_slice == 0) & (remapped > 0)
            canvas_slice[write_mask] = remapped[write_mask]

        except Exception as e:
            print(f"  [dedup] Warning: error processing {tile['path'].name}: {e}")
            global_id_ranges[tile["path"]] = (0, 0)

    return global_mask, tile_info, global_id_ranges


def _export_deduplicated_fovs(
    global_mask: np.ndarray,
    tile_info: list,
    global_id_ranges: dict,
    output_dir: Path,
    tiff_pattern: str,
    min_cell_size: int,
) -> None:
    """Re-slice the global canvas and save deduplicated per-FOV masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile in tqdm(tile_info, desc="  Exporting FOVs", leave=False):
        try:
            x, y, w, h = tile["x"], tile["y"], tile["w"], tile["h"]
            dedup_slice = global_mask[y: y + h, x: x + w]

            id_start, id_end = global_id_ranges.get(tile["path"], (0, 0))
            filtered = np.zeros_like(dedup_slice, dtype=np.uint32)
            if id_start != 0 or id_end != 0:
                valid = (dedup_slice >= id_start) & (dedup_slice < id_end)
                filtered[valid] = dedup_slice[valid]

            if min_cell_size > 0 and filtered.max() > 0:
                props = regionprops(filtered)
                small_ids = [p.label for p in props if p.area < min_cell_size]
                if small_ids:
                    filtered[np.isin(filtered, small_ids)] = 0

            out_path = output_dir / tiff_pattern.format(FOV_Name=tile["fov_name"])
            tifffile.imwrite(str(out_path), filtered, dtype=np.uint32)
        except Exception as e:
            print(f"  [dedup] Warning: error exporting {tile['fov_name']}: {e}")


def deduplicate_slide(
    slide_dir: str,
    overlap_threshold: float = 0.1,
    min_cell_size: int = 0,
    tiff_pattern: str = "{FOV_Name}_whole_cell.tiff",
    clear_borders: bool = True,
    border_buffer: int = 2,
    seg_subdir: str = "deepcell_output",
) -> None:
    """
    Run full deduplication pipeline for one slide.

    Stage 1 (optional): Border clearing - remove cells touching tile edges.
    Stage 2: Overlap deduplication - resolve cells segmented in two adjacent tiles.

    Parameters
    ----------
    slide_dir : str
        Slide root directory containing segmentation/ and fov_coordinates.csv.
    overlap_threshold : float
        Cells with this fraction of pixels already claimed by a prior tile
        are discarded as duplicates (default 0.1 = 10%).
    min_cell_size : int
        Cells with fewer pixels than this are removed (0 = disabled).
    tiff_pattern : str
        Filename pattern; {FOV_Name} is replaced with the actual FOV name.
    clear_borders : bool
        Run border clearing before deduplication (default True).
        Removes incomplete cells that touch tile edges.
    border_buffer : int
        Pixel buffer from edge for border clearing (default 2).
    seg_subdir : str
        Subdirectory under segmentation/ for mask files (default "deepcell_output").
    """
    slide_dir = Path(slide_dir)
    csv_file = slide_dir / "fov_coordinates.csv"
    tile_dir = slide_dir / "segmentation" / seg_subdir
    tmp_dir  = slide_dir / "segmentation" / f"{seg_subdir}_tmp"

    if not csv_file.exists():
        raise FileNotFoundError(f"fov_coordinates.csv not found: {csv_file}")

    # Stage 1: Border clearing (in-place on deepcell_output/)
    if clear_borders:
        print(f"[dedup] Stage 1: Clearing border cells ...")
        border_cleared_dir = tile_dir.parent / f"{seg_subdir}_borders_cleared"
        clear_tile_borders(str(tile_dir), str(border_cleared_dir), buffer_size=border_buffer)
        # Swap: border-cleared becomes the input to deduplication
        bak_borders = Path(str(tile_dir) + "_bak_borders")
        if bak_borders.exists():
            shutil.rmtree(bak_borders)
        os.rename(tile_dir, bak_borders)
        os.rename(border_cleared_dir, tile_dir)

    print(f"[dedup] Building global canvas for {slide_dir.name} ...")
    global_mask, tile_info, id_ranges = _build_global_canvas(
        csv_file, tile_dir, tiff_pattern, overlap_threshold
    )

    _export_deduplicated_fovs(
        global_mask, tile_info, id_ranges, tmp_dir, tiff_pattern, min_cell_size
    )

    # Atomic swap: original -> _bak, tmp -> final
    bak_dir = Path(str(tile_dir) + "_bak")
    if bak_dir.exists():
        shutil.rmtree(bak_dir)
    os.rename(tile_dir, bak_dir)
    os.rename(tmp_dir, tile_dir)
    print(f"[dedup] Done. Deduplicated masks saved to {tile_dir}")


def deduplicate_experiment(
    experiment_dir: str,
    slide_list: Optional[list] = None,
    overlap_threshold: float = 0.1,
    min_cell_size: int = 0,
    tiff_pattern: str = "{FOV_Name}_whole_cell.tiff",
    clear_borders: bool = True,
    border_buffer: int = 2,
    seg_subdir: str = "deepcell_output",  
) -> None:
    """
    Run deduplication for all (or selected) slides in an experiment.

    Parameters
    ----------
    experiment_dir : str
        Root experiment folder.
    slide_list : list of str, optional
        Slide folder names. If None, all subdirectories are processed.
    overlap_threshold, min_cell_size, tiff_pattern : see deduplicate_slide().
    """
    experiment_dir = Path(experiment_dir)

    if slide_list is None:
        slide_dirs = [d for d in sorted(experiment_dir.iterdir()) if d.is_dir() and not d.name.startswith('.')]
    else:
        slide_dirs = [experiment_dir / s for s in slide_list]

    for slide_dir in slide_dirs:
        print(f"\n>>> Processing: {slide_dir.name}")
        try:
            deduplicate_slide(
                str(slide_dir), overlap_threshold, min_cell_size, tiff_pattern,
                clear_borders=clear_borders, border_buffer=border_buffer,
                seg_subdir=seg_subdir,             
            )
        except Exception as e:
            print(f"  [dedup] Error: {e}")

    print("\n[dedup] All slides processed.")
