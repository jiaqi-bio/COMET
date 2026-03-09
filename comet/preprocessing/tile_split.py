"""
tile_split.py - COMET preprocessing step 1
Splits a whole-slide mIHC image (qptiff/ome.tiff) into overlapping tiles
and records each tile's coordinates in fov_coordinates.csv.
"""

import numpy as np
import tifffile
import pandas as pd
from pathlib import Path



def _squeeze_to_chw(img: np.ndarray, path: str = "") -> np.ndarray:
    """
    Normalise tifffile output to (C, H, W).

    tifffile can return ome.tiff as (C, H, W), (H, W, C), (C, Z, H, W),
    or (T, C, H, W) depending on metadata. This function always returns
    (C, H, W) for consistent downstream processing.
    """
    if img.ndim == 2:
        return img[np.newaxis, :, :]
    if img.ndim == 3:
        # (H, W, C) if channel count is last and smaller than both spatial dims
        if img.shape[-1] < img.shape[0] and img.shape[-1] < img.shape[1]:
            return np.transpose(img, (2, 0, 1))
        return img  # already (C, H, W)
    if img.ndim == 4:
        # (C, Z, H, W) or (T, C, H, W): take the first Z-plane or timepoint
        img = img[:, 0, :, :]
        return img
    raise ValueError(
        f"Cannot normalise array with ndim={img.ndim}, shape={img.shape}. "
        f"File: {path}"
    )


def is_tissue(patch: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Return True if the tile contains valid tissue signal.

    Uses standard deviation of the patch as the signal criterion.
    Blank or near-blank tiles have very low std and are discarded.

    Parameters
    ----------
    patch : np.ndarray
        Tile image array. Shape (C, H, W) or (H, W).
    threshold : float
        Minimum std required to keep the tile (default 10).
    """
    return float(np.std(patch)) >= threshold


def tile_slide(
    slide_path: str,
    output_dir: str,
    tile_size: int = 1024,
    overlap: int = 102,
    tissue_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Tile a whole-slide image into overlapping FOVs.

    Parameters
    ----------
    slide_path : str
        Path to the input qptiff / ome.tiff slide.
    output_dir : str
        Directory under which <slide_name>/Tiles/ will be created.
    tile_size : int
        Width and height of each tile in pixels (default 1024).
    overlap : int
        Overlap in pixels between adjacent tiles (default 102, ~10% of 1024).
    tissue_threshold : float
        Tiles with std below this value are discarded (default 10).

    Returns
    -------
    pd.DataFrame
        Table with columns [FOV_Name, X, Y, Width, Height].
    """
    slide_path = Path(slide_path)
    slide_name = slide_path.name.split(".")[0]  # Slides1.ome.tif -> Slides1

    tile_out_dir = Path(output_dir) / slide_name / "Tiles"
    tile_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {slide_path} ...")
    img = tifffile.imread(str(slide_path))

    img = _squeeze_to_chw(img, str(slide_path))
    C, H, W = img.shape
    print(f"  Image shape: {img.shape}, dtype: {img.dtype}")

    step = tile_size - overlap
    fov_count = 0
    coordinates = []

    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            patch = img[:, y:y_end, x:x_end]

            # Pad edges to full tile_size
            if patch.shape[1] < tile_size or patch.shape[2] < tile_size:
                pad_y = tile_size - patch.shape[1]
                pad_x = tile_size - patch.shape[2]
                patch = np.pad(patch, ((0, 0), (0, pad_y), (0, pad_x)),
                               mode="constant", constant_values=0)

            if not is_tissue(patch, tissue_threshold):
                continue

            fov_name = f"FOV{fov_count}"
            tifffile.imwrite(
                str(tile_out_dir / f"{fov_name}.tif"),
                patch,
                photometric="minisblack",
            )

            coordinates.append({
                "FOV_Name": fov_name,
                "X": x,
                "Y": y,
                "Width": tile_size,
                "Height": tile_size,
            })
            fov_count += 1

    coords_df = pd.DataFrame(coordinates)
    csv_path = Path(output_dir) / slide_name / "fov_coordinates.csv"
    coords_df.to_csv(str(csv_path), index=False)
    print(f"[tile_slide] {slide_name}: {fov_count} FOVs -> {tile_out_dir}")
    return coords_df


def tile_experiment(
    experiment_dir: str,
    suffix: str = ".ome.tif",
    tile_size: int = 1024,
    overlap: int = 102,
    tissue_threshold: float = 10.0,
) -> None:
    """
    Tile all slides in an experiment folder.

    Parameters
    ----------
    experiment_dir : str
        Root folder containing slide files.
    suffix : str
        File extension to glob for (e.g. '.ome.tif', '.tiff').
    tile_size, overlap, tissue_threshold : see tile_slide().
    """
    experiment_dir = Path(experiment_dir)
    slides = sorted(experiment_dir.glob(f"*{suffix}"))

    if not slides:
        raise FileNotFoundError(
            f"No slides with suffix '{suffix}' found in {experiment_dir}"
        )

    for slide in slides:
        tile_slide(
            slide_path=str(slide),
            output_dir=str(experiment_dir),
            tile_size=tile_size,
            overlap=overlap,
            tissue_threshold=tissue_threshold,
        )
