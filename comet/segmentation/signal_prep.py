"""
signal_prep.py - COMET segmentation step 1
Normalizes and fuses nuclear + membrane channel signals into a
2-channel input image for CellposeSAM.

COMET improvement: per-channel percentile normalization before fusion
ensures nuclear markers (e.g. DAPI, EOMES) and membrane markers
(e.g. CD3, CD4) contribute equally.

Output: segmentation/cellpose_input/<FOV_Name>.tiff  [2, H, W] uint16
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List


def normalize_channel(
    image_data: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Percentile-based normalization on non-zero pixels.
    Returns float32 array clipped to [0, 1].
    """
    img_flat = image_data.flatten()
    nonzero = img_flat[img_flat > 0]

    if len(nonzero) == 0:
        return np.zeros_like(image_data, dtype=np.float32)

    p_low = np.percentile(nonzero, lower_percentile)
    p_high = np.percentile(nonzero, upper_percentile)

    if p_high - p_low == 0:
        return np.zeros_like(image_data, dtype=np.float32)

    img_norm = (image_data.astype(np.float32) - p_low) / (p_high - p_low)
    return np.clip(img_norm, 0.0, 1.0)


def _read_marker(fov_path: Path, marker: str) -> np.ndarray:
    """Try marker.tif then marker.tiff. Returns None if not found."""
    for ext in (".tif", ".tiff"):
        p = fov_path / f"{marker}{ext}"
        if p.exists():
            return tifffile.imread(str(p))
    return None


def prepare_cellpose_input_fov(
    fov_path: str,
    output_path: str,
    nuclear_markers: List[str],
    membrane_markers: List[str],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    normalize: bool = True,
) -> None:
    """
    Fuse nuclear and membrane channels for a single FOV.

    Parameters
    ----------
    fov_path : str
        Directory containing per-channel tiff files (e.g. image_data/FOV0/).
    output_path : str
        Full path for the output 2-channel tiff (e.g. cellpose_input/FOV0.tiff).
    nuclear_markers : list of str
        Marker names to fuse into nuclear channel (channel 0).
    membrane_markers : list of str
        Marker names to fuse into membrane channel (channel 1).
    lower_percentile, upper_percentile : float
        Percentile range for normalization.
    normalize : bool
        If True (default), apply per-channel percentile normalization before fusion.
        If False, channels are scaled to [0, 1] by dividing by 65535 without
        per-channel normalization (Baseline A condition).
    """
    fov_path = Path(fov_path)

    # Infer image dimensions from first available file (.tif or .tiff)
    sample_files = list(fov_path.glob("*.tif")) + list(fov_path.glob("*.tiff"))
    if not sample_files:
        raise FileNotFoundError(f"No tif files found in {fov_path}")

    sample = tifffile.imread(str(sample_files[0]))
    h, w = sample.shape[-2], sample.shape[-1]
    out_img = np.zeros((2, h, w), dtype=np.float32)

    for marker in nuclear_markers:
        img_data = _read_marker(fov_path, marker)
        if img_data is not None:
            if normalize:
                out_img[0] += normalize_channel(img_data, lower_percentile, upper_percentile)
            else:
                out_img[0] += img_data.astype(np.float32) / 65535.0
        else:
            print(f"  [signal_prep] Warning: nuclear marker '{marker}' not found in {fov_path.name}")
    out_img[0] = np.clip(out_img[0], 0.0, 1.0)

    for marker in membrane_markers:
        img_data = _read_marker(fov_path, marker)
        if img_data is not None:
            if normalize:
                out_img[1] += normalize_channel(img_data, lower_percentile, upper_percentile)
            else:
                out_img[1] += img_data.astype(np.float32) / 65535.0
        else:
            print(f"  [signal_prep] Warning: membrane marker '{marker}' not found in {fov_path.name}")
    out_img[1] = np.clip(out_img[1], 0.0, 1.0)

    out_img_16bit = (out_img * 65535).astype(np.uint16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), out_img_16bit, imagej=True)


def prepare_cellpose_inputs(
    base_dir: str,
    nuclear_markers: List[str],
    membrane_markers: List[str],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    normalize: bool = True,
    output_subdir: str = "cellpose_input",
) -> None:
    """
    Prepare CellposeSAM input for all FOVs under one slide directory.

    Parameters
    ----------
    base_dir : str
        Slide root directory (e.g. experiment/Slides1/).
    nuclear_markers : list of str
        Markers to fuse as nuclear signal (e.g. ["DAPI", "EOMES"]).
    membrane_markers : list of str
        Markers to fuse as membrane signal (e.g. ["CD3", "CD4", "CD8a", "TCR"]).
    lower_percentile, upper_percentile : float
        Percentile normalization range.
    normalize : bool
        If True (default), apply per-channel percentile normalization.
        If False, use raw scaling without per-channel normalization (Baseline A).
    output_subdir : str
        Subdirectory under segmentation/ for output files (default "cellpose_input").
    """
    base_dir = Path(base_dir)
    image_dir = base_dir / "image_data"
    output_dir = base_dir / "segmentation" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    fov_folders = sorted(
        [f for f in image_dir.iterdir() if f.is_dir()],
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
    )

    if not fov_folders:
        raise FileNotFoundError(f"No FOV folders found in {image_dir}")

    for fov_dir in fov_folders:
        print(f"  [signal_prep] Processing: {fov_dir.name}")
        output_path = output_dir / f"{fov_dir.name}.tiff"
        prepare_cellpose_input_fov(
            fov_path=str(fov_dir),
            output_path=str(output_path),
            nuclear_markers=nuclear_markers,
            membrane_markers=membrane_markers,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            normalize=normalize,
        )

    print(f"[signal_prep] Done. {len(fov_folders)} FOVs -> {output_dir}")
