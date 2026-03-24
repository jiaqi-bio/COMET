"""
channel_extract.py - COMET preprocessing step 2
Reads each multi-channel tile and saves individual channels as named tiff files.

Output layout:
  <experiment>/<slide>/image_data/<FOV_Name>/<channel_name>.tif
"""

import tifffile
import numpy as np
from pathlib import Path
from typing import List, Optional



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
        if img.shape[-1] < img.shape[0] and img.shape[-1] < img.shape[1]:
            return np.transpose(img, (2, 0, 1))
        return img
    if img.ndim == 4:
        img = img[:, 0, :, :]
        return img
    raise ValueError(
        f"Cannot normalise array with ndim={img.ndim}, shape={img.shape}. "
        f"File: {path}"
    )


def get_channel_names_from_metadata(slide_path: str) -> Optional[List[str]]:
    """
    Attempt to read channel names from ome.tif metadata.
    Returns a list of channel names if found, otherwise None.
    """
    try:
        with tifffile.TiffFile(slide_path) as tif:
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                channels = root.findall(".//ome:Channel", ns)
                if channels:
                    return [ch.get("Name", f"channel{i}") for i, ch in enumerate(channels)]
    except Exception:
        pass
    return None


def print_channel_metadata(slide_path: str) -> None:
    """Print channel metadata to help users set channel_names."""
    names = get_channel_names_from_metadata(slide_path)
    if names:
        print(f"[channel metadata] {Path(slide_path).name}:")
        for i, name in enumerate(names):
            print(f"  Channel {i}: {name}")
    else:
        print(f"[channel metadata] Could not read channel names from {slide_path}.")
        print("  Please specify channel_names manually.")



def resolve_channel_names(
    slide_path: str,
    channel_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Resolve channel names for Stage 1 extraction.

    If channel_names is provided, blank values are removed and the remaining
    names are used in order. When fewer names are provided than appear in the
    OME metadata, only the leading channels are exported and trailing channels
    are ignored. If channel_names is omitted, metadata names are used.
    """
    cleaned_names = [
        str(name).strip()
        for name in (channel_names or [])
        if str(name).strip()
    ]
    metadata_names = get_channel_names_from_metadata(slide_path)

    if cleaned_names:
        if metadata_names and len(cleaned_names) > len(metadata_names):
            raise ValueError(
                f"{Path(slide_path).name} exposes {len(metadata_names)} metadata channels, "
                f"but {len(cleaned_names)} names were provided."
            )
        if metadata_names and len(cleaned_names) < len(metadata_names):
            ignored = metadata_names[len(cleaned_names):]
            print(
                f"[channel metadata] Using the first {len(cleaned_names)} channels and "
                f"ignoring the remaining {len(ignored)} trailing channels: {ignored}"
            )
        return cleaned_names

    if metadata_names:
        print("[channel metadata] No CHANNEL_NAMES provided. Using OME metadata names.")
        return metadata_names

    raise ValueError(
        "CHANNEL_NAMES is empty and channel names could not be read from OME metadata."
    )



def _resolve_export_names(
    n_channels: int,
    channel_names: Optional[List[str]],
    fov_name: str,
) -> List[str]:
    """Choose the channel names to export for one FOV."""
    if not channel_names:
        print(
            f"  Warning: no channel names provided for {fov_name}. "
            "Using channel1, channel2, ..."
        )
        return [f"channel{i+1}" for i in range(n_channels)]

    if len(channel_names) > n_channels:
        raise ValueError(
            f"{fov_name} has {n_channels} channels, but {len(channel_names)} names were provided."
        )

    if len(channel_names) < n_channels:
        print(
            f"  Note: {len(channel_names)} names provided but {fov_name} has {n_channels} channels. "
            f"Exporting the first {len(channel_names)} channels and ignoring the remaining "
            f"{n_channels - len(channel_names)}."
        )

    return list(channel_names)



def extract_channels_slide(
    slide_dir: str,
    channel_names: Optional[List[str]],
) -> None:
    """
    Split channels for all tiles belonging to one slide.

    Parameters
    ----------
    slide_dir : str
        Path to the slide subfolder containing Tiles/.
    channel_names : list of str, optional
        Channel names in order. If fewer names than channels are supplied,
        only the leading channels are exported. If omitted, generic names are
        used.
    """
    slide_dir = Path(slide_dir)
    tiles_dir = slide_dir / "Tiles"
    image_data_dir = slide_dir / "image_data"

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    # glob both .tif and .tiff, sort numerically by FOV index
    fov_files = sorted(
        list(tiles_dir.glob("FOV*.tif")) + list(tiles_dir.glob("FOV*.tiff")),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
    )
    if not fov_files:
        raise FileNotFoundError(f"No FOV*.tif files found in {tiles_dir}")

    print(f"Found {len(fov_files)} FOVs. Starting channel splitting...")

    for fov_file in fov_files:
        fov_name = fov_file.stem  # e.g. "FOV0"
        fov_out_dir = image_data_dir / fov_name
        fov_out_dir.mkdir(parents=True, exist_ok=True)

        fov_img = tifffile.imread(str(fov_file))
        fov_img = _squeeze_to_chw(fov_img, str(fov_file))

        n_channels = fov_img.shape[0]
        actual_names = _resolve_export_names(n_channels, channel_names, fov_name)

        for c_idx, name in enumerate(actual_names):
            out_path = fov_out_dir / f"{name}.tif"
            tifffile.imwrite(str(out_path), fov_img[c_idx], photometric="minisblack")

    print(f"[extract_channels] {slide_dir.name}: channel splitting complete.")



def extract_channels_experiment(
    experiment_dir: str,
    channel_names: Optional[List[str]],
) -> None:
    """
    Split channels for all slides in an experiment folder.

    Parameters
    ----------
    experiment_dir : str
        Root experiment folder.
    channel_names : list of str, optional
        Channel names in order. If fewer names than channels are supplied,
        only the leading channels are exported.
    """
    experiment_dir = Path(experiment_dir)
    slide_dirs = [
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (d / "Tiles").exists()
    ]

    if not slide_dirs:
        raise FileNotFoundError(
            f"No slide folders with Tiles/ found in {experiment_dir}"
        )

    for slide_dir in sorted(slide_dirs):
        extract_channels_slide(str(slide_dir), channel_names)
