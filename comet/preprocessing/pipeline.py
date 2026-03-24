"""High-level Stage 1 preprocessing helpers."""

from pathlib import Path
from typing import List, Optional

from .tile_split import tile_experiment
from .channel_extract import (
    extract_channels_experiment,
    print_channel_metadata,
    resolve_channel_names,
)
from ..segmentation.signal_prep import prepare_cellpose_inputs



def _get_raw_slides(base_dir: Path, slide_suffix: str) -> List[Path]:
    """Return raw slide paths under the experiment root."""
    base_dir = Path(base_dir)
    raw_slides = sorted(base_dir.glob(f"*{slide_suffix}"))
    if not raw_slides:
        raise FileNotFoundError(
            f"No raw slide files matching *{slide_suffix} were found in {base_dir}. "
            "Stage 1 expects whole-slide OME-TIFF files in the experiment root."
        )
    return raw_slides



def inspect_experiment_metadata(
    base_dir: Path,
    slide_suffix: str = ".ome.tif",
) -> Path:
    """
    Print channel metadata for the first raw slide in an experiment folder.

    Returns the Path of the slide that was inspected so notebooks can surface
    which file was used as the metadata reference.
    """
    raw_slides = _get_raw_slides(Path(base_dir), slide_suffix)
    reference_slide = raw_slides[0]

    print("==== Stage 1A | Inspect channel metadata on the first raw slide ====")
    print(f"Reference slide: {reference_slide.name}")
    print_channel_metadata(str(reference_slide))
    return reference_slide



def run_signal_preparation_pipeline(
    base_dir: Path,
    channel_names: Optional[List[str]],
    nuclear_markers: List[str],
    membrane_markers: List[str],
    slide_suffix: str = ".ome.tif",
    tile_size: int = 1024,
    overlap: int = 102,
    tissue_threshold: float = 10.0,
    normalize: bool = True,
) -> None:
    """
    Run the full Stage 1 preprocessing path from raw slide to Cellpose input.

    Run inspect_experiment_metadata() first in interactive workflows so the
    user can review the OME channel order before filling CHANNEL_NAMES.
    If the channel list is shorter than the raw image channel count, only the
    leading channels are exported and trailing channels are ignored.
    """
    base_dir = Path(base_dir)
    raw_slides = _get_raw_slides(base_dir, slide_suffix)
    reference_slide = raw_slides[0]

    resolved_channel_names = resolve_channel_names(
        slide_path=str(reference_slide),
        channel_names=channel_names,
    )
    print("==== Stage 1B | Channel names selected for extraction ====")
    for idx, name in enumerate(resolved_channel_names):
        print(f"  Channel {idx}: {name}")

    print("==== Stage 1C | Tile whole-slide images into tissue-rich FOVs ====")
    tile_experiment(
        experiment_dir=str(base_dir),
        suffix=slide_suffix,
        tile_size=tile_size,
        overlap=overlap,
        tissue_threshold=tissue_threshold,
    )

    print("==== Stage 1D | Split each tiled FOV into named marker images ====")
    extract_channels_experiment(
        experiment_dir=str(base_dir),
        channel_names=resolved_channel_names,
    )

    print("==== Stage 1E | Build Cellpose-ready nuclear and membrane composites ====")
    for sample_dir in sorted(base_dir.iterdir()):
        if sample_dir.is_dir() and (sample_dir / "image_data").exists():
            print(f"==== Start processing sample folder: {sample_dir.name} ====")
            prepare_cellpose_inputs(
                base_dir=str(sample_dir),
                nuclear_markers=nuclear_markers,
                membrane_markers=membrane_markers,
                normalize=normalize,
            )
