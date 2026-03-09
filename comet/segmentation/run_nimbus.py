"""
run_nimbus.py - COMET segmentation step 4
Thin wrapper for batch NIMBUS marker classification.

Reads:  image_data/FOV*/  and  segmentation/deepcell_output/FOV*_whole_cell.tif
Writes: nimbus_output/nimbus_cell_table.csv
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional


def run_nimbus_slide(
    slide_dir: str,
    include_channels: List[str],
    checkpoint: str = "V1.pt",
    batch_size: int = 4,
    test_time_aug: bool = True,
    quantile: float = 0.999,
    n_subset: int = 50,
    seg_subdir: str = "deepcell_output",
    output_subdir: str = "nimbus_output",
) -> None:
    """
    Run NIMBUS marker classification for one slide.

    Parameters
    ----------
    slide_dir : str
        Slide root directory.
        Reads:  image_data/  and  segmentation/deepcell_output/
        Writes: nimbus_output/nimbus_cell_table.csv
    include_channels : list of str
        Channel names to classify (must match image_data/ subdirectory names).
    checkpoint : str
        NIMBUS model checkpoint filename (default 'V1.pt').
    batch_size : int
        Batch size (default 4; reduce if GPU OOM).
    test_time_aug : bool
        Enable test-time augmentation (default True).
    quantile : float
        Normalization quantile (default 0.999).
    n_subset : int
        Number of FOVs to use for normalization dict (default 50).
    seg_subdir : str
        Subdirectory under segmentation/ for mask files (default "deepcell_output").
    output_subdir : str
        Subdirectory under slide_dir/ for NIMBUS output (default "nimbus_output").
    """
    try:
        from nimbus_inference.nimbus import Nimbus, prep_naming_convention
        from nimbus_inference.utils import MultiplexDataset
        from alpineer import io_utils
    except ImportError:
        raise ImportError(
            "nimbus-inference is required. Install with: pip install nimbus-inference"
        )

    slide_dir = Path(slide_dir)
    tiff_dir = slide_dir / "image_data"
    seg_dir  = slide_dir / "segmentation" / seg_subdir
    output_dir = slide_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    io_utils.validate_paths([str(slide_dir), str(tiff_dir), str(seg_dir)])

    # Only include subdirectories (FOV0, FOV1, ...), skip stray files
    fov_names = sorted(
        [f for f in os.listdir(str(tiff_dir))
         if (tiff_dir / f).is_dir() and not f.startswith(".")],
        key=lambda s: int("".join(filter(str.isdigit, s)) or "0"),
    )
    if not fov_names:
        raise FileNotFoundError(f"No FOV subdirectories found in {tiff_dir}")

    fov_paths = [str(tiff_dir / fov) for fov in fov_names]
    n_fovs = len(fov_names)
    print(f"  [nimbus] {n_fovs} FOVs found.")

    seg_naming = prep_naming_convention(str(seg_dir))

    dataset = MultiplexDataset(
        fov_paths=fov_paths,
        suffix=".tif",
        include_channels=include_channels,
        segmentation_naming_convention=seg_naming,
        output_dir=str(output_dir),
    )

    nimbus = Nimbus(
        dataset=dataset,
        save_predictions=True,
        batch_size=batch_size,
        test_time_aug=test_time_aug,
        input_shape=[1024, 1024],
        device="auto",
        checkpoint=checkpoint,
        output_dir=str(output_dir),
    )
    nimbus.check_inputs()

    print("  [nimbus] Preparing normalization dictionary ...")
    effective_n_subset = min(n_subset, n_fovs)
    if effective_n_subset < n_subset:
        print(f"  [nimbus] n_subset capped to {effective_n_subset} (only {n_fovs} FOVs available).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress verbose NIMBUS/torch warnings
        dataset.prepare_normalization_dict(
            quantile=quantile,
            n_subset=effective_n_subset,
            clip_values=(0, 2),
            multiprocessing=True,
            overwrite=True,
        )

        print("  [nimbus] Running predictions ...")
        cell_table = nimbus.predict_fovs()

    out_csv = output_dir / "nimbus_cell_table.csv"
    cell_table.to_csv(str(out_csv), index=False)
    print(f"  [nimbus] Cell table saved -> {out_csv} (shape: {cell_table.shape})")


def run_nimbus_experiment(
    experiment_dir: str,
    include_channels: List[str],
    slide_list: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Run NIMBUS for all (or selected) slides in an experiment.

    Parameters
    ----------
    experiment_dir : str
        Root experiment folder.
    include_channels : list of str
        Marker channels to classify.
    slide_list : list of str, optional
        Specific slide names. If None, processes all subdirectories.
    **kwargs
        Passed to run_nimbus_slide().
    """
    experiment_dir = Path(experiment_dir)

    if slide_list is None:
        slide_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir()])
    else:
        slide_dirs = [experiment_dir / s for s in slide_list]

    for slide_dir in slide_dirs:
        print(f"\n[nimbus] Processing: {slide_dir.name}")
        try:
            run_nimbus_slide(str(slide_dir), include_channels, **kwargs)
        except Exception as e:
            print(f"  [nimbus] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n[nimbus] Batch processing complete.")
