"""
run_cellpose.py - COMET segmentation step 2
Thin wrapper for batch CellposeSAM segmentation.

Input:  segmentation/cellpose_input/FOV*.tiff   (from signal_prep.py)
Output: segmentation/deepcell_output/FOV*_whole_cell.tif

The output directory and naming convention match what deduplication.py
and run_nimbus.py expect, so no manual file renaming is needed.
"""

import os
import re
import glob
import time
import shutil
from pathlib import Path
from tqdm import tqdm




def run_cellpose_slide(
    slide_dir: str,
    channels: list = None,
    diameter: float = 25.0,
    flow_threshold: float = 0.8,
    cellprob_threshold: float = 0.0,
    batch_size: int = 32,
    model_type: str = "cpsam",
    input_subdir: str = "cellpose_input",
    raw_subdir: str = "cellpose_output",
    final_subdir: str = "deepcell_output",
) -> None:
    
    """
    Run CellposeSAM segmentation for one slide.

    Parameters
    ----------
    slide_dir : str
        Slide root directory.
        Reads:  segmentation/cellpose_input/FOV*.tiff
        Writes: segmentation/deepcell_output/FOV*_whole_cell.tif
    channels : list of int
        [cytoplasm_channel, nuclear_channel]. Default [1, 0].
    diameter : float
        Expected cell diameter in pixels (default 25).
    flow_threshold : float
        Flow error threshold (default 0.8).
    cellprob_threshold : float
        Cell probability threshold (default 0.0).
    batch_size : int
        Batch size for GPU inference (default 32).
    model_type : str
        Cellpose model name (default 'cpsam').
    input_subdir : str
        Subdirectory under segmentation/ to read inputs from (default "cellpose_input").
    raw_subdir : str
        Subdirectory under segmentation/ for raw cellpose output (default "cellpose_output").
    final_subdir : str
        Subdirectory under segmentation/ for renamed masks (default "deepcell_output").
    """
    try:
        from cellpose import models, io, core
    except ImportError:
        raise ImportError(
            "cellpose is required. Install with: pip install cellpose\n"
            "See https://github.com/MouseLand/cellpose for GPU setup."
        )

    if channels is None:
        channels = [1, 0]  # [Cyto, Nuclei]

    slide_dir = Path(slide_dir)
    input_dir  = slide_dir / "segmentation" / input_subdir
    raw_dir    = slide_dir / "segmentation" / raw_subdir
    final_dir  = slide_dir / "segmentation" / final_subdir
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Detect GPU
    use_gpu = core.use_gpu()
    if use_gpu:
        try:
            import torch
            print(f"  [cellpose] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("  [cellpose] GPU detected but could not read device name.")
    else:
        print("  [cellpose] Warning: GPU not found, running on CPU (slow).")

    print(f"  [cellpose] Loading model '{model_type}' ...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)

    # Collect input files - FOV naming is uppercase (FOV0, FOV1, ...)
    files = sorted(
        glob.glob(str(input_dir / "FOV*.tiff")) + glob.glob(str(input_dir / "FOV*.tif")),
        key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))) or "0"),
    )
    if not files:
        raise FileNotFoundError(f"No FOV*.tiff found in {input_dir}")

    print(f"  [cellpose] Segmenting {len(files)} FOVs ...")
    imgs = [io.imread(f) for f in files]

    masks, flows = [], []
    start = time.time()
    for img in tqdm(imgs, desc="  Segmenting", unit="FOV"):
        m, f, _ = model.eval(
            img,
            batch_size=batch_size,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            resample=False,
            augment=False,
            do_3D=False,
        )
        masks.append(m)
        flows.append(f)

    elapsed = time.time() - start
    print(f"  [cellpose] Done in {elapsed:.1f}s. Saving masks ...")

    # io.save_masks writes to raw_dir with cellpose's own naming (FOV0_cp_masks.tif)
    io.save_masks(
        imgs, masks, flows, files,
        channels=channels,
        png=False,
        tif=True,
        savedir=str(raw_dir),
        save_flows=False,
        save_outlines=False,
    )

    # -----------------------------------------------------------------------
    # Rename and copy to deepcell_output with the naming convention that
    # deduplication.py and run_nimbus.py both expect:
    #   cellpose_output/FOV0_cp_masks.tif  ->  deepcell_output/FOV0_whole_cell.tif
    # -----------------------------------------------------------------------
    cp_masks = list(raw_dir.glob("*_cp_masks.tif"))
    if not cp_masks:
        # Fallback: some cellpose versions use _seg.tif
        cp_masks = list(raw_dir.glob("*_seg.tif"))
    if not cp_masks:
        raise FileNotFoundError(
            f"No *_cp_masks.tif or *_seg.tif found in {raw_dir}. "
            "Check cellpose output."
        )

    for src in cp_masks:
        # FOV0_cp_masks.tif -> FOV0
        fov_name = re.sub(r"(_cp_masks|_seg)$", "", src.stem)
        dst = final_dir / f"{fov_name}_whole_cell.tif"
        shutil.copy2(str(src), str(dst))

    print(f"  [cellpose] {len(cp_masks)} masks -> {final_dir}")


def run_cellpose_experiment(
    experiment_dir: str,
    slide_list: list = None,
    **kwargs,
) -> None:
    """
    Run CellposeSAM for all (or selected) slides in an experiment.

    Parameters
    ----------
    experiment_dir : str
        Root experiment folder.
    slide_list : list of str, optional
        Specific slide folder names. If None, processes all subdirectories.
    **kwargs
        Passed to run_cellpose_slide().
    """
    experiment_dir = Path(experiment_dir)

    if slide_list is None:
        slide_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    else:
        slide_dirs = [experiment_dir / s for s in slide_list]

    for slide_dir in slide_dirs:
        print(f"\n[cellpose] Processing: {slide_dir.name}")
        try:
            run_cellpose_slide(str(slide_dir), **kwargs)
        except Exception as e:
            print(f"  [cellpose] Error: {e}")
