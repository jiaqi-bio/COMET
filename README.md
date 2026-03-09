# COMET — COmbinatorial Marker Expression Typing

A computational pipeline for accurate quantification of rare immune cells in multiplexed immunohistochemistry (mIHC) whole-slide images.

Developed to support the identification of IL-10+ CD4-CD8- double-negative T cells (DNTs) in IBD tissue, COMET integrates [CellposeSAM](https://github.com/MouseLand/cellpose) and [NIMBUS](https://github.com/angelolab/nimbus-inference) with three targeted improvements for multi-marker rare-cell phenotyping.

---

## Core Improvements

| Improvement | Problem Solved |
|---|---|
| **Normalized signal fusion** (`signal_prep`) | Direct channel addition lets bright markers dominate segmentation input; per-channel percentile normalization ensures equal contribution |
| **Tile overlap deduplication** (`deduplication`) | CellposeSAM has no built-in tile stitching; the same cell gets double-counted in overlapping regions |
| **OTSU x factor thresholding** (`threshold`) | Standard OTSU alone fails when positive cells are rare (<1%); a per-marker multiplicative factor adjusts stringency for each marker |

---

## Pipeline

```
qptiff / ome.tiff
    |
    |-- 1. tile_split        Whole-slide -> 1024x1024 tiles (102px overlap)
    |-- 2. channel_extract   Multi-channel tile -> per-marker .tif files
    |-- 3. signal_prep       Nuclear + membrane fusion -> CellposeSAM input
    |-- 4. CellposeSAM       Cell segmentation
    |-- 5. deduplication     Border clearing + overlap deduplication
    |-- 6. NIMBUS            Per-cell marker probability scoring
    |-- 7. threshold         OTSU x factor classification + cell typing
    `-- 8. qupath_export     Export results to QuPath for visualization
```

---

## Installation

```bash
# 1. Create environment (Python 3.10 recommended)
conda create -n comet python=3.10
conda activate comet

# 2. Install CellposeSAM (GPU recommended)
pip install cellpose
# For GPU: follow https://github.com/MouseLand/cellpose#gpu-version

# 3. Install NIMBUS
pip install nimbus-inference

# 4. Install COMET
pip install -e .
```

---

## Quick Start

```python
import comet

# Step 1: Tile the slides
comet.tile_experiment(
    experiment_dir="my_experiment",
    suffix=".ome.tif",
    tile_size=1024,
    overlap=102,
)

# Step 2: Extract channels (check channel order first)
comet.print_channel_metadata("my_experiment/Patient1.ome.tif")
comet.extract_channels_experiment(
    experiment_dir="my_experiment",
    channel_names=["DAPI", "EOMES", "CD3", "CD4", "CD8a", "TCR", "CXCR4"],
)

# Step 3: Prepare CellposeSAM input
comet.prepare_cellpose_inputs(
    base_dir="my_experiment/Patient1",
    nuclear_markers=["DAPI", "EOMES"],
    membrane_markers=["CD3", "CD4", "CD8a", "TCR"],
)

# Step 4: Run CellposeSAM (GPU recommended)
comet.run_cellpose_slide("my_experiment/Patient1")

# Step 5: Border clearing + overlap deduplication
comet.deduplicate_slide("my_experiment/Patient1")

# Step 6: NIMBUS marker probability scoring
# Note: include_channels must exactly match the channel filenames in image_data/
comet.run_nimbus_slide(
    slide_dir="my_experiment/Patient1",
    include_channels=["CD3", "CD4", "CD8a", "TCR", "CXCR4", "EOMES"],
)

# Step 7: Classify and assign cell types
# Default factors: CD3=1.0, TCR=0.6, EOMES=1.6, CD4=0.4, CD8a=0.5, CD45=1.0
# NIMBUS column names: Prob_{channel_name} (e.g. Prob_CD3, Prob_CD8a)
result = comet.threshold_slide(
    nimbus_csv="my_experiment/Patient1/nimbus_output/nimbus_cell_table.csv",
    markers=["CD3", "CD4", "CD8a", "TCR", "EOMES"],
    # col_map is only needed if NIMBUS column names differ from Prob_{marker}
    # e.g. col_map={"CD8a": "Prob_CD8"} if your NIMBUS outputs Prob_CD8
)

# Step 8: Export to QuPath for visualization
comet.export_to_qupath(
    classified_csv="my_experiment/Patient1/nimbus_output/nimbus_cell_table_classified.csv",
)
```

---

## Marker naming and NIMBUS columns

NIMBUS names probability columns as `Prob_{channel_name}`. The channel_name is
whatever you passed to `include_channels`. Example:

```python
# If you run NIMBUS with:
include_channels=["CD3", "CD8a", "TCR"]
# NIMBUS outputs columns: Prob_CD3, Prob_CD8a, Prob_TCR

# threshold_slide expects these column names by default.
# If your NIMBUS output uses different names (e.g. from an older analysis),
# use col_map to remap:
result = comet.threshold_slide(
    nimbus_csv="...",
    markers=["CD3", "CD8a", "TCR"],
    col_map={"CD8a": "Prob_CD8"},   # only if NIMBUS named it Prob_CD8
)
```

---

## Cell type classification

`threshold_slide` assigns a `Cell_Type` column based on this hierarchy (from highest to lowest priority):

| Cell_Type | Criteria |
|---|---|
| `gd+T` | CD3+ & TCR- |
| `ab+CD4+T` | CD3+ & TCR+ & CD4+ |
| `ab+CD8+T` | CD3+ & TCR+ & CD8a+ |
| `CD45+CD3-` | CD3- & CD45+ |
| `ab+EOMES-DNT` | CD3+ & TCR+ & CD4- & CD8a- & EOMES- |
| `ab+EOMES+DNT` | CD3+ & TCR+ & CD4- & CD8a- & EOMES+ |
| `CD45+ Cells` | CD45+ (fallback) |

CD4/CD8a double-positives are resolved by flipping the weaker signal to negative
before classification.

---

## Output layout

```
my_experiment/
|-- Patient1.ome.tif
`-- Patient1/
    |-- fov_coordinates.csv           # tile positions in WSI coordinates
    |-- Tiles/                        # multi-channel tiles
    |   `-- FOV0.tif, FOV1.tif ...
    |-- image_data/                   # per-marker channel files
    |   `-- FOV0/
    |       |-- DAPI.tif
    |       |-- EOMES.tif
    |       `-- CD3.tif ...
    |-- segmentation/
    |   |-- cellpose_input/           # 2-channel fused input
    |   |   `-- FOV0.tiff ...
    |   |-- cellpose_output/          # raw CellposeSAM masks (intermediate)
    |   |   `-- FOV0_cp_masks.tif ...
    |   |-- deepcell_output/          # deduplicated masks (NIMBUS input)
    |   |   `-- FOV0_whole_cell.tiff ...
    |   `-- deepcell_output_bak/      # backup of pre-dedup masks
    `-- nimbus_output/
        |-- nimbus_cell_table.csv               # NIMBUS probabilities
        |-- nimbus_cell_table_classified.csv    # final cell table with Cell_Type
        |-- thresholds_used.csv                 # thresholds applied (for reporting)
        `-- threshold_distributions.png         # QC plots
```

---

## Repository structure

```
comet/
|-- __init__.py
|-- preprocessing/
|   |-- tile_split.py       # WSI tiling
|   `-- channel_extract.py  # channel demultiplexing
|-- segmentation/
|   |-- signal_prep.py      # normalized signal fusion
|   |-- run_cellpose.py     # CellposeSAM wrapper
|   |-- deduplication.py    # border clearing + overlap dedup
|   `-- run_nimbus.py       # NIMBUS wrapper
|-- classification/
|   `-- threshold.py        # OTSU x factor thresholding + cell typing
`-- export/
    `-- qupath_export.py    # QuPath TSV export
```

---

## Citation

If you use COMET in your research, please cite:

> [Manuscript in preparation] Spatial niche regulates IL-10-dependent efferocytosis during resolution of intestinal tissue inflammation.

---

## Acknowledgements

- [CellposeSAM](https://github.com/MouseLand/cellpose) — Stringer et al.
- [NIMBUS](https://github.com/angelolab/nimbus-inference) — Angelo Lab
