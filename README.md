# COMET — COmbinatorial Marker Expression Typing

A computational pipeline for accurate quantification of rare immune cell populations in multiplexed immunohistochemistry (mIHC) whole-slide images.

COMET integrates [CellposeSAM](https://github.com/MouseLand/cellpose) and [NIMBUS](https://github.com/angelolab/nimbus-inference) with three targeted improvements designed for multi-marker rare-cell phenotyping in tissue sections.

Developed in the **[Francis Chan Lab](https://francischanlab.com/)**, Liangzhu Laboratory, Zhejiang University.

---

## Core Improvements

| Improvement | Problem Solved |
|---|---|
| **Normalized signal fusion** (`signal_prep`) | Direct channel addition allows bright markers to dominate segmentation input; per-channel percentile normalization ensures equal contribution from each marker |
| **Tile overlap deduplication** (`deduplication`) | CellposeSAM has no built-in tile stitching; cells in overlapping regions are double-counted without correction |
| **Otsu × factor thresholding** (`threshold`) | Standard Otsu thresholding fails when positive cells are rare (<1%); a per-marker multiplicative correction factor adjusts classification stringency |

---

## Pipeline

![COMET Pipeline](docs/COMET.png)

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

# Step 1: Tile the whole-slide image
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
    channel_names=["DAPI", "NuclearMarker", "MemMarker1", "MemMarker2"],
)

# Step 3: Prepare CellposeSAM input (normalized signal fusion)
comet.prepare_cellpose_inputs(
    base_dir="my_experiment/Patient1",
    nuclear_markers=["DAPI", "NuclearMarker"],
    membrane_markers=["MemMarker1", "MemMarker2"],
)

# Step 4: Run CellposeSAM (GPU recommended)
comet.run_cellpose_slide("my_experiment/Patient1")

# Step 5: Border clearing + overlap deduplication
comet.deduplicate_slide("my_experiment/Patient1")

# Step 6: NIMBUS marker probability scoring
comet.run_nimbus_slide(
    slide_dir="my_experiment/Patient1",
    include_channels=["MemMarker1", "MemMarker2", "NuclearMarker"],
)

# Step 7: Classify and assign cell types
result = comet.threshold_slide(
    nimbus_csv="my_experiment/Patient1/nimbus_output/nimbus_cell_table.csv",
    markers=["MemMarker1", "MemMarker2", "NuclearMarker"],
    # col_map only needed if NIMBUS column names differ from Prob_{marker}
    # e.g. col_map={"MemMarker1": "Prob_Mem1"}
)

# Step 8: Export to QuPath
comet.export_to_qupath(
    classified_csv="my_experiment/Patient1/nimbus_output/nimbus_cell_table_classified.csv",
)
```

---

## Marker naming and NIMBUS columns

NIMBUS names probability columns as `Prob_{channel_name}`, where `channel_name` is whatever was passed to `include_channels`:

```python
# If you run NIMBUS with:
include_channels=["CD3", "CD8a", "Marker"]
# NIMBUS outputs: Prob_CD3, Prob_CD8a, Prob_Marker

# Use col_map only if your NIMBUS output uses different names:
result = comet.threshold_slide(
    nimbus_csv="...",
    markers=["CD3", "CD8a", "Marker"],
    col_map={"CD8a": "Prob_CD8"},   # only if needed
)
```

---

## Output layout

```
my_experiment/
|-- Patient1.ome.tif
`-- Patient1/
    |-- fov_coordinates.csv           # tile positions in WSI coordinates
    |-- Tiles/                        # multi-channel tiles
    |   `-- FOV0.tif, FOV1.tif ...
    |-- image_data/                   # per-marker single-channel files
    |   `-- FOV0/
    |       |-- DAPI.tif
    |       `-- Marker.tif ...
    |-- segmentation/
    |   |-- cellpose_input/           # 2-channel fused input
    |   |-- cellpose_output/          # raw CellposeSAM masks (intermediate)
    |   |-- deepcell_output/          # deduplicated masks (NIMBUS input)
    |   `-- deepcell_output_bak/      # backup of pre-dedup masks
    `-- nimbus_output/
        |-- nimbus_cell_table.csv               # NIMBUS probabilities
        |-- nimbus_cell_table_classified.csv    # final cell table with Cell_Type
        |-- thresholds_used.csv                 # thresholds applied
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
|   |-- deduplication.py    # border clearing + overlap deduplication
|   `-- run_nimbus.py       # NIMBUS wrapper
|-- classification/
|   `-- threshold.py        # Otsu × factor thresholding + cell typing
`-- export/
    `-- qupath_export.py    # QuPath TSV export
```

---

## Contributing

COMET is open for community use. If you encounter bugs, have questions, or want to suggest improvements, please open a [GitHub Issue](https://github.com/jiaqi-bio/COMET/issues). Pull requests are also welcome.

---

## Citation

If you use COMET in your research, please cite:

> [Spatial niche regulates IL-10–dependent efferocytosis during resolution of intestinal tissue inflammation] — Francis Chan Lab, Liangzhu Laboratory, Zhejiang University.
> Code will be made available upon publication.

---

## Contributing

COMET is open for community use. If you encounter bugs, have questions, or want to suggest improvements, please open a [GitHub Issue](https://github.com/jiaqi-bio/COMET/issues). Pull requests are also welcome.

---


## Acknowledgements

- [CellposeSAM](https://github.com/MouseLand/cellpose) — Pachitariu et al., 2025
- [NIMBUS](https://github.com/angelolab/nimbus-inference) — Rumberger, Greenwald et al., *Nature Methods*, 2025
