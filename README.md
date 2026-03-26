# COMET — COmbinatorial Marker Expression Typing

A computational pipeline for accurate quantification of rare immune cell populations in multiplexed immunohistochemistry (mIHC) whole-slide images.

COMET integrates [CellposeSAM](https://github.com/MouseLand/cellpose) and [NIMBUS](https://github.com/angelolab/nimbus-inference) with three targeted improvements designed for multi-marker rare-cell phenotyping in tissue sections.

Developed in the **[Francis Chan Lab](https://francischanlab.com/)**, Liangzhu Laboratory, Zhejiang University.

---

## Core Improvements

| Improvement | Problem Solved |
|---|---|
| **Normalized signal fusion** (`signal_prep`) | Direct channel addition allows bright markers to dominate segmentation input; per-channel percentile normalization ensures equal contribution from each marker |
| **Tile overlap deduplication** (`deduplication`) | Nimbus has no built-in tile stitching; cells in overlapping regions are double-counted without correction |
| **Otsu × factor thresholding** (`threshold`) | Standard Otsu thresholding on predictions from Nimbus output fails when positive cells are rare (<1%); a per-marker multiplicative correction factor adjusts classification stringency |

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

## Notebooks

For interactive use, JupyterLab or Jupyter Notebook is recommended.

COMET now provides two notebook entry points, each with a separate responsibility.

### Notebook 1: preprocessing, segmentation, deduplication, and NIMBUS inference

```text
./notebooks/1_COMET_Workflow.ipynb
```

Use notebook 1 to move from raw multiplex whole-slide images to `nimbus_cell_table.csv`.

It covers:
1. Stage 1: inspect OME metadata from the first raw slide, confirm channel order, then run image tiling, channel extraction, and Cellpose input preparation
2. Stage 2: run CellposeSAM segmentation on the 2-channel fused TIFF inputs
3. Stage 3: remove edge-touching masks and deduplicate overlapping cells across neighboring FOVs
4. Stage 4: run NIMBUS marker probability inference on the final masks and per-marker TIFFs

#### NIMBUS checkpoint troubleshooting

Current NIMBUS releases try to download the latest model checkpoint from the official Hugging Face repository:

```text
https://huggingface.co/JLrumberger/Nimbus-Inference
```

If that automatic download fails on your machine, manually download `V1.pt` from the official NIMBUS source and place it in the local `nimbus_inference/assets/` directory used by your Python environment. Common locations are:

```text
<conda-env>/lib/python3.x/site-packages/nimbus_inference/assets/V1.pt
<Nimbus-Inference-repo>/src/nimbus_inference/assets/V1.pt   # editable install
```

COMET does not redistribute third-party NIMBUS checkpoint files.

### Notebook 2: thresholding, cell classification, and optional CSV export

```text
./notebooks/2_Cell_Classification.ipynb
```

Use notebook 2 after notebook 1 has created `nimbus_cell_table.csv` for each slide.

It covers:
1. Previewing marker columns from one slide and confirming the marker names to threshold
2. Computing Otsu x factor thresholds and reviewing threshold plots
3. Applying optional mutex correction for marker pairs that should not remain double-positive
4. Assigning ordered cell types, where the first matched rule becomes `Cell_Type`
5. Writing `nimbus_cell_table_classified.csv` and optional `nimbus_cell_table_classified_qupath.csv`

When defining `cell_type_rules`, place more specific classes before broader parent classes. The first matched rule becomes `Cell_Type`; later matches are written to `Additional_Labels`.

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
comet.print_channel_metadata("my_experiment/Slide1.ome.tif")
comet.extract_channels_experiment(
    experiment_dir="my_experiment",
    channel_names=["DAPI", "NuclearMarker", "MemMarker1", "MemMarker2", "Marker"],
)

# Step 3: Prepare CellposeSAM input (normalized signal fusion)
comet.prepare_cellpose_inputs(
    base_dir="my_experiment/Slide1",
    nuclear_markers=["DAPI", "NuclearMarker"],
    membrane_markers=["MemMarker1", "MemMarker2"],
)

# Step 4: Run CellposeSAM (GPU recommended)
comet.run_cellpose_slide("my_experiment/Slide1")

# Step 5: Border clearing + overlap deduplication
comet.deduplicate_slide("my_experiment/Slide1")

# Step 6: NIMBUS marker probability scoring
comet.run_nimbus_slide(
    slide_dir="my_experiment/Patient1",
    include_channels=["MemMarker1", "MemMarker2", "NuclearMarker", "Marker"],
)

# Step 7: Classify and assign cell types
result = comet.threshold_slide(
    nimbus_csv="my_experiment/Slide1/nimbus_output/nimbus_cell_table.csv",
    markers=["CD3", "CD4", "CD8a", "CD68", "EOMES"],
    # col_map only needed if NIMBUS column names differ from marker names
    # e.g. col_map={"TCR": "TCRbeta"}
    # TCR -> the name you want; TCRbeta -> the name used in nimbus column
    mutex_pairs=[("CD4", "CD8a"), ("CD3", "CD68")],
    # Optionally applies mutual-exclusion correction to configured marker pairs.
    cell_type_rules=[
        {"name": "CD4_TCell", "positive": ["CD3", "CD4"], "negative": ["CD68", "CD8a"]},
        {"name": "CD8_TCell", "positive": ["CD3", "CD8a"], "negative": ["CD68", "CD4"]},
        {"name": "TCell", "positive": ["CD3"], "negative": ["CD68"]},
        {"name": "EOMES_Pos", "positive": ["EOMES"], "negative": []},
    ],
)

# Cell-type rule behavior:
# - Rules are evaluated in the order you provide.
# - Put more specific classes before broader parent classes.
# - The first matched rule becomes Cell_Type.
# - Any later matched rules are stored in Additional_Labels as a semicolon-delimited string.
# - Cells that match no rule keep Cell_Type="Unknown".

# Step 8: Export a fov/label-keyed CSV for downstream review or QuPath-side integration
comet.export_to_qupath(
    classified_csv="my_experiment/Slide1/nimbus_output/nimbus_cell_table_classified.csv",
)
```

---

## QuPath integration

COMET includes two QuPath-side scripts, depending on whether you want to import raw NIMBUS measurements or final COMET cell classes.

### Script 1: import masks plus NIMBUS measurements

```text
./Qupath/Import COMET masks and NIMBUS predictions into QuPath.groovy
```

Use this script after segmentation, deduplication, and NIMBUS inference are complete. It uses:

- `fov_coordinates.csv` to place each field of view back into whole-slide coordinates
- `segmentation/deepcell_output/` to import the final whole-cell masks
- `nimbus_output/nimbus_cell_table.csv` to attach NIMBUS prediction values to each imported cell as QuPath measurements

Typical usage in QuPath:

1. Open the corresponding whole-slide image.
2. Edit the `slideDir` variable in the script so it points to the COMET slide output directory, for example `my_experiment/Slide1`.
3. Run `Import COMET masks and NIMBUS predictions into QuPath.groovy` from the QuPath script editor.

After import, each cell detection in QuPath retains the NIMBUS-derived prediction columns as measurements. These imported values can be used for manual gating, measurement-based filtering, or ad hoc rule building inside QuPath.

### Script 2: import masks plus final COMET class labels

```text
./Qupath/Import COMET masks and cell class into Qupath.groovy
```

Use this script after notebook 2 has generated `nimbus_cell_table_classified_qupath.csv`. It uses:

- `fov_coordinates.csv` to place each field of view back into whole-slide coordinates
- `segmentation/deepcell_output/` to import the final whole-cell masks
- `nimbus_output/nimbus_cell_table_classified_qupath.csv` to assign the exported `Class` values to QuPath detections

This script is intended for cases where you want QuPath to open directly with final COMET class labels already assigned, rather than importing the raw NIMBUS probabilities first.

It now includes two configuration switches:

- `importClass = true/false` controls whether the `Class` column is assigned as the QuPath PathClass
- `importMeasurements = true/false` controls whether numeric columns from `nimbus_cell_table_classified_qupath.csv` are also imported as QuPath measurements

This makes script 2 usable in three modes:

- class only
- measurements only
- class plus measurements

If you want to refine classification interactively, QuPath can be used to create new PathClasses, threshold prediction columns, or define additional rule-based cell classes on top of either imported NIMBUS measurements or imported COMET class labels.

---

## Output layout

```text
my_experiment/
|-- Slide1.ome.tif
|-- Slide1/
|   |-- fov_coordinates.csv
|   |-- Tiles/
|   |   |-- FOV0.tif, FOV1.tif, ...
|   |-- image_data/
|   |   |-- FOV0/
|   |   |   |-- DAPI.tif
|   |   |   |-- Marker.tif
|   |   |   |-- ...
|   |-- segmentation/
|   |   |-- cellpose_input/          # 2-channel fused TIFF inputs for CellposeSAM
|   |   |-- cellpose_output/         # raw Cellpose output artifacts
|   |   |-- deepcell_output/         # renamed whole-cell masks; deduplicated in place later
|   |   |-- deepcell_output_bak/     # backup created during deduplication
|   |-- nimbus_output/
|   |   |-- nimbus_cell_table.csv
|   |   |-- nimbus_cell_table_classified.csv
|   |   |-- nimbus_cell_table_classified_qupath.csv   # optional
|   |   |-- thresholds_used.csv
|   |   |-- threshold_distributions.png
```

---

## Repository structure

```text
comet/
|-- __init__.py
|-- preprocessing/
|   |-- __init__.py
|   |-- tile_split.py       # WSI tiling
|   |-- channel_extract.py  # channel extraction and metadata-based channel handling
|   |-- pipeline.py         # notebook-facing Stage 1 helpers
|-- segmentation/
|   |-- __init__.py
|   |-- signal_prep.py      # normalized signal fusion
|   |-- run_cellpose.py     # CellposeSAM wrapper
|   |-- deduplication.py    # border clearing + overlap deduplication
|   |-- run_nimbus.py       # NIMBUS wrapper
|-- classification/
|   |-- __init__.py
|   |-- threshold.py        # Otsu × factor thresholding + cell typing
|-- export/
|   |-- __init__.py
|   |-- qupath_export.py    # QuPath CSV export
```

---

## Contributing

COMET is open for community use. If you encounter bugs, have questions, or want to suggest improvements, please open a [GitHub Issue](https://github.com/jiaqi-bio/COMET/issues). Pull requests are also welcome.

---

## Citation

If you use COMET in your research, please cite:

> [An IL-10 Driven Spatial Niche Regulates Efferocytosis During Resolution of Intestinal Tissue Inflammation] — Francis Chan Lab, Liangzhu Laboratory, Zhejiang University.

---


## Acknowledgements

- [CellposeSAM](https://github.com/MouseLand/cellpose) — Pachitariu et al., 2025
- [NIMBUS](https://github.com/angelolab/nimbus-inference) — Rumberger, Greenwald et al., *Nature Methods*, 2025
