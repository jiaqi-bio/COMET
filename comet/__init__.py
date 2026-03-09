"""
COMET - COmbinatorial Marker Expression Typing
"""
from .preprocessing.tile_split import tile_slide, tile_experiment
from .preprocessing.channel_extract import (
    extract_channels_slide,
    extract_channels_experiment,
    print_channel_metadata,
)
from .segmentation.signal_prep import prepare_cellpose_inputs
from .segmentation.run_cellpose import run_cellpose_slide, run_cellpose_experiment
from .segmentation.deduplication import deduplicate_slide, deduplicate_experiment
from .segmentation.run_nimbus import run_nimbus_slide, run_nimbus_experiment
from .classification.threshold import (
    threshold_slide,
    compute_thresholds,
    classify_all_markers,
    assign_cell_types,
    plot_marker_thresholds,
)
from .export.qupath_export import (
    export_to_qupath,
    write_qupath_selection_script,
)

__version__ = "0.1.0"
