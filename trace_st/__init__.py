"""TraCE-ST"""

# Core / utils (formerly mcastle_utils_vBT.py)
from . import castle_core as mcastle

# Data I/O / xarray helpers
from .data_io import extract_box, get_data_mcastle

# Trajectory helpers
from .trajectory import get_mcastle_box

# Movement / clustering helpers
# NOTE: only re-export the stable public API. Internal helper functions are
# intentionally not re-exported to avoid breaking imports when refactoring.
from .movement import (
    pick_direction_then_weighted_delta,
    snap_delta_to_stencil,
    summarize_clusters,
)

# Plotting helpers
from .plotting import plot_stencil, plot_stencil_on_map, plot_stencil_graph

__all__ = [
    # namespace
    "mcastle",

    # data_io
    "extract_box", "get_data_mcastle",

    # trajectory
    "get_mcastle_box",

    # movement
    "pick_direction_then_weighted_delta",
    "snap_delta_to_stencil",
    "summarize_clusters",

    # plotting
    "plot_stencil",
    "plot_stencil_on_map",
    "plot_stencil_graph",
]
