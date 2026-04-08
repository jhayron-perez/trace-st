# NOTE: Auto-organized from original CausalBTs.py and mcastle_utils_vBT.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_stencil(
    stencil_graph: np.ndarray,
    stencil_val_matrix: np.ndarray = None,
    label_var_names: bool = True,
    show_colorbar: bool = False,
    label_colorbars: bool = False,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.pyplot.Axes = None,
    style: str = "color",
    monochrome_edge_color: str = "black",
):
    """
    Plots a custom graph based on the provided stencil graph and value matrix.

    Parameters:
    -----------
    stencil_graph : numpy.ndarray
        An adjacency matrix representing the stencil graph. The shape of this array should be (9, 9, 2). The first two
         dimensions are for each position of the 3x3 stencil. The third dimension corresponds to two possible time lags
         (tau and tau-1).
    stencil_val_matrix : np.ndarray, optional
        The value matrix corresponding to the stencil graph. If provided, it will be used to determine the color of the links.
    show_colorbar : bool, optional
        Whether to show the colorbar in the plot. Default is False.
    label_colorbars : bool, optional
        Whether to label the colorbar in the plot. Default is False.
    label_var_names : bool, optional
        Whether to label nodes with var_names. Default is True.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis object to use for the plot. If None, a new axis will be created. Default is None.
    style : str, optional
        The style of the plot. Can be "color" or "monochrome". Default is "color".
    monochrome_edge_color : str, optional
        The color for the monochrome style. Can be "black" or "white". Default is "black".

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis object containing the plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if label_var_names:
        var_names = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
    else:
        var_names = [""] * 9

    x_pos = list(np.array([[i for i in range(3)] for j in range(3)]).flatten())
    y_pos = [i for i in range(3) for j in range(3)]
    y_pos.reverse()
    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }

    if style == "color":
        tp.plot_graph(
            fig_ax=(fig, ax),
            graph=stencil_graph,
            val_matrix=stencil_val_matrix,
            link_label_fontsize=0.0,
            var_names=var_names,
            cmap_edges='seismic',
            node_pos=node_positions,
            show_colorbar=show_colorbar,
            link_colorbar_label="Cross-Dependence" if label_colorbars else None,
            node_colorbar_label="Inter-Dependence" if label_colorbars else None,
        )
    elif style == "monochrome":
        cmap_N = 256
        white_vals = np.ones((cmap_N, 4))
        black_vals = np.zeros((cmap_N, 4))
        white_cmap = ListedColormap(white_vals)
        black_cmap = ListedColormap(black_vals)

        if monochrome_edge_color == "white":
            cmap_edges = white_cmap
        else:
            cmap_edges = black_cmap

        tp.plot_graph(
            fig_ax=(fig, ax),
            val_matrix=stencil_val_matrix.round(),
            graph=stencil_graph,
            link_label_fontsize=0.0,
            arrowhead_size=80,
            cmap_edges=cmap_edges,
            cmap_nodes="binary",
            var_names=var_names,
            node_pos=node_positions,
            show_colorbar=show_colorbar,
            link_colorbar_label="Cross-Dependence" if label_colorbars else None,
            node_colorbar_label="Inter-Dependence" if label_colorbars else None,
        )

    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)

    ax.patch.set_alpha(0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax

def plot_stencil_on_map(
    ax,
    center_lat,
    center_lon,
    graph,
    val_matrix=None,
    lag_index=1,
    arrow_scale=3,
    arrow_width=0.008,
    cmap='seismic',
    vmin=None,
    vmax=None,
    node_color='black',
    return_scalar_mappable=False,
    spacing_deg = 6.5):
    """
    Plots a 3x3 stencil graph as arrows on top of a Cartopy map with optional continuous coloring and colorbar.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axes object with Cartopy projection where the stencil will be plotted.

    center_lat : float
        Latitude for the center of the stencil.

    center_lon : float
        Longitude for the center of the stencil.

    graph : np.ndarray of shape (9, 9, 2)
        Binary adjacency matrix with 1/0 or '-->' representing directed edges.

    val_matrix : np.ndarray of shape (9, 9, 2), optional
        Value matrix to color arrows by strength.

    lag_index : int
        Lag to visualize (0 or 1).

    arrow_scale : float
        Scaling for arrow length.

    arrow_width : float
        Width of arrows.

    cmap : str or Colormap
        Colormap to use for arrow coloring.

    vmin, vmax : float, optional
        Value range for color normalization. If None, inferred from val_matrix.

    node_color : str
        Color of the central node.

    return_scalar_mappable : bool
        If True, returns a ScalarMappable for external colorbar plotting.

    Returns:
    --------
    sm : ScalarMappable (only if return_scalar_mappable is True)
    """
    # Define stencil relative positions
    stencil_offsets = {
        0: (-1,  1),  # NW
        1: ( 0,  1),  # N
        2: ( 1,  1),  # NE
        3: (-1,  0),  # W
        4: ( 0,  0),  # C
        5: ( 1,  0),  # E
        6: (-1, -1),  # SW
        7: ( 0, -1),  # S
        8: ( 1, -1),  # SE
    }

    
    lat_pos = {k: center_lat + spacing_deg * dy for k, (_, dy) in stencil_offsets.items()}
    lon_pos = {k: center_lon + spacing_deg * dx for k, (dx, _) in stencil_offsets.items()}

    # Color normalization
    if val_matrix is not None:
        if vmin is None:
            vmin = np.nanmin(val_matrix[:, :, lag_index])
        if vmax is None:
            vmax = np.nanmax(val_matrix[:, :, lag_index])
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmax+vmin)/2, vmax=vmax)
        # cmap_obj = cm.get_cmap(cmap)
        cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    else:
        cmap_obj = None
        norm = None

    # Draw arrows
    for source in range(9):
        for target in range(9):
            if graph[source, target, lag_index] == '-->':
                lon_start, lat_start = lon_pos[source], lat_pos[source]
                lon_end, lat_end = lon_pos[target], lat_pos[target]
                dx, dy = lon_end - lon_start, lat_end - lat_start

                if val_matrix is not None:
                    value = val_matrix[source, target, lag_index]
                    color = cmap_obj(norm(value))
                else:
                    color = 'black'
                if (source==4)&(target==4):
                    value = val_matrix[4, 4, lag_index] if val_matrix is not None else np.nan
                    if np.isfinite(value) and np.abs(value) > 0:  # only plot if significant
                        color = cmap_obj(norm(value))
                        ax.plot(
                            lon_pos[4], lat_pos[4],
                            'o',
                            color=color,
                            transform=ccrs.PlateCarree(),
                            markersize=20,  # larger than the center node
                            zorder=0
                        )
                else:
                    ax.arrow(
                        lon_start, lat_start,
                        dx / arrow_scale, dy / arrow_scale,
                        transform=ccrs.PlateCarree(),
                        width=arrow_width,
                        head_width=arrow_width * 4,
                        head_length=arrow_width * 6,
                        color=color,
                        alpha=0.9,
                        length_includes_head=True,
                        zorder=20
                    )

    # Plot center node
    ax.plot(
        lon_pos[4], lat_pos[4],
        'o', color=node_color,
        transform=ccrs.PlateCarree(),
        markersize=5,
        zorder=21
    )

    # Return scalar mappable for colorbar
    if return_scalar_mappable and val_matrix is not None:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        return sm

def plot_stencil_graph(
    stencil_graph,
    stencil_val_matrix=None,
    head_width=2,
    head_length=1,
    tail_width=0.5,
    show_colorbar=False,
    var_names=None,
    directional_var_names=False,
    label_var_names=True,
    fig=None,
    ax=None,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=-1,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="RdBu_r",
    link_colorbar_label="MCI",
    node_colorbar_label="auto-MCI",
    node_label_size=None
):
    """
    Plots a stencil graph based on the provided stencil and value matrix.

    This version is independent of stencil size. It assumes that `stencil_graph`
    corresponds to a single stencil of size K x K nodes, where K = sqrt(N_nodes).

    Parameters
    ----------
    stencil_graph : np.ndarray
        Adjacency matrix of shape (K, K, tau_max+1).
    stencil_val_matrix : np.ndarray, optional
        Value matrix of same shape as stencil_graph.
    directional_var_names : bool, optional
        If True AND K == 3, uses ["NW","N","NE","W","C","E","SW","S","SE"].
        For other K, this option is ignored and generic names (or blanks) are used.
    var_names : list of str, optional
        Names per node (length K*K). If None, generic or directional names are used.
    label_var_names : bool, optional
        If False, node labels are suppressed.
    """

    from matplotlib.artist import Artist
    from tigramite import plotting as tp
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import matplotlib  # for matplotlib.text.Text

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # --- Infer stencil size and grid dimension ---
    n_nodes = stencil_graph.shape[0]
    assert stencil_graph.shape[0] == stencil_graph.shape[1], \
        "stencil_graph must be square in (nodes x nodes)"

    grid_len = int(math.sqrt(n_nodes))
    assert grid_len * grid_len == n_nodes, \
        f"Number of nodes ({n_nodes}) is not a perfect square; cannot form a grid."

    # --- Build node positions on a grid ---
    # x: 0..grid_len-1 repeated per row
    x_pos = np.tile(np.arange(grid_len), grid_len)

    # y: 0..grid_len-1 per row, but reversed so that higher index is 'north'
    # (top row = 'north')
    y_indices = np.arange(grid_len)[::-1]    # e.g., [2,1,0] for grid_len=3
    y_pos = np.repeat(y_indices, grid_len)

    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }

    # --- Handle node names ---
    if var_names is not None:
        # User provided explicit names: must match n_nodes
        assert len(var_names) == n_nodes, \
            "Length of var_names must match number of nodes in stencil_graph."
    else:
        if directional_var_names and grid_len == 3:
            # Classic 3x3 directional names
            var_names = ["NW", "N", "NE",
                         "W",  "C", "E",
                         "SW", "S", "SE"]
        else:
            # Generic names: indices or blanks
            var_names = [f"{i}" for i in range(n_nodes)]

    if not label_var_names:
        var_names_for_plot = [""] * n_nodes
    else:
        var_names_for_plot = var_names

    # --- Plot using Tigramite's plot_graph ---
    tp.plot_graph(
        fig_ax=(fig, ax),
        graph=stencil_graph,
        val_matrix=stencil_val_matrix,
        link_label_fontsize=0.0,
        var_names=var_names_for_plot,
        node_pos=node_positions,
        show_colorbar=show_colorbar,
        vmin_edges=vmin_edges,
        vmax_edges=vmax_edges,
        edge_ticks=edge_ticks,
        cmap_edges=cmap_edges,
        vmin_nodes=vmin_nodes,
        vmax_nodes=vmax_nodes,
        node_ticks=node_ticks,
        cmap_nodes=cmap_nodes,
        link_colorbar_label=link_colorbar_label,
        node_colorbar_label=node_colorbar_label,
        node_label_size=node_label_size,
    )
    # keep only edge colorbar
    if show_colorbar:
        for ax_i in fig.axes:
            if ax_i.get_ylabel() == node_colorbar_label:
                ax_i.remove()

    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            txt = child.get_text()
    
            if txt == "1":
                x, y = child.get_position()
    
                # node labels sit exactly on integer grid locations
                if not (float(x).is_integer() and float(y).is_integer()):
                    Artist.set_visible(child, False)

    ax.set_aspect("equal")
    return fig, ax
