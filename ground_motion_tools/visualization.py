import matplotlib.pyplot as plt
import numpy as np
import warnings


def show_gm(
    gm_data: np.ndarray,
    time_step: float,
    save_path: str = None,
    *,
    y_label: str = None,
    show_plot: bool = True,
    component_names: list = None,
    title: str = None,
    **kwargs,
) -> None:
    """
    Visualize ground motion data

    Parameters:
    gm_data : 1D or 2D array
        Ground motion data
    time_step : float
        Time step in seconds
    save_path : str, optional
        Path to save the figure
    **kwargs:
        title : str
            Title of the figure
        show_plot : bool
            Whether to show the plot (default: True)
        component_names : list of str
            Names of components for legend (required for 2D arrays)
    """
    plt.rcParams["font.family"] = (
        " Times New Roman, SimSun"  # 设置字体族，中文为SimSun，英文为Times New Roman
    )
    plt.rcParams["mathtext.fontset"] = "stix"  # 设置数学公式字体为stix

    if gm_data.ndim == 1:
        gm_data = gm_data.reshape(1, -1)

    # Prepare
    batch_size = gm_data.shape[0]
    n_steps = gm_data.shape[1]
    time = np.arange(n_steps) * time_step
    time_len = n_steps * time_step

    # Set figure size (6cm x 10cm)
    fig, ax = plt.subplots(figsize=(12 / 2.54, 5 / 2.54), dpi=150)

    # Create colormap for multiple components
    cmap = plt.cm.Pastel1
    # For discrete colormaps, use integer indices instead of linspace
    colors = [cmap(i % cmap.N) for i in range(batch_size)]

    for i in range(batch_size):
        ax.plot(time, gm_data[i, :], linewidth=0.7, color=colors[i])

    # Set x-axis limit to 60s
    ax.set_xlim(0, time_len)

    # Set labels
    ax.set_xlabel("Time (s)", fontsize=10.5)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10.5)
    else:
        ax.set_ylabel("Acceleration", fontsize=10.5)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=10.5)

    # Add legend for multiple components
    if component_names:
        if batch_size == len(component_names):
            ax.legend(component_names, loc="upper right", fontsize=9)
        else:
            warnings.warn(
                f"Number of component names ({len(component_names)}) does not match batch size ({batch_size}). Legend will not be shown."
            )

    # Set tick label font size
    ax.tick_params(axis="both", which="major", labelsize=10.5)

    # Set Y-axis to scientific notation
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add grid with gray dashed lines
    ax.grid(True, linestyle="--", color="gray")

    # Set linewidth of the axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Set ticks direction inward
    ax.tick_params(direction="in")

    # Adjust layout with compact padding
    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)

    # Show plot if show_plot is True
    show_plot = kwargs.get("show_plot", True)
    if show_plot:
        plt.show()

    # Close the figure to free memory
    plt.close(fig)


def show_gm_spectrum(
    spectrum_data: np.ndarray,
    save_path: str = None,
    *,
    y_label: str = None,
    show_plot: bool = True,
    component_names: list = None,
    **kwargs,
) -> None:
    """
    Visualize ground motion spectrum data

    Parameters:
    spectrum_data : 1D or 2D array
        Spectrum data (period vs spectral acceleration)
    save_path : str, optional
        Path to save the figure
    y_label : str, optional
        Y-axis label
    show_plot : bool
        Whether to show the plot (default: True)
    component_names : list of str
        Names of components for legend (required for 2D arrays)
    **kwargs:
        title : str
            Title of the figure
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    if spectrum_data.ndim == 1:
        spectrum_data = spectrum_data.reshape(1, -1)

    # Prepare
    batch_size = spectrum_data.shape[0]
    # Set figure size (8cm x 6cm)
    fig, ax = plt.subplots(figsize=(8 / 2.54, 6 / 2.54), dpi=150)

    # Create colormap for multiple components
    cmap = plt.cm.Pastel1
    # For discrete colormaps, use integer indices instead of linspace
    colors = [cmap(i % cmap.N) for i in range(batch_size)]

    # Plot each component
    for i in range(batch_size):
        ax.plot(spectrum_data[i, :], linewidth=0.7, color=colors[i])

    # Set labels
    ax.set_xlabel("Period (s)", fontsize=10.5)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10.5)
    else:
        ax.set_ylabel("Spectral Acceleration", fontsize=10.5)

    # Add title if provided
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=10.5)

    # Add legend for multiple components
    if component_names:
        if batch_size == len(component_names):
            ax.legend(component_names, loc="upper right", fontsize=9)
        else:
            warnings.warn(
                f"Number of component names ({len(component_names)}) does not match batch size ({batch_size}). Legend will not be shown."
            )

    # Set tick label font size
    ax.tick_params(axis="both", which="major", labelsize=10.5)

    # Set Y-axis to scientific notation
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add grid with gray dashed lines
    ax.grid(True, linestyle="--", color="gray")

    # Set linewidth of the axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Set ticks direction inward
    ax.tick_params(direction="in")

    # Adjust layout with compact padding
    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)

    # Show plot if show_plot is True
    if show_plot:
        plt.show()

    # Close the figure to free memory
    plt.close(fig)
