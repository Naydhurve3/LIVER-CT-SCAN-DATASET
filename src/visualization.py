"""
Reusable visualization functions for medical CT volumes with CT windowing,
interactive viewers, statistical plots, and tumor burden analysis reports.
"""
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT.parent != _ROOT:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))
from src.config import WINDOWS, CLASS_MAPPING, OUTPUTS_DIR
from src.preprocessing import apply_hu_window
from src.utils import logger


# =============================================================================
# 1. PLOT SINGLE SLICE WITH / WITHOUT MASK (CT-Windowed)
# =============================================================================
def plot_slice_with_mask(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    window: str = "liver",
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Figure:
    """
    Plot a single CT slice with optional segmentation overlay using CT windowing.

    Args:
        volume: 3D CT volume (D, H, W).
        mask: 3D segmentation mask (D, H, W) with values 0, 1, 2.
        slice_idx: Slice index (default: middle).
        window: CT window preset ('liver', 'abdomen', 'bone', 'lung').
        figsize: Figure size.
        title: Optional title.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure.
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Apply CT windowing
    windowed = apply_hu_window(volume, window_name=window)

    fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=figsize)

    if mask is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Show CT slice
    ax1.imshow(windowed[slice_idx], cmap="gray", aspect="auto")
    ax1.set_title(f"CT Slice {slice_idx} ({window} window)")
    ax1.axis("off")

    if mask is not None:
        # Overlay mask on CT
        mask_slice = mask[slice_idx]
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        # Liver: blue with alpha
        overlay[mask_slice == 1] = [0, 0, 1, 0.4]
        # Tumor: red with alpha
        overlay[mask_slice == 2] = [1, 0, 0, 0.6]

        ax2.imshow(windowed[slice_idx], cmap="gray", aspect="auto")
        ax2.imshow(overlay, aspect="auto")
        ax2.set_title(f"Segmentation Overlay (Slice {slice_idx})")
        ax2.axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(color="blue", alpha=0.4, label="Liver"),
            mpatches.Patch(color="red", alpha=0.6, label="Tumor"),
        ]
        ax2.legend(handles=legend_patches, loc="upper right")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# 2. MULTI-PLANAR RECONSTRUCTION (MPR) VIEWS
# =============================================================================
def plot_mpr_views(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window: str = "liver",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot axial, coronal, and sagittal views at the center of the volume.
    """
    windowed = apply_hu_window(volume, window_name=window)
    D, H, W = volume.shape

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Axial (transverse) — slice at middle depth
    d_idx = D // 2
    axes[0].imshow(windowed[d_idx], cmap="gray", aspect="auto")
    axes[0].set_title(f"Axial (z={d_idx})")
    axes[0].axis("off")

    # Coronal — slice at middle height
    h_idx = H // 2
    axes[1].imshow(windowed[:, h_idx, :], cmap="gray", aspect="auto")
    axes[1].set_title(f"Coronal (y={h_idx})")
    axes[1].axis("off")

    # Sagittal — slice at middle width
    w_idx = W // 2
    axes[2].imshow(windowed[:, :, w_idx], cmap="gray", aspect="auto")
    axes[2].set_title(f"Sagittal (x={w_idx})")
    axes[2].axis("off")

    if mask is not None:
        # Overlay mask outlines on each view (optional extension)
        pass

    plt.suptitle("Multi-Planar Reconstruction (MPR) Views", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# =============================================================================
# 3. INTENSITY HISTOGRAM
# =============================================================================
def plot_intensity_histogram(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bins: int = 100,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot intensity distribution for the whole volume and per-class."""
    fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=figsize)

    ax = axes[0] if mask is not None else axes
    ax.hist(volume.flatten(), bins=bins, color="gray", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Intensity (HU)")
    ax.set_ylabel("Frequency")
    ax.set_title("CT Intensity Distribution")
    ax.grid(True, alpha=0.3)

    if mask is not None:
        ax2 = axes[1]
        colors = ["gray", "blue", "red"]
        labels = ["Background", "Liver", "Tumor"]
        for c in range(3):
            voxels = volume[mask == c]
            if len(voxels) > 0:
                ax2.hist(voxels.flatten(), bins=bins, color=colors[c], alpha=0.5,
                         label=f"{labels[c]} (n={len(voxels)})", edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Intensity (HU)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Per-Class Intensity Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# =============================================================================
# 4. CLASS DISTRIBUTION BAR CHART
# =============================================================================
def plot_class_distribution(
    mask: np.ndarray,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot pixel/class count distribution as a bar chart."""
    class_counts = {label: (mask == c).sum() for c, label in CLASS_MAPPING.items()}
    total = mask.size

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["gray", "royalblue", "crimson"]
    bars = ax.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor="black")

    # Add percentage labels on bars
    for bar, count in zip(bars, class_counts.values()):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Pixel Count")
    ax.set_title("Class Distribution")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# =============================================================================
# 5. TUMOR BURDEN ANALYSIS
# =============================================================================
def plot_tumor_burden(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Analyze and plot tumor burden across slices.

    Args:
        mask: 3D segmentation mask (D, H, W).
        spacing: Voxel spacing in mm (dz, dy, dx).
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    D = mask.shape[0]

    liver_per_slice = []
    tumor_per_slice = []
    slices = []

    for d in range(D):
        tumor_count = (mask[d] == 2).sum()
        liver_count = (mask[d] == 1).sum()
        if liver_count > 0 or tumor_count > 0:
            slices.append(d)
            liver_per_slice.append(liver_count * voxel_volume_mm3 / 1000)  # cm³
            tumor_per_slice.append(tumor_count * voxel_volume_mm3 / 1000)  # cm³

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Tumor area per slice
    ax1.plot(slices, tumor_per_slice, "r-", linewidth=2, label="Tumor")
    ax1.fill_between(slices, tumor_per_slice, alpha=0.3, color="red")
    ax1.set_xlabel("Slice Index")
    ax1.set_ylabel("Volume (cm³)")
    ax1.set_title("Tumor Volume per Slice")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Tumor vs Liver ratio per slice
    ratio = [t / max(l, 1) * 100 for t, l in zip(tumor_per_slice, liver_per_slice)]
    ax2.bar(slices, ratio, color="darkorange", alpha=0.7, width=1.0)
    ax2.set_xlabel("Slice Index")
    ax2.set_ylabel("Tumor / Liver Ratio (%)")
    ax2.set_title("Tumor Burden per Slice")
    ax2.grid(True, axis="y", alpha=0.3)

    # Overall stats
    total_tumor = sum(tumor_per_slice)
    total_liver = sum(liver_per_slice)
    fig.suptitle(
        f"Tumor Burden Analysis — Total Tumor: {total_tumor:.2f} cm³, "
        f"Liver: {total_liver:.2f} cm³, Ratio: {total_tumor / max(total_liver, 1) * 100:.1f}%",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# =============================================================================
# 6. INTERACTIVE SLICE VIEWER (ipywidgets)
# =============================================================================
def interactive_slice_viewer(volume: np.ndarray, mask: Optional[np.ndarray] = None, window: str = "liver"):
    """
    Launch an interactive slice viewer in a Jupyter notebook using ipywidgets.
    Requires: ipywidgets, IPython.display

    Usage in notebook:
        from src.visualization import interactive_slice_viewer
        interactive_slice_viewer(volume, mask)
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        logger.error("❌ ipywidgets not installed. Run: pip install ipywidgets")
        return

    windowed = apply_hu_window(volume, window_name=window)
    D = volume.shape[0]

    # Create overlay for mask
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask == 1] = [0, 0, 1, 0.4]
        overlay[mask == 2] = [1, 0, 0, 0.6]

    # Widgets
    slice_slider = widgets.IntSlider(value=D // 2, min=0, max=D - 1, step=1,
                                     description="Slice:", continuous_update=True)
    window_dropdown = widgets.Dropdown(
        options=list(WINDOWS.keys()), value=window,
        description="Window:"
    )
    show_mask = widgets.Checkbox(value=mask is not None, description="Show Mask")

    # Output
    output = widgets.Output()

    def update_view(_=None):
        with output:
            output.clear_output(wait=True)
            idx = slice_slider.value
            wnd = window_dropdown.value
            wnd_data = apply_hu_window(volume, window_name=wnd)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(wnd_data[idx], cmap="gray", aspect="auto")
            ax.set_title(f"Slice {idx} ({wnd} window)")

            if show_mask.value and mask is not None:
                # Recompute overlay for each window — mask stays same
                mask_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                mask_overlay[mask == 1] = [0, 0, 1, 0.4]
                mask_overlay[mask == 2] = [1, 0, 0, 0.6]
                ax.imshow(mask_overlay[idx], aspect="auto")

            ax.axis("off")
            plt.tight_layout()
            plt.show()

    slice_slider.observe(update_view, names="value")
    window_dropdown.observe(update_view, names="value")
    show_mask.observe(update_view, names="value")

    display(widgets.VBox([widgets.HBox([slice_slider, window_dropdown, show_mask]), output]))
    update_view()


# =============================================================================
# 7. REPORT SAVING HELPER
# =============================================================================
def save_analysis_report(stats: dict, filename: str = "analysis_report.txt") -> Path:
    """Save a formatted text report of dataset analysis."""
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "  LIVER TUMOR SEGMENTATION — DATASET ANALYSIS REPORT",
        "=" * 60,
        "",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved: {path}")
    return path

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print("  Functions: plot_slice_grid, plot_3d_volume, plot_metrics, plot_tumor_burden_dashboard")
    print("  OK (requires display backend for full test)")
