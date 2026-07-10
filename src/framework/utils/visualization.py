from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def plot_slice_with_mask(image: np.ndarray, mask: np.ndarray,
                          pred: Optional[np.ndarray] = None,
                          uncertainty: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None,
                          title: str = ""):
    n_cols = 2 + (pred is not None) + (uncertainty is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Image")
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    col = 2
    if pred is not None:
        axes[col].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[col].set_title("Prediction")
        axes[col].axis('off')
        col += 1
    if uncertainty is not None:
        im = axes[col].imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
        axes[col].set_title("Uncertainty")
        axes[col].axis('off')
        plt.colorbar(im, ax=axes[col], fraction=0.046)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tumor_burden(slice_indices: np.ndarray, tumor_areas: np.ndarray,
                      save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(slice_indices, tumor_areas, 'b-', linewidth=1.5)
    ax.fill_between(slice_indices, tumor_areas, alpha=0.3)
    ax.set_xlabel("Slice Index")
    ax.set_ylabel("Tumor Area (pixels)")
    ax.set_title("Tumor Burden per Slice")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_calibration_curve(confidences: np.ndarray, accuracies: np.ndarray,
                            bin_counts: list, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confidences, accuracies, 'o-', linewidth=2, label='Model')
    ax.fill_between(confidences, confidences, accuracies, alpha=0.2)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (c, a, cnt) in enumerate(zip(confidences, accuracies, bin_counts)):
        if cnt > 0:
            ax.annotate(str(cnt), (c, a), fontsize=8, ha='center')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
