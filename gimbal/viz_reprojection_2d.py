"""2D keypoint reprojection visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def plot_2d_reprojections(
    y_2d: np.ndarray,
    parents: np.ndarray,
    output_path: Path,
    camera_idx: int = 0,
    num_frames: int = 9,
    image_size: Tuple[int, int] = (1280, 720),
) -> None:
    """Plot 2D keypoints overlaid on a grid for one camera.

    Parameters
    ----------
    y_2d : np.ndarray, shape (C, T, K, 2)
        2D observations
    parents : np.ndarray
        Parent indices for each joint
    output_path : Path
        Output file path
    camera_idx : int
        Camera index to visualize
    num_frames : int
        Number of frames to plot
    image_size : tuple of int
        (width, height) of image
    """
    T = y_2d.shape[1]
    K = y_2d.shape[2]
    img_w, img_h = image_size

    # Select evenly spaced frames
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    grid_size = int(np.ceil(np.sqrt(num_frames)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 12))
    axes = axes.flatten() if num_frames > 1 else [axes]

    for plot_idx, t in enumerate(frame_indices):
        ax = axes[plot_idx]

        # Draw bones
        for k in range(1, K):
            parent = parents[k]
            if parent >= 0:
                p1 = y_2d[camera_idx, t, parent]
                p2 = y_2d[camera_idx, t, k]

                # Only draw if both points are valid
                if not (np.any(np.isnan(p1)) or np.any(np.isnan(p2))):
                    ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=2, alpha=0.6
                    )

        # Draw keypoints
        valid_mask = ~np.isnan(y_2d[camera_idx, t, :, 0])
        if np.any(valid_mask):
            ax.scatter(
                y_2d[camera_idx, t, valid_mask, 0],
                y_2d[camera_idx, t, valid_mask, 1],
                c="red",
                s=50,
                zorder=5,
                label="Valid",
            )

        missing_mask = np.isnan(y_2d[camera_idx, t, :, 0])
        if np.any(missing_mask):
            # For missing points, show a marker at center (just for visualization)
            ax.scatter(
                [img_w / 2] * np.sum(missing_mask),
                [img_h / 2] * np.sum(missing_mask),
                c="gray",
                s=30,
                marker="x",
                alpha=0.3,
                label="Missing",
            )

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # Flip Y axis (image convention)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {t}", fontsize=10)
        ax.grid(True, alpha=0.2)

        if plot_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"2D Reprojections - Camera {camera_idx}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
