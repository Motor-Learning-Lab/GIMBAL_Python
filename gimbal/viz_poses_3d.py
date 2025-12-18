"""3D skeleton pose snapshots visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List


def plot_skeleton_poses_3d(
    x_true: np.ndarray,
    parents: np.ndarray,
    output_path: Path,
    num_frames: int = 9,
    camera_positions: Optional[List[np.ndarray]] = None,
    camera_targets: Optional[List[np.ndarray]] = None,
) -> None:
    """Plot grid of 3D skeleton poses at evenly spaced timesteps.

    Parameters
    ----------
    x_true : np.ndarray, shape (T, K, 3)
        Joint positions over time
    parents : np.ndarray
        Parent indices for each joint
    output_path : Path
        Output file path
    num_frames : int
        Number of frames to plot (default 9 for 3x3 grid)
    camera_positions : list of np.ndarray, optional
        Camera positions for visualization
    camera_targets : list of np.ndarray, optional
        Camera look-at targets for visualization
    """
    T = x_true.shape[0]

    # Select evenly spaced frames
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_frames)))

    fig = plt.figure(figsize=(15, 15))

    for plot_idx, t in enumerate(frame_indices):
        ax = fig.add_subplot(grid_size, grid_size, plot_idx + 1, projection="3d")

        # Draw skeleton bones
        for k in range(1, len(parents)):
            parent = parents[k]
            if parent >= 0:
                p1 = x_true[t, parent]
                p2 = x_true[t, k]
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "b-", linewidth=2
                )

        # Draw joints
        ax.scatter(
            x_true[t, :, 0], x_true[t, :, 1], x_true[t, :, 2], c="red", s=50, zorder=5
        )

        # Add camera visualization if provided
        if camera_positions is not None and camera_targets is not None:
            for cam_idx, (position, target) in enumerate(
                zip(camera_positions, camera_targets)
            ):
                # Draw camera position
                ax.scatter(
                    position[0],
                    position[1],
                    position[2],
                    c="purple",
                    marker="^",
                    s=200,
                    edgecolors="black",
                    linewidths=2,
                    zorder=10,
                )

                # Draw view direction arrow
                direction = target - position
                direction = direction / np.linalg.norm(direction) * 10
                ax.quiver(
                    position[0],
                    position[1],
                    position[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    color="purple",
                    arrow_length_ratio=0.3,
                    linewidth=2,
                    alpha=0.7,
                )

                # Add camera label
                ax.text(
                    position[0],
                    position[1],
                    position[2] + 5,
                    f"Cam{cam_idx}",
                    fontsize=10,
                    weight="bold",
                    color="purple",
                    ha="center",
                )

        # Set consistent axis limits across all subplots
        all_coords = x_true.reshape(-1, 3)
        margin = 5
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)

        ax.set_title(f"Frame {t}")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle("3D Pose Snapshots", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
