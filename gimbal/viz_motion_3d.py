"""3D skeleton motion trajectory visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


def plot_skeleton_motion_3d(
    x_true: np.ndarray,
    joint_names: List[str],
    parents: np.ndarray,
    output_path: Path,
    selected_joints: Optional[List[int]] = None,
    camera_positions: Optional[List[np.ndarray]] = None,
    camera_targets: Optional[List[np.ndarray]] = None,
) -> None:
    """Plot 3D trajectories of selected joints with optional camera visualization.

    Parameters
    ----------
    x_true : np.ndarray, shape (T, K, 3)
        Joint positions over time
    joint_names : list of str
        Names of joints
    parents : np.ndarray
        Parent indices for each joint
    output_path : Path
        Output file path
    selected_joints : list of int, optional
        Joint indices to plot. If None, plots root + leaf nodes.
    camera_positions : list of np.ndarray, optional
        Camera positions for visualization
    camera_targets : list of np.ndarray, optional
        Camera look-at targets for visualization
    """
    if selected_joints is None:
        # Auto-select: root + all leaf nodes (joints with no children)
        K = len(parents)
        children = set()
        for k in range(K):
            if parents[k] >= 0:
                children.add(parents[k])

        leaf_nodes = [k for k in range(K) if k not in children]
        selected_joints = [0] + leaf_nodes  # Root + leaves

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot joint trajectories
    for j_idx in selected_joints:
        traj = x_true[:, j_idx, :]  # (T, 3)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            label=joint_names[j_idx],
            alpha=0.7,
            linewidth=2,
        )

    # Add camera visualization if provided
    if camera_positions is not None and camera_targets is not None:
        for cam_idx, (position, target) in enumerate(
            zip(camera_positions, camera_targets)
        ):
            # Draw camera position as a pyramid
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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton Motion Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
