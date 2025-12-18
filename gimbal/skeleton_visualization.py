"""Visualization functions for skeletal motion data.

This module provides focused visualization functions for displaying skeletal motion:
- 3D motion trajectories
- 3D pose snapshots
- 2D keypoint reprojections
- State sequence timelines

These functions work with any skeletal data (synthetic or real motion capture).
"""

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


def plot_state_timeline(
    z_true: np.ndarray,
    transition_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot state sequence over time and transition matrix.

    Parameters
    ----------
    z_true : np.ndarray, shape (T,)
        Hidden state sequence
    transition_matrix : np.ndarray, shape (S, S)
        State transition probability matrix
    output_path : Path
        Output file path
    """
    T = len(z_true)
    S = transition_matrix.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # State timeline
    ax1.plot(z_true, drawstyle="steps-post", linewidth=2, color="blue")
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("State", fontsize=12)
    ax1.set_title("Hidden State Sequence", fontsize=14)
    ax1.set_yticks(range(S))
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T - 1)

    # Add state duration annotations (optional, for clarity)
    current_state = z_true[0]
    duration_start = 0
    for t in range(1, T):
        if z_true[t] != current_state:
            duration = t - duration_start
            ax1.axvspan(
                duration_start,
                t,
                alpha=0.2,
                color=f"C{current_state}",
                label=f"State {current_state}" if duration_start == 0 else "",
            )
            current_state = z_true[t]
            duration_start = t
    # Last segment
    ax1.axvspan(duration_start, T, alpha=0.2, color=f"C{current_state}")

    # Transition matrix heatmap
    im = ax2.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax2.set_xlabel("To State", fontsize=12)
    ax2.set_ylabel("From State", fontsize=12)
    ax2.set_title("State Transition Matrix", fontsize=14)
    ax2.set_xticks(range(S))
    ax2.set_yticks(range(S))

    # Add probability annotations
    for i in range(S):
        for j in range(S):
            text = ax2.text(
                j,
                i,
                f"{transition_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if transition_matrix[i, j] > 0.5 else "black",
                fontsize=10,
            )

    plt.colorbar(im, ax=ax2, label="Transition Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
