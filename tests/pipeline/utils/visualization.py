"""Visualization functions for synthetic dataset validation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .config_generator import GeneratedDataset


def plot_3d_skeleton_motion(
    dataset: GeneratedDataset,
    output_path: Path,
    selected_joints: Optional[list[int]] = None,
) -> None:
    """Plot 3D trajectories of selected joints.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_path : Path
        Output file path
    selected_joints : list of int, optional
        Joint indices to plot. If None, plots root + leaf nodes.
    """
    if selected_joints is None:
        # Auto-select: root + all leaf nodes (joints with no children)
        parents = dataset.skeleton.parents
        K = len(parents)
        children = set()
        for k in range(K):
            if parents[k] >= 0:
                children.add(parents[k])

        leaf_nodes = [k for k in range(K) if k not in children]
        selected_joints = [0] + leaf_nodes  # Root + leaves

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_true = dataset.x_true
    joint_names = dataset.skeleton.joint_names

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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton Motion Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3D motion plot to {output_path}")


def plot_3d_pose_snapshots(
    dataset: GeneratedDataset, output_path: Path, num_frames: int = 9
) -> None:
    """Plot grid of 3D skeleton poses at evenly spaced timesteps.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_path : Path
        Output file path
    num_frames : int
        Number of frames to plot (default 9 for 3x3 grid)
    """
    x_true = dataset.x_true
    T = x_true.shape[0]
    parents = dataset.skeleton.parents

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

    print(f"Saved 3D pose snapshots to {output_path}")


def plot_2d_reprojection_montage(
    dataset: GeneratedDataset,
    output_path: Path,
    camera_idx: int = 0,
    num_frames: int = 9,
) -> None:
    """Plot 2D keypoints overlaid on a grid for one camera.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_path : Path
        Output file path
    camera_idx : int
        Camera index to visualize
    num_frames : int
        Number of frames to plot
    """
    y_2d = dataset.y_2d
    T = y_2d.shape[1]
    K = y_2d.shape[2]
    parents = dataset.skeleton.parents

    # Get image size
    if camera_idx < len(dataset.camera_metadata):
        img_w, img_h = dataset.camera_metadata[camera_idx]["image_size"]
    else:
        img_w, img_h = 1280, 720

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

    print(f"Saved 2D reprojection montage to {output_path}")


def plot_missingness_outlier_summary(
    dataset: GeneratedDataset, output_path: Path
) -> None:
    """Plot heatmaps showing missingness and outlier patterns.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_path : Path
        Output file path
    """
    y_2d = dataset.y_2d
    C, T, K, _ = y_2d.shape

    joint_names = dataset.skeleton.joint_names
    camera_names = [f"Cam{c}" for c in range(C)]
    if dataset.camera_metadata:
        camera_names = [meta["name"] for meta in dataset.camera_metadata]

    # Compute missingness per camera/joint
    nan_mask = np.isnan(y_2d).any(axis=3)  # (C, T, K)
    missingness_rate = nan_mask.mean(
        axis=1
    )  # (C, K) - fraction missing per camera/joint

    # Outlier detection (simplified: just mark as "potential outliers")
    # For visualization, we'll show observation std per camera/joint
    y_valid = np.where(np.isnan(y_2d), 0, y_2d)
    obs_std = np.std(y_valid, axis=(1, 3))  # (C, K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Missingness heatmap
    im1 = ax1.imshow(missingness_rate, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax1.set_xlabel("Joint", fontsize=12)
    ax1.set_ylabel("Camera", fontsize=12)
    ax1.set_title("Missingness Rate (Fraction NaN)", fontsize=14)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels(joint_names, rotation=45, ha="right")
    ax1.set_yticks(range(C))
    ax1.set_yticklabels(camera_names)

    # Add text annotations
    for c in range(C):
        for k in range(K):
            text = ax1.text(
                k,
                c,
                f"{missingness_rate[c, k]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.colorbar(im1, ax=ax1, label="Missingness Rate")

    # Observation std heatmap (proxy for outlier detection)
    im2 = ax2.imshow(obs_std, aspect="auto", cmap="YlOrRd")
    ax2.set_xlabel("Joint", fontsize=12)
    ax2.set_ylabel("Camera", fontsize=12)
    ax2.set_title("Observation Std Dev (pixels)", fontsize=14)
    ax2.set_xticks(range(K))
    ax2.set_xticklabels(joint_names, rotation=45, ha="right")
    ax2.set_yticks(range(C))
    ax2.set_yticklabels(camera_names)

    # Add text annotations
    for c in range(C):
        for k in range(K):
            text = ax2.text(
                k,
                c,
                f"{obs_std[c, k]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.colorbar(im2, ax=ax2, label="Std Dev (px)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved missingness/outlier summary to {output_path}")


def plot_state_timeline(dataset: GeneratedDataset, output_path: Path) -> None:
    """Plot state sequence over time and transition matrix.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_path : Path
        Output file path
    """
    z_true = dataset.z_true
    T = len(z_true)
    S = dataset.config["dataset_spec"]["states"]["num_states"]
    trans_matrix = np.array(
        dataset.config["dataset_spec"]["states"]["transition_matrix"]
    )

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
    im = ax2.imshow(trans_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
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
                f"{trans_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if trans_matrix[i, j] > 0.5 else "black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax2, label="Transition Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved state timeline to {output_path}")


def generate_all_figures(dataset: GeneratedDataset, output_dir: Path) -> None:
    """Generate all standard figures for a dataset.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to visualize
    output_dir : Path
        Output directory for figures (will create figures/ subdirectory)
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures for {dataset.config['meta']['name']}...")

    plot_3d_skeleton_motion(dataset, figures_dir / "motion_3d.png")
    plot_3d_pose_snapshots(dataset, figures_dir / "poses_3d.png")
    plot_2d_reprojection_montage(dataset, figures_dir / "reprojection_2d.png")
    plot_missingness_outlier_summary(dataset, figures_dir / "missingness.png")
    plot_state_timeline(dataset, figures_dir / "states.png")

    print(f"All figures saved to {figures_dir}")
