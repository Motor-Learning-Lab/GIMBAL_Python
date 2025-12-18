"""Visualization functions for synthetic dataset validation.

This module wraps the focused visualization functions from gimbal.viz_* modules
for use in the pipeline. The focused functions in gimbal/ do not depend on
GeneratedDataset and can be reused elsewhere.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .config_generator import GeneratedDataset

# Import focused visualization functions from gimbal
from gimbal.skeleton_visualization import (
    plot_skeleton_motion_3d,
    plot_skeleton_poses_3d,
    plot_2d_reprojections,
    plot_state_timeline as plot_state_timeline_focused,
)


def plot_3d_skeleton_motion(
    dataset: GeneratedDataset,
    output_path: Path,
    selected_joints: Optional[list[int]] = None,
) -> None:
    """Plot 3D trajectories of selected joints with camera visualization.

    Wraps gimbal.viz_motion_3d.plot_skeleton_motion_3d for pipeline use.
    """
    # Extract camera positions and targets
    camera_positions = None
    camera_targets = None
    if dataset.camera_metadata:
        camera_positions = [
            np.array(cam["position"]) for cam in dataset.camera_metadata
        ]
        camera_targets = [np.array(cam["target"]) for cam in dataset.camera_metadata]

    plot_skeleton_motion_3d(
        x_true=dataset.x_true,
        joint_names=dataset.skeleton.joint_names,
        parents=dataset.skeleton.parents,
        output_path=output_path,
        selected_joints=selected_joints,
        camera_positions=camera_positions,
        camera_targets=camera_targets,
    )
    print(f"Saved 3D motion plot to {output_path}")


def plot_3d_pose_snapshots(
    dataset: GeneratedDataset, output_path: Path, num_frames: int = 9
) -> None:
    """Plot grid of 3D skeleton poses at evenly spaced timesteps with cameras.

    Wraps gimbal.viz_poses_3d.plot_skeleton_poses_3d for pipeline use.
    """
    # Extract camera positions and targets
    camera_positions = None
    camera_targets = None
    if dataset.camera_metadata:
        camera_positions = [
            np.array(cam["position"]) for cam in dataset.camera_metadata
        ]
        camera_targets = [np.array(cam["target"]) for cam in dataset.camera_metadata]

    plot_skeleton_poses_3d(
        x_true=dataset.x_true,
        parents=dataset.skeleton.parents,
        output_path=output_path,
        num_frames=num_frames,
        camera_positions=camera_positions,
        camera_targets=camera_targets,
    )
    print(f"Saved 3D pose snapshots to {output_path}")


def plot_2d_reprojection_montage(
    dataset: GeneratedDataset,
    output_path: Path,
    camera_idx: int = 0,
    num_frames: int = 9,
) -> None:
    """Plot 2D keypoints overlaid on a grid for one camera.

    Wraps gimbal.viz_reprojection_2d.plot_2d_reprojections for pipeline use.
    """
    # Get image size
    if camera_idx < len(dataset.camera_metadata):
        img_w, img_h = dataset.camera_metadata[camera_idx]["image_size"]
    else:
        img_w, img_h = 1280, 720

    plot_2d_reprojections(
        y_2d=dataset.y_2d,
        parents=dataset.skeleton.parents,
        output_path=output_path,
        camera_idx=camera_idx,
        num_frames=num_frames,
        image_size=(img_w, img_h),
    )
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

    Wraps gimbal.viz_state_timeline.plot_state_timeline for pipeline use.
    """
    trans_matrix = np.array(
        dataset.config["dataset_spec"]["states"]["transition_matrix"]
    )

    plot_state_timeline_focused(
        z_true=dataset.z_true,
        transition_matrix=trans_matrix,
        output_path=output_path,
    )
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
