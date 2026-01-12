"""
Stage C: Triangulation for L00_minimal dataset

Triangulate 3D joint positions from cleaned 2D keypoints using DLT method.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gimbal


def plot_3d_skeleton_wireframe(
    x_3d: np.ndarray,
    parents: np.ndarray,
    joint_names: list[str],
    output_path: Path,
    frame_indices: list[int] = None,
    title: str = "3D Skeleton",
) -> None:
    """Plot 3D skeleton wireframe at selected frames."""
    T, K, _ = x_3d.shape

    if frame_indices is None:
        # Sample 4 evenly spaced frames
        frame_indices = [int(T * f) for f in [0.1, 0.35, 0.6, 0.85]]

    fig = plt.figure(figsize=(16, 4))

    for i, t in enumerate(frame_indices):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_title(f"Frame {t}")

        # Plot joints
        x_t = x_3d[t]  # (K, 3)
        ax.scatter(x_t[:, 0], x_t[:, 1], x_t[:, 2], c="red", s=50, alpha=0.8)

        # Plot bones
        for k in range(K):
            if parents[k] >= 0:
                parent_pos = x_t[parents[k]]
                child_pos = x_t[k]
                ax.plot(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]],
                    "b-",
                    linewidth=2,
                    alpha=0.6,
                )

        # Label joints
        for k, name in enumerate(joint_names):
            ax.text(x_t[k, 0], x_t[k, 1], x_t[k, 2], name, fontsize=8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_3d_trajectories(
    x_3d: np.ndarray, joint_names: list[str], output_path: Path
) -> None:
    """Plot 3D trajectories for all joints."""
    T, K, _ = x_3d.shape

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("3D Joint Trajectories")

    for dim, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.set_title(f"{label} Coordinate")
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{label} (mm)")

        for k in range(K):
            ax.plot(x_3d[:, k, dim], label=joint_names[k], alpha=0.7)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_triangulation_metrics(
    x_3d: np.ndarray,
    x_true: np.ndarray,
    parents: np.ndarray,
    bone_lengths_true: np.ndarray,
) -> Dict[str, Any]:
    """Compute triangulation quality metrics."""
    T, K, _ = x_3d.shape

    # NaN statistics
    n_nans = int(np.sum(np.isnan(x_3d)))
    nan_fraction = float(n_nans / (T * K * 3))

    # RMSE vs ground truth (ignore NaNs)
    diff = x_3d - x_true
    rmse_per_joint = [float(np.sqrt(np.nanmean(diff[:, k, :] ** 2))) for k in range(K)]
    rmse_global = float(np.sqrt(np.nanmean(diff**2)))

    # Bone length errors
    bone_length_errors = []
    for k in range(K):
        if parents[k] >= 0:
            parent_idx = parents[k]
            # Compute bone vectors
            bones = x_3d[:, k, :] - x_3d[:, parent_idx, :]  # (T, 3)
            lengths = np.sqrt(np.sum(bones**2, axis=1))  # (T,)

            # Compare to true length
            true_length = bone_lengths_true[k]
            if true_length > 0:
                errors = np.abs(lengths - true_length) / true_length
                mean_error = float(np.nanmean(errors))
                max_error = float(np.nanmax(errors))
                bone_length_errors.append(
                    {
                        "joint": k,
                        "mean_error_pct": mean_error * 100,
                        "max_error_pct": max_error * 100,
                    }
                )

    return {
        "nan_statistics": {
            "n_nans": n_nans,
            "nan_fraction": nan_fraction,
            "note": "NaNs indicate triangulation failure (insufficient cameras or poor conditioning)",
        },
        "rmse_vs_ground_truth": {
            "global_mm": rmse_global,
            "per_joint_mm": rmse_per_joint,
        },
        "bone_length_consistency": bone_length_errors,
        "summary": {
            "triangulation_success_rate": float(1.0 - nan_fraction),
            "mean_bone_length_error_pct": (
                float(np.mean([e["mean_error_pct"] for e in bone_length_errors]))
                if bone_length_errors
                else 0.0
            ),
        },
    }


def run_stage_c(
    dataset_dir: Path, stage_b_output_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    """
    Run Stage C: Triangulation

    Parameters
    ----------
    dataset_dir : Path
        Directory containing original dataset (for ground truth)
    stage_b_output_dir : Path
        Directory containing Stage B outputs (y_2d_clean.npz)
    output_dir : Path
        Directory for outputs (must contain figures/ subdirectory)

    Returns
    -------
    metrics : dict
        Triangulation results and metrics
    """
    print("=" * 80)
    print("STAGE C: Triangulation")
    print("=" * 80)

    # Load config
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    joint_names = config["dataset_spec"]["skeleton"]["joint_names"]
    parents = np.array(config["dataset_spec"]["skeleton"]["parents"])
    bone_lengths_true = np.array(
        config["dataset_spec"]["skeleton"]["lengths"], dtype=float
    )

    # Load cleaned 2D data from Stage B
    stage_b_data = np.load(stage_b_output_dir / "y_2d_clean.npz")
    y_2d_clean = stage_b_data["y_2d_clean"]

    # Load camera matrices and ground truth from original dataset
    dataset = np.load(dataset_dir / "dataset.npz", allow_pickle=True)
    camera_proj = dataset["camera_proj"]
    x_true = dataset["x_true"]

    C, T, K, _ = y_2d_clean.shape
    print(f"Cleaned 2D keypoints shape: {y_2d_clean.shape}")
    print(f"Cameras: {C}, Frames: {T}, Joints: {K}")

    # Triangulate
    print("\n[1/4] Triangulating 3D positions using DLT...")
    x_3d = gimbal.triangulate_multi_view(
        y_2d_clean, camera_proj, min_cameras=2, condition_threshold=1e6
    )
    print(f"  Triangulated shape: {x_3d.shape}")

    n_nans = np.sum(np.isnan(x_3d))
    print(f"  NaN points: {n_nans} / {T * K * 3} ({100 * n_nans / (T * K * 3):.2f}%)")

    # Compute metrics
    print("\n[2/4] Computing triangulation metrics...")
    metrics = compute_triangulation_metrics(x_3d, x_true, parents, bone_lengths_true)

    print(
        f"  RMSE vs ground truth: {metrics['rmse_vs_ground_truth']['global_mm']:.2f} mm"
    )
    print(
        f"  Triangulation success rate: {metrics['summary']['triangulation_success_rate'] * 100:.2f}%"
    )
    print(
        f"  Mean bone length error: {metrics['summary']['mean_bone_length_error_pct']:.2f}%"
    )

    # Quality checks
    quality_passed = True
    if metrics["summary"]["triangulation_success_rate"] < 0.95:
        print(f"  ⚠️  WARNING: Success rate < 95%")
        quality_passed = False
    else:
        print(f"  ✓ Success rate > 95%")

    if metrics["summary"]["mean_bone_length_error_pct"] > 5.0:
        print(f"  ⚠️  WARNING: Bone length error > 5%")
        quality_passed = False
    else:
        print(f"  ✓ Bone length error < 5%")

    # Plot 3D skeleton
    print("\n[3/4] Plotting 3D skeleton wireframes...")
    figures_dir = output_dir / "figures"
    plot_3d_skeleton_wireframe(
        x_3d, parents, joint_names, figures_dir / "triangulation_skeleton_snapshots.png"
    )
    print(f"  Saved skeleton snapshots")

    # Plot trajectories
    print("\n[4/4] Plotting 3D trajectories...")
    plot_3d_trajectories(
        x_3d, joint_names, figures_dir / "triangulation_trajectories.png"
    )
    print(f"  Saved trajectory plots")

    # Compile full metrics
    full_metrics = {
        "stage": "C_triangulation",
        "method": "DLT",
        "parameters": {"min_cameras": 2, "condition_threshold": 1e6},
        "metrics": metrics,
        "quality_check": {
            "success_rate_threshold": 0.95,
            "bone_error_threshold_pct": 5.0,
            "passed": quality_passed,
        },
    }

    # Save metrics
    output_path = output_dir / "triangulation_metrics.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage C complete. Metrics saved to {output_path}")

    # Save triangulated data for next stage
    triangulated_data_path = output_dir / "x_3d_triangulated.npz"
    np.savez(triangulated_data_path, x_3d=x_3d)
    print(f"✓ Triangulated 3D data saved to {triangulated_data_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP fix

    # Paths
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    stage_b_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    # Run stage
    metrics = run_stage_c(dataset_dir, stage_b_dir, output_dir)

    # Report
    if metrics["quality_check"]["passed"]:
        print("\n" + "=" * 80)
        print("STAGE C: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE C: FAILED - Quality thresholds not met")
        print("=" * 80)
        import sys

        sys.exit(1)
