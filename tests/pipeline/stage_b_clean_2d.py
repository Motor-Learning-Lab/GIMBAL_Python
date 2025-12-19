"""
Stage B: 2D Cleaning for L00_minimal dataset

Apply cleaning to 2D keypoints. Since L00 is clean data, this stage tests that
cleaning doesn't degrade quality (per Q4 in clarifications).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gimbal
from gimbal.data_cleaning import CleaningConfig


def plot_2d_cleaning_comparison(
    y_before: np.ndarray,
    y_after: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
    joint_names: list[str],
    camera_idx: int = 0,
) -> None:
    """Plot before/after cleaning comparison for one camera."""
    T = y_before.shape[1]
    K = len(joint_names)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Camera {camera_idx}: 2D Cleaning Comparison")

    # Select a representative joint (joint 1 - proximal)
    joint_idx = 1
    joint_name = joint_names[joint_idx]

    # X coordinate
    axes[0, 0].set_title(f"{joint_name} - X Coordinate")
    axes[0, 0].plot(
        y_before[camera_idx, :, joint_idx, 0],
        "b-",
        label="Before",
        alpha=0.7,
        linewidth=1,
    )
    axes[0, 0].plot(
        y_after[camera_idx, :, joint_idx, 0],
        "r--",
        label="After",
        alpha=0.7,
        linewidth=1,
    )
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Pixel X")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Y coordinate
    axes[0, 1].set_title(f"{joint_name} - Y Coordinate")
    axes[0, 1].plot(
        y_before[camera_idx, :, joint_idx, 1],
        "b-",
        label="Before",
        alpha=0.7,
        linewidth=1,
    )
    axes[0, 1].plot(
        y_after[camera_idx, :, joint_idx, 1],
        "r--",
        label="After",
        alpha=0.7,
        linewidth=1,
    )
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Pixel Y")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pixel-wise difference
    axes[1, 0].set_title(f"{joint_name} - Pixel Difference Magnitude")
    diff_mag = np.sqrt(
        np.sum(
            (
                y_after[camera_idx, :, joint_idx, :]
                - y_before[camera_idx, :, joint_idx, :]
            )
            ** 2,
            axis=1,
        )
    )
    axes[1, 0].plot(diff_mag, "g-", linewidth=1)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Euclidean Distance (pixels)")
    axes[1, 0].axhline(y=1.0, color="orange", linestyle="--", label="1px threshold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Valid frame mask
    axes[1, 1].set_title("Frame Validity")
    axes[1, 1].plot(valid_mask[camera_idx], "k-", linewidth=1)
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Valid (1) / Invalid (0)")
    axes[1, 1].set_ylim([-0.1, 1.1])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"clean_2d_camera_{camera_idx}_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def compute_cleaning_metrics(
    y_before: np.ndarray,
    y_after: np.ndarray,
    valid_mask: np.ndarray,
    cleaning_summary: Dict,
) -> Dict[str, Any]:
    """Compute metrics comparing before/after cleaning."""
    C, T, K, _ = y_before.shape

    # Pixel-wise RMSE (ignoring NaNs)
    diff = y_after - y_before
    rmse_per_camera = [float(np.sqrt(np.nanmean(diff[c] ** 2))) for c in range(C)]
    rmse_global = float(np.sqrt(np.nanmean(diff**2)))

    # Max difference (ignoring NaNs)
    diff_mag = np.sqrt(np.sum(diff**2, axis=-1))  # (C, T, K)
    max_diff_per_camera = [float(np.nanmax(diff_mag[c])) for c in range(C)]
    max_diff_global = float(np.nanmax(diff_mag))

    # NaN statistics
    nans_before = int(np.sum(np.isnan(y_before)))
    nans_after = int(np.sum(np.isnan(y_after)))

    # Valid frames
    invalid_frames_per_camera = [int(np.sum(~valid_mask[c])) for c in range(C)]

    return {
        "rmse": {"global": rmse_global, "per_camera": rmse_per_camera},
        "max_difference": {
            "global": max_diff_global,
            "per_camera": max_diff_per_camera,
        },
        "nan_counts": {
            "before": nans_before,
            "after": nans_after,
            "introduced": nans_after - nans_before,
        },
        "frame_validity": {
            "invalid_frames_per_camera": invalid_frames_per_camera,
            "total_invalid": int(np.sum(invalid_frames_per_camera)),
        },
        "cleaning_summary": {
            "n_jump_outliers": int(cleaning_summary["n_jump_outliers"]),
            "n_bone_outliers": int(cleaning_summary["n_bone_outliers"]),
            "n_interpolated": int(cleaning_summary["n_interpolated"]),
            "n_invalid_frames": int(cleaning_summary["n_invalid_frames"]),
        },
    }


def run_stage_b(dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Run Stage B: 2D Cleaning

    Parameters
    ----------
    dataset_dir : Path
        Directory containing dataset.npz
    output_dir : Path
        Directory for outputs (must contain figures/ subdirectory)

    Returns
    -------
    metrics : dict
        Cleaning results and metrics
    """
    print("=" * 80)
    print("STAGE B: 2D Cleaning")
    print("=" * 80)

    # Load config for metadata
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    joint_names = config["dataset_spec"]["skeleton"]["joint_names"]
    parents = np.array(config["dataset_spec"]["skeleton"]["parents"])

    # Load dataset
    dataset_path = dataset_dir / "dataset.npz"
    data = np.load(dataset_path, allow_pickle=True)
    y_2d = data["y_2d"]  # (C, T, K, 2)

    C, T, K, _ = y_2d.shape
    print(f"2D keypoints shape: {y_2d.shape}")
    print(f"Cameras: {C}, Frames: {T}, Joints: {K}")

    # Configure cleaning (conservative for L00 clean data)
    cleaning_config = CleaningConfig(
        jump_z_thresh=3.0,  # Conservative - only catch extreme outliers
        bone_z_thresh=3.0,
        max_gap=5,
        max_bad_joint_fraction=0.3,
    )
    print(f"\nCleaning config:")
    print(f"  Jump Z-threshold: {cleaning_config.jump_z_thresh}")
    print(f"  Bone Z-threshold: {cleaning_config.bone_z_thresh}")
    print(f"  Max gap: {cleaning_config.max_gap}")
    print(f"  Max bad joint fraction: {cleaning_config.max_bad_joint_fraction}")

    # Apply cleaning
    print("\n[1/3] Applying 2D cleaning...")
    y_2d_clean, valid_mask, cleaning_summary = gimbal.clean_keypoints_2d(
        y_2d, parents, cleaning_config
    )

    print(f"  Cleaning summary:")
    print(f"    - Jump outliers detected: {cleaning_summary['n_jump_outliers']}")
    print(f"    - Bone outliers detected: {cleaning_summary['n_bone_outliers']}")
    print(f"    - Points interpolated: {cleaning_summary['n_interpolated']}")
    print(f"    - Invalid frames: {cleaning_summary['n_invalid_frames']}")

    # Compute metrics
    print("\n[2/3] Computing cleaning metrics...")
    metrics = compute_cleaning_metrics(y_2d, y_2d_clean, valid_mask, cleaning_summary)

    print(f"  RMSE (pixels): {metrics['rmse']['global']:.4f}")
    print(f"  Max difference (pixels): {metrics['max_difference']['global']:.4f}")
    print(f"  NaNs introduced: {metrics['nan_counts']['introduced']}")

    # Quality check for clean data
    if metrics["rmse"]["global"] > 1.0:
        print(
            f"  ⚠️  WARNING: RMSE > 1px on clean data - cleaning may be too aggressive"
        )
    else:
        print(f"  ✓ RMSE < 1px - cleaning preserved data quality")

    # Plot comparisons
    print("\n[3/3] Generating plots...")
    figures_dir = output_dir / "figures"

    # Plot for each camera
    for c in range(C):
        plot_2d_cleaning_comparison(
            y_2d, y_2d_clean, valid_mask, figures_dir, joint_names, camera_idx=c
        )

    print(f"  Saved {C} comparison plots to {figures_dir}")

    # Compile full metrics
    full_metrics = {
        "stage": "B_2d_cleaning",
        "dataset_path": str(dataset_path),
        "cleaning_config": {
            "jump_z_thresh": cleaning_config.jump_z_thresh,
            "bone_z_thresh": cleaning_config.bone_z_thresh,
            "max_gap": cleaning_config.max_gap,
            "max_bad_joint_fraction": cleaning_config.max_bad_joint_fraction,
        },
        "metrics": metrics,
        "quality_check": {
            "rmse_threshold_px": 1.0,
            "passed": metrics["rmse"]["global"] <= 1.0,
            "note": "For clean L00 data, cleaning should have minimal effect",
        },
    }

    # Save metrics
    output_path = output_dir / "cleaning_2d_metrics.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage B complete. Metrics saved to {output_path}")

    # Save cleaned data for next stage
    cleaned_data_path = output_dir / "y_2d_clean.npz"
    np.savez(cleaned_data_path, y_2d_clean=y_2d_clean, valid_mask_2d=valid_mask)
    print(f"✓ Cleaned 2D data saved to {cleaned_data_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP fix

    # Paths
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    # Run stage
    metrics = run_stage_b(dataset_dir, output_dir)

    # Report
    if metrics["quality_check"]["passed"]:
        print("\n" + "=" * 80)
        print("STAGE B: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE B: WARNING - High RMSE on clean data")
        print("=" * 80)
