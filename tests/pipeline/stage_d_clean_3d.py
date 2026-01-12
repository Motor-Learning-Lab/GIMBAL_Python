"""
Stage D: 3D Cleaning for L00_minimal dataset

Apply cleaning to triangulated 3D positions with skeleton-aware smoothing.
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


def plot_3d_cleaning_comparison(
    x_before: np.ndarray, x_after: np.ndarray, joint_names: list[str], output_path: Path
) -> None:
    """Plot before/after 3D cleaning comparison."""
    T, K, _ = x_before.shape

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("3D Cleaning Comparison")

    # Select representative joint (joint 1 - proximal)
    joint_idx = 1
    joint_name = joint_names[joint_idx]

    # X, Y, Z coordinates
    for dim, (label, ax_row) in enumerate(zip(["X", "Y", "Z"], axes)):
        # Before
        ax_row[0].set_title(f"{joint_name} - {label} Before")
        ax_row[0].plot(x_before[:, joint_idx, dim], "b-", linewidth=1, alpha=0.7)
        ax_row[0].set_xlabel("Frame")
        ax_row[0].set_ylabel(f"{label} (mm)")
        ax_row[0].grid(True, alpha=0.3)

        # After
        ax_row[1].set_title(f"{joint_name} - {label} After")
        ax_row[1].plot(x_after[:, joint_idx, dim], "r-", linewidth=1, alpha=0.7)
        ax_row[1].set_xlabel("Frame")
        ax_row[1].set_ylabel(f"{label} (mm)")
        ax_row[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bone_lengths_over_time(
    x_3d: np.ndarray,
    parents: np.ndarray,
    bone_lengths_true: np.ndarray,
    joint_names: list[str],
    output_path: Path,
    title: str = "Bone Lengths Over Time",
) -> None:
    """Plot bone length trajectories vs expected lengths."""
    T, K, _ = x_3d.shape

    fig, axes = plt.subplots(K - 1, 1, figsize=(14, 3 * (K - 1)))
    if K == 2:
        axes = [axes]

    fig.suptitle(title)

    bone_idx = 0
    for k in range(K):
        if parents[k] >= 0:
            # Compute bone vectors
            bones = x_3d[:, k, :] - x_3d[:, parents[k], :]
            lengths = np.sqrt(np.sum(bones**2, axis=1))

            true_length = bone_lengths_true[k]

            ax = axes[bone_idx]
            ax.plot(lengths, "b-", linewidth=1, alpha=0.7, label="Computed")
            ax.axhline(
                y=true_length,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"True ({true_length:.1f}mm)",
            )
            ax.set_title(f"Bone: {joint_names[parents[k]]} → {joint_names[k]}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Length (mm)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            bone_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_3d_cleaning_metrics(
    x_before: np.ndarray,
    x_after: np.ndarray,
    valid_mask: np.ndarray,
    use_for_stats_mask: np.ndarray,
    cleaning_summary: Dict,
    parents: np.ndarray,
    bone_lengths_true: np.ndarray,
) -> Dict[str, Any]:
    """Compute 3D cleaning metrics."""
    T, K, _ = x_before.shape

    # Position RMSE
    diff = x_after - x_before
    rmse_per_joint = [float(np.sqrt(np.nanmean(diff[:, k, :] ** 2))) for k in range(K)]
    rmse_global = float(np.sqrt(np.nanmean(diff**2)))

    # Max difference
    diff_mag = np.sqrt(np.sum(diff**2, axis=-1))
    max_diff_per_joint = [float(np.nanmax(diff_mag[:, k])) for k in range(K)]
    max_diff_global = float(np.nanmax(diff_mag))

    # NaN statistics
    nans_before = int(np.sum(np.isnan(x_before)))
    nans_after = int(np.sum(np.isnan(x_after)))

    # Bone length consistency after cleaning
    bone_errors = []
    for k in range(K):
        if parents[k] >= 0:
            bones = x_after[:, k, :] - x_after[:, parents[k], :]
            lengths = np.sqrt(np.sum(bones**2, axis=1))
            true_length = bone_lengths_true[k]

            if true_length > 0:
                errors = np.abs(lengths - true_length) / true_length
                bone_errors.append(
                    {
                        "joint": k,
                        "mean_error_pct": float(np.nanmean(errors) * 100),
                        "std_error_pct": float(np.nanstd(errors) * 100),
                        "max_error_pct": float(np.nanmax(errors) * 100),
                    }
                )

    # Valid frame statistics
    n_valid_frames = int(np.sum(valid_mask))
    n_stats_usable_points = int(np.sum(use_for_stats_mask))

    return {
        "rmse_change": {
            "global_mm": rmse_global,
            "per_joint_mm": rmse_per_joint,
            "max_diff_global_mm": max_diff_global,
            "max_diff_per_joint_mm": max_diff_per_joint,
        },
        "nan_statistics": {
            "before": nans_before,
            "after": nans_after,
            "introduced": nans_after - nans_before,
        },
        "bone_length_consistency": bone_errors,
        "frame_validity": {
            "valid_frames": n_valid_frames,
            "invalid_frames": T - n_valid_frames,
            "valid_fraction": float(n_valid_frames / T),
        },
        "stats_mask": {
            "usable_points": n_stats_usable_points,
            "total_points": T * K,
            "usable_fraction": float(n_stats_usable_points / (T * K)),
        },
        "cleaning_summary": {
            "n_jump_outliers": int(cleaning_summary["n_jump_outliers"]),
            "n_bone_outliers": int(cleaning_summary["n_bone_outliers"]),
            "n_interpolated": int(cleaning_summary["n_interpolated"]),
            "n_invalid_frames": int(cleaning_summary["n_invalid_frames"]),
        },
    }


def run_stage_d(
    dataset_dir: Path, stage_c_output_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    """
    Run Stage D: 3D Cleaning

    Parameters
    ----------
    dataset_dir : Path
        Directory containing original dataset (for skeleton config)
    stage_c_output_dir : Path
        Directory containing Stage C outputs (x_3d_triangulated.npz)
    output_dir : Path
        Directory for outputs

    Returns
    -------
    metrics : dict
        Cleaning results and metrics
    """
    print("=" * 80)
    print("STAGE D: 3D Cleaning")
    print("=" * 80)

    # Load config
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    joint_names = config["dataset_spec"]["skeleton"]["joint_names"]
    parents = np.array(config["dataset_spec"]["skeleton"]["parents"])
    bone_lengths_true = np.array(
        config["dataset_spec"]["skeleton"]["lengths"], dtype=float
    )

    # Load triangulated 3D from Stage C
    stage_c_data = np.load(stage_c_output_dir / "x_3d_triangulated.npz")
    x_3d = stage_c_data["x_3d"]

    T, K, _ = x_3d.shape
    print(f"Triangulated 3D shape: {x_3d.shape}")
    print(f"Frames: {T}, Joints: {K}")

    # Configure cleaning
    cleaning_config = CleaningConfig(
        jump_z_thresh=3.0, bone_z_thresh=3.0, max_gap=5, max_bad_joint_fraction=0.3
    )
    print(f"\nCleaning config:")
    print(f"  Jump Z-threshold: {cleaning_config.jump_z_thresh}")
    print(f"  Bone Z-threshold: {cleaning_config.bone_z_thresh}")
    print(f"  Max gap: {cleaning_config.max_gap}")
    print(f"  Max bad joint fraction: {cleaning_config.max_bad_joint_fraction}")

    # Apply cleaning
    print("\n[1/4] Applying 3D cleaning...")
    x_3d_clean, valid_mask, use_for_stats_mask, cleaning_summary = (
        gimbal.clean_keypoints_3d(x_3d, parents, cleaning_config)
    )

    print(f"  Cleaning summary:")
    print(f"    - Jump outliers detected: {cleaning_summary['n_jump_outliers']}")
    print(f"    - Bone outliers detected: {cleaning_summary['n_bone_outliers']}")
    print(f"    - Points interpolated: {cleaning_summary['n_interpolated']}")
    print(f"    - Invalid frames: {cleaning_summary['n_invalid_frames']}")

    # Compute metrics
    print("\n[2/4] Computing cleaning metrics...")
    metrics = compute_3d_cleaning_metrics(
        x_3d,
        x_3d_clean,
        valid_mask,
        use_for_stats_mask,
        cleaning_summary,
        parents,
        bone_lengths_true,
    )

    print(f"  RMSE change: {metrics['rmse_change']['global_mm']:.4f} mm")
    print(
        f"  Valid frames: {metrics['frame_validity']['valid_frames']} / {T} ({metrics['frame_validity']['valid_fraction']*100:.1f}%)"
    )
    print(f"  Stats-usable points: {metrics['stats_mask']['usable_fraction']*100:.1f}%")

    mean_bone_error = np.mean(
        [e["mean_error_pct"] for e in metrics["bone_length_consistency"]]
    )
    print(f"  Mean bone length error: {mean_bone_error:.2f}%")

    # Quality check
    quality_passed = True
    if metrics["rmse_change"]["global_mm"] > 1.0:
        print(f"  ⚠️  WARNING: RMSE change > 1mm")
        quality_passed = False
    else:
        print(f"  ✓ RMSE change < 1mm")

    if mean_bone_error > 5.0:
        print(f"  ⚠️  WARNING: Bone length error > 5%")
        quality_passed = False
    else:
        print(f"  ✓ Bone length error < 5%")

    # Plot comparisons
    print("\n[3/4] Plotting cleaning comparison...")
    figures_dir = output_dir / "figures"
    plot_3d_cleaning_comparison(
        x_3d, x_3d_clean, joint_names, figures_dir / "clean_3d_comparison.png"
    )
    print(f"  Saved comparison plot")

    # Plot bone lengths
    print("\n[4/4] Plotting bone length consistency...")
    plot_bone_lengths_over_time(
        x_3d_clean,
        parents,
        bone_lengths_true,
        joint_names,
        figures_dir / "clean_3d_bone_lengths.png",
        title="Bone Lengths After 3D Cleaning",
    )
    print(f"  Saved bone length plot")

    # Compile full metrics
    full_metrics = {
        "stage": "D_3d_cleaning",
        "cleaning_config": {
            "jump_z_thresh": cleaning_config.jump_z_thresh,
            "bone_z_thresh": cleaning_config.bone_z_thresh,
            "max_gap": cleaning_config.max_gap,
            "max_bad_joint_fraction": cleaning_config.max_bad_joint_fraction,
        },
        "metrics": metrics,
        "quality_check": {
            "rmse_threshold_mm": 1.0,
            "bone_error_threshold_pct": 5.0,
            "passed": quality_passed,
        },
    }

    # Save metrics
    output_path = output_dir / "cleaning_3d_metrics.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage D complete. Metrics saved to {output_path}")

    # Save cleaned data for next stages
    cleaned_data_path = output_dir / "x_3d_clean.npz"
    np.savez(
        cleaned_data_path,
        x_3d_clean=x_3d_clean,
        valid_frame_mask=valid_mask,
        use_for_stats_mask=use_for_stats_mask,
    )
    print(f"✓ Cleaned 3D data saved to {cleaned_data_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    stage_c_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    metrics = run_stage_d(dataset_dir, stage_c_dir, output_dir)

    if metrics["quality_check"]["passed"]:
        print("\n" + "=" * 80)
        print("STAGE D: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE D: WARNING - Quality thresholds not met")
        print("=" * 80)
