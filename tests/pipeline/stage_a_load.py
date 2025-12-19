"""
Stage A: Load & Validation for L00_minimal dataset

This module loads the synthetic dataset from Step 3 and performs validation checks.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def load_dataset(dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load dataset from .npz file."""
    data = np.load(dataset_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def validate_shapes(
    dataset: Dict[str, np.ndarray], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate dataset shapes match expectations."""
    T = metadata["num_frames"]
    K = metadata["num_joints"] - 1  # Exclude root
    C = metadata["num_cameras"]

    errors = []
    checks = {}

    # Check 2D keypoints
    if "y_2d" in dataset:
        expected = (C, T, K + 1, 2)
        actual = dataset["y_2d"].shape
        passed = expected == actual
        checks["y_2d_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(f"y_2d shape mismatch: expected {expected}, got {actual}")

    # Check 3D ground truth
    if "x_true" in dataset:
        expected = (T, K + 1, 3)
        actual = dataset["x_true"].shape
        passed = expected == actual
        checks["x_true_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(f"x_true shape mismatch: expected {expected}, got {actual}")

    # Check camera matrices
    if "camera_proj" in dataset:
        expected = (C, 3, 4)
        actual = dataset["camera_proj"].shape
        passed = expected == actual
        checks["camera_proj_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(
                f"camera_proj shape mismatch: expected {expected}, got {actual}"
            )

    # Check parents
    if "parents" in dataset:
        expected = (K + 1,)
        actual = dataset["parents"].shape
        passed = expected == actual
        checks["parents_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(f"parents shape mismatch: expected {expected}, got {actual}")

    # Check bone lengths
    if "bone_lengths" in dataset:
        expected = (K + 1,)
        actual = dataset["bone_lengths"].shape
        passed = expected == actual
        checks["bone_lengths_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(
                f"bone_lengths shape mismatch: expected {expected}, got {actual}"
            )

    # Check HMM states
    if "z_true" in dataset:
        expected = (T,)
        actual = dataset["z_true"].shape
        passed = expected == actual
        checks["z_true_shape"] = {
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            errors.append(f"z_true shape mismatch: expected {expected}, got {actual}")

    return {"passed": len(errors) == 0, "errors": errors, "checks": checks}


def check_nans(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Check for NaN values in dataset."""
    nan_counts = {}
    for key in ["y_2d", "x_true", "camera_proj", "bone_lengths"]:
        if key in dataset:
            count = np.sum(np.isnan(dataset[key]))
            nan_counts[key] = int(count)

    total_nans = sum(nan_counts.values())
    return {
        "passed": total_nans == 0,
        "total_nans": total_nans,
        "nan_counts_by_array": nan_counts,
    }


def check_camera_invertibility(
    camera_proj: np.ndarray, tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Check that camera matrices are well-formed and invertible (via SVD)."""
    C = camera_proj.shape[0]
    results = {}
    errors = []

    for c in range(C):
        P = camera_proj[c]  # (3, 4)
        # Check rank via SVD
        _, s, _ = np.linalg.svd(P, full_matrices=False)
        rank = np.sum(s > tolerance)
        min_singular_value = s[-1]

        results[f"camera_{c}"] = {
            "rank": int(rank),
            "min_singular_value": float(min_singular_value),
            "invertible": bool(rank == 3),
        }

        if rank < 3:
            errors.append(f"Camera {c} has rank {rank} < 3 (singular)")

    return {"passed": len(errors) == 0, "errors": errors, "camera_results": results}


def plot_2d_trajectories(
    y_observed: np.ndarray, output_dir: Path, joint_names: list[str]
) -> None:
    """Plot 2D keypoint trajectories for each camera."""
    C, T, K_plus_1, _ = y_observed.shape

    for c in range(C):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Camera {c}: 2D Keypoint Trajectories")

        # X coordinates
        axes[0].set_title("X Coordinates")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Pixel X")
        for k in range(K_plus_1):
            axes[0].plot(y_observed[c, :, k, 0], label=joint_names[k], alpha=0.7)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Y coordinates
        axes[1].set_title("Y Coordinates")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Pixel Y")
        for k in range(K_plus_1):
            axes[1].plot(y_observed[c, :, k, 1], label=joint_names[k], alpha=0.7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"load_2d_camera_{c}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def run_stage_a(dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Run Stage A: Load & Validation

    Parameters
    ----------
    dataset_dir : Path
        Directory containing dataset.npz and metadata.json
    output_dir : Path
        Directory for outputs (must contain figures/ subdirectory)

    Returns
    -------
    metrics : dict
        Validation results
    """
    print("=" * 80)
    print("STAGE A: Load & Validation")
    print("=" * 80)

    # Load config (metadata)
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    # Extract metadata from config
    metadata = {
        "dataset_name": config["meta"]["name"],
        "num_frames": config["meta"]["T"],
        "num_joints": len(config["dataset_spec"]["skeleton"]["joint_names"]),
        "num_cameras": len(config["dataset_spec"]["cameras"]["cameras"]),
        "joint_names": config["dataset_spec"]["skeleton"]["joint_names"],
    }

    print(f"Dataset: {metadata['dataset_name']}")
    print(
        f"Frames: {metadata['num_frames']}, Joints: {metadata['num_joints']}, Cameras: {metadata['num_cameras']}"
    )

    # Load dataset
    dataset_path = dataset_dir / "dataset.npz"
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} arrays: {list(dataset.keys())}")

    # Validate shapes
    print("\n[1/3] Validating shapes...")
    shape_validation = validate_shapes(dataset, metadata)
    if shape_validation["passed"]:
        print("  ✓ All shapes correct")
    else:
        print("  ✗ Shape errors:")
        for error in shape_validation["errors"]:
            print(f"    - {error}")

    # Check NaNs
    print("\n[2/3] Checking for NaNs...")
    nan_check = check_nans(dataset)
    if nan_check["passed"]:
        print("  ✓ No NaNs found")
    else:
        print(f"  ✗ Found {nan_check['total_nans']} NaN values:")
        for key, count in nan_check["nan_counts_by_array"].items():
            if count > 0:
                print(f"    - {key}: {count} NaNs")

    # Check camera invertibility
    print("\n[3/3] Checking camera matrices...")
    camera_check = check_camera_invertibility(dataset["camera_proj"])
    if camera_check["passed"]:
        print("  ✓ All cameras invertible")
    else:
        print("  ✗ Camera errors:")
        for error in camera_check["errors"]:
            print(f"    - {error}")

    # Plot 2D trajectories
    print("\n[4/4] Plotting 2D trajectories...")
    figures_dir = output_dir / "figures"
    joint_names = metadata["joint_names"]
    plot_2d_trajectories(dataset["y_2d"], figures_dir, joint_names)
    print(f"  Saved {metadata['num_cameras']} plots to {figures_dir}")

    # Compile metrics
    metrics = {
        "stage": "A_load_validation",
        "dataset_path": str(dataset_path),
        "metadata": metadata,
        "validation": {
            "shape_validation": shape_validation,
            "nan_check": nan_check,
            "camera_check": camera_check,
        },
        "summary": {
            "all_passed": (
                shape_validation["passed"]
                and nan_check["passed"]
                and camera_check["passed"]
            )
        },
    }

    # Save metrics
    output_path = output_dir / "load_validation.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Stage A complete. Metrics saved to {output_path}")

    return metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP fix

    # Paths
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    # Run stage
    metrics = run_stage_a(dataset_dir, output_dir)

    # Report
    if metrics["summary"]["all_passed"]:
        print("\n" + "=" * 80)
        print("STAGE A: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE A: FAILED ✗")
        print("=" * 80)
        import sys

        sys.exit(1)
