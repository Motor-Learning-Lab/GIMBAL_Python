"""Metrics computation for synthetic dataset validation."""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import json

from .config_generator import GeneratedDataset


def compute_dataset_metrics(dataset: GeneratedDataset) -> Dict[str, Any]:
    """Compute comprehensive metrics for generated dataset.

    Parameters
    ----------
    dataset : GeneratedDataset
        Generated dataset

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics
    """
    metrics = {}

    # === Bone length consistency ===
    T, K, _ = dataset.x_true.shape
    skeleton = dataset.skeleton

    bone_length_deviations = []
    for t in range(T):
        for k in range(1, K):  # Skip root
            parent = skeleton.parents[k]
            actual_length = np.linalg.norm(
                dataset.x_true[t, k] - dataset.x_true[t, parent]
            )
            expected_length = skeleton.bone_lengths[k]
            if expected_length > 0:
                deviation = abs(actual_length - expected_length) / expected_length
                bone_length_deviations.append(deviation)

    metrics["bone_length"] = {
        "max_relative_deviation": float(np.max(bone_length_deviations)),
        "mean_relative_deviation": float(np.mean(bone_length_deviations)),
        "std_relative_deviation": float(np.std(bone_length_deviations)),
    }

    # === Direction normalization health ===
    direction_norms = np.linalg.norm(dataset.u_true[:, 1:], axis=2)  # (T, K-1)

    metrics["direction_normalization"] = {
        "mean_norm": float(np.mean(direction_norms)),
        "std_norm": float(np.std(direction_norms)),
        "min_norm": float(np.min(direction_norms)),
        "max_norm": float(np.max(direction_norms)),
        "near_zero_count": int(np.sum(direction_norms < 0.01)),
    }

    # === Smoothness metrics ===
    dt = dataset.config["meta"]["dt"]

    # Velocity (finite differences)
    velocities = np.diff(dataset.x_true, axis=0) / dt  # (T-1, K, 3)
    speeds = np.linalg.norm(velocities, axis=2)  # (T-1, K)

    # Acceleration
    accelerations = np.diff(velocities, axis=0) / dt  # (T-2, K, 3)
    accel_mags = np.linalg.norm(accelerations, axis=2)  # (T-2, K)

    # Jerk
    jerks = np.diff(accelerations, axis=0) / dt  # (T-3, K, 3)
    jerk_mags = np.linalg.norm(jerks, axis=2)  # (T-3, K)

    metrics["smoothness"] = {
        "speed": {
            "mean": float(np.mean(speeds)),
            "std": float(np.std(speeds)),
            "p95": float(np.percentile(speeds, 95)),
            "max": float(np.max(speeds)),
        },
        "acceleration": {
            "mean": float(np.mean(accel_mags)),
            "std": float(np.std(accel_mags)),
            "p95": float(np.percentile(accel_mags, 95)),
            "max": float(np.max(accel_mags)),
        },
        "jerk": {
            "mean": float(np.mean(jerk_mags)),
            "std": float(np.std(jerk_mags)),
            "p95": float(np.percentile(jerk_mags, 95)),
            "max": float(np.max(jerk_mags)),
        },
    }

    # === State sanity ===
    z_true = dataset.z_true
    S = dataset.config["dataset_spec"]["states"]["num_states"]

    # Dwell times per state
    dwell_times = {s: [] for s in range(S)}
    current_state = z_true[0]
    dwell_start = 0

    for t in range(1, len(z_true)):
        if z_true[t] != current_state:
            dwell_times[current_state].append(t - dwell_start)
            current_state = z_true[t]
            dwell_start = t
    dwell_times[current_state].append(len(z_true) - dwell_start)

    # Transition counts
    transition_counts = np.zeros((S, S), dtype=int)
    for t in range(1, len(z_true)):
        transition_counts[z_true[t - 1], z_true[t]] += 1

    metrics["states"] = {
        "num_states": S,
        "dwell_times": {
            s: {"mean": float(np.mean(times)) if times else 0.0, "count": len(times)}
            for s, times in dwell_times.items()
        },
        "transition_counts": transition_counts.tolist(),
        "single_state_check": {
            "is_single_state": (S == 1),
            "actual_unique_states": int(len(np.unique(z_true))),
        },
    }

    # === 2D observation sanity ===
    y_2d = dataset.y_2d
    C, T, K, _ = y_2d.shape

    # Count NaNs (missingness)
    nan_mask = np.isnan(y_2d).any(axis=3)  # (C, T, K)
    nan_fraction = float(np.mean(nan_mask))

    # Count valid observations and check bounds
    image_sizes = [cam["image_size"] for cam in dataset.camera_metadata]
    bounds_violations = 0
    total_valid = 0

    for c in range(C):
        w, h = image_sizes[c] if c < len(image_sizes) else [1280, 720]
        for t in range(T):
            for k in range(K):
                if not np.any(np.isnan(y_2d[c, t, k])):
                    total_valid += 1
                    u, v = y_2d[c, t, k]
                    if not (0 <= u <= w and 0 <= v <= h):
                        bounds_violations += 1

    metrics["2d_observations"] = {
        "total_observations": C * T * K,
        "valid_observations": total_valid,
        "nan_fraction": nan_fraction,
        "bounds_violations": bounds_violations,
        "bounds_violation_rate": (
            bounds_violations / total_valid if total_valid > 0 else 0.0
        ),
    }

    # Add config-specified noise levels for reference
    obs_spec = dataset.config["dataset_spec"]["observation"]
    metrics["2d_observations"]["config"] = {
        "noise_px": obs_spec["noise_px"],
        "outliers_enabled": obs_spec.get("outliers", {}).get("enabled", False),
        "missingness_enabled": obs_spec.get("missingness", {}).get("enabled", False),
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics to JSON file.

    Parameters
    ----------
    metrics : dict
        Computed metrics
    output_path : Path
        Output file path (should end in .json)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {output_path}")


def check_metrics_thresholds(
    metrics: Dict[str, Any], thresholds: Dict[str, Any]
) -> tuple[bool, list[str]]:
    """Check if metrics pass validation thresholds.

    Parameters
    ----------
    metrics : dict
        Computed metrics
    thresholds : dict
        Threshold values for each metric

    Returns
    -------
    passed : bool
        True if all thresholds passed
    failures : list of str
        List of failure messages (empty if passed=True)
    """
    failures = []

    # Bone length consistency
    if metrics["bone_length"]["max_relative_deviation"] > thresholds.get(
        "bone_length_max_dev", 0.01
    ):
        failures.append(
            f"Bone length max deviation {metrics['bone_length']['max_relative_deviation']:.6f} "
            f"exceeds threshold {thresholds.get('bone_length_max_dev', 0.01):.6f}"
        )

    # Direction normalization
    if abs(metrics["direction_normalization"]["mean_norm"] - 1.0) > thresholds.get(
        "direction_norm_tolerance", 0.01
    ):
        failures.append(
            f"Direction norm mean {metrics['direction_normalization']['mean_norm']:.6f} "
            f"deviates from 1.0 by more than {thresholds.get('direction_norm_tolerance', 0.01):.6f}"
        )

    if metrics["direction_normalization"]["near_zero_count"] > thresholds.get(
        "near_zero_directions", 0
    ):
        failures.append(
            f"Found {metrics['direction_normalization']['near_zero_count']} near-zero directions "
            f"(threshold: {thresholds.get('near_zero_directions', 0)})"
        )

    # State sanity (for single-state configs)
    if metrics["states"]["single_state_check"]["is_single_state"]:
        if metrics["states"]["single_state_check"]["actual_unique_states"] != 1:
            failures.append(
                f"Config specifies 1 state but got {metrics['states']['single_state_check']['actual_unique_states']} unique states"
            )

    # 2D observations sanity
    if metrics["2d_observations"]["bounds_violation_rate"] > thresholds.get(
        "bounds_violation_rate", 0.05
    ):
        failures.append(
            f"Bounds violation rate {metrics['2d_observations']['bounds_violation_rate']:.4f} "
            f"exceeds threshold {thresholds.get('bounds_violation_rate', 0.05):.4f}"
        )

    # Smoothness checks (jerk should not be extreme)
    # These thresholds are dataset-dependent, so we allow them to be optional
    if "jerk_p95_max" in thresholds:
        if metrics["smoothness"]["jerk"]["p95"] > thresholds["jerk_p95_max"]:
            failures.append(
                f"Jerk 95th percentile {metrics['smoothness']['jerk']['p95']:.2f} "
                f"exceeds threshold {thresholds['jerk_p95_max']:.2f}"
            )

    return len(failures) == 0, failures
