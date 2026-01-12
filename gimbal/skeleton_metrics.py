"""Metrics for skeletal motion validation and quality assessment.

This module provides focused metric functions for analyzing skeletal motion data:
- Bone length consistency
- Direction normalization
- Smoothness (speed, acceleration, jerk)
- State sequence analysis
- 2D observation quality

These functions work with any skeletal data (synthetic or real motion capture).
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from .skeleton_config import SkeletonConfig


def compute_bone_length_metrics(
    x_true: np.ndarray, skeleton: SkeletonConfig
) -> Dict[str, Any]:
    """Compute bone length consistency metrics.

    Parameters
    ----------
    x_true : np.ndarray, shape (T, K, 3)
        Joint positions over time
    skeleton : SkeletonConfig
        Skeleton configuration

    Returns
    -------
    metrics : dict
        Dictionary with max_relative_deviation, mean_relative_deviation, std_relative_deviation
    """
    T, K, _ = x_true.shape
    bone_length_deviations = []

    for t in range(T):
        for k in range(1, K):  # Skip root
            parent = skeleton.parents[k]
            if parent < 0:
                continue

            actual_length = np.linalg.norm(x_true[t, k] - x_true[t, parent])
            expected_length = skeleton.bone_lengths[k]

            if expected_length > 0:
                deviation = abs(actual_length - expected_length) / expected_length
                bone_length_deviations.append(deviation)

    if not bone_length_deviations:
        return {
            "max_relative_deviation": 0.0,
            "mean_relative_deviation": 0.0,
            "std_relative_deviation": 0.0,
        }

    return {
        "max_relative_deviation": float(np.max(bone_length_deviations)),
        "mean_relative_deviation": float(np.mean(bone_length_deviations)),
        "std_relative_deviation": float(np.std(bone_length_deviations)),
    }


def compute_direction_metrics(u_true: np.ndarray) -> Dict[str, Any]:
    """Compute direction normalization health metrics.

    Parameters
    ----------
    u_true : np.ndarray, shape (T, K, 3)
        Normalized joint directions over time

    Returns
    -------
    metrics : dict
        Dictionary with mean_norm, std_norm, min_norm, max_norm, near_zero_count
    """
    # Compute norms for all non-root directions (skip first column which is root)
    direction_norms = np.linalg.norm(u_true[:, 1:], axis=2)  # (T, K-1)

    return {
        "mean_norm": float(np.mean(direction_norms)),
        "std_norm": float(np.std(direction_norms)),
        "min_norm": float(np.min(direction_norms)),
        "max_norm": float(np.max(direction_norms)),
        "near_zero_count": int(np.sum(direction_norms < 0.01)),
    }


def compute_smoothness_metrics(x_true: np.ndarray, dt: float) -> Dict[str, Any]:
    """Compute smoothness metrics: speed, acceleration, jerk.

    Parameters
    ----------
    x_true : np.ndarray, shape (T, K, 3)
        Joint positions over time
    dt : float
        Time step between frames

    Returns
    -------
    metrics : dict
        Nested dictionary with speed/acceleration/jerk statistics
    """
    # Velocity (finite differences)
    velocities = np.diff(x_true, axis=0) / dt  # (T-1, K, 3)
    speeds = np.linalg.norm(velocities, axis=2)  # (T-1, K)

    # Acceleration
    accelerations = np.diff(velocities, axis=0) / dt  # (T-2, K, 3)
    accel_mags = np.linalg.norm(accelerations, axis=2)  # (T-2, K)

    # Jerk
    jerks = np.diff(accelerations, axis=0) / dt  # (T-3, K, 3)
    jerk_mags = np.linalg.norm(jerks, axis=2)  # (T-3, K)

    return {
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


def compute_state_metrics(z_true: np.ndarray, num_states: int) -> Dict[str, Any]:
    """Compute state sequence metrics: dwell times, transitions.

    Parameters
    ----------
    z_true : np.ndarray, shape (T,)
        Hidden state sequence
    num_states : int
        Number of states in the HMM

    Returns
    -------
    metrics : dict
        Dictionary with dwell times, transition counts, and sanity checks
    """
    # Dwell times per state
    dwell_times = {s: [] for s in range(num_states)}
    current_state = z_true[0]
    dwell_start = 0

    for t in range(1, len(z_true)):
        if z_true[t] != current_state:
            dwell_times[current_state].append(t - dwell_start)
            current_state = z_true[t]
            dwell_start = t
    dwell_times[current_state].append(len(z_true) - dwell_start)

    # Transition counts
    transition_counts = np.zeros((num_states, num_states), dtype=int)
    for t in range(1, len(z_true)):
        transition_counts[z_true[t - 1], z_true[t]] += 1

    return {
        "num_states": num_states,
        "dwell_times": {
            s: {"mean": float(np.mean(times)) if times else 0.0, "count": len(times)}
            for s, times in dwell_times.items()
        },
        "transition_counts": transition_counts.tolist(),
        "single_state_check": {
            "is_single_state": (num_states == 1),
            "actual_unique_states": int(len(np.unique(z_true))),
        },
    }


def compute_observation_metrics(
    y_2d: np.ndarray,
    image_sizes: List[Tuple[int, int]],
    config_observation_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute 2D observation quality metrics.

    Parameters
    ----------
    y_2d : np.ndarray, shape (C, T, K, 2)
        2D observations with potential noise/outliers/missingness
    image_sizes : list of (width, height) tuples
        Image size for each camera
    config_observation_spec : dict
        Observation specification from config (noise_px, outliers, missingness)

    Returns
    -------
    metrics : dict
        Dictionary with NaN fraction, bounds violations, and config reference
    """
    C, T, K, _ = y_2d.shape

    # Count NaNs (missingness)
    nan_mask = np.isnan(y_2d).any(axis=3)  # (C, T, K)
    nan_fraction = float(np.mean(nan_mask))

    # Count valid observations and check bounds
    bounds_violations = 0
    total_valid = 0

    for c in range(C):
        w, h = image_sizes[c] if c < len(image_sizes) else (1280, 720)
        for t in range(T):
            for k in range(K):
                if not np.any(np.isnan(y_2d[c, t, k])):
                    total_valid += 1
                    u, v = y_2d[c, t, k]
                    if not (0 <= u <= w and 0 <= v <= h):
                        bounds_violations += 1

    return {
        "total_observations": C * T * K,
        "valid_observations": total_valid,
        "nan_fraction": nan_fraction,
        "bounds_violations": bounds_violations,
        "bounds_violation_rate": (
            bounds_violations / total_valid if total_valid > 0 else 0.0
        ),
        "config": {
            "noise_px": config_observation_spec["noise_px"],
            "outliers_enabled": config_observation_spec.get("outliers", {}).get(
                "enabled", False
            ),
            "missingness_enabled": config_observation_spec.get("missingness", {}).get(
                "enabled", False
            ),
        },
    }
