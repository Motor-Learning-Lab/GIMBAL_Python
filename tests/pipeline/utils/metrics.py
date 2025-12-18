"""Metrics computation for synthetic dataset validation.

This module wraps the focused metric functions from gimbal.metrics_* modules
for use in the pipeline. The focused functions in gimbal/ do not depend on
GeneratedDataset and can be reused elsewhere.
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import json

from .config_generator import GeneratedDataset

# Import focused metric functions from gimbal
from gimbal.metrics_bone_length import compute_bone_length_metrics
from gimbal.metrics_direction import compute_direction_metrics
from gimbal.metrics_smoothness import compute_smoothness_metrics
from gimbal.metrics_state import compute_state_metrics
from gimbal.metrics_observation import compute_observation_metrics


def compute_dataset_metrics(dataset: GeneratedDataset) -> Dict[str, Any]:
    """Compute comprehensive metrics for generated dataset.

    Wraps focused metric functions from gimbal.metrics_* modules.

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

    # Bone length consistency
    metrics["bone_length"] = compute_bone_length_metrics(
        x_true=dataset.x_true, skeleton=dataset.skeleton
    )

    # Direction normalization health
    metrics["direction_normalization"] = compute_direction_metrics(
        u_true=dataset.u_true
    )

    # Smoothness metrics
    dt = dataset.config["meta"]["dt"]
    metrics["smoothness"] = compute_smoothness_metrics(x_true=dataset.x_true, dt=dt)

    # State sanity
    num_states = dataset.config["dataset_spec"]["states"]["num_states"]
    metrics["states"] = compute_state_metrics(
        z_true=dataset.z_true, num_states=num_states
    )

    # 2D observation sanity
    image_sizes = [tuple(cam["image_size"]) for cam in dataset.camera_metadata]
    obs_spec = dataset.config["dataset_spec"]["observation"]
    metrics["2d_observations"] = compute_observation_metrics(
        y_2d=dataset.y_2d, image_sizes=image_sizes, config_observation_spec=obs_spec
    )

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
