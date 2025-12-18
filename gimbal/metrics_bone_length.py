"""Bone length consistency metrics for skeletal motion validation."""

import numpy as np
from typing import Dict, Any

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
