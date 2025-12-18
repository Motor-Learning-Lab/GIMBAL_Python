"""Observation quality metrics for 2D keypoint validation."""

import numpy as np
from typing import Dict, Any, List, Tuple


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
