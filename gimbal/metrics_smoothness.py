"""Smoothness metrics (speed, acceleration, jerk) for skeletal motion validation."""

import numpy as np
from typing import Dict, Any


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
