"""Direction normalization metrics for skeletal motion validation."""

import numpy as np
from typing import Dict, Any


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
