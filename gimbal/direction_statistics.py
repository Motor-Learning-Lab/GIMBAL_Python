"""Direction statistics for 3D joint position data.

This module computes empirical directional statistics (mean direction and
concentration) for bone orientations in 3D space using von Mises-Fisher (vMF)
distribution approximation.

v0.2.1 addition for data-driven priors pipeline.
"""

from typing import Dict, List, Optional

import numpy as np


def _compute_vmf_mean_direction(directions: np.ndarray) -> np.ndarray:
    """
    Compute mean direction on the unit sphere.

    Parameters
    ----------
    directions : ndarray, shape (N, 3)
        Unit vectors on the sphere

    Returns
    -------
    mu : ndarray, shape (3,)
        Mean direction (unit vector). Returns NaN if resultant length is zero.
    """
    # Compute resultant vector
    R = np.sum(directions, axis=0)  # (3,)
    R_length = np.linalg.norm(R)

    if R_length < 1e-6:
        # No preferred direction (uniform distribution)
        return np.full(3, np.nan)

    # Normalize to unit vector
    mu = R / R_length
    return mu


def _compute_vmf_concentration(directions: np.ndarray) -> float:
    """
    Estimate concentration parameter kappa for vMF distribution.

    Uses the approximation from Banerjee et al. (2005):
        kappa ≈ R_bar * (3 - R_bar^2) / (1 - R_bar^2)
    where R_bar is the mean resultant length.

    Parameters
    ----------
    directions : ndarray, shape (N, 3)
        Unit vectors on the sphere

    Returns
    -------
    kappa : float
        Estimated concentration parameter. Returns NaN if N < 3 or R_bar ≈ 1.
    """
    N = directions.shape[0]
    if N < 3:
        return np.nan

    # Compute resultant vector
    R = np.sum(directions, axis=0)
    R_length = np.linalg.norm(R)

    # Mean resultant length
    R_bar = R_length / N

    # Avoid division by zero (R_bar ≈ 1 means perfect concentration)
    if R_bar > 0.999:
        # Very high concentration, return large value
        return 100.0

    if R_bar < 0.001:
        # Very low concentration (near uniform)
        return 0.0

    # Banerjee et al. (2005) approximation for d=3
    numerator = R_bar * (3 - R_bar**2)
    denominator = 1 - R_bar**2

    kappa = numerator / denominator
    return max(0.0, kappa)  # Ensure non-negative


def compute_direction_statistics(
    positions_3d: np.ndarray,
    parents: np.ndarray,
    use_for_stats_mask: np.ndarray,
    joint_names: List[str],
    min_samples: int = 10,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute empirical directional statistics for each joint.

    For each joint (except root), computes the mean direction and concentration
    of the bone vector from parent to child. Uses only frames marked as valid
    in use_for_stats_mask (excludes outliers and interpolated data).

    Parameters
    ----------
    positions_3d : ndarray, shape (T, K, 3)
        3D joint positions over time
    parents : ndarray, shape (K,)
        Parent indices for skeleton tree
    use_for_stats_mask : ndarray, shape (T, K), bool
        Statistical validity mask (True = use for statistics)
    joint_names : list of str, length K
        Names of joints in order
    min_samples : int, optional
        Minimum number of valid samples required per joint. Default: 10.
        Joints with fewer samples get NaN statistics.

    Returns
    -------
    stats : dict
        Dictionary with joint names as keys. Each value is a dict with:
        - mu : ndarray, shape (3,) - Mean direction (unit vector)
        - kappa : float - Concentration parameter
        - n_samples : int - Number of valid samples used
        Joints with insufficient data or no parent have NaN values.

    Notes
    -----
    - Root joint (parent < 0) is skipped and returns NaN statistics
    - Bone directions are computed as (child - parent) / ||child - parent||
    - Only samples where both parent and child are valid (per use_for_stats_mask)
      and the resulting direction is non-NaN are included
    - Uses vMF mean direction (normalized resultant) and Banerjee approximation
      for concentration estimation
    """
    T, K, _ = positions_3d.shape

    if len(joint_names) != K:
        raise ValueError(f"joint_names length ({len(joint_names)}) must match K ({K})")

    stats = {}

    for k in range(K):
        joint_name = joint_names[k]
        p = parents[k]

        # Initialize with NaN
        stats[joint_name] = {
            "mu": np.full(3, np.nan),
            "kappa": np.nan,
            "n_samples": 0,
        }

        # Skip root (no parent)
        if p < 0:
            continue

        # Collect valid bone directions
        directions = []
        for t in range(T):
            # Check if both parent and child are valid for statistics
            if not (use_for_stats_mask[t, k] and use_for_stats_mask[t, p]):
                continue

            # Get positions
            pos_child = positions_3d[t, k, :]
            pos_parent = positions_3d[t, p, :]

            # Check for NaN
            if np.any(np.isnan(pos_child)) or np.any(np.isnan(pos_parent)):
                continue

            # Compute bone vector
            bone_vec = pos_child - pos_parent
            bone_length = np.linalg.norm(bone_vec)

            # Skip if length is too small (numerical issues)
            if bone_length < 1e-6:
                continue

            # Normalize to unit direction
            direction = bone_vec / bone_length
            directions.append(direction)

        # Convert to array
        if len(directions) < min_samples:
            # Insufficient data
            continue

        directions = np.array(directions)  # (N, 3)

        # Compute mean direction
        mu = _compute_vmf_mean_direction(directions)

        # Compute concentration
        kappa = _compute_vmf_concentration(directions)

        # Store results
        stats[joint_name] = {
            "mu": mu,
            "kappa": kappa,
            "n_samples": len(directions),
        }

    return stats
