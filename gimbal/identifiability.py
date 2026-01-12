"""Camera identifiability checking and optimization for 3D reconstruction.

This module provides a 3-tier system for ensuring camera configurations
support reliable 3D reconstruction via triangulation:

Tier 1: check_identifiability() - Validation with diagnostics
Tier 2: iteratively_adjust_cameras() - Optimization-based adjustment
Tier 3: auto_place_cameras() - Intelligent initial placement

The optimization objective (Tier 2) uses three normalized terms:
- J_θ: Ray-angle geometry (minimum pairwise angles)
- J_d: Camera distance shell (preferred working distance)
- J_F: Coverage fraction (percentage of well-constrained points)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import softplus


@dataclass
class IdentifiabilityConfig:
    """Configuration for identifiability checking and optimization.

    Attributes
    ----------
    min_cameras : int
        Minimum cameras that must see each point (default 2)
    min_angle_deg : float
        Minimum pairwise ray angle in degrees (default 10)
    target_fraction : float
        Target fraction of points meeting criteria (default 0.95)
    preferred_distance : float
        Preferred camera distance from root (default None = auto from skeleton)
    distance_std : float
        Standard deviation for distance penalty (default 10.0)
    angle_scale : float
        Scale parameter for angle penalty softplus (default 0.1)
    fraction_scale : float
        Scale parameter for fraction penalty softplus (default 0.05)
    max_iter : int
        Maximum optimization iterations (default 100)
    """

    min_cameras: int = 2
    min_angle_deg: float = 10.0
    target_fraction: float = 0.95
    preferred_distance: Optional[float] = None
    distance_std: float = 10.0
    angle_scale: float = 0.1
    fraction_scale: float = 0.05
    max_iter: int = 100


def check_identifiability(
    x_3d: np.ndarray,
    camera_positions: np.ndarray,
    config: Optional[IdentifiabilityConfig] = None,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """Check identifiability of 3D points under given camera configuration.

    Tier 1: Validation function that samples points and checks geometric criteria.

    Parameters
    ----------
    x_3d : np.ndarray, shape (T, K, 3)
        3D joint positions over time
    camera_positions : np.ndarray, shape (C, 3)
        Camera positions in world coordinates
    config : IdentifiabilityConfig, optional
        Configuration parameters
    num_samples : int
        Number of time-point-joint samples to check (default 100)

    Returns
    -------
    results : dict
        Dictionary with:
        - passed : bool (True if criteria met)
        - fraction_good : float (fraction of samples meeting criteria)
        - mean_min_angle : float (mean minimum pairwise angle in degrees)
        - std_min_angle : float (std of minimum pairwise angles)
        - diagnostics : dict (detailed per-sample info)
    """
    if config is None:
        config = IdentifiabilityConfig()

    T, K, _ = x_3d.shape
    C = camera_positions.shape[0]

    # Sample points uniformly
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(T * K, size=min(num_samples, T * K), replace=False)
    t_samples = sample_indices // K
    k_samples = sample_indices % K

    min_angles = []
    camera_counts = []
    passes = []

    for t, k in zip(t_samples, k_samples):
        point = x_3d[t, k]

        # Compute viewing rays
        rays = camera_positions - point[None, :]  # (C, 3)
        distances = np.linalg.norm(rays, axis=1, keepdims=True)
        rays_normalized = rays / (distances + 1e-10)  # (C, 3)

        # Count cameras that can see this point (all cameras for now, could add FOV check)
        num_cameras = C
        camera_counts.append(num_cameras)

        # Compute pairwise angles
        pairwise_angles = []
        for i in range(C):
            for j in range(i + 1, C):
                dot_product = np.clip(
                    np.dot(rays_normalized[i], rays_normalized[j]), -1.0, 1.0
                )
                angle_rad = np.arccos(dot_product)
                angle_deg = np.degrees(angle_rad)
                pairwise_angles.append(angle_deg)

        if pairwise_angles:
            min_angle = np.min(pairwise_angles)
        else:
            min_angle = 0.0

        min_angles.append(min_angle)

        # Check if this point passes criteria
        passes.append(
            num_cameras >= config.min_cameras and min_angle >= config.min_angle_deg
        )

    min_angles = np.array(min_angles)
    camera_counts = np.array(camera_counts)
    passes = np.array(passes)

    fraction_good = np.mean(passes)
    passed = fraction_good >= config.target_fraction

    return {
        "passed": passed,
        "fraction_good": float(fraction_good),
        "mean_min_angle": float(np.mean(min_angles)),
        "std_min_angle": float(np.std(min_angles)),
        "min_min_angle": float(np.min(min_angles)),
        "max_min_angle": float(np.max(min_angles)),
        "mean_cameras": float(np.mean(camera_counts)),
        "diagnostics": {
            "num_samples": len(passes),
            "num_passed": int(np.sum(passes)),
            "num_failed": int(np.sum(~passes)),
        },
    }


def _compute_identifiability_objective(
    camera_positions_flat: np.ndarray,
    x_3d_samples: np.ndarray,
    root_pos: np.ndarray,
    config: IdentifiabilityConfig,
) -> float:
    """Compute identifiability objective for optimization.

    Objective: J = J_θ + J_d + J_F

    Parameters
    ----------
    camera_positions_flat : np.ndarray, shape (C * 3,)
        Flattened camera positions for optimization
    x_3d_samples : np.ndarray, shape (P, 3)
        Sampled 3D points to evaluate
    root_pos : np.ndarray, shape (3,)
        Root position at t=0 for distance reference
    config : IdentifiabilityConfig
        Configuration parameters

    Returns
    -------
    objective : float
        Total objective value (sum of three terms)
    """
    C = len(camera_positions_flat) // 3
    camera_positions = camera_positions_flat.reshape(C, 3)
    P = x_3d_samples.shape[0]

    # === J_θ: Ray-angle geometry term ===
    angle_penalties = []
    point_scores = []

    for p in range(P):
        point = x_3d_samples[p]

        # Compute viewing rays
        rays = camera_positions - point[None, :]
        distances = np.linalg.norm(rays, axis=1, keepdims=True)
        rays_normalized = rays / (distances + 1e-10)

        # Compute all pairwise angles
        pairwise_angles = []
        for i in range(C):
            for j in range(i + 1, C):
                dot_product = np.clip(
                    np.dot(rays_normalized[i], rays_normalized[j]), -1.0, 1.0
                )
                angle_rad = np.arccos(dot_product)
                pairwise_angles.append(angle_rad)

        if not pairwise_angles:
            angle_penalties.append(1.0)
            point_scores.append(0.0)
            continue

        # Soft minimum over pairwise angles
        # Use negative log-sum-exp trick for numerical stability
        angles_array = np.array(pairwise_angles)
        beta = 10.0  # Temperature for softmin (higher = closer to true min)
        softmin_angle = -np.log(np.mean(np.exp(-beta * angles_array))) / beta

        # Penalty for angles below threshold
        theta_0 = np.radians(config.min_angle_deg)
        penalty = softplus((theta_0 - softmin_angle) / config.angle_scale)
        angle_penalties.append(penalty)

        # Point score: 1 if meets criteria, 0 otherwise (for J_F)
        min_angle = np.min(pairwise_angles)
        point_scores.append(1.0 if min_angle >= theta_0 else 0.0)

    J_theta = np.mean(angle_penalties)

    # === J_d: Camera distance shell term ===
    camera_distances = np.linalg.norm(camera_positions - root_pos[None, :], axis=1)

    # Use preferred distance if specified, otherwise use median of current distances
    d_0 = (
        config.preferred_distance
        if config.preferred_distance is not None
        else np.median(camera_distances)
    )

    distance_penalties = ((camera_distances - d_0) / config.distance_std) ** 2
    J_d = np.mean(distance_penalties)

    # === J_F: Coverage fraction term ===
    mean_score = np.mean(point_scores)
    J_F = softplus((config.target_fraction - mean_score) / config.fraction_scale)

    return J_theta + J_d + J_F


def iteratively_adjust_cameras(
    x_3d: np.ndarray,
    camera_positions_init: np.ndarray,
    config: Optional[IdentifiabilityConfig] = None,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """Iteratively adjust camera positions to improve identifiability.

    Tier 2: Optimization-based adjustment using smooth differentiable objective.

    Parameters
    ----------
    x_3d : np.ndarray, shape (T, K, 3)
        3D joint positions over time
    camera_positions_init : np.ndarray, shape (C, 3)
        Initial camera positions
    config : IdentifiabilityConfig, optional
        Configuration parameters
    num_samples : int
        Number of points to sample for optimization

    Returns
    -------
    results : dict
        Dictionary with:
        - camera_positions : np.ndarray, shape (C, 3) (optimized)
        - objective_initial : float
        - objective_final : float
        - converged : bool
        - diagnostics : dict (per-term objectives, iteration count, etc.)
    """
    if config is None:
        config = IdentifiabilityConfig()

    T, K, _ = x_3d.shape
    C = camera_positions_init.shape[0]

    # Sample points
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(T * K, size=min(num_samples, T * K), replace=False)
    t_samples = sample_indices // K
    k_samples = sample_indices % K
    x_3d_samples = x_3d[t_samples, k_samples, :]  # (P, 3)

    # Root position at t=0 for distance reference
    root_pos = x_3d[0, 0, :]

    # Initial objective
    camera_flat_init = camera_positions_init.flatten()
    obj_init = _compute_identifiability_objective(
        camera_flat_init, x_3d_samples, root_pos, config
    )

    # Optimize
    result = minimize(
        _compute_identifiability_objective,
        camera_flat_init,
        args=(x_3d_samples, root_pos, config),
        method="L-BFGS-B",
        options={"maxiter": config.max_iter, "disp": False},
    )

    camera_positions_opt = result.x.reshape(C, 3)
    obj_final = result.fun

    return {
        "camera_positions": camera_positions_opt,
        "objective_initial": float(obj_init),
        "objective_final": float(obj_final),
        "converged": result.success,
        "diagnostics": {
            "iterations": result.nit,
            "message": result.message,
            "improvement": float(obj_init - obj_final),
            "improvement_pct": float(100 * (obj_init - obj_final) / (obj_init + 1e-10)),
        },
    }


def auto_place_cameras(
    x_3d_t0: np.ndarray,
    num_cameras: int,
    config: Optional[IdentifiabilityConfig] = None,
) -> np.ndarray:
    """Automatically place cameras in intelligent initial configuration.

    Tier 3: Intelligent initial placement using two-ring geometry.

    Cameras are placed on two rings at heights h/2 and 2h, evenly distributed
    around a circle at radius 3w, where:
    - h = skeleton height (max_z - root_z at t=0)
    - w = skeleton radius (max horizontal distance from root at t=0)

    Parameters
    ----------
    x_3d_t0 : np.ndarray, shape (K, 3)
        Joint positions at t=0
    num_cameras : int
        Number of cameras to place
    config : IdentifiabilityConfig, optional
        Configuration parameters (not used for Tier 3, but kept for consistency)

    Returns
    -------
    camera_positions : np.ndarray, shape (C, 3)
        Camera positions
    """
    if config is None:
        config = IdentifiabilityConfig()

    root_pos = x_3d_t0[0, :]  # Root is first joint

    # Compute skeleton dimensions
    z_coords = x_3d_t0[:, 2]
    h = np.max(z_coords) - root_pos[2]  # Height

    xy_offsets = x_3d_t0[:, :2] - root_pos[:2][None, :]
    w = np.max(np.linalg.norm(xy_offsets, axis=1))  # Horizontal radius

    # Camera placement radius
    camera_radius = 3 * w

    # Two ring heights
    height_low = root_pos[2] + h / 2
    height_high = root_pos[2] + 2 * h

    # Distribute cameras between two rings
    # If odd number, put extra camera at high ring
    num_low = num_cameras // 2
    num_high = num_cameras - num_low

    camera_positions = []

    # Low ring cameras
    for i in range(num_low):
        angle = 2 * np.pi * i / num_low
        x = root_pos[0] + camera_radius * np.cos(angle)
        y = root_pos[1] + camera_radius * np.sin(angle)
        camera_positions.append([x, y, height_low])

    # High ring cameras
    # Offset by half angle to stagger with low ring
    angle_offset = np.pi / num_high if num_high > 0 else 0
    for i in range(num_high):
        angle = 2 * np.pi * i / num_high + angle_offset
        x = root_pos[0] + camera_radius * np.cos(angle)
        y = root_pos[1] + camera_radius * np.sin(angle)
        camera_positions.append([x, y, height_high])

    return np.array(camera_positions)
