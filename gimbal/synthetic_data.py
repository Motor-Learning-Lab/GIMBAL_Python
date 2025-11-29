"""Synthetic data generation utilities for GIMBAL testing and demos.

This module provides functions to generate synthetic skeletal motion data
with known ground truth for testing and demonstration purposes.
"""

import numpy as np
from typing import NamedTuple, Optional
from .skeleton_config import SkeletonConfig


class SyntheticDataConfig(NamedTuple):
    """Configuration for synthetic data generation.

    Attributes
    ----------
    T : int
        Number of timesteps
    C : int
        Number of cameras
    S : int
        Number of hidden states
    kappa : float
        Concentration parameter for directional noise (higher = less noise)
    obs_noise_std : float
        Standard deviation of 2D observation noise in pixels
    occlusion_rate : float
        Fraction of observations to mark as occluded (0 to 1)
    root_noise_std : float
        Standard deviation of root position random walk
    random_seed : int | None
        Random seed for reproducibility (None = no seeding)
    """

    T: int = 60
    C: int = 3
    S: int = 3
    kappa: float = 8.0
    obs_noise_std: float = 5.0
    occlusion_rate: float = 0.05
    root_noise_std: float = 1.0
    random_seed: Optional[int] = 42


class SyntheticMotionData(NamedTuple):
    """Generated synthetic motion data.

    Attributes
    ----------
    x_true : np.ndarray
        True 3D joint positions, shape (T, K, 3)
    u_true : np.ndarray
        True joint direction vectors (unit), shape (T, K, 3)
    true_states : np.ndarray
        True hidden state sequence, shape (T,)
    canonical_mu : np.ndarray
        Canonical direction for each state and joint, shape (S, K, 3)
    y_observed : np.ndarray
        Observed 2D projections with noise and occlusions, shape (C, T, K, 2)
        Occluded observations are NaN
    camera_proj : np.ndarray
        Camera projection matrices, shape (C, 3, 4)
    trans_probs : np.ndarray
        State transition probability matrix, shape (S, S)
    skeleton : SkeletonConfig
        Skeleton configuration used for generation
    config : SyntheticDataConfig
        Configuration parameters used for generation
    """

    x_true: np.ndarray
    u_true: np.ndarray
    true_states: np.ndarray
    canonical_mu: np.ndarray
    y_observed: np.ndarray
    camera_proj: np.ndarray
    trans_probs: np.ndarray
    skeleton: SkeletonConfig
    config: SyntheticDataConfig


def generate_canonical_directions(
    S: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate canonical directions for each state.

    Creates distinct pose states suitable for HMM demonstration:
    - State 0: Upright (mostly +z direction)
    - State 1: Forward lean (+x and +z components)
    - State 2: Sideways lean (+y and +z components)

    For S > 3, generates random diverse directions.

    Parameters
    ----------
    S : int
        Number of states
    K : int
        Number of joints
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    canonical_mu : np.ndarray
        Canonical directions, shape (S, K, 3)
        canonical_mu[s, k] is the mean direction for joint k in state s
    """
    canonical_mu = np.zeros((S, K, 3))

    if S >= 1:
        # State 0: Upright (mostly +z)
        canonical_mu[0, 1:, 2] = 1.0

    if S >= 2:
        # State 1: Forward lean
        canonical_mu[1, 1:, 0] = 0.6  # Forward component
        canonical_mu[1, 1:, 2] = 0.8  # Still mostly up

    if S >= 3:
        # State 2: Sideways lean
        canonical_mu[2, 1:, 1] = 0.7  # Sideways component
        canonical_mu[2, 1:, 2] = 0.7  # Still mostly up

    # For additional states, generate random directions
    for s in range(3, S):
        for k in range(1, K):
            # Random direction with upward bias
            direction = rng.normal([0, 0, 0.5], [0.3, 0.3, 0.2], 3)
            canonical_mu[s, k] = direction

    # Normalize all directions to unit vectors
    for s in range(S):
        for k in range(1, K):  # Skip root
            norm = np.linalg.norm(canonical_mu[s, k])
            if norm > 0:
                canonical_mu[s, k] /= norm

    return canonical_mu


def generate_state_sequence(
    T: int,
    S: int,
    trans_probs: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a hidden state sequence with persistence.

    Parameters
    ----------
    T : int
        Number of timesteps
    S : int
        Number of states
    trans_probs : np.ndarray, optional
        State transition probability matrix, shape (S, S)
        If None, uses default high-persistence transitions
    rng : np.random.Generator, optional
        Random number generator (default: creates new one)

    Returns
    -------
    states : np.ndarray
        State sequence, shape (T,), values in [0, S-1]
    """
    if rng is None:
        rng = np.random.default_rng()

    if trans_probs is None:
        # Default: high persistence (80-85% stay probability)
        trans_probs = np.ones((S, S)) * (0.15 / (S - 1) if S > 1 else 0)
        np.fill_diagonal(trans_probs, 0.85)

    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(S)

    for t in range(1, T):
        states[t] = rng.choice(S, p=trans_probs[states[t - 1]])

    return states


def generate_skeletal_motion(
    skeleton: SkeletonConfig,
    true_states: np.ndarray,
    canonical_mu: np.ndarray,
    kappa: float,
    root_noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 3D skeletal motion from state sequence.

    Parameters
    ----------
    skeleton : SkeletonConfig
        Skeleton configuration
    true_states : np.ndarray
        State sequence, shape (T,)
    canonical_mu : np.ndarray
        Canonical directions, shape (S, K, 3)
    kappa : float
        Concentration parameter for directional noise
    root_noise_std : float
        Standard deviation of root position random walk
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    x_true : np.ndarray
        Joint positions, shape (T, K, 3)
    u_true : np.ndarray
        Joint directions, shape (T, K, 3)
    """
    T = len(true_states)
    K = len(skeleton.joint_names)

    x_true = np.zeros((T, K, 3))
    u_true = np.zeros((T, K, 3))

    for t in range(T):
        s = true_states[t]

        # Root position (random walk)
        if t == 0:
            x_true[t, 0] = [0, 0, 100]  # Start at height 100
        else:
            x_true[t, 0] = x_true[t - 1, 0] + rng.normal(0, root_noise_std, 3)

        # Generate directions with noise around canonical direction
        for k in range(1, K):
            # Add noise to canonical direction
            u_noisy = canonical_mu[s, k] + rng.normal(0, 1.0 / kappa, 3)
            u_noisy /= np.linalg.norm(u_noisy) + 1e-8
            u_true[t, k] = u_noisy

            # Compute position from parent
            parent = skeleton.parents[k]
            bone_length = (
                skeleton.bone_lengths[k] if skeleton.bone_lengths is not None else 10.0
            )
            x_true[t, k] = x_true[t, parent] + bone_length * u_true[t, k]

    return x_true, u_true


def generate_camera_matrices(
    C: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic camera projection matrices.

    Cameras are positioned in a circle around the scene, looking at the origin.
    Uses simplified projection for demonstration purposes.

    Parameters
    ----------
    C : int
        Number of cameras
    rng : np.random.Generator
        Random number generator (for potential future randomization)

    Returns
    -------
    camera_proj : np.ndarray
        Camera projection matrices, shape (C, 3, 4)
    """
    camera_proj = np.zeros((C, 3, 4))

    for c in range(C):
        # Camera positioned in a circle around the scene
        angle = 2 * np.pi * c / C
        camera_pos = np.array([150 * np.cos(angle), 150 * np.sin(angle), 100])

        # Simple projection matrix (orthographic-like for simplicity)
        # In practice, use proper camera calibration
        focal_length = 500
        A = np.eye(3) * focal_length
        b = -A @ camera_pos

        camera_proj[c] = np.column_stack([A, b])

    return camera_proj


def generate_observations(
    x_true: np.ndarray,
    camera_proj: np.ndarray,
    obs_noise_std: float,
    occlusion_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate noisy 2D observations with occlusions.

    Parameters
    ----------
    x_true : np.ndarray
        True 3D positions, shape (T, K, 3)
    camera_proj : np.ndarray
        Camera projection matrices, shape (C, 3, 4)
    obs_noise_std : float
        Standard deviation of observation noise in pixels
    occlusion_rate : float
        Fraction of observations to mark as occluded
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    y_observed : np.ndarray
        Observed 2D projections, shape (C, T, K, 2)
        Occluded observations are NaN
    """
    T, K, _ = x_true.shape
    C = camera_proj.shape[0]

    y_observed = np.zeros((C, T, K, 2))

    for c in range(C):
        for t in range(T):
            for k in range(K):
                # Homogeneous coordinates
                x_h = np.append(x_true[t, k], 1)

                # Project
                y_proj = camera_proj[c] @ x_h

                # Perspective division (for orthographic, this is identity)
                if y_proj[2] != 0:
                    y_2d = y_proj[:2] / y_proj[2]
                else:
                    y_2d = y_proj[:2]

                # Add noise
                y_observed[c, t, k] = y_2d + rng.normal(0, obs_noise_std, 2)

    # Add random occlusions (NaN values)
    n_occlusions = int(occlusion_rate * C * T * K)
    for _ in range(n_occlusions):
        c_occ = rng.integers(0, C)
        t_occ = rng.integers(0, T)
        k_occ = rng.integers(0, K)
        y_observed[c_occ, t_occ, k_occ] = np.nan

    return y_observed


def generate_demo_sequence(
    skeleton: SkeletonConfig,
    config: Optional[SyntheticDataConfig] = None,
) -> SyntheticMotionData:
    """Generate complete synthetic motion sequence for GIMBAL demonstration.

    This is the main entry point for generating synthetic data. It creates:
    - A state sequence with persistence
    - 3D skeletal motion with state-dependent poses
    - Multi-camera 2D observations with noise and occlusions

    Parameters
    ----------
    skeleton : SkeletonConfig
        Skeleton configuration to use
    config : SyntheticDataConfig, optional
        Generation parameters (uses defaults if None)

    Returns
    -------
    data : SyntheticMotionData
        Complete synthetic dataset with ground truth and observations

    Examples
    --------
    >>> from gimbal.skeleton_config import DEMO_V0_1_SKELETON
    >>> from gimbal.synthetic_data import generate_demo_sequence
    >>> data = generate_demo_sequence(DEMO_V0_1_SKELETON)
    >>> print(f"Generated {data.config.T} timesteps with {data.config.S} states")
    """
    if config is None:
        config = SyntheticDataConfig()

    # Initialize RNG
    rng = np.random.default_rng(config.random_seed)

    K = len(skeleton.joint_names)

    # Generate canonical directions for each state
    canonical_mu = generate_canonical_directions(config.S, K, rng)

    # Generate state transition probabilities
    trans_probs = np.ones((config.S, config.S)) * (
        0.15 / (config.S - 1) if config.S > 1 else 0
    )
    np.fill_diagonal(trans_probs, 0.85)

    # Generate state sequence
    true_states = generate_state_sequence(config.T, config.S, trans_probs, rng)

    # Generate 3D skeletal motion
    x_true, u_true = generate_skeletal_motion(
        skeleton, true_states, canonical_mu, config.kappa, config.root_noise_std, rng
    )

    # Generate camera matrices
    camera_proj = generate_camera_matrices(config.C, rng)

    # Generate 2D observations
    y_observed = generate_observations(
        x_true, camera_proj, config.obs_noise_std, config.occlusion_rate, rng
    )

    return SyntheticMotionData(
        x_true=x_true,
        u_true=u_true,
        true_states=true_states,
        canonical_mu=canonical_mu,
        y_observed=y_observed,
        camera_proj=camera_proj,
        trans_probs=trans_probs,
        skeleton=skeleton,
        config=config,
    )
