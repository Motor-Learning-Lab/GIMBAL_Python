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
    """Generate 3D skeletal motion from state sequence (legacy simple noise model).

    **LEGACY:** This function uses simple directional noise and is kept for backward
    compatibility. For continuous smooth motion, use `generate_skeletal_motion_continuous()`.

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


def generate_skeletal_motion_continuous(
    skeleton: SkeletonConfig,
    true_states: np.ndarray,
    state_params: dict,
    root_params: dict,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate continuous 3D skeletal motion using second-order attractor dynamics.

    This implements a state-dependent second-order dynamical system:
        a_t = -omega_k^2 * (q_t - mu_k) - 2 * zeta_k * omega_k * v_t + noise
        v_{t+1} = v_t + a_t * dt
        q_{t+1} = q_t + v_{t+1} * dt

    Where q_t is a concatenation of raw bone direction vectors (not normalized).
    After integration, directions are normalized before FK.

    Parameters
    ----------
    skeleton : SkeletonConfig
        Skeleton configuration
    true_states : np.ndarray
        State sequence, shape (T,)
    state_params : dict
        Per-state motion parameters with keys:
        - 'mu' : np.ndarray, shape (S, num_bones, 3) - attractor positions
        - 'omega' : np.ndarray, shape (S,) or (S, num_bones) - natural frequencies
        - 'zeta' : np.ndarray, shape (S,) or (S, num_bones) - damping ratios
        - 'sigma_process' : np.ndarray, shape (S,) or (S, num_bones) - process noise
    root_params : dict
        Root motion parameters with keys:
        - 'mu' : np.ndarray, shape (S, 3) - attractor positions for root
        - 'omega' : float or np.ndarray, shape (S,) - natural frequency
        - 'zeta' : float or np.ndarray, shape (S,) - damping ratio
        - 'sigma_process' : float or np.ndarray, shape (S,) - process noise
        - 'init_pos' : np.ndarray, shape (3,) - initial root position (optional)
    dt : float
        Timestep in seconds
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    x_true : np.ndarray
        Joint positions, shape (T, K, 3)
    u_true : np.ndarray
        Joint directions (normalized), shape (T, K, 3)
    a_true : np.ndarray
        Acceleration vectors, shape (T, K, 3)
        For root: 3D position acceleration
        For joints: raw direction acceleration (before normalization)
    """
    T = len(true_states)
    K = len(skeleton.joint_names)
    num_bones = K - 1  # Exclude root from directional dynamics

    # Extract state-dependent parameters
    mu_dirs = state_params["mu"]  # (S, num_bones, 3)
    omega = state_params["omega"]  # (S,) or (S, num_bones)
    zeta = state_params["zeta"]  # (S,) or (S, num_bones)
    sigma_proc = state_params["sigma_process"]  # (S,) or (S, num_bones)

    # Root parameters
    root_mu = root_params["mu"]  # (S, 3)
    root_omega = np.atleast_1d(root_params["omega"])  # (S,) or scalar
    root_zeta = np.atleast_1d(root_params["zeta"])
    root_sigma = np.atleast_1d(root_params["sigma_process"])

    # Initialize
    x_true = np.zeros((T, K, 3))
    u_true = np.zeros((T, K, 3))
    a_true = np.zeros((T, K, 3))

    # Initial state (state 0 attractor)
    s0 = true_states[0]
    q_dirs = mu_dirs[s0].copy()  # (num_bones, 3) - raw directions
    v_dirs = np.zeros((num_bones, 3))  # Zero initial velocity

    # Root initial state
    x_true[0, 0] = root_params.get("init_pos", root_mu[s0])
    v_root = np.zeros(3)

    # Generate motion
    for t in range(T):
        s = true_states[t]

        # === Root dynamics (simple second-order system) ===
        # Broadcast parameters
        omega_r = root_omega[s] if root_omega.size > 1 else root_omega[0]
        zeta_r = root_zeta[s] if root_zeta.size > 1 else root_zeta[0]
        sigma_r = root_sigma[s] if root_sigma.size > 1 else root_sigma[0]

        # Compute root acceleration
        a_root = (
            -(omega_r**2) * (x_true[t, 0] - root_mu[s])
            - 2 * zeta_r * omega_r * v_root
            + rng.normal(0, sigma_r, 3)
        )
        a_true[t, 0] = a_root

        # Update root velocity and position
        v_root = v_root + a_root * dt
        if t + 1 < T:
            x_true[t + 1, 0] = x_true[t, 0] + v_root * dt

        # === Joint directional dynamics ===
        # Broadcast omega, zeta, sigma to per-bone if needed
        # Handle scalar, per-state scalar, or per-bone arrays

        def get_param_per_bone(param, param_name):
            """Extract per-bone parameter values for current state."""
            if np.isscalar(param):
                # Single scalar for all states and bones
                return np.full(num_bones, float(param))
            elif param.ndim == 1:
                # Per-state scalar: param[s]
                if param.size > s:
                    return np.full(num_bones, float(param[s]))
                else:
                    # Single value, use it
                    return np.full(num_bones, float(param[0]))
            elif param.ndim == 2:
                # Per-state per-bone: param[s, b]
                if param.shape[0] > s:
                    return param[s]
                else:
                    return param[0]
            else:
                raise ValueError(f"Unexpected shape for {param_name}: {param.shape}")

        omega_b = get_param_per_bone(omega, "omega")
        zeta_b = get_param_per_bone(zeta, "zeta")
        sigma_b = get_param_per_bone(sigma_proc, "sigma_proc")

        # Compute acceleration for each bone direction (vectorized)
        a_dirs = np.zeros((num_bones, 3))
        for b in range(num_bones):
            a_dirs[b] = (
                -omega_b[b] ** 2 * (q_dirs[b] - mu_dirs[s, b])
                - 2 * zeta_b[b] * omega_b[b] * v_dirs[b]
                + rng.normal(0, sigma_b[b], 3)
            )

        # Update velocity
        v_dirs = v_dirs + a_dirs * dt

        # Update position
        q_dirs = q_dirs + v_dirs * dt

        # Normalize directions and perform FK
        for k in range(1, K):  # Skip root
            bone_idx = k - 1
            u_normalized = q_dirs[bone_idx] / (np.linalg.norm(q_dirs[bone_idx]) + 1e-8)
            u_true[t, k] = u_normalized

            # Store acceleration (for joints, this is direction acceleration)
            a_true[t, k] = a_dirs[bone_idx]

            # Forward kinematics
            parent = skeleton.parents[k]
            bone_length = (
                skeleton.bone_lengths[k] if skeleton.bone_lengths is not None else 10.0
            )
            x_true[t, k] = x_true[t, parent] + bone_length * u_normalized

    return x_true, u_true, a_true


def generate_camera_matrices(
    C: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic camera projection matrices with perspective projection.

    Uses full perspective camera model: P = K[R|t] where:
    - K: intrinsic matrix (focal length, principal point)
    - R: rotation matrix (camera orientation via look-at)
    - t: translation vector (t = -R @ camera_pos)

    Cameras are positioned around the scene with proper orientations:
    - Camera 0: Front view (from +X axis, looking at scene center)
    - Camera 1: Side view (from +Y axis, looking at scene center)
    - Camera 2: Overhead view (from above, looking down at scene center)

    Scene center is [0, 0, 100], skeleton has ~42 unit extent.

    Parameters
    ----------
    C : int
        Number of cameras
    rng : np.random.Generator
        Random number generator (for potential future randomization)

    Returns
    -------
    camera_proj : np.ndarray
        Camera projection matrices P = K[R|t], shape (C, 3, 4)
        Projection is: [u', v', w'] = P @ [x, y, z, 1]
                       u = u'/w', v = v'/w' (perspective division)
    """
    from .camera_utils import build_projection_matrix

    camera_proj = np.zeros((C, 3, 4))

    # Scene center: where skeleton lives
    scene_center = np.array([0.0, 0.0, 100.0])

    # World up direction for look-at calculations
    up_world = np.array([0.0, 0.0, 1.0])

    # Focal length: with skeleton at distance ~80 units, f=10 gives ~300 pixel extent
    focal_length = 10.0

    # Define camera positions
    camera_positions = []

    if C >= 1:
        # Camera 0: Front view (from +X, looking at center)
        camera_positions.append(scene_center + np.array([80, 0, 0]))

    if C >= 2:
        # Camera 1: Side view (from +Y, looking at center)
        camera_positions.append(scene_center + np.array([0, 80, 0]))

    if C >= 3:
        # Camera 2: Overhead view (from above, looking down)
        camera_positions.append(scene_center + np.array([0, 0, 80]))

    # Additional cameras in a circle at varying heights
    for c in range(3, C):
        angle = 2 * np.pi * (c - 3) / max(C - 3, 1) + np.pi / 4
        radius = 80
        # Vary height: alternate between slightly above and below center
        height_offset = 20 * ((c - 3) % 2 - 0.5)
        camera_positions.append(
            scene_center
            + np.array([radius * np.cos(angle), radius * np.sin(angle), height_offset])
        )

    # Build projection matrices with proper orientation
    for c in range(C):
        camera_pos = camera_positions[c]

        # Build P = K[R|t] with camera looking at scene_center
        camera_proj[c] = build_projection_matrix(
            camera_pos=camera_pos,
            target_pos=scene_center,
            focal_length=focal_length,
            up_world=up_world,
            principal_point=(0.0, 0.0),
        )

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

                # Project: [u', v', w'] = P @ [x, y, z, 1]
                y_proj = camera_proj[c] @ x_h

                # Perspective division: u = u'/w', v = v'/w'
                # This matches the PyMC model and DLT triangulation
                w = y_proj[2]
                if np.abs(w) > 1e-6:
                    y_2d = y_proj[:2] / w
                else:
                    # Point behind or at camera - mark as invalid
                    y_2d = np.array([np.nan, np.nan])

                # Add noise
                if not np.any(np.isnan(y_2d)):
                    y_observed[c, t, k] = y_2d + rng.normal(0, obs_noise_std, 2)
                else:
                    y_observed[c, t, k] = y_2d

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
