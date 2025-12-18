"""Config-driven synthetic dataset generation for v0.2.1.

This module provides JSON config loading and dataset generation using the
continuous second-order attractor motion model.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from gimbal.skeleton_config import SkeletonConfig
from gimbal.synthetic_data import generate_skeletal_motion_continuous
from gimbal.camera_utils import build_projection_matrix


@dataclass
class GeneratedDataset:
    """Container for generated synthetic dataset."""

    # Ground truth
    x_true: np.ndarray  # (T, K, 3) joint positions
    u_true: np.ndarray  # (T, K, 3) normalized joint directions
    a_true: np.ndarray  # (T, K, 3) accelerations
    z_true: np.ndarray  # (T,) hidden state sequence

    # Observations
    y_2d: np.ndarray  # (C, T, K, 2) 2D observations with noise/outliers/missingness

    # Metadata
    skeleton: SkeletonConfig
    camera_proj: np.ndarray  # (C, 3, 4) camera projection matrices
    camera_metadata: list  # List of dicts with camera info
    config: Dict[str, Any]  # Original config dict
    config_hash: str  # SHA256 hash of config


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate JSON config file.

    Parameters
    ----------
    config_path : Path
        Path to JSON config file

    Returns
    -------
    config : dict
        Loaded configuration
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove comment keys (start with _comment)
    def remove_comments(d):
        if isinstance(d, dict):
            return {
                k: remove_comments(v)
                for k, v in d.items()
                if not k.startswith("_comment")
            }
        elif isinstance(d, list):
            return [remove_comments(item) for item in d]
        else:
            return d

    config = remove_comments(config)

    # Basic validation
    required_keys = ["meta", "dataset_spec", "output"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    return config


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute stable SHA256 hash of config.

    Parameters
    ----------
    config : dict
        Configuration dict

    Returns
    -------
    hash_str : str
        Hex digest of SHA256 hash
    """
    # Sort keys for stable hash
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def calibrate_sigma_process(
    mu: np.ndarray,
    omega: float,
    zeta: float,
    sigma_pose_target: float,
    dt: float,
    T_calibration: int,
    rng: np.random.Generator,
) -> float:
    """Calibrate sigma_process to achieve target within-state dispersion.

    Runs a short simulation and adjusts process noise to match target pose variance.

    Parameters
    ----------
    mu : np.ndarray
        Attractor position, shape (num_bones, 3) or (3,) for root
    omega : float
        Natural frequency
    zeta : float
        Damping ratio
    sigma_pose_target : float
        Target RMS deviation from attractor
    dt : float
        Timestep
    T_calibration : int
        Number of timesteps for calibration run
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    sigma_process : float
        Calibrated process noise standard deviation
    """
    # Start with provisional sigma_process
    sigma_proc = sigma_pose_target * omega / 2.0  # Heuristic starting point

    # Run short simulation
    q = mu.copy()
    v = np.zeros_like(mu)

    trajectory = []
    for _ in range(T_calibration):
        a = (
            -(omega**2) * (q - mu)
            - 2 * zeta * omega * v
            + rng.normal(0, sigma_proc, mu.shape)
        )
        v = v + a * dt
        q = q + v * dt
        trajectory.append(q.copy())

    trajectory = np.array(trajectory)  # (T, ...)

    # Measure realized dispersion
    deviations = trajectory - mu
    rms_deviation = np.sqrt(np.mean(deviations**2))

    # Scale sigma_process
    if rms_deviation > 1e-6:
        scale_factor = sigma_pose_target / rms_deviation
        sigma_proc *= scale_factor

    return sigma_proc


def generate_from_config(
    config: Dict[str, Any], calibrate_noise: bool = True
) -> GeneratedDataset:
    """Generate synthetic dataset from JSON config.

    Parameters
    ----------
    config : dict
        Configuration loaded from JSON
    calibrate_noise : bool
        If True, calibrate sigma_process to match sigma_pose targets

    Returns
    -------
    dataset : GeneratedDataset
        Generated dataset with ground truth and observations
    """
    meta = config["meta"]
    spec = config["dataset_spec"]

    # Initialize RNG
    seed = meta.get("seed")
    rng = np.random.default_rng(seed)

    # Extract parameters
    T = meta["T"]
    dt = meta["dt"]

    # === Build skeleton ===
    skel_spec = spec["skeleton"]
    skeleton = SkeletonConfig(
        joint_names=skel_spec["joint_names"],
        parents=np.array(skel_spec["parents"]),
        bone_lengths=np.array(skel_spec["lengths"]),
        up_axis=np.array([0, 0, 1]),  # Default +Z axis
    )
    K = len(skeleton.joint_names)
    num_bones = K - 1

    # === Generate state sequence ===
    states_spec = spec["states"]
    S = states_spec["num_states"]
    trans_matrix = np.array(states_spec["transition_matrix"])

    z_true = np.zeros(T, dtype=int)
    z_true[0] = 0  # Start in state 0
    for t in range(1, T):
        z_true[t] = rng.choice(S, p=trans_matrix[z_true[t - 1]])

    # === Prepare motion parameters ===
    motion_spec = spec["motion"]
    per_state = motion_spec["per_state_params"]

    # Collect mu for all states
    mu_dirs_list = []
    for s in range(S):
        state_key = str(s)
        mu_raw = np.array(per_state[state_key]["mu"])  # (num_bones, 3)
        # Normalize to unit vectors
        mu_normalized = mu_raw / (np.linalg.norm(mu_raw, axis=1, keepdims=True) + 1e-8)
        mu_dirs_list.append(mu_normalized)

    mu_dirs = np.array(mu_dirs_list)  # (S, num_bones, 3)

    # Extract omega, zeta, sigma_pose (assume scalar for now)
    omega_vals = np.array([per_state[str(s)]["omega"] for s in range(S)])
    zeta_vals = np.array([per_state[str(s)]["zeta"] for s in range(S)])
    sigma_pose_vals = np.array([per_state[str(s)]["sigma_pose"] for s in range(S)])

    # Calibrate sigma_process if requested
    if calibrate_noise:
        sigma_process_vals = np.zeros(S)
        for s in range(S):
            sigma_process_vals[s] = calibrate_sigma_process(
                mu=mu_dirs[s, 0],  # Use first bone as representative
                omega=omega_vals[s],
                zeta=zeta_vals[s],
                sigma_pose_target=sigma_pose_vals[s],
                dt=dt,
                T_calibration=200,
                rng=rng,
            )
    else:
        # Use sigma_pose directly (less accurate)
        sigma_process_vals = sigma_pose_vals * omega_vals / 2.0

    state_params = {
        "mu": mu_dirs,
        "omega": omega_vals,
        "zeta": zeta_vals,
        "sigma_process": sigma_process_vals,
    }

    # Root parameters
    root_spec = motion_spec["root_params"]
    root_mu_list = []
    for s in range(S):
        root_mu_list.append(np.array(root_spec["per_state"][str(s)]["mu"]))

    root_params = {
        "mu": np.array(root_mu_list),  # (S, 3)
        "omega": root_spec["omega"],
        "zeta": root_spec["zeta"],
        "sigma_process": root_spec.get("sigma_pose", 5.0) * root_spec["omega"] / 2.0,
        "init_pos": np.array(root_spec.get("init_pos", root_mu_list[0])),
    }

    # === Generate motion ===
    x_true, u_true, a_true = generate_skeletal_motion_continuous(
        skeleton=skeleton,
        true_states=z_true,
        state_params=state_params,
        root_params=root_params,
        dt=dt,
        rng=rng,
    )

    # === Build cameras ===
    cam_specs = spec["cameras"]["cameras"]
    C = len(cam_specs)
    camera_proj = np.zeros((C, 3, 4))
    camera_metadata = []

    for c, cam_spec in enumerate(cam_specs):
        K_mat = np.array(cam_spec["K"])
        R_mat = np.array(cam_spec["R"])
        t_vec = np.array(cam_spec["t"])

        # Build P = K[R|t]
        Rt = np.hstack([R_mat, t_vec.reshape(3, 1)])
        camera_proj[c] = K_mat @ Rt

        camera_metadata.append(
            {
                "name": cam_spec["name"],
                "K": K_mat,
                "R": R_mat,
                "t": t_vec,
                "image_size": cam_spec.get("image_size", [1280, 720]),
            }
        )

    # === Generate 2D observations ===
    obs_spec = spec["observation"]
    noise_px = obs_spec["noise_px"]

    y_2d = np.zeros((C, T, K, 2))

    # Clean projection first
    for c in range(C):
        for t in range(T):
            for k in range(K):
                x_h = np.append(x_true[t, k], 1)
                y_proj = camera_proj[c] @ x_h
                w = y_proj[2]
                if np.abs(w) > 1e-6:
                    y_2d[c, t, k] = y_proj[:2] / w
                else:
                    y_2d[c, t, k] = np.nan

    # Add noise
    if noise_px > 0:
        noise = rng.normal(0, noise_px, (C, T, K, 2))
        y_2d = np.where(np.isnan(y_2d), y_2d, y_2d + noise)

    # Add outliers
    outlier_spec = obs_spec.get("outliers", {})
    if outlier_spec.get("enabled", False):
        frac = outlier_spec["fraction"]
        scale = outlier_spec["noise_scale"]
        n_outliers = int(frac * C * T * K)

        for _ in range(n_outliers):
            c_out = rng.integers(0, C)
            t_out = rng.integers(0, T)
            k_out = rng.integers(0, K)
            if not np.any(np.isnan(y_2d[c_out, t_out, k_out])):
                y_2d[c_out, t_out, k_out] += rng.normal(0, scale, 2)

    # Add missingness
    miss_spec = obs_spec.get("missingness", {})
    if miss_spec.get("enabled", False):
        rate = miss_spec["rate"]
        n_missing = int(rate * C * T * K)

        for _ in range(n_missing):
            c_miss = rng.integers(0, C)
            t_miss = rng.integers(0, T)
            k_miss = rng.integers(0, K)
            y_2d[c_miss, t_miss, k_miss] = np.nan

    # Compute config hash
    config_hash = compute_config_hash(config)

    return GeneratedDataset(
        x_true=x_true,
        u_true=u_true,
        a_true=a_true,
        z_true=z_true,
        y_2d=y_2d,
        skeleton=skeleton,
        camera_proj=camera_proj,
        camera_metadata=camera_metadata,
        config=config,
        config_hash=config_hash,
    )


def save_dataset(dataset: GeneratedDataset, output_dir: Path) -> None:
    """Save generated dataset to .npz file.

    Parameters
    ----------
    dataset : GeneratedDataset
        Dataset to save
    output_dir : Path
        Output directory (will create if needed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to .npz
    npz_path = output_dir / "dataset.npz"
    np.savez_compressed(
        npz_path,
        x_true=dataset.x_true,
        u_true=dataset.u_true,
        a_true=dataset.a_true,
        z_true=dataset.z_true,
        y_2d=dataset.y_2d,
        camera_proj=dataset.camera_proj,
        # Serialize config as JSON string
        config_json=json.dumps(dataset.config, indent=2),
        config_hash=dataset.config_hash,
        # Skeleton metadata
        joint_names=np.array(dataset.skeleton.joint_names, dtype=object),
        parents=dataset.skeleton.parents,
        bone_lengths=dataset.skeleton.bone_lengths,
    )

    print(f"Saved dataset to {npz_path}")


def load_dataset(dataset_path: Path) -> GeneratedDataset:
    """Load dataset from .npz file.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset.npz file

    Returns
    -------
    dataset : GeneratedDataset
        Loaded dataset
    """
    data = np.load(dataset_path, allow_pickle=True)

    config = json.loads(str(data["config_json"]))

    skeleton = SkeletonConfig(
        joint_names=list(data["joint_names"]),
        parents=data["parents"],
        bone_lengths=data["bone_lengths"],
        up_axis=np.array([0, 0, 1]),
    )

    # Camera metadata needs to be reconstructed
    # For now, return minimal version
    camera_metadata = []

    return GeneratedDataset(
        x_true=data["x_true"],
        u_true=data["u_true"],
        a_true=data["a_true"],
        z_true=data["z_true"],
        y_2d=data["y_2d"],
        skeleton=skeleton,
        camera_proj=data["camera_proj"],
        camera_metadata=camera_metadata,
        config=config,
        config_hash=str(data["config_hash"]),
    )
