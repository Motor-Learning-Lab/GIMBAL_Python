"""
Ground Truth Parameter Extraction

Extract complete ground truth parameters from synthetic data to match
all 42 parameters in the PyMC model.

This ensures we can test the model at its own generating parameters.
"""

import numpy as np
import pymc as pm
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence


def extract_complete_ground_truth(
    synth_data, skeleton, S: int
) -> Dict[str, np.ndarray]:
    """
    Extract ALL ground truth parameters needed for the PyMC model.

    The model has these parameters (42 total for S=2, K=6):
    - eta2_root: scalar (root position variance)
    - rho: (K-1,) array (bone correlation)
    - sigma2: (K-1,) array (bone variance)
    - x_root: (T, 3) (root positions)
    - raw_u_k: (T, 3) for k=1..K-1 (bone direction vectors)
    - length_k: (T,) for k=1..K-1 (bone lengths)
    - obs_sigma: scalar (observation noise)
    - inlier_prob: scalar (inlier probability)
    - HMM parameters: mu, kappa, init_logits, trans_logits

    Parameters
    ----------
    synth_data : SyntheticMotionData
        Output from generate_demo_sequence()
    skeleton : SkeletonConfig
        Skeleton configuration
    S : int
        Number of HMM states

    Returns
    -------
    ground_truth : dict
        Dictionary with ALL 42 parameters in constrained space
    """
    K = len(skeleton.joint_names)
    T = synth_data.config.T

    gt = {}

    print(f"Extracting ground truth for K={K} joints, T={T} timesteps, S={S} states")
    print()

    # ========================================================================
    # SKELETON STRUCTURE PARAMETERS
    # ========================================================================

    # Root positions from joint 0
    gt["x_root"] = synth_data.x_true[:, 0, :]  # (T, 3)
    print(f"[OK] x_root: shape {gt['x_root'].shape}")

    # Compute bone vectors and lengths from joint positions
    for k in range(1, K):
        parent_idx = skeleton.parents[k]

        # Bone vector (direction × length)
        bone_vec = synth_data.x_true[:, k, :] - synth_data.x_true[:, parent_idx, :]
        gt[f"raw_u_{k}"] = bone_vec

        # Bone length (from skeleton config - these are constants)
        bone_length = skeleton.bone_lengths[k - 1]  # bone_lengths is (K-1,)
        gt[f"length_{k}"] = np.full(T, bone_length)  # Repeat for all timesteps

        print(
            f"[OK] raw_u_{k}: shape {gt[f'raw_u_{k}'].shape}, "
            f"mean length {np.mean(np.linalg.norm(bone_vec, axis=1)):.3f}"
        )
        print(
            f"[OK] length_{k}: shape {gt[f'length_{k}'].shape}, "
            f"constant value {bone_length:.3f}"
        )

    # Compute hyperparameters from bone vectors
    # eta2_root: variance of root position changes
    root_diff = np.diff(gt["x_root"], axis=0)
    gt["eta2_root"] = np.mean(np.var(root_diff, axis=0))
    print(f"[OK] eta2_root: {gt['eta2_root']:.4f} (computed from root variance)")

    # rho: autocorrelation for each bone
    # sigma2: innovation variance for each bone
    # These are harder to estimate from data, use reasonable defaults
    gt["rho"] = np.full(K - 1, 0.95)  # High persistence
    gt["sigma2"] = np.full(K - 1, 0.1)  # Small innovations
    print(f"[OK] rho: shape {gt['rho'].shape}, default value 0.95")
    print(f"[OK] sigma2: shape {gt['sigma2'].shape}, default value 0.1")

    # ========================================================================
    # OBSERVATION MODEL PARAMETERS
    # ========================================================================

    # Observation noise (from config)
    gt["obs_sigma"] = synth_data.config.obs_noise_std
    print(f"[OK] obs_sigma: {gt['obs_sigma']:.4f}")

    # Inlier probability (from config)
    gt["inlier_prob"] = 1.0 - synth_data.config.occlusion_rate
    print(f"[OK] inlier_prob: {gt['inlier_prob']:.4f}")

    # ========================================================================
    # HMM PARAMETERS (if available)
    # ========================================================================

    if hasattr(synth_data, "canonical_mu") and synth_data.canonical_mu is not None:
        # Canonical directions (S, K, 3)
        canonical_mu = synth_data.canonical_mu
        print(f"[OK] Found canonical directions: shape {canonical_mu.shape}")

        for s in range(S):
            for k in range(K):
                gt[f"dir_hmm_mu_raw_s{s}_k{k}"] = canonical_mu[s, k, :]

        # Kappa (concentration) - from config
        kappa_value = synth_data.config.kappa
        for s in range(S):
            for k in range(K):
                gt[f"dir_hmm_kappa_s{s}_k{k}"] = kappa_value
        print(f"[OK] dir_hmm_kappa: constant value {kappa_value:.4f}")

        # Initial state distribution (uniform for simplicity)
        gt["dir_hmm_init_dist"] = np.ones(S) / S
        print(f"[OK] dir_hmm_init_dist: uniform {gt['dir_hmm_init_dist']}")

        # Transition matrix (from data)
        if hasattr(synth_data, "trans_probs") and synth_data.trans_probs is not None:
            gt["dir_hmm_trans_mat"] = synth_data.trans_probs
            print(f"[OK] dir_hmm_trans_mat: shape {gt['dir_hmm_trans_mat'].shape}")
        else:
            # Default: high persistence
            trans_mat = np.ones((S, S)) * (0.15 / (S - 1) if S > 1 else 0)
            np.fill_diagonal(trans_mat, 0.85)
            gt["dir_hmm_trans_mat"] = trans_mat
            print(f"[OK] dir_hmm_trans_mat: default high persistence")
    else:
        print("[WARNING] No HMM parameters in synthetic data")

    print()
    print(f"Total parameters extracted: {len(gt)}")

    return gt


def transform_to_unconstrained_space(
    gt_constrained: Dict[str, np.ndarray], model: pm.Model
) -> Dict[str, np.ndarray]:
    """
    Transform ground truth from constrained to unconstrained space.

    PyMC samples in unconstrained space with transformations:
    - Positive reals: log transform (e.g., obs_sigma -> obs_sigma_log__)
    - Unit interval: logit transform (e.g., inlier_prob -> inlier_prob_logodds__)
    - Simplexes: stick-breaking or log-odds

    Parameters
    ----------
    gt_constrained : dict
        Ground truth in constrained space
    model : pm.Model
        PyMC model (to get transformation info)

    Returns
    -------
    gt_unconstrained : dict
        Ground truth in unconstrained space matching model.free_RVs
    """
    print("=" * 80)
    print("Transforming to Unconstrained Space")
    print("=" * 80)
    print()

    gt_unc = {}

    # Get model's free RVs and their names
    with model:
        model_vars = {rv.name for rv in model.free_RVs}

    print(f"Model expects {len(model_vars)} parameters")
    print()

    # Direct assignments (no transformation)
    for key in ["x_root", "raw_u_1", "raw_u_2", "raw_u_3", "raw_u_4", "raw_u_5"]:
        if key in gt_constrained and key in model_vars:
            gt_unc[key] = gt_constrained[key]
            print(f"[OK] {key}: direct copy")

    for key in ["length_1", "length_2", "length_3", "length_4", "length_5"]:
        if key in gt_constrained and key in model_vars:
            gt_unc[key] = gt_constrained[key]
            print(f"[OK] {key}: direct copy")

    # Log-transformed parameters (positive reals)
    if "eta2_root" in gt_constrained and "eta2_root" in model_vars:
        # Check if model uses log transform
        gt_unc["eta2_root_log__"] = np.log(gt_constrained["eta2_root"])
        gt_unc["eta2_root"] = gt_constrained["eta2_root"]
        print(
            f"[OK] eta2_root: {gt_constrained['eta2_root']:.4f} "
            f"-> log__ = {gt_unc['eta2_root_log__']:.4f}"
        )

    if "rho" in gt_constrained and "rho" in model_vars:
        gt_unc["rho_log__"] = np.log(gt_constrained["rho"])
        gt_unc["rho"] = gt_constrained["rho"]
        print(f"[OK] rho: log transform applied")

    if "sigma2" in gt_constrained and "sigma2" in model_vars:
        gt_unc["sigma2_log__"] = np.log(gt_constrained["sigma2"])
        gt_unc["sigma2"] = gt_constrained["sigma2"]
        print(f"[OK] sigma2: log transform applied")

    if "obs_sigma" in gt_constrained and "obs_sigma" in model_vars:
        val = gt_constrained["obs_sigma"]
        gt_unc["obs_sigma_log__"] = np.log(val)
        gt_unc["obs_sigma"] = val
        print(f"[OK] obs_sigma: {val:.4f} -> log__ = {gt_unc['obs_sigma_log__']:.4f}")

    # Logit-transformed parameters (unit interval)
    # Note: model uses "logodds_inlier" as the free parameter
    if "inlier_prob" in gt_constrained:
        prob = gt_constrained["inlier_prob"]

        # Critical check: ensure prob is in (0, 1) strictly
        if prob <= 0 or prob >= 1:
            print(f"[WARNING] inlier_prob = {prob} is at boundary!")
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            print(f"          Clipped to {prob}")

        logodds = np.log(prob / (1 - prob))
        gt_unc["logodds_inlier"] = logodds  # This is the actual model parameter name

        print(f"[OK] inlier_prob: {prob:.4f} -> logodds_inlier = {logodds:.4f}")

        if np.isinf(logodds):
            print(f"[ERROR] logodds_inlier is INFINITE!")

    # HMM parameters
    # Canonical directions (no transform)
    for key in gt_constrained:
        if key.startswith("dir_hmm_mu_raw_"):
            gt_unc[key] = gt_constrained[key]
            print(f"[OK] {key}: direct copy")

    # Kappa (HalfNormal, so both constrained and log-transformed versions)
    for key in gt_constrained:
        if key.startswith("dir_hmm_kappa_"):
            gt_unc[key] = gt_constrained[key]  # Constrained value
            gt_unc[f"{key}_log__"] = np.log(gt_constrained[key])  # Unconstrained
            print(f"[OK] {key}: both constrained and log-transformed")

    # HMM init/trans (convert from probability to logits)
    if "dir_hmm_init_dist" in gt_constrained:
        probs = gt_constrained["dir_hmm_init_dist"]
        # Convert to logits relative to first state
        logits = np.log(probs / probs[0])
        gt_unc["dir_hmm_init_logits"] = logits
        print(f"[OK] dir_hmm_init_logits: converted from probabilities")

    if "dir_hmm_trans_mat" in gt_constrained:
        trans = gt_constrained["dir_hmm_trans_mat"]
        # Convert to logits relative to first column
        logits = np.log(trans / trans[:, 0:1])
        gt_unc["dir_hmm_trans_logits"] = logits
        print(f"[OK] dir_hmm_trans_logits: converted from transition matrix")

    print()
    print(f"Unconstrained parameters created: {len(gt_unc)}")

    # Verify we have all required model parameters
    missing = model_vars - set(gt_unc.keys())
    if missing:
        print()
        print(f"[WARNING] Missing {len(missing)} model parameters:")
        for name in sorted(missing):
            print(f"  - {name}")
    else:
        print("[OK] All model parameters have ground truth values!")

    print()

    return gt_unc


def verify_ground_truth_coverage(
    gt_unconstrained: Dict[str, np.ndarray], model: pm.Model
) -> Tuple[bool, list, list]:
    """
    Verify that ground truth covers all model parameters.

    Returns
    -------
    is_complete : bool
        True if all model parameters are covered
    missing : list
        List of model parameters not in ground truth
    extra : list
        List of ground truth parameters not in model
    """
    with model:
        model_params = {rv.name for rv in model.free_RVs}

    gt_params = set(gt_unconstrained.keys())

    missing = model_params - gt_params
    extra = gt_params - model_params

    is_complete = len(missing) == 0

    return is_complete, sorted(missing), sorted(extra)


if __name__ == "__main__":
    """
    Test the extraction and transformation functions.
    """
    from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
    from tests.diagnostics.v0_2_1_divergence.test_utils import build_test_model

    print("=" * 80)
    print("Testing Ground Truth Parameter Extraction")
    print("=" * 80)
    print()

    # Generate synthetic data
    config = SyntheticDataConfig(
        T=50,
        C=4,
        S=2,
        kappa=10.0,
        obs_noise_std=0.5,
        occlusion_rate=0.05,
        random_seed=42,
    )
    synth_data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    print("[OK] Generated synthetic data")
    print()

    # Extract ground truth
    gt_constrained = extract_complete_ground_truth(synth_data, DEMO_V0_1_SKELETON, S=2)

    # Build model
    data_dict = {
        "observations_uv": synth_data.y_observed,
        "camera_matrices": synth_data.camera_proj,
        "joint_positions": synth_data.x_true,
        "joint_names": DEMO_V0_1_SKELETON.joint_names,
        "parents": DEMO_V0_1_SKELETON.parents,
        "bone_lengths": DEMO_V0_1_SKELETON.bone_lengths,
        "true_states": synth_data.true_states,
        "config": config,
    }
    model = build_test_model(data_dict, use_directional_hmm=True, S=2)
    print("[OK] Built model")
    print()

    # Transform to unconstrained
    gt_unconstrained = transform_to_unconstrained_space(gt_constrained, model)

    # Verify coverage
    print("=" * 80)
    print("Verifying Ground Truth Coverage")
    print("=" * 80)
    print()

    is_complete, missing, extra = verify_ground_truth_coverage(gt_unconstrained, model)

    if is_complete:
        print("[SUCCESS] Ground truth covers ALL model parameters!")
    else:
        print(f"[FAIL] Missing {len(missing)} parameters:")
        for name in missing:
            print(f"  - {name}")

    if extra:
        print(
            f"\n[INFO] {len(extra)} extra parameters in ground truth (OK if not used):"
        )
        for name in extra[:5]:  # Show first 5
            print(f"  - {name}")
        if len(extra) > 5:
            print(f"  ... and {len(extra) - 5} more")

    print()
    print("=" * 80)
