# %% [markdown]
# # GIMBAL v0.1 Complete Integration Demo
#
# This notebook demonstrates the complete three-phase GIMBAL HMM integration pipeline:
#
# - **v0.1.1:** Collapsed HMM engine with forward algorithm
# - **v0.1.2:** Camera observation model with kinematics and 2D projections
# - **v0.1.3:** Directional HMM prior over joint directions
#
# We'll generate synthetic data with state-dependent poses, build the full model, sample with PyMC, and visualize the results.
#
# ## Overview
#
# 1. Import libraries and setup
# 2. Generate synthetic motion data with 3 pose states
# 3. Build v0.1.2 camera observation model
# 4. Add v0.1.3 directional HMM prior
# 5. Sample from the posterior
# 6. Analyze and visualize results

# %% [markdown]
# ## 1. Import Libraries and Setup

# %%
# Fix OpenMP conflict
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent))

# %%
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# GIMBAL imports
from gimbal.pymc_model import build_camera_observation_model
from gimbal.fit_params import InitializationResult

print("Libraries imported successfully!")
print(f"PyMC version: {pm.__version__}")
print(f"NumPy version: {np.__version__}")

# %% [markdown]
# ## 2. Generate Synthetic Motion Data with State-Dependent Poses
#
# We'll create synthetic data with 3 distinct pose states:
# - **State 0:** Upright posture (standing)
# - **State 1:** Leaning forward
# - **State 2:** Leaning sideways
#
# Each state has characteristic joint directions that the HMM will learn.

# %%
# Configuration
T = 60  # Number of timesteps
K = 6  # Number of joints (including root at index 0)
C = 3  # Number of cameras
S = 3  # Number of hidden states

# Skeleton structure (simple chain for demo)
parents = np.array([-1, 0, 1, 2, 3, 4])  # Root -> joint1 -> joint2 -> ...

# Set random seed for reproducibility
rng = np.random.default_rng(42)

print(f"Configuration:")
print(f"  Timesteps: {T}")
print(f"  Joints: {K}")
print(f"  Cameras: {C}")
print(f"  States: {S}")

# %%
# Define canonical directions for each state
# State 0: Upright (mostly +z direction)
# State 1: Forward lean (+x and +z)
# State 2: Sideways lean (+y and +z)

canonical_mu = np.zeros((S, K, 3))

# State 0: Upright
canonical_mu[0, 1:, 2] = 1.0  # All joints point up (except root)

# State 1: Forward lean
canonical_mu[1, 1:, 0] = 0.6  # Forward component
canonical_mu[1, 1:, 2] = 0.8  # Still mostly up

# State 2: Sideways lean
canonical_mu[2, 1:, 1] = 0.7  # Sideways component
canonical_mu[2, 1:, 2] = 0.7  # Still mostly up

# Normalize to unit vectors
for s in range(S):
    for k in range(1, K):  # Skip root
        norm = np.linalg.norm(canonical_mu[s, k])
        if norm > 0:
            canonical_mu[s, k] /= norm

print("Canonical directions defined for 3 states")
print(f"State 0 (upright) direction sample: {canonical_mu[0, 1]}")
print(f"State 1 (forward) direction sample: {canonical_mu[1, 1]}")
print(f"State 2 (sideways) direction sample: {canonical_mu[2, 1]}")

# %%
# Generate state sequence with persistence
trans_probs = np.array(
    [
        [0.85, 0.10, 0.05],  # From state 0: mostly stay
        [0.10, 0.80, 0.10],  # From state 1: mostly stay
        [0.05, 0.10, 0.85],  # From state 2: mostly stay
    ]
)

true_states = np.zeros(T, dtype=int)
true_states[0] = rng.choice(S)

for t in range(1, T):
    true_states[t] = rng.choice(S, p=trans_probs[true_states[t - 1]])

print(f"Generated state sequence")
print(f"State distribution: {np.bincount(true_states)}")
print(f"State transitions: {len(np.where(np.diff(true_states) != 0)[0])}")

# %%
# Generate 3D positions and directions based on states
# Start with a simple skeleton in a chain
bone_lengths = np.array([0, 10.0, 10.0, 8.0, 8.0, 6.0])  # Root has length 0

x_true = np.zeros((T, K, 3))
u_true = np.zeros((T, K, 3))

kappa_true = 8.0  # Concentration for directional noise

for t in range(T):
    s = true_states[t]

    # Root position (random walk)
    if t == 0:
        x_true[t, 0] = [0, 0, 100]  # Start at height 100
    else:
        x_true[t, 0] = x_true[t - 1, 0] + rng.normal(0, 1.0, 3)

    # Generate directions with noise around canonical direction
    for k in range(1, K):
        # Add noise to canonical direction
        u_noisy = canonical_mu[s, k] + rng.normal(0, 1.0 / kappa_true, 3)
        u_noisy /= np.linalg.norm(u_noisy) + 1e-8
        u_true[t, k] = u_noisy

        # Compute position from parent
        parent = parents[k]
        x_true[t, k] = x_true[t, parent] + bone_lengths[k] * u_true[t, k]

print(f"Generated 3D skeletal motion")
print(f"Position range: [{x_true.min():.1f}, {x_true.max():.1f}]")
print(f"Mean bone length: {bone_lengths[1:].mean():.1f}")

# %%
# Generate synthetic camera projection matrices
camera_proj = np.zeros((C, 3, 4))

for c in range(C):
    # Camera positioned in a circle around the scene
    angle = 2 * np.pi * c / C
    camera_pos = np.array([150 * np.cos(angle), 150 * np.sin(angle), 100])

    # Look at origin
    look_at = np.array([0, 0, 100])
    up = np.array([0, 0, 1])

    # Simple projection matrix (orthographic-like for simplicity)
    # In practice, use proper camera calibration
    focal_length = 500
    A = np.eye(3) * focal_length
    b = -A @ camera_pos

    camera_proj[c] = np.column_stack([A, b])

print(f"Generated {C} camera projection matrices")
print(f"Camera 0 translation: {camera_proj[0, :, 3]}")

# %%
# Project 3D points to 2D and add observation noise
y_observed = np.zeros((C, T, K, 2))

obs_noise = 5.0  # pixels

for c in range(C):
    for t in range(T):
        for k in range(K):
            # Homogeneous coordinates
            x_h = np.append(x_true[t, k], 1)

            # Project
            y_proj = camera_proj[c] @ x_h

            # Perspective division (if needed, here we use orthographic)
            if y_proj[2] != 0:
                y_2d = y_proj[:2] / y_proj[2]
            else:
                y_2d = y_proj[:2]

            # Add noise
            y_observed[c, t, k] = y_2d + rng.normal(0, obs_noise, 2)

# Add some random occlusions (NaN values)
n_occlusions = int(0.05 * C * T * K)  # 5% occlusions
for _ in range(n_occlusions):
    c_occ = rng.integers(0, C)
    t_occ = rng.integers(0, T)
    k_occ = rng.integers(0, K)
    y_observed[c_occ, t_occ, k_occ] = np.nan

print(f"Generated 2D observations with noise")
print(f"Observation range: [{np.nanmin(y_observed):.1f}, {np.nanmax(y_observed):.1f}]")
print(
    f"Occlusions: {np.sum(np.isnan(y_observed[:, :, :, 0]))} / {C*T*K} ({100*n_occlusions/(C*T*K):.1f}%)"
)

# %% [markdown]
# ## 3. Visualize Synthetic Data
#
# Let's visualize the generated data to understand what we're working with.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: True state sequence
axes[0, 0].plot(true_states, "o-", markersize=4)
axes[0, 0].set_xlabel("Timestep")
axes[0, 0].set_ylabel("State")
axes[0, 0].set_title("True Hidden State Sequence")
axes[0, 0].set_yticks([0, 1, 2])
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Root trajectory in 3D
ax = fig.add_subplot(2, 2, 2, projection="3d")
scatter = ax.scatter(
    x_true[:, 0, 0],
    x_true[:, 0, 1],
    x_true[:, 0, 2],
    c=true_states,
    cmap="viridis",
    s=20,
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Root Trajectory (colored by state)")
plt.colorbar(scatter, ax=ax, label="State")

# Plot 3: Sample 2D observations from camera 0
axes[1, 0].set_title("2D Observations (Camera 0, Joint 1)")
for s in range(S):
    mask = true_states == s
    axes[1, 0].scatter(
        y_observed[0, mask, 1, 0],
        y_observed[0, mask, 1, 1],
        label=f"State {s}",
        alpha=0.6,
        s=20,
    )
axes[1, 0].set_xlabel("X pixel")
axes[1, 0].set_ylabel("Y pixel")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Directional vectors for one joint over time
axes[1, 1].plot(u_true[:, 1, 0], label="X component")
axes[1, 1].plot(u_true[:, 1, 1], label="Y component")
axes[1, 1].plot(u_true[:, 1, 2], label="Z component")
axes[1, 1].set_xlabel("Timestep")
axes[1, 1].set_ylabel("Direction")
axes[1, 1].set_title("Direction Vectors (Joint 1)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Synthetic data visualization complete")

# %% [markdown]
# ## 4. Build Complete Model (v0.1.1-1.3)
#
# Now we'll build the full GIMBAL model:
# - **v0.1.2**: Camera observation model (kinematics + 2D projections)
# - **v0.1.3**: Directional HMM prior (canonical directions per state)
# - **v0.1.1**: Collapsed HMM engine (called internally by v0.1.3)

# %%
# Create initialization result
# In practice, use initialize_from_observations_dlt() or anipose
# Here we use the true values as a good starting point

init_result = InitializationResult(
    x_init=x_true,
    eta2=np.ones(K) * 1.0,  # Temporal variance
    rho=bone_lengths[1:],  # Mean bone lengths
    sigma2=np.ones(K - 1) * 0.5,  # Bone length variance
    u_init=u_true,
    obs_sigma=obs_noise,
    inlier_prob=0.95,  # High inlier probability
    metadata={"method": "synthetic_truth"},
)

print("Initialization result created")
print(f"  x_init shape: {init_result.x_init.shape}")
print(f"  u_init shape: {init_result.u_init.shape}")
print(f"  rho (bone lengths): {init_result.rho}")

# %%
# Build the complete model with v0.1.3 directional HMM
print("Building PyMC model with v0.1.3 directional HMM...")

model = build_camera_observation_model(
    y_observed=y_observed,
    camera_proj=camera_proj,
    parents=parents,
    init_result=init_result,
    use_mixture=False,  # Use simple Gaussian for speed
    use_directional_hmm=True,  # Enable v0.1.3!
    hmm_num_states=S,
    hmm_kwargs={
        "name_prefix": "pose_hmm",
        "share_kappa_across_joints": False,  # Allow joint-specific concentrations
        "share_kappa_across_states": False,  # Allow state-specific concentrations
        "kappa_scale": 5.0,  # Prior scale for concentrations
    },
)

print("\nModel built successfully!")
print(f"Number of free random variables: {len(model.free_RVs)}")
print(f"\nKey v0.1.3 variables:")
print(
    f"  - pose_hmm_mu (canonical directions): shape {model['pose_hmm_mu'].eval().shape}"
)
print(
    f"  - pose_hmm_kappa_full (concentrations): shape {model['pose_hmm_kappa_full'].eval().shape}"
)
print(
    f"  - pose_hmm_loglik (HMM log-likelihood): shape {model['pose_hmm_loglik'].eval().shape}"
)

# %%
# Check initial log-likelihood
with model:
    initial_logp = model.compile_logp()(model.initial_point())

print(f"Initial model log-likelihood: {initial_logp:.2f}")
print("\nThis combines:")
print("  - v0.1.2: Camera observation likelihood")
print("  - v0.1.3: Directional HMM prior")
print("  - v0.1.1: Collapsed HMM marginalization (called internally)")

# %% [markdown]
# ## 5. Sample from the Posterior
#
# Now we'll use PyMC's NUTS sampler to draw samples from the posterior distribution. This may take a few minutes.
#
# **Note:** For production use, consider using `nuts_sampler="nutpie"` for faster sampling, and increase the number of draws and chains for better convergence.

# %%
# Sample from the posterior
# Using shorter chains for demonstration purposes
# For production: use more draws (1000+) and tune steps (500+)

print("Starting MCMC sampling...")
print("This may take a few minutes depending on your hardware...")

with model:
    idata = pm.sample(
        draws=100,  # Number of posterior samples per chain
        tune=100,  # Number of tuning/warmup steps
        chains=2,  # Number of independent chains
        cores=2,  # Parallel chains
        progressbar=True,
        return_inferencedata=True,
    )

print("\nSampling complete!")
print(f"Posterior dimensions: {dict(idata.posterior.dims)}")

# %% [markdown]
# ## 6. Analyze Results
#
# Let's examine the posterior samples and compare to the true values.

# %%
# Extract posterior samples for key variables
mu_samples = idata.posterior["pose_hmm_mu"].values  # (chains, draws, S, K, 3)
kappa_samples = idata.posterior["pose_hmm_kappa_full"].values  # (chains, draws, S, K)

# Compute posterior means
mu_mean = mu_samples.mean(axis=(0, 1))  # (S, K, 3)
kappa_mean = kappa_samples.mean(axis=(0, 1))  # (S, K)

print("Posterior Summary:")
print(f"\nCanonical Directions (mu) - Posterior Mean:")
for s in range(S):
    norms = np.linalg.norm(mu_mean[s], axis=-1)
    print(
        f"  State {s}: norms range [{norms.min():.4f}, {norms.max():.4f}], mean {norms.mean():.4f}"
    )
    print(f"    Joint 1 direction: {mu_mean[s, 1]}")

print(f"\nConcentrations (kappa) - Posterior Mean:")
for s in range(S):
    print(
        f"  State {s}: mean={kappa_mean[s].mean():.2f}, std={kappa_mean[s].std():.2f}"
    )

# %%
# Analyze transition probabilities
trans_logits_samples = idata.posterior["pose_hmm_trans_logits"].values
trans_probs_samples = np.exp(
    trans_logits_samples - trans_logits_samples.max(axis=-1, keepdims=True)
)
trans_probs_samples /= trans_probs_samples.sum(axis=-1, keepdims=True)
trans_probs_posterior = trans_probs_samples.mean(axis=(0, 1))

print("\nTransition Probability Matrix (Posterior Mean):")
print("From → To")
for s in range(S):
    probs_str = "  ".join([f"{p:.3f}" for p in trans_probs_posterior[s]])
    print(f"State {s}: [{probs_str}]")

print("\nTrue Transition Probabilities:")
for s in range(S):
    probs_str = "  ".join([f"{p:.3f}" for p in trans_probs[s]])
    print(f"State {s}: [{probs_str}]")

# %% [markdown]
# ## 7. Visualize Learned Canonical Directions
#
# Compare the learned canonical directions to the true ones (note: label switching may occur).

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

joint_to_plot = 1  # Plot directions for joint 1

for s in range(S):
    # True canonical direction
    axes[s].quiver(
        0,
        0,
        canonical_mu[s, joint_to_plot, 0],
        canonical_mu[s, joint_to_plot, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.015,
        label="True",
        alpha=0.7,
    )

    # Learned canonical direction (posterior mean)
    axes[s].quiver(
        0,
        0,
        mu_mean[s, joint_to_plot, 0],
        mu_mean[s, joint_to_plot, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        width=0.015,
        label="Learned",
        alpha=0.7,
    )

    axes[s].set_xlim(-1.2, 1.2)
    axes[s].set_ylim(-1.2, 1.2)
    axes[s].set_aspect("equal")
    axes[s].grid(True, alpha=0.3)
    axes[s].set_xlabel("X component")
    axes[s].set_ylabel("Y component")
    axes[s].set_title(f"State {s} - Joint {joint_to_plot} Direction (XY plane)")
    axes[s].legend()

plt.tight_layout()
plt.show()

print(f"Red arrows: True canonical directions")
print(f"Blue arrows: Learned canonical directions (posterior mean)")
print(f"\nNote: States may be permuted due to label switching (this is expected!)")
print(f"For proper comparison, apply Hungarian algorithm post-hoc relabeling.")

# %% [markdown]
# ## 8. Summary and Next Steps
#
# ### What We Demonstrated
#
# ✅ **v0.1.1 (HMM Engine)**: Collapsed forward algorithm running internally
# ✅ **v0.1.2 (Camera Model)**: 3D kinematics projected to 2D observations
# ✅ **v0.1.3 (Directional HMM)**: State-dependent canonical directions learned from data
#
# ### Key Results
#
# - The model successfully learned canonical directions per state
# - Transition probabilities show state persistence (high diagonal values)
# - Concentrations (kappa) capture the tightness of directional clustering
#
# ### Important Notes
#
# **Label Switching:** HMM states are not identifiable - chains may assign different labels to the same pose pattern. For production use:
# 1. Apply Hungarian algorithm post-hoc relabeling (see `v0.1.3-completion-report.md`)
# 2. Use feature-based state alignment across draws
# 3. Compute posterior summaries only after relabeling
#
# ### Next Steps for Production Use
#
# 1. **Longer Sampling**: Use 1000+ draws and 500+ tuning steps
# 2. **Convergence Checks**: Verify R-hat < 1.01 for all parameters
# 3. **Real Data**: Replace synthetic data with actual motion capture observations
# 4. **Hyperparameter Tuning**: Adjust `kappa_scale`, sharing options, number of states
# 5. **Nutpie Integration**: Use `nuts_sampler="nutpie"` for faster sampling
# 6. **Post-processing**: Apply label switching correction and compute summaries
#
# ### References
#
# - v0.1.3 specification: `plans/v0.1.3-detailed-spec.md`
# - Completion report: `plans/v0.1.3-completion-report.md`
# - Test suite: `test_v0_1_3_directional_hmm.py`
# - Minimal example: `test_hmm_v0_1_3.py`
