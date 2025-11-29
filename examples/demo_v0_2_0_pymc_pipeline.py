"""
Minimal PyMC HMM Pipeline Demo

This script demonstrates the complete GIMBAL v0.1 PyMC pipeline in a
minimal, easy-to-understand form. It's designed as a "hello world" example
for new users and AI coding assistants.

The pipeline consists of three stages:
1. Stage 1: Collapsed HMM engine (hmm_pytensor.py)
2. Stage 2: Camera observation model (pymc_model.py)
3. Stage 3: Directional HMM prior (hmm_directional.py)

This script generates synthetic data, builds the model, and runs a brief
sampling to validate the full pipeline works.
"""

import os
# Fix OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pymc as pm

# Import GIMBAL PyMC pipeline
import gimbal
from gimbal import (
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
    generate_demo_sequence,
    build_camera_observation_model,
    add_directional_hmm_prior,
)


def main():
    """Run minimal PyMC HMM pipeline demo."""
    
    print("=" * 70)
    print("GIMBAL v0.1 PyMC HMM Pipeline - Minimal Demo")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate Synthetic Data
    # -------------------------------------------------------------------------
    print("\nStep 1: Generating synthetic motion data...")
    
    config = SyntheticDataConfig(
        T=20,   # Short sequence for speed
        C=2,    # Two cameras (minimal)
        S=2,    # Two pose states (minimal)
        kappa=8.0,           # Directional noise concentration
        obs_noise_std=5.0,   # 5 pixels observation noise
        occlusion_rate=0.05, # 5% occlusions
        random_seed=42,      # Reproducibility
    )
    
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    
    print(f"✓ Generated {config.T} timesteps")
    print(f"  - Skeleton: {len(DEMO_V0_1_SKELETON.joint_names)} joints")
    print(f"  - Cameras: {config.C}")
    print(f"  - Hidden states: {config.S}")
    print(f"  - Ground truth state sequence: {data.true_states}")
    
    # -------------------------------------------------------------------------
    # Step 2: Build Stage 2 - Camera Observation Model
    # -------------------------------------------------------------------------
    print("\nStep 2: Building Stage 2 camera observation model...")
    
    with pm.Model() as model:
        model_result, U, x_all, y_pred, log_obs_t = build_camera_observation_model(
            y_obs=data.y_observed,
            proj_param=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
        )
        
        print(f"✓ Stage 2 built successfully")
        print(f"  - U (directions): {U.type.shape}")
        print(f"  - x_all (positions): {x_all.type.shape}")
        print(f"  - y_pred (2D projections): {y_pred.type.shape}")
        print(f"  - log_obs_t (observation likelihood): {log_obs_t.type.shape}")
        
        # ---------------------------------------------------------------------
        # Step 3: Add Stage 3 - Directional HMM Prior
        # ---------------------------------------------------------------------
        print("\nStep 3: Adding Stage 3 directional HMM prior...")
        
        hmm_vars = add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=config.S,
            name_prefix="dir_hmm",
            share_kappa_across_joints=False,
            share_kappa_across_states=False,
            kappa_scale=5.0,
        )
        
        print(f"✓ Stage 3 added successfully")
        print(f"  - mu (canonical directions): {hmm_vars['mu'].type.shape}")
        print(f"  - kappa (concentrations): {hmm_vars['kappa'].type.shape}")
        print(f"  - HMM log-likelihood added to model")
        
        # ---------------------------------------------------------------------
        # Step 4: Validate Model Structure
        # ---------------------------------------------------------------------
        print("\nStep 4: Validating model structure...")
        
        n_free_vars = len(model.free_RVs)
        print(f"✓ Model has {n_free_vars} free random variables")
        
        # Check that model is well-formed (no shape errors)
        try:
            model.debug()
            print("✓ Model graph is valid (no shape errors)")
        except Exception as e:
            print(f"⚠ Model validation warning: {e}")
        
        # ---------------------------------------------------------------------
        # Step 5: Run Prior Predictive Sampling
        # ---------------------------------------------------------------------
        print("\nStep 5: Running prior predictive sampling (quick test)...")
        
        try:
            prior_pred = pm.sample_prior_predictive(
                samples=10,
                random_seed=42,
            )
            
            print("✓ Prior predictive sampling successful")
            print(f"  - Sampled {len(prior_pred.prior.chains)} chain(s)")
            print(f"  - Variables: {list(prior_pred.prior.data_vars)[:5]}...")
            
        except Exception as e:
            print(f"⚠ Prior predictive sampling failed: {e}")
            print("  (This is non-critical for model structure validation)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nThe full GIMBAL v0.1 PyMC pipeline has been successfully built:")
    print("  ✓ Stage 1: Collapsed HMM engine (via collapsed_hmm_loglik)")
    print("  ✓ Stage 2: Camera observation model")
    print("  ✓ Stage 3: Directional HMM prior")
    print("\nNext steps:")
    print("  - For full inference, use pm.sample() with nutpie")
    print("  - See notebook/demo_v0_1_complete.ipynb for detailed walkthrough")
    print("  - See plans/v0.1-overview.md for architecture documentation")
    print("\nFor production sampling, you would run:")
    print("  compiled_model = pm.Model.compile(model)")
    print("  idata = nutpie.sample(compiled_model, chains=4, draws=1000)")


if __name__ == "__main__":
    main()
