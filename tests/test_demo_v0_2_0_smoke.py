"""
Smoke test for v0.1 complete pipeline.

This test validates that the full PyMC HMM pipeline (Stage 1-3) can be
built and that the model graph is valid. It does not test convergence,
only that:
- Imports work correctly
- Shapes match expectations
- The model can be constructed without errors
- Prior predictive sampling runs

This provides a fast regression test to catch obvious breakage.
"""

import pytest
import numpy as np
import pymc as pm

from gimbal import (
    build_camera_observation_model,
    add_directional_hmm_prior,
    generate_demo_sequence,
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
)


def test_v0_1_pipeline_builds():
    """Test that the full v0.1 pipeline can be built."""
    # Generate minimal synthetic data
    config = SyntheticDataConfig(
        T=10,  # Minimal timesteps for speed
        C=2,   # Minimal cameras
        S=2,   # Minimal states
        random_seed=42,
    )
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    
    # Validate synthetic data shapes
    assert data.x_true.shape == (10, 6, 3)
    assert data.u_true.shape == (10, 6, 3)
    assert data.true_states.shape == (10,)
    assert data.y_observed.shape == (2, 10, 6, 2)
    assert data.camera_proj.shape == (2, 3, 4)
    
    # Build Stage 2 model
    with pm.Model() as model:
        model_result, U, x_all, y_pred, log_obs_t = build_camera_observation_model(
            y_obs=data.y_observed,
            proj_param=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
        )
        
        # Validate Stage 2 shapes
        assert U.type.shape == (10, 6, 3)  # (T, K, 3)
        assert x_all.type.shape == (10, 6, 3)  # (T, K, 3)
        assert y_pred.type.shape == (2, 10, 6, 2)  # (C, T, K, 2)
        assert log_obs_t.type.shape == (10,)  # (T,)
        
        # Add Stage 3 directional HMM prior
        hmm_vars = add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=2,  # Minimal states
        )
        
        # Validate Stage 3 variables exist
        assert "mu" in hmm_vars
        assert "kappa" in hmm_vars
        assert "hmm_loglik" in hmm_vars
        assert "logp_emit" in hmm_vars
        
        # Validate Stage 3 shapes
        assert hmm_vars["mu"].type.shape == (2, 6, 3)  # (S, K, 3)
        assert hmm_vars["kappa"].type.shape == (2, 6)  # (S, K)
        
        # Check that model has the expected number of free variables
        # This is a rough check - exact number depends on DLT initialization
        assert len(model.free_RVs) > 0
        
        # Validate model graph is buildable (no shape errors)
        # This will raise if there are issues with the model
        model.debug()


def test_v0_1_prior_predictive_sampling():
    """Test that prior predictive sampling works."""
    config = SyntheticDataConfig(T=5, C=2, S=2, random_seed=42)
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    
    with pm.Model() as model:
        _, U, x_all, y_pred, log_obs_t = build_camera_observation_model(
            y_obs=data.y_observed,
            proj_param=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
        )
        
        add_directional_hmm_prior(U=U, log_obs_t=log_obs_t, S=2)
        
        # Run minimal prior predictive sampling
        prior_pred = pm.sample_prior_predictive(
            samples=10,
            random_seed=42,
        )
        
        # Check that we got samples
        assert prior_pred.prior is not None
        # Check that key variables are present
        assert "dir_hmm_mu" in prior_pred.prior
        assert "dir_hmm_kappa_full" in prior_pred.prior


def test_v0_1_backward_compatibility():
    """Test that v0.1 behavior is preserved (no priors changed)."""
    config = SyntheticDataConfig(T=5, C=2, S=2, random_seed=42)
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    
    # Build model with default parameters (v0.1 behavior)
    with pm.Model() as model_v1:
        _, U, x_all, y_pred, log_obs_t = build_camera_observation_model(
            y_obs=data.y_observed,
            proj_param=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
        )
        add_directional_hmm_prior(U=U, log_obs_t=log_obs_t, S=2)
    
    # Build model with explicit prior_config=None (should be identical)
    with pm.Model() as model_v2:
        _, U, x_all, y_pred, log_obs_t = build_camera_observation_model(
            y_obs=data.y_observed,
            proj_param=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
        )
        add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=2,
            prior_config=None,  # Explicit None should be same as default
        )
    
    # Both models should have same structure
    assert len(model_v1.free_RVs) == len(model_v2.free_RVs)
    assert set(rv.name for rv in model_v1.free_RVs) == set(rv.name for rv in model_v2.free_RVs)


def test_synthetic_data_deterministic():
    """Test that synthetic data generation is deterministic with seed."""
    config = SyntheticDataConfig(T=10, C=2, S=2, random_seed=42)
    
    data1 = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    data2 = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    
    # Should produce identical results with same seed
    np.testing.assert_array_equal(data1.x_true, data2.x_true)
    np.testing.assert_array_equal(data1.u_true, data2.u_true)
    np.testing.assert_array_equal(data1.true_states, data2.true_states)
    
    # Observations may differ slightly due to occlusion randomness,
    # but non-NaN values should be close
    mask1 = ~np.isnan(data1.y_observed)
    mask2 = ~np.isnan(data2.y_observed)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_allclose(
        data1.y_observed[mask1],
        data2.y_observed[mask2],
        rtol=1e-10,
    )


if __name__ == "__main__":
    # Run tests directly
    test_v0_1_pipeline_builds()
    print("✓ Pipeline builds successfully")
    
    test_v0_1_prior_predictive_sampling()
    print("✓ Prior predictive sampling works")
    
    test_v0_1_backward_compatibility()
    print("✓ Backward compatibility maintained")
    
    test_synthetic_data_deterministic()
    print("✓ Synthetic data generation is deterministic")
    
    print("\nAll smoke tests passed!")
