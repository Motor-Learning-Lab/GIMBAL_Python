"""
Test: Synthetic projection consistency.

Verifies that the synthetic generator (generate_observations) produces
results consistent with project_points_numpy, ensuring that:

1. Generative and inference paths use the same projection model
2. P matrices are correctly formed
3. Perspective division is applied consistently
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from gimbal.camera_utils import project_points_numpy


def test_projection_consistency():
    """
    Test that synthetic observations match project_points_numpy.

    This verifies that:
    - generate_observations() uses perspective division
    - project_points_numpy() mirrors PyTensor implementation
    - Both produce identical results (before noise/occlusions)
    """

    print("=" * 70)
    print("TEST: Synthetic Projection Consistency")
    print("=" * 70)

    # Generate synthetic data WITHOUT noise or occlusions
    config = SyntheticDataConfig(
        T=50,
        C=3,
        S=3,
        kappa=10.0,  # Tight distribution
        obs_noise_std=0.0,  # No noise
        occlusion_rate=0.0,  # No occlusions
        random_seed=42,
    )

    print(f"\nGenerating synthetic data:")
    print(f"  T={config.T}, C={config.C}, K={len(DEMO_V0_1_SKELETON.joint_names)}")
    print(f"  obs_noise_std={config.obs_noise_std} (no noise)")
    print(f"  occlusion_rate={config.occlusion_rate} (no occlusions)")

    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Project using NumPy projector
    print(f"\nProjecting with project_points_numpy...")
    y_numpy = project_points_numpy(data.x_true, data.camera_proj)  # (C, T, K, 2)

    # Compare with synthetic observations
    y_synthetic = data.y_observed  # (C, T, K, 2)

    print(f"\nComparing projections:")
    print(f"  y_numpy shape: {y_numpy.shape}")
    print(f"  y_synthetic shape: {y_synthetic.shape}")

    # Compute differences
    diff = y_numpy - y_synthetic

    # Statistics (should be near zero with no noise)
    rmse = np.sqrt(np.nanmean(diff**2))
    max_error = np.nanmax(np.abs(diff))
    mean_error = np.nanmean(np.abs(diff))

    print(f"\n  RMSE: {rmse:.6f} pixels")
    print(f"  Max error: {max_error:.6f} pixels")
    print(f"  Mean |error|: {mean_error:.6f} pixels")

    # Check for NaNs
    n_nan_numpy = np.sum(np.isnan(y_numpy))
    n_nan_synthetic = np.sum(np.isnan(y_synthetic))
    print(f"\n  NaNs in y_numpy: {n_nan_numpy}")
    print(f"  NaNs in y_synthetic: {n_nan_synthetic}")

    # Success criteria
    threshold = 1e-6  # Should match to machine precision

    if rmse < threshold and max_error < threshold:
        print(f"\n✅ SUCCESS: Projections match within {threshold:.1e} pixels")
        print(f"   Synthetic generator and project_points_numpy are consistent!")
        return True
    else:
        print(f"\n❌ FAILURE: Projections differ by more than {threshold:.1e} pixels")
        print(f"   This indicates generative and inference paths disagree!")

        # Show some example differences
        print(f"\n   Sample differences (first 5 points):")
        for c in range(min(3, config.C)):
            for k in range(min(3, len(DEMO_V0_1_SKELETON.joint_names))):
                t = 0
                print(f"     Camera {c}, Joint {k}, t=0:")
                print(
                    f"       NumPy:     [{y_numpy[c, t, k, 0]:.3f}, {y_numpy[c, t, k, 1]:.3f}]"
                )
                print(
                    f"       Synthetic: [{y_synthetic[c, t, k, 0]:.3f}, {y_synthetic[c, t, k, 1]:.3f}]"
                )
                print(
                    f"       Diff:      [{diff[c, t, k, 0]:.6f}, {diff[c, t, k, 1]:.6f}]"
                )

        return False


def test_camera_centers():
    """Verify camera centers can be extracted correctly."""

    print("\n" + "=" * 70)
    print("TEST: Camera Center Extraction")
    print("=" * 70)

    from gimbal.camera_utils import camera_center_from_proj

    # Generate cameras
    config = SyntheticDataConfig(T=10, C=3, random_seed=42)
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Extract centers
    centers = camera_center_from_proj(data.camera_proj)

    print(f"\nExtracted camera centers:")
    scene_center = np.array([0.0, 0.0, 100.0])
    for c in range(data.config.C):
        dist = np.linalg.norm(centers[c] - scene_center)
        print(
            f"  Camera {c}: [{centers[c, 0]:7.2f}, {centers[c, 1]:7.2f}, {centers[c, 2]:7.2f}]"
        )
        print(f"    Distance from scene center: {dist:.2f} units")

    print(f"\n✅ Camera centers extracted successfully")
    return True


if __name__ == "__main__":
    success1 = test_projection_consistency()
    success2 = test_camera_centers()

    print("\n" + "=" * 70)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        exit(1)
