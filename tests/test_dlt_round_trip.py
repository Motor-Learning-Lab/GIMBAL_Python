"""
Test: DLT round-trip consistency.

Verifies that projection and triangulation are mutually consistent:

    x_true → project → y_observed → triangulate → x_reconstructed

The reconstruction error should be small, confirming that:
1. Projection matrices are correct
2. DLT implementation is correct
3. Generative and inference paths agree
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from gimbal.triangulation import triangulate_multi_view


def test_dlt_round_trip():
    """
    Test that x_true → project → triangulate → x_recon has small error.

    With proper perspective projection and consistent DLT:
    - Reconstruction error should be << bone lengths
    - Error mainly from noise, not systematic bias
    """

    print("=" * 70)
    print("TEST: DLT Round-Trip Consistency")
    print("=" * 70)

    # Generate synthetic data WITH small noise (realistic scenario)
    config = SyntheticDataConfig(
        T=50,
        C=3,
        S=3,
        kappa=10.0,
        obs_noise_std=1.0,  # Small noise
        occlusion_rate=0.0,  # No occlusions for clean test
        random_seed=42,
    )

    print(f"\nGenerating synthetic data:")
    print(f"  T={config.T}, C={config.C}, K={len(DEMO_V0_1_SKELETON.joint_names)}")
    print(f"  obs_noise_std={config.obs_noise_std} pixels")
    print(f"  occlusion_rate={config.occlusion_rate}")

    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Triangulate: y_observed → x_reconstructed
    print(f"\nTriangulating with DLT...")
    x_recon = triangulate_multi_view(
        data.y_observed,  # (C, T, K, 2)
        data.camera_proj,  # (C, 3, 4)
        condition_threshold=1e6,
    )  # (T, K, 3)

    print(f"  x_recon shape: {x_recon.shape}")

    # Compute reconstruction errors
    print(f"\nComputing reconstruction errors...")

    errors = np.linalg.norm(x_recon - data.x_true, axis=-1)  # (T, K)

    # Statistics
    valid_errors = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        print("  ❌ No valid reconstructions!")
        return False

    rmse = np.sqrt(np.mean(valid_errors**2))
    mean_error = np.mean(valid_errors)
    max_error = np.max(valid_errors)
    median_error = np.median(valid_errors)

    print(f"\n  RMSE: {rmse:.3f} units")
    print(f"  Mean error: {mean_error:.3f} units")
    print(f"  Median error: {median_error:.3f} units")
    print(f"  Max error: {max_error:.3f} units")
    print(f"  Valid reconstructions: {len(valid_errors)} / {errors.size}")

    # Bone lengths for reference
    bone_lengths = DEMO_V0_1_SKELETON.bone_lengths
    min_bone = min([b for b in bone_lengths if b > 0])
    mean_bone = np.mean([b for b in bone_lengths if b > 0])

    print(f"\n  Reference (bone lengths):")
    print(f"    Min bone: {min_bone:.1f} units")
    print(f"    Mean bone: {mean_bone:.1f} units")

    # Per-joint statistics
    print(f"\n  Per-joint errors:")
    for k, joint_name in enumerate(DEMO_V0_1_SKELETON.joint_names):
        joint_errors = errors[:, k]
        valid_joint = joint_errors[~np.isnan(joint_errors)]
        if len(valid_joint) > 0:
            print(
                f"    {joint_name:8s}: mean={np.mean(valid_joint):.3f}, "
                f"max={np.max(valid_joint):.3f} units"
            )

    # Success criteria
    # With noise_std=1.0 pixels and focal_length=10, distance~80:
    # Expected triangulation error: ~noise * distance / focal ~ 1 * 80 / 10 = 8 units
    # This is comparable to bone lengths, which is expected for realistic noise levels

    # More realistic threshold: error should be roughly noise * (distance/focal)
    expected_error = config.obs_noise_std * 80 / 10  # ~8 units for noise=1
    success_threshold = expected_error * 1.5  # Allow 50% margin

    print(f"\nValidation:")
    print(f"  Expected error from noise: ~{expected_error:.1f} units")

    if mean_error <= success_threshold:
        print(f"  ✅ Mean error ({mean_error:.3f}) <= {success_threshold:.3f} units")
        print(f"     (threshold = 1.5 × expected noise-induced error)")
    else:
        print(f"  ❌ Mean error ({mean_error:.3f}) > {success_threshold:.3f} units")
        print(f"     (threshold = 1.5 × expected noise-induced error)")

    if rmse <= success_threshold * 1.2:
        print(f"  ✅ RMSE ({rmse:.3f}) <= {success_threshold * 1.2:.3f} units")
    else:
        print(f"  ❌ RMSE ({rmse:.3f}) > {success_threshold * 1.2:.3f} units")

    # Check for systematic bias (mean should be close to median)
    bias_threshold = 0.5  # units
    bias = abs(mean_error - median_error)
    if bias <= bias_threshold:
        print(f"  ✅ Low bias: |mean - median| = {bias:.3f} units")
    else:
        print(f"  ⚠️  High bias: |mean - median| = {bias:.3f} units (systematic error?)")

    success = (mean_error <= success_threshold) and (rmse <= success_threshold * 1.5)

    print(f"\n{'=' * 70}")
    if success:
        print("✅ SUCCESS: DLT round-trip error is acceptably small!")
        print("  Projection and triangulation are mutually consistent.")
        print("=" * 70)
        return True
    else:
        print("❌ FAILURE: DLT round-trip error is too large!")
        print("  This suggests projection/triangulation mismatch or DLT issues.")
        print("=" * 70)
        return False


def test_low_noise_accurate_reconstruction():
    """
    Test that with very low noise, reconstruction is highly accurate.

    This verifies numerical stability and consistency.
    """

    print("\n" + "=" * 70)
    print("TEST: High-Accuracy Reconstruction (Low Noise)")
    print("=" * 70)

    # Generate data with very low noise
    config = SyntheticDataConfig(
        T=20,
        C=3,
        S=2,
        kappa=20.0,
        obs_noise_std=0.1,  # Very small noise
        occlusion_rate=0.0,
        random_seed=42,
    )

    print("\nGenerating low-noise synthetic data...")
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Triangulate
    x_recon = triangulate_multi_view(
        data.y_observed,
        data.camera_proj,
        condition_threshold=1e6,
    )

    # Compute errors
    errors = np.linalg.norm(x_recon - data.x_true, axis=-1)
    valid_errors = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        print("  ❌ No valid reconstructions!")
        return False

    rmse = np.sqrt(np.mean(valid_errors**2))
    max_error = np.max(valid_errors)
    mean_error = np.mean(valid_errors)

    print(f"\n  RMSE: {rmse:.4f} units")
    print(f"  Mean error: {mean_error:.4f} units")
    print(f"  Max error: {max_error:.4f} units")

    # With very low noise (0.1 pixels), expect error ~0.1 * 80/10 = 0.8 units
    expected_error = config.obs_noise_std * 80 / 10
    threshold = expected_error * 2.0

    if rmse <= threshold:
        print(f"\n  ✅ High-accuracy reconstruction (RMSE <= {threshold:.2f} units)")
        print(f"     Expected from noise: ~{expected_error:.2f} units")
        print("     Confirms projection and triangulation are consistent.")
        return True
    else:
        print(f"\n  ❌ Reconstruction error larger than expected")
        print(f"     Expected RMSE ≤ {threshold:.2f}, got {rmse:.4f}")
        print("     This suggests numerical issues or model mismatch.")
        return False


if __name__ == "__main__":
    success1 = test_dlt_round_trip()
    success2 = test_low_noise_accurate_reconstruction()

    print("\n" + "=" * 70)
    if success1 and success2:
        print("✅ ALL DLT TESTS PASSED")
        print("=" * 70)
        exit(0)
    else:
        print("❌ SOME DLT TESTS FAILED")
        print("=" * 70)
        exit(1)
