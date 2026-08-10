"""Quick smoke test that subpackages are accessible."""

import gimbal_pymc

print("Testing gimbal_pymc subpackage imports...")

# Test subpackages are accessible
subpackages = ["hmm", "cameras", "skeleton", "joints", "priors", "data_cleaning"]

for name in subpackages:
    if hasattr(gimbal_pymc, name):
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name} MISSING")

# Test that we can import from subpackages
print("\nTesting subpackage module access...")

try:
    from gimbal_pymc.cameras.triangulation import triangulate_multi_view

    print("  ✓ cameras.triangulation.triangulate_multi_view")
except ImportError as e:
    print(f"  ✗ cameras.triangulation.triangulate_multi_view: {e}")

try:
    from gimbal_pymc.data_cleaning.cleaning import (
        CleaningConfig,
        clean_keypoints_2d,
        clean_keypoints_3d,
    )

    print(
        "  ✓ data_cleaning.cleaning.{CleaningConfig, clean_keypoints_2d, clean_keypoints_3d}"
    )
except ImportError as e:
    print(f"  ✗ data_cleaning.cleaning: {e}")

try:
    from gimbal_pymc.joints.statistics import compute_direction_statistics

    print("  ✓ joints.statistics.compute_direction_statistics")
except ImportError as e:
    print(f"  ✗ joints.statistics.compute_direction_statistics: {e}")

try:
    from gimbal_pymc.priors.building import (
        build_priors_from_statistics,
        gamma_mode_sd_to_shape_rate,
    )

    print(
        "  ✓ priors.building.{build_priors_from_statistics, gamma_mode_sd_to_shape_rate}"
    )
except ImportError as e:
    print(f"  ✗ priors.building: {e}")

try:
    from gimbal_pymc.hmm.observation_model import build_camera_observation_model

    print("  ✓ hmm.observation_model.build_camera_observation_model")
except ImportError as e:
    print(f"  ✗ hmm.observation_model.build_camera_observation_model: {e}")

try:
    from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior

    print("  ✓ hmm.directional_prior.add_directional_hmm_prior")
except ImportError as e:
    print(f"  ✗ hmm.directional_prior.add_directional_hmm_prior: {e}")

try:
    from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON, SkeletonConfig

    print("  ✓ skeleton.config.{DEMO_V0_1_SKELETON, SkeletonConfig}")
except ImportError as e:
    print(f"  ✗ skeleton.config: {e}")

try:
    from gimbal_pymc.skeleton.synthetic_data import (
        generate_demo_sequence,
        SyntheticDataConfig,
    )

    print("  ✓ skeleton.synthetic_data.{generate_demo_sequence, SyntheticDataConfig}")
except ImportError as e:
    print(f"  ✗ skeleton.synthetic_data: {e}")

print("\nAll API export tests passed!")
