"""Simple test runner for v0.1.3 tests without pytest."""

import sys
from pathlib import Path

# Add parent directory to path to import gimbal
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_v0_1_3_directional_hmm import (
    test_kappa_sharing_options,
    test_directional_hmm_shapes,
    test_numerical_stability_extreme_log_obs,
    test_gradient_computation,
    test_logp_normalization,
    test_integration_with_stage2,
)


def run_tests():
    """Run all v0.1.3 tests."""
    tests = [
        ("Kappa sharing options", test_kappa_sharing_options),
        ("Directional HMM shapes", test_directional_hmm_shapes),
        (
            "Numerical stability with extreme log_obs",
            test_numerical_stability_extreme_log_obs,
        ),
        ("Gradient computation", test_gradient_computation),
        ("LogP normalization", test_logp_normalization),
        ("Integration with v0.1.2", test_integration_with_stage2),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"Running: {name}...", end=" ")
            test_fn()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
