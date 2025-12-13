"""
GIMBAL v0.2.1 Divergence Test Suite Runner

This script runs all divergence diagnostic tests and generates a comprehensive report.
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.diagnostics.v0_2_1_divergence.test_baseline import run_baseline_tests
from tests.diagnostics.v0_2_1_divergence.test_hmm_effect import run_hmm_effect_tests
from tests.diagnostics.v0_2_1_divergence.test_state_count import run_state_count_tests
from tests.diagnostics.v0_2_1_divergence.test_likelihood_scale import (
    run_likelihood_scale_tests,
)
from tests.diagnostics.v0_2_1_divergence.test_diagnostics import (
    run_divergence_diagnostics,
)
from tests.diagnostics.v0_2_1_divergence.test_root_variance import (
    run_root_variance_tests,
)
from tests.diagnostics.v0_2_1_divergence.test_bone_length_variance import (
    run_bone_length_tests,
)
from tests.diagnostics.v0_2_1_divergence.test_runtime_scaling import (
    run_runtime_scaling_tests,
)
from tests.diagnostics.v0_2_1_divergence.test_group_9_root_fixed import (
    run_group_9_root_fixed,
)
from tests.diagnostics.v0_2_1_divergence.test_group_10_direct_3d import (
    run_group_10_direct_3d,
)
from tests.diagnostics.v0_2_1_divergence.test_group_11_redundancy_fixed import (
    run_group_11_redundancy_fixed,
)
from tests.diagnostics.v0_2_1_divergence.test_group_11_redundancy_priors import (
    run_group_11_redundancy_priors,
)
from tests.diagnostics.v0_2_1_divergence.report_generator import generate_report


def main():
    """Run all divergence tests and generate report."""
    print("=" * 80)
    print("GIMBAL v0.2.1 Divergence Test Suite")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_results = []

    # Test Group 1: Baseline (HMM OFF)
    print("\n" + "=" * 80)
    print("Test Group 1: Baseline Model Tests (HMM OFF)")
    print("=" * 80)
    try:
        results = run_baseline_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} baseline tests")
    except Exception as e:
        print(f"[FAIL] Error in baseline tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 2: HMM Effect (HMM ON)
    print("\n" + "=" * 80)
    print("Test Group 2: HMM Effect Tests (HMM ON)")
    print("=" * 80)
    try:
        results = run_hmm_effect_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} HMM effect tests")
    except Exception as e:
        print(f"[FAIL] Error in HMM effect tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 3: State Count Comparison
    print("\n" + "=" * 80)
    print("Test Group 3: State Count Comparison (S=1,2,3)")
    print("=" * 80)
    try:
        results = run_state_count_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} state count tests")
    except Exception as e:
        print(f"[FAIL] Error in state count tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 4: Likelihood Scale
    print("\n" + "=" * 80)
    print("Test Group 4: Likelihood Scale Comparison")
    print("=" * 80)
    try:
        results = run_likelihood_scale_tests(all_results)
        all_results.extend(results)
        print(f"[OK] Completed likelihood scale analysis")
    except Exception as e:
        print(f"[FAIL] Error in likelihood scale tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 5: Divergence Localization
    print("\n" + "=" * 80)
    print("Test Group 5: Divergence Localization Diagnostics")
    print("=" * 80)
    try:
        run_divergence_diagnostics(all_results)
        print(f"[OK] Completed divergence diagnostics")
    except Exception as e:
        print(f"[FAIL] Error in divergence diagnostics: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 6: Root Variance Sensitivity
    print("\n" + "=" * 80)
    print("Test Group 6: Root Variance Sensitivity")
    print("=" * 80)
    try:
        results = run_root_variance_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} root variance tests")
    except Exception as e:
        print(f"[FAIL] Error in root variance tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 7: Bone Length Variance Sensitivity
    print("\n" + "=" * 80)
    print("Test Group 7: Bone Length Variance Sensitivity")
    print("=" * 80)
    try:
        results = run_bone_length_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} bone length tests")
    except Exception as e:
        print(f"[FAIL] Error in bone length tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 8: Runtime Scaling
    print("\n" + "=" * 80)
    print("Test Group 8: Runtime Scaling Tests")
    print("=" * 80)
    try:
        results = run_runtime_scaling_tests()
        all_results.extend(results)
        print(f"[OK] Completed {len(results)} runtime scaling tests")
    except Exception as e:
        print(f"[FAIL] Error in runtime scaling tests: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 9: Root RW Funnel Diagnostic
    print("\n" + "=" * 80)
    print("Test Group 9: Root RW Funnel Diagnostic")
    print("=" * 80)
    try:
        results = run_group_9_root_fixed()
        all_results.append(results)
        print(f"[OK] Completed Test Group 9 (Root RW)")
    except Exception as e:
        print(f"[FAIL] Error in Test Group 9: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 10: Camera Likelihood Conditioning
    print("\n" + "=" * 80)
    print("Test Group 10: Camera Likelihood Conditioning Diagnostic")
    print("=" * 80)
    try:
        results = run_group_10_direct_3d()
        all_results.append(results)
        print(f"[OK] Completed Test Group 10 (Camera Conditioning)")
    except Exception as e:
        print(f"[FAIL] Error in Test Group 10: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 11.1: Parameter Redundancy (Fixed)
    print("\n" + "=" * 80)
    print("Test Group 11.1: Parameter Redundancy Diagnostic (Fixed)")
    print("=" * 80)
    try:
        results = run_group_11_redundancy_fixed()
        all_results.append(results)
        print(f"[OK] Completed Test Group 11.1 (Redundancy - Fixed)")
    except Exception as e:
        print(f"[FAIL] Error in Test Group 11.1: {e}")
        import traceback

        traceback.print_exc()

    # Test Group 11.2: Parameter Redundancy (Strong Priors)
    print("\n" + "=" * 80)
    print("Test Group 11.2: Parameter Redundancy Diagnostic (Strong Priors)")
    print("=" * 80)
    try:
        results = run_group_11_redundancy_priors()
        all_results.append(results)
        print(f"[OK] Completed Test Group 11.2 (Redundancy - Strong Priors)")
    except Exception as e:
        print(f"[FAIL] Error in Test Group 11.2: {e}")
        import traceback

        traceback.print_exc()

    # Generate Report
    print("\n" + "=" * 80)
    print("Generating Report")
    print("=" * 80)
    try:
        report_path = generate_report(all_results)
        print(f"[OK] Report generated: {report_path}")
    except Exception as e:
        print(f"[FAIL] Error generating report: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tests run: {len(all_results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
