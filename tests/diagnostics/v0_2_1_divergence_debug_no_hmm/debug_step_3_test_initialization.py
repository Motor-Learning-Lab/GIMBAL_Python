"""
Debug Step 3: Test Initialization and Log Probability

This script tests if we can evaluate the model's log probability at the
initial point. This is the first step that pm.sample() performs before
gradient compilation.

Expected outcome: Should get finite log probability. If we get -inf or crash,
this indicates initialization issues.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add test utilities path
sys.path.insert(0, str(project_root / "tests" / "diagnostics" / "v0_2_1_divergence"))

import numpy as np
import pymc as pm
from test_utils import get_standard_synth_data, build_test_model


def test_initialization():
    """Test if we can evaluate log probability at initial point."""

    print("=" * 60)
    print("DEBUG STEP 3: Test Initialization and Log Probability")
    print("=" * 60)
    print()

    results = {
        "step": 3,
        "description": "Test initialization and log probability evaluation",
        "checks": {},
    }

    # Step 1: Generate data and build model
    print("[1/5] Generating data and building model...")
    try:
        T = 100
        C = 3
        S = 3
        seed = 42

        synth_data = get_standard_synth_data(T=T, C=C, S=S, seed=seed)

        model = build_test_model(
            synth_data=synth_data,
            use_directional_hmm=False,
            S=S,
            eta2_root_sigma=0.5,
            sigma2_sigma=0.2,
        )

        print("[OK] Model built")
        results["checks"]["model_build"] = {
            "status": "PASS",
            "message": "Model built successfully",
        }
    except Exception as e:
        print(f"[FAIL] Model building failed: {e}")
        results["checks"]["model_build"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 2: Get initial point
    print("[2/5] Getting initial point...")
    try:
        with model:
            initial_point = model.initial_point()
            n_vars = len(initial_point)

            print(f"  - Variables: {n_vars}")

            results["checks"]["initial_point"] = {
                "status": "PASS",
                "n_variables": n_vars,
                "variables": list(initial_point.keys()),
            }

        print("[OK] Initial point obtained")
    except Exception as e:
        print(f"[FAIL] Getting initial point failed: {e}")
        results["checks"]["initial_point"] = {
            "status": "FAIL",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return results

    # Step 3: Check for NaN/inf in initial point
    print("[3/5] Checking initial point values...")
    try:
        has_nan = False
        has_inf = False
        bad_vars = []

        for key, val in initial_point.items():
            if isinstance(val, np.ndarray):
                if np.isnan(val).any():
                    has_nan = True
                    bad_vars.append(f"{key}: has NaN")
                if np.isinf(val).any():
                    has_inf = True
                    bad_vars.append(f"{key}: has inf")
            else:
                if np.isnan(val):
                    has_nan = True
                    bad_vars.append(f"{key}: is NaN")
                if np.isinf(val):
                    has_inf = True
                    bad_vars.append(f"{key}: is inf")

        if has_nan or has_inf:
            print(f"[WARN] Found invalid values in initial point:")
            for var in bad_vars:
                print(f"  - {var}")
            results["checks"]["initial_values"] = {
                "status": "WARN",
                "has_nan": has_nan,
                "has_inf": has_inf,
                "bad_variables": bad_vars,
            }
        else:
            print("[OK] All initial values are finite")
            results["checks"]["initial_values"] = {
                "status": "PASS",
                "has_nan": False,
                "has_inf": False,
            }
    except Exception as e:
        print(f"[FAIL] Checking initial values failed: {e}")
        results["checks"]["initial_values"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 4: Compile log probability function
    print("[4/5] Compiling log probability function...")
    try:
        with model:
            # This is what pm.sample does first
            logp_fn = model.compile_logp()

        print("[OK] Log probability function compiled")
        results["checks"]["logp_compilation"] = {
            "status": "PASS",
            "message": "Log probability compiled successfully",
        }
    except Exception as e:
        print(f"[FAIL] Compiling logp failed: {e}")
        results["checks"]["logp_compilation"] = {
            "status": "FAIL",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return results

    # Step 5: Evaluate log probability at initial point
    print("[5/5] Evaluating log probability at initial point...")
    try:
        with model:
            # Evaluate at initial point
            logp_value = logp_fn(initial_point)

            print(f"  - Log probability: {logp_value}")

            is_finite = np.isfinite(logp_value)

            if not is_finite:
                print(f"[WARN] Log probability is not finite!")
                results["checks"]["logp_evaluation"] = {
                    "status": "WARN",
                    "logp_value": str(logp_value),
                    "is_finite": False,
                }
            else:
                print(f"[OK] Log probability is finite")
                results["checks"]["logp_evaluation"] = {
                    "status": "PASS",
                    "logp_value": float(logp_value),
                    "is_finite": True,
                }
    except Exception as e:
        print(f"[FAIL] Evaluating logp failed: {e}")
        results["checks"]["logp_evaluation"] = {
            "status": "FAIL",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return results

    print()
    print("=" * 60)

    # Determine overall status
    all_passed = all(
        check.get("status") == "PASS" for check in results["checks"].values()
    )

    if all_passed:
        print("[SUCCESS] All initialization checks passed")
        results["overall_status"] = "PASS"
    else:
        has_failures = any(
            check.get("status") == "FAIL" for check in results["checks"].values()
        )
        if has_failures:
            print("[FAIL] Some initialization checks failed")
            results["overall_status"] = "FAIL"
        else:
            print("[WARN] Some initialization checks have warnings")
            results["overall_status"] = "WARN"

    print("=" * 60)
    print()

    return results


if __name__ == "__main__":
    results = test_initialization()

    # Save results
    output_file = Path(__file__).parent / "results_step_3_test_initialization.json"
    print(f"Saving results to {output_file.name}...")

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Deep convert all values
    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [deep_convert(item) for item in d]
        else:
            return convert_for_json(d)

    results_converted = deep_convert(results)

    with open(output_file, "w") as f:
        json.dump(results_converted, f, indent=2)

    print("[OK] Results saved")
    print()

    # Exit with appropriate code
    overall_status = results.get("overall_status")
    if overall_status == "PASS":
        sys.exit(0)
    elif overall_status == "WARN":
        sys.exit(0)  # Warnings are OK for now
    else:
        sys.exit(1)
