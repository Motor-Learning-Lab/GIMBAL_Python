"""
Debug Step 4: Test Gradient Compilation

This script tests gradient compilation, which is where the crash likely occurs.
We'll try to compile the gradient function that NUTS needs for sampling.

This is expected to be where we encounter the infinite recursion / hang.
"""

import sys
import json
from pathlib import Path
import signal
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add test utilities path
sys.path.insert(0, str(project_root / "tests" / "diagnostics" / "v0_2_1_divergence"))

import numpy as np
import pymc as pm
import pytensor
from test_utils import get_standard_synth_data, build_test_model


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Operation timed out")


def test_gradient_compilation():
    """Test if we can compile gradient functions."""

    print("=" * 60)
    print("DEBUG STEP 4: Test Gradient Compilation")
    print("=" * 60)
    print()

    results = {
        "step": 4,
        "description": "Test gradient compilation for NUTS sampler",
        "checks": {},
    }

    # Step 1: Build model
    print("[1/3] Building model...")
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
        results["checks"]["model_build"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Model building failed: {e}")
        results["checks"]["model_build"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 2: Get initial point
    print("[2/3] Getting initial point...")
    try:
        with model:
            initial_point = model.initial_point()

        print("[OK] Initial point obtained")
        results["checks"]["initial_point"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Getting initial point failed: {e}")
        results["checks"]["initial_point"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 3: Try to compile gradient function (THIS IS WHERE CRASH OCCURS)
    print("[3/3] Compiling gradient function...")
    print("  NOTE: This may hang or crash. Will timeout after 30 seconds.")
    print()

    try:
        with model:
            # This is what pm.sample does when initializing NUTS
            print("  - Attempting to compile dlogp function...")
            start_time = time.time()

            # Try to compile gradients
            # This is where PyTensor shape inference goes into infinite recursion
            try:
                dlogp_fn = model.compile_dlogp()
                compile_time = time.time() - start_time

                print(f"  - Compilation succeeded in {compile_time:.2f}s")
                print("[OK] Gradient function compiled")

                results["checks"]["gradient_compilation"] = {
                    "status": "PASS",
                    "compile_time": compile_time,
                }

                # Try to evaluate gradients
                print("  - Attempting to evaluate gradients...")
                grad_start = time.time()
                gradients = dlogp_fn(initial_point)
                grad_time = time.time() - grad_start

                print(f"  - Gradient evaluation succeeded in {grad_time:.2f}s")
                print(f"  - Number of gradient arrays: {len(gradients)}")

                # Check for NaN/inf in gradients
                has_nan = False
                has_inf = False
                for i, grad in enumerate(gradients):
                    if isinstance(grad, np.ndarray):
                        if np.isnan(grad).any():
                            has_nan = True
                        if np.isinf(grad).any():
                            has_inf = True

                if has_nan or has_inf:
                    print(f"  [WARN] Gradients contain NaN or inf")
                    results["checks"]["gradient_evaluation"] = {
                        "status": "WARN",
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                    }
                else:
                    print(f"  [OK] All gradients are finite")
                    results["checks"]["gradient_evaluation"] = {
                        "status": "PASS",
                        "has_nan": False,
                        "has_inf": False,
                    }

            except KeyboardInterrupt:
                print()
                print("[FAIL] Gradient compilation interrupted by user")
                results["checks"]["gradient_compilation"] = {
                    "status": "FAIL",
                    "error": "Interrupted by user (likely hung)",
                }
                return results
            except TimeoutError:
                print()
                print("[FAIL] Gradient compilation timed out")
                results["checks"]["gradient_compilation"] = {
                    "status": "FAIL",
                    "error": "Timed out after 30 seconds",
                }
                return results
            except Exception as e:
                print(f"[FAIL] Gradient compilation failed: {e}")
                print(f"  Error type: {type(e).__name__}")
                results["checks"]["gradient_compilation"] = {
                    "status": "FAIL",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                return results

    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        results["checks"]["gradient_compilation"] = {
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
        print("[SUCCESS] All gradient compilation checks passed")
        results["overall_status"] = "PASS"
    else:
        has_failures = any(
            check.get("status") == "FAIL" for check in results["checks"].values()
        )
        if has_failures:
            print("[FAIL] Gradient compilation failed")
            results["overall_status"] = "FAIL"
        else:
            print("[WARN] Some gradient checks have warnings")
            results["overall_status"] = "WARN"

    print("=" * 60)
    print()

    return results


if __name__ == "__main__":
    results = test_gradient_compilation()

    # Save results
    output_file = Path(__file__).parent / "results_step_4_test_gradients.json"
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
