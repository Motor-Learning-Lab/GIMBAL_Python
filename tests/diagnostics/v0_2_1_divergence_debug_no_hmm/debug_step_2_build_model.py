"""
Debug Step 2: Test Model Building in Isolation

This script attempts to build the PyMC model WITHOUT sampling to isolate
whether the issue is in model specification or during gradient compilation.

Expected outcome: Model should build successfully. The crash likely occurs
during gradient compilation when sampling is initiated.
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


def test_model_building():
    """Test if we can build the model without crashing."""

    print("=" * 60)
    print("DEBUG STEP 2: Test Model Building")
    print("=" * 60)
    print()

    results = {
        "step": 2,
        "description": "Test model building in isolation",
        "checks": {},
    }

    # Step 1: Generate synthetic data
    print("[1/4] Generating synthetic data...")
    try:
        T = 100
        C = 3
        S = 3
        seed = 42

        synth_data = get_standard_synth_data(T=T, C=C, S=S, seed=seed)

        observations_uv = synth_data["observations_uv"]
        camera_matrices = synth_data["camera_matrices"]
        joint_positions = synth_data["joint_positions"]
        parents = synth_data["parents"]

        print("[OK] Data generated")
        results["checks"]["data_generation"] = {
            "status": "PASS",
            "message": "Data generated successfully",
        }
    except Exception as e:
        print(f"[FAIL] Data generation failed: {e}")
        results["checks"]["data_generation"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 2: Build model with HMM OFF
    print("[2/4] Building model with use_directional_hmm=False...")
    try:
        model = build_test_model(
            synth_data=synth_data,
            use_directional_hmm=False,  # THIS IS THE KEY PARAMETER
            S=S,
            eta2_root_sigma=0.5,
            sigma2_sigma=0.2,
        )

        print("[OK] Model built successfully")
        results["checks"]["model_build"] = {
            "status": "PASS",
            "message": "Model built without errors",
        }
    except Exception as e:
        print(f"[FAIL] Model building failed: {e}")
        results["checks"]["model_build"] = {
            "status": "FAIL",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return results

    # Step 3: Inspect model structure
    print("[3/4] Inspecting model structure...")
    try:
        with model:
            # Count free random variables
            free_rvs = model.free_RVs
            n_free = len(free_rvs)

            print(f"  - Free RVs: {n_free}")
            print("  - Variables:")
            for rv in free_rvs:
                print(f"    - {rv.name}: shape={rv.type.shape}")

            results["checks"]["model_structure"] = {
                "status": "PASS",
                "n_free_rvs": n_free,
                "free_rvs": [rv.name for rv in free_rvs],
            }

        print("[OK] Model structure inspected")
    except Exception as e:
        print(f"[FAIL] Model inspection failed: {e}")
        results["checks"]["model_structure"] = {"status": "FAIL", "error": str(e)}
        return results

    # Step 4: Try to get initial point (test_point)
    print("[4/4] Getting initial test point...")
    try:
        with model:
            initial_point = model.initial_point()

            print(f"  - Got initial point with {len(initial_point)} variables")
            print("  - Sample values:")
            for key, val in list(initial_point.items())[:3]:
                if isinstance(val, np.ndarray):
                    print(
                        f"    - {key}: shape={val.shape}, range=[{val.min():.2f}, {val.max():.2f}]"
                    )
                else:
                    print(f"    - {key}: {val}")

            results["checks"]["initial_point"] = {
                "status": "PASS",
                "n_variables": len(initial_point),
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

    print()
    print("=" * 60)
    print("[SUCCESS] All model building checks passed")
    print("=" * 60)
    print()

    results["overall_status"] = "PASS"
    return results


if __name__ == "__main__":
    results = test_model_building()

    # Save results
    output_file = Path(__file__).parent / "results_step_2_build_model.json"
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
    if results.get("overall_status") == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)
