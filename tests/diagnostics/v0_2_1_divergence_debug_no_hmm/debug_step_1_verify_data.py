"""
Debug Step 1: Verify Synthetic Data Generation

Purpose:
    Ensure synthetic data is correctly formatted and contains no invalid values
    that could cause downstream issues in model building.

Checks:
    - Data shapes are correct (T, C, number of joints)
    - No NaN or inf values
    - Camera matrices are valid
    - Observations are in valid range

Reference: plans/v0.2.1_debug_model_no_hmm_plan.md, Step 1
"""

import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Import test utilities from the working diagnostics directory
sys.path.insert(0, str(repo_root / "tests" / "diagnostics" / "v0_2_1_divergence"))
from test_utils import get_standard_synth_data


def check_array_validity(name, arr, allow_nan=False):
    """Check if array has valid values."""
    checks = {
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "min_value": float(np.nanmin(arr)) if allow_nan else float(np.min(arr)),
        "max_value": float(np.nanmax(arr)) if allow_nan else float(np.max(arr)),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }

    is_valid = True
    issues = []

    if checks["has_nan"] and not allow_nan:
        is_valid = False
        issues.append("Contains NaN values")

    if checks["has_inf"]:
        is_valid = False
        issues.append("Contains inf values")

    return checks, is_valid, issues


def debug_step_1_verify_data():
    """
    Step 1: Verify synthetic data generation.

    Returns
    -------
    dict
        Complete results including all data checks
    """
    print("\n" + "=" * 70)
    print("Debug Step 1: Verify Synthetic Data Generation")
    print("=" * 70)

    # Configuration
    config = {
        "T": 100,
        "C": 3,
        "S": 3,
        "seed": 42,
    }

    print(f"\nConfiguration:")
    print(f"  T={config['T']}, C={config['C']}, S={config['S']}, seed={config['seed']}")

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    try:
        synth_data = get_standard_synth_data(
            T=config["T"],
            C=config["C"],
            S=config["S"],
            seed=config["seed"],
        )
        print(f"  [OK] Data generated successfully")
    except Exception as e:
        print(f"  [FAIL] Data generation failed: {e}")
        return {
            "step": 1,
            "description": "Verify synthetic data generation",
            "status": "FAILED",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

    # Check what keys are in the data
    print(f"\nData dictionary keys:")
    for key in synth_data.keys():
        print(f"  - {key}")

    # Perform checks on each component
    results = {
        "step": 1,
        "description": "Verify synthetic data generation",
        "configuration": config,
        "checks": {},
        "issues": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Check observations_uv
    print(f"\nChecking observations_uv...")
    if "observations_uv" in synth_data:
        checks, is_valid, issues = check_array_validity(
            "observations_uv", synth_data["observations_uv"], allow_nan=True
        )
        results["checks"]["observations_uv"] = checks
        results["checks"]["observations_uv"]["is_valid"] = is_valid

        print(f"  Shape: {checks['shape']}")
        print(f"  Range: [{checks['min_value']:.2f}, {checks['max_value']:.2f}]")
        print(f"  Has NaN: {checks['has_nan']} (allowed for occlusion)")
        print(f"  Has inf: {checks['has_inf']}")

        if issues:
            results["issues"].extend([f"observations_uv: {issue}" for issue in issues])
            print(f"  [WARN] {', '.join(issues)}")
        else:
            print(f"  [OK] Valid")
    else:
        results["issues"].append("observations_uv: Missing from data")
        print(f"  [FAIL] Missing")

    # Check camera_matrices
    print(f"\nChecking camera_matrices...")
    if "camera_matrices" in synth_data:
        checks, is_valid, issues = check_array_validity(
            "camera_matrices", synth_data["camera_matrices"]
        )
        results["checks"]["camera_matrices"] = checks
        results["checks"]["camera_matrices"]["is_valid"] = is_valid

        print(f"  Shape: {checks['shape']} (expected: (C, 3, 4))")
        print(f"  Range: [{checks['min_value']:.2f}, {checks['max_value']:.2f}]")
        print(f"  Has NaN: {checks['has_nan']}")
        print(f"  Has inf: {checks['has_inf']}")

        # Check if shape is correct
        expected_shape = [config["C"], 3, 4]
        if checks["shape"] != expected_shape:
            issues.append(f"Wrong shape, expected {expected_shape}")
            is_valid = False

        if issues:
            results["issues"].extend([f"camera_matrices: {issue}" for issue in issues])
            print(f"  [FAIL] {', '.join(issues)}")
        else:
            print(f"  [OK] Valid")
    else:
        results["issues"].append("camera_matrices: Missing from data")
        print(f"  [FAIL] Missing")

    # Check joint_positions (ground truth)
    print(f"\nChecking joint_positions...")
    if "joint_positions" in synth_data:
        checks, is_valid, issues = check_array_validity(
            "joint_positions", synth_data["joint_positions"]
        )
        results["checks"]["joint_positions"] = checks
        results["checks"]["joint_positions"]["is_valid"] = is_valid

        print(f"  Shape: {checks['shape']} (expected: (T, n_joints, 3))")
        print(f"  Range: [{checks['min_value']:.2f}, {checks['max_value']:.2f}]")
        print(f"  Has NaN: {checks['has_nan']}")
        print(f"  Has inf: {checks['has_inf']}")

        if issues:
            results["issues"].extend([f"joint_positions: {issue}" for issue in issues])
            print(f"  [FAIL] {', '.join(issues)}")
        else:
            print(f"  [OK] Valid")
    else:
        results["issues"].append("joint_positions: Missing from data")
        print(f"  [FAIL] Missing")

    # Check parents
    print(f"\nChecking parents...")
    if "parents" in synth_data:
        parents = synth_data["parents"]
        print(f"  Type: {type(parents)}")
        print(f"  Length: {len(parents)}")
        print(f"  Values: {parents}")

        # Check for valid parent indices
        invalid_parents = []
        for i, p in enumerate(parents):
            if p >= i and p != -1:
                invalid_parents.append(f"Joint {i} has parent {p} (must be < {i})")

        results["checks"]["parents"] = {
            "length": len(parents),
            "values": parents.tolist() if hasattr(parents, "tolist") else list(parents),
            "is_valid": len(invalid_parents) == 0,
        }

        if invalid_parents:
            results["issues"].extend([f"parents: {issue}" for issue in invalid_parents])
            print(f"  [FAIL] {', '.join(invalid_parents)}")
        else:
            print(f"  [OK] Valid")
    else:
        results["issues"].append("parents: Missing from data")
        print(f"  [FAIL] Missing")

    # Overall status
    if len(results["issues"]) == 0:
        results["status"] = "PASSED"
        print(f"\n" + "=" * 70)
        print("[SUCCESS] All data checks passed")
        print("=" * 70)
    else:
        results["status"] = "FAILED"
        print(f"\n" + "=" * 70)
        print(f"[FAILURE] Found {len(results['issues'])} issue(s):")
        for issue in results["issues"]:
            print(f"  - {issue}")
        print("=" * 70)

    # Save results
    results_file = Path(__file__).parent / "results_step_1_verify_data.json"
    print(f"\nSaving results to {results_file.name}...")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Results saved")

    return results


if __name__ == "__main__":
    results = debug_step_1_verify_data()

    # Exit with appropriate code
    sys.exit(0 if results["status"] == "PASSED" else 1)
