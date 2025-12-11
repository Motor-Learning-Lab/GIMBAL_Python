"""
Debug Step 5: Test Actual Sampling

Now that we know gradients compile (albeit slowly), let's test if pm.sample()
actually works end-to-end with a small number of samples.

Expected outcome: Should complete successfully, possibly with divergences.
"""

import sys
import json
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add test utilities path
sys.path.insert(0, str(project_root / "tests" / "diagnostics" / "v0_2_1_divergence"))

import numpy as np
import pymc as pm
from test_utils import get_standard_synth_data, build_test_model


def test_sampling():
    """Test if we can actually sample from the model."""

    print("=" * 60)
    print("DEBUG STEP 5: Test Actual Sampling")
    print("=" * 60)
    print()

    results = {
        "step": 5,
        "description": "Test if pm.sample() completes successfully",
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

    # Step 2: Sample from model
    print("[2/3] Sampling from model...")
    print("  Configuration:")
    print("    - draws: 20")
    print("    - tune: 100")
    print("    - chains: 1")
    print("    - random_seed: 42")
    print()
    print("  NOTE: Initial gradient compilation takes ~45 seconds.")
    print("        Please be patient...")
    print()

    try:
        start_time = time.time()

        with model:
            trace = pm.sample(
                draws=20,
                tune=100,
                chains=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=True,
            )

        total_time = time.time() - start_time

        print()
        print(f"[OK] Sampling completed in {total_time:.1f}s")

        # Get sampling statistics
        n_draws = len(trace.posterior.draw)
        n_chains = len(trace.posterior.chain)

        # Check for divergences
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = trace.sample_stats.diverging.values
            n_divergences = int(divergences.sum())
            divergence_rate = n_divergences / (n_draws * n_chains)
        else:
            n_divergences = 0
            divergence_rate = 0.0

        print(f"  - Total draws: {n_draws}")
        print(f"  - Chains: {n_chains}")
        print(f"  - Divergences: {n_divergences} ({divergence_rate*100:.1f}%)")

        results["checks"]["sampling"] = {
            "status": "PASS",
            "total_time": total_time,
            "n_draws": int(n_draws),
            "n_chains": int(n_chains),
            "n_divergences": n_divergences,
            "divergence_rate": float(divergence_rate),
        }

    except Exception as e:
        print(f"[FAIL] Sampling failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        results["checks"]["sampling"] = {
            "status": "FAIL",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return results

    # Step 3: Check trace quality
    print("[3/3] Checking trace quality...")
    try:
        # Get summary statistics for a few key parameters
        summary = pm.summary(
            trace, var_names=["eta2_root", "obs_sigma", "logodds_inlier"]
        )

        print("  Sample parameter statistics:")
        print(summary[["mean", "sd", "r_hat"]].to_string())

        # Check for problematic r_hat values
        max_rhat = summary["r_hat"].max()
        if max_rhat > 1.1:
            print(f"  [WARN] High R-hat detected: {max_rhat:.3f}")
            results["checks"]["trace_quality"] = {
                "status": "WARN",
                "max_rhat": float(max_rhat),
                "message": "Some parameters have high R-hat (> 1.1)",
            }
        else:
            print(f"  [OK] All R-hat values < 1.1")
            results["checks"]["trace_quality"] = {
                "status": "PASS",
                "max_rhat": float(max_rhat),
            }

    except Exception as e:
        print(f"[FAIL] Trace quality check failed: {e}")
        results["checks"]["trace_quality"] = {"status": "FAIL", "error": str(e)}
        return results

    print()
    print("=" * 60)

    # Determine overall status
    sampling_status = results["checks"]["sampling"]["status"]

    if sampling_status == "PASS":
        divergence_rate = results["checks"]["sampling"]["divergence_rate"]
        if divergence_rate > 0.5:
            print(f"[WARN] High divergence rate: {divergence_rate*100:.1f}%")
            print("       This is expected for Test Group 1 (no HMM)")
            results["overall_status"] = "PASS_WITH_DIVERGENCES"
        else:
            print("[SUCCESS] Sampling completed successfully")
            results["overall_status"] = "PASS"
    else:
        print("[FAIL] Sampling failed")
        results["overall_status"] = "FAIL"

    print("=" * 60)
    print()

    return results


if __name__ == "__main__":
    results = test_sampling()

    # Save results
    output_file = Path(__file__).parent / "results_step_5_test_sampling.json"
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
    if overall_status in ["PASS", "PASS_WITH_DIVERGENCES"]:
        sys.exit(0)
    else:
        sys.exit(1)
