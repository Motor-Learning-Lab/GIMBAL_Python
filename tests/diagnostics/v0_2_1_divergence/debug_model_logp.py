"""
Debug Model Log Probabilities - Step 1

Comprehensive debugging function to identify which parameter(s) cause -inf
log probability in the GIMBAL v0.2.1 model.

Uses PyMC's model.debug() to systematically check each parameter's:
- Value at test point
- Log probability contribution
- Gradient/derivative (if available)
"""

import numpy as np
import pymc as pm
import sys
from pathlib import Path
from datetime import datetime
from io import StringIO
import pandas as pd

# Add repository root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from test_utils import build_test_model
from extract_ground_truth import (
    extract_complete_ground_truth,
    transform_to_unconstrained_space,
    verify_ground_truth_coverage,
)


def debug_model_logp_by_parameter(model, test_point=None, verbose=False):
    """
    Debug model by checking log probability of each parameter individually.

    Parameters
    ----------
    model : pm.Model
        PyMC model to debug
    test_point : dict, optional
        Specific point to test. If None, uses model's default initialization
    verbose : bool
        Whether to show verbose PyTensor output

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns: parameter, value_shape, logp, is_finite, notes
    """
    print("=" * 80)
    print("Debugging Model Log Probabilities by Parameter")
    print("=" * 80)
    print()

    # Get test point
    if test_point is None:
        print("Using model's default initialization point...")
        with model:
            test_point = model.initial_point()
    else:
        # Filter test_point to only include model parameters
        with model:
            model_params = {rv.name for rv in model.free_RVs}
        test_point = {k: v for k, v in test_point.items() if k in model_params}

    print(f"Test point has {len(test_point)} parameters")
    print()

    # Check overall log probability first
    print("Checking overall log probability...")
    with model:
        try:
            total_logp = model.compile_logp()(test_point)
            print(f"  Total log probability: {total_logp}")
            if np.isfinite(total_logp):
                print("  [OK] Total logp is FINITE")
            else:
                print("  [FAIL] Total logp is INFINITE - investigating parameters...")
            print()
        except Exception as e:
            print(f"  [ERROR] Could not compute total logp: {e}")
            print()
            total_logp = np.nan

    # Debug each parameter's contribution
    print("Checking log probability contribution of each parameter...")
    print()

    results = []

    with model:
        # Get all random variables
        rvs = model.free_RVs

        for i, rv in enumerate(rvs, 1):
            rv_name = rv.name
            print(f"[{i}/{len(rvs)}] {rv_name}")
            print("-" * 60)

            result = {
                "parameter": rv_name,
                "value_shape": None,
                "value_min": None,
                "value_max": None,
                "value_mean": None,
                "logp": None,
                "is_finite": None,
                "has_nan": None,
                "has_inf": None,
                "notes": [],
            }

            # Get parameter value
            if rv_name in test_point:
                val = test_point[rv_name]
                result["value_shape"] = val.shape if hasattr(val, "shape") else "scalar"

                # Check for problematic values
                if np.isscalar(val):
                    result["value_min"] = val
                    result["value_max"] = val
                    result["value_mean"] = val
                    result["has_nan"] = np.isnan(val)
                    result["has_inf"] = np.isinf(val)
                else:
                    result["value_min"] = np.min(val)
                    result["value_max"] = np.max(val)
                    result["value_mean"] = np.mean(val)
                    result["has_nan"] = np.any(np.isnan(val))
                    result["has_inf"] = np.any(np.isinf(val))

                print(f"  Value shape: {result['value_shape']}")
                print(
                    f"  Value range: [{result['value_min']:.4f}, {result['value_max']:.4f}]"
                )
                print(f"  Value mean:  {result['value_mean']:.4f}")

                if result["has_nan"]:
                    print("  [WARNING] Value contains NaN!")
                    result["notes"].append("contains_nan")
                if result["has_inf"]:
                    print("  [WARNING] Value contains inf!")
                    result["notes"].append("contains_inf")
            else:
                print(f"  [WARNING] Parameter not in test point!")
                result["notes"].append("not_in_test_point")

            # Try to compute log probability for this variable alone
            try:
                # Use PyMC's debug method for this specific variable
                # We'll capture the output
                old_stdout = sys.stdout
                mystdout = sys.stdout = StringIO()

                model.debug(point=test_point, fn="logp", verbose=verbose)

                output = mystdout.getvalue()
                sys.stdout = old_stdout

                # Extract logp for this variable from output (if possible)
                # For now, compute it directly
                logp_fn = model.compile_logp(vars=[rv], sum=True)
                logp_val = logp_fn(test_point)
                result["logp"] = logp_val
                result["is_finite"] = np.isfinite(logp_val)

                print(f"  Log probability: {logp_val:.4f}")

                if result["is_finite"]:
                    print("  [OK] Log probability is FINITE")
                else:
                    print("  [FAIL] Log probability is INFINITE")
                    result["notes"].append("infinite_logp")

            except Exception as e:
                print(f"  [ERROR] Could not compute logp: {e}")
                result["logp"] = np.nan
                result["is_finite"] = False
                result["notes"].append(f"error: {str(e)[:50]}")

            print()
            results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df["notes"] = df["notes"].apply(lambda x: ", ".join(x) if x else "")

    return df, total_logp


def debug_with_model_debug_method(model, test_point=None):
    """
    Use PyMC's built-in model.debug() method to diagnose issues.

    This will show which parameters have issues and provide detailed
    PyTensor graph information.
    """
    print("=" * 80)
    print("Using PyMC's model.debug() Method")
    print("=" * 80)
    print()

    if test_point is None:
        print("Using model's default initialization point...")
        with model:
            test_point = model.initial_point()
    else:
        # Filter test_point to only include model parameters
        with model:
            model_params = {rv.name for rv in model.free_RVs}
        test_point = {k: v for k, v in test_point.items() if k in model_params}

    print("Running model.debug() for log probability...")
    print()

    with model:
        model.debug(point=test_point, fn="logp", verbose=True)

    print()
    print("If you see any ERROR or FAIL messages above, those parameters")
    print("are causing the -inf log probability.")
    print()


def check_prior_specifications(model):
    """
    Check the prior specifications for all parameters to identify
    potential issues (e.g., unbounded priors, priors that allow zeros, etc.)
    """
    print("=" * 80)
    print("Checking Prior Specifications")
    print("=" * 80)
    print()

    with model:
        for rv in model.free_RVs:
            print(f"{rv.name}:")
            print(f"  Type: {type(rv.owner.op).__name__}")

            # Try to get distribution info
            if hasattr(rv.owner.op, "dist"):
                dist = rv.owner.op.dist
                print(f"  Distribution: {type(dist).__name__}")

                # Check for transformation
                if hasattr(rv.owner.op, "transform"):
                    transform = rv.owner.op.transform
                    if transform is not None:
                        print(f"  Transform: {type(transform).__name__}")

            # Check shape
            try:
                shape = rv.eval().shape if hasattr(rv.eval(), "shape") else "scalar"
                print(f"  Shape: {shape}")
            except:
                print(f"  Shape: unknown")

            print()


def create_diagnostic_report(results_df, total_logp, output_dir):
    """
    Create a formatted report of the debugging results.
    """
    report_path = output_dir / "debug_logp_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Log Probability Debug Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Overall Results\n\n")
        f.write(f"- **Total log probability**: {total_logp}\n")
        if np.isfinite(total_logp):
            f.write("- **Status**: [OK] FINITE\n\n")
        else:
            f.write("- **Status**: [FAIL] INFINITE\n\n")

        f.write(f"- **Parameters checked**: {len(results_df)}\n\n")

        # Count issues
        infinite_logp = results_df[~results_df["is_finite"]]
        has_nan = results_df[results_df["has_nan"] == True]
        has_inf = results_df[results_df["has_inf"] == True]

        f.write("### Issues Found\n\n")
        f.write(f"- **Parameters with infinite logp**: {len(infinite_logp)}\n")
        f.write(f"- **Parameters with NaN values**: {len(has_nan)}\n")
        f.write(f"- **Parameters with inf values**: {len(has_inf)}\n\n")

        if len(infinite_logp) > 0:
            f.write("## Parameters with Infinite Log Probability\n\n")
            f.write("These parameters are causing the -inf total log probability:\n\n")
            f.write("| Parameter | Value Range | Value Mean | Notes |\n")
            f.write("|-----------|-------------|------------|-------|\n")
            for _, row in infinite_logp.iterrows():
                f.write(f"| `{row['parameter']}` | ")
                f.write(f"[{row['value_min']:.4f}, {row['value_max']:.4f}] | ")
                f.write(f"{row['value_mean']:.4f} | ")
                f.write(f"{row['notes']} |\n")
            f.write("\n")

        if len(has_inf) > 0:
            f.write("## Parameters with Infinite Values\n\n")
            f.write("These parameters have inf values at the test point:\n\n")
            for _, row in has_inf.iterrows():
                f.write(f"- **`{row['parameter']}`**: {row['notes']}\n")
            f.write("\n")

        if len(has_nan) > 0:
            f.write("## Parameters with NaN Values\n\n")
            f.write("These parameters have NaN values at the test point:\n\n")
            for _, row in has_nan.iterrows():
                f.write(f"- **`{row['parameter']}`**: {row['notes']}\n")
            f.write("\n")

        f.write("## All Parameters\n\n")
        f.write("Complete list of all parameters and their log probabilities:\n\n")
        f.write("| Parameter | Shape | Log P | Finite | Value Range | Notes |\n")
        f.write("|-----------|-------|-------|--------|-------------|-------|\n")
        for _, row in results_df.iterrows():
            status = "[OK]" if row["is_finite"] else "[FAIL]"
            f.write(f"| `{row['parameter']}` | ")
            f.write(f"{row['value_shape']} | ")
            f.write(f"{row['logp']:.2f} | ")
            f.write(f"{status} | ")
            f.write(f"[{row['value_min']:.4f}, {row['value_max']:.4f}] | ")
            f.write(f"{row['notes']} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        if len(infinite_logp) > 0:
            f.write("### Investigate These Parameters\n\n")
            for param in infinite_logp["parameter"].values:
                f.write(f"1. **`{param}`**\n")
                f.write(f"   - Check prior specification in model code\n")
                f.write(f"   - Verify transformation (log, logit, etc.) is correct\n")
                f.write(f"   - Ensure parameter values are in valid range\n")
                f.write(f"   - Look for boundary conditions (0, 1, etc.)\n\n")
        else:
            f.write("All parameters have finite log probabilities individually.\n")
            f.write("The issue may be in the interaction between parameters or ")
            f.write("in the observed data likelihood.\n\n")

    print(f"[OK] Debug report saved to {report_path}")


def run_comprehensive_debug():
    """
    Main function: Run complete debugging workflow on ground truth data.
    """
    print("\n")
    print("=" * 80)
    print("COMPREHENSIVE MODEL DEBUG - Step 1")
    print("Per-Parameter Log Probability Analysis")
    print("=" * 80)
    print("\n")

    # Setup
    output_dir = Path(__file__).parent / "debug_outputs" / "model_logp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    T = 50
    S = 2
    C = 4

    print(f"Configuration: T={T}, S={S}, C={C}")
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    config = SyntheticDataConfig(
        T=T,
        C=C,
        S=S,
        kappa=10.0,
        obs_noise_std=0.5,
        occlusion_rate=0.05,
        random_seed=42,
    )
    synth_data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    print("[OK] Data generated")
    print()

    # Prepare data dictionary
    data_dict = {
        "observations_uv": synth_data.y_observed,
        "camera_matrices": synth_data.camera_proj,
        "joint_positions": synth_data.x_true,
        "joint_names": DEMO_V0_1_SKELETON.joint_names,
        "parents": DEMO_V0_1_SKELETON.parents,
        "bone_lengths": DEMO_V0_1_SKELETON.bone_lengths,
        "true_states": synth_data.true_states,
        "config": config,
    }

    # Build model
    print("Building PyMC model...")
    model = build_test_model(
        synth_data=data_dict,
        use_directional_hmm=True,
        S=S,
    )
    print("[OK] Model built")
    print()

    # Initialize ground truth variables
    results_df_gt = None
    total_logp_gt = None
    csv_path_gt = None

    # Extract ground truth parameters
    print("=" * 80)
    print("EXTRACTING GROUND TRUTH PARAMETERS")
    print("=" * 80)
    print()

    gt_constrained = extract_complete_ground_truth(synth_data, DEMO_V0_1_SKELETON, S=S)
    gt_unconstrained = transform_to_unconstrained_space(gt_constrained, model)
    is_complete, missing, extra = verify_ground_truth_coverage(gt_unconstrained, model)

    if not is_complete:
        print(f"[WARNING] Ground truth missing {len(missing)} parameters")
        print("Will only test default initialization")
        test_ground_truth = False
    else:
        print("[OK] Ground truth complete - will test both scenarios")
        test_ground_truth = True
    print()

    # SCENARIO A: Test at ground truth (most stringent test)
    if test_ground_truth:
        print("\n")
        print("=" * 80)
        print("SCENARIO A: Testing at GROUND TRUTH Parameters")
        print("=" * 80)
        print()
        print("This is the most stringent test - the model should give finite logp")
        print("at its own generating parameters.")
        print()

        # Filter ground truth to only model parameters
        with model:
            model_params = {rv.name for rv in model.free_RVs}
        gt_filtered = {k: v for k, v in gt_unconstrained.items() if k in model_params}

        results_df_gt, total_logp_gt = debug_model_logp_by_parameter(
            model, test_point=gt_filtered, verbose=False
        )

        csv_path_gt = output_dir / "parameter_logp_ground_truth.csv"
        results_df_gt.to_csv(csv_path_gt, index=False)
        print(f"[OK] Results saved to {csv_path_gt}")

        if not np.isfinite(total_logp_gt):
            print(f"\n[CRITICAL] Ground truth gives -inf log probability!")
            print(f"Total logp: {total_logp_gt}")

            infinite_params = results_df_gt[~results_df_gt["is_finite"]]
            if len(infinite_params) > 0:
                print(f"\nCulprit parameters:")
                for param in infinite_params["parameter"].values:
                    print(f"  - {param}")
        else:
            print(f"\n[OK] Ground truth gives finite logp: {total_logp_gt}")
        print()

    # SCENARIO B: Test at default initialization
    print("\n")
    print("=" * 80)
    print("SCENARIO B: Testing at DEFAULT Initialization")
    print("=" * 80)
    print()
    print("Testing whether default PyMC initialization produces finite logp.")
    print()

    results_df, total_logp = debug_model_logp_by_parameter(model, verbose=False)

    # Save results
    csv_path = output_dir / "parameter_logp_default.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Results saved to {csv_path}")
    print()

    # STEP 2: Use PyMC's debug method (on default init)
    # NOTE: Skipping model.debug() due to PyTensor iteration issue
    # The per-parameter analysis above is more informative anyway
    print("\n")
    print("STEP 2: PyMC's model.debug() skipped (technical issue)")
    print("=" * 80)
    print()
    print("[INFO] Skipping model.debug() - per-parameter analysis above is sufficient")
    print()

    # STEP 3: Check prior specifications
    print("\n")
    print("STEP 3: Checking prior specifications")
    print("=" * 80)
    print()

    check_prior_specifications(model)

    # Create report
    print("\n")
    print("=" * 80)
    print("Creating diagnostic report...")
    print("=" * 80)
    print()

    create_diagnostic_report(results_df, total_logp, output_dir)

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if test_ground_truth and results_df_gt is not None:
        print("GROUND TRUTH Test:")
        print(f"  Total log probability: {total_logp_gt}")
        infinite_params_gt = results_df_gt[~results_df_gt["is_finite"]]
        if len(infinite_params_gt) > 0:
            print(f"  [FAIL] {len(infinite_params_gt)} parameters with -inf logp:")
            for param in infinite_params_gt["parameter"].values[:5]:
                print(f"    - {param}")
        else:
            print(f"  [OK] All parameters finite at ground truth!")
        print()

    print("DEFAULT INITIALIZATION Test:")
    print(f"  Total log probability: {total_logp}")
    infinite_params = results_df[~results_df["is_finite"]]
    if len(infinite_params) > 0:
        print(f"  [FAIL] {len(infinite_params)} parameters with -inf logp:")
        for param in infinite_params["parameter"].values[:5]:
            print(f"    - {param}")
    else:
        print(f"  [INTERESTING] All parameters finite individually!")
        print(f"  Issue may be in parameter interactions or likelihood.")

    print()
    print("Output files:")
    if test_ground_truth:
        print(f"  - {csv_path_gt}")
    print(f"  - {csv_path}")
    print(f"  - {output_dir / 'debug_logp_report.md'}")
    print()


if __name__ == "__main__":
    # Note: This will generate a lot of output!
    # Consider redirecting to a file or running in a notebook

    run_comprehensive_debug()
