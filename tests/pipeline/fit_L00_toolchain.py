"""Full Pipeline Runner: fit_L00_toolchain.py

Executes all stages (A-J) in sequence for L00_minimal dataset validation.

This validates the complete v0.2.1 fitting toolchain:
- Data loading & validation
- 2D/3D cleaning
- Empirical prior building
- PyMC model construction
- MCMC sampling
- Posterior diagnostics
- Ground truth comparison

Outputs final report with pass/fail against Q7 criteria.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import stage modules
from tests.pipeline import (
    stage_a_load,
    stage_b_clean_2d,
    stage_c_triangulation,
    stage_d_clean_3d,
    stage_e_directions,
    stage_f_priors,
    stage_g_model_build,
    stage_h_fitting,
    stage_i_diagnostics,
    stage_j_ground_truth,
)


def run_full_pipeline():
    """Run all stages A-J in sequence."""

    print("=" * 80)
    print(" " * 20 + "GIMBAL v0.2.1 L00_minimal Fitting Toolchain")
    print("=" * 80)
    print()

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    fits_dir = base_dir / "tests" / "pipeline" / "fits"
    output_dir = fits_dir / "v0.2.1_L00_minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage results
    results = {}
    all_passed = True

    # Stage A: Load validation
    print("\n" + "=" * 80)
    print("Running Stage A: Load Validation")
    print("=" * 80)
    try:
        results["stage_a"] = stage_a_load.run_stage_a(dataset_dir, output_dir)
        stage_a_pass = results["stage_a"]["validation_summary"]["all_passed"]
        results["stage_a"]["passed"] = stage_a_pass
        all_passed = all_passed and stage_a_pass
    except Exception as e:
        print(f"\n✗ Stage A FAILED with exception: {e}")
        results["stage_a"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage B: 2D cleaning
    print("\n" + "=" * 80)
    print("Running Stage B: 2D Cleaning")
    print("=" * 80)
    try:
        results["stage_b"] = stage_b_clean_2d.run_stage_b(dataset_dir, output_dir)
        stage_b_pass = results["stage_b"]["quality_check"]["passed"]
        results["stage_b"]["passed"] = stage_b_pass
        all_passed = all_passed and stage_b_pass
    except Exception as e:
        print(f"\n✗ Stage B FAILED with exception: {e}")
        results["stage_b"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage C: Triangulation
    print("\n" + "=" * 80)
    print("Running Stage C: Triangulation")
    print("=" * 80)
    try:
        results["stage_c"] = stage_c_triangulation.run_stage_c(
            dataset_dir, output_dir, output_dir
        )
        stage_c_pass = results["stage_c"]["quality_check"]["passed"]
        results["stage_c"]["passed"] = stage_c_pass
        all_passed = all_passed and stage_c_pass
    except Exception as e:
        print(f"\n✗ Stage C FAILED with exception: {e}")
        results["stage_c"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage D: 3D cleaning
    print("\n" + "=" * 80)
    print("Running Stage D: 3D Cleaning")
    print("=" * 80)
    try:
        results["stage_d"] = stage_d_clean_3d.run_stage_d(
            dataset_dir, output_dir, output_dir
        )
        stage_d_pass = results["stage_d"]["quality_check"]["passed"]
        results["stage_d"]["passed"] = stage_d_pass
        all_passed = all_passed and stage_d_pass
    except Exception as e:
        print(f"\n✗ Stage D FAILED with exception: {e}")
        results["stage_d"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage E: Direction statistics
    print("\n" + "=" * 80)
    print("Running Stage E: Direction Statistics")
    print("=" * 80)
    try:
        results["stage_e"] = stage_e_directions.run_stage_e(
            dataset_dir, output_dir, output_dir
        )
        # Stage E always passes (warnings acceptable)
        results["stage_e"]["passed"] = True
    except Exception as e:
        print(f"\n✗ Stage E FAILED with exception: {e}")
        results["stage_e"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage F: Prior building
    print("\n" + "=" * 80)
    print("Running Stage F: Prior Building")
    print("=" * 80)
    try:
        results["stage_f"] = stage_f_priors.run_stage_f(
            dataset_dir, output_dir, output_dir
        )
        stage_f_pass = results["stage_f"]["validation"]["all_priors_valid"]
        results["stage_f"]["passed"] = stage_f_pass
        all_passed = all_passed and stage_f_pass
    except Exception as e:
        print(f"\n✗ Stage F FAILED with exception: {e}")
        results["stage_f"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage G: Model building
    print("\n" + "=" * 80)
    print("Running Stage G: Model Building")
    print("=" * 80)
    try:
        results["stage_g"] = stage_g_model_build.run_stage_g(
            dataset_dir, fits_dir, output_dir
        )
        stage_g_pass = results["stage_g"]["model_diagnostics"]["compilation_success"]
        results["stage_g"]["passed"] = stage_g_pass
        all_passed = all_passed and stage_g_pass
    except Exception as e:
        print(f"\n✗ Stage G FAILED with exception: {e}")
        results["stage_g"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage H: Fitting/sampling
    print("\n" + "=" * 80)
    print("Running Stage H: Fitting/Sampling")
    print("=" * 80)
    print("NOTE: This will take 5-10 minutes for MCMC sampling...")
    try:
        results["stage_h"] = stage_h_fitting.run_stage_h(
            dataset_dir, fits_dir, output_dir
        )
        stage_h_pass = (
            results["stage_h"]["criteria_Q7"]["rhat_pass"]
            and results["stage_h"]["criteria_Q7"]["divergence_pass"]
            and results["stage_h"]["criteria_Q7"]["ess_pass"]
        )
        results["stage_h"]["passed"] = stage_h_pass
        all_passed = all_passed and stage_h_pass
    except Exception as e:
        print(f"\n✗ Stage H FAILED with exception: {e}")
        results["stage_h"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage I: Posterior diagnostics
    print("\n" + "=" * 80)
    print("Running Stage I: Posterior Diagnostics")
    print("=" * 80)
    try:
        results["stage_i"] = stage_i_diagnostics.run_stage_i(
            dataset_dir, fits_dir, output_dir
        )
        stage_i_pass = results["stage_i"]["criteria_Q7"]["reprojection_pass"]
        results["stage_i"]["passed"] = stage_i_pass
        all_passed = all_passed and stage_i_pass
    except Exception as e:
        print(f"\n✗ Stage I FAILED with exception: {e}")
        results["stage_i"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    # Stage J: Ground truth comparison
    print("\n" + "=" * 80)
    print("Running Stage J: Ground Truth Comparison")
    print("=" * 80)
    try:
        results["stage_j"] = stage_j_ground_truth.run_stage_j(
            dataset_dir, fits_dir, output_dir
        )
        stage_j_pass = results["stage_j"]["criteria_Q7"]["bone_length_pass"]
        results["stage_j"]["passed"] = stage_j_pass
        all_passed = all_passed and stage_j_pass
    except Exception as e:
        print(f"\n✗ Stage J FAILED with exception: {e}")
        results["stage_j"] = {"passed": False, "error": str(e)}
        all_passed = False
        return results, all_passed

    return results, all_passed


def generate_final_report(results, all_passed, output_dir):
    """Generate markdown report summarizing all stages."""

    report_path = output_dir / "fit_report.md"

    with open(report_path, "w") as f:
        f.write("# GIMBAL v0.2.1 L00_minimal Fitting Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("## Executive Summary\n\n")

        if all_passed:
            f.write("✅ **ALL STAGES PASSED**\n\n")
        else:
            f.write("❌ **SOME STAGES FAILED**\n\n")

        f.write("## Stage Results\n\n")

        stage_names = {
            "stage_a": "Stage A: Load Validation",
            "stage_b": "Stage B: 2D Cleaning",
            "stage_c": "Stage C: Triangulation",
            "stage_d": "Stage D: 3D Cleaning",
            "stage_e": "Stage E: Direction Statistics",
            "stage_f": "Stage F: Prior Building",
            "stage_g": "Stage G: Model Building",
            "stage_h": "Stage H: Fitting/Sampling",
            "stage_i": "Stage I: Posterior Diagnostics",
            "stage_j": "Stage J: Ground Truth Comparison",
        }

        for stage_key, stage_name in stage_names.items():
            if stage_key in results:
                passed = results[stage_key].get("passed", False)
                status = "✅ PASSED" if passed else "❌ FAILED"
                f.write(f"### {stage_name}: {status}\n\n")

                # Add key metrics for each stage
                if stage_key == "stage_b" and "cleaning_metrics" in results[stage_key]:
                    rmse = results[stage_key]["cleaning_metrics"]["rmse_px"]
                    f.write(f"- **RMSE:** {rmse:.2f} px\n\n")

                elif stage_key == "stage_c" and "triangulation" in results[stage_key]:
                    success_rate = results[stage_key]["triangulation"]["success_rate"]
                    rmse = results[stage_key]["comparison"]["rmse_mm"]
                    f.write(f"- **Success Rate:** {success_rate:.1f}%\n")
                    f.write(f"- **RMSE vs GT:** {rmse:.4f} mm\n\n")

                elif stage_key == "stage_h" and "convergence" in results[stage_key]:
                    max_rhat = results[stage_key]["convergence"]["max_rhat"]
                    div = results[stage_key]["convergence"]["divergences"]
                    min_ess = results[stage_key]["convergence"]["min_ess"]
                    f.write(f"- **Max R-hat:** {max_rhat:.4f}\n")
                    f.write(f"- **Divergences:** {div}\n")
                    f.write(f"- **Min ESS:** {min_ess:.0f}\n\n")

                elif stage_key == "stage_i" and "reprojection" in results[stage_key]:
                    rmse = results[stage_key]["reprojection"]["rmse_px"]
                    f.write(f"- **Reprojection RMSE:** {rmse:.2f} px\n\n")

                elif stage_key == "stage_j" and "bone_lengths" in results[stage_key]:
                    max_error = results[stage_key]["bone_lengths"]["max_error_pct"]
                    f.write(f"- **Max Bone Length Error:** {max_error:.2f}%\n\n")
            else:
                f.write(f"### {stage_name}: ⚠️  NOT RUN\n\n")

        f.write("## Q7 Criteria Checklist\n\n")
        f.write("- [ ] Convergence: R-hat < 1.05\n")
        f.write("- [ ] No divergences\n")
        f.write("- [ ] Effective sample size > 100\n")
        f.write("- [ ] Reprojection RMSE < 5px\n")
        f.write("- [ ] Bone length error < 10%\n\n")

        f.write("---\n\n")
        f.write(f"**Output Directory:** `{output_dir}`\n")

    print(f"\n✓ Final report saved to {report_path}")

    return report_path


if __name__ == "__main__":
    # Run full pipeline
    results, all_passed = run_full_pipeline()

    # Save results JSON
    output_dir = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "pipeline"
        / "fits"
        / "v0.2.1_L00_minimal"
    )
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "all_passed": all_passed,
                "timestamp": datetime.now().isoformat(),
                "stages": results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Pipeline results saved to {results_path}")

    # Generate final report
    report_path = generate_final_report(results, all_passed, output_dir)

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print(" " * 20 + "✅ PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print(" " * 20 + "❌ PIPELINE COMPLETED WITH FAILURES")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)
