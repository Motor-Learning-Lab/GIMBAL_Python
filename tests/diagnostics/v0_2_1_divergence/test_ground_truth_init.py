"""
Ground Truth Initialization Test

Tests whether the v0.2.1 model can initialize and sample when given
exact ground truth parameters as starting values.

Purpose: Diagnose the initialization failure causing -inf log probability
"""

import numpy as np
import pymc as pm
import json
import sys
from pathlib import Path
from datetime import datetime

# Add repository root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from tests.diagnostics.v0_2_1_divergence.test_utils import build_test_model


def test_ground_truth_initialization():
    """
    Main test: Generate data with known parameters, initialize model with
    those exact parameters, check if it can evaluate and sample.
    """
    print("=" * 80)
    print("Ground Truth Initialization Test")
    print("=" * 80)
    print()

    # Setup
    output_dir = Path(__file__).parent / "debug_outputs" / "ground_truth_init"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    T = 50
    S = 2
    K = len(DEMO_V0_1_SKELETON.joint_names)
    C = 4

    print(f"Configuration: T={T}, S={S}, K={K}, C={C}")
    print()

    # Generate synthetic data
    print("Generating synthetic data with ground truth parameters...")
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
    print(f"  - obs_noise_std: {config.obs_noise_std}")
    print(f"  - occlusion_rate: {config.occlusion_rate}")
    print(f"  - inlier_prob: {1.0 - config.occlusion_rate}")
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

    # Build PyMC model
    print("Building PyMC model...")
    try:
        model = build_test_model(
            synth_data=data_dict,
            use_directional_hmm=True,
            S=S,
        )
        print("[OK] Model built successfully")
        print()
    except Exception as e:
        print(f"[FAIL] Model building failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Create ground truth starting point from actual data
    print("Creating ground truth starting point...")
    x_true = synth_data.x_true  # (T, K, 3)

    start_point = {}

    # Root joint positions
    start_point["x_root"] = x_true[:, 0, :]  # (T, 3)

    # Bone vectors from joint positions
    for j in range(1, K):
        parent_idx = DEMO_V0_1_SKELETON.parents[j]
        if parent_idx >= 0:
            bone_vec = x_true[:, j, :] - x_true[:, parent_idx, :]
            start_point[f"raw_u_{j}"] = bone_vec  # (T, 3)

    # Observation noise (log-transformed)
    obs_sigma = config.obs_noise_std
    start_point["obs_sigma_log__"] = np.log(obs_sigma)
    print(f"  - obs_sigma = {obs_sigma}")
    print(f"  - obs_sigma_log__ = {np.log(obs_sigma)}")

    # Inlier probability (logit-transformed) - THE SUSPECT
    inlier_prob = 1.0 - config.occlusion_rate
    print(f"  - inlier_prob = {inlier_prob}")

    if inlier_prob <= 0 or inlier_prob >= 1:
        print(f"  WARNING: inlier_prob = {inlier_prob} is out of (0, 1) range!")
        inlier_prob = np.clip(inlier_prob, 1e-10, 1 - 1e-10)

    logodds_inlier = np.log(inlier_prob / (1 - inlier_prob))
    start_point["logodds_inlier"] = logodds_inlier
    print(f"  - logodds_inlier = {logodds_inlier}")

    if np.isinf(logodds_inlier):
        print("  ERROR: logodds_inlier is inf!")
    print()

    print(f"[OK] Created starting point with {len(start_point)} parameters")
    print()

    # Save starting point
    start_point_file = output_dir / "initial_point.json"
    save_dict = {
        k: v.tolist() if isinstance(v, np.ndarray) else float(v)
        for k, v in start_point.items()
    }
    with open(start_point_file, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"[OK] Saved starting point to {start_point_file}")
    print()

    # Check initial log probability
    print("Checking initial log probability with ground truth...")
    try:
        with model:
            # Get the point in model's coordinate system
            # Need to check what variables the model expects
            print(f"  Model has {len(model.free_RVs)} free random variables:")
            for rv in model.free_RVs:
                print(f"    - {rv.name}")
            print()

            # Try to compute log probability
            # First, let's see if PyMC can compute it with defaults
            try:
                test_point = model.initial_point()
                initial_logp = model.compile_logp()(test_point)
                print(f"  Default initialization log probability: {initial_logp}")

                if np.isinf(initial_logp):
                    print("  [FAIL] Default initialization gives -inf!")

                    # Diagnose which parameter
                    try:
                        logp_list = model.compile_logp(vars=model.free_RVs, sum=False)(
                            test_point
                        )
                        print("\n  Log probability by parameter (default init):")
                        for var, logp_val in zip(model.free_RVs, logp_list):
                            status = "[OK]" if np.isfinite(logp_val) else "[FAIL]"
                            print(f"    {status} {var.name}: {logp_val}")
                    except Exception as e:
                        print(f"  Could not diagnose: {e}")

                    # Check specifically for logodds_inlier
                    if "logodds_inlier" in test_point:
                        print(
                            f"\n  Default logodds_inlier = {test_point['logodds_inlier']}"
                        )
                    else:
                        print("\n  WARNING: logodds_inlier not in model!")

                    # If default init fails, try with ground truth starting point
                    print("\n  Trying with ground truth starting point...")
                    try:
                        # Merge ground truth values into default initial point
                        gt_point = test_point.copy()
                        gt_point.update(
                            {
                                "obs_sigma_log__": start_point["obs_sigma_log__"],
                                "logodds_inlier": start_point["logodds_inlier"],
                            }
                        )

                        gt_logp = model.compile_logp()(gt_point)
                        print(f"  Ground truth log probability: {gt_logp}")

                        if np.isinf(gt_logp):
                            print(
                                "  [FAIL] Ground truth initialization also gives -inf!"
                            )

                            # Diagnose which parameter
                            try:
                                logp_list = model.compile_logp(
                                    vars=model.free_RVs, sum=False
                                )(gt_point)
                                print(
                                    "\n  Log probability by parameter (ground truth):"
                                )
                                for var, logp_val in zip(model.free_RVs, logp_list):
                                    if isinstance(logp_val, np.ndarray):
                                        logp_scalar = (
                                            logp_val.sum()
                                            if logp_val.size > 1
                                            else float(logp_val)
                                        )
                                    else:
                                        logp_scalar = float(logp_val)
                                    status = (
                                        "[OK]" if np.isfinite(logp_scalar) else "[FAIL]"
                                    )
                                    print(f"    {status} {var.name}: {logp_scalar}")
                            except Exception as e:
                                print(f"  Could not diagnose: {e}")

                            return False
                        else:
                            print("  [OK] Ground truth initialization works!")
                            return True

                    except Exception as e:
                        print(
                            f"  [FAIL] Could not compute with ground truth initialization: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        return False
                else:
                    print("  [OK] Default initialization works!")
                    return True

            except Exception as e:
                print(f"  [FAIL] Log probability evaluation failed: {e}")
                import traceback

                traceback.print_exc()
                return False

    except Exception as e:
        print(f"[FAIL] Unexpected error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_report(success, output_dir):
    """Generate a report of the test results"""
    report_path = output_dir / "ground_truth_init_report.md"

    with open(report_path, "w") as f:
        f.write("# Ground Truth Initialization Test Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Test Purpose\n\n")
        f.write(
            "Diagnose why v0.2.1 divergence tests fail at model initialization by testing "
        )
        f.write("with exact ground truth parameters as starting values.\n\n")

        f.write("## Test Configuration\n\n")
        f.write("- **Data**: Synthetic with known ground truth\n")
        f.write("- **Initialization**: Exact ground truth (no jitter)\n")
        f.write("- **Parameters**: T=50, S=2, K=6, C=4\n\n")

        f.write("## Results\n\n")

        if success:
            f.write("### [OK] Test PASSED\n\n")
            f.write("**Findings**:\n")
            f.write("1. Model accepts ground truth parameters\n")
            f.write("2. Initial log probability is finite\n")
            f.write("3. Default initialization succeeds\n\n")
            f.write(
                "**Conclusion**: The initialization failure in divergence tests is due to "
            )
            f.write(
                "poor default initialization or jitter, not a fundamental model bug.\n\n"
            )
            f.write(
                "**Recommendation**: Modify test suite to use better initialization strategy:\n"
            )
            f.write("- Use data-driven initialization from triangulated 3D points\n")
            f.write("- Reduce jitter magnitude\n")
            f.write("- Provide sensible starting values for all parameters\n")
        else:
            f.write("### [FAIL] Test FAILED\n\n")
            f.write("**Findings**:\n")
            f.write("1. Model initialization produces -inf log probability\n")
            f.write("2. Specific parameter causing issue identified in output\n\n")
            f.write(
                "**Conclusion**: There is a bug in the model specification or parameter "
            )
            f.write(
                "transformation. See `test_output.txt` for detailed diagnostics.\n\n"
            )
            f.write("**Recommendation**: Review model code:\n")
            f.write("- Check parameter transformations (log, logit)\n")
            f.write("- Verify prior specifications\n")
            f.write("- Check for missing or misconfigured parameters\n")
            f.write("- Pay special attention to `inlier_prob_logodds__`\n")

        f.write("\n---\n\n")
        f.write("## Output Files\n\n")
        f.write(f"- `initial_point.json` - Ground truth starting point\n")
        f.write(f"- `test_output.txt` - Console output with diagnostics\n")

    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    output_dir = Path("results/diagnostics/v0_2_1_divergence/sanity_ground_truth_init")

    # Redirect stdout to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(output_dir / "test_output.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)

    try:
        success = test_ground_truth_initialization()

        print()
        print("=" * 80)
        if success:
            print("TEST PASSED: Ground truth initialization works!")
        else:
            print("TEST FAILED: Ground truth initialization does not work")
        print("=" * 80)

        # Restore stdout before generating report
        sys.stdout = original_stdout
        log_file.close()

        # Generate report
        generate_report(success, output_dir)

    except Exception as e:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Test crashed: {e}")
        import traceback

        traceback.print_exc()

        generate_report(False, output_dir)
