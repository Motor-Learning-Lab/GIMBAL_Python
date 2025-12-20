"""Group 1: Build-Only Sanity Check

Tests whether the model builds successfully and evaluates sensible
log-probabilities at the initial point.

Usage:
    pixi run python test_group_1_build_only_sanity.py --run_id local_run
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import pymc as pm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import gimbal
from _diag_utils import (
    make_paths,
    collect_environment,
    write_json,
    save_text,
    safe_point_logps,
    summarize_worst_terms,
    format_worst_terms_table,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Group 1: Build-Only Sanity Check")
    parser.add_argument(
        "--dataset_name", default="v0.2.1_L00_minimal", help="Dataset name"
    )
    parser.add_argument(
        "--max_T", type=int, default=None, help="Maximum number of time frames"
    )
    parser.add_argument(
        "--max_K", type=int, default=None, help="Maximum number of joints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_id", default=None, help="Run identifier (default: ISO timestamp)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    return parser.parse_args()


def load_dataset(dataset_name, max_T=None, max_K=None):
    """Load the dataset."""
    dataset_dir = project_root / "tests" / "pipeline" / "datasets" / dataset_name
    dataset_path = dataset_dir / "dataset.npz"

    with np.load(dataset_path, allow_pickle=True) as f:
        data = {
            "y_2d": f["y_2d"],
            "camera_proj": f["camera_proj"],
            "parents": f["parents"],
            "bone_lengths": f["bone_lengths"],
            "joint_names": [str(name) for name in f["joint_names"]],
        }

    # Subsample if requested
    if max_T is not None:
        data["y_2d"] = data["y_2d"][:, :max_T, :, :]
    if max_K is not None:
        data["y_2d"] = data["y_2d"][:, :, :max_K, :]
        data["parents"] = data["parents"][:max_K]
        data["joint_names"] = data["joint_names"][:max_K]

    return data, dataset_dir


def build_model(data):
    """Build the PyMC model with the failing configuration."""
    C, T, K, _ = data["y_2d"].shape

    # Initialize from triangulation
    from gimbal.fit_params import InitializationResult

    # Simple initialization (use mean of observed data as proxy)
    x_init = np.random.randn(T, K, 3) * 10 + 100  # Rough estimate
    rho_init = (
        data["bone_lengths"][: K - 1]
        if len(data["bone_lengths"]) >= K - 1
        else np.ones(K - 1) * 10
    )

    init_result = InitializationResult(
        x_init=x_init,
        eta2=np.ones(K),
        rho=rho_init,
        sigma2=rho_init * 0.01,
        u_init=np.zeros((T, K, 3)),
        obs_sigma=2.0,
        inlier_prob=0.95,
        metadata={"method": "simple_init"},
    )

    # Build model with failing configuration
    with pm.Model() as model:
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=True,
            image_size=(1280, 720),
            use_directional_hmm=False,
            validate_init_points=False,
        )

        # Extract variables
        U = model["U"]
        log_obs_t = model["log_obs_t"]

        # Add Stage 3 directional HMM with K=1
        gimbal.add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=1,  # K=1 HMM
            joint_names=data["joint_names"],
        )

    return model, init_result


def run_group_1(args):
    """Run Group 1 diagnostics."""
    # Setup paths
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_paths("group_1_build_only_sanity", args.run_id)

    # Initialize results
    results = {
        "group": "group_1_build_only_sanity",
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(),
        "environment": collect_environment(),
        "config": {
            "dataset_name": args.dataset_name,
            "max_T": args.max_T,
            "max_K": args.max_K,
            "seed": args.seed,
        },
        "success": False,
        "error": None,
    }

    try:
        # Set seed
        np.random.seed(args.seed)

        # Load dataset
        print(f"Loading dataset: {args.dataset_name}")
        data, dataset_dir = load_dataset(args.dataset_name, args.max_T, args.max_K)
        C, T, K, _ = data["y_2d"].shape
        print(f"  Cameras: {C}, Frames: {T}, Joints: {K}")

        results["data_shape"] = {"C": C, "T": T, "K": K}

        # Build model
        print("\nBuilding PyMC model...")
        model, init_result = build_model(data)
        print(f"  Model built successfully")
        print(f"  Free RVs: {len(model.free_RVs)}")
        print(f"  Total parameters: {sum(rv.size for rv in model.free_RVs)}")

        # Calculate total params carefully (handle TensorVariables)
        total_params = 0
        for rv in model.free_RVs:
            try:
                if hasattr(rv.size, "eval"):
                    total_params += rv.size.eval()
                else:
                    total_params += int(rv.size)
            except:
                # If we can't compute size, use ndim as fallback
                total_params += int(
                    np.prod(rv.shape.eval() if hasattr(rv.shape, "eval") else rv.shape)
                )

        results["model_info"] = {
            "n_free_rvs": len(model.free_RVs),
            "total_params": int(total_params),
            "rv_names": [rv.name for rv in model.free_RVs],
        }

        # Get initial point
        print("\nEvaluating at initial point...")
        initial_point = model.initial_point()

        # Evaluate logp
        logp_result = safe_point_logps(model, initial_point)

        results["initial_point_logp"] = {
            "total_logp": logp_result["total_logp"],
            "has_nan": logp_result["has_nan"],
            "has_inf": logp_result["has_inf"],
            "error": logp_result["error"],
        }

        print(f"  Total logp: {logp_result['total_logp']}")
        print(f"  Has NaN: {logp_result['has_nan']}")
        print(f"  Has Inf: {logp_result['has_inf']}")

        # Get worst terms
        if logp_result["point_logps"]:
            worst_terms = summarize_worst_terms(logp_result["point_logps"], n=10)
            results["worst_terms"] = worst_terms

            print(f"\nWorst 10 log-probability terms:")
            for i, term in enumerate(worst_terms, 1):
                print(f"  {i}. {term['name']}: {term['value']:.4f}")
        else:
            results["worst_terms"] = []

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\nERROR: {e}")
        print(traceback.format_exc())

    # Write results JSON
    print(f"\nWriting results to: {paths['results_json']}")
    write_json(paths["results_json"], results)

    # Write markdown report
    print(f"Writing report to: {paths['report_md']}")
    report = generate_report(results, paths)
    save_text(paths["report_md"], report)

    return results


def generate_report(results, paths):
    """Generate markdown report."""
    lines = [
        "# Group 1: Build-Only Sanity Check",
        "",
        f"**Run ID:** {results['run_id']}",
        f"**Timestamp:** {results['timestamp']}",
        f"**Status:** {'✓ SUCCESS' if results['success'] else '✗ FAILED'}",
        "",
        "## Configuration",
        "",
        f"- **Dataset:** {results['config']['dataset_name']}",
        f"- **Max T:** {results['config']['max_T']}",
        f"- **Max K:** {results['config']['max_K']}",
        f"- **Seed:** {results['config']['seed']}",
        "",
    ]

    if "data_shape" in results:
        lines.extend(
            [
                "## Data Shape",
                "",
                f"- **Cameras:** {results['data_shape']['C']}",
                f"- **Frames:** {results['data_shape']['T']}",
                f"- **Joints:** {results['data_shape']['K']}",
                "",
            ]
        )

    if "model_info" in results:
        lines.extend(
            [
                "## Model Information",
                "",
                f"- **Free RVs:** {results['model_info']['n_free_rvs']}",
                f"- **Total Parameters:** {results['model_info']['total_params']}",
                "",
            ]
        )

    if "initial_point_logp" in results:
        logp = results["initial_point_logp"]
        lines.extend(
            [
                "## Initial Point Log-Probability",
                "",
                f"- **Total logp:** {logp['total_logp']}",
                f"- **Has NaN:** {logp['has_nan']}",
                f"- **Has Inf:** {logp['has_inf']}",
            ]
        )

        if logp.get("error"):
            lines.append(f"- **Error:** {logp['error']}")

        lines.append("")

    if results.get("worst_terms"):
        lines.extend(
            [
                "## Worst 10 Log-Probability Terms",
                "",
                format_worst_terms_table(results["worst_terms"]),
                "",
            ]
        )

    if results.get("error"):
        lines.extend(
            [
                "## Error Details",
                "",
                f"```",
                results.get("traceback", results["error"]),
                f"```",
                "",
            ]
        )

    lines.extend(
        [
            "## Environment",
            "",
            f"- **Python:** {results['environment']['python_version'].split()[0]}",
            f"- **PyMC:** {results['environment']['pymc_version']}",
            f"- **PyTensor:** {results['environment']['pytensor_version']}",
            f"- **ArviZ:** {results['environment']['arviz_version']}",
            f"- **Platform:** {results['environment']['platform']}",
            "",
            "---",
            "",
            f"**Results JSON:** `{paths['results_json']}`",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    results = run_group_1(args)

    if results["success"]:
        print("\n" + "=" * 80)
        print("GROUP 1: PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("GROUP 1: FAILED")
        print("=" * 80)
        sys.exit(1)
