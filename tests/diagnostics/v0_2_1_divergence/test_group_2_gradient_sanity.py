"""Group 2: Gradient Sanity Check

Tests whether gradients are finite and well-behaved at the initial point
and under small perturbations.

Usage:
    pixi run python test_group_2_gradient_sanity.py --run_id local_run
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
    compile_logp_and_grad,
    format_grad_components_table,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Group 2: Gradient Sanity Check")
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

    # Initialize using library estimator (not ad-hoc)
    print("  Initializing from observations (DLT triangulation)...")
    from gimbal.fit_params import initialize_from_observations_dlt

    # Reshape y_2d from (C, T, K, 2) as expected by library
    init_result = initialize_from_observations_dlt(
        y_observed=data["y_2d"],
        camera_proj=data["camera_proj"],
        parents=data["parents"],
    )

    print(f"  Initialization complete")
    print(
        f"    Triangulation rate: {init_result.metadata.get('triangulation_rate', 'N/A'):.2%}"
    )
    print(f"    Bone lengths (rho): {init_result.rho}")
    print(f"    Bone variances (sigma2): {init_result.sigma2}")

    # Build model
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

        U = model["U"]
        log_obs_t = model["log_obs_t"]

        gimbal.add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=1,
            joint_names=data["joint_names"],
        )

    return model, init_result


def compute_gradient_stats(grad_dict):
    """Compute statistics on gradient dictionary."""
    all_grads = []
    grad_components = []

    for name, grad_val in grad_dict.items():
        if isinstance(grad_val, np.ndarray):
            flat = grad_val.flatten()
            for i, val in enumerate(flat):
                all_grads.append(val)
                grad_components.append(
                    {
                        "name": name,
                        "index": str(i) if len(flat) > 1 else "",
                        "value": float(val),
                        "abs_value": float(np.abs(val)),
                    }
                )
        else:
            all_grads.append(grad_val)
            grad_components.append(
                {
                    "name": name,
                    "index": "",
                    "value": float(grad_val),
                    "abs_value": float(np.abs(grad_val)),
                }
            )

    all_grads = np.array(all_grads)

    stats = {
        "grad_norm_l2": float(np.linalg.norm(all_grads)),
        "grad_norm_linf": float(np.max(np.abs(all_grads))),
        "nan_count": int(np.sum(np.isnan(all_grads))),
        "inf_count": int(np.sum(np.isinf(all_grads))),
        "total_components": len(all_grads),
    }

    # Sort by absolute value
    grad_components.sort(key=lambda x: x["abs_value"], reverse=True)

    return stats, grad_components


def perturb_point(point, scale=1e-3, seed=None):
    """Generate a perturbed version of the point."""
    if seed is not None:
        np.random.seed(seed)

    perturbed = {}
    for name, val in point.items():
        if isinstance(val, np.ndarray):
            noise = np.random.randn(*val.shape) * scale
            perturbed[name] = val + noise
        else:
            noise = np.random.randn() * scale
            perturbed[name] = val + noise

    return perturbed


def run_group_2(args):
    """Run Group 2 diagnostics."""
    # Setup paths
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_paths("group_2_gradient_sanity", args.run_id)

    # Initialize results
    results = {
        "group": "group_2_gradient_sanity",
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

        # Get initial point
        print("\nEvaluating gradients at initial point...")
        initial_point = model.initial_point()

        # Compile gradient function
        logp_fn, dlogp_fn = compile_logp_and_grad(model)

        # Evaluate gradient (returns array, not dict)
        grad_array = dlogp_fn(initial_point)

        # Convert to dict using free RV names
        grad_dict = {}
        free_rv_shapes = [
            rv.shape.eval() if hasattr(rv.shape, "eval") else rv.shape
            for rv in model.free_RVs
        ]
        free_rv_names = [rv.name for rv in model.free_RVs]

        idx = 0
        for rv_name, rv_shape in zip(free_rv_names, free_rv_shapes):
            rv_size = int(np.prod(rv_shape))
            grad_dict[rv_name] = grad_array[idx : idx + rv_size].reshape(rv_shape)
            idx += rv_size

        # Compute statistics
        grad_stats, grad_components = compute_gradient_stats(grad_dict)

        results["gradient_stats"] = grad_stats
        results["top_20_abs_grad_components"] = grad_components[:20]

        print(f"  Gradient L2 norm: {grad_stats['grad_norm_l2']:.6e}")
        print(f"  Gradient L-inf norm: {grad_stats['grad_norm_linf']:.6e}")
        print(f"  NaN count: {grad_stats['nan_count']}")
        print(f"  Inf count: {grad_stats['inf_count']}")

        # Perturbation check
        print("\nTesting gradient under perturbations...")
        perturbation_results = []

        for i in range(3):
            print(f"  Perturbation {i+1}/3...")
            perturbed = perturb_point(initial_point, scale=1e-3, seed=args.seed + i + 1)

            try:
                pert_logp = float(logp_fn(perturbed))
                pert_grad_array = dlogp_fn(perturbed)

                # Convert gradient array to dict
                pert_grad_dict = {}
                idx = 0
                for rv_name, rv_shape in zip(free_rv_names, free_rv_shapes):
                    rv_size = int(np.prod(rv_shape))
                    pert_grad_dict[rv_name] = pert_grad_array[
                        idx : idx + rv_size
                    ].reshape(rv_shape)
                    idx += rv_size

                pert_stats, _ = compute_gradient_stats(pert_grad_dict)

                perturbation_results.append(
                    {
                        "perturbation_id": i + 1,
                        "logp": pert_logp,
                        "logp_finite": bool(np.isfinite(pert_logp)),
                        "grad_l2_norm": pert_stats["grad_norm_l2"],
                        "grad_linf_norm": pert_stats["grad_norm_linf"],
                        "grad_nan_count": pert_stats["nan_count"],
                        "grad_inf_count": pert_stats["inf_count"],
                        "grad_finite": pert_stats["nan_count"] == 0
                        and pert_stats["inf_count"] == 0,
                    }
                )

                print(
                    f"    Logp: {pert_logp:.2f}, Grad finite: {perturbation_results[-1]['grad_finite']}"
                )

            except Exception as e:
                perturbation_results.append(
                    {
                        "perturbation_id": i + 1,
                        "error": str(e),
                        "logp_finite": False,
                        "grad_finite": False,
                    }
                )
                print(f"    ERROR: {e}")

        results["perturbation_checks"] = perturbation_results

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
        "# Group 2: Gradient Sanity Check",
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

    if "gradient_stats" in results:
        stats = results["gradient_stats"]
        lines.extend(
            [
                "## Gradient Statistics at Initial Point",
                "",
                f"- **L2 Norm:** {stats['grad_norm_l2']:.6e}",
                f"- **L-infinity Norm:** {stats['grad_norm_linf']:.6e}",
                f"- **NaN Count:** {stats['nan_count']} / {stats['total_components']}",
                f"- **Inf Count:** {stats['inf_count']} / {stats['total_components']}",
                "",
            ]
        )

    if results.get("top_20_abs_grad_components"):
        lines.extend(
            [
                "## Top 20 Gradient Components (by Absolute Value)",
                "",
                format_grad_components_table(
                    results["top_20_abs_grad_components"], n=20
                ),
                "",
            ]
        )

    if results.get("perturbation_checks"):
        lines.extend(
            [
                "## Perturbation Analysis",
                "",
                "Testing gradient behavior under small perturbations (scale=1e-3):",
                "",
                "| Perturbation | Logp | Logp Finite | Grad L2 Norm | Grad Finite |",
                "|--------------|------|-------------|--------------|-------------|",
            ]
        )

        for pert in results["perturbation_checks"]:
            pid = pert["perturbation_id"]
            if "error" in pert:
                lines.append(f"| {pid} | ERROR | ✗ | - | ✗ |")
            else:
                logp = f"{pert['logp']:.2f}" if pert.get("logp") else "N/A"
                logp_ok = "✓" if pert.get("logp_finite") else "✗"
                grad_norm = (
                    f"{pert['grad_l2_norm']:.2e}" if pert.get("grad_l2_norm") else "N/A"
                )
                grad_ok = "✓" if pert.get("grad_finite") else "✗"
                lines.append(
                    f"| {pid} | {logp} | {logp_ok} | {grad_norm} | {grad_ok} |"
                )

        lines.append("")

    if results.get("error"):
        lines.extend(
            [
                "## Error Details",
                "",
                "```",
                results.get("traceback", results["error"]),
                "```",
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
            "",
            "---",
            "",
            f"**Results JSON:** `{paths['results_json']}`",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    results = run_group_2(args)

    if results["success"]:
        print("\n" + "=" * 80)
        print("GROUP 2: PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("GROUP 2: FAILED")
        print("=" * 80)
        sys.exit(1)
