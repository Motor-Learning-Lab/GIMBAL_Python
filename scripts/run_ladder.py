"""C1 Capability Ladder Runner.

Generates a synthetic dataset from a ladder YAML config, runs the full
fitting pipeline (stages A-J), and writes results + a brief markdown
summary to the diagnosis directory.

Usage (from repo root):
    python scripts/run_ladder.py L00_minimal
    python scripts/run_ladder.py L00_minimal --method default --stages A-G
    python scripts/run_ladder.py L00_minimal --skip-generation

Output structure:
    diagnosis/results/<YYYY-MM-DD>/<step>/<method>/
        dataset/          (generated .npz + config.json)
        pipeline_results.json
        report.md

Pixi shortcut:
    pixi run ladder-run L00_minimal
    pixi run ladder-smoke          (L00_minimal stages A-G, no MCMC)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure UTF-8 output on Windows (stage files print unicode checkmarks)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.pipeline.utils.config_generator import (
    generate_from_config,
    save_dataset,
)
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

# ─── Stage registry ──────────────────────────────────────────────────────────

STAGE_NAMES = {
    "A": "Load Validation",
    "B": "2D Cleaning",
    "C": "Triangulation",
    "D": "3D Cleaning",
    "E": "Direction Statistics",
    "F": "Prior Building",
    "G": "Model Building",
    "H": "Fitting/Sampling",
    "I": "Posterior Diagnostics",
    "J": "Ground Truth Comparison",
}

ALL_STAGES = list(STAGE_NAMES.keys())


def parse_stages(stages_str: str) -> list[str]:
    """Parse '--stages A-G' or '--stages A,B,C' into a list of stage letters."""
    if "-" in stages_str and "," not in stages_str:
        start, end = stages_str.split("-")
        start_i = ALL_STAGES.index(start.upper())
        end_i = ALL_STAGES.index(end.upper())
        return ALL_STAGES[start_i : end_i + 1]
    return [s.strip().upper() for s in stages_str.split(",")]


# ─── Config loading ───────────────────────────────────────────────────────────

def load_yaml_config(step_name: str) -> dict:
    """Load YAML ladder config, returning a dict compatible with generate_from_config."""
    ladder_dir = REPO_ROOT / "datasets" / "ladder"
    yaml_path = ladder_dir / f"{step_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No ladder config found for '{step_name}'. "
            f"Expected: {yaml_path}\n"
            f"Available: {sorted(p.stem for p in ladder_dir.glob('*.yaml'))}"
        )
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Validate required keys
    for key in ("meta", "dataset_spec", "output"):
        if key not in config:
            raise ValueError(f"Ladder config '{step_name}' missing required key: '{key}'")
    return config


# ─── Dataset generation ───────────────────────────────────────────────────────

def generate_step_dataset(config: dict, dataset_dir: Path) -> None:
    """Generate synthetic dataset and save to dataset_dir."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save config as JSON so downstream stages can read it
    config_path = dataset_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Generating synthetic data (T={config['meta']['T']}, "
          f"seed={config['meta'].get('seed', 'random')})...")
    dataset = generate_from_config(config, calibrate_noise=True)
    save_dataset(dataset, dataset_dir)
    print(f"  Saved dataset to {dataset_dir}/")


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def run_stages(
    stages: list[str],
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[dict, bool]:
    """Run requested pipeline stages, return (results_dict, all_passed)."""
    results = {}
    all_passed = True

    # All intermediate outputs live in the same output_dir.
    # Stages that ask for a "previous stage output dir" get output_dir too.
    fits_dir = output_dir

    for stage in stages:
        label = STAGE_NAMES[stage]
        print(f"\n{'='*70}")
        print(f"  Stage {stage}: {label}")
        print(f"{'='*70}")
        try:
            if stage == "A":
                # run_stage_a(dataset_dir, output_dir)
                r = stage_a_load.run_stage_a(dataset_dir, output_dir)
                passed = r.get("summary", r.get("validation_summary", {})).get("all_passed", False)
            elif stage == "B":
                # run_stage_b(dataset_dir, output_dir)
                r = stage_b_clean_2d.run_stage_b(dataset_dir, output_dir)
                passed = r["quality_check"]["passed"]
            elif stage == "C":
                # run_stage_c(dataset_dir, stage_b_output_dir, output_dir)
                r = stage_c_triangulation.run_stage_c(dataset_dir, output_dir, output_dir)
                passed = True  # Ladder records metrics; bone-length warnings are not a gate
            elif stage == "D":
                # run_stage_d(dataset_dir, stage_c_output_dir, output_dir)
                r = stage_d_clean_3d.run_stage_d(dataset_dir, output_dir, output_dir)
                passed = True  # Ladder records metrics; bone-length warnings are not a gate
            elif stage == "E":
                # run_stage_e(dataset_dir, stage_d_output_dir, output_dir)
                r = stage_e_directions.run_stage_e(dataset_dir, output_dir, output_dir)
                passed = True  # Warnings acceptable
            elif stage == "F":
                # run_stage_f(stage_e_output_dir, output_dir)
                r = stage_f_priors.run_stage_f(output_dir, output_dir)
                v = r.get("validation", {})
                passed = v.get("all_priors_valid",
                               v.get("all_non_root_have_priors", False)
                               and v.get("reasonable_parameters", False))
            elif stage == "G":
                # run_stage_g(dataset_dir, fits_dir, output_dir)
                r = stage_g_model_build.run_stage_g(dataset_dir, fits_dir, output_dir)
                passed = r["model_diagnostics"]["compilation_successful"]
            elif stage == "H":
                # run_stage_h(dataset_dir, fits_dir, output_dir)
                r = stage_h_fitting.run_stage_h(dataset_dir, fits_dir, output_dir)
                passed = (
                    r["criteria_Q7"]["rhat_pass"]
                    and r["criteria_Q7"]["divergence_pass"]
                    and r["criteria_Q7"]["ess_pass"]
                )
            elif stage == "I":
                # run_stage_i(dataset_dir, fits_dir, output_dir)
                r = stage_i_diagnostics.run_stage_i(dataset_dir, fits_dir, output_dir)
                passed = r["criteria_Q7"]["reprojection_pass"]
            elif stage == "J":
                # run_stage_j(dataset_dir, fits_dir, output_dir)
                r = stage_j_ground_truth.run_stage_j(dataset_dir, fits_dir, output_dir)
                passed = r["criteria_Q7"]["bone_length_pass"]
            else:
                raise ValueError(f"Unknown stage: {stage}")

            r["passed"] = passed
            results[f"stage_{stage.lower()}"] = r
            status = "PASSED" if passed else "FAILED"
            print(f"  -> Stage {stage}: {status}")
            all_passed = all_passed and passed

        except Exception as exc:
            print(f"  -> Stage {stage}: EXCEPTION: {exc}")
            results[f"stage_{stage.lower()}"] = {"passed": False, "error": str(exc)}
            all_passed = False
            # Abort on first exception — later stages depend on earlier outputs
            break

    return results, all_passed


# ─── Report generation ───────────────────────────────────────────────────────

def write_report(
    step_name: str,
    method: str,
    stages_run: list[str],
    results: dict,
    all_passed: bool,
    output_dir: Path,
    config: dict,
) -> Path:
    """Write a brief markdown summary to output_dir/report.md."""
    report_path = output_dir / "report.md"

    with open(report_path, "w") as f:
        f.write(f"# GIMBAL C1 Ladder — {step_name}  ({method})\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Description:** {config['meta'].get('description', '').strip()}\n\n")

        complexity = config["meta"]
        skel = config["dataset_spec"]["skeleton"]
        obs = config["dataset_spec"]["observation"]
        states = config["dataset_spec"]["states"]
        cams = config["dataset_spec"]["cameras"]["cameras"]

        f.write("## Complexity profile\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| T | {complexity['T']} |\n")
        f.write(f"| Joints | {len(skel['joint_names'])} |\n")
        f.write(f"| States | {states['num_states']} |\n")
        f.write(f"| Cameras | {len(cams)} |\n")
        f.write(f"| Noise (px) | {obs['noise_px']} |\n")
        f.write(f"| Outliers | {obs.get('outliers', {}).get('fraction', 0) * 100:.0f}% "
                f"({'on' if obs.get('outliers', {}).get('enabled') else 'off'}) |\n")
        miss = obs.get("missingness", {})
        f.write(f"| Missingness | {miss.get('rate', 0) * 100:.0f}% "
                f"({'on' if miss.get('enabled') else 'off'}) |\n\n")

        f.write("## Stage results\n\n")
        overall = "ALL PASSED" if all_passed else "SOME FAILED"
        f.write(f"**Overall: {overall}**\n\n")
        f.write("| Stage | Name | Result |\n|---|---|---|\n")
        for stage in stages_run:
            key = f"stage_{stage.lower()}"
            name = STAGE_NAMES[stage]
            if key in results:
                passed = results[key].get("passed", False)
                status = "PASSED" if passed else "FAILED"
                if "error" in results[key]:
                    status = f"ERROR: {results[key]['error'][:60]}"
            else:
                status = "NOT RUN"
            f.write(f"| {stage} | {name} | {status} |\n")

        f.write("\n---\n")
        f.write(f"Output directory: `{output_dir}`\n")

    print(f"\n  Report written to {report_path}")
    return report_path


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a C1 capability ladder step end-to-end."
    )
    parser.add_argument(
        "step",
        help="Ladder step name (e.g. L00_minimal). Must match datasets/ladder/<step>.yaml",
    )
    parser.add_argument(
        "--method",
        default="default",
        help="Method label for output directory (default: 'default')",
    )
    parser.add_argument(
        "--stages",
        default="A-J",
        help="Stages to run: range 'A-G' or list 'A,B,C' (default: A-J)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation; use previously generated data in output dir",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override date tag for output path (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()

    date_tag = args.date or datetime.now().strftime("%Y-%m-%d")
    stages = parse_stages(args.stages)

    # ── Output paths ────────────────────────────────────────────────────────
    results_root = REPO_ROOT / "diagnosis" / "results" / date_tag / args.step / args.method
    dataset_dir = results_root / "dataset"
    output_dir = results_root

    print(f"\n{'='*70}")
    print(f"  GIMBAL C1 Ladder Runner")
    print(f"  Step:   {args.step}")
    print(f"  Method: {args.method}")
    print(f"  Stages: {', '.join(stages)}")
    print(f"  Output: {results_root.relative_to(REPO_ROOT)}")
    print(f"{'='*70}\n")

    # ── Load config ─────────────────────────────────────────────────────────
    config = load_yaml_config(args.step)

    # ── Generate dataset ─────────────────────────────────────────────────────
    if args.skip_generation:
        npz_path = dataset_dir / "dataset.npz"
        if not npz_path.exists():
            print(f"ERROR: --skip-generation requested but no dataset found at {npz_path}")
            sys.exit(1)
        print(f"  Using existing dataset at {dataset_dir}/")
    else:
        generate_step_dataset(config, dataset_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run pipeline stages ──────────────────────────────────────────────────
    results, all_passed = run_stages(stages, dataset_dir, output_dir)

    # ── Save results JSON ────────────────────────────────────────────────────
    results_json_path = output_dir / "pipeline_results.json"
    with open(results_json_path, "w") as f:
        json.dump(
            {
                "step": args.step,
                "method": args.method,
                "stages_run": stages,
                "all_passed": all_passed,
                "timestamp": datetime.now().isoformat(),
                "stages": results,
            },
            f,
            indent=2,
            default=str,
        )

    # ── Write markdown report ────────────────────────────────────────────────
    write_report(args.step, args.method, stages, results, all_passed, output_dir, config)

    # ── Exit code ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if all_passed:
        print("  ALL STAGES PASSED")
    else:
        print("  SOME STAGES FAILED  (see report.md for details)")
    print(f"{'='*70}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
