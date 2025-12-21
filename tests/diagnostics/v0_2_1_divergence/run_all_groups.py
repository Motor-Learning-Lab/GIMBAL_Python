"""Batch runner for all diagnostic test groups.

Runs Groups 3-7 sequentially and generates a summary report.

Usage:
    pixi run python run_all_groups.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
tests_dir = project_root / "tests" / "diagnostics" / "v0_2_1_divergence"

GROUPS = [
    {
        "id": 3,
        "name": "baseline_minimal",
        "script": "test_group_3_sampling_baseline_minimal.py",
        "run_id": "batch_baseline",
    },
    {
        "id": 4,
        "name": "mixture_only",
        "script": "test_group_4_sampling_mixture_only.py",
        "run_id": "batch_mixture",
    },
    {
        "id": 5,
        "name": "hmm_only",
        "script": "test_group_5_sampling_hmm_only.py",
        "run_id": "batch_hmm",
    },
    {
        "id": 6,
        "name": "full_truncated",
        "script": "test_group_6_sampling_full_truncated.py",
        "run_id": "batch_full",
    },
    {
        "id": 7,
        "name": "likelihood_only_freeze_latents",
        "script": "test_group_7_likelihood_only_freeze_latents.py",
        "run_id": "batch_frozen",
    },
]


def run_group(group_info):
    """Run a single group test."""
    print(f"\n{'=' * 80}")
    print(f"Running Group {group_info['id']}: {group_info['name']}")
    print("=" * 80)

    script_path = tests_dir / group_info["script"]
    run_id = f"{group_info['run_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        "pixi",
        "run",
        "python",
        str(script_path),
        "--run_id",
        run_id,
        "--max_T",
        "80",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=False,
    )

    success = result.returncode == 0
    print(f"\nGroup {group_info['id']}: {'✓ SUCCESS' if success else '✗ FAILED'}")

    return {
        "group_id": group_info["id"],
        "group_name": group_info["name"],
        "run_id": run_id,
        "success": success,
        "returncode": result.returncode,
    }


def generate_summary_report(results):
    """Generate summary report across all groups."""
    lines = [
        "# Divergence Debugging Groups 3-7 Summary",
        "",
        f"**Run Date:** {datetime.now().isoformat()}",
        "",
        "## Results Overview",
        "",
    ]

    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        lines.append(
            f"- **Group {result['group_id']}** ({result['group_name']}): {status}"
        )

    lines.extend(
        [
            "",
            "## Comparative Analysis",
            "",
            "### Divergence Pattern",
            "",
        ]
    )

    # Add analysis placeholder
    lines.append("(Review individual reports for divergence rates and diagnostics)")

    lines.extend(
        [
            "",
            "## Interpretation Guidelines",
            "",
            "- **If Group 3 stable**: Core kinematics are sound",
            "- **If Group 4 unstable**: Mixture likelihood introduces issues",
            "- **If Group 5 unstable**: HMM Op/gradients cause problems",
            "- **If Group 6 unstable but 4 & 5 stable**: Interaction between mixture and HMM",
            "- **If Group 7 stable but 3 unstable**: Latent geometry is the issue",
            "",
            "## Next Steps",
            "",
            "1. Review individual group reports",
            "2. Identify first group to exhibit high divergence rate",
            "3. Focus debugging efforts on that specific component",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    """Run all groups and generate summary."""
    print("=" * 80)
    print("GIMBAL v0.2.1 Divergence Debugging: Groups 3-7 Batch Run")
    print("=" * 80)

    results = []

    for group_info in GROUPS:
        result = run_group(group_info)
        results.append(result)

    # Generate summary report
    summary = generate_summary_report(results)

    summary_path = tests_dir / "results" / "batch_summary.md"
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"\n{'=' * 80}")
    print("BATCH RUN COMPLETE")
    print("=" * 80)
    print(f"\nSummary report: {summary_path}")

    # Print summary to console
    print("\n" + summary)

    # Exit with error code if any test failed
    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
