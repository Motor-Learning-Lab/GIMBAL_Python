"""Batch runner for Group 8 temporal scaling tests.

Runs Groups 8a-8e sequentially to identify temporal scaling threshold.

Usage:
    pixi run python run_group_8_temporal.py
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
    {"id": "8a", "name": "T80", "script": "test_group_8a_temporal_T80.py", "T": 80},
    {"id": "8b", "name": "T200", "script": "test_group_8b_temporal_T200.py", "T": 200},
    {"id": "8c", "name": "T500", "script": "test_group_8c_temporal_T500.py", "T": 500},
    {"id": "8d", "name": "T1000", "script": "test_group_8d_temporal_T1000.py", "T": 1000},
    {"id": "8e", "name": "T1800", "script": "test_group_8e_temporal_T1800.py", "T": 1800},
]


def run_group(group_info):
    """Run a single group test."""
    print(f"\n{'=' * 80}")
    print(f"Running Group {group_info['id']}: {group_info['name']} (T={group_info['T']})")
    print('=' * 80)
    
    script_path = tests_dir / group_info["script"]
    run_id = f"batch_{group_info['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "pixi", "run", "python", str(script_path),
        "--run_id", run_id,
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
        "T": group_info["T"],
        "run_id": run_id,
        "success": success,
        "returncode": result.returncode,
    }


def generate_summary_report(results):
    """Generate summary report across all temporal tests."""
    lines = [
        "# Group 8 Temporal Scaling Analysis Summary",
        "",
        f"**Run Date:** {datetime.now().isoformat()}",
        "",
        "## Purpose",
        "",
        "Identify at what sequence length (T) the full model (mixture + HMM) begins to diverge.",
        "Group 6 showed stability at T=80, but Stage H fails at T=1800.",
        "",
        "## Results Overview",
        "",
        "| Group | T (frames) | Status | Notes |",
        "|-------|-----------|--------|-------|",
    ]
    
    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        lines.append(f"| **{result['group_id']}** | {result['T']} | {status} | - |")
    
    lines.extend([
        "",
        "## Analysis",
        "",
        "### Divergence Threshold",
        "",
        "(Review individual reports for divergence rates at each T value)",
        "",
        "### Computational Scaling",
        "",
        "- **Expected complexity:** O(T) for observation model, O(T²) for HMM transitions",
        "- **Memory scaling:** Linear with T (state vectors)",
        "- **Gradient depth:** Scales with T (backprop through sequence)",
        "",
        "### Interpretation",
        "",
        "**If divergences appear gradually:**",
        "- Accumulated numerical error (floating point precision)",
        "- Need better numerical stability in HMM Op",
        "",
        "**If divergences appear suddenly at threshold:**",
        "- Identifiability collapse (too many latents vs constraints)",
        "- Parameter space becomes ill-conditioned",
        "- May need stronger priors or regularization",
        "",
        "**If T=1800 stable:**",
        "- Stage H issue is not temporal scaling",
        "- Check: different priors, different data, different sampling config",
        "",
        "## Next Steps",
        "",
        "1. Plot divergence rate vs T",
        "2. Plot sampling time vs T",
        "3. Identify critical T threshold",
        "4. If threshold < 1800: strengthen model for longer sequences",
        "5. If no threshold: debug Stage H specific configuration",
        "",
    ])
    
    return "\n".join(lines)


def main():
    """Run all Group 8 temporal tests and generate summary."""
    print("=" * 80)
    print("GIMBAL v0.2.1 Temporal Scaling Tests: Group 8a-8e")
    print("=" * 80)
    
    results = []
    
    for group_info in GROUPS:
        result = run_group(group_info)
        results.append(result)
        
        # If a test fails catastrophically, ask whether to continue
        if not result["success"]:
            print(f"\nGroup {group_info['id']} failed. Continuing to next test...")
    
    # Generate summary report
    summary = generate_summary_report(results)
    
    summary_path = tests_dir / "results" / "group_8_temporal_summary.md"
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    summary_path.write_text(summary, encoding="utf-8")
    
    print(f"\n{'=' * 80}")
    print("TEMPORAL SCALING TESTS COMPLETE")
    print('=' * 80)
    print(f"\nSummary report: {summary_path}")
    
    # Print summary to console
    print("\n" + summary)
    
    # Exit with error code if any test failed
    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
