"""
Test Group 12: Root Variance Constraint on Top of Strong Redundancy Priors

Purpose:
    TG11b (strong GT-based priors on directions/lengths) eliminated divergences but
    still failed the Stage H gate on exactly one parameter: eta2_root (root
    random-walk variance), R-hat 1.054 / ESS 35.5, per
    plans/v0.2.2/v0.2.2_C1_findings.md. Every other parameter passed cleanly.

    This test holds TG11b's configuration fixed and sweeps eta2_root_sigma tighter,
    to check whether constraining root variance closes that one remaining gap and
    produces a configuration that passes the repo's Stage H gate (0 divergences,
    max R-hat <= 1.05, min ESS >= 100) -- a candidate for a real, honestly-converging
    L0.

Configuration:
    - Same as TG11b: T=100, C=3, S=3, seed=42, strong priors (length SD 2%,
      raw direction SD 0.05)
    - Sweeps eta2_root_sigma: [0.5 (TG11b baseline), 0.1, 0.02]
    - Pilot run: reduced draws/tune for a fast first signal, not a final result

Reference: plans/v0.2.2/v0.2.2_C1_findings.md, "Open questions" #1

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_tg12_i1i3_root_variance_constrained.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_tg12_i1i3_root_variance_constrained.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_tg12_i1i3_root_variance_constrained.md
* [x] Reuses test_tg11b's build_test_model_strong_priors unmodified (imported, not copied)
* [x] Does not claim the problem is solved -- reports what results show
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))

from test_tg11b_i3_redundancy_priors import build_test_model_strong_priors
from test_utils import extract_metrics, get_standard_synth_data, sample_model


def run_variant(synth_data, config: Dict[str, Any], eta2_root_sigma: float) -> Dict[str, Any]:
    label = f"eta2_root_sigma={eta2_root_sigma}"
    print("\n" + "-" * 70)
    print(f"Variant: {label}")
    print("-" * 70)

    model = build_test_model_strong_priors(
        synth_data,
        eta2_root_sigma=eta2_root_sigma,
        length_relative_sd=config["length_relative_sd"],
        raw_direction_sd=config["raw_direction_sd"],
    )
    print(f"  [OK] Model has {len(model.free_RVs)} free RVs")

    print(f"Sampling (draws={config['draws']}, tune={config['tune']}, chains={config['chains']})...")
    start = time.time()
    trace = sample_model(model, draws=config["draws"], tune=config["tune"], chains=config["chains"])
    runtime = time.time() - start
    print(f"  [OK] Sampling completed in {runtime:.1f}s")

    metrics = extract_metrics(trace, runtime)
    max_rhat = max(v for v in metrics["rhat_max"].values() if v == v)  # skip NaN
    min_ess = min(v for v in metrics["ess_bulk"].values() if v == v)
    worst_rhat_param = max(
        (k for k, v in metrics["rhat_max"].items() if v == v), key=lambda k: metrics["rhat_max"][k]
    )
    worst_ess_param = min(
        (k for k, v in metrics["ess_bulk"].items() if v == v), key=lambda k: metrics["ess_bulk"][k]
    )

    passed = metrics["divergences"] == 0 and max_rhat <= 1.05 and min_ess >= 100
    print(
        f"  div={metrics['divergences']} max_rhat={max_rhat:.4f} ({worst_rhat_param}) "
        f"min_ess={min_ess:.1f} ({worst_ess_param}) -> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "eta2_root_sigma": eta2_root_sigma,
        "divergences": metrics["divergences"],
        "total_samples": metrics["total_samples"],
        "max_rhat": float(max_rhat),
        "worst_rhat_param": worst_rhat_param,
        "min_ess": float(min_ess),
        "worst_ess_param": worst_ess_param,
        "runtime_seconds": runtime,
        "gate_pass": bool(passed),
        "trace": trace,
    }


def main(pilot: bool = True, sweep: list = None):
    config = {
        "test_group": 12,
        "description": "Root variance constraint on top of TG11b strong priors",
        "T": 100,
        "C": 3,
        "S": 3,
        "seed": 42,
        "length_relative_sd": 0.02,
        "raw_direction_sd": 0.05,
        "draws": 100 if pilot else 500,
        "tune": 200 if pilot else 500,
        "chains": 2,
    }
    print("=" * 70)
    print("Test Group 12: Root Variance Constraint")
    print(f"Mode: {'PILOT (reduced draws)' if pilot else 'FULL'}")
    print("=" * 70)

    synth_data = get_standard_synth_data(T=config["T"], C=config["C"], S=config["S"], seed=config["seed"])

    sweep = sweep if sweep is not None else [0.5, 0.1, 0.02]
    results = [run_variant(synth_data, config, s) for s in sweep]

    # Save traces for post-hoc diagnosis (one per swept eta2_root_sigma)
    for r in results:
        sigma_tag = str(r["eta2_root_sigma"]).replace(".", "p")
        trace_path = Path(__file__).parent / f"trace_tg12_sigma_{sigma_tag}.nc"
        r["trace"].to_netcdf(trace_path)
        print(f"[OK] Trace saved to {trace_path.name}")

    results_for_json = [{k: v for k, v in r.items() if k != "trace"} for r in results]

    out = {
        "test_group": 12,
        "description": config["description"],
        "mode": "pilot" if pilot else "full",
        "configuration": config,
        "sweep": results_for_json,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = Path(__file__).parent / "results_tg12_i1i3_root_variance_constrained.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {results_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"  eta2_root_sigma={r['eta2_root_sigma']:<6} div={r['divergences']:<4} "
            f"max_rhat={r['max_rhat']:<10.4f} min_ess={r['min_ess']:<8.1f} "
            f"{'PASS' if r['gate_pass'] else 'FAIL'}"
        )


if __name__ == "__main__":
    pilot = "--full" not in sys.argv
    sweep = None
    if "--sweep" in sys.argv:
        idx = sys.argv.index("--sweep")
        sweep = [float(x) for x in sys.argv[idx + 1].split(",")]
    main(pilot=pilot, sweep=sweep)
