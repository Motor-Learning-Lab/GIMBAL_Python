"""
Test Group 14: Non-Centered Reparameterization of Root Random Walk vs. eta2_root

Purpose:
    TG13 (non-centered length_k/sigma2_k) passed the divergence and R-hat gates and
    left exactly one residual failure: eta2_root (R-hat 1.042, ESS 39.0) -- almost
    exactly TG11a's own residual failure (R-hat 1.054, ESS 35.5), despite TG13 not
    fixing anything to ground truth.

    x_root is built with pm.GaussianRandomWalk(sigma=sqrt(eta2_root)), which has the
    same centered-funnel shape that length_k/sigma2_k just had: eta2_root controls the
    scale of every one of the T-1 random-walk increments directly. This test applies
    the identical non-centering fix used in TG13, isolated to just this one variable,
    on top of TG13's already-fixed length_k/sigma2_k.

    Non-centered form:
        x_root_steps_raw[t] ~ Normal(0, 1)   for t = 1..T-1
        x_root[0] = x0   (same init_dist as GaussianRandomWalk's default, Normal(0,100))
        x_root[t] = x0 + cumsum(x_root_steps_raw[:t]) * sqrt(eta2_root)

Reference: plans/v0.2.2/v0.2.2_C1_findings.md, TG13 section

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_tg14_i1_noncentered_root.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_tg14_i1_noncentered_root.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_tg14_i1_noncentered_root.md
* [x] Built on TG13's non-centered length_k, changing only x_root's parameterization
* [x] Does not claim the problem is solved -- reports what results show
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).parent))

import gimbal_pymc.priors.initialization as gp_init
from test_tg11b_i3_redundancy_priors import compute_ground_truth_directions
from test_utils import extract_metrics, get_standard_synth_data, sample_model


def build_test_model_noncentered_root_and_length(
    synth_data: Dict[str, Any],
    eta2_root_sigma: float = 0.5,
    length_relative_sd: float = 0.02,
    raw_direction_sd: float = 0.05,
) -> pm.Model:
    """TG13's non-centered length_k, plus non-centered x_root vs eta2_root."""
    init_result = gp_init.initialize_from_observations_dlt(
        y_observed=synth_data["observations_uv"],
        camera_proj=synth_data["camera_matrices"],
        parents=synth_data["parents"],
    )

    x_true = synth_data["joint_positions"]
    T, K, _ = x_true.shape
    C = synth_data["camera_matrices"].shape[0]

    u_true = compute_ground_truth_directions(x_true, synth_data["parents"])
    lengths_true = synth_data["bone_lengths"]

    with pm.Model() as model:
        eta2_root = pm.HalfNormal(
            "eta2_root", sigma=eta2_root_sigma, initval=init_result.eta2[0]
        )

        # --- NON-CENTERED root random walk ---
        x_root0 = pm.Normal(
            "x_root0", mu=0.0, sigma=100.0, shape=(3,), initval=init_result.x_init[0, 0, :]
        )
        x_root_steps_raw = pm.Normal(
            "x_root_steps_raw", mu=0.0, sigma=1.0, shape=(T - 1, 3), initval=np.zeros((T - 1, 3))
        )
        x_root_steps = x_root_steps_raw * pt.sqrt(eta2_root)
        x_root = pm.Deterministic(
            "x_root",
            pt.concatenate(
                [x_root0[None, :], x_root0[None, :] + pt.cumsum(x_root_steps, axis=0)], axis=0
            ),
        )

        rho_list = []
        sigma2_list = []
        for k in range(K - 1):
            length_gt = lengths_true[k + 1]
            length_sd = length_relative_sd * length_gt
            rho_k = pm.Normal(f"rho_{k}", mu=length_gt, sigma=length_sd, initval=length_gt)
            rho_list.append(rho_k)
            sigma2_k = pm.HalfNormal(
                f"sigma2_{k}", sigma=0.01 * length_gt, initval=0.01 * length_gt
            )
            sigma2_list.append(sigma2_k)
        rho = pt.stack(rho_list)
        sigma2 = pt.stack(sigma2_list)

        x_all_list = [x_root]
        u_list = []
        for k in range(1, K):
            parent = int(synth_data["parents"][k])
            u_true_k = u_true[:, k, :]

            raw_u_k = pm.Normal(
                f"raw_u_{k}", mu=u_true_k, sigma=raw_direction_sd, shape=(T, 3), initval=u_true_k
            )
            norm_u_k = pt.sqrt((raw_u_k**2).sum(axis=-1, keepdims=True) + 1e-8)
            u_k = pm.Deterministic(f"u_{k}", raw_u_k / norm_u_k)
            u_list.append(u_k)

            # non-centered length_k (from TG13)
            length_k_raw = pm.Normal(
                f"length_{k}_raw", mu=0.0, sigma=1.0, shape=(T,), initval=np.zeros(T)
            )
            length_k = pm.Deterministic(
                f"length_{k}", rho[k - 1] + length_k_raw * pt.sqrt(sigma2[k - 1])
            )

            x_parent = x_all_list[parent]
            x_k = pm.Deterministic(f"x_{k}", x_parent + u_k * length_k[:, None])
            x_all_list.append(x_k)

        x_all = pm.Deterministic("x_all", pt.stack(x_all_list, axis=1))

        proj_param = synth_data["camera_matrices"]
        y_pred_list = []
        for c in range(C):
            A_c = proj_param[c, :, :3]
            b_c = proj_param[c, :, 3]
            uvw = pt.dot(x_all, A_c.T) + b_c[None, None, :]
            # Depth goes through a softplus first, giving it a smooth positive lower
            # bound instead of an unprotected division -- see
            # plans/v0.2.2/v0.2.2_C1_findings.md, "TG14 full-run chain divergence: a
            # camera cheirality singularity". Without this, a point drifting behind a
            # camera hits a real 1/w pole that HMC can diverge on (this is exactly what
            # trapped this run's chain 1).
            depth = pt.softplus(uvw[:, :, 2]) + 1e-9
            u = uvw[:, :, 0] / depth
            v = uvw[:, :, 1] / depth
            y_c = pt.stack([u, v], axis=-1)
            y_pred_list.append(y_c)
        y_pred = pm.Deterministic("y_pred", pt.stack(y_pred_list, axis=0))

        obs_sigma = pm.HalfNormal("obs_sigma", sigma=10.0, initval=init_result.obs_sigma)
        pm.Normal(
            "y_obs", mu=y_pred, sigma=obs_sigma, observed=synth_data["observations_uv"]
        )

    return model


def run_variant(synth_data, config: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "-" * 70)
    print("Variant: non-centered root + non-centered length_k")
    print("-" * 70)

    model = build_test_model_noncentered_root_and_length(
        synth_data,
        eta2_root_sigma=config["eta2_root_sigma"],
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
    max_rhat = max(v for v in metrics["rhat_max"].values() if v == v)
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


def main(pilot: bool = True):
    config = {
        "test_group": 14,
        "description": "Non-centered root random walk on top of TG13 non-centered length",
        "T": 100,
        "C": 3,
        "S": 3,
        "seed": 42,
        "eta2_root_sigma": 0.5,
        "length_relative_sd": 0.02,
        "raw_direction_sd": 0.05,
        "draws": 100 if pilot else 500,
        "tune": 200 if pilot else 500,
        "chains": 2,
    }
    print("=" * 70)
    print("Test Group 14: Non-Centered Root Random Walk")
    print(f"Mode: {'PILOT' if pilot else 'FULL'}")
    print("=" * 70)

    synth_data = get_standard_synth_data(T=config["T"], C=config["C"], S=config["S"], seed=config["seed"])
    result = run_variant(synth_data, config)
    trace = result.pop("trace")

    trace_path = Path(__file__).parent / "trace_tg14_i1_noncentered_root.nc"
    trace.to_netcdf(trace_path)
    print(f"Trace saved to {trace_path}")

    out = {
        "test_group": 14,
        "description": config["description"],
        "mode": "pilot" if pilot else "full",
        "configuration": config,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = Path(__file__).parent / "results_tg14_i1_noncentered_root.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    pilot = "--full" not in sys.argv
    main(pilot=pilot)
