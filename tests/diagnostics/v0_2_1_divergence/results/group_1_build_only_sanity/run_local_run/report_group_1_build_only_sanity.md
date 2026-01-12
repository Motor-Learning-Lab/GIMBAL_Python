# Group 1: Build-Only Sanity Check

**Run ID:** local_run
**Timestamp:** 2025-12-20T00:31:34.954206
**Status:** ✓ SUCCESS

## Configuration

- **Dataset:** v0.2.1_L00_minimal
- **Max T:** None
- **Max K:** None
- **Seed:** 42

## Data Shape

- **Cameras:** 3
- **Frames:** 1800
- **Joints:** 4

## Model Information

- **Free RVs:** 15
- **Total Parameters:** 27025

## Initial Point Log-Probability

- **Total logp:** -inf
- **Has NaN:** False
- **Has Inf:** True

## Worst 10 Log-Probability Terms

| Rank | Variable Name | Log-Probability |
|------|---------------|-----------------|
| 1 | `rho` | -∞ |
| 2 | `sigma2` | -∞ |
| 3 | `length_1` | -∞ |
| 4 | `eps_root` | -554834.1700 |
| 5 | `y_obs` | -361359.3400 |
| 6 | `dir_hmm_potential` | -361359.3400 |
| 7 | `raw_u_1` | -4962.2700 |
| 8 | `raw_u_2` | -4962.2700 |
| 9 | `raw_u_3` | -4962.2700 |
| 10 | `dir_hmm_mu_raw` | -11.0300 |

## Environment

- **Python:** 3.11.14
- **PyMC:** 5.26.1
- **PyTensor:** 2.35.1
- **ArviZ:** 0.22.0
- **Platform:** Windows-10-10.0.26200-SP0

---

**Results JSON:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_1_build_only_sanity\run_local_run\results_group_1_build_only_sanity.json`