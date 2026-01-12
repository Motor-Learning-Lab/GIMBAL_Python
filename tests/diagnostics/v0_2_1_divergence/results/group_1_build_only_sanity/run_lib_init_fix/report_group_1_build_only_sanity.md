# Group 1: Build-Only Sanity Check

**Run ID:** lib_init_fix
**Timestamp:** 2025-12-20T13:11:27.699803
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

- **Total logp:** -251431.7894159169
- **Has NaN:** False
- **Has Inf:** False

## Worst 10 Log-Probability Terms

| Rank | Variable Name | Log-Probability |
|------|---------------|-----------------|
| 1 | `y_obs` | -124220.6200 |
| 2 | `dir_hmm_potential` | -124220.6200 |
| 3 | `raw_u_1` | -5862.2700 |
| 4 | `raw_u_2` | -5862.2700 |
| 5 | `raw_u_3` | -5862.2700 |
| 6 | `dir_hmm_mu_raw` | -11.0300 |
| 7 | `sigma2` | -3.3600 |
| 8 | `logodds_inlier` | -2.9500 |
| 9 | `dir_hmm_kappa` | -2.9000 |
| 10 | `x0_root` | -2.7600 |

## Environment

- **Python:** 3.11.14
- **PyMC:** 5.26.1
- **PyTensor:** 2.35.1
- **ArviZ:** 0.22.0
- **Platform:** Windows-10-10.0.26200-SP0

---

**Results JSON:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_1_build_only_sanity\run_lib_init_fix\results_group_1_build_only_sanity.json`