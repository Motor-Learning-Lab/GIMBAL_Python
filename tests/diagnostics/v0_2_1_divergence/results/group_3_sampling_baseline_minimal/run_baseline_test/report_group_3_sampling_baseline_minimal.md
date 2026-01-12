# Group 3: Baseline Sampling - Minimal Model

**Run ID:** baseline_test
**Timestamp:** 2025-12-20T13:31:41.236429
**Status:** ✗ FAILED

## Configuration

- **Dataset:** v0.2.1_L00_minimal
- **Max T:** 80
- **Max K:** None
- **Use Mixture:** False
- **Use Directional HMM:** False
- **Chains:** 2
- **Tune:** 300
- **Draws:** 100

## Data Shape

- **Cameras:** 3
- **Frames:** 80
- **Joints:** 4

## Initial Point Log-Probability

- **Total logp:** -108363.6556879498
- **Has NaN:** False
- **Has Inf:** False

## Worst 10 Log-Probability Terms

| Rank | Variable Name | Log-Probability |
|------|---------------|-----------------|
| 1 | `y_obs` | -108224.1800 |
| 2 | `raw_u_1` | -260.5500 |
| 3 | `raw_u_2` | -260.5500 |
| 4 | `raw_u_3` | -260.5500 |
| 5 | `sigma2` | -3.3600 |
| 6 | `x0_root` | -2.7600 |
| 7 | `rho` | -2.1900 |
| 8 | `eta2_root` | -0.9000 |
| 9 | `obs_sigma` | -0.1700 |
| 10 | `length_1` | 110.8100 |

## Gradient Statistics

- **L2 Norm:** 3.052514e+05
- **L-infinity Norm:** 2.110687e+05
- **NaN Count:** 0
- **Inf Count:** 0

## Interpretation

Sampling did not complete successfully.

## Next Action

Fix build/initialization errors before sampling.

---

**Results JSON:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_3_sampling_baseline_minimal\run_baseline_test\results_group_3_sampling_baseline_minimal.json`
**Trace:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_3_sampling_baseline_minimal\run_baseline_test/trace.nc`