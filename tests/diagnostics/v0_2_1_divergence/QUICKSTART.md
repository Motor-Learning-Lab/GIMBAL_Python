# Test Suite Quick Start Guide

## Running Tests with Pixi

This workspace uses `pixi` for environment management. Always run Python commands through pixi.

### Run Full Test Suite

```powershell
pixi run python tests/v0_2_1_divergence_tests/test_runner.py
```

### Run Individual Test Groups

```powershell
pixi run python tests/v0_2_1_divergence_tests/test_baseline.py
pixi run python tests/v0_2_1_divergence_tests/test_hmm_effect.py
pixi run python tests/v0_2_1_divergence_tests/test_state_count.py
pixi run python tests/v0_2_1_divergence_tests/test_root_variance.py
pixi run python tests/v0_2_1_divergence_tests/test_bone_length_variance.py
pixi run python tests/v0_2_1_divergence_tests/test_runtime_scaling.py
```

### Test Import

```powershell
pixi run python -c "from tests.v0_2_1_divergence_tests import test_utils; print('✓ Imports successful')"
```

## Output Locations

- **Report**: `tests/v0.2.1-divergence-report.md`
- **Plots**: `tests/v0.2.1-diagnostics/`

## Expected Runtime

- Full test suite: ~30-60 minutes (depending on hardware)
- Individual test groups: ~5-10 minutes each
- Runtime scaling tests use reduced samples for faster execution

## Troubleshooting

If you encounter import errors, ensure pixi environment is set up:
```powershell
pixi install
```
