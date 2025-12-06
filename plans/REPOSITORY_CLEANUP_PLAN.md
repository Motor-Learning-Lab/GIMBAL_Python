# GIMBAL Repository Cleanup and Organization Plan

**Date**: 2025-12-06  
**Purpose**: Establish clear naming conventions, file locations, and organization principles for the GIMBAL repository

---

## Current Problems

1. **Test files scattered in root directory** (test_*.py files should be in tests/)
2. **Debug scripts scattered in root directory** (debug_*.py files lack organization)
3. **Documentation files in root directory** (*.md files should be organized)
4. **Diagnostic outputs in plans/** (should have dedicated location)
5. **Unclear file naming conventions** (inconsistent prefixes and purposes)
6. **Multiple versions of similar files** (confusion about which is current)

---

## Proposed Directory Structure

```
GIMBAL_Python/
├── gimbal/                          # Core library code (DO NOT CHANGE)
│   ├── *.py                        # Core modules
│   └── torch_legacy/               # Legacy PyTorch implementation
│
├── tests/                          # All test code
│   ├── unit/                       # NEW: Unit tests for core modules
│   │   ├── test_camera_utils.py
│   │   ├── test_triangulation.py
│   │   └── ...
│   ├── integration/                # NEW: Integration tests
│   │   ├── test_v0_1_3_directional_hmm.py
│   │   ├── test_v0_2_1_data_driven_priors.py
│   │   └── ...
│   ├── smoke/                      # NEW: Quick smoke tests
│   │   ├── test_v0_2_0_smoke.py
│   │   ├── test_v0_2_1_smoke.py
│   │   └── ...
│   ├── diagnostics/                # NEW: Diagnostic test suites
│   │   └── v0_2_1_divergence/     # Divergence testing for v0.2.1
│   │       ├── __init__.py
│   │       ├── test_runner.py
│   │       ├── test_baseline.py
│   │       ├── test_hmm_effect.py
│   │       └── ...
│   └── README.md                   # Test organization guide
│
├── debug/                          # NEW: Debug/exploration scripts
│   ├── camera/                     # Camera-related debugging
│   │   ├── debug_camera_orient.py
│   │   ├── debug_skeleton_projection.py
│   │   └── visualize_camera_fix.py
│   ├── triangulation/              # Triangulation debugging
│   │   ├── debug_triangulate.py
│   │   ├── debug_triangulation.py
│   │   └── analyze_skeleton_3d.py
│   ├── misc/                       # Other debugging
│   │   ├── debug_zero_noise.py
│   │   └── check_skeleton_extents.py
│   └── README.md                   # Debug scripts guide
│
├── results/                        # NEW: Test/debug outputs
│   ├── diagnostics/                # Diagnostic plots and outputs
│   │   └── v0_2_1_divergence/     # v0.2.1 divergence diagnostics
│   ├── benchmarks/                 # Performance benchmarks
│   └── figures/                    # Generated figures
│
├── examples/                       # Example usage scripts (KEEP AS IS)
│   ├── run_gimbal_demo.py
│   ├── demo_v0_2_0_pymc_pipeline.py
│   └── demo_v0_2_1_data_driven_priors.py
│
├── notebook/                       # Jupyter notebooks (KEEP AS IS)
│   └── *.ipynb
│
├── plans/                          # Planning docs and reports (KEEP AS IS)
│   ├── v*.md                       # Version planning/completion docs
│   └── *.md                        # Design docs
│
├── docs/                           # NEW: Current documentation
│   ├── camera_projection_technical_report.md
│   ├── CAMERA_FIX_SUMMARY.md
│   ├── DIVERGENCE_DEBUGGING_GUIDE.md
│   ├── DIVERGENCE_TEST_RESULTS.md
│   └── IMPLEMENTATION_SUMMARY_v0_2_1.md
│
├── .gitignore
├── .gitattributes
├── pixi.toml
├── pixi.lock
├── README.md                       # Main project README
├── GIMBAL spec.md                  # Core specification
└── LICENSE
```

---

## Naming Conventions

### Test Files

| Type | Prefix | Location | Example |
|------|--------|----------|---------|
| Unit tests | `test_<module>.py` | `tests/unit/` | `test_camera_utils.py` |
| Integration tests | `test_<feature>.py` | `tests/integration/` | `test_v0_2_1_data_driven_priors.py` |
| Smoke tests | `test_<version>_smoke.py` | `tests/smoke/` | `test_v0_2_1_smoke.py` |
| Diagnostic suites | `test_<aspect>.py` | `tests/diagnostics/<suite>/` | `test_baseline.py` |

### Debug Scripts

| Type | Prefix | Location | Example |
|------|--------|----------|---------|
| Camera debugging | `debug_camera_*.py` | `debug/camera/` | `debug_camera_orient.py` |
| Visualization | `visualize_*.py` | `debug/camera/` | `visualize_camera_fix.py` |
| Analysis | `analyze_*.py` | `debug/triangulation/` | `analyze_skeleton_3d.py` |
| Checking | `check_*.py` | `debug/misc/` | `check_skeleton_extents.py` |

### Documentation

| Type | Format | Location | Example |
|------|--------|----------|---------|
| Implementation summaries | `IMPLEMENTATION_SUMMARY_<version>.md` | `docs/` | `IMPLEMENTATION_SUMMARY_v0_2_1.md` |
| Technical reports | `<topic>_technical_report.md` | `docs/` | `camera_projection_technical_report.md` |
| Debugging guides | `<TOPIC>_DEBUGGING_GUIDE.md` | `docs/` | `DIVERGENCE_DEBUGGING_GUIDE.md` |
| Test results | `<TOPIC>_TEST_RESULTS.md` | `docs/` | `DIVERGENCE_TEST_RESULTS.md` |
| Planning docs | `v<version>-<type>.md` | `plans/` | `v0.2.1-completion-report.md` |

### Output Files

| Type | Location | Example |
|------|----------|---------|
| Diagnostic plots | `results/diagnostics/<suite>/` | `results/diagnostics/v0_2_1_divergence/*.png` |
| Test reports | `results/diagnostics/<suite>/` | `results/diagnostics/v0_2_1_divergence/report.md` |
| Benchmark results | `results/benchmarks/` | `results/benchmarks/v0_2_1_performance.json` |
| Figures | `results/figures/` | `results/figures/camera_comparison.png` |

---

## Migration Plan

### Phase 1: Create New Directory Structure

```bash
mkdir -p tests/unit
mkdir -p tests/integration  
mkdir -p tests/smoke
mkdir -p tests/diagnostics/v0_2_1_divergence
mkdir -p debug/camera
mkdir -p debug/triangulation
mkdir -p debug/misc
mkdir -p results/diagnostics/v0_2_1_divergence
mkdir -p results/benchmarks
mkdir -p results/figures
mkdir -p docs
```

### Phase 2: Move Test Files

**From root to tests/smoke/**:
- `test_v0_2_1_smoke.py` → `tests/smoke/test_v0_2_1_smoke.py`
- `test_projection_quick.py` → `tests/smoke/test_projection_quick.py`
- `test_new_cameras.py` → `tests/smoke/test_new_cameras.py`
- `test_camera_projection.py` → `tests/smoke/test_camera_projection.py`
- `test_camera_positioning.py` → `tests/smoke/test_camera_positioning.py`

**Reorganize existing tests/**:
- `tests/test_v0_2_1_data_driven_priors.py` → `tests/integration/test_v0_2_1_data_driven_priors.py`
- `tests/test_v0_1_3_directional_hmm.py` → `tests/integration/test_v0_1_3_directional_hmm.py`
- `tests/test_demo_v0_2_0_smoke.py` → `tests/smoke/test_v0_2_0_smoke.py`
- Keep unit tests in `tests/unit/`: `test_pymc_utils.py`, `test_model_init.py`, etc.

**Divergence test suite**:
- `tests/v0_2_1_divergence_tests/` → `tests/diagnostics/v0_2_1_divergence/`

### Phase 3: Move Debug Scripts

**Camera debugging**:
- `debug_camera_orient.py` → `debug/camera/debug_camera_orient.py`
- `debug_skeleton_projection.py` → `debug/camera/debug_skeleton_projection.py`
- `visualize_camera_fix.py` → `debug/camera/visualize_camera_fix.py`
- `show_camera_differences.py` → `debug/camera/show_camera_differences.py`

**Triangulation debugging**:
- `debug_triangulate.py` → `debug/triangulation/debug_triangulate.py`
- `debug_triangulation.py` → `debug/triangulation/debug_triangulation.py`
- `analyze_skeleton_3d.py` → `debug/triangulation/analyze_skeleton_3d.py`

**Misc debugging**:
- `debug_zero_noise.py` → `debug/misc/debug_zero_noise.py`
- `check_skeleton_extents.py` → `debug/misc/check_skeleton_extents.py`

### Phase 4: Move Documentation

**To docs/**:
- `IMPLEMENTATION_SUMMARY_v0_2_1.md` → `docs/IMPLEMENTATION_SUMMARY_v0_2_1.md`
- `camera_projection_technical_report.md` → `docs/camera_projection_technical_report.md`
- `CAMERA_FIX_SUMMARY.md` → `docs/CAMERA_FIX_SUMMARY.md`
- `DIVERGENCE_DEBUGGING_GUIDE.md` → `docs/DIVERGENCE_DEBUGGING_GUIDE.md`
- `DIVERGENCE_TEST_RESULTS.md` → `docs/DIVERGENCE_TEST_RESULTS.md`

**Keep in root**:
- `README.md` (main project readme)
- `GIMBAL spec.md` (core specification)
- `LICENSE`

### Phase 5: Move Output Files

**Diagnostic outputs**:
- `plans/v0.2.1-diagnostics/` → `results/diagnostics/v0_2_1_divergence/`
- `plans/v0.2.1-divergence-report.md` → `results/diagnostics/v0_2_1_divergence/report.md`

**Figures**:
- `*.png` files in root → `results/figures/`

### Phase 6: Update Import Paths

After moving files, update all import statements in:
- Test files (update relative imports)
- Debug scripts (update module imports)
- README files (update file paths)

---

## Divergence Testing Organization

### Current State (Problematic)
```
tests/v0_2_1_divergence_tests/     # Test code
plans/v0.2.1-diagnostics/          # Output plots (WRONG!)
plans/v0.2.1-divergence-report.md  # Output report (WRONG!)
```

### Proposed State (Clear)
```
tests/diagnostics/v0_2_1_divergence/          # Test code
    ├── __init__.py
    ├── test_runner.py                        # Main test runner
    ├── test_baseline.py                      # Test modules
    ├── test_hmm_effect.py
    └── ...

results/diagnostics/v0_2_1_divergence/        # Test outputs
    ├── report.md                             # Generated report
    ├── plots/                                # Diagnostic plots
    │   ├── divergence_summary.png
    │   ├── baseline_parallel.png
    │   └── ...
    └── data/                                 # Raw test data (optional)
        └── test_results.json
```

**Key Principle**: Test CODE in `tests/`, test OUTPUTS in `results/`

---

## Benefits of This Organization

1. **Clear separation**: Code vs. outputs vs. documentation
2. **Discoverable**: Obvious where to find things
3. **Version-specific**: Easy to track what's been tested for each version
4. **Scalable**: Can add new test suites, debug scripts without clutter
5. **Professional**: Standard Python project structure
6. **Git-friendly**: Can .gitignore results/ without losing test code

---

## Implementation Checklist

- [ ] Create new directory structure
- [ ] Move test files and update imports
- [ ] Move debug scripts and update imports
- [ ] Move documentation files
- [ ] Move output files
- [ ] Update README.md with new structure
- [ ] Update .gitignore for results/
- [ ] Create README.md in each major directory
- [ ] Update test runner paths
- [ ] Run tests to verify everything works
- [ ] Commit and document the reorganization

---

## Notes

- This is a **reorganization**, not a rewrite - functionality remains the same
- All moved files should update their internal imports
- Consider adding `.gitignore` entries for `results/` to avoid committing outputs
- Add README files in key directories to explain their purpose
