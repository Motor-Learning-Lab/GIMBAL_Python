# GIMBAL Import Refactoring Inventory

This document lists ALL files that currently import from `gimbal_pymc` and need to be updated to use explicit subpackage imports.

**Goal**: Remove all convenience exports from `gimbal_pymc.__init__.py` and require explicit subpackage imports.

## Current State (v0.2.2-dev)

`gimbal_pymc.__init__.py` currently exports ONLY subpackages:
```python
from . import hmm, cameras, skeleton, joints, priors, data_cleaning
```

However, many files still expect old-style convenience imports that no longer exist.

## Required Import Changes

### Pattern 1: Old-style convenience imports
```python
# OLD (no longer supported)
from gimbal_pymc import build_camera_observation_model

# NEW (explicit subpackage)
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
```

### Pattern 2: Alias imports
```python
# OLD
import gimbal_pymc as gp
x = gp.build_camera_observation_model(...)

# NEW
import gimbal_pymc as gp
x = gp.hmm.observation_model.build_camera_observation_model(...)
```

---

## Files Requiring Updates

### 1. Examples Directory (`examples/`)

#### `examples/demo_v0_2_0_pymc_pipeline.py`
**Current imports:**
```python
import gimbal_pymc as gp
from gimbal_pymc import (
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
    generate_demo_sequence,
    build_camera_observation_model,
    add_directional_hmm_prior,
)
```

**Required changes:**
```python
import gimbal_pymc as gp
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior
```

---

#### `examples/demo_v0_2_1_data_driven_priors.py`
**Current imports:**
```python
import gimbal_pymc as gp
from gimbal_pymc import DEMO_V0_1_SKELETON
from gimbal_pymc.synthetic_data import generate_demo_sequence, SyntheticDataConfig
```

**Required changes:**
```python
import gimbal_pymc as gp
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import generate_demo_sequence, SyntheticDataConfig
```

**Additional notes:**
- Line 39: `from gimbal_pymc.priors.initialization import initialize_from_groundtruth` ✓ (already correct)
- Line 86: `from gimbal_pymc.cameras.projection import camera_center_from_proj` ✓ (already correct)
- Line 1867: `from gimbal_pymc import build_camera_observation_model` → `from gimbal_pymc.hmm.observation_model import build_camera_observation_model`
- Line 2223: `from gimbal_pymc.hmm.engine import collapsed_hmm_loglik` ✓ (already correct)

---

### 2. Tests Directory (`tests/`)

#### `tests/smoke/test_full_pipeline_smoke.py`
**Current imports:**
```python
from gimbal_pymc import (
    build_camera_observation_model,
    add_directional_hmm_prior,
    generate_demo_sequence,
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
)
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
```

**Required changes:**
```python
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior
from gimbal_pymc.skeleton.synthetic_data import generate_demo_sequence, SyntheticDataConfig
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt  # ✓ already correct
```

---

#### `tests/smoke/test_v0_2_1_smoke.py`
**Current imports:**
```python
import gimbal_pymc as gp
# Uses: gp.triangulate_multi_view, gp.CleaningConfig, gp.clean_keypoints_2d, etc.
```

**Required changes:**
Replace all `gp.` references with explicit subpackage paths:
- `gp.triangulate_multi_view` → `gp.cameras.triangulation.triangulate_multi_view`
- `gp.CleaningConfig` → `gp.data_cleaning.CleaningConfig`
- `gp.clean_keypoints_2d` → `gp.data_cleaning.clean_keypoints_2d`
- `gp.clean_keypoints_3d` → `gp.data_cleaning.clean_keypoints_3d`
- `gp.compute_direction_statistics` → `gp.joints.statistics.compute_direction_statistics`
- `gp.build_priors_from_statistics` → `gp.priors.building.build_priors_from_statistics`
- `gp.get_gamma_shape_rate` → `gp.priors.building.get_gamma_shape_rate`

---

#### `tests/smoke/test_api_exports_smoke.py`
**Current imports:**
```python
import gimbal_pymc as gp
# Tests exports: gp.triangulate_multi_view, gp.CleaningConfig, etc.
```

**Required changes:**
Same as `test_v0_2_1_smoke.py` - update all `gp.` references to use explicit subpackage paths.

---

#### `tests/test_stage1_collapsed_hmm.py`
**Current imports:**
```python
from gimbal_pymc.hmm.engine import forward_log_prob_single, collapsed_hmm_loglik
```

**Status:** ✓ Already correct (uses explicit subpackage imports)

---

#### `tests/test_stage2_camera_model.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.priors.utils import validate_stage2_outputs
```

**Status:** ✓ Already correct (uses explicit subpackage imports)

---

#### `tests/unit/test_dlt_round_trip.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.triangulation import triangulate_multi_view
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.triangulation import triangulate_multi_view  # ✓ already correct
```

---

#### `tests/unit/test_model_init.py`
**Current imports:**
```python
from gimbal_pymc.joints.distributions import VonMisesFisher
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
from gimbal_pymc.cameras.projection import build_projection_matrix
```

**Status:** ✓ Already correct

---

#### `tests/unit/test_pymc_utils.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import initialize_from_groundtruth
from gimbal_pymc.priors.utils import (...)
from gimbal_pymc.joints.distributions import VonMisesFisher
```

**Status:** ✓ Already correct

---

#### `tests/unit/test_dlt_init.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
from gimbal_pymc.cameras.projection import build_projection_matrix
```

**Status:** ✓ Already correct

---

#### `tests/integration/test_stage3_directional_hmm.py`
**Current imports:**
```python
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior, _build_kappa
```

**Status:** ✓ Already correct

---

#### `tests/integration/test_data_driven_priors.py`
**Current imports:**
```python
import gimbal_pymc as gp
# Uses gp.* throughout
```

**Required changes:**
Review all `gp.` references and update to explicit subpackage paths. Common patterns:
- `gp.triangulate_multi_view` → `gp.cameras.triangulation.triangulate_multi_view`
- `gp.DEMO_V0_1_SKELETON` → `gp.skeleton.config.DEMO_V0_1_SKELETON`
- `gp.SyntheticDataConfig` → `gp.skeleton.synthetic_data.SyntheticDataConfig`

---

### 3. Pipeline Tests (`tests/pipeline/`)

#### `tests/pipeline/stage_b_clean_2d.py`
**Current imports:**
```python
import gimbal_pymc
from gimbal_pymc.data_cleaning import CleaningConfig
```

**Required changes:**
```python
import gimbal_pymc as gp  # Add alias for consistency
from gimbal_pymc.data_cleaning import CleaningConfig  # ✓ already correct
```

---

#### `tests/pipeline/stage_c_triangulation.py`
**Current imports:**
```python
import gimbal_pymc as gp
# Uses gp.* references
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths.

---

#### `tests/pipeline/stage_d_clean_3d.py`
**Current imports:**
```python
import gimbal_pymc
from gimbal_pymc.data_cleaning import CleaningConfig
```

**Required changes:**
```python
import gimbal_pymc as gp
from gimbal_pymc.data_cleaning import CleaningConfig  # ✓ already correct
```

---

#### `tests/pipeline/stage_e_directions.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths.

---

#### `tests/pipeline/stage_f_priors.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths.

---

#### `tests/pipeline/stage_g_model_build.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths.

---

#### `tests/pipeline/stage_h_fitting.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths (line 36 and beyond).

---

#### `tests/pipeline/stage_i_diagnostics.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths (line 30 and beyond).

---

#### `tests/pipeline/utils/config_generator.py`
**Current imports:**
```python
from gimbal_pymc.skeleton.config import SkeletonConfig
from gimbal_pymc.skeleton.synthetic_data import generate_skeletal_motion_continuous
from gimbal_pymc.cameras.projection import build_projection_matrix
```

**Status:** ✓ Already correct

---

#### `tests/pipeline/utils/metrics.py`
**Current imports:**
```python
from gimbal_pymc.skeleton.metrics import (...)
from gimbal_pymc.cameras.identifiability import (...)
```

**Status:** ✓ Already correct

---

### 4. Diagnostic Tests (`tests/diagnostics/v0_2_1_divergence/`)

#### `tests/diagnostics/v0_2_1_divergence/test_ground_truth_init.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `tests/diagnostics/v0_2_1_divergence/extract_ground_truth.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `tests/diagnostics/v0_2_1_divergence/debug_model_logp.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `tests/diagnostics/v0_2_1_divergence/test_sanity_posterior_geometry.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
```

**Status:** ✓ Already correct

---

#### `tests/diagnostics/v0_2_1_divergence/test_tg09_i1_root_fixed.py`
**Current imports:**
```python
import gimbal_pymc as gp
# Uses: gp.fit_params.initialize_from_observations_dlt (line 54)
```

**Required changes:**
```python
import gimbal_pymc as gp
# Update: gp.fit_params.initialize_from_observations_dlt → gp.priors.initialization.initialize_from_observations_dlt
```

---

#### `tests/diagnostics/v0_2_1_divergence/test_tg11b_i3_redundancy_priors.py`
**Current imports:**
```python
import gimbal_pymc
from gimbal_pymc.prior_building import get_gamma_shape_rate
```

**Required changes:**
```python
import gimbal_pymc as gp
from gimbal_pymc.priors.building import get_gamma_shape_rate
```

---

#### `tests/diagnostics/v0_2_1_divergence/test_tg11a_i3_redundancy_fixed.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths (line 58).

---

#### `tests/diagnostics/v0_2_1_divergence/test_tg10_i2_direct_3d.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths (line 56).

---

#### Tests with `import gimbal_pymc from _diag_utils import ...` (SYNTAX ERRORS)

These files have broken import statements:
- `test_group_1_build_only_sanity.py` (line 25)
- `test_group_2_gradient_sanity.py` (line 25)
- `test_group_3_sampling_baseline_minimal.py` (line 28)
- `test_group_4_sampling_mixture_only.py` (line 28)
- `test_group_5_sampling_hmm_only.py` (line 28)
- `test_group_6_sampling_full_truncated.py` (line 28)
- `test_group_7_likelihood_only_freeze_latents.py` (line 31)
- `test_group_8a_temporal_T80.py` (line 29)
- `test_group_8b_temporal_T200.py` (line 28)
- `test_group_8c_temporal_T500.py` (line 28)
- `test_group_8d_temporal_T1000.py` (line 28)
- `test_group_8e_temporal_T1800.py` (line 28)

**Current (BROKEN):**
```python
import gimbal_pymc from _diag_utils import (...)
```

**Required fix:**
```python
import gimbal_pymc as gp
from _diag_utils import (...)
```

---

### 5. Debug Scripts (`debug/`)

#### `debug/misc/debug_zero_noise.py`
**Current imports:**
```python
from gimbal_pymc import *
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `debug/misc/check_skeleton_extents.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `debug/triangulation/debug_triangulate.py`
**Current imports:**
```python
from gimbal_pymc import *
from gimbal_pymc.triangulation import triangulate_multi_view
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.triangulation import triangulate_multi_view
```

---

#### `debug/triangulation/analyze_skeleton_3d.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.projection import camera_center_from_proj
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.projection import camera_center_from_proj  # ✓ already correct
```

---

#### `debug/triangulation/debug_triangulation.py`
**Current imports:**
```python
import gimbal_pymc as gp
```

**Required changes:**
Update all `gp.` references to explicit subpackage paths (line 51).

---

#### `debug/camera/visualize_camera_fix.py`
**Current imports:**
```python
from gimbal_pymc import (
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
    generate_demo_sequence,
)
from gimbal_pymc.cameras.projection import camera_center_from_proj, project_points_numpy
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
from gimbal_pymc.cameras.projection import camera_center_from_proj, project_points_numpy  # ✓ already correct
```

---

#### `debug/camera/show_camera_differences.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `debug/camera/debug_skeleton_projection.py`
**Current imports:**
```python
from gimbal_pymc import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
```

**Required changes:**
```python
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig, generate_demo_sequence
```

---

#### `debug/camera/debug_camera_orient.py`
**Current imports:**
```python
from gimbal_pymc.synthetic_data import generate_camera_matrices
from gimbal_pymc.cameras.projection import camera_center_from_proj
```

**Required changes:**
```python
from gimbal_pymc.skeleton.synthetic_data import generate_camera_matrices
from gimbal_pymc.cameras.projection import camera_center_from_proj  # ✓ already correct
```

---

### 6. Notebooks (`notebook/`)

#### `notebook/demo_v0_1_complete.py`
**Current imports:**
```python
from gimbal_pymc import (
    build_camera_observation_model,
    add_directional_hmm_prior,
    generate_demo_sequence,
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
)
from gimbal_pymc.priors.initialization import InitializationResult
```

**Required changes:**
```python
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior
from gimbal_pymc.skeleton.synthetic_data import generate_demo_sequence, SyntheticDataConfig
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.priors.initialization import InitializationResult  # ✓ already correct
```

---

#### `notebook/demo_v0_2_1_data_driven_priors.py`
**Current imports:**
```python
import gimbal_pymc as gp
from gimbal_pymc import (
    DEMO_V0_1_SKELETON,
    # ... many more
)
from gimbal_pymc.priors.initialization import initialize_from_groundtruth
from gimbal_pymc.cameras.projection import camera_center_from_proj
# ... and more throughout the file (lines 1859, 1867, 2223)
```

**Required changes:**
Multiple sections need updating - this is a large file with many imports. Key areas:
- Line 24-40: Initial imports block
- Line 86: camera_center_from_proj (already correct)
- Line 1859-1867: Module imports for inspection
- Line 2223: collapsed_hmm_loglik import (already correct)

See notes in `examples/demo_v0_2_1_data_driven_priors.py` section above for patterns.

---

### 7. Tools (`tools/`)

#### `tools/test_config_refactor.py`
**Current imports:**
```python
from gimbal_pymc import SyntheticDataConfig
```

**Required changes:**
```python
from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig
```

---

### 8. Internal Package Files (`gimbal_pymc/`)

#### `gimbal_pymc/hmm/observation_model.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import InitializationResult
from gimbal_pymc.priors.utils import (...)
from gimbal_pymc.priors.building import get_gamma_shape_rate as gamma_from_mode_sd
```

**Status:** ✓ Already correct (uses explicit subpackage imports)

---

#### `gimbal_pymc/hmm/directional_prior.py`
**Current imports:**
```python
from gimbal_pymc.hmm.engine import collapsed_hmm_loglik
```

**Status:** ✓ Already correct

---

#### `gimbal_pymc/skeleton/synthetic_data.py`
**Current imports:**
```python
from gimbal_pymc.skeleton.config import SkeletonConfig
```

**Status:** ✓ Already correct

---

#### `gimbal_pymc/skeleton/metrics.py`
**Current imports:**
```python
from gimbal_pymc.skeleton.config import SkeletonConfig
```

**Status:** ✓ Already correct

---

#### `gimbal_pymc/priors/utils.py`
**Current imports:**
```python
from gimbal_pymc.priors.initialization import InitializationResult
```

**Status:** ✓ Already correct

---

#### `gimbal_pymc/hmm/gaussian_example.py`
**Current imports:**
```python
from gimbal_pymc.hmm.engine import collapsed_hmm_loglik
```

**Status:** ✓ Already correct

---

## Summary Statistics

### By Directory:
- **examples/**: 2 files need updates
- **tests/smoke/**: 3 files need updates
- **tests/unit/**: 1 file needs updates (4 already correct)
- **tests/integration/**: 1 file needs updates (1 already correct)
- **tests/pipeline/**: 7 files need updates (2 already correct)
- **tests/diagnostics/**: 16 files need updates (including 12 with syntax errors)
- **debug/**: 8 files need updates (3 already correct)
- **notebook/**: 2 files need updates
- **tools/**: 1 file needs updates
- **gimbal_pymc/ (internal)**: 0 files need updates (all already correct!)

### Total:
- **41 files need import updates**
- **12 files have CRITICAL syntax errors** (broken import statements)
- **11 files already use correct imports**

## Common Import Mapping Table

| Old Import | New Import |
|-----------|-----------|
| `from gimbal_pymc import DEMO_V0_1_SKELETON` | `from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON` |
| `from gimbal_pymc import SyntheticDataConfig` | `from gimbal_pymc.skeleton.synthetic_data import SyntheticDataConfig` |
| `from gimbal_pymc import generate_demo_sequence` | `from gimbal_pymc.skeleton.synthetic_data import generate_demo_sequence` |
| `from gimbal_pymc import build_camera_observation_model` | `from gimbal_pymc.hmm.observation_model import build_camera_observation_model` |
| `from gimbal_pymc import add_directional_hmm_prior` | `from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior` |
| `gp.triangulate_multi_view` | `gp.cameras.triangulation.triangulate_multi_view` |
| `gp.CleaningConfig` | `gp.data_cleaning.CleaningConfig` |
| `gp.clean_keypoints_2d` | `gp.data_cleaning.clean_keypoints_2d` |
| `gp.clean_keypoints_3d` | `gp.data_cleaning.clean_keypoints_3d` |
| `gp.compute_direction_statistics` | `gp.joints.statistics.compute_direction_statistics` |
| `gp.build_priors_from_statistics` | `gp.priors.building.build_priors_from_statistics` |
| `gp.get_gamma_shape_rate` | `gp.priors.building.get_gamma_shape_rate` |
| `gp.fit_params.*` | `gp.priors.initialization.*` |
| `from gimbal_pymc.synthetic_data import *` | `from gimbal_pymc.skeleton.synthetic_data import *` |
| `from gimbal_pymc.triangulation import *` | `from gimbal_pymc.cameras.triangulation import *` |
| `from gimbal_pymc.prior_building import *` | `from gimbal_pymc.priors.building import *` |

## Next Steps

1. **Fix CRITICAL syntax errors** in 12 diagnostic test files (highest priority)
2. **Update test files** (tests/ directory) - 11 files
3. **Update examples** - 2 files
4. **Update debug scripts** - 8 files
5. **Update notebooks** - 2 files
6. **Update tools** - 1 file
7. **Run full test suite** to verify no regressions
8. **Update documentation** to reflect new import patterns

## Validation Checklist

After updates, verify:
- [ ] All files can be imported without errors
- [ ] `pixi run pytest tests/` passes
- [ ] Example scripts run without errors
- [ ] Diagnostic tests can execute
- [ ] No remaining `from gimbal_pymc import <function>` patterns
- [ ] `import gimbal_pymc as gp` only uses `gp.subpackage.module.function` patterns
