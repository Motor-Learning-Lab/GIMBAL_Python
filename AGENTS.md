# AGENTS.md — GIMBAL Python

Guidance for coding agents working in this repository. Human contributors should read
this too. Format follows the [AGENTS.md](https://agents.md/) convention.

**Last verified against the codebase:** 2026-08-11 (v0.2.2, after P0 module reorganization)

---

## What this project is

GIMBAL infers 3D skeletal motion from multi-camera 2D keypoint observations using a
Bayesian HMM, implemented in PyMC. The library is `gimbal_pymc/`.

The three-stage pipeline:

1. **Collapsed HMM engine** — `gimbal_pymc/hmm/engine.py` → `collapsed_hmm_loglik()`
2. **Camera observation model** — `gimbal_pymc/hmm/observation_model.py` → `build_camera_observation_model()`
3. **Directional HMM prior** — `gimbal_pymc/hmm/directional_prior.py` → `add_directional_hmm_prior()`

## Environment: Pixi, not conda or pip

Never run `pip install` or `conda` directly. Use [Pixi](https://pixi.sh):

```bash
pixi install                 # set up environment
pixi run <task>              # run a task
pixi run python <script>     # run a script in the environment
```

Dependencies go in `pixi.toml` under `[dependencies]`. Do not put machine-specific
absolute paths in `pixi.toml` — a hardcoded Windows `PYTENSOR_FLAGS` compiledir there
previously created a literal `C:/` directory on Linux machines.

## Imports: always import from the leaf module

**This is the single most common source of breakage in this repo.** After the v0.2.2
P0 reorganization there are no convenience re-exports anywhere:

- `gimbal_pymc/__init__.py` imports the six subpackages and nothing else.
- Every subpackage `__init__.py` contains only a docstring — **zero** re-exports.

So the only import form that works is the fully-qualified leaf module:

```python
# CORRECT
from gimbal_pymc.data_cleaning.cleaning import clean_keypoints_2d, CleaningConfig
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
import gimbal_pymc.cameras.triangulation as tri
```

```python
# BROKEN — these raise AttributeError / ImportError at runtime, not import time
import gimbal_pymc as gp
gp.clean_keypoints_2d(...)                              # no top-level re-export
from gimbal_pymc.data_cleaning import CleaningConfig    # no subpackage re-export
```

The broken forms often survive review because the module still *imports* fine; the
failure only appears when the call executes. If you change an import, run the code.

## Package layout

| Subpackage | Contents |
|---|---|
| `gimbal_pymc/hmm/` | `engine.py`, `observation_model.py`, `directional_prior.py`, `gaussian_example.py` |
| `gimbal_pymc/cameras/` | `projection.py`, `triangulation.py`, `identifiability.py` |
| `gimbal_pymc/skeleton/` | `config.py`, `synthetic_data.py`, `metrics.py`, `visualization.py` |
| `gimbal_pymc/priors/` | `building.py`, `initialization.py`, `utils.py` |
| `gimbal_pymc/joints/` | `distributions.py`, `statistics.py` |
| `gimbal_pymc/data_cleaning/` | `cleaning.py` |

## Model-building conventions

- Model code builds **PyTensor tensors** (`import pytensor.tensor as pt`), not NumPy arrays.
- Use `pt.logsumexp()` for log-space reductions and `.dimshuffle()` to add dimensions
  (not NumPy `None` indexing). Both are used throughout `gimbal_pymc/hmm/`.
- **Stage order matters:** build the observation model (Stage 2) before the directional
  prior (Stage 3) — `add_directional_hmm_prior(U, log_obs_t, S, ...)` consumes the `U`
  and `log_obs_t` that Stage 2 returns.
- Nutpie sampling needs explicit initialization first; see `gimbal_pymc/priors/initialization.py`.
- Scripts that import PyMC on Windows set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`
  before the import, to avoid an OpenMP conflict.

## Running and testing

```bash
pixi run test-p0            # smoke + unit + integration + stage tests
pixi run test-all           # full pytest suite
pixi run demo-priors        # v0.2.1 data-driven priors demo
```

**Trap:** `tests/diagnostics/` contains files named `test_*.py` that are *not* pytest
tests — they are standalone scripts behind `if __name__ == "__main__":` with no
`def test_` functions. `pixi run test-diagnostics` therefore collects zero tests and
passes vacuously. Run them directly:

```bash
pixi run python tests/diagnostics/v0_2_1_divergence/test_tg11b_i3_redundancy_priors.py
```

`tests/pipeline/stage_*.py` are likewise pipeline stages, not pytest tests. They are
imported and called by the ladder runner on the `c1-capability-ladder` branch.

## Where truth lives (precedence)

When sources disagree, prefer in this order:

1. **GitHub issues** — current scope and dependencies. Authoritative for what to build.
2. **`plans/`** — design rationale and recorded findings. Explains *why*, may lag scope.
3. **Code** — implementation truth. Verify claims here before relying on docs.

`plans/README.md` indexes the planning documents. Anything under `plans/archive/` is
superseded — check the archive notice at the top before using it.

## Known state — do not rediscover

The sampler **does not currently converge**, even on the simplest synthetic dataset.
This is a known, diagnosed problem; do not treat it as a new discovery or a bug in
your own change.

- Root causes were isolated in December 2025: camera projection conditioning (primary)
  and parameter redundancy between root motion, directions, and bone lengths (co-equal).
- See `plans/v0.2.1/v0.2.1_divergence_plan_2.md` and the `report_tg*.md` files in
  `tests/diagnostics/v0_2_1_divergence/`.
- The planned remedies are issues A1/A2/A3 (informative camera, segment-length, and
  anchoring priors). They are not yet implemented.

## Conventions

- Diagnostic work follows a three-file pattern (`test_*.py`, `results_*.json`,
  `report_*.md`). See `IMPORTANT_FILE_NAMING_CONVENTIONS.md`.
- Do not commit run artifacts (traces, figures, generated datasets, result reports)
  to git. A stale committed `report.md` previously described a failure that no longer
  reproduced and misled debugging.
- Do not claim a result in a commit message unless an artifact demonstrates it. Report
  what results *show*, not that a problem is solved.
- One branch per issue, named `<issue-number>-<slug>`. Merge or close it; do not leave
  long-lived parallel branches — `main` was once 16 commits behind its own feature branch.
