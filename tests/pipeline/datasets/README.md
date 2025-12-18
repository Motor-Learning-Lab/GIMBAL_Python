# Dataset Artifact Policy

## What Gets Versioned

**Configs (versioned):**
- `config.json` - Defines each dataset's parameters (skeleton, motion dynamics, noise, camera geometry)
- `config_template.json` - Template for creating new dataset configs

**Rationale:** Configs are small, stable, and define the "ground truth" specification of each dataset. They enable reproducible generation.

## What Gets Generated (not versioned)

**Generated artifacts (regenerated on demand):**
- `dataset.npz` - Generated motion data, observations, ground truth (~several MB)
- `metrics.json` - Quality metrics computed from dataset
- `figures/` - Diagnostic visualizations (motion_3d.png, poses_3d.png, reprojection_2d.png, states.png, missingness.png)

**Rationale:** These are deterministic outputs from configs (given same seed) and can be regenerated quickly with:
```bash
pixi run generate-datasets v0.2.1_L00_minimal v0.2.1_L01_noise v0.2.1_L02_outliers v0.2.1_L03_missingness
```

## Canonical Datasets (L00-L03)

The v0.2.1 canonical datasets are defined in:
- `tests/pipeline/datasets/v0.2.1_L00_minimal/` - Clean baseline (1 state, no noise)
- `tests/pipeline/datasets/v0.2.1_L01_noise/` - Observation noise
- `tests/pipeline/datasets/v0.2.1_L02_outliers/` - Outlier keypoints
- `tests/pipeline/datasets/v0.2.1_L03_missingness/` - Missing data (NaN)

**First-time setup:** Generate datasets before running tests:
```bash
pixi run generate-datasets v0.2.1_L00_minimal v0.2.1_L01_noise v0.2.1_L02_outliers v0.2.1_L03_missingness
```

**Testing:** Integration tests expect these datasets to exist:
```bash
pixi run test-pipeline
```

## Adding New Datasets

1. Create directory: `tests/pipeline/datasets/v0.2.1_<name>/`
2. Add `config.json` (use `config_template.json` as reference)
3. Generate: `pixi run generate-datasets v0.2.1_<name>`
4. Commit only the `config.json`, not the generated artifacts
