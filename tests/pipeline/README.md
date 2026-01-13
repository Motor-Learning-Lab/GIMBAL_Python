# Pipeline Tests

End-to-end synthetic data generation and validation pipeline.

## Camera Configuration Schema

As of v0.2.1, pipeline camera configurations use **position/target specification only** (no rotation matrices or translation vectors).

### Required Fields

```json
{
  "cameras": {
    "num_cameras": 4,
    "camera_position": [[x1, y1, z1], [x2, y2, z2], ...],
    "target_position": [x, y, z],
    "focal_length": 1000.0,
    "image_size": [1920, 1080]
  }
}
```

**Field Descriptions:**
- `camera_position`: List of [x, y, z] positions for each camera
- `target_position`: Single [x, y, z] point that all cameras look at (shared)
- `focal_length`: Camera focal length in pixels
- `image_size`: [width, height] of image plane

### Projection Matrix Generation

Camera projection matrices are built internally using:

```python
import gimbal_pymc as gp

proj_matrix = gp.camera_utils.build_projection_matrix(
    camera_position=cam_pos,
    target_position=target,
    focal_length=focal_length,
    image_size=image_size
)
```

This function:
1. Computes camera orientation from position → target direction
2. Builds intrinsic matrix K from focal length and image center
3. Constructs 3x4 projection matrix P = K[R|t]

### Validation

The `config_generator.py` validates configurations:
- Checks for legacy R/t/K fields (raises `ValueError` if found)
- Projects skeleton bounds to ensure >50% visibility
- Raises error if projection results in excessive out-of-bounds points

### Example Configuration

See `tests/pipeline/configs/v0.2.1/L00_minimal.json` for a minimal working example with 4-camera ring geometry.

## Pipeline Stages

The pipeline consists of 10 stages (A-J):

- **Stage A (Load)**: Load synthetic dataset from config
- **Stage B (Clean 2D)**: Clean 2D keypoint observations
- **Stage C (Triangulation)**: Triangulate to 3D
- **Stage D (Clean 3D)**: Clean 3D reconstructed keypoints
- **Stage E (Directions)**: Compute joint directions
- **Stage F (Priors)**: Build priors from data statistics
- **Stage G (Model Build)**: Construct PyMC model
- **Stage H (Fitting)**: Sample with Nutpie
- **Stage I (Diagnostics)**: Compute metrics
- **Stage J (Ground Truth)**: Compare to ground truth

Run individual stages via:
```powershell
pixi run pipeline-clean        # Stage B
pixi run pipeline-triangulate  # Stage C
pixi run pipeline-full         # All stages (A-J)
```

## Synthetic Data Generation

Generate datasets from configs:

```powershell
pixi run generate-datasets
```

This reads all `*.json` files in `configs/v0.2.1/` (except `_template.json`) and generates corresponding datasets in `datasets/`.
