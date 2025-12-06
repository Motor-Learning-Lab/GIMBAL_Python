# Camera Projection System - Technical Report

**Date**: November 30, 2025  
**Issue**: All three cameras show similar "looking down" views for vertical skeleton poses  
**Status**: User reports this behavior doesn't make sense geometrically

---

## Executive Summary

The camera projection system uses an **orthographic-like projection** with the formula:
```
pixel_coords = focal_length * (point_3d - camera_position)
```

**Critical Issue Identified by User**: When the skeleton is vertical (state 0), all three cameras show similar views as if "looking down/up" at the skeleton. User correctly points out that with three non-collinear cameras, at least one should see the skeleton as a series of dots (viewing from the side), but this is NOT what we observe.

---

## 1. Camera Generation Function

### Function Signature
```python
def generate_camera_matrices(config: SyntheticDataConfig) -> np.ndarray
```

**Location**: `gimbal/synthetic_data.py`, lines 250-340

### Implementation Details

```python
def generate_camera_matrices(config: SyntheticDataConfig) -> np.ndarray:
    """
    Generate camera projection matrices P = [A | b].
    
    Returns:
        camera_proj: (C, 3, 4) array where camera_proj[c] @ [x, y, z, 1] = [u, v, w]
                     Pixel coordinates are (u, v)
    """
    C = config.C
    scene_center = np.array([0.0, 0.0, 100.0])  # Skeleton root position
    
    camera_positions = []
    
    # Camera 0: Front-below
    camera_positions.append(scene_center + np.array([80, 0, -30]))
    
    # Camera 1: Side-level
    camera_positions.append(scene_center + np.array([0, 80, 20]))
    
    # Camera 2: Above-angled
    camera_positions.append(scene_center + np.array([-60, -60, 70]))
    
    focal_length = 10.0
    
    camera_proj = np.zeros((C, 3, 4))
    for c in range(C):
        camera_pos = camera_positions[c]
        
        # A = focal_length * I
        A = np.eye(3) * focal_length
        
        # b = -A * camera_pos
        b = -A @ camera_pos
        
        # P = [A | b]
        camera_proj[c] = np.column_stack([A, b])
    
    return camera_proj
```

### Current Camera Positions
```
Scene center (skeleton root): [0, 0, 100]

Camera 0: [80, 0, 70]    (Front-below, 30 units below skeleton center)
Camera 1: [0, 80, 120]   (Side-level, 20 units above skeleton center)  
Camera 2: [-60, -60, 170] (Above-angled, 70 units above skeleton center)
```

### Distances from Scene Center
```
Camera 0: distance = sqrt(80^2 + 0^2 + (-30)^2) = 85.4 units
Camera 1: distance = sqrt(0^2 + 80^2 + 20^2) = 82.5 units
Camera 2: distance = sqrt((-60)^2 + (-60)^2 + 70^2) = 110.9 units
```

---

## 2. Projection Formula

### Mathematical Definition

For a 3D point `P = [x, y, z]` and camera position `C = [cx, cy, cz]`:

```
Step 1: Homogeneous coordinates
P_h = [x, y, z, 1]

Step 2: Apply projection matrix
[u, v, w]^T = P_matrix @ P_h

Step 3: Extract pixel coordinates
pixel_u = u
pixel_v = v
(w is ignored for orthographic projection)
```

Where `P_matrix = [A | b]` with:
- `A = focal_length * I_3x3` (3×3 identity scaled by focal length)
- `b = -A @ camera_position` (3×1 translation vector)

### Expanded Formula

```
[u]   [f  0  0 | -f*cx]   [x]   [f*(x - cx)]
[v] = [0  f  0 | -f*cy] @ [y] = [f*(y - cy)]
[w]   [0  0  f | -f*cz]   [z]   [f*(z - cz)]
                          [1]
```

Final pixel coordinates:
```
u = focal_length * (x - camera_x)
v = focal_length * (y - camera_y)
```

**KEY OBSERVATION**: This is NOT a true perspective projection. There's no division by depth (z-coordinate). This is more like a **scaled orthographic projection** where the camera's z-position doesn't affect the pixel coordinates.

---

## 3. Observation Generation Function

### Function Signature
```python
def generate_observations(
    positions_3d: np.ndarray,      # (T, K, 3)
    camera_proj: np.ndarray,       # (C, 3, 4)
    config: SyntheticDataConfig,
) -> np.ndarray                    # (C, T, K, 2)
```

**Location**: `gimbal/synthetic_data.py`, lines 342-402

### Implementation (Key Lines 380-395)

```python
# For each camera c, timestep t, joint k:
x = positions_3d[t, k, :]  # (3,) 3D position
x_h = np.append(x, 1.0)    # (4,) homogeneous coordinates

# Project to camera
y_proj = camera_proj[c] @ x_h  # (3,) result [u, v, w]

# CRITICAL FIX (v0.2.1): Removed incorrect perspective division
# OLD (BUG): y_2d = y_proj[:2] / y_proj[2]
# NEW (CORRECT): 
y_2d = y_proj[:2]  # No perspective division for orthographic projection

# Add observation noise
noise = rng.normal(0, config.obs_noise_std, size=2)
y_2d = y_2d + noise

# Apply occlusions
if rng.random() < config.occlusion_rate:
    y_2d = np.array([np.nan, np.nan])

y_observed[c, t, k, :] = y_2d
```

---

## 4. Skeleton Configuration

### Structure
```python
DEMO_V0_1_SKELETON = SkeletonConfig(
    joint_names=['root', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5'],
    parents=[-1, 0, 1, 2, 3, 4],
    bone_lengths=[0, 10, 10, 8, 8, 6]
)
```

Total skeleton height: 42 units (sum of bone lengths excluding root)

### HMM State Definitions

**Location**: `gimbal/synthetic_data.py`, lines 82-130

```python
# State 0: Upright (bones point straight up)
canonical_mu[0, 1:, 2] = 1.0  # All bones: direction = [0, 0, 1]

# State 1: Forward lean (bones point forward-up)
canonical_mu[1, 1:, 0] = 0.6  # x-component
canonical_mu[1, 1:, 2] = 0.8  # z-component
# Normalized: approximately [0.6, 0, 0.8]

# State 2: Sideways lean (bones point sideways-up)
canonical_mu[2, 1:, 1] = 0.7  # y-component  
canonical_mu[2, 1:, 2] = 0.7  # z-component
# Normalized: approximately [0, 0.7, 0.7]
```

---

## 5. Test Results

### Test 1: `test_camera_projection.py`

**Purpose**: Validate projection math independently

**Test Cases**:
1. Simple orthographic projection (camera at origin)
2. Camera at offset position
3. Verify formula: `pixel = focal_length * (point_3d - camera_pos)`
4. Full skeleton projection

**Results**: ✅ All tests passed
- Projection formula verified mathematically correct
- Skeleton extent with focal_length=10 at distance ~100: ~200 pixels
- Test output visualization: `test_camera_projection_output.png`

### Test 2: `debug_skeleton_projection.py`

**Purpose**: Compare projected vs observed coordinates to find bugs

**Critical Bug Found**: Perspective division error
```python
# BUG (lines 385-389): 
if y_proj[2] != 0:
    y_2d = y_proj[:2] / y_proj[2]  # Dividing by ~1000!
```

**Bug Impact**:
- Projected coordinates: `[-800, 0]`
- After division by y_proj[2] ≈ 1000: `[-0.8, 0]`
- After noise addition: `[-4.8, -0.4]`
- Result: Skeleton appeared 100× too small

**Fix Applied**: Removed perspective division
```python
y_2d = y_proj[:2]  # Correct for orthographic projection
```

**Verification After Fix**:
```
Camera 0 Projected: [-800.0, 0.0]
Camera 0 Observed:  [-802.1, -0.4]  ✅ Match!
```

### Test 3: `check_skeleton_extents.py`

**Purpose**: Analyze skeleton 3D structure and 2D extents at different timesteps

**Results**:
```
Timestep 0 (state=0, upright):
  3D extent: X=1.99, Y=1.81, Z=41.63 units
  Camera 0 (front-below): 2D extent u=17.7, v=14.2 pixels
  Camera 1 (side-level):  2D extent u=28.5, v=27.5 pixels
  Camera 2 (above):       2D extent u=17.1, v=20.4 pixels

Timestep 25 (state=2, sideways lean):
  3D extent: X=1.40, Y=28.92, Z=30.13 units
  Camera 0: 2D extent u=21.2, v=287.8 pixels  ✅ Much better
  Camera 1: 2D extent u=23.4, v=297.3 pixels
  Camera 2: 2D extent u=19.1, v=297.9 pixels

Timestep 50 (state=1, forward lean):
  3D extent: X=22.99, Y=2.80, Z=34.81 units
  Camera 0: 2D extent u=223.5, v=36.2 pixels  ✅ Much better
  Camera 1: 2D extent u=232.7, v=30.6 pixels
  Camera 2: 2D extent u=236.1, v=27.5 pixels

Timestep 75 (state=0, upright):
  3D extent: X=2.68, Y=1.67, Z=41.60 units
  Camera 0: 2D extent u=28.6, v=19.8 pixels
  Camera 1: 2D extent u=28.9, v=25.0 pixels
  Camera 2: 2D extent u=31.7, v=16.1 pixels
```

---

## 6. THE PROBLEM: User's Valid Concern

### User's Observation
"It looks like all three cameras are looking straight down on the body at time 0 and 75. Does that make sense?"

"No matter what direction the skeleton is pointing, no two cameras should be looking down/up at it unless they are colinear. Given three cameras, at least one should see a series of dots."

### User is Correct: Geometric Analysis

**Three camera positions**:
- Camera 0: `[80, 0, 70]` - Positioned to the +X side, slightly below center
- Camera 1: `[0, 80, 120]` - Positioned to the +Y side, above center
- Camera 2: `[-60, -60, 170]` - Positioned to -X/-Y side, well above center

**Skeleton at t=0 (state 0)**:
- Root: `[0, 0, 100]`
- Extends upward 42 units to `[0, 0, 142]`
- Direction: Pure Z-axis (vertical)

**Expected Behavior**:
1. Camera 0 (from +X side): Should see skeleton as vertical line in X-Z plane
2. Camera 1 (from +Y side): Should see skeleton as vertical line in Y-Z plane
3. Camera 2 (from above): Should see skeleton as point (top-down view)

**BUT TEST RESULTS SHOW**: All three cameras produce similar small 2D extents (~15-30 pixels), suggesting they all have similar views.

### Why This Doesn't Make Sense

With the given camera positions:
- Camera 0 at `[80, 0, 70]` is 30 units below the root at z=100
- Camera 1 at `[0, 80, 120]` is 20 units above the root
- Camera 2 at `[-60, -60, 170]` is 70 units above the root

These are NOT collinear. They should produce different views:
- At least one horizontal camera (0 or 1) should see the full vertical extent
- Camera 2 from above should see minimal extent

**But the projection formula**:
```
u = focal_length * (x - camera_x)
v = focal_length * (y - camera_y)
```

**DOES NOT USE THE Z-COORDINATE FOR PROJECTION!**

This means the camera's Z-position doesn't affect which part of the skeleton is "visible" - it's not a true perspective projection.

---

## 7. HYPOTHESIS: Orthographic Projection Issue

### The Core Problem

The projection formula treats all three dimensions equally:
```python
pixel_u = focal_length * (point_x - camera_x)
pixel_v = focal_length * (point_y - camera_y)
```

This is essentially:
1. Translate the 3D world so camera is at origin: `P' = P - C`
2. Scale by focal length: `pixel = f * P'`
3. **Drop the z-coordinate**

This is **orthographic projection** (parallel projection), NOT perspective projection.

### Why This Causes the Observed Behavior

For a vertical skeleton at `x=0, y=0, z=100 to 142`:

**Camera 0 at `[80, 0, 70]`**:
```
For each joint at [0, 0, z]:
  u = 10 * (0 - 80) = -800
  v = 10 * (0 - 0) = 0
```
All joints project to the same (u, v) because they have the same (x, y) coordinates!

**Camera 1 at `[0, 80, 120]`**:
```
For each joint at [0, 0, z]:
  u = 10 * (0 - 0) = 0
  v = 10 * (0 - 80) = -800
```
All joints project to the same (u, v) again!

**Camera 2 at `[-60, -60, 170]`**:
```
For each joint at [0, 0, z]:
  u = 10 * (0 - (-60)) = 600
  v = 10 * (0 - (-60)) = 600
```
All joints project to the same (u, v) yet again!

### Conclusion

**With pure orthographic projection, a vertical skeleton where all joints have the same (x, y) coordinates will project to a SINGLE POINT in every camera view, regardless of camera position.**

The small 2D extents we see (~15-30 pixels) are due to:
1. Random walk noise in root position
2. Small deviations in bone directions (kappa=5.0, not infinite)
3. Observation noise (std=3.0 pixels)

---

## 8. Verification Needed

### Manual Calculation for Timestep 0

From `check_skeleton_extents.py`, we should have:
- Root position at t=0
- Joint positions at t=0
- For each camera, compute expected pixel coordinates

### Expected vs Actual

If the skeleton is truly vertical with all joints at the same (x, y), then:
- All joints should project to nearly the same pixel coordinate
- This would appear as a cluster of dots
- This matches what we observe

### The Geometric Inconsistency

**User's concern is valid**: The projection formula doesn't match typical camera geometry where:
- A camera "looking from the side" should see a vertical line
- A camera "looking from above" should see a point

Instead, our orthographic projection:
- Projects all points with same (x, y) to same pixel location
- Camera z-position doesn't affect the image
- This is physically unrealistic for cameras

---

## 9. Potential Issues and Questions

### Q1: Is orthographic projection intended?

Most motion capture systems use **perspective projection**:
```
u = f * (x - cx) / (z - cz)
v = f * (y - cy) / (z - cz)
```

Our formula lacks the division by z-coordinate.

### Q2: Should we be using camera orientation?

Current implementation assumes camera always looks toward scene center. Real cameras have:
- Position (translation)
- Orientation (rotation matrix)
- Intrinsic parameters (focal length, principal point)

Full perspective projection:
```
P = K [R | t]
where K = intrinsic matrix (3×3)
      R = rotation matrix (3×3)
      t = translation vector (3×1)
```

### Q3: Why was perspective division removed?

In the bug fix (lines 385-390), we removed:
```python
y_2d = y_proj[:2] / y_proj[2]
```

This was correct for the current P matrix definition, but maybe P should be different?

### Q4: Is the P matrix correct?

Current: `P = [f*I | -f*C]`

Should it be: `P = K [R | -R^T * C]` where R is rotation matrix?

---

## 10. Recommendations for Investigation

1. **Verify skeleton positions at t=0**: Print actual (x, y, z) coordinates of all joints
2. **Manual projection calculation**: For each camera, manually compute expected pixel coords
3. **Compare with ground truth**: Check if `data.x_true` matches expectations
4. **Review projection assumptions**: Decide if orthographic is intended or if perspective needed
5. **Add camera orientation**: If perspective projection desired, add rotation matrices

---

## 11. Code Locations Summary

| Component | File | Lines | Function |
|-----------|------|-------|----------|
| Camera generation | `gimbal/synthetic_data.py` | 250-340 | `generate_camera_matrices()` |
| Projection formula | `gimbal/synthetic_data.py` | 380-395 | `generate_observations()` |
| Skeleton states | `gimbal/synthetic_data.py` | 82-130 | `generate_canonical_directions()` |
| Test: Projection math | `test_camera_projection.py` | 1-230 | Multiple test functions |
| Test: Debug projection | `debug_skeleton_projection.py` | 1-101 | Main script |
| Test: Extent analysis | `check_skeleton_extents.py` | 1-150 | Main script |

---

## 12. Test Commands

```powershell
# Run projection validation
pixi run python test_camera_projection.py

# Debug observed vs projected coords  
pixi run python debug_skeleton_projection.py

# Analyze skeleton extents
pixi run python check_skeleton_extents.py

# Run notebook demo
# Open: notebook/demo_v0_2_1_data_driven_priors.ipynb
```

---

## Appendix: Expected vs Actual Behavior

### What We Expected
- Camera 0 (side view): See skeleton as vertical line → Large v-extent, small u-extent
- Camera 1 (side view): See skeleton as vertical line → Large u-extent, small v-extent  
- Camera 2 (top view): See skeleton as point → Small u and v extents

### What We Observe
- Camera 0: Small extents in both u and v (~17×14 pixels)
- Camera 1: Small extents in both u and v (~28×27 pixels)
- Camera 2: Small extents in both u and v (~17×20 pixels)

### Conclusion
All cameras see similar "point cluster" views, which makes sense for orthographic projection of a vertical line, but NOT for perspective cameras at different positions looking at a 3D object.

---

**END OF INITIAL REPORT**

---

# ADDENDUM: Fix Implementation

**Date**: December 1, 2025  
**Status**: ✅ **FIXED**

## Summary of Changes

The camera projection system has been refactored to use **proper perspective projection with camera orientations**, fixing the fundamental issue identified in the original report.

### Root Cause

The original implementation used:
- **Axis-aligned cameras**: All cameras shared the same orientation (optical axis along global +Z)
- **Orthographic projection**: No perspective division (`y = P[:2]` instead of `y = P[:2]/P[2]`)
- **Mismatch**: Synthetic generator (orthographic) disagreed with PyMC model (perspective)

This caused all cameras to show similar "looking down" views of vertical skeletons.

### Solution Implemented

#### 1. Added Camera Utilities (`gimbal/camera_utils.py`)

New module with:
- `project_points_numpy()`: NumPy implementation of perspective projection matching PyTensor
- `build_look_at_matrix()`: Constructs rotation matrix R from camera position and target
- `build_projection_matrix()`: Creates full P = K[R|t] with proper orientation
- `camera_center_from_proj()`: Extracts camera position from P matrix

#### 2. Refactored Camera Generation

**File**: `gimbal/synthetic_data.py`, `generate_camera_matrices()`

**Old approach**:
```python
A = focal_length * I  # Axis-aligned
b = -A @ camera_pos
P = [A | b]
```

**New approach**:
```python
R = build_look_at_matrix(camera_pos, scene_center, up_world)
K = intrinsic_matrix(focal_length, principal_point)
t = -R @ camera_pos
P = K @ [R | t]
```

**Camera positions**:
- Camera 0: `[80, 0, 100]` - Front view (looks from +X toward center)
- Camera 1: `[0, 80, 100]` - Side view (looks from +Y toward center)
- Camera 2: `[0, 0, 180]` - Overhead view (looks from +Z down at center)

#### 3. Updated Observation Generation

**File**: `gimbal/synthetic_data.py`, `generate_observations()`

**Old (orthographic)**:
```python
y_proj = camera_proj[c] @ x_h
y_2d = y_proj[:2]  # No division
```

**New (perspective)**:
```python
y_proj = camera_proj[c] @ x_h
w = y_proj[2]
if abs(w) > 1e-6:
    y_2d = y_proj[:2] / w  # Perspective division
```

### Verification Tests

#### Test 1: Projection Consistency ✅
**File**: `tests/test_synthetic_projection_consistency.py`

Verified that `generate_observations()` and `project_points_numpy()` produce identical results:
- RMSE: 0.000000 pixels (machine precision)
- Confirms generative and inference paths agree

#### Test 2: Vertical Skeleton Views ✅
**File**: `tests/test_vertical_skeleton_views.py`

For a purely vertical skeleton (42 units tall, ~0 units wide):
- Camera 0 (front): **5.25 pixels extent** ✅ (sees vertical line)
- Camera 1 (side): **5.25 pixels extent** ✅ (sees vertical line)
- Camera 2 (overhead): **0.00 pixels extent** ✅ (sees point)

This confirms cameras now produce geometrically correct views!

#### Test 3: DLT Round-Trip ✅
**File**: `tests/test_dlt_round_trip.py`

Verified projection and triangulation are mutually consistent:
- With 1.0 pixel noise: Mean error = 8.35 units (expected ~8 from noise geometry)
- With 0.1 pixel noise: Mean error = 0.84 units (expected ~0.8)
- Reconstruction error matches theoretical prediction from noise level

### Key Results

| Metric | Old (Orthographic) | New (Perspective) |
|--------|-------------------|-------------------|
| Camera orientations | All axis-aligned | Proper look-at with rotations |
| Vertical skeleton view | All show ~15px clusters | Front/side: 5px lines, overhead: 0px point |
| Synthetic vs PyMC | **Mismatched** | ✅ **Matched** |
| DLT round-trip | Large systematic error | Error matches noise level |

### Impact on Demo Notebook

The notebook `demo_v0_2_1_data_driven_priors.ipynb` will now show:
- **Correct camera views**: Different cameras show different perspectives
- **Accurate triangulation**: 3D reconstruction matches ground truth within noise bounds
- **Consistent pipeline**: All stages (projection, triangulation, inference) use same model

### Files Modified

1. **New**: `gimbal/camera_utils.py` (208 lines) - Camera geometry utilities
2. **Modified**: `gimbal/synthetic_data.py` - Camera generation and observation projection
3. **New**: `tests/test_synthetic_projection_consistency.py` - Verify generator/projector match
4. **New**: `tests/test_vertical_skeleton_views.py` - Verify geometric correctness
5. **New**: `tests/test_dlt_round_trip.py` - Verify projection/triangulation consistency

### Running the Tests

```powershell
# All tests pass:
pixi run python tests/test_synthetic_projection_consistency.py  # ✅
pixi run python tests/test_vertical_skeleton_views.py            # ✅
pixi run python tests/test_dlt_round_trip.py                     # ✅
```

### Theoretical Validation

For perspective projection with focal length `f`, distance `d`, and pixel noise `σ`:
- **Depth uncertainty**: `Δz ≈ σ · d / f`
- **With our parameters**: `Δz ≈ 1.0 · 80 / 10 = 8 units`
- **Observed**: Mean error = 8.35 units ✅

This confirms the implementation is mathematically correct.

---

## Conclusion

The camera projection system now implements **proper perspective projection with camera orientations**, making it consistent with:
- Standard computer vision (DLT triangulation)
- PyMC inference model (perspective division)
- Geometric expectations (different cameras see different views)

The original issue - all cameras showing similar "looking down" views - is completely resolved. Each camera now has its own orientation determined by a look-at matrix, and perspective division is applied consistently throughout the pipeline.

**Status**: Production-ready. All tests passing.

**END OF REPORT**
