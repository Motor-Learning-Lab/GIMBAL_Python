# Blended Local Coordinate Frames for Anatomically Meaningful Joint Priors

This document describes a stable and differentiable method for constructing **local anatomical coordinate frames** for each joint in a kinematic chain, suitable for use in Stage‑7 of the GIMBAL–PyMC model. The design combines:

1. **Anatomical frames** derived from parent and grandparent bone directions (v₂, v₁)
2. **Parallel‑transported frames** that move smoothly with the parent bone over time
3. **Smooth blending** between the two, avoiding discontinuities and degeneracies when v₁ and v₂ become nearly colinear.

The result is a local frame that is:
- Anatomically meaningful where possible
- Smooth and stable during singular configurations
- Fully deterministic (not sampled)
- Differentiable for PyTensor and HMC
- Suitable for defining canonical relative directions and hierarchical priors

---

## 1. Inputs

For each joint **k** at each time **t**, we assume:

- **Parent direction**:
  $$\mathbf{v}_2(t) = \text{direction of parent bone (unit vector)}$$

- **Grandparent direction**:
  $$\mathbf{v}_1(t) = \text{direction of grandparent bone (unit vector)}$$

- **Previous transported frame**:
  $$F_{t-1}^{PT} = [\mathbf{e}_x^{PT}(t-1), \mathbf{e}_y^{PT}(t-1), \mathbf{e}_z^{PT}(t-1)]$$

At time **t = 1**, we initialize:

$$F_1^{PT} = F_1^{anat}$$

---

## 2. Anatomical Frame from v₁ and v₂

Based on the construction in `coordinates.md`, the local anatomical coordinate system is:

1. **x-axis aligned with parent bone:**
   $$
   \mathbf{e}_x^{anat} = \frac{\mathbf{v}_2}{\|\mathbf{v}_2\|}
   $$

2. **y-axis in the plane of v₁ and v₂:**
   $$
   \mathbf{y}_0 = \mathbf{v}_1 - (\mathbf{v}_1 \cdot \mathbf{e}_x^{anat}) \, \mathbf{e}_x^{anat}
   $$
   $$
   \mathbf{e}_y^{anat} = \frac{\mathbf{y}_0}{\|\mathbf{y}_0\|}
   $$

3. **z-axis by right-hand rule:**
   $$
   \mathbf{e}_z^{anat} = \mathbf{e}_x^{anat} \times \mathbf{e}_y^{anat}
   $$

This frame is anatomically meaningful **when v₁ and v₂ are not colinear**.

---

## 3. Parallel Transport Frame

Parallel transport produces a smooth frame that moves consistently with the parent bone direction.

Given:
- Previous x-axis: $$\mathbf{e}_x^{PT}(t-1)$$
- New parent direction: $$\mathbf{v}_2(t)$$

Define:

1. **New x-axis:**
   $$
   \mathbf{e}_x' = \frac{\mathbf{v}_2(t)}{\|\mathbf{v}_2(t)\|}
   $$

2. **Minimal rotation** mapping old x-axis to new:
   - Let axis:
     $$
     \mathbf{a} = \mathbf{e}_x^{PT}(t-1) \times \mathbf{e}_x'
     $$
   - Let angle:
     $$
     \theta = \arccos(\mathbf{e}_x^{PT}(t-1) \cdot \mathbf{e}_x')
     $$

   - Rodrigues formula:
     $$
     R_t = I + [\mathbf{a}]_\times + \frac{[\mathbf{a}]_\times^2}{1 + \mathbf{e}_x^{PT}(t-1) \cdot \mathbf{e}_x'}
     $$

3. **Transport the frame:**
   $$
   F_t^{PT} = R_t\,F_{t-1}^{PT}
   $$

This frame is always defined and always smooth.

---

## 4. Blending Anatomical and Transported Frames

We now construct a **blended frame**:

$$
F_t = [\mathbf{e}_x(t), \mathbf{e}_y(t), \mathbf{e}_z(t)]
$$

### 4.1 Colinearity Measure

We detect how reliable the anatomical frame is by measuring:

$$
 c_t = \| \mathbf{v}_1(t) \times \mathbf{v}_2(t) \|
$$

- If **cₜ ≈ 1**, bones are well-separated → anatomical frame is good.
- If **cₜ → 0**, bones are colinear → anatomical frame becomes unstable.

Define a smooth weight:

$$
\alpha_t = \frac{c_t}{c_t + \varepsilon}
$$

with \(\varepsilon\) small (e.g., 1e−2).

### 4.2 Blending

Use the same x-axis for both frames:

$$
\mathbf{e}_x(t) = \mathbf{e}_x^{anat}(t) = \mathbf{e}_x^{PT}(t)
$$

Blend the transverse directions:

$$
\tilde{\mathbf{e}}_y(t) = \alpha_t \, \mathbf{e}_y^{anat}(t)
                      + (1-\alpha_t) \, \mathbf{e}_y^{PT}(t)
$$

Normalize:

$$
\mathbf{e}_y(t) = \frac{\tilde{\mathbf{e}}_y(t)}{\| \tilde{\mathbf{e}}_y(t) \|}
$$

Complete the frame:

$$
\mathbf{e}_z(t) = \mathbf{e}_x(t) \times \mathbf{e}_y(t)
$$

This ensures:
- When anatomical data is reliable, \(F_t\) is almost anatomical
- When anatomical data is degenerate, \(F_t\) transitions smoothly to the transported frame
- Frames vary smoothly with time and pose

---

## 5. Using the Blended Frame for Canonical Priors

For the child joint direction:

$$\mathbf{u}_k(t)$$

Compute local coordinates:

$$
\mathbf{w}_k(t) = F_t^T\mathbf{u}_k(t)
$$

Define a hierarchical prior:

- Canonical direction:
  $$\mu_k \sim \mathcal{N}(m_k, \tau^2 I)$$

- Stiffness:
  $$\kappa_k \sim \text{HalfNormal}(\sigma_\kappa)$$

- vMF-like likelihood:
  $$
  \log p(\mathbf{w}_k(t) \mid \mu_k, \kappa_k)
  \propto \kappa_k \, (\mathbf{w}_k(t) \cdot \mu_k)
  $$

Combined with temporal smoothness on \(\mathbf{u}_k(t)\), this forms the core of Stage‑7 biomechanical priors.

---

## 6. Summary

The **blended local frame** approach:
- Preserves anatomical meaning when v₁ and v₂ are not colinear
- Remains stable and smooth during degenerate configurations
- Produces deterministic, differentiable frames for PyMC
- Supports clean hierarchical priors over canonical pose and stiffness

This design is recommended for Stage‑7 of the GIMBAL–PyMC model.

