Yes — you can do this *entirely* with dot- and cross-product formulas and **no conditional logic**, by constructing an orthonormal basis that satisfies your orientation constraints.

You want a coordinate system:

* **x-axis** : along **v₂**
* **y-axis** : in the span of {v₁, v₂} and having **positive dot product with v₁**
* **z-axis** : right-hand rule from (x, y)

Below is a clean construction.

---

# Step 1 — Normalize v₂ to get the x-axis

[
\mathbf{e}_x = \frac{\mathbf{v}_2}{|\mathbf{v}_2|}
]

---

# Step 2 — Construct the y-axis as the normalized component of v₁ orthogonal to v₂

Project v₁ onto the plane perpendicular to v₂:

[
\mathbf{y}_0 = \mathbf{v}_1 - (\mathbf{v}_1 \cdot \mathbf{e}_x), \mathbf{e}_x
]

Normalize:

[
\mathbf{e}_y = \frac{\mathbf{y}_0}{|\mathbf{y}_0|}
]

Because **y₀ is v₁ with its v₂-component removed**, and has positive dot product with v₁ automatically, you don’t need conditional logic.

---

# Step 3 — Define the z-axis using the right-hand rule

[
\mathbf{e}_z = \mathbf{e}_x \times \mathbf{e}_y
]

This is automatically orthonormal if v₁ and v₂ are not collinear.

---

# Step 4 — Coordinates of w in this basis

For any vector **w**, its coordinates (**wₓ**, **wᵧ**, **w_z**) in the {eₓ, eᵧ, e_z} system are:

[
\begin{aligned}
w_x &= \mathbf{w}\cdot\mathbf{e}_x[4pt]
w_y &= \mathbf{w}\cdot\mathbf{e}_y[4pt]
w_z &= \mathbf{w}\cdot\mathbf{e}_z
\end{aligned}
]

All dot and cross products, no conditionals.

---

# Final compact formula set

[
\boxed{
\mathbf{e}_x=\frac{\mathbf{v}_2}{|\mathbf{v}_2|}
}
]

[
\boxed{
\mathbf{e}_y=\frac{\mathbf{v}_1 - (\mathbf{v}_1\cdot \mathbf{e}_x)\mathbf{e}_x}
{\left|\mathbf{v}_1 - (\mathbf{v}_1\cdot \mathbf{e}_x)\mathbf{e}_x\right|}
}
]

[
\boxed{\mathbf{e}_z = \mathbf{e}_x \times \mathbf{e}_y}
]

[
\boxed{
(w_x,w_y,w_z)=
(\mathbf{w}\cdot\mathbf{e}_x,;
\mathbf{w}\cdot\mathbf{e}_y,;
\mathbf{w}\cdot\mathbf{e}_z)
}
]

---

# Notes

* **No sign flips or branching are needed**; the constructed eᵧ is always the “positive” vector in the v₁–v₂ plane because it is literally the projection of v₁ onto the orthogonal direction.
* Works for any non-collinear v₁, v₂.
* (Optional) If v₁ and v₂ *are* collinear, then the construction degenerates; but you explicitly asked for no conditional logic, so this is the standard branch-free formulation.

---

If you want, I can also give you:

* a single matrix expression
* code in NumPy / PyTorch / JAX
* a fully normalized derivative-stable version (important for GIMBAL-like applications)
