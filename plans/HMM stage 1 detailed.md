# Stage 1 — Detailed Specification: Collapsed HMM Engine in PyTensor/PyMC

This document provides a **complete, implementation-ready specification** of Stage 1. It is intended for Copilot / Sonnet 4.5 and contains algorithms, file structure, tests, diagnostics, naming conventions, and detailed step-by-step instructions.

Stage 1 is fully **independent** of:

* Cameras
* Skeletons
* 3D kinematic chains
* Joint directions

Its purpose is to produce a reusable **collapsed HMM log-likelihood** module that can be dropped into later stages.

---

# 1. Mathematical Definition

We define a standard HMM with:

* States: `z_t ∈ {0,…,S−1}` for `t = 0,…,T−1`
* Observations: `y_t`
* Initial state distribution: `π[s] = p(z_0 = s)`
* Transition matrix: `A[i,j] = p(z_t = j | z_{t−1} = i)`
* Emission distribution: `p(y_t | z_t = s, θ)`

We do **not** sample the states. Instead we compute the collapsed likelihood:

```
log p(y | π, A, θ) = log Σ_z p(z_0) Π p(z_t | z_{t−1}) Π p(y_t | z_t)
```

via the **forward algorithm in log-space**.

## Inputs

* `logp_emit[t, s]`: log p(y_t | z_t=s, θ)
* `logp_init[s]`    = log π_s
* `logp_trans[i,j]` = log A_{i→j}

## Output

A scalar PyTensor variable representing the collapsed log-likelihood.

---

# 2. Required Shapes

Let:

* `T` = number of time steps
* `S` = number of states

Then:

* `logp_emit.shape == (T, S)`
* `logp_init.shape == (S,)`
* `logp_trans.shape == (S, S)`

Stage 1 supports **one sequence only** (batch dimension optional later).

---

# 3. File Structure

Create a new file:

```
hmm_pytensor.py
```

which defines the following public API:

* `forward_log_prob_single(logp_emit, logp_init, logp_trans)`
* `collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)`

No references to the rest of the GIMBAL model appear in Stage 1.

---

# 4. Implementation Details

## 4.1 `forward_log_prob_single`

**Purpose:** perform the log-space forward recursion.

### Function Signature

```
def forward_log_prob_single(
    logp_emit,   # (T, S)
    logp_init,   # (S,)
    logp_trans,  # (S, S)
):
    """
    Compute collapsed HMM log-likelihood for one sequence using the
    forward algorithm in log-space.
    Returns a scalar PyTensor variable.
    """
```

### Step 1 — Initialization

```
alpha_prev = logp_init + logp_emit[0]    # shape (S,)
```

### Step 2 — Define one forward step

```
def step(alpha_prev, logp_emit_t):
    # alpha_prev: (S,)
    # logp_trans: (S, S)

    # alpha_prev[i] + log A[i,j]
    alpha_pred = pt.logsumexp(
        alpha_prev.dimshuffle(0, "x") + logp_trans,
        axis=0,
    )

    alpha_t = logp_emit_t + alpha_pred
    return alpha_t
```

### Step 3 — Run recursion with `pt.scan`

```
alpha_all, _ = pt.scan(
    fn=step,
    sequences=logp_emit[1:],
    outputs_info=alpha_prev,
)
```

### Step 4 — Final state (`T=1` handled automatically)

```
alpha_last = pt.switch(
    pt.eq(logp_emit.shape[0], 1),
    alpha_prev,
    alpha_all[-1],
)
```

### Step 5 — Collapse over final states

```
logp = pt.logsumexp(alpha_last)
return logp
```

---

## 4.2 `collapsed_hmm_loglik`

```
def collapsed_hmm_loglik(logp_emit, logp_init, logp_trans):
    return forward_log_prob_single(logp_emit, logp_init, logp_trans)
```

This wrapper is provided for readability and future extension.

---

# 5. PyMC Example Model (Gaussian Emissions)

This demonstrates how to use the collapsed HMM in PyMC.

### Public API

Implement in a new file or notebook:

* `build_gaussian_hmm_model(y, S)`
* `simulate_gaussian_hmm(T, S, mu_true, sigma_true, pi_true, A_true)`

## 5.1 Gaussian Emission Model Builder

```
def build_gaussian_hmm_model(y, S):
    T = y.shape[0]
    with pm.Model() as model:
        # Initial and transition logits
        init_logits = pm.Normal("init_logits", 0, 1, shape=S)
        trans_logits = pm.Normal("trans_logits", 0, 1, shape=(S, S))

        # Normalize to log-probabilities
        logp_init = init_logits - pm.math.logsumexp(init_logits)
        logp_trans = trans_logits - pm.math.logsumexp(trans_logits, axis=1, keepdims=True)

        # Emission parameters
        mu = pm.Normal("mu", 0, 5, shape=S)
        sigma = pm.Exponential("sigma", 1.0)

        # Construct logp_emit (T, S)
        y_t = pt.shape_padright(y, 1)   # (T, 1)
        mu_s = mu.dimshuffle("x", 0)    # (1, S)
        logp_emit = pm.logp(pm.Normal.dist(mu_s, sigma), y_t)  # (T, S)

        # Collapsed HMM
        hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
        pm.Potential("hmm_loglik", hmm_ll)

    return model
```

---

# 6. Synthetic Gaussian HMM Generator

```
def simulate_gaussian_hmm(T, S, mu_true, sigma_true, pi_true, A_true, random_state=None):
    rng = np.random.default_rng(random_state)
    z = np.zeros(T, dtype=int)
    y = np.zeros(T)

    # Initial
    z[0] = rng.choice(S, p=pi_true)
    sigma_arr = np.broadcast_to(sigma_true, (S,))
    y[0] = rng.normal(mu_true[z[0]], sigma_arr[z[0]])

    # Transitions
    for t in range(1, T):
        z[t] = rng.choice(S, p=A_true[z[t-1]])
        y[t] = rng.normal(mu_true[z[t]], sigma_arr[z[t]])

    return y, z
```

---

# 7. Demo Notebook — `hmm_demo_gaussians.ipynb`

This notebook should:

1. Simulate data (`simulate_gaussian_hmm`).
2. Build the model (`build_gaussian_hmm_model`).
3. Sample with nutpie if available, otherwise PyMC NUTS.
4. Plot diagnostics:

   * ESS, R-hat
   * Posterior means for `mu`, `sigma`
5. Optional: perform Viterbi decoding outside PyMC for qualitative inspection.

---

# 8. Validation Tests

## 8.1 Tiny HMM brute-force

* Choose `S=2`, `T=3`.
* Manually enumerate 8 state sequences.
* Compare NumPy log p(y | params) to PyTensor result.
* Must match within `1e-6`.

## 8.2 T=1 edge case

* `logp_emit` is `(1, S)`.
* Check:

```
log p = logsumexp(logp_init + logp_emit[0])
```

## 8.3 Gradient sanity check

* Call `model.compile_dlogp()` and evaluate.
* All gradients must be finite; no NaNs.

## 8.4 Performance sanity check

* Use moderate `T` (100–300) and `S` (2–4).
* Check sampling speed under nutpie and stability of logp.

---

# 9. Coding Style & Conventions

* Import PyTensor as `import pytensor.tensor as pt`.
* Public functions:

  * `forward_log_prob_single`
  * `collapsed_hmm_loglik`
  * `build_gaussian_hmm_model`
  * `simulate_gaussian_hmm`
* Avoid references to skeleton, cameras, or directions.
* Keep Stage 1 orthogonal to all later stages.

---

# 10. Completion Criteria

Stage 1 is *complete* when the following are true:

* `hmm_pytensor.py` exists and passes tiny-HMM and T=1 tests.
* Gaussian emission model compiles and samples without NaNs or divergences.
* Demo notebook shows correct parameter recovery on synthetic data.
* Forward algorithm has stable gradients.

At that point, Stage 2 (refactor of emissions and kinematic pipelines) can begin.
