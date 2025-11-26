# Stage 1 Specification: Generic Collapsed HMM in PyTensor/PyMC

This document specifies **Stage 1** of the new HMM architecture. It is intended for direct use by Copilot / Sonnet 4.5. It contains complete algorithms, code specifications, naming conventions, and testing procedures.

---

# 1. Mathematical Specification

We construct a **collapsed Hidden Markov Model (HMM)** in PyTensor:

* Discrete states (z_t \in {0,\dots,S-1})
* Observations (y_t)
* Emissions parameterized by arbitrary parameters (\theta)

We do **not** sample (z_t). Instead, we compute:

[
\log p(y_{0:T-1} \mid \pi, A, \theta)
= \log \sum_{z_{0:T-1}} p(z_0)\prod_{t=1}^{T-1}p(z_t|z_{t-1})\prod_{t=0}^{T-1}p(y_t|z_t)
]

using a **forward recursion in log-space**.

Inputs:

* `logp_emit[t, s]  =  log p(y_t | z_t = s, θ)`
* `logp_init[s]     =  log π_s`
* `logp_trans[i, j] =  log A_{ij}`

Forward recursion:

* Initialization:

[
\alpha_{0,s} = \log \pi_s + \ell_{0,s}
]

* Recursion:

[
\alpha_{t,j} = \ell_{t,j} + \log\sum_i\exp(\alpha_{t-1,i} + \log A_{ij})
]

* Final collapsed log-likelihood:

[
\log p(y|\theta) = \log \sum_j \exp(\alpha_{T-1,j})
]

---

# 2. Data & Shape Conventions

Let:

* `T` = number of time steps
* `S` = number of states

Tensor shapes:

* `logp_emit`: `(T, S)`
* `logp_init`: `(S,)`
* `logp_trans`: `(S, S)`

Stage 1 supports a **single sequence**.

---

# 3. Module Layout

Create a new module:

```
hmm_pytensor.py
```

containing at least two functions:

* `forward_log_prob_single` — core forward algorithm
* `collapsed_hmm_loglik` — wrapper

---

# 4. Function Specifications

## 4.1 `forward_log_prob_single`

**Location:** `hmm_pytensor.py`

**Signature:**

```python
import pytensor.tensor as pt

def forward_log_prob_single(
    logp_emit: pt.TensorVariable,   # shape (T, S)
    logp_init: pt.TensorVariable,   # shape (S,)
    logp_trans: pt.TensorVariable,  # shape (S, S)
) -> pt.TensorVariable:
    """
    Compute collapsed HMM log-likelihood for one sequence using
    the forward algorithm in log-space.

    Returns
    -------
    logp : scalar TensorVariable
        The collapsed log-likelihood log p(y[0:T-1] | params)
    """
```

**Implementation details:**

1. Extract shapes:

```python
T = logp_emit.shape[0]
S = logp_emit.shape[1]
```

2. Initialization:

```python
alpha_prev = logp_init + logp_emit[0]   # (S,)
```

3. Define step function:

```python
def step(alpha_prev, logp_emit_t):
    # alpha_prev: (S,)
    # logp_trans: (S, S)
    # alpha_prev[:, None] + logp_trans: (S, S)
    alpha_pred = pt.logsumexp(alpha_prev.dimshuffle(0, "x") + logp_trans, axis=0)
    alpha_t = logp_emit_t + alpha_pred
    return alpha_t
```

4. Run recursion using `pt.scan`:

```python
alpha_all, _ = pt.scan(
    fn=step,
    sequences=logp_emit[1:],
    outputs_info=alpha_prev,
)
alpha_last = alpha_all[-1]
logp = pt.logsumexp(alpha_last)
return logp
```


---

## 4.2 `collapsed_hmm_loglik`

A small wrapper for clarity.

**Signature:**

```python
def collapsed_hmm_loglik(
    logp_emit: pt.TensorVariable,   # (T, S)
    logp_init: pt.TensorVariable,   # (S,)
    logp_trans: pt.TensorVariable,  # (S, S)
) -> pt.TensorVariable:
    """Public API for collapsed HMM log-likelihood."""
```

**Implementation:**

```python
return forward_log_prob_single(logp_emit, logp_init, logp_trans)
```

---

# 5. PyMC Wrapper Example

Below is an example model for **1D Gaussian emission HMM**.

**Model builder:**

```python
import pymc as pm
import pytensor.tensor as pt
from hmm_pytensor import collapsed_hmm_loglik

def build_gaussian_hmm_model(y, S):
    T = y.shape[0]

    with pm.Model() as model:
        # Unconstrained initial + transition logits
        init_logits = pm.Normal("init_logits", 0, 1, shape=S)
        trans_logits = pm.Normal("trans_logits", 0, 1, shape=(S, S))

        # Normalize to log-probabilities
        logp_init = init_logits - pm.math.logsumexp(init_logits)
        logp_trans = trans_logits - pm.math.logsumexp(trans_logits, axis=1, keepdims=True)

        # Emission parameters
        mu = pm.Normal("mu", 0, 5, shape=S)
        sigma = pm.Exponential("sigma", 1.0)

        # Build logp_emit (T, S)
        y_t = pt.shape_padright(y, 1)   # (T,1)
        mu_s = mu.dimshuffle("x", 0)  # (1,S)
        logp_emit = pm.logp(pm.Normal.dist(mu_s, sigma), y_t)  # (T,S)

        # Collapsed HMM likelihood
        hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
        pm.Potential("hmm_loglik", hmm_ll)

    return model
```

---

# 6. Synthetic Data Generator

In the demo notebook, define:

```python
import numpy as np

def simulate_gaussian_hmm(T, S, mu_true, sigma_true, pi_true, A_true, random_state=None):
    rng = np.random.default_rng(random_state)

    z = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)

    # Initial
    z[0] = rng.choice(S, p=pi_true)
    y[0] = rng.normal(mu_true[z[0]], sigma_true)

    for t in range(1, T):
        z[t] = rng.choice(S, p=A_true[z[t-1]])
        y[t] = rng.normal(mu_true[z[t]], sigma_true)

    return y, z
```

---

# 7. Demo Notebook: `hmm_demo_gaussians`

Workflow:

1. Simulate data using the generator.
2. Build PyMC model with `build_gaussian_hmm_model`.
3. Sample using nutpie or NUTS.
4. Check parameter recovery.
5. (Optional) Posterior decoding using forward–backward outside PyMC.

---

# 8. Tests

Add tests either in a file or notebook:

### 8.1 Forward algorithm correctness

* For tiny HMM (S=2, T=3):

  * Manually enumerate all state sequences in NumPy.
  * Compare with `forward_log_prob_single`.

### 8.2 Numerical stability

* Use random parameters; ensure finite log-likelihood.

### 8.3 PyMC integration

* Build small model; evaluate logp & grads; ensure no NaNs.

---

# 9. Coding Style and Names

* Use PyTensor as `pt`.
* Function names:

  * `forward_log_prob_single`
  * `collapsed_hmm_loglik`
  * `build_gaussian_hmm_model`
  * `simulate_gaussian_hmm`

---

# 10. Summary of Required Outputs

1. `hmm_pytensor.py` with:

   * `forward_log_prob_single`
   * `collapsed_hmm_loglik`

2. A PyMC model builder using the collapsed HMM.

3. Synthetic HMM generator.

4. Notebook demonstrating end-to-end recovery.

This completes **Stage 1** and prepares the system for Stage 2.
