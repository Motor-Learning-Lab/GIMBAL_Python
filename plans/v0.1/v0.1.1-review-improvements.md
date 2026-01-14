# Stage 1 Specification Improvements: Generic Collapsed HMM in PyTensor/PyMC

This document contains recommended improvements to the Stage 1 HMM implementation plan before beginning development.

---

## Overall 3-Stage Plan Review

### Strengths
- Clear separation of concerns across stages
- Building blocks approach (generic HMM → refactoring → integration)
- Maintains PyTensor/nutpie compatibility throughout

### Suggested Improvements for 3-Stage Plan

**Stage 1 Enhancements:**
1. **Add batch support from the start** — Design `logp_emit` to support `(B, T, S)` batches for multiple sequences (will help Stage 3 with multiple joints)
2. **Include Viterbi decoding** — Add `viterbi_decode` function for most likely state sequence recovery
3. **Specify numerical stability requirements** — Document behavior with extreme log-probabilities

**Stage 2 Clarifications:**
1. **Clarify "parent-relative" semantics** — Specify handling for root joint (no parent)
2. **Add intermediate milestone** — Split into 2a (refactor emission) and 2b (coordinate transforms)
3. **Specify coordinate frame convention** — Choose rotation representation (Rodrigues/quaternions/matrices)

**Stage 3 Critical Additions:**
1. **State initialization strategy** — How to initialize K×S canonical directions? (Random? K-means on observed directions?)
2. **Dimension scalability** — For K=20, S=5: 300 direction parameters. Consider hierarchical priors or tied states
3. **Label switching problem** — Need post-hoc relabeling, ordered constraints, or reference state with fixed direction

---

## Stage 1 Detailed Specification Improvements

### Critical Issues to Address

**1. Gradient Validation (Add to Section 8)**
```
### 8.4 Gradient correctness
- Use pytensor.grad to compute gradients w.r.t. logp_init, logp_trans, mu, sigma
- Compare with finite differences
- Ensure gradients flow correctly through logsumexp operations
- Test gradient w.r.t. emission parameters
```

**2. Edge Case Handling (Add to Section 4.1)**

After initialization step, add:
```python
# Handle T=1 edge case (no recursion needed)
if T == 1:
    return pt.logsumexp(logp_init + logp_emit[0])

# Ensure logp_init is normalized
logp_init_norm = logp_init - pt.logsumexp(logp_init)
alpha_prev = logp_init_norm + logp_emit[0]
```

**3. Emission Function Flexibility (Enhance Section 5)**

Add more general model builder:
```python
def build_generic_hmm_model(y, S, logp_emit_fn):
    """
    Build HMM model with custom emission function.
    
    Parameters
    ----------
    y : array-like
        Observed data
    S : int
        Number of states
    logp_emit_fn : callable
        Function (y, state_params) -> (T, S) log-emission tensor
    """
    with pm.Model() as model:
        # ... HMM priors ...
        logp_emit = logp_emit_fn(y, state_params)
        hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
        pm.Potential("hmm_loglik", hmm_ll)
    return model
```

**4. State-Specific vs. Shared Emission Parameters (Section 5)**

Current code has shared `sigma` across states. Clarify options:
```python
# Option A: Shared variance (current)
sigma = pm.Exponential("sigma", 1.0)  # scalar

# Option B: State-specific variance (more flexible)
sigma = pm.Exponential("sigma", 1.0, shape=S)
logp_emit = pm.logp(pm.Normal.dist(mu_s, sigma.dimshuffle("x", 0)), y_t)
```

**5. Enhanced Demo Notebook Diagnostics (Section 7)**

Add to workflow:
```
6. Check effective sample size (ESS) for all parameters
7. Visualize posterior predictive: sample z_t sequences, compare to true
8. Plot transition matrix posterior as heatmap with credible intervals
9. Check parameter identifiability (label switching check)
10. Trace plots for init_logits and trans_logits convergence
```

---

## New Sections to Add

### Section 4.3: Viterbi Decoding

Add after `collapsed_hmm_loglik`:

```python
def viterbi_decode(
    logp_emit: pt.TensorVariable,   # shape (T, S)
    logp_init: pt.TensorVariable,   # shape (S,)
    logp_trans: pt.TensorVariable,  # shape (S, S)
) -> pt.TensorVariable:
    """
    Compute most likely state sequence using Viterbi algorithm.
    
    Returns
    -------
    z_map : (T,) integer tensor
        Most likely state at each time step
        
    Notes
    -----
    Uses max-product algorithm (log-domain) with backtracking.
    """
    # Implementation details:
    # 1. Forward pass with max instead of logsumexp
    # 2. Track backpointers
    # 3. Backward pass to reconstruct path
    
    # Can be deferred to Stage 1b if too complex initially
```

### Section 11: Performance Benchmarks

```markdown
# 11. Performance Benchmarks

Expected performance for Stage 1 validation:

**Compilation:**
- Forward pass compilation: < 5 seconds
- Model compilation with nutpie: < 10 seconds

**Evaluation:**
- Forward pass evaluation: < 0.1 seconds for T=100, S=5
- Forward pass evaluation: < 1 second for T=1000, S=10

**Sampling:**
- 1000 draws with nutpie: < 60 seconds for 1D Gaussian HMM
- Effective sample size (ESS): > 100 for mu parameters
- R-hat: < 1.01 for all parameters

**Failure modes to check:**
- T > 5000: May need chunked forward algorithm
- S > 20: Dense transition matrix becomes unwieldy
- Extreme log-probabilities (< -1e10): Check for NaN/Inf
```

### Section 12: Known Limitations

```markdown
# 12. Known Limitations

Stage 1 deliberately excludes features needed for later stages:

**Scope limitations:**
1. Single sequence only (no batch dimension for multiple sequences)
2. Dense transition matrix (no sparsity support for large S)
3. No backward pass (no posterior state marginals p(z_t | y))
4. No input-dependent transitions (assumes stationary dynamics)
5. No structured state spaces (no factorial HMMs, DBNs, etc.)

**To be addressed in:**
- Batch support: Stage 1b or early Stage 2
- Backward pass: Stage 3 (for smooth state inference)
- Sparse transitions: Future optimization if S > 50
```

---

## Code Corrections

### Section 4.1: Clarify Summation in Step Function

```python
def step(alpha_prev, logp_emit_t):
    """
    Forward step: alpha_t[j] = logp_emit[t,j] + log sum_i exp(alpha[t-1,i] + log A[i,j])
    
    Parameters
    ----------
    alpha_prev : (S,) tensor
        Log forward probabilities at time t-1
    logp_emit_t : (S,) tensor
        Log emission probabilities at time t
        
    Returns
    -------
    alpha_t : (S,) tensor
        Log forward probabilities at time t
    """
    # alpha_prev: (S,)
    # logp_trans: (S, S) where logp_trans[i,j] = log p(z_t=j | z_{t-1}=i)
    # Broadcast: alpha_prev[:, None] gives (S, 1), logp_trans is (S, S)
    # Sum over axis=0 sums over previous states i
    alpha_pred = pt.logsumexp(alpha_prev[:, None] + logp_trans, axis=0)  # (S,)
    alpha_t = logp_emit_t + alpha_pred
    return alpha_t
```

### Section 6: Add Label-Switching Check to Generator

```python
def simulate_gaussian_hmm(T, S, mu_true, sigma_true, pi_true, A_true, random_state=None):
    """
    Simulate 1D Gaussian HMM.
    
    Parameters
    ----------
    T : int
        Number of time steps
    S : int
        Number of states
    mu_true : array (S,)
        State-specific means
    sigma_true : float
        Observation noise (shared across states)
    pi_true : array (S,)
        Initial state probabilities (must sum to 1)
    A_true : array (S, S)
        Transition matrix (rows must sum to 1)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    y : array (T,)
        Observed sequence
    z : array (T,) of int
        True hidden state sequence
        
    Notes
    -----
    States are arbitrary labels. Permuting (mu, pi, A) gives equivalent model.
    """
    rng = np.random.default_rng(random_state)
    
    # Validate inputs
    assert np.isclose(pi_true.sum(), 1.0), "pi must sum to 1"
    assert np.allclose(A_true.sum(axis=1), 1.0), "A rows must sum to 1"
    
    z = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)

    # Initial state
    z[0] = rng.choice(S, p=pi_true)
    y[0] = rng.normal(mu_true[z[0]], sigma_true)

    # Sequence
    for t in range(1, T):
        z[t] = rng.choice(S, p=A_true[z[t-1]])
        y[t] = rng.normal(mu_true[z[t]], sigma_true)

    return y, z
```

---

## Questions to Clarify Before Implementation

1. **Target sequence length T?**
   - If T > 1000, scan-based forward may be slow
   - Consider chunked algorithm or document T < 500 requirement

2. **Need backward pass in Stage 1?**
   - If Stage 3 needs smooth posteriors p(z_t | y), add forward-backward now
   - Otherwise defer to Stage 3

3. **Prior specification for Stage 3?**
   - Normal(0,1) on logits may be too vague for many states
   - Consider Dirichlet priors on transition rows
   - Consider hierarchical priors on state-specific parameters

4. **Testing label switching?**
   - Add test: permute (mu, trans) and verify identical likelihood
   - Important for Stage 3 where label switching is problematic

5. **Coordinate frame for Stage 2?**
   - Rotation matrices (3×3): standard but 9 parameters
   - Quaternions (4D): compact but normalized constraint
   - Axis-angle (3D): compact but singularities at 0/2π

---

## Recommended Implementation Order

```
Phase 1: Core Algorithm
1. hmm_pytensor.py with forward_log_prob_single (basic version)
2. Unit test: T=3, S=2, manual enumeration validation
3. Add T=1 edge case handling
4. Add numerical stability checks

Phase 2: PyMC Integration  
5. PyMC wrapper with 1D Gaussian emissions (shared sigma)
6. Synthetic data generator with validation
7. Demo notebook with basic sampling

Phase 3: Validation
8. Gradient tests (pytensor.grad vs finite differences)
9. Label switching test (permutation invariance)
10. Performance benchmarks

Phase 4: Extensions (Optional for Stage 1)
11. Add Viterbi decoding
12. Add batch dimension support
13. Add state-specific emission parameters option
```

---

## Summary of Critical Additions

**Must have for Stage 1:**
1. ✅ Gradient validation tests
2. ✅ T=1 edge case handling
3. ✅ Numerical stability documentation
4. ✅ Performance benchmarks
5. ✅ Label switching discussion

**Nice to have for Stage 1:**
1. ⚠️ Viterbi decoding (can defer to Stage 1b)
2. ⚠️ Batch dimension support (can defer to Stage 2)
3. ⚠️ Backward pass (can defer to Stage 3)

**Must clarify before Stage 2:**
1. ❗ Parent-relative coordinate frame representation
2. ❗ State initialization strategy for Stage 3
3. ❗ Label switching mitigation approach

The Stage 1 plan is **fundamentally sound** but needs these additions for robustness and to avoid rework in later stages.
