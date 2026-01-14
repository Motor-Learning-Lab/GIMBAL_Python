# Review of Updated HMM Plans - Second Iteration

This document reviews the updated 3-stage HMM plan and Stage 1 detailed specification, identifying what's improved and what critical issues remain before implementation.

---

## Overall Assessment

**Status:** Better, but not ready for implementation yet.

The updated plans show significant improvements and address several concerns from the first review. However, critical issues remain around gradient validation, label switching, dimensionality scaling, and validation infrastructure.

---

## What's Improved ✅

### 1. Global Frame Clarification
- Explicitly states canonical directions are in **global coordinates**, not parent-relative
- Removes ambiguity about coordinate frames
- Makes Stage 2 refactoring simpler

### 2. T=1 Edge Case Handling
Section 4.1 now includes:
```python
alpha_last = pt.switch(
    pt.eq(logp_emit.shape[0], 1),
    alpha_prev,
    alpha_all[-1],
)
```
This properly handles single-timestep sequences.

### 3. Stage 2 Split into 2a and 2b
- Stage 2a: Factor out observation model
- Stage 2b: Clarify direction pipeline
- Better granularity for incremental progress

### 4. Cleaner Completion Criteria
Section 10 provides clear exit conditions for Stage 1.

### 5. Improved Synthetic Generator
Now uses `np.broadcast_to(sigma_true, (S,))` for proper sigma handling.

---

## Critical Issues Remaining ⚠️

### Issue 1: Insufficient Gradient Validation (Section 8.3)

**Current plan:**
> Call `model.compile_dlogp()` and evaluate. All gradients must be finite; no NaNs.

**Problem:** This only checks for NaNs, not correctness.

**Required fix:**
```python
def test_gradients():
    """Test analytical gradients against finite differences."""
    # 1. Build model
    model = build_gaussian_hmm_model(y_obs, S=2)
    
    # 2. Get gradient function
    grad_fn = model.compile_dlogp()
    
    # 3. Test point
    test_point = model.initial_point()
    
    # 4. Check finite
    grads = grad_fn(test_point)
    for param, grad in grads.items():
        assert np.isfinite(grad).all(), f"{param} has non-finite gradient"
    
    # 5. CRITICAL: Compare with finite differences
    eps = 1e-5
    for param in ['mu', 'sigma', 'init_logits', 'trans_logits']:
        numerical_grad = compute_finite_diff_gradient(model, param, test_point, eps)
        analytical_grad = grads[param]
        
        # Must match to relative tolerance
        assert np.allclose(numerical_grad, analytical_grad, rtol=1e-4, atol=1e-6), \
            f"{param} gradient mismatch: analytical={analytical_grad}, numerical={numerical_grad}"

def compute_finite_diff_gradient(model, param_name, point, eps):
    """Compute gradient via finite differences."""
    logp_fn = model.compile_logp()
    base_logp = logp_fn(point)
    
    param_value = point[param_name]
    grad = np.zeros_like(param_value)
    
    for idx in np.ndindex(param_value.shape):
        point_plus = point.copy()
        point_plus[param_name][idx] += eps
        logp_plus = logp_fn(point_plus)
        
        grad[idx] = (logp_plus - base_logp) / eps
    
    return grad
```

**Why this matters:** PyTensor's autodiff through `pt.scan` and `pt.logsumexp` is complex. You need numerical validation to ensure correctness.

**Add to Stage 1 spec:** New subsection 8.3 with complete finite-difference test code.

---

### Issue 2: Normalization Requirements Unclear (Section 4.1)

**Current Step 1:**
```python
alpha_prev = logp_init + logp_emit[0]
```

**Problem:** Assumes `logp_init` is already normalized, but this isn't documented or enforced.

**Required fix:** Add to Section 4.1 after function signature:

```markdown
**Input Requirements:**
- `logp_init` must satisfy: `logsumexp(logp_init) ≈ 0` (normalized log-probabilities)
- Each row of `logp_trans` must satisfy: `logsumexp(logp_trans[i,:]) ≈ 0`
- If inputs are not normalized, results will be incorrect by a constant factor

**Note:** The PyMC wrapper in Section 5 handles normalization via:
```python
logp_init = init_logits - pm.math.logsumexp(init_logits)
logp_trans = trans_logits - pm.math.logsumexp(trans_logits, axis=1, keepdims=True)
```

**Alternative approach (more robust):** Normalize inside forward algorithm:
```python
# Step 1 — Initialization with normalization
logp_init_norm = logp_init - pt.logsumexp(logp_init)
alpha_prev = logp_init_norm + logp_emit[0]
```
```

**Recommendation:** Document requirements clearly OR normalize internally. Current approach is fragile.

---

### Issue 3: Label Switching Unaddressed (Stage 3)

**Current Stage 3 plan:**
```
mu[k, s, :] from normalized Gaussian
kappa[k, s] ~ HalfNormal(...)
```

**Problem:** For K=20, S=5, you have 100 canonical directions with S! = 120 equivalent posterior modes.

**Missing from plan:**
1. Initialization strategy for `mu[k,s]`
2. Post-hoc relabeling approach
3. Visualization strategy for permutation-invariant posteriors

**Required addition to Stage 3:**

```markdown
## Label Switching Mitigation

The HMM posterior exhibits label switching: permuting state indices gives equivalent likelihood.

### Initialization Strategy

```python
def initialize_canonical_directions(U_observed, S, parents):
    """
    Initialize mu[k,s] using K-means clustering on observed directions.
    
    Parameters
    ----------
    U_observed : array (T, K, 3)
        Observed unit direction vectors
    S : int
        Number of states
    parents : array (K,)
        Parent indices
        
    Returns
    -------
    mu_init : array (K, S, 3)
        Initial canonical directions (unit vectors)
    kappa_init : array (K, S)
        Initial concentration parameters
    """
    from sklearn.cluster import KMeans
    
    mu_init = np.zeros((K, S, 3))
    kappa_init = np.full((K, S), 10.0)
    
    for k in range(K):
        if k == 0:  # root has no direction
            continue
            
        # Cluster observed directions for this joint
        u_k = U_observed[:, k, :]  # (T, 3)
        kmeans = KMeans(n_clusters=S, random_state=42)
        kmeans.fit(u_k)
        
        # Normalize cluster centers
        centers = kmeans.cluster_centers_  # (S, 3)
        mu_init[k] = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        
        # Estimate concentration from cluster variance
        for s in range(S):
            mask = kmeans.labels_ == s
            if mask.sum() > 1:
                variance = np.var(u_k[mask], axis=0).sum()
                kappa_init[k, s] = 1.0 / max(variance, 0.01)
    
    return mu_init, kappa_init
```

### Post-Hoc Relabeling

```python
def relabel_hmm_posterior(trace, reference_joint=1):
    """
    Relabel states to canonical order based on mean canonical direction.
    
    Strategy: Sort states by z-component of mu[reference_joint, s, 2]
    """
    # Extract posterior mean canonical directions
    mu_mean = trace.posterior['mu'].mean(dim=['chain', 'draw']).values  # (K, S, 3)
    
    # Sort states by z-component of reference joint
    z_components = mu_mean[reference_joint, :, 2]
    state_order = np.argsort(z_components)[::-1]  # descending order
    
    # Permute all state-indexed variables
    trace_relabeled = trace.copy()
    for var in ['mu', 'kappa', 'trans_logits']:
        if var in trace.posterior:
            # Apply permutation to state dimension
            trace_relabeled.posterior[var] = trace.posterior[var].sel(state=state_order)
    
    return trace_relabeled
```

### Alternative: Ordered Constraint

Add constraint during model building to eliminate label switching:
```python
# Force canonical directions to be ordered by z-component
for k in range(1, K):
    for s in range(S-1):
        # Constrain: mu[k, s, 2] > mu[k, s+1, 2]
        pm.Potential(
            f"ordering_k{k}_s{s}",
            pt.switch(
                mu[k, s, 2] > mu[k, s+1, 2],
                0.0,
                -1e10,  # strong penalty
            )
        )
```

**Trade-off:** This eliminates label switching but may hurt mixing if true states aren't naturally ordered.
```

**Add entire section to Stage 3 plan.**

---

### Issue 4: Dimensionality Explosion (Stage 3)

**Current Stage 3:**
```
mu[k, s, :] ~ normalized Gaussian  # K×S×3 unconstrained params
kappa[k, s] ~ HalfNormal(...)       # K×S params
```

**Problem:** For K=20, S=5:
- 100 canonical directions = **300 unconstrained parameters** (raw_u)
- 100 kappa parameters
- 25 transition parameters
- **Total: ~425 continuous parameters**

**Missing:**
- Expected sampling time
- Gradual scaling strategy
- Hierarchical prior options

**Required addition to Stage 3:**

```markdown
## Computational Complexity and Scaling

Stage 3 parameter count scales as O(K×S). Plan gradual increase:

### Stage 3a: Minimal Skeleton (K=3, S=2)
**Parameters:**
- 6 canonical directions (18 unconstrained via raw_u)
- 6 kappa parameters  
- 4 transition parameters
- **Total: ~28 parameters**

**Expected performance:**
- Nutpie compilation: ~10 seconds
- 4 chains × 1000 draws: ~2 minutes
- ESS > 100 for most parameters

**Purpose:** Validate HMM integration, test label switching handling.

### Stage 3b: Medium Skeleton (K=10, S=3)
**Parameters:**
- 30 canonical directions (90 unconstrained)
- 30 kappa parameters
- 9 transition parameters
- **Total: ~129 parameters**

**Expected performance:**
- Nutpie compilation: ~20 seconds
- 4 chains × 2000 draws: ~10 minutes
- ESS > 50 acceptable due to increased complexity

**Purpose:** Test scalability, identify bottlenecks.

### Stage 3c: Full Skeleton (K=20, S=5)
**Parameters:**
- 100 canonical directions (300 unconstrained)
- 100 kappa parameters
- 25 transition parameters
- **Total: ~425 parameters**

**Expected performance:**
- Nutpie compilation: ~30-60 seconds
- 4 chains × 2000 draws: ~30-60 minutes
- ESS > 20 acceptable for this complexity

**Optimization strategies if too slow:**
1. Hierarchical priors: Share kappa across symmetric joint pairs
2. Fewer states: Try S=3 instead of S=5
3. Tied parameters: Same mu/kappa for left/right limbs
4. Increase target_accept to 0.95 (slower but more stable)

### Hierarchical Prior Example

```python
# Instead of: kappa[k, s] ~ HalfNormal(sigma=10)
# Use hierarchical structure:

kappa_global_scale ~ HalfNormal(sigma=5)
kappa_state_scale ~ HalfNormal(sigma=1, shape=S)
kappa ~ HalfNormal(sigma=kappa_global_scale * kappa_state_scale, shape=(K, S))

# This reduces effective parameters and improves identifiability
```
```

**Add entire section to Stage 3 plan.**

---

### Issue 5: Viterbi Decoding Missing (Section 7)

**Current plan:**
> Optional: perform Viterbi decoding outside PyMC for qualitative inspection.

**Problem:** No implementation guidance. You need this for validation.

**Required addition to Section 7:**

```python
def viterbi_decode_numpy(logp_emit, logp_init, logp_trans):
    """
    Decode most likely state sequence using Viterbi algorithm.
    
    Parameters
    ----------
    logp_emit : array (T, S)
        Log emission probabilities
    logp_init : array (S,)
        Log initial state probabilities (normalized)
    logp_trans : array (S, S)
        Log transition matrix (rows normalized)
        
    Returns
    -------
    z_map : array (T,) of int
        Most likely state at each timestep
    logp_path : float
        Log probability of most likely path
        
    Notes
    -----
    Uses max-product algorithm in log space with backtracking.
    """
    T, S = logp_emit.shape
    
    # Forward pass: compute max probabilities and backpointers
    delta = logp_init + logp_emit[0]  # (S,)
    psi = np.zeros((T, S), dtype=int)  # backpointers
    
    for t in range(1, T):
        # For each current state j, find best previous state i
        trans_scores = delta[:, None] + logp_trans  # (S, S)
        psi[t] = np.argmax(trans_scores, axis=0)  # (S,)
        delta = logp_emit[t] + np.max(trans_scores, axis=0)  # (S,)
    
    # Backward pass: backtrack to find most likely path
    z_map = np.zeros(T, dtype=int)
    z_map[-1] = np.argmax(delta)
    logp_path = delta[z_map[-1]]
    
    for t in range(T-2, -1, -1):
        z_map[t] = psi[t+1, z_map[t+1]]
    
    return z_map, logp_path


# Add to notebook after sampling:
def visualize_decoded_states(y, z_true, z_decoded, mu_posterior):
    """Visualize Viterbi decoded states vs true states."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot observations colored by true state
    ax = axes[0]
    for s in range(S):
        mask = z_true == s
        ax.scatter(np.where(mask)[0], y[mask], label=f'True state {s}', alpha=0.6)
    ax.set_ylabel('Observation')
    ax.set_title('Observations colored by true state')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot observations colored by decoded state
    ax = axes[1]
    for s in range(S):
        mask = z_decoded == s
        ax.scatter(np.where(mask)[0], y[mask], label=f'Decoded state {s}', alpha=0.6)
    ax.set_ylabel('Observation')
    ax.set_title('Observations colored by Viterbi decoded state')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # State sequence comparison
    ax = axes[2]
    ax.plot(z_true, 'o-', label='True states', alpha=0.6)
    ax.plot(z_decoded, 's--', label='Decoded states', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('State sequence comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute accuracy
    accuracy = (z_true == z_decoded).mean()
    print(f"Viterbi decoding accuracy: {accuracy:.1%}")
```

**Add complete implementation to Section 7 of Stage 1 spec.**

---

### Issue 6: Emission Construction Bug (Stage 3)

**Current Stage 3 Integration Steps:**
```
3. Add observation logp: logp_emit = log_dir_emit + logp_obs[:,None]
```

**Problem:** This suggests `logp_obs` is a vector, but earlier text implies scalar. The shape is ambiguous.

**Required clarification:**

```markdown
### Step 3: Construct Total Emission Log-Likelihood

```python
# Directional emission (state-dependent)
# Shape: (T, S) where each entry is log p(U_t | z_t=s)
log_dir_emit = directional_log_emissions(U, mu, kappa)

# Observation likelihood (state-independent)
# Shape: (T,) where each entry is log p(y_t | x_t, cameras)
# This comes from Stage 2a API
logp_obs = log_obs_likelihood(
    y_obs=y_observed,
    y_pred=y_projected,
    obs_sigma=obs_sigma,
    inlier_prob=inlier_prob,
    occlusion_mask=occlusion_mask
)  # returns (T,) scalar per timestep

# Total emission: broadcast observation likelihood to all states
# Shape: (T, S)
logp_emit = log_dir_emit + logp_obs[:, None]

# Interpretation:
# logp_emit[t, s] = log p(y_t, U_t | z_t=s, x_t)
#                 = log p(y_t | x_t) + log p(U_t | z_t=s)
# Assumes conditional independence: y_t ⊥ z_t | x_t
```

**Replace ambiguous step 3 in Stage 3 plan with clear version above.**

---

### Issue 7: No Performance Benchmarks

**Problem:** No quantitative targets for Stage 1 validation.

**Required addition:**

```markdown
# 11. Performance Benchmarks

Stage 1 must meet these targets to be considered working correctly:

## Forward Algorithm Performance

**Compilation:**
- First compilation: < 5 seconds
- Subsequent evaluations: < 0.01 seconds (graph cached)

**Evaluation:**
- T=100, S=5: < 0.1 seconds per evaluation
- T=500, S=10: < 0.5 seconds per evaluation
- T=1000, S=10: < 1 second per evaluation

**Scaling:**
- Should scale as O(T × S²) due to transition matrix
- If significantly worse, check for unnecessary recompilation

## Sampling Performance (1D Gaussian HMM)

**Test configuration:**
- T=200, S=3
- 4 chains × 1000 draws
- Ground truth: μ = [-2, 0, 2], σ = 0.5, diagonal-dominant transition matrix

**Expected results:**
- Nutpie compilation: < 10 seconds
- Sampling time: < 60 seconds total
- ESS for μ parameters: > 100
- ESS for σ: > 50
- R-hat for all parameters: < 1.05
- No divergences

## Diagnostic Checks

```python
def check_performance_benchmarks(trace, sampling_time):
    """Validate sampling performance meets targets."""
    import arviz as az
    
    # ESS checks
    ess = az.ess(trace)
    assert ess['mu'].values.min() > 100, f"mu ESS too low: {ess['mu'].values}"
    assert ess['sigma'].values > 50, f"sigma ESS too low: {ess['sigma'].values}"
    
    # R-hat checks  
    rhat = az.rhat(trace)
    assert rhat.max() < 1.05, f"R-hat too high: {rhat.max().values}"
    
    # Time check
    assert sampling_time < 60, f"Sampling too slow: {sampling_time:.1f}s"
    
    print("✓ All performance benchmarks passed")
```

## Failure Mode Checks

**If forward algorithm is slow:**
1. Check dtype: should be float64
2. Check for repeated compilations (use `pytensor.function` caching)
3. Profile with `pytensor.printing.debugprint()`

**If sampling fails:**
1. Verify logp is finite at initial point
2. Check gradients are O(1) magnitude
3. Reduce S to 2 for debugging
4. Try smaller T (e.g., T=50)

**If ESS is low:**
1. Check for label switching (multiple modes)
2. Increase tune steps to 2000
3. Increase target_accept to 0.95
4. Consider reparameterization
```

**Add Section 11 to Stage 1 spec.**

---

### Issue 8: Missing Troubleshooting Guide

**Problem:** Plans assume everything will work. Need contingency guidance.

**Required addition:**

```markdown
# 12. Common Issues and Solutions

## Issue: NaN in forward algorithm

**Symptoms:**
- `forward_log_prob_single` returns NaN
- logp evaluation crashes

**Diagnosis:**
```python
# Check inputs
print(f"logp_emit range: [{logp_emit.min()}, {logp_emit.max()}]")
print(f"logp_emit contains -inf: {pt.isinf(logp_emit).any()}")
print(f"logp_init sum: {pt.logsumexp(logp_init)}")  # should be ~0
```

**Solutions:**
1. Clip extreme log-probabilities:
   ```python
   logp_emit = pt.clip(logp_emit, -1e10, 0)
   logp_init = pt.clip(logp_init, -1e10, 0)
   ```
2. Check emission model isn't producing -inf for valid data
3. Add small regularization to avoid exact zeros in probabilities

## Issue: Sampling doesn't move (ESS < 10)

**Symptoms:**
- All chains stuck at initialization
- Trace plots show flat lines
- R-hat cannot be computed (identical chains)

**Diagnosis:**
```python
# Check logp and gradients at initial point
initial_point = model.initial_point()
logp_fn = model.compile_logp()
grad_fn = model.compile_dlogp()

print(f"Initial logp: {logp_fn(initial_point)}")
grads = grad_fn(initial_point)
for param, grad in grads.items():
    print(f"{param} gradient magnitude: {np.abs(grad).mean():.2e}")
```

**Solutions:**
1. Check logp_init and logp_trans are normalized (logsumexp ≈ 0)
2. Reduce S (try S=2 first)
3. Check gradient magnitude (should be O(1), not O(1e-10) or O(1e10))
4. Increase tune steps to 2000
5. Try different initialization:
   ```python
   initial_point = {
       'mu': np.linspace(-3, 3, S),
       'sigma': 1.0,
       'init_logits': np.zeros(S),
       'trans_logits': np.eye(S),  # diagonal-ish transition
   }
   ```

## Issue: Label switching in posterior

**Symptoms:**
- Trace plots show "jumps" between chains
- Multiple modes in μ posterior
- State identities swap mid-sampling

**Diagnosis:**
```python
# Check for multi-modality
mu_posterior = trace.posterior['mu'].values  # (chain, draw, S)
for chain in range(4):
    mu_chain = mu_posterior[chain, :, :]
    print(f"Chain {chain} mu means: {mu_chain.mean(axis=0)}")
# If means differ significantly between chains → label switching
```

**Solutions:**
1. Apply post-hoc relabeling (see Issue 3 above)
2. Use ordered constraint during model building
3. Initialize with K-means on observed data
4. Reduce S if problem persists

## Issue: Stage 3 too slow (> 1 hour sampling)

**Symptoms:**
- Progress bar moves very slowly
- nutpie compilation takes > 2 minutes

**Solutions:**
1. **Reduce complexity first:**
   - Try S=2 instead of S=5
   - Use K=10 instead of K=20
   - Reduce T if possible (T=100 instead of T=200)

2. **Hierarchical priors:**
   ```python
   # Share information across joints
   kappa_global ~ HalfNormal(5)
   kappa_local ~ HalfNormal(1, shape=(K, S))
   kappa = kappa_global * kappa_local
   ```

3. **Increase target_accept:**
   ```python
   # Slower but more stable
   trace = nutpie.sample(compiled_model, target_accept=0.95)
   ```

4. **Profile to find bottleneck:**
   ```python
   import cProfile
   cProfile.run('nutpie.sample(compiled_model, draws=100)', 'profile.stats')
   ```
```

**Add Section 12 to Stage 1 spec.**

---

## Summary of Required Changes

### Critical (Must Fix Before Implementation)

1. **✅ Add gradient validation with finite differences** (Issue 1)
   - Complete test code in Section 8.3
   - Include `compute_finite_diff_gradient` helper

2. **✅ Document normalization requirements** (Issue 2)
   - Add to Section 4.1 function docstring
   - Clarify logsumexp requirements

3. **✅ Fix emission construction** (Issue 6)
   - Clarify shapes in Stage 3
   - Show explicit broadcasting

4. **✅ Add Viterbi decoding** (Issue 5)
   - Complete implementation in Section 7
   - Include visualization helper

### Important (Should Add Before Stage 3)

5. **⚠️ Add label switching section** (Issue 3)
   - Complete initialization strategy
   - Post-hoc relabeling code
   - Add to Stage 3 plan

6. **⚠️ Add complexity scaling plan** (Issue 4)
   - Stages 3a/3b/3c with parameter counts
   - Expected sampling times
   - Hierarchical prior examples

7. **⚠️ Add performance benchmarks** (Issue 7)
   - Section 11 with quantitative targets
   - Diagnostic checking code

8. **⚠️ Add troubleshooting guide** (Issue 8)
   - Section 12 with common issues
   - Diagnostic commands
   - Solution strategies

---

## Recommended Next Steps

1. **For Stage 1 spec:** Add Sections 11-12 and enhance Section 8.3 with gradient testing
2. **For Stage 3 plan:** Add label switching section and complexity scaling section
3. **Review updated plans** with these additions before beginning implementation
4. **Start with Stage 1 only** - don't attempt Stage 2/3 until Stage 1 is solid

The core algorithms are sound, but the validation and scaling infrastructure needs to be in place before implementation begins.
