Here’s what I’ll give you in this message:

1. **A self-contained Markdown spec** of GIMBAL focused on implementation.
2. **A ready-to-paste Copilot prompt** that tells it exactly what to do with that spec.

You can copy section (1) into a file like `gimbal_spec.md` in your repo, and use section (2) as your Copilot Chat prompt.

---

## 1. Implementation-focused Markdown spec (copy this into a `.md` file)

### GIMBAL: Implementation-Oriented Specification

This document specifies the **GIMBAL** algorithm (“Geometric Manifolds for Body Articulation and Localization”) for multi-view 3D animal pose reconstruction from 2D keypoints using a hierarchical von Mises–Fisher–Gaussian model. 

The goal is to implement:

* A **generative model** of 3D poses and 2D observations.
* An **MCMC inference algorithm** (HMC + Gibbs) to sample from the posterior over latent variables given 2D keypoints.
* A **parameter-initialization procedure** based on a small “ground truth” dataset with 3D motion capture.

The implementation can be in Python using **NumPy + JAX**, **PyTorch**, or another autodiff framework.

---

### 1. Data, cameras, and notation

* Time steps: (t = 1,\dots,T)
* Keypoints: (k = 1,\dots,K)
* Cameras: (c = 1,\dots,C)

**3D variables**

* (x_{t,k} \in \mathbb{R}^3): 3D position of keypoint (k) at time (t), in world coordinates.
* Skeleton graph: a tree
  [
  G = {(\pi(k), k)}_{k=2}^K
  ]
  where keypoint 1 is the root, and (\pi(k) < k) is the parent of keypoint (k). 
* (\rho_k > 0): average 3D distance between keypoint (k) and its parent (\pi(k)) (estimated from training data).
* (\sigma_k^2): variance of this distance for edge ((\pi(k),k)).

**Directional variables**

* (u_{t,k} \in \mathbb{S}^2): unit vector from parent to child,
  conceptually proportional to (x_{t,k} - x_{t,\pi(k)}).

**Temporal and pose variables**

* (s_t \in {1,\dots,S}): discrete pose state at time (t).
* (h_t \in [-\pi, \pi)): heading angle (rotation in the (xy)-plane) at time (t). 

**Observation variables**

* (y_{t,k,c} \in \mathbb{R}^2): 2D keypoint position (pixels) in camera (c).
* (z_{t,k,c} \in {0,1}): outlier indicator for observation (y_{t,k,c}).

**Camera model**

For each camera (c), we have a calibrated projective mapping (f_c : \mathbb{R}^3 \to \mathbb{R}^2) of the form:
[
(u, v, w)^\top = A_c x + b_c,\quad
f_c(x) = \frac{1}{w} (u, v)^\top
]
where (A_c \in \mathbb{R}^{3\times 3}) and (b_c \in \mathbb{R}^3) are known from calibration. 

---

### 2. Generative model

The joint distribution is over:
[
x_{1:T,1:K},; u_{1:T,2:K},; s_{1:T},; h_{1:T},; z_{1:T,1:K,1:C},; y_{1:T,1:K,1:C}
]
given parameters (skeleton, temporal variances, vMF pose priors, HMM transitions, outlier mixture parameters).

Below everything is **conditioned on fixed camera parameters** and on **hyperparameters** that are either hand-set or fit from training data.

---

#### 2.1 Root keypoint dynamics

Keypoint 1 is the root of the skeleton. For (t=1):
[
x_{1,1} \sim \mathcal{N}(\mu_1, \sigma_1^2 I_3)
]
For (t>1):
[
x_{t,1} \mid x_{t-1,1} \sim \mathcal{N}(x_{t-1,1}, \eta_1^2 I_3)
]
where (\eta_1^2) is the temporal variance of the root node.

---

#### 2.2 Hierarchical vMFG prior for non-root keypoints

For each child keypoint (k>1) and time (t), the conditional distribution combines:

* Temporal smoothness:
  [
  x_{t,k} \mid x_{t-1,k} \sim \mathcal{N}(x_{t-1,k}, \eta_k^2 I_3)
  ]
* Skeletal constraint:
  [
  x_{t,k} \mid x_{t,\pi(k)}, u_{t,k} \sim \mathcal{N}(x_{t,\pi(k)} + \rho_k u_{t,k},; \sigma_k^2 I_3)
  ]

Combining these gives the conditional
[
p(x_{t,k} \mid x_{t-1,k}, x_{t,\pi(k)}, u_{t,k})
= \mathcal{N}(\tilde\mu_{t,k}, \tilde\sigma_k^2 I_3)
]
with 

[
\alpha_k = \frac{\eta_k^{-2}}{\eta_k^{-2} + \sigma_k^{-2}},\quad
\tilde\sigma_k^2 = \frac{1}{\eta_k^{-2} + \sigma_k^{-2}},
]
[
\tilde\mu_{t,k} = \alpha_k x_{t-1,k}

* (1-\alpha_k),\big(x_{t,\pi(k)} + \rho_k u_{t,k}\big).
  ]

This is the “hierarchical von Mises–Fisher–Gaussian” part: spatial offsets plus temporal smoothing.

---

#### 2.3 Pose-state-dependent directional prior

Let (R(h_t)) be the 3D rotation matrix that rotates by angle (h_t) in the (xy) plane (leaving (z) fixed). 

For each time (t) and keypoint (k>1):
[
u_{t,k} \mid s_t, h_t \sim \mathrm{vMF}!\left(R(h_t),\nu_{s_t,k},; \kappa_{s_t,k}\right),
]
where (\nu_{s,k} \in \mathbb{S}^2) and (\kappa_{s,k} \ge 0) are pose-state-specific mean directions and concentrations learned from training data.

---

#### 2.4 Pose state dynamics

Pose states follow an HMM:

* Initial state:
  [
  s_1 \sim \text{Uniform}{1,\dots,S}
  ]
* Transitions:
  [
  s_t \mid s_{t-1} \sim \text{Categorical}(\Lambda_{s_{t-1},:}),
  ]
  where (\Lambda \in [0,1]^{S\times S}) is the transition matrix, rows summing to 1. 

---

#### 2.5 Heading prior

A simple prior is uniform on the circle:

[
h_t \sim \text{Uniform}(-\pi,\pi)
]

Equivalently, a von Mises distribution with zero concentration: (h_t \sim \mathrm{vM}(0, 0)). 

---

#### 2.6 Robust observation model

For each (t,k,c):

1. Outlier indicator:
   [
   z_{t,k,c} \sim \text{Bernoulli}(\beta_{k,c})
   ]
2. Error distribution (2D):
   [
   \varepsilon_{t,k,c} \mid z_{t,k,c} = z
   \sim \mathcal{N}(\mu_{k,c,z},; \omega_{k,c,z}^2 I_2),\quad z \in {0,1}
   ]
3. Observation:
   [
   y_{t,k,c} = f_c(x_{t,k}) + \varepsilon_{t,k,c}.
   ]

Typically (\mu_{k,c,0}\approx 0) and (\mu_{k,c,1}\approx 0); the key difference is (\omega_{k,c,1}^2 \gg \omega_{k,c,0}^2).

---

### 3. Posterior inference via MCMC

We want to sample from the posterior
[
p(x,u,s,h,z \mid y)
]
given parameters. The algorithm alternates:

1. **HMC update for all 3D positions (x)**
2. **Gibbs updates for directional vectors (u)**
3. **Gibbs updates for headings (h)**
4. **Forward-filtering backward-sampling for pose states (s)**
5. **Bernoulli updates for outlier indicators (z)**

The log-posterior and its gradient w.r.t. (x) are computed using autodiff.

---

#### 3.1 Sampling 3D positions (x) with HMC

We treat (x = {x_{t,k}}) as a single large vector in (\mathbb{R}^{3TK}).

The conditional density is
[
p(x \mid y,u,s,h,z) \propto
p(x \mid u,s,h) ;
p(y \mid x,z),
]
where:

* (p(x \mid u,s,h)) is given by the root dynamics and hierarchical Gaussians (Sections 2.1–2.2).
* (p(y \mid x,z)) is a product of Gaussians over all ((t,k,c)) using the error model in Section 2.6.

Implement HMC as follows (pseudocode):

1. Define `log_p_x(x)` returning (\log p(x \mid y,u,s,h,z)).
2. Use autodiff to compute (\nabla_x \log p_x(x)).
3. At each MCMC iteration:

   * Sample momentum (p \sim \mathcal{N}(0, I)).
   * Run (L) leapfrog steps with step size (\epsilon).
   * Accept/reject via Metropolis–Hastings.

The paper uses ~10 leapfrog steps per iteration and adapts (\epsilon) during burn-in (e.g., NUTS-style dual averaging), but any reasonable HMC tuning is acceptable. 

---

#### 3.2 Sampling directions (u_{t,k})

Given (x,h,s), each direction (u_{t,k}) for (k>1) is independent and has a vMF conditional: 

[
u_{t,k} \mid x,s,h \sim \mathrm{vMF}(\tilde\nu_{t,k}, \tilde\kappa_{t,k}),
]
where
[
\mathbf{a}*{t,k}
= \kappa*{s_t,k} R(h_t)\nu_{s_t,k}

* \frac{\rho_k}{\sigma_k^2} (x_{t,k} - x_{t,\pi(k)}),
  ]
  [
  \tilde\kappa_{t,k} = |\mathbf{a}*{t,k}|,\quad
  \tilde\nu*{t,k} = \mathbf{a}*{t,k} / \tilde\kappa*{t,k}.
  ]

Implementation detail:

* Compute (\mathbf{a}_{t,k}) as a 3-vector.
* Set (\tilde\kappa_{t,k} = \max(|\mathbf{a}_{t,k}|, \varepsilon)) with a small (\varepsilon) to avoid divide-by-zero.
* Use a standard vMF sampler on (\mathbb{S}^2).

---

#### 3.3 Sampling headings (h_t)

Given (u_{t,k}) and (s_t), the posterior for (h_t) is a von Mises distribution:

[
h_t \mid u_t, s_t \sim \mathrm{vM}(\theta_t, \tau_t),
]
with
[
\tilde{y}*t = \sum*{k=2}^K
\sin(\hat xz(u_{t,k})) ,
\sin(\hat xz(\nu_{s_t,k})) ,
\sin(\Delta_{t,k}),
]
[
\tilde{x}*t = \sum*{k=2}^K
\sin(\hat xz(u_{t,k})) ,
\sin(\hat xz(\nu_{s_t,k})) ,
\cos(\Delta_{t,k}),
]
[
\Delta_{t,k} = \hat xy(u_{t,k}) - \hat xy(\nu_{s_t,k}),
]
[
\theta_t = \arctan!\left(\frac{\tilde{y}_t}{\tilde{x}_t}\right),\quad
\tau_t = \sqrt{\tilde{x}_t^2 + \tilde{y}_t^2}.
]

Here (\hat xz(v)) and (\hat xy(v)) are azimuth and polar angles of unit vector (v\in\mathbb{S}^2).

Implementation:

* Write helper functions to convert a unit vector to ((\hat xz,\hat xy)).
* Compute (\tilde{x}_t, \tilde{y}_t) and then sample from vM((\theta_t,\tau_t)).

---

#### 3.4 Sampling pose states (s_{1:T}) (HMM FFBS)

Conditioned on (u) and (h), pose states follow an HMM with:

* Initial log-probabilities: (\log p(s_1) = -\log S).
* Transition probabilities: (\Lambda).
* Emission log-likelihood:
  [
  \log p(u_t \mid s_t, h_t) =
  \sum_{k=2}^K \log \mathrm{vMF}\big(u_{t,k}
  \mid R(h_t)\nu_{s_t,k},\kappa_{s_t,k}\big).
  ] 

Use standard **forward filtering–backward sampling**:

* Forward pass: compute filtered log-probabilities (\alpha_t(s)).
* Backward sampling: sample (s_T) then (s_{T-1},\dots,s_1).

---

#### 3.5 Sampling outlier indicators (z_{t,k,c})

Given (x) and (y), define residuals
[
\varepsilon_{t,k,c} = y_{t,k,c} - f_c(x_{t,k}).
]

The conditional posterior for (z_{t,k,c}) is Bernoulli with: 

[
z_{t,k,c} \mid y,x \sim \text{Bernoulli}(\tilde\beta_{t,k,c}),
]
[
\tilde\beta_{t,k,c}
= \sigma\Big(
\text{logit}(\beta_{k,c})

* \log \mathcal{N}(\varepsilon_{t,k,c} \mid \mu_{k,c,1}, \omega_{k,c,1}^2 I_2)

- \log \mathcal{N}(\varepsilon_{t,k,c} \mid \mu_{k,c,0}, \omega_{k,c,0}^2 I_2)
  \Big),
  ]
  where (\sigma(\cdot)) is the logistic sigmoid and (\text{logit}(\beta)=\log\frac{\beta}{1-\beta}).

---

#### 3.6 Overall MCMC scheme

For iteration (m = 1,\dots,M):

1. Sample (x^{(m)} \sim p(x \mid y,u^{(m-1)},s^{(m-1)},h^{(m-1)},z^{(m-1)})) using HMC.
2. For all (t,k>1), sample (u_{t,k}^{(m)}) from their vMF conditionals.
3. For each (t), sample (h_t^{(m)}) from (\mathrm{vM}(\theta_t,\tau_t)).
4. Sample pose state sequence (s_{1:T}^{(m)}) using FFBS.
5. For all (t,k,c), sample (z_{t,k,c}^{(m)}) from their Bernoulli conditionals.

Discard burn-in, then use the remaining samples to compute posterior means for (x_{t,k}) (3D keypoints) and to estimate uncertainty.

---

### 4. Parameter initialization from ground truth data

We assume access to a **small training dataset** where:

* 3D motion capture gives ground truth (x_{t,k}^{\text{GT}}).
* Calibrated cameras allow projection of ground truth into each view.

We outline a practical initialization procedure.

#### 4.1 Temporal variances (\eta_k^2)

For each keypoint (k), compute frame-to-frame displacements in training data:
[
\Delta x_{t,k} = x_{t,k}^{\text{GT}} - x_{t-1,k}^{\text{GT}}.
]
Set
[
\eta_k^2 = \text{Var}*t(|\Delta x*{t,k}|)^2
]
or a robust estimate (e.g. based on median). In practice the paper uses small, fixed values such as (\eta_k^2=50) or 100 depending on model variant. 

#### 4.2 Skeletal parameters (\rho_k,\sigma_k^2)

For each edge ((\pi(k),k)):
[
d_{t,k} = \big|x_{t,k}^{\text{GT}} - x_{t,\pi(k)}^{\text{GT}}\big|.
]
Set:
[
\rho_k = \mathbb{E}*t[d*{t,k}],\quad
\sigma_k^2 = \text{Var}*t(d*{t,k}).
] 

#### 4.3 Outlier mixture parameters (\beta_{k,c},\mu_{k,c,z},\omega_{k,c,z}^2)

1. For each (k,c), compute 2D reprojection error magnitudes:
   [
   e_{t,k,c} = \big|y_{t,k,c}^{\text{obs}}

   * f_c(x_{t,k}^{\text{GT}})\big|.
     ]
2. Fit a 2-component Gaussian mixture model to ({e_{t,k,c}}) using EM, initialized with:

   * means near 0 for both components,
   * inlier variance (\omega_{k,c,0}^2 \approx 1),
   * outlier variance (\omega_{k,c,1}^2 \approx 100^2),
   * outlier mixing weight (\beta_{k,c}) initialized to the fraction of errors above e.g. 15 px. 
3. Convert GMM parameters into (\beta_{k,c}), (\mu_{k,c,0/1}), and (\omega_{k,c,0/1}^2) in the 2D vector domain (errors in x,y).

#### 4.4 Pose priors (\nu_{s,k},\kappa_{s,k},\Lambda)

From ground-truth 3D data:

1. For each frame and child keypoint, compute direction:
   [
   v_{t,k} = \frac{x_{t,k}^{\text{GT}} - x_{t,\pi(k)}^{\text{GT}}}
   {\big|x_{t,k}^{\text{GT}} - x_{t,\pi(k)}^{\text{GT}}\big|}
   ]
2. Treat ({v_{t,k}}) across time as samples of body pose, ignoring heading (or after aligning headings).
3. Fit a **hidden Markov model** or mixture of vMF distributions on these directions to obtain:

   * pose states (s_t),
   * per-state, per-keypoint vMF parameters ((\nu_{s,k},\kappa_{s,k})),
   * state transition matrix (\Lambda).

A simpler approximation is to:

* Cluster frames into (S) clusters using k-means on concatenated directions ([\dots, v_{t,k}, \dots]).
* For each cluster (s), and keypoint (k), set (\nu_{s,k}) to the normalized mean direction over cluster members; set (\kappa_{s,k}) from the resultant length.
* Estimate (\Lambda) from empirical transition counts in the assigned state sequence.

---

This completes the implementation-oriented specification of GIMBAL: all variables, conditionals, and algorithmic steps required for a working sampler and 3D reconstruction pipeline.
