# Product Requirements Document (PRD): `r-elegans`

**Project Name:** `r-elegans`  
**Target Stack:** JAX / Diffrax / Gymnax (Vectorized & Differentiable RL Environment)  
**Primary Objective:** Build a GPU-accelerated, biophysically constrained Reinforcement Learning (RL) environment and differentiable simulator for *Caenorhabditis elegans* (*C. elegans*) continuous neural dynamics, neuromuscular activation, and low-Reynolds hydrodynamics.

---

## 1. Project Overview & High-Level Goals

### Strategic Goal
Create an open-source, ultra-fast, end-to-end differentiable simulation and RL platform that allows researchers and AI agents to study biological sensorimotor control, neural circuit optimization, and chemotaxis behavior in *C. elegans*.

### Core Technical Objectives
* **JAX-Native & Hardware Acceleration:** Entire pipeline (brain ODEs, body physics, environment step logic) must run natively on JAX to support vectorization (`vmap`), parallel multi-environment rollout (`pmap`), and GPU acceleration.
* **End-to-End Differentiability:** Enable reverse-mode automatic differentiation through time (via `diffrax`) to allow gradient-based optimization of neural parameters (e.g., synaptic weights, gap conductances) alongside traditional RL policies.
* **Biological Grounding:** Constrain the agent's neural architecture strictly to the empirical 302-neuron connectome topology and biophysical membrane potential dynamics.
* **Gymnax/Gymnasium Compatibility:** Expose a standardized interface (`reset`, `step`) for seamless integration with modern JAX RL frameworks (e.g., PureJaxRL, CleanRL).

---

## 2. System Architecture & System Boundaries

The system is structured as a closed-loop sensorimotor framework:

```
                  ┌─────────────────────────────────────────┐
                  │              ENVIRONMENT                │
                  │                                         │
                  │  ┌──────────────┐     ┌──────────────┐  │
                  │  │ External     │     │ 2D Planar    │  │
                  │  │ Gradients    │     │ Hydrodynamics│  │
                  │  │ (Food, Odor) │     │ (Viscous)    │  │
                  │  └──────┬───────┘     └──────▲───────┘  │
                  └─────────┼────────────────────┼──────────┘
                            │ Sensory            │ Muscle Actuations
                            │ Stimuli (I_ext)    │ (Dorsal/Ventral)
                            ▼                    │
                  ┌──────────────────────────────┴──────────┐
                  │                 AGENT                   │
                  │                                         │
                  │      Constrained 302-Neuron Brain       │
                  │     (Continuous Graded Potentials)      │
                  │                                         │
                  │  Policy Parameters:                     │
                  │  - Synaptic Weights (W_chem ⊙ M_chem)   │
                  │  - Gap Conductances (G_gap ⊙ M_gap)     │
                  │  - Channel Reversal Potentials (E_syn)  │
                  └─────────────────────────────────────────┘
```

### Module Interface Specifications

| Interface Element | Component | Definition / Bounds |
| :--- | :--- | :--- |
| **Observation Space ($S_t$)** | Environment → Agent | - Continuous membrane voltages ($V_i, i \in [1..302]$)<br>- Segment body bend angles ($\theta_k, k \in [1..N-1]$)<br>- Local chemical gradient concentration ($C(x,y)$) |
| **Action Space ($A_t$)** | Agent → Environment | - 95 muscle activation potentials, or high-level $N$-segment dorsal/ventral bending moments ($M_d, M_v$) |
| **Policy Parameters ($\theta$)** | Optimizer → Agent | - Synaptic weight matrix ($W_{\text{chem}}$)<br>- Gap junction conductance matrix ($G_{\text{gap}}$)<br>- Element-wise masked by empirical connectome matrices ($M_{\text{chem}}, M_{\text{gap}}$) |

---

## 3. Directory Structure & Setup Commands

To initialize the repository layout execute:

```bash
mkdir -p r-elegans/{r_elegans/{data,brain,body,envs,utils},tests,notebooks,scripts}
touch r-elegans/pyproject.toml
touch r-elegans/README.md
touch r-elegans/LICENSE
```

Target repository layout:

```
r-elegans/
├── pyproject.toml
├── README.md
├── LICENSE
├── r_elegans/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── connectome_loader.py    # Downloads/parses Cook 2019 & PyOpenWorm
│   │   └── eigenworm_loader.py     # Parses Stephens et al. PCA shape vectors
│   ├── brain/
│   │   ├── __init__.py
│   │   └── dynamics.py             # Diffrax ODE continuous graded dynamics
│   ├── body/
│   │   ├── __init__.py
│   │   └── mechanics.py            # 2D low-Reynolds Resistive Force Theory
│   ├── envs/
│   │   ├── __init__.py
│   │   └── foraging_env.py         # Gymnax / Gym-compatible JAX environment
│   └── utils/
│       ├── __init__.py
│       └── metrics.py              # Eigenworm posture MSE & trajectory loss
├── tests/
│   ├── test_connectome.py
│   ├── test_brain.py
│   └── test_differentiability.py
└── scripts/
    └── extract_data.py             # CLI runner for connectome preprocessing
```

---

## 4. Dependency Specification (`pyproject.toml`)

```toml
[build-system]
requires = ["flit_core >=3.2,<4.0"]
build-backend = "flit_core.buildapi"

[project]
name = "r-elegans"
version = "0.1.0"
description = "A JAX-native RL environment and differentiable simulator for C. elegans"
authors = [{ name = "r-elegans Contributors" }]
license = { file = "LICENSE" }
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.25",
    "jaxlib>=0.4.25",
    "diffrax>=0.5.0",
    "equinox>=0.11.0",
    "gymnax>=0.0.8",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pyopenworm>=0.11.0",
    "matplotlib>=3.7.0",
]
```

---

## 5. Detailed Component Specifications

### 5.1 Data Pipeline & Connectome Ingestion
* **Source Dataset:** Empirical hermaphrodite connectome from Cook et al. (2019) / PyOpenWorm.
* **Ingestion Task:** Automated parser (`r_elegans/data/connectome_loader.py`) that downloads, filters, and formats adjacency matrices into JAX arrays.
* **Key Artifacts Generated:**
  * `mask_chem`: $302 \times 302$ binary mask of chemical synapses.
  * `weight_chem`: $302 \times 302$ initial weight matrix derived from physical synapse counts.
  * `mask_gap`: $302 \times 302$ symmetric binary mask of electrical gap junctions.
  * `weight_gap`: $302 \times 302$ initial gap junction conductance matrix.

### 5.2 Continuous Neural Dynamics (Brain)
* **Mathematical Model:** Continuous non-spiking (graded potential) differential equations solved via `diffrax` (`r_elegans/brain/dynamics.py`).
* **State Variable:** Membrane voltage vector $V \in \mathbb{R}^{302}$.
* **Governing Equation:**
  $$\frac{dV_i}{dt} = \frac{1}{\tau_i} \left[ (E_L - V_i) + I_{\text{gap}, i} + I_{\text{chem}, i} + I_{\text{ext}, i} \right]$$
* **Mechanisms:**
  * **Gap Junction Currents:** Bi-directional flow proportional to voltage differences between connected neurons, masked by $M_{\text{gap}}$:
    $$I_{\text{gap}, i} = \sum_j G_{ij} \cdot M_{\text{gap}, ij} \cdot (V_j - V_i)$$
  * **Chemical Synaptic Currents:** Unidirectional flow governed by a sigmoidal activation function operating on presynaptic voltages, masked by $M_{\text{chem}}$:
    $$I_{\text{chem}, i} = \sum_j W_{ij} \cdot M_{\text{chem}, ij} \cdot \sigma\left(\frac{V_j - V_{\text{th}}}{V_{\text{slope}}}\right) \cdot (E_{\text{rev}, j} - V_i)$$
  * **External Input ($I_{\text{ext}}$):** Sensory current injected into specific sensory neurons (e.g., AWA, AWC, ASE) based on environmental stimuli.

### 5.3 Low-Reynolds Hydrodynamics (Body Physics)
* **Kinematic Structure:** $N$-link planar chain ($N \approx 12\text{--}24$ segments) representing the worm's longitudinal midline (`r_elegans/body/mechanics.py`).
* **Physics Framework:** Resistive Force Theory (RFT) in low-Reynolds-number viscous regimes (e.g., fluid or agar medium).
* **Force Decomposition:**
  * Perpendicular drag force: $F_\perp = -C_\perp \cdot v_\perp$
  * Parallel drag force: $F_\parallel = -C_\parallel \cdot v_\parallel$
  * Drag anisotropy ratio: $\frac{C_\perp}{C_\parallel} \approx 1.5 \text{ to } 40.0$ (reflecting physical media properties).

### 5.4 Vectorized Environment Logic (Foraging & Chemotaxis)
* **Task:** Navigate a 2D spatial plane to locate chemical sources via gradient ascent (`r_elegans/envs/foraging_env.py`).
* **Reward Structure:**
  * **Chemotaxis Progress:** Positive reward proportional to distance decreased toward peak gradient concentrations ($r_t = d_{t-1} - d_t$).
  * **Posture Smoothness:** Regularization penalty on rapid, unphysiological changes in segment bending.
  * **Energy Efficiency:** Penalty on excessive muscle activation magnitude.
* **Termination Conditions:** Reaching target food source radius, exceeding maximum step budget, or numerical instability bounds.

---

## 6. Phased Execution & Development Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASED EXECUTION ROADMAP                           │
└─────────────────────────────────────────────────────────────────────────┘
  Phase 1: Project Setup & Automated Connectome Extraction
  ├── Establish repository structure & dependency locks (pyproject.toml)
  └── Implement automated downloading & matrix formatting for Cook et al. (2019)
  
  Phase 2: Differentiable Neural Engine (Diffrax ODE)
  ├── Implement graded potential ODE vector fields
  └── Build differentiability test suite to verify gradient flow through diffrax solvers
  
  Phase 3: Body Physics & Hydrodynamics (Resistive Force Theory)
  ├── Implement 2D N-link kinematics and RFT drag calculations
  └── Validate undulatory locomotion dynamics (e.g., wave propagation)
  
  Phase 4: Gymnax Environment & Closed-Loop Reinforcement Learning
  ├── Integrate Brain + Body + Environment into standardized Gymnax API
  └── Implement benchmark training pipeline (PPO/Gradient-based optimization)
```

---

## 7. Acceptance & Verification Criteria

To declare the PRD implementation complete, the codebase must pass three functional acceptance gates:

1. **Connectome Topology Gate:** The processed adjacency matrices must map exactly to 302 canonical neuron identifiers, with non-zero entries strictly matching the empirical chemical synapse and gap junction masks.
2. **Gradient Flow & Differentiability Gate:** Reverse-mode automatic differentiation through the `diffrax` ODE solver must execute without generating `NaN` values or zero-gradient deadlocks across a 1,000-step unroll.
3. **Locomotion & Chemotaxis Gate:** The closed-loop environment must achieve stable undulatory locomotion in 2D space, demonstrating non-zero forward displacement driven by asymmetric dorsal/ventral muscle contractions.
