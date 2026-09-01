# r-elegans

`r-elegans` is an experimental JAX-native foundation for a differentiable,
connectome-constrained *C. elegans* simulator.

The current milestone contains three independently testable systems:

- a conductance-based, single-compartment neuron with the 17 ionic-current
  terms used in the published AWCON and RMD models;
- a 302-neuron-compatible graded-potential model with masked chemical and
  electrical connectivity, plus Diffrax integration;
- an overdamped planar body model that uses resistive-force theory (RFT) to
  turn prescribed joint-angle waves into locomotion.

Biophysical calibration currently applies only to AWCON and RMD. Empirical
connectome ingestion, the neuromuscular adapter, Gymnax environments, and
closed-loop chemotaxis are later milestones described in [PRD.md](PRD.md).

## Single-compartment electrophysiology

`r_elegans.brain.single_compartment` implements the voltage-gated K, Ca,
Ca-regulated K, NCA, and leak currents described by Nicoletti et al. (2019),
using the neuron-calibrated parameter values from its supplement. The model
uses mV, ms, pA, nS, pF, and micromolar units, exposes each ionic current
separately, supports current-clamp integration, and remains differentiable in
JAX.

The AWCON and RMD conductance compositions are external data, loaded with:

```python
from r_elegans.data import load_neuron_parameters

awcon = load_neuron_parameters("AWCON")
```

Every conductance has an evidence label. The published AWCON/RMD values are
whole-cell fits, not direct measurements of channel abundance. For other
neuron classes, transcript expression can identify plausible channel presence
but cannot by itself determine maximal conductance. Those classes remain
explicitly uncalibrated until electrophysiology or a documented fit constrains
them.

Primary model source: [Nicoletti et al., PLOS ONE 2019](https://doi.org/10.1371/journal.pone.0218738).

## Muscle-driven body

The planar body now accepts one dorsal and one ventral activation value per
joint. Opposing muscles generate active bending moments; elastic and viscous
body terms resist bending; and resistive-force theory solves the rigid-body
translation and rotation required by force and torque balance. A complete
muscle-driven rollout is:

```python
from r_elegans.body import (
    default_muscle_body_params,
    simulate_muscle_wave,
)

params = default_muscle_body_params(num_segments=12)
final_state, trajectory = simulate_muscle_wave(params, steps=500, dt=0.01)
```

The default parameters are normalized and intended for controller development,
gradient tests, and integration. They are not yet an adult-worm mechanical
calibration. The parameter interface separates substrate drag, passive bending
stiffness and damping, muscle moment scale, activation time constant, and the
maximum joint angle so each can later be fitted to external biomechanical data.

Primary mechanics references: [Fang-Yen et al., PNAS 2010](https://doi.org/10.1073/pnas.1003016107)
and [Shen et al., Biophysical Journal 2012](https://doi.org/10.1016/j.bpj.2012.05.012).

## External scientific data

Raw data, processed arrays, learned parameters, and simulation results are
never stored in this Git repository. Select an external directory explicitly:

```bash
export R_ELEGANS_DATA_DIR="~/OneDrive/r-elegans"
```

On macOS development machines, `~/OneDrive` may itself be a stable alias for
the institution-managed OneDrive location.

The code rejects a data root located inside the source checkout. The standard
external layout separates immutable source artifacts, derived data, manifests,
caches, and results:

```text
r-elegans/
├── raw/
│   ├── connectome/cook2019/
│   ├── neurotransmitters/
│   ├── physiology/
│   └── functional/
├── processed/
│   ├── connectome/
│   └── parameters/
├── manifests/
├── cache/
└── results/
```

Every acquired artifact must have a manifest recording its source URL, DOI or
citation, retrieval date, license, SHA-256 digest, and processing-code version.
Only those small metadata manifests may be copied into Git for reproducibility.

The initial physiology data root contains the paper's equation and parameter
supplements, digitized AWCL/AIYL/AVAL/AWAL/RIML/VD05 validation traces, a
checksummed physiology manifest, and the AWCON/RMD parameter catalog. None of
these artifacts are tracked by Git.

## Development

Use Python 3.10 or newer, create an isolated environment, and install:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Conventions

- Connectivity matrices use `[postsynaptic, presynaptic]` indexing.
- Chemical and gap-junction strengths are nonnegative after transformation.
- Gap-junction masks and strengths must be symmetric.
- Body coordinates use a center-of-geometry frame internally and world-space
  position plus heading externally.
