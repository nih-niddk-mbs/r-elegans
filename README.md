# r-elegans

`r-elegans` is an experimental JAX-native foundation for a differentiable,
connectome-constrained *C. elegans* simulator.

The current milestone contains four independently testable systems:

- a conductance-based, single-compartment neuron with the 17 ionic-current
  terms used in the published AWCON and RMD models;
- a 302-neuron-compatible graded-potential model with masked chemical and
  electrical connectivity, plus Diffrax integration;
- an overdamped planar body model that uses resistive-force theory (RFT) to
  turn muscle activation into locomotion;
- a polarity-aware adapter from 302 neural voltages through the 95 anatomical
  body-wall muscles to the body's 11 bending joints.

Biophysical calibration currently applies only to AWCON and RMD. Whole-animal
parameter fitting, Gymnax environments, and closed-loop chemotaxis are later
milestones described in [PRD.md](PRD.md).

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

## Neuromuscular adapter

Cook et al.'s hermaphrodite chemical-connectivity matrix is ingested in fixed
`[95 muscles, 302 neurons]` order. Its 956 reported neuron-to-muscle edges
contain 5,515 serial-section contacts and reach every body-wall muscle. The
muscles retain their dorsal/ventral quadrant and anterior/posterior row, then a
fixed local projection maps them to the 11 joints of the default 12-segment
body.

```python
import jax.numpy as jnp

from r_elegans.body import (
    NeuromuscularParams,
    build_muscle_projection,
    default_muscle_body_params,
    initialize_muscle_body,
    neuromuscular_body_step,
)
from r_elegans.data import (
    load_neuromuscular_connectome,
)

connectome = load_neuromuscular_connectome()
params = NeuromuscularParams(
    synapse_weights=connectome.chemical_counts,
    synapse_signs=connectome.synapse_signs,
    neuron_threshold=jnp.full((302,), -20.0),
    neuron_slope=jnp.full((302,), 5.0),
    muscle_threshold=jnp.full((95,), 0.05),
    muscle_slope=jnp.full((95,), 0.25),
)
state, muscle_activity = neuromuscular_body_step(
    initialize_muscle_body(12),
    default_muscle_body_params(12),
    jnp.full((302,), -60.0),
    params,
    build_muscle_projection(11),
    dt=0.01,
)
```

Polarity is intentionally separate from contact count. Wang et al.'s updated
transmitter atlas supports acetylcholine-only edges as excitatory and GABA-only
edges as inhibitory; ambiguous or receptor-dependent transmitter classes stay
zero instead of being guessed. This assigns a supported sign to 880 of 956
edges (5,142 of 5,515 section contacts). The sigmoid thresholds, gains, and
effective contact strengths remain calibration parameters.

Primary anatomy sources: [Cook et al., Nature 2019](https://doi.org/10.1038/s41586-019-1352-7)
and [Wang et al., eLife 2024](https://doi.org/10.7554/eLife.95402).

## Body motion fitting

The first motor-learning stage fits a compact periodic controller directly
through the differentiable body. It emits all 95 muscle activations while
learning wave amplitude, frequency, wavelength, phase, and steering bias:

```bash
python scripts/fit_body_controller.py --iterations 100
```

The objective rewards positive-x displacement and penalizes lateral drift,
heading drift, activation energy, and excessive curvature. Use `--output` to
write the fitted trajectory to the external results directory; generated
rollouts must not be committed to Git.

## External scientific data

Raw data, processed arrays, learned parameters, and simulation results are
never stored in this Git repository. Select an external directory explicitly:

```bash
export R_ELEGANS_DATA_DIR="$HOME/OneDrive/r-elegans"
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
