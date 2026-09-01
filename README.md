# r-elegans

`r-elegans` is an experimental JAX-native foundation for a differentiable,
connectome-constrained *C. elegans* simulator.

## Run the bundled model

The repository ships a compact runtime checkpoint, so cloning and installing
the package is enough to run the current motor-teacher chemotaxis baseline:

```bash
python -m pip install -e .
r-elegans-demo
```

No external data directory is required for this demo. The bundled checkpoint
contains the canonical 302-neuron ordering, 3,709 directed chemical edges,
1,093 undirected gap-junction pairs, 956 neuron-to-muscle edges, conservative
polarity for 880 of those edges, the fitted neural motor coefficients, body
gait, sensory policy, and the calibrated AWCON/RMD single-compartment
parameters. Use `--help` to select the source position, heading, duration,
direct-body mode, or an output archive.

The demo does **not** yet use the recurrent 302-neuron connectome to find food.
Food sensing is performed by a fitted seven-parameter controller that converts
head concentration history and gait phase into `[speed, steering]`. In neural
mode, a separate supervised motor teacher converts that command and gait phase
into 302 voltage outputs, which drive the anatomical neuron-to-muscle map. The
bundled neuron-to-neuron topology is available for the next recurrent stage but
is not on this demo's active computation path. See
[CURRENT_MODEL.md](CURRENT_MODEL.md) for the complete data flow, fitted
parameters, results, and limitations.

Raw publications, electrophysiology traces, fit histories, and generated
trajectories remain external because they are needed for auditing or
retraining—not inference. See [MODEL_ASSET_PROVENANCE.md](MODEL_ASSET_PROVENANCE.md)
for the bundled checkpoint's sources and limitations.

The current milestone contains five independently testable systems:

- a conductance-based, single-compartment neuron with the 17 ionic-current
  terms used in the published AWCON and RMD models;
- a 302-neuron-compatible graded-potential model with masked chemical and
  electrical connectivity, plus Diffrax integration;
- an overdamped planar body model that uses resistive-force theory (RFT) to
  turn muscle activation into locomotion;
- a polarity-aware adapter from 302 neural voltages through the 95 anatomical
  body-wall muscles to the body's 11 bending joints.
- a circular Petri-dish environment with a finite diffusing food pulse and an
  engineered-controller closed-loop chemotaxis baseline.

Biophysical calibration currently applies only to AWCON and RMD. The body and
Petri dish use normalized units; whole-animal parameter calibration and a
recurrent connectome-constrained controller remain later milestones described
in [PRD.md](PRD.md).

## Single-compartment electrophysiology

`r_elegans.brain.single_compartment` implements the voltage-gated K, Ca,
Ca-regulated K, NCA, and leak currents described by Nicoletti et al. (2019),
using the neuron-calibrated parameter values from its supplement. The model
uses mV, ms, pA, nS, pF, and micromolar units, exposes each ionic current
separately, supports current-clamp integration, and remains differentiable in
JAX.

The AWCON and RMD conductance compositions are bundled runtime parameters,
loaded with:

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

The continuous behavioral action interface is fitted over a grid of forward,
reverse, left, and right commands with:

```bash
python scripts/fit_commanded_body_controller.py --iterations 300
```

Its action is `[speed, steering]` in `[-1, 1]²`; zero speed produces zero
muscle activation. The internal controller maintains gait phase and converts
each command into the full 95-muscle trajectory.

## Supervised neural motor fitting

The second motor-learning stage fits bounded output channels for all 302 neuron
identifiers so that the fixed Cook neuron-to-muscle topology reproduces the
motion library:

```bash
python scripts/fit_neural_motor_controller.py \
  --data-root "$R_ELEGANS_DATA_DIR" \
  --iterations 500
```

Only the 129 neurons with signed neuromuscular projections are trainable in
this stage. A compact 13-feature command/phase map generates their voltages;
Cook contact counts and Wang transmitter signs remain fixed. These values are
called “neural outputs” because they are indexed by neuron identity, but they
are not generated by membrane dynamics or neuron-to-neuron connections. This
is a supervised motor teacher, not the recurrent 302-neuron brain. Its saved
voltage trajectories are intended as targets for fitting that recurrent model.

## Virtual Petri dish and chemotaxis

The first closed-loop task places the modeled worm in a circular dish and a
finite food pulse at a randomized location. Food follows the analytic solution
for a two-dimensional Gaussian diffusion pulse. An engineered head-local sensor
observes concentration and its recent history; source coordinates are available
to the training loss but never to the controller. The fitted seven-parameter
controller modulates the continuous `[speed, steering]` interface using sensory
adaptation and gait-phase sampling, a compact approximation of klinotaxis.

```bash
python scripts/train_petri_chemotaxis.py \
  --data-root "$R_ELEGANS_DATA_DIR" \
  --iterations 300
```

Training differentiates directly through the 95-muscle body, then evaluates
the learned commands through the fitted 302-voltage motor teacher and the fixed
Cook/Wang neuromuscular projection. This is behavior optimization, not yet RL:
RL becomes useful when the task includes richer sensory state, choices, and
delayed reward. It is also not yet a complete biological sensory circuit. The
head sensor and compact policy stand in for amphid transduction and recurrent
interneuron dynamics.

The seven fitted quantities are base speed; sine and cosine gains for adapted
log-concentration; sine and cosine gains for its temporal derivative; steering
bias; and food-proximity slowing. Body-direct fitting used 16 randomized source
locations and headings. On 24 held-out trials it reached the target radius in
91.7% of episodes, compared with 16.7% before fitting. Passing the same commands
through the supervised motor teacher and anatomical NMJs reached 62.5%. Neither
score measures recurrent-brain chemotaxis.

The initial environment uses normalized dimensions, clips the body center at
the dish margin, and treats diffusion as an unbounded Gaussian rather than a
no-flux circular-domain solution. Those approximations are explicit so they can
be replaced independently without changing the policy/body interface. Generated
fit artifacts and trajectories are written beneath `results/behavior/` in the
external data root and must not be committed.

## External scientific data

The small inference checkpoint is distributed with the package. Raw data,
complete training artifacts, and simulation results are never stored in this
Git repository. Select an external directory when reproducing fits or auditing
source data:

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
