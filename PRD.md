# Product Requirements Document: `r-elegans`

**Status:** Living specification

**Last updated:** 2026-09-01

**Target stack:** JAX, Diffrax, and a vectorized JAX environment interface

**Long-term objective:** A differentiable, anatomically constrained model in
which environmental stimuli enter identified sensory neurons, recurrent
302-neuron dynamics generate motor-neuron activity, anatomical neuromuscular
connections activate the body, and behavior changes the sensed environment.

## 1. Scope and scientific claim

`r-elegans` is a staged research simulator. It combines experimentally derived
anatomical topology with explicit models for neural dynamics, muscles, body
mechanics, sensory environments, and learning.

The project must distinguish four different forms of completeness:

1. **Anatomical completeness:** canonical cells and observed connection
   topology are present.
2. **Dynamical completeness:** every modeled cell and edge has enough numerical
   parameters to integrate the equations.
3. **Behavioral completeness:** the closed loop can generate locomotion and
   task-directed behavior.
4. **Biological calibration:** numerical parameters are constrained by
   measurements with documented evidence and uncertainty.

Possessing a 302-neuron adjacency matrix satisfies only part of anatomical
completeness. It does not by itself create a functioning or biologically
calibrated brain.

## 2. Current implementation baseline

The repository currently provides:

- canonical ordering and sparse counts for 302 neurons, 3,709 directed chemical
  edges, and 1,093 undirected gap-junction pairs;
- 956 neuron-to-body-wall-muscle edges, with conservative polarity assigned to
  880;
- general differentiable graded-potential recurrent equations constrained by
  chemical and gap-junction masks;
- detailed single-compartment channel equations, with published whole-cell
  parameter fits for AWCON and RMD only;
- a 95-muscle, 12-segment planar body with normalized resistive-force mechanics;
- a fitted continuous `[speed, steering]` gait;
- a supervised 13-feature command/phase-to-302-output motor teacher;
- a circular Petri dish, finite diffusing food pulse, and head-local
  concentration measurement;
- a fitted seven-parameter engineered sensory controller that produces
  `[speed, steering]` from concentration history and gait phase.

The current food-finding demo does not pass sensory input through the recurrent
connectome. Its active path is:

```text
food at head
  → engineered seven-parameter sensory controller
  → [speed, steering]
  → either fitted body gait directly
     or supervised 302-output motor teacher + anatomical NMJs
  → 95 muscles
  → body mechanics
```

The neuron-to-neuron topology is bundled and loadable but is not active in this
baseline. See [CURRENT_MODEL.md](CURRENT_MODEL.md) for exact equations,
parameters, fit protocol, and measured results.

## 3. Target closed-loop architecture

The target system is:

```text
diffusing chemical field
    ↓ local receptor stimulus
identified chemosensory neurons
    ↓ recurrent chemical synapses and gap junctions
302-neuron graded-potential network
    ↓ anatomical neuron-to-muscle connections
95 body-wall muscles
    ↓ local dorsal/ventral projection
12–24 segment body and substrate mechanics
    ↓ movement changes head and body position
diffusing chemical field
```

No target coordinate, bearing, distance-to-food, reward, or privileged
environment state may be supplied as neural input. Those quantities may be
used by a training objective or evaluator only.

## 4. Functional requirements

### 4.1 Runtime and packaging

- A fresh clone must run a representative checkpoint after installing declared
  Python dependencies.
- The compact runtime asset must be versioned in the repository and include all
  inference-critical topology, ordering, and learned parameters.
- Raw publications, spreadsheets, validation traces, full optimizer histories,
  and generated trajectories must remain outside Git.
- Every bundled or external scientific artifact must have source attribution,
  transformation notes, indexing conventions, checksum, and schema version.
- The runtime loader must validate array shapes, identifiers, finite values,
  nonnegative counts, and gap-junction symmetry.

### 4.2 Connectome and cell identity

- All neuron-to-neuron matrices use `[postsynaptic, presynaptic]` indexing.
- Chemical masks are directed; gap-junction masks and effective conductances are
  symmetric.
- Canonical ordering must be shared by recurrent state, sensory inputs, motor
  outputs, and neuromuscular matrices.
- Anatomical contact count must remain distinct from fitted effective strength.
- Neurotransmitter identity must remain distinct from receptor-dependent edge
  effect. Unknown polarity must remain explicit rather than guessed.

### 4.3 Neural dynamics

The recurrent baseline uses continuous graded potentials:

```text
dV_i/dt = (I_leak,i + I_gap,i + I_chem,i + I_ext,i) / tau_i
```

Required mechanisms:

- leak toward a neuron-specific reversal potential;
- bidirectional gap current proportional to connected voltage differences;
- presynaptic sigmoid activation for chemical transmission;
- reversal-potential-dependent chemical current;
- external sensory current delivered only to identified sensory neurons;
- stable JAX/Diffrax integration and differentiability through time.

The model must expose anatomical masks separately from trainable chemical
strengths and gap conductances. It must support parameter sharing by neuron
class as well as neuron-specific values.

### 4.4 Sensory system

The first biological sensory milestone is chemotaxis. It must specify:

- which amphid neurons receive food-related input;
- whether each neuron responds to absolute level, increases, decreases,
  derivatives, or temporal filters;
- receptor/transduction kinetics and adaptation state;
- left/right or head-motion geometry where supported;
- stimulus-to-current units and saturation;
- sensory delays, noise, and uncertainty where evidence exists.

The current engineered seven-parameter controller is a behavioral teacher and
benchmark. It is not an acceptable final sensory-neural implementation.

### 4.5 Neuromuscular system

- Neural voltage must be transformed through the fixed `[95, 302]`
  neuron-to-muscle topology.
- Contact counts, polarity, neural thresholds, neural slopes, muscle thresholds,
  and muscle slopes must remain independently inspectable.
- The 95 anatomical muscle identities and dorsal/ventral quadrants must be
  preserved.
- Local projection to body joints must not destroy anterior/posterior ordering.
- Unknown NMJ signs must be excluded, marginalized, or fitted under explicit
  uncertainty; they must not silently become excitatory.

### 4.6 Body mechanics

- The body is an articulated planar midline with explicit joint angles and
  dorsal-minus-ventral activation state.
- Passive bending stiffness, damping, muscle moment, activation kinetics,
  substrate drag, and joint limits are separate parameters.
- Low-Reynolds force and torque balance determines rigid-body translation and
  rotation from shape change.
- The current normalized mechanics are suitable for controller development but
  must not be described as an adult-worm biomechanical calibration.
- A later calibration must fit posture wavelength, frequency, speed, turning
  radius, and medium-dependent drag to measured trajectories.

### 4.7 Petri-dish environment

- The environment contains a circular boundary, body pose, chemical source,
  concentration field, and time.
- The initial food field may use an analytic Gaussian diffusion pulse.
- A later version should solve diffusion with a no-flux dish boundary and
  configurable source deposition/consumption.
- Sensory functions may query concentration only at modeled receptor locations.
- Training/evaluation may access source distance, but the neural controller may
  not.
- Episodes must support randomized source location, initial heading, posture,
  concentration width, diffusion coefficient, and noise.

### 4.8 Learning strategy

Training proceeds in inexpensive stages:

1. Fit body motion primitives and the continuous action interface.
2. Fit neural motor outputs to the muscle/body motion library.
3. Fit a compact engineered sensory controller as a closed-loop behavioral
   teacher.
4. Fit the recurrent network by supervised trajectory matching to sensory and
   motor teachers while preserving connectome masks.
5. Fine-tune the recurrent closed loop through differentiable behavior loss.
6. Use RL only for tasks with delayed reward, partial observability, competing
   objectives, or exploration that supervised/differentiable objectives do not
   cover.

RL is not required merely to demonstrate locomotion or gradient following.

## 5. Current quantitative baseline

The current sensory-controller fit used 16 randomized training episodes and 24
held-out episodes, each 500 steps at `dt=0.02`.

| Path | Held-out food-zone success | Meaning |
| --- | ---: | --- |
| Unfitted sensory controller | 16.7% | Initialization baseline |
| Fitted controller → body gait | 91.7% | Engineered policy/body performance |
| Fitted controller → motor teacher → NMJs → body | 62.5% | Surrogate neural-output validation |
| Sensory neurons → recurrent connectome → NMJs → body | Not implemented | Target brain-driven behavior |

Success means minimum head-to-source distance below `0.12` normalized body
lengths. These scores must not be relabeled as recurrent-brain training.

## 6. Roadmap and status

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | Repository, JAX foundation, external-data policy | Complete |
| 2 | Canonical neuron and neuromuscular topology | Complete |
| 3 | Graded-potential and single-compartment equations | Implemented; sparsely calibrated |
| 4 | Muscle body and continuous action space | Implemented; normalized/fitted baseline |
| 5 | Supervised 302-output motor teacher | Complete baseline; not recurrent |
| 6 | Petri dish and engineered chemotaxis teacher | Complete baseline |
| 7 | Biological sensory transduction into identified neurons | Not started |
| 8 | Recurrent connectome fit to neural/motor teachers | Not started |
| 9 | Closed-loop recurrent chemotaxis | Not started |
| 10 | Gymnax-style batched RL tasks | Not started |
| 11 | Whole-animal biological calibration and validation | Not started |

## 7. Acceptance gates

### Gate A: self-contained inference

- A clean installation runs `r-elegans-demo` without an external data root.
- Bundled topology and parameter assets pass schema and checksum validation.
- No training trajectory or raw source file is required for inference.

### Gate B: anatomical integrity

- Exactly 302 canonical neuron identifiers are present.
- Chemical nonzeros match the derived directed topology.
- Gap-junction matrices are symmetric.
- NMJ arrays retain 95 muscles in canonical order.

### Gate C: numerical integrity

- Neural, muscle, body, and environment steps are finite and JIT-compatible.
- Gradients through representative unrolls are finite and nonzero where
  expected.
- Batched execution produces the same result as individual execution within
  numerical tolerance.

### Gate D: motor reconstruction

- The supervised motor teacher reproduces held-out command trajectories within
  documented muscle and endpoint error tolerances.
- Evaluation passes through the fixed neuromuscular topology and body, not only
  a direct command comparison.

### Gate E: recurrent sensory-motor closure

This gate is **not yet satisfied**. It requires:

- environmental concentration converted to currents in identified sensory
  neurons;
- recurrent 302-neuron integration using the anatomical chemical and gap masks;
- recurrent voltages driving the anatomical NMJ projection;
- movement changing subsequent sensory input;
- held-out food finding without the engineered sensory controller or supervised
  command-to-voltage motor teacher.

### Gate F: biological validation

This gate is **not yet satisfied**. It requires comparison with experimental
neural traces, locomotor statistics, sensory ablations, and behavior across
multiple stimulus geometries, with uncertainty and provenance reported.

## 8. Explicit nonclaims

Until Gates E and F are passed, documentation and results must not claim that:

- the recurrent 302-neuron brain has been trained;
- the connectome itself finds food;
- all neuron-to-neuron weights or signs are known;
- all 302 neurons have calibrated ion-channel compositions;
- the body is quantitatively adult-worm accurate;
- the current behavior is an RL result;
- topology alone constitutes a complete functional brain.
