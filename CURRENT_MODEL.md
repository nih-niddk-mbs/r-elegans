# Current model: exact scope and computation path

This document describes what the repository executes today. It distinguishes
anatomical data, implemented equations, fitted surrogate controllers, and the
future recurrent brain so that “302-neuron model” is not mistaken for “the
connectome currently controls behavior.”

## Executive summary

The current food-finding baseline has three learned layers:

1. A 14-parameter body gait maps `[speed, steering]` to 95 body-wall muscle
   activations.
2. A supervised motor teacher maps command and gait phase to 302 neuron-indexed
   voltage outputs, then the fixed anatomical neuromuscular projection maps
   those outputs to the same 95 muscles.
3. A seven-parameter engineered sensory controller maps head concentration
   history and gait phase to `[speed, steering]`.

That seven-parameter controller can be fitted two ways: by differentiating
directly through the body/gait simulator (the baseline described below), or
by treating the same simulator as a black-box Gymnax environment and applying
model-free reinforcement learning (see "Body-direct reinforcement learning").
Both produce the identical controller shape and can be evaluated
interchangeably; only the fitting method differs.

The 302×302 chemical and gap-junction topology is bundled and the graded-
potential recurrent equations are implemented, but that recurrent network is
not in the active food-finding path. Its weights, cell parameters, sensory
transduction, and motor behavior have not been fitted as one closed loop.

A real, hand-selected 14-neuron *subset* of that connectome (chemosensory
AWC/ASE through interneurons AIY/AIZ, an RIA integrator, to dorsal/ventral
RMD head-motor readout) has been fitted as a closed loop and can replace the
seven-parameter controller as the body-direct actor -- see "Connectome-
subcircuit chemotaxis" below. This is not the 302-neuron recurrent network
described in the paragraph above; it is a small, literature-guided slice of
it, fitted by supervised imitation and then reinforcement-learning
fine-tuning, with speed control still handled by the same engineered formula
as the analytic controller. It should not be read as evidence that "the
connectome" performs chemotaxis.

## Status table

| Component | Present | Fitted | Active in food demo |
| --- | --- | --- | --- |
| 302 canonical neuron identifiers | Yes | Not applicable | Used for ordering |
| Chemical and gap-junction topology | Yes | No | No |
| Graded-potential recurrent equations | Yes | No | No |
| AWCON/RMD conductance models | Yes | Published whole-cell fits | No |
| Other neuron conductances | Interface only | No | No |
| 302-channel motor-output teacher | Yes | Supervised | Yes, neural mode |
| 956 neuron-to-muscle edges | Yes | Counts fixed; 880 signs assigned | Yes, neural mode |
| 95-muscle/12-segment body | Yes | SI liquid/agar mechanics; active gait fitted only in legacy normalized mode | Yes, legacy mode |
| Diffusing food field | Yes | Hand-specified normalized parameters | Yes |
| Biological sensory-neuron model | No | No | No |
| Seven-parameter sensory controller | Yes | Differentiable behavior fit | Yes |
| RL policy (analytic controller) | Yes, body-direct only | PPO or A2C (Gymnax) | Optional alternative to the row above |
| 14-neuron chemotaxis subcircuit (real anatomy, hand-selected) | Yes | Supervised pretrain + RL fine-tune | Optional alternative body-direct actor |
| Sensory transduction into the subcircuit | Engineered signal into real neurons | Trainable gains, not biophysical | Yes, when the subcircuit actor is used |

## Active food-finding computation

The body-direct training path is:

```text
food field at head
    → engineered concentration memory
    → fitted seven-parameter sensory controller
    → [speed, steering]
    → fitted traveling-wave gait
    → 95 muscle activations
    → 11 joint commands
    → 12-segment body mechanics
    → new head position
```

The neural validation path inserts the supervised motor teacher:

```text
[speed, steering] + gait phase
    → fixed 13-feature encoder
    → fitted 302-channel voltage output
    → fixed Cook/Wang neuron-to-muscle projection
    → 95 muscle activations
    → body mechanics
```

The 302×302 neuron-to-neuron chemical and gap-junction matrices do not appear in
either path. Therefore the current food-finding result is not evidence that the
recurrent connectome can perform chemotaxis.

## Food field and sensed variables

Food is a finite two-dimensional Gaussian pulse. For source position (x_s),
head position (x), mass (M), initial variance (v_0), and diffusion
coefficient (D):

```text
v(t) = v0 + 2 D t
C(x,t) = M / (2 π v(t)) · exp(-||x - xs||² / (2 v(t)))
```

The current normalized defaults are dish radius `1.5`, `M=1`, `v0=0.16`, and
`D=0.005`. The analytic pulse is an unbounded-domain approximation; it does not
yet impose a no-flux boundary at the circular dish wall.

At each step the engineered sensor computes:

- log concentration at the modeled head;
- difference between log concentration and an adapting baseline;
- first temporal difference of log concentration;
- sine and cosine of the locomotor gait phase.

The adaptation time constant is `0.75` normalized seconds. There are no modeled
receptor kinetics, amphid geometry, bilateral sensory neurons, or sensory noise
in this baseline.

## What the seven-parameter fit learned

Let `r` be adapted log-concentration response, `dr/dt` be its temporal
difference, and `phi` be gait phase. Steering is:

```text
tanh(
    response_sine_gain     · r     · sin(phi)
  + response_cosine_gain   · r     · cos(phi)
  + 0.1 derivative_sine_gain   · dr/dt · sin(phi)
  + 0.1 derivative_cosine_gain · dr/dt · cos(phi)
  + steering_bias
)
```

Speed is:

```text
base_speed · (1 - food_slowing · relative_concentration)
```

The fitted physical parameters are:

| Parameter | Value |
| --- | ---: |
| Base speed | 0.7140385 |
| Adapted-response sine gain | 7.978845 |
| Adapted-response cosine gain | -7.581171 |
| Concentration-change sine gain | -10.060525 |
| Concentration-change cosine gain | -8.544154 |
| Steering bias | 0.041312 |
| Food-proximity slowing | 0.125010 |

The controller infers direction from temporal concentration changes during the
body-driven head sweep. This is an engineered approximation to klinotaxis, not
a reconstruction of the biological sensory circuit.

## Sensory-controller fitting protocol

The seven raw parameters were optimized with Adam for 300 iterations at
learning rate `0.02`. Each episode contained 500 steps of `0.02` normalized
seconds. Training used 16 reproducibly randomized source positions and initial
headings; evaluation used 24 held-out combinations.

The mean episode objective was:

```text
minimum source distance
+ 0.25 · final source distance
+ 0.001 · mean squared steering
```

Source coordinates were used only by this objective. The controller received
no source coordinates, bearing, or distance.

Results using a target radius of `0.12` body-length units:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Unfitted sensory controller, held out | 16.7% | 0.3584 |
| Fitted controller through body gait, training | 87.5% | 0.0672 |
| Fitted controller through body gait, held out | 91.7% | 0.0451 |
| Same controller through motor teacher and NMJs, held out | 62.5% | 0.2028 |

The reduction from 91.7% to 62.5% measures accumulated approximation error in
the motor teacher and neuromuscular transform. It is not a recurrent-brain
score.

## Body-direct reinforcement learning (alternative fitting method)

`r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv` exposes the identical
body-direct food field, gait, and body mechanics through a Gymnax-compatible
`reset`/`step` interface. It reveals only the same four sensed quantities
(adapted-response, its derivative, and gait-phase sine/cosine) plus relative
concentration, and accepts the same `[speed, steering]` command; the
environment is agnostic to how that command is produced.

`r_elegans.rl` fits the seven raw parameters against this environment with
model-free reinforcement learning instead of differentiating through the
simulator, using generalized advantage estimation and a small MLP critic that
supplies the value baseline used to compute advantages; the critic is
discarded after training and is not part of the deployed controller. Two
update rules are implemented, sharing the same rollout collection and GAE:

- **PPO** (`r_elegans.rl.make_ppo_train_step`, the default): the clipped
  surrogate objective. Each update collects one on-policy rollout, then runs
  several epochs over randomly shuffled environment-sequence minibatches,
  clipping the policy-ratio so a batch can be reused for multiple gradient
  steps without drifting too far off-policy. Time order is preserved within
  every sequence for recurrent actors.
- **A2C** (`r_elegans.rl.make_a2c_train_step`): plain policy gradient with the
  same GAE baseline but no ratio clipping, taking exactly one full-batch
  gradient step per freshly collected rollout.

The mean action reproduces `r_elegans.envs.petri_dish.decode_sensory_policy`
exactly (`r_elegans.rl.policy.action_mean`); a state-independent learned
log-standard-deviation drives exploration only during training.

Reward per step is `10 · (previous distance − new distance) − 0.001 ·
steering² + 5 · [distance < 0.12]`; an episode terminates early on reaching the
target radius and truncates at the configured step budget. Source position and
initial heading are randomized every reset; the controller never observes
them, matching the differentiable-fit protocol above.

Natural terminations have zero bootstrap value. Time-limit truncations instead
bootstrap from Gymnax's pre-reset final observation, and GAE is cut at the
episode boundary so returns never leak into the automatically reset episode.
Training success is successes divided by completed episodes, not the fraction
of individual timesteps carrying a success flag.

The following numbers are historical results from the earlier normalized-body
RL implementation. They have not yet been reproduced after the corrected
time-limit handling, recurrent BPTT update, and body-mechanics upgrade, and are
not current WormGym benchmarks.

A run of `scripts/train_rl_chemotaxis.py` with 64 parallel environments, 250
steps per episode, and 300 updates (learning rate `0.003`, 4 epochs per
update; PPO additionally split each batch into 4 minibatches with clip range
`0.2`) reached, on the same 24 held-out source/heading pairs used for the
differentiable fit:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Unfitted sensory controller, held out | 16.7% | 0.3584 |
| A2C-trained controller, held out | 75.0% | 0.0658 |
| PPO-trained controller, held out | 87.5% | 0.0523 |
| Differentiable-fit controller, held out (for reference) | 91.7% | 0.0451 |

Both RL update rules reach a large majority of held-out sources; PPO's
ratio clipping lets it use its rollouts more effectively than A2C's
full-batch updates at the same training budget, closing most of the gap to
the differentiable fit, which back-propagates an exact analytic gradient
through every simulated step rather than estimating a policy gradient from
sampled rollouts. This gap is the expected price of treating the simulator as
a black box; it is not evidence that the RL path is broken. Both paths deploy
the same seven-number controller shape, so an RL-trained checkpoint can be
evaluated, validated through the motor teacher, or visualized with the same
functions (`simulate_petri_dish`, `simulate_neural_petri_dish`) used for the
differentiable-fit checkpoint. `scripts/demo_rl_chemotaxis.py` trains a
controller this way and animates the resulting 12-segment body moving through
the dish.

## Connectome-subcircuit chemotaxis (partial, body-direct only)

The seven-parameter controller can also be replaced entirely by a real,
hand-selected 14-neuron subcircuit of the bundled connectome, trained by
supervised imitation and then reinforcement learning, while everything else
in the body-direct pipeline (food field, gait, body mechanics, reward) stays
identical. This is a step toward, not the completion of, the "connectome-
driven worm" described below -- 14 of 302 neurons, not the full recurrent
network.

**The subcircuit** (`r_elegans.brain.circuit.SUBCIRCUIT_NEURON_NAMES`):
chemosensory `AWCL/AWCR/ASEL/ASER`, interneurons `AIYL/AIYR` then
`AIZL/AIZR`, integrator `RIAL/RIAR`, and dorsal/ventral head-motor readout
`RMDDL/RMDDR/RMDVL/RMDVR`. This is not an assumed pathway: the bundled
connectome asset contains 63 real chemical synapses and 8 real gap junctions
among exactly these 14 neurons (e.g. `AIYL→AIZL` count 67, `RIAL→RMDDL` count
48, `RIAR→RMDDR` count 53), matching the textbook AWC/ASE→AIY→AIZ→RIA→RMD
klinotaxis pathway. The dorsal/ventral RMD subclasses are read out (rather
than the more obviously bilateral RMDL/RMDR) because the simulator's
`steering` command is itself a dorsal-minus-ventral bending bias, and RIA is
the documented site where left/right sensory asymmetry becomes exactly that
bias.

**Dynamics**: the existing graded-potential equations
(`r_elegans.brain.dynamics.neural_rhs`) integrated with a fixed-step RK4
(`integrate_neural_fixed_step`), in the same normalized unit system as the
rest of the body-direct simulator (not the separate, literal-mV/ms
`single_compartment.py` model, which is calibrated for only 2 of 302 neuron
classes and has no cross-neuron coupling). Anatomical chemical/gap masks are
protected from gradient updates (`jax.lax.stop_gradient` in
`effective_chemical_weights`/`effective_gap_weights`) so training can never
drift the fixed 0/1 topology into arbitrary floats; per-neuron time constant
and activation slope are trained through a positivity floor rather than as
bare unconstrained values, matching this repository's existing `decode_*`
convention. `synapse_reversal` (excitatory/inhibitory character) has no real
per-edge sign dataset in this repository; it is initialized mostly excitatory
with one literature fact hand-set (AIY inhibits AIZ) and is otherwise
trainable -- a disclosed judgment call, not a fitted quantity.

**Sensory transduction and readout are only partly biological.** `response`
and `derivative` (adapted log-concentration and its rate) are the same
signal real ASE/AWC neurons are documented to encode as ON/OFF adaptation,
injected as current into the real sensory neuron indices through trainable
gains -- but AWC and ASE currently receive the *same* underlying signal, and
`sin(phase)/cos(phase)` (an explicit stand-in for proprioceptive/CPG
coupling, since gait phase is not itself a modeled biological quantity here)
is injected the same way. Only `steering` is connectome-driven; `speed` keeps
the analytic controller's plain food-proximity-slowing formula, to keep the
biological claim (about steering/klinotaxis specifically) falsifiable and to
keep failure modes separable.

**Training proceeded in two stages**, per the roadmap in "What remains for a
connectome-driven worm": supervised pretraining first
(`scripts/pretrain_connectome_circuit.py`), then RL fine-tuning
(`scripts/train_rl_chemotaxis.py --actor connectome`). Pretraining
back-propagates an MSE loss through a full within-episode unroll of the
subcircuit's own recurrence to imitate the bundled differentiable-fit
controller's trajectories -- this stage is not subject to the RL rollout's
stop-gradient boundary and differentiates through time. Corrected recurrent
PPO/A2C now also backpropagates through the actor recurrence while continuing
to treat environment dynamics as a black box. The following measurements are
historical results from before that RL correction and require reproduction:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Untrained subcircuit, held out | 4.2% | 0.6372 |
| Supervised-pretrained subcircuit, held out | 45.8% | 0.1888 |
| + gentle PPO fine-tuning (`lr=1e-4`), held out | 45.8% | 0.1702 |
| Seven-parameter PPO, held out (for reference) | 87.5% | 0.0523 |

Supervised pretraining alone recovered 45.8% of held-out sources
from a real anatomical subcircuit with mostly-engineered sensory input --
substantially below the seven-parameter controller, consistent with a
14-neuron slice being a much more constrained function class than a
hand-designed 7-parameter formula tuned specifically for this task. RL
fine-tuning at the same learning rate used for the analytic controller's
from-scratch training (`3e-3`) *degraded* the pretrained checkpoint (down to
8.3% success) rather than improving it: the actor starts from a good
supervised optimum while its critic starts randomly initialized, and early
noisy value estimates produced large, destabilizing policy updates. A
substantially smaller fine-tuning learning rate (`1e-4`, with a matching
smaller entropy coefficient) avoided this degradation and modestly improved
mean minimum distance, though it did not raise the held-out success rate in
this run. This is reported as an honest, current limitation, not smoothed
over: further tuning of the fine-tuning learning-rate schedule, reward
shaping, or a critic warm-start are the most likely paths to closing more of
the remaining gap to the analytic controller.

`r_elegans.rl.actor_interface.ActorInterface` lets PPO/A2C train either actor
architecture. The connectome voltage is carried alongside environment state
during collection. During each update the actor is unrolled again over each
time-ordered environment sequence, resetting at episode boundaries and using
backpropagation through time. PPO shuffles whole sequences between minibatches,
never isolated recurrent timesteps.

## Supervised motor teacher

The motor teacher receives `[speed, steering]` and gait phase. Its 13 features
encode a bias; forward and reverse magnitudes; their sine/cosine phase terms;
and direction-specific steering and steering-phase interactions. A learned
`[302, 13]` coefficient matrix produces bounded voltages:

```text
V = -60 + 80 · sigmoid(features · coefficientsᵀ)
```

Only 129 neuron identities with conservatively signed neuromuscular outputs
were trainable. The remaining channels stay near rest. The fixed Cook contact
counts and Wang-derived signs then produce 95 muscle activations. Training used
12 forward/reverse/turn commands and achieved muscle RMSE `0.0570` and body
endpoint RMSE `0.0215` normalized body lengths.

The teacher has no neuron-to-neuron recurrence, membrane ODE, sensory current,
or internal neural state. “302-channel neural output” is the precise term;
“trained 302-neuron brain” is not.

## Anatomical topology in the repository

The bundled runtime asset contains:

- 302 neurons in one canonical order;
- 3,709 directed neuron-to-neuron chemical edges and contact counts;
- 1,093 undirected gap-junction pairs and counts;
- 956 neuron-to-body-wall-muscle edges and counts;
- conservative polarity for 880 neuromuscular edges;
- 95 anatomical body-wall muscle identifiers.

All matrices use `[postsynaptic, presynaptic]` indexing. The topology is needed
to construct the future recurrent model, but topology alone does not determine
synaptic conductance, receptor sign, membrane dynamics, sensory encoding, or a
behavioral policy.

## What remains for a connectome-driven worm

The next model must replace the engineered sensory controller and motor teacher
with a single recurrent computation:

1. Map food concentration and its dynamics to identified chemosensory neurons
   such as ASE, AWA, and AWC using documented receptor/transduction models.
2. Assign or fit neuron-specific membrane parameters beyond AWCON and RMD.
3. Fit chemical strengths, gap conductances, reversal potentials, thresholds,
   and time constants while preserving the anatomical masks.
4. Drive the fixed neuromuscular projection from recurrent voltages.
5. Train first against the existing sensory-controller and motor-teacher
   trajectories, then fine-tune closed-loop behavior.
6. Evaluate held-out source locations, initial postures, concentration fields,
   ablations, and robustness without exposing source coordinates.

RL is optional at that stage. Supervised trajectory matching and differentiable
behavior optimization are cheaper first steps; RL becomes appropriate for
delayed reward, state uncertainty, competing objectives, and richer tasks.

"Connectome-subcircuit chemotaxis" above is a narrow, partial step along
items 1, 3, and 5 for a hand-selected 14-neuron slice only -- not items 2, 4,
or 6, and not the full 302-neuron network. Item 1 in particular is only
partly addressed: sensory current reaches real chemosensory neuron indices,
but through the same engineered adaptation signal the analytic controller
uses, not a documented receptor/transduction model. The fixed neuromuscular
projection (item 4) is still not driven by recurrent voltages; the subcircuit
drives `[speed, steering]` into the existing fitted gait, exactly like the
analytic controller.

## Terminology used by this project

- **Anatomical connectome:** fixed neuron identifiers and edge masks/counts.
- **Recurrent brain model:** membrane dynamics coupled through the anatomical
  neuron-to-neuron graph.
- **Motor teacher:** non-recurrent command/phase-to-302-output surrogate.
- **Sensory controller:** engineered seven-parameter concentration-to-command
  policy.
- **Current chemotaxis baseline:** sensory controller plus body gait, optionally
  validated through the motor teacher and neuromuscular anatomy.
