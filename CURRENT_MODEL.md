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
| 95-muscle/12-segment body | Yes | Gait fitted; mechanics normalized | Yes |
| Diffusing food field | Yes | Hand-specified normalized parameters | Yes |
| Biological sensory-neuron model | No | No | No |
| Seven-parameter sensory controller | Yes | Differentiable behavior fit | Yes |
| RL policy | Yes, body-direct only | On-policy actor-critic (Gymnax) | Optional alternative to the row above |

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

`r_elegans.rl` fits the seven raw parameters against this environment with an
on-policy actor-critic policy-gradient update (generalized advantage
estimation, no importance-ratio clipping) instead of differentiating through
the simulator. A small MLP critic supplies the value baseline used to compute
advantages; it is discarded after training and is not part of the deployed
controller. The mean action reproduces
`r_elegans.envs.petri_dish.decode_sensory_policy` exactly
(`r_elegans.rl.policy.action_mean`); a state-independent learned log-standard-
deviation drives exploration only during training.

Reward per step is `10 · (previous distance − new distance) − 0.001 ·
steering² + 5 · [distance < 0.12]`; an episode terminates early on reaching the
target radius and truncates at the configured step budget. Source position and
initial heading are randomized every reset; the controller never observes
them, matching the differentiable-fit protocol above.

A run of `scripts/train_rl_chemotaxis.py` with 64 parallel environments, 250
steps per episode, and 300 updates (learning rate `0.003`, 4 full-batch epochs
per update) reached, on the same 24 held-out source/heading pairs used for the
differentiable fit:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Unfitted sensory controller, held out | 16.7% | 0.3584 |
| RL-trained controller, held out | 75.0% | 0.0658 |
| Differentiable-fit controller, held out (for reference) | 91.7% | 0.0451 |

RL reaches a large majority of held-out sources but does not match the
differentiable fit, which back-propagates an exact analytic gradient through
every simulated step rather than estimating a policy gradient from sampled
rollouts. This gap is the expected price of treating the simulator as a
black box; it is not evidence that the RL path is broken. Both paths deploy
the same seven-number controller shape, so an RL-trained checkpoint can be
evaluated, validated through the motor teacher, or visualized with the same
functions (`simulate_petri_dish`, `simulate_neural_petri_dish`) used for the
differentiable-fit checkpoint. `scripts/demo_rl_chemotaxis.py` trains a
controller this way and animates the resulting 12-segment body moving through
the dish.

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

## Terminology used by this project

- **Anatomical connectome:** fixed neuron identifiers and edge masks/counts.
- **Recurrent brain model:** membrane dynamics coupled through the anatomical
  neuron-to-neuron graph.
- **Motor teacher:** non-recurrent command/phase-to-302-output surrogate.
- **Sensory controller:** engineered seven-parameter concentration-to-command
  policy.
- **Current chemotaxis baseline:** sensory controller plus body gait, optionally
  validated through the motor teacher and neuromuscular anatomy.
