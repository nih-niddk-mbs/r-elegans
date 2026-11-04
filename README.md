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

The [behavior catalog](BEHAVIOR_CATALOG.md) maps documented animal behaviors to
measurable assays and missing simulator capabilities. The companion
[WormGym training protocols](TRAINING_PROTOCOLS.md) define the proposed
imitation/RL curriculum, biological controls, and validation gates.

Full recurrent training is now available separately from the bundled demo:
`scripts/train_timed.py --actor full-network` runs all 302 recurrent neurons
and anatomical NMJs to 95 muscles, with no fitted gait or motor teacher.
Whole-network optimization is the default; optional parameter subsets leave
all 302 neurons in the simulation. This path is smoke-tested; full-network
food-finding and biological behavioral fidelity remain unvalidated.

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

Biophysical calibration applies to AWCON and RMD, and the body now has explicit
adult liquid-swimming and agar-crawling calibrations in SI units. The existing
food task still uses its legacy normalized body while its gait and dish are
re-fitted; it must not be mixed with an SI-valued body. A recurrent
connectome-constrained controller remains a later milestone described in
[PRD.md](PRD.md).

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

The planar body accepts one dorsal and one ventral activation value per joint.
Opposing muscles generate active bending moments. A coupled overdamped solve
then balances those moments against body elasticity, internal viscosity, and
environmental resistance while imposing zero net external force and torque.
Finite-segment rotational drag is included as well as centerline translation.
A normalized muscle-driven rollout is:

```python
from r_elegans.body import (
    default_muscle_body_params,
    simulate_muscle_wave,
)

params = default_muscle_body_params(num_segments=12)
final_state, trajectory = simulate_muscle_wave(params, steps=500, dt=0.01)
```

`default_muscle_body_params` remains the backwards-compatible normalized mode.
For physical simulations, construct an adult body explicitly:

```python
from r_elegans.body import physical_muscle_body_params

swimmer = physical_muscle_body_params(12, medium="liquid")
crawler = physical_muscle_body_params(12, medium="agar")
```

These factories use meters, seconds, newton-meters, and kilogram-based drag
units. Liquid drag is computed from viscosity, body radius, and gait wavelength
with Lighthill local RFT. Agar uses published effective whole-worm coefficients
with perpendicular/parallel drag ratio 40. Absolute load now affects joint
dynamics; viscosity is no longer canceled out of the actuation model. See
[PHYSICS.md](PHYSICS.md) for equations, provenance, tests, and validity limits.

Primary mechanics references: [Fang-Yen et al., PNAS 2010](https://doi.org/10.1073/pnas.1003016107)
and [Boyle et al., Frontiers in Computational Neuroscience 2012](https://doi.org/10.3389/fncom.2012.00010).

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

## Body-direct reinforcement learning

The same seven-parameter controller and body-direct simulator are also
reachable through a Gymnax-compatible environment,
`r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv`, so they can be fitted
with model-free reinforcement learning instead of by differentiating through
the simulator. `r_elegans.rl` implements two update rules around the
identical controller formula in
`r_elegans.envs.petri_dish.decode_sensory_policy`, both using generalized
advantage estimation with a small MLP value baseline used only during
training: **PPO** (the clipped surrogate objective, default) and a simpler
**A2C** policy gradient with no ratio clipping. Only this body-direct path is
covered; the supervised motor teacher and neuromuscular projection are not
part of the RL loop.

```bash
pip install -e ".[env,demo]"
python scripts/train_rl_chemotaxis.py --num-updates 300            # PPO (default)
python scripts/train_rl_chemotaxis.py --num-updates 300 --algorithm a2c
python scripts/demo_rl_chemotaxis.py --num-updates 200
```

`train_rl_chemotaxis.py` trains and reports held-out success/distance exactly
as `train_petri_chemotaxis.py` does, so the fitting methods are directly
comparable. `demo_rl_chemotaxis.py` trains (or loads a saved policy via
`--load`) and animates the resulting 12-segment body moving through the dish
toward the food source. In historical normalized-body runs on the same 24
held-out source/heading pairs used for the differentiable fit, 300 updates reached 87.5% success (mean minimum
distance 0.0523) with PPO and 75.0% (0.0658) with A2C, versus 91.7% (0.0451)
for the differentiable fit and 16.7% for the unfitted controller -- both RL
rules find a strong majority of sources without ever differentiating through
the physics, and PPO's clipped surrogate lets it reuse each rollout more
effectively than A2C at the same budget, but neither matches an exact
analytic gradient. These numbers predate the corrected recurrent/time-limit
training logic and body-physics upgrade, so they must be rerun before being
treated as current WormGym benchmarks. See `CURRENT_MODEL.md` for scope.

## Full recurrent network training

Start with supervised motor milestones before food seeking. The first
implemented milestone fits **one small static dorsal bend**, with all 302
neurons running and only recurrent neural parameters learning. NMJ strengths,
muscle transfer parameters and sensory gains stay fixed. There is no RL or
teacher/clock input to the brain:

```bash
python scripts/train_motor_bend.py \
  --output-dir "$HOME/OneDrive/r-elegans/results/motor_curriculum/bend-$(date -u +%Y%m%dT%H%M%SZ)"
```

The target is a mild, synthetic 95-muscle pattern around 40% activation, with
a smooth dorsal/ventral difference of at most four percentage points per side.
Training uses 64 steps (1.28 s); verification free-runs the brain for twice as
long, then drives the body with its actual muscle outputs. Passing requires
muscle RMSE <0.03, body-angle relative RMSE <20%, last-quarter hold error <20%,
and the correct bend direction. Failed gates return exit code 2 and retain
artifacts for inspection; they do not advance the curriculum.

A seed-0 Mac run (500 supervised updates, about 18 s) reduced evaluation muscle
RMSE from 0.455 to 0.0099 and body-angle relative error from 109% to 12.6%,
passing the longer-hold check. This establishes one autonomous held bend in
normalized mechanics. That run did not establish command-controlled movement,
alternating bends, swimming or biological fidelity. The subsequent milestone is
controlled alternating bends (below), followed by a slow traveling wave, then
forward/reverse motion and turns. Food-seeking RL comes after those motor
policies work. The longer RL commands below remain available for later use.

For command-controlled alternating bends, start with the held-bend checkpoint:

```bash
python scripts/train_motor_alternation.py \
  --initial-actor /external/bend/actor_best.npz \
  --output-dir /external/alternation --iterations 2000
```

This stage supplies a scalar dorsal/ventral command as fixed, artificial
currents to NMJ-projecting neurons. The distribution is derived from signed
outgoing muscle contacts. It is a research stimulation interface, not an
identified biological command circuit. Commands never bypass the recurrent
neurons or the fixed NMJs. Sensory gains and all NMJ/muscle transfer parameters
stay frozen; recurrent neural parameters learn. A held-out slower sinusoid and
changed stop/reverse command sequence run twice as long as training. Each must
pass muscle RMSE <0.025, body-angle relative RMSE <25%, bend correlation >0.9,
and direction agreement >95%. A constant bend cannot pass these gates. This
does not establish an autonomous oscillator or locomotion.

A seed-0 Mac continuation passed these checks after 1,200 supervised updates
(about four minutes): body-angle error 16.9%/18.6%, correlation 0.997/0.992 and
direction agreement 100% on the two unseen sequences. This is one fitted seed;
the Linux ensemble below is intended to assess reliability across seeds.

The full-network dish and these motor checks now explicitly map anterior
muscle rows to the head end of the body. The old projection reversed that
relationship; it remains available for reproducing historical fitted gaits.
Wave targets must use the corrected ordering.

For an unattended, approximately one-hour supervised run on mbslnx2, in the
activated `r-elegans` environment:

```bash
git pull --ff-only
nohup python -u scripts/run_motor_curriculum.py --minutes 60 \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-curriculum.log 2>&1 < /dev/null &
```

It runs four independent seeds, each on four disjoint physical CPU cores.
Each seed must pass held-bend fitting before it can train alternating bends.
Checkpoints, losses, command sequences and muscle/body verification arrays
remain outside Git. The time budget includes the initial bend stage; final
verification can extend slightly beyond it. A failed bend gate retains that
seed's artifacts and stops its progression. Exit code 2 means a motor gate
failed (or the worker was interrupted); numerical errors are reported in each
stage's summary. Fresh runs stop at alternating bends; the explicit resume
options below start wave fitting. The runner never starts food-seeking RL.
For single-device GPU fitting, invoke the
stage script directly with `CUDA_VISIBLE_DEVICES=0` and CPU pinning omitted.

After all four alternating-bend seeds pass, continue them into slow traveling
bends on mbslnx2. Activate the existing conda environment before launching:

```bash
source /home/carsonc/miniconda3/etc/profile.d/conda.sh
conda activate r-elegans
git pull --ff-only
nohup python -u scripts/run_motor_curriculum.py --resume-latest --minutes 60 \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-wave.log 2>&1 < /dev/null &
```

`--resume-latest` chooses the newest `motor-*` directory with passed alternation
summaries and matching checkpoints for every requested seed. It reuses each
seed's own `alternation/actor_best.npz`; optimizers reset. Use `--resume-run
/path/to/motor-RUN` to select a run explicitly. Results go to a fresh `wave-*`
directory; earlier runs are preserved. Dependency preflight rejects a Python
environment missing JAX/Optax/Flax before launching workers.

The wave stage uses the same scalar artificial stimulation command, with no
phase, spatial delay or target-muscle input supplied to the brain. Only the
supervised target delays that command progressively along the anatomical
muscle rows: 0.4 seconds from anterior to posterior at the initial amplitude
of four percentage points per side. Commands are slow sinusoids or held
bends; the target is causal and uses zero command before time zero. This
tests a small externally paced traveling bend, not autonomous oscillation or
net locomotion. NMJ/muscle parameters remain fixed; neural dynamics must learn
the spatial timing. Validation changes command rate, phase, levels and order,
and doubles the training horizon.

In addition to the alternating-bend gates, both muscle and body signals must
show posterior following anterior by at least 40 ms, and their lags must match
the corresponding target within 60 ms. Overlap-normalized cross-correlation
searches both directions, after initial settling. Tests reject uniform bends
and reversed waves. Passing does not establish propulsion; net displacement,
forward/reverse gaits and turning remain subsequent milestones.

A warm-started local seed passed after 3,000 wave updates (about ten minutes
on the Mac). Body-angle error was 10.7%/16.9% and anterior/posterior body-band
lags were 160/140 ms against 140 ms targets on the two longer validation
sequences. The first 1,500-update fit failed one propagation check; continuing
the same checkpoint passed the fixed criteria. This is one externally paced
traveling-bend fit, not demonstrated autonomous swimming or an ensemble result.

The full-network actor runs all 302 neurons, 3,709 chemical edges, 1,093 gap
pairs and the 956-edge projection to 95 muscles. It uses generic normalized
graded-potential cells and engineered chemical sensors. Anatomical counts
initialize relative strengths. Known NMJ signs and topology stay fixed;
unknown NMJ signs can learn. This is an experimental training model, without
validated biological behavior or a calibrated physical gait.

Benchmark this larger workload on mbslnx2 before choosing CPU versus GPU:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
python scripts/benchmark_rl.py --actor full-network --require-gpu \
  --num-envs 32 --num-steps 128 > /home/carsonc/r-elegans-results/full-a100.json
```

For a one-hour run on one A100, from the activated `r-elegans` environment:

```bash
run_dir="/home/carsonc/r-elegans-results/full-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python -u scripts/train_timed.py --actor full-network --train-group all \
  --minutes 60 --num-envs 32 --episode-steps 250 --learning-rate 1e-4 \
  --output-dir "$run_dir" > "$run_dir/run.log" 2>&1 < /dev/null &
echo "PID=$! log=$run_dir/run.log"
```

It saves finite agent/optimizer/RNG checkpoints every five minutes, plus
initial, validation-selected best, and final actors. The budget includes
compilation and evaluation; a running update/final evaluation can extend past
the deadline. Earlier CPU/GPU throughput results used the 14-neuron actor and
do not predict this workload. A Mac smoke benchmark (32 environments, 128 steps,
two PPO epochs) measured about 2,794 steps/s; this is throughput, not learning
success. A three-update functional smoke run remained finite and had zero
food-zone successes; no full-network behavioral success is claimed.

For four independent CPU seeds, use `scripts/run_cpu_training.py --actor
full-network --minutes 60 --output-dir /home/carsonc/r-elegans-results` under
`nohup`. Defaults now select the full network, 32 environments per worker,
250-step episodes and learning rate `1e-4`. Each worker gets four physical
cores. Pass `--actor connectome` explicitly to reproduce the old pilot.

To train a subset, add `--train-group neural` (or `nmj`, `sensory`) and optionally
`--train-neurons AWCL AWCR ...`. All 302 cells continue evolving. Start a later
stage with `--initial-actor /external/run/actor_final.npz`; omit subset options
to optimize the entire model. This transfers actor parameters and resets the
critic/optimizer. Parameters that cannot influence the current task may have
zero gradients; locomotion alone cannot constrain every circuit. See
[CURRENT_MODEL.md](CURRENT_MODEL.md) for exact parameter selection and dynamics.

## Connectome-subcircuit chemotaxis

For compute qualification, `python scripts/benchmark_rl.py --actor connectome`
reports PPO compilation time separately from synchronized update throughput.
Use `--require-gpu` on GPU hosts and `CUDA_VISIBLE_DEVICES=0` to benchmark one
device. Run each batch size in a separate process with identical precision and
training settings. This measures the legacy normalized dish and 14-neuron actor;
it does not train or validate the full 302-neuron model. Progress is written to
stderr and the JSON report to stdout; no checkpoints are saved.

For unattended Linux CPU comparisons, use
`python -u scripts/benchmark_cpu.py --output-dir /path/outside/repo/results`.
The sequential sweep compares 4/16/32 allowed physical cores (one hardware
thread per core when topology is available), with 32/128 environments. Each
case gets a fresh process, CPU-only JAX, an affinity restriction, bounded BLAS
threads, a ten-minute timeout, and separate compile/training timings. Results
go to a unique timestamped directory with individual logs and `summary.json`.
Affinity limits eligibility; actual CPU utilization must still be measured.
Run after other performance jobs finish. The Mac is appropriate for development
and runs lasting at most a few hours; use the Linux workstations for longer jobs.

To measure independent-worker scaling, use
`python -u scripts/benchmark_cpu_parallel.py --output-dir /path/outside/repo/results`.
It compares a single four-core worker with four concurrent workers on disjoint
physical cores, using 128 environments and 20 measured updates per worker.
Compilation/warmup completes before a common start signal. Aggregate throughput
uses total steps divided by the time span covering all workers' measurements,
including start skew, rather than summing individual throughput estimates.
Each cohort has a ten-minute timeout; worker logs/reports and aggregate speedup
are retained in a unique timestamped directory. Run after other benchmarks finish.

For an exploratory one-hour training pilot on mbslnx2, use
`python -u scripts/run_cpu_training.py --actor connectome --minutes 60 --output-dir /path/outside/repo/results`
under `nohup`. It launches four seeds on disjoint four-core CPU allocations,
each with 128 environments, 250-step episodes, two PPO epochs and learning rate
`1e-3`. The wall-clock budget includes compilation and periodic validation,
with up to one minute reserved for final evaluation; finalization may extend
slightly beyond the requested duration. The supervisor allows five minutes of
grace before requesting a stop, then kills an unresponsive worker after thirty
seconds. Each seed writes a progress log, metrics, five-minute atomic training
checkpoints (agent, optimizer, RNG, update count), and initial/best/final actor
archives. Best-model selection uses eight validation trials (seed 37); initial,
final and selected-best actors are tested on the same separate 24 trials (seed
19). A failed numerical update cannot overwrite the previous finite state.
This pilot trains only the existing 14-neuron steering baseline in the legacy
normalized dish. Its outcomes are exploratory and do not validate the full
302-neuron model or the physical gait calibration.

The seven-parameter controller can also be replaced by a real, hand-selected
14-neuron subcircuit of the bundled connectome
(`r_elegans.brain.circuit.SUBCIRCUIT_NEURON_NAMES`: chemosensory
`AWCL/AWCR/ASEL/ASER` → interneurons `AIYL/AIYR` → `AIZL/AIZR` → integrator
`RIAL/RIAR` → dorsal/ventral head-motor readout `RMDDL/RMDDR/RMDVL/RMDVR`),
using the real graded-potential equations
(`r_elegans.brain.dynamics.neural_rhs`) restricted to real anatomical
connectivity (63 chemical synapses, 8 gap junctions among exactly these 14
neurons in the bundled asset) rather than the analytic formula. Only
`steering` is connectome-driven; `speed` keeps the analytic controller's
food-proximity-slowing formula. Everything else in the body-direct
pipeline -- food field, gait, body mechanics, reward -- is unchanged, and
`r_elegans.rl.actor_interface.ActorInterface` lets the same PPO/A2C
implementation train either actor.

Training proceeds in two stages: supervised pretraining (imitating the
bundled differentiable-fit controller by backpropagating through a full
episode unroll of the subcircuit's own recurrence), then optional RL
fine-tuning.

```bash
pip install -e ".[env,demo]"
python scripts/pretrain_connectome_circuit.py --iterations 600
python scripts/train_rl_chemotaxis.py --actor connectome \
    --pretrained-checkpoint results/behavior/connectome_pretrain_v1.npz \
    --learning-rate 1e-4 --entropy-coef 1e-4 --num-updates 200
python scripts/demo_rl_chemotaxis.py --actor connectome \
    --load results/behavior/petri_chemotaxis_connectome_ppo_v1.npz
```

On the same 24 held-out source/heading pairs: an untrained subcircuit
succeeds 4.2% of the time (mean minimum distance 0.6372); supervised
pretraining alone reaches 45.8% (0.1888) -- well below the seven-parameter
controller, consistent with 14 real neurons being a far more constrained
function class than a formula hand-designed for this task. RL fine-tuning at
the same learning rate used for the analytic controller's from-scratch
training actively hurt the pretrained checkpoint (down to 8.3%): its critic
starts randomly initialized against an already-good actor, and early noisy
value estimates drove destabilizing updates. A substantially gentler
fine-tuning learning rate (`1e-4` vs. `3e-3`) avoided that regression and
slightly improved mean minimum distance (0.1702) without raising success in
this run. This gap and the fine-tuning instability are reported as current,
honest limitations, not resolved results -- see `CURRENT_MODEL.md`'s
"Connectome-subcircuit chemotaxis" section for the full account of what is
and is not biologically grounded here.

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
