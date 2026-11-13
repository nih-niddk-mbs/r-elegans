# r-elegans

E068 server audit completed successfully (user-reported stdout; run
`output-path-audit-20261113T041119Z-939399`). Under identical ordinary scoring,
the hybrid has larger T/M/H cycles than old/old by55.4%/23.4%/115.2%.
Old/new and hybrid printed amplitudes match exactly. Most tail-joint oscillatory
signal survives dorsal/ventral subtraction at the observed motor frequency;
the strongest cancellation is the final head joint (only8–9% survives in the
hybrid/candidates). Cancellation alone therefore does not explain weak tail
bends. This ratio concerns one frequency, not broadband energy or causal gain.
Detailed NMJ/muscle/filter stdout has now been received; findings follow.
No new model qualification or training.
Evidence: [E068 server summary](experiments/records/E068-server-summary.json).

E068 detailed stage report now received: muscle activation near-extreme fractions
are0 in all printed windows, with responsive local sigmoid slopes. For hybrid
45–60s, net command T/M/H=.00175222/.0701773/.022169; filtered activation=
.000998/.0398816/.0126051 (~57% retained), and joint cycles=
4.68955e-5/.000244504/.000266448rad. Tail command is only~1/40 of middle.
Thus weak posterior command and downstream mechanical attenuation both matter;
sigmoid clipping is not supported. These regional ratios do not establish causal
gain or a physics error. Next proposed check: body torque/response across timing
and spatial patterns, keeping the preserved neural models intact. No new run.
See [stage evidence](experiments/records/E068-server-stage-details.json).

`r-elegans` is an experimental JAX-native foundation for a differentiable,
connectome-constrained *C. elegans* simulator.

**Research objective:** fit one shared embodied model against compatible
physiology, neural activity, movement, sensory responses and perturbation data,
then test held-out animals and experiments. The initial
[data-target ledger](experiments/data_targets_v1.json) tracks available sources
and missing capabilities; this is not yet an achieved biological model or a
completed joint-data trainer. Training remains incremental.

E066 full compact result (2026-11-12): the300s unpaced case retains37
repeating voltage and90 muscle channels in its late half. Eight joint signals
repeat under the two-window detector, but no joint passes the stricter five-
window eligibility screen there. Direct cyclic excursions remain nonzero:
tail/middle/head8.78e-5/1.99e-4/1.82e-4rad versus4.28e-5/2.66e-4/2.89e-4rad
at60s. Thus body amplitude changes over time; this is no settled gait claim.
Nonzero cyclic amplitudes differ less than1% under either half-step refinement.
The held-voltage control has no repeated signals or cyclic bends. Original
regional/rhythm qualification remains failed. PC1=1 in60s cases is computed
from only joint10, not evidence of whole-body synchrony or wave propagation.
See [complete compact evidence](experiments/records/E066-server-summary.json).

Decision: retain the hybrid as a separate neural-rhythm scaffold and build body
amplitude incrementally with channel-density scales initially fixed. Score raw
cyclic excursions over appropriately settled/multiple windows, preserve neural
repetitions and long-run validation, keep actual proprioception and no periodic
drive. A possible next small rung is tail1e-4,middle/head3e-4rad; this is a
proposal, not an implemented objective or training result. E062 and old/old
references remain separate. No new simulation or fitting launched this turn.

Historical partial E066 report (full result now above): execution valid but rhythm/regional
validation failed. Candidate hash f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e.
Both initial-voltage perturbation rows retain37 repeating voltage,90 muscle and
11 joint channels. Direct tail/middle/head cyclic excursions are about
4.28e-5/2.66e-4/2.89e-4rad, well below the .003rad engineering comparison.
Zero historical window-gated medians do not mean zero physical motion. Neural
refinement checks pass; gated regional comparisons can compare zeros with
zeros and do not establish convergence of nonzero cycle amplitudes. PC1=1 over
eligible joints alone does not establish whole-body synchrony or propagation.
At that earlier entry the long/baseline/refinement rows were missing; the full
compact report subsequently supplied is recorded above.
See [partial server evidence](experiments/records/E066-server-partial-summary.json).

Historical E066 preparation (server execution now reported above).
The old-channel/new-connection candidate is pinned by its saved hash and exact
family reconstruction. Eight simulations test its own initial-voltage,
neural/body timestep, held-voltage and300s unpaced persistence controls, plus
regional amplitudes and raw-supported phase/PCA descriptions. Four workers
run concurrently. No fitting or automatic model promotion. Local8-case12s
software smoke and282 tests passed; the older local fixture fails the stricter
tail persistence gate, retained as a negative result. Server execution and full cycle table now reported; original screening remains
failed and the proposed next rung is documented above. See [method and local checks](experiments/records/E066-hybrid-validation.json).

E065 full server table (2026-11-12):60s comparisons isolate the suppressing
parameter family in this setup. Old channel scales + old connections yield
54 repeating voltage/92 muscle/11 joint channels; old scales + fitted connections
37/90/11. Fitted channel scales yield0/0/0 with either connection family. Cue
and sham agree in these counts; current-model early reference checks pass.
Old/old repeats with half neural step, initial voltage +/-1mV and both tested
precisions; feedback blocking changes54 to53 voltage channels, retaining
muscle/joint repetition. Fixed kinetics, leak, capacitance and topology were
unchanged. This implicates fitted non-leak density scales for these parents,
not a specific channel, universal absence of oscillators or biological cause.
See [full compact evidence](experiments/records/E065-server-summary.json).

Preserve old/old as tested reference and old-channel/new-connection as a separate
candidate. The hybrid still needs its own refinement/perturbation checks and
amplitude/coordination assessment; old/old controls cannot substitute. More
repeating neurons is not automatically better fidelity. Proposed next fitting
stage freezes recovered channel scales initially, retaining an explicit rhythm
guard before later channel adaptation. No new simulation or fitting launched.

Historical partial E065 report (full table now above): successful execution and common float64 initial
states for rhythm-comparison-20261112T185742Z-865067. Three complete control
rows were supplied: old/old neural-half and initial-voltage +/-1mV each retain
54 repeating voltage channels,3 calcium,92 muscles and11 command/drive/joint
channels under the diagnostic floors. This supports the earlier oscillatory
parent as a starting point under tested controls; it does not identify the
parameter family suppressing current-model rhythms. At this earlier entry primary hybrid and precision/feedback rows were still
missing; the complete table subsequently supplied is recorded above.
Evidence is user-reported partial stdout; no artifacts independently retrieved.
See [partial result](experiments/records/E065-server-partial-summary.json).

Historical E065 preparation (partial result reported above)
(2026-11-12). Four full-network combinations separate old/new channel scales
from old/new chemical+electrical strengths. Fixed fields, leak, anatomy and
muscle readout must match exactly. All primary cases share the current E063
sensory gains, initialization and wet-substrate proxy.15 independent60s
simulations include cue/sham, initial-state, precision, feedback and neural-step
controls, with four concurrent workers on disjoint four-physical-core groups.
No fitting or periodic drive. Local15-case12s software smoke and279-test suite
passed; this used an older local fitted model, not actual server3fabef... .
Server execution and the full family table are now reported; E062 remains separate.
See [E065 method and verification](experiments/records/E065-rhythm-comparison.json).

E064 server diagnosis (2026-11-12): no detected qualifying repeated signal
at any of the six measured stages in all26 saved traces. Both baseline and best
have zero reported voltage-reversing channels across302 neurons in4–12s, under
the explicit reporting floors. Their nonzero voltage/body ranges are drift,
not demonstrated rhythms. The absence is already visible upstream of muscles;
this does not prove a unique causal defect or exclude downstream limitations.
Held voltage has no repeated pattern; residual passive body relaxation remains.
Input hashes were reported unchanged and the diagnostic exited0. Evidence is
user-pasted stdout, not independently retrieved trajectories. See
[server diagnosis](experiments/records/E064-server-summary.json).

Decision: first recheck the earlier preserved oscillatory checkpoint with the
same float64 precision, initialization and sensory/body coupling. Its old
rhythm must not be assumed to survive that comparison. If confirmed, build
from that independent parent and prioritize neural repetitions before body
alternation. Preserve E062's separate sensory-response skill. No new fitting
objective, comparison simulation or training run has been launched.

Historical E064 readiness (server result above): the diagnostic was implemented and locally verified
(2026-11-12). It reads saved E063 baseline/best and all validation active/sham
signals, counts raw extrema across every channel, and writes per-channel tables
and raw-signal figures externally. Inputs are hash-checked before and after;
no simulation, fitting or JAX/GPU initialization. The local fixture is an older
checkpoint, not the eight-hour server model. Server analysis subsequently completed; see the dated result above. No next training objective or job has been started.
See [diagnostic record](experiments/records/E064-signal-chain-diagnostic.json).

E063 eight-hour server outcome (2026-11-12): no qualifying alternation after
252 proposals; selected251,496.43min including validation. Reported joint has
zero peaks/troughs in every case, and unforced pattern fails. Loss falls only
1.63606 ->1.58909 (about2.9%). Detrended range .0005523rad exceeds1e-5 target
without actual reversals; it must not be interpreted as oscillation amplitude.
Numerical amplitude differences are inside tolerance and cue-effect convergence
passes; pattern refinement fails because the finer pattern remains unqualified.
The working explanation is a search plateau: raw turning-count penalties supply
no gradual reward before reversals appear, while residual curved drift improves
the window-ratio term. This is an inference from code and compact stdout, not a
trajectory diagnosis. Preserve E062's independent recovery skill and E063's
negative result. Inspect saved signal-chain traces before another long fit;
consider a smaller one-reversal objective and parallel short seed screens after
that review. No new run launched. See [server record](experiments/records/E063-server-summary.json).

E062 server result (reported 2026-11-11; UTC run name20261112): the float64
bend-and-recover skill passes the full engineering gate. Low-dose amplitudes
are3.06561e-5rad (coarse),3.06503e-5rad (matched refinement) and3.06565e-5rad
(neural quarter), all above3e-5rad. Chemical-blocked and held-voltage controls
remove the cue response. Preserve checkpoint `16146cc539d47277cb2230e1cfced9a00282d9dd8e4e3b18112fde46515a042b`
from `sensorimotor64-20261112T020300Z-784477`. This is a small sensory-evoked
reversible bend, with numerical checks; locomotion and long autonomous rhythms
remain unproved. Evidence is user-reported stdout, not independent retrieval.
See [E062 record](experiments/records/E062-server-summary.json).

Historical E063 plan (outcome reported 2026-11-12): repeated post-cue bends after one finite
odor exposure, without periodic pacing. A60min float64 pilot uses the E062
parameters with fresh seed6300 and preserves the recovery skill separately.
Full302-neuron topology, fixed NMJ/body/kinetics, unknown channel densities and
connection strengths plus three sensory gains remain the fitting scope.
The subsequent eight-hour server run failed; see the E063 outcome above.

Paired neural/behavior data preparation now includes 21 identified baseline
recordings with frozen date-group holdouts. Data remain external. An initial joint-statistics fitter is now implemented;
exact trajectory fitting and matched experimental mechanics remain pending. See
[E049](experiments/records/E049-paired-data-admission.json).

**Latest autonomous milestone (2026-11-10):**server winner7001 passes both
tail motor/muscle/bend and full-body regional amplitude checks with zero external
drive. Tail,middle and head all meet the defined target. Selected parameters and the assessment are now hash-verified in external Mac
OneDrive;bundled demo unchanged. Coordinated locomotion remains undemonstrated.
See [evidence and scope](CURRENT_MODEL.md#qualified-autonomous-distributed-amplitude-e044-2026-11-10).

**Required locomotion capabilities:** swimming in water and crawling on a moist,
compliant substrate, with initiation, stopping and restarting. Both environments
need separate validation; eventual adaptation must occur in the running animal.
The E046 wet-agar model is an experimental reference environment; the liquid
amplitude milestone remains specific to its original medium. See
[wet-surface mechanics and launch instructions](PHYSICS.md#wet-surface-reference-assessment-e046).

An opt-in [embodied sensory loop](CURRENT_MODEL.md#embodied-sensory-connections-e048-2026-11-10)
now connects head-sampled chemical cues and actual body bending to identified
neurons in the autonomous channel brain. Water and wet-agar reference diagnostics
include input-block controls. Transduction is experimental and unfitted; learned
food finding, regulated start/stop and biological sensing remain unvalidated.
The historical no-input model and its checkpoints remain reproducible.
The server sensory diagnostic reports completion with similar pathway effects;
its small chemical-to-body effect still needs matched numerical refinement
before it can count as reliable behavioral control.

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

An experimental opt-in intrinsic motor-neuron model now adds regenerative
calcium and delayed potassium dynamics to DA/VA and DB/VB cells. Isolated-cell
tests demonstrate autonomous A-type and steady-drive-dependent B-type rhythms.
These are reduced demonstration dynamics, not fitted cell-specific channel
models or a validated autonomous gait; existing skill training keeps its prior
membrane equations. See [the implementation and assay contract](CURRENT_MODEL.md#self-drive-and-proprioception-gap-reviewed-2026-11-07).

Physical-unit [channel and synapse fitting tools](CURRENT_MODEL.md#conductance-fitting-branch-e025-2026-11-07)
are also available separately. The first VD5 clamp candidate predicts held-out
current traces, with non-unique conductances and transferred-kinetics assumptions.
Motor-oscillator channels and biological synaptic strengths remain unfitted;
this candidate does not alter the running worm.

The new [all-cell calcium-fitting branch](CURRENT_MODEL.md#all-cell-channel-and-calcium-branch-e026-2026-11-07)
gives every neuron channel gates and intracellular calcium and fits against
identified GCaMP traces starting from a saved full-network actor. Most channel
compositions are transferred priors, and no spiking rule is imposed. Initial
short-window fits have not established predictive accuracy or preserved motor
skills; use the existing operating model until this branch is qualified.

An [unpaced activity assay](CURRENT_MODEL.md#sustained-channel-dynamics-qualification-e027-2026-11-07)
now demonstrates persistent channel-driven oscillations in an exploratory RMD
profile, including three RMD cells in the full connected network. Calcium-current
ablation removes the qualifying rhythms. This establishes a dynamical candidate,
not physiological calibration or autonomous locomotion. Calcium fitting can start
from its saved channel checkpoint and optionally check that activity persists.

The channel brain now also drives the frozen anatomical muscle/body readout.
The bundled Mac reference now has user-reported server evidence of three rhythmic
RMD cells and small unpaced body bends at 600 simulated seconds. Both controls
stop the activity; useful autonomous locomotion remains undemonstrated. An
earlier server candidate with different parameters settled to a stationary state.
A [bounded mbslnx2 diagnostic launcher](CURRENT_MODEL.md#launch-the-bounded-mbslnx2-diagnostic-yourself)
runs neural/muscle/body controls under `nohup`, with external results and
a 45-minute budget. Pull and run `sh scripts/launch_embodied_mbslnx2.sh` on
mbslnx2: it uses the bundled exact Mac-tested model and matching parent readout
(about 104 KB), verifies their hashes and needs no transfer. The user launches it. Server
results are user-reported; full remote artifacts have not been independently retrieved.

An [autonomous amplitude fit](CURRENT_MODEL.md#autonomous-amplitude-fitting-e030-2026-11-08)
now removes all injected compensation currents and fits uncertain RMD channel
densities. A local short-window fit increases sustained bend amplitude ~11x
while keeping physiology constants, connections and body mechanics fixed.
Independent 300-s local validation retains ~0.03-rad bends with zero injected
current. The fitted pair is bundled as `autonomous_wiggle_v1`; this first
milestone is a larger head/anterior wiggle, not whole-body propulsion.

The first [autonomous forward-movement search](CURRENT_MODEL.md#first-autonomous-forward-movement-search-e031-2026-11-08)
starts from that preserved wiggle. Run `sh scripts/launch_autonomous_movement_mbslnx2.sh`
after pulling on mbslnx2 for four independent searches, up to four hours including
validation. It fits neuron densities with zero external drive and rewards repeated
net forward progress after warmup. The first four-seed server search retained
wiggling but failed forward-motion validation in every seed; broader motor-circuit
training is the proposed next stage.

Movement training now fits [neurons and existing recurrent connections](CURRENT_MODEL.md#joint-neuron-and-connection-movement-fitting-e032-2026-11-08):
9,616 eligible parameters across all 302 neurons, while anatomy, physiological
constants, muscle/body parameters and zero injected current remain fixed. The
same four-hour movement launcher explicitly selects this broader E032 scope;
the earlier restricted E031 result is preserved. The first four-hour joint
search also failed movement validation (0/4); a synthetic audit found short-window
phase sensitivity in its selection score. Scoring/trajectory analysis is the
next priority before another long run.

A [read-only trajectory analyzer](CURRENT_MODEL.md#trajectory-analysis-e034-2026-11-08)
measures spatial bend amplitude, phase, shape variance and translation at
matching wiggle phases. The saved local pilot is mostly synchronous anterior
bending. User-reported server analysis now confirms that all four-hour
candidates remain anterior-dominated and nearly in phase, with no robust
cycle-matched forward progress.

The current [intrinsic body-wave curriculum](CURRENT_MODEL.md#autonomous-body-wave-curriculum-e035-2026-11-08)
first recruits coherent middle/posterior bending, then spatial phase progression.
Run `sh scripts/launch_autonomous_wave_mbslnx2.sh` after pulling on mbslnx2
(four workers, four-hour cap including validation). Neurons and connections
remain trainable with zero external drive. Complete-cycle position averaging
replaces endpoint-based progress for the later, gated movement stage. The first
E035 server cohort improves regional recruitment but still fails posterior
amplitude and wave qualification; seed 2/3 are retained as continuation candidates.

The current [body-amplitude-only stage](CURRENT_MODEL.md#body-amplitude-only-learned-parameter-forks-e037-2026-11-09)
defers phase coordination and propulsion. It continues learned seeds 2 and 3
with four fresh random forks while preserving the original parameter bounds.
Pull and run `sh scripts/launch_body_amplitude_mbslnx2.sh` on mbslnx2 (four-hour
cap including validation). Legacy movement/wave launchers delegate to this stage.

Calcium fitting now uses a [conservative parameter policy](CURRENT_MODEL.md#conservative-fitting-policy-e029-2026-11-07):
shared physiological properties and reference leak stay fixed; only explicitly
eligible unknown channel densities and existing synaptic strengths fit. Each run
saves its exact masks and distinguishes measured constraints from fixed estimates.

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

## Laboratory records and replication

The current strategy is **skill coverage first**: preserve separately trained
full-network controllers and defer shared-weight consolidation. The
[skill repertoire](SKILL_REPERTOIRE.md) lists motor primitives, search/state
compositions, later sensory tasks and organ/3D/developmental extensions,
mapping all 62 entries in the biological behavior catalog. It marks observed,
candidate and blocked capabilities without claiming a working skill selector.

Standalone shallow-steering training is now available with
`scripts/run_motor_curriculum.py --steering`. Each worker fits its own signed
turn from the matching forward-wave checkpoint; even seeds receive positive
muscle bias and odd seeds negative bias. Parent models are preserved, with no
shared forward/reverse retention requirement. The trainer differentiates
muscle, body-joint and centroid-trajectory errors, and selects on actual
centroid-path/body-axis turning as well as waveform quality. Full final
assays retain the 0.001 progress floor, timestep/refinement and rest/isotropic
translation controls. This is externally paced, normalized steering.

On mbslnx2, after activating conda `r-elegans` and pulling the current code:

```bash
nohup python -u scripts/run_motor_curriculum.py --steering --minutes 480 \
  --resume-run /home/carsonc/r-elegans-results/wave-20261105T214835887935Z \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-steering.log 2>&1 < /dev/null &
```

Four workers use four physical CPU cores each, two candidates per turn sign.
Fresh `steering-*` cohorts include complete provenance and independent actor
checkpoints; no other skill is launched on success. Archive a finished cohort
with `archive_motor_run.py --latest-under <results> --pattern 'steering-*'
--output-dir <archives>`. Local development and results are recorded in E019
of the notebook. These controllers are not a calibrated biological turning
range or a runtime skill-switching implementation.

The subsequent server run reports supervisor exit code 0 (E020), indicating
all configured steering workers passed. Only a truncated result tail has been
received; full per-seed summaries and remote checkpoint verification remain
pending. Head sweeps are the next standalone skill target.

The stationary head-sweep stage is implemented as `train_head_sweep.py` and
`run_motor_curriculum.py --head-sweep`. It fits an anterior-localized scan with
the posterior trunk quiet, using the same full recurrent brain and fixed NMJs.
Its artificial pacing current is weighted by anterior NMJ contacts. The target
is not supplied as muscle output; the input map does provide an explicit
anterior-localization prior. Positive/negative scan phases
occur within each policy; these are not two separate skills or a locomotor
turn. Physical tests use head angle/lateral motion relative to the trunk,
localization, bounded drift, periodicity, rest and timestep controls. No minimum
forward progress is required for this deliberately stationary skill.

Run the independent four-seed head-sweep cohort on mbslnx2 after pulling:

```bash
nohup python -u scripts/run_motor_curriculum.py --head-sweep --minutes 480 \
  --resume-run /home/carsonc/r-elegans-results/wave-20261105T214835887935Z \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-head-sweep.log 2>&1 < /dev/null &
```

Fresh `head-sweep-*` directories preserve all previous locomotion/steering
models. Archive the finished run with `--pattern 'head-sweep-*'`. This is
externally driven, normalized scanning, with no biological amplitude/frequency
calibration or runtime policy handoff. E021 in the notebook records the
predeclared protocol and development results; the agent cannot launch the
password-authenticated server command itself.

The local head policy now passes both tuning and physical controls after
3,000 staged updates. Two earlier below-gate fits are retained. Head excursion
is 0.1595–0.1750 rad peak-to-peak with a quiet posterior trunk and negligible
cycle drift. This is one source lineage; the four-seed server run remains
unverified. All artifacts and replay inputs are preserved in OneDrive.

E022 subsequently reports that all four server head-sweep seeds pass both
tuning and physical tests at every pacing rate. Stationary driven head
scanning therefore joins the reported replicated repertoire. Complete remote
artifacts/provenance remain to archive; this is not autonomous head selection
or biological validation.

[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) is the chronological experiment record,
including hypotheses, methods, negative results, decisions, code/artifact
lineage and explicit evidence gaps. Small [reported-result summaries](experiments/records)
stay in Git; raw data and checkpoints remain external. Future training and
assessment invocations automatically save exact commands, package/platform
records, numerical settings and source snapshots under `run_records/`.

Preserve a finished server cohort and all checkpoint dependencies with
`scripts/archive_motor_run.py`; see the notebook for archive/verification and
replay procedures. A same-disk archive is not an off-machine backup, and
collection-time metadata does not replace missing historical run environments.

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

To assess sustained locomotion of the latest passed wave ensemble, with
weights frozen, use:

```bash
python -u scripts/run_motor_curriculum.py --resume-latest --assess-locomotion \
  --output-dir /home/carsonc/r-elegans-results
```

This is a short diagnostic, not another hour of training. It runs three pacing
periods (2.4, 3.0, 3.7 s), discards four startup cycles, and measures six complete
cycles at matching command phases. Progress uses the segment-center centroid
and actual tail-to-head axis, rather than head-tip movement. Shape closure,
cycle consistency, lateral motion and yaw are reported. Timestep refinement
(20 → 10 ms), zero-command rest and isotropic-drag controls check numerical
drift. All trajectories stay outside Git.

`detectable_translation` requires periodic shape, consistent signed progress,
agreement between timesteps and translation above numerical/control floors.
The stronger `passed` gate also requires approximately straight movement and
at least 0.001 body lengths per cycle at every pacing period. These are declared
engineering criteria in normalized mechanics, not measured biological speeds.
Completed diagnostics can return exit code 2 for insufficient progress, with
`error: null`; startup or numerical failures are distinct.

A local wave checkpoint shows real but small forward progress:
0.00032–0.00042 body lengths/cycle, rest drift zero, isotropic-drag drift below
2e-10 body lengths/cycle, and 0.5–1.4% timestep differences. Increasing the
synthetic target amplitude from 0.04 to 0.05 and fitting 1,000 further updates
passes wave checks and raises progress to 0.00051–0.00066 body lengths/cycle,
about 59% more. Both remain below the stronger 0.001 threshold. This is one
local seed, not a completed locomotion ensemble or natural swimming result.

To make that same small amplitude step on the passed mbslnx2 wave ensemble:

```bash
nohup python -u scripts/run_motor_curriculum.py --resume-latest \
  --increase-amplitude 0.05 --minutes 60 \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-locomotion.log 2>&1 < /dev/null &
```

Each seed is first assessed without training. Only real, stable, approximately
straight baseline propulsion advances to fitting. The target changes by at
most 30% per stage; neural parameters learn while the stimulation gain, NMJs
and muscle transfer remain fixed. Only a passed wave fit advances to a new
locomotion assay. A result that still falls below the progress threshold stays
at this milestone; no RL is launched. Passing wave checkpoints are preserved
even if the stronger locomotion check fails, allowing another small amplitude
step later. `--resume-latest` selects the newest wave cohort with matching
passed wave checkpoints, including such partially completed cohorts. Use an
explicit `--resume-run` to choose an earlier cohort. Always activate the
`r-elegans` conda environment first.

The completed mbslnx2 0.05 ensemble retained passed wave fits and stable forward
propulsion for all four seeds, improving progress by 52–67% to
0.000535–0.000605 body lengths/cycle. The readiness threshold was not reached.
Continue from those passed wave checkpoints with `--increase-amplitude 0.06`;
the increase is 20%, with pacing, target propagation delay and gates unchanged.
A local 1,000-update continuation passes wave checks and reaches
0.000749–0.000965 body lengths/cycle, 45–46% above its 0.05 baseline, while
remaining below the 0.001 readiness threshold. Replication across seeds remains
the purpose of the next mbslnx2 run. Use `--resume-run /path/to/completed-0.05-run`
to select that cohort explicitly, preserving all previous artifacts.

The subsequent mbslnx2 0.06 ensemble passed all wave fits and reached
0.000772–0.000922 body lengths/cycle. At 0.07 all four wave fits passed,
and three seeds passed locomotion readiness at every pacing rate. Seed 2
reached 0.0009956 at the slowest rate, narrowly below the unchanged 0.001 gate.
All conditions remained periodic, consistent, approximately straight and
forward; progress improved 29–35% over 0.06. These are externally paced,
normalized-model results, not biological speed calibration.

The next stage tests **stop and restart with frozen weights** before adding
training. It carries neural and body state continuously through six pacing
cycles each of moving, zero command, and moving again. The last three cycles
of each block are measured; the first three allow settling. Both centroid
path length and body-angle speed during the stop must fall below 10% of the
initial moving window, and restarted forward progress must recover to
80–120%. Moving windows retain the existing propulsion, straightness and
0.001 body-length/cycle gates. The assay repeats at 2.4/3.0/3.7-second periods,
20/10-ms timesteps, and with isotropic drag. It measures late settling,
not immediate stopping or a biological response latency.

Run on mbslnx2 using the completed 0.07 cohort explicitly:

```bash
nohup python -u scripts/run_motor_curriculum.py --stop-restart \
  --resume-run /home/carsonc/r-elegans-results/wave-20261105T214835887935Z \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-stop-restart.log 2>&1 < /dev/null &
```

This is a short diagnostic, with no optimizer or RL. Results go into a fresh
`stop-restart-*` directory. `control_passed` distinguishes successful stopping
and recovery from the stronger overall `passed` gate, which also retains
the progress floor. Seed 2 may still miss that floor. A failed gate preserves
all source checkpoints and does not automatically launch another stage.

The completed mbslnx2 stop/restart assay subsequently passed `control_passed`
for all four seeds at all three rates, without training. Residual shape speed
was <0.024% of its moving value; restarted progress recovered within 0.1%.
Seeds 0/1/3 passed overall readiness. Seed 2 retained the same slow-rate
progress-floor miss, independently of successful stopping and restarting.

For the next supervised stage, `--reverse` continues the forward-wave
checkpoints into a shared forward/reverse policy. A fixed artificial binary
direction context adds reproducible +/-0.05 currents to NMJ-projecting
neurons in reverse mode; forward mode retains the previous scalar input.
This is an experimental interface, not an identified biological command
pathway. Reverse targets propagate causally from posterior to anterior at
the source checkpoint's amplitude/delay. All 302 neurons evolve, only neural
parameters learn, and NMJ/muscle parameters remain fixed. Equal-weight
forward rehearsal matches the frozen source actor, including stop/hold
commands. Validation changes timing and command levels and checks both
reverse wave propagation and forward retention.

```bash
nohup python -u scripts/run_motor_curriculum.py --reverse --minutes 180 \
  --resume-run /home/carsonc/r-elegans-results/wave-20261105T214835887935Z \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-reverse.log 2>&1 < /dev/null &
```

Four independent seeds use four CPU cores each. About 179 minutes are
available for fitting, leaving a minute for final assays; compilation and
shutdown can extend elapsed time slightly. Checkpoints are saved every 100
updates. Fresh `reverse-*` folders preserve the original wave checkpoints.
Each finite completed candidate, including one that misses its imitation
gate, receives forward and reverse go/stop/restart diagnostics with timestep
and isotropic controls. Overall success requires the fit, forward retention,
and reverse propulsion/stop/restart gates. Reverse progress must be negative
along the body axis and meet the unchanged 0.001 magnitude floor. These
assays hold direction fixed; switching between directions during motion is
a later test. No RL is launched.

The completed three-hour reverse ensemble learned backward translation in all
four seeds and passed stop/restart control in both directions at all pacing
rates. Reverse readiness passed all rates for seeds 2/3, and forward readiness
only for seed 3. Reverse imitation passed seeds 0/2; seed 1 failed a changed
hold-command sequence and seed 3 exceeded wave-lag tolerance. No seed passed
every gate. Training loss alone did not protect forward propulsion.

Continue these models for eight hours with the refinement mode:

```bash
nohup python -u scripts/run_motor_curriculum.py --refine-reverse \
  --resume-latest --minutes 480 \
  --output-dir /home/carsonc/r-elegans-results \
  > /home/carsonc/r-elegans-results/motor-reverse-refinement.log 2>&1 < /dev/null &
```

This mode selects the latest **completed reverse cohort**, including candidates
that missed imitation/readiness gates, provided all seeds completed both motor
control assays successfully. It is continuation for learning, not promotion.
The original forward reference must still exist and match its saved checksum.
It is recovered from each candidate's provenance; the weaker continued forward
policy is never substituted as teacher. `--resume-run` selects a cohort
explicitly. Confirm the chosen directory in the startup log.

`scripts/refine_reverse.py` trains eight command sequences with four pacing
periods, varied phases/levels and stop/hold intervals. Training horizons are
7.68 seconds and tuning horizons 15.36 seconds, versus 3.84/7.68 in the first
reverse run. Random minibatches contain two cases. Forward relative muscle
error has weight 4 and reverse error weight 1; the learning rate is 0.0003.
Every 500 updates, both directions undergo body/muscle wave validation on
eight different sequences. The best checkpoint is chosen by failed validation
case count, then worst normalized error, then mean error, rather than training
loss. Validation data used for selection are tuning data, not an independent
test set. All scientific gates and motor-input definitions remain unchanged.

Checkpoints save every 100 updates. Approximately 479 minutes are reserved
for fitting, followed by independent long forward and reverse physics assays
including stop/restart, timestep refinement and isotropic controls. The fresh
`reverse-*` folder preserves both earlier cohorts. The iteration cap is one
million, so the wall-clock budget normally governs; increase `--minutes` for
a longer run.

The eight-hour result has now been reported: imitation passed seeds 0/1/2,
and forward readiness passed 0/1. All seeds retained reverse translation and
stop/restart control, but every reverse readiness assay missed the 0.001
progress floor (go magnitudes 0.0005968–0.0009962 body lengths/cycle).
Seed 3 also missed one forward tuning case. Every stage completed without
errors; no model met every criterion. The remaining mismatch is imitation
versus actual propulsion. No further learning-objective change has been run.
See [E015 in the notebook](LAB_NOTEBOOK.md#e015--reported-2026-11-06-eight-hour-refinement)
for per-seed updates, selection and evidence limitations.

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

Latest reported amplitude cohort (E037,2026-11-09): head reaches target in
4/4 seeds, middle3/4, tail0/4; no distributed-amplitude qualification. Next is
a read-only posterior activity audit, with control/propulsion still deferred.
See [current model](CURRENT_MODEL.md) and [lab notebook](LAB_NOTEBOOK.md).

E038 server audit reported (2026-11-09): posterior neural voltage/release
oscillations are weak, and posterior median individual-muscle sustained
excursion is zero in all four workers. Deficit exists before the body-mechanics
stage. Working hypothesis: insufficient posterior neural recruitment; channel,
recurrent and NMJ causes remain unresolved. Next proposed curriculum targets
posterior recruitment/amplitude with zero external drive, preserving independent
head/middle candidates. No new training or skill qualification. Full parsed
reports: `experiments/records/E038-server-20261109.json`.

E039 short tail trial is prepared: `sh scripts/launch_tail_amplitude_mbslnx2.sh`
starts a20-minute nohup cohort on mbslnx2, from learned E037 parents1001/1003.
Targets autonomous tail amplitude with a weak neural-recruitment auxiliary
score; no external drive. Qualification is separate from body-wide amplitude
and locomotion. See [protocol](TRAINING_PROTOCOLS.md#short-autonomous-tail-amplitude-trial-e039-2026-11-09).

E039 server result reported (2026-11-09):0/4 short tail qualifications,151
proposals total. Best fitting tail amplitude gain2.63% (seed2000); others~0.3%.
Best long-validation tail median2.905e-5rad remains103x below0.003rad target;
all active checks unqualified, held-voltage excursions zero. Existing head and
middle amplitude remain above target in these checks. No new skill qualified.
Next proposed diagnostic: posterior-cell autonomous oscillation feasibility
within current bounds before extending unchanged fitting. No new run launched.
Exact parsed results: `experiments/records/E039-server-20261109.json`.

E040 local excitability audit found no isolated rhythms in5,396 bounded
cell/profile/initial-state trials;full-network coupling supports only weak
tail-projecting activity. No candidate promoted. Actual server-parent repeat:
`sh scripts/launch_posterior_excitability_mbslnx2.sh` (nohup diagnostic, no training).
See [current model](CURRENT_MODEL.md#posterior-excitability-diagnostic-e040-2026-11-09).

E040 actual server-parent results reported (2026-11-09):diagnostic exit0,
0 isolated-screen rhythms and0/4047 alternate-basin cell trials. Current
full-network selected rhythms7 (maximum7.61054mV),NMJ-defined isolated
subcircuit0,strong-inward candidate6 (maximum3.43791mV). These selected cells
have tail projections;none of the reported rhythmic cells is DA/DB/DD/VA/VB/VD/AS.
This supports connection-dependent activity but not posterior motor recruitment.
Finite screen does not exclude other densities/recurrent loops. No candidate
promoted or body skill qualified. Next proposed search targets posterior motor/
muscle recruitment from the existing autonomous rhythm;external drive remains
zero. See `experiments/records/E040-server-20261109.json`.

Option1,E041: `sh scripts/launch_posterior_motor_mbslnx2.sh` prepares a30-minute
background server cohort targeting26 strongly posterior motor neurons,net muscle
activation and bending. Zero external drive,original bounds preserved. See
[current model](CURRENT_MODEL.md#targeted-posterior-motor-recruitment-e041-2026-11-09).

E041 server result reported (2026-11-09):0/4 qualifications after295 proposals,
but seed5003 shows strong subthreshold gains:tail bend7.17x to0.0001874rad,
net-muscle excursion32.68x to0.0004799. Gains persist under perturbations and
finer brain/body steps (brain-refined bend+0.008%,muscle+1.54%). Remaining
shortfall16x bend and6.25x muscle. Seed5002 offers a smaller stable gain;5000/5001
improve muscle activity without improving bend. Prefer independent continuation
from5003,with5002 backup;preserve original anchors/bounds and zero input. No new
run launched or skill qualified. Exact record:`experiments/records/E041-server-20261109.json`.

Three-hour continuation(E042) prepared:
`sh scripts/continue_posterior_motor_mbslnx2.sh` starts3 forks from E041 seed5003
and1 from5002,with original bounds,zero drive and300s long validation.
See [current model](CURRENT_MODEL.md#three-hour-posterior-motor-continuation-e042-plan-2026-11-09).

E042 server continuation reported (2026-11-09):2008 proposals,0/4 full passes,
but best6002 increases tail bending7.28x to0.00136388rad and net-muscle activity
7.34x to0.00352342. Net-muscle amplitude exceeds0.003 in all five active checks;
tail bending remains~2.2x below0.003rad target. Long validation0.00139634rad/
0.00361156 muscle;brain refinement changes bend-1.84%,muscle-1.49%. All four
workers improve,held-voltage scored excursions0. Prefer future continuation from
6002,preserve6003 alternate lineage/other backups. Same objective,original bounds,
zero drive;no full skill qualified and no new run launched. Exact record:
`experiments/records/E042-server-20261109.json`.

Current continuation launcher now uses an8-hour budget(E043):
`sh scripts/continue_posterior_motor_mbslnx2.sh`,three forks from reviewed6002
and one from6003. Same objective/original bounds/zero drive,300s long validation.
The previous3-hour version is preserved in commit`cb99867`.

E043 duration amended (2026-11-10):latest user instruction is3hours/180min,
superseding the8-hour preparation and intervening4-hour request. Same launcher,
6002x3+6003x1 parents,seeds7000–7003,20s fit/300s long validation,original bounds
and zero drive. Eight-hour source preserved in commit`30ff375`. No E043 server
launch confirmed;editing launcher does not change an existing process.

Read-only full-body assessment of winner7001(E044) is prepared:
`sh scripts/assess_winner_mbslnx2.sh`. It prints region amplitudes and exports
scientific PNGs plus a standalone interactive motion.html preview. No training
or checkpoint changes;actual server assessment results remain pending.

E045 physics audit:current liquid dynamics are a dimensioned local-RFT
approximation for unbounded Newtonian water. Basic physics tests pass;real
Petri-dish boundary effects and quantitative waveform→speed/force calibration
remain unvalidated. See [physics scope](PHYSICS.md#medium-physics-audit-status-e045-2026-11-10).

## Paired neural and movement fitting pilot

On mbslnx2, launch a short detached run with the dedicated conda interpreter:

```sh
git pull --ff-only
sh scripts/launch_paired_fit_mbslnx2.sh 20
```

The launcher prints the log and result paths. The fitting budget is 20 minutes,
plus first-run data download/preparation and final timestep refinement. It fetches
the pinned ~569 MB source archive/labels, verifies hashes and frozen E049 splits,
and stores data in `/home/carsonc/r-elegans-data` (override with
`R_ELEGANS_DATA_DIR`). It uses the preserved server winner7001, two training
recordings, four physical CPU cores and no external neural drive.

This E050 pilot fits calcium fluctuation and coarse bend statistics with one
shared set of unknown channel densities and synaptic strengths. It is an
incremental objective, with assumed reporter brightness and a wet-agar mechanical
proxy. Mounted-medium physics, observation noise, animal initial states and
exact trajectory agreement remain unresolved. It does not promote a new bundled
model or use held-out targets. See [current scope](CURRENT_MODEL.md).

### Continue a completed paired search

On mbslnx2, continue the latest completed paired fit for three additional hours:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/continue_paired_fit_mbslnx2.sh 180
```

The wrapper prints the selected parent and new log path. It preserves the saved
best offsets, PCG64 state, original anchor/bounds and proposal numbering. It is a
continuation of a completed random search, not an in-flight process restoration.
The 180-minute budget excludes setup and final timestep refinement. The job runs
under nohup using the dedicated conda interpreter. No held-out target is fitted.

Latest paired-data result: the E052 server continuation reduced the training
objective by 5.3%, with a gain retained on timestep refinement. Bend fluctuations
increased but remain far too small; nominal calcium error worsened slightly.
This is an exploratory tradeoff on two short training windows. See the
[E052 evidence](experiments/records/E052-server-summary.json).

E053 eight-hour continuation: training loss decreased11.1%, with the gain
retained on timestep refinement. Coarse bend fluctuations grew29–126% but remain
roughly11–34 times too small; calcium agreement was essentially unchanged.
Most scalar improvement came from mean-bend error. See the
[E053 analysis](experiments/records/E053-server-summary.json).

### Inspect sustained motion and parameter bounds

After the E053 fit, run the read-only diagnostic on mbslnx2:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/launch_paired_diagnostic_mbslnx2.sh 40
```

It prints the selected checkpoint, log, bound report and motion-preview paths.
The detached job runs40 simulated seconds each for active, refined and held-voltage
controls. It records dense neural/muscle/body signals and separates repeated
late fluctuations from posture and drift. No training or bound widening occurs.
The E054 server diagnostic completed: no sustained repeats passed the late-window
criteria in active or refined cases, and no fitted coordinate was near its bound.
This supports a new sustained-amplitude objective while retaining current bounds;
it does not establish absence of slower rhythms or zero variance. See the
[E054 result and limitations](experiments/records/E054-server-summary.json).
The next curriculum remains a specification; its trainer has not been implemented.

### Sensory–motor foundation curriculum (E055)

Train sensory responses alongside motor skills, beginning with one reversible
bend following a finite odor exposure. The full302-neuron model and anatomical
readout remain intact. Unknown neural/synaptic parameters and three effective
sensory gains are eligible within original bounds; kinetics and body parameters
remain fixed. Later alternation, propagation, movement and regulation stages
are [planned separately](experiments/sensorimotor_curriculum_v1.json).

On mbslnx2, launch the first-stage search in the background:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/launch_sensorimotor_mbslnx2.sh 20 &&
tail -n 40 -F "$(ls -1t /home/carsonc/r-elegans-results/sensorimotor-*.log | head -n 1)"
```

The script uses the dedicated conda interpreter and latest completed paired fit,
verifies its original anchor, and preserves the parent recursively outside Git.
Budget:20minutes search including baseline, plus compilation/validation overhead.
Exit0 means the stated engineering cue-response gates passed; exit2 means the
search completed without qualification. This is a new objective with fresh RNG,
not an exact continuation of paired fitting. Selected/final models, sensory gains,
search state, source/environment records and trajectories remain external.

Each8-second trial contains one exposure at2–3seconds and recovery at6–8seconds.
Selection compares cue and sham trajectories, rewarding bend amplitude with
recovery; it does not prescribe a bending side or waveform. Validation includes
dose/onset variations, half timesteps, chemical/voltage controls and a descriptive
proprioception block. Unforced activity is recorded separately. A successful
evoked response does not establish autonomous wiggling, navigation or biological
fidelity; empirical sensory kinetics and substrate calibration remain unresolved.

The initial cue-specific target is only .00003rad, to establish reliable signal
transmission before increasing amplitude. The planned ladder is .00003, .0001,
.0003, .001, .003rad, with review at every step. This first target is far below a
biologically realistic bend. Only the first rung is currently selected; changing
a target is a new objective/configuration, not exact continuation.

E055 server result partially reported (2026-11-11):23.58minutes, time-budget
stop; checkpoint437b227e... from confirmed E053 parent98ea6655.... An unlabeled
validation case shows1.274e-5rad cue-specific bend (42.5% of the3e-5 target),
19.5% residual and6.025mV AWA recruitment with recovery. Chemical and held-voltage
blocks give zero differential response; engineering refinement gate passes
(2.750e-6rad versus7.5e-6 tolerance). No promotion: run ID, baseline/best,
iterations, full active-case results and final exit code were not supplied.
The labeled proprioception block is descriptive; compare it with the missing
active case before interpreting feedback's contribution. See the
[partial server record](experiments/records/E055-server-partial-20261111.json).

E055 complete compact server summary reported:48 proposals, best selected at48;
qualification failed. Active cue-specific bend grew51.1% (9.462e-6 ->1.430e-5rad)
and objective fell23.6%. Active recovery residual fraction improved14.0% ->10.7%,
though absolute residual increased. Lowest active validation amplitude is
1.233e-5rad (41.1% of target); all active variants remain below3e-5rad. The
previously unlabeled row is confirmed refined. Proprioception block increases
this cue response7.6%, so its benefit is not established in this assay. Retain
feedback and first-rung target. A60min continuation is planned with both neural
and sensory offsets and PCG64 restored; server launch pending. See the
[complete compact report](experiments/records/E055-server-summary.json).

Continue the latest completed sensory–motor search, preserving its neural and
sensory parameters, original bounds, post-search RNG and global iteration count:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/continue_sensorimotor_mbslnx2.sh 60 &&
tail -n 40 -F "$(ls -1t /home/carsonc/r-elegans-results/sensorimotor-*.log | head -n 1)"
```

This budgets60minutes including baseline/search, plus validation. It re-evaluates
the parent best objective and aborts on configuration/checkpoint/state mismatch.
The original launch command without a resume parent still starts from the latest
paired fit; use the continuation command to retain sensory–motor learning.

E056 continuation server tail reported (2026-11-11):exit2 after63.87minutes;
refined bend2.803e-5rad (93.4% of target), shifted onset2.832e-5rad (94.4%).
Both remain unqualified despite adequate recovery; chemical/voltage controls
and engineering refinement pass. Chemical sensory multiplier is now exactly2,
its upper bound; neural bound occupancy is unknown. Refined amplitude increased
120.0% versus E055's matched refined case. Candidate180c2b93... resumes confirmed
E055437b227e.... Run ID, proposal counts, active/dose results and baseline/best
are missing from this paste. Retain target/bounds and request the compact report
before choosing another fit. No new training launched. See the
[E056 partial record](experiments/records/E056-server-partial-20261111.json).

Get a compact report of the latest sensory–motor run without starting training:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
/home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.report_sensorimotor
```

Future training logs print this compact report after the full JSON. It includes
baseline/best, every validation case, global proposal counts and sensory gains.

E056 full compact report:148 proposals (global48 ->196), selected195. Active
bend grew102.1% to2.890e-5rad (96.3% target), low dose2.503e-5rad (83.4% target);
high dose passes. Controls/recovery and engineering refinement pass. Remain on
first rung. E057 prepares worst-case selection over nominal and.8dose, since
nominal amplitude alone would stop earning reward at target. Keep the target,
original parameter bounds and learned neural/sensory parameters; use fresh
seed5700/global0 for the changed objective. Low dose becomes a training condition;
high dose, timing shift and refinement stay outside selection. No new server
training launched. See [E056 complete compact record](experiments/records/E056-server-summary.json).

Launch the robust first-rung objective on mbslnx2:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/launch_robust_sensorimotor_mbslnx2.sh 60 &&
tail -n 28 -F "$(ls -1t /home/carsonc/r-elegans-results/sensorimotor-*.log | head -n 1)"
```

This starts a new60minute background search plus validation, inheriting selected
neural parameters and sensory gains from the latest completed sensory–motor run.
It preserves original bounds and uses fresh seed5700 for the revised objective.
Each candidate shares one sham across nominal and .8dose cue trials; selection
uses the larger response loss plus the original parameter prior. Dose-low is now
a training condition, not an unseen validation cue. No waveform or motor current
is supplied. The original numerical/recovery/control gates still apply.

`continue_sensorimotor_mbslnx2.sh` automatically retains the selected search's
training-cue mode and RNG for later continuations. A direct `--resume-run` with
a changed cue mode is rejected; use `--warm-start-run` for a new objective.

E057 server result:99 proposals, selected98,63.80minutes, qualification false.
Nominal, high-dose, shifted-onset and nominal-refined responses now pass the
current gates. Low dose reaches2.93875611e-5rad (97.96% of target), still below
3e-5; controls and engineering refinement pass. Retain candidate1d864fdb... and
same robust objective. Plan E05860min completed-search continuation with saved
neural/sensory parameters and final RNG after99 (not selected98). No target or
bound changes, stage promotion or new server launch. Near-threshold low-dose
performance needs numerical care: current refinement is nominal-only, not a
separate fine-step low-dose qualification. See
[E057 server report](experiments/records/E057-server-summary.json).

Continue this robust search with saved parameters and RNG; the continuation
launcher reads the original cue mode automatically:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/continue_sensorimotor_mbslnx2.sh 60 &&
tail -n 28 -F "$(ls -1t /home/carsonc/r-elegans-results/sensorimotor-*.log | head -n 1)"
```

This continues the current objective. Repeating the robust-stage launch command
would instead warm-start another fresh search. Keep the selected E057 model and
its sensory parameters independently even if the continuation does not improve.

E058 first-rung engineering PASS (user-reported):99additional proposals,
global99 ->198, selected167. All required active variants and recovery/control
criteria pass. Low-dose bend3.03782455e-5rad is101.26% of target, nominal116.42%.
Preserve winner6cd7b0fa... with its sensory parameters independently. This is a
tiny sensory-evoked reversible response, not autonomous wiggling or a biological
reflex fit. E059 is a read-only low-dose timestep/margin assay before promotion:
current refinement tested nominal cue only. No retraining or threshold change.
See [E058 report](experiments/records/E058-server-summary.json).

Run the pinned winner's read-only low-dose resolution assay on mbslnx2:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/assess_sensorimotor_resolution_mbslnx2.sh &&
tail -n 20 -F "$(ls -1t /home/carsonc/r-elegans-results/cue-resolution-*.log | head -n 1)"
```

It pins E058's reported path and SHA, preserves recursive dependencies, and runs
8second cue/sham pairs at original, half and quarter body/neural timesteps. No
weights/gains change. Results report each response gate, successive cue-effect
differences and whether the fine-grid threshold margin exceeds the last grid
difference. This is descriptive numerical evidence, not a rigorous error bound
or retrospective revision of the original stage pass. Assay exit0 means complete;
read its summary to assess all-grid response and margin confidence.

E059 server numerical assessment:original low-dose coarse gate passes, but
half/quarter fail at2.796e-5/2.537e-5rad versus3e-5 target. Quarter amplitude is
16.47% below coarse; successive peak changes have not shrunk. Recovery remains
acceptable. Retain E058's original protocol pass, but do not promote or continue
fitting until this sensitivity is understood. E060 prepares a fixed-parameter
split brain-step versus body/sensory/coupling-step assay; no model changes or
server launch. Full comparison arrays and actual convergence remain unknown.
See [E059 server record](experiments/records/E059-server-summary.json).

Run the pinned winner's independent step-family diagnostic:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/assess_sensorimotor_steps_mbslnx2.sh &&
tail -n 20 -F "$(ls -1t /home/carsonc/r-elegans-results/cue-split-*.log | head -n 1)"
```

This is a five-case read-only8-second low-dose cue/sham assay, under nohup.
The brain branch refines neural steps at fixed body step; the body branch
refines mechanics/sensory sampling/coupling at fixed neural step. This does not
isolate mechanics alone. All weights, gains, original thresholds and numerical
equations stay fixed; no new fitting or automatic skill promotion occurs.
The source remains pinned to E0586cd7b0fa..., and full traces/source records
remain outside Git. Diagnostic exit0 means completion; inspect its case ratios
and within-branch differences before selecting a numerical fix or training grid.

E060 server result localizes the tested sensitivity to neural stepping:
brain-half/quarter bend ratios .9207/.8352, body/sensory/coupling .99981/.99985.
Body-branch trajectory differences shrink; neural differences grow. Keep the
original E058 gate pass, but defer promotion and fitting. E061 prepares a fixed-
parameter float64 neural-step sweep against saved E060 float32 traces, with
legacy input/initial-state values promoted rather than regenerated. This tests
precision sensitivity; it is not a solver fix or a biological qualification.
See [E060 record](experiments/records/E060-server-summary.json).

Run the pinned E058 winner's float64 neural-step comparison:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/assess_sensorimotor_precision_mbslnx2.sh &&
tail -n 20 -F "$(ls -1t /home/carsonc/r-elegans-results/cue-precision-*.log | head -n 1)"
```

Read-only, under nohup:fixed body step.002s and neural steps.1/.05/.025ms.
The existing E060 trace files provide the float32 reference and are preserved
recursively. Stored parameter values, legacy templates and starting states are
widened exactly; derived functions run in64, with trace dtypes verified. Host
wet mechanics is already float64 in both modes. No weights, gains or thresholds
are fit. Source verification occurs under legacy float32 before promotion, so
parameter-coordinate reconstruction cannot silently change the source model.
A precision-sensitive result still requires a convergence decision before
training or skill promotion.

The float64 assay uses an integer-step clock to retain the original float32
ligand on/off bins; a fixture verifies that equivalence. This prevents clock
rounding from confounding the precision comparison. Historical/default clock
behavior is unchanged.

Historical E061 finding and E062 plan (superseded by the E062 result above):
float64 peaks remain2.85671–2.85674e-5rad as
neural step shrinks. Peak variation0.00116%; full-trajectory differences halve
6.10e-9 ->3.05e-9rad. Float32 arithmetic strongly contributes to the prior
sensitivity in this test. Better-resolved amplitude is95.22% of target, still
unqualified. E062 prepares float64 fitting, preserving stored E058 parameters,
original coordinate bounds and3e-5 acceptance threshold. Optimize a declared
2% training-only margin; add low-dose numerical checks. Fresh seed6200 because
precision/objective changed; not an exact float32 continuation. No server job
launched. See [E061 report](experiments/records/E061-server-summary.json).

Start the new float64 fitting stage on mbslnx2:

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/launch_sensorimotor64_mbslnx2.sh 60 &&
tail -n 28 -F "$(ls -1t /home/carsonc/r-elegans-results/sensorimotor64-*.log | head -n 1)"
```

The pinned E058 checkpoint is promoted without regenerating its stored weights.
An immutable operational reference/origin preserves that exact starting model
while retaining the original coordinate box and prior; inherited float32
encoding residual is logged. Sensory coordinates migrate to their old
operational gain values. Legacy-rounded body/mapping templates and initial
states are promoted, and event bins use the verified canonical clock. A baseline
low-dose check against the pinned E061 float64 reference aborts on migration
mismatch before fitting. No muscle/body parameters are trained.

Validation threshold remains3e-5rad. Selection uses3.06e-5rad (2% training
margin), with the worse nominal/.8dose loss and unchanged prior. New validation
includes low-dose half-step and neural-quarter replay in addition to original
gates/controls. Losses differ from the old float32 objective. This is a fresh
seed6200 search, not old optimizer continuation. Budget60minutes plus validation.
Full source/RNG/selected/final references and traces stay outside Git.

Later continuation uses `sh scripts/continue_sensorimotor64_mbslnx2.sh 60`; it
restores float64 reference, coordinates and RNG, rejecting precision/margin
changes and checksum mismatch. The compact reporter finds both training families.
Original float32 scripts and historical passes remain separate.

## Post-cue alternation pilot (E063)

The E062 recovery checkpoint remains independent. Launch the next60min search
in the dedicated conda environment; the launcher pins its parent SHA, selects
four physical CPU cores and detaches with nohup. Validation adds wall time.

```sh
cd /home/carsonc/github/r-elegans &&
git pull --ff-only &&
sh scripts/launch_alternation64_mbslnx2.sh 60 &&
tail -n 28 -F "$(ls -1t /home/carsonc/r-elegans-results/alternation64-*.log | head -n 1)"
```

Read the final report with the dedicated interpreter:

```sh
/home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.report_sensorimotor
```

This is a fresh seed6300 search from E062, not an exact optimizer continuation.
Both nominal and low doses train unknown full-network channel densities and
existing synapse strengths plus sensory gains, within original bounds. No
periodic input or prescribed motor waveform is supplied. One exposure2–3s is
followed by observation4–12s. At least one joint must show two raw peaks and two
troughs, a turning point in both late windows, >=1e-5rad detrended peak-to-peak
in each window and minimum/maximum window amplitude>=.5. Training adds2% margin.
Matched refinements must preserve the same joint's qualification, amplitudes
within25% and peak/trough counts within1, alongside cue-effect checks. Chemical
block removes cue recruitment; held voltage must remove qualifying motion.
Sham spontaneous repetitions are described separately. Success would establish
only this short-window prerequisite; slow oscillatory ringdown can still pass.
It does not establish long autonomy, propagation, navigation
or a biologically calibrated gait. Local smoke uses an older local parent;
actual E062 winner results await the user's server launch.

E067 server result (2026-11-12): all four connections-only searches finished
without meeting the small amplitude rung. Channels stayed fixed; every active
validation retained the motor guard, and held-voltage cyclic bends were0.
Across249 proposals,126 were accepted. Each seed made only62–63 proposals.
Ordinary tail gains were3.9–14.4%; middle/head changes were smaller or mixed.
Seed6703 reaches roughly44%/85%/89% of short-window T/M/H targets, but is not
declared a full-objective winner without complete losses. Long windows still
miss targets. Numerical cycle changes under half-step tests are below1%.
See [full compact evidence](experiments/records/E067-server-summary.json).

Decision: preserve all trained candidates and the original hybrid. Do not infer
a connectome limit from this small costly search. Before another long run,
compare already saved old/old vs old/new amplitudes with identical scoring and
inspect dorsal/ventral cancellation and the fixed NMJ transfer. NMJ weights,
polarities and muscle transfer were held fixed here, so brain-weight fitting
could not explore that interface. These are hypotheses to test, not diagnosed
causes. No new model changes or experiment have been launched.

Historical E067 preparation (completed server result above).
Four independent seeds6700..6703 use the pinned restored hybrid f024bd14... .
All channel scales, NMJ/body, sensory gains and topology remain fixed. Targets
are tail1e-4,middle/head3e-4rad direct cyclic excursion, scored over30–45/45–60s
with a retained motor-cycle guard. Original-anchor connection factor2 bounds
apply, with declared inherited numerical slack only. Each240min baseline/search
budget is followed by60s perturbation/refinement and300s validation. No periodic
drive. Per user request for a command within3min, no Mac worm simulations or
end-to-end fit validation were performed; compilation/shell and quick synthetic
scoring checks only. Server parent-hash/state/reference checks precede proposals.
This is a new objective/seeded stage, not an optimizer continuation or biological
qualification. All selected/final weights, references, RNG and traces external.
