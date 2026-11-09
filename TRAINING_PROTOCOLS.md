# WormGym: training and validation protocols

Design baseline: 2026-11-03. Companion inventory:
[BEHAVIOR_CATALOG.md](BEHAVIOR_CATALOG.md). All P00–P13 recipes below are proposed;
the existing Gaussian-cue Gymnax environment is the only current RL environment.
This is a plan for implementation, not a report of completed training.

Implementation update (2026-11-04): `scripts/train_timed.py --actor full-network`
now trains all 302 recurrent neurons and the anatomical 95-muscle readout in a
separate normalized-dish environment. The default train group is `all`;
`--train-group` and `--train-neurons` freeze parameters without removing nodes.
`--initial-actor` carries parameters between stages with a fresh optimizer and
critic. The deployed full actor has no fitted gait or phase input. This is
training infrastructure with smoke-test validation, not a completed P00–P13
curriculum or evidence of real-animal behavioral agreement. Known NMJ signs
and anatomical edge masks remain constraints; unknown NMJ signs can learn.

## Intrinsic motor-cell qualification (E024, 2026-11-07)

The opt-in reduced active-membrane model is an additional mechanistic branch;
it does not replace the preserved driven-skill policies. Run
`python -m scripts.assay_intrinsic_motor --output-dir /external/fresh/path`.
It checks isolated autonomous DA/VA-like dynamics, DB/VB-like steady-drive
dependence and withdrawal, calcium-current ablation, initial-state perturbation,
substep convergence, and finite connected-network execution. The equation form
is inspired by [Morris and Lecar (1981)](https://pubmed.ncbi.nlm.nih.gov/7260316/);
its demonstration coefficients are not worm-specific channel measurements.
Gao/Xu evidence motivates the functional distinction, not these numerical values.
No optimizer or fitting is involved. The A regime has an explicit fixed bias;
the B assay uses constant current as a permissive-input proxy. It does not
establish the AVB-specific electrical-coupling mechanism.

Require at least three final-window cycles and >0.2 normalized peak-to-peak
voltage for A and driven B; <1e-4 peak-to-peak for quiet/withdrawn B and
calcium-ablated A; maximum state discrepancy <0.002 when doubling substeps.
Windows, inputs, initial states, fixed coefficients and all traces are recorded.
Do not equate those gates with biological electrophysiology or locomotion.
Future motor-skill fitting must qualify the new dynamics independently and
preserve configuration with checkpoints; legacy NPZs do not encode this mode.
In the initial connected network some A rhythms are suppressed by coupling;
coordinated muscle waves, proprioception and propulsion are still outstanding.
This is recorded evidence, not a reason to inject a hidden periodic command.

## Physiological fitting before behavioral inference (E025)

Fit identifiable membrane constraints before synaptic strengths to reduce
compensation between cell excitability and connectivity. This is a staged
estimation strategy, not evidence that every conductance becomes identifiable.
Preserve fixed channel kinetics/reversals, separate whole held-out voltage
steps, and report data-only sensitivity rank alongside regularized optima.
Cross-class priors remain assumptions; zero expression constraints must be
supported by evidence rather than inferred from a small optimized conductance.
The E025 VD05 candidate passes provisional current-prediction and numerical
gates but cannot identify all 17 conductances. It is not a fitted DA/VA/DB/VB
model and is not installed in the normalized actor.

For synapses, require known pre/post voltage, outward intrinsic/capacitance-
subtracted postsynaptic current, fixed release/reversal parameters and confirmed
isolation of omitted cells. Hold out complete independent stimulation trials;
respect chemical masks, share each electrical conductance between both ends,
and do not interpret indirect network responses as direct edge conductance.
Current synaptic recovery evidence is **synthetic software validation only**.
Behavioral training may later infer effective missing parameters under priors;
such values must be labeled behavior-inferred rather than electrophysiologically
measured. Do not transfer physical nS into normalized actor units implicitly.

## Calcium-constrained full-network inference (E026)

The user directs fitting calcium recordings from the current parameter settings,
with channels in all neurons. This does not require every neuron to spike.
Use the all-cell 17-current branch and the recorded normalized-to-physical
migration, preserving the exact parent weights/identities/config externally.
Published whole-cell class models are priors; most assigned channel profiles
remain transferred and uncertain. The Nicoletti 2024 VA/VB/VD/interneuron models
are additional available evidence; the current library has not yet ported their
extra currents. Earlier local-data gaps do not establish absence from literature.

Fit predicted intracellular calcium through an effective indicator model to
identified fluorescence traces. Do not compare voltage directly to GCaMP or
interpret fluorescence peaks alone as proven action potentials. Keep source
reporter, compartment, normalization and label confidence explicit; do not infer
unresolved left/right identities from desired activity. Only training samples
may set normalization, observation readout or nuisance parameters. Reserve
contiguous future frames and eventually entire animals/interventions.

The first fit trains channel multipliers and existing chemical/gap strengths,
with fixed masks/kinetics/calcium/release/reversal/indicator parameters and no
target-derived current. Record both loss against the initial model and against
a training-mean baseline; a small training improvement with poor validation is
a failed predictive fit. Distinguish numerical integration error from statistical
error by doubling substeps. Whole-network gradients do not make every cell or
edge identifiable from partial head recordings. Requalify motor oscillation and
body motion before promoting any calcium-fitted candidate into skill execution.

## Preserve self-sustained dynamics before further activity fitting (E027)

The user prioritizes channel kinetics capable of sustained activity. Existing
Ca/K gate feedback is sufficient in a literature-inspired RMD conductance regime;
additional channel names alone do not ensure it. First qualify persistence without
periodic input, separating an oscillatory attractor from a plateau or long transient.
Use washout plus five late windows, >=2 mV amplitude in each, amplitude retention
0.8..1.2, >=5 peaks and bounded voltage. These gates target the tested fast RMD
regime; slow brain states need longer assays, not an assumption of universal failure.

Confirm from multiple initial conditions, remove the regenerative calcium current,
retain ordinary non-oscillatory controls, then reconnect the anatomical network.
RMD's CCA1-dependent candidate must not be mislabeled as an UNC2-dependent A-motor
model. Compare refinement by both raw trajectories and phase-aligned late cycles:
period error <1%, amplitude error <2%, waveform RMS <0.5 mV. Retain phase drift
rather than claiming pointwise identity over arbitrarily many cycles.

`fit_calcium_network.py --initial-network /external/network_candidate.npz` retains
the candidate's full channel/config initialization. Add `--qualify-rhythms` to run
a separate 20 s unpaced persistence assay on initial and final parameters; both
must retain at least one qualifying RMD cell for a zero exit code. This guard does
not promise all six RMD cells oscillate or that a candidate matches biology.
Use small integration steps and refine in each fitted regime. The current candidate
is not promoted into motor execution; body feedback and gait qualification remain
required. Do not trade away the recorded E026 calcium-fit failures or call a
one-update integration check a successful recording fit.

## Unpaced brain-to-body diagnostic before training (E028)

Run the fixed channel candidate through physical release, preserved 956-edge
NMJ/muscle transfer and the 95-muscle body, with no pacing or parameter fitting.
Keep physical release fields explicit; do not feed mV into old normalized
thresholds. Retain the parent's nonzero resting tone and signs/multipliers.
The shared execution loop has no proprioceptive feedback yet.

The server queue contrasts normalized/liquid/agar bodies, +/-1 mV initial-state
perturbations, RMD CCA1 removal, held voltage, body timestep and brain integration
refinement. It uses four disjoint four-core CPU workers, a 45-minute global budget
and bounded cases. Save all time chunks and complete latest/final neural/body
states, input hashes/source/environment and completion/omission status. This is
not an RL run or a fit of channel/capacitance/synapse values.

Default pull-and-run launch: `sh scripts/launch_embodied_mbslnx2.sh`. Use the
pinned `r_elegans/assets/mac_rhythm_v1/` network and matching parent, verify both
hashes, then copy them to the external run directory. These are the exact Mac
20-s test inputs. A server-local head-sweep checkpoint is no longer selected
automatically. Explicit `--parent-actor` reconstruction and exact `--inputs`
overrides remain available. Keep the reference fixed when testing longer
persistence or comparing hosts; a matching parameter hash does not promise
bitwise identical numerical trajectories across JAX/platform versions.

Late-window flags: maximum muscle activation p2p >1e-4 for dynamic output, maximum
joint p2p >1e-4 rad for dynamic shape, centroid displacement >1e-5 body lengths
for measurable drift. Neural persistence uses E027 gates. These are numerical
diagnostic flags, not empirical animal thresholds. Report full amplitudes,
path/net displacement, heading, saturation and joint limits alongside flags;
neither a static bend, startup relaxation nor nonzero drift establishes gait.
The local liquid candidate produces small periodic bends but negligible net
displacement. Useful locomotion and body-feedback qualification remain pending.

First server queue (user-reported `embodied-20261107T185913027018Z`): nine
finite exit-0 cases, eight at 600 s and brain-refined at 374 s (case budget).
All final measurement windows are stationary, including neural voltage and
muscle output. This is a negative persistence result despite successful software
execution. The server used a different parent from the 20-s Mac test.

Second queue (user-reported `embodied-20261108T035729978291Z`, bundled-reference
workflow): three rhythmic RMD cells persist in final windows at 600 s in all
active cases, including +/-1 mV initial perturbations and body refinement.
Controls have zero late neural/muscle/body variation. Brain-refined case retains
three rhythmic cells at 368 s before its budget. Active liquid joint p2p is
0.003352 rad and late net motion 2.64e-6 body lengths. This qualifies longer
unpaced activity in the model, not useful locomotion or empirical physiology.
Full trajectories/input hashes remain unretrieved; compact output omits hashes.
Next priority: measure spatial phase and traveling-wave coordination in saved
outputs, preserving the fixed rhythmic reference. No new fitting run launched.


## Fix stereotypical properties; fit explicit unknowns (E029)

New calcium-fitting default supersedes the all-conductance E026/E027 experiments.
Use `unknown-only-v1`: fixed channel kinetics, reversals, calcium handling,
capacitance, leak reversal, topology, release fields and observation/readout.
Fix reference LEAK conductance as a scale anchor, RMD class/exploratory profiles
as held hypotheses, and unobserved cell channel priors for each recording.
Only remaining unknown observed-cell channel densities and unmeasured strengths
on existing chemical/gap edges vary. This preserves whole-network simulation;
an unobserved prior being held is not evidence that it is biologically known.

Store policy/masks, sources/reasons and fixed-versus-trainable counts, and verify
frozen fields remain exactly unchanged after optimization. Additional known channel
or edge freezes are external provenance-bearing constraints; electrical pairs
freeze symmetrically. Do not use synapse contact counts as conductance measurements
or force arbitrary bilateral equality. The E028 workstation run remains entirely
fixed-parameter and is unaffected by these optimizer defaults.

## Working hypothesis: motor curriculum before task RL

[H1 in the laboratory notebook](LAB_NOTEBOOK.md#h1--build-the-motor-repertoire-before-task-rl)
records the user's hypothesis: establish and retain an effective motor
repertoire through small motor-learning stages (currently supervised) before
downstream sensory-task RL.
The literature terms are **curriculum learning** and **motor-skill pretraining**;
learned skill priors and hierarchical RL are related potential extensions.
The hypothesis is specific to this project, not a universal prerequisite for RL.
Its predicted benefit is more effective exploration and better task learning
per total compute/sample budget, counting pretraining costs.

Current motion remains externally paced. Reliable skill retention, endogenous
rhythm and sensory-guided choice are separate milestones. Neither a hierarchy
nor a learned probabilistic skill prior has been implemented. A controlled
comparison of incremental pretraining, abrupt pretraining and no pretraining
is proposed in H1; it has not been run. Preserve physical progress and old-skill
gates at each step rather than assuming imitation scores establish readiness.

## Current motor milestone (updated 2026-11-06)

**Current priority, user revision H1.1:** expand the independently validated
[skill repertoire](SKILL_REPERTOIRE.md). Separate full-network skill policies
and checkpoints are acceptable. Do not block learning steering/head sweeps
on consolidating forward/reverse into one weight set. Retain physical gates
per skill and save prior successful policies; track interference and address
consolidation later. The intended downstream RL action is a skill identifier
plus validated parameters/duration. No high-level selector is implemented yet.
This update supersedes the consolidation-first proposals below; completed
experiments remain historical evidence.

Implemented next standalone fit: `--steering --minutes 480` from the original
0.07 wave cohort. Four seed-matched policies, signs alternating by seed parity,
expand MOT-08 independently. Target amplitude 0.07 plus signed muscle bias
0.006; same pacing gain and 0.4-second target propagation delay. Learning rate
0.0003. Objective: normalized muscle MSE plus 0.5 times the mean normalized
joint-angle/centroid-position trajectory error through the physical body.
No cross-skill retention loss. Initial local 0.06 development advances target
amplitude to 0.07 (16.7%); server parents already use 0.07.

Every 250 updates, selection checks longer waveform commands and a 4-startup/
6-measured-cycle actual turning probe. Final tests at 2.4/3.0/3.7 s, 20/10 ms,
rest and isotropic drag require positive progress >=0.001 body lengths/cycle,
signed centroid/body turning >=0.00005 rad/cycle and at least half the oracle
turn, consistent signs and agreement above numerical floors. Isotropic drag
checks translation, not the absence of geometric rotation. The coarse tuning
probe also requires >=0.001, giving a conservative additional selection gate.
Detailed predeclaration, development failures and results are in E019. No
autonomous rhythm, biological turning calibration, or inter-policy state
handoff is claimed. The server run is not launched by the coding agent.

Subsequent user report E020 shows `MOTOR_CURRICULUM_EXIT_CODE=0`: all configured
steering workers passed according to the supervisor. The pasted tail includes
a valid negative-turn condition but not the complete per-seed results. Preserve
the aggregate reported success and retrieve full archives before reporting
seed statistics or independent checkpoint verification. Head sweeps are the
next proposed standalone coverage target, not a fit already run.

Head-sweep implementation E021: `--head-sweep --minutes 480`, using the
original wave cohort. Four independent full-brain policies train at-rest,
bidirectional anterior scans with target amplitude 0.04 and envelope width
0.18 along anatomical position. Input is a fixed anterior-NMJ-weighted
artificial current times scalar pacing. No forward/reverse/steering retention
is required. Parent models, all neural nodes and fixed NMJs are preserved.
The shared standalone trainer keeps muscle/body-joint/centroid trajectory
losses and validation-based selection, with separate head-specific metrics.

Head gates: angle peak-to-peak >=0.02 rad and half the oracle; head-center
lateral excursion relative to trunk >=0.0005 body lengths and half the oracle;
posterior/anterior joint dynamic RMS ratio <=0.25; cycle centroid drift
<=0.0001 body lengths and trunk rotation drift <=0.002 rad. Require periodic
shape, stable cycle amplitudes, timestep agreement and <=10% residual scan
under zero pacing. This intentionally replaces propulsion gates with
stationary-scan gates, not a relaxed test of locomotion. Isotropic drag is
recorded as a load control; an isolated head oscillation need not disappear
under isotropic drag. Commands, targets, source, environment, input/checkpoint
hashes and full physical trajectories remain external and reproducible.

Local E021 outcome: three 1,000-update stages with fresh optimizers produced
a policy passing both longer-command tuning (body errors 0.230/0.249) and all
physical head tests. Earlier stages passed physical tests but missed the
unchanged trajectory gate and remain archived. This establishes one driven,
stationary head-scan policy, not multi-seed or biological validation; server
replication is the next run.

Server E022 now reports four-seed success: physical and tuning gates pass for
every seed and all 12 pacing conditions. Preserve the selected actors and
their startup/source/environment records at the reported run path in the
notebook. This completes the reported stationary head-scan cohort, pending
independent archive inspection; no gate was relaxed or new fit launched here.

For the chronological protocol/outcome record and replication checklist, see
[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md). The historical stages below describe
progression, not a claim that all current models pass every stage.

The externally paced 302-neuron 0.07 ensemble has passed forward locomotion
readiness for three of four seeds. All four pass wave fitting; seed 2 misses
the 0.001 body-length/cycle floor at its slowest pacing rate (0.0009956).
No natural sensory input or autonomous rhythm is demonstrated.

The next implemented protocol is `assess_stop_restart.py`, also exposed by
`run_motor_curriculum.py --stop-restart --resume-run <wave-cohort>`.
Test existing weights first: continuous go/zero/go commands, six cycles per
block, last three measured, at three pacing periods. Keep neural and body
state across transitions. Require <=10% residual centroid path length and
joint-angle speed during the late stop window, 80–120% recovered progress,
and the unchanged moving-window propulsion gates, timestep refinement and
isotropic controls. Record `control_passed` separately from overall readiness
so a borderline propulsion floor is not confused with inability to stop.
The subsequent remote diagnostic passed stop/recovery for all four seeds;
overall readiness remained passed for seeds 0/1/3, with seed 2 still below
the slow-rate progress floor. No additional training was needed.
If it fails, use the failed criterion to design the next supervised fit;
do not discard the established forward-motion checkpoint.

Next implemented fit: `run_motor_curriculum.py --reverse --minutes 180`.
Use the 0.07 forward cohort explicitly. Teach a posterior-to-anterior target
with a fixed artificial direction context, while rehearsing the original
actor's forward outputs with equal loss weight. Keep the source amplitude,
delay, pacing gain, full 302-neuron simulation and fixed NMJ/muscle transform.
Validate reverse lags and forward imitation on longer unseen commands, then
run go/stop/restart physics assays independently in each direction. Negative
signed progress is required for reverse; the same absolute progress floor,
straightness, stop/recovery and numerical controls remain. Overall success
requires forward retention as well as reverse behavior. Save every outcome;
an unsuccessful fit must not replace the original forward checkpoint.
This is supervised motor construction, with held direction per assay; rapid
direction switching and sensory-task RL remain later stages.

Three-hour outcome: reverse translation and stop/restart control in both
directions passed for all four seeds, but no seed passed every imitation and
propulsion gate. Forward progress weakened in seeds 0–2; seed 1 failed changed
hold commands and seed 3 missed reverse wave timing. Preserve these learned
models and the original forward references.

The next implemented protocol, `--refine-reverse --resume-latest --minutes
480`, resumes the reverse candidates with the **original** forward actor as
teacher (checksum required). Use eight longer training sequences at multiple
pacing rates and eight different tuning sequences at twice the horizon.
Give forward relative muscle error weight 4 and reverse error weight 1;
use a smaller learning rate of 0.0003. Select checkpoints jointly on forward
and reverse body/muscle validation every 500 updates, prioritizing failed
case count and worst error. Do not call tuning data used for selection an
independent test. After fitting, run the existing long physical go/stop/go
assays in both directions. Keep all previous thresholds, fixed NMJ/muscle
mapping and artificial direction context. No automatic turning or RL follows.

Eight-hour outcome: tuning imitation passed seeds 0/1/2 and forward readiness
passed 0/1, but no seed passed reverse propulsion readiness. All retained
backward translation and stop/restart control, and every stage completed
without errors. Reverse go magnitudes remained 0.0005968–0.0009962 body
lengths/cycle, below 0.001; seed 3 also missed one forward tuning case.
Do not equate passing imitation with passing behavior. The earlier proposal
was to improve joint movement optimization/selection. H1.1 instead prioritizes
independent skill coverage now; physical movement criteria still apply to
each skill. Neither a new movement objective nor the new skill fits have
been implemented or run by this strategy update.

## Scientific objective

Autonomous-rhythm prerequisite identified in E023: account for distributed
motor oscillators, descending state drive and proprioceptive coupling, and
close the body-to-neuron execution loop. Current body-loss differentiation
does not provide that feedback. A future test should require sustained,
coordinated motion with no externally periodic signal, starting with a tonic
permissive command or brief trigger, while measuring load/posture robustness.
This is distinct from sensory-guided action selection. It is a proposed
mechanistic milestone, not a new implemented model or replacement of H1.1.

Locomotion is now assessed separately from wave imitation: frozen neural
weights, complete phase-matched cycles after startup, actual body centroid and
axis, timestep refinement, rest and isotropic-drag controls. One local seed
shows small sustained forward translation. A 0.04 → 0.05 supervised amplitude
step preserves wave gates and increases displacement by about 59%, but remains
below the declared 0.001 body-length/cycle readiness threshold. The Linux
runner can assess passed wave ensembles or perform one guarded amplitude step
and re-assess. These engineering criteria in normalized mechanics do not
substitute for measured animal locomotion or SI calibration.

The next motor stage is implemented as `train_motor_alternation.py --stage
wave`: a causal 0.4-second anterior-to-posterior target delay with the same
scalar artificial stimulation input. No phase/delayed muscle pattern enters
the forward neural computation. Both muscle and body propagation are gated,
in addition to trajectory matching. `run_motor_curriculum.py --resume-latest`
continues each passed alternation seed into this stage, with fixed NMJs and
fresh optimizers. This is externally paced wave fitting; the later assays
above establish model propulsion, while autonomous rhythm and real-animal
fidelity remain unestablished.

Current first motor step: `scripts/train_motor_bend.py` performs supervised
fitting of one static dorsal bend, with only full-network neural parameters
learning and a fixed NMJ/muscle transform. It passed a longer-hold body check
for seed 0. This precedes alternating bends, traveling waves, forward/reverse
motion and turns. Do not use food-seeking RL as the first motor-learning task.
The static bend is an infrastructure milestone, not a completed biological
behavior protocol. The next implemented stage fits commanded alternating
bends with artificial currents to NMJ-projecting neurons, while keeping NMJs
and muscle transfer parameters fixed. Validation changes command order, levels,
phase and rate, at twice the training horizon. The mbslnx2 motor curriculum
runner stops a seed when its stage fails; it never launches food-seeking RL.

Test whether a constrained embodied model can reproduce the animal's behavior
across stimuli, context, and experience. Task reward establishes usefulness;
agreement with independent real-animal observations establishes behavioral
fidelity. Similar behavior alone cannot identify unique synaptic weights or
prove that the animal uses the same learning algorithm.

Three distinct optimization levels must remain visible:

1. **Physical identification:** estimate body/medium parameters from passive
   mechanics and locomotor measurements, with uncertainty.
2. **Offline policy fitting:** imitation and RL optimize a controller across
   virtual animals/tasks. This is model construction, not observed learning by
   an individual worm.
3. **Within-animal experience:** the deployed controller changes its responses
   during conditioning and testing with the optimizer frozen. Recurrent state,
   adaptation, neuromodulation, or a trained plasticity rule can mediate this.
   Reset at animal boundaries, not between conditioning and its memory test.

## Network sufficiency and weight specificity

The primary neural-control question is whether the specified network can match
the behavior suite, regardless of whether its fitted weights resemble the
unknown biological weights. Initialize from available anatomical/physiological
priors, then optimize behavior. Distinguish an initialization-only prior from
an ongoing regularization penalty; run the former as the main sufficiency
experiment and the latter as a separately labeled constrained comparison.
Declare which topology, sign, neuron-dynamics and parameter-range constraints
are fixed in every experiment. Sufficiency is conditional on those choices.

Train ensembles from independent initial weights and compare:

| Experiment | Readout |
| --- | --- |
| Independent fits with the same topology and budget | Fraction meeting all held-out behavioral criteria; training cost and failure modes |
| Successful weight ensembles | Diversity of effective conductances and required weight relationships, accounting for parameter scaling/nonidentifiability |
| Individual-edge and distributed perturbations | Behavioral degradation versus perturbation magnitude, including variation across successful networks |
| Neuron silencing and excitation/inhibition changes | Necessary components and compensatory routes across tasks |
| Topology-shuffled controls matched for size, degree and applicable signs | Whether anatomical wiring improves achievable behavior, learning efficiency or robustness |

Measure perturbations in effective weights rather than only unconstrained raw
optimizer parameters. Keep body, sensors, effectors and readout authority
matched, and report frozen-weight damage separately from recovery after
retraining. Broadly different successful solutions and robustness within each
solution are separate properties. A failed optimization run does not establish
network insufficiency; evaluate alternative seeds, budgets and optimization
methods before interpreting failures. These experiments are proposed, not run.

## Shared environment contract

- Use SI units for position, time, medium properties and actuation. Pair world
  and body calibrations. Version normalized legacy tasks separately.
- Agents receive local receptor signals and allowed proprioception/internal
  feedback. Source coordinates, distance-to-goal, full gradients, and task labels
  belong to the evaluator unless a protocol explicitly justifies them.
- Replace the current source-relative concentration channel with a calibrated
  local measurement for biological tests; normalize using fixed training-set
  constants, not the unknown source peak or current source distance.
- Specify chemical species, diffusion/advection/boundaries, source emission,
  bacterial patch geometry, and ingestion separately. Introduce mass conservation
  and no-flux boundaries where relevant before long dish assays.
- Supply distinct chemical, mechanical, thermal, gas and light interfaces.
  An arbitrary scalar “food sensor” cannot stand in for all modalities.
- Use recorded stimulus waveforms and locations. Calibration, organism
  measurements and simulator assumptions must have separate provenance.
- Each task defines initial conditions, stimulus schedule, reward, event-based
  termination, time-limit truncation, policy reset, horizon, metrics, and held-out
  conditions. Register these before comparing policies.
- Stochasticity must be seeded; randomize plausible physical/sensory parameters
  within measured uncertainty. Hold out some loads and cue configurations.

## Actions and controller comparisons

The existing Gymnax bounds are `[0,1]` speed and `[-1,1]` steering. That excludes
active reversal and is insufficient for the full chemotaxis/escape repertoire.
Build and validate a signed-speed interface, pause, and a distinct deep-turn
primitive with contact-aware mechanics. Smooth steering alone is insufficient.

Use two action levels with separate claims:

- **Behavioral-policy benchmark:** signed speed, steering, and validated turn
  selection drive a frozen motor-skill library. This isolates task selection.
- **Neural-control benchmark:** sensory input drives recurrent neurons, their
  outputs drive anatomical NMJs, and the 95 muscles drive the body. Score the
  same behaviors, but do not insert a hidden command-to-gait teacher and call
  the result fully neural control.

The 14-neuron actor is a steering baseline. Compare it with an analytic policy,
an unconstrained recurrent policy, and later a suitably equipped connectome
controller. Match observations, action authority, simulator budget and training
trials. Its speed channel is currently engineered, so a success is not evidence
of connectome-driven speed regulation.

## Objective design

For a proposed imitation-plus-RL curriculum, begin with the schematic objective

```text
maximize expected task return
  - lambda_motion * discrepancy(real, simulated motion features)
  - lambda_sequence * discrepancy(real, simulated event sequences)
  - lambda_effort * actuator effort
```

Feature discrepancies may compare curvature spectra, speed, bout distributions,
transition hazards and stimulus-response curves. Their weights and scales must
be fitted on training/validation data and locked before final evaluation.
Distribution matching needs batches or windows: do not pretend a single
instantaneous reward can measure a whole bout-duration distribution. Use
demonstration likelihood, an auxiliary sequence loss, or an appropriately
validated trajectory discriminator where needed.

Do not maximize displacement, food residence, or energy conservation alone:
each admits behavior unlike the animal. Penalize simulator exploits such as
body penetration and numerical limit cycling. Reward stochastic exploration
only to measured behavioral extent; training entropy is not proof of realistic
behavioral variability.

## Protocol recipes

### P00 — Physical and numerical qualification

Scope: prerequisites for all MOV entries. No RL.

Fit passive bending/relaxation and drag using independent measurements; then
test force/torque balance, dissipation, timestep convergence, segment-resolution
convergence, viscosity dependence and heading/position invariance. Include
nonpenetration tests before contacts/deep turns. Run prescribed waveforms
through the medium and compare predicted motion with recordings. Literature
parameter sets are starting points, not a quantitative validation certificate.

Gate: independently held-out mechanics/kinematics agree within predeclared
measurement-based tolerances. Keep residual errors and parameter uncertainty
visible. Current algebraic physics tests cover only part of this gate.

### P01 — Motor repertoire and gait transfer

Catalog: MOV-01–09. Train skill models on real centerline sequences or derived
muscle targets; use RL only to close residual control gaps.

- Setup: uniform agar or defined liquid; fit forward, reverse, pause, shallow
  left/right turns and later omega/delta turns separately. Do not impose exact
  left/right equality if the experimental cohort has asymmetries.
- Curriculum: short straight runs → speed changes → reversals → shallow turns →
  contact-qualified deep turns → transitions between skills and loads.
- Observation: local curvature/stretch history and muscle/internal state;
  commanded skill is allowed only for this motor-training stage.
- Reward/loss: track real curvature dynamics and target motion, with effort and
  shape/contact constraints. Random target headings are control exercises, not
  evidence of spontaneous biological turn selection.
- Timing: resolve the shortest measured muscle/shape timescale; use substeps
  justified by convergence. Episodes span several gait cycles and transitions;
  report physical seconds, not merely RL steps.
- Evaluation: speed and slip versus frequency, wavelength/body length,
  head/tail amplitude, turn exit-angle and reversal-duration distributions.
  Hold out animals, sequences and some medium loads.

Gate: a frozen motor library covers the needed repertoire in the intended
medium. Do not tune physics to compensate for a defective policy.

### P02 — Acute escape and stimulus-response tuning

Catalog: SEN-01–05, SEN-10; CHE-05/06 variants after chemical transduction exists.

- Setup: sparse, randomized anterior/posterior/nose mechanical stimuli, or
  calibrated light/aversive chemical pulses. Include sham trials and both
  moving and paused starting states.
- Observation: stimulus only through spatially appropriate sensory channels;
  mechanical contact comes from the physical stimulus/body interaction.
- Training: imitate stimulus-aligned response sequences, then RL for escape
  from a local hazardous region with bounded effort. Keep modality-specific
  parameters and anatomical input locations explicit.
- Evaluation: response probability and latency, movement direction, reverse
  distance, acceleration, turn completion and recovery; stratify by intensity,
  stimulation site and pre-stimulus movement phase/state.
- Controls: no-stimulus behavior, held-out stimulus strengths, sensor silencing,
  anterior/posterior swap, and motor-competence tests after silencing.

Horizon must include pre-stimulus baseline and post-response recovery. True
terminal danger/success rules are distinct from an observation time limit.

### P03 — Food encounter and slowing

Catalog: FOR-01/02.

- Setup: a stable bacterial patch emitting cues; train matched well-fed and
  deprived virtual cohorts with explicit nutritional history. Include
  chemical-only and texture-only controls where the reference assay supports
  them.
- Observation: chemical/texture signals and physiological feeding feedback.
  Do not hand the actor “on food” or “deprived” labels as sensory shortcuts.
- Reward: nutritional benefit minus movement/feeding costs, with an imitation
  term for entry/exit dynamics. Before pharynx implementation, a contact-based
  ingestion surrogate is permitted only as an explicitly labeled baseline.
- Evaluation: speed before/after edge crossing, enhanced versus basal slowing,
  response latency, exit probability and recovery with refeeding.
- Holdout: new patch sizes, cue strengths and deprivation histories. Stop
  giving distance shaping during final evaluation.

### P04 — Spontaneous states and local-to-global search

Catalog: FOR-03–06, MOV-07.

- Setup: long uniform-food recordings and separate abrupt food-removal trials.
  Keep roaming/dwelling on food distinct from local/global search off food.
- Training: learn sequence/state-duration statistics from reference tracks,
  then optimize energy acquisition in randomized patch landscapes. Internal
  memory should respond to actual food encounter/removal; do not supply a
  countdown that labels the expected search state.
- Observation: local sensory history, proprioception and physiological state.
- Evaluation: conditional speed/reversal/turn distributions, covered area,
  state occupancies, duration survival curves, autocorrelation and transition
  hazards versus time since food loss. Fit state labels on reference/training
  data and lock the classifier for tests.
- Timing: simulated durations must cover minute-scale state changes and the
  reference food-removal window. Resetting every few seconds cannot test this.
- Controls: identical landscapes with different food histories, stationary
  random-walk baselines, and removal of the memory/state mechanism.

### P05 — Chemotaxis using both steering and pirouettes

Catalog: CHE-01–05.

- Setup: start with a characterized single chemical gradient; randomize source
  location and initial heading. Add multiple sources and changing gradients
  only after matching the baseline assay. Volatile and soluble assays use
  their own physical delivery models.
- Curriculum: replay concentration steps/ramps to calibrate sensory response →
  spatial gradients with motor library → mixed sensory/motor closed loop.
- Reward: arrival/benefit plus optional potential-based shaping during
  training, using `gamma*Phi(s_next)-Phi(s)` with explicit terminal handling.
  Match steering and turn statistics through the behavioral objective.
- Evaluation: chemotaxis index, first-passage survival curve (retain failures
  as censored trials), path efficiency, curvature versus transverse gradient,
  pirouette hazard versus experienced concentration change, and exit-angle
  distributions. Evaluator may compute gradients even though the agent cannot
  observe them directly.
- Controls: flat field, cue-location swap, matched baseline odor level,
  steering-disabled and reversal-disabled policies, sensory noise changes,
  and new gradients. Success near the source alone does not pass the gate.

The current seven-parameter model supplies a baseline for smooth steering. Its
forward-only action and source-relative cue preclude calling it a full
biological chemotaxis benchmark.

### P06 — Patch selection, risk, and food quality

Catalog: FOR-07/08, CHE-07; CHE-08 through P09.

Give multiple patches different edible biomass, ingestion efficiency and
delayed harm. Vary distance, depletion and prior experience independently of
odor intensity. Reward modeled nutritional gain and physiological harm, with
real leaving/preference dynamics constraining the trajectory objective.
Evaluate sampling, patch-leaving hazard, preference changes and risk/benefit
tradeoffs; include safe, nutritious and dangerous conditions separately.
Predator-cue escape and persistent egg-laying changes need different effectors
and timescales; defer the latter until P11.

### P07 — Thermal navigation and history

Catalog: SEN-06/07, LEA-05.

Expose virtual cohorts to controlled thermal/cultivation histories, then assay
them on gradients from the corresponding experiment. Train with sensory
history and internal memory; do not give source coordinates or a preferred
temperature label as the biological actor input. Compare navigation sign,
temperature occupancy, tracking duration and temperature variance conditional
on starting temperature, gradient slope and feeding state. An experiment that
does not show positive thermotaxis must not be “corrected” by rewarding it.
Test reversal of the gradient and shifts in thermal history with optimizer
frozen. Separate preference/acclimation from noxious-temperature escape.

### P08 — Gas/light/humidity and other modality extensions

Catalog: SEN-08–12; light escape also uses P02.

Create one independently calibrated field/transduction model per modality.
Use gas steps plus gradients with feeding/genotype controls; light pulses with
wavelength/intensity controls; humidity conditioning with thermal/hydration
coupling; and electric-field reversal for electrotaxis. Train stimulus-response
imitation first, then navigation if appropriate. Compare dose-response curves,
latency, occupancy and cross-cue conflict responses. Targets are assay-specific:
there is no universal gas optimum or universal humidity preference. Flow-driven
orientation should first be tested as a physics effect with a fixed controller.

### P09 — Learning, retention, and recovery

Catalog: LEA-01–05 and CHE-08.

This protocol trains the *capacity* for experience-dependent change over many
offline histories, then freezes optimizer updates for each test animal.
Conditioning modifies only the deployed model's allowed internal/plastic state.

| Assay | Conditioning/test sequence | Required controls and readouts |
| --- | --- | --- |
| Chemical adaptation | Baseline → sustained/repeated cue → washout → cue | Attenuation and recovery; different cue where supported; sensor response versus motor fatigue |
| Touch habituation | Baseline → repeated harmless stimuli → delay → test | Response probability and reverse distance separately; spontaneous recovery, dishabituation and independent motor-competence control |
| Salt/deprivation association | Paired salt/deprivation → delayed gradient choice | Salt alone, deprivation alone and temporally unpaired controls; retention and reacquisition |
| Pathogen-odor association | Odor/ingestion harm pairing → delayed choice | Safe-food, cue-only and unpaired exposure; preference change with nutritional/malaise controls |
| Thermal-history shift | Controlled prior temperature → gradient → altered history → retest | Gradient slope/feeding controls and persistence after delay; no optimizer intervention |

Interstimulus intervals and retention delays must match the selected reference
experiment in physical time. Do not reset policy memory between conditioning
and testing; reset/reinitialize between animals. Hold out conditioning schedules
and cue/outcome mappings. Reject a model that responds only to elapsed time or
trial number when contingency controls disagree with it.

### P10 — Quiescence, sleep, and internal-state control

Catalog: FOR-09, STA-01–03.

Build nutritional state, recovery/stress and stage clocks before training these
behaviors. Fit joint movement/feeding state sequences under matched conditions.
For sleep, require reversibility, altered stimulus responsiveness, and the
appropriate homeostatic response; immobility alone is insufficient. Compare
quiescent-bout distributions, arousal thresholds and deprivation/recovery
curves. Lethargus requires stage-specific anatomy and timing; it is not an adult
food-search task. Any fitness benefit assigned to sleep is a modeling hypothesis
to test, not evidence that a stillness reward explains sleep biology.

### P11 — Organ motor programs and their coordination

Catalog: ORG-01–06; persistent reproductive consequences of CHE-07.

Add pharyngeal/enteric/vulval effectors and nutrient/reproductive state. Fit
intrinsic organ dynamics and neural-to-organ coupling using event times and
organ recordings. Begin with supervised timing/sequence reconstruction; RL can
later optimize ingestion or safe oviposition in a world with meaningful
consequences. Evaluate pumping/peristalsis coordination, spitting particle flow,
defecation order and variability, egg-laying bouts and cross-program timing.
Do not give these event clocks to the policy as hidden answers. Validate neural
and organ perturbations separately to distinguish control from passive dynamics.

### P12 — 3D movement and dauer dispersal

Catalog: MOV-10/11 and STA-04.

Develop contact, torsion and calibrated 3D substrate mechanics. Train burrowing
on volumetric tracks across gel properties, measuring plane rotation, progress
and energy. Nictation needs dauer anatomy, tail attachment and a suitable
substrate/carrier assay. Reward physically realized carrier transfer only after
validating the tail-supported posture and waving distribution. A two-dimensional
line cannot qualify either behavior.

### P13 — Social and sex-specific programs

Catalog: SOC-01/02, SEX-01–05.

Use separate genotype/sex/stage configurations. First test multiple worms with
pheromone/gas production and physical interactions; score aggregation, lawn
bordering and cue-dependent preference against matched cohorts. Then introduce
male anatomy, connectivity, tail sensors, reproductive muscles and partner
contact. Train mating steps individually by imitation before optimizing their
sequence. Report per-step success, retries, latency and failure locations as
well as complete mating success. An arbitrary proximity bonus cannot establish
social behavior or successful reproduction.

## Data split and acceptance rules

1. Split by animal and recording/session, not random frames. Keep conditioning
   and its test within one split. Reserve a final cohort and assay variants.
2. Estimate real-versus-real variability from independent matched cohorts.
   Predeclare acceptable model discrepancies based on that variability and
   scientific effect sizes. Do not invent universal speed/success thresholds.
3. Use multiple independent training seeds (initial planning default: five),
   paired evaluation scenarios and animal-level confidence intervals. Determine
   final sample size from pilot variance/power; correlated frames are not
   independent animals.
4. Report four outcomes separately: task performance, motion realism,
   behavioral-sequence realism, and held-out biological perturbation effects.
   More food found does not compensate for missing reversals or wrong memory.
5. Include negative controls and challenge assays never used for fitting.
   Match silencing/genetic interventions only where the simulator represents
   the affected mechanism; keep a motor-competence control for sensory deficits.
6. Compare recurrence, sensory adaptation and plasticity ablations. For neural
   recordings, include calcium indicator/measurement dynamics and identify
   neurons rather than treating calcium traces as membrane voltages.
7. Retain legacy results as historical; rerun all current comparisons after
   physical/gait changes. A passing software test is not biological validation.

## Suggested first release and implementation order

| Gate | Deliverable | Blocking work | Initial benchmark IDs |
| --- | --- | --- | --- |
| 0 | Matched reference data and feature pipeline | Audit metadata/licensing; extract wild-type distributions | All first-release tasks |
| 1 | Validated physical motor library | P00/P01; signed speed, pause and turn interfaces | MOV-01–04, MOV-06–09 |
| 2 | Agar world and sensory interfaces | Chemical transport, food/cue separation, contact/touch and basic nutritional state | SEN-01–03, FOR-01/02, CHE-01–06 |
| 3 | Behavioral selection benchmarks | P02–P05; real transition/bout targets and long rollouts | FOR-03–06, CHE-03/04 |
| 4 | Within-animal learning benchmark | P09; memory/plastic-state semantics and controls | LEA-01–03 |
| 5 | Extended modalities and organs | P06–P13 capabilities and matching datasets | Thermal/gas, physiology, sleep, social, male/dauer |

Proposed first environment suite: `Crawl`, `Reverse`, `ShallowTurn`, `TouchEscape`,
`FoodEncounter`, `SearchAfterFoodLoss`, `Chemotaxis`, and `TouchHabituation`.
These names are a roadmap, not registered task IDs. Add task versioning only
when observation/action/reset/reward definitions are executable and reviewed.

The first research question is whether one shared sensory/motor controller can
match locomotion, touch escape, food slowing, search and both chemotaxis
strategies across held-out conditions. That is a more informative next milestone
than a high score on the existing single food-source task.

## P00 — Autonomous amplitude before wave coordination (E030)

User-directed first milestone: spontaneous wiggling amplitude, without external
drive, a traveling-wave teacher, propulsion reward or RL. All 302 injected
compensation currents are zero in this branch, including those retained by the
earlier Mac reference. The 20-s zero-bias baseline retains three rhythmic RMD
cells and small bends, so injected tonic current is not needed for that short
model assay.

`brain/autonomous_wiggle.py` fits 78 existing, non-LEAK channel-density values
in the six RMD neurons. The other 296 neurons remain in the full network.
Capacitance, leak, gating kinetics, ionic reversals, calcium handling, synaptic
strengths, release settings and NMJ/muscle/body parameters remain fixed.
RMD density freezing in the calcium-fitting policy was a model assumption;
this explicitly scoped amplitude fit relaxes it without changing that policy.
Densities are bounded to 0.25–4 times the starting value. No absent channel
is introduced. Derivative-free search avoids differentiating the full stiff
network and stops searching at the small engineering target (0.01 rad p2p).

The objective measures repeated late joint-angle variation, using the weakest
of five late amplitude windows at the strongest varying joint. Static postures,
decaying transients, unbounded voltages, clipped joints and excessive muscle
saturation cannot pass. At least one RMD neuron must retain a qualified rhythm.
This milestone permits a local/head wiggle; it does not demand whole-body waves.

Local seed-0 short verification selected candidate 2: sustained bend amplitude
0.002305 -> 0.026224 rad in the 12-s fit window (~11.4x). Twenty-second held-out
validation and +/-1 mV initial perturbations and finer brain timestep retain
three rhythmic cells and amplitudes 0.0261–0.0279 rad. Strongest bends are near
the head. All currents are zero; all frozen fields compare exactly. Learned
density factors are only 0.858–1.225 of the prior in this candidate.
Independent 300-s validation passed: three RMD rhythms, sustained joint
amplitude 0.02823 rad and maximum joint p2p 0.02951 rad. Every 2-s head-joint
window after the first 12 s exceeds 0.02681 rad. Holding voltage constant
stops all late motion; no clipping or muscle saturation occurred.
These are model dynamics, not animal-calibrated channel densities.

The exact fitted pair is bundled in `r_elegans/assets/autonomous_wiggle_v1/`,
with hashes/provenance. To run the long fixed-parameter diagnostic on mbslnx2:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_embodied_mbslnx2.sh --inputs r_elegans/assets/autonomous_wiggle_v1
```

This is the same nine-case, 45-minute-cap diagnostic; it does not retrain.
For four independent fresh amplitude searches instead:
`sh scripts/launch_autonomous_wiggle_mbslnx2.sh` starts four four-core CPU workers,
each with a 60-minute cap and a target of 0.01 rad. They may finish much sooner
once the target is found and validated. Each worker validates its selection for
300 s, +/-1 mV initial states and a refined brain timestep. Exit 2 means the
full amplitude/validation criteria were not met (or execution failed); inspect
summary/logs to distinguish these. Raw histories, all accepted candidates, RNG
search metadata, source/environment and validation traces stay external.
The CLI does not implement exact mid-search resume. The agent launched no
remote job; the user launches the desired diagnostic or training run.

## First autonomous forward movement search (E031, 2026-11-08)

Start from the validated `autonomous_wiggle_v1` amplitude primitive; preserve it
as an independent fallback. All 302 neurons evolve with zero injected currents.
This first small movement stage fits the same 78 unknown non-LEAK RMD channel
densities; it does not silently broaden into whole-network or body-parameter
fitting. Connections, other neurons, capacitance, leak, channel kinetics,
reversals, calcium handling and NMJ/body mechanics remain fixed. Fit density
factors stay within 0.25–4 of this reference. No periodic stimulation, gait
teacher, timed command or sensory input is supplied. No RL yet.

Use 30-s trial simulations, scoring three separate blocks in the late half.
Forward is the mean tail-to-head body axis at the beginning of that late window,
held fixed for the measurement. Reward the minimum signed centroid speed across
those blocks, retaining >=0.01-rad sustained wiggle and <0.35-rad axis turning.
Distance traveled back and forth and startup posture relaxation do not satisfy
the movement criterion. Initial numerical goal: >=1e-4 body lengths/s in each
block, >=5x positive baseline speed and >=2% forward/path efficiency. These
are small engineering milestones, not animal-speed calibration.

Four independent seeds each receive a three-hour wall budget including reserved
validation time. The search uses that budget rather than stopping at the
amplitude target. After search, validate 300 simulated seconds, +/-1 mV initial
states, a finer neural timestep, a finer body timestep, and held voltage. All
active validation cases must pass the motion criterion; refined-timestep
speeds must also agree within 25% of the matched short-window speed (with a
target-speed tolerance floor). Held voltage must stop late motion. A candidate that improves short-horizon progress but fails these
checks remains experimental and is not promoted. If this subset cannot produce
progress, a later experiment can explicitly broaden the motor-circuit scope.

On mbslnx2:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_autonomous_movement_mbslnx2.sh
```

The launcher uses the dedicated r-elegans Python and `nohup`, prints a unique
log path, and assigns four disjoint physical CPU cores to each worker. Results
stay in `/home/carsonc/r-elegans-results/autonomous-movement-*`. Workers preserve
all candidate parameters/metrics, every accepted checkpoint and trajectory,
search RNG metadata, full source/environment and final validation traces.
`fitted_inputs/` contains the selected exact pair and checksum manifest for
further assays, including when the movement criterion is not met. Check
`summary.json` and its `passed` field before treating it as a successful skill.
Exit 2 means not all validation gates passed or an execution failure; inspect
worker logs/summary to distinguish those outcomes.

The three-hour cap covers each worker's search and validation; startup and
supervisor termination grace can add about two minutes. You may disconnect
after launch. The agent has prepared the run; no remote launch is assumed.

E031 server outcome (user-reported `autonomous-movement-20261108T154427901428Z`):
**0/4 seeds pass**, after 350/347/350/347 proposals (1,394 total). Each retains
0.017–0.031-rad sustained bends. All four long-validation minimum forward speeds
are negative (~-1.6e-6 to -3.2e-7 body lengths/s). Small positive speeds in some
perturbation/body-refined cases reverse sign in other checks; maximum reported
forward/path efficiency is only 0.00763, below the 0.02 criterion. Held voltage
stops all late motion. Exit 2 here reports unmet movement criteria; no missing
summary or execution exception is evidenced in this compact report.

This limited search did not convert the anterior oscillator into reliable
propulsion. It is not evidence that the full 302-neuron network is insufficient:
only 78 RMD channel densities changed, with all other neuron parameters and
connection strengths fixed. Neither an impossibility result for this subset nor the unique cause of
failure is established. Preserve the original amplitude reference and the failed
searches. Proposed next experiment: broaden unknown body-motor neuron densities
and/or existing synaptic strengths to develop distributed coordination; keep
known/fixed physiological properties and zero external drive. Do not merely
repeat the same restricted three-hour search or weaken its motion criteria.
No broader experiment has been launched. Full remote input/source/environment,
search history and raw trajectories remain unretrieved. Numerical evidence:
`experiments/records/E031-server-20261108.json`.

## Joint neuron-and-connection movement fitting (E032, 2026-11-08)

User requests training both neurons and connections after the restricted E031
search failed. Movement training now defaults to `--fit-scope neurons-connections`,
starting from the preserved `autonomous_wiggle_v1` parameter pair.

Eligible unknowns across the full 302-neuron network:

- 4,814 nonzero, non-LEAK channel-density scales across all 302 neurons.
- 3,709 strengths on existing directed chemical edges.
- 1,093 independent undirected electrical pairs.

Total: **9,616 independently eligible parameters**. LEAK, capacitance, gating
kinetics, ionic reversals/calcium handling, anatomy, release/reversal settings,
NMJ readout and body mechanics stay fixed. Missing channel terms and absent
edges stay unchanged; gap-junction conductance stays symmetric. All injected
currents remain zero; no sensory drive, timer, gait teacher or body forcing.
These are behavior-fitted effective parameters, not inferred biological values.

Bounded log multipliers act on existing densities/effective synaptic conductance
(0.25–4 of the initial value); chemical strength remains nonnegative. Search
uses cell-local blocks, coherent channel proposals, connection-only blocks and
mixed blocks, rather than perturbing all 9,616 values simultaneously. Each
parameter remains independent: shared proposal directions do not tie weights.
Eligibility does not mean every parameter changes on every iteration. Each run
saves the allowed masks, parameter-vector ordering, proposed/accepted coverage,
candidate vectors, RNG and selected/final checkpoints so actual changes can be
examined. Zero-offset application preserves the starting model exactly.

The E031 movement objective/criteria and validation matrix remain unchanged.
The prior restricted failure is retained; it did not test this broader scope.
Use `--fit-scope rmd` only for an explicit reproduction of the old restricted
experiment. The current movement launcher pins the broader scope:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_autonomous_movement_mbslnx2.sh
```

Four independent four-core CPU searches, **up to four hours including
validation**, under nohup. The launcher prints PID/log path; disconnect after
launch. Records remain under `/home/carsonc/r-elegans-results/autonomous-movement-*`.
Source, input hashes and scope are recorded per worker; `summary.json` reports
trainable count and proposal/update coverage, alongside movement validation.
Existing wiggle/model assets are not overwritten. No remote launch or movement
success is claimed until actual server results are reported.

E032 four-hour joint-search outcome (user-reported run
`autonomous-movement-20261108T205220165775Z`): **0/4 movement passes**,
1,886 proposals total. Cumulative accepted parameter updates per seed:
5,744 / 5,122 / 5,656 / 2,030. All long-validation minimum forward speeds
are negative (~-1.65e-6 to -1.14e-6 body lengths/s); bends persist at
0.026–0.031 rad. Best short-condition speed is only 6.91e-6, below the
1e-4 target, and maximum efficiency 0.01039 is below 0.02. Held voltage
stops motion. This broader search failed; topology sufficiency is not disproved.
Full raw histories/input/source/environment hashes remain unretrieved.

E033 local scoring audit demonstrates a training-objective vulnerability. A
bounded periodic centroid trace with exactly zero complete-cycle translation
gets minimum block speed +1.1155e-4 body lengths/s at 30 s, but -9.9046e-5
at 300 s. The efficiency/long validation gates correctly reject the trace.
Nevertheless, the search score itself rewards its short-window endpoint phase.
This is a synthetic metric example, not a worm hydrodynamics simulation or
proof of E032's failure mechanism. Next priority is phase-robust/cycle-averaged
progress scoring and spatial neural/muscle/body trajectory analysis before
repeating the long search. No code, parameters or thresholds changed in this
audit, and no new server run launched. Exact records: E032-server-20261108.json
and E033.json under `experiments/records/`.

## Trajectory analysis (E034, 2026-11-08)

The four-hour E032 cohort remains on mbslnx2; noninteractive SSH failed
authentication, so its actual trajectories have not been analyzed locally.
Preserved local E032 two-proposal pilot and E030 300-s reference traces were
analyzed without changing any model or objective.

Local E032 pilot: 99.04% of dynamic joint variance is in the three anterior
joints; one shape component explains 99.94%. Four joints exceed 10% of maximum
amplitude (with a 1e-4-rad floor); others bend weakly. Coherent offsets span
only ~1.57 ms in a 268-ms cycle, consistent with mostly synchronous anterior
bending rather than a distributed traveling wave. Spectral phase is descriptive,
not neural causality. Equal-reference-joint-phase centroid regression is
-4.05e-7 body lengths/s, changing to -1.13e-8 under body timestep refinement:
negligible, nonrobust progress. Median whole-shape closure residual between
phase samples is 3.47e-5 rad; one joint does not define a full gait phase.
The original E030 300-s reference is similar: 99.05% anterior variance,
99.93% first-component variance, -4.62e-7 body lengths/s phase drift. These
findings cannot be substituted for the four-hour candidates.

The read-only analyzer supports a cohort, a worker, or a chunked diagnostic.
It writes input hashes, numerical reports and phase samples; `--plots` adds
optional matplotlib figures. It never modifies inputs or trains anything.
Analyze the actual server cohort with:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
/home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.analyze_autonomous_trajectories \
  --run /home/carsonc/r-elegans-results/autonomous-movement-20261108T205220165775Z \
  --output-dir "/home/carsonc/r-elegans-results/trajectory-analysis-$(date -u +%Y%m%dT%H%M%SZ)" \
  --body-length 0.001
```

Body length is explicit in the input's position units (0.001 m for these liquid
assays), not inferred from displacement. The command prints compact per-case
findings and saves `summary.json`. Numeric analysis does not need matplotlib.
Use actual cohort findings to separate localized bending, phase progression,
cycle drift and endpoint effects before more training.
