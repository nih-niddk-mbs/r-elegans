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

## Current motor milestone (2026-11-05)

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
This is a frozen diagnostic, not completed remote validation or added RL.
If it fails, use the failed criterion to design the next supervised fit;
do not discard the established forward-motion checkpoint.

## Scientific objective

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
