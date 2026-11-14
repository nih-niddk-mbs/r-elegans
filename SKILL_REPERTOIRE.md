# r-elegans skill repertoire and action-space plan

2026-11-14 E076 diagnostic completed (user-reported):19.08min total, no timeout,
all steps/refinement complete. RIB/RIM depolarizing.1nS probe reduced ordinary
body range98.57%, but movement did not spontaneously recover. Neural/body-half
checks likewise found no body cycles in50–60s (probe removed40s), with raw ranges
1.40e-6/2.21e-6rad. Strict reversible-suppression gate failed; no learned-stop or
swimming qualification. This is strong imposed body-motion suppression, not proof of
neural oscillator silencing or a stable rest attractor. Sensory-integrator
excitation.1nS reduced motion85.42% with recovery on the ordinary grid only.
Next proposed diagnostic: explicit restart after suppression, matched no-restart
and longer observation; no new run launched. [Outcome](experiments/records/E076-server-summary.json).


2026-11-14 E075 saved-data audit completed (user-reported). Stop cue changes
ASEL by.640mV, but selected control responses are small: AIYL.00772mV,
AVBL.00365mV, RIS.000134mV; body motor maximum.000745mV. START changes AWA
by8.364mV. This supports cue access with weak downstream influence under these
conditions, not learned switching. Four actual workers made566 proposals,
accepted248, errors0; selected rest range dropped~10.07% from first reported
best, but E074 no-cue comparison shows no stopping.995/1318 selected trainables
changed; one near bound. No new skill or swimming qualification. Preserve parents.
Proposed next step is a short test of control access to the oscillator, followed
by cue-specific suppression training if feasible; no new experiment launched.
[Audit outcome](experiments/records/E075-server-summary.json).


2026-11-14 E074 outcome (user-reported):559.98min, no timeout, process exit0.
Selected seed7200/checkpoint2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f
failed all positive control cases. Nominal rest range.0352063rad versus1e-4rad
limit (~352×); no-cue.0352062644rad, with persistent~1.901Hz body rhythm.
No stop latency detected; movement2ms after START reflects ongoing movement,
not a learned restart. Full control case set reported complete, but control
qualification failed; pipeline validation remains incomplete because swimming
never started. Preserve E071 and failed candidates. No propulsion qualification.
Reported controller-group labels do not establish learned-control dependence
when unlesioned control already fails; reporting logic corrected prospectively.
Next proposed step: cue-to-motor signal-flow audit, not another blind long fit.
[Outcome](experiments/records/E074-server-summary.json).


2026-11-13 E074 planned follow-up: the controlled-swimming launcher now has a
**600-minute TOTAL ceiling** (setup, search, validation and cleanup). Control
search uses the remaining budget minus one full validation suite and10min
reporting margin, ending early on a nominal pass. Validate the best control
candidate even if it misses the nominal training gate. Full control validation
still gates any swimming. Swimming proceeds only if enough time remains for its
own validation. This is a fresh search from unchanged E071 seed6901 with seeds
7200–7203, not a resume of failed E072 optimizer states. Prior failed outcomes
remain preserved. No new server execution is evidenced yet.


2026-11-13 E072 server outcome (user-reported): four control fits exited0 after
18–19 proposals each; all remained in the stop stage and failed the nominal
control gate. Total77.6min; no watchdog timeout. Full control validation and E073
swimming training did not start. No learned stopping or swimming qualification.
The coordinator reserved later-phase time and ended after the control-search
allocation; unused budget was not reassigned. Best rest-motion values are not yet
reported; inspect saved metrics before choosing the next experiment. Preserve
E071 unchanged and all E072 candidates. [Outcome](experiments/records/E072-server-summary.json).


2026-11-13: H1.2 adds sensory initiation, stopping/rest and restart to every
future behavior qualification. E072/E073 were implemented as a planned pipeline; the first E072 server
attempt is now reported above. Learned control remains unverified. The E071 winner below remains
unchanged. The next run first learns and validates sensory stop/restart, then
trains physical forward swimming only after the exact selected checkpoint passes
its complete control gate. See [control policy](experiments/behavior_control_policy_v1.json).


E071 seed6901 PASSED the small autonomous water-model timing/amplitude rung:
~1.8727Hz body rhythm; all active validations have tail cycles>.003rad, with
minimum .00316012rad; held-voltage cycles0. Total job192.91min<240min. Preserve
checkpoint8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a
and its independent lineage. This does not qualify travelling-wave swimming or
propulsion. E072/E073 now target sensory control followed by gated propulsion.

E070 reported water-model rhythms meet the1.8Hz±10% timing rung in seeds6800,
6802 and6803, but every seed misses the .003rad tail-amplitude rung. No new full
skill qualification. Preserve all independent checkpoints;6802 is the proposed
follow-up candidate at1.8797Hz and91.5% tail amplitude. Long/timestep/control
results support persistent unforced wiggling, not travelling-wave swimming or
propulsion. Source hashes/full guards remain to be read before continuation.

E068 read-only server audit completed (user-reported stdout). It adds no trained
skill or E067 promotion. Hybrid ordinary regional cycles exceed old/old under
the same estimator; final-head-joint cancellation is strong, while weak tail
bends remain unexplained. Preserve all independent checkpoints. NMJ/filter
detail now shows responsive unsaturated muscle transfer, weak tail command
and small mechanical response. No new skill or training is claimed.

Strategy update: 2026-11-06, user-directed, recorded as H1.1 / E018 in
[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md). **Priority: cover useful behaviors with
individually validated skills; consolidate shared weights later.** Separate
full-302-neuron policies/checkpoints are acceptable. This document defines a
training backlog and future interface, not a new trained skill library.

The [biological catalog](BEHAVIOR_CATALOG.md) supplies stable behavior IDs,
primary references and required simulator capabilities. Its 62 current entries
are all mapped below. That inventory is expandable, not an exhaustive list of
every behavior in every sex, stage, strain and environment. Adult hermaphrodite
planar locomotion is the initial scope; male, dauer, organ and 3D programs
require explicit extensions.

## Meaning of coverage and skill status

User clarification (2026-11-10): the ultimate objective is one shared model
consistent with compatible neural, physiological and behavioral data, including
perturbations. H1.1 permits independent checkpoints during discovery; successful
separate policies do not establish that final cross-data model. Retain every
qualified parent while joint fitting and later consolidation develop. See
`experiments/data_targets_v1.json` for the initial data-readiness ledger.

User clarification after E046 (2026-11-10):both swimming in water and crawling
on a moist substrate are required. Initiation, pause and restart are required
alongside sustained movement. The experimental wet-agar reference is assessed
separately; MOT-16/17 retain their original liquid amplitude qualification.
No transfer to crawling or new skill is inferred from changing drag laws.
See [the substrate assessment](PHYSICS.md#wet-surface-reference-assessment-e046).

Coverage-first H1.1 still allows independent skill checkpoints during learning.
A later claim of one animal that starts, stops and adapts between media requires
continuous-state tests with one parameter set, sensory/internal state input and
no checkpoint swaps, state resets, voltage clamps or muscle-output overrides.
MOT-05/06 driven stop/restart evidence does not establish these capabilities in
the current autonomous conductance-network branch. Brief pauses, prolonged
quiescence and sleep remain different biological targets.

The goal is to map a useful, physically admissible repertoire under the model's
constraints. Coverage means documented effects over tested initial states,
parameters and media. It does not establish the complete mathematical action
space, all possible policies or biological equivalence. Waveform matching and
actual displacement remain separate measurements.

Status labels:

- **Driven evidence:** a full recurrent brain has produced the effect with
  external pacing/current inputs in normalized mechanics; evidence may be
  user-reported. This is not autonomous movement or calibrated crawling.
- **Autonomous amplitude evidence:**zero-drive recurrent channel network passes
  the specified muscle/bend assay;does not establish coordination or locomotion.
  Server reports remain labeled until raw artifact verification.
- **Candidate:** train/test separately; not demonstrated through the recurrent
  brain. A legacy analytic-gait demo does not change this status.
- **Composition:** later selection/sequencing/feedback over motor skills,
  rather than an additional independent muscle action.
- **Extension:** missing contact, organ, sensory, developmental or 3D capability
  prevents a credible biological assay today.

Within an individual skill, retain existing physical/numerical criteria and
record failures. A failed shared-policy retention gate does not invalidate a
previously saved successful standalone policy. Keep each good checkpoint and
its input contract, and continue exploring other skills. Consolidation and
interference mitigation are deferred; failure to move or invalid mechanics
still cannot count as a skill.

## A. Core motor primitives and parameter families

These are the initial lower-level skill candidates. Left/right and short/long
variants are parameter settings, not claims of additional discrete biological
categories. No numeric biological parameter ranges are assumed without data.

| Skill ID | Skill / controllable parameters | Observable acceptance target | Present status and biological links |
| --- | --- | --- | --- |
| MOT-01 | Held body bend; signed curvature, location, duration | Target curvature and stable hold without spurious translation | Driven evidence, E005; synthetic calibration primitive supporting MOV-01/02/03/06 |
| MOT-02 | Alternating body bends; amplitude, frequency, phase | Repeated coordinated bends through actual body mechanics | Driven evidence, E006; synthetic precursor to MOV-01/08 |
| MOT-03 | Forward run; duration and movement intensity | Positive centroid progress, correct traveling wave, straightness and numerical controls | Driven evidence; some checkpoints pass readiness, E012/E015; MOV-01 |
| MOT-04 | Backward run; duration and movement intensity | Negative centroid progress with opposite propagation and matching controls | Driven evidence; E014 seeds 2/3 passed all reverse pacing rates, E015 below progress floor; MOV-02 |
| MOT-05 | Pause/stop; requested duration | Low residual centroid path and body-angle speed after settling | Driven evidence in fixed-direction go/zero/go assays, E013–E015; MOV-07 |
| MOT-06 | Start/restart a run; direction, duration | Recover progress without resetting brain/body between stop and restart | Driven evidence within a held direction, E013–E015; MOV-07 |
| MOT-07 | Speed/intensity modulation; target level and ramp time | Measured displacement response versus input, including slowing/acceleration | Amplitude continuations show changes between fitted models; continuous speed control in one recurrent policy remains candidate; MOV-01/02/09, FOR-01/02 |
| MOT-08 | Shallow steering; signed turn angle/rate, duration | Controlled heading change during progress, with limited unwanted drift | Local +/-0.006 variants pass three rates/controls (E019); server supervisor reports cohort success (E020, truncated stdout); full seed tables/checkpoint verification and broader ranges pending; MOV-03 |
| MOT-09 | Head sweep/local head bend; extent, rate, hold | Head excursion relative to trunk with controlled body displacement | Local E021 passes; server E022 reports all four seeds and all three rates pass tuning/physical tests; raw remote artifact verification pending; driven stationary scan only; MOV-06 |
| MOT-10 | Forward/backward transition; requested direction and switching time | Reverse signed progress with a valid continuous-state transition | Candidate; separate fixed-direction assays do not establish switching; MOV-02/07 |
| MOT-11 | Omega-shaped deep turn; signed exit angle and completion time | High-curvature reorientation with valid head/body contact and exit heading | Extension: contact-qualified mechanics needed; MOV-04 |
| MOT-12 | Delta-shaped deep reorientation; geometry and exit angle | Reconstructed coiled path with valid geometry and contact | Extension: self-occlusion/contact and potentially out-of-plane mechanics; MOV-05 |
| MOT-13 | Crawling execution profile; load, amplitude, frequency | Reference-matched waveform, slip and speed on specified substrate | Extension/calibration: SI factories exist; current normalized driven runs are not calibrated crawling; MOV-01 |
| MOT-14 | Swimming execution profile; viscosity, amplitude, frequency | Reference-matched liquid waveform, propulsion and turning | Extension/calibration: SI liquid factory exists; active gait remains uncalibrated; MOV-08 |
| MOT-15 | Load/gait adaptation; medium/load context | Appropriate waveform and propulsion across tested mechanical loads | Candidate after medium-specific skill calibration; MOV-09 |
| MOT-16 | Autonomous posterior oscillatory bending;fixed fitted parameters | Tail median sustained bend>=0.003rad AND net-muscle excursion>=0.003,perturbation/refinement and quiet held-voltage checks | Autonomous amplitude evidence,user-reported E043 seed7001 passes all checks;selected parameters/assessment copy hash-verified locally;raw server traces retained separately;synthetic precursor to MOV-01/08,not locomotion |
| MOT-17 | Autonomous distributed oscillatory bending;fixed fitted parameters | Tail/middle/head regional median sustained bends>=0.003rad,all active checks,regional refinement and quiet held-voltage control | Autonomous amplitude evidence,user-reported E044 recheck passes on same7001 checkpoint;no spatial phase/propulsion criterion;synthetic prerequisite to MOV-01/08 |

MOT-01/02 are useful identification tasks, not additional claims of natural
behavior categories. MOT-13/14 are medium-conditioned variants of the movement
primitives, not proof that every behavior needs a different neural policy.
Signed steering is defined in the planar world frame; do not impose equal
dorsal/ventral performance or symmetric biological turn distributions.

MOT-08 standalone fitting is implemented as `train_steering.py`; local positive
and negative variants passed E019; E020 reports server cohort success, with
complete remote records still to retrieve. Priority for further coverage:
retain the successful MOT-09 stationary scans (E022), and independently fit
forward/backward intensity variants (MOT-07). Moving head scans remain a later
composition/coverage variant.
Do not first spend another run forcing all successful behaviors into one
parameter set. This order is a proposed backlog; no new fit is launched here.

## B. Movement sequences and search modes

These expand the repertoire of effects available to a later selector. First
validate constituent motions and transitions; then test sequence outcomes.
Their biological occurrence/selection statistics remain distinct from merely
executing a commanded pattern.

| Skill ID | Sequence/state | Motor dependencies and outcome | Status / catalog mapping |
| --- | --- | --- | --- |
| SEQ-01 | Reversal followed by reorientation and forward resumption; pirouette family | MOT-03/04/08/10, optionally MOT-11; net exit angle, reverse bout and resumed run | Composition; MOV-02/04/07, CHE-04 |
| SEQ-02 | Short versus long reversal bouts | Parameterized MOT-04 plus transition/restart; bout distance/duration | Candidate parameter coverage; MOV-02 |
| SEQ-03 | Scanning while moving or paused | MOT-05/08/09; scan envelope and coverage without losing body control | Composition; MOV-06/07 |
| SEQ-04 | Local / area-restricted search | Frequent reversals/reorientations; spatial confinement and trajectory statistics | Composition; FOR-03 |
| SEQ-05 | Global / dispersal search | Longer runs and changed reorientation statistics; area explored | Composition; FOR-04 |
| SEQ-06 | Roaming on food | Longer exploratory bouts under defined food/internal context | Composition; FOR-05 |
| SEQ-07 | Dwelling on food | Slow restricted movement, reversals and state persistence | Composition; FOR-06 |
| SEQ-08 | Patch departure and return | Skill sequencing across patch boundary; departure/return probabilities | Composition with food/environment model; FOR-07 |

Roaming, dwelling and search modes are persistent policy/state patterns.
They must not be presented to downstream RL as interchangeable instantaneous
muscle commands. Controllers may acquire them hierarchically after the motor
repertoire exists; manually prescribing their sequence is only a motor assay.

## C. Sensory-guided control and later tasks

These require the relevant environmental field, transduction and context.
They select/modulate motor skills. Training locomotion alone does not establish
correct sensation, avoidance or food finding. Food cues, biomass and ingestion
must remain separate model quantities.

| Skill/task ID | Behavior family | Required skill selection / assay | Status / catalog mapping |
| --- | --- | --- | --- |
| CTL-01 | Food-dependent slowing and resumption | MOT-07/05/06 across a food edge, with feeding-history controls | Composition + food/internal-state extension; FOR-01/02 |
| CTL-02 | Soluble/volatile attractant seeking | Run/steer/reorient using local chemical history and head sampling | Composition + calibrated sensory extension; CHE-01/02 |
| CTL-03 | Weathervaning / klinotaxis | Modulate MOT-08 using sampled transverse gradients | Composition; CHE-03 |
| CTL-04 | Pirouette / klinokinetic navigation | Modulate SEQ-01 probability using experienced chemical change | Composition; CHE-04 |
| CTL-05 | Repellent and osmotic escape | Select backward/forward escape and reorientation versus cue/site/intensity | Composition + sensory extension; CHE-05/06 |
| CTL-06 | Gentle-touch, nose-touch, harsh-touch and tap responses | Map calibrated stimulus position/intensity to appropriate withdrawal/escape | Composition + distributed mechanical sensing/contact; SEN-01/02/03/04/05 |
| CTL-07 | Temperature-dependent navigation/tracking | Steer and select runs conditional on thermal field and cultivation history | Composition + temperature/history model; SEN-06/07 |
| CTL-08 | Oxygen/CO2 navigation | State/context-dependent movement under defined gas fields | Composition + gas/internal-state model; SEN-08/09 |
| CTL-09 | Light avoidance | Calibrated light input to escape choices | Composition + phototransduction; SEN-10 |
| CTL-10 | Humidity orientation / electrotaxis assays | Movement under controlled modality-specific fields; conditioned context where required | Composition + additional stimulus physics; SEN-11/12 |
| CTL-11 | Dietary choice, sampling and pathogen avoidance | Patch choice, ingestion and history-dependent reorientation/leaving | Composition + food/organ/learning models; FOR-08, CHE-08 |
| CTL-12 | Predator-cue response | Escape and coordinated changes in other physiological programs | Composition + sensory/internal/organ models; CHE-07 |

The engineered food-finding demo is a separate surrogate baseline. None of
these rows is marked learned by the full recurrent motor actor on that basis.

## D. Physiological, learning, social and anatomical extensions

Preserve these in the overall animal repertoire while acknowledging missing
effectors or mechanisms. They are not unlocked by adding more body-wall-muscle
training. Some involve intrinsic organ dynamics or learning assays, rather
than an additional low-level action for a fixed adult hermaphrodite body.

| Extension ID | Capability / behavior | Necessary extension and target | Catalog mapping |
| --- | --- | --- | --- |
| EXT-01 | Pharyngeal pumping | Pharynx mechanics/control; pump timing and ingestion | ORG-01 |
| EXT-02 | Isthmus peristalsis | Pharyngeal compartment dynamics; coordinated transport | ORG-02 |
| EXT-03 | Spitting / feeding rejection | Pharyngeal switching and fluid/particle handling | ORG-03 |
| EXT-04 | Defecation motor program | Gut/enteric effectors and physiological clock; ordered expulsion sequence | ORG-04 |
| EXT-05 | Egg laying | Reproductive anatomy, vulval/uterine effectors and context; event/bout statistics | ORG-05 |
| EXT-06 | Coordination of feeding, gut, reproduction and movement | Coupled organ/internal-state model; cross-program timing | ORG-06 |
| EXT-07 | Satiety, sleep-like quiescence and sleep homeostasis | Internal state, arousability, stress/development and interruption/recovery assays | FOR-09, STA-01/02/03 |
| EXT-08 | Adaptation, habituation and associative learning | History/plasticity mechanisms with paired/unpaired and fatigue controls | LEA-01/02/03/04/05 |
| EXT-09 | Burrowing | 3D body, contact and substrate mechanics; progress through gel | MOV-10 |
| EXT-10 | Torsional rolls / 3D reorientation | Body torsion and out-of-plane mechanics; orientation changes | MOV-11 |
| EXT-11 | Dauer nictation / carrier transfer | Dauer-specific anatomy, substrate attachment and carrier interaction | STA-04 |
| EXT-12 | Aggregation/bordering and pheromone responses | Multiple animals, food/gas/chemical context and interaction | SOC-01/02 |
| EXT-13 | Male mate-cue search | Male-specific anatomy/physiology and social chemical sensing | SEX-01 |
| EXT-14 | Male contact response/backing and mate-end turn | Male tail/contact effectors and partner geometry | SEX-02/03 |
| EXT-15 | Vulva location, insertion and sperm transfer | Male reproductive effectors and partner interaction | SEX-04/05 |

All extension rows are unimplemented or incomplete for the full behavioral
assay. A stopping motor primitive is not evidence of sleep or satiety, and a
planar self-crossing drawing is not a validated deep turn.

## Future skill interface and coverage ledger

Proposed action representation:

```text
action = (skill_id, parameters, requested_duration)
```

Parameters are skill-specific: e.g. intensity, signed steering angle, scan
extent, or bout duration. Preserve this as a structured, potentially continuous
space. Sampling ranges must be fitted/validated and tagged with medium/units;
the current amplitude parameter is a muscle target modulation, not measured
animal speed. No option API, latent action model or high-level selector is
implemented by this document.

Each separately learned skill must eventually have a registry record with:

- Biological IDs, version, evidence source and checkpoint/source hashes.
- Full model/topology, effectors, unit system and environment assumptions.
- Input contract, including external pacing/direction context; later mark
  internally generated rhythm separately from autonomous task selection.
- Valid initial states, parameter support, requested/actual duration,
  termination/completion conditions and failure conditions.
- Actual body effects: signed displacement, turn angle, posture, stopping
  residuals, numerical controls and variation across seeds/conditions.
- Separate statuses for waveform fitting, physical validity/readiness,
  biological agreement and composability. Report coverage per skill/parameter
  range rather than treating one jointly successful controller as the metric.
- Original and selected checkpoints, independent validation and external
  artifact/archive locations. Keep good standalone versions even if a later
  mixed-skill fit forgets them.

Switching between different weight sets can invalidate a recurrent hidden
state. Separate checkpoints establish candidate skill coverage, not immediate
safe runtime composition. Define/reset/map state and validate transitions
before claiming a working high-level skill selector. Consolidation into one
conditioned network is a later optional experiment.

## Evidence connections

The full biological/source map remains in [BEHAVIOR_CATALOG.md](BEHAVIOR_CATALOG.md).
In particular, [Broekmans et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5030097/)
separates omega and delta reorientation; [Iino and Yoshida (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6665864/)
distinguishes continuous steering and pirouette-based chemotaxis. These support
separate repertoire/selection entries rather than labeling all navigation a
single movement. Developmental, organ and sex-specific rows inherit their
explicit references and capability requirements from the biological catalog.

This catalog is a versioned hypothesis about a useful policy decomposition.
It is not a claim that the animal uses our option boundaries, random direction
context, parameterization or learning algorithm. Coverage and downstream RL
benefit must be measured.

### E030 — Intrinsic anterior wiggle amplitude (2026-11-08)

Independent channel-model reference `autonomous_wiggle_v1`: all 302 injected
compensation currents zero; fit 78 unknown non-LEAK RMD channel densities,
keeping connections and physiology/muscle/body constants fixed. Local 300-s
liquid assay retains three rhythmic RMD cells and ~0.03-rad peak-to-peak bends,
mainly anterior. Initial-state perturbations and neural timestep refinement
retain the short-window amplitude; held-voltage control stops motion. This is
a self-generated amplitude primitive, not whole-body wave coordination, useful
locomotion, or a consolidated autonomous behavior policy. Exact pair is bundled;
raw records external. Original driven skills and Mac reference remain intact.

E030 server update: user-reported 600-s liquid amplitude 0.02950885 rad and
three rhythmic cells; perturbation/body refinement checks retain it and both
controls stop it. E031 autonomous forward movement is the next planned stage,
using the same limited unknown-density scope and a three-hour CPU budget.
Amplitude qualification does not imply successful E031 motion.

E031 server outcome: 0/4 seeds pass forward-motion validation after 1,394
proposals. Sustained anterior wiggle retained, but drift is weak and changes
sign across checks. No autonomous forward skill added. Full-network sufficiency
remains untested because only the RMD density subset was trainable. Preserve
`autonomous_wiggle_v1`; broader motor-circuit coordination is proposed next.

E032 planned: broaden forward-movement fitting to all eligible neuron densities
and existing chemical/electrical strengths (9,616 independent unknowns),
retaining the autonomous wiggle source, zero injected drive, fixed anatomy and
physiological/body constants. No new forward-motion skill is yet validated.

E032 joint-search result: 0/4 movement passes after 1,886 proposals, with
thousands of accepted parameter updates and persistent bends. No forward
skill added. E033 synthetic audit identifies phase-sensitive short-window
selection as a possible search failure route; actual trajectories remain
unretrieved. Preserve the autonomous amplitude reference and audit scoring
before repeating compute.

E034 actual E032 cohort analysis received: ~99% anterior variance and
~99.9% one shape pattern across all four long cases, only small phase offsets
and negligible/negative cycle-matched drift. No forward skill added. Next
proposed primitive is autonomous body-wide bend coverage and phase progression
before forward-speed optimization; no external drive or teacher is needed
for the proposed behavioral objective. Implementation/results remain pending.

E035 implemented, not yet qualified: autonomous regional bend recruitment
followed by phase-coordinated body wave, with no external drive or waveform
teacher. All neuron/connection unknowns eligible; original wiggle preserved.
Only a validated wave checkpoint may start E036 complete-cycle movement
optimization. No new skill claimed from the short software/pipeline pilot.

E035 server result (2026-11-09): 0/4 full wave passes. Seed 2 middle amplitude
is near its target, posterior amplitude remains ~38x too small. Seed 3 offers
more shape/phase diversity but also lacks posterior amplitude. Preserve both
as unqualified recruitment candidates; no new body-wave or propulsion skill
added, and E036 remains gated. Do not confuse log coverage with physical
amplitude completion.

E037 current curriculum: distributed amplitude first, control later. Independent
per-joint regional medians allow standing, opposing and asynchronous sustained
bends. Continue learned E035 seeds2/3 using original anchors/bounds and fresh
RNG forks. No phase, heading or propulsion criteria. A future amplitude pass
is not automatically a wave/control skill; none is yet claimed.

E037 server cohort (2026-11-09):0/4 distributed-amplitude qualifications.
Anterior reaches target4/4 and middle3/4; posterior0/4. Best tail remains
~106x below target. Preserve candidates, especially1001; no new body-wide
amplitude, autonomous wave or locomotion skill added. Audit posterior activity
before more training; phase/control remain deferred.

E038 server audit reported (2026-11-09): posterior neural voltage/release
oscillations are weak, and posterior median individual-muscle sustained
excursion is zero in all four workers. Deficit exists before the body-mechanics
stage. Working hypothesis: insufficient posterior neural recruitment; channel,
recurrent and NMJ causes remain unresolved. Next proposed curriculum targets
posterior recruitment/amplitude with zero external drive, preserving independent
head/middle candidates. No new training or skill qualification. Full parsed
reports: `experiments/records/E038-server-20261109.json`.

E039 planned: independent short-assay tail amplitude primitive. Original parents
retained. Tail median sustained joint p2p>=0.003rad, with perturbation, timestep
and held-voltage checks. No phase/control or propulsion requirement.20-minute
server trial prepared; no new qualification until actual results are reported.

E039 server result reported (2026-11-09):0/4 short tail qualifications,151
proposals total. Best fitting tail amplitude gain2.63% (seed2000); others~0.3%.
Best long-validation tail median2.905e-5rad remains103x below0.003rad target;
all active checks unqualified, held-voltage excursions zero. Existing head and
middle amplitude remain above target in these checks. No new skill qualified.
Next proposed diagnostic: posterior-cell autonomous oscillation feasibility
within current bounds before extending unchanged fitting. No new run launched.
Exact parsed results: `experiments/records/E039-server-20261109.json`.

E040 local diagnostic: no new skill. Isolated channel profiles settle rather
than oscillate across the finite tested bounds/basins;weak connected rhythms
vanish under the direct-NMJ subcircuit ablation. These are model diagnostics,
not evidence against biological oscillators or all possible model parameters.
Working model unchanged;actual server-parent confirmation prepared.

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

E041 planned independent posterior motor/muscle/bend primitive:net-muscle and
physical tail median sustained excursions must both reach0.003,with perturbation,
refinement and held-voltage checks. No phase/propulsion gate.30-minute server
trial prepared from E039 parents;no new qualification claimed.

E041 server result reported (2026-11-09):0/4 qualifications after295 proposals,
but seed5003 shows strong subthreshold gains:tail bend7.17x to0.0001874rad,
net-muscle excursion32.68x to0.0004799. Gains persist under perturbations and
finer brain/body steps (brain-refined bend+0.008%,muscle+1.54%). Remaining
shortfall16x bend and6.25x muscle. Seed5002 offers a smaller stable gain;5000/5001
improve muscle activity without improving bend. Prefer independent continuation
from5003,with5002 backup;preserve original anchors/bounds and zero input. No new
run launched or skill qualified. Exact record:`experiments/records/E041-server-20261109.json`.

E042 planned180-minute continuation:3 independent forks of E041 seed5003 plus
1 of5002;no new skill yet. Preserve original anchors/bounds;extend long
validation to300s. Same muscle AND bend amplitude qualification;phase/control/
propulsion deferred. No server execution claimed.

E042 server continuation reported (2026-11-09):2008 proposals,0/4 full passes,
but best6002 increases tail bending7.28x to0.00136388rad and net-muscle activity
7.34x to0.00352342. Net-muscle amplitude exceeds0.003 in all five active checks;
tail bending remains~2.2x below0.003rad target. Long validation0.00139634rad/
0.00361156 muscle;brain refinement changes bend-1.84%,muscle-1.49%. All four
workers improve,held-voltage scored excursions0. Prefer future continuation from
6002,preserve6003 alternate lineage/other backups. Same objective,original bounds,
zero drive;no full skill qualified and no new run launched. Exact record:
`experiments/records/E042-server-20261109.json`.

E043 planned8-hour continuation:three independent forks of6002 plus one6003,
original anchors/bounds retained,no new skill yet. Muscle component already
reached in6002;physical bend target still required. Same muscle AND bend
qualification and numerical controls;phase/control/propulsion deferred.

E043 duration amended (2026-11-10):latest user instruction is3hours/180min,
superseding the8-hour preparation and intervening4-hour request. Same launcher,
6002x3+6003x1 parents,seeds7000–7003,20s fit/300s long validation,original bounds
and zero drive. Eight-hour source preserved in commit`30ff375`. No E043 server
launch confirmed;editing launcher does not change an existing process.

E043 result(2026-11-10):MOT-16 is the first reported qualified autonomous
posterior amplitude primitive,seed7001. All five active checks pass;held voltage
quiet. Preserve independent winner;whole-body amplitude,phase/heading/propulsion
remain unassessed. Source path and missing raw verification recorded in
`experiments/records/E043-server-20261110.json`. No new run launched.

E044 next assessment(read-only):inspect MOT-16 winner7001 head/middle retention,
full-body amplitude/refinement consistency and saved motion preview. Tail pass
remains preserved;no new whole-body,coordination or locomotion skill is claimed
until actual server evidence. No training launched.

E044 server launch reported(2026-11-10):preflight validates MOT-16 winner7001
SHA256`417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14`.
Actual source files remain server-only;full-body assessment completion/results
pending. See`experiments/records/E044-server-launch-20261110.json`.

E044 completed by server report(2026-11-10):MOT-17 distributed amplitude
qualifies on preserved7001;minimum regional medians[0.00303260,0.03331643,
0.01103545]rad. Near-single-mode shape(~99.9% PC1) and tiny reported drift do
not qualify a traveling wave or locomotion. Inspect actual preview/full report
before proposing phase coordination;no new training launched.

E044 local artifact verification(2026-11-10):user copied the actual server bundle
to external Mac OneDrive. All256 artifact hashes match;selected network.npz
independently matches417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14.
Fixed fields equal original anchor,all bias currents zero,frozen parameter
entries unchanged,and frozen readout hash82f418ac...8247dcb0 matches parent.
Selected parameters/reference and assessment are now locally preserved;raw300s
neural/body arrays and full recursive server history are not included.

Actual plot/preview frames inspected:~2.56Hz (~0.3905s period),middle/head mostly
synchronous;posterior joint2 is nearly opposite phase with an abrupt transition.
Low-amplitude end joints omitted from spectral display. PC1~99.9%,PC2~0.1%,
shape cycle nearly retraces one mode. Stable tiny negative same-phase drift
is not useful locomotion qualification. Next proposed step:small smooth spatial
phase progression while retaining existing regional amplitudes/zero drive.
Winner7001 unchanged;no new training. Exact verification:
`experiments/records/E044-local-artifact-review-20261110.json`.

### E048 — Sensory connections as infrastructure (2026-11-10)

The opt-in channel-brain branch now receives local chemical and actual bending
signals at identified neurons. Paired block controls and continuous-memory
checks distinguish the connection from mere anatomical cell presence. This is
infrastructure, not a new behavioral skill or retrospective promotion of the
older normalized chemotaxis demo. Input gains, kinetics and receptive fields
remain experimental. MOT-16/17 retain their original no-input liquid qualification;
MOT-05/06 physiological stop/restart and MOT-15 environment adaptation still
require new evidence in this branch. No new skill checkpoint is registered.

## Sensory–motor foundation curriculum (E055, 2026-11-11)

Status: complete compact server result reported; first-rung qualification failed
on amplitude. Active response reaches47.7% of target, weakest active validation
41.1%, with adequate reported recovery and passing chemical/voltage controls
and engineering refinement. E056 continuation on the same rung is planned.
See [stage definitions](experiments/sensorimotor_curriculum_v1.json). Keep the
full302-neuron network during this curriculum. Sensory transduction gains and
unknown neural/connection strengths can adapt; anatomy, response signs, receptor
kinetics, muscle readout and mechanics remain fixed.

1. Reversible sensory-evoked bend: finite chemical cue, matched sham, recovery.
2. Alternating bends: actual-body feedback, no periodic cue pacing.
3. Propagation: adjacent-region recruitment and distributed persistent motion.
4. Translation: useful displacement in water and on a wet substrate.
5. Regulation: continuous-state initiation, stopping, reversing and turning.

Stages2–5 are plans, not implemented trainers or qualified skills. Preserve each
qualified independent checkpoint and its sensory configuration before promotion.
A sensory-evoked bend does not count as spontaneous oscillation or a validated
biological diacetyl reflex. Stage-specific numerical/physical criteria remain
required; shared-weight consolidation remains deferred under H1.1.

E055 complete server update:48proposals increased active cue-specific bend51.1%;
all active validation cases remain below the3e-5rad engineering target. Retain
contract_relax status unqualified and preserve checkpoint437b227e.... E056 plans
60minutes of completed-search continuation, including saved sensory gains/RNG;
no promotion to alternation. See experiments/records/E055-server-summary.json.

E056 continuation tail:refined response reaches93.4% of the first-rung target
and shifted-onset94.4%, with controls/recovery passing. Still unqualified; do not
promote. Chemical gain has reached2x bound. Active/dose results and full summary
are needed before another fit. Evidence: E056-server-partial-20261111.json.

E056 full report supersedes missing-field status:148additional proposals,
active96.3%, low-dose83.4%, high-dose107.1% of target. First rung unqualified.
E057 plans robust nominal/.8dose training from this checkpoint, with fresh RNG
and unchanged bounds/target. Dose-low ceases to be an unseen cue; all original
validation gates remain. No promotion to alternation or new biological claim.

E057 server update:nominal/high-dose/onset/nominal-refined gates pass, low-dose
97.96% of target remains unqualified. Retain1d864fdb...; E058 same-objective
continuation planned with RNG after99 and selected model98. Do not promote to
alternation or infer autonomous/biological skill. Low-dose fine-timestep behavior
is not separately established by the nominal refinement case.

E058: **first-rung engineering pass**, user-reported; preserve selected model
6cd7b0fa95c545c354f83334fe32aaf0551b8dd4e3d4e7f98babc944849669bf and its
sensory gains under sensorimotor-20261111T193810Z-773573. All original required
active variants/recovery/control gates pass. Capability is a tiny reversible
sensory-evoked bend; no autonomous rhythm, propagation, propulsion or biological
reflex fit. No shared-skill consolidation claim. Further promotion pending E059
read-only low-dose timestep/margin assay; original E058 pass remains recorded.

E059 outcome:the original first-rung engineering pass is preserved, but low-dose
half/quarter refinement fails. Do not promote to alternation. E060 read-only
step-family isolation planned before further training; no certified converged
amplitude or biological validation established. See E059-server-summary.json.

E060 server step isolation:brain-only refinement changes peak7.93–16.48%,
body/sensory/coupling under.02% for tested steps. First rung remains promotion-
deferred. E061 read-only float64 neural sweep against E060 reference planned;
no new learned skill or numerical fix yet. Original E058 pass retained.

E061:float64 neural-step amplitude nearly invariant (0.00116% range) but only
95.22% of target. No numerical skill promotion yet. At that entry E062 planned float64 robust
fitting with original validation threshold,2% training-only margin and explicit
low-dose refinement checks, preserving all previous checkpoints independently.
No autonomous rhythm or biological reflex match claimed.

## E062 sensory skill qualification and E063 alternation pilot — 2026-11-11

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
The subsequent eight-hour server run failed; see the dated outcome below.

## E063 server outcome — 2026-11-12

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

## E064 diagnostic readiness — 2026-11-12

E064 read-only signal-chain diagnostic is implemented and locally verified
(2026-11-12). It reads saved E063 baseline/best and all validation active/sham
signals, counts raw extrema across every channel, and writes per-channel tables
and raw-signal figures externally. Inputs are hash-checked before and after;
no simulation, fitting or JAX/GPU initialization. The local fixture is an older
checkpoint, not the eight-hour server model. Actual server analysis remains
pending the user launch. No next training objective or job has been started.
See [diagnostic record](experiments/records/E064-signal-chain-diagnostic.json).

## E064 server diagnosis — 2026-11-12

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

## E065 controlled rhythm comparison readiness — 2026-11-12

E065 controlled rhythm comparison is ready for the user to launch on mbslnx2
(2026-11-12). Four full-network combinations separate old/new channel scales
from old/new chemical+electrical strengths. Fixed fields, leak, anatomy and
muscle readout must match exactly. All primary cases share the current E063
sensory gains, initialization and wet-substrate proxy.15 independent60s
simulations include cue/sham, initial-state, precision, feedback and neural-step
controls, with four concurrent workers on disjoint four-physical-core groups.
No fitting or periodic drive. Local15-case12s software smoke and279-test suite
passed; this used an older local fitted model, not actual server3fabef... .
Server comparison remains unexecuted here. E062 recovery skill remains separate.
See [E065 method and verification](experiments/records/E065-rhythm-comparison.json).

## E065 partial server controls — 2026-11-12

E065 server footer reports successful execution and common float64 initial
states for rhythm-comparison-20261112T185742Z-865067. Three complete control
rows were supplied: old/old neural-half and initial-voltage +/-1mV each retain
54 repeating voltage channels,3 calcium,92 muscles and11 command/drive/joint
channels under the diagnostic floors. This supports the earlier oscillatory
parent as a starting point under tested controls; it does not identify the
parameter family suppressing current-model rhythms. Primary hybrid rows and
precision/feedback results remain unreported in this excerpt. Request the
complete compact table from the saved summary; do not launch another run.
Evidence is user-reported partial stdout; no artifacts independently retrieved.
See [partial result](experiments/records/E065-server-partial-summary.json).

## E065 full family comparison — 2026-11-12

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

## E066 hybrid validation readiness — 2026-11-12

E066 independent hybrid validation is ready for user launch (2026-11-12).
The old-channel/new-connection candidate is pinned by its saved hash and exact
family reconstruction. Eight simulations test its own initial-voltage,
neural/body timestep, held-voltage and300s unpaced persistence controls, plus
regional amplitudes and raw-supported phase/PCA descriptions. Four workers
run concurrently. No fitting or automatic model promotion. Local8-case12s
software smoke and282 tests passed; the older local fixture fails the stricter
tail persistence gate, retained as a negative result. Actual server60/300s
candidate validation remains pending. See [method and local checks](experiments/records/E066-hybrid-validation.json).

## E066 partial server amplitude result — 2026-11-12

E066 server excerpt (2026-11-12) reports execution valid but rhythm/regional
validation failed. Candidate hash f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e.
Both initial-voltage perturbation rows retain37 repeating voltage,90 muscle and
11 joint channels. Direct tail/middle/head cyclic excursions are about
4.28e-5/2.66e-4/2.89e-4rad, well below the .003rad engineering comparison.
Zero historical window-gated medians do not mean zero physical motion. Neural
refinement checks pass; gated regional comparisons can compare zeros with
zeros and do not establish convergence of nonzero cycle amplitudes. PC1=1 over
eligible joints alone does not establish whole-body synchrony or propagation.
The long-run/baseline/refinement rows remain unreported in this excerpt.
Read the complete existing compact report before fitting; no new run launched.
See [partial server evidence](experiments/records/E066-server-partial-summary.json).

## E066 complete compact result — 2026-11-12

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

## E067 connections-only amplitude stage — 2026-11-12

E067 is ready for a user-launched four-hour connections-only amplitude search.
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

## E067 completed amplitude search — 2026-11-12

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

2026-11-14: E075 saved-data sensory-to-motor audit completed in user-reported stdout;
no new skill qualification. Adds no trained or qualified skill; all E074 control failures remain.

E076 planned diagnostic follow-up (2026-11-14):
`sh scripts/launch_control_access_mbslnx2.sh` screens additive conductance probes
in four candidate interneuron groups at.01/.1nS and0/−80mV reversal. Sham plus16
probes;60s per case, input30–40s, recovery measured50–60s. Strongest safe matched-
baseline candidate gets matched-sham neural/body half-step tests. All parent
weights and mechanics fixed; sensing, all neurons and NMJs continue evolving.
No sensory-control, learned-stop, swimming or biological pass is conferred by
imposed input. Reports distinguish body-motion suppression from neural activity.
45min maximum TOTAL under nohup, four CPU workers; source/reference checks and
all outputs external. No new training launches automatically. See E076-plan.json
in experiments/records; reported outcome in experiments/records/E076-server-summary.json.
