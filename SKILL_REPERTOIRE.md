# r-elegans skill repertoire and action-space plan

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

The goal is to map a useful, physically admissible repertoire under the model's
constraints. Coverage means documented effects over tested initial states,
parameters and media. It does not establish the complete mathematical action
space, all possible policies or biological equivalence. Waveform matching and
actual displacement remain separate measurements.

Status labels:

- **Driven evidence:** a full recurrent brain has produced the effect with
  external pacing/current inputs in normalized mechanics; evidence may be
  user-reported. This is not autonomous movement or calibrated crawling.
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
| MOT-08 | Shallow steering; signed turn angle/rate, duration | Controlled heading change during progress, with limited unwanted drift | Candidate; legacy steering is not full-network evidence; MOV-03 |
| MOT-09 | Head sweep/local head bend; extent, rate, hold | Head excursion relative to trunk with controlled body displacement | Candidate; MOV-06 |
| MOT-10 | Forward/backward transition; requested direction and switching time | Reverse signed progress with a valid continuous-state transition | Candidate; separate fixed-direction assays do not establish switching; MOV-02/07 |
| MOT-11 | Omega-shaped deep turn; signed exit angle and completion time | High-curvature reorientation with valid head/body contact and exit heading | Extension: contact-qualified mechanics needed; MOV-04 |
| MOT-12 | Delta-shaped deep reorientation; geometry and exit angle | Reconstructed coiled path with valid geometry and contact | Extension: self-occlusion/contact and potentially out-of-plane mechanics; MOV-05 |
| MOT-13 | Crawling execution profile; load, amplitude, frequency | Reference-matched waveform, slip and speed on specified substrate | Extension/calibration: SI factories exist; current normalized driven runs are not calibrated crawling; MOV-01 |
| MOT-14 | Swimming execution profile; viscosity, amplitude, frequency | Reference-matched liquid waveform, propulsion and turning | Extension/calibration: SI liquid factory exists; active gait remains uncalibrated; MOV-08 |
| MOT-15 | Load/gait adaptation; medium/load context | Appropriate waveform and propulsion across tested mechanical loads | Candidate after medium-specific skill calibration; MOV-09 |

MOT-01/02 are useful identification tasks, not additional claims of natural
behavior categories. MOT-13/14 are medium-conditioned variants of the movement
primitives, not proof that every behavior needs a different neural policy.
Signed steering is defined in the planar world frame; do not impose equal
dorsal/ventral performance or symmetric biological turn distributions.

Priority for new coverage: independently fit MOT-08 shallow steering and
MOT-09 head sweeps, alongside independent forward/backward intensity variants.
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
