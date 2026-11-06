# C. elegans behavior catalog for r-elegans

Research and design baseline: 2026-11-03. This is a broad, expandable inventory
of documented behavior families and assay variants, not a claim that a finite
list captures every natural behavior. Proposed training recipes live in
[TRAINING_PROTOCOLS.md](TRAINING_PROTOCOLS.md). Neither document adds executable
environments or trained behaviors.

Operational update (2026-11-06): [SKILL_REPERTOIRE.md](SKILL_REPERTOIRE.md) maps
every current behavior ID to a motor skill, composition or explicit model
extension, with status and observable targets. The strategy is coverage first,
using independent policies when needed; shared-weight consolidation is deferred.

## Scope and interpretation

The first reference animal should be an age-matched adult N2 hermaphrodite in a
specified feeding condition on a specified agar substrate. Wild isolates,
males, larvae, and dauers require separate reference cohorts. A laboratory
assay can reveal a natural capacity without reproducing its ecological setting;
electrotaxis is included as an assay, not assumed to be an everyday ecological
activity. N2 alone cannot define the species' entire repertoire.

Entries distinguish motor primitives, stimulus responses, persistent states,
learning, and organ programs. They overlap: a reversal can participate in
search, escape, and chemotaxis. Training an isolated primitive does not establish
that it is selected appropriately in those contexts.

### Simulator capability keys

| Key | Required capability | Current repository status |
| --- | --- | --- |
| M | Physical crawling/swimming with muscle control | Coupled planar RFT and normalized driven motion exist; SI active gaits still need experimental fitting |
| R | Reverse, pause, shallow turn, deep reorientation | Driven full-network forward/reverse/stop recorded; switching and shallow turns unvalidated; contact-qualified deep turns absent |
| C | Local chemical transduction and distinct modalities | Gaussian cue and engineered memory exist; calibrated receptor/neuron responses absent |
| K | Mechanical contact and distributed touch | Absent; center clipping at a dish margin is not contact mechanics |
| T | Temperature field and thermosensation | Absent |
| G | Oxygen and carbon-dioxide fields/transduction | Absent |
| L | Light stimulus and phototransduction | Absent |
| I | Food ingestion, energy, satiety, arousal, neuromodulation | Absent as physiological state; neural recurrence alone does not supply these variables |
| P | Experience-dependent adaptation/plasticity | Sensory adaptation and recurrent carry exist; no validated model of behavioral learning |
| O | Pharynx, gut, enteric/vulval muscles, reproduction | Not represented by the 95 body-wall-muscle model |
| 3 | Three-dimensional mechanics, torsion, surface attachment | Absent |
| S | Multiple worms and their contact/chemical/gas coupling | Absent |
| D | Stage-specific anatomy, physiology, and developmental clocks | Absent; adult hermaphrodite topology is the baseline |
| X | Additional stimulus physics, e.g. humidity or electric field | Absent |

“Requires” below lists capabilities needed for a credible test, not a statement
that they are all implemented. Stable IDs are intended for future task manifests.
Every row has an assay/target and a corresponding proposed protocol family.

## 1. Movement and posture

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| MOV-01 | Forward crawling | Uniform agar; centerline curvature waves, speed, amplitude, wavelength, frequency, slip | M | P01 / [gait mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC2996704/) |
| MOV-02 | Backward crawling/reversal | Spontaneous and evoked bouts; reverse wave propagation, duration, distance, forward restart | M,R | P01 / [navigation circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC546636/) |
| MOV-03 | Shallow turning | Open arena; curvature and heading changes during continuing locomotion | M,R | P01 / [shallow turns](https://pubmed.ncbi.nlm.nih.gov/21490263/) |
| MOV-04 | Omega turn | Large reorientation; curvature pulse, head-body proximity, exit angle | M,R,K | P01 / [coiled postures](https://pmc.ncbi.nlm.nih.gov/articles/PMC5030097/) |
| MOV-05 | Delta turn | Deep, self-occluding reorientation; reconstructed centerline and exit direction | M,R,K,3 | P01 / [coiled postures](https://pmc.ncbi.nlm.nih.gov/articles/PMC5030097/) |
| MOV-06 | Head sweeps and exploratory bending | Head motion relative to body during movement/search; amplitude and phase distributions | M,R | P01 / [navigation circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC546636/) |
| MOV-07 | Brief pauses and locomotor transitions | Unstimulated recordings; pause durations and transition probabilities, distinct from sleep | M,R | P01 / [behavioral time series](https://pmc.ncbi.nlm.nih.gov/articles/PMC3962822/) |
| MOV-08 | Swimming | Defined Newtonian liquid; waveform, speed, beat frequency and turning distribution | M,R | P01 / [gait mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC2996704/) |
| MOV-09 | Gait adaptation to load | Viscosity series and medium changes; waveform and frequency as functions of mechanical resistance | M,I | P01 / [gait mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC2996704/) |
| MOV-10 | Burrowing | Three-dimensional gel; progress, wave shape and turning versus gel properties | M,K,3 | P12 / [burrowing assay](https://pmc.ncbi.nlm.nih.gov/articles/PMC4444045/) |
| MOV-11 | Torsional rolls/reorientation in 3D | Volumetric trajectories; undulation-plane rotation and orientation changes | M,3 | P12 / [roll maneuvers](https://pmc.ncbi.nlm.nih.gov/articles/PMC5910807/) |

Delta turns require special treatment: a rendered planar line crossing must not
be interpreted as a physically valid body crossing. Contact and out-of-plane
geometry must be resolved before assigning a biomechanical pass.

## 2. Food, foraging, and persistent locomotor states

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| FOR-01 | Basal slowing on food | Well-fed animal crossing a bacterial lawn edge; change in speed | M,C,K,I | P03 / [food slowing](https://www.sciencedirect.com/science/article/pii/S089662730081199X) |
| FOR-02 | Enhanced slowing after deprivation | Matched lawn-entry trials after controlled food deprivation; additional slowing and recovery | M,C,K,I,P | P03 / [food slowing](https://www.sciencedirect.com/science/article/pii/S089662730081199X) |
| FOR-03 | Local/area-restricted search | Immediately after food removal; elevated reversal/reorientation statistics and spatial confinement | M,R,C,I,P | P04 / [navigation circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC546636/) |
| FOR-04 | Global/dispersal search | Longer time away from food; longer excursions and changed reversal/turn hazard | M,R,C,I,P | P04 / [navigation circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC546636/) |
| FOR-05 | Roaming on food | Sustained faster exploratory bouts; occupancy, run length and bout duration | M,R,C,I | P04 / [state regulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3942133/) |
| FOR-06 | Dwelling on food | Slow, spatially restricted feeding-associated movement; reversals and state durations | M,R,C,I | P04 / [state regulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3942133/) |
| FOR-07 | Food-patch leaving | Patches differing in ingestion quality; departure hazard and return probability | M,R,C,I | P06 / [dietary choice](https://pmc.ncbi.nlm.nih.gov/articles/PMC1352325/) |
| FOR-08 | Dietary preference and sampling | Two bacterial foods; time-dependent occupancy and ingestion, conditioned on prior diet | M,R,C,I,P,O | P06 / [dietary choice](https://pmc.ncbi.nlm.nih.gov/articles/PMC1352325/) |
| FOR-09 | Satiety-associated quiescence | Deprivation/refeeding; feeding and movement cessation and return to activity | M,I,O | P10 / [satiety](https://pmc.ncbi.nlm.nih.gov/articles/PMC3786678/) |

## 3. Chemical navigation and avoidance

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| CHE-01 | Soluble-attractant chemotaxis | Salt/tastant gradient; navigation index, endpoints, paths and stimulus-response curve | M,R,C | P05 / [two navigation mechanisms](https://pmc.ncbi.nlm.nih.gov/articles/PMC6665864/) |
| CHE-02 | Volatile-odor attraction | Odor gradient with defined delivery; approach, adaptation and concentration dependence | M,R,C | P05 / [olfactory navigation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3422662/) |
| CHE-03 | Klinotaxis/weathervaning | During forward runs; continuous curvature response to the transverse gradient sampled by head motion | M,R,C | P05 / [two navigation mechanisms](https://pmc.ncbi.nlm.nih.gov/articles/PMC6665864/) |
| CHE-04 | Klinokinesis/pirouette strategy | Reorientation hazard versus experienced concentration change; run and turn sequences | M,R,C | P05 / [pirouettes](https://pubmed.ncbi.nlm.nih.gov/10531458/) |
| CHE-05 | Odor/soluble-repellent avoidance | Defined octanol or soluble aversive cue; reversal probability, latency and escape trajectory | M,R,C | P02,P05 / [polymodal avoidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC544906/) |
| CHE-06 | Osmotic avoidance | Local osmotic step/barrier; response probability and crossing/withdrawal | M,R,C | P02 / [polymodal avoidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC544906/) |
| CHE-07 | Predator-cue defense | Predator sulfolipid cue; escape plus later suppression of egg laying | M,R,C,I,O | P06,P11 / [predator cues](https://pmc.ncbi.nlm.nih.gov/articles/PMC5859177/) |
| CHE-08 | Pathogen-associated food avoidance | Exposure and subsequent food choice; acquired odor avoidance and lawn leaving | M,R,C,I,P,O | P09 / [pathogen learning](https://www.nature.com/articles/nature04216) |

“Food” must be split into bacterial biomass, nutritional benefit, soluble
chemicals, and volatile chemicals. A stationary bacterial patch can emit a
diffusing cue; bacterial mass itself should not automatically behave like the
current unbounded Gaussian chemical pulse. Different compounds and delivery
methods need separate transduction and transport parameters.

## 4. Mechanical, thermal, gas, and other responses

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| SEN-01 | Anterior gentle-touch escape | Local anterior touch; backward response and recovery | M,R,K | P02 / [touch circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565008/) |
| SEN-02 | Posterior gentle-touch escape | Local posterior touch; forward acceleration/escape | M,R,K | P02 / [touch circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565008/) |
| SEN-03 | Nose-touch withdrawal | Nose contact; reversal and response tuning across stimuli | M,R,K | P02 / [nose-touch experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC3145979/) |
| SEN-04 | Harsh-touch escape | Calibrated stronger body stimulation; response versus position/intensity | M,R,K | P02 / [nose/body mechanosensory experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC3145979/) |
| SEN-05 | Tap-evoked withdrawal | Distributed mechanical disturbance; reversal probability, distance and developmental dependence | M,R,K,D | P02,P09 / [tap responses](https://pubmed.ncbi.nlm.nih.gov/2376729/) |
| SEN-06 | Thermotaxis | Controlled gradient and cultivation history; warm/cold navigation under matching assay conditions | M,R,T,I,P | P07 / [defined thermal stimuli](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758190/) |
| SEN-07 | Isothermal tracking | Near cultivation temperature; duration and thermal deviation of aligned runs | M,R,T,P | P07 / [tracking trajectories](https://journals.biologists.com/jeb/article-pdf/209/23/4652/1257565/4652.pdf) |
| SEN-08 | Oxygen-dependent navigation | Gas gradient; occupancy and movement response with food/genotype controls | M,R,G,I | P08 / [oxygen experiments](https://pubmed.ncbi.nlm.nih.gov/15220933/) |
| SEN-09 | Carbon-dioxide response | Gas steps/gradient; acute escape and feeding-history dependence | M,R,G,I,P | P08 / [acute CO2 response](https://pmc.ncbi.nlm.nih.gov/articles/PMC2430355/) |
| SEN-10 | Short-wavelength-light avoidance | Calibrated light pulses/patches; escape versus wavelength, intensity, and stimulation site | M,R,L | P02,P08 / [light responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC2652401/) |
| SEN-11 | Humidity orientation after conditioning | Humidity gradient with starvation/conditioning controls; preference and trajectory | M,T,K,I,P,X | P08 / [humidity experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC4050571/) |
| SEN-12 | Electrotaxis (laboratory assay) | Electric field and polarity changes; cathode orientation and movement | M,R,X | P08 / [electrosensory experiments](https://pubmed.ncbi.nlm.nih.gov/17626220/) |

Thermotaxis must not be reduced to universal attraction to a fixed temperature.
For example, the cited defined-gradient experiments distinguish negative
thermotaxis and tracking, and did not observe positive thermotaxis in their
tested regime. Gas responses likewise depend on food, genotype, and history.

## 5. Experience and memory

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| LEA-01 | Sensory adaptation to repeated/sustained chemical cues | Stimulus history and washout; sensory response attenuation and recovery | C,P | P09 / [ASH adaptation](https://pmc.ncbi.nlm.nih.gov/articles/PMC544906/) |
| LEA-02 | Touch-response habituation | Repeated stimuli and recovery tests; response decrement with exposure | M,R,K,P | P09 / [touch experience](https://pmc.ncbi.nlm.nih.gov/articles/PMC4012311/) |
| LEA-03 | Salt/food-deprivation association | Paired versus unpaired salt/deprivation; later chemotactic preference | M,R,C,I,P | P09 / [salt conditioning](https://pubmed.ncbi.nlm.nih.gov/11316496/) |
| LEA-04 | Pathogen-odor associative learning | Exposure, delayed choice and safe-food controls; preference change and retention | M,R,C,I,P,O | P09 / [pathogen learning](https://www.nature.com/articles/nature04216) |
| LEA-05 | Thermal-history-dependent navigation | Shift cultivation temperature; later change in tracking/navigation regime | M,R,T,I,P | P07,P09 / [thermal history](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758190/) |

These are not interchangeable. Adaptation, fatigue, persistent neural activity,
and associative memory can all change a response; control experiments must
distinguish them. Online synaptic plasticity is one candidate model, not an
established requirement for every assay.

## 6. Feeding, gut, and reproduction

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| ORG-01 | Pharyngeal pumping | Food series and neural perturbations; contraction timing, pump frequency and ingestion | C,I,O | P11 / [pumping experiments](https://pubmed.ncbi.nlm.nih.gov/26976078/) |
| ORG-02 | Isthmus peristalsis | Pharyngeal imaging; peristalsis rate and temporal relationship to pumps | C,I,O | P11 / [feeding coordination](https://pmc.ncbi.nlm.nih.gov/articles/PMC3463504/) |
| ORG-03 | Spitting/feeding rejection | Aversive stimulation; pharyngeal state switch and particle ejection | C,L,I,O | P11 / [spitting mechanism](https://elifesciences.org/articles/59341) |
| ORG-04 | Defecation motor sequence | Fed recordings; posterior contraction, anterior contraction, expulsion and cycle variability | I,O | P11 / [defecation/locomotion coupling](https://pmc.ncbi.nlm.nih.gov/articles/PMC4657007/) |
| ORG-05 | Egg-laying bouts and context dependence | Adult reproductive recordings; event timing, bout structure and relation to locomotion/food | M,I,O | P11 / [whole-organism profiling](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347390/) |
| ORG-06 | Coordination of locomotion, feeding, and egg laying | Simultaneous multichannel recording; cross-program timing and state-dependent coupling | M,I,O | P11 / [whole-organism profiling](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347390/) |

These programs need specialized effectors. Body-wall-muscle contractions alone
cannot establish ingestion, expulsion, or egg-laying competence. Some rhythms
also involve intrinsic organ dynamics, so neural RL should act on an explicit
organ model rather than being rewarded to invent every clock from scratch.

## 7. Quiescence and stage-specific behavior

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| STA-01 | Developmentally timed sleep/lethargus | Larval stage transitions; reversible quiescence, reduced responsiveness and homeostasis | M,I,O,D | P10 / [lethargus experiments](https://pubmed.ncbi.nlm.nih.gov/18185515/) |
| STA-02 | Stress-induced sleep | Defined stress and recovery; quiescent bouts and arousability | M,I,O | P10 / [stress-induced sleep](https://pubmed.ncbi.nlm.nih.gov/25264259/) |
| STA-03 | Sleep homeostasis | Controlled interruption; subsequent change in sleep drive/structure | M,I,P,D | P10 / [sleep homeostasis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4273442/) |
| STA-04 | Dauer nictation and carrier dispersal | Appropriate substrate and carrier encounter; tail-supported waving and transfer | M,K,3,D | P12 / [nictation experiments](https://www.nature.com/articles/nn.2975) |

Dauer entry/exit, molting/ecdysis, and aging change which behaviors are available
and how they are expressed. Their biological timing belongs in a developmental
model before it becomes an RL task. They are scope extensions, not additional
actions for an adult controller.

## 8. Social and male-specific behavior

| ID | Behavior | Assay and observable target | Requires | Protocol / evidence |
| --- | --- | --- | --- | --- |
| SOC-01 | Aggregation and lawn bordering | Multiple animals with specified genotype, oxygen and food; cluster/border occupancy | M,R,C,G,I,S | P13 / [social circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC2760495/) |
| SOC-02 | Pheromone attraction/repulsion | Specified ascaroside/blend, sex and concentration; preference and avoidance | M,R,C,I,S | P13 / [social chemical signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254649/) |
| SEX-01 | Mate-cue attraction | Male sensory apparatus and hermaphrodite cues; approach and concentration tuning | M,R,C,D | P13 / [sex-dependent pheromone response](https://www.sciencedirect.com/science/article/pii/S0896627312006204) |
| SEX-02 | Contact response and backing along mate | Tail contact with partner; initiation and maintained contact | M,R,K,3,S,D | P13 / [mating experiments](https://pubmed.ncbi.nlm.nih.gov/7826644/) |
| SEX-03 | Turning around mate | Partner-end encounter; turn trajectory and contact retention | M,R,K,3,S,D | P13 / [mating experiments](https://pubmed.ncbi.nlm.nih.gov/7826644/) |
| SEX-04 | Vulva location | Scanning along partner; stopping accuracy and search retries | M,R,K,O,S,D | P13 / [mating experiments](https://pubmed.ncbi.nlm.nih.gov/7826644/) |
| SEX-05 | Spicule insertion and sperm transfer | Reproductive contact; insertion success and transfer sequence | K,O,3,S,D | P13 / [mating experiments](https://pubmed.ncbi.nlm.nih.gov/7826644/) |

The adult hermaphrodite 302-neuron topology cannot be used unchanged to claim a
male-mating model. Social behavior also requires matching strain: solitary N2
and aggregating backgrounds must not be pooled into a single target.

## Evidence gaps and expansion queue

The inventory currently contains 62 entries. Before calling it a comprehensive
ethogram, curate additional primary-assay records for noxious heat/cold escape,
pH-specific responses, odor discrimination and appetitive/context-dependent
learning, obstacle following, body self-righting, mating-related male food
leaving, ecdysis, and ecological collective behaviors. Split broad entries by
compound, stage, and assay only when they add a distinct measurable question.

Rheotaxis is a candidate physics benchmark: near-wall flow can orient a swimmer
through hydrodynamic effects, so upstream alignment must not automatically be
credited to a learned sensory policy. See [rheotaxis mechanism](https://www.pnas.org/content/pnas/112/12/3606.full.pdf).
Magnetotaxis is excluded from the core suite because independent work failed to
elicit a robust response: [replication study](https://pubmed.ncbi.nlm.nih.gov/36871106/).

## Candidate reference data

| Resource | Intended use | Qualifications |
| --- | --- | --- |
| [Yemini et al. behavioral database](https://pmc.ncbi.nlm.nih.gov/articles/PMC3962822/) | Wild-type posture, speed, reversals and feature definitions | Separate matched wild-type records from mutant phenotypes; confirm record-level metadata and download availability |
| [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) | Common extraction of trajectories, posture and behavior features | Software is not a reference cohort; use the same definitions on model and real trajectories |
| [WormWideWeb datasets](https://wormwideweb.org/about/datasets/) and [Atanas & Kim et al.](https://www.sciencedirect.com/science/article/pii/S0092867423008504) | Simultaneous behavioral and identified-neuron recordings | Match recording geometry; calcium signals require an observation model, not direct equality to membrane voltage |
| [Developing-worm curvature data](https://datadryad.org/dataset/doi%3A10.5061/dryad.stqjq2c8p) | Stage-specific waveform/reference variability | Published archive lists time and ten curvature channels; stage/assay matching remains necessary |
| Linked primary-study supplements | Stimulus timing, individual trajectories, response curves and organ events | Availability and permissions must be audited before ingestion; paper plots alone do not establish that raw data are available |

No new datasets were downloaded for this catalog. Large data remain external to
Git under the configured r-elegans data root. Protocols must record dataset
version, source, license, units, strain, sex, stage, age, feeding/cultivation
history, medium, stimulus delivery, extraction method, and exclusions. Missing
metadata are an evidence gap, not a default biological value.
