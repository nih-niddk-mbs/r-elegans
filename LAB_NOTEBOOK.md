# r-elegans laboratory notebook

Established 2026-11-06 at the user's request. Scientific question: can the
anatomically constrained recurrent network reproduce animal behavior, and how
specific do its effective connection weights have to be? Behavioral agreement
is the objective; agreement with unknown biological weights is not assumed.

This is a retrospective reconstruction through E015, followed by ongoing
records. Dates before this notebook was established come from commit dates,
run labels and reported logs; exact execution times are not reconstructed when
missing. Commit dates are not proof that a remote run used that commit.

## Evidence and storage conventions

- **Local artifact verified:** files, manifests or lineage records inspected in
  the local archive. Source revision/dirty status below comes from manifests.
- **Server result reported:** user-supplied server stdout, parsed and preserved
  with its SHA-256. This is evidence of the reported result, not independent
  verification of server checkpoints, source revision or installed environment.
- **Historical reconstruction:** Git history and earlier conversation/document
  summaries; missing original logs and commands remain explicit gaps.
- **Implemented/tested:** software validation, not a successful scientific run.

Raw data, model checkpoints, optimizer states, trajectories and stdout remain
outside Git. Small result summaries and evidence hashes are metadata in
[experiments/records](experiments/records). Exact pasted stdout is preserved
under `notebook/evidence/<entry>/reported-server-results.txt` relative to the
OneDrive r-elegans directory. The [evidence index](experiments/records/evidence-index.json)
records hashes; it does not prove an off-machine backup of the server files.

Storage roots:

- Local: `/Users/carsonc/OneDrive/r-elegans`, resolving to the institution's
  OneDrive folder. Selected motor archives: `results/motor_curriculum/`.
- Server: `/home/carsonc/github/r-elegans` and
  `/home/carsonc/r-elegans-results` on `mbslnx2.niddk.nih.gov`.
- New server archives: `/home/carsonc/r-elegans-archives`, separate from live
  results. Same-disk copies are not off-machine backups.

## H1 — build the motor repertoire before task RL

Recorded explicitly 2026-11-06 from the user's working hypothesis, following
E015–E016. **Status: hypothesis, not an established result of this project.**

**H1.1, user-directed revision on 2026-11-06:** prioritize coverage of the
motor repertoire, preserving successful independent full-network checkpoints.
Joint retention in one parameter set is no longer a prerequisite for adding
the next skill. The intended downstream design is a parameterized skill action
space for a high-level RL selector; the runtime hierarchy is not implemented.
Consolidation/interference mitigation is deferred. This supersedes the
consolidation-first next-step proposals after E014–E017; those historical
results and decisions remain recorded below. The operational list is
[SKILL_REPERTOIRE.md](SKILL_REPERTOIRE.md).

For this embodied connectome model, establish and retain useful motor policies
through small, progressively harder steps before asking task RL to discover
sensory-guided behavior. A viable, accessible motor repertoire should make
exploration more productive: task learning can refine or coordinate movements
that already affect the environment rather than also discovering locomotion
from ineffective muscle outputs. The user's phrase "sketch out the policy
space" refers to constructing that repertoire/initialization, not enumerating
the complete mathematical policy space or fixing the final solution.

**Literature connections:**

- **Curriculum learning:** sequence training from easier to harder problems,
  related to behavioral shaping in animal learning.
  Bengio et al. (2009), [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf).
- **Motor-skill pretraining:** learn useful behaviors before downstream tasks.
  Florensa, Duan and Abbeel (2017), [Stochastic Neural Networks for Hierarchical
  Reinforcement Learning](https://arxiv.org/abs/1704.03012).
- **Learned skill priors:** use previous behavioral experience to guide later
  exploration. Pertsch, Lee and Lim, CoRL 2020 / PMLR 2021, [Accelerating
  Reinforcement Learning with Learned Skill Priors](https://proceedings.mlr.press/v155/pertsch21a.html).
  Our current controller has no learned probabilistic skill-prior distribution;
  this is a related direction, not an algorithm already implemented here.
- **Hierarchical RL/options:** a future task policy could select temporally
  extended skills. Sutton, Precup and Singh (1999), [Between MDPs and semi-MDPs](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).
  We have no implemented high-level skill selector or formal options with
  initiation/termination rules. H1.1 selects skill-based downstream control as
  the intended direction; runtime composition and later consolidation remain
  to be designed and tested.

The most accurate current description is **curriculum learning with motor-skill
pretraining**. This is not a claim that all successful RL requires pretraining,
nor that motor pretraining alone will resolve interference, objective mismatch,
autonomous rhythm or sensory encoding.

**Project observations motivating H1:** direct food-task training produced no
success in E004; staged supervision subsequently produced driven motion and
stop/restart. These experiments were not a controlled comparison, so they do
not establish that a curriculum causes better downstream RL performance.
E014–E015 also show that new skills can interfere and imitation can fail to
preserve propulsion. Under H1.1, preserve independent successful checkpoints
and track interference; do not require joint retention to expand the library.
Each individual skill must still demonstrate physically valid effects.

**Proposed sequence, not yet tested:**
preserve forward/backward/stop checkpoints; independently add steering, head
sweeps and other motor variants in small increments; extend mechanics for
contact-dependent/3D or specialized organ behaviors; progressively replace
external rhythmic pacing with internally generated motor activity; then train
skill selection/composition for sensory tasks. These tracks can progress
without forcing one shared weight set to perform every skill first.
Commandable internally generated movement is distinct from autonomous action
selection. Present results remain externally driven.

**Falsifiable test, planned only:** compare task RL from random initialization,
abrupt full-skill pretraining, and incremental motor pretraining, with the same
network, physical model, sensory information and task distribution. Hold total
compute/environment interaction budgets comparable and charge pretraining to
its budget. Use multiple independent seeds, separate tuning/test tasks, RL
success versus samples/time, and motor retention/robustness after task learning.
If incremental pretraining gives no reliable advantage, or restricts task
performance, revise H1 rather than treating it as a prerequisite by definition.
No such downstream comparison has been run.

## Fixed scope and acceptance criteria

The active motor model simulates all 302 generic graded-potential neurons,
3,709 directed chemical edges, 1,093 gap pairs, and 956 projections to 95
muscles. Anatomical counts are not calibrated conductances. Motor-stage fits
train neural parameters while keeping topology and NMJ/muscle fields fixed.
AWCON/RMD electrophysiology fits are available separately; the full motor
actor does not use neuron-specific calibrated ion channels for all cells.

These experiments use the normalized 12-rod, 11-joint body, length 1, coupled
overdamped RFT mechanics and head-aligned muscle projection. They do not use
the SI liquid/agar factories. The longitudinal head is at the final segment.
Physics uses highest-precision matrix products. See [PHYSICS.md](PHYSICS.md).

Muscle targets are synthetic: baseline 0.4 plus a smooth dorsal/ventral
modulation. Pacing supplies artificial currents to NMJ-projecting neurons,
not a biological sensory pathway. Reverse mode adds a fixed random-sign
direction context (seed 719, gain 0.05). Negative pacing alone changes phase,
not propagation direction. Forward and reverse direction are held separately
in the assays; transitions between them have not been validated.

Wave gates: muscle RMSE <0.025, relative body-angle RMSE <0.25, bend correlation
>0.9, active-frame agreement >0.95, correct propagation sign, muscle/body lags
of magnitude >=40 ms and within 60 ms of target (float32 boundary epsilon).
Targets have 0.4-second end-to-end delay; measured front/rear-band target lag
is usually 0.14 seconds. These are different quantities.

Locomotion gates: phase-matched centroid displacement on the shape-derived
tail-to-head axis, periodic shape, consistent cycles, timestep agreement,
rest/isotropic controls, straightness, and >=0.001 body lengths per cycle in
magnitude at every pacing period. The floor is an engineering readiness
criterion, not measured biological speed. Tests use 2.4/3.0/3.7-second pacing
and 20/10-ms timesteps. Stop/restart blocks contain six cycles each of
go/zero/go, first three settling and last three measured. Stop path length
and shape speed must be <=10% of the moving values, and restarted progress
80–120%. This does not measure immediate stopping latency.

Exit code 2 means an acceptance gate was not met when status is completed and
error is null. Crashes/missing summaries must be distinguished from such
results. No food-seeking RL is launched automatically by the motor curriculum.

## Chronological record

### E001 — 2026-08-31 to 2026-09-04: initial model and surrogate baseline

**Purpose/method:** acquire anatomy and physiology externally; establish
single-compartment neurons, muscle/body dynamics, anatomical NMJs, fitted
motion commands, neural-output teacher and virtual-dish food cues. Bundle
compact derived topology/parameters so the demo runs without raw downloads.
Relevant commits: `c688f20`, `797f88f`, `b4d9819`, `48a6c25`, `0371a37`,
`1432c6c`, `801fa87`, `09d9b36`, `96b5d12`, `84746f2`.

**Result/interpretation:** the food demo uses an engineered sensory controller
and fitted gait or nonrecurrent motor-output teacher. A separate 14-neuron
subcircuit was developed. These results do not demonstrate food finding by
the full 302-neuron recurrent network. Anatomy, polarity and cellular evidence
are documented in [MODEL_ASSET_PROVENANCE.md](MODEL_ASSET_PROVENANCE.md).
**Evidence/gap:** historical reconstruction; exact early fit invocations and
all raw acquisition volumes have not been recovered into this notebook.

### E002 — 2026-09-04 and 2026-11-03: mechanics, RL and accelerator corrections

**Hypothesis/problem:** prescribed joint rates hid effects of environmental
load. **Change:** coupled generalized RFT force/torque balance, rod rotational
drag and implicit elasticity (`022dc05`); RL corrections (`5e8f816`); GPU
precision/dependency fixes (`6249738`). Keep previous normalized controllers
separate from SI mechanics. Reduced-order RFT is not resolved CFD.

**Result:** GPU tests initially showed resistance asymmetry and locomotion
differences from reduced-mantissa operations; request HIGHEST precision rather
than relaxing tolerances. Missing `six` had hidden Gymnax tests behind skips.
User reported 107 GPU tests passed in 184.06 seconds after fixes.
**Evidence/gap:** Git and reported console results; original environment
snapshot absent. **Decision:** validate mechanics before further behavioral RL.

### E003 — 2026-11-04: choose hardware from measurements

**Method:** synchronized throughput benchmarks excluding compilation;
independent CPU workers pinned to disjoint physical cores. Source commits
`6ec5193`, `dcfa890`, `b392212`. Server has two A100 80-GB GPUs and 128 logical
CPUs (64 physical cores identified). Conda environment `r-elegans`, Python
3.11/JAX 0.6.2; do not modify the separate scRNA environment.

**Reported results:** small 14-neuron workload: A100 about 6,204 steps/s;
four CPU cores/128 environments about 18,402; four concurrent four-core
workers aggregate 61,303 (3.32x single-worker speed, 82.9% efficiency).
**Interpretation:** not a benchmark of the later full-network workload.
**Decision:** use mbslnx2 CPU workers for unattended ensembles; Mac for short
fits. **Evidence/gap:** historical console/JSON reports, not a fresh measurement.

### E004 — 2026-11-04: direct full-network food-seeking pilot

**Question/method:** can the full recurrent actor learn the food task directly?
Implementation `2efd321`, timed pilot infrastructure `21ad2f1`; four seeds,
about one hour each, roughly 7.2 million steps per seed, 28,888,000 total.
**Reported result:** all held-out success rates remained 0%; some distances
improved. **Interpretation:** this establishes neither useful policy nor
network insufficiency. **Decision:** follow the user's correction: establish
small motor primitives by supervision before sensory-task RL.
**Gap:** exact original remote run directory/manifests not independently recovered.

### E005 — 2026-11-04: full-network static bend

**Hypothesis/method:** first fit a small held bend through the recurrent brain
and fixed NMJs. Commit `01eb8e3`; seed 0, amplitude 0.04, 500 updates, 64-step
training and 128-step verification, no external pacing. About 18 seconds on Mac.
**Result:** muscle RMSE 0.455 ->0.0099; body error about 12.6%, longer hold
13.7%. Static-bend gates passed. **Decision:** add controlled alternation.
**Local verified archive:** `bend-01eb8e3-seed0`, clean revision in manifest.

### E006 — 2026-11-04: controlled alternation and projection correction

**Method:** scalar artificial motor stimulation; sine and level commands;
validate changed order/rate/phase at twice the horizon. Correct anterior
muscles to the last-segment head; preserve historical projection defaults for
old demo reproducibility. Commit `fb731d0`.
**Result:** local 1,000-update warmup plus 200 aligned updates passed; final
one-update clean-code verification in archive. Later reported four-seed
Linux fits passed, body errors about 5.1–5.8% after about 18k updates each.
**Incident:** first server launch used `/usr/bin/python`, missing JAX. Add
environment preflight and require conda activation. Failed launches are not
scientific negative results. **Local verified archive:**
`alternation-fb731d0-seed0/lineage.json` distinguishes warmup/aligned/verified
and their dirty/clean source status. **Decision:** teach propagation next.

### E007 — 2026-11-04: causal traveling wave

**Hypothesis/method:** same scalar pacing, with delayed muscle pattern only in
the supervision. Anterior-to-posterior target delay 0.4 seconds; no delayed
pattern supplied to neurons. Implement `04bc1fc`, record `06a386a`.
**Result:** first local 1,500 updates failed a changed level-command lag;
another 1,500 passed. Clean one-update verification is archived separately.
Later reported four-seed Linux ensemble passed wave fits, body errors roughly
7.1–12.8%. **Decision:** distinguish wave imitation from actual propulsion.
**Local verified archive:** `wave-06a386a-seed0`; failed first attempt retained.

### E008 — 2026-11-05: establish sustained translation

**Method:** frozen brain, 4 startup plus 6 measured cycles, actual centroid and
shape-derived axis, 20/10-ms refinement, zero-command and isotropic controls.
Implementation `e9f3dc1`. **Local result, amplitude 0.04:**
0.000321/0.000384/0.000420 body lengths/cycle at the three periods; rest zero,
isotropic drift <2e-10, timestep differences about 0.5–1.4%. Real forward
translation, below the unchanged 0.001 readiness floor.
**Decision:** increase target amplitude gradually, not restart RL.
**Local verified archive:** `locomotion-e9f3dc1-seed0/baseline`.

### E009 — 2026-11-05: local amplitude continuations

**Method:** 0.04 ->0.05 ->0.06, 1,000 supervised updates at each increment,
fixed pacing gain and target delay. **Result:** 0.05 gives
0.000511/0.000610/0.000665 body lengths/cycle (about +59%); 0.06 gives
0.000749/0.000888/0.000965 (about +45–46%). Both retain wave and numerical
checks but miss readiness. **Evidence:** verified local archives
`locomotion-e9f3dc1-seed0/amplitude-fit`, `amplitude-assay`, and
`amplitude-006-e9f3dc1-seed0/{fit,assay}`. The 0.05 fit used a dirty source
overlay; that overlay is retained, so its commit alone is not sufficient.
The 0.06 manifests identify clean `e9f3dc1`. **Decision:** replicate ensembles.

### E010 — 2026-11-05: server amplitude 0.05 ensemble

**Reported result:** all four wave fits pass, stable forward progress
0.000535–0.000605 body lengths/cycle, +52–67% over baseline, all below readiness.
Root reported as `wave-20261105T164155192027Z` under server results.
**Evidence:** [E010 metadata](experiments/records/E010.json); exact stdout hash
in evidence index. **Decision:** continue each matching seed to 0.06.

### E011 — 2026-11-05: server amplitude 0.06 ensemble

**Reported result:** all wave fits pass; 0.000772–0.000922 body lengths/cycle,
+42–57% over 0.05; all retain periodicity, consistent direction and straightness
but remain below readiness. Duplicate pasted results were recognized rather
than counted as a second experiment. **Evidence:** [E011](experiments/records/E011.json).
**Decision:** one further small amplitude increment, to 0.07.

### E012 — 2026-11-05: server amplitude 0.07 ensemble

**Reported result:** all wave fits pass, progress increases 29–35%.
Seeds 0/1/3 pass all locomotion rates; seed 2 misses the slowest rate at
0.0009956 body lengths/cycle. All remain stable, straight and forward.
**Source cohort for later stages:** `wave-20261105T214835887935Z`.
**Evidence:** [E012](experiments/records/E012.json).
**Decision:** preserve these forward checkpoints; test stopping before training it.

### E013 — 2026-11-05: frozen stop/restart

**Question/method:** does zero pacing already stop the learned brain/body?
Commit `a073406`, no optimizer, no state reset between command blocks.
**Local check:** saved 0.06 model stops/restarts, but keeps its previous
below-floor propulsion; archive `stop-restart-local-006-seed0` records dirty
source plus snapshot. **Server 0.07 result:** all four seeds pass stop/recovery
at all rates; residual shape speed <0.024% of moving value; restart within
0.1% of original progress. Overall readiness passes 0/1/3; seed 2 retains the
same slow-rate floor miss. **Evidence:** [E013](experiments/records/E013.json).
**Incident:** GitHub temporarily rejected publication (`commit_refs_rules`);
bundle prepared, then normal push succeeded. Initial server `--stop-restart`
rejection was an old checkout, fixed by pulling. **Decision:** teach reverse.

### E014 — 2026-11-05 to 2026-11-06: three-hour reverse run

**Hypothesis/method:** a shared recurrent policy can support both directions.
Associated implementation `8f2c326` (remote revision still needs its manifest).
Start from each 0.07 forward checkpoint; reverse context as above; teach a
posterior-to-anterior wave while equally rehearsing frozen forward outputs.
192 training steps, 384 validation steps; learning rate 0.001; four seeds,
180-minute supervisor budget (about 179 fitting). Save every 100 updates;
select by training loss, then run both physics assays.

**Reported outcomes:** 26,713–26,940 updates/seed; every stage completed without
errors. All models show actual forward and backward translation and stop/restart
control in both directions. Readiness and imitation diverge:

| Seed | Reverse imitation | Forward readiness | Reverse readiness |
| --- | --- | --- | --- |
| 0 | Pass | Fail | Fail |
| 1 | Failed changed hold sequence | Fail | Passed 2/3 pacing rates |
| 2 | Pass | Fail | Pass |
| 3 | Lag outside tolerance | Pass | Pass |

**Interpretation:** reverse learned, but forward weakened in 0–2; no seed meets
every gate. This is not a crash or proof of anatomical insufficiency.
**Decision:** broaden commands, strengthen original-forward retention and
select jointly on tuning validation. **Evidence:** [E014](experiments/records/E014.json).
Server run directory is not recovered from this pasted summary; discover it
from the server manifest rather than inventing a label.

### E015 — reported 2026-11-06: eight-hour refinement

**Hypothesis:** broader, longer commands and stronger original-forward
rehearsal will fix generalization and retain useful propulsion in both modes.
**Associated implementation:** `ddd2d7e`; actual remote revision/environment
await archive verification. Resume E014 candidates with fresh optimizers;
recover each original 0.07 forward teacher by saved path/hash.

**Protocol:** 8 training sequences, 384 steps (7.68 s), periods 2.1/2.7/3.3/4.1 s
plus phase/level/stop variation; minibatches of 2. Eight separate 768-step
tuning sequences, periods 2.3/3.1/3.9/4.5 s. Forward normalized muscle loss
weight 4, reverse weight 1; learning rate 0.0003. Validate every 500 updates,
select by failed-case count, worst normalized error, then mean. These are
tuning data, not an independent test set. Save every 100 updates. Requested
supervisor budget 480 minutes; final long physics assays are separate from
selection. Thresholds, amplitude and direction-context definition unchanged.

**Exact reported progress:**

| Seed | Updates | Selected update | Failed tuning cases | Forward readiness | Reverse readiness |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 36,125 | 19,500 | 0 | Pass | Fail |
| 1 | 35,774 | 35,000 | 0 | Pass | Fail |
| 2 | 36,063 | 35,500 | 0 | Failed slowest rate | Fail |
| 3 | 35,807 | 2,000 | 1 (forward) | Fail | Fail |

All stages completed, error null, stop reason time budget. Every model retains
detectable, periodic, consistent, straight motion in the commanded direction,
and successful stopping/recovery. Reverse magnitudes are
0.0005968–0.0009962 body lengths/cycle across go windows; all miss the 0.001
floor in at least one condition. Seeds 0/1/2 pass imitation despite failing
reverse propulsion readiness. The overall exit code is 2, not a crash.

**Interpretation:** stronger rehearsal recovers forward readiness in 0/1 but
does not preserve sufficiently strong backward propulsion. Selecting on body
wave imitation still does not optimize the movement metric. More elapsed time
on this objective alone is not established as a solution. No joint success
and no inference about unique biological weights or network insufficiency.
**Decision proposed, not run:** incorporate actual bidirectional displacement
and retention into optimization/checkpoint selection while preserving stop,
coordination and numerical gates. Do not advance to turning or food RL yet.
**Evidence:** [E015](experiments/records/E015.json), hashed user-supplied stdout;
remote checkpoints/logs/manifest and backup remain unverified locally.

### E016 — 2026-11-06: establish replication records

**Problem:** status documents are not a chronological notebook, and historical
environment/command capture is incomplete. **Action:** establish this notebook,
retain hashed reported evidence and machine-readable summaries; correct stale
eight-hour status; add automatic invocation/environment/source snapshots to
motor/PPO trainers, physics assessors and the CPU supervisor. New invocation
records never overwrite earlier invocations, including repeated output paths.
Archive tool preserves finished cohorts and recursive checkpoint dependencies,
a repository bundle, source snapshot, file inventory and explicit gaps.
**Validation:** 154 tests passed in 56.32 seconds. A one-update local refinement
invocation exercised the new startup records and source snapshot successfully;
its expected below-gate exit code 2 is not a new scientific result. Archive
tests cover dependency closure, missing checkpoints, dirty source preservation,
separate invocation records, and corruption detection. Agent instructions in
`AGENTS.md` require this recordkeeping for subsequent scientific work.
**Status:** implementation/testing recorded in the accompanying commit; remote
archive creation and off-machine copying are still pending user execution.
This infrastructure does not change the learning objective or claim a new fit.

### E017 — 2026-11-06: state H1 and connect it to the literature

**User proposal:** RL may become tractable when a useful policy repertoire is
established first, in small steps. **Action:** record H1 above with its scope,
literature links, predicted benefit, present evidence and a falsifiable future
comparison. The broad name is curriculum learning; motor-skill pretraining
describes the current preparation. Learned skill priors and hierarchical RL
are related approaches with additional machinery not currently implemented.
**Decision:** preserve this as the organizing hypothesis, requiring physical
behavior and old-skill retention checks at each stage. Distinguish driven
control, endogenous rhythm and sensory-driven task choice. **Outcome/status:**
documentation and literature review only; no new learning run or proof of H1.

### E018 — 2026-11-06: prioritize repertoire coverage over consolidation

**User direction:** build the worm's skill repertoire and map a structured
action space before downstream autonomous-motion/search RL; do not make
interference between skills the immediate bottleneck. **Strategy H1.1:**
learn and preserve separate policies/checkpoints when useful, measure each
skill's physical effects and parameter support, and defer consolidation into
one network parameter set. Interference is recorded, not a gate blocking
the next standalone skill. All 302 neurons may remain in every candidate;
separate policies do not imply pruning the brain or granting biological
fidelity. Existing physical/numerical criteria remain.

**Action:** add the operational [skill repertoire](SKILL_REPERTOIRE.md), with
50 entries (15 motor, 8 sequence/state, 12 sensory-control and 15 extension
families), mapping all 62 entries of the existing biological inventory.
Candidate variants have parameters, observable targets and evidence/status.
Deep turns/contact, organ programs, male/dauer and 3D behavior require the
listed model extensions. Roaming, search and navigation are policy patterns,
not additional atomic muscle commands. Standalone controllers still need
state-handoff/transition validation before runtime composition.

**Next coverage priorities proposed:** shallow signed steering, head sweeps
and independent movement-intensity variants, preserving the better previous
forward and reverse checkpoints. No new fit is launched by this documentation
change. **Validation:** all 62 biological IDs mapped, all 50 repertoire IDs
unique, relative documentation links resolved, and whitespace checks passed.
**Outcome/status:** strategy/catalog update only; no new behavioral
competence, options implementation or autonomous policy demonstrated.

### E019 — 2026-11-06: standalone shallow-steering protocol and local results

**Question:** can an independently fitted full recurrent brain drive a shallow
curved forward path, expanding MOT-08 coverage without consolidating earlier
skills? **Parent:** preserve the original forward-wave checkpoints; local
development uses the archived 0.06 seed-0 forward fit and server training will
use the original 0.07 wave cohort. No changes to parent files or NMJ/muscle
mapping. Artificial pacing remains; autonomy is not claimed.

**Stage A, simulator feasibility (planned):** apply a synthetic forward muscle
wave plus a constant mild dorsal/ventral bias. Examine bias magnitudes
0.003/0.006/0.012 and both signs, before fitting any neural controller. Establish
whether the actual body centroid follows a turning trajectory, rather than
counting a changed head-tip pose as steering. This is oracle calibration, not
a learned-skill result. If these fail physical criteria, revise the protocol
explicitly before a neural fit; do not silently relabel the outcome.

**Stage B, neural fits (planned):** separate signed-bias policies are acceptable;
all 302 cells evolve and only neural fields learn, with fixed NMJs. Supervision
is a muscle target; no target muscle command enters the neural forward path.
Select/check candidate effects in the actual body. Preserve forward/reverse
skill checkpoints rather than enforcing cross-skill retention.

**Predeclared physical checks:** four startup plus six measured phase-matched
cycles at periods 2.4/3.0/3.7 s; 20/10-ms refinement; zero pacing and isotropic
drag controls. Forward progress floor remains 0.001 body lengths/cycle.
Signed body-axis yaw and signed centroid-path turning must agree with the
oracle and each other. Use >=0.00005 rad/cycle detectable turn as an engineering
floor, require the signal above numerical/control differences, periodic shape
and consistent cycle signs. A fitted candidate must recover at least half the
oracle's turn magnitude. No biological turning range is inferred from these
normalized engineering criteria.

**Development amendment before the next fit:** the initial muscle-only local
pilot creates the desired turn but its tuning probe remains just below the
forward-progress floor after 1,000 updates. Preserve that attempt rather than
loosening the floor. Add a differentiable body-trajectory objective (joint
angles and actual centroid positions relative to the same oracle) alongside
muscle supervision, and test finite gradients before fitting. Default planned
body-loss weight 0.5, learning rate 0.0003. Physical checkpoint selection and
independent refinement/control assays still decide success. Isotropic drag
controls translation; it is not assumed to forbid geometric body rotation.
**Local outcomes (artifact verified):** direct-body oracle calibration passed
at all three pacing rates for target amplitude 0.07 and bias 0.006; the sign
of the bias correctly sets the sign of both body and centroid-path rotation.
The initial calibration screen examined the predeclared smaller/larger biases;
0.006 was chosen to leave margin above the minimum detectable-turn criterion.
That exploratory screen used five phase increments and survives in tool
stdout, not as a complete archived run. The archived selected-bias oracle and
all fitted-policy acceptance assays use the declared six measured cycles.
The oracle is not itself a neural controller.

| Trial | Updates / selected | Result | Interpretation |
| --- | --- | --- | --- |
| Positive, muscle-only objective | 1,200 / 1,200 | Overall fail; coarse tuning-probe progress below 0.001 | Final fine-timestep physical assays pass, but conservative coarse selection gate remains unmet; do not erase this borderline attempt |
| Positive, body-trajectory objective | 300 / 100 | Pass, all three rates and controls | Independently learned positive steering |
| Negative, body-trajectory objective | 300 / 100 | Pass, all three rates and controls | Independently learned negative steering |

The body-objective fits took 71.81 and 73.93 seconds on the Mac. Positive
signed body turns are 0.0001065/0.0001418/0.0001787 rad/cycle at
2.4/3.0/3.7 s; negative turns are -0.0000928/-0.0001013/-0.0000923. Forward
progress across the two policies is 0.0011967–0.0015584 body lengths/cycle.
Centroid-path turning agrees in sign and magnitude with body-axis turning;
20/10-ms refinement, zero-pacing rest and isotropic translation controls pass.
Both neural candidates pass the longer waveform checks and actual coarse
turning/progress selection probe. NMJ/muscle/sensory/noise fields and topology
were verified unchanged; parent checkpoint bytes are preserved.

**Evidence:** [E019 metadata](experiments/records/E019.json). Full trial records,
all four calibration/fit attempts, startup code/environment snapshots, input
checkpoint copy, raw trajectories, plots and checksums are archived at
`results/motor_curriculum/steering-20261106-local-seed0` under OneDrive r-elegans.
Original upstream forward-training history remains in the earlier motor
archives; the included input copy is sufficient to repeat these steering
fits with rebased paths. Dirty development source snapshots, rather than a
clean commit alone, identify the actual trial code. The muscle-only and body
fits are a development comparison, not a matched-budget causal study.

**Software validation:** 162 tests passed, including gradient checks through
actual body trajectories, rejection of pose-only/stationary false turns,
progress/numerical controls and independent signed server dispatch.
**Coverage:** MOT-08 now has two independently learned driven variants from
one parent model, not a calibrated biological turning range, autonomous skill,
multi-seed replication, or a validated controller handoff. The originals and
the below-gate pilot remain available. **Next prepared run:** four independent
mbslnx2 workers, source 0.07 wave cohort, even seeds positive/odd negative bias,
480-minute budget and complete final physical assays. Not launched remotely
by the agent. Head sweeps remain a subsequent coverage target.

### E020 — 2026-11-06: server steering success reported (partial transcript)

**Parent/protocol:** intended E019 mbslnx2 standalone-steering cohort, four
independent candidates with alternating bias sign and 480-minute budget.
**Reported aggregate result:** `MOTOR_CURRICULUM_EXIT_CODE=0`. Under the
published supervisor this means every configured worker's steering stage
passed with completed status. This is reported cohort success, not independent
inspection of each remote checkpoint or confirmation of the exact invocation.

The pasted excerpt begins inside a JSON record. It does not identify the
visible seed or pacing rate; these remain unknown rather than inferred from
print order. Its visible fine-step condition shows forward progress
0.0011677308 body lengths/cycle, signed body-axis turn -0.0001492804 rad/cycle
and centroid-path turn -0.0001411836 rad/cycle. Axis/path direction, consistency,
agreement and numerical controls are true. Rest forward drift is 1.39e-9 and
isotropic forward drift 7.41e-11 body lengths/cycle. Visible stage elapsed
time is 28,782.01 seconds with stop reason `time_budget`.

**Evidence:** [E020 metadata](experiments/records/E020.json) and checksummed
partial stdout preserved in the OneDrive evidence directory. Full per-seed
summaries, selected iterations, actual run path, manifests and checkpoint
hashes still require server archive retrieval. No ranges or seed statistics
are reconstructed from this tail alone. **Interpretation:** standalone driven
steering passes the server supervisor's configured gates, supporting skill
coverage beyond the local pilot; no autonomy or skill-switching claim.
**Next coverage target:** independent head sweeps (MOT-09), retaining all prior
policies. This entry records results only and does not launch a new fit.

### E021 — 2026-11-06: stationary head-sweep protocol and local development

**User request:** run the next independent skill after reported steering
success. **Question:** can the full brain oscillate the head relative to a
quiet trunk, adding MOT-09 without consolidating earlier skills?
**Parent:** original forward-wave checkpoints; local development uses the
archived 0.06 seed-0 forward input. All 302 neurons remain recurrent, topology
and NMJ/muscle fields remain fixed. Separate head-sweep weights are acceptable.

**Planned target/interface:** synthetic bidirectional head oscillation around
0.4 muscle activation, amplitude 0.04, anterior envelope exp(-(p/0.18)^2),
where anatomical position p=0 is head. Applied artificial pacing currents
are weighted by anterior NMJ contacts; no target muscle pattern enters the
brain as a supplied muscle output. Anterior localization is partly specified
by this experimental input map, not discovered de novo; it is a deliberate
scaffold, not a known biological command pathway. No forward locomotion is
requested. Retain scalar pacing, body-trajectory
supervision, independent checkpoints and complete run/environment/source records.

**Predeclared engineering gates:** four settling plus six measured cycles,
2.4/3.0/3.7-second pacing, 20/10-ms refinement. Require head-angle peak-to-peak
>=0.02 rad and >=half the oracle, head-center lateral excursion relative to
the trunk >=0.0005 body lengths and >=half the oracle. Posterior eight-joint
dynamic RMS must be <=25% of anterior three-joint RMS; static head bending
and rigid whole-body rotation must not count. Require periodic shape closure
<0.1, stable sweep amplitudes, bounded phase-matched centroid drift <=0.0001
body lengths/cycle, and net trunk orientation drift <=0.002 rad/cycle.
Amplitude signals must exceed numerical/rest differences; zero-pacing rest
must have <=10% residual sweep amplitude. The 0.001 propulsion minimum from
locomotion is inapplicable to a deliberately stationary skill. Isotropic drag
is recorded as a mechanics control, not used to require disappearance of a
head oscillation. No biological amplitude/frequency calibration is claimed.

**Plan:** validate the body-only oracle first; test finite gradients and false
positives; fit/check a local pilot; then prepare the four-seed eight-hour
mbslnx2 run from the original forward cohort. No runtime handoff or autonomous
head motion is claimed. **Status:** planned; outcomes to be appended.

**Local development update:** the body-only oracle passes all three periods.
A 1,000-update local neural pilot passes every physical head-scan test and
control, but longer-command body trajectory errors 0.352/0.360 exceed the
unchanged 0.25 imitation gate. Preserve that below-gate attempt. Continue its
selected checkpoint for another 1,000 updates with a fresh optimizer, same
target/input/objective and all gates unchanged. This is a staged continuation,
not exact optimizer-state resume, and does not justify reducing the gate.

**Continuation outcome / final local refinement plan:** the additional 1,000
updates again pass all physical tests; trajectory errors improve to
0.253/0.269, still above 0.25. Preserve both below-gate fits and run one final
1,000-update local continuation with identical settings, fresh optimizer and
unchanged criteria. The server stage is already software-tested; these local
continuations assess fit convergence, not new biological hypotheses.

**Final local result (artifact verified):** the third 1,000-update stage passes
both tuning and all physical tests; its selected checkpoint is update 1,000
(3,000 total staged updates from one source model). Longer-command trajectory
errors are 0.230/0.249, head correlations 0.997/0.995, below/above the unchanged
respective gates. Fits took 215.57/210.78/210.03 seconds; each continuation
started a fresh optimizer and the previous selected actor, not the original
forward actor. The two earlier below-gate attempts remain preserved.

At 2.4/3.0/3.7-second pacing, the final head-angle peak-to-peak excursions are
0.1595/0.1693/0.1750 rad; lateral head-center excursions relative to the trunk
are 0.01885/0.02003/0.02071 body lengths. Posterior/anterior dynamic joint RMS
ratios are 0.0558/0.0559/0.0561 (limit 0.25). Maximum phase-matched centroid
drift across these conditions is 1.23e-6 body lengths/cycle, well below 1e-4.
Rest suppression, periodicity, stable amplitudes and 20/10-ms controls pass.
NMJ/muscle/sensory/noise fields and anatomical masks were verified unchanged.

**Evidence/archive:** [E021 metadata](experiments/records/E021.json); OneDrive
`results/motor_curriculum/head-sweep-20261106-local-seed0` contains the oracle,
all three fits, command/source/environment snapshots, input copy, checkpoints,
trajectories, plots, lineage mapping and checksums. The earliest oracle
manifest retains unused steering fields from the shared entrypoint; startup
source confirms they were ignored by the head target, and subsequent manifests
remove them. Raw records are preserved unchanged. Dirty startup source
snapshots identify the exact trial code. Upstream forward-training history
remains in earlier archives; included replay inputs cover these head fits.

**Validation/status:** 167 software tests passed. Local MOT-09 stationary scan
is demonstrated in one driven full-network policy/parent lineage, not three
independent seeds. Template fit is close to the 0.25 boundary in one held-out
sequence; multi-seed reliability is not established. This is not biological
calibration, endogenous rhythm or sensor-driven head selection. The four-seed
480-minute mbslnx2 command was published with implementation `d7c668e`; its
actual remote launch/outcome is not observed by the agent. All earlier skills
remain unchanged.

### E022 — 2026-11-07: four-seed server head-sweep success reported

**Method/run:** E021 stationary head-sweep protocol; reported directory
`/home/carsonc/r-elegans-results/head-sweep-20261107T003628901581Z`.
The intended budget was 480 minutes; actual elapsed times and exact execution
revision were not printed in this user-supplied summary and remain unverified.

**Reported outcome:** all four seeds have passed=true, physical_passed=true,
tuning_passed=true, status=completed and error=null. All 12 seed/pacing-rate
conditions pass. The selected checkpoint is not necessarily the final update:

| Seed | Updates executed | Selected update | Physical / tuning |
| --- | ---: | ---: | --- |
| 0 | 132,777 | 76,750 | Pass / pass |
| 1 | 132,428 | 131,500 | Pass / pass |
| 2 | 133,469 | 16,750 | Pass / pass |
| 3 | 132,004 | 52,750 | Pass / pass |

Total executed updates: 530,678. Across the three pacing periods (2.4/3.0/3.7 s),
head-angle peak-to-peak excursion is 0.16349–0.18499 rad (about 9.4–10.6 degrees).
Posterior/anterior dynamic joint RMS ratios are 0.03965–0.04958, below the
0.25 limit. Maximum cycle centroid drift is 5.124e-6 body lengths, below the
1e-4 limit. These are normalized-model measurements, not animal calibration.

**Evidence:** [E022 metadata](experiments/records/E022.json), transcribed from
the user's inline output. A normalized transcript copy is stored externally;
its checksum is not presented as a hash of the original stdout bytes. Raw
server manifests, command/environment snapshots, trajectories and checkpoint
hashes still require archive retrieval. No missing times or source revisions
are reconstructed from iteration counts or directory names.

**Interpretation/decision:** stationary, externally paced head scanning now
has reported four-seed replication with tuning and physical criteria met.
The trunk remains quiet by the defined bending/drift measures. Preserve each
selected actor and its lineage; do not overwrite prior locomotion/steering
skills or collapse this into a shared-weight controller. This does not show
autonomous rhythm, sensory selection, calibrated biological behavior or safe
runtime skill handoff. Next proposed standalone coverage target: movement
intensity/speed modulation (MOT-07); no new training is launched in this entry.

### E023 — 2026-11-07: biological self-drive and the missing feedback loop

**User question:** how does the real network sustain movement without our
external pacing? **Action/status:** primary-literature review and code-path
inspection, not a new training experiment or implemented autonomy mechanism.

**Evidence from the animal:**

- Distributed forward rhythm generation is supported by perturbation/ablation
  experiments: [Fouad et al. (2018)](https://elifesciences.org/articles/29913).
  A single master head clock is not a sufficient general description.
- Descending AVB electrical input can shift B-type motor neurons into an
  oscillatory regime; proprioceptive coupling coordinates propagating forward
  bends: [Xu et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29686107/).
  Their model uses sustained depolarized AVB drive, not a supplied sinusoid.
- A-class motor neurons can generate intrinsic oscillatory activity for
  backward locomotion, with UNC-2 calcium-current involvement:
  [Gao et al. (2018)](https://elifesciences.org/articles/29915).
- Body bending provides motor-circuit proprioceptive input that coordinates
  waves; B-type neurons transduce such signals:
  [Wen et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/).
- State selection is more complex than a symmetric AVA/AVB switch. Tonic and
  phasic AVA effects act on different timescales:
  [Meng et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38598630/).
  Additional circuit context and physiological conditions matter; this review
  does not claim a complete, uniquely established locomotor mechanism.

**Current implementation check:** `commanded_muscle_sequence` carries neural
voltage and artificial command only. `muscle_sequence_body` subsequently
simulates the resulting muscle trace. `full_network_actor` accepts applied
currents (or engineered chemical observations); these motor drivers do not
compute body-dependent proprioceptive input. Neural recurrence is present,
but motor execution is open-loop with respect to body feedback. Differentiating
the body loss teaches parameters; it does not feed sensed curvature back at
execution time. The generic graded-potential full actor also lacks validated
A/B-specific pacemaking-channel physiology. Its nonlinear recurrent dynamics
might support oscillations; their existence, robustness and biological
mechanism have not been established by the driven skill tests.

**Interpretation:** wiring/counts alone do not specify mechanosensory
transduction, cell-specific dynamics or extrasynaptic state modulation. A
biological autonomous motor model must account for these functions. Preserved
driven skills remain useful reference trajectories, not proof of self-drive.

**Proposed future milestone, not implemented/run:** couple neural and body
state in one execution loop, provide evidence-grounded proprioception, and
test coordinated movement under a constant permissive command or brief
trigger with no periodic input. Separate rhythm generation, propagation,
load/posture robustness, stopping and direction selection; retain numerical
controls and old skill archives. Autonomous sensory-guided task selection
is a further milestone. No learning objective, priority or model was changed
by this literature review; H1.1 coverage-first records remain in force.

### E024 — 2026-11-07: intrinsic motor-neuron dynamics (local assay completed)

**Question/authorization:** user requests intrinsic oscillation in the motor
neurons. Implement an opt-in reduced calcium/potassium conductance model for
DA/VA (autonomous regime) and DB/VB (steady-drive-dependent regime), within the
full 302-node actor. This is a qualitative dynamical hypothesis, not a fitted
UNC-2 model or measured channel inventory. Evidence: Gao 2018 and Xu 2018,
linked in E023. No sinusoid, phase clock, or periodic forcing in the assay.

**Plan:** membrane voltage plus a slow potassium activation state; instantaneous
calcium activation provides regenerative inward current. Fixed Morris–Lecar
type demonstration coefficients and an explicit normalized-voltage transform;
no claim of worm-specific mV/ms calibration. B steady injected current is a
permissive-drive proxy, not an implemented AVB gating mechanism. Preserve the
existing default actor/checkpoints exactly; new dynamics require explicit opt-in.
No parent fitted checkpoint: use bundled topology with initialization seed 0.
No optimization: all parameters fixed in the experiment; existing anatomical
chemical/gap masks and NMJs retained in full-network integration.

**Planned acceptance:** isolated A cells sustain multiple cycles after transient
with zero applied current; isolated B cells relax without drive, oscillate under
constant drive, and relax after withdrawal. Removing the regenerative calcium
conductance abolishes the sustained A rhythm. Check timestep convergence,
finite differentiable full-network muscle outputs, canonical cell selection,
and unchanged legacy behavior. These are cellular/software gates, not evidence
of coordinated propulsion or biologically fitted membrane physiology. Exact
assay commands, runtime/source records and external output paths follow below
when run. No server training scheduled by this entry.

**Implementation/result (local verification):** `brain/intrinsic_motor.py`
implements the planned reduced active-membrane dynamics, opt-in through
`make_full_network_actor_interface(intrinsic_motor=IntrinsicMotorConfig())`.
The equation form follows the calcium/potassium mechanism of
[Morris and Lecar (1981)](https://pubmed.ncbi.nlm.nih.gov/7260316/), originally a
barnacle-muscle model; the illustrative numerical coefficients are NOT fitted
C. elegans channel properties. All 302 neurons remain, including 21 DA/VA and
18 DB/VB active-membrane cells. A bias is fixed at 90 model-current units;
B tonic assay current is 90 (equivalent to 0.1 applied normalized actor input).
State comprises voltage and slow potassium activation, without a phase clock.
No calcium concentration, UNC-2-specific activation kinetics, neuromodulation,
or body proprioception has been added. Existing AWC/RMD models are unchanged.

**Commands actually run (Mac, no server or optimizer):**

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_intrinsic_motor --output-dir /tmp/r-elegans-E024-intrinsic-v1
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_intrinsic_motor --output-dir /tmp/r-elegans-E024-intrinsic-final
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q tests/test_intrinsic_motor.py tests/test_full_network.py tests/test_motor_curriculum.py
```

The second assay adds connected-cell summary diagnostics, with unchanged
equations/coefficients. Both passed all nine assay gates. Each trajectory is
30 normalized simulation seconds, dt=0.02, four RK4 substeps; comparison uses
eight substeps, with a maximum state difference of 1.76914e-5 (<0.002 gate).
In the final ten seconds A and tonic B each have five upward voltage crossings
and 0.827397 normalized peak-to-peak voltage; quiet B, withdrawn B (withdrawal
at 15 s) and calcium-ablated A have zero final-window voltage variation.
A starting-voltage perturbation (-0.35 to -0.34) retains five final-window cycles.
These are simulation-clock results, not measured animal frequencies or voltages.

**Connected network finding/limitation:** with anatomical recurrence enabled,
seed 0, and zero applied input, 19/21 A cells have at least three final-window
upward crossings; two do not. A voltage peak-to-peak spans 0.004644–0.811298.
B cells have no upward crossings and 0.000466–0.010382 peak-to-peak activity.
Coupling can suppress intrinsic oscillators; the isolated-cell result does
not promise every cell oscillates under arbitrary synaptic load. Full-network
execution/muscle outputs are finite. No body motion was assessed or optimized.
There is no fitted AVB-to-B enabling mechanism or coordinated autonomous gait.

**Software:** targeted suite passed 20 tests, including canonical masks,
calcium ablation, substep convergence, differentiable recurrent muscle output,
frozen NMJs, legacy integration identity and NPZ roundtrip. Full suite:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q` — **171 passed
in 50.35 s**. Active-membrane coefficients are fixed;
selected generic leak/tau fields no longer affect selected cell voltage.
Legacy trainers remain on their previous dynamics. Any future intrinsic-mode
checkpoint must retain its config: the legacy NPZ format does not encode it.

**Artifacts:** both invocations, exact runtime/source snapshots, summaries and
raw trajectories copied to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E024/`; 261 copied files
were SHA-256 verified. Manifest `SHA256SUMS.json` hash:
`5975d7efd692f186a26e5a531e493f549d7e38534298767ccfc1548e8514a606`.
Small results and artifact hashes: `experiments/records/E024.json`.
Run records honestly retain the pre-commit dirty-tree state/source snapshot.
Local copy verification does not prove OneDrive cloud synchronization.

**Decision:** reduced intrinsic excitability is now implemented and cell-assayed.
The next mechanistic work is evidence-grounded descending control and embodied
coordination/proprioception, followed by propulsion assays. No old skill has
been overwritten or claimed to transfer, and H1.1 skill coverage remains useful.

### E025 — 2026-11-07: conductance fitting and synaptic inference tools (local results)

**Authorization/question:** fit ion channels and synapses after E024. First
audit the available measurements, then implement physical-unit clamp fitting
with explicit provenance, held-out sweeps and identifiability diagnostics.
No behavior-only optimum will be described as a measured channel inventory.

**Local inventory:** seven digitized trace files cover AWA (two files), AWC,
AVA, AIY, RIM and VD05; no DA/VA/DB/VB recordings and no paired synaptic traces
found under the configured raw/physiology or raw/functional directories.
Existing AWCON/RMD parameter sets are published fits, not new recordings.
Pinned upstream BAAIWorm commit
`b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc` source was read as text only:
`eworm/model_figure/single_neuron_features.py` and `fig_single_neuron.ipynb`.
The source converts voltage-clamp response A to nA (factor 1e9), uses holding
-60 mV for VD05, steps -10..40 mV and duration 1210 ms. VD05 file headers
agree with those levels. AWCL headers (0,10,30,..110 mV) disagree with upstream
simulation levels (-10,10,..110); block automatic AWC fitting pending resolution.
The holding voltage is an upstream simulation assumption, not independently
verified original experimental metadata. Retain this limitation with results.

**Planned first fit:** VD05 candidate using existing 17-current/35-gate library,
AWCON prior (explicit cross-class transfer), fixed kinetics/reversals/calcium
parameters and nonnegative maximal conductances only. Real unit conversions
are explicit. Fit interior pulse samples excluding edges/capacitance, with
baseline subtraction matching padded zero baselines. Hold out complete voltage
steps (0 and 30 mV) before optimization. Report train/validation RMSE, numerical
substep convergence and unregularized sensitivity rank; no unique-composition
claim. Poor held-out prediction remains a failed candidate, not a reason to
relax acceptance after fitting. Predefined provisional adequacy gate: held-out
RMSE <=20% of the training current RMS; convergence <=1% of that scale.

**Synaptic stage:** implement a constrained conductance fit for recordings of
pre/post voltage and residual postsynaptic current with known reversal/release
parameters, fixed anatomical topology and shared gap-junction pairs. Test with
synthetic data for recovery, but do not claim a real synaptic fit without paired
measurements. This assay is distinct from fitting effective synapses to behavior.
Do not overwrite E024, old skill weights, or the bundled published presets.
Save inputs, source/environment, candidates and diagnostics externally.

**Implemented:** `brain/conductance_fitting.py` supplies fixed-kinetics voltage
clamp prediction in mV/ms/pA/nS, positive maximal-conductance estimation,
data-only sensitivity diagnostics, and sparse nonnegative synaptic conductance
estimation. `scripts/fit_membrane_synapses.py` adds audited inputs, complete
sweep/trial holdouts, candidates, diagnostics and source/environment capture.
SciPy is declared in the optional `fit` dependencies. No candidate is installed
in the running actor; E024 coefficients and all existing policies are unchanged.

**Executed channel commands (Mac):**

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_membrane_synapses channels --data-root /Users/carsonc/OneDrive/r-elegans --output-dir /tmp/r-elegans-E025-vd5-fit-v1 --max-evaluations 100
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_membrane_synapses channels --data-root /Users/carsonc/OneDrive/r-elegans --output-dir /tmp/r-elegans-E025-vd5-fit-final --max-evaluations 100
```

The final run corrects sensitivity-rank tolerance to use the Jacobian's actual
float32 precision; coefficient fitting is unchanged. The first run's apparent
rank 17 and ~9.14e9 condition number were not evidence of identifiability.
Both run records are retained. VD05 trace hash matches the original manifest;
the AWCON prior catalog is copied with its hash. Existing 35 gates plus bulk
calcium evolve under piecewise constant voltage; exponential frozen-target
updates (eight substeps) are compared against sixteen. Sample interval 2 ms
(fivefold decimation of 0.4 ms digitized samples); loss uses 5..1200 ms of the
1210 ms pulse. Holding-current subtraction uses -25..-5 ms model baseline.
Train levels -10,10,20,40 mV; validation levels 0,30 mV. Only 17 maximal
conductances are optimized, all other fields fixed. Positive g bounded
1e-6..100 nS, log-conductance prior weight 0.001; prior zeros receive a 0.01 nS
floor, explicitly allowing unsupported channel expression in this exploratory
candidate. No expression-based composition claim follows. Deterministic SciPy
least-squares uses JAX forward sensitivities; 18 evaluations, optimizer success.

**Measured-trace candidate outcome:** training RMSE 59.4297 -> 11.8596 pA;
held-out RMSE 56.9954 -> 8.44905 pA, against training RMS current 138.614 pA.
Refinement RMS discrepancy 0.000515587 pA. All three provisional gates pass.
Data-only sensitivity has numerical rank 16/17 and practical rank 15/17 at
relative threshold 1e-3. Baseline-subtracted NCA and leak currents have the same
ohmic voltage dependence, so these recordings cannot separate their strengths.
Other compensating contributions and transferred kinetics further limit the
inference. Plots show residual low-voltage adaptation mismatch despite passing
the aggregate criterion. This is a conditional VD05 current-response candidate,
not a unique composition, a fitted motor oscillator, or a calibrated full worm.

**Synaptic software control:** constructed an explicitly synthetic isolated
AVAL/LUAL two-cell clamp fixture on real bundled topology, seed 17, three
independent trials of 80 voltage samples, fixed sigmoid release (-35 mV,
slope 6 mV), zero chemical reversal, chemical strengths 0.3 nS in each existing
direction and a shared gap strength 0.12 nS. Current generated from analytic
outward-positive chemical/electrical expressions; no intrinsic current present.

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_membrane_synapses synapses --recordings /tmp/r-elegans-E025-synaptic-fixture/recordings.npz --metadata /tmp/r-elegans-E025-synaptic-fixture/metadata.json --heldout-trial 0 --output-dir /tmp/r-elegans-E025-synaptic-control
```

All three known conductances recover; train RMSE 1.49e-14 pA, held-out trial
RMSE 1.39e-14 pA, sensitivity rank 3/3. This validates the inference code under
its assumptions, **not biological synaptic values**. No real synaptic fit was
run: paired pre/post clamp currents are absent locally. CLI requires units,
subtraction provenance, known reversals/release, and explicit isolation for
subsets. For large designs it reports skipped dense sensitivity diagnostics
rather than claiming unique edges. A subsequent CLI guard rejects nonfinite
held-out currents and returns nonzero on optimizer failure.

**Verification:** targeted channel/synapse and single-compartment tests:
12 passed. Full suite: `MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q`
— **175 passed in 53.23 s**, before the final CLI guard-only edit; the targeted
suite was repeated afterward with **12 passed in 8.05 s**. No models
were trained with new dynamics or replaced.

**Artifacts:** originals, both channel runs, the synthetic fixture/control,
clamp-fit PNG/PDF with plotting source, upstream pinned protocol-source text,
dataset audit and exact invocation/environment/source snapshots are copied to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E025/`. Verified SHA-256
for 418 copied files; manifest `SHA256SUMS.json` hash:
`c79e7c396516deed199b5cb7dcb76f05fad8db9989f5cc34b44890b3b6593100`.
Local copy verification does not prove cloud synchronization. The dirty source
snapshots are authoritative for the runs; later documentation/CLI guards do not
retroactively change them. Small summary/hashes: `experiments/records/E025.json`.

**Remaining scientific choice:** direct motor-cell data are missing. Asked the
user whether to infer their unknown channel/synaptic values from established
motor behaviors with physiological priors. No such behavior-inference fit was
executed in E025, and inferred parameters must retain that evidence label.

### E026 — 2026-11-07: channels in all neurons and calcium-recording fitting

**User direction:** fit calcium recordings from current parameters; give every
neuron membrane channels, without forcing graded-potential cells to spike.
Use a versioned conductance-network branch, preserve current fitted synaptic
parameters and explicit migration assumptions, and do not overwrite old skills.
Joint calcium fitting supersedes any implication that only paired clamp
recordings can constrain the network. Chloride/mixed-cation receptor currents
are relevant in addition to K/Ca/Na-permeable membrane currents. No classical
voltage-gated sodium spike current should be invented for worm neurons.

**Plan/data:** Atanas & Kim 2023 identified baseline calcium/behavior recording,
processed HDF5/NeuroPAL labels from Zenodo record 19388374, DOI reference in
WormWideWeb. Download processed archive (568,776,589 bytes), not image volumes,
to external OneDrive and verify published MD5 plus local SHA-256. Audit label
confidence, timestamps, reporter and trace normalization before fitting. Select
canonical, uniquely labeled neurons only; exclude unresolved LR identities.
Do not treat calcium fluorescence as membrane voltage or substitute behavior
features as direct neural inputs without labeling that assumption.

**Model plan:** vectorize the existing 17-current/35-gate model over all 302
cells with intracellular calcium and a saturating, low-pass indicator. Existing
AWCON/RMD presets are measured-model priors only for their classes; use explicitly
transferred weak priors elsewhere, not invented cell-specific measurements.
Preserve parent chemical/gap relative strengths and release fields through an
explicit normalized-to-physical-unit mapping. Non-spiking cells have channels
but no hard spike/reset mechanism or required oscillatory target. Unknown
receptor polarity remains unresolved; parent reversal priors are assumptions.

**Planned first assay:** initialize from a preserved current full-network actor
checkpoint; fit bounded channel conductance multipliers and existing anatomical
synaptic strengths against an identified short calcium window. Preserve initial
model/config/parent hashes, optimization state and final candidate. Reserve a
later contiguous interval for validation; determine scaling/observation gain
using training data only. No teacher forcing from target fluorescence and no
per-timestep free neural currents. Verify all 302 have active channel states,
gradients, numerical convergence and checkpoint roundtrip. Report initial/final
train/held-out error even if no generalization; improvement on one short window
is not electrophysiological identifiability or behavioral fidelity. Exact
methods/commands/results to follow; no server launch assumed.

**Implementation:** `brain/conductance_network.py` now vectorizes all 17 current
components/35 gates over all 302 cells, with bulk calcium and an effective
Hill/low-pass indicator. No spike/reset rule. `data/calcium.py` loads processed
HDF5 traces and one-based NeuroPAL label mappings, excluding noncanonical,
ambiguous, low-confidence and duplicate labels. `scripts/fit_calcium_network.py`
fits channel multipliers plus existing chemical/gap strengths. All cells and
anatomical edges remain simulated, even when not observed. Channel coefficient
and effective synaptic inference from calcium is conditional and non-unique.
The preserved NMJ parent is copied with each run, but no body-motion assay is
part of this fit and the new brain has not replaced the operating motor actor.

**Data acquisition:** public website API returned HTTP 403; switched to the
public Zenodo source repository rather than requiring credentials. Downloaded
processed archive 568,776,589 bytes, SHA-256
`c93e0cea3941b9e72bd7f09a36d7d26c8c5eca0e26fc2694dac739bbfa51c3c8`, and
NeuroPAL labels 16,169 bytes, SHA-256
`b371e3dae9a0f78c6f6b9ba63c7076a55aa8fccecb650f579f53918d10402684`.
Both passed published Zenodo MD5 verification. Extracted only the selected
baseline HDF5 using an exact basename and streamed contents, not arbitrary
archive paths. Its SHA-256
`aabca135ef82e4be6821bf515cd2e936ae2f78014bf2979ed8a10aeaa7c4ac11`
matches the pinned paper manifest (WormWideWeb-data commit
`0422546192ef58d0de4740a0aa91fd38de22da2e`). Sources and Zenodo record retained
under external `raw/calcium/atanas_kim_2023`. H5py 3.16.0 installed in .venv;
`fit` extra now declares HDF5 reader/optimizer dependencies.

**Recording:** `2022-06-14-01`, 1600 frames x 109 extracted ROIs; 79 annotated
labels, 55 retained at confidence >=4 with exact canonical identities and finite
nonconstant training traces. Excluded 24 labels; this is head/pharyngeal coverage,
not a simultaneous recording of all 302. Raw ratiometric fluorescence, not
the full-session z-scored field, is standardized using training moments only.
GCaMP7f is nuclear-localized in the source experiment; our effective Hill
Kd=0.3 uM, exponent 2 and low-pass tau=0.5 s are uncertain demonstration
observation priors, not calibrated GCaMP7f or nuclear transport kinetics.

**Parent/migration:** preserved local head-sweep refined actor:
`results/motor_curriculum/head-sweep-20261106-local-seed0/refined/actor_best.npz`
in OneDrive, SHA-256
`82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0`.
Existing chemical/gap raw values and anatomical masks are copied exactly.
Physical unit migration: 100*v-25 mV, 0.03*softplus(weight) nS, capacitance
1000*tau*0.03 pF. Because learned normalized intrinsic fields can be far outside
plausible physical ranges, bound initial/leak voltage [-80,-35] mV, effective
synaptic reversal [-85,0], release threshold [-65,-20], slope [2,20], capacitance
[1,20] pF. These bounds and g-unit are initialization hypotheses, not measured
conversion factors. An exploratory unbounded migration reached -208.9 mV in a
short smoke check; it was rejected before fitting. Fixed bias balances initial
intrinsic currents only, not future or recurrent currents. No E024 clock scale
or artificial periodic forcing is carried into the physical membrane equations.

Each RMD cell starts with the RMD conductance library; all others start with
AWCON conductances scaled to 1%, with 1e-5 nS floor and a 0.03 nS leak in every
cell. Capacitance/leak reversal come from the parent mapping, so even RMD is
not an unchanged published whole-cell model. AWCON fate is not assigned as a
known left/right identity. These are transferable starting priors, not claims
of true cell-specific expression or an exhaustive worm channel inventory.

**Fitting method:** all 302*17 log-conductance multipliers, existing chemical
edges and symmetric gap strengths may change. Multipliers bounded 0.1..10;
synaptic raw fields within parent +/-2; absent edges remain absent. Adam LR .01,
global gradient clipping 1; parameter-deviation prior weight .001. Kinetics,
reversals/release, calcium handling, biases and observation parameters fixed.
The neural trajectory receives no fluorescence/behavior teacher forcing,
sinusoid, sensory replay or arbitrary learned input sequence. Actual animal
inputs and unobserved initial/internal state remain a limitation. Full-network
source/units/fixed fields are stored in versioned candidate checkpoints, with
canonical cell IDs in the finalized format. The initial prototype checkpoint
preceded the ID field; its exact source snapshot must be used to replay it.

16 frames after the first reference timestamp (~9.67 s), first 12 train and
last four held out. Nonuniform source intervals are integrated directly, with
a common substep count chosen from the maximum interval. The fixed observation
offset is the initial model's training mean; gain is 1/max(initial model training
std,0.01), capped at 100 by construction. Observed training mean/std define
target standardization; no held-out data calibrate the readout. Baseline is
the observed training mean (zero in these standardized units).

**First executed fit:**

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_calcium_network --recording /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/2022-06-14-01-data.h5 --labels /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/neuropal_label.json.bz2 --uid 2022-06-14-01 --initial-actor /Users/carsonc/OneDrive/r-elegans/results/motor_curriculum/head-sweep-20261106-local-seed0/refined/actor_best.npz --output-dir /tmp/r-elegans-E026-calcium-v1 --frames 16 --iterations 10 --max-step-ms 2 --minutes 15
```

Locally completed, 22.55 s. Standardized training MSE 1.050135 -> 1.049237;
held-out MSE 4.353773 -> 4.365976 (worse), training-mean held-out baseline
4.387815. Training-mean training MSE is 1.0, so even the improved training fit
does not beat that trivial baseline. Maximum log-channel/chemical/gap parameter
changes ~0.101/0.101/0.100; channel and synapse optimization is active.
Voltage range -80.25..-30.09 mV; finite trajectories. 321 substeps, maximum
step ~1.999 ms; doubling substeps changes standardized predicted fluorescence
by 0.01747 RMS. This is a failed predictive fit and an insufficiently resolved
numerical candidate for mechanistic interpretation, not evidence of successful
physiology or of network insufficiency.

**Finer run/verification:** rerun from the SAME parent/priors (not a continuation),
50 updates, max step 1 ms, output `/tmp/r-elegans-E026-calcium-fine`; exact command
captured by its source/environment record. Held-out values are not used to
select parameters. Outcomes and archival hashes follow below. Unit tests first
found 0.657 mV coarse/fine discrepancy with 0.5/0.25 ms substeps in an active-cell
test; retained 0.2 mV criterion and refined to 0.125/0.0625 ms (0.144 mV discrepancy).
Targeted suite: five tests pass, covering all-cell states, exact old current
equations, parent topology, differentiable channel/chemical/gap effects, numerical
controls, passive graded responses without spikes, train-only normalization,
identity exclusion and full checkpoint roundtrip. No broad spiking-cell
classification is inferred from these simulation tests.

**Additional literature correction prompted by user:** patch-clamp does exist
for selected neurons, including VA5/VB6/VD5, AVA/AIY/RIM. The current local
inventory was not evidence of absence in the literature. Nicoletti et al. 2024
(doi:10.1371/journal.pone.0298105) supplies more class models and adds EXP-2,
UNC-103 and KQT-1 currents; those equations/priors are NOT yet ported. The
current all-cell library is intentionally labeled incomplete. Clear action
potentials in AWA/AVL/DVB, RMD plateaus, graded responses and motor oscillations
are different evidence categories. A complete 302-cell patch/phenotype atlas
was not found in the sources reviewed. Failure to observe spikes is not proof
of impossibility under all conditions. Literature and code gaps remain explicit.

**Interpretation:** adding channels to every neuron is implemented. Fitting
actual calcium is implemented and attempted, but neither the first small
training improvement nor the existence of channels establishes accurate activity,
unique compositions, preserved skill execution, or physiological spike phenotypes.
No candidate replaces the original actor. Use improved class priors/observation
models and multiple recordings/perturbations before large training or promotion.

**Finer fit outcome (local verification):** completed 50 updates in 189.88 s,
642 substeps per frame (maximum step 0.999512 ms), with the same parent and
priors. Standardized training MSE 1.050225 -> 1.036041; held-out MSE
4.353875 -> 4.377726 (worse). Training-mean baseline remains 1.0 train and
4.387815 held out. Thus the model still does not beat the constant baseline
on training or improve future-frame prediction relative to its initial state.
Maximum log-channel/chemical/gap parameter changes 0.607/0.582/0.557; both
membrane and connection parameters really learn. Voltage range -80.37..-17.46
mV, finite all-cell traces. Doubling substeps changes standardized predicted
fluorescence by only 0.00013550 RMS, so integration discrepancy is much smaller
than the statistical error at this candidate. No claim that this proves
spike-level fidelity, calibrated calcium handling or correct nuclear transport.
Plots show the main failure clearly: many model traces remain almost flat while
recorded activity varies. No candidate is promoted. Initial/final networks,
fixed fields/unit configuration/IDs, copied parent/data, final optimizer state,
source/environment, exact commands, complete logs and trajectory arrays retained.
No RNG is consumed in these deterministic fits; the parent hash defines the
starting weights. The CLI starts fresh optimizers and is not exact mid-run resume.

**Software verification:** new targeted suite 5 passed; full repository suite
**180 passed in 62.47 s**. The only later source change adds the already-used
Flax serialization dependency explicitly to the `fit` extra. Software tests
do not convert either calcium-fit failure into physiological validation.

**Archive:** both complete run directories, copied source HDF5/labels/parent,
initial/final network checkpoints, optimizer states, full stdout logs, numerical
trajectories, example PNG/PDF with plotting source, exact source/environment
snapshots and upstream metadata copied to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E026/`.
All 303 copied files verified by SHA-256. Manifest `SHA256SUMS.json` hash:
`b549d5839623663a0ac23c7699dde8c85dee83ad20ebf0b95128d37d1ecd7f4e`.
The 569 MB source archive remains separately under external `raw/calcium`;
it is not duplicated into Git. Small summary/hashes: `experiments/records/E026.json`.
Run source records retain the actual dirty pre-commit state. Verified local
OneDrive copies do not establish completed cloud synchronization.

### E027 — 2026-11-07: sustained conductance dynamics (local qualification)

**User direction:** get enough channel kinetics for self-sustained dynamics,
before further activity fitting. Investigate channel balance/kinetics rather
than assuming adding channel names guarantees oscillations. All 302 channel
states and old skills remain preserved. No periodic command or target-derived
current may masquerade as self-drive.

**Evidence/plan:** Nicoletti 2019 reports stable periodic solutions in its RMD
17-current model when leak/CCA1 balance changes (Fig.11, ~200 ms periods, ~5 mV
oscillations around -30 mV). First reproduce a sustained regime using existing
kinetics and published parameter scales; distinguish this modeled RMD regime
from an experimentally fitted DA/VA pacemaker. Gao 2018 implicates UNC-2 in
A-motor rhythm; a CCA1-dependent RMD cycle is not a substitute for that phenotype.
Nicoletti 2024 motor source inspected at commit
`84726a72a37230f02cecc1814c896232ead195ec`; motor models include isolated SLO2
with calcium-dependent activation and fitted recovery multiplier absent from
our nanodomain-only SLO2 library. Source text downloaded externally, not run.

**Planned first experiment:** isolated RMD prior, leak grid 0.03..0.12 nS,
CCA1 0.3..3.1 nS, other published conductances/capacitance/calcium settings fixed;
no applied or compensation current. Start in depolarized basin (-35 mV) to test
whether activity persists, then repeat from rest/perturbed states. Initial
voltage is a condition, not pacing. 10 s screening with >=5 s washout, then
longer confirmation for candidates. Require nondecaying amplitude across final
windows, multiple cycles, bounded voltage/calcium and integration convergence;
remove regenerative calcium current to test causality. Retain stable/non-
oscillatory controls and negatives. After cellular qualification test the full
anatomical network, noting coupling may quench or alter oscillations. Do not
claim a propagating gait or calcium-recording fit from activity alone.

**Implementation plan:** add explicit optional per-cell kinetic parameters and
isolated SLO channel kinetics based on source evidence where necessary; version
new checkpoints and preserve old behavior/config. Record which kinetic fields
are published versus hypothetical/tuned. Exact run records and outcomes follow.

**Screening correction:** the first 49-case grid (CCA1 <=3.1 nS) produced no
persistent cycles. Inspection of the original Fig.11 image resolved mathematical
labels missing from the web text: its CCA1 axis reaches ~70 nS, while its leak
axis is 0.03..0.12 nS. The text's approximate periodic interval was not an
appropriate CCA1 grid. Repeat with CCA1 [0.3,3.1,10,20,30,50,65] nS and the
same leak levels/initial state/protocol; preserve the failed original screen.
These large conductances are a modeled bifurcation regime, not fitted animal
values or a justification for assigning them to motor neurons generally.

**Implemented/decision:** the current 17-current/35-gate machinery is already
sufficient for persistent cycles in an appropriate conductance regime. Added
`brain/rhythm_qualification.py` and `scripts/assay_channel_rhythms.py` for isolated
screens, causal/initial-state controls, and anatomical-network tests. No new
rate constant, phase oscillator, periodic current or synthetic waveform was
introduced. The planned isolated-SLO/kinetic extensions remain future work for
better motor-class fidelity; they were not required for this dynamical proof.
This is not claimed to reproduce the full published bifurcation diagram or to
fit an animal's channel composition. E024's reduced oscillator is separate.

**Run commands (Mac, external output; each invocation has its own source/env record):**

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_channel_rhythms --output-dir /tmp/r-elegans-E027-rmd-grid
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_channel_rhythms --output-dir /tmp/r-elegans-E027-rmd-grid-corrected
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_channel_rhythms --output-dir /tmp/r-elegans-E027-confirm --confirm --duration 30
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_channel_rhythms --initial-network /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E026/r-elegans-E026-calcium-fine/network_initial.npz --coupling-scale 1 --duration 20 --max-step-ms 0.1 --output-dir /tmp/r-elegans-E027-connected-default
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assay_channel_rhythms --initial-network /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E026/r-elegans-E026-calcium-fine/network_initial.npz --remove-rmd-cca1 --duration 20 --max-step-ms 0.1 --output-dir /tmp/r-elegans-E027-connected-cca-removed
```

**Screen outcomes:** original grid 0/49 pass, all finite; corrected grid 6/49
pass. Passing candidates have leak 0.03/0.045 nS, CCA1 10..50 nS, with various
combinations. No relaxation of persistence criteria was used: voltage bounded
(-120,80) mV, final half divided into five windows with amplitude >=2 mV in
each, final/first late amplitude 0.8..1.2, >=5 late peaks (prominence >=1 mV).
These gates target a fast RMD-mode assay, not all possible neural timescales.

**30 s isolated confirmation:** selected leak 0.045 nS, CCA1 30 nS, all other
RMD preset conductances/passive/calcium values intact. Bias and synapses zero.
Starting voltages -35/-34/-70 mV all sustain cycles; final 15 s contain 69 peaks,
~22.03 mV peak-to-peak, median period 0.220 s, late amplitude retention within
0.002% of one. Starting from -70 mV includes an initial large transient that is
excluded by washout; the subsequent late cycle is the same regime. CCA1=0 and
ordinary RMD (CCA1=3.1, leak=0.4 nS) controls settle; neither passes. Calcium is
finite, 0.0502..1.0292 uM across all confirmation conditions. The first confirmation
summary accidentally labels the ordinary control leak as 0.045 because its grid
metadata preceded the explicit 0.4 override. Actual simulated leak is 0.4 in the
captured source; a separate metadata-correction record retains that fact. Raw
summary/arrays are not rewritten. Future CLI output reports the corrected pair.

**Full-network outcome:** parent E026 initial network (before its unsuccessful
calcium fit), with six RMD cells reset to the selected full RMD profile and
published passive settings; their log multipliers set to zero and compensation
bias removed. Other 296 cells, all chemical/gap raw values/masks, and global
synaptic unit 0.03 nS unchanged. Other cells retain their constant parent intrinsic-
current compensation, so this is unpaced activity under steady conditions, not
a claim that every current in the network is zero. No applied time-varying input,
teacher signal, phase variable or body feedback is used.

In 20 s simulations, RMDL/RMDR/RMDVL pass late persistence gates at original and
refined steps. Each has ~46 peaks in the final ten seconds, median period
0.216 s; late amplitudes ~21.17,20.35,11.64 mV. Calcium peak-to-peak amplitudes
in the final ten seconds are 0.2643,0.2537,0.1278 uM. The other RMD cells are
quenched or fail the gates. Full-network CCA1 removal from the six RMD cells
eliminates all qualifying rhythms (0/302), with finite calcium. This is a positive
model-level causal control, not a biological CCA1-knockout prediction or evidence
for the UNC2-dependent DA/VA phenotype. All full anatomical weights remain
present; they are not fitted in these rhythm assays.

**Numerical controls:** isolated step 0.05 vs 0.025 ms; network step 0.1 vs
0.05 ms. Passing cells remain passing at refinement. Small frequency errors
accumulate phase over 20–30 s: raw maximum voltage differences are 17.42 mV
isolated and 19.51 mV connected, so pointwise trajectories are not identical.
Post hoc cycle analysis aligns and averages the last five peak-to-peak cycles,
retains raw errors, and requires period change <1%, amplitude change <2%,
phase-aligned waveform RMS <0.5 mV. Isolated positives: relative period change
0.183%, amplitude <=0.126%, waveform RMS <=0.167 mV. Connected positives:
period <=0.371%, amplitude <=0.274%, waveform RMS <=0.121 mV. All pass. This
supports a persistent oscillatory regime, not exact long-horizon phase prediction.
Analysis JSONs are labeled post hoc and kept alongside original trajectories;
these tests were not used to lower the original persistence gates.

**Calcium-pipeline integration:** fitter now accepts `--initial-network`,
preserving fixed channels/kinetics/config instead of rebuilding from a generic
parent. `--qualify-rhythms` independently runs a 20 s unpaced assay before/after
fitting and returns nonzero if no RMD cell qualifies. It does not install a model
or assert animal fidelity. Example actually run:

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_calcium_network --recording /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/2022-06-14-01-data.h5 --labels /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/neuropal_label.json.bz2 --uid 2022-06-14-01 --initial-network /tmp/r-elegans-E027-connected-default/network_candidate.npz --qualify-rhythms --experiment-id E027 --output-dir /tmp/r-elegans-E027-calcium-integration --frames 8 --iterations 1 --learning-rate .001 --max-step-ms .1 --minutes 15
```

This is a one-update integration/control test, not a meaningful fitted activity
model. Parent SHA-256
`870444a4a1122818d9124af242242c506778b25c0079d4ee8a441cf8c64fe63a`;
saved initial checkpoint has the same hash. Final SHA-256
`9ce41cce63a56fbd6e4527dad6887ca81d8e6a148cc00439505abaee83d60234`.
Three RMD cells qualify before and after; periods can change under fitting.
Train MSE 1.094283 -> 1.094188 and held-out 5.100195 -> 5.099837, still worse
than the constant training-mean predictor (1 train, 5.071310 held out). Thus no
successful calcium fit is claimed. The run's elapsed_seconds=41.72 covers fitting
and refinement only, before the two persistence assays; later code corrects
total timing and retains fit/refinement time separately. No total wall time is
invented for the earlier run. Initial/final weights, optimizer state, exact
input/source/environment and data traces retained.

**Software:** initial targeted suite 8 passed, full suite **183 passed in 65.68 s**.
The three rhythm tests repeated after phase-comparison assertions: **3 passed
in 4.14 s**. Later edits clarify metadata/CLI errors and timing, not integration
equations. Tests include sustained full-channel cycles, calcium-current removal,
phase-aligned convergence, plateau/transient/blowup rejection, and exact preservation
of unmodified cells and anatomical weights by the profile application.

**Conclusion/limits:** a self-sustained, calcium-cycling conductance-network
candidate now exists and can seed subsequent fitting without an imposed clock.
This does not establish animal-matched kinetics/composition, A/B motor physiology,
coordinated muscle waves, proprioceptive control, locomotion, or a predictive
calcium fit. Preserve it as an experimental branch; the operating driven skills
remain unchanged. Artifact/archive identifiers and checksums follow below.

**Archive:** all six run directories, original complete stdout, copied parent
networks/data, initial/final models and optimizer state where fitting occurred,
raw/refined trajectories, post hoc cycle-comparison JSON, explicit prototype
metadata corrections, source/environment snapshots and plot PNG/PDF with source
copied to `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E027/`.
Upstream 2019 HTML/figure and pinned 2024 source text kept under `source_evidence`
for audit, with citation/repository metadata. All 897 copied files SHA-256
verified; manifest `SHA256SUMS.json` hash:
`4b7dd1f606fc1cdac13db26eb65d17c78f1eb2d36ff44015e6f8864c635dd6b0`.
Candidate for future inference:
`r-elegans-E027-connected-default/network_candidate.npz` inside that archive.
Small summary/hashes in `experiments/records/E027.json`. The archival records
retain actual dirty pre-commit source. Rebase `/tmp` parent paths to copied
archive equivalents only in replay copies, verifying hashes. Local OneDrive
verification is not evidence of completed cloud synchronization.

### E028 — 2026-11-07: unpaced channel brain → muscles → body diagnostic (planned)

**User authorization:** finish and locally verify the channel-brain/body bridge,
then prepare a bounded 30–60 minute nohup diagnostic on mbslnx2. No training,
RL, food task or periodic input. Preserve the E027 active candidate and the
saved parent NMJ/muscle-transfer parameters exactly; all raw data external.

**Plan:** use the channel network's physical voltage and physical release
threshold/slope to compute transmitter release, then the parent's frozen
956-edge NMJ gains/signs and muscle sigmoid (same nonzero resting tone as the
full actor) to obtain 95 activations. Project anatomically (head at last segment)
and integrate the overdamped body in the same execution loop. No body-to-neuron
feedback is added; a common loop is not proprioception. State is checkpointable
between chunks. Start from E027's original active checkpoint, not the one-update
calcium-fit candidate; parent NMJ SHA-256 recorded below when run.

**Diagnostic matrix:** normalized development body plus SI liquid/agar bodies;
unperturbed and small initial-voltage perturbations; original versus RMD-CCA1
ablation controls; frozen voltage/muscle output control for transient posture
relaxation. Compare body timestep refinement while retaining brain integration
accuracy. Distinguish sustained neural activity, dynamic muscle output, dynamic
shape, gross drift/rotation and useful locomotion. Static bends or startup
relaxation do not count as a gait. Predeclare late-window thresholds/flags,
report raw metrics, and retain negative outcomes without retuning readout.

**Operations:** select disjoint CPU cores before numerical imports, use the
explicit r-elegans conda Python, write summaries/checkpoints/logs externally,
enforce wall-clock budget and capture source/environment/input hashes. Prepare
a small checksummed input bundle for transfer if SSH remains password-only.
Do not claim remote launch until a remote process/run directory is evidenced.
Local bridge checks and server readiness/commands/results follow below.

**Implemented:** `body/channel_bridge.py` joins the complete neural state,
physical release, frozen parent NMJs/muscle sigmoid and overdamped body in one
scan. All 302 gates/calcium and the body's pose/shape/activation continue between
chunks; held-voltage control freezes the brain. No feedback to neurons, inferred
gait, tonic retuning or optimization. Physical threshold/slope come from the
channel network; NMJ strengths/signs and muscle fields come from the preserved
parent. This is a deliberate explicit unit contract, not an assertion that the
transferred readout is physiologically calibrated.

`scripts/diagnose_channel_body.py` records input/evaluated models, body constants,
full initial/latest/final state, chunk traces, progress and late-window metrics.
Latest state writes are atomic. Latest snapshots support audit/reconstruction;
an exact checkpoint-resume CLI is not exposed. Metrics use the final half of
up to the last 30 simulated seconds. Dynamic muscle/shape flags require max
p2p >1e-4 activation/>1e-4 rad; measurable displacement >1e-5 BL is a drift flag,
not a gait criterion. The script never claims locomotion from drift alone.
Late muscle saturation and joint-limit fractions are also reported by final code.

**Inputs:** original E027 active candidate, SHA-256
`870444a4a1122818d9124af242242c506778b25c0079d4ee8a441cf8c64fe63a`, plus
the preserved E026 parent actor/NMJ state, SHA-256
`82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0`.
The input bundle uses exactly these, not the one-update fitted E027 candidate.
All 956 anatomical NMJ edges and 95 muscles remain. Muscle transfer has the
parent's nonzero resting tone rather than a new baseline-subtracted readout.

**Local experiments actually run (Mac):** four 20 s SI-liquid cases: active,
RMD-CCA1 removal, held voltage, body timestep refinement. Commands are captured
in each run record, equivalent to:

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.diagnose_channel_body --network /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E027/r-elegans-E027-connected-default/network_candidate.npz --parent-actor /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E026/r-elegans-E026-calcium-fine/parent_actor.npz --output-dir /tmp/r-elegans-E028-liquid-smoke --medium liquid --duration 20 --chunk-seconds 2 --minutes 5
```

Control output directories `/tmp/r-elegans-E028-cca-control` and
`/tmp/r-elegans-E028-held-voltage` add `--condition cca1_removed` and
`--condition held_voltage`. Refined body run `/tmp/r-elegans-E028-body-refined`
adds `--dt .001`. All use max brain step .1 ms; chunk duration is 2 s.
Active runtime ~17.40 s locally (not a server timing prediction).

**Results:** all finite. Active: three rhythmic RMD cells, max muscle p2p
0.0584111, RMS muscle p2p 0.0109022; max joint p2p 0.00306485 rad, RMS joint
p2p 0.00125922 rad. Late displacement 4.8499e-6 BL over ten seconds; late path
0.000568758 BL, average path speed 5.6887e-5 BL/s, heading change -0.0003263 rad.
Thus direct brain→muscle→body influence is demonstrated as a small unpaced wiggle,
**not useful net locomotion or a calibrated gait**. CCA1 removal and held voltage
both give zero late neural/muscle/shape variation and displacement. Body dt
.001 gives max muscle p2p 0.0584131, max joint p2p 0.00307098 rad (~0.20%
difference), displacement 4.9813e-6 BL. Refinement supports the small body effect;
it does not turn negligible drift into locomotion. No parameters were fitted.

**Software verification:** targeted bridge/streaming/readout/checksum tests
**4 passed in 5.57 s**; full repository suite **187 passed in 66.29 s**.
Checks cover release units/resting tone, NMJ locality, chunk continuation,
neural equivalence with no body feedback, held-voltage control, finite SI body
state, drift-versus-gait distinction and bundle tamper detection. Final metadata
fields/queue heartbeat were added around verification; numerical equations are
unchanged. The Linux supervisor is not claimed executed on the Mac or server.

**Server preparation:** `run_embodied_diagnostics.py` is stdlib-only until
worker launch, verifies checksummed inputs/dependencies, assigns disjoint
physical CPU groups and uses sys.executable. Four CPU workers x four cores;
nine cases: active normalized/liquid/agar, liquid initial +/-1 mV perturbations,
liquid CCA1/held-voltage controls, body dt refinement, brain integration
refinement. 45-minute global budget, 12-minute case budgets, 600 simulated-second
ceilings; may finish sooner. Workers save progress every chunk; supervisor emits
30 s heartbeats, individual logs and a queue summary including pending cases.
Graceful stop retains completed chunks; forced cleanup may add 45 s. A finite
negative movement result is diagnostic evidence, not a process crash.

`prepare_embodied_bundle.py` packages two NPZ inputs, their checksums, tested
source hashes, and a launcher. The launcher fast-forwards Git (no reset), checks
data/source, explicitly uses `/home/carsonc/miniconda3/envs/r-elegans/bin/python`,
and nohups the supervisor with unique log/PID. It does not alter conda environments.
No HDF5 imaging dataset is needed for this fixed-parameter server diagnostic.

**Remote status:** SSH probe to `carsonc@mbslnx2.niddk.nih.gov` timed out before
authentication; no remote command/job launched. User then explicitly directed
"Just push and I'll launch myself." Honor that: publish code/bundle and launch
instructions, do not retry or claim a remote run. Local outcomes and server
plans stay separate until user supplies actual run evidence.

**Parameter question during work:** user notes channels and connections can
both fit and asks about leak/capacitance. Current calcium fitter jointly adjusts
channel conductances (including leak) and chemical/electrical strengths; C,
leak reversal and kinetic/calcium constants are currently fixed. Bundled AWCON
whole-cell model priors: C=3.1 pF, gLeak=0.27 nS, ELeak=-90 mV; RMD: C=1.2 pF,
gLeak=0.40 nS, ELeak=-80 mV. Active exploratory RMD profile: C=1.2 pF,
gLeak=0.045 nS, ELeak=-80 mV. Both presets use EK=-80, ECa=60, ENa=30 mV,
calcium equilibrium .05 uM, removal 50 ms. Most other cells have transferred
priors with bounded parent-derived C, not measured individual values. Fitting
capacitance/reversal would require explicit trainable fields and prior/identifiability
controls; it was not silently added to this fixed-parameter diagnostic.

Artifact/transfer paths and checksums follow after archival preparation.

**Local archive:** all four complete run directories, original stdout, source/
environment records, copied inputs/evaluated network/body parameters and all
chunk/state files copied to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E028/`.
Verified 667 files against `LOCAL_SHA256SUMS.json`, manifest SHA-256
`5f2132fa5c223ee519a0cb62cabd679d4e9fd3db22c3cd9485920705da4dd8b5`.
This manifest covers the local evidence, before adding the separately checksummed
transfer bundle. Final targeted recheck after diagnostic metadata additions:
4 passed in 5.71 s. No remote process started. Local OneDrive verification does
not establish completed cloud synchronization.

**E028 transfer bundle finalized after E029 source commit:**
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E028/transfer/r-elegans-E028.tar.gz`,
104,253 bytes, SHA-256
`fc13e63b7ccf9cd83326ee8610c727746120569fc5b28a0a3b63bd8f5c1adc99`.
Prepared from committed source `dd9aa10`, with exact builder invocation/environment
records alongside. All 54 source and two input hashes were verified, launcher
passed `sh -n`, and archive membership was checked: directory E028 plus
network.npz, parent_actor.npz, inputs.json and launch.sh only. This bundle is
separate from the local evidence manifest above. Copy/extract/launch commands
are in the E028 section of CURRENT_MODEL.md. No remote job has been launched;
user self-launches.

### E029 — 2026-11-07: freeze stereotypical properties, fit explicit unknowns (verified)

**User direction:** "fix as many things as we can", "only fit what is truly
unknown", and "fix parameters that are stereotypical." This changes fitting
defaults; E028 remains a fixed-parameter diagnostic, not a training run.

**Policy:** keep shared channel gating laws/rate constants, ionic reversals,
calcium handling, capacitance, leak reversal, topology, release/reversal priors,
NMJ/muscle/body readout and indicator constants fixed. Additionally anchor LEAK
conductance in every cell to prevent it drifting against the conductance/time
scale. Keep established RMD class profiles and the exploratory active RMD
profile fixed; the latter remains a model hypothesis, not measured biology.
Hold unobserved cells' channel priors fixed for each recording. Fit remaining
unknown channel densities only in observed, unanchored cells, and unmeasured
strengths on existing chemical/electrical edges. Counts/topology are known;
edge conductances are not established by contact count.

**Evidence distinctions:** many capacitance/leak/reversal values and transferred
profiles are current model estimates, not universal empirical constants. Fixing
them is the user's conservative modeling choice; do not relabel guesses as
known or force bilateral cells to be identical without evidence (ASE/AWC
asymmetry remains relevant). Additional externally supplied channel/edge freezes
must retain a source and reason. Gap freezes apply to both directions. No
kinetic or capacitance/reversal optimizer fields are added.

**Implementation/verification plan:** save a per-run fit-policy manifest and
exact trainable masks. Mask forward/gradient paths and project every optimizer
update back onto frozen values, preserving them bit-for-bit even with optimizer
momentum. Anchor priors and bounds to the supplied initial checkpoint, rather
than treating loaded nonzero log multipliers as deviations from zero. Test
mask/gradient/optimizer behavior, symmetry, invalid constraints, and a one-update
real-recording integration check. Preserve known/held-by-assumption labels and
report exact trainable counts/frozen changes. No long fit or remote launch is
authorized by this policy update.

**Implementation:** `brain/fit_constraints.py`, integrated into the calcium
fitter as its default with no broad all-parameter override. Every run saves
`fit_policy.json`, `trainable_masks.npz` and summary counts/frozen changes.
Masks act in the forward/gradient path and after each optimizer update. LEAK
is now fixed too, superseding the earlier statement during E028 that leak
conductance was trainable under the old default. Channel/edge prior penalties
and log-scale bounds are relative to the supplied initial checkpoint.

**Real-recording integration command:**

```bash
MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.fit_calcium_network --recording /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/2022-06-14-01-data.h5 --labels /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/neuropal_label.json.bz2 --uid 2022-06-14-01 --initial-network /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E027/r-elegans-E027-connected-default/network_candidate.npz --qualify-rhythms --experiment-id E029 --output-dir /tmp/r-elegans-E029-constraints --frames 8 --iterations 1 --learning-rate .001 --max-step-ms .1 --minutes 15
```

Only one update to test constraint enforcement, not a new physiological fit.
55 identified cells: 848 eligible channel conductances (53 unanchored observed
cells x16 non-LEAK components), 3709 existing chemical strengths, 1093 independent
gap strengths. All 302 neurons still evolve. Frozen change counts are exactly
zero for channel, chemical and electrical masks. Separately compared 20 fixed
checkpoint fields (cell constants, fixed network fields, config, IDs/schema):
all bit-identical. Capacitance, leak/reference and reversals were not fitted.
No identities or topology changed. RMD profile remains a fixed hypothesis.

Train MSE 1.094283 -> 1.094277; held-out 5.100195 -> 5.100143, still not a good
predictive fit or better than the constant baseline. Independent 20 s rhythm
qualification passes before and after. This verifies the restricted optimizer
path and persistence check, not the correctness/identifiability of fitted values.

**Tests:** targeted policy tests 3 passed in 3.31 s; full suite **190 passed
in 69.82 s**. Tests enforce class/unobserved/reference anchors, zero gradients,
fixed values despite deliberately injected optimizer momentum, symmetric known
gap freezes, and source/schema validation. No long training/server run started.

**Archive:** `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E029/` contains
the full integration run, original stdout, policy/masks, input/final states,
optimizer, traces and fixed-field audit. 163 copied files SHA-256 verified;
manifest `SHA256SUMS.json` hash
`6ab2197a36693f146fee675038bce9d952c56412fe8daad1cafd602acc092be1`.
Small summary/hashes: `experiments/records/E029.json`. Exact source/environment
snapshots retain the dirty pre-commit state. Local copy verification does not
establish cloud synchronization. E028 diagnostic inputs/equations are unchanged.

## Replication and recordkeeping procedure

For every future experiment, add an entry with: question, predicted outcome,
parent entry/checkpoint hashes, exact command, actual run manifest/revision,
fixed/trainable fields, data/model asset versions, seeds, environment/backend,
numerical settings, predefined gates, outcome including failures, interpretation,
decision and artifact/archive locations. Preserve original entries; date
corrections and link superseding entries rather than erasing negative results.

Future invocations store `run_records/<UTC-id>/command.json`,
`environment.json`, `requirements-recorded.txt`, `source_record.json` and a
source snapshot in each run folder. Environment records identify CPU affinity,
Python, installed package versions, platform and an allowlist of numerical
environment settings. They do not capture arbitrary secrets or claim original
wheel/container hashes. The scientific manifest remains authoritative for
effective settings overridden from checkpoint provenance.

Archive the latest finished reverse cohort on mbslnx2 after pulling this change:

```bash
python scripts/archive_motor_run.py \
  --latest-under /home/carsonc/r-elegans-results \
  --output-dir /home/carsonc/r-elegans-archives
```

The archive copies entire dependency cohorts, including original forward
references, rather than keeping only the latest actor. It checks dependency
hashes when recorded, verifies copied contents, preserves Git history in
`repository.bundle`, and records collection-time environment separately.
`archive.json` contains original-to-archive path mapping, SHA-256 inventory
and historical gaps. `COMPLETE` appears only after verification. Missing or
changed required checkpoints cause an error; no source files are deleted.

Verify before/after transfer with:

```bash
python scripts/archive_motor_run.py --verify /path/to/finished-archive
```

Copy the completed archive to separate storage/OneDrive and verify it there.
Until that is done, server records are not independently backed up. Keep the
archive's original manifests unchanged. On an isolated replay workspace,
restore source/dependency folders using `archive.json` mappings; preserve the
original paths or rebase reference paths only in replay copies. Record any
rebasing and confirm checkpoint hashes remain identical.

For a fresh repeat: check out the recorded code revision from the bundle (or
use the captured source snapshot for dirty source); restore original input
checkpoint dependencies; recreate the recorded Python/package environment and
numerical/backend settings; execute the recorded command with the same seed
and a fresh output directory; repeat independent physics assays and compare
metrics within declared numerical tolerance. A different CPU/GPU/JAX stack
does not promise bitwise identical results. Do not reuse tuning cases as the
only behavioral test or lower a criterion after observing failure.

Current continuations start fresh optimizers when loading `--initial-actor`.
Although optimizer and RNG checkpoints are retained, exact mid-run optimizer
resume is not exposed by every CLI. This is distinct from repeating a run
from its original initial checkpoint and seed.

Historical gaps remain: original early commands/environments and complete
remote files have not all been recovered; pasted stdout is not a checkpoint
backup; package version lists are not immutable wheel images; some dirty-code
experiments require archived overlays. The new records reduce future gaps,
but do not retroactively make every past experiment exactly reproducible.

## 2026-11-07 — E028 launch workflow correction: pull and run

User requested restoration of the familiar Git pull/launch workflow. The earlier
transfer bundle reproduced the exact Mac input, but mandatory transfer was
unnecessary when a successful full-network head-sweep parent already exists on
mbslnx2. Historical bundle hashes above remain evidence of that earlier method.

Planned server method: after `git pull --ff-only`, run
`sh scripts/launch_embodied_mbslnx2.sh`. It uses the dedicated r-elegans Python,
selects the newest head-sweep cohort containing a passed stage and an existing
actor checkpoint, preferring the lowest successful seed within that cohort,
and reconstructs the all-cell channel candidate with the fixed E027 RMD profile.
No download, random fallback or parameter fitting occurs. Preserve the selected
parent path/hash, exact copied actor, resulting model/hash, source/environment
and all diagnostic outputs externally. A different server parent may change
the result; equivalence with the Mac candidate is not assumed. Explicit parent
and exact-input overrides remain available. The nohup launcher prints PID and
a unique log path. Queue budget is 45 minutes, input setup up to two additional
minutes and cleanup up to 45 seconds. No remote job has been launched.

Local verification: executed
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.bootstrap_embodied_inputs --parent-actor /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E026/r-elegans-E026-calcium-fine/parent_actor.npz --output-dir /tmp/r-elegans-E028-pull-run-check`.
The preserved parent SHA256 is
`82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0`;
reconstructed network SHA256 is
`870444a4a1122818d9124af242242c506778b25c0079d4ee8a441cf8c64fe63a`,
byte-identical to the existing E027 candidate when using that same parent.
This validates reconstruction, not biological accuracy or a server outcome.
Raw input copies, manifest, source/environment snapshots remain in the external
`/tmp/r-elegans-E028-pull-run-check` directory (temporary local verification,
not an independently backed-up server run). Shell syntax check passed; all five
channel/body tests passed, including success-gated parent selection and fallback
only to an older successful checkpoint if newer stages fail or files are absent.

The reconstruction verification directory was also copied to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E028/pull-run-verification/`
for local retention. This confirms a local copy, not completed cloud sync.

Final local software validation: full `.venv/bin/python -m pytest -q` suite
passed **191 tests in 68.81 s**. `git diff --check` and shell syntax passed.
The Linux queue remains unexecuted pending the user launch.

## 2026-11-07 — E028 first server result: stationary late state

Evidence: user-pasted summary stdout for
`/home/carsonc/r-elegans-results/embodied-20261107T185913027018Z`.
Attachment SHA256 `580080f8d449e6ade669e9a7b1385c4b640092e332a180ae38f59636654570b3`.
Small extracted record: `experiments/records/E028-server-20261107.json`. This is
user-reported execution, not independent retrieval of remote raw artifacts.
Actual server source/environment snapshots remain on the server; their revision
is not present in this summary, so no exact revision is inferred.

All nine planned cases finished, exit 0, finite, none pending or interrupted.
Queue took 1397.130883 s (23.29 min), below 45-minute budget. Eight cases reached
600 simulated seconds. Brain-refined case stopped at its own time budget after
374 simulated seconds / 724.189057 wall seconds. This is partial simulated
duration, not an error or a complete 600-s refinement test.

In every final measurement window (last 15 s of retained 30 s): zero maximum
neural voltage p2p, zero rhythmic RMD cells, zero muscle p2p, zero joint p2p,
zero net/path displacement and heading change. Active-liquid muscle activation
remained nonzero and constant (0.135769 to 0.948988 across muscles). Thus the
network/body settled rather than demonstrating persistent unpaced behavior.
Same qualitative late result for normalized/liquid/agar, +/-1 mV initial
perturbations, calcium/held-voltage controls and both timestep refinements.

Parent `beb069dd051763b6fdc3de977e4a2f77c9db73827cb4e8dfb2f88d899ea27078`;
network `95fafcabfce62aa0a2c37dc9c49b475a23ed96264e8d8899640bb9ad30da4882`.
Parent selected from passed seed-0 server head-sweep cohort
`head-sweep-20261107T003628901581Z`. This differs from the Mac-tested parent;
that short 20-s outcome is not a matched comparison. It remains unknown whether
the server ever oscillated transiently or the Mac candidate persists for 600 s.
Do not attribute failure uniquely to synaptic suppression, channel kinetics or
medium. No body-to-neuron feedback exists, so changing medium alone cannot
rescue this neural fixed point. No parameters were trained in this diagnostic.

Decision: do not extend this same fixed-state diagnostic merely to spend more
time, or proceed directly to RL. Next proposed analysis is early/middle/late
chunk inspection and matched-parent/matched-duration neural-only comparisons,
keeping channel/connection parameters fixed initially to isolate the cause.
These follow-ups are proposed, not launched. Existing driven skills remain
separate checkpoints, unaffected by this negative autonomous-branch result.

## 2026-11-07 — Bundle the exact Mac reference at user request

User instruction: "Then just copy those parameters into the repo?" This
explicitly authorizes two small parameter-checkpoint assets as exceptions to
the external-checkpoint rule; raw scientific data and other outputs remain
external. Copied without fitting or numerical reconstruction:

- E027 `network_candidate.npz` from OneDrive evidence into
  `r_elegans/assets/mac_rhythm_v1/network.npz`, 53,165 bytes, SHA256
  `870444a4a1122818d9124af242242c506778b25c0079d4ee8a441cf8c64fe63a`.
- E026 `parent_actor.npz` from OneDrive evidence into the same asset directory,
  50,664 bytes, SHA256
  `82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0`.

Manifest retains source paths, sizes/hashes, exploratory status and the 20-s
qualification limit. The diagnostic and nohup launcher default to this exact
pair, verify checksums and copy both into each external run directory. No
automatic discovery of a different server parent remains in the default path.
Explicit parent reconstruction/other input overrides are still opt-in.
The already-reported negative server result is preserved; changing the default
is not a retroactive change to that experiment. No new server job is launched.

Re-run command on mbslnx2: `git pull --ff-only` then
`sh scripts/launch_embodied_mbslnx2.sh` from the repository. Same fixed nine-case
diagnostic and timing caps; no parameter training or RL. Purpose is now to
test the exact short-Mac candidate over longer durations. All physiology and
connection parameters remain fixed. A longer persistence outcome is pending.

Local verification: `MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q tests/test_channel_body.py`
passed all six tests in 7.11 s. The new reference test validates pinned hashes,
loads both complete checkpoints and confirms exact equality of all neural
parameter/fixed arrays against reconstruction from this same parent and E027
profile. It also checks default versus explicit input/parent selection. Shell
syntax and `git diff --check` passed. No new physiological outcome is inferred
from these software checks.

## 2026-11-08 — E028 Mac-reference workflow retains late rhythms at 600 s

Evidence: user-pasted compact case metrics and preceding completion log for
`/home/carsonc/r-elegans-results/embodied-20261108T035729978291Z`.
Queue exit 0; all nine cases completed execution, approximate wall time 23 min
(exact final elapsed time not provided). Extracted numerical evidence is in
`experiments/records/E028-server-20261108.json`. Full raw trajectories, source,
environment and input hashes have not been independently retrieved. The user
ran the bundled-reference workflow, but compact output omits hashes; identity
is expected from that workflow rather than independently verified for this run.

Six active cases reach 600 simulated seconds with three rhythmic RMD cells
in final windows: normalized, liquid, agar, liquid +/-1 mV initial perturbations
and refined body dt. Maximum voltage p2p ~21.20 mV, muscle p2p ~0.05846.
Liquid: joint p2p 0.00335177034 rad; late displacement 2.6387122e-6 body lengths.
Agar late displacement 1.00981988e-5 barely exceeds the diagnostic drift flag,
not a useful locomotion result. Refined body: joint p2p 0.00336031616 rad
(~0.255% above ordinary liquid); neural refinement: three rhythmic cells at
368 s, 21.25097 mV p2p, then case time budget. Different observation durations
mean this is qualitative persistence evidence, not matched-duration convergence.
Both calcium-removal and held-voltage controls reach 600 s with zero late
voltage/muscle/joint variation and zero displacement.

Interpretation: the Mac reference workflow produces longer-lived unpaced neural
activity transmitted into small bends on mbslnx2. The earlier negative candidate
is preserved; host performance alone cannot explain the earlier lack of rhythm.
Parameter-specific differences are implicated, but individual weight/channel
causality is untested. End-window summaries do not prove uninterrupted rhythms
throughout all 600 s. Engineered E027 conductance profile, transferred priors,
fixed compensation currents and lack of body-to-neuron feedback remain. This
is not biological calibration, new learned skill, or useful autonomous gait.

Decision: preserve this fixed parameter reference; next proposed work is spatial
phase/traveling-wave analysis of saved neural/muscle/body trajectories before
training coordinated propulsion. No new experiment, fitting or RL launched.
Documentation and record JSON validation only; no simulation code changed.

## 2026-11-08 — E030 planned: amplitude-only autonomous wiggle fitting

User asks to train neuron parameters for wiggle amplitude, explicitly without
external drive. Start from the exact bundled Mac reference; zero all injected
compensation currents, not just periodic forcing. No sensory input, timing
signal, gait teacher, NMJ retuning or body force injection. No propulsion or RL
objective. First assay the zero-bias baseline locally.

Fit uncertain non-LEAK RMD channel densities, bounded multiplicatively around
the reference, leaving capacitance/leak/kinetics/reversals/calcium handling,
recurrent strengths/topology and NMJ/muscle/body parameters fixed. This explicitly
relaxes the earlier model-assumption freeze of the exploratory RMD densities
for this behavioral fit; those densities were not established measured values.
All 302 neurons remain in the simulation. Use bounded derivative-free search
to avoid backpropagating through the full stiff channel network. Score late
cyclic joint-angle variation rather than static curvature or displacement.
Initial engineering target: 0.01 rad peak-to-peak at a joint, retaining persistent
neural rhythm, finite bounded voltages and no joint-limit clipping. This is a
small developmental target, not an animal-calibrated amplitude.

Planned records: exact source/environment, parent/input hashes, allowed mask,
all candidate parameters/metrics and RNG search state, selected/final network
checkpoints, optimizer/search history and held-out longer validation traces.
Server launch remains user-operated under nohup. No new server job is launched
by the agent. Scientific success remains pending actual fitting and validation.

E030 execution update (local): zero-bias baseline command
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.diagnose_channel_body --network /tmp/r-elegans-E030-zero-bias-input/network.npz --parent-actor r_elegans/assets/mac_rhythm_v1/parent_actor.npz --output-dir /tmp/r-elegans-E030-zero-bias-baseline --duration 20 --minutes 2`.
Zero-bias input SHA256
`ac3976ff6bc89b8b9c3ccd92068882ea2cce7201384b4eabc4cc87138ce3e98c`.
Three rhythmic RMD cells, max voltage p2p 21.18092 mV, max joint p2p
0.00294807553 rad; no clipping/saturation. All injected currents zero.
Raw output and stdout at the corresponding /tmp paths.

Short fitting verification launched locally with seed 0 (default):
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.train_autonomous_wiggle --output-dir /tmp/r-elegans-E030-fit-smoke --minutes 4 --iterations 8 --validation-seconds 20`
(stdout `/tmp/r-elegans-E030-fit-smoke.log`). Initial short-window sustained
amplitude 0.00230509043 rad; candidate 2 reached 0.0262243822 rad and retained
three rhythmic cells. This is provisional until validation. The search source
was snapshotted at invocation; subsequently added stop-at-target behavior is
not part of that executing search. Planned additional independent local
300-s liquid diagnostic of its selected model, plus held-voltage control,
to establish persistence beyond the short-window fit without changing it.

E030 short verification completed: passed=True, 8 evaluated proposals, selected
iteration 2, 137.4422277 s wall. This pre-stop-at-target source is captured in
its run record. All 78 fit values are existing non-LEAK conductances in six
RMD neurons; selected density factors range 0.857854–1.224696. Independent
20-s validation sustained amplitude 0.0279118828 rad; +1/-1 mV initial-state
perturbation checks 0.0263734451 / 0.0261019589 rad; 0.05-ms brain timestep
0.0263551883 rad. All four checks retain three qualified RMD rhythms.
Strongest joint is anterior/head, not a whole-body traveling wave.
Frozen parameter arrays compare exactly; all 302 injected currents remain zero.

Independent longer diagnostic command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.diagnose_channel_body --network /tmp/r-elegans-E030-fit-smoke/network_best.npz --parent-actor r_elegans/assets/mac_rhythm_v1/parent_actor.npz --output-dir /tmp/r-elegans-E030-fit-long --duration 300 --minutes 10`.
A separate held-voltage control uses the same selected network, duration 20,
`--condition held_voltage`, output `/tmp/r-elegans-E030-fit-held-control`; all
late neural/muscle/joint variation and displacement are zero. Raw stdout stays
at the corresponding `.log` paths.

The selected model is bundled for pull-and-run reproducibility as
`r_elegans/assets/autonomous_wiggle_v1/network.npz`, SHA256
`482995e03f28cfd62704b20630c28c2fb1706aec87dbd4e2cc6eecd9315a2ff6`.
The corresponding parent readout remains byte-identical to the Mac reference.
These compact fitted parameters serve the requested working wiggle; raw
scientific data, search trajectories and other checkpoints stay external.
The original Mac asset and its default diagnostic remain preserved.

Software validation: full suite 194 passed in 66.50 s; subsequent added
bundled-fit integrity test and all amplitude tests passed (3 tests in 1.33 s).
Both launcher shell syntax checks, module CLI help and `git diff --check` pass.
Final search behavior stops at target before independent validation; a worker
that does not meet all validation gates retains its candidate but exits 2.
These are software tests, not experimental biological validation.

E030 independent 300-s validation completed locally: finite throughout, three
rhythmic RMD cells, maximum voltage p2p 27.72645 mV, muscle p2p 0.2397614,
maximum joint p2p 0.0295088887 rad, sustained five-window amplitude
0.0282294517 rad. Every saved 2-s head-joint window after 12 s has p2p
>=0.0268120915 rad. This inspection establishes persistent larger bends across
the sampled post-warmup run, not only its final window. No joint clipping or
muscle saturation. Late net displacement 2.6557840e-5 body lengths; the drift
flag is not a useful propulsion claim. The corresponding held-voltage control
has zero late neural/muscle/joint variation.

Result: first autonomous amplitude milestone achieved locally with all injected
currents zero, no external pacing, no gait teacher and no synapse/body retuning.
This is mainly anterior/head bending, not a whole-body traveling gait or a
biologically calibrated channel fit. Preserve this model as an independent
skill/reference; coordinated body waves remain a later objective. Exact compact
results: `experiments/records/E030.json`. Source/output provenance and all raw
local files are archived under OneDrive evidence/E030; this is local retention,
not a claim of completed cloud synchronization or independent remote backup.
No mbslnx2 job has been launched by the agent.

Verified local OneDrive archive: 875 files, `LOCAL_SHA256SUMS.json` SHA256
`0bb644e6bf045b0c57c213f7dc71c89c109cf9d1acf6f4a08d4c20c24d60c669`.

## 2026-11-08 — E030 user-reported 600-s server amplitude validation

Run `/home/carsonc/r-elegans-results/embodied-20261108T141954839072Z`,
user-pasted compact metrics and completion log; queue exit 0. Active liquid
bend p2p 0.0295088477 rad, muscle p2p 0.239748418, three rhythmic RMD cells
at 600 s, matching local 300-s behavior. Agar 0.0235053953 rad; normalized
0.00766909309 rad. Liquid +/-1 mV initial perturbations: 0.0295133293 /
0.0294779725 rad. Body refinement at 600 s: 0.0295425355 rad (~0.114% above
ordinary liquid). Brain refinement: 0.0294816662 rad at 370 s before its case
budget; not a completed matched-duration 600-s comparison. All active cases
retain three rhythmic cells. Both held-voltage and RMD calcium-removal controls
reach 600 s with zero late rhythm, muscle and joint variation. Full remote
source/environment/input hashes and raw trajectories not independently retrieved.
Thus the zero-drive amplitude reference has user-reported longer server support;
useful propulsion remains a separate goal.

## E031 planned — first autonomous movement, three-hour server budget

User authorizes movement training again, with a three-hour run. Preserve the
working `autonomous_wiggle_v1` checkpoint and begin each independent search
there, with all injected currents zero and the full 302-neuron circuit active.
First small stage retains the same 78 unknown RMD density parameters; all
connection weights, other cells, stereotypical physiology and body/NMJ fields
stay fixed. If this restricted stage fails, broaden motor-circuit parameters
in a later recorded experiment rather than silently changing this scope.

Optimize repeated signed forward centroid progress in SI liquid after warmup,
not path length, static posture or displacement during startup relaxation.
Keep the >=0.01-rad sustained wiggle as a viability gate. Initial engineering
target: >=1e-4 body lengths/s in each of three late blocks, >=5x positive
baseline speed, >=2% forward/path efficiency, <0.35-rad heading variation.
No timed stimulation, target waveform, sensory input, body forcing or RL.
Use the existing bounded density search, four independent seeds on four
disjoint four-core CPU workers, up to 180 wall minutes each. Reserve time for
300-s validation, +/-1 mV initial states, neural and body timestep refinement,
and a held-voltage control. No success claim until those checks pass.

All parent/model hashes, candidate histories, selected checkpoints, search RNG,
full source/environment and validation trajectories remain external; compact
results and lab notes in Git. The user launches on mbslnx2 under nohup; agent
will prepare/test/push and will not start a remote job.

E031 local pilot started before server launch:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.train_autonomous_wiggle --objective movement --output-dir /tmp/r-elegans-E031-movement-smoke --minutes 6 --iterations 2 --validation-seconds 30`
(stdout `/tmp/r-elegans-E031-movement-smoke.log`). This is a short pipeline
verification, not the planned three-hour search. Baseline repeated-block
minimum signed forward speed is -1.8236493e-5 body lengths/s; initial candidate
1 improves it to -1.3028506e-5 while keeping >=0.0245-rad sustained wiggle, but
still moves backward in at least one block and is not a movement success.
The local pilot captures its exact pre-refinement-agreement source snapshot.
A subsequent promotion guard also requires matched short-window speed agreement
within 25% under body/neural timestep refinement (target speed tolerance floor);
the synthetic metric tests cover rejecting timestep-dominated translation.
This is a numerical tolerance choice, not a biological variability estimate.

E031 local pilot completed in 210.027 s: status completed,
passed=False, expected exit 2, two proposals and all six validation cases
finished. Frozen fields unchanged and injected currents zero. The standardized
`fitted_inputs/` checksum manifest verifies, including for this unpromoted
candidate. No forward-motion skill is claimed. Compact pilot metrics recorded
in `experiments/records/E031.json`; source snapshot/RNG/history/selected models
and all validation traces retained externally. Full suite: 196 passed in
66.08 s. Subsequently added timestep-agreement guard and movement/amplitude
tests: 4 passed in 1.34 s; shell syntax, CLI help and diff whitespace checks pass.
Server three-hour run remains prepared, not launched by the agent.

E031 OneDrive local archive verified: 203 files; manifest SHA256
`311a3f51a4b4fe9a8c09de57df7c56d61269c3b49a12f981ce97f20c3fc16527`.
Local retention only; cloud synchronization has not been independently verified.

## 2026-11-08 — E031 three-hour server result: no forward-motion qualification

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

The pasted summary does not include per-worker elapsed time, baseline/best
training metrics, hashes or the source revision; those are not inferred. The
reported run identifier and iterations are preserved exactly. A multi-part
forward-motion criterion failed; no specific numerical artifact is established
from these scalar summaries alone. This update changes records/docs only.

## 2026-11-08 — E032 planned: fit neuron densities and recurrent connections

User explicitly requests training neurons and connections after E031 failed.
Broaden eligibility across all 302 cells: 4,814 existing non-LEAK channel-density
scales, all 3,709 anatomical chemical strengths and 1,093 independent electrical
pairs (9,616 unknown parameters total). Start from the preserved autonomous
wiggle, not the failed E031 candidates. All injected currents remain zero.
Fixed: leak, capacitance, channel kinetics/reversals/calcium handling, anatomical
masks, synaptic reversal/release settings and NMJ/muscle/body fields. Maximal
conductances/connection strengths remain unknown effective model parameters,
not experimentally inferred physiological values. Preserve missing channels
and absent anatomical edges. Keep electrical coupling symmetric.

Use a flat vector of bounded log multipliers (0.25–4 of initial density or
effective connection strength). Use stochastic blocks of individual parameters
and coherent channel/connection-type proposals; all unknowns are eligible,
without requiring every parameter to move on each proposal. Preserve exact
zero-offset identity and byte-identical frozen/off-topology fields. Retain
E031 scoring and promotion criteria so changing trainable scope is explicit.
Four independent three-hour CPU workers; same 30-s search horizon and longer
perturbation/timestep/held-voltage checks. No external drive, gait teacher,
body-parameter tuning or RL. Prepare/test/push; user launches remotely.
Before publishing, test masks, symmetry, effective-factor bounds, no-drive
invariant, parameter counts and an end-to-end local trial/export. No E032
movement success is assumed. Raw search records and source stay external.

E032 local pipeline pilot command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.train_autonomous_wiggle --objective movement --fit-scope neurons-connections --output-dir /tmp/r-elegans-E032-joint-smoke --minutes 6 --iterations 2 --validation-seconds 30`
(stdout `/tmp/r-elegans-E032-joint-smoke.log`). The first accepted proposal
changes 58 density scales, 58 directed chemical strengths and 12 electrical
pairs (24 symmetric raw entries), with zero injected current and preserved
wiggle. Minimum signed speed improves from -1.8236493e-5 to -1.4114976e-5
body lengths/s but is still negative, so this is not a movement success.
Policy explicitly reports all 9,616 eligible values; proposal and accepted
coverage are tracked separately rather than claiming every value changed.
Metadata coverage fields were expanded after the pilot started; its original
source snapshot is preserved, so those late metadata additions are not
retroactively attributed to its executable source. No optimization equation
was changed after pilot launch.

Full software suite: 198 passed in 70.00 s. Joint/density/movement targeted
tests: 6 passed in 2.75 s. Tests verify exact zero-offset identity, all-cell
eligibility, existing topology, multiplier bounds and symmetric electrical
conductance, as well as frozen fields. Shell syntax, CLI help and diff
whitespace checks pass. These validate software, not biological fidelity.

E032 local pilot completed in 216.546 s, status
completed, expected exit 2 because the forward-motion criterion is not met.
Two proposals, selected iteration 1; all six validation cases and fitted-pair
export completed. Both density and chemical/electrical strengths changed in
the selected candidate, while frozen fields compare exactly and all injected
currents stay zero. Exact saved parameter pair checksum verifies. Compact
results are in `experiments/records/E032.json`; full source, histories, masks,
coverage, selected models and validation traces remain external. No server
run was started by the agent, and no new movement skill is claimed.

E032 OneDrive archive verified locally: 210 files; checksum manifest SHA256
`d2c5aa3eb902859af71de9c191b35ac06abbf83f8ed6a2185a02e0ca1fa0c3ef`.
This confirms local retention, not completed cloud synchronization.

## 2026-11-08 — E032 budget extended to four hours

User explicitly requests four hours. Movement launcher now passes
`--minutes 240`, replacing 180, for each of four concurrent joint neuron/
connection searches. This includes reserved validation time, plus startup/
termination grace. All masks, objective and acceptance criteria unchanged.
Applies to new launches; an already-running process is not modified. No
remote process was started or restarted by the agent. Shell syntax and diff
whitespace checks passed; no simulation rerun needed for this budget change.

## 2026-11-08 — E032 user-reported joint search fails movement validation

Run `autonomous-movement-20261108T205220165775Z`; reported scope
neurons-connections. All four seeds exit 2, passed=False. Iterations
473/470/473/470 (1,886 total). Reported parameters_updated
5744/5122/5656/2030 are cumulative accepted-update coverage, not necessarily
final nonzero offsets or per-kind coverage. Per-kind breakdown is absent.
All long-validation minimum speeds and efficiencies are negative; active
sustained bends remain roughly 0.025–0.031 rad. Positive short-condition speeds
peak at 6.91e-6 body lengths/s, still about 14.5x below the 1e-4 target; maximum
reported forward/path efficiency 0.01039 remains below 0.02. Held-voltage
controls have zero speed, efficiency and bend variation in every seed.

Full raw server histories, waveforms, input/source/environment hashes, baseline
and selected training metrics have not been retrieved. No exact elapsed time
or revision is inferred from the compact output. This is evidence of a failed
search, not full-network insufficiency or a unique failure mechanism.

E033 planned diagnostic audit, not a new training run: inspect whether the
fixed-endpoint, short-block score can reward phase alignment of bounded cyclic
centroid motion. Use synthetic periodic traces at 30/300 s, test current scoring
and promotion separately. This does not model actual hydrodynamics or establish
that E032 exploited this route. Record exact script/source and numerical outputs
externally. No model parameters or acceptance thresholds will change in audit.

E033 executed locally:
`MPLCONFIGDIR=/tmp/r-elegans-mpl PYTHONPATH=/Users/carsonc/github/r-elegans .venv/bin/python /tmp/r-elegans-E033-phase-audit.py`
stdout `/tmp/r-elegans-E033-phase-audit.log`. Script copied into the raw
assay directory and source/environment/invocation captured. Period
5/23.05 = 0.2169197397 s, centroid amplitude 0.003 body lengths.
Zero complete-cycle translation by construction. At 30 s the current search
score is +0.00011155016, although efficiency 0.00279556 fails promotion.
At 300 s the score is -0.00009904649 and efficiency -0.000596876. Thus
selection score and promotion are misaligned for this synthetic trace.
It does not establish that an actual hydrodynamic solution exploited this
loophole; the chosen centroid trace is an input to a metric, not a forward
body simulation. Propose robust cycle/phase-averaged progress and inspect real
selected trajectories before further compute. Preserve objective versions
and negative runs; do not retrospectively relabel them or weaken thresholds.

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

E033 archive verified locally: 178 files, checksum manifest SHA256
`2da4b717c8796ccd82fdbdb5bac0d3c543e774b00d3838753b701a92363fed11`.
Local retention does not establish completed cloud synchronization. JSON
record consistency and diff whitespace checks passed. No simulation code
changed and no full test-suite rerun was needed for these records.

## 2026-11-08 — E034 planned trajectory analysis

User requests analysis of trajectories after E032 joint training failed. The
four-hour cohort `autonomous-movement-20261108T205220165775Z` is not locally
available. Noninteractive read-only SSH fails authentication; no password or
key requested/read, no server files retrieved. Analyze preserved local E032
pilot validations and E030 300-s reference where available, clearly separated
from the four-hour candidates. Prepare identical read-only analysis for user
execution on the server. Do not infer remote waveforms from scalar summaries.

Measures: late neural and joint/motor activation variation along tail-to-head
body; dominant frequency and coherent phase offsets of joints with meaningful
amplitude; centroid displacement between repeated upward crossings of the
strongest-joint waveform; full-shape closure and heading changes at crossings;
cycle-sampled drift regression versus fixed-endpoint short-block scores. Plot
waveform/spatial amplitude/phase and raw centroid versus equal-phase samples.
A single joint's phase is not proof that the entire body shape closes: report
that residual explicitly. Timing from per-case recorded metadata; SI body
length supplied explicitly (0.001 m for these recorded liquid worker assays).
This is analysis, not training or biological validation. No model, objective
or acceptance threshold changes. Exact analysis source/environment, input
hashes, numerical outputs and figures remain external; compact findings in Git.

E034 local execution: `MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.analyze_autonomous_trajectories --run /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E032/joint-smoke --output-dir /tmp/r-elegans-E034-local-pilot --body-length .001 --plots`.
Same CLI for `.../E030/fit-long`, destination
`/tmp/r-elegans-E034-local-reference`. Both completed: all six pilot cases,
all 150 chunks of the 300-s reference. Exact source/environment/commands and
input hashes are in those outputs, stdout at corresponding `.log` paths.

Local pilot late window: anterior3 variance 99.044388%, PCA1 99.939185%,
four significant joints, ~3.733 Hz. Joint10/9/8/7 lags relative to joint10
~0/0.160/1.340/1.569 ms versus mean cycle period 268.056 ms. Most shape
variation is one anterior pattern rather than a body-spanning wave. Three
neurons exceed 2-mV p2p (RMDs): descriptive voltage variation, not a biological
spike classification. Across 55 phase-matched cycles, median full-shape
closure RMS 3.4657e-5 rad. Forward phase-position regression -4.04816e-7 BL/s
versus raw minimum endpoint-block speed -1.41150e-5. Body refinement gives
-1.12755e-8; neural refinement -4.08388e-7. No robust propulsion is demonstrated.
Original 300-s E030 reference: anterior variance 99.054%, PCA1 99.931%, phase
drift -4.62370e-7 BL/s. These are local findings, not four-hour server data.

Targeted tests: 9 passed in 2.17 s, checking bounded oscillation versus known
translation recovery, standing versus phase-shifted shape, and static controls.
Cohort CLI verified with a one-worker layout fixture linking the same local
pilot (not a new seed/server run), long and held-voltage cases. A generated
plot was visually inspected. All input files unmodified; no model/fitness
change and no new training. Compact evidence `experiments/records/E034.json`;
raw reports/phase samples/figures archived externally. Actual server trajectory
analysis still requires user execution of the provided read-only command.

E034 OneDrive local archive verified: 572 files, checksum manifest SHA256
`245a97032280bfabeeff38debb01e7e80bfc0e85c6378105e422bf2236f95c6b`.
This confirms a local preserved copy; cloud synchronization is not verified.
Server run remains pending user-side read-only analysis because the agent
cannot authenticate to obtain the raw files. No four-hour findings invented.

## 2026-11-08 — E034 actual four-hour candidate analysis received

E034 actual-cohort update (user-reported server analysis): the four-hour E032
selections have now been analyzed on mbslnx2, with all 24 worker/case results
reported. Analysis directory `trajectory-analysis-20261109T020051Z`; original
training cohort `autonomous-movement-20261108T205220165775Z`. Full raw files
and detailed hash-bearing reports have not been independently retrieved.

All four long cases concentrate 98.95–99.09% of dynamic joint variance in
the three anterior joints. One shape pattern explains 99.88–99.99%. Four
joints exceed the meaningful-amplitude threshold. Dominant frequencies
3.41–3.81 Hz (periods ~262–293 ms), with maximum active lag only 1.08–2.98 ms
in long cases and <=5.15 ms in other active cases. The learned movement
remains nearly synchronous anterior bending, not a body-spanning wave.

Long equal-phase drift (BL/s), seeds 0..3:
-5.002e-7 / -2.854e-8 / -3.911e-7 / -4.470e-7. Small positive short endpoint
scores reverse or shrink after matching phase in seeds 0,2,3; seed 1 retains
small short-case positive phase drift but loses it in the long case. Body
refinement changes the tiny residual substantially, sometimes flipping sign.
Seed 1 positive-cycle fractions of 1 in short cases do not qualify it: speeds
remain ~5e-7–9e-7 and long drift is slightly negative. Held-voltage controls
have no cycles or motion. No robust autonomous forward skill is established.

The local diagnosis is supported by actual candidate summaries. The missing
primitive is distributed, phase-progressing bending. Proposed next curriculum:
first reward self-generated bend coverage into middle/posterior joints and
nontrivial spatial phase coordination, then cycle-averaged forward progress.
Neuron/connection eligibility remains broad; zero injected drive and fixed
physiological/body properties remain required. A single cause for optimization
failure or full-network insufficiency is not established. Preserve original
wiggle and failed candidates; no new long search launched in this analysis.
See `experiments/records/E034-server-20261108.json`.

Raw console attachment SHA256 `6c7e77d7d5961ea6975012c6d0068c35f173f13625b8aac0c54ea6e64fa19ee3`. UTC analysis
directory date is Nov9; client-local record date remains Nov8. No source or
input model hash is invented from the console. This updates analysis evidence
and proposed next steps only; no simulation code or objective changed.

User-provided console preserved locally under OneDrive evidence/E034/
server-analysis-report/console.txt with SHA256.json. This preserves the
reported analysis, not the full remote trajectories or a verified cloud copy.
JSON extraction contains 24 unique worker/case records; whitespace checks
passed. No code changes or simulation rerun for this records-only update.

## 2026-11-08 — E035 planned intrinsic body-wave curriculum

User authorizes the proposed next curriculum after actual four-hour trajectory
analysis found ~99% anterior bending and tiny endpoint-phase-dependent progress.
Implement a body-wave objective before further propulsion optimization. Start
from autonomous_wiggle_v1, keep all 9,616 eligible unknown neuron/connection
parameters, zero all injected currents, and freeze anatomy, leak/C/kinetics/
reversals/calcium/release, NMJs and body mechanics. No time-varying teacher,
sensory current, forced oscillator or body actuation is inserted.

Stage1 spreads coherent spontaneous oscillations into posterior/middle regions.
Stage2 rewards ordered spatial phase progression in both filtered antagonistic
activation and joint bending. Regions are equal contiguous body thirds
(tail-to-head ordering). Engineering targets: harmonic joint and drive p2p
>=0.003 per region, common-frequency power fraction>=0.5 and phase-locking>=0.8;
phase span>=pi/3, consistent direction in both spatial increments, no increment
>0.8pi or <0.1 rad, shape PCA second component>=0.02. Either propagation
direction allowed for this first primitive. Preserve >=0.01-rad sustained bend
somewhere, bounded voltages/no clipping/saturation. These are not animal fits.
A rhythm in any neuron (>=0.05-mV engineering threshold, including graded
activity) may support the wave; no RMD-only identity requirement is imposed.

Use the observed spontaneous frequency, not a prescribed clock. Before spatial
phase is optimized, continuous log amplitude coverage supplies a small-stage
reward; phase score becomes available after coverage reaches its target.
Use anatomy only to focus some parameter proposals on cells projecting to
middle/posterior muscles and their existing inputs; all unknowns remain eligible.
This proposal prior is not runtime neural drive or a muscle teacher.

Also implement cycle-averaged translation diagnostics: integrate centroid
position over complete rising/falling-reference cycles, then regress averaged
position over three late blocks. Require agreement across phase origins and
report whole-shape closure, rather than rewarding endpoint displacement.
A future cycle-movement fit must start from a validated body-wave checkpoint.
Wave qualification does not imply propulsion. Preserve separate checkpoints.

Four independent four-core CPU searches, four-hour cap including validation.
Validate 300s, +/-1mV initial states, finer neural/body timesteps and held voltage.
Require qualified active cases plus numerical agreement; held voltage must stop
the motion. Test synthetic head-only, standing, traveling, drifting and static
cases, then run a short local end-to-end trial before push. Agent does not
launch remote jobs; user retains pull-and-nohup workflow. No success inferred.

E035 pre-pilot scope clarification: body-wave/cycle-movement objectives use
0.0625–16x density/effective-connection bounds by default (explicit CLI
override retained), versus 0.25–4x for historical amplitude/endpoint objectives.
Most non-RMD priors were initialized at1% transferred channel densities. The
wider unknown-parameter neighborhood is a recruitment hypothesis, not an
empirical physiological bound. LEAK, capacitance, kinetic equations/reversals
and body/NMJ constants remain fixed. Viability/voltage/saturation/clipping
checks still apply. All changes precede the new fitting pilot.
Geometry guard is retained: late mean body-axis norm>.5 and turn<.35 rad.
Search can stop at a short-window wave target and spend the remainder on
independent validation; failed long checks are not promoted.

E035 local pilot executed:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.train_autonomous_wiggle --objective body-wave --output-dir /tmp/r-elegans-E035-wave-pilot --minutes 6 --iterations 2 --validation-seconds 30`
stdout `/tmp/r-elegans-E035-wave-pilot.log`. Completed215.494557 s, two
proposals, selected1, expected exit2 because no wave qualified. All six short
validation cases completed, frozen fields unchanged, all injected currents zero.
Coverage score2.09238e-5 ->1.36574e-4; absolute regional posterior joint p2p
3.04639e-6 ->3.40285e-6 rad, drive5.61378e-7 ->1.00748e-6. These are tiny
recruitment changes, far below0.003 targets; do not call this a learned wave.
103 anatomically focused cells; all9616 values remain eligible. Six validations
remain in spread stage, and the held-voltage control has no sustained rhythms.
All accepted checkpoints, histories/masks/coverage/RNG, source/environment and
full trajectories retained externally. Original model assets remain untouched.

Full suite205 passed in69.60s. Subsequent targeted wave/cycle/joint tests6
passed in2.27s include observed slow-frequency recognition, graded non-RMD
rhythm, standing/in-phase/head-only/drifting shape rejection, known cycle-mean
translation recovery at several phase origins, numerical refinement agreement
and anatomy-focused proposals without freezing remaining cells.
E036 precondition verified: a source without passed body-wave metadata is
rejected before network_initial creation; no E036 simulation was run.
Both launcher shell checks, CLI help and diff checks are required before push.
The old movement launcher now delegates to E035; explicit legacy objective
remains reproducible from recorded old source. No server job started by agent.

E035 OneDrive local archive verified:405 files, manifest SHA256
`bbe831c3ec46f89b471f5c41b7edc2a810a306f75eddc841d979f56136392187`.
Local preservation is verified; cloud synchronization is not. Pilot fitted-pair
hashes/identity and no-drive/frozen-field flags verify. No body-wave skill
is yet qualified. Final progress-log additions report regional amplitudes
and cycle-mean speed explicitly; numerical objectives unchanged after pilot.

## 2026-11-09 — E035 recruitment improves, posterior target unmet

E035 server result, user-reported 2026-11-09:
`autonomous-wave-20261109T032054849657Z`, 471/466/471/466 proposals
(1,874 total), all exit 2/passed=False and remain in spread stage.
No full body-wave checkpoint is qualified and E036 must not start.

Seed 2 has strongest coverage 0.38437. Long regional harmonic joint p2p
[posterior,middle,anterior] = [7.96799e-5,0.00258090,0.02159237] rad.
Middle is 86.0% of the 0.003 target; posterior is 2.66% (37.65x too small).
These amplitudes are similar across perturbations/refinement, but phase span
remains below target (0.253 rad long,0.072–0.106 rad in short checks).
The logarithmic coverage score 0.384 does not mean 38.4% amplitude completion.

Seed 3 retains a larger second shape mode (0.004287 =0.429%, versus 2% target)
and stable ~1.06 rad total phase span, but posterior p2p 5.53e-5 rad remains
~54x below target; middle 0.001848rad is 61.6% of target. Preserve it as a
separate phase-diverse continuation candidate, not a validated traveling wave.
Phase span alone does not certify direction, smoothness, drive phase or power/
locking gates; those detailed fields are absent. Seeds 0/1 also lack posterior
amplitude. Held-voltage cases remain unqualified with feature values omitted;
full null-control amplitudes are not shown by this compact command.

Decision: preserve all candidates, prioritize posterior recruitment from
seed 2 while retaining seed 3 separately; do not restart blindly from the
original head-only reference or advance to propulsion. Before choosing
continuation bounds, inspect original search state and bound saturation;
a continuation must record its parent, RNG mode and any change of reference
bounds. No new search is launched by this records update. Exact server
artifacts/input/source/environment remain unretrieved. Numeric record:
`experiments/records/E035-server-20261109.json`.

Raw console SHA256 `f955597abb23bb08e33e1ae0eae5093146203e284e9cac4c00ca4d5e02c28ddf`.
No exact elapsed time or baseline-to-final gain is inferred from this compact
validation-only output. No thresholds lowered and no source/model changes.

User-reported console preserved locally under OneDrive evidence/E035/
server-report-20261109/console.txt with SHA256.json. This retains the report,
not a copy of full remote traces/checkpoints or proof of cloud synchronization.
Extraction verified4 workers x6 cases and1874 proposals. Records/docs only;
JSON consistency and diff whitespace checks passed, no code/physics change.

## 2026-11-09 — E037 planned amplitude-only parameter forks

User corrects the next stage: build autonomous amplitude first, leave control
for later. Remove phase progression/locking, common-frequency power, PCA,
heading/straightness and translation from this new objective and qualification.
Allow standing, opposing and asynchronous sustained bends. Preserve physical
finite-voltage/no clipping/no excessive saturation checks, zero injected
currents, full 302-neuron forward model and all9,616 unknown densities/edges.

Measure each joint/filtered antagonist drive independently. Require repeated
late excursions, not a static bend or one settling transient. Use minimum
p2p across5 late windows and repeated peaks/troughs; regional medians across
posterior/middle/anterior joints prevent one hinge dominating a region. Target
0.003rad regional median bend and0.003 regional median filtered drive, plus
a viable sustained max bend>=0.01rad. No waveform, frequency, phase, direction
or velocity target. Scores are not comparable with E035 coherent-mean scores.

Continue from E035 seeds2 and3 in cohort autonomous-wave-20261109T032054849657Z,
with four independent fresh-RNG parameter-search forks. Preserve the original
anchor/16x bounds: load source_inputs and saved absolute offsets rather than
re-centering around selected values. Verify reconstruction equals parent best,
match allowed masks, preserve the full completed parent worker inside the new
record, and re-evaluate baseline under the new objective. This is parameter
warm start, not exact optimizer/RNG or physical-state resume. Use seeds1000..1003
with parents2/3 round-robin. Retain all originals, no in-place overwrite.

Four-hour cap including validation; 300s, +/-1mV initial states, neural/body
refinement and held-voltage checks. Qualification is amplitude only; no wave or
locomotion claim. Before server launch, test phase-insensitive grading and
original-anchor restoration/bounds, and run a local warm-start pipeline trial.
Raw histories/checkpoints/provenance remain external; prepare/test/push and
leave remote launch to user. No parent hashes or successes are invented.

E037 pre-pilot clarification: remove the inherited >=0.01-rad maximum-head
amplitude requirement too. A balanced body-wide0.004-rad wiggle should meet
the0.003 regional goal even if no hinge retains the previous large head bend.
Optimization viability uses repeated joint excursion>1e-5rad and sustained
neural variation>=0.05mV; qualification still requires every regional median
joint/drive amplitude>=0.003. The original stronger head checkpoint remains
unchanged separately. This avoids treating preservation of one head skill as
a control target while learning distributed amplitude.

E037 final pre-pilot simplification: only physical joint-bend amplitudes enter
the amplitude objective and promotion. Filtered muscle-drive amplitudes remain
diagnostics, not targets or refinement gates. Passive transmission of neurally
generated bending is acceptable here; controlled muscle recruitment is deferred.
All runtime forces still originate from the unchanged neuron->NMJ->muscle->body
loop, with no artificial body forcing. This aligns the new stage strictly with
the user's amplitude-first instruction. No E037 fitting trial has run yet.

E037 local warm-start pipeline executed:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.train_autonomous_wiggle --objective body-amplitude --warm-start-worker /tmp/r-elegans-E035-wave-pilot --output-dir /tmp/r-elegans-E037-amplitude-fork --seed 1000 --minutes 6 --iterations 1 --validation-seconds 30`
stdout `/tmp/r-elegans-E037-amplitude-fork.log`. Completed192.687235 s, one
accepted proposal, expected exit2/no qualification. Full parent worker copied;
old offsets/masks exactly reproduce incoming best parameters at original
source anchor hash482995e03f28cfd62704b20630c28c2fb1706aec87dbd4e2cc6eecd9315a2ff6.
Original factor16 bounds preserved; parent had128 nonzero offsets and0 at
bounds. This local parent is not the actual four-hour server seed2/3.

New amplitude baseline score0.0573764 ->0.0581275. Regional posterior/middle/
anterior median sustained p2p changes [3.87430e-6,1.64837e-4,0.0204343] to
[3.93391e-6,1.58483e-4,0.0202728] rad. Tiny posterior increase, far below0.003
target; do not call a body-amplitude success. All six short validation cases
complete; zero injected currents/frozen-field checks and fitted-pair checksum
verify. No wave or movement claim. The pilot's RMD-count log default0 was
not an RMD measurement: this objective doesn't compute that count. Final
logger uses null for unmeasured RMD and reports sustained total neural rhythms.
This logging correction does not change scoring or dynamics.

Full suite209 passed in69.71s. Targeted amplitude/warm-start tests verify
phase/frequency-independent amplitude, ramps/transients rejection, full parent
copy/checksum/mutation protection, original-anchor reconstruction and bounds.
New per-region medians must not be directly compared to previous coherent
regional-mean amplitude scores. No exact RNG/physical-state resume claim.
Source, histories, all checkpoints and validation traces retained externally.
Both recent standard launchers delegate to the amplitude-only stage; no
remote process started by agent.

E037 local archive verified:448 files, manifest SHA256
`e973d8097743caa6ba06e63d1117b33589f270eb24a6f60b235b2f76066ed2d7`.
Incoming parent parameters/fixed fields/configuration compare exactly to
network_initial in the new run; fitted-pair hashes and all six validations
verify. No physical/RNG exact-resume or amplitude-success claim. Targeted
amplitude/warm-start tests after final guards:4 passed in1.65s. Offset arrays
are cast explicitly to their original anchor dtype; default float32 dynamics
unchanged. All three shell launchers, CLI help and diff whitespace checks pass.
This verifies local preservation, not completed cloud synchronization.

## 2026-11-09 — E037 reported server amplitude results; E038 readout audit planned

User-reported cohort `autonomous-body-amplitude-20261109T152102003705Z`
completed with seeds1000–1003 all exit2 / passedFalse. Long validation regional
median sustained joint excursions [posterior,middle,anterior] rad:
1000 [1.05425715e-5,0.005640725,0.037973873] (467 proposals);
1001 [2.83159316e-5,0.046748955,0.013630696] (463);
1002 [4.93228436e-6,0.002148852,0.022814460] (469);
1003 [2.61440873e-5,0.032198064,0.007909603] (463).
Head exceeds0.003 in4/4, middle3/4, posterior0/4; best tail is105.95x below
target. Coverage0.0703–0.2354 is a nonlinear score, not fraction of skills
completed. All five active validations remain unqualified; held-voltage scored
excursions are zero. Perturbation/refinement preserve the posterior deficit.
No actual server baseline, source/checkpoint hashes or raw trajectories were
provided; no exact gain or causal explanation can be inferred. These medians
must not be directly compared with E035 harmonic regional amplitudes.

E038 planned LOCAL static diagnostic: inspect the shared bundled anatomical
NMJ→muscle→joint readout for missing posterior projections or zero sensitivity.
Use `autonomous_wiggle_v1` network/parent, bundled topology, JAX autodiff of the
frozen readout at initial membrane voltages. Report anatomical inputs and
Jacobian norms by joint; this is not the learned server state or an embodied
simulation. Fixed all parameters; no fitting, seeds or external runtime drive.
Command: `MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python /tmp/e038_readout.py`.
Accept the audit only as a structural/sensitivity check; a nonzero Jacobian
cannot demonstrate sustained tail activity. Do not launch more training until
actual posterior neural/muscle/drive traces and search coverage are inspected.

E038 trace-analysis plan extension: add a read-only server utility computing
late-window sustained amplitudes along voltage→release→individual muscles→
projected dorsal/ventral→difference→filtered drive→joint angle. Reconstruct
saved muscles from checkpoint/frozen readout as an alignment check; report
per-joint input-neuron counts, largest posterior voltage/release excursions,
D/V peak-to-peak cancellation diagnostics and parameter coverage/bounds if
available. Use saved long validation, no simulation or fitting. Smoke-check on
the existing local E037 pilot. Outputs outside source run/repo, input hashes
and environment recorded; diagnostic output is not qualification.

E038 executed locally: static audit JAX0.6.2 CPU; parent/network hashes in
`experiments/records/E038-posterior-audit.json`. Each posterior joint has16–58
anatomical presynaptic neurons and nonzero initial-voltage net-drive Jacobian
L1 norms0.00978–0.01519 per mV. No structural disconnection found. This is
not an actual server-state sensitivity measurement or proof of tail movement.

Read-only trace utility smoke command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.analyze_posterior_activity --run /tmp/r-elegans-E037-amplitude-fork --output-dir /tmp/r-elegans-E038-posterior-audit-final`
Exit0. Sampled saved muscles reconstruct within1e-5 (exact measured error
in compact record). At the LOCAL one-proposal pilot, posterior regional median
of each joint's maximum input sustained voltage amplitude0.00170135mV,
release0.0000173151; corresponding middle0.25523mV and anterior27.8411mV.
Posterior joint median amplitude3.93391e-6rad; no posterior joint had an input
above the0.05mV rhythm threshold. Zero of9,616 offsets at original16x bounds.
This suggests weak neural recruitment in that pilot, not a demonstrated cause
of the four server outcomes. No new simulation or fit ran. Server audit pending.

Preserved script/static output/full local trace-analysis output and environment
at `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E038` (199 files;
manifest SHA256 c62423ba3e8d061969e1a86c460089e6e4ec873c3b71e8310d54a1a1f77217c1).
Local copy verified; cloud sync not independently checked. E037 raw server
artifacts remain on mbslnx2, not retrieved. Decision: audit saved server long
validations along voltage→release→muscle→net drive→joint before deciding
whether to target posterior excitability, recurrent recruitment or NMJ transfer.
Amplitude-first criterion and zero external drive remain unchanged.

## 2026-11-09 — E038 user-reported server posterior audit

User pasted all four worker reports from the read-only utility after pulling
5b87117. Exact parsed reports and attachment SHA256 are committed in
`experiments/records/E038-server-20261109.json`. Server output directory's
expanded timestamp, full report/input hashes and final exit status were not
provided. A CUDA plugin discovery traceback preceded four CPU result lines;
this did not prevent these analyses from returning. No training occurred.

For seeds1000/1001/1002/1003, posterior regional medians of each joint's
maximum input sustained voltage excursion are0.01110/0.23174/0.000816/0.11172mV;
release excursions3.44e-5/0.005066/1.01e-7/0.002521. Corresponding anterior
voltage32.6–34.2mV and release0.652–0.672. Posterior median rhythmic input
counts per joint0/1/0/1 versus anterior21/23/11/31 (inputs can repeat across
joints; these are not unique regional neuron counts). Posterior median of
per-joint median individual-muscle sustained excursions is0 in all seeds;
net-drive medians6.23e-6/5.96e-8/2.98e-8/1.47e-5, filtered-drive medians
4.34e-6/0/0/1.05e-5. Head/middle activity is substantially stronger.

Interpretation: the posterior deficit already exists in presynaptic neural
activity/release and muscle activation, rather than only appearing in body
mechanics. This supports weak posterior neural recruitment as the next working
hypothesis. It does not prove which channels/connections cause it, rule out
NMJ attenuation/cancellation or mean every tail muscle is static. Small tail
angles despite zero median filtered-drive score can include mechanical
coupling and different channels contributing to medians. No causal experiment
was performed. Bounds hit22/18/16/11 of9616; not global saturation, but
individual influential parameters could be bound-limited.

Decision: keep amplitude first, no external drive, no phase/control objective.
Next proposed change is a posterior-focused recruitment/amplitude curriculum
using learned candidates, retaining fixed stereotypical parameters and original
bounds until per-parameter evidence justifies changing them. Preserve existing
head/middle candidates independently. No new training was launched or new
qualification claimed. Raw trajectories/full server report remain on mbslnx2.

Exact pasted stdout and compact parsed report archived locally under OneDrive
`notebook/evidence/E038/server-report-20261109`; manifest SHA256
56624ce317b23106b8f54ccdfe310da78ddb868d9dbefa65797c672f08d65134.
Cloud synchronization not verified. Earlier E038 manifest covers the earlier
local-audit files only; this server-report subdirectory has its own manifest.
