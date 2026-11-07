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
