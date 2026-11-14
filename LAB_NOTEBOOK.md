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

## 2026-11-09 — E039 planned short autonomous tail recruitment trial

User authorized a short run to get the tail moving. Plan:20-minute mbslnx2
cohort, four independent four-core workers, fresh RNG seeds2000–2003 forked
round-robin from reviewed E037 seed1001/seed1003. Preserve original anchors,
16x bounds, entire lineage and all existing fixed fields; all302 neurons and
9616 unknown parameters remain eligible. All injected currents stay zero.
No phase, heading, speed or control objective; no NMJ/body changes.

Tail-only scoring:95% logarithmic sustained joint-amplitude coverage in the
posterior four joints (same median definition, target0.003rad) plus5% capped
posterior motor-neuron sustained voltage recruitment, to provide a weak search
signal before measurable tail motion. Recruitment is a fitting heuristic, not
an electrophysiological target or a qualification gate. Keep body-wide metrics
separate. Tail qualification requires validity and posterior median>=0.003rad;
no other region has to improve. Proposals:90% sparse tail-projecting neurons'
channel densities or existing incoming chemical/electrical edges,10% original
full-space proposals. Independent neurons/edges remain untied. No bounds change.
Fit20 simulated seconds; long validation60s plus initial-voltage perturbations,
finer brain/body steps and held-voltage control. Same25% amplitude refinement
tolerance, tail only. Passing this short assay is a provisional tail primitive,
not long-duration qualification, body-wide motion or locomotion.

Before remote launch: targeted synthetic criterion/proposal tests and a local
2-proposal smoke run using `/tmp/r-elegans-E037-amplitude-fork`, fit12s,
validation20s, budget6minutes; record outcome separately. Actual E037 server
parents are unavailable locally; never substitute local parents for claimed
server results. Server launch remains user-run after push via nohup script.

E039 LOCAL pilot executed with the planned command above (stdout
`/tmp/r-elegans-E039-tail-pilot.log`). Completed95.8146s, two proposals both
rejected, selected iteration0; exit2, unqualified. This is not improvement
evidence. All six validations completed. All external currents zero and frozen
fields unchanged; original-anchor parameter reconstruction passed. Anatomical
focus71 distinct presynaptic cells across the four posterior joint projections.
Baseline posterior joint median2.86847e-6rad; no accepted parameter changes.
Learned incoming parameters preserved. Best checkpoint SHA256
1ce277c4e5be0d28ef0324de8c55fea0e225167f16cf49d1a67e85e0da294815.
Long validation20s is deliberately only a smoke check; actual server uses60s.

Verification:10 targeted tests passed2.81s; full suite212 passed68.75s;
launcher shell syntax passed. Synthetic tests reject head-only and neural-only
motion claims, allow independent tail qualification, check refinement and
bounded sparse proposals. Software tests do not establish biological validity.

External archive `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E039`:
685 files including recursive parent, all selected/final checkpoints, RNG/state,
full history, traces, source/environment records, stdout and implementation.
Manifest SHA2567aa6100c42a70f5eeb3a35bd221faabbb4dd01c86709fd5d6292b287b2174680.
Copy verified locally; cloud synchronization not independently verified. Compact
record `experiments/records/E039-tail-amplitude.json` contains exact invocation,
parent hashes, summary and limitations. Decision: push prepared20-minute server
launcher for user execution; no server launch or tail qualification claimed.

## 2026-11-09 — E039 short server trial reported, tail remains below target

User reported completed cohort
`/home/carsonc/r-elegans-results/autonomous-tail-amplitude-20261109T200140261630Z`.
All four workers completed with exit2/passedFalse: seeds2000–2003 ran37,38,38,38
proposals (151 total). This is a criterion failure, not a reported worker crash.
Parsed before/after and all six validations, plus exact attachment hash, are in
`experiments/records/E039-server-20261109.json`. No independently retrieved raw
traces, checkpoint hashes or source/environment records in this excerpt.

Fit posterior median sustained joint p2p rad, baseline→best:
2000 2.83010304e-5→2.90460885e-5 (+2.6326%);
2001 2.60770321e-5→2.61515379e-5 (+0.2857%);
2002 2.83010304e-5→2.83867121e-5 (+0.3028%);
2003 2.60770321e-5→2.61515379e-5 (+0.2857%).
Neural recruitment score baseline→best:
2000 0.06667194→0.06790449;
2001 0.09046980→0.09049947;
2002 0.06667194→0.06675016;
2003 0.09046980→0.09045196 (slightly down; mixed score permits this).
This auxiliary score is neither firing rate nor a biological measurement.

Best long-validation tail amplitude2.90498137e-5rad (seed2000),103.27x below
0.003 target. All five active checks unqualified in every seed; posterior
amplitude deficit persists under perturbation/refinement. Held-voltage scored
excursions zero. Head/middle amplitudes remain above0.003rad in all active
checks, but no retention/control claim follows. Small nonzero tail excursions
are not proof of active posterior-muscle recruitment (mechanical coupling is
possible); neural/muscle diagnostics were not rerun here.

Interpretation: sparse posterior proposals provide tiny positive fitting gains,
not a demonstrated tail primitive or evidence a longer identical run will
solve recruitment. Only37–38 proposals per worker; no convergence or structural
impossibility claim. Proposed next diagnostic: determine whether posterior-cell
models can sustain autonomous oscillations within existing conductance bounds,
then consider informed initialization rather than another blind extension.
Isolated-cell feasibility would not by itself establish network/body behavior.
No new simulation, training or parameter-bound change has been performed.

Exact server stdout and parsed summary archived under external OneDrive
`notebook/evidence/E039/server-report-20261109`; its separate manifest SHA256
45fac83c955cc3b5e3dc03720c52d40e012ba65652756f5cf2d580a3a60275f6.
Local copy completed; cloud synchronization not independently verified.

## 2026-11-09 — E040 planned posterior excitability/oscillation feasibility audit

User authorized the proposed diagnostic. Question: do posterior channel models
support autonomous rhythms within the current original-anchor density bounds,
and does anatomical coupling suppress or enable the same configurations?
Diagnostic interventions are explicitly separate from the working model;
no learned checkpoint is overwritten, no current is injected, no leak/C/kinetic,
release/NMJ/body field is fitted or changed.

Local parent `/tmp/r-elegans-E039-tail-pilot` (local smoke lineage, not actual
E039 server seeds). Target cells: all71 anatomical inputs to posterior four
joint projections; include their original identities. All density candidates
bounded to original source log-scales +/-log16; missing terms and LEAK fixed.
Probe current, original anchor, all active terms high/low, inward-high with
outward-low, inward-high with outward unchanged, individual calcium/NCA high,
and fixed-seed log-uniform combinations. This finite screen is not a proof of
impossibility. Isolated simulations use the existing gate/Ca equations and
exponential voltage integrator, checked against the network with zero edges.
12s screen, sample2ms, integration0.1ms, no drive; selected candidate gets
30s confirmation, +/-1mV initial voltage, half integration step and full302-cell
anatomical-network comparison. Baseline full network and isolated posterior
subcircuit comparison retain current parameters. Isolated/subcircuit graph
ablations remove edges only for diagnostics; they do not alter the saved model.

Candidate criterion: existing repeated late-window excursion detector >=0.05mV,
finite voltage in(-120,80)mV; screen count, magnitude, release excursion and
resting voltage reported separately. Select by number of qualifying cells,
then median excursion; verify every screen candidate remains within bounds.
Do not require every real motor neuron to oscillate intrinsically; network
oscillations can be emergent. No biological validity or body movement claim.

Build reusable script with optional completed worker parent and pinned server
parent from E039 seed2000; retain all source/checkpoint/input hashes and exact
profile arrays/seed/RNG state externally. Run local audit now, then push for
user to repeat against the actual server parent. Planned command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl JAX_PLATFORMS=cpu .venv/bin/python -m scripts.assay_posterior_excitability --worker /tmp/r-elegans-E039-tail-pilot --output-dir /tmp/r-elegans-E040-excitability --random-profiles 8 --seed 4000`

E040 adaptive diagnostic extension planned before execution: initial19 isolated
profiles produced no repeated late-window excursion. Test all19 again from
uniform initial voltages-65,-50,-35mV to check alternative attraction basins;
these are initial-state changes only, not sustained external input. Use same
71-cell identities,12s screening,0.1ms integration and saved exact profile
arrays. Preserve all traces and outputs externally; finite negative findings
remain insufficient to prove absence of oscillations. Command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl JAX_PLATFORMS=cpu .venv/bin/python -m scripts.assay_posterior_basins --audit-dir /tmp/r-elegans-E040-excitability --output-dir /tmp/r-elegans-E040-basins`

E040 LOCAL main assay completed118.3476s, exit0;19 profiles x71 cells=1349
screened cases, no isolated rhythms. Selected-profile30s confirmations at
nativeinitial,+1,-1mV and half-step also0 rhythms. Implementation selection
ranks count then maximum excursion (planned text said median);all screened
excursions were zero so this did not affect selection. Predeclared
inward-high/outward-low probe chosen explicitly as an unsuccessful diagnostic,
not a fitted regime. All screen/confirmation voltages finite and in bounds.

Alternate initial-basin assay completed107.7144s, exit0;19 profiles x3 initial
voltages x71 cells=4047 cases,0 rhythms, all valid. Together5396 primary
isolated trials, plus284 selected-profile confirmation cell trials. No currents
injected.12s screens/30s confirmations do not rule out slower rhythms, other
initial states or untested conductance combinations. This is not an exhaustive
parameter feasibility result.

Current connected network has3 qualifying tail-projecting neurons (SMDVR,
SMBVL,SIBVR), largest sustained excursion0.255234mV;total network18 rhythms.
All39 selected DA/DB/DD/VA/VB/VD/AS-class cells remain below0.05mV, maximum
0.00345612mV. The selected-cell-only subcircuit has0 rhythms. Strong-inward
candidate in fullnetwork has1 selected rhythm, max0.0569344mV, with depolarized
plateaus;not an improvement. Isolated strong-profile late mean voltages range
-42.85 to47.65mV;excitation alone is possible without sustained cycles.
NMJ-defined selection includes non-VNC neurons and can exclude necessary
interneurons;do not equate this graph intervention with the complete posterior
functional circuit. No body simulation or locomotion qualification performed.
No candidate promoted or production model changed.

Interpretation: isolated oscillator assumption is unsupported in this finite
bounded local screen;weak selected rhythms depend on the rest of the network.
This does not show that every posterior neuron should be an intrinsic oscillator,
that the channel library is insufficient,or that full-network oscillator loops
cannot work. It argues against indiscriminate inward-density gain as a fix.
Actual server parent differs from local smoke lineage;confirmation launcher
pins reviewed E039 seed2000 and saves all diagnostics via nohup on four cores.
No server launch is claimed;standing workflow is push and user launches.

Verification:5 targeted tests passed3.25s,including zero-edge integration
agreement and profile bounds/frozen leak;full suite214 passed71.26s. Both CLI
assays executed successfully;launcher shell syntax passed. Raw local outputs,
all profiles,parent lineage,RNG/state,source/environment records and stdout
archived at `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E040`:
1190 files,manifest SHA256
8280865b8af2950b997b4dd13a680c960f0262de8c0cd801a53e4d9f2739b1bc.
Local archive copy completed;cloud sync not independently verified. Compact
record `experiments/records/E040-posterior-excitability.json` includes hashes,
case counts,connected comparisons and explicit limitations.

## 2026-11-09 — E040 actual server-parent diagnostic completed (user report)

Run `posterior-excitability-20261109T214558Z-615489`, pinned E039 seed2000
confirmed by user-pasted process command. Exit0: diagnostic completed, not an
oscillator/behavior pass. User summary reports0 isolated-screen rhythmic cell
trials; selected inward_high_outward_low; robust isolated rhythm list empty.
Alternate-initial-state screen4047 cell trials,0 rhythms,145.291797s.
No full source/checkpoint/profile hashes pasted; raw artifacts remain on server.
Compact transcription `experiments/records/E040-server-20261109.json`.

Current full network:7 rhythmic selected tail-projecting cells (AVKR,SMDDL,
SMDDR,SMDVL,SMDVR,SIADL,SIAVR),maximum sustained p2p7.61054039mV.
Selected-cell-only subcircuit:0 rhythms,0 excursion. Full network with bounded
strong-inward/weak-outward density intervention:6 selected rhythms (same except
SIADL),maximum3.43791199mV. Maximum excursion fell~54.8%;this compares maxima,
not necessarily amplitudes of the same neuron. Actual server numbers supersede
local smoke values3→0→1 for interpretation of the trained E039 parent; keep
both records as distinct initializations.

The population is defined by any direct NMJ contribution to the posterior four
joint projections, not by posterior motor-cell identity. No reported rhythmic
cell is in the DA/DB/DD/VA/VB/VD/AS classes. Thus a seven-cell selected rhythm
count does not establish recruitment of those motor classes. A neural reward
averaged over this broad population can improve without useful tail drive.
Subcircuit graph ablation can remove necessary interneurons, so do not infer
absence of a complete posterior functional oscillator. Isolated quiescence
also does not imply that each motor neuron must become a pacemaker.

Decision: no diagnostic candidate promoted, no new training launched. Next
proposed curriculum should recruit effective posterior motor/muscle activity
from the existing internally generated rhythm using targeted recurrent and
channel-density blocks, retaining zero external input/fixed stereotypical fields.
Avoid another unchanged hill-climb extension or indiscriminate inward gain.
Keep amplitude first;phase,control and propulsion still deferred. No biological
impossibility or model-sufficiency claim follows from this finite screen.

Clearly labeled server-summary transcription archived under external OneDrive
`notebook/evidence/E040/server-report-20261109`;separate manifest SHA256
 ae2e353b30a9734bd6281c8086819a897e93a0dddcc5d335d358f69b955f2f13.
This is a transcription of user console messages,not independently retrieved
raw stdout or server artifact backup. Cloud synchronization not verified.

## 2026-11-09 — E041 planned option1 posterior motor recruitment trial

User selected option1. Replace the broad71-cell neural auxiliary with a
reproducible strong-posterior motor target: canonical DA/DB/DD/VA/VB/VD/AS
classes, >=50% of normalized NMJ-to-joint contribution in posterior four joints,
retaining cells accounting for95% of eligible posterior contribution. Weight
by the frozen parent effective NMJ readout (unsigned anatomical magnitude).
Audit exact selected names/contributions before fitting; no biological cell-
specific physiology claim. An initial anatomy probe failed on a read-only
NumPy view; corrected to nonmutating division. It found appropriate strongly
posterior projections (VD13,VA12,DA09,DD06,VB11,etc), distinct from E040 rhythmic
SMD/SIA/AVK population. No fit occurred in this metadata check.

Score60% tail bend coverage,35% actual projected dorsal-minus-ventral muscle
excursion,5% contribution-weighted sustained release from selected motor cells.
Both physical targets are median per-joint sustained p2p>=0.003 in posterior
four joints; release auxiliary is heuristic,never qualifies a skill. Common-mode
muscle fluctuations or passive tail bending alone cannot qualify. No prescribed
frequency,phase,control or propulsion. All runtime currents zero. Frozen leak,
C,kinetics,reversals,Ca,release,NMJs/body; original16x bounds and anchors retained.
All9616 uncertain existing density/recurrent strengths remain eligible.

Proposals:10% original full-space;45% existing one/two-hop chemical paths from
current internally rhythmic transmitter-release sources into a weighted posterior
motor target;25% coordinated existing incoming-chemical blocks;20% motor density
and neighboring symmetric gap blocks. Paths use existing anatomy only, no
new edges or forcing. Current best trace supplies driver identities; record them
in search state and source policy. Joint density/connection changes allowed,
independent values remain untied. This tests recruitment of an existing rhythm,
not a requirement for isolated motor-cell pacemakers. No driven teacher.

Planned server trial30min incl validation,4workers x4cores, seeds5000–5003
round-robin from reviewed E039 seeds2000/2001. Fit20s,long60s plus perturbation,
finer brain/body steps and held-voltage. Refined tail bend AND net-muscle drive
must agree within25%;no head/middle retention constraint,parent skills preserved
independently. Short pass provisional,not long-duration or locomotion evidence.

Before server launch: meaningful synthetic tests for target specificity,
common-mode/passive rejection and anatomical path/bound preservation;local
6-proposal fit12s/validation20s/budget6min using cached local E039 smoke parent.
Command planned:
`MPLCONFIGDIR=/tmp/r-elegans-mpl JAX_PLATFORMS=cpu .venv/bin/python -m scripts.train_autonomous_wiggle --objective body-amplitude --posterior-motor-focus --fit-scope neurons-connections --warm-start-worker /tmp/r-elegans-E039-tail-pilot --output-dir /tmp/r-elegans-E041-motor-pilot --seed 5000 --minutes 6 --iterations 6 --fit-seconds 12 --validation-seconds 20`
Server stays user-launched after push;actual server parents unavailable locally.

E041 LOCAL trial executed as planned,133.960614s,exit2/unqualified. Six proposals,
one nominal-score acceptance at iteration3,six parameters changed:VA10 EGL36,
EGL19,SLO1_UNC2,KCNL densities plus existing AVER→AVAR→VA10 chemical path.
This is a joint intervention,not attribution of benefit to an individual term.
Baseline internally rhythmic release sources RMDDR,RMDDL,RMDVR,AVER;initial
one/two-hop paths reach12/26 targets. All prior unknowns remain globally eligible.
Policy selects26 posterior motors (full names/weights in compact record),none
of the E040 SMD/SIA/AVK rhythmic population. Frozen fields unchanged,all injected
currents zero,original anchor/bounds and parent reconstruction verified.

Nominal training posterior bend median2.86847e-6→3.39746e-6rad,net-muscle
median4.35114e-6→7.51019e-6 (dimensionless). These tiny gains are NOT robust
actuation evidence:brain-refined validation bend2.90573e-6rad,median net-muscle
and motor-release excursion scores both0. Long20s validation bend5.11855e-6,
net muscle7.36117e-6. All five active validations fail qualification;held-voltage
scored excursions0. All six cases completed. Nominal objective can still favor
numerically sensitive small amplitudes;independent refinement gates reject
such candidates. Search remains nominal timestep,not a proven numerically
robust optimizer. No skill pass,realistic animal behavior or model sufficiency
claim. Server trial is exploratory;do not automatically extend if it stalls.

Targets use0.003rad bend and0.003 dimensionless net-muscle excursion. Both
required;release auxiliary alone cannot qualify. Software verification12
focused tests passed3.05s,full suite218 passed71.91s;added drive-only rejection
assertion then4 focused tests passed1.11s. Shell syntax passed. Tests exercise
nonmotor/anterior exclusion,head-only reward invariance,common-mode/passive/
neural-only/drive-only rejection,refinement,driver identity and existing directed
path/bound preservation. They validate software,not the biology.

Best local checkpoint SHA256
4cc72b2702f4c3b04f988e21df1b54d4d9d87c1814644e0df82506e0a1eeabd5.
Raw local artifacts,complete copied lineage,selected/final checkpoints,RNG and
history,all validations,source/environment,stdout and new implementation archived
at `/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E041` (934 files),manifest
SHA256391c05f8f95385d43bd872c64a9a4cb939177b8552747e8dd0c0ab19559ff620.
Copy completed locally;cloud synchronization not independently verified.
Compact record `experiments/records/E041-posterior-motor.json` preserves exact
invocation,hashes,policy and summary. Decision:push30-minute nohup launcher from
reviewed actual E039 parents for user launch;no server execution claimed.

## 2026-11-09 — E041 server option1 trial: robust subthreshold recruitment gain

User reported cohort
`/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261109T222806260061Z`.
Seeds5000–5003 all completed exit2/passedFalse,77/71/77/70 proposals (295 total).
All six validation rows provided per worker. Exact parsed fields and attachment
SHA256 in `experiments/records/E041-server-20261109.json`;raw server histories,
checkpoints,source/environment hashes and parameter changes not retrieved.

Strongest candidate5003:fit tail median bend2.61515379e-5→1.87389553e-4rad
(7.1655x);net-muscle median1.46850944e-5→4.79921699e-4 (32.6809x).
Motor release auxiliary0.0103797→0.0866077 (heuristic,not a physical release
amplitude/firing rate). Long validation1.87478960e-4rad /4.80100513e-4 muscle.
Perturb±1mV retain gains. Brain-refined1.87404454e-4rad /4.87297773e-4 muscle:
+0.00795% bend,+1.5369% muscle relative to nominal training. Body-refined
1.87397003e-4rad /4.79951501e-4 muscle. Held-voltage scored excursions0.
The dominant gain persists across all provided checks,unlike the tiny local
pilot's coarse-step-only changes. Still16.01x below0.003rad bend and6.251x
below0.003 activation,so this is recruitment progress,not a skill qualification.

Seed5002 gives smaller stable gain:fit bend2.90460885e-5→5.29140234e-5rad
(1.8217x),net muscle0→5.64455986e-5;brain-refined5.26886433e-5rad /
5.58793545e-5. Preserve as alternate independent candidate. Seeds5000/5001
improve net muscle/release while bending falls slightly;composite score tradeoff
must not be called a bending improvement. Every active validation remains below
qualification in every worker;held-voltage scored excursions0 for all.

Interpretation:targeted motor/path proposal and physical muscle score produced
substantially better,short-window numerically consistent posterior recruitment
in one seed. Joint training design does not isolate which parameter/proposal
caused it. No proof of natural biomechanics,long-duration stability,phase control
or locomotion. Numerical consistency of below-target amplitudes is not an
acceptance pass;qualification still requires both physical thresholds.

Decision:preserve all parent/new independent candidates;prefer fresh-RNG
continuations of5003,with5002 as alternate,retaining original source anchors,
16x bounds,zero drive and fixed fields. No new training launched or continuation
budget selected here. Inspect future outcomes rather than blindly extending
all workers. Successful checkpoint remains on server,not bundled or promoted
as qualified in Git.

Exact pasted stdout and parsed compact summary copied to external OneDrive
`notebook/evidence/E041/server-report-20261109`;its separate manifest SHA256
7967054f7e7c76936a4b6f785793c7a5b4cc415285b62c04c9231ed17c01a0ee.
Local copy completed;cloud sync and independent server-artifact backup remain
unverified. No implementation changes in this results-recording update.

## 2026-11-09 — E042 planned three-hour E041 continuation

User selected3hours instead of proposed2. Operational continuation of unchanged
E041 posterior-motor objective,not a new fitting algorithm. Three independent
fresh-RNG workers from E041 seed5003 and one from5002,using cohort
`autonomous-posterior-motor-20261109T222806260061Z`. Seeds6000/6001/6002 map
to5003;6003 maps to5002. Four disjoint physical four-core CPU groups on mbslnx2.
Preserve original anchor,bounds16x,all9616 eligible unknowns,full parent copies,
RNG/history/checkpoints/source/env. No re-centering or exact optimizer resume.

Budget180min perworker incl setup/search/validation,plus supervisor termination
grace. Fit20simulated seconds unchanged;long validation extended60→300s for
stronger persistence check. Perturbation and finer brain/body/held-voltage
checks remain20s. Both tail bending>=0.003rad AND net-muscle excursion>=0.003
required in all active tests;25% refinement agreement;held-voltage quiet.
Zero external drive,fixed leak/C/kinetics/reversals/Ca/release/NMJs/body. No
phase,heading,forward motion or RL added. No head/middle retention constraint;
preserve independent parent skill checkpoints.

Command planned: `sh scripts/continue_posterior_motor_mbslnx2.sh` after pull.
Default parents pinned,three repeated paths explicitly implement3:1 allocation.
Preflight validates completed posterior-motor parents and saved checksums/bounds
before nohup;exact source/parent metadata captured by existing worker/supervisor.
Actual hashes/environment/run timestamps cannot be recorded until server launch.
Launcher creates matching `autonomous-posterior-motor-continue-*.log`;cohort
folder still uses E041's `autonomous-posterior-motor-*` prefix. Worker/supervisor
experiment labels remain E041 to preserve algorithm identity;E042 labels this
planned continuation in notebook. No server launch claimed;user launches.

E042 preparation verified locally:launcher `sh -n` passed;supervisor CLI help
recognizes every flag. Executed actual shell with a temporary interpreter/
preflight shim;captured argv verifies3:1 allocation,180min,20s fit,300s long
validation,seeds6000+. No actual scientific preflight,simulation,training or
remote launch occurred. Source training implementation unchanged;no redundant
simulation/full-suite rerun for this launcher-only change. Existing218-test
verification remains documented under E041.

Plan,exact launcher and stub command-construction evidence archived at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E042`;separate manifest SHA256
23a47c094878faf331792c0baf5ae6ef4a2f2bc497de0c646d462cdee774a817.
Archive is preparation evidence,not server experiment output or a checkpoint
backup. Local copy completed;cloud synchronization not independently verified.

## 2026-11-09 — E042 three-hour server continuation: muscle threshold reached

User reported cohort
`/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261109T232143970960Z`.
Seeds6000–6003 all completed exit2/passedFalse,503/502/504/499 proposals
(2008 total). All six validation rows provided. Parsed results and exact
attachment SHA256 in `experiments/records/E042-server-20261109.json`.
No independent retrieval of source/environment records,selected genotypes,
bound saturation,full history or raw trajectories. Runtime/duration settings
remain from the planned launcher;not directly printed in this excerpt.

Best6002 from5003:fit median tail bend1.87389553e-4→0.00136388093rad
(7.2783x),net-muscle median4.79921699e-4→0.00352342427 (7.3417x).
Long validation bend0.00139633566rad,net muscle0.00361156464. Perturb±1mV:
bend0.00135914609/0.00136641040,net muscle0.00351200998/0.00353859365.
Brain-refined bend0.00133876503(-1.8415% vs fit),net muscle0.00347095728
(-1.4891%);body-refined0.00136598572rad/0.00352346152. Held-voltage scored
excursions0. Motor release score rises0.0866077→0.2086125,with refined0.189943;
this is a normalized heuristic,not an absolute transmitter amplitude.

The NET-MUSCLE AMPLITUDE COMPONENT reaches>=0.003 in all five active tests.
Full posterior-motor qualification remains FALSE:tail bend is45.46% of target
at fit,44.63% in brain-refined case. Needs2.20x nominal or2.24x using minimum
active validation to reach0.003rad. Do not lower the bend threshold or label
muscle-component completion a full skill pass. Numerical/perturbation consistency
supports this subthreshold improvement within tested windows,not general
stability,biological fidelity,phase control or locomotion.

Other workers improve both physical metrics:6000 fit tail0.000670701/net muscle
0.00171800;6001 fit tail0.000676252/net muscle0.00171223;6003 fit tail0.000608586/
net muscle0.00102966 (its parent5002 was0.0000529140/0.0000564456).
All remain unqualified;held-voltage scored excursions0.6003 preserves progress
from the alternate lineage;retain all independent checkpoints rather than
consolidating. Similar outcomes6000/6001 do not demonstrate a plateau without
learning curves and bound occupancy.

Decision:prefer a subsequent unchanged-objective continuation from6002,retain
6003 as alternative lineage and6001 as a sibling backup. Keep original anchors,
16x bounds,zero drive,fixed fields and long/refined/held controls. Already-capped
muscle coverage term leaves bend reward to encourage further physical amplitude;
no new phase/control/propulsion requirement. No new training launched or new
budget selected in this results-recording turn. Preserve actual successful
subthreshold genotypes on server;do not promote a qualified asset in Git.

Exact user-pasted stdout and parsed report copied to external OneDrive
`notebook/evidence/E042/server-report-20261109`;its separate manifest SHA256
0e03a972ba411e9d485b8ced94b3c76b35175c89abd166beaeb68aab6f3f8d41.
Copy completed locally;cloud sync and independent raw-server checkpoint backup
not verified. Results-only update;no implementation or acceptance criteria changed.

## 2026-11-09 — E043 planned eight-hour posterior motor continuation

User authorized another8hours. Keep E041 algorithm/objective unchanged. Pin
reviewed E042 cohort`autonomous-posterior-motor-20261109T232143970960Z`:
three fresh-RNG workers7000–7002 from6002,one7003 from6003 to retain the
alternate lineage. Four disjoint four-core physical CPU groups on mbslnx2.
Full302-neuron simulation,all9616 uncertain existing density/recurrent strengths
eligible,original anchor16x bounds retained. Fixed leak/C/kinetics/reversals/Ca/
release/NMJ/body fields and anatomical topology;all external currents zero.
No RL,phase,heading or propulsion objective;no consolidation/retention gate.

Budget480min perworker incl setup/search/validation;existing supervisor grace
can extend slightly and qualification may stop a worker earlier. Fit20s,long
validation300s,other perturbation/refinement/held controls20s unchanged. Both
tail bending>=0.003rad and net-muscle excursion>=0.003 required in active
checks;25% refinement agreement,held-voltage quiet. Parent6002 has reached
muscle threshold but remains below bend threshold;no full pass presumed.

Update the same `scripts/continue_posterior_motor_mbslnx2.sh` to latest reviewed
parents,seed range and480min. Exact E042 three-hour version remains in commit
cb99867 and its external preparation archive;historical replay must use that
source instead of current launcher. No optimizer-state resume,re-centering or
parent overwrite. Preserve full lineage/checkpoints/RNG/history/source/env via
existing warm-start and recording machinery. Preflight validates parent hashes,
bounds and completed posterior-motor flags before nohup. Actual server hashes,
source/environment and start timestamp will be captured at launch,not invented
here. Worker/supervisor algorithm labels remain E041;E043 is this planned cohort.

Command planned after pull:`sh scripts/continue_posterior_motor_mbslnx2.sh`.
Verify shell syntax and actual generated command via stub interpreter before
push. No new scientific simulation needed for this launcher-only change. Server
launch remains user-run;do not claim an eight-hour run has started locally.

E043 launcher verification completed:POSIX shell syntax passes;executed actual
shell with interpreter/preflight shim and captured argv. Confirmed480minutes,
reviewed6002 repeated3x +6003 once,seeds7000+,four4-core workers,fit20s,long
validation300s and posterior-motor objective. No real parent validation,simulation
or training occurred in this check. Implementation unchanged;no redundant
scientific trial or full-suite run for this launcher-only configuration change.

Exact launcher,plan and stub verification archived externally under
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E043`;manifest SHA256
11d9125d23d083ea5abe5dc60c5c8ccead4aaeaa5f71c2723861ce3411d0e7c7.
Preparation evidence only,not server output/checkpoint backup. Local copy
completed;cloud sync not independently verified. Ready for user pull/launch;
no server execution claimed. Earlier E042 source remains preserved atcb99867.

## 2026-11-10 — E043 plan amended to three hours before confirmed launch

Latest user steering:4hours,immediately corrected to3. Four-hour request was
superseded before edits/execution. Update current E043 launcher budget480→180min;
keep reviewed6002x3 +6003x1 allocation,seeds7000–7003,all training settings,
original anchors/bounds and zero drive unchanged. Exact eight-hour version
preserved in commit30ff375 and external E043 preparation archive. Previous
user-pasted output was E042's completed6000–6003 cohort,not evidence E043 ran.
No E043 launch or completion confirmed;configuration edits do not alter any
already-running process. Guard against duplicate starts when launching.

Fit20s,long validation300s,all other controls20s unchanged. Budget includes
setup/search/validation with existing termination grace and early qualification.
Command remains`sh scripts/continue_posterior_motor_mbslnx2.sh` after pull.
Verify shell syntax and actual generated argv with a shim;no new training or
parent validation planned locally for this duration-only amendment.

Three-hour amendment verified:shell syntax and actual argv with stub interpreter
confirm180min,6002x3+6003x1,seeds7000+,20s fits/300s longvalidation. No real
parent preflight,simulation or training launched. External archive
`notebook/evidence/E043/three-hour-amendment-20261110` manifest SHA256
590d5193971785cb502a90956457ca781945e92fcf20afeef87eba1c7f6be8fd.
Local copy completed;cloud sync unverified. Duration-only configuration update;
no redundant training-code test suite rerun.

## 2026-11-10 — E043 first reported qualified autonomous posterior amplitude primitive

User console reports cohort
`/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z`.
All workers completed:7000/7001/7002/7003 ran501/382/501/493 proposals (1877).
7001 exits0,passedTrue. Others exit2/False. This is the first reported pass of
the predeclared E041 posterior motor/muscle/bend assay,1/4 cohort candidates.
An overall cohort exit2 can coexist with this individual success.

Winner7001 from6002:fit tail median0.00136388093→0.00303739123rad;
net-muscle0.00352342427→0.00646743365. Long validation0.00303769857rad /
0.00646781549. Perturb±1mV both qualify. Brain-refined0.00303260423rad /
0.00646974146,relative changes-0.15760% bend,+0.03568% muscle versus fit;
body-refined0.00303931907rad /0.00646751747. All FIVE active validations
qualifiedTrue;held-voltage excursions0 and qualifiedFalse as expected for the
negative control. Parent/source helper's control gate is quietness,not an active
amplitude pass. Reported worker passedTrue confirms combined control/validation.

Worst active bend0.00303260423 is1.0868% above0.003rad threshold;minimum active
net-muscle0.00646745972 is above0.003. Accept under predeclared criteria without
post-hoc goal changes. Limited amplitude margin does not imply broad robustness.
Long300s /other20s settings are from launcher;durations/source/environment hashes
not directly printed in excerpt. No further independent raw verification done.

Winner path(server only):
`.../autonomous-posterior-motor-20261110T140125786528Z/seed-7001/network_best.npz`.
SHA256 not supplied yet,do not invent it or claim an independent checkpoint
backup/asset promotion. Complete source,history,parents,allowed masks,RNG and
validation traces should remain in that immutable worker directory;no overwrites
made by agent. Other candidates7000/7002 are near bend target but do not pass;
7003 improves from alternate lineage yet still fails both physical thresholds.
Preserve independent candidates as H1.1 requires.

Scientific scope:small autonomous posterior amplitude through the channel brain,
anatomical readout and liquid body under existing zero-drive protocol. This is
not a coordinated traveling wave,forward/backward motion,swimming,natural-animal
fidelity,whole-body amplitude qualification or a reusable command-conditioned
skill selector. Current head/middle amplitudes are absent in this console;
retention was never a gate. Full302-cell network remains simulated;no new
runtime drive or change to physical parameters was introduced in this run.

Decision:register qualified posterior amplitude evidence in SKILL_REPERTOIRE;
preserve7001 rather than replacing it in future experiments. Next inspect full-
body regional amplitudes and trajectories,then choose amplitude completion or
coordination based on evidence. No new simulation,RL or training launched here.
Exact values transcribed from direct user message (no attachment bytes for hash)
in `/tmp/r-elegans-E043-server-transcription.txt`;compact parsed record
`experiments/records/E043-server-20261110.json` labels transcription and missing
checkpoint/source hashes explicitly. This is a user-reported server result,
not an independently downloaded/recomputed assay.

Manual user-console transcription and parsed summary archived at external
OneDrive`notebook/evidence/E043/server-report-20261110`;separate manifest SHA256
40ec9029c8fb4f60269e132290e9cbdff95088b6c1b8bce5dcc12892f1c0e73a.
This preserves reported evidence only,not the actual winner genotype/trajectories
or independent server verification. Local archive copy completed;cloud sync
unverified. Winner remains server-only;no checkpoint hash fabricated. Docs
register MOT-16 autonomous posterior amplitude evidence,explicitly scoped to
these assays. Results-only update,no training/implementation changes.

## 2026-11-10 — E044 planned read-only full-body assessment of E043 winner

User authorized proceeding after tail qualification. Parent pinned to
`autonomous-posterior-motor-20261110T140125786528Z/seed-7001` on mbslnx2.
No training,simulation,external drive,parameter changes or checkpoint overwrite.
Question:are head,middle and tail amplitudes all sufficient in this same
checkpoint,and what shape/translation does its saved motion actually show?

Read all six saved validation trajectories,verify completed qualified worker
and parent hashes via read_warm_start. Recompute sustained per-joint and filtered
muscle-drive amplitudes using unchanged E037 detector,regional medians [tail,
middle,head],compare with recorded values. Report recorded body-wide amplitude
qualification separately from tail pass. Reuse existing shape PCA,spectral lag,
same-reference-joint-phase translation analysis;no automatic locomotion claim.
Verify quiet control remains distinct from active qualification. Explicit SI
body length0.001m matches current twelve-rod liquid assay.

Create external JSON hashes/report,scientific PNG figures and self-contained
HTML motion preview of final5s. Render exact12-segment geometry with equal axis
scale,tail/head labels;true-scale world view plus centered shape view. Optional
oscillatory-deviation gain explicitly labeled visual-only,default1x;do not
silently exaggerate movement. Preview subsampling labeled;full metrics use all
samples. Compact parameter-reference copy outside Git verifies selected model
and matching frozen parent;it is not an independent raw-lineage backup.

Source remains untouched;output manifest and source/environment captured.
Criterion is measurement/reporting integrity,not inventing new behavioral
acceptance. Rendered motion/position alone does not prove propulsion,closed
shape cycles or animal fidelity. Do not assume head/middle retention.

Local verification on cached UNQUALIFIED E041 pilot via explicit override:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m scripts.assess_full_body --worker /tmp/r-elegans-E041-motor-pilot --allow-unqualified --output-dir /tmp/r-elegans-E044-body-assessment`
Test region handling,static-vs-repeated excursions and geometry against existing
mechanics. Actual7001 files unavailable locally;server assessment stays
user-launched after push. Preserve statuses as pending until real output arrives.

E044 scope clarification before final assessment execution:tail training only
requires tail refinement agreement. To assess full-body readiness,also reapply
existing E037 regional25% refinement tolerance to head/middle/tail against the
recorded best-training amplitudes,plus the quiet held-voltage control. Report
this separately from each case's stored body_amplitude_qualified flag;all cases
above target alone do not certify the missing regional refinement comparison.
No training acceptance criterion changed. Pure analysis import initially pulled
JAX via brain package initialization;move the detector unchanged to a top-level
NumPy/SciPy module and re-export from original location. Function text SHA256
49f361f6eeb373b4863c71b1f6cdb5939d719c90d4099acb162963229966116e
is byte-identical. This avoids GPU-plugin initialization in read-only assessment.

E044 LOCAL assessment executed on explicit unqualified E041 pilot override,
all six cases completed;reference copies and source-checksum stability verified.
Default qualified-parent guard rejects the pilot before creating output. Local
minimum active region medians [2.90573e-6,0.000153831,0.0192844]rad;full-body
assay recheckfalse,expected. These measurements are NOT7001 results. Actual
winner head/middle retention and motion remain server-assessment pending.

Final local command/output:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -u -m scripts.assess_full_body --worker /tmp/r-elegans-E041-motor-pilot --allow-unqualified --output-dir /tmp/r-elegans-E044-body-assessment-reviewed`
stdout`/tmp/r-elegans-E044-body-assessment-reviewed.log`,exit0. Original pilot
raw inputs already archived under external E041/motor-pilot;new outputs hash
references instead of duplicating all raw traces. Parameter-reference copies
include winner/readout/original anchor and metadata,not an independent full
raw-lineage backup. No scientific model/qualification math changed.

Verification:final suite223 passed71.15s. Tests cover region ordering/repeated
amplitudes,static/ramp rejection,rod lengths/centering/world transformation against
existing mechanics,true-scale defaults/frame cap,HTML labeling,self-contained
preview,import without JAX,and full-body regional refinement/quiet-control gates.
Pure-analysis import no longer initializes JAX. Detector function moved intact,
old import path re-exported. Scientific PNG visually inspected for the LOCAL
pilot. Actual preview JavaScript executed via macOS JavaScriptCore with mocked
DOM/canvas:finite coordinates,playback/scrub and20x visual gain labeling verified;
278 frames in local5s preview. This is not a browser screenshot or actual7001
preview. Output HTML hash matches the version exercised by the control check.
Launcher POSIX shell syntax passes;server parent preflight remains pending.

Raw local reports/PNGs/HTML,parameter references,source/environment,logs and
preview verification archived externally at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E044` (264 files),manifest
SHA2569f561d5b9dde708787063740ade5dea152bc57ad4d09d8260252516b915adeb9.
Local copy completed;cloud sync unverified. Compact record
`experiments/records/E044-full-body-assessment.json`. No actual server artifact
backup or winner assessment claimed. Decision:push read-only nohup launcher
pinned to7001;user runs and returns compact regional/shape reports,then select
next training stage from the measured body-wide result. Preserve qualified tail
primitive regardless of later coverage work.

## 2026-11-10 — E044 server assessment launch reported; winner hash now supplied

User launch stdout identifies correct qualified E043 seed7001 and preflight
SHA256417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14.
This is remote preflight verification reported by user,not independent local
retrieval/recomputation. E043 compact record updated with this provenance;
historical missing-hash statements describe earlier evidence availability.

Assessment output `/home/carsonc/r-elegans-results/full-body-assessment-20261110T174954Z-687519`,
log samepath`.log`,PID687526,CPUs0,1,2,3. Expected motion.html and
long/trajectory.png paths printed. No actual report contents or completion
status provided;do not infer regional amplitudes,body-wide qualification,
visual motion or raw-data backup. Reference record
`experiments/records/E044-server-launch-20261110.json`. No new training or model
change. Wait for same exact log rather than selecting an older analysis run.

Reported launch metadata archived externally at
`notebook/evidence/E044/server-launch-20261110`;manifest SHA256
2d9b32657becf5d6ebdbd8cd482735c18d8e0d5016e87c2adbae0289fcb38c18.
Local copy completed;cloud sync unverified. This archives metadata only,not
actual assessment outputs or winner checkpoint bytes.

## 2026-11-10 — E044 server full-body amplitude recheck passes

User reports exact E044 assessment log completed with
FULL_BODY_ASSESSMENT_EXIT_CODE=0 and full_body_amplitude_assay_passTrue.
Checkpoint SHA256 matches registered7001:
417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14.
Input/output references unchanged from launch. Parsed direct-console JSON
transcription in`experiments/records/E044-server-20261110.json`;not independently
downloaded/recomputed from original raw artifacts.

All five active cases pass stored full-body amplitude checks and all three
regional targets. Reapplied E037 regional refinement/quiet-control aggregate
reports pass. Minimum active sustained median bends[tail,middle,head]rad:
[0.00303260423,0.03331642691,0.01103544980]. Long validation:
[0.00303769857,0.03338550963,0.01105994731]. Long filtered-drive medians:
[0.00367019512,0.04026140645,0.01334451139]. Held-voltage all excursions0;
active qualificationFalse in that control is expected. Regional medians do not
require every single joint/muscle to exceed the threshold.

Register MOT-17 autonomous distributed bend-amplitude prerequisite on the SAME
7001 checkpoint,alongside MOT-16. Head/middle activity survived tail training
within the reported assay windows and existing regional refinement limits.
This is amplitude evidence,not command-conditioned control,animal fidelity or
propulsion. No new fitting/simulation or parameter change.

Shape PC1 explains99.898% of long-window bend variance;other active checks
~99.897–99.900%. This indicates near-one-dimensional shape variation and
suggests limited independent phase structure. It does not alone prove a standing
wave or exclude small traveling components. Strongest joint7 in all active
cases. Equal-reference-joint-phase speed is about-1e-5body-lengths/s,negative
and tiny;body-refined-8.90546e-6 vs long-9.98797e-6. Full-shape closure and
phase profiles are not pasted,so no coordinated wave or useful forward/backward
locomotion qualification follows. Avoid interpreting any centroid drift as a
trained behavior.

Preview/PNG/full report exist on server;actual images and parameter-reference
bytes not independently fetched/viewed. Existing local E044 images are only
unqualified pipeline tests,not7001. Next:preserve/download assessment bundle
(includes compact parameter reference with hashes) to external Mac OneDrive,
inspect actual preview and phase/shape evidence,then propose gradual spatial
phase coordination while retaining amplitude and zero drive. No phase fitting
or new training launched in this results-recording turn. Complete server raw
lineage remains separately retained;compact reference is not that full backup.

Reported console JSON transcription and parsed compact summary archived externally
at`notebook/evidence/E044/server-report-20261110`;manifest SHA256
 ae69f9400011913c1bf3563f73cfc81c698b4f7fe2942c86eb7b558256207612.
This is reported evidence only,not actual preview/report/checkpoint-byte download.
Local copy completed;cloud sync unverified. To complete visual inspection and
retain an independent compact parameter reference,user can SCP the assessment
folder from a Mac Terminal to external OneDrive E044 directory. Agent cannot
supply the user's interactive SSH password;standing workflow is user-run transfer,
never password sharing. No additional login setup or training requested here.

## 2026-11-10 — E044 copied server bundle: independent local artifact inspection

User reports SCP completed. Local copied bundle found at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E044/full-body-assessment-20261110T174954Z-687519`.
Read-only verification checks all256 manifest entries,then independently hashes
parameter_reference/network.npz against known417254ad...6a417f14. All match.
This establishes matching copied selected parameters/reference/report artifacts,
not an independent recomputation from raw300s voltage/body arrays (not included).

Inspect actual long/trajectory.png and full report,not cached pilot. Fixed fields
match original anchor exactly after same zero-bias reference operation;frozen
parameter entries unchanged;all external bias currents0;302x17 channel-scale
shape and9616 eligible unknowns. Frozen parent readout hash82f418ac...8247dcb0
matches previously preserved parent. Copied worker metadata reports9488 parameters
differ from original anchor,3779 from incoming parent;counts are differences,
not evidence of physiological identification or necessity of those changes.

Actual full-body recheck passes. Long300s analysis dominant frequency2.56Hz,
383 late-window reference cycles;PC1 fraction0.9989816896,PC2~0.00099765.
Middle/head active-joint lags near0 (roughly-4.19 to+2.49ms versusjoint7);
posterior joint3 lag-12.10ms,posterior joint2 lag-180.69ms (~opposed),not a smooth
phase ramp. Low-amplitude joints0/1/10 excluded by the existing spectral display
threshold;do not extrapolate phase through them. Coherence near1 at reference
frequency means shared rhythm,not useful propagating coordination.
Same-phase regression speed-9.98797e-6BL/s,positive-cycle fraction0;median
shape closure RMS4.41812e-8rad. Thus same-reference samples nearly close shape
here,but observed drift remains tiny and numerical convergence of propulsion
has not been qualified. No natural swimming/backward-run claim.

Planned supplemental visualization only:extract true1x/explicit20x centered
rod frames already in copied HTML,choose four samples separated by roughly a
quarter of stored reference-cycle period,render phase snapshots outside Git.
No new integration,parameters or amplitude/locomotion gate. Exact frame times
and input hashes recorded;frames are subsampled and rounded to8decimals in
body-length coordinates. Do not use this preview to requalify the300s assay.
Preserve script/results with independent artifact-review record.

Copied actual parameter-reference/model passes local load/shape/finite validation:
302neurons,17 channel terms,all bias currents0;fixed fields and frozen parameter
entries equal original anchor. Independent selected-file SHA256 exactly matches
server-reported417254ad...6a417f14. Parent readout independently hashes to
82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.
Worker metadata differences from original anchor:4745 channel terms,3652
chemical edges,1091 electrical pairs (9488 total);3779 from incoming parent.
These counts are copied metadata,not proof all changes are necessary or measured.

Supplemental true1x/explicit20x shape snapshots produced without new integration
at295.092,295.182,295.290,295.380s using preview frames. Stored reference period
0.390515s. Actual plot and snapshots visually inspected:oscillations around a
persistent curved mean posture,with largely common middle/head bending and
posterior phase reversal.20x panel explicitly illustrative;no amplified motion
used in measurements or acceptance. Source preview278 frames covering
295.002–299.988s,full numerical report uses300s source. No browser animation
screenshot claimed;inspected saved plot and numerical preview frame geometry.

Independent review script,figure,source/environment and full/compact review
archived at external OneDrive
`notebook/evidence/E044/local-artifact-review-20261110` (232 files),manifest
SHA25668c38eaa9c36c12a7da32e9a5c53a3d8de75c1bf7ec0ac1bfff6f519ecf6195f.
Copied actual server bundle is preserved separately at its original named folder
under E044. Compact reference is now on Mac,not Git;raw full neural/body arrays
and complete historical lineage remain on server. Cloud sync not independently
verified. Source implementation unchanged;no new training or redundant software
test suite. Verification record
`experiments/records/E044-local-artifact-review-20261110.json`.

Decision:inspection supports the next proposed baby step of a small smooth
spatial phase progression,with regional amplitude floors and zero external
input retained,keeping7001 immutable as baseline. Do not jump directly to RL
or claim locomotion from current drift. No phase-training implementation or
launch was authorized/executed by the user's copy confirmation alone.

## 2026-11-10 — Literature check: is2.56Hz biologically plausible?

User asks whether winner7001's2.56Hz motion is in the C.elegans ballpark.
Checked primary sources online,not new simulation or parameter fitting:

- Pierce-Shimomura etal.(2008),PNAS,Genetic analysis of crawling and swimming
  locomotory patterns,doi10.1073/pnas.0810359105,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2634943/ :neck-curvature spectral
  peaks~0.8Hz crawling and~2.1Hz swimming in their comparison. Spatial propagation,
  amplitude and muscle activity also distinguish their measured gaits.
- Magnes,Susman,Eells(2012),JVisExp,Quantitative Locomotion Study of Freely
  Swimming Micro-organisms Using Laser Diffraction,doi10.3791/4412,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3499068/ :reported representative
  free-swim frequencies~2.5Hz(video)and~2.7Hz(diffraction acquisition);cites
  mean2.37Hz in5mm cuvette. Geometry/confinement matters;these are experimental
  examples,not a universal allowed frequency interval.

Inference:2.56Hz is plausible for a liquid/swimming-timescale oscillation.
It is faster than the cited crawling comparison. This contextual plausibility
is not a physiological calibration,validation of waveform/propulsion or proof
of correct neuron-specific channel parameters. Assay uses liquid body model;
existing near-synchronous/phase-reversal pattern remains the immediate mismatch.
Recommendation:retain emergent rhythm for next proposed phase-coordination
trial rather than changing frequency solely to hit a single literature number.
No frequency target,parameter change,new training or asset promotion introduced.

## 2026-11-10 — E045 planned read-only medium-physics status audit

User asks whether medium physics is correct. Scope:inspect current calibration,
force balance,tests and primary-source assumptions;calculate drag numbers and
Reynolds-scale estimates from copied7001 preview. No new integration,training,
external drive or parameter changes. Do not equate software tests and literature
constants with experimental validation of the complete body/medium model.

Implementation reviewed:physical_muscle_body_params(12,medium='liquid') in trainer;
L1mm,radius40um,viscosity0.001Pa s,density1000kg/m3,wavelength1.5mm parameter
in Lighthill drag closure. Coupled Schur-complement torque-driven overdamped
body,force/torque-free rigid motion,finite rod rotational drag,implicit passive
elasticity. Current tests check SI conversion,SPD/precision,residual force/torque,
load response and Re calculations. Existing223-test suite previously passed;
new audit question justifies targeted physics-test confirmation if run.

Read-only numerical plan:report local/whole-body drag ratio,material relaxation
upper-bound timescale,and approximate node speeds/Re from saved true-scale HTML
frames;velocity finite differences are subsampled estimates,not full raw maxima.
Use both radius and length scales,avoid judging inertial relevance from tiny
centroid drift alone. Primary sources checked online:Boyle etal2012 RFT formula,
Fang-Yen etal2010 passive bending and chamber wall correction,Sznitman etal2010
swimming,with empirical limitations distinguished. Finite planar local RFT omits
nonlocal flow,walls/free surface,body taper,self-contact and fluid rheology beyond
Newtonian viscosity. No benchmark/gait fit or wall correction is being silently
added. Acceptance:document actual approximation/validation scope,not certify
physical completeness. Planned command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python /tmp/r-elegans-medium-audit.py`.

E045 read-only audit executed successfully;no neural/body integration or model
changes. Lighthill per-length drag0.003290414/0.005215252kg/(m s),ratio1.584983;
whole-body3.290414e-6/5.215252e-6kg/s. These match cited convention and correct
length conversion. Passive relaxation upper-bound ratioav/b0.005263s,muscle
time constant0.1s;active preferred-curvature moment scale9.5e-10Nm is a closure.
Preview278 true-world frames yield sampled maximum node speed1.94225e-4m/s,
Re(radius)0.007769,Re(length)0.194225;radius-based unsteady parameter0.025736.
These approximated node speeds are not exact peak fluid velocities and cannot
certify hydrodynamic error. Tiny centroid drift must not be substituted for
lateral body speed when assessing viscous dominance.

Targeted existing tests:`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q tests/test_physical_calibration.py`,8 passed2.06s. Covers formula anisotropy,
SI parameters,SPD/precision,force/torque residual,load response and Re. Does
not measure accuracy against real swimmer thrust,speed,confinement or rheology.
No coefficient/unit mismatch identified by this limited audit;no exhaustive
physical correctness claim. No complete RFT-versus-Stokes or spatial-resolution
benchmark performed. Qualified7001 retained unchanged in its original model.

Sources:Boyle2012doi10.3389/fncom.2012.00010 verifies the formula;Fang-Yen2010
doi10.1073/pnas.1003016107 documents150um chamber transverse drag~9.7x bulk.
Author-hosted PDF provides primary methods;some PMC pages triggered CAPTCHA.
Paper OCR occasionally corrupts superscripts/units;SI beam conversion here
checked dimensionally and against existing primary coefficient convention.
No new PDF/raw dataset stored in Git. Geometry-specific wall factors must not
be applied by globally rescaling water viscosity without a matching geometry.

Plan/source/environment/numeric output/script archived externally at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E045` (232 files),manifest
SHA2562e1b67e3b2b56beaad1fe9186b113be92e029d21b53909a6014eddda6db04c69.
Local copy completed;cloud sync unverified. Compact record
`experiments/records/E045-medium-physics-audit.json`. Decision:classify current
medium as unbounded Newtonian RFT approximation;propose prescribed-wave and
matched-geometry propulsion/force/convergence benchmarks before further
quantitative locomotion claims. No additional training or physics changes.

## 2026-11-10 — E046 planned wet-surface reference and fixed-winner assessment

User approves making a moist compliant substrate the primary natural-behavior
 target. Question: how does unchanged E043/E044 seed7001 respond to a measured
wet-agar force law, and is that reference implementation numerically reliable?
Preserve liquid winner SHA256
417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14.
No channel, synapse, NMJ, muscle or passive-body fitting; no external drive,
proprioception or sensory currents added. This is an assessment, not training.

Method planned: independent float64 overdamped reference solver with local
quadrature forces, force/torque balance and implicit passive elasticity.
Use Rabets et al. 2014 (doi:10.1016/j.bpj.2014.09.006) published power-law
fits for standard 1.7% NGM agar. Whole-1mm-worm force in microNewtons equals
a*(speed in micrometers/second)^gamma: tangential a=.32,gamma=.58; normal
a=6.8,gamma=.31. These are published fitted curves, not new raw measurements
or a complete lubrication/contact solver. Other measured concentrations remain
available for sensitivity, not interpolated into a fictitious natural substrate.

Regularize the unmeasured near-zero-speed divergence smoothly with a stated
speed scale; test .01/.1/1 micrometer/s. Reject unconverged force balances;
do not hide Newton failures with zero motion. Validate straight towing force,
positive dissipation, linear-law reduction against the existing solver and
passive relaxation. Compare unchanged liquid/legacy-agar/new wet-surface body
responses to exactly the same generated autonomous motor output. Since this
branch has no body-to-brain feedback, recording then replaying endogenous
muscle commands preserves the one-way model; explicitly label the replay.
Start all bodies straight/relaxed, keep muscle kinetics and projection fixed.
Use half body timestep with piecewise-held identical muscle commands; record
that this checks body integration, not refined neural integration or spatial
convergence. Retain traces, states, source, environment, exact commands and
input hashes outside Git. No optimizer/RNG exists for this deterministic assay.

Planned default local assay:20 simulated seconds with existing dt=.002s;
server launch may extend to60s under nohup using the absolute dedicated Python.
Acceptance for implementation:relative nonlinear residual <=1e-8; linear-law
 equivalence within floating-point/regularization tolerances; finite traces,
document timestep and low-speed sensitivity without calling either a biological
pass. Existing regional .003rad amplitudes remain descriptive; no thresholds
relaxed. Independent held-voltage control included. Primary training-environment
target becomes wet-surface crawling; promotion to training awaits assessment.
Status planned until execution is recorded below.

E046 first local20s assessment executed (not a server report):
`MPLCONFIGDIR=/tmp/r-elegans-mpl JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u -m scripts.assess_wet_surface --reference /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E044/full-body-assessment-20261110T174954Z-687519/parameter_reference --output-dir /tmp/r-elegans-E046-local --duration 20`.
Exit0; all8 cases completed. Liquid-reference sustained regional bends reproduce
~.003037/.033373/.011056rad. Wet1.7% assay yields sustained0/.000457/0rad;
held-voltage sustained amplitudes0. These zeros mean the repeated/sustained
criterion was not met, not that every sample was constant. Low-speed smoothing
matters:middle sustained amplitude .000415/.000457/.000848rad at smoothing
.01/.1/1um/s. No .003rad regional threshold was changed and no training occurred.
Four new tests passed1.63s: published towing units/whole length, positive/odd
resistance and analytic derivatives, linear mechanics reduction against JAX,
passive energy decrease/zero-command control. These are software/closure tests.

Decision:20s may retain slow substrate/posture transients. Extend unchanged
assessment locally to60s before interpreting loss of sustained regional motion.
Same command with --duration60 and --output-dir /tmp/r-elegans-E046-local-60s;
final runner also records raw amplitudes (not only sustained detector outputs),
force curves, per-case progress and direct low-speed sensitivity. This is a
longer diagnostic, not optimization or a new checkpoint. Exact initial/final
neural states, all command samples, body states/trajectories and captured source
remain external. Completion/results pending at this entry.

E046 local60s comparison completed,exit0; replay/plot stage121.898s after the
neural-generation stage. All8 cases saved with exact source/environment,
commands, neural states and per-case body trajectories. No seed/optimizer:
this is deterministic evaluation of pinned7001. Standalone Linux launcher uses
absolute dedicated Python,disjoint four-physical-core allocation and nohup;
it is prepared but has not been executed on mbslnx2 by this agent.

Liquid float64 reference versus original JAX maximum joint error1.00356e-7rad;
regional sustained .00303760/.03338529/.01105988rad. Legacy constant agar
.00414992/.00375516/.00247043rad does not meet all-region. Measured nonlinear
wet-agar and refinements/smoothing cases have sustained regional detector
outputs0/0/0 at60s. Raw default late-half excursions .01473234/.01235087/
.01234939rad are mostly slow postural relaxation,not repeated cyclic amplitude.
Held-voltage also relaxes,raw .01094044/.00843806/.00893424rad,with sustained0.
Inspected actual wet-agar trajectory.png:small oscillations ride on a slow
posture trend; PC1drift dominates. No zero-movement or gait-frequency claim
from detector0 or low-frequency FFT bins. Equal-phase propulsion undefined
without sufficient crossings; raw drift cannot establish locomotion.

Nonlinear residual max9.99946e-9. Half-step raw regional amplitude relative
changes [1.14e-6,2.45e-5,1.57e-7]; 4-to8-point quadrature
[1.37e-4,9.00e-5,3.02e-4]. Low-speed sensitivity much larger:0.01um/s versus
0.1 gives raw changes22.48/20.74/15.43%;1um/s gives48.63/47.84/15.94%.
Default62.19% late local velocity components below0.1um/s. These are signs of
unmeasured near-zero extrapolation and incomplete settling,not robust physical
identification. Maximum point speed default3.72611e-5m/s includes startup;
whole-worm constant-speed towing fits are not new independent measurements.
Reference solver passes its implementation checks; no biological crawling pass.

`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q`:
227 passed69.47s. Shell syntax and git diff whitespace checks passed. Current
winner,original priors/bounds,and previous liquid environments unchanged.
Decision:preserve liquid skill;new wet-surface model is experimental reference,
not promoted to training. Before a long fit,address low-speed uncertainty and
settling,verify a fast implementation against this reference,then aim for
surface amplitude first. No longer training run started or promised.

E046 evidence archived to
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E046`:
1096 files verified against archive_manifest.json, SHA256
873ec5d0b2870f8c35fc7236c7e7ce1b4f57d7f59332ab63d9f3e0b25e4fe9b3.
Includes both local runs,stdout,all source/environment snapshots,pinned inputs,
all body/neural traces,scientific figures/HTML,tests,and primary Rabets2014 PDF
SHA25694811389f685052b1e697676473a25f21aa727dff04b59e8e43771a303beb423.
Local archive copy and all1096 checksums verified;OneDrive cloud synchronization
not independently verified. Original server raw historical lineage remains
separate;this diagnostic archive is not an optimizer resume or new lineage
backup. Compact committed record:experiments/records/E046-wet-surface-assessment.json.

## 2026-11-10 — User clarifies both environments and action-state control

User requires swimming in water AND movement on moist substrate, then asks what
triggers swimming and whether the animal rests and starts/stops actions. This
supersedes the sole-primary-environment wording from E046, without altering its
experimental results. Both media remain required. Separate intermediate skills
remain permitted under H1.1; eventual one-animal adaptation needs continuous
state with one parameter set, no manual checkpoint swaps.

Read-only code confirmation:conductance_network.initial_network_state uses
fixed.initial_voltage_mV and initialized channels/calcium; channel_body_chunk
advances the same unpaced network regardless of the chosen body load. There is
no immersion stimulus, body feedback or explicit serotonin/dopamine/FLP-11
state in that branch. Persistent rhythm is the outcome of the selected dynamics
from initialization, not a demonstrated decision to start swimming. Historical
driven MOT-05/06 stop/restart is not transferred evidence for this branch.

Primary literature checked online:Vidal-Gadea et al.2011,PMID21969584,
doi10.1073/pnas.1108673108,serotonin-dependent swim initiation and dopamine-linked
return to crawling; Ghosh and Emmons2008,PMID19011210,doi10.1242/jeb.023606,
prolonged swimming alternates with quiescence even after command-interneuron
ablation; Steuer Costa et al.2019,PMID31506439,doi10.1038/s41467-019-12098-5,
RIS activation stops locomotion with muscle tone retained, GABA/FLP-11 roles.
Do not generalize these findings into a single water-on sensor, a universal
pause mechanism or identification of all immobility as sleep.

Requirements recorded:separate movement generation from onset, maintenance,
stopping, rest and restart. Future cue-to-bout/pause/restart assays must preserve
continuous brain/body state and use genuine neural control, with no output
zeroing, frozen-voltage trick or reset. Maintain autonomous motor dynamics;
sensory cues and internal state may regulate action selection without pacing
each bend. Require documented cues, fixed/trainable mechanisms, transition
metrics and numerical/biological criteria before an actual experiment. No new
training run or scientific simulation executed in this documentation update.

## 2026-11-10 — Planned E047 neuron-role and sensory-pathway audit

User requests identification of control versus motor neurons and asks whether
sensory neurons, olfaction and proprioception are present. Plan:annotate a
literature-supported locomotion/sensory subset against all302 canonical IDs,
allow overlapping motor/proprioceptive roles and explicitly mark unreviewed
neurons. Do not pretend to have a complete binary classification. Count existing
NMJ contacts separately from functional roles; an NMJ edge does not prove that
a cell controls the current behavior. Verify source-backed group names,indices
and memberships using the bundled runtime topology. No integration,weights,
channel parameters,input currents or training masks changed; no checkpoint
needed beyond existing bundle provenance. No stochastic trial/seed/optimizer.

Code audit already identifies two distinct branches:the autonomous conductance
network has sensory cells anatomically but no environmental/proprioceptive
inputs; older full-network normalized actors inject three engineered chemical
features into AWCL/AWCR/ASEL/ASER. The14-cell subcircuit additionally uses
sin/cos of a gait phase as an engineered proxy,not physical proprioception.
Chemosensory cell presence or transferred AWCON channel priors do not establish
an identified sensory-receptor model in the current autonomous worm.

Primary sources checked online:Bargmann et al.1993 odor-selective AWA/AWC,
Bargmann/Horvitz1991 ASE,Chalfie et al.1985 touch/premotor circuit,
Xu et al.2018 AVB-B descending coupling,Wen et al.2012 B-type proprioception,
Li et al.2006 DVA stretch/TRP-4,Yeon et al.2018 SMDD sensory-motor steering,
Steuer Costa et al.2019 RIS stopping. Their reported functions are biological
annotations,not validated roles of the fitted simulator. Prefer cited anatomical
body motor classes over assigning every muscle-projecting cell a single role.
Acceptance:all annotated members exist uniquely in canonical order;B-type and
SMDD overlap retained;unreviewed status explicit;input-pathway audit distinguishes
active code from intended future sensory feedback. Record exact audit command,
source/environment and outputs externally,small counts/summary in Git.

E047 annotation audit executed locally:
`.venv/bin/python -m scripts.audit_neuron_roles --output-dir /tmp/r-elegans-E047-neuron-role-audit`.
All302 canonical IDs retained;113 annotated,189 unreviewed. Curated counts:
14 sensory (8 chemical,6 touch),20 interneurons (10 premotor,1 RIS state control
among them),79 selected motor neurons,21 overlapping proprioceptive annotations.
These are subset annotation counts,not total sensory/motor counts for the animal.
All groups map to existing unique IDs. Two targeted invariant tests pass0.02s;
no integration or fitted-network causal perturbation was performed. The full227
suite passed during E046;not rerun for this metadata-only addition.

Audit directly confirms AWC/ASE nodes have no anatomical NMJs in this bundle,
while AVBR has2 and DVA1;do not define exclusive neuron types from any NMJ edge.
Controls and motor neurons remain in the same recurrent model,with all other
neurons retained. Annotations alone do not change dynamics,trainable subsets,
input currents or the preserved checkpoint. Core source paths and old/new branch
input differences documented in CURRENT_MODEL.md and the detailed audit report.

Decision:use explicit overlapping identities and test actual fitted-network
causality later. World-to-sensory transduction and body-to-neuron feedback remain
missing in the autonomous branch. No claim of new olfaction,proprioception,
AVB recruitment or RIS stopping in the simulated worm. Future assays should
separate local sensory encoding,descending state regulation,motor rhythms and
mechanics. Exact source-backed role records and scientific evidence remain
external;compact summary is committed under E047.

E047 external archive:
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E047`.
240 archived files checksum-verified;archive manifest SHA256
06d96e55c294dc0e5948206014171eae8c19ad94bfb84a76b5f2193b8dcd5425,
report SHA2561b90190cd732f5f6b5eff943eb95e2ba6f13ded5ee8df940b78b8166ec180143.
Local copy verified;cloud sync unverified. Compact record:
experiments/records/E047-neuron-role-audit.json. This is an informational audit,
not a new fit or a behavioral qualification. No server run launched.

## 2026-11-10 — Planned E048 embodied sensory connections

User authorizes connecting the missing environmental and proprioceptive inputs.
Implement a separate explicit closed-loop entry point around the existing302
channel brain, anatomical NMJs,95 muscles and SI body; preserve the old no-input
entry point and immutable7001 checkpoint. Existing recurrent control-to-motor
edges are retained, not replaced by a synthetic controller or new graph edges.
No training, optimizer or random proposals; deterministic wiring/causality tests.

Plan:sample independent odor and salt test fields at the true head endpoint in
SI coordinates. Use adaptive bounded transduction into AWA/AWC/ASE (distinct
cue channels and ASE left/right response signs). Feed actual body curvature into
DB/VB via anterior receptive fields,plus DVA bending-magnitude and optional
SMDD head-bend proxies. Bending feedback must depend on physical shape,not gait
phase or a desired wave. Receptive-field locations derived from anatomical NMJ
output distributions are explicit approximations,not measured stretch-sensor
positions. Unknown gains/kinetics/normalization are declared experimental.

Apply inputs as nonnegative membrane conductances with stated reversals in the
voltage solve,not forced voltage, periodic current or direct muscle control.
Detailed receptor cascades,water sensing,contact sensing,serotonin/dopamine and
RIS peptide regulation remain separate work. Biological response directions
are supported by primary experiments; numerical transduction constants are not
physiological fits. Preserve all sensory memory across streamed chunks and
capture per-neuron conductances/currents and sampled cues.

Proposed diagnostics:zero gains versus historical no-input execution;known odor
on/off and salt up/down;positive/negative imposed shape (mechanistic probe,not
normal behavior);controlled uniform field/translated source;paired closed-loop
versus sensory/proprioceptive blocks. Validate that feedback affects neurons,
can propagate through existing connectivity and reaches muscles/body,without
claiming food-finding or start/stop skill. Compare streaming with uninterrupted
execution;finite states and input locality;check body timestep and save exact
initial/final state and source/environment. Targeted/full tests appropriate to
new integration. Use liquid JAX body and genuinely interleaved wet-agar reference
body (not replay after neural recording,which is invalid once feedback exists).
Pinned7001 exact hash/readout,all trainable and passive parameters fixed. New
external diagnostic archives and launchable mbslnx2 commands will be recorded
when executed. Any default transduction parameters are exploratory,not promoted
into the old calibrated/no-input asset. Acceptance is correct physical-signal
routing and numerical consistency,not biological behavior qualification.

E048 initial local diagnostic executed:5 simulated seconds in10 cases (both
media,four sensory-block conditions plus half-body-timestep controls),exit0.
Command:
`MPLCONFIGDIR=/tmp/r-elegans-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu .venv/bin/python -u -m scripts.diagnose_embodied_senses --reference /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E044/full-body-assessment-20261110T174954Z-687519/parameter_reference --output-dir /tmp/r-elegans-E048-local --duration 5 --refine`.
Recorded native checkpoint hash/readout,full302 recurrent dynamics and actual
interleaved wet-surface reference;all cases finite. This is an initial software
causality diagnostic,not a trained behavior or biological validation.

Initial6 sensory tests passed7.74s; expanded integration tests (including
legacy channel-body,conductance network and wet reference)22 passed17.25s.
Tests include chemical response signs/locality,pose-dependent inputs,membrane
conductance analytic solution,zero-gain legacy equivalence,nonzero-memory
chunk equivalence and downstream neural/muscle response. Single-step wet body
agrees with the existing reference replay in a numerical fixture. JSON config
round-trip normalization was then added for reproducible parameter restoration;
future diagnostics also explicitly summarize coarse/fine timestep differences.

Next planned local verification:20s diagnostic with the finalized code to check
finite dynamics after the stimulus pulses end,using the same unknown gains and
immutable checkpoint (no tuning). Same command with --duration20 and output
/tmp/r-elegans-E048-local-20s. Run full test suite because membrane integration
now has a new optional conductance input. Preserve each run's source snapshot;
older and final diagnostic code versions are distinguishable. No behavior
acceptance threshold will be inferred from these software results.

E048 finalized local20s diagnostic completed in194.966s,exit0;10 cases finite.
Exact command is the planned invocation above with --duration20 and
/tmp/r-elegans-E048-local-20s. Network SHA256 remains
417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14;
frozen readout82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.
No original channels,connections,NMJs,mechanics or intrinsic bias changed.

Maximum paired differences (over complete records,not behavior scores):
water closed versus chemical block:3.30556mV neural,1.03891e-4 muscle activation,
2.23368e-5rad joints;versus proprioceptive block:7.23340mV,0.0319275 muscle,
0.0166179rad joints. Wet closed versus chemical block:3.43927mV,1.08361e-4
muscle,9.42463e-6rad joints;versus proprioceptive block:4.08121mV,0.0104352
muscle,0.00316331rad joints. These support active causal connections through
this fitted network. They do not establish correct receptor physiology,
control-neuron necessity/sufficiency,food finding or functional locomotion.
Maximum input currents~0.24955pA water/~0.14051pA wet;all-blocked currents0.
Wet Newton residuals<1e-8. AVBR/RIS voltage changes are recorded,but are not
interpreted as proof of their biological control roles.

Half-body-timestep versus coarse endpoint-aligned trajectories:max water
voltage difference0.0925666mV/joint0.00101866rad;wet0.0198468mV/0.000152679rad.
This comparison includes differing discretization of bath onsets/washout and
body/sensory updates;it is descriptive,not a convergence or biological pass.
Brain maximum substep stays0.1ms. Source schedules/head samples and all memory
are retained;no hidden phase signal or field-coordinate input reaches neurons.
Wet dynamics were genuinely interleaved;using an open-loop replay here would
have been incorrect. No low-speed uncertainty from E046 has been removed.

Short-window sustained regional bends water~.003017/.032738/.011060rad;wet
0/.000543/0rad. These do not requalify the prior300s skill or establish wet
crawling. No gains were tuned to these outcomes. The old no-input endpoint is
unchanged;zero-gain equivalence and continuity are verified in tests.
Full suite after implementation/JSON round-trip fix:
`MPLCONFIGDIR=/tmp/r-elegans-mpl .venv/bin/python -m pytest -q`,
236 passed73.48s. Shell syntax/whitespace checks pass. Server launcher prepared
with pinned references,absolute dedicated Python,four physical CPUs and nohup;
this agent has not launched it on mbslnx2. Primary sources and unresolved
transduction assumptions are documented in CURRENT_MODEL.md.

E048 archived externally at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E048`:903 files verified,
archive manifest SHA25611c93740fc4ecc8e57d05f1567b1c44936b5fb9bf5d2467ccd85ce54a958c239.
Final20s summary SHA256e0997258a835711e7cd207a48b0b9ff284d0de7f3f9ed7b3ec3e353dea67957d;
configuration SHA2560e4f2316379db89e0d9bd7b2ea05e86d23db2961419694bc4697ee88ca003285.
Both runs,logs,states,all chunk/full trajectories,HTML previews,immutable inputs,
source/environment snapshots and full test log are retained. Local copies and
checksums verified;OneDrive cloud synchronization not independently verified.
Compact record:experiments/records/E048-embodied-sensory-connections.json.
These evaluation inputs do not replace the separately preserved historical
training lineage. No optimizer/resume state exists for these deterministic tests.

## 2026-11-10 — E048 server results reported by user

User pasted completed diagnostic summary from mbslnx2:
/home/carsonc/r-elegans-results/embodied-senses-20261110T220423Z-694360.
Evidence is user-reported stdout,not an independently copied/hashed server
archive. Full server summary/configuration/source/environment,case durations,
checkpoint hashes and complete raw traces have not been inspected locally.
Do not invent these from the launcher defaults. Prior local comparison uses
its explicit20s source/configuration records. Compare reported numeric summaries
descriptively;exact cross-platform protocol identity remains unverified.

Planned read-only reconciliation:store all6 reported block comparisons and2
reported timestep comparisons in a separate server evidence record;calculate
numeric differences from local20s aggregates and timestep/chemical-effect scale
ratios. No simulation,parameter change,threshold change or new training.
Scientific caution:coarse/fine maximum trajectory differences are not error
bars or a rigorous error bound. A chemical-effect maximum smaller than this
sensitivity requires matched active/blocked refinements before robust motor
or behavior claims. Software routing evidence from the existing locality and
zero-gain tests is separate. Preserve the source run path and unknown metadata.

E048 server reconciliation executed locally:
`python3 /tmp/r-elegans-E048-reconcile-server.py`. It reads the local E048 compact
record and explicitly transcribed user-reported metrics,not remote files.
All6 matching block-effect voltage/muscle/joint maxima differ from local20s
aggregates by at most1.13088% relative. Reported neural half-timestep maxima
(0.0345781mV water,0.00999451mV wet) differ materially from local maxima
(0.0925666/0.0198468mV);do not claim identical numerical trajectories.
No source of these cross-host differences has been established from stdout.

Server chemical-block neural effects3.30557/3.43927mV, muscle effects
0.000104904/0.000109017, joint effects2.22623e-5/9.42171e-6rad in water/wet.
Proprioception-block muscle effects0.0318671/0.0103724 and joints
0.0166096/0.00316322rad are substantially larger at these experimental gains.
This is a model/parameter-dependent response,not a biological modality ranking.

Reported half-step joint sensitivity0.00101866/0.000152679rad is45.757/16.205
 times the chemical-induced joint effect. This is not an error estimate or proof
that the chemical effect is numerical noise. It identifies an unresolved need:
refine active AND blocked runs to assess the stability of their difference.
Connections/locality are tested,but the small chemical motor effect is not
numerically qualified as a useful behavior. Existing local summaries that said
nonzero body effects must be read with this qualification,not as locomotion.

Decision:record successful reported execution and qualitative software routing
agreement;retain experimental status. Propose matched chemical/block refinement
and causal AVB recruitment/release tests before long behavioral fitting. No
parameters,weights,acceptance thresholds or simulations changed. No new server
job launched. Compact record:experiments/records/E048-server-20261110.json.

Reported-results reconciliation archived at
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/E048/server-report-20261110`:
248 files verified;manifest SHA256
babc2e62b6c0aab704ee546a6868ab36d56bf95b6eb99708174f15966b6c597d.
This preserves transcribed summary metrics,local reference record,analysis code
and source/environment. It is not a copy of server raw trajectories/configuration.
Local archive verified;cloud sync unverified. Documentation/record-only change:
whitespace and structured JSON checks performed;no software test rerun needed.

## 2026-11-10 — Proprioception hypothesis and genetic validation literature

User emphasizes that proprioception may be important for motor patterns and
movement, and asks about C. elegans knockout studies. Literature review only;
no new simulation,fit or genetic perturbation implemented. Existing E048
feedback-block results remain software/model evidence,not biological validation.
The biological literature supports testing amplitude,coordination,steering and
propagation separately,not treating proprioception as a uniform excitatory gain.

Primary sources checked online:

- Li et al.2006,Nature,doi10.1038/nature04538,PMID16572173,PMC2865900:
  trp-4 loss alters/exaggerates body bending;DVA-specific rescue and DVA ablation
  distinguish channel function from whole-cell function. DVA can modulate motor
  output positively and negatively. A whole-cell feedback deletion is not a
  faithful molecular trp-4 knockout.
- Wen et al.2012,Neuron,doi10.1016/j.neuron.2012.08.039,PMID23177960,
  PMC3508473:genetic/optogenetic/microfluidic experiments identify B-type motor
  neurons as proprioceptive transducers for anterior-to-posterior coupling.
  vab-7 mutations alter DB axon orientation and impair dorsal wave propagation;
  this is a developmental/morphological perturbation,not a pure receptor loss.
  Cholinergic/B-type suppression also affects motor output,so it does not isolate
  sensory transduction alone. unc-8 loss did not abolish the tested coupling,
  an important negative result;do not label the B feedback proxy as UNC-8.
- Yeon et al.2018,PLOS Biology,doi10.1371/journal.pbio.2004929,
  PMID29883446:trp-1(sy690) trp-2(sy691) double mutants exhibit ventral circling
  and misregulated SMDD activity. Either single mutant lacks the circling
  phenotype,consistent with redundancy. Cell-directed rescue supports SMD/SMDD
  involvement. SMDD/SMDV ablations cause opposite steering biases;ablations and
  channel lesions are distinct interventions. Activity is misregulated rather
  than universally abolished by the double mutation.
- Xu et al.2018,PNAS,doi10.1073/pnas.1717022115,PMID29686107,
  PMC5948959:unc-7/unc-9 null mutations and AVB ablation impair descending
  facilitation of B-type oscillations/propagation. Key locomotion assays use
  A-type motor silencing and1Pa s liquid;do not generalize those quantitative
  phenotypes to an unmodified worm in water. A selected AVB-B edge deletion
  models a pathway lesion,not the entire pleiotropic unc-7/unc-9 genotype.
- Das et al.2021,Science Advances,doi10.1126/sciadv.abg4617,PMID34533987,
  PMC8448456:DVA-specific cytoskeletal/channel experiments distinguish stretch
  from compression,implicate UNC-70 spectrin,TRP-4 and TWK-16,and find residual
  curvature-dependent Ca signal in trp-4(sy695). This qualifies the older simple
  claim that TRP-4 loss abolishes all DVA proprioception. Directional/local
  opposing mechanotransduction is not represented by our global positive RMS
  curvature-to-DVA proxy. This source is a future mechanism/validation target,
  not an implemented extension or an exhaustive statement of current knowledge.

Inference/decision:proprioception may supply coordination and negative as well
as positive feedback;loss can preserve movement yet produce excessive bends,
circling or poor propagation. Proposed validation should compare intact and
selective-feedback lesions with unchanged neural/muscle/mechanical parameters,
and test restoration,without retraining each lesion to hide its effect. Match
strain,medium,intervention and observables before interpreting a gene phenotype.
Use these as held-out checks of causal mechanism,not amplitude-only training
rewards. Current DVA/SMDD proxies cannot be called gene-faithful knockouts;
SMDD feedback is disabled by default and a zero-gain SMDD deletion would be
uninformative. No new motor skill or start/stop capability is established.

## 2026-11-10 — Unified model fitting objective clarified by user

User says the model should behave similarly to the biological perturbations,
then clarifies the overall goal:train a worm matching all existing data.
Interpretation:one shared biological model consistent with compatible datasets
and experimental variability,with held-out predictive validation. Preserve H1.1
incremental training and selected skill parents;final scientific claim cannot
be a different policy/parameter set independently fitted to each assay.
Record conditions,genotypes and unsupported developmental/organ/3D extensions
rather than forcing incompatible experiments into one unexplained target.

Read-only local inventory performed,not a fitting experiment or new acquisition.
Core22 and physiology10 manifest artifacts all exist and match recorded SHA256.
Existing digitized electrophysiology and transferred priors must be distinguished
from raw direct measurements. A single extracted calcium HDF5 has1600x109 raw
channels and1600-element behavioral streams plus30/62-column angle arrays.
There are55 unique canonical label candidates at confidence>=4;no claim all109
channels are identified or all55 pass later signal-quality checks. Existing
label metadata covers40 UIDs,but remaining archived recordings were not admitted
or counted as ready cohorts. Original archive568776589bytes was observed but
not rehashed/extracted in this audit. The raw/functional directory is empty.

Exact inventory command:
`.venv/bin/python /tmp/r-elegans-joint-data-inventory.py`.
Source/output at /tmp/r-elegans-joint-data-inventory.py and
/tmp/r-elegans-joint-data-inventory/inventory.json. It checks existing manifests,
small extracted HDF5/label hashes and array shapes,not model residuals.
No data split or target-score inspection/tuning performed. Units and shared
array lengths do not prove synchronization,physical calibration or spatial
correspondence to11 model joints. No model training,tests or parameters changed.

Created experiments/data_targets_v1.json:nonexhaustive readiness ledger for
anatomy/transmitters,electrophysiology,paired calcium/behavior,sensory response,
mechanics/media,genetic/circuit perturbations and action-state repertoire.
Updated PRD objective/learning/validation gates and corrected its stale prohibition
on truthfully reporting full-network training attempts (biological validation
remains absent). This is a requirements/data audit,not an implemented joint
trainer. All source arrays,checkpoints and raw archives remain outside Git.

Decision:admit datasets with identity,condition,units,alignment,observation model,
uncertainty and source-dependency checks;freeze grouped animal/trial holdouts
before fitting. Balance family losses without allowing large datasets to hide
failed modalities;report each family and missing capabilities. Keep documented
stereotypical parameters fixed and fit unknowns conservatively. Mutations map to
mechanistic interventions with no independent mutant refits;reserve held-out
interventions where feasible. Preserve discrepant evidence and use failures to
revise model mechanisms. Do not claim matching every raw observation exactly or
exhaustive data coverage. Immediate next preparation is paired neural/behavior
data admission and additional-animal inventory,not a blind longer motor-only run.

Parameter identifiability remains part of the original research question:even
joint data fits may admit multiple weight/channel configurations. Keep
admissible candidate ensembles and report their uncertainty/robustness,while
requiring each candidate to explain the assays with shared parameters. Do not
choose a different ensemble member after seeing each held-out experiment.

Inventory/requirements archive:
`/Users/carsonc/OneDrive/r-elegans/notebook/evidence/unified-data-objective-20261110`,
249 files verified;manifest SHA256
f9eae55423e9273d467b297bb677b97cb2daee50e38063559cffbc8fbbebe3c3.
Includes inventory source/output,initial data-target ledger and source/environment
snapshot. Original raw data remains in its established external location,not
copied into Git. Local archive verified;cloud synchronization unverified.
Structured JSON and whitespace checks pass;no runtime code changed or software
test rerun. No joint trainer,optimization run or comprehensive data census is
claimed by this documented objective and source-readiness audit.

## 2026-11-10 — Planned E049 paired neural/behavior data admission

User authorizes preparing the paired recordings for joint fitting. Scope:verify
source archive/labels against publisher checksums,inventory unscrumbled original
HDF5 members,inspect authoritative extraction/processing code and cohort CSV,
resolve units/timing/DV/segment conventions,produce an external processed dataset
and an immutable group split with explicit prior-exposure exclusions. Preserve
all raw data. No model fitting,weight change or behavior qualification in this
preparation stage. Reference model remains adult hermaphrodite,11-joint planar
body with E048 input mechanisms;do not force62 source angles into11 joints until
their meaning and ordering are established.

Local source header review already shows no HDF5 unit attributes. Publisher
Zenodo metadata points to the Atanas/Kim Cell2023 analysis/source repositories
and WormWideWeb metadata CSV. It explicitly requires per-dataset
head-angle DV correction,so file names or array shapes alone cannot establish
orientation. Metadata label keys use Julia one-based ROI numbering. Need to
inspect actual upstream writer for array and index conventions,including
potential ROI skipping and signal variants before admitting neural labels.

Planned admission criteria:publisher checksum agreement,member path safety,
strict finite/monotonic seconds timebase,clear resampling lineage,physical unit
and DV evidence,unique high-confidence canonical identities,per-signal missing
masks and documented exclusions. Preserve unidentifiable channels as excluded,
not guessed. Fix animal/trial groups independently of model residuals;the
previously explored2022-06-14-01 must not become a blind test animal. Prior
metadata-only inspection of other files is different from fitting them;track
actual exposure and avoid claiming a never-viewed or prospective test set.
Use published shared provenance/group IDs to avoid same-animal leakage. If a
metadata field cannot be established,mark the affected target ineligible rather
than inventing a conversion. Raw/derived files stay under external OneDrive;
commit only code,small schemas/manifests and notebook summaries. Capture exact
commands/source/environment and all original checksums. Freeze rules before
looking at model/data residuals. Numeric acceptance derives from source schema,
not arbitrary thresholds selected to improve model fit.

E049 local outcome (2026-11-10): verified the original (not scrambled) archive
and label file against publisher MD5, then all 68 HDF5 files against the pinned
publisher CSV SHA256 values. Safe streaming extraction rejected the archive's
checksum CSV on the first attempt; the explicit allowlist was corrected to
include that regular file, then extraction completed (68 HDF5 plus checksum CSV).
No source files were changed. Publisher metadata revision:
0422546192ef58d0de4740a0aa91fd38de22da2e.

Prepared 21 identified baseline recordings, 33,630 frames, 130 distinct canonical
neurons across the cohort, 25–86 confidence>=4 identities per recording. All
21 ROI skip maps are identity maps. The importer rejects duplicate identities,
low confidence, noncanonical labels and unverified nonidentity index spaces.
It preserves NaNs with explicit masks, author-aligned confocal behavior arrays,
absolute timestamps and source split/gap boundaries. One recording has two
segments; segments remain in the same date fold. No normalization, interpolation,
parameter fitting, target-driving current or body-joint conversion was added.

Frozen metadata-only split: 17 train, 2 validation, 2 test recordings, grouped
by date. June 14, previously fitted, is forced to train. Date groups are not
verified independent animal IDs; these are retrospective held-out assignments,
not claimed prospectively unseen data. Forty-seven other recordings remain in
the inventory with explicit quarantine reasons (heat, unlabeled or GFP).

The published methods establish one-day adult hermaphrodites and NeuroPAL/GCaMP
SWF702 cohort, with food-containing M9 on 2% NGM agar, bead-spaced coverslip and
five-minute settling. Exact per-UID genotype and recording temperature remain
unverified. This preparation identifies a real condition mismatch with both
unconfined water and the implemented 1.7% wet-agar calibration. Published source
processing includes ROI interpolation and full-record bleaching correction;
finite masks cannot recover the missing original ROI masks. Original source
angles are radians, signed velocity mm/s; publisher orientation correction
applies to head angle and angular velocity only. No source-angle-to-11-joint
mapping is asserted. No absolute calcium or voltage calibration is asserted.

Reproduction command (external artifacts, immutable output directory):
```
.venv/bin/python -m scripts.prepare_paired_recordings \
  --recordings /tmp/r-elegans-E049-extracted \
  --metadata /tmp/r-elegans-E049-sources/atanas_kim_2023.csv \
  --labels /Users/carsonc/OneDrive/r-elegans/raw/calcium/atanas_kim_2023/neuropal_label.json.bz2 \
  --output-dir /tmp/r-elegans-E049-reproducible
```
The earlier preparation invocation's manifest was augmented while inspecting
methods; a subsequent clean invocation now reproduces that manifest and split
byte-for-byte. Source/environment snapshots, extraction script, pinned publisher
source excerpts, original methods PDF, metadata and stdout are preserved
externally. Artifact hashes and upstream revisions are in
experiments/records/E049-paired-data-admission.json.

Verification: three targeted admission tests passed; full suite **239 passed
in 76.07 seconds**. The subsequent CLI provenance/protocol addition was executed
against the full real archive and reproduced manifest and split hashes. These
are software/integrity checks, not evidence of physiological model agreement.
Next: implement verified spatial and calcium observation mappings, represent
mounted medium conditions, then stage joint fitting with shared unknowns. This
is prepared data with explicit remaining admission conditions, not a joint fit.

E049 archive completion: copied and SHA256-verified 346 extracted, processed,
source and run-record files into external OneDrive storage. Prepared manifest
SHA256 36456db8732bebc18f328d91bd7b07035e0a31218c5afdfd0cc92aca1dc72780;
split SHA256 85e11be2d695a37974496e4b8b16031f0399cd544f98d717b5461bab7cbbe85b.
Local copied-file integrity is verified; cloud synchronization is not independently
verified. No server launch is needed for this preparation step.

## 2026-11-10 — Planned E050 paired observation and joint-fit pilot

Question: can the existing full channel network, without target-driving current,
be evaluated and incrementally fitted against neural and movement observations
from the same training recording? First implement publisher-defined spatial
observables and an explicit calcium observation likelihood, preserving E049 folds.
Verify source spatial scale and head/tail orientation before mapping. Preserve
raw angle observations; never treat 62 tangents as 11 joint angles.

Parent: preserved winner7001 if locally available and hash verified; no random
fallback. Fit only unknown non-leak channel densities and existing chemical/gap
strengths, with symmetric electrical pairs and bounded multiplicative changes.
Fix leak, capacitance, reversal potentials, channel kinetics, reporter kinetics,
NMJ/body parameters and anatomical masks. Any transferred indicator or mechanical
parameters remain exploratory assumptions, not measurements. Zero tonic bias,
no injected target current or target body forcing.

Stage the pilot using short training-only windows and a coarse observation
objective before seeking long-record trajectory agreement. Keep animal initial
conditions and indicator calibration explicit. Existing mounted-medium mechanics
are not calibrated to these recordings: compare available mechanical hypotheses
as exploratory proxies, not biological fits or certified bounds. Do not use test
records for initialization, scaling, selection or debugging targets.

Acceptance: synthetic geometry/observation tests, missing-data masks, no fold
leakage, frozen-field checks, finite locally executed baseline and candidate
losses, saved lineage/RNG/source records. Success is an executable constrained
joint-data fitting path, not demonstrated biological fidelity. Exact invocation,
seed, numerical settings, data hashes and result will follow execution. No long
server run launched from this Mac.

E050 method refinement during implementation: use a joint *summary-statistic
objective*, not a calibrated observation likelihood. Available source metadata
cannot identify optical offsets/noise, precise per-neuron exposure times,
per-animal initial states or mounted-medium drag. Relative brightness is fixed
basal brightness 0.1 plus the existing Hill/lag reporter state. Constant optical
gain cancels in coefficient of variation and lag-1 correlation; additive offsets,
reference-channel fluctuation and observation noise remain unmodeled. Fixed
weights 0.1 CV, 0.5 correlation and 0.1 rad are optimization scales, not measured
uncertainty. This is the next incremental fitting stage, not exact trajectory
prediction or an all-data biological validation.

Source geometry: publication repository f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3,
ANTSUN_NeuroPAL notebook uses seven binned pixels per nominal segment; unit.jl
specifies 791 unbinned pixels/mm and segmentation downsamples by two. Source
medial axes start at the nose. Nominal interval = 17.699 micrometers; Julia
points [1,9,17,25,33,41,49] define five unsigned chord bends. Source endpoint
quantization and temporal averaging prevent exact reconstruction. Synthetic
constant-curvature comparison with 12 rods has <0.011 rad discretization
error for a 1mm curvature radius; this is a software example, not a global
biological error bound. Head-angle reconstruction from source points [1,5,8]
on the eight-frame June14-01 training window agrees within 0.0011781 rad.
Chirality, locomotion direction and exact pointwise joint angles are not fit.

Publisher nir_vec_to_confocal averages within volume-associated NIR ranges.
The released target lacks those exact ranges, so midpoint-defined exposure bins
are an explicit sampling hypothesis, with point sampling as a sensitivity option.
Model fine-step tangent angles are unwrapped and averaged before source-space
chords are reconstructed; target observations are not resampled. The source's
low-rate paired imaging cannot alone establish high-frequency gait identity.

Local pilot 1 command:
```
JAX_PLATFORMS=cpu .venv/bin/python -u -m scripts.fit_paired_recordings \
 --prepared-data /Users/carsonc/OneDrive/r-elegans/processed/calcium/paired_E049_v1 \
 --reference /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E044/full-body-assessment-20261110T174954Z-687519/parameter_reference \
 --output-dir /tmp/r-elegans-E050-pilot --frames 8 --iterations 2 --minutes 4 \
 --media liquid wet-agar --refine
```
Seed50, default 2s burn-in, 2ms body step, <=0.1ms neural substep, boxed sample
bins, maximum unknown conductance factor1.2. Completed in63.31s. Baseline shared
objective35.53717; neither proposal improved it and no parameter was changed.
Refined objective35.54648. Existing parent SHA417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14
was retained exactly. Both proxies finite. Water and 1.7% wet agar are hypotheses,
not matched2% coverslip conditions or certified bounds. On this window, measured
coarse-bend standard deviations are0.315–0.449rad versus wet-proxy0.00338–0.01088rad;
these are coarse unsigned chord statistics, not the earlier11-joint p2p metric.

Local pilot2 planned/executing: same parent and settings, two shared training
windows (June14-01 and June14-07), eight proposals, wet-agar proxy, maximum factor2.
No held-out targets are loaded. Exact invocation/source/environment captured in
its external run records; local result to follow. No server experiment started.

E050 local pilot2 outcome: completed eight bounded proposals on the same
full302-neuron network shared by June14-01 and June14-07 training windows.
Command matches pilot1 except `--uids 2022-06-14-01 2022-06-14-07`,
`--output-dir /tmp/r-elegans-E050-shared-pilot`, `--iterations 8`,
`--media wet-agar`, `--maximum-factor 2`; seed50 and all numerical settings
unchanged. Fresh greedy search from parent, not continuation of pilot1.
Elapsed157.57s. Shared objective47.48417 ->47.38850 (0.2015% decrease), with940
nonzero offset coordinates among9616 eligible independent unknowns (4814
channel densities,3709 chemical strengths,1093 electrical pairs). All frozen
fields unchanged. Selected checkpoint SHA256
9328de15ac3a05e9c5133cfa0f58655478df445698bdecbdda18fa7f70a40fcd.
Both data families contribute, but their amplitude discrepancy remains large;
this tiny short-window decrease is not a qualified skill or biological fit.
No held-out recording was used. Source head-chord checks agree within0.0013362rad
across the two selected windows; that check does not calibrate source arclength
or mounted mechanics.

Read-only numerical check: pilot2 already evaluated its selected network at
half body timestep and half maximum neural substep. A separate source-recorded
invocation `.venv/bin/python /tmp/r-elegans-E050-refinement-check.py` then evaluated
the original parent at the same refined settings without changing any parameters.
Refined parent objective47.49928 vs selected47.41260, preserving the improvement
in this check. This is not a certified integration error bound or biological
acceptance. The source helper, exact environment/invocation and predictions are
archived alongside the pilot. Current `--refine` now records both parent and
selected objectives automatically for subsequent runs.

Verification: full suite **244 passed in81.40s**, then all five new observation,
split-integrity and safe-extraction tests passed after a package-import cleanup.
Server launcher shell syntax passed. Cached pinned-data fetch CLI executed
successfully against existing artifacts through an external staging root;
fresh server HTTP download and launch have not been executed locally. New data
fetch uses pinned URLs/checksums, refuses changed cached sources, keeps extracted
HDF5 outside Git and recreates the frozen E049 split before fitting. Unsafe
archive links and path traversal are rejected in tests.

Decision: preserve this candidate independently, leave bundled models unchanged,
and offer a short20-minute detached server pilot on two16-frame training windows.
It reports separate neural/movement losses and supports the existing CPU strategy.
A20-minute fitting budget excludes first download/preparation and final refinement.
No long server run was launched. This is an incremental statistics-fitting tool;
next scientific requirements are broader training windows, optical/initial-state
models, independently calibrated mounted-medium mechanics and later held-out tests.

E050 archive completion:819 local run/source/evidence files copied externally
and copied pilot files SHA256-verified under OneDrive/r-elegans/notebook/evidence/E050.
Both pilot outputs, full stdout, RNG/search state, selected predictions, comparison
figure, refinement helper and pinned parent parameter-reference lineage preserved.
Local copy integrity is verified; cloud synchronization is not independently
verified. Future fit runs also copy recursive server warm-start history or the
verified compact reference into their external inputs, preserving lineage before
fitting. The independent selected candidate is not a bundled model promotion.

## 2026-11-10 — Planned E051 mbslnx2 paired-workload capacity study

Question: which CPU/SMT, two-A100 or mixed process layout maximizes completed
E050 candidate evaluations per second on mbslnx2? The previous PPO benchmark
cannot establish performance for this sequential NumPy wet-surface reference.
The server has 128 logical processors and two A100 80GB devices (user-reported);
physical topology, available memory and per-backend timings must be measured.

Method: run the same paired training windows, fixed parent and candidate on all
workers. Pin affinity before importing numerical libraries. Warm up every worker,
then synchronize measurements with a shared gate; exclude compilation from steady
throughput and separately report warm-up and end-to-end wall time. Probe CPU
worker counts, physical-core vs SMT layouts, both GPUs and a mixed CPU/GPU case.
Reserve the first four physical cores for the existing pilot and a separate CPU
allocation for GPU host work. Cap CPU process count using available memory and
observed single-worker peak RSS. Compare numerical outputs to the CPU reference
before recommending a GPU layout. Failure/timeout is recorded, not a speedup.

Fixed: winner7001, E049 frozen train folds, same 16-frame windows, identical
bounded deterministic candidate, E050 reporter/spatial/temporal assumptions,
physical equations and timestep. No fitting or held-out target selection during
capacity measurement. An optional subsequent independent-seed fitting ensemble
uses the selected measured layout, preserving each worker separately. Separate
seeds increase search throughput; they do not accelerate one sequential rollout
or constitute a shared-gradient optimizer. No scientific checkpoint promotion.

Local verification: process orchestration, gating, affinity planning, throughput
arithmetic, memory caps and failure handling with synthetic subprocesses; run
real local worker if feasible. Only the user can launch the hardware measurement
on password-authenticated mbslnx2; no remote timings or GPU utilization invented.

E051 shelved at user's explicit request to leave processor tuning aside and
launch a three-hour fit instead. No hardware-capacity run or new GPU performance
was measured on mbslnx2. Uncommitted capacity-tool work is preserved separately
in /tmp/r-elegans-E051-shelved; it is not a deployed launcher. Local synthetic
orchestration tests passed; one local unpinned CPU workload check took22.96s
per evaluation after24.24s warm-up, not a server scaling result. No capacity
benchmark or ensemble has been launched remotely.

## 2026-11-10 — Planned E052 three-hour paired-fit continuation

Resume the completed server E050 search from its best saved offsets and saved
PCG64 RNG state, preserving original winner7001 anchor/bounds and all fixed
fields. Do not repeat the first34 proposals. Verify completed summary, checkpoint
hash, offsets, RNG, frozen data split and unchanged numerical/objective settings.
Copy recursive parent results into the new external run. Three-hour additional
fitting budget, same two16-frame training windows, no held-out targets, no drive;
setup and final refinement take additional time. Preserve independent winner and
bundled model. Local resume validation and short integration check before push;
user launches the nohup server job. No claim of in-flight process continuation
or biological qualification.

E050 server result received from user stdout (2026-11-10): completed34 proposals,
1356.11s including refinement; fit budget20min. Objective47.93021 ->47.51395
(~0.87% decrease); refined parent47.94112 ->selected47.56418, same improvement
direction.2439 offset coordinates changed, frozen fields reported unchanged.
Reported checkpoint SHA256
c435ee69645f27163af21ef5590b05a30151a38df7389a6f2fe70ef4db1ee481.
PAIRED_FIT_EXIT_CODE=0. Model coarse-bend fluctuation remains about20–75 times
smaller than measured. These are training statistics, not biological validation.
No exact server run directory, source capsule or complete checkpoint was copied
back; do not invent a run identifier or claim independent checkpoint verification.
Compact user-reported evidence: experiments/records/E050-server-20261110.json.

E052 continuation implementation restores original-anchor offsets and PCG64 RNG
from a completed run, checks matching objective/data/integration settings and
saved network consistency, copies the parent recursively and continues global
proposal numbering. No mid-candidate process state is claimed. Baseline and
refined-baseline now refer to this continuation's starting candidate. Default
three-hour wrapper discovers the latest completed paired-fit directory and
verifies its checkpoint hash before launching under nohup. The source run is
never modified. E051 capacity work remains shelved.

E052 local verification: six focused observation/resume tests passed. A real
one-proposal continuation from the eight-proposal local E050 pilot restored its
starting objective exactly (47.38850166772256), used the saved RNG and advanced
global proposal8 ->9. Frozen fields remained unchanged. Command used
`python -m scripts.fit_paired_recordings --resume-run /tmp/r-elegans-E050-shared-pilot`
with matching two eight-frame wet-proxy windows, factor2 bounds and one additional
proposal; full exact arguments/source/environment in external run records. The
server continuation will instead restore the user's completed34-proposal,
16-frame search after verifying its own saved configuration. No remote launch
or server checkpoint verification is claimed yet.

## 2026-11-10 — E052 server continuation result (user-reported)

Server run /home/carsonc/r-elegans-results/paired-fit-20261111T003225Z-698419
reports completed, stop_reason=time_budget,335 additional proposals from global
34 to369. Elapsed10941.68s includes setup/refinement. Starting objective
47.51395069275865 matches the previously selected E050 server objective exactly,
consistent with the intended continuation rather than a fresh anchor restart.
Selected objective44.9933180382051:5.305% reduction. Refined start47.56417529122987
vs selected45.002973328269036:5.385% reduction, preserving the gain. This paired
refinement is a descriptive numerical check, not a certified error bound.

All ten sampled coarse chord-bend standard deviations increased by23.3–49.5%,
but measured/model amplitude ratios remain14.77–50.90. This is unsigned coarse
chord fluctuation, not joint p2p amplitude, coordinated propagation, speed or a
new qualified autonomous skill. Both normalized calcium CV and lag1 errors
worsen on both training windows at the nominal timestep; their equally weighted
neural component increases about1.61%. The falling scalar objective therefore
trades some neural agreement for better body statistics. We cannot report joint
physiological fidelity or infer a successful calcium fit.

9474 of9616 offset coordinates now differ from the original winner7001 anchor;
this is neither the count of newly changed coordinates in this continuation nor
evidence that bounds were reached. Frozen fields reported unchanged. Reported
checkpoint SHA256
ffa4ea6addd9309382204160855bdf3b21fbd17b1ff0b5a669062b66cab8c41d.
The summary was pasted by the user; no complete checkpoint, numerical source/
environment record, history or raw predictions were independently copied back.
No exact server revision or unreported exit code is invented. Compact manually
transcribed evidence and locally computed comparisons are in
experiments/records/E052-server-summary.json.

Decision: retain candidate and original skill checkpoints. Before another long
fit, inspect bound occupancy and activity/calcium-to-muscle-to-body transfer to
locate the remaining amplitude bottleneck. The fixed reporter/NMJ/mechanics and
mounted2% agar mismatch remain hypotheses/limitations. Do not infer which link
is limiting from these aggregate statistics alone. Hardware tuning remains
shelved; no new run, held-out evaluation or model promotion was launched.
Verification in this turn is JSON/arithmetic and continuation-baseline consistency;
no simulation was rerun and no tests are needed for metadata-only updates.

## 2026-11-10 — Planned E053 eight-hour paired-fit continuation

User authorizes eight additional fitting hours after the reported E052 result.
Question: does continued bounded neuron/connection search further reduce the
same paired-statistics objective, and how do body and calcium components trade
against each other? This is continued exploration, not a new biological claim.

Expected parent: latest completed server paired fit, currently reported as
/home/carsonc/r-elegans-results/paired-fit-20261111T003225Z-698419,
checkpoint SHA256 ffa4ea6addd9309382204160855bdf3b21fbd17b1ff0b5a669062b66cab8c41d,
last global proposal369, selected nominal objective44.9933180382051. The wrapper
must discover and verify actual saved artifacts; the parent is user-reported,
not independently retrieved. If a newer completed run exists, the launcher
prints that actual parent and its new run record documents the lineage.

Method/settings: `sh scripts/continue_paired_fit_mbslnx2.sh 480`, from the updated
server checkout, using the absolute r-elegans conda interpreter under nohup.
Continue saved PCG64 RNG and best offsets; no seed reset or replay of earlier
proposals. Preserve original winner7001 anchor and factor2 bounds. Same two
16-frame training windows, frozen E049 folds, same objective weights, reporter
and sampling assumptions, wet-agar proxy, body timestep0.002s and maximum brain
substep0.1ms. All stereotypical fields, anatomy, NMJ/muscle/body and sensory gains
stay fixed; only existing unknown non-leak channel densities and chemical/gap
strengths remain eligible. No external neural drive or held-out data fitting.

Budget480 additional fitting minutes; setup and final paired timestep refinement
are extra. Retain recursive parent lineage, selected checkpoint, search/RNG
state, history, exact source/environment and predictions outside Git. Completion
means a finite preserved search and unchanged frozen fields, not successful
animal behavior. Compare each family and both refined parent/selected objectives;
calcium agreement need not improve merely because total loss falls. Parameter
bounds and amplitude-transfer diagnostics remain outstanding. No hardware-tuning
run or bundled-model promotion. Status planned: user will launch on mbslnx2;
no remote execution or result is claimed in this entry.

## 2026-11-11 — E053 eight-hour continuation result (user-reported)

Run /home/carsonc/r-elegans-results/paired-fit-20261111T040502Z-706911 reports
completed, time_budget,865 additional proposals, global369 ->1234, elapsed
28945.09s including setup/refinement. Starting objective44.9933180382051 matches
E052 selected loss exactly. Best40.01228735191045:11.071% reduction. Matched
refined parent45.002973328269036 ->selected40.01460331530774:11.085% reduction.
The finer-timestep check preserves the gain; it does not certify all temporal
statistics, long-autonomous behavior or biological fidelity.

All ten sampled unsigned coarse chord-bend standard deviations increased by
29.3–126.3%. Measured/model fluctuation-amplitude ratios remain10.83–33.82.
Nominal equal-weight neural objective component2.01549045 ->2.01600179 (+0.0254%):
essentially unchanged overall, with mixed per-neuron-summary changes. This is
not evidence of a meaningful calcium fit improvement. These are two short
training windows; eight hours is optimization wall time, not simulated trajectory
length or an eight-hour survival/behavior test.

Exact objective decomposition from the reported metrics and existing code:
for two recordings, each reported calcium_cv, calcium_lag1, bend_mean and
bend_std contributes one quarter of its summed per-recording error. Of the
4.98103 total loss decrease, mean-bend error contributes4.62949 (~92.94%),
bend-fluctuation error0.352075 (~7.07%), calcium CV0.000557, calcium lag1-0.001068;
the prior contributes about-0.0000269. Thus the scalar objective predominantly
improves average unsigned bending. Mean unsigned bending includes both posture
and dynamic excursions: this decomposition alone does not prove a static
posture or isolate a specific physiological bottleneck. Actual sampled bend
fluctuations increased substantially but remain too small.

All9616 eligible offsets reported nonzero relative to the original anchor;
this is not a new-coordinate count or evidence of bound saturation. Frozen
fields reported unchanged. Reported checkpoint SHA256
98ea66551777dd8b228bf0ee59d97043ce32ec1b8d7f2e04f139887ecd26d6e2.
Exact source revision, full server trajectories, RNG/history and checkpoint
have not been independently retrieved. Source attachment SHA256 and all parsed
summary sections are preserved in experiments/records/E053-server-summary.json.
No unreported exit code or raw-data backup is asserted.

Decision: retain E053 alongside previous skill checkpoints. Before another
unchanged long continuation, inspect actual trajectories and parameter-bound
occupancy, and make sustained dynamic amplitude explicit in the next curriculum
objective while tracking neural metrics separately. A long-autonomy/perturbation
assessment remains necessary for a qualified skill; future changes must retain
physical/numerical criteria and explicit reporter/mounted-medium assumptions.
No revised objective, new simulation or server launch was executed in this turn.
Verification: parsed the pasted output, checked369+865=1234 and matched E052/E053
boundary losses, recomputed all percentages and component weights. Metadata-only
updates do not require a new simulation/test suite.

E053 evidence preservation completed: pasted server stdout, parsing/arithmetic
script, compact analysis and file hashes copied to external OneDrive directory
notebook/evidence/E053. This preserves the supplied report, not the full server
checkpoint or trajectories. Cloud synchronization is not independently verified.

## 2026-11-11 — Planned E054 sustained-motion and bound diagnostic

User authorizes inspecting trajectories/bounds and preparing a sustained-amplitude
curriculum after E053. First deliver a read-only diagnostic on the actual latest
completed server paired checkpoint (expected E05398ea6655...); do not silently
replace it with an older local model. Full E053 artifacts are not yet on this Mac.

Question: do oscillations persist after settling, where along the neural -> NMJ
-> filtered muscle -> body chain is dynamic range lost, and how many unknowns
approach original factor2 bounds? Preserve the original anchor, fixed fields,
zero tonic bias, and proprioceptive feedback. No fitting, external gait teacher,
periodic input, parameter widening or biological promotion. Constant-voltage
control is a diagnostic intervention only. Test selected network over40 simulated
seconds, matched half timesteps, and held-voltage control; record dense voltage,
calcium/indicator, NMJ activation, signed/filtered muscle commands and actual pose.
Use late-window repeated-peak/trough sustained excursions, alongside std, linear
trend and mean posture. No selected oscillation frequency or phase target.

Acceptance: source/checkpoint/offset/hash consistency; exact input preservation;
finite trajectories; no frozen-field changes; separate read-only reports for
active, refined and held controls. Numeric diagnostics are descriptive, not a
measurement of true biological channel densities or certified integration bounds.
Keep original source/prepared data external, record full source/env/invocation,
and archive the parent recursively. Locally verify software and an older local
checkpoint smoke run, labeling it separately from E053. The user launches the
actual server diagnostic. Define the next amplitude objective after this evidence;
no further unchanged eight-hour fitting run in this step.

E054 local software outcome:247 tests passed in78.52s. Two new tests cover bound
occupancy (nonzero is not near-bound) and sustained excursion rejection of ramps,
static offsets and decays. Both existing paired training recordings reproduce
their saved indicator, joint and bend prediction arrays exactly with the default
rollout path. Dense recording/held controls are opt-in and not used by fitting.

A two-simulated-second smoke diagnostic on the older local E052 smoke candidate
completed active, refined and held-voltage cases. Its checkpoint is not E053.
The duration is intentionally too short for a sustained-motion qualification;
zero reported repeated joints cannot be extrapolated to40s or the server model.
That older local model had no offsets near original factor2 bounds. Full-trace
refinement differences included max joint0.0001543rad, filtered drive0.001711,
and max instantaneous voltage7.026mV; aggregate objective convergence must not
be mistaken for pointwise voltage agreement. This observation is descriptive,
not a certified timestep error or an E053 measurement. Reports include both full
and late-window differences for future runs. Numerical reporting floors are
explicit per-unit engineering thresholds, not biological acceptance criteria.

Actual local diagnostic command:
```
JAX_PLATFORMS=cpu .venv/bin/python -u -m scripts.diagnose_paired_motion \
 --run /tmp/r-elegans-E052-resume-smoke \
 --output-dir /tmp/r-elegans-E054-local-final --duration 2
```
Source/environment captured automatically; trajectory hashes and exact invocation
stay external. Local default-path comparison command and stdout preserved too.

Prepared server command `sh scripts/launch_paired_diagnostic_mbslnx2.sh 40` selects
and verifies the latest completed paired search, preserves it recursively, and
runs three read-only cases under nohup. The expected latest source remains E053;
only the actual server report will establish its bound occupancy and sustained
signal-chain behavior. A proposed curriculum specification is saved in
experiments/sustained_amplitude_stage_v1.json; its trainer is not implemented or
launched. Revised selection is a new stage, not an exact E050 continuation.

E054 evidence copied and local-final files SHA256-verified under external
OneDrive/r-elegans/notebook/evidence/E054, including default-path comparison and
full test output. The preliminary smoke without numerical reporting floors is
also preserved; its tiny-amplitude repeat counts are not biological rhythms.
Local archives do not constitute retrieval of E053 or verified cloud sync.

## 2026-11-11 — E054 server result reported

Evidence: user-pasted stdout, not independent server artifact retrieval. Run:
`/home/carsonc/r-elegans-results/paired-diagnostic-20261111T145342Z-766260`.
PAIRED_DIAGNOSTIC_EXIT_CODE=0; active, refined and held-voltage each completed40
simulated seconds. The pasted report omits checkpoint hash/source capsule and
wall duration; expected latest E053 identity is not independently established.
Exact invocation/environment and dense trajectories remain in server artifacts;
no backup or cloud synchronization is asserted. Small transcription:
experiments/records/E054-server-summary.json.

All three cases report zero repeated joints, muscles and voltage neurons, and
zero qualified sustained excursions throughout the printed signal chain. This
is not evidence of zero variance: the detector examines the late20seconds,
requires at least three peaks and three troughs, amplitude persistence across
five subwindows, and explicit numerical reporting floors. Slower oscillations,
transients or drift require full traces/late std/slopes to distinguish. Active
and refined PC1 fractions .9983135 and .9983145 describe nearly one-dimensional
shape variation, not periodicity. Held-control PC1 .9999720 is not an oscillator.
No muscle samples were reported within .01 of activation extremes.

All9616 offsets nonzero, none near lower/upper log bounds. Channel factors span
.5323–1.5319, chemical .6542–1.4740, electrical .6633–1.7650, inside .5–2.
Thus near-bound clipping is not supported as the explanation for failed rhythm
criteria. Agreement of the binary active/refined outcome does not establish
pointwise voltage convergence; full/late difference metrics were not supplied.

Decision: retain bounds and all prior independent checkpoints. Prepare a new
sustained-oscillation amplitude stage, with fixed body/NMJ/readout and no external
periodic drive. Separate dynamic amplitude from mean posture; establish repeated
activity before propulsion/control. Current evidence does not determine which
mechanism causes the failure, nor prove that all rhythms have disappeared. No
new fit, simulation or server launch in this reporting turn. Documentation and
JSON consistency checked locally; no software implementation changed.

## 2026-11-11 — Planned E055 sensory–motor foundation curriculum

User authorizes proceeding through reversible contraction, alternation,
propagation, translation and regulation, with sensory learning alongside motor
learning. This is a functional curriculum inspired by evolutionary simplicity,
not a reconstruction of nematode ancestry. Implement the first stage now; later
stages remain gated plans, not successful skills or automatic promotions.

Question: can a finite environmental chemical exposure elicit a reversible body
bend through existing AWA routing and recurrent/NMJ anatomy, with actual-body
proprioception active, while the same network is evaluated without a cue?
Use a nonperiodic diacetyl field onset/washout, never target-angle or motor current
injection. Cue minus matched sham separates evoked response from spontaneous
activity. No preferred left/right bend, waveform or gait frequency is specified.
This generic chemical response is an engineering prerequisite, not evidence that
real worms execute a particular reflex to diacetyl. Existing effective chemical
and proprioceptive closures remain uncalibrated biological hypotheses.

Parent: verify latest completed paired fit and its original winner7001 anchor,
checkpoint/search offsets, fixed fields, runtime anatomy and readout hashes on
execution. Preserve recursive lineage outside Git and retain all earlier skills.
Fresh stage/seed5500: bounded greedy channel and chemical/electrical search,
original factor2 limits. Also expose exploratory chemical, B and DVA transduction
gains within factor2 of their original nonzero defaults. Keep receptor kinetics,
response signs/routing, leak/capacitance/reversals, reporter, NMJ and mechanics
fixed. No sensory signal, target or reward may directly enter a motor command.
New objective and added sensory gains mean fresh optimizer/RNG, not E053 resume.

First-stage assay: 8 simulated seconds, cue2–3s, recovery6–8s; matched sham,
small cue-dose and onset perturbations, half integration timesteps, chemical
block and held-voltage control. Score cue-specific reversible bend and AWA
recruitment; penalize residual bend and joint-limit approaches. Initial operational
target .003rad cue-specific bend and .1mV AWA depolarization, residual <=25% of
response amplitude; these are engineering gates, not measured biological values.
Require all active validation variants and causal loss of response in blocked
controls before reporting a stage pass. Record unforced dynamics separately;
evoked response is not spontaneous oscillation, locomotion or a fitted calcium
recording. Keep the wet-agar proxy mismatch explicit.

Planned server command: `sh scripts/launch_sensorimotor_mbslnx2.sh 20` after pull.
Default20min search plus baseline/validation/compilation overhead, not a hard
wall-clock20min guarantee. Detailed configuration, invocation/environment/source,
parent hashes, PCG64 state, proposal history, selected/final networks, sensory
parameters and best/validation trajectories external. No server launch yet.

E055 pre-final-verification refinement: require AWA recruitment to recover as
well as the selected body joint (each residual <=25% of its response). Report a
proprioception-blocked intervention descriptively, without requiring the initial
reflex to depend on feedback. Require refined cue-minus-sham joint trajectories
to agree within .00075rad (25% of the .003rad engineering target); this does not
certify voltage convergence. Source calcium-data agreement is not optimized in
this exploratory stage; preserve paired-fit parents for later joint constraints.
The sensory–motor curriculum precedes the proposed sustained-amplitude stage,
rather than treating an evoked bend as spontaneous wiggling.

E055 planned target correction after preliminary local software smoke: the older
local parent produced AWA recruitment4.30mV but only6.54e-6rad selected
cue-specific bend (strongest raw joint1.08e-5rad). One proposal increased the
selected response to6.92e-6rad; chemical and held-voltage controls were exactly
zero. This is not E053 server evidence or a qualified skill. A .003rad first
target would demand hundreds-fold growth, contrary to the user's baby-step
strategy. Therefore set the initial engineering target to .00003rad and plan
a reviewed amplitude ladder [.00003,.0001,.0003,.001,.003]rad. Nothing is promoted
automatically. Refinement tolerance scales with target (.0000075rad initially).
The earlier .003/.00075rad plan is superseded for the first rung, not silently
removed from history. Re-run final software smoke with the new target before
publishing; the preceding local smoke(s) retain their original source/settings.

E055 local verification outcome (not server execution): full suite251 passed in
92.87s; after lowering the first-rung target, three targeted protocol tests passed
in0.50s. Existing two local paired-recording indicator/joint/bend arrays remained
bit-for-bit identical with the default disabled chemical input. Shell syntax and
Python compilation checks passed. Tests cover stimulus locality/washout, selective
gain scaling/blocking, rejection of static/pre-cue differences, non-recovering
body or sensory responses, and matched-sham spontaneous movement.

A final8-second-per-case, one-proposal smoke on the OLDER local E052 candidate
(parent SHA9328de15ac3a05e9c5133cfa0f58655478df445698bdecbdda18fa7f70a40fcd)
completed in189.89s with the new .00003rad first-rung target. It did NOT qualify:
selected cue response1.134e-5rad, recovery ratio .2791 (target<=.25), AWA response
4.503mV. Both neural and sensory gains were updated;128 neural/connection offset
coordinates changed plus all three exploratory gains. Chemical-block and held
controls gave exactly zero differential bend/voltage response. Refined body cue
effect differed by9.806e-7rad, below7.5e-6rad tolerance. Proprioception-blocked
case is descriptive, not a required causal prerequisite for this first reflex.
No realistic motion, spontaneous oscillation, calcium-data fit or E053 server
performance is established by this smoke.

Final local command:
```
JAX_PLATFORMS=cpu .venv/bin/python -u -m scripts.train_sensorimotor \
 --parent-run /tmp/r-elegans-E052-resume-smoke \
 --output-dir /tmp/r-elegans-E055-first-rung --iterations 1 --minutes 5
```
Selected/final saved network arrays agree; original parent checksum unchanged;
neural offsets and sensory multipliers remain within factor2 bounds. Preserve
all preliminary and final runs with their source snapshots, RNG/state/history,
selected/final checkpoints and actual-pose previews externally. Compact evidence:
experiments/records/E055-local-verification.json. Server run remains pending;
launcher defaults to20minutes including baseline/search, plus subsequent
validation overhead. It uses the dedicated conda environment and nohup, verifies
the actual latest paired parent, and prints its hash in stdout. No server launch
was performed. Review the first rung before raising its target or implementing
subsequent stages; no automatic stage progression or shared-skill claim.

E055 evidence preservation completed: all three local verification runs and full
stdout, source/environment capsules, recursive parents, selected/final networks,
RNG/search states, trajectories, previews, test stdout and the legacy-comparison
helper copied to OneDrive/r-elegans/notebook/evidence/E055. Every copied run file
was SHA256-compared against its /tmp source; archive manifest covers3460 files.
This verifies local copies, not OneDrive cloud synchronization or retrieval of
the actual E053 server checkpoint. The earlier report helper briefly used system
python without NumPy and failed before mutation; rerun with .venv succeeded.

## 2026-11-11 — E055 partial server result reported

Evidence: user-pasted summary tail; run ID, iteration count, baseline/best, full
active/dose rows, source capsule and final exit-code digits are absent. Do not
reconstruct these as facts. Exact artifacts remain on server, not independently
retrieved. Partial transcription: experiments/records/E055-server-partial-20261111.json.
Reported elapsed1414.7473s=23.5791minutes, stop_reason time_budget. Candidate SHA
437b227e2fdcb5ff376c9b5324ede09a49d1dbc2676c6886846bd10abc2e624d; parent SHA
98ea66551777dd8b228bf0ee59d97043ce32ec1b8d7f2e04f139887ecd26d6e2 confirms
the E053 parent for THIS run. Original anchor417254ad... unchanged as reported;
frozen_fields_unchanged=True.3949 neural/connection coordinates changed versus
parent; that does not measure bound occupancy. Sensory multipliers chemical
1.42231, B .99205, DVA .83114, all within original .5–2 exploratory gain bounds.

An unlabeled validation row shows joint4 peak1.274003e-5rad, residual2.4871e-6rad
(19.52%), AWA depolarization6.02456mV with residual .0026474mV (.04394%),
pre-cue difference0. Peak is42.47% of the3e-5rad first-rung target, requiring
2.355fold more amplitude in that case. Its label is missing: do not silently
assign active/refined or use it as the baseline for a proprioceptive comparison.
Chemical-blocked and held-voltage rows have exactly zero cue-specific bend and
AWA recruitment; blocked_controls_passed=True. Refined cue-effect difference
2.75044e-6rad passes the7.5e-6 engineering tolerance; this is not a voltage-error
bound or biological pass. Proprioception-blocked row shows1.53880e-5rad peak
and12.55% residual; full active row is needed for paired interpretation.

Interpretation: sensory recruitment and recovery operate in the reported case,
but its motor amplitude remains below the deliberately small first target.
Cannot quantify optimization improvement without matching baseline/best. No
autonomous oscillation or biological qualification; no next stage launched.
Decision: retain this candidate, remain at contract_relax, and obtain the compact
complete summary before choosing further training or an objective change. No
new run or code experiment executed in this reporting turn. Metadata/JSON and
arithmetic checked locally; software tests not repeated for record-only edits.

## 2026-11-11 — E055 complete server summary and planned E056 continuation

User reports full compact summary for
/home/carsonc/r-elegans-results/sensorimotor-20261111T154821Z-767568.
Passed=False,48 proposals, selected_iteration48. Baseline selected joint4 bend
9.46244995e-6rad ->1.430110702e-5rad; recovery ratio .139702 ->.106643;
AWA4.33661 ->6.02619mV. Scalar objective .68461635 ->.52338115. Exact percentages
and all reported validation rows will be recorded in a compact E055 full record.
The previously unlabeled row is now explicitly identified as refined. Preserve
E055 partial record as historical evidence and link the correction. Parent and
selected hashes match the earlier partial tail. No server artifact retrieval.

Decision: remain on the .00003rad contract_relax rung; learning is measurable but
all active validation cases remain below target. Proprioceptive block increases
peak slightly in this particular assay, so feedback benefit has not been shown;
do not remove proprioception based on one engineering reflex. No promotion or
claim of autonomous oscillation. Plan E056:60additional minutes of the SAME
objective from selected E055 parameters AND learned sensory gains, original
winner7001 bounds and all fixed parameters unchanged. Restore saved post-search
PCG64 state and global proposal numbering; verify baseline against parent best.
This is continuation of a completed greedy search, not an interrupted simulation
or exact bitwise replay across machines. Preserve full recursive lineage, selected
and final checkpoints, RNG/state, source/env/commands externally. Validation and
acceptance criteria unchanged. Seed inherited5500, not a fresh random restart.
Planned command: sh scripts/continue_sensorimotor_mbslnx2.sh 60. User launches;
no server launch authorized through SSH or performed here. Test continuation
locally on the older local smoke candidate, explicitly separate from E055 server.

E055 complete-summary arithmetic: active amplitude+51.135%, objective-23.551%;
active target fraction .47670, minimum active validation fraction .41095
(dose-low), requiring2.433fold more amplitude in that case. Relative recovery
residual improved13.970% ->10.664%, while absolute residual increased1.322e-6
->1.525e-6rad; do not conflate these. Refined peak is10.916% below active.
Proprioception-blocked peak is7.600% above active; its recovery ratio12.547% is
worse than active10.664%. Both remain amplitude-unqualified. This does not imply
feedback should be removed from later rhythmic/propagating skills. Sensory gain
and neural changes are confounded; training gain cannot be attributed to network
weights alone without further interventions. Full compact transcript recorded
in experiments/records/E055-server-summary.json; partial record remains dated.

E056 implementation: verified completed E055 restoration checks selected/final
network, neural and sensory offset agreement, effective transduction gain values,
original bounds, objective/numerics and PCG64 state. It restores the final search
RNG even when the final proposal was rejected. Re-evaluated starting loss must
match prior selected loss or abort. Recursive original paired parent and prior
E055 search retained separately. New summary distinguishes additional proposals
from global iteration and newly changed coordinates from total change versus
the original paired parent. This is not mid-simulation state resume.

E056 local verification complete:255 tests passed in91.08s; targeted continuation
+protocol suite7 passed in0.61s. Shell syntax and Python compilation passed.
One additional proposal on the older local E055 smoke restored global1 ->2;
starting objective exactly equals its prior best. The proposal was rejected,
which verifies that the saved post-proposal RNG advances even without a new
selected model. Independently reconstructed its next joint proposal and sensory
noise, matching saved final PCG64 state exactly. Saved baseline offsets/gains
match the parent's selected values; parent checksum unchanged. Full8-case
validation completed, still amplitude-unqualified, as expected for this software
smoke. This is not the reported server checkpoint437b227e.... Numerical objective
and body/sensory equations were not changed. See E056-local-verification.json
for exact invocation, wall duration, source captures and hashes. No server launch.

E056 evidence archived and copied run files SHA256-verified under
OneDrive/r-elegans/notebook/evidence/E056 (2308manifest entries), including source
capsules, recursive lineage, selected/final checkpoints, search/RNG states,
trajectories, previews, stdout/tests and compact E055 full server transcription.
Local copy integrity is verified; OneDrive cloud synchronization and the original
server artifacts are not independently verified.

## 2026-11-11 — E056 continuation server tail reported

Evidence: user attachment193a9636-bdf2-46fa-9925-b956e6ab4515, truncated stdout,
not independently retrieved server artifacts. Complete onset-shift/refined/blocked
rows and terminal metadata parsed with JSONDecoder; prefix lacks run ID,
proposal counts, baseline/best and active/dose rows. Preserve missing values as
unknown. Compact transcription and source SHA: experiments/records/E056-server-
partial-20261111.json. SENSORIMOTOR_EXIT_CODE=2 explicitly supplied; failed skill
qualification, not evidence of a crash. Elapsed3832.163929s=63.8694minutes; stop
time_budget. Candidate180c2b9379ae3fff4b66458a511e044211241d257abdbe9276cbc59424182abf;
resume437b227e... matches E055, original paired parent98ea6655... and
anchor417254ad... retained as reported. Frozen fields unchanged=True;7848
coordinates changed versus original paired parent,6574 this continuation.
These counts are not bound occupancy or proof of the cause of improvement.

Refined bend2.803115366e-5rad=93.44% of3e-5 target, recovery10.85%; shifted
onset2.831803042e-5rad=94.39%, recovery8.26%. AWA responses8.22055/8.22237mV
with small recovery residuals. Refined amplitude+120.02% versus matched E055
refined case; shifted-onset+96.39% versus E055 shifted case. Neither qualifies.
Chemical/held controls exactly zero differential bend and AWA recruitment;
blocked_controls_passed=True. Engineering refinement difference3.68683e-6rad
<7.5e-6; not a certified voltage or biology error bound. Proprioception-blocked
peak2.97083e-5rad remains below target; missing active row prevents paired
interpretation. Sensory multipliers[2.0,.866362744,.768239748]: chemical gain is
AT its original upper bound. This is not NMJ activation saturation; unknown
neural bounds must be inspected before declaring overall parameter constraints.

Decision: retain candidate, first-rung threshold and bounds. Obtain full compact
active/dose/baseline/best/count data before choosing further training. Existing
objective scores active case only and clips amplitude benefit at target: if
active has passed while validation remains below target, investigate robust
selection rather than simply prolonging a capped objective. That condition is
not established by the paste. No new experiment or server launch this turn.

Reporting fix only: add standard-library read-only scripts.report_sensorimotor
and print its compact baseline/best/validation/count/gain report after final JSON
in future logs, preventing required evidence from being lost in tail truncation.
Verified report against completed local E056 smoke and Python compilation. No
numerical model/objective changes; full simulation tests not repeated for a
reporting-only edit. Earlier255-test result remains the numerical-code evidence.

Exact pasted E056 server-tail attachment, parsed summary and local compact-report
verification copied and SHA256-compared under external notebook/evidence/E056/
server-tail-20261111, with manifest. This preserves the reported evidence; it does
not retrieve the original server checkpoint, source capsule or missing stdout.
Cloud synchronization is unverified.

## 2026-11-11 — E056 compact complete report and planned E057 robust cue selection

User reports run sensorimotor-20261111T163646Z-769197:148additional proposals,
global48 ->196, best195, passed=False. Baseline .523381147, best .0369130066
(printed rounded). Active bend1.4301107e-5 ->2.88992698e-5rad (96.3% target),
recovery .1066 ->.0632. Low dose2.50309199e-5rad=83.4% target; high dose passes
at3.21280778e-5rad. Shifted onset/refined remain below target. Full compact
record will reconcile previous partial output; exact floats in compact text
are rounded and must not be represented as full precision checkpoint data.

Decision: select next candidates for the worse of nominal and .8dose response
losses. Existing single-cue objective caps amplitude reward at its target, so
nominal success alone could stall robustness; active has not quite reached this
cap yet. This is a proactive curriculum change, not proof of an observed plateau.
E057 planned60min search, seed5700 fresh PCG64 and proposal numbering; WARM START
from selected E056 neural and sensory offsets. Original anchor417254ad...,
selected source180c2b93..., original neural factor2 and sensory .5–2 bounds,
channel/receptor/reporting constants, NMJ/body/physics and .00003rad target
unchanged. Chemical gain already at2x; no bound widening. Train nominal and
.8dose using same-state cue/sham pairs with one shared sham per candidate.
Low dose becomes a training condition, no longer an unseen robustness check.
High dose, onset shift and timestep refinement stay outside selection; all
existing validation gates and blocked controls still required. Minimize maximum
response loss plus unchanged prior, not mean amplitude or stronger odor alone.
Save both training traces and original best/final model lineage externally.

New objective means warm start with fresh optimizer/RNG, explicitly distinct
from E056 same-objective continuation. Preserve E056 RNG and checkpoint rather
than overwrite. Planned server command:
sh scripts/launch_robust_sensorimotor_mbslnx2.sh 60
No server launch; user launches after push. Local one-proposal smoke will use
older local E056 candidate, not the actual180c2b93... server network.

E056 complete compact arithmetic (rounded display values): active amplitude
+102.077%, objective-92.947%, active96.331% of target, weakest low-dose83.436%;
low-dose requires19.852% more amplitude. The nominal cue has not yet reached
the cap; revised selection anticipates the mismatch between a nominal-only
objective and multi-condition qualification rather than claiming a demonstrated
plateau. Reconcile exact tail metadata/hashes with the complete compact record
in experiments/records/E056-server-summary.json. No source revision invented.

E057 warm-start software audit on older local E056 smoke: initial selected
neural/sensory offsets equal parent, initial PCG64 equals fresh seed5700, and
new initial global count0 (parent global2). Nominal response metrics exactly
equal old selected nominal metrics; new low-dose training metrics exactly equal
old low-dose validation metrics. Revised baseline is precisely max of these two
response losses plus the unchanged prior, deliberately different from the old
scalar objective. One proposed update accepted in software smoke; no qualified
skill or actual E056 server model validation inferred. Nominal-only runs retain
their previous objective; mismatched cue-mode resumes are rejected.

E057 verification complete:257 tests passed in92.08s; targeted9 passed in0.93s.
New tests ensure a failed low-dose case remains visible after nominal success
and reject a cue-selection change under same-objective resume (legacy nominal
default stays compatible). Shell syntax/Python compilation passed. One-proposal
local warm-start smoke completed full8-case validation, still unqualified on the
older local model. New robust selected state also passes the completed-search
validator for future same-objective continuation. Baseline/best totals exactly
equal worst training-case loss plus original prior. Saved nominal and low-dose
selected traces, original parent hashes, neural/sensory offsets and fresh RNG
checked. This is not a new E056 server-model result. Full provenance and exact
invocation in experiments/records/E057-local-verification.json and external
source/env capsules; no server run launched. The proposed60min server run
retains original thresholds, independent validation and causal controls.

E057 local evidence copied and all copied run files SHA256-compared under
OneDrive/r-elegans/notebook/evidence/E057;3473manifest entries include recursive
parents, all selected/final networks and neural/sensory state, RNG/history,
training/validation trajectories, previews, source/environment capsules, stdout,
test results and compact E056 server transcription. Local integrity verified;
cloud synchronization and original server checkpoint retrieval unverified.

## 2026-11-11 — E057 robust server result and planned E058 continuation

User-pasted compact stdout and full-precision terminal metadata; server artifacts
not independently retrieved. Run sensorimotor-20261111T181101Z-771391,99
proposals global0 ->99, selected98, elapsed3828.107879s (63.80min), time_budget,
exit2 and passed=False. State hashes and rounded baseline/best/validation metrics
in experiments/records/E057-server-summary.json. Candidate1d864fdb..., sensory
filedebccda0..., warm parent180c2b93..., original paired parent98ea6655... and
anchor417254ad... as reported. Frozen fields unchanged=True.3936coordinates
changed this run,8608 versus original paired parent; counts are not bound usage.

Robust objective .165858002 ->.0207356124. Active3.38929e-5rad now passes,
low-dose2.93875611e-5rad remains below3e-5 with recovery6.67%; high-dose,
shifted-onset and nominal-refined cases pass current gates. Chemical and held
controls give exactly zero cue effect, correctly passing the separate control
criterion even though their ordinary motor-goal passed flags are false.
Nominal refinement difference3.86318e-6rad passes7.5e-6 tolerance. Sensory
multipliers[2.,.862942136,.546141090]; chemical stays at its upper bound,
DVA remains above its .5 lower bound. Proprioception-blocked cue passes too;
this is not evidence of feedback dependence or successful propagating motion.

Low-dose response improved17.405% from2.50309199e-5, reaching97.9585% of target;
it needs2.084% growth (shortfall6.124389e-7rad), not permission to round the gate
down. This gap is smaller than the maximum nominal refinement difference, but
that difference is neither a certified error bound nor a direct low-dose error
estimate. Fine-step low-dose response is not separately measured by current
validation. Keep numerical verification explicit before promoting a near-edge
result; no biological/autonomous claim.

Plan E058:60additional minutes, same worst-of-nominal/.8dose objective, original
3e-5rad target, neural and sensory factor2 bounds, all fixed fields unchanged.
Parent selected checkpoint1d864fdb... plus learned sensory gains; restore final
PCG64 after global99 rather than selected98. Preserve lineage and all
selected/final checkpoints/RNG/source/env/commands externally. This is completed
search continuation, not a fresh seed or mid-simulation resume. Seed5700 history
continues. Existing software already supports this mode; no numerical code or
acceptance criteria changed and no new tests/simulation needed for this metadata
update. Command: sh scripts/continue_sensorimotor_mbslnx2.sh 60. User launches;
no server launch performed, no promotion to next motor skill.

E057 compact server transcription copied and SHA256-compared under external
notebook/evidence/E057/server-summary-20261111, with manifest. This archives
reported summary data, not original server stdout, checkpoints or source
capsules; those remain on mbslnx2. Cloud synchronization unverified.

## 2026-11-11 — E058 first-rung engineering pass; planned E059 resolution assay

User reports sensorimotor-20261111T193810Z-773573, passed=True, exit0,99
additional proposals global99 ->198, selected167; elapsed63.77min (rounded
display). Same robust nominal/.8dose objective. Nominal3.49248654e-5rad, low
dose3.03782455e-5rad now pass, as do high dose, onset shift and nominal refined
case. Recovery/control and current refinement gates pass. This is a valid PASS
of the originally stated first-rung engineering protocol, not autonomous
oscillation, realistic-amplitude locomotion, or biological calibration. Preserve
selected checkpoint6cd7b0fa95c545c354f83334fe32aaf0551b8dd4e3d4e7f98babc944849669bf
and learned sensory gains[1.901149620,.788503654,.603620720] independently.

E059 question: does the near-threshold .8dose response persist under matched
body/neural timestep refinement? Current low-dose margin3.782455e-7rad is much
smaller than nominal full-trajectory coarse/fine difference6.142204e-6rad, which
is not a low-dose error estimate or rigorous numerical error bound. Do not
retroactively revoke E058's defined pass, but assess numerical confidence before
promoting to alternation. No additional fitting or changed target/bounds.

Method: read-only8s cue/sham .8dose at original dt=.002s,brain=.1ms; half
(.001,.05) and quarter(.0005,.025) timesteps, all fixed initial/protocol/medium
settings and learned parameters. Verify full selected network/offsets/gains,
original anchor/readout/anatomy and source SHA; copy recursive inputs outside
Git. Record raw active/sham dense trajectories, original response gates,
successive cue-effect differences and margin-versus-difference diagnostic.
Report all-grid engineering pass separately from numerical margin support;
three grids cannot certify the ODE solution. No periodic input beyond the
original finite ligand exposure, no direct muscle/voltage target. No source
mutations, no optimizer or RNG draws needed. Preserve exact source/env/commands
and hash records externally. Server launcher pins the reported E058 root and
SHA; no latest-checkpoint substitution. Local software smoke may use the older
unqualified E057 model and is labeled separately. Planned command:
sh scripts/assess_sensorimotor_resolution_mbslnx2.sh
Diagnostic exit0 means completion, not biological or automatic promotion. No
server assay launched in this turn. All neuronal/NMJ/body/sensory parameters
fixed; no training seed because no stochastic simulation/proposal is used.

E059 local verification:258 tests passed in96.78s, including new comparison test
that cancels distinct sham postures, aligns nested timestamps, isolates an
injected per-joint response difference and rejects incomplete fine-grid coverage.
Python compilation and launcher shell syntax passed. Full three-grid assay
completed on older UNQUALIFIED local E057 candidate (not server E058); all
three response gates and margin-confidence check are false, as expected for
that model. Source neural/sensory file hashes unchanged. Successive cue-effect
differences9.091e-7 then1.071e-6rad are non-monotonic; do not report certified
convergence from a binary tolerance pass. Exact invocation/results in
experiments/records/E059-local-verification.json, full traces/source capsules
external. The server launcher pins E058 path+6cd7b0fa... SHA and requires its
reported original-stage pass before nohup launch. New output prefix cue-resolution
is deliberately outside sensorimotor-* so diagnostics cannot be mistaken for
training candidates by existing latest-run/resume/report helpers. No model,
training objective, threshold or previous pass changed. Actual E058 numerical
assay remains pending user launch.

E059 local evidence archived and all copied run files SHA256-compared under
OneDrive/r-elegans/notebook/evidence/E059 (3779manifest entries): recursive old
local parent, fixed-grid active/sham traces, actual-pose previews, source/env
capsules, summaries and test stdout. Compact user-reported E058 pass record also
archived; actual server E058 checkpoint not retrieved and cloud sync unverified.

## 2026-11-11 — E059 server resolution result; planned E060 split-step assay

User reports cue-resolution-20261111T213856Z-776777, exit0 (assay completed).
Same pinned E058 winner as launch plan; no checkpoint hash is printed in this
stdout, so identity remains enforced by launcher rather than independently
retrieved artifacts. Low-dose coarse3.0378245549425587e-5rad passes; half
2.7961579055602037e-5 and quarter2.5373644852610955e-5 fail the unchanged
3e-5 target. Recovery .05865/.11925/.16174 remains below.25. All-grid pass=False
and fine-margin support=False. No successive full-trajectory comparison values
were pasted; leave them unknown. Preserve the original E058 engineering pass
as reported but do not promote: numerical sensitivity is materially larger than
its narrow amplitude margin. No evidence yet that quarter resolution has
converged, nor a basis to label the entire signal a numerical artifact.

Plan E060 read-only five-case .8dose cue/sham assay on same6cd7b0fa... winner:
coarse(.002s,.1ms), brain-half(.002,.05), brain-quarter(.002,.025),
body-half(.001,.1), body-quarter(.0005,.1). Brain branch varies neural substeps
at fixed body/coupling step; body branch varies mechanics, transduction sampling
and coupling hold intervals at fixed neural step. Thus this isolates two step
families, not pure mechanics versus all other causes. Compare within each
branch against its own nested predecessor. Keep all parameters, gains,
thresholds, field, initial states and readout fixed. No training, new RNG or
changed solver equations. Preserve recursive selected-parent dependencies, full
source/env/commands and dense traces external. Use separate cue-split-* output
prefix. Exit0 means diagnostic completion; do not assign a biological pass.
Question: which step family changes sensory recruitment/body response most,
and does either branch show stabilizing differences? No automatic cause
classification or promotion; use results to choose integration/precision work
or a stable training grid. Local full assay uses older E057 smoke, explicitly
not actual E058. Planned server command:
sh scripts/assess_sensorimotor_steps_mbslnx2.sh
No new server assay or training launched in this turn.

E060 local verification:259 tests passed in91.82s; targeted2 in0.01s. Added
plan test fixes one step family per branch, correct within-branch reference
comparisons, and original matched-grid defaults. Full five-case assay completed
on older unqualified local E057 model, not actual E058. Coarse response metrics
exactly match its earlier E059 coarse result; source checkpoint/sensory hashes
unchanged. On this older model, brain-step peak ratios .9550/.9837 versus
coarse are non-monotonic; body/coupling ratios .99665/1.00078. Body-branch
full-trajectory differences shrink3.285e-7 ->1.328e-7rad, brain differences
8.614e-7 ->1.044e-6 do not. This supports the assay's discriminating value,
not attribution of the actual E058 sensitivity. Script inspection confirms
advance_network uses a frozen-conductance first-order exponential update;
truncation bias, float precision and nonlinear trajectory sensitivity remain
untested mechanisms until actual-source diagnosis. No neuronal/physical model
changes, additional fitting or actual server assay. See E060-local-verification
record for exact command, all metrics, hashes and external artifacts. Python
compilation/shell syntax passed. Only module docstring changed after numerical
smoke; numerical implementation unchanged.

E060 evidence archived and copied run files SHA256-compared under external
notebook/evidence/E060 (3790manifest entries), including recursive older local
parent, all five cue/sham traces/previews, source/env capsules, tests/stdout and
compact E059 server transcription. This verifies local copied evidence, not
retrieval of actual E058/E059 server artifacts or cloud synchronization.

## 2026-11-11 — E060 server split result; planned E061 precision test

User reports cue-split-20261111T224206Z-778307, exit0. Brain-half/quarter fail
at2.796966782e-5/2.537050273e-5rad, ratios .920714/.835154 to coarse. Body-half/
quarter pass at3.037249697e-5/3.037358620e-5, ratios .999811/.999847. Body/coupling
full-trace differences shrink6.499e-8 ->2.902e-8rad; neural differences increase
5.543e-6 ->6.928e-6. This localizes the observed sensitivity to neural stepping
in these tested conditions, not a proof of a particular integration or float
precision bug. Preserve E058 original pass but defer promotion and fitting.
No source hash/coarse row/runtime metadata pasted; do not invent missing fields.

Plan E061: fixed selected E058 model, low dose .8, body/coupling .002s, neural
steps .1/.05/.025ms, evaluate JAX float64 arithmetic versus existing E060 float32
reference. Verify checkpoint and original-anchor offsets under legacy float32
first; then promote stored parameter/readout/model/template values and original
initial states to64, preserving their numeric values. Build body/mapping/field
and initial-state templates in legacy precision before promotion so changed
parameter generation/initialization does not masquerade as accumulation error.
Derived currents/kinetics are reevaluated in64 with the same equations. Host
wet mechanics already uses NumPy float64 in both modes. Record actual neural,
calcium and muscle dtypes; no fitting, parameter changes, RNG draws or promotion.
Use fixed initial states, original thresholds and all original cue timings.
Copy recursive checkpoint and existing E060 reference assay dependencies outside
Git; verify source/ref SHA and configs/grids. Keep float32 reference distinct
from new float64 results, including complete cue/sham traces and comparisons.
If64 changes the step sensitivity, precision is implicated; residual64 variation
still requires integration convergence investigation. Do not declare a solver
fix from this single assay. Source/env/command/hash records external. Planned:
sh scripts/assess_sensorimotor_precision_mbslnx2.sh
Server launcher pins E0586cd7b0fa... and E060 reference path. No actual server
launch. Local software smoke uses older E057/E060 local model, labeled separately.

E061 control amendment before final verification: changing JAX precision can
change comparisons at ligand on/off boundaries because wet-body time was
accumulated in float64 but passed through a float32 transducer previously. A
float64 comparison can distinguish tiny clock drift that float32 rounded away.
Use integer-step timestamps in the float64 assay to preserve the original
float32 cue bins at2/3s, with a fixture test proving that equivalence. Keep this
an opt-in diagnostic timing control; historical/default rollouts unchanged.
The preliminary float64 smoke is retained separately; re-run the final assay
with controlled event timing rather than silently attribute all differences
to neural accumulation. No actual server precision assay has run yet.

E061 final local outcome:261 tests passed in90.47s; four targeted tests passed
in0.17s. Pure promotion test preserves stored float32 values exactly in64 and
keeps indices/names; event-bin fixture matches legacy float32 at2/3s for the
tested body steps. Older two local paired-recording prediction arrays remain
bit-for-bit unchanged under default float32 rollout. Full controlled-clock
float64 three-grid assay completed on older UNQUALIFIED E057 smoke model, not
actual E058. Reloaded voltage/calcium/indicator/muscle arrays are float64; their
initial neural and pose values equal widened reference values exactly. Source
network/sensory hashes unchanged. Within64 peak ratios1.000460/1.000649;
trajectory differences1.123e-7 ->5.755e-8rad roughly halve, unlike older32
brain-step variation. This supports investigating precision on actual E058,
not declaring it fixed or qualified. Old model still fails motor gate.

Exact command/source/env, full case/precision comparisons and test hashes are
in E061-local-verification.json and external run capsules. Preliminary float64
run without canonical event clock remains distinct; final results supersede it
for the planned experiment. No solver equations, fitted parameters, thresholds
or default training precision changed. Additional output/reference path guards
and precision metadata do not alter numerical simulation. Server assay remains
pending the user's pinned launcher command; no new training.

E061 preliminary and final controlled-clock runs archived and copied files
SHA256-compared under notebook/evidence/E061 (15140manifest entries), including
recursive source/reference dependencies, all raw traces, source/env capsules,
legacy default-path proof, tests/stdout and compact E060 server transcription.
Original server E058/E060 files were not retrieved by this Mac; cloud sync is
not independently verified. Local copied evidence is self-contained and hashed.

## 2026-11-11 — E061 server precision result; planned E062 float64 fitting

User reports cue-precision-20261112T004158Z-781583 (UTC-named artifact; recorded
on client date2026-11-11 America/New_York). Float64 low-dose peaks are
2.8567109789101597e-5,2.8567330569093108e-5,2.8567440622559448e-5rad at
.1/.05/.025ms. Ratios1.000007728/1.000011581; successive full-trajectory
cue-effect differences6.09990e-9 ->3.04921e-9rad approximately halve.
All-grid motor pass=False; quarter recovery .0107233. This strongly implicates
float32 arithmetic in prior neural-step sensitivity for this tested model;
remaining float64 step variation is small but not a rigorous global error bound.
Original float32 protocol pass retained, not a numerically robust promotion.

Plan E062:60min fresh seed6200, same nominal/.8dose worst-case response task,
float64 arithmetic/canonical event clock, legacy rounded templates and starting
states preserved exactly as E061. Warm start E0586cd7b0fa... actual stored
parameters, not a newly reconstructed64 version of old32 coordinate transforms.
Keep original coordinate box and prior .001mean offsets², fixed physiology,
NMJ/body/transduction assumptions. Validation bend threshold3e-5 unchanged;
optimize a declared2% training-only margin3.06e-5 to avoid razor-edge selection.
Add independent low-dose half-step and neural-quarter checks to original
validation, preserving original pass history rather than silently redefining it.
Low dose is training; high dose/timing/refinements remain outside selection.

Float64 parameterization: retain immutable promoted operational source network
S and original-coordinate origin x0; decode candidate as apply(S,x-x0), with x
still in original +/-log2 box. This exactly preserves the stored source at
baseline; do not recompute S using64 transforms of the old anchor. The tiny
legacy32 encoding residual relative to the original anchor is fixed and logged,
not a new physical bound relaxation. Sensory coordinates migrate to log of the
old operational float32-rounded gains, then optimize in64 within .5–2; record
this representation migration. Preserve operational reference network/origin
hashes and recursive parent/RNG records. New precision and margin mean a NEW
search, not exact E058 RNG continuation. Support later exact completed64-search
continuations with reference and dtype verification; never silently downcast.

Compare initial64 low-dose amplitude with E061 reference before claiming model
migration fidelity. All source/env/commands, references, selected/final networks,
RNG/state/history and raw traces external. Server launcher pins source6cd7b0fa...
and original E061 reference. Local one-proposal test uses older local model,
not actual E058. Planned command sh scripts/launch_sensorimotor64_mbslnx2.sh 60.
No actual server job launched; no claim of autonomous movement or biological fit.

E062 local verification:265 tests passed in91.26s; targeted9 in1.65s. Tests
verify operational-parent zero delta preserves all stored legacy values exactly
in64, proposal coordinate bounds/frozen fields/electrical symmetry, reference
hash protection, migration baseline mismatches, and rejection of precision or
margin changes under exact continuation. Full one-proposal older-model smoke
completed all10 validation conditions in438.08s; no qualified skill expected
or demonstrated. Proposal rejected; selected0, final1 RNG preserved. Baseline
low-dose1.030689472175661e-5rad exactly equals older E061 float64 coarse peak.
All saved validation voltages float64; operational reference/actual selected
parameters reconstructed exactly via new loader, including64 dtype despite
legacy32 verification context. Encoding residual6.18324e-7 raw units on this
older model is recorded, not silently rounded or an intentional bound change.
The reference low-dose helper was executed against the completed baseline;
optional CLI reference-copy branch added after smoke is tested at helper level.
Actual server launch will check pinned E061 before proposals. Exact source/env/
invocation and selected/final/RNG/reference files external; compact evidence
E062-local-verification.json. No actual E058/E062 server fit launched.

Only numerical precision, explicit operational encoding and declared training
margin change in this new stage; fixed biological assumptions and original
validation target remain. Independent low-dose half-step and neural-quarter
validation are added prospectively, not used to rewrite E058's old pass. Source
float64 cannot be silently downgraded; completed64 continuations preserve
reference/origin hashes, coordinates and PCG64. Prefix sensorimotor64-* keeps
legacy32 selection separate, while compact reports recognize both families.

E062 supplemental verification after the broad suite:targeted9 tests passed
in1.63s with explicit NONZERO updated64 network serialization/restoration,
proving the operational decoder and64 dtype survive beyond the zero-delta
baseline. No numerical implementation changed in this supplement.

E062 local run and supplemental checks archived and copied run files SHA256-
compared under notebook/evidence/E062 (4660manifest entries), including complete
operational references/origins, recursive parent, selected/final networks,
search/RNG history, all10 validation traces/previews, source/env capsules and
compact E061 server transcription. Precision baseline reference summary/config
also copied. This verifies local evidence, not retrieval of actual E058/E061
server artifacts or cloud synchronization. All actual-server source paths
remain pinned and unmodified by these tools.

## 2026-11-11 — E062 float64 first-rung pass; planned E063 alternation pilot

User reports sensorimotor64-20261112T020300Z-784477 (UTC-named; client date
2026-11-11 America/New_York), passed=True,45 proposals global0 ->45, selected41,
73.24min rounded. Nominal/low/high/onset and nominal/low-dose refined and
low-dose neural-quarter cases pass, as do separate blocked controls. Low-dose
3.06561362e-5rad, refined3.06503171e-5 and neural-quarter3.06564917e-5 are all
above unchanged3e-5 target. Preserve16146cc539d47277cb2230e1cfced9a00282d9dd8e4e3b18112fde46515a042b
and sensory gains[2.,.8414655104,.5875771484] independently. This is a small,
numerically checked sensory-evoked bend/recovery primitive, not locomotion or
long autonomous rhythm. High-dose selected-joint recovery .2245 passes .25;
don't infer monotonic joint recruitment from selected peak values alone.

E063 planned exploratory60min next skill: at least two peaks and two troughs in
one joint after a SINGLE finite2–3s odor exposure, observation4–12s. No periodic
stimulus, injected neuronal/muscle pattern, desired frequency or phase. Use
12s trials, float64/canonical clock, full302-neuron topology and frozen NMJ/body/
channel constants. Eligible unknown densities/connections/sensory gains and
original coordinate bounds unchanged; fresh seed6300/global0 from E062 stored
parameters, preserving operational reference/origin. New objective/protocol,
not exact optimizer continuation. H1.1 permits recovery-skill interference:
retain E062; don't require the same trial to both stop and sustain alternation.

Engineering initial target1e-5rad minimum detrended peak-to-peak across two late
windows (2% training-only margin), window-amplitude ratio>=.5, each window at
least one raw turning point, total>=2 raw peaks and>=2 troughs at prominence
.1target. Detrended range supplies dense partial progress; raw extrema and
persistence gates reject static/ramping/single-bend motion and fast decay. Slow
oscillatory ringdown can still qualify: this stage does not establish a limit
cycle. These finite-window criteria can miss slow rhythms, and are not a
biological gait/frequency fit.
Keep actual-body proprioception and AWA recruitment during cue. Score worse
nominal/.8dose case. Require original dose/timing/refined variants and low-dose
refinements to meet NEW alternation gates; held-voltage must not have qualifying
pattern. Chemical block must remove chemical cue recruitment, but spontaneous
patterns may remain: initiation/stop gating is deferred. Analyze sham pattern
separately; short unforced repetitions do not establish long autonomy. Do not
require cue-specific phase shifts or penalize useful spontaneous oscillations.

Record actual post-cue joint dynamics (not cue-minus-sham amplitude) for this
skill, with matched sham for sensory causality/descriptive unforced behavior.
Save all selected/final/RNG/reference/source/env/commands and traces external.
No server training launched. Local numerical smoke will use older local E062,
explicitly not the actual16146cc... winner. Planned command:
sh scripts/launch_alternation64_mbslnx2.sh 60

E062 detailed transcription: experiments/records/E062-server-summary.json.
Actual source revision, full environment, seed confirmation, exact seconds,
sensory-file hash and low-dose full-trajectory comparison details were not
supplied. Do not infer retrieval, backup or exact reproduction from stdout.
Changed neural coordinates2991 this run,9130 versus original paired parent;
gains[2.,.8414655104152531,.5875771484181291]. Nominal refinement maximum
cue-effect difference6.974503366508067e-8rad. High-dose recovery .2245 is close
to .25 limit; chemical gain reaches its existing2x bound. Keep both constraints.

E063 prospective numerical gate: for nominal and low-dose refinements, the
same coarse-selected joint must qualify, minimum-window amplitude difference
<=25% of max(target,coarse amplitude), peak and trough counts each differ<=1.
Also retain the original cue-minus-sham trajectory checks. A joint switch
cannot conceal loss of a coarse qualifying pattern. These are exploratory
finite-window consistency gates, not a phase-error or biological bound.

E063 local verification outcome: full suite267 passed in92.37s before adding
one resume guard test; final targeted10 passed in.57s. Initial extra test had
an unsaved fixture stage and was corrected; no numerical fix was needed.
One-proposal end-to-end run /tmp/r-elegans-E063-alternation-smoke completed in
652.764164082997s with all10 validation cases, blocked controls and three pattern
refinement checks passing. Selected iteration0, global1: the older LOCAL E062
fixture already has this short-window pattern; the proposal was not accepted.
It is NOT the actual server16146... model or a newly learned alternation skill.
Nominal joint1 has21 peaks/21 troughs and .00053071578rad minimum-window range,
window ratio .6225; sham also qualifies, consistent with spontaneous transient
activity. No long persistence conclusion. Refined and neural-quarter amplitude
differences stay inside the prospectively declared25% bounds, counts within1.
Held-voltage detrended range .000044703829rad alone exceeds target but has zero
raw extrema and correctly fails the pattern gate. Chemical block removes AWA
recruitment but leaves spontaneous pattern; this is expected for this protocol.
Saved network/reference/origin, final/selected state, post-rejection RNG,
float64 trajectories and unchanged local parent verified by /tmp/audit_E063.py.
Reporter/title/explicit-target metadata/docstring edits after run start do not
change numerical execution; original run-start source capsule is authoritative,
final reviewed source archived separately. Local record E063-local-verification.
Actual E062 winner alternation remains server-pending. Decision: launch one
independent60min pilot, then review finite-window/refinement evidence before
longer persistence, distributed waves or propulsion. No server job launched.

E063 evidence archived to /Users/carsonc/OneDrive/r-elegans/notebook/evidence/E063:
6167 copied files SHA256-compared, manifest a8bb34480835a2947d4878ee1d1b63d93714d01e4c0f61cb80b25cabf9000a60.
Includes recursive local parents, selected/final/RNG/source/env/invocations,
full stdout/tests/audit and final reviewed source. This is local archive-copy
verification, not actual server retrieval or confirmation of cloud sync.
See E063-evidence-archive.json. Archive confirmation was added after source
capture; it does not change the numerical source or the archived manifest.

## 2026-11-12 — E063 eight-hour server negative outcome

User had extended the planned60min pilot to480min before launching; this
amendment was given in conversation but not entered in the notebook at that
time. This paragraph reconstructs that instruction, not a prospective record.
Requested command: sh scripts/launch_alternation64_mbslnx2.sh 480. Actual argv,
source/environment capsule and seed were not supplied; planned seed6300 is not
independently confirmed. User subsequently reported alternation64-20261112T042107Z-791518:
252 proposals global0->252, selected251,496.43min rounded, time_budget, passedFalse.
Compact transcription: experiments/records/E063-server-summary.json. No actual
server trajectories/checkpoints independently retrieved or copied to OneDrive.
SHA3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b,
warm parent16146cc539d47277cb2230e1cfced9a00282d9dd8e4e3b18112fde46515a042b.
Neural coordinates changed8837 this run,9506 versus original paired parent.

Loss1.63606397 ->1.58909052 (~2.87% improvement), but selected joint has zero
raw peaks and troughs in all10 validation conditions. Unforced pattern fails.
Nominal detrended p2p .000552317988rad (~55.23 times target) is not evidence of
wiggling: removing a linear trend leaves range even for curved monotonic drift.
Window ratio .3644 ->.4114 stays below.5. Inspection of the loss shows the zero-
extrema penalty is flat before the first raw reversal; the small improvement
is consistent with window-ratio reward rather than alternation. This is a
code/stdout-supported hypothesis; without traces we cannot identify cell-level
mechanism, all-joint extrema, saturation, or where rhythms disappear.

Numerical checks: nominal/low matched amplitude errors4.2649e-7rad are below
~1.3808e-4 allowance; neural-quarter low9.4476e-10rad also below allowance;
turn-count consistencyTrue. Pattern checksFalse because fine qualificationFalse,
not because error tolerances are exceeded. Cue-effect trajectory difference
7.563261752996375e-8rad <7.5e-6 passes. Blocked controls pass, AWA recruitment
persists (~8.04mV nominal), but this does not demonstrate a motor rhythm.
Whole-state/long-time convergence and biological fidelity remain unestablished.

Decision: no blind eight-hour continuation. Preserve E062 successful recovery
skill independently; preserve E063 selected/final/RNG/lineage externally on the
server, without claiming backup verification. Next proposed read-only analysis:
compare saved active/sham voltage, NMJ, muscle drive and raw joint traces across
parent/best and resolutions, including per-joint raw extrema. Then consider a
smaller first-reversal objective with dense progress toward reversal, followed
by short independent seed screens on separate physical core groups. Proposed
analysis/training has NOT executed; no new job or numerical changes this turn.
Earlier CPU/GPU PPO benchmarks do not establish scaling of this host-loop fitter.

## 2026-11-12 — Planned E064 read-only E063 signal-chain diagnostic

Question: where do post-cue raw reversals fail along voltage/calcium -> recorded
muscle activation -> signed body command -> filtered drive -> joints? Analyze
existing saved traces, not a new simulation. Pinned E063 source
/home/carsonc/r-elegans-results/alternation64-20261112T042107Z-791518,
checkpoint3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b.
Compare baseline selected0, best selected251 and all saved validation active/sham
conditions, including low-dose refinements and held-voltage. No training, model,
objective, RNG or integration changes; no experiment seeds consumed. Preserve
E062 recovery checkpoint and existing E063 files. The read-only analysis will
hash inputs before/after and save its own source/env/argv outside Git.

Method: per-channel raw extrema with explicit engineering prominence floors,
raw/detrended ranges, raw net drift, positive/negative variation and minimum
window range/ratio over4–12s. Retain every channel's counts, with descriptive
named motor-neuron groups, rather than only the selected joint. No per-edge
NMJ trace was saved: muscle activation is the first available downstream
readout; do not claim direct synaptic localization. Analyze raw activation
near extrema descriptively, not as a biological saturation criterion. Plot raw
voltage, calcium, muscle activation, command, filtered drive and joint signals
with cue and observation intervals. No prescribed frequency/phase.

Software acceptance: synthetic curved monotonic drift must have zero raw
reversals despite nonzero detrended range; a rhythm confined to neural channels
must be distinguished from downstream flat motion; held-voltage residual body
relaxation must not become an oscillation claim. Reject inconsistent times,
nonfinite arrays, mismatched neuron/channel counts, changed checkpoint hash,
missing required traces or output inside source/repo. Full existing local smoke
will exercise saved-file analysis, explicitly NOT the actual server checkpoint.
Scientific output is descriptive; counts/floors cannot prove causal mechanism,
whole-state convergence, autonomous limit cycles or biological fidelity.

Planned command: sh scripts/launch_alternation_diagnostic_mbslnx2.sh
Runtime expected to be much shorter than training (file I/O/statistics only),
not yet measured. No server analysis or new training executed at this entry.

E064 local outcome: final saved-file analysis /tmp/r-elegans-E064-signal-audit-final
completed26 trace analyses over27 unique inputs from older local E063 fixture;
all before/after hashes equal, JAX not imported. Baseline/best local fixture
already has51 voltage,4 calcium,92 muscle,11 command,11 drive and9 joint
channels passing these finite-window reporting gates; held-voltage has zero
at every stage. This verifies detector behavior on a previously oscillatory
LOCAL source; it says nothing new about actual server3fabef... dynamics.
Figures visually inspected: raw oscillations and body drift remain separate
visible signals. Every channel's counts retained, not only selected joint.
Broad suite273 passed in95.78s before extra subprocess guard test/final path-
report changes; final targeted6 passed in1.31s. Tests include curved monotonic
range with zero extrema, neural rhythm without muscle/joint rhythm, invalid
sampling, source inventory/path checks and wrong hash rejection before output.

Two preliminary full saved-file analyses retained. An indirect data-path import
loaded JAX in the first; replaced it with explicit external-output path checks.
Final complete run confirms no JAX imported. No simulation or training ran.
Each run has source/env/invocation; final diagnostic code frozen before final
run. Exact local command and evidence paths in E064-signal-chain-diagnostic.json.
No new fitting objective implemented; await actual server signal-chain evidence.

E064 local evidence archived in OneDrive notebook/evidence/E064:1378 copied
files SHA256-compared;27 source inputs verified against earlier E063 local
archive. Manifest04d96b241d2273ad94b858bf73cde144b0291c820b2b787ac2ca6c71247e3632;
dependency record c68ca4d400a7b522f2a49c4544729663b5bf6025c7cf0fb78ede3f72c640512b.
No actual server artifact retrieval or cloud-sync confirmation. Archive receipt
added after final reviewed-source capture; numerical diagnostic source unchanged.

## 2026-11-12 — E064 server diagnosis: neural rhythm absent in current lineage

User-pasted stdout from alternation-signals-20261112T171621Z-862622,
exit0, input hashes reported unchanged. Complete26 trace count rows and four
named-group rollups transcribed to E064-server-summary.json. Raw transcription
hash recorded there; actual summary/source/trajectories not independently
retrieved. Exact argv/revision/env/runtime not supplied. Expected checkpoint
from pinned launcher3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b;
actual verification SHA was not printed, so it is not independently transcribed.

Every trace has zero qualifying repetitions at all six measured stages.
Both baseline and best have zero reported raw voltage-reversing channels across
302 neurons over4–12s at.1mV reporting floor/.01mV prominence. Baseline has one
calcium and one muscle reversing channel but no repetitions; best has neither.
Baseline/best maximum raw voltage ranges4.339994864919701/5.393828777474768mV,
joint ranges .05142383248289345/.04885415558907638rad: nonzero drift, not
constant silence and not oscillation amplitude. Onset-shift has one command
and one filtered-drive reversal, no repeated pattern. Held-voltage joint traces
have two reversing channels but zero repeats, consistent with passive relaxation
rather than driven oscillation. Other held signals flat apart from command
roundoff9.71445e-17. Proprioceptive blocking changes drift magnitudes but does
not restore a neural or motor pattern.

Interpretation: earliest observed absence of a repeated pattern is already
neural in current E062 baseline and E063 fitted model. The hypothesis that a
working oscillator merely needs more body amplitude is unsupported for this
lineage. Counts are finite-window/floor-dependent, not proof of no smaller or
slower rhythm. Summary alone cannot identify responsible neurons/channels,
prove convergence to an equilibrium, or exclude downstream constraints. The
older local fixture used to test E064 was separately oscillatory and was never
evidence that this actual server parent oscillated.

Decision correction: before a new first-body-reversal objective, recheck the
independently preserved earlier oscillatory parent under identical float64,
initial voltages/states and sensory/body coupling. Do not assume earlier
float32/autonomy results survive this controlled comparison. If reproducible,
use that independent parent for neural-rhythm-first fitting, then motor
alternation. Preserve E062 sensory-response checkpoint and E063 negative run.
No new objective, numerical code, comparison simulation or training executed
this turn. Record proposed comparison as planned before eventual execution.

E064 server transcription archived under notebook/evidence/E064-server-20261112T171621Z-862622:
three copied files SHA256-compared (pasted stdout, parsed compact record and
transcription script). Raw stdout SHAe0687c8d1bbded4d9bb20c7582a387eb8128129e022027991196de01b53a12c3;
manifest ed08c9ec185e54a0deb1290da88028dca87d06f1cf694512932d455f8fde0e22.
This preserves user-reported text only; no actual server trajectories/model
retrieved, and no cloud-sync assertion. No software tests repeated because
this turn changes only records/documentation; JSON/count/diff checks pass.

## 2026-11-12 — Planned E065 rhythm recovery and channel/connection comparison

Question: does the independently preserved E043/44 seed7001 oscillatory parent
retain neural repetition in today's closed-loop float64 setup, and do channel
scales or chemical/electrical connection strengths explain the current E063
model's missing rhythm? No optimization. Primary2x2: old/old, old/new, new/old,
new/new channel scales vs chemical+gap strengths. Old checkpoint SHA417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14;
new E063 SHA3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b.
The new model here is the E063 fitted model, not E062; family contrasts do not
date when suppression arose. E062 sensory skill remains separate and intact.

Precondition: exact canonical neuron order, topology, all fixed cell/channel/
release/initial-voltage/bias fields and network config agree after exact dtype
promotion. Abort instead of averaging/replacing incompatible fixed fields.
Initial local compatibility attempt could not read the former /tmp E063
fixture (temporary files no longer present); no compatibility result obtained.
Use its previously archived copy for local verification. Actual server
compatibility remains unverified until execution. Frozen readout SHA82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.

Plan: each of four hybrids gets60s nominal single2–3s ligand exposure and60s
sham, same E063 sensory gains/maps, wet-substrate body and zero injected bias.
No periodic pacing. All floating legacy template/initial values built once at
float32 then widened; actual stored fitted64 parameters preserved. Same body/
brain steps .002s/.1ms, canonical event clock. Keep actual-body proprioception
active for primary comparisons. Raw4–12s and late-half neural/muscle/command/
drive/joint statistics use E064 explicit floors and raw peak/trough gates;
require >=2 peaks/troughs and stable two-window amplitude to report repetitions.
Neural repetitions and motor repetitions are separate results, not movement or
biological qualification. Do not require every cell or isolated neurons to cycle.

Additional controls: old/old sham with initial voltage +/-1mV (only voltage
shifted at t0; gates/calcium/body/sensory memory held as initialized); old/old
and new/new sham with half neural step; old/old sham with proprioception blocked;
old/old float32 sham with feedback active and blocked.15 independent simulations
in total. Float32 contrasts are current matched pipeline diagnostics, not exact
historical replay. No RNG draws/seeds or trainable parameters.4 concurrent
workers on disjoint4-physical-core groups on mbslnx2, each sequential trajectory;
this does not establish a parallel speedup benchmark. Source input/model hashes
checked before/after; save hybrid checkpoints, initial state, raw traces, all
source/env/argv and compact tables outside Git. Forward inputs self-contained;
original training lineage stays in source run, not an optimizer-resume capsule.

Local software smoke will use actual archived old parent and OLDER LOCAL E063
fixture (not current server3fabef...),12s per case, explicit no-affinity mode.
These cannot establish server rhythm recovery or60s persistence. Tests must
verify family mixing identity, fixed-field mismatch rejection, source hashes,
control inventory/initialization and no fitting. Plan recorded before execution.
Planned server command: sh scripts/launch_rhythm_comparison_mbslnx2.sh
No server simulations or fitting launched at this entry.

E065 prospective clarification before simulation: old/old float32 sham with
proprioception blocked completes the2x2 precision/feedback contrast (15 total
simulations, including the controls already listed above). These are matched current-pipeline comparisons, not historical
replays. Current new/new nominal/sham must also reproduce its saved first12s
traces within per-field roundoff tolerances before interpretation. Check initial
state fingerprints across unperturbed64 cases. Fixed leak scale equality is
required in addition to fixed-tree equality. Preserve self-contained forward
inputs and operational-reference/origin files; original training lineage stays
intact in its external source, without claiming an exact optimizer continuation.

E065 local outcome: archived old E043/44 model and older archived LOCAL E063
fixture have compatible fixed config/tree/leak/readout; frozen input copies
hash-verified.15-case12s local cohort /tmp/r-elegans-E065-local-cohort completed
successfully, queue elapsed131.7626327920002s (excludes preparation; no CPU
scaling claim because explicit no-affinity mode). All unperturbed64 state
fingerprints match. Post-run audit verifies every evaluated channel/chemical/
gap family exactly matches requested parent, input/source hashes unchanged,
all initial numeric values identical across32/64, and perturbation changes only
one302-value voltage leaf by exactly +/-1mV. Current local-model nominal/sham
first12s match saved fields EXACTLY in all6 signal stages. Local fixture was
already oscillatory; these are software controls, not actual server recovery.

Full279 tests pass in102.74s before startup-guard hardening. During local cohort,
worker x64 reset was moved before model imports and child environment explicitly
set false for legacy-template construction; no observed initial-value changes.
Final targeted11 passed in1.34s. Additional final-source new/new-sham12s worker
with inherited JAX_ENABLE_X64=true again gives exact reference parity, proving
legacy-template construction survives that external setting. Per-process source
capsules retain dirty snapshots; final-source extra worker is authoritative for
that startup correction. Figure visually inspected: no ligand cue is unshaded,
late-half analysis marker at6s for this12s smoke. Numerical integration functions
and default E064 plotting behavior otherwise unchanged.

No actual server3fabef... comparison,60s persistence or new training executed
here. Plan remains15 independent60s cases with4 workers/4 physical cores each.
Preparation verifies actual server compatibility before launching simulations.
Command and full local method/evidence in E065-rhythm-comparison.json. Original
models unchanged; no optimization, RNG draws or periodic pacing. The graphical
preview is an older local model and must not be confused with E063 server data.

E065 local evidence archived in OneDrive notebook/evidence/E065:6716 files
SHA256-compared;10 original frozen source-input files reverified unchanged.
Manifest69293290c5dacf90fb8cd187074b460f0a37230046519e6a03e681e2d8cee8d5.
Contains all local cases/evaluated models/initial states/raw trajectories and
full source/env/argv/stdout/tests/audit; earlier archived training dependencies
remain intact. No actual server3fabef... retrieval/evaluation or cloud-sync
confirmation. Archive receipt added after source capture, without numerical
source changes. Server launch remains for user after push.

## 2026-11-12 — E065 partial server controls reported

User reports rhythm-comparison-20261112T185742Z-865067 footer execution_validTrue,
common_float64_initial_stateTrue, exit0. Three complete rows supplied:
old-old-sham-neural-half, old-old-sham-voltage-minus, old-old-sham-voltage-plus.
Each has54 repeating voltage,3 calcium,92 muscle,11 command,11 filtered-drive
and11 joint channels, including17 numbered body motor names,8 RMD/SMD/SMB
and8 AVA/AVB/AVD/AVE/PVC names, none AWA. Exact names retained in compact record
E065-server-partial-summary.json. Beginning fragment has no case identifier;
not assigned. Requested60s default is known from launcher, actual duration,
source/env/argv and individual primary hybrid rows not supplied in excerpt.
Footer validity is recorded as reported, not substituted for missing rows.

Interpretation: old parent retains finite-window repetitions through smaller
neural step and +/-1mV initial shifts under the planned64 controls. This is
stronger support for the old circuit as an oscillatory starting point; not
proof of limit-cycle convergence, certified numerical robustness or biological
fidelity. Full four-way primary and precision/feedback tables must be read
before attributing suppression to channels or synapses. No new training,
restoration or simulation launched; next action is read existing summary.
Original actual server files not retrieved or independently backed up here.
Records/docs only this turn; JSON/diff checks, no numerical test rerun needed.

## 2026-11-12 — E065 complete family table: density scales suppress rhythms

User supplied all15 compact rows and duration60.0s, execution_validTrue.
Full compact transcription E065-server-summary.json supersedes incomplete
family information in the earlier partial entry, which is preserved. Every
case exit0; new/new cue and sham reference_matchTrue. Metadata/source/trajectories
not independently retrieved. Expected source hashes remain plan-derived;
actual hybrid hash was not printed and remains unknown.

Primary matrix (repeating voltage/muscle/joint counts): old-channel/old-synapse
54/92/11; old-channel/new-synapse37/90/11; new-channel/old-synapse0/0/0;
new-channel/new-synapse0/0/0. Cue and sham each give the same counts. Channels
here means fitted conductance-density log scales; kinetics, leak, capacitance,
release properties and anatomical topology are fixed and compatibility-checked.
Within this controlled setup, density-scale replacement is sufficient to
suppress detected late repetition with either synaptic-strength family; new
chemical+electrical strengths still permit repetition with old channel scales.
This does not identify individual responsible channels, date changes within
the training lineage or establish biological causality. Count changes54->37
must not be interpreted as reduced/increased fidelity without waveform/data
and recruitment evidence.

Old/old controls:64 half-neural-step and +/-1mV initial shifts each54/92/11;
float32 feedback-on54/92/11, feedback-off53/92/11;64 feedback-off53/92/11.
New/new half-neural-step0/0/0. Supports old reference repetitions across tested
settings, not a certified integration bound or asymptotic limit-cycle proof.
Old-new hybrid lacks its own refinement/perturbation controls in this suite.

Decision: retain validated old/old reference and independent old-channel/new-
connection candidate at known case output path. Candidate checkpoint hash must
be read and verified before promotion; do not overwrite original source model
or call this a greedy-search continuation (hybrid has no matching optimizer
state). Next proposed work validates its numerical/initial-state sensitivity,
amplitude/persistence and regional phase relations. Then create an explicitly
initialized independent fitting stage with recovered channel scales held fixed
while optimizing coordination, preserving neural repetition. Broader channel
training can return later with bounds/retention guards. No new comparison,
objective implementation, restore or fitting this turn. E062 remains separate.
Records/docs only; JSON/count/diff checks, no numerical test rerun needed.

## 2026-11-12 — Planned E066 independent old-channel/new-connection validation

Question: does the E065 old-new candidate retain raw motor repetitions under
its own numerical/initial-state controls and longer observation, and what are
its regional amplitudes and phase relations? No fitting or automatic promotion.
Source /home/carsonc/r-elegans-results/rhythm-comparison-20261112T185742Z-865067,
old-new-sham/evaluated_network.npz. Read its actual hash from saved case config
and verify bytes, source execution validity and exact reconstruction from the
pinned old417254ad... channel scales/new3fabef... chemical+gap strengths. Hash
unknown in supplied stdout; never invent it. Preserve source run and original
training lineage externally. Freeze candidate, old/new forward inputs, reference
trace, source config and evaluated hybrid under a new external result directory.

Plan8 float64 cases on4 disjoint4-core workers: sham60s, one cue2–3s60s,
initial voltage +/-1mV60s, neural half-step (.002s/.05ms)60s, body-only half-step
(.001s/.1ms)60s, sham300s, held-voltage60s negative control. All other initial
state values, frozen kinetics/leak/capacitance/topology/NMJ/body, actual-body
proprioception and E063 sensory gains identical to E065. No periodic drive,
parameter optimization or RNG draws. Initial perturbation changes voltage only.
Body-only refinement also refines sensory/body co-simulation, explicitly not
pure mechanics error isolation. Shared baseline initial-state fingerprints
must match source; unperturbed baseline and long-prefix must reproduce saved
first12s exactly within declared E065 per-field tolerance.

Engineering rhythm gate: baseline has at least one repeated named body/head
motor voltage channel and a repeated joint in each tail/middle/head region;
each active validation retains that, including the same strongest baseline
repeating motor cell. Baseline/fine same-cell minimum-window voltage amplitude
relative difference <=20%, median raw-peak frequency difference <=10%; regional
sustained bend amplitude difference <=20% of max(baseline,1e-5rad). Held voltage
must have no qualifying repeated voltage or joint channels. Initial perturbations
and300s run test repetition retention, not phase locking or exact trajectories.
Thresholds are prospective engineering screening gates, not certified error or
biological criteria. All cases validly executed and reproduction/initial-state
checks must pass before interpretation. Keep amplitude qualification separate:
existing historical E037/E044 .003rad median sustained bend in all3 regions is
reported, not assumed from11 repeating joints. No automatic locomotion claim.

Coordination method: raw-supported sustained joint extrema select eligible
joints; use strongest sustained-amplitude joint's median raw peak interval for
frequency, then detrended Fourier phase/coherence and shape PCA. Reject
monotonic drift before spectral interpretation. Report phase modulo period,
PCA and unphased net displacement descriptively; do not label a travelling wave
or useful propulsion from count/phase span alone. Synthetic standing vs travelling
waves and curved drift must be distinguished. Source fields/weights immutable;
all case models, initial states, raw traces, figures, source/env/argv preserved.

Planned server command: sh scripts/launch_hybrid_validation_mbslnx2.sh
Local smoke uses prior E065 LOCAL cohort and shorter12s cases/long case12s,
explicit local-short-source flag, not actual server candidate or300s validation.
No server simulation or fitting launched at this plan entry.

E066 method clarification before server execution: add descriptive raw
turn-to-turn excursion amplitudes alongside the historical sustained-window
ranges. Per joint, median upward/downward opposite-extremum excursions are
computed; their smaller value reports cyclic amplitude, avoiding drift inflation.
Report regional medians and a separately labelled .003rad comparison. This does
not replace the predeclared rhythm/numerical gates. Add tests with a small sine
on slow monotonic posture drift and static curved drift. Initial local cohort
uses original descriptive fields; preserve it, then verify final source and
reanalyze its saved traces externally without changing original outputs.

E066 local software outcome:8-case12s smoke /tmp/r-elegans-E066-local-cohort
completed execution_validTrue, common initial-state fingerprintsTrue; queue
elapsed78.08647320899945s excluding preparation. This is the older LOCAL fitted
fixture, not the actual server candidate. RMDR amplitude/frequency and regional
bend refinement comparisons pass; held-voltage negative control passes. Regional
persistent-tail coverage fails (0 tail qualifying joints; middle2/head1), so
rhythm_validation_passFalse. Historical .003rad regional amplitude checkFalse.
These negative findings were retained, not relaxed;12s startup windows cannot
establish actual server60/300s qualification. No parameter optimization ran.

Full282 tests passed in93.62s before descriptive cycle-excursion additions;
final targeted15 passed in1.22s. Added direct raw opposite-extremum excursion
measure rejects drift inflation and synthetic standing/travelling phase/PCA
tests distinguish shapes. Reanalyzed all8 saved traces to separate external
/tmp/r-elegans-E066-reanalysis without altering original outputs, hashes equal.
Final-source preparation /tmp/r-elegans-E066-release-inputs freezes candidate,
initial state and original manifest/config; final standalone12s worker reproduces
all6 reference fields exactly and matches final reanalysis metrics. Local
candidateSHA9e87f27b78b267ef7a8d1a0beebc39ac996375e6bb8d8c912f70427e7e362e15;
not the unreported actual server candidate hash. Final metadata-only supervisor
footer exposes candidate hash and prevents short smoke being labelled full-
duration validation. No numerical integration or existing E065 defaults changed;
plot previews decimated with explicit stride labels, statistics full-rate.

Source/env/argv per process preserved; full final reviewed source also recorded
separately. Actual server candidate numerical/initial-state validation,300s
persistence, amplitude/coordination and any next fitting parent remain pending
user launch. Original source models, E062 separate recovery skill and old/old
reference unchanged. No automatic promotion or new optimization follows.

E066 local evidence archived under OneDrive notebook/evidence/E066:5179 copied
files SHA256-compared;17 original source input files reverified unchanged.
Manifest08003d5eedfd0040ed05d45ea40f99995e935b1b14c21becd0711eb49b73eaca.
Includes every local model/state/trajectory, reanalysis and final source capsules,
not actual server hybrid retrieval or cloud-sync confirmation. Archive receipt
added after source capture; numerical source unchanged. User launches after push.

## 2026-11-12 — E066 partial server result and planned read-only report audit

User supplies two complete rows (hybrid-voltage-minus/plus), a preceding
unassigned fragment, and footer for hybrid-validation-20261112T202746Z-868392.
Footer execution_validTrue, common initialTrue, candidate
f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e,
rhythm_validation_passFalse, historical amplitudeFalse, duration_protocol_metTrue,
planned-duration rhythm validationFalse, both refinement screensTrue, heldTrue,
exit0. Each complete perturbation row has37 voltage,4 calcium,90 muscle and11
command/drive/joint repeating channels, gated regional medians all0; direct
cyclic excursions tail~4.28e-5,middle~2.66e-4,head~2.89e-4rad. Selected-joint
frequency2.6455026455026447Hz, eligible-shape PC1=1 and phase span null. Do not
infer300s persistence, global standing-wave shape or zero physical motion from
this truncated excerpt. Actual long/baseline/refinement rows are not supplied.

Read-only method audit: sustained_excursions masks amplitudes to0 for failed
raw-extrema/prominence/persistence checks; regional medians can be0 even when
some joints qualify. Its peak prominence depends on total raw late range,
which can hide small cycles on drifting posture. Current regional numerical
check can compare0 with0, so it does not independently establish convergence
of the NONZERO direct cyclic excursion amplitudes. Preserve original failed
screen and its criteria; do not retrospectively relabel it as passed.
PC1=1 over only one eligible joint is trivial, not evidence of whole-body
synchronization; null phase span may mean too few coherent eligible joints.

Plan: add standard-library read-only compact reporter of all8 saved case rows,
per-region coverage, eligible joint IDs, direct cyclic excursions and timings.
Compute descriptive nonzero cycle-amplitude relative differences for neural/
body refinements and long run versus baseline. No new pass threshold or model
change. Reads existing summary only; no simulation, optimizer or RNG. Local
verification uses saved prior E066 reanalysis outputs under a new temporary
summary (explicit older local fixture), not actual server candidate. Original
raw inputs/results remain unchanged. Await full actual table before choosing
body-amplitude fitting; freeze recovered channel scales if subsequent evidence
supports that next stage. Command will be python -m scripts.report_hybrid_validation
with explicit actual run path. No new experiment or fitting executed here.

E066 partial evidence transcribed to E066-server-partial-summary.json. Original
screen remains failed. Read-only reporter implemented: standard library only,
prints all8 case rows including duration/coverage/eligible joints and descriptive
cycle-relative errors, no pass criterion added. Verified on new temporary copy
of prior LOCAL E066 summary with saved reanalysis fields restored; original
summary/traces unchanged. Fixture clearly older12s source, not f024... server
candidate. CLI output /tmp/r-elegans-E066-compact-report.log; fixture/source/env/
argv under /tmp/r-elegans-E066-compact-report-fixture. Zero-baseline guard checked
(undefined relative error, no division by zero); fresh import confirms no JAX.
No integration/physics/objective change or new simulation; no broad numerical
tests repeated for this reporting-only addition. Source and JSON/diff checks.
Actual full table still awaited before promoting candidate or amplitude fitting.

Report audit evidence archived in OneDrive notebook/evidence/E066-report-audit:
345 copied files SHA256-compared, manifest82b8fd16a54ab6bba0a87f9bccb2c29ee78174291fa007b4a03c037017dde4d4.
Contains structured partial server transcription, older local fixture/source/
environment/invocation, initial/final reporter stdout and final standalone
reporter source. Final formatting adds coherent/eligible counts and phase span;
no numerical model or acceptance changes. No actual server trajectories or
complete summary retrieved; no cloud-sync assertion.

## 2026-11-12 — E066 full compact report: persistent neural basis, weak body

User reports full8-case table for hybrid-validation-20261112T202746Z-868392,
candidate f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e,
ordinary60s/long300s, execution_validTrue. Manual compact transcription in
E066-server-summary.json at printed precision. Reporter git pull3187012 is
evidenced; simulation revision, runtime, exact simulation argv and environment
remain unreported. No independent summary/model/trajectory retrieval. Earlier
partial entries remain historical; their missing-table status is superseded.

In the300s case, late150–300s has37 repeating voltage,90 muscle,8 joint channels
by E064 two-window reporting, but no joint passes stricter five-window eligibility
and region coverage0/0/0. Cyclic excursions tail8.78392e-5,middle.000199279,
head.000181941rad remain nonzero. At60s sham they are4.28399e-5,.000266032,
.000288509rad; late windows30–60 versus150–300 differ +105%/-25.1%/-36.9%.
Hence a finite-time endogenous neural/muscle rhythm is demonstrated, but body
amplitude stationarity, usable gait and locomotion are not. Long-case frequency
is not estimated by eligibility-based reporter (None); do not assign2.65Hz to
it from the short run.

True cycle-amplitude refinement errors are small: neural-half .910/.769/.880%,
body-half .793/.228/.283% (T/M/H). This adds meaningful nonzero-amplitude
sensitivity evidence; it is not a certified convergence bound. Selected joint
frequency at60s2.6455026455026447Hz, neural-half2.6595744680850726Hz. All active
60s cases retain37/90/11 count signals, but regional coverage0/0/1, eligible
joint[10], coherence count1. PC1=1 is therefore a one-column PCA, not a global
standing-wave inference. No phase-span/travelling-wave evidence from that set.
Held-voltage signals/cyclic bends all0. Original screening remains failed and
is not retrospectively relabelled. Nonzero cycles and failed window gate are
compatible measurements with different definitions.

Decision: candidate is a useful independent neural-rhythm scaffold, not a
qualified coordinated/body-amplitude/biological policy. Next proposed stage
holds restored channel-density scales fixed while building direct raw cyclic
body amplitude through eligible unknown connections, with actual feedback and
no periodic pacing. Use settled/multiple-window scoring and retain neural-cycle
and long validation guards, because body amplitude is still evolving. Proposed
small first rung tail1e-4,middle/head3e-4rad (not a jump to .003 everywhere).
No new experiment, objective implementation or fitting has run; final trainable
mask, settling protocol, bounds, seeds, command and acceptance must be entered
before actual next execution. Preserve E062 and old/old independent skills.
Records/docs only; no numerical test rerun. JSON/count/diff checks performed.

Complete compact server transcription and transcription script archived under
OneDrive notebook/evidence/E066-server-20261112T202746Z-868392:2 copied files
SHA256-compared, manifest66542904f93664fa9579782faf50e4bb73fca0a1318602d5bf1fe0207a8bf8f8.
This preserves user-reported compact evidence only, not independently retrieved
server trajectories/checkpoint or confirmed cloud synchronization.

## 2026-11-12 — Planned E067 four-hour connections-only cyclic-amplitude run

User authorizes a4h run. Parent is the independent restored-channel hybrid
f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e in
/home/carsonc/r-elegans-results/hybrid-validation-20261112T202746Z-868392/inputs/candidate.npz.
E066 failed regional gate is retained; neural/muscle repetition at300s supports
an amplitude-building attempt, not a coordinated locomotion claim.

Question: can existing chemical+electrical strengths increase genuine cyclic
bends while fixed old channel scales retain the oscillator? Full302 neurons
and95 muscles active. Train only3709 existing chemical strengths and1093
symmetric electrical pairs; no channels, gains, topology, release/kinetic/leak/
capacitance/reversal, NMJ, muscle, medium or injected-bias parameters change.
Use actual-body proprioception and sham world, no cue pacing or target waveform.
Fresh independent seeds6700..6703, four concurrent disjoint4-physical-core workers.
240min per-worker baseline/search budget, full validation follows (wall time can
exceed4h). This is a fresh optimization stage, not an E066 optimizer resume.

Primary objective: tail1e-4,middle/head3e-4rad direct cyclic excursion, measured
in actual30–45s and45–60s windows from a60s standard-initial-state rollout.
Use smaller median upward/downward opposite-extremum excursions, region medians;
score worst/mean log target deficits over both windows, plus small change prior.
These windows are not claimed fully settled;300s validation tests later behavior.
No state reconstructed from incomplete raw traces and no hidden warm restart.
Keep selected common motor neuron from baseline/300s source (normally RMDR)
repeating in both windows, with >=3 peaks/troughs, >=.1mV and >=half baseline
excursion and broad .5..2x baseline frequency guard; max absolute joint<.5rad.
No requirement that all cells oscillate. Reject lost-rhythm candidates outright.

Use sparse/coherent seeded connection proposals, all existing edges eligible,
with optional anatomical motor/premotor focus, initial log step.025. Original
E043 anchor417254ad... supplies effective-strength factor2 bounds, not a fresh
factor2 around the hybrid. Preserve exact operational hybrid at zero delta;
derive and save its global effective-strength coordinates. Abort inherited
bound excess >2e-6 log units; any smaller representational excess is recorded
and preserved as per-coordinate inherited slack, not silently clipped. Frozen
fields checked and all selected/final models/RNG/coordinates/reference saved.

Acceptance: both training windows meet all three small amplitude targets with
motor guard intact, plus60s initial +/-1mV, neural-half, body-half and300s no-
drive validation. Long score windows150–225 and225–300s use same targets/guard.
Numerical validation compares direct cycle amplitudes (20%) and reference motor
frequency (10%) to60s baseline, with meaningful nonzero fields. Held voltage
must have zero repeated voltage/joint signals and no qualifying cycle amplitude.
No automatic promotion to locomotion/biology or downstream RL. Original E062,
old/old and restored hybrid checkpoints remain untouched and independent.

Every source/preparation/worker has exact source/env/argv; preserve immutable
input hashes, references/origins/bounds, selected/final models and post-rejection
RNG, stdout and validation trajectories outside Git. Local end-to-end smoke
uses older archived LOCAL E066 fixture, short explicit software-only durations,
never actual server parent or a4h claim. Record actual outcomes after execution.
Planned server command: sh scripts/launch_connection_amplitude_mbslnx2.sh 240

E067 implementation prepared. User then explicitly requested command within3min
and waived Mac validation. No worm simulation or end-to-end fit validation run
locally. Quick compilation/shell and synthetic scoring/bounds checks performed;
monotonic drift and flat motor voltage cannot pass the target/guard. Server
preflight pins source f024..., compares initial state and full baseline trajectory
before any proposal; all channel/inactive fields checked during decoding and
final saved fixed fields checked. Per-worker source/env/argv/RNG/state/checkpoints
are enabled. Four-hour search followed by full validation; source untouched.
This is prepared/unexecuted evidence, not a successful fit or reproducibility
claim for a server run that has not launched. See E067-connection-amplitude.json.

## 2026-11-12 — E067 completed four-search outcome: small gains, rhythm preserved

User supplied all four per-seed summaries from connection-amplitude-20261112T221938Z-871781.
All passedFalse, channels frozenTrue. Seeds6700/6701/6702/6703 made63/62/62/62
proposals and accepted32/29/35/30:249 total,126 accepted. Entire compact cycle
vectors transcribed to E067-server-summary.json. Actual checkpoint hashes,
window-level losses, search/total runtimes, numerical-criterion flags, source
revision/environment and cohort exit footer not printed; keep unknown. Earlier
status snapshots showed all searches finished and all four on long validation;
subsequent DONE/summary results establish completion, not missing exact timing.

Compared to identical baseline3.8513936356829315e-5/.00024450367971642906/
.0002664478765474626rad, ordinary gains(T/M/H):6700 +7.267/+3.107/+3.776%;
6701 +3.901/-.017/-2.066%;6702 +14.360/+3.481/-3.277%;6703 +14.182/+4.136/+.116%.
6703 ordinary4.39761e-5/.000254617/.000266758rad (~44/85/89% targets). Strongest
ordinary tail is6702;6700 has largest reported long head bend. Do not declare
winner from min-window vectors alone: objective uses full two-window deficits
and prior. All candidates remain independent; none qualified. Long target
misses persist. Never compare E067 min-window long numbers directly to E066
single whole-late-window median as an exact gain; estimator definitions differ.

All six active cases per seed retain neural guard; held voltage has zero cycles
and fails neural guard as intended. Direct half-step amplitude changes computed
from provided vectors are<1% for every region/seed, unlike earlier vacuous
window-gated0 comparisons. Perturbations barely change cycles. Fixed channels
therefore avoided the earlier loss of neural rhythm; body progress remains
small and regional tradeoffs visible. This is neither useful propulsion nor
biological validation. Few62–63 proposals per chain cannot establish a hard
limit of the connectome, parameters, body or original bounds.

Code inspection: channel_bridge computes signed, fan-in-normalized NMJ drive,
then frozen muscle sigmoid; local dorsal and ventral activations are projected
and subtracted for net joint command. These NMJ multipliers, unknown signs,
thresholds/slopes and mechanics were fixed in E067. Possible compression or
antagonist cancellation is a hypothesis, not a measured result from the pasted
amplitudes. Next proposed read-only work compares saved E065 old/old and old/new
traces under E067's SAME windows and quantifies D/V oscillatory components/net
command. Avoid another blind long run before that. Faster evaluations from
settled states would require all gates/calcium/body/sensory/solver continuation
state, not reconstruction from partial dense recordings. No new simulation,
optimizer or parameter changes this turn. Records/JSON/diff checks only.

E067 full compact transcription and method archived under OneDrive
notebook/evidence/E067-server-20261112T221938Z-871781:2 copied files SHA256-
compared, manifest a3304280cd4166a50ab925b0e696a6c24cc3bb21aa1ae85c7ce5bdd13da11613.
This preserves user-reported evidence only; no actual server weights/trajectories
retrieved or cloud sync verified. All original run artifacts remain external.

### E068 — planned read-only saved output-path audit (2026-11-12)

Question: are E067's modest body cycles associated with weak muscle activation,
antagonist cancellation, or attenuation after the net command? Is old/old better
than old/new under the same estimator? These are hypotheses, not findings.
Method: analyze existing E065 old-old/old-new sham, E066 hybrid sham/long and
E067 seeds6700–6703 ordinary/long trajectories. No simulation, optimizer, RNG
or parameter changes; all neuron/channel/connection/readout/body parameters fixed.
Parent inputs: E065 rhythm-comparison-20261112T185742Z-865067; E066 hybrid-
validation-20261112T202746Z-868392 (candidate f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e);
E067 connection-amplitude-20261112T221938Z-871781. Check each declared evaluated/
selected model hash and identical actor hashes; hash every consumed input before/
after. Source/environment/argv capsule enabled. All results remain external.
No fresh random seeds. Server command: `sh scripts/launch_output_audit_mbslnx2.sh`.

Ordinary cycles use exactly E067 windows30–45 and45–60s and the existing raw
extremum estimator. Long uses150–225/225–300s, descriptive drift comparison only.
Reconstruct dorsal/ventral projection and require agreement with recorded net
command within1e-7 absolute (legacy32 interpolation/reduction differences).
Report each stage's raw cycle amplitudes, D/V phase and net/(|D|+|V|) Fourier
coefficient ratio at an observed RMDR frequency, muscle near-extreme fraction,
and frozen sigmoid local derivative. Undefined carrier ratios remain null.
These are descriptive; no new skill or biological acceptance gate. Static
sigmoid derivative and signal attenuation do not establish closed-loop causality.
Per-edge NMJ currents were not recorded: no causal synapse inference planned.
Execution criteria: complete saved inputs, verified hashes/projection, finite
signals, unchanged sources, no simulator imports. Software checks planned:
synthetic cancellation/anti-phase/static/wrong-map examples and analysis of
existing local12s E065/E066 smoke traces, explicitly not actual server evidence.

E068 local software verification, 2026-11-12:2 focused tests passed (synthetic
same-phase cancellation, opposite-phase survival, static nonmotion and wrong-
projection rejection). Shell syntax and diff checks passed. No worm simulation
run on the Mac. Existing local12s E065/E066 traces were analyzed:4 cases, source
hashes unchanged, JAX absent, maximum reconstructed-command error4.82719494e-8,
maximum fixed NMJ activation reconstruction error3.33066907e-16. Input/output
capsules are /tmp/r-elegans-E068-output-audit-transfer-smoke (plus earlier saved-
projection smoke /tmp/r-elegans-E068-output-audit-smoke). These local short sources
have a different newer checkpoint from the actual server and establish software
execution only, not server physiology or candidate ranking.

Initial attempt exposed a reader mistake: E066 cohort summary does not contain
candidate_sha256; corrected to read the verified candidate hash from its input
manifest. Final saved-file checks succeeded after correction. Fixed NMJ drive
is now algebraically reconstructed from recorded voltages, physical release
thresholds/slopes, packaged contact/sign map and frozen actor, with activation
agreement required within1e-7 absolute. This never evolves neuronal dynamics.
Raw per-muscle drive p2p and per-neuron voltage p2p supplement the D/V analysis.
Per-edge currents remain unavailable; static readout reconstruction does not
identify which parameter adjustment would improve closed-loop motion.

Decision: E068 prepared for user launch, actual server execution/results pending.
No model/optimizer changes; no E067 promotion, extra training or biological claim.

E068 local verification outputs archived to OneDrive notebook/evidence/
E068-local-output-audit:718 files copied and individually SHA256-compared;
archive manifest SHA256 c2feb1370d07ea2dd87ce56a5f798d5b0a82478717df6b9c1ac0f1de8c959941.
This includes both successful analysis invocation/source/environment capsules;
referenced input trajectories stay in the existing E065/E066 external archives.
No server files retrieved and no cloud sync status inferred.

### E068 server outcome — user-reported stdout received 2026-11-12 local context

Reported run output-path-audit-20261113T041119Z-939399, summary.json and adjacent
.log under /home/carsonc/r-elegans-results. Run identifier contains a UTC date;
no independent execution timestamp, source revision, environment or backup status
was supplied. OUTPUT_AUDIT_EXIT_CODE=0;12 case lines and final summary path.
This is user-reported server stdout, not retrieved server JSON/trajectories.
Compact vectors and attachment SHA256: experiments/records/E068-server-summary.json.
All underlying checkpoints, trajectories, input hashes and original source/
environment capsules remain on server; no new simulation/training occurred.

Same-estimator30–45/45–60s minimum regional cycles(T/M/H) old/old:
2.47807595e-5/.000198156681/.000123828498rad; old/new and hybrid both
3.85139364e-5/.000244503680/.000266447877rad. Thus hybrid ordinary improvements
are55.419/23.389/115.175% approximately. Prefer hybrid to old/old on THIS measured
regional amplitude comparison, not a general fitness ranking. All four E067
ordinary/long vectors match their previous user-reported values. No full-rung
pass inferred. Different long windows describe drift, not same-window gains.

Last-window Fourier net/(|dorsal|+|ventral|) magnitude ratio at observed RMDR
frequency is ~.95–.98 at joints0–2 and4–7 for hybrid/candidates; joint3~.79–.81.
Final head joint10 is .08566 for hybrid ordinary and .08208–.08844 for E067
ordinary; old/old .16708. Strong cancellation is localized at that final head
joint, not a demonstrated general tail bottleneck. These are single-frequency
amplitude coefficient ratios, not percentages of broadband energy, and do not
establish causal effects of changing one synapse. No imposed carrier was used.

Maximum printed anatomical projection residual4.827194938e-8, below protocol
1e-7. All12 cases reach audit completion, but actual NMJ activation residuals,
input hash lists and fixed-readout identity were not printed in this transcript.
Per-muscle NMJ p2p, activation p2p/extreme fractions/static derivatives and
D/V/net/filter/joint cycles ARE in generated summary but absent from pasted
stdout. Cannot yet identify weak neural/NMJ drive, sigmoid compression, filtering
or mechanics as the remaining cause. Added read-only stdlib reporter to expose
already computed detail; no new experiment, acceptance criteria or optimizer.
Local verification: reporter reads the previously verified E068 local smoke
JSON; synthetic inputs/source simulations were not rerun. This validates report
format only, not the actual server values. No model parameters were changed.

Decision: retain all independent models and inspect full stage amplitudes before
another long fit. Neither blindly training longer nor fitting muscle gains is
justified by these few printed fields alone.

E068 server transcription archived to OneDrive notebook/evidence/
E068-server-20261113T041119Z-939399:2 files SHA256-compared; archive manifest
8e8636ccccaabbb16d6ad16d43dc1d7e65688fb50aec57b43f526ce4b16a1b47.
Original pasted stdout is preserved verbatim; no actual server artifacts
retrieved and no OneDrive cloud synchronization claimed.

### E068 detailed output-stage evidence — user-reported server stdout

Received follow-up reporter output for the SAME run
output-path-audit-20261113T041119Z-939399. Not a new experiment. Software smoke
False; frozen actor SHA82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.
Five cases printed: old/old, hybrid ordinary/long,6703 ordinary/long, two windows
each. Compact stage vectors and attachment SHA:
experiments/records/E068-server-stage-details.json. Full JSON/models/trajectories
not retrieved. Source revision/environment not independently established.

Numerical checks: muscle reconstruction residuals3.33e-16 to4.44e-16;
projection residuals~4.81e-8. All reported min/median/max fractions near muscle
activation extremes are0. For hybrid45–60s the frozen sigmoid's mean local slope
across95 muscles min/median/max=.347534/2.39822/2.49974. Activation raw p2p
min/median/max=4.37084e-5/.00195674/.109812; signed normalized NMJ drive p2p=
1.79384e-5/.00083513/.0824891. This argues against near0/1 saturation or uniformly
flat sigmoid transfer, not against every possible readout limitation.

Hybrid45–60s regional raw cycles(T/M/H): net command .00175222/.0701773/.022169;
filtered activation .000998/.0398816/.0126051; joints4.68955e-5/.000244504/
.000266448rad. Filter/command amplitude ratios~.570/.568/.569. Tail command is
~.02497 times middle (~40x smaller). Active synaptic output is spatially uneven.
6703 increases ordinary tail bend even though its tail command cycle is smaller
than hybrid: .00158766 vs .00175222. Similarly hybrid has smaller tail command
than old/old but larger tail bends. Therefore raw regional signal magnitude
alone is insufficient: spatial pattern, coupling, posture and timing matter.
Do not infer per-joint gain by dividing separately aggregated regional medians.

Large attenuation between normalized filtered activation and angular body motion
is visible, but those have DIFFERENT units. This is neither percent energy loss
nor proof of incorrect mechanics. Hybrid motor reference~2.6455Hz; body temporal
filter/drag/stiffness and spatial forcing are plausible constraints. Static
muscle moment calibration in current source is tied to bending modulus/preferred
curvature, not independently fitted force data (body/calibration.py). Existing
wet-surface closure also has explicit assumptions. No source error established
and no empirical physics parameter changed on the basis of this report.

Decision: no blind long extension or global muscle gain adjustment. Propose a
separate body transfer/torque and timing diagnostic (including slower rhythms),
then train posterior recruitment/coordinated neural timing with mechanics held
fixed unless calibration evidence justifies a change. A body-only diagnostic
would isolate transfer; only a subsequent unforced network rollout could qualify
autonomous movement. This is a proposed next step, not a launched experiment or
new numbered plan. Preserve full-network parents, all candidate states and
existing skills. Local checks this turn only parse evidence/JSON and diff; no
new simulator or tests required for recordkeeping.

Follow-up evidence archived externally at OneDrive notebook/evidence/
E068-server-stage-details-20261113T041119Z-939399:2 files copied and SHA256-
compared; archive manifest08532f04417a5881ac0a23ca50eb7ba4ec90c89a938e8c42a2d2a2143fdc0d92.
This archives pasted stdout and derived summary, not retrieved server artifacts;
cloud sync is not verified.

### E069 — planned saved-pattern body timing/torque diagnostic

User authorized proceeding after E068 interpretation. Question: with the SAME
recorded spatial muscle command and fixed mechanics, do slower playback rates
produce larger body bends? Separate body transfer from neural recruitment.
Parent: E066 hybrid-validation-20261112T202746Z-868392, candidate
f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e; frozen actor
82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.
No trainable parameters, optimizer, neural simulation or random seeds. Body-only
replay is a diagnostic input, not evidence of a new autonomous neural skill.

Planned method: use saved hybrid-sham45–60s joint command and recorded body state
at45s. Preserve piecewise-held source commands and unchanged amplitudes/spatial
pattern; replay speeds1,.5,.25,.125 at bodydt=.002s. Simulated durations15,30,60,
120s respectively; rates are engineering probes, not biological targets. Same
initial q/filter/pose for every case.1x must reproduce original recorded joints
within1e-7rad and filtered activation within1e-10 BEFORE interpreting retiming.
Require unchanged body implementation hashes versus source invocation capsule.
Use exact legacy32 physical body parameters widened to64 and existing1.7% wet-
agar closure. Record full parameter values and assumptions. Include1x/slowest
half-dt checks and held-command control. Four independent CPU workers after
baseline check; no neural/GPU workload. Launch under nohup; no4h search budget.

Scoring: raw-cycle estimator in final two quarter-duration windows, per-joint
and T/M/H amplitudes. Record net/filter/joints and active, elastic, internal-
viscous torques; environmental moment inferred from balance is bookkeeping,
not independent drag validation. Require finite state, no joint-limit clipping,
Newton residual<=1e-8, source hashes unchanged. Half-step cycle differences must
be<=5% with1e-7rad absolute floor per joint for descriptive numerical agreement.
No new physical/biological/locomotion pass. Do not change mechanics or train
neurons solely to exploit a diagnostic. Preserve source trace/model hashes,
segment/initial state, checkpoints and recursive external dependencies; capture
exact source/env/argv and case results. Command planned:
`sh scripts/launch_body_timing_mbslnx2.sh`. Server execution pending.

E069 implementation/local verification: source body-code hashes are checked
against its archived invocation capsule. Saved source15s segment, copied neural/
actor parent files, exact scalar body parameters and initial-state arrays form
an immutable external input capsule. Every worker hashes it before/after;
supervisor also verifies original inputs after completion. Source commands are
held over original intervals; floating-point source clock boundaries are snapped
within1e-7 of an interval index to avoid accidental one-step offsets. Four
single-physical-core slots, single-thread BLAS; baseline completes before other
cases start. CPU-only JAX is used for unchanged legacy32 body-template setup;
body evolution uses the existing NumPy reference solver, with no neuronal update.

Half-step checks were strengthened to require late raw waveform agreement as
well as cycle amplitudes; report number of supported cycles. This prevents an
all-zero cycle gate from hiding different posture drift. Focused tests:2 passed,
covering original/slow/refined command indexing, clock roundoff and rejection of
different raw drift despite zero cycle estimates. Shell/diff checks passed.

Final local smoke /tmp/r-elegans-E069-body-timing-verified-smoke used existing
local E066 source (different newer checkpoint from actual server). Only .4s of
source was used: .4s normal, .8s half-speed, .4s refined body replays. No neural
simulation or full-duration Mac run. Normal reproduction maximum q error
8.32667268e-17rad and filter error5.55111512e-17; Newton residual maxima<1e-8.
Refined late raw waveform maximum difference5.86058064e-6rad within5% raw-range
criterion; zero supported cyclic joints in this very short sample, which cannot
establish sustained amplitude. No new biological/autonomous-motion result.
Earlier local smoke capsules retained alongside final verification; no server
execution has occurred. Seven full diagnostic cases remain planned for user
launch. No trained parameters or existing solver implementations changed.

E069 local verification archived at OneDrive notebook/evidence/
E069-local-body-timing:4380 files copied and individually SHA256-compared;
manifest a4738894baa196dd7e614cf5e1c5f252a6a60f3666a3f610b6073078ac255a68.
Includes three local software invocation cohorts and source/environment/input
capsules; referenced older E066 artifacts remain in their existing archives.
No actual server files retrieved, full-duration run or cloud-sync claim.

### E069 server result and E070 planned eight-hour neural timing search

User reported body-timing-20261113T044423Z-940182/summary.json and
BODY_TIMING_EXIT_CODE=0. Seven cases, execution_valid True. Source replay match
numbers were truncated/not supplied; supervisor requires baseline match before
sweep. Ordinary/half/quarter/eighth T/M/H cycles(rad):
4.78720449e-5/.000232994029/.000253594616;
.000249702642/.000739769277/.000717062114;
.000732313664/.001494195497/.001384009528;
.001445459553/.002167606644/.002085724257.
Eighth speed increases T/M/H~30.2/9.30/8.22x. Normal/slow refined cycles agree
closely; both waveform+cycle refinement gates pass with11 supported joints,
maximum waveform differences4.09500778e-6/3.33451527e-6rad. Held command has0
cycles. Newton residual maxima<1e-8. Fraction below drag regularization .556–.623
for rhythmic cases: conclusions depend on this modeled low-speed regime.
This is controlled retimed BODY replay, not autonomous slowing or locomotion.
No actual server artifacts/source revision/environment independently retrieved.

E070 PLAN, user-authorized8h search: fit intrinsic unknown channel densities and
existing neural chemical/electrical strengths in the full302-neuron network to
produce slower autonomous rhythms with useful cyclic body amplitude. Parent is
pinned E066 hybrid f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e;
actor82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0.
Fresh searches, NOT exact optimizer continuation; preserve all independent
parents/E067 candidates. Train nonzero non-LEAK channel-density coordinates and
all existing chemical/gap pairs; factor2 effective bounds from original anchor,
retaining documented inherited connection roundoff slack. Fixed capacitance,
leak, reversals, channel kinetics, release, topology, NMJ, muscle transfer, body
mechanics, sensory maps/gains. No injected bias, ligand cue, clock multiplier,
prescribed motor pattern or external pacing. Actual body proprioception active.

Seeds6800..6803, target reference-motor frequencies .33/.5/.66/1Hz respectively;
engineering probes, not biological targets. Baseline/search480min per worker,
then full seven-case validation (additional wall time). Four workers, disjoint
four-physical-core groups. Standard common source state for every rollout;
float64,60s training, raw cyclic excursions over30–45/45–60s. Preserve baseline
neural oscillation amplitude>=half and regional body cycles>=half, no .5x
baseline FREQUENCY lower bound (would prohibit the requested slowing). Loss:
existing mean+worst regional log deficits to1e-4/3e-4/3e-4rad, plus mean absolute
log frequency error and small parameter prior. Frequency target tolerance25%
for this engineering rung. Raw peaks/troughs guard against monotonic drift.
Bounds/frozen fields and baseline source trajectory are verified. Accepted/final
models, optimizer coordinates, RNG and source/env/argv preserved externally.

Full validation: ordinary, initialvoltage±1mV, half-neural step, half-body step,
300s long, held-voltage control. Require active amplitude/timing guards and
half-step frequency/amplitude consistency; held-voltage has no cycles. No
locomotion, steady-gait or biological promotion. Checkpoint selection remains
training-only; no automatic next-stage training. Command planned:
`sh scripts/launch_neural_timing_mbslnx2.sh 480`. Local verification will be
synthetic/static setup checks; no long Mac neural simulations. Server pending.

E070 plan CORRECTION BEFORE SERVER EXECUTION, user steering: r-elegans must
operate on the true worm timescale. Supersedes the preceding proposed four
reference-neuron target frequencies .33/.5/.66/1Hz. Those were engineering probes
and were never executed. Keep physical simulation seconds, neural milliseconds
and all fixed kinetic/capacitance constants unchanged. No training clock factor.
Source inspection confirms advance_network uses h=dt_s*1000/substeps; changing
integration resolution does not change physical duration. Computation wall time
is separate from simulated animal time.

Primary literature checked: Fang-Yen et al2010, DOI10.1073/pnas.1003016107,
https://pmc.ncbi.nlm.nih.gov/articles/PMC2996704/ Table1/Methods. Reported body-
undulation mean±SEM .30±.02Hz crawling on2% agarose in NGM;1.76±.07Hz swimming
in NGM buffer, N>10 each. Reference values describe those conditions, not a
universal worm frequency. Current1.7% wet-NGM modeled drag is a condition mismatch;
use .30Hz only as a provisional literature benchmark, not an exact calibrated
biological gate. ±25% is an explicitly chosen engineering tolerance, NOT SEM
or a measured inter-animal range. Source temperature not inferred.

ALL four seeds now use the SAME body-rhythm target .30Hz. Loss reads supported
raw JOINT cycles/peak intervals in tail, middle and head, with per-region median
frequencies over both actual15s windows. Quiet or monotonic posture does not
supply a frequency. Reference neuron retains its rhythmic-amplitude guard;
its frequency is NOT forced to equal body frequency. Neurons may have faster
signals while the body undulates more slowly. Half-step validation now checks
body frequency consistency as well as motor/body amplitude. This targets real-
time gait output through parameters rather than slower movie playback. Body-
amplitude floors and fixed physics/readout remain in place. No biological,
wave-propagation or useful propulsion claim from this engineering rung.

Software verification:5 tests passed (2 timing objective/proposal tests including
slow body with a fast neuron;3 existing channel/connection fit tests). Initial
verification command named a nonexistent test file and collected no tests;
corrected to existing files before reporting success. No long Mac neural run.
Static actual local parent decode check: identity unchanged, all fixed fields
unchanged, symmetric effective gap strengths,4814 nonleak channel densities /
3709 chemical strengths /1093 electrical pairs eligible. Every eligible density
changed under a tiny test offset, masked/leak fields unchanged. Static fixture
checkpoint d13901b0c7817deb969d17aab52cde00304072225bf91688f560447fe9c08aa9
is a software artifact ONLY, never a training parent. Inputs and exact static
verification script/capsule are external under /tmp/r-elegans-E070-static-check
and /tmp/r-elegans-E070-setup-inputs. Static decoding does not evolve neurons.
CLI parsing, compilation, shell syntax and diff checks passed. E070 remains
prepared, server execution pending. Launcher links and verifies the completed
same-parent E069 evidence and numerical controls before starting searches.

E070 FINAL PLAN STEERING BEFORE EXECUTION (2026-11-13): user requests swimming
in water first, then explicitly "set up a training run now targeting swimming
at1.8Hz." Supersedes crawling .30Hz and all earlier multi-rate plans, none of
which ran. Do not infer a measured resonance from E069's monotonically rising
response at slower playback: the current body is overdamped and the sweep
shows frequency-dependent attenuation, not a located resonance peak. The user
selected swimming-first; no evolutionary precedence claim is established here.

Actual training medium is now LIQUID (unbounded Newtonian water model,1e-3Pa s,
existing Lighthill local RFT), not retimed wet-agar commands. All seeds6800..6803
fit BODY frequency1.8Hz in physical time. Acceptance frequency tolerance is now
10% engineering tolerance, not25% or the literature SEM. Initial density/neural
connection parameter parent stays E066, all original unknown-density bounds and
fixed kinetics/capacitance/leak/readout/body parameters remain as planned.
Changing the medium changes drag, not the clock. Source wet-agar trajectory
reproduction is checked SEPARATELY against old saved data. New water baseline
sets amplitude-preservation floors, and all seven subsequent validations use
water. Setup retains exact legacy32 templates promoted to64 and records actual
liquid mechanical/muscle parameter scalars. No liquid source reference is
invented; startup tests the unchanged parent in water before proposing weights.

Amplitude target .003rad T/M/H is the existing small whole-body engineering rung,
not a measured animal swimming amplitude. Frequency loss uses supported raw body
joint cycles per region, not forcing neuronal voltages to share a single rate.
Fast neurons with1.8Hz body motion are allowed. Actual body proprioception stays
interleaved; no replayed command, external pacing or clock/capacitance/kinetic
rescaling during fitting. Current water closure differs from the cited thin-
chamber assay; record that condition mismatch and retain biological_fidelity=False.
Targeting swim timing does not itself demonstrate travelling-wave propulsion.

Verification:5 objective/proposal/parameter-fit tests pass for the1.8Hz objective;
static local water/source rollout factories construct successfully, identity
parameter decode/fixed fields and effective-gap symmetry checked. No forward
neural rollout or full Mac training run performed. Source/parameter static
artifact will be archived independently of earlier unexecuted designs. Final
launcher defaults480min baseline/search on four independent four-core workers,
with validation afterward. No server launch by agent. This is the concrete run
for the user to pull and start; recorded exact commands/inputs/RNG/checkpoints
remain enabled. Revised E070 record supersedes earlier design but earlier plans
remain above for scientific history.

E070 final swimming release archived externally: OneDrive notebook/evidence/
E070-swimming-release-20261113,741 files including complete staged dirty-source
snapshot, environment, static verification script/results and compact E069/E070
records; manifest9fffd787924030b18e733b529ed954b236bd9f85c7d9f1ebbb7e7833f0c98296.
Earlier unexecuted draft/static records remain at E070-preparation-20261113,
758 files, manifestbf92593dd889fa77116ac1118799e134896e5b6c489b180683a79cc276b75943.
These preserve preparation evidence, not server training results or cloud-sync
verification. Final target is1.8Hz in water; prior crawling/multi-rate plans were
superseded before launch. No actual server run has been started by the agent.

### E070 swimming search outcome — user-reported server stdout

Attachment SHA256 and exact parsed28 validation rows are in
experiments/records/E070-server-summary.json. No run identifier, source revision,
environment, elapsed timings, baseline-vs-best comparison, checkpoint hashes,
complete guards or parameter-bound occupancy were supplied. Run path stays
unknown; do not fabricate it from other experiment timestamps. All four workers
exit0 and report passedFalse:6800 iter198,6801 iter193,6802 iter197,6803 iter193,
total781. Cohort execution-valid/footer was not printed. No server files retrieved.
E070 planned ordinary/long durations are60/300s; actual duration not in this
excerpt, so refer to the printed ordinary/long case names rather than independently
verified simulated times. No direct parameter-update counts printed.

Ordinary body frequency Hz / regional T/M/H cyclic excursions(rad):
6800:1.86567 / .00216351,.04400890,.01570984;
6801:2.21239 / .00196843,.03722991,.01249750;
6802:1.87970 / .00274598,.03894839,.01335845;
6803:1.74216 / .00212667,.04113992,.01392926.
Seeds6800/6802/6803 are within target1.62–1.98Hz in all printed active cases.
6801 exceeds the band. Every active case in every seed misses tail .003rad;
all middle/head regional amplitudes exceed that engineering threshold. Failure
is therefore expected without relaxing criteria. Full neural/safety/coverage
flags were not printed, so tail failure is demonstrated but not proved the
ONLY failed gate. No weights or criteria modified to turn a failure into a pass.

6802's ordinary tail is91.5326% of target; to reach .003rad requires9.25064%
increase from current amplitude, not merely8.4674% (the deficit as target fraction).
Its frequency is4.4277% above1.8Hz. Strongest reported tail with acceptable timing
makes it a useful next parent candidate, not a proved best total-objective model.
No baseline comparison available: cannot attribute the full water/wet difference
to training or quantify improvement over its untrained water parent.

Numerical descriptions computed directly from printed rows: across all seeds,
half-step cycle differences<.23245% and frequency differences<.75758% relative
to ordinary, small compared with protocol tolerances. Long vs ordinary regional
amplitude differences<1.87e-6 relative. Voltage± perturbation results remain
close to ordinary. Held-voltage body frequencies allNone and excursions all0.
The held control's passedFalse is expected for an inactive negative control;
it is not by itself a failure of control specificity. Reported checks support
persistent cyclic body dynamics at the intended physical timescale under the
implemented unforced water/proprioception protocol. Same regional frequencies
do NOT establish phase progression, wavelength, useful propulsion or swimming
biological fidelity; no trajectory/displacement report was supplied.

Decision: preserve all four full-network checkpoints and source/RNG/optimizer
lineage externally. Favor6802 for a proposed focused tail-amplitude continuation
with a hard swim-timescale guard and retained middle/head activity. Before any
launch, retrieve full source summary/hash/guard/bound details from the server;
no new experiment/optimizer or automatic next stage launched this turn. Current
model/readme/protocol/repertoire statuses updated; earlier failed stages retained.
Local work only parses supplied stdout, derives comparisons, validates JSON
and checks diffs. No simulation or new tests needed for recordkeeping.

E070 reported results archived externally at OneDrive notebook/evidence/
E070-server-user-report:2 files copied and SHA256-compared (verbatim attachment
plus compact parsed summary); manifest05d23abd8222aaab873d3bea6866bbfab7e126c1879b6bec2838802932c2bdd2.
Folder intentionally has no invented run ID. Actual server checkpoints, model
hashes, run capsules and trajectories were not retrieved or backed up by this
step. No OneDrive cloud-sync claim.

### E071 planned — focused tail continuation, FOUR HOURS TOTAL

User explicitly requires setup+search+validation+cleanup to finish within4h,
superseding earlier search-only budgets. Plan: discover the completed E070
seed6802 matching reported tail .00274597939 and1.87969925Hz; verify its exact
checkpoint/selected-coordinate hash, full neural/safety/timing/control/refinement
flags, original input capsule and fixed fields before fitting. No arbitrary run
ID/hash invented. Copy parent selected checkpoint, coordinates, selected RNG,
trajectory, summary/config and full inherited input capsule outside Git. Fresh
optimizer/RNG seeds6900..6903 from that selected model, NOT exact mid-run resume.
All original E070 states remain preserved. Record actual selected source path/hash.

Objective: tail .00315rad training margin, .003rad validation threshold; hard
1.62–1.98Hz body-frequency guard; middle/head retain>=max(.003rad,80% parent).
Keep neural rhythmic amplitude and .5rad posture safety guards. Unknown nonleak
densities and existing neural chemical/gap strengths remain trainable within
original-anchor factor2 bounds. No changes to leak, capacitance, kinetics,
release/topology, NMJ/muscle, water mechanics, sensory gains or physical clock.
Posterior-NMJ-projecting neurons receive proposal emphasis; all eligible neural
coordinates remain possible. Four workers with disjoint four-core groups.

Total monotonic deadline starts in the nohup watchdog before preparation. Reserve
validation using actual E070 total-minus-search duration:>=45min, otherwise
2x measured parent validation+10min; retain additional evaluation/finalization
margin. Search stops before that reserve, including a conservative look-ahead
from measured slowest evaluation. Cooperative deadline checks stop rollouts
between chunks; a process-group watchdog provides a hard ceiling and cleans
up all descendants. Preparation/baseline/source copying/validation count INSIDE
240min. If a required step cannot finish, record incomplete/unqualified with
saved best/final states; never silently skip validation or run overtime.

Validation remains full ordinary, voltage±, half-neural, half-body,300s long,
held-voltage. Report numerical refinement and body timing/amplitude gates.
No propulsion/biological promotion. Exact source/env/argv, selected/final states,
RNG/coordinates and timeout receipts remain external. Planned command:
`sh scripts/launch_swim_continuation_mbslnx2.sh` (240min TOTAL, no extra phase).
Tests planned: deadline scheduling, real subprocess descendant cleanup under
short timeout, tail/timing/coverage guards, static source-coordinate identity.
No long Mac training or forward validation. Server run not launched by agent.

E071 local verification:9 focused/regression tests passed in6.30s, including real
short subprocess tests that verify the descendant started and cannot outlive
both timeout and early leader exit. Search scheduling preserves validation time;
tail scoring rejects improved amplitude outside the swim-frequency band or
with lost middle/head activity. Existing density/gap fit tests also pass.
No long Mac simulation or training. Static continuation fixture exactly rebuilds
its selected-parent parameters from copied coordinates, and an expired deadline
raises BEFORE any neural rollout. Fixture is the earlier local software model,
not the unknown-hash server6802. Exact verification script/capsule under
/tmp/r-elegans-E071-static-check, parent fixtureSHA
 d13901b0c7817deb969d17aab52cde00304072225bf91688f560447fe9c08aa9.
This model is never chosen by the production launcher. Actual server source path
and checkpoint hash will be printed and recorded only when auto-discovery runs.

Watchdog stops coordinator work before4h, with remaining margins for process-
group cleanup and receipt writing. Worker validation has cooperative cutoff and
saves an incomplete summary/missing-case list if it cannot finish. Full source
and coordinate copies, baseline/source reproduction, per-selection model/RNG,
final model and timeout receipt stay external. Four-hour budget is job wall
clock; physical neural/body times are untouched. Existing E067/E070 search-only
launchers retain their defaults; the new E071 entry point enforces TOTAL budget.
Source compilation/shell syntax/diff checks passed. Server run still pending.

E071 preparation archive: OneDrive notebook/evidence/
E071-total-budget-preparation-20261113,1141 files including staged dirty-source
snapshot, environment, static reconstruction/deadline script and input capsule;
manifest78514353b68be9606494a4d944937e77e094dca6139b6e574ca08fd325f1643d.
This is local software/preparation evidence only. Actual E0706802 checkpoint
and new E071 run identity are resolved on the server at launch; neither has
been invented or retrieved here. No OneDrive cloud-sync status claimed.

### E071 server outcome — qualified small water-model rhythm/amplitude skill

User-pasted run swim-continuation-20261113T150202Z-1002622. Budget receipt:
192.91min (printed rounded), all_steps_finishedTrue,validation_completeTrue,
timed_outFalse,exit_code0. Thus total<=240min demonstrated by reported receipt,
with no extra validation phase. Exact elapsed seconds, search/validation split,
iterations, environment/source revision and raw server artifacts not retrieved.
The pasted reporting command has terminal garbling, but final receipt/model/
validation lines are intact. Archive the attachment verbatim; do not reconstruct
the damaged command as an exact invocation. Actual invocation capsules remain
on server. Parsed28 validation rows and evidence SHA are recorded in
experiments/records/E071-server-summary.json.

Reported parent SHA e42a56a329c04d747171470d5d8909fa8db22559da3a52aaeadd61862c351aa8
identifies the E0706802 model selected by the continuation. This adds later
checkpoint evidence to E070; its original cohort run ID is STILL unknown here.
All four E071 workers exit0/statuscompleted.6901 passedTrue,6900/6902/6903
passedFalse. WinnerSHA
8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a;
expected path from implemented layout: <run>/seed-6901/network_best.npz.
No parameter file has been retrieved/copied to Git. Retain all selected/final
checkpoints, RNG, original-anchor masks/bounds, source/env/argv and recursive
input dependencies externally; successful model remains independent under H1.1.

6901 ordinary frequencies all~1.8727Hz (printed4decimals), T/M/H raw cyclic
excursions .003166766375/.038753944498/.013208665027rad. All six active cases
passed target gates; weakest tail is neural-half .003160119242rad,5.3373% above
.003rad. Ordinary tail increased15.32375% vs reported E0706802 .002745979392rad,
with same scoring convention/medium. Relative to ordinary, refined neural/body
frequencies and amplitudes remain close; long tail .003166779514rad stays above
threshold. Held-voltage all frequenciesNone and excursions0; its own positive-
behavior passedFalse is expected, while overall worker qualification isTrue.

Other candidates are preserved as negative outcomes:6900 tail .002889697567rad;
6902 .002886450677rad;6903 .002976334219rad, with6903 refined tail .002964903019rad.
Do not round .002976 to .003 to promote a failure. All have acceptable printed
frequencies but miss tail amplitude. One of four observed searches qualified;
not a statistical estimate of success rate across weight space.

Interpretation: under implemented E071's unchanged clock, real-body feedback,
fixed water/physiology/readout and no external pacing, the reported result meets
the preregistered small engineering rhythm/amplitude rung. Not evidence of
biological channel accuracy, neural causality, phase propagation, wavelength,
useful swimming/forward locomotion or all natural skills. Matching regional
frequencies alone does not establish a travelling wave. The .003rad threshold
is an engineering milestone, not real-animal swimming amplitude. Numerical/
perturbation/held-voltage validation supports this scope without broad robustness
claims. Actual parameter changes/bound occupancy were not printed.

Decision: pin/preserve6901; stop extending tail-amplitude training for this rung.
Next proposed read-only work: inspect the saved shape cycle, head-to-tail phase
and equal-phase displacement for travelling-wave/nonreciprocal propulsion. No
new experiment or training launch this turn; do not silently promote to swimming.
Updated CURRENT_MODEL/README/TRAINING_PROTOCOLS/SKILL_REPERTOIRE from pending to
qualified engineering status; earlier negatives remain. Local verification is
parsing supplied stdout, JSON integrity and diff checks only; no new tests or
simulations required for these record updates.

E071 server evidence archive: OneDrive notebook/evidence/
E071-server-20261113T150202Z-1002622,2 files copied and SHA256-compared (verbatim
pasted transcript including terminal garbling, compact parsed summary);
manifestd1c1272c9d6a1fb94b3d2710a1da514a1873330bceb2145cae990694dd4e4459.
This archives reported evidence, not actual server parameter/trajectory files;
no checkpoint retrieval or OneDrive cloud-sync claim.

### H1.2 / E072–E073 PLAN — controllable actions before propulsion

User requires sensory inputs to decision/control neurons to start and stop the
current behavior, verification before actual swimming, and start/stop training
for every future behavior. Preserve the E071 amplitude/timing winner separately;
that historical engineering pass does NOT establish controllability.

E072 planned sensory-state skill: full302-neuron network, parent E0716901 SHA
8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a at
swim-continuation-20261113T150202Z-1002622/seed-6901. Brief bath cues enter ONLY
existing AWA (diacetyl) and ASE (salt) transduction. Learn start vs stop cue
associations; no asserted innate odor/salt stop reflex. No direct interneuron,
motor or muscle commands, global freeze, desired-state input, 1.8Hz teacher or
clock rescaling. Proprioception and all intrinsic dynamics continue during rest.

Biological sources reviewed: RIS stopping, https://pmc.ncbi.nlm.nih.gov/articles/PMC6736843/;
locomotor initiation circuits, https://pmc.ncbi.nlm.nih.gov/articles/PMC3233480/.
Premotor/sensory-integration roles overlap and are distributed; they are not a
separate hardwired decision-cell class. Candidate groups: AIY/AIB/AIZ/RIA,
RIB/RIM, AVB/PVC and RIS. RIS full GABA/FLP-11 mechanisms are NOT implemented.
Local inherited fixed RIS reversal is exploratory(-21.5778675mV), not a validated
RIS inhibitory-synapse model. Do not silently flip its sign or claim physiological
RIS stopping. Functional learned control must be tested through existing edges
and a candidate-interneuron efferent-block intervention with pre-cue movement
preserved; group labels alone are not a mechanism demonstration.

Nominal90s: initial wiggle30–40s; stop cue40–42; score rest48–58; start cue58–60;
score resumed wiggle68–88 in two10s windows. Rest is low RAW joint range, not
posture-zero or detrended-range cancellation. Active windows retain amplitude
and1.8Hz±10% frequency. Repeat140s, long-rest118s, shifted onset, dose changes,
redundant cues/idempotence, cue-blocked/no-cue and held-voltage controls, half-
timesteps, and candidate-control output blocks. Require at least one controller
block to disrupt cue control while preserving initial movement; no positive
claim if lesions simply paralyze the model. Cues are environmental bath events,
not phase/current/motor targets. Labels are seen only by the objective.

E072 training first emphasizes channel densities in sensory/control cells and
existing edges incident on control interneurons; qualified motor-cell density
settings remain fixed. Original full-network unknown-coordinate factor2 bounds
stay; NMJ/body, leak/capacitance/kinetics/reversals, sensory routing/signs and
receptor kinetics/gains fixed. Full topology remains present. Fresh RNG seeds
7200..7203, selected/final models and coordinate/RNG state external.

E073 planned ONLY AFTER complete E072 validation: train physical forward swimming
with the same sensory start/stop protocol and hard state-control guards. Full
unknown density/connection coordinate space eligible; mechanics/clocks remain
fixed. First forward-speed engineering rung .01BL/s during active episodes,
matched-phase shape closure and positive directed displacement required; this
is not a measured adult swimming-speed target. Stop/start validation reruns on
EVERY new selected swimming checkpoint. A parent gate is not transferable to
changed weights. Preserve both the qualified control parent and any swim child.

Pipeline retains the user's240min TOTAL cap, including source preparation,
training and all required validation. Four search workers, validation cases
parallelized for the selected checkpoint. Deadline/partial outcomes explicit.
No swimming stage starts on a failed/unverified sensory-state gate or inadequate
remaining budget. Exact source/env/argv, input hashes, full selected/final
checkpoints, RNG and recursive lineage external. No server execution yet.

E072/E073 local preparation verification (2026-11-13): 16 focused software tests
passed in6.61s (behavior-control protocol/scoring/causal gates/snapshot tests,
wall budgets, timing helpers and existing sensory transduction). Compilation,
launcher shell syntax and diff whitespace checks passed. No Mac forward-neural
simulation or server run was performed. Static adapter fixture reconstructed its
own parent exactly and instantiated bath/fixed-field overrides; an expired wall
deadline prevented any neural simulation. Fixture SHA
 d13901b0c7817deb969d17aab52cde00304072225bf91688f560447fe9c08aa9
is NOT the server E071 winner. Static mask count:1318 E072 trainable coordinates
of9616 full coordinates; RMD densities excluded. These checks establish software
wiring only. Actual sensory-state control and forward swimming remain pending.

Future rule H1.2 is recorded in AGENTS.md and behavior_control_policy_v1.json:
sensory initiation, sustained action, stop/rest and restart must be trained and
validated on each changed behavior checkpoint. A parent's pass cannot qualify a
child. This user-authorized E072→E073 sequence permits automatic propulsion
training only after the full exact-model control gate; other future transitions
are not implicitly authorized. E072 seeds7200–7203; E073 seeds7300–7303. Command:
`sh scripts/launch_controlled_swimming_mbslnx2.sh` after pulling from the repo.
Raw artifacts: launcher-generated controlled-swimming-<UTC>-<PID> external
results directory. Both phases/all checks share240min TOTAL. Feature-level
refinement includes independent state passes, amplitude/frequency/latency checks;
pointwise periodic phase drift is descriptive. The8/10s transition allowances
and .01BL/s first propulsion target are engineering rungs, not natural reflex or
adult swimming-speed measurements. Full18-case validation includes independent
start/stop omissions and non-paralyzing control-group interventions.

E072/E073 preparation archive: OneDrive notebook/evidence/
E072-E073-control-preparation-20261113,388 files plus manifest, each archived file
SHA256-checked. Manifest SHA256
4bb50cef1cb505bf0ab70446465776d86059811a4b119de105d540c36491e4d8.
Includes the staged dirty-source snapshot, environment/argv and local static
fixture result/check script. Notebook receipt added after source capture; Git
preserves this final receipt. No actual server checkpoint retrieval or cloud-sync
completion is claimed.

### E072 outcome / E073 blocked — user-reported server stdout (2026-11-13)

Run controlled-swimming-20261113T215141Z-18194 under external server results.
Reporter invocation: cd /home/carsonc/github/r-elegans &&
/home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.report_controlled_swimming.
Reported77.6min, process exit0, timed_out=False, all_steps_finished=False,
validation_complete=False. Four worker exits0:7200/7202 each19 proposals,
7201/7203 each18 (74 total). Every worker remained epoch=stop and failed nominal
control. No full control validation or swimming search launched; scientific gate
failure is distinct from process success. No new behavior qualified. Preserve
E071 parent unchanged and every external selected/final E072 model and lineage.

Code inspection: initial control search is allocated60% of remaining time after
reserving two validation suites and600s; control gate failure ends coordinator.
Unused downstream reserve is not reassigned to control. This explains early
termination structurally; exact recorded reserves/worker timing unavailable here.
77.6min is reported elapsed, not evidence of a240min optimization. Four workers
were used concurrently; low proposal count is consistent with costly full
rollouts, but exact per-proposal timing has not been retrieved. Do not infer
biological impossibility, no improvement or a particular failing numerical
constraint from a binary gate report.

Decision: inspect stored best rest-motion/active-state metrics, accepted proposals
and errors before another experiment. Reporter now includes these fields and the
per-stage reserve. Source baseline in control fits is no-cue; best is cued, so
these are not a matched before/after cue-response comparison. Actual checkpoint
hashes, source revision/environment and raw traces remain unreported. E073 is
blocked, not executed. Small structured summary in E072-server-summary.json;
no raw data/weights/stdout added to Git. No new training authorized/launched here.

### E074 planned extended control/swimming run — maximum10h TOTAL

User first requested8h then authorized up to10h. Set launcher600min TOTAL, all
phases/cleanup included. Question: does longer control search reach validated
sensory stopping/rest/restart, enabling subsequent controlled physical swimming?
Parent remains E0716901 SHA8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a.
Fresh searches7200..7203(control),7300..7303(swim); this intentionally restarts
from the qualified motor parent, not a failed-state optimizer resume. Same seeds
can repeat the earlier proposal prefix; no saved E072 candidate improvement is
assumed because detailed results remain unreported. Prior failed outcomes kept.
Fixed/trainable fields and numerical/physical/causal gates unchanged from E072/
E073; no external motor drive, disabled dynamics or softened stopping criterion.
Control1318 coordinates; swim9616 unknown densities/neural connections.

Scheduler correction: control may search until finish minus one full selected-
checkpoint validation reserve and600s reporting margin. A nominal pass stops
search early; successful full validation can release remaining time to swimming.
Even if no nominal pass, validate the best process-successful control candidate
to obtain complete diagnostic cases. No successful worker => explicit failure.
Swimming only after full exact-model control pass and sufficient remaining
validation budget. Hard outer watchdog unchanged; deadline cannot qualify a
partial model. It may finish early if stages complete or cannot proceed safely
within remaining budget. Command after pull:
`sh scripts/launch_controlled_swimming_mbslnx2.sh`. Exact run ID is launcher-
generated, not known here. Full external source/env/argv, lineage hashes, selected/
final models, coordinate/RNG/optimizer state retained. No server run launched.

E072 reporting-only update local verification: module compilation and synthetic
read-only report fixture passed; no simulation. E074 scheduler verification below.

E074 local software verification:13 focused wall-budget/control tests passed
in6.08s, including watchdog descendant cleanup, the additional6h becoming6h of
additional available control search, and retention of validation/report reserves.
Coordinator compilation, launcher shell syntax and diff checks passed. These are
software checks; no new neural forward simulation, learned control or physical
swimming result. Exact source/environment/invocation capture remains enabled in
the server launcher/coordinator/workers. E074 remains planned pending user launch.

### E074 outcome — failed sensory stopping, full control case set reported (2026-11-14)

User-reported server stdout attachment e44a2b3a-e9b0-4f46-a4ee-b5b41ddd9f6f;
SHA256 in experiments/records/E074-server-summary.json. Run external results/
controlled-swimming-20261114T034629Z-30860,559.98min, exit0, no timeout.
Validation reserve50.24min/stage. Selected control fitseed7200, checkpoint
2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f.
Control gate complete=True (all18 case results reported), but positive/control/
sensory-specificity/refinement gatesFalse. Pipeline all_steps_finished and
validation_completeFalse; swimming blocked/not started. Software execution
success does not qualify behavior. Raw server models/trajectories not retrieved;
actual source/env, proposal histories and per-cell responses remain unknown.

Nominal rest max raw joint range.03520628649325283rad vs1e-4rad (~352.06×).
No-cue and chemical-blocked each.0352062644382099rad. Difference in this scalar
range is only2.2055e-8rad; this is NOT a full-waveform or phase-error bound.
Nominal active regional frequencies~1.90114Hz before/after cues. No settled stop
latency; first movement.002s after START is expected from ongoing oscillation,
not evidence of restart. Nominal directed speed~−3.883e-6BL/s: no meaningful
forward swimming demonstrated. Repeated/dose/shift/long-rest fail likewise.
Held-voltage has zero motion as expected, not learned stopping. Neural/body-half
rest ranges.03515895/.03520488rad still fail; refinement_pass=False alone cannot
be read as numerical instability because its gate also requires positive control.

Correction to interpretation/reporting: old control_report listed three causal
groups merely because their output-blocked models moved initially and also failed
control. Since unlesioned control itself fails, this establishes no loss of a
learned function. Require positive control and cue specificity before listing
causal groups. Retain historical reported fields in structured outcome with this
explicit correction; no earlier outcome is silently promoted or removed.
Controller interventions alter movement (sensory-integrator block rest range
.003434rad), supporting a general dynamical influence only, not sensory stopping
or a particular RIS mechanism. Overall control qualification was alreadyFalse;
this correction does not change the failed outcome.

Decision: retain qualified E071 motor parent and every failed E074 selected/final
model and lineage. Do not launch more long fitting yet. Proposed read-only audit:
cue-minus-sham ASE/AWA conductance/voltage, candidate-control responses and
transmission into rhythmic motor cells; inspect fixed excitatory/inhibitory
signs, parameter movement and objective sensitivity, then choose targeted
training changes. Detailed audit protocol/experiment remains to be specified
before execution. No new server run or physiology claim. Current records/docs
updated, historical plan/negative entries preserved. Invocation reported:
/home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.report_controlled_swimming
from /home/carsonc/github/r-elegans.

Local verification of causal-report correction:9 focused behavior-control
software tests passed in2.74s, including a regression case where unlesioned
control fails and no causal groups may be claimed. Structured evidence parsed
all18 cases from supplied stdout; no server forward simulation or new fit.

E074 evidence archive: OneDrive notebook/evidence/
E074-server-20261114T034629Z-30860; verbatim supplied transcript and compact
interpreted summary byte-checked against originals; manifest SHA256
abdfa01602a8ac8ccd17dcd97857094e08c164dbc3e1cb40e4b4d311005b13f0.
No actual server checkpoint/trajectory retrieval or cloud-sync claim.

### E075 PLAN — saved-data cue-to-motor signal-flow audit (2026-11-14)

User approved the short audit after E074's failed stop/control gate. Question:
is there appreciable cue transduction and sensory response; where does the
paired response attenuate before reaching oscillating motor cells? Parent is
E074 selected7200 SHA2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f,
source controlled-swimming-20261114T034629Z-30860. This is deterministic saved-
trajectory analysis, not a forward simulation or fit; no RNG/seeds, all parameters
fixed, original models/trajectories read-only. No new behavior qualification.

Method: hash-check the entire source control_inputs capsule and pinned selected
model; load verified nominal/no-cue/chemical-blocked/stop-blocked/start-blocked
traces on identical grids. Quantify cue-minus-sham voltage/calcium in chemical
sensory cells, AIY/AIB/AIZ/RIA, RIB/RIM, AVB/PVC, RIS, body/head motor groups;
rank uncurated responses too via top20 whole-network cells. Compare muscle,
command/filter/joint outputs across pre, stop, recovery/rest and start windows.
Reconstruct release sigmoids and group-to-group instantaneous inward chemical/
electrical currents at sampled recorded voltages; report topology/strengths and
fixed source reversal potentials. These are descriptive sampled drives, not
measured neural-substep currents, intrinsic channel-current fits or causal
mediation. Full ionic gates were not saved, so do not invent their histories.

Replay only chemical receptor/adaptation equations offline with the preserved
initial sensory memory/config/gains and scheduled bath. Explicitly label this
intended-input reconstruction, not recorded applied conductance; floating-point
event boundaries can differ. No direct neuron inputs added or altered. Validate
selected coordinates/frozen fields/bounds against the original input decoder.
Summarize all four workers' accepted proposals/errors and first/last reported
best stopping metrics. First history row is after first proposal, not an initial
cued baseline. Do not compare no-cue baseline to selected cued result as pure
training improvement. Model/coordinates/origin snapshots, source/env/argv and
input hashes external; preserve the complete source run/recursive dependencies.

Acceptance is data integrity and finite aligned diagnostics; no arbitrary neural
amplitude threshold or biological pass. Require exact source hashes stable across
analysis, coordinate-model equality and unchanged frozen fields/bounds. Command:
`sh scripts/launch_control_signal_audit_mbslnx2.sh`. Dedicated conda interpreter,
CPU0..3 under nohup, timeout20min. Expected lightweight saved-data processing;
actual runtime unknown. External results/control-signal-audit-<UTC>-<PID> path
is launcher-generated; no execution/result has been evidenced yet. Next training
choice depends on these responses and is not automatically launched.

Local software checks:3 tests passed in1.60s: paired-window transient localization,
post/pre matrix orientation/current sign, and offline chemical replay matching
actual transduction memory updates (sensory-only test, no neuron/body dynamics).
A malformed single-timestamp replay is rejected. No Mac neural forward simulation
or server artifact analysis occurred; actual data remain on mbslnx2.

E075 preparation archive: OneDrive notebook/evidence/E075-audit-preparation-20261114,
394 files plus manifest, all archived hashes verified. Manifest SHA256
b5ee8d38368c054e866211b2751f36ea0a3a3d0ca16a6df32f2719697fd1e7fd.
Includes staged source/environment/verification-invocation record; this receipt
was added after capture. Compilation, launcher shell syntax and diff checks also
passed. No cloud-sync completion or server execution is claimed.

### E075 outcome — stop cue sensed, weak observed downstream influence (2026-11-14)

User-reported audit stdout: control-signal-audit-20261114T135353Z-96957, parent
E074 controlled-swimming-20261114T034629Z-30860, checkpoint
2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f.
Audit completion reported; actual elapsed time/source/env/files not retrieved.
Pre-cue voltage difference0 supports the paired initial-condition comparison.
No forward neural simulation or fit, no new behavior qualification. Full parsed
small outcome in E075-server-summary.json; original attachment/hash external.

STOP cue maxima: ASEL.640063mV; AIYL.00772170mV; RIML.00195693mV;
AVBL.00364579mV; RIS.000133542mV; DB07.000744943mV; RMDR.0676000mV.
These are group-wise pointwise cue-minus-sham maxima, not matched-path transfer
gains; groups peak at potentially different times, and phase changes contribute.
AFTER STOP: ASER.470816mV; RIS.000150323mV. During REST: ASER.0322625mV,
RIS.000109058mV. START: AWAL8.36399mV; AIZR.903729mV; AVBL.0318359mV;
RIS.00347536mV; VB01.0100078mV; RMDR.828837mV. AWA/ASE cues use different
ligands/pathways, so their ratio is not a calibrated receptor-gain comparison.
Evidence supports a functioning cue-to-sensory response and small influence on
selected control/body motor cells, not a complete sensory disconnection. It does
not identify a unique faulty synapse or establish which cells can stop the model.

Offline chemical replay (not recorded applied conductance): STOP ASEL peak
.000380235nS; START AWA.00418258nS. Adaptation/rectification produces ASER
response after salt removal. These transients are followed by a target sustained
rest state; that state was not learned. Sampled group net chemical-drive changes
(pA): sensory→integration.000267667; integration→body motor.0000236083;
integration→head.000403301. Direct sensory-group→RIS/forward/locomotor and
RIS→body motor net changes reported0, but zero net difference alone does not
prove zero anatomical edges. Full topology/release profiles remain in server
summary, not supplied here. RIS fixed reversal−21.5778675mV is exploratory;
without operating voltages and mechanism validation it is not proof of inhibitory
control. No unsupported reversal/sign/peptide change made.

Optimizer:143/140/143/140 proposals across7200..7203 (566 total), accepted
68/68/56/56 (248), reported proposal errors0. All remain stop epoch. Selected
7200 first-reported-best rest.03914883224rad to.03520628649rad (~10.07% decrease).
First history row is after first proposal, not initial cued baseline. E074's
almost identical no-cue range means this is not evidence of cue-specific stopping;
it is consistent with merely weakening the ongoing oscillation. Loss5.97251 to
5.86665. Selected995 of1318 eligible coordinates changed out of9616 total;
relative selected-coordinate factors~.633..1.324, one near original bound,
frozen fields preserved. This does not suggest widespread bound saturation;
actual channel physiology and ability to switch the oscillator remain uncertain.

Minor audit reporting bug: seed-* glob also listed seed-*.log regular files as
zero-proposal workers. Those are not additional workers. Correct prospectively
by requiring directories; preserve the raw historical transcript. No selected
model/statistical outcome changes. Decision: retain E071 motor parent, E074
failed variants and full lineages. Propose a short, graded control-access test:
can candidate interneurons suppress ongoing dynamics at all while the remainder
of the model stays active? Any imposed perturbation is diagnostic only, not a
learned-stop success. Specify a new protocol before execution. If control access
exists, train cue-specific suppression relative to matched no-cue and preserve
pre-cue movement, then restart; avoid rewarding only global amplitude reduction.
Do not run another long identical search or widen physiological bounds blindly.
No new perturbation/training run launched in this turn.

E075 server evidence archive: OneDrive notebook/evidence/
E075-server-20261114T135353Z-96957, verbatim supplied stdout and compact interpreted
summary byte-checked. Manifest SHA256
 a88a1e4646a36b940360161ea38b8109a292fa79fe390a79d7e6df94679ed215.
No actual server trajectories/checkpoint retrieval or cloud-sync claim. Local
verification: parsed4 actual workers/4 phases; module compilation and diff checks
passed for the directory-only reporting correction. No new dynamic tests needed
for that filter, no model parameters changed.

### E076 PLAN — graded interneuron control-access screen (2026-11-14)

User approved the proposed short control-access diagnostic. Question: with
E074's parameters unchanged, can candidate interneurons reduce ongoing body
motion reversibly while all other neurons, muscles and feedback remain active?
Parent selected7200 SHA2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f
from controlled-swimming-20261114T034629Z-30860. No parameter training, optimizer
or RNG seeds; this is a deterministic intervention experiment. Preserve E071,
all E074 variants, selected/final/optimizer/RNG and complete recursive lineage.

Method: unchanged no-cue sham plus16 probes, four groups (AIY/AIB/AIZ/RIA;
RIB/RIM; AVB/PVC; RIS), per-cell conductance.01/.1nS, reversal0/−80mV.
These imposed conductances are diagnostic depolarizing/hyperpolarizing hypotheses,
not identified native channels, measured optogenetic strengths or learned cues.
No pacing waveform or clock rescaling. Each60s rollout: assess pre20–30s,
intervene30–40s, score last5s35–40, assess recovery50–60. All normal sensory/
proprioceptive transduction, brain dynamics, NMJs and mechanics remain active;
no voltage freeze or muscle zeroing. Optional diagnostic hook defaultsNone and
is never supplied by behavior trainers. It adds conductance to normal membrane
input, leaving inherited parameters/topology/bias unchanged.

Integrity before interpretation: input capsule hashes, exact global coordinates/
model/fixed reconstruction and stable parent hashes; new ordinary sham must
match archived E074 no-cue voltage/calcium/muscles/filter/joints within1e-8.
Screen pre-probe voltage/joints must match same-grid sham within1e-8 and all
regional baseline cycles>=.003rad. Quiet criterion max raw joint range<=1e-4rad
in35–40s. Recovery regional cycles>=max(.003rad,.8 matched sham) and1.62–1.98Hz
in50–60s. Joint magnitude<.5rad, voltage between−100 and60mV, finite outputs.
Report target voltages versus sham and muscle/command/filter raw ranges so a
stationary body is not assumed to mean all neural oscillators are silenced.
This diagnostic alone cannot establish natural stopping physiology or a learned
sensory-controlled skill. Negative results exclude only tested groups, doses and
timing, not every possible network control mechanism.

Rank safe baseline-matched candidates by during/sham body-motion range ratio;
strongest gets neural-half(.05ms) and body-half(.001s), each with its own sham.
Both refined comparisons must independently pass suppression/recovery and ratio
agreement within max(.02,20% ordinary ratio) for diagnostic-access flag. Always
keep learned_stopping_qualified=False, even if externally imposed probes work.
No automatic fitting/promotion. All17 screen results and4 refinement trajectories
external, with exact source/env/argv, configurations and hashes. Preserve input
source run; no selected/final model file is overwritten.

Command after pull: `sh scripts/launch_control_access_mbslnx2.sh`. Dedicated conda
interpreter under nohup/private-process-group watchdog, four workers x4 physical
cores.45min MAXIMUM TOTAL includes source checks, compilation,17 screen cases,
up to4 refinement cases, reports and cleanup. Reserve15min for refinement; mark
unstarted/failed/time-budget cases explicitly. Child absolute deadlines and outer
watchdog remain active; partial results never certify access. Exact output path
control-access-<UTC>-<PID> unknown until launcher runs. No server execution here.

Local software verification:20 focused tests passed in6.43s: probe strengths/
target/time-window/JIT routing, membrane addition with normal neural/NMJ path
preserved (mocked advance, no forward neural test), matched sham/recovery/quiet
scoring including rejection of fast recovery, existing behavior gates and
wall-watchdog cleanup, chemical replay. Module compilation, launcher shell syntax
and diff checks passed. No new Mac forward neural simulation or claimed server
result. Server sham reconstruction remains a required execution check.

E076 preparation archive: OneDrive notebook/evidence/E076-probe-preparation-20261114,
401 files plus manifest, hashes verified. Manifest SHA256
 d423f759182a5a89b9f587b7bbb7cc81e32c4040a9ff9a66de1456aac0139173.
Staged source/environment/verification-invocation preserved; this archive receipt
added after capture. No server execution or OneDrive cloud-sync completion claim.
