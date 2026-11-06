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
  initiation/termination rules. Direct fine-tuning of the same network is
  another possible route; architecture has not been decided.

The most accurate current description is **curriculum learning with motor-skill
pretraining**. This is not a claim that all successful RL requires pretraining,
nor that motor pretraining alone will resolve interference, objective mismatch,
autonomous rhythm or sensory encoding.

**Project observations motivating H1:** direct food-task training produced no
success in E004; staged supervision subsequently produced driven motion and
stop/restart. These experiments were not a controlled comparison, so they do
not establish that a curriculum causes better downstream RL performance.
E014–E015 also show that new skills can interfere and imitation can fail to
preserve propulsion. Retention must be tested at each step, rather than
assuming progress is monotonic.

**Proposed sequence, not yet tested:**
retain forward/reverse/stop control; add turning in small increments; then
progressively replace external rhythmic pacing with internally generated
motor activity; finally introduce sensory tasks using the established motor
repertoire. Commandable internally generated movement is distinct from
autonomous action selection. Present results remain externally driven.

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
