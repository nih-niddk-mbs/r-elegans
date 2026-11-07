# Current model: exact scope and computation path

This document describes what the repository executes today. It distinguishes
anatomical data, implemented equations, fitted surrogate controllers, and the
full recurrent training path so anatomical availability, trainable models,
and demonstrated behavior are distinguished.

## Executive summary

**Training strategy update, 2026-11-06:** prioritize coverage of independent
[motor skills](SKILL_REPERTOIRE.md), preserving separate full-network policies
where needed. Interference in a shared weight set no longer blocks adding
steering, head sweeps or other skills. Each skill still needs valid physical
effects and a recorded input/parameter contract. Downstream skill selection
and parameter modulation are the intended RL interface; no runtime hierarchy
or autonomous repertoire has been implemented. Earlier consolidation-first
recommendations in the historical training discussion are superseded by H1.1.

The current food-finding baseline has three learned layers:

1. A 14-parameter body gait maps `[speed, steering]` to 95 body-wall muscle
   activations.
2. A supervised motor teacher maps command and gait phase to 302 neuron-indexed
   voltage outputs, then the fixed anatomical neuromuscular projection maps
   those outputs to the same 95 muscles.
3. A seven-parameter engineered sensory controller maps head concentration
   history and gait phase to `[speed, steering]`.

That seven-parameter controller can be fitted two ways: by differentiating
directly through the body/gait simulator (the baseline described below), or
by treating the same simulator as a black-box Gymnax environment and applying
model-free reinforcement learning (see "Body-direct reinforcement learning").
Both produce the identical controller shape and can be evaluated
interchangeably; only the fitting method differs.

The bundled demo retains this baseline. A separate full-network trainer now
puts all 302 recurrent neurons and anatomical NMJs in the control path. It has
passed integration and gradient checks; successful food finding and biological
behavior have not yet been established for this full-network actor.

A real, hand-selected 14-neuron *subset* of that connectome (chemosensory
AWC/ASE through interneurons AIY/AIZ, an RIA integrator, to dorsal/ventral
RMD head-motor readout) has been fitted as a closed loop and can replace the
seven-parameter controller as the body-direct actor -- see "Connectome-
subcircuit chemotaxis" below. This is not the 302-neuron recurrent network
described in the paragraph above; it is a small, literature-guided slice of
it, fitted by supervised imitation and then reinforcement-learning
fine-tuning, with speed control still handled by the same engineered formula
as the analytic controller. It should not be read as evidence that "the
connectome" performs chemotaxis.

## Self-drive and proprioception gap (reviewed 2026-11-07)

The skill drivers are recurrent in neural voltage but open-loop with respect
to the body: they first generate a muscle trace, then simulate its mechanical
effect. Body curvature/stretch is not supplied back to the neurons during
these executions. Body-trajectory gradients used in fitting are different
from runtime proprioceptive feedback. The full actor uses generic neural
dynamics; intrinsic motor-oscillator physiology is not calibrated or validated.
Thus full topology and successful paced movement do not establish endogenous
rhythm. E023 in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) records the biological
evidence and a proposed closed-loop, nonperiodically driven motor test.
That body-feedback loop has not been implemented.

**E024 implementation, 2026-11-07:** an opt-in reduced calcium/potassium
membrane model now gives the 21 DA/VA cells autonomous oscillatory dynamics
and the 18 DB/VB cells conditional excitability. This is a Morris–Lecar-type
demonstration model, not fitted UNC-2 kinetics or a full channel inventory.
Fixed coefficients, baseline A bias, voltage transform, and clock scaling are
explicit in `brain/intrinsic_motor.py`. Isolated A cells sustain oscillations
without periodic input; isolated B cells oscillate under constant current and
settle after withdrawal. Calcium-current removal abolishes the A rhythm.
In the initial connected network coupling suppresses some A rhythms; B cells
remain below the oscillatory regime at zero applied input. No coordinated gait
or AVB-specific enabling mechanism has been demonstrated with this model.

Use `make_full_network_actor_interface(...,
intrinsic_motor=IntrinsicMotorConfig(), input_mode="current")` to opt in.
Carry is `[2, 302]`: normalized voltage and potassium recovery. All anatomical
edges and NMJs remain in the execution path; the other 263 cells retain generic
voltage dynamics. For the 39 selected cells, generic leak-reversal/time-constant
fields are superseded by fixed active-membrane coefficients; generic release
threshold/slope and chemical/electrical strengths remain active. No channel
coefficients were fitted. Carry recovery is actual recurrent state, not an
external phase clock. This model uses RK4 rather than the legacy exponential
integrator and needs new convergence checks for each fitted regime.

Legacy actor NPZ contents/default interfaces and all existing skill trainers
remain unchanged. Old skills do not transfer automatically to changed membrane
dynamics. Intrinsic mode/configuration is **not encoded by the legacy NPZ**:
retain the config alongside any future intrinsic-mode checkpoint and reconstruct
the matching interface explicitly. The E024 assay saves the complete config,
source/environment records and trajectories externally. Run:

```bash
python -m scripts.assay_intrinsic_motor --output-dir /path/outside/repo/intrinsic-assay
```

The assay is a fixed-parameter cellular/full-network smoke experiment, not RL
or a body-motion test. See E024 in the notebook for results and limitations.

## Conductance fitting branch (E025, 2026-11-07)

`scripts/fit_membrane_synapses.py` fits **physical-unit candidates**, separately
from both the normalized full actor and E024's reduced oscillator coefficients.
It never installs them into the actor or modifies published presets.

- The first local VD05 voltage-clamp candidate fits 17 nonnegative maximal
  current conductances using fixed transferred AWCON kinetics, reversal and
  calcium parameters. Four voltage steps train; complete 0 and 30 mV sweeps
  are held out. Training RMSE falls from 59.43 to 11.86 pA; held-out RMSE falls
  from 57.00 to 8.45 pA. Doubling substeps changes predictions by 0.000516 pA RMS.
- These are conditional whole-cell estimates, not a unique channel composition:
  sensitivity rank is 16/17 at numerical precision, practical rank 15/17 at
  relative cutoff 1e-3. In particular, holding-subtracted ohmic leak and NCA
  contributions cannot be separated by this protocol. A good current fit does
  not remove that degeneracy or validate transferred channel kinetics.
- VD05 holding voltage (-60 mV) comes from upstream simulation configuration,
  not independently recovered original experimental metadata. Baseline current
  subtraction is assumed to match padded zero baselines; capacitive/series-
  resistance effects are omitted and pulse edges excluded from fitting.
- Local digitized recordings cover AWA, AWC, AVA, AIY, RIM and VD05, **not
  DA/VA/DB/VB**. AWCL auto-fitting is blocked by disagreement between file
  voltage labels and upstream simulation levels. No motor-oscillator channels
  have been fitted to measurements by this branch.
- A second command fits nonnegative chemical and shared electrical conductances
  to paired voltage/current clamp recordings while intrinsic currents, release
  functions and reversals are fixed. It preserves anatomical masks, holds out
  entire trials and requires explicit units, subtraction provenance and isolation
  for a subset. The CLI recovered known synthetic conductances; **no real paired
  synaptic recordings are present locally and no biological synapses were fitted**.

Install the fitting dependency with `python -m pip install -e '.[fit]'`. Example:

```bash
python -m scripts.fit_membrane_synapses channels \
  --data-root /path/to/external/r-elegans \
  --output-dir /path/to/external/fresh-vd5-fit
```

Synapse input is an external NPZ with `neuron_ids`, `voltage_mV` and
`residual_current_pA` (both `[trial,time,neuron]`), `reversal_mV` (`[post,pre]`),
and per-neuron `threshold_mV`/`slope_mV`. Residual currents must be outward
positive, with intrinsic and capacitive currents already subtracted; fluorescence
is not an interchangeable input. Provide a metadata JSON with `source`,
`voltage_unit: "mV"`, `current_unit: "pA"`,
`current_definition: "outward_intrinsic_and_capacitance_subtracted"`,
`intrinsic_subtraction_source`, and `isolated_subcircuit: true` for subsets:

```bash
python -m scripts.fit_membrane_synapses synapses \
  --recordings /external/paired_clamp.npz --metadata /external/provenance.json \
  --heldout-trial 0 --output-dir /external/fresh-synaptic-fit
```

These inference tools do not make the 302 neurons fully biophysical. Measured
cell fits, behavior-inferred effective parameters, and E024 demonstration
coefficients must retain separate evidence labels and units. Physical conductance
candidates cannot be copied into normalized actor weights without a documented
mapping and independent validation. See E025 for exact inputs and artifacts.

## All-cell channel and calcium branch (E026, 2026-11-07)

`brain/conductance_network.py` now instantiates the existing 17-current library
in **all 302 neurons**, with 35 channel gates, intracellular calcium, and a
saturating low-pass fluorescence-indicator state in every cell. There is no
hard spike threshold/reset and no requirement for a cell to spike. Graded
responses, oscillation and regenerative activity depend on conductances and
inputs. This is an additional inference branch; the preserved motor-skill actor
and bundled demonstration have not been replaced or requalified under it.

All neurons have a channel parameterization, **not an established measured
composition**. RMD cells use the RMD library prior; others use a weak transferred
AWCON prior (1% maximal conductance, 1e-5 nS floor). Leak is initialized at
0.03 nS. AWCON identity is not assigned to both AWC cells as a known phenotype.
The library is incomplete: it does not yet include the EXP-2, UNC-103 and KQT-1
extensions in [Nicoletti et al. 2024](https://doi.org/10.1371/journal.pone.0298105),
nor a complete receptor, mechanosensory, neuromodulatory or ion-homeostasis model.
Effective chemical currents retain parent reversal priors, rather than newly
identified receptor-specific chloride/cation conductances. Published evidence
supports spiking in AWA/AVL/DVB and other distinct regenerative/graded regimes;
no complete experimentally established 302-cell spike/nonspike map is assumed.

**Starting-parameter contract:** the fitter requires a saved full-network actor;
it copies its chemical/gap raw strengths and masks exactly. A migration maps
normalized voltage to `100*v - 25 mV`, synaptic weights to `0.03*softplus(raw) nS`
on the same edges, and passive tau to capacitance. Parent fits can have values
outside plausible membrane ranges, so migration bounds initial/leak voltage to
[-80,-35] mV, effective synapse reversal to [-85,0], release threshold to
[-65,-20] mV, slope to [2,20] mV, and capacitance to [1,20] pF. These are disclosed
initialization assumptions, not experimentally fitted unit conversions. A fixed
bias balances initial intrinsic current only. There is no target-driven current
or learned per-timestep input. The E024 reduced oscillator equations are not
silently combined with the 17-current equations; their phenotypes must be
retested in this branch.

**Recording/fitting path:** `scripts/fit_calcium_network.py` loads a required
preserved parent, an external processed HDF5 calcium recording and NeuroPAL
labels, rejecting ambiguous/noncanonical/duplicate identities. Source label
indices are one-based; observed traces are mapped by canonical neuron name,
not by spatial rank. All 302 evolve; only recorded, confidently identified cells
enter the loss. The tested baseline recording `2022-06-14-01` has 55 included
labels at confidence >=4. Its HDF5 SHA-256 agrees with the source manifest.
Reported GCaMP7f fluorescence is ratiometric, not voltage or absolute calcium.
The Hill/low-pass indicator is an uncalibrated effective observation model,
not a measured nuclear-calcium transport model.

The first assay uses 16 frames (~9.67 s), training on the first 12 and holding
out the last four. Only training frames set fluorescence normalization and
the fixed observation scale/offset. All channel multipliers and existing
chemical/electrical strengths are trainable under priors; masks, channel
kinetics, current biases, release/reversal fields and indicator constants stay
fixed. No behavior, target fluorescence, phase clock, or inferred latent current
is fed to neurons. Missing animal inputs are a limitation. Checkpoint NPZs
store the complete fixed membrane network, fitted parameters, unit config and
canonical neuron ordering; preserve the separately copied NMJ/parent actor too.

```bash
python -m pip install -e '.[fit]'
python -m scripts.fit_calcium_network \
  --recording /external/2022-06-14-01-data.h5 \
  --labels /external/neuropal_label.json.bz2 --uid 2022-06-14-01 \
  --initial-actor /external/preserved-skill/actor_best.npz \
  --output-dir /external/fresh-calcium-fit \
  --frames 16 --iterations 50 --max-step-ms 1 --minutes 15
```

Short-window loss reduction is separate from successful held-out prediction,
unique conductance inference and behavior. The initial 10-update test slightly
reduced training error but worsened held-out error; it did not establish a good
calcium fit. Numerical convergence must be tested in each parameter regime:
frozen-conductance integration is first order, and fast active cells need finer
steps than the older generic membrane model. E026 records complete results and
artifacts; **no fit has been installed as the operating worm**.

The finer 50-update assay (1 ms maximum integration step) reduces training MSE
1.05023 -> 1.03604, but worsens held-out MSE 4.35387 -> 4.37773 and still loses
to a constant training-mean predictor on training (MSE 1). Substep refinement
changes standardized predictions by only 0.0001355 RMS; the poor predictive fit
is not explained by that small numerical difference. Many predicted traces are
nearly flat. All 180 software tests pass, but the candidate is **not a successful
calcium fit** and has not been promoted. Better class/indicator priors and
input/initial-state inference are needed before large fitting runs.

## Status table

Standalone steering development now uses `train_steering.py` (MOT-08).
An inherited forward brain is fitted independently for one signed curvature
bias, with fixed anatomical NMJs and muscle transfer. A differentiated
body-joint/centroid-trajectory loss supplements normalized muscle error;
there is no shared-policy rehearsal loss. Tuning uses a real turning/progress
probe, and final tests require body-axis and centroid-path turning to agree
through all three pacing rates and numerical controls. The original models
are not overwritten. H1.1 permits separate policies for the two signs.
See E019/E020 and [SKILL_REPERTOIRE.md](SKILL_REPERTOIRE.md) for evidence/status.
The server steering run subsequently reported supervisor exit code 0, indicating
all configured workers passed. The provided stdout is a truncated tail;
complete seed-level tables and checkpoint/source verification remain pending.

Local E019 body-trajectory fits now pass for both signs, each from the same
archived 0.06 forward parent with target amplitude raised to 0.07. Each ran
300 updates and selected update 100. Both pass all three pacing rates,
actual centroid/body turn agreement, the unchanged progress floor, timestep
and rest/isotropic controls. This expands driven normalized steering coverage
in two separate policies; it does not establish autonomy, broad angle control,
biological gait calibration, a shared controller or multi-seed replication.

Stationary head-sweep training (MOT-09) is implemented in
`scripts/train_head_sweep.py` using the standalone trainer with a localized
head target and anterior-NMJ-weighted pacing. Relative head/trunk geometry
and quiet-posterior/drift tests prevent global rotation or whole-body bends
from counting as a scan. No locomotion progress floor applies at rest.
Local E021 now passes tuning and physical tests after three 1,000-update stages
with fresh optimizers. Head excursions span 0.1595–0.1750 rad peak-to-peak,
posterior/anterior dynamic RMS ratio is about 0.056, and cycle centroid drift
is <=1.23e-6 body lengths. All three pacing rates and controls pass; fixed
NMJ/muscle fields and topology are unchanged. This is one parent lineage,
with the server cohort prepared but not launched by the agent.

E022 subsequently reports all four server head-sweep seeds passed tuning and
physical tests at all three pacing rates (12/12 conditions). Head-angle
peak-to-peak excursion spans 0.16349–0.18499 rad; posterior/anterior dynamic
RMS ratios are 0.03965–0.04958; maximum cycle drift is 5.124e-6 body lengths.
Selected checkpoints occur at updates 76,750/131,500/16,750/52,750 for seeds
0–3. This is reported replicated driven head scanning; original remote files
and execution provenance still require archive inspection. No autonomous
rhythm, sensory selection or inter-policy transition is established.

| Component | Present | Fitted | Active in food demo |
| --- | --- | --- | --- |
| 302 canonical neuron identifiers | Yes | Not applicable | Used for ordering |
| Chemical and gap-junction topology | Yes | No | No |
| Graded-potential recurrent equations | Yes | No | No |
| AWCON/RMD conductance models | Yes | Published whole-cell fits | No |
| Other neuron conductances | Interface only | No | No |
| 302-channel motor-output teacher | Yes | Supervised | Yes, neural mode |
| 956 neuron-to-muscle edges | Yes | Counts fixed; 880 signs assigned | Yes, neural mode |
| 95-muscle/12-segment body | Yes | SI liquid/agar mechanics; active gait fitted only in legacy normalized mode | Yes, legacy mode |
| Diffusing food field | Yes | Hand-specified normalized parameters | Yes |
| Biological sensory-neuron model | No | No | No |
| Seven-parameter sensory controller | Yes | Differentiable behavior fit | Yes |
| RL policy (analytic controller) | Yes, body-direct only | PPO or A2C (Gymnax) | Optional alternative to the row above |
| 14-neuron chemotaxis subcircuit (real anatomy, hand-selected) | Yes | Supervised pretrain + RL fine-tune | Optional alternative body-direct actor |
| Sensory transduction into the subcircuit | Engineered signal into real neurons | Trainable gains, not biophysical | Yes, when the subcircuit actor is used |
| Full 302-neuron recurrent actor with 95-muscle output | Yes, separate training path | Smoke-tested PPO; behavior unvalidated | No, use full-network trainer |

## Full-network training path

### First supervised motor milestone

`scripts/train_motor_bend.py` fits one small static dorsal bend before any RL.
All 302 neurons evolve, but only neural parameters learn. Anatomical NMJ
strengths/signs, muscle transfer fields and sensory gains remain fixed.
The brain free-runs from its resting voltage with zero external input; the
synthetic target muscles appear only in the loss, never in the forward path.

The target is 40% baseline muscle activation plus/minus a four-percentage-point
smooth dorsal/ventral envelope. A 64-step training sequence is verified over
128 steps through the actual muscle-to-body dynamics. Seed 0, 500 updates on
the Mac, took about 18 seconds: muscle RMSE 0.455 → 0.0099, body-angle relative
RMSE 109% → 12.6%, and longer-hold error 13.7%. The bend gate requires <0.03
muscle RMSE, <20% body/hold relative error, and correct bend direction. These
checks reject straight or oppositely bent bodies. This run established only a
held bend; the following stage adds controlled alternation. Locomotion remains
unproven.

Actor archives, optimizer checkpoints, loss history, provenance and predicted
versus target muscle/body trajectories are stored outside Git. The best actor
is compatible with full-network actor loading. It is not yet a viable
food-seeking initialization: establish controlled bends, a slow traveling wave,
forward/reverse locomotion and turns before progressing to sensory-task RL.

### Controlled alternating bends

`scripts/train_motor_alternation.py` continues from the bend checkpoint, using
slow signed commands (dorsal, neutral, ventral). A fixed anatomy-derived vector
applies external currents only to NMJ-projecting neurons. This is artificial
stimulation for system identification, not a biological command pathway or a
chemical sensory model. All 302 nodes and recurrent edges remain present;
only neural parameters learn. NMJs and muscle transfer fields freeze. There
is no command-to-muscle bypass. Targets are used only for supervised loss.

Training has a sinusoid and a stop/reverse sequence. Validation changes rate,
phase, command levels and order and doubles the horizon. Each sequence must
satisfy muscle RMSE <0.025, body-angle relative RMSE <25%, mean bend correlation
>0.9, and bend-direction agreement >95% during active bends. This rejects
constant outputs and inverted responses. It does not demonstrate an autonomous
rhythm, traveling-wave locomotion, or natural sensory control.

A seed-0 continuation of the held-bend checkpoint passed after 1,200 supervised
updates (about four minutes on the Mac). On the slower sinusoid and unseen
stop/reverse sequence, respectively: muscle RMSE 0.0041/0.0054, body-angle
relative error 16.9%/18.6%, bend correlation 0.997/0.992, and direction agreement
100%/100%. These are synthetic command-following checks for one fitted seed,
not evidence of biological fidelity or ensemble reliability. Longer mbslnx2
runs subsequently completed for all four seeds: each passed both motor stages
after roughly 17,800–18,000 alternating-bend updates. Held-out body-angle error
was 5.1–5.8%, correlation exceeded 0.9998, and active-bend direction agreement
was 100% on both sequences for every seed. These synthetic checks support
advancing to the small traveling-bend stage; they do not establish propulsion.

`scripts/run_motor_curriculum.py` supplies an unattended mbslnx2 CPU ensemble:
four disjoint four-core workers, separate seeds, bend then alternation, with a
wall-clock budget and checkpoints. A failed stage stops that seed's progression;
the runner does not start RL or advance to unimplemented motor stages.

### Slow traveling-bend fitting

The wave stage is `scripts/train_motor_alternation.py --stage wave`. It takes
the same scalar neural stimulation as controlled alternation; only the
supervision target changes. Anatomical muscle rows follow a causal delayed
version of the command, with a default 0.4-second anterior-to-posterior delay
across the body. No delayed input or target array is given to the brain. The
302-cell network must learn spatial timing with fixed NMJs and muscle transfer.
This stage remains externally paced, not an autonomous oscillator.

The existing muscle/body matching gates are augmented with propagation gates:
both muscle and body anterior/posterior bands must have positive lag >=40 ms,
within 60 ms of their respective target lags. Lag is estimated by normalized
overlap correlation over positive and negative shifts, excluding initial
settling. Both changed-rate sinusoids and unseen stop/reverse commands must
pass over a horizon twice as long as training. Uniform and backward waves
fail unit tests. Net propulsion is not a gate yet: it remains a separate
locomotion milestone after a small traveling bend works.

One local seed continued from the Mac alternating-bend checkpoint and passed
after 3,000 wave updates (two 1,500-update fits, about ten minutes total). The
first fit failed the stop/reverse propagation gate; the continuation passed
without changing the criteria. On the two longer validation sequences:
body-angle error 10.7%/16.9%, body-band lag 160/140 ms versus a 140 ms target,
and muscle-band lag 180/100 ms versus a 140 ms target. Both sequences passed
the shape, direction, correlation and propagation gates. These band delays
are distinct from the 400 ms target delay across the entire muscle axis.
This establishes a small externally paced traveling bend for one local seed;
the resumed mbslnx2 ensemble is needed to assess reliability. Autonomous
oscillation, useful propulsion and reverse/turn primitives remain outstanding.

On mbslnx2, `scripts/run_motor_curriculum.py --resume-latest` resumes the newest
passed four-seed alternation run under the selected output root into waves.
Each seed uses its own checkpoint; all source gates and files are checked
before launch. Fresh wave directories preserve earlier artifacts. Explicit
`--resume-run` selects a particular run. No food-seeking RL is launched.

### Locomotion validation and small amplitude steps

`scripts/assess_locomotion.py` freezes the fitted brain and drives the actual
body for four startup plus six measured cycles at periods 2.4, 3.0 and 3.7 s.
Centroid positions are compared at identical command phases. Progress is
projected on the tail-to-head segment-center axis: mechanics' `heading` alone
is the first rod's orientation, not the worm's axis. Shape closure and cycle
consistency distinguish steady strokes from startup deformation.

The assay halves dt from 20 to 10 ms and includes zero-command and isotropic-
drag controls. Detectable translation must exceed 1e-6 body lengths/cycle,
five times control drift and three times timestep discrepancy, with consistent
direction and shape closure. The stronger readiness gate additionally requires
approximately straight travel and >=0.001 body lengths/cycle at every pacing
period. These thresholds are engineering criteria, not biological calibration.

A local seed's 0.04-amplitude wave gives sustained forward progress of
0.00032–0.00042 body lengths/cycle. The rest control stays still, isotropic-drag
drift is <2e-10, and timestep discrepancies are 0.5–1.4%. A supervised amplitude
step to 0.05 (1,000 updates, neural parameters only) retains passed wave gates
and raises progress to 0.00051–0.00066 body lengths/cycle, approximately 59%
more. Shape closes and progress repeats across measured cycles. Neither
amplitude reaches the stronger readiness threshold; natural swimming and
autonomous rhythm have not been established.

The four-seed mbslnx2 0.05 continuation subsequently passed every wave fit and
showed stable, approximately straight forward translation in all three pacing
conditions. Progress was 0.000535–0.000605 body lengths/cycle, 52–67% above each
seed's baseline. All assays completed without errors but remained below the
0.001 readiness threshold; exit code 2 indicated this unmet gate.

A local 0.05 → 0.06 continuation (1,000 supervised updates, same pacing and
0.4-second target delay) passes both longer wave checks: body error 10.6%/17.2%.
Its sustained progress is 0.000749/0.000888/0.000965 body lengths/cycle at periods
2.4/3.0/3.7 s, respectively, another 45–46% increase. Shape and cycle direction
remain stable; rest/isotropic controls are negligible and timestep differences
are 0.4–1.3%. This is one local seed and still below the unchanged readiness
gate. The corresponding four-seed continuation uses the completed 0.05 wave
checkpoints; it does not restart the curriculum or introduce RL.

`run_motor_curriculum.py --resume-latest --assess-locomotion` selects passed
wave checkpoints for a frozen four-seed diagnostic. `--increase-amplitude 0.05`
instead checks baseline propulsion, trains the next small amplitude only when
the baseline is valid, gates the wave fit, then repeats the locomotion assay.
The increase is capped at 30% per stage. Earlier artifacts and successful
intermediate wave checkpoints remain external and available for continuation.
Exit code 2 with completed status and no error means an unmet motor/readiness
gate, not a software crash. No RL or sensory-task reward is introduced.

### Forward readiness and continuous stop/restart

The mbslnx2 0.06 ensemble reached 0.000772–0.000922 body lengths/cycle, with
all wave fits passed. The 0.07 ensemble then improved progress by 29–35%:
seeds 0, 1 and 3 passed all locomotion readiness conditions, while seed 2
missed the unchanged 0.001 floor only at 3.7-second pacing (0.0009956).
All conditions retained stable, periodic, approximately straight forward
motion. This establishes an externally paced forward primitive in normalized
mechanics, without establishing natural-animal speed or autonomous rhythm.

`scripts/assess_stop_restart.py` is the next diagnostic. With weights frozen,
one continuous neural/body trajectory contains six moving cycles, six
zero-command cycles and six restarted moving cycles. Each block's first
three cycles settle; its last three are measured. Stop centroid path length
and RMS joint-angle speed must be <=10% of the initial moving window, so
zero net displacement cannot hide reciprocal motion. Restarted forward
progress must be 80–120% of its original value. Both moving windows also
retain the existing propulsion and 0.001 progress gates, with 20/10-ms
refinement and isotropic controls at all three pacing rates.

`run_motor_curriculum.py --stop-restart --resume-run <wave-cohort>` runs all
requested seed checkpoints without training and saves fresh external
trajectories and summaries. `control_passed` reports stop/recovery success
separately from overall readiness. No neuron/body state is reset between
command blocks, and the checkpoint's stimulation gain is preserved.
This measures settling after 7.2–11.1 seconds of zero command, not an animal's
stopping latency. Testing the 0.07 remote ensemble remains necessary.

A local frozen 0.06 checkpoint passes `control_passed` at all three periods:
late stop centroid motion is below the saved trajectory's floating-point
resolution, RMS body-angle speed is <0.000004 of the moving value, and
restarted progress recovers to within 0.002% of the initial window.
Overall `passed` remains false because that checkpoint's forward progress
is below 0.001. This checks the new assay against a real fitted brain/body
trajectory; it is not a result for the remote 0.07 cohort.

The remote 0.07 stop/restart ensemble subsequently passed control checks for
all four seeds at all three pacing periods. Maximum residual shape speed was
0.024% of the moving value; restart progress differed by <0.1%. Overall gates
passed for seeds 0/1/3. Seed 2 missed only the unchanged slow-rate propulsion
floor, at approximately 0.000995 body lengths/cycle. These results required
no new training and do not establish immediate stopping latency.

### Reverse-wave fitting with forward rehearsal

`train_motor_alternation.py --stage reverse` loads the forward checkpoint's
amplitude, stimulation gain and propagation delay. Its reverse muscle target
delays the scalar command by distance from the posterior, yielding negative
front/rear lags. The existing positive-lag gates are sign-aware for this mode.
A new fixed direction context supplies +/-0.05 normalized current to
NMJ-projecting neurons (deterministic random signs, seed 719). Forward mode
uses the old input exactly; reverse mode adds the context, including during
stops. It is an artificial research interface, not a biological command model
and not a target phase/delay supplied to the recurrent network.

The supervised loss equally weights reverse target matching and rehearsal of
the frozen source actor's forward outputs. Both use sine and stop/hold command
sequences. All recurrent neural parameters can learn; topology and NMJ/muscle
parameters stay fixed. Longer held-out sequences check reverse wave lags and
forward trajectory retention. The best training-loss checkpoint, latest actor,
optimizer and final actor remain outside Git.

`run_motor_curriculum.py --reverse --minutes 180 --resume-run <wave-cohort>`
fits four seeds for approximately 179 minutes, then assesses forward and
reverse go/stop/restart independently. Both timesteps, isotropic controls,
actual centroid progress and existing magnitude/straightness/stop/recovery
gates apply. All fit and behavioral gates are required for overall success.
Completed finite fits receive diagnostics even when imitation fails, without
automatic promotion. The original forward checkpoints are preserved.
This stage is implemented and tested; successful reverse learning in the
remote ensemble is not yet established. Direction switching within a single
trajectory is not tested yet. No RL is introduced.

### Eight-hour bidirectional refinement

The three-hour reverse ensemble completed without errors: all four seeds
showed actual reverse translation and passed stop/restart control in both
directions at every rate. Reverse imitation passed seeds 0/2. Seed 1 failed
the changed hold-command sequence; seed 3 overshot target wave timing.
Reverse propulsion readiness passed seeds 2/3, while forward readiness passed
only seed 3. Seeds 0–2 retained forward translation but fell below its progress
floor. No model met every criterion. The original forward checkpoints remain
available and unchanged.

`scripts/refine_reverse.py` addresses these failures with longer, broader
commands and stronger original-forward retention. It resumes the learned
reverse candidate but loads the original forward teacher from its provenance,
verifying its checksum and input/body conventions. Fixed NMJ/muscle fields
must still equal their original initialization. All 302 recurrent neurons run;
only neural parameters learn. The amplitude, delay and direction context stay
unchanged. Normalized forward muscle error has weight 4 versus 1 for reverse,
with learning rate 0.0003 and fresh optimizer.

Eight 384-step training cases cover four pacing rates and varied phases,
levels and stops; random minibatches contain two cases. Eight separate
768-step tuning cases use different rates and command sequences. Every 500
updates, forward and reverse wave/body validation chooses the best checkpoint
by failed-case count, worst normalized criterion error, then mean error.
Training loss does not select the best actor. Initial, best, latest and final
actors, optimizer/RNG state, validation history and command banks stay outside
Git. Existing wave and propulsion thresholds are unchanged.

`run_motor_curriculum.py --refine-reverse --resume-latest --minutes 480`
selects a completed reverse cohort with successful motor control in both
directions, even if stronger imitation/readiness gates were missed. This is
permission to refine, not a successful stage declaration. The supervisor
checks the original forward reference before launching workers. After roughly
479 minutes of fitting, both independent long physics assays still run and
must pass for overall success.

The eight-hour results were reported on 2026-11-06: seeds 0/1/2 passed tuning
imitation, forward readiness passed 0/1, and every seed missed reverse
readiness. Reverse go progress magnitudes were 0.0005968–0.0009962 body
lengths/cycle, below the unchanged 0.001 floor. All seeds retained real reverse
motion and stop/restart control; all stages completed without errors. Seed 3
missed one forward tuning case. Selected updates were 19,500/35,000/35,500/2,000
for seeds 0–3, versus about 36k updates actually executed each. Thus more
training did not guarantee the selected model passed the movement criteria.
No seed met every gate. Optimizing and selecting directly for displacement is
the proposed next change, not an experiment already run. Direction switching
and RL remain later stages.

The chronological [laboratory notebook](LAB_NOTEBOOK.md) records this history,
exact reported summaries and evidence gaps. Future invocations also capture
commands, installed packages, numerical environment and source snapshots;
`archive_motor_run.py` preserves dependency cohorts and checksums externally.

### Anatomical muscle ordering (projection convention)

The full-network dish and supervised motor body now explicitly map anterior
muscle rows to the final body joint, adjacent to the segment used by
`head_position`. The previous projection mapped anterior rows to the opposite
end. Symmetric static targets concealed this mismatch; traveling-wave targets
would propagate in the wrong anatomical direction. The historical bundled
gait baselines retain their old projection convention so their fitted numbers
remain reproducible. New full-network manifests record the corrected ordering.

### Whole-network and subset optimization

`scripts/train_timed.py` and `scripts/run_cpu_training.py` now default to
`--actor full-network`. Explicit `--actor connectome` retains the historical
14-neuron steering baseline. `scripts/benchmark_rl.py --actor full-network`
benchmarks the new workload; earlier throughput figures do not apply to it.

```text
head concentration → adapted response, derivative, relative concentration
  → trainable current injection into AWCL/AWCR/ASEL/ASER
  → 302 recurrent normalized graded-potential neurons
  → anatomical 956-edge NMJ map → 95 muscle activations
  → local joint projection → overdamped RFT body → next head concentration
```

There is no gait phase input, speed formula, fitted gait, motor teacher, or
direct sensory-to-muscle readout in this actor. During exploration, Gaussian
noise is applied to muscle logits; deterministic evaluation uses brain-driven
muscle activations. PPO recomputes full recurrent sequences for temporal
gradients, with voltage reset at episode boundaries.

All anatomical chemical strengths and symmetric gap conductances are eligible
to learn, along with membrane fields, four sensory input rows, NMJ magnitudes,
unknown NMJ polarities, muscle transfer parameters and exploration scales.
Chemical/gap masks and known NMJ signs remain fixed. Chemical excitation versus
inhibition is determined by trainable presynaptic reversal potentials. Counts
initialize relative strengths; these are not measured conductances. All cells
use normalized generic dynamics, not the AWCON/RMD ion-channel fits.

`--train-group all|neural|nmj|sensory` and optional `--train-neurons AWCL AWCR ...`
restrict gradients, while the same 302-node model continues to run. Neuron
selection trains incoming chemical rows, incident gap edges, selected cell
fields and corresponding sensory/NMJ columns; muscle-wide fields and noise
freeze with a neuron subset. Use `--initial-actor /external/path/actor_final.npz`
to carry fitted parameters into another stage; the optimizer and critic reset.
Dropping subset restrictions enables whole-network optimization. Eligibility
does not imply every parameter receives a useful task gradient: disconnected
or behaviorally irrelevant circuits require additional organs/tasks, and no
anatomical edges are invented to make them influence locomotion.

Integration uses four frozen-conductance exponential-Euler substeps per 20 ms
interval. Scalar relaxation remains stable at large positive conductances;
accuracy is first order and needs substep convergence checks after fitting.
Tests compare to a fine-step RK4 reference and check stability, temporal
gradients, fixed masks, parameter freezing, body movement and actual PPO updates.
The NMJ transfer uses normalized cell thresholds and a smooth sigmoid with
resting tone, rather than the older mV motor teacher's rest-subtracted transfer.

The dish and body remain in the legacy normalized configuration. Sensors are
engineered proxies, and motion/food-finding in this model does not yet establish
real-worm behavioral fidelity. Checkpoints and generated results stay outside
the repository. See README for background launch commands.

## Active food-finding computation

The body-direct training path is:

```text
food field at head
    → engineered concentration memory
    → fitted seven-parameter sensory controller
    → [speed, steering]
    → fitted traveling-wave gait
    → 95 muscle activations
    → 11 joint commands
    → 12-segment body mechanics
    → new head position
```

The neural validation path inserts the supervised motor teacher:

```text
[speed, steering] + gait phase
    → fixed 13-feature encoder
    → fitted 302-channel voltage output
    → fixed Cook/Wang neuron-to-muscle projection
    → 95 muscle activations
    → body mechanics
```

The 302×302 neuron-to-neuron chemical and gap-junction matrices do not appear in
either path. Therefore the current food-finding result is not evidence that the
recurrent connectome can perform chemotaxis.

## Food field and sensed variables

Food is a finite two-dimensional Gaussian pulse. For source position (x_s),
head position (x), mass (M), initial variance (v_0), and diffusion
coefficient (D):

```text
v(t) = v0 + 2 D t
C(x,t) = M / (2 π v(t)) · exp(-||x - xs||² / (2 v(t)))
```

The current normalized defaults are dish radius `1.5`, `M=1`, `v0=0.16`, and
`D=0.005`. The analytic pulse is an unbounded-domain approximation; it does not
yet impose a no-flux boundary at the circular dish wall.

At each step the engineered sensor computes:

- log concentration at the modeled head;
- difference between log concentration and an adapting baseline;
- first temporal difference of log concentration;
- sine and cosine of the locomotor gait phase.

The adaptation time constant is `0.75` normalized seconds. There are no modeled
receptor kinetics, amphid geometry, bilateral sensory neurons, or sensory noise
in this baseline.

## What the seven-parameter fit learned

Let `r` be adapted log-concentration response, `dr/dt` be its temporal
difference, and `phi` be gait phase. Steering is:

```text
tanh(
    response_sine_gain     · r     · sin(phi)
  + response_cosine_gain   · r     · cos(phi)
  + 0.1 derivative_sine_gain   · dr/dt · sin(phi)
  + 0.1 derivative_cosine_gain · dr/dt · cos(phi)
  + steering_bias
)
```

Speed is:

```text
base_speed · (1 - food_slowing · relative_concentration)
```

The fitted physical parameters are:

| Parameter | Value |
| --- | ---: |
| Base speed | 0.7140385 |
| Adapted-response sine gain | 7.978845 |
| Adapted-response cosine gain | -7.581171 |
| Concentration-change sine gain | -10.060525 |
| Concentration-change cosine gain | -8.544154 |
| Steering bias | 0.041312 |
| Food-proximity slowing | 0.125010 |

The controller infers direction from temporal concentration changes during the
body-driven head sweep. This is an engineered approximation to klinotaxis, not
a reconstruction of the biological sensory circuit.

## Sensory-controller fitting protocol

The seven raw parameters were optimized with Adam for 300 iterations at
learning rate `0.02`. Each episode contained 500 steps of `0.02` normalized
seconds. Training used 16 reproducibly randomized source positions and initial
headings; evaluation used 24 held-out combinations.

The mean episode objective was:

```text
minimum source distance
+ 0.25 · final source distance
+ 0.001 · mean squared steering
```

Source coordinates were used only by this objective. The controller received
no source coordinates, bearing, or distance.

Results using a target radius of `0.12` body-length units:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Unfitted sensory controller, held out | 16.7% | 0.3584 |
| Fitted controller through body gait, training | 87.5% | 0.0672 |
| Fitted controller through body gait, held out | 91.7% | 0.0451 |
| Same controller through motor teacher and NMJs, held out | 62.5% | 0.2028 |

The reduction from 91.7% to 62.5% measures accumulated approximation error in
the motor teacher and neuromuscular transform. It is not a recurrent-brain
score.

## Body-direct reinforcement learning (alternative fitting method)

`r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv` exposes the identical
body-direct food field, gait, and body mechanics through a Gymnax-compatible
`reset`/`step` interface. It reveals only the same four sensed quantities
(adapted-response, its derivative, and gait-phase sine/cosine) plus relative
concentration, and accepts the same `[speed, steering]` command; the
environment is agnostic to how that command is produced.

`r_elegans.rl` fits the seven raw parameters against this environment with
model-free reinforcement learning instead of differentiating through the
simulator, using generalized advantage estimation and a small MLP critic that
supplies the value baseline used to compute advantages; the critic is
discarded after training and is not part of the deployed controller. Two
update rules are implemented, sharing the same rollout collection and GAE:

- **PPO** (`r_elegans.rl.make_ppo_train_step`, the default): the clipped
  surrogate objective. Each update collects one on-policy rollout, then runs
  several epochs over randomly shuffled environment-sequence minibatches,
  clipping the policy-ratio so a batch can be reused for multiple gradient
  steps without drifting too far off-policy. Time order is preserved within
  every sequence for recurrent actors.
- **A2C** (`r_elegans.rl.make_a2c_train_step`): plain policy gradient with the
  same GAE baseline but no ratio clipping, taking exactly one full-batch
  gradient step per freshly collected rollout.

The mean action reproduces `r_elegans.envs.petri_dish.decode_sensory_policy`
exactly (`r_elegans.rl.policy.action_mean`); a state-independent learned
log-standard-deviation drives exploration only during training.

Reward per step is `10 · (previous distance − new distance) − 0.001 ·
steering² + 5 · [distance < 0.12]`; an episode terminates early on reaching the
target radius and truncates at the configured step budget. Source position and
initial heading are randomized every reset; the controller never observes
them, matching the differentiable-fit protocol above.

Natural terminations have zero bootstrap value. Time-limit truncations instead
bootstrap from Gymnax's pre-reset final observation, and GAE is cut at the
episode boundary so returns never leak into the automatically reset episode.
Training success is successes divided by completed episodes, not the fraction
of individual timesteps carrying a success flag.

The following numbers are historical results from the earlier normalized-body
RL implementation. They have not yet been reproduced after the corrected
time-limit handling, recurrent BPTT update, and body-mechanics upgrade, and are
not current WormGym benchmarks.

A run of `scripts/train_rl_chemotaxis.py` with 64 parallel environments, 250
steps per episode, and 300 updates (learning rate `0.003`, 4 epochs per
update; PPO additionally split each batch into 4 minibatches with clip range
`0.2`) reached, on the same 24 held-out source/heading pairs used for the
differentiable fit:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Unfitted sensory controller, held out | 16.7% | 0.3584 |
| A2C-trained controller, held out | 75.0% | 0.0658 |
| PPO-trained controller, held out | 87.5% | 0.0523 |
| Differentiable-fit controller, held out (for reference) | 91.7% | 0.0451 |

Both RL update rules reach a large majority of held-out sources; PPO's
ratio clipping lets it use its rollouts more effectively than A2C's
full-batch updates at the same training budget, closing most of the gap to
the differentiable fit, which back-propagates an exact analytic gradient
through every simulated step rather than estimating a policy gradient from
sampled rollouts. This gap is the expected price of treating the simulator as
a black box; it is not evidence that the RL path is broken. Both paths deploy
the same seven-number controller shape, so an RL-trained checkpoint can be
evaluated, validated through the motor teacher, or visualized with the same
functions (`simulate_petri_dish`, `simulate_neural_petri_dish`) used for the
differentiable-fit checkpoint. `scripts/demo_rl_chemotaxis.py` trains a
controller this way and animates the resulting 12-segment body moving through
the dish.

## Connectome-subcircuit chemotaxis (partial, body-direct only)

The seven-parameter controller can also be replaced entirely by a real,
hand-selected 14-neuron subcircuit of the bundled connectome, trained by
supervised imitation and then reinforcement learning, while everything else
in the body-direct pipeline (food field, gait, body mechanics, reward) stays
identical. This is a step toward, not the completion of, the "connectome-
driven worm" described below -- 14 of 302 neurons, not the full recurrent
network.

**The subcircuit** (`r_elegans.brain.circuit.SUBCIRCUIT_NEURON_NAMES`):
chemosensory `AWCL/AWCR/ASEL/ASER`, interneurons `AIYL/AIYR` then
`AIZL/AIZR`, integrator `RIAL/RIAR`, and dorsal/ventral head-motor readout
`RMDDL/RMDDR/RMDVL/RMDVR`. This is not an assumed pathway: the bundled
connectome asset contains 63 real chemical synapses and 8 real gap junctions
among exactly these 14 neurons (e.g. `AIYL→AIZL` count 67, `RIAL→RMDDL` count
48, `RIAR→RMDDR` count 53), matching the textbook AWC/ASE→AIY→AIZ→RIA→RMD
klinotaxis pathway. The dorsal/ventral RMD subclasses are read out (rather
than the more obviously bilateral RMDL/RMDR) because the simulator's
`steering` command is itself a dorsal-minus-ventral bending bias, and RIA is
the documented site where left/right sensory asymmetry becomes exactly that
bias.

**Dynamics**: the existing graded-potential equations
(`r_elegans.brain.dynamics.neural_rhs`) integrated with a fixed-step RK4
(`integrate_neural_fixed_step`), in the same normalized unit system as the
rest of the body-direct simulator (not the separate, literal-mV/ms
`single_compartment.py` model, which is calibrated for only 2 of 302 neuron
classes and has no cross-neuron coupling). Anatomical chemical/gap masks are
protected from gradient updates (`jax.lax.stop_gradient` in
`effective_chemical_weights`/`effective_gap_weights`) so training can never
drift the fixed 0/1 topology into arbitrary floats; per-neuron time constant
and activation slope are trained through a positivity floor rather than as
bare unconstrained values, matching this repository's existing `decode_*`
convention. `synapse_reversal` (excitatory/inhibitory character) has no real
per-edge sign dataset in this repository; it is initialized mostly excitatory
with one literature fact hand-set (AIY inhibits AIZ) and is otherwise
trainable -- a disclosed judgment call, not a fitted quantity.

**Sensory transduction and readout are only partly biological.** `response`
and `derivative` (adapted log-concentration and its rate) are the same
signal real ASE/AWC neurons are documented to encode as ON/OFF adaptation,
injected as current into the real sensory neuron indices through trainable
gains -- but AWC and ASE currently receive the *same* underlying signal, and
`sin(phase)/cos(phase)` (an explicit stand-in for proprioceptive/CPG
coupling, since gait phase is not itself a modeled biological quantity here)
is injected the same way. Only `steering` is connectome-driven; `speed` keeps
the analytic controller's plain food-proximity-slowing formula, to keep the
biological claim (about steering/klinotaxis specifically) falsifiable and to
keep failure modes separable.

**Training proceeded in two stages**, per the roadmap in "What remains for a
connectome-driven worm": supervised pretraining first
(`scripts/pretrain_connectome_circuit.py`), then RL fine-tuning
(`scripts/train_rl_chemotaxis.py --actor connectome`). Pretraining
back-propagates an MSE loss through a full within-episode unroll of the
subcircuit's own recurrence to imitate the bundled differentiable-fit
controller's trajectories -- this stage is not subject to the RL rollout's
stop-gradient boundary and differentiates through time. Corrected recurrent
PPO/A2C now also backpropagates through the actor recurrence while continuing
to treat environment dynamics as a black box. The following measurements are
historical results from before that RL correction and require reproduction:

| Evaluation | Success | Mean minimum distance |
| --- | ---: | ---: |
| Untrained subcircuit, held out | 4.2% | 0.6372 |
| Supervised-pretrained subcircuit, held out | 45.8% | 0.1888 |
| + gentle PPO fine-tuning (`lr=1e-4`), held out | 45.8% | 0.1702 |
| Seven-parameter PPO, held out (for reference) | 87.5% | 0.0523 |

Supervised pretraining alone recovered 45.8% of held-out sources
from a real anatomical subcircuit with mostly-engineered sensory input --
substantially below the seven-parameter controller, consistent with a
14-neuron slice being a much more constrained function class than a
hand-designed 7-parameter formula tuned specifically for this task. RL
fine-tuning at the same learning rate used for the analytic controller's
from-scratch training (`3e-3`) *degraded* the pretrained checkpoint (down to
8.3% success) rather than improving it: the actor starts from a good
supervised optimum while its critic starts randomly initialized, and early
noisy value estimates produced large, destabilizing policy updates. A
substantially smaller fine-tuning learning rate (`1e-4`, with a matching
smaller entropy coefficient) avoided this degradation and modestly improved
mean minimum distance, though it did not raise the held-out success rate in
this run. This is reported as an honest, current limitation, not smoothed
over: further tuning of the fine-tuning learning-rate schedule, reward
shaping, or a critic warm-start are the most likely paths to closing more of
the remaining gap to the analytic controller.

`r_elegans.rl.actor_interface.ActorInterface` lets PPO/A2C train either actor
architecture. The connectome voltage is carried alongside environment state
during collection. During each update the actor is unrolled again over each
time-ordered environment sequence, resetting at episode boundaries and using
backpropagation through time. PPO shuffles whole sequences between minibatches,
never isolated recurrent timesteps.

## Supervised motor teacher

The motor teacher receives `[speed, steering]` and gait phase. Its 13 features
encode a bias; forward and reverse magnitudes; their sine/cosine phase terms;
and direction-specific steering and steering-phase interactions. A learned
`[302, 13]` coefficient matrix produces bounded voltages:

```text
V = -60 + 80 · sigmoid(features · coefficientsᵀ)
```

Only 129 neuron identities with conservatively signed neuromuscular outputs
were trainable. The remaining channels stay near rest. The fixed Cook contact
counts and Wang-derived signs then produce 95 muscle activations. Training used
12 forward/reverse/turn commands and achieved muscle RMSE `0.0570` and body
endpoint RMSE `0.0215` normalized body lengths.

The teacher has no neuron-to-neuron recurrence, membrane ODE, sensory current,
or internal neural state. “302-channel neural output” is the precise term;
“trained 302-neuron brain” is not.

## Anatomical topology in the repository

The bundled runtime asset contains:

- 302 neurons in one canonical order;
- 3,709 directed neuron-to-neuron chemical edges and contact counts;
- 1,093 undirected gap-junction pairs and counts;
- 956 neuron-to-body-wall-muscle edges and counts;
- conservative polarity for 880 neuromuscular edges;
- 95 anatomical body-wall muscle identifiers.

All matrices use `[postsynaptic, presynaptic]` indexing. The topology is needed
to initialize the recurrent training model, but topology alone does not determine
synaptic conductance, receptor sign, membrane dynamics, sensory encoding, or a
behavioral policy.

## What remains for a connectome-driven worm

The full-network trainer now replaces the engineered controller and motor
teacher with recurrent computation. Biological validation still requires:

1. Map food concentration and its dynamics to identified chemosensory neurons
   such as ASE, AWA, and AWC using documented receptor/transduction models.
2. Assign or fit neuron-specific membrane parameters beyond AWCON and RMD.
3. Fit chemical strengths, gap conductances, reversal potentials, thresholds,
   and time constants while preserving the anatomical masks.
4. Drive the fixed neuromuscular projection from recurrent voltages.
5. Train first against the existing sensory-controller and motor-teacher
   trajectories, then fine-tune closed-loop behavior.
6. Evaluate held-out source locations, initial postures, concentration fields,
   ablations, and robustness without exposing source coordinates.

RL is optional at that stage. Supervised trajectory matching and differentiable
behavior optimization are cheaper first steps; RL becomes appropriate for
delayed reward, state uncertainty, competing objectives, and richer tasks.

"Connectome-subcircuit chemotaxis" above is a narrow, partial step along
items 1, 3, and 5 for a hand-selected 14-neuron slice only -- not items 2, 4,
or 6, and not the full 302-neuron network. Item 1 in particular is only
partly addressed: sensory current reaches real chemosensory neuron indices,
but through the same engineered adaptation signal the analytic controller
uses, not a documented receptor/transduction model. In that 14-neuron baseline,
the neuromuscular projection (item 4) is not driven by recurrent voltages; it
uses the fitted gait. The separate full-network actor now drives muscles from
recurrent voltages and supports closed-loop PPO, but lacks biological sensory
and cellular calibration and a successful behavioral validation result.

## Terminology used by this project

- **Anatomical connectome:** fixed neuron identifiers and edge masks/counts.
- **Recurrent brain model:** membrane dynamics coupled through the anatomical
  neuron-to-neuron graph.
- **Motor teacher:** non-recurrent command/phase-to-302-output surrogate.
- **Sensory controller:** engineered seven-parameter concentration-to-command
  policy.
- **Current chemotaxis baseline:** sensory controller plus body gait, optionally
  validated through the motor teacher and neuromuscular anatomy.
