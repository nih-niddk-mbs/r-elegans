# Current model: exact scope and computation path

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


E071 PASSED the small autonomous water-model timing/amplitude engineering rung
in seed6901 (user-reported server results). All setup/search/validation/cleanup
finished in192.91min reported, inside the240min TOTAL budget; no timeout.
Ordinary body rhythm~1.8727Hz, T/M/H cycles .00316677/.03875394/.01320867rad.
All six active cases pass, including the long and refined timesteps; smallest
active tail bend .00316012rad is5.34% above the .003rad validation threshold.
Held voltage produces0 cycles. Three other seeds remain unqualified.

Qualified checkpoint SHA256:
`8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a`.
Run: `/home/carsonc/r-elegans-results/swim-continuation-20261113T150202Z-1002622`.
Preserve seed6901 and its complete external lineage; no consolidation or new
training launched. This establishes the engineering rhythm/amplitude skill,
not travelling-wave swimming, net propulsion or biological fidelity. The current authorized next step is sensory control followed by gated propulsion training.
Evidence: [E071 reported outcome](experiments/records/E071-server-summary.json).

E070 server results received: four workers completed (781 reported proposals),
all unqualified on the full engineering rung. Seeds6800/6802/6803 have body
rhythms within1.8Hz±10%;6801 remains fast at2.212Hz. Every active validation
misses the .003rad TAIL amplitude requirement while middle/head exceed it.
Seed6802 is the preferred follow-up candidate by reported timing and tail
amplitude:1.8797Hz, T/M/H=.00274598/.0389484/.0133585rad. Tail is91.5% of
threshold, requiring~9.25% gain from its present amplitude. This is not a full-
objective or propulsion ranking. Across reported half-step cases, cycle changes
are<.233% and frequency changes<.758%; long amplitudes remain close to ordinary.
Held voltage produces0 cycles. These support autonomous water-model wiggling
under E070's unforced protocol, not coordinated swimming or biological fidelity.

Preserve all four checkpoints; proposed next step is focused posterior-amplitude
fitting from6802 while retaining the swim-timescale band. No new run launched.
Run ID, checkpoint hashes, baseline gains and complete guard/bound reports were
not supplied; verify those before a continuation. Evidence:
[reported E070 summary](experiments/records/E070-server-summary.json).

E069's earlier body-only timing replay showed larger modeled wet-substrate bends
at slower playback. It did not establish a resonance peak or an autonomous skill.

E068 server audit completed successfully (user-reported stdout; run
`output-path-audit-20261113T041119Z-939399`). Under identical ordinary scoring,
the hybrid has larger T/M/H cycles than old/old by55.4%/23.4%/115.2%.
Old/new and hybrid printed amplitudes match exactly. Most tail-joint oscillatory
signal survives dorsal/ventral subtraction at the observed motor frequency;
the strongest cancellation is the final head joint (only8–9% survives in the
hybrid/candidates). Cancellation alone therefore does not explain weak tail
bends. This ratio concerns one frequency, not broadband energy or causal gain.
Detailed NMJ/muscle/filter stdout has now been received; findings follow.
No new model qualification or training.
Evidence: [E068 server summary](experiments/records/E068-server-summary.json).

E068 detailed stage report now received: muscle activation near-extreme fractions
are0 in all printed windows, with responsive local sigmoid slopes. For hybrid
45–60s, net command T/M/H=.00175222/.0701773/.022169; filtered activation=
.000998/.0398816/.0126051 (~57% retained), and joint cycles=
4.68955e-5/.000244504/.000266448rad. Tail command is only~1/40 of middle.
Thus weak posterior command and downstream mechanical attenuation both matter;
sigmoid clipping is not supported. These regional ratios do not establish causal
gain or a physics error. Next proposed check: body torque/response across timing
and spatial patterns, keeping the preserved neural models intact. No new run.
See [stage evidence](experiments/records/E068-server-stage-details.json).

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

Historical E067 preparation (completed server result above).
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

Historical partial E066 report (full result now above): execution valid but rhythm/regional
validation failed. Candidate hash f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e.
Both initial-voltage perturbation rows retain37 repeating voltage,90 muscle and
11 joint channels. Direct tail/middle/head cyclic excursions are about
4.28e-5/2.66e-4/2.89e-4rad, well below the .003rad engineering comparison.
Zero historical window-gated medians do not mean zero physical motion. Neural
refinement checks pass; gated regional comparisons can compare zeros with
zeros and do not establish convergence of nonzero cycle amplitudes. PC1=1 over
eligible joints alone does not establish whole-body synchrony or propagation.
At that earlier entry the long/baseline/refinement rows were missing; the full
compact report subsequently supplied is recorded above.
See [partial server evidence](experiments/records/E066-server-partial-summary.json).

Historical E066 preparation (server execution now reported above).
The old-channel/new-connection candidate is pinned by its saved hash and exact
family reconstruction. Eight simulations test its own initial-voltage,
neural/body timestep, held-voltage and300s unpaced persistence controls, plus
regional amplitudes and raw-supported phase/PCA descriptions. Four workers
run concurrently. No fitting or automatic model promotion. Local8-case12s
software smoke and282 tests passed; the older local fixture fails the stricter
tail persistence gate, retained as a negative result. Server execution and full cycle table now reported; original screening remains
failed and the proposed next rung is documented above. See [method and local checks](experiments/records/E066-hybrid-validation.json).

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

Historical partial E065 report (full table now above): successful execution and common float64 initial
states for rhythm-comparison-20261112T185742Z-865067. Three complete control
rows were supplied: old/old neural-half and initial-voltage +/-1mV each retain
54 repeating voltage channels,3 calcium,92 muscles and11 command/drive/joint
channels under the diagnostic floors. This supports the earlier oscillatory
parent as a starting point under tested controls; it does not identify the
parameter family suppressing current-model rhythms. At this earlier entry primary hybrid and precision/feedback rows were still
missing; the complete table subsequently supplied is recorded above.
Evidence is user-reported partial stdout; no artifacts independently retrieved.
See [partial result](experiments/records/E065-server-partial-summary.json).

Historical E065 preparation (partial server result now reported above)
(2026-11-12). Four full-network combinations separate old/new channel scales
from old/new chemical+electrical strengths. Fixed fields, leak, anatomy and
muscle readout must match exactly. All primary cases share the current E063
sensory gains, initialization and wet-substrate proxy.15 independent60s
simulations include cue/sham, initial-state, precision, feedback and neural-step
controls, with four concurrent workers on disjoint four-physical-core groups.
No fitting or periodic drive. Local15-case12s software smoke and279-test suite
passed; this used an older local fitted model, not actual server3fabef... .
Server execution and the full family table are now reported; E062 remains separate.
See [E065 method and verification](experiments/records/E065-rhythm-comparison.json).

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

Historical E064 readiness (server result above): the diagnostic was implemented and locally verified
(2026-11-12). It reads saved E063 baseline/best and all validation active/sham
signals, counts raw extrema across every channel, and writes per-channel tables
and raw-signal figures externally. Inputs are hash-checked before and after;
no simulation, fitting or JAX/GPU initialization. The local fixture is an older
checkpoint, not the eight-hour server model. Server analysis subsequently completed; see the dated result above. No next training objective or job has been started.
See [diagnostic record](experiments/records/E064-signal-chain-diagnostic.json).

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
The subsequent eight-hour server run failed; see the E063 outcome above.

Historical E061 finding and E062 plan (superseded by the E062 result above):
float64 peaks remain2.85671–2.85674e-5rad as
neural step shrinks. Peak variation0.00116%; full-trajectory differences halve
6.10e-9 ->3.05e-9rad. Float32 arithmetic strongly contributes to the prior
sensitivity in this test. Better-resolved amplitude is95.22% of target, still
unqualified. E062 prepares float64 fitting, preserving stored E058 parameters,
original coordinate bounds and3e-5 acceptance threshold. Optimize a declared
2% training-only margin; add low-dose numerical checks. Fresh seed6200 because
precision/objective changed; not an exact float32 continuation. At that planning entry no server job had been
launched. See [E061 report](experiments/records/E061-server-summary.json).

E060 server result localizes the tested sensitivity to neural stepping:
brain-half/quarter bend ratios .9207/.8352, body/sensory/coupling .99981/.99985.
Body-branch trajectory differences shrink; neural differences grow. Keep the
original E058 gate pass, but defer promotion and fitting. E061 subsequently completed its fixed-parameter float64 comparison, with
legacy input/initial-state values promoted rather than regenerated. This tests
precision sensitivity; it is not a solver fix or a biological qualification.
See [E060 record](experiments/records/E060-server-summary.json).

E059 server numerical assessment:original low-dose coarse gate passes, but
half/quarter fail at2.796e-5/2.537e-5rad versus3e-5 target. Quarter amplitude is
16.47% below coarse; successive peak changes have not shrunk. Recovery remains
acceptable. Retain E058's original protocol pass, but do not promote or continue
fitting until this sensitivity is understood. E060 subsequently ran the split-step assay; its result is recorded above.
No model parameters were changed. Full comparison arrays and actual convergence remain unknown.
See [E059 server record](experiments/records/E059-server-summary.json).

E058 first-rung engineering PASS (user-reported):99additional proposals,
global99 ->198, selected167. All required active variants and recovery/control
criteria pass. Low-dose bend3.03782455e-5rad is101.26% of target, nominal116.42%.
Preserve winner6cd7b0fa... with its sensory parameters independently. This is a
tiny sensory-evoked reversible response, not autonomous wiggling or a biological
reflex fit. E059 subsequently assessed low-dose timestep sensitivity; its result is above.
The original training refinement tested nominal cue only. No retraining or threshold change.
See [E058 report](experiments/records/E058-server-summary.json).

E057 server result:99 proposals, selected98,63.80minutes, qualification false.
Nominal, high-dose, shifted-onset and nominal-refined responses now pass the
current gates. Low dose reaches2.93875611e-5rad (97.96% of target), still below
3e-5; controls and engineering refinement pass. Retain candidate1d864fdb... and
same robust objective. E058 subsequently executed completed-search continuation with saved
neural/sensory parameters and final RNG after99 (not selected98). No target or
bound changes were made; its pass is recorded above. Near-threshold low-dose
performance needs numerical care: current refinement is nominal-only, not a
separate fine-step low-dose qualification. See
[E057 server report](experiments/records/E057-server-summary.json).

E056 full compact report:148 proposals (global48 ->196), selected195. Active
bend grew102.1% to2.890e-5rad (96.3% target), low dose2.503e-5rad (83.4% target);
high dose passes. Controls/recovery and engineering refinement pass. Remain on
first rung. E057 subsequently executed worst-case selection over nominal and.8dose, since
nominal amplitude alone would stop earning reward at target. Keep the target,
original parameter bounds and learned neural/sensory parameters; use fresh
seed5700/global0 for the changed objective. Low dose becomes a training condition;
high dose, timing shift and refinement stay outside selection. That result is reported above. See [E056 complete compact record](experiments/records/E056-server-summary.json).

E056 continuation server tail reported (2026-11-11):exit2 after63.87minutes;
refined bend2.803e-5rad (93.4% of target), shifted onset2.832e-5rad (94.4%).
Both remain unqualified despite adequate recovery; chemical/voltage controls
and engineering refinement pass. Chemical sensory multiplier is now exactly2,
its upper bound; neural bound occupancy is unknown. Refined amplitude increased
120.0% versus E055's matched refined case. Candidate180c2b93... resumes confirmed
E055437b227e.... Those fields were missing from that initial paste; the full compact record above
now supplies them and supersedes the request for a report. Retain target/bounds. No new training launched. See the
[E056 partial record](experiments/records/E056-server-partial-20261111.json).

E055 complete compact server summary reported:48 proposals, best selected at48;
qualification failed. Active cue-specific bend grew51.1% (9.462e-6 ->1.430e-5rad)
and objective fell23.6%. Active recovery residual fraction improved14.0% ->10.7%,
though absolute residual increased. Lowest active validation amplitude is
1.233e-5rad (41.1% of target); all active variants remain below3e-5rad. The
previously unlabeled row is confirmed refined. Proprioception block increases
this cue response7.6%, so its benefit is not established in this assay. Retain
feedback and first-rung target. E056 subsequently ran with neural/sensory offsets and PCG64 restored; its
partial result is recorded above. See the
[complete compact report](experiments/records/E055-server-summary.json).

E055 server result partially reported (2026-11-11):23.58minutes, time-budget
stop; checkpoint437b227e... from confirmed E053 parent98ea6655.... An unlabeled
validation case shows1.274e-5rad cue-specific bend (42.5% of the3e-5 target),
19.5% residual and6.025mV AWA recruitment with recovery. Chemical and held-voltage
blocks give zero differential response; engineering refinement gate passes
(2.750e-6rad versus7.5e-6 tolerance). At that reporting point run ID, baseline/best, iterations and active rows were
missing; the complete compact report above now supplies them. Exit digits remain
unreported.
The labeled proprioception block is descriptive; compare it with the missing
active case before interpreting feedback's contribution. See the
[partial server record](experiments/records/E055-server-partial-20261111.json).

E055 now defines a sensory–motor curriculum: reversible cue response ->
alternation -> proprioceptive propagation -> translation -> sensory regulation.
The first-stage trainer is implemented; partial server results are recorded below. It trains
unknown channel/synapse strengths and three bounded exploratory sensory gains
through the existing full302-neuron network, with fixed sensory routing/kinetics,
NMJ and mechanics. A single diacetyl exposure/washout is sampled through actual
head position; matched sham, dose/timing changes, timestep refinement and causal
blocks separate sensory-evoked movement from spontaneous motion. The initial
.00003rad reversible-bend and .1mV AWA thresholds are engineering gates, not measured
animal reflex targets. No periodic teacher or direct muscle commands are supplied.
Later stages are planned, not implemented or demonstrated; advancement requires
review. See [curriculum specification](experiments/sensorimotor_curriculum_v1.json).

E054 server diagnostic completed (user-reported, 2026-11-11): active and
refined40-second simulations both report no sustained repeated voltage, muscle
or joint excursions under the detector criteria; held-voltage control also
reports none. None of9616 fitted coordinates is within1% of its original
log-coordinate bound. This does not justify widening the factor2 bounds.
Zero qualified excursions is not zero variance or proof against slower rhythms:
the late20-second detector requires three peaks/troughs, persistence across five
subwindows, and explicit numerical reporting floors. Full traces and timestep
difference metrics have not been retrieved. The pasted diagnostic does not
include its checkpoint hash; latest E053 identity remains expected, unverified.
See [E054 report](experiments/records/E054-server-summary.json).
The next sustained-amplitude stage remains a specification, with no trainer or
new training run launched. Preserve prior checkpoints and fixed mechanics/readout.

E053 eight-hour continuation (user-reported, 2026-11-11): 865 additional
proposals (global369 to1234) reduced the paired training objective11.1%, from
44.993 to40.012. Matched refinement retained the gain. Sampled coarse bend
fluctuations increased29–126% but remain about11–34 times below the recordings;
the nominal neural-error component was essentially unchanged. Approximately93%
of the scalar loss decrease came from mean unsigned bend error and7% from
bend-fluctuation error. Mean bend includes posture and dynamic excursions;
this does not establish a static-posture explanation. Full traces and parameter
bound summary is now recorded in E054; full trajectories remain unavailable locally. No held-out or long-autonomy qualification, new
bundled model, or new run is claimed. See the
[E053 record](experiments/records/E053-server-summary.json). Next: inspect remaining trajectory statistics and make sustained dynamic amplitude explicit in the curriculum.


E052 server continuation (user-reported): 335 additional proposals (global
34 to 369) reduced the training objective by 5.3%, from 47.514 to 44.993.
The matched finer-timestep comparison preserved the gain. Coarse bend
fluctuations increased 23–49% but remain about 15–51 times below the recordings.
The nominal calcium-error component worsened about 1.6%, so the aggregate loss
improvement is a tradeoff, not improved agreement in every data family. Frozen
fields were reported unchanged; no held-out, long-autonomy or locomotion
qualification is demonstrated. The full checkpoint/source/history have not been
independently retrieved. See the [E052 report](experiments/records/E052-server-summary.json).
Next: inspect parameter-bound occupancy and neural-to-muscle-to-body transfer
before another long fit. Existing bundled models and qualified skills remain.


E050 local pilot result: one shared candidate fitted two eight-frame training
windows, reducing the exploratory objective by about 0.2% (47.484 to 47.389).
The reduction persisted with both brain and body timesteps halved. This leaves
large neural and bend-amplitude discrepancies and provides no held-out biological
qualification. The selected candidate is archived separately; winner7001 and
bundled model assets remain unchanged. See the
[E050 record](experiments/records/E050-paired-joint-pilot.json).

E050 adds an executable **joint neural/movement statistics pilot**:
`python -m scripts.fit_paired_recordings`. It shares unknown non-leak channel
densities and existing chemical/electrical strengths across selected training
recordings; stereotypical fields, anatomy, NMJs and mechanics remain fixed.
No target-driven current or periodic input is supplied; actual-body
proprioception remains active. The first objective compares reporter fluctuation
CV/lag correlation with five unsigned coarse chord-bend means/amplitudes.
It does not fit exact phases, chirality, speed or pointwise neural trajectories.
Optical gain cancels in the neural statistics under the stated emission model;
additive offsets/noise and transferred reporter kinetics remain assumptions.
Publisher nominal spatial spacing and approximate imaging-volume averaging are
explicit. The 1.7% wet-agar proxy remains mechanically unmatched to the mounted
2% recording condition. No held-out evaluation or biological qualification is
claimed. The checkpoint remains separate from the established winner and demos.


Paired-data preparation (E049, 2026-11-10): all 68 released HDF5 files now
match the publisher metadata hashes. The 21 identified baseline recordings are
prepared outside Git, with fixed date-group splits (17 train / 2 validation /
2 test), source timestamps, segment boundaries, confidence-filtered identities
and finite-value masks. Their union covers 130 named neurons; coverage is
25–86 per recording. Previously fitted June 14 recordings remain training data.
These are author-processed fluorescence ratios, not measured voltage. The paper
uses food-containing buffer on 2% NGM agar under a bead-spaced coverslip; our
water and 1.7% wet-agar models are not calibrated to that mounting. Joint fitting
still requires a spatial observation mapping, calcium likelihood and explicit
condition handling. No model parameters changed. See the
[E049 admission record](experiments/records/E049-paired-data-admission.json).


This document describes what the repository executes today. It distinguishes
anatomical data, implemented equations, fitted surrogate controllers, and the
full recurrent training path so anatomical availability, trainable models,
and demonstrated behavior are distinguished.

## Executive summary

**Latest autonomous milestone,2026-11-10:**winner7001 now passes both tail
motor/muscle/bend and distributed head/middle/tail amplitude checks (E043/E044),
with zero external drive. Server-reported checkpoint;coordinated locomotion
remains undemonstrated. See [distributed amplitude evidence](#qualified-autonomous-distributed-amplitude-e044-2026-11-10).

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

## Sustained channel dynamics qualification (E027, 2026-11-07)

The existing channel kinetics **can sustain activity** in an appropriate
conductance regime. E027 adds `brain/rhythm_qualification.py` and
`scripts/assay_channel_rhythms.py` to distinguish persistent cycles from plateaus,
decaying transients and numerical instability. No new channel or phase clock
was needed for this demonstration. This does not establish complete kinetics
for the real motor system; isolated SLO channels and other motor-class priors
from the 2024 models remain unported.

A literature-inspired RMD profile (full existing RMD preset, leak 0.045 nS,
CCA1 30 nS) oscillates with **zero applied/compensation current** for 30 s,
from -35, -34 and -70 mV initial conditions. After the first 15 s, 69 cycles
remain, period ~0.219 s and amplitude ~22 mV without decay. CCA1 removal and
the ordinary RMD preset settle rather than cycle. This is a dynamical-model
candidate, not a measured channel composition; its high CCA1 conductance and
waveform are not claimed to reproduce animal RMD or DA/VA physiology. It is
not the reduced Morris–Lecar model of E024.

Starting from E026's preserved initial channel network, apply this profile only
to six RMD cells, zero their compensation bias, retain all other cell parameters
and **all original anatomical chemical/gap weights** at the original 0.03 nS
weight-unit scale. RMDL, RMDR and RMDVL retain sustained cycles in the connected
302-neuron simulation over 20 s (~46 cycles in the final ten seconds; period
~0.216 s). Other RMD cells are quenched or fail the persistence gates under
coupling. Removing CCA1 from the six profile cells eliminates qualifying rhythms
in the network control. Other cells retain their original constant compensation
bias; no time-varying neural input or target forcing is introduced.

Doubling substeps preserves qualifying rhythms. Small period changes accumulate
large pointwise phase differences over long recordings, so raw errors are retained
and cycle periods/amplitudes/phase-aligned waveform errors are reported separately.
Connected qualifying cells differ by <=0.371% in period, <=0.274% in amplitude,
and <=0.121 mV in phase-aligned waveform RMS under refinement. This supports the
existence/shape of the simulated cycles, not pointwise long-horizon equality.

The candidate is saved as a complete external `network_candidate.npz` with its
actual channels/configuration. `fit_calcium_network.py --initial-network` loads
it directly; `--qualify-rhythms` checks initial and final candidates in a separate
20 s unpaced assay and returns nonzero if all RMD activity collapses. This is a
persistence guard, not a successful calcium-fit or biological-validity criterion.
It never installs a candidate automatically. Example qualification:

```bash
python -m scripts.assay_channel_rhythms \
  --initial-network /external/preserved-channel-network.npz \
  --duration 20 --max-step-ms .1 --output-dir /external/fresh-rhythm-assay
```

The positive finding establishes a self-sustained **neural** dynamical regime.
Coordinated muscle waves, proprioceptive control, correct A/B motor physiology,
animal-matched calcium activity and body propulsion remain separate requirements.
The operating driven-skill model remains unchanged.

## Unpaced channel-brain/body diagnostic (E028, 2026-11-07)

The channel branch is now connected to the 95 muscles and overdamped body via
`body/channel_bridge.py`. Each execution step advances the complete neural
state, computes physical-unit transmitter release, applies the preserved parent
NMJ multipliers/signs and muscle sigmoid, and advances the body. Anatomical
projection uses head-at-end ordering. There is **no body-to-brain feedback**;
sharing an execution loop does not implement proprioception. No oscillator,
gait teacher, periodic input, optimization or readout retuning is inserted.

The channel network's physical release thresholds/slopes are used explicitly.
The parent's NMJ/muscle fields stay frozen; its old normalized neural voltage
is not supplied to the new membrane equations. The nonzero resting muscle tone
matches the parent full actor. SI liquid/agar mechanics are available alongside
the normalized body, but this transferred neuromuscular readout is uncalibrated
for the changed neural dynamics.

Local 20 s liquid test: three RMD cells remain rhythmic, maximum late muscle
peak-to-peak activation 0.05841, maximum joint variation 0.003065 rad. Late net
displacement is only 4.85e-6 body lengths over ten seconds: **a small unpaced
wiggle, not demonstrated useful locomotion**. Both RMD-CCA1 removal and held-
voltage controls have zero late muscle/joint variation. Halving body timestep
changes maximum joint variation to 0.003071 rad (~0.2% difference). E028 records
source/input hashes, trajectories, state snapshots and exact local commands.

### Launch the bounded mbslnx2 diagnostic yourself

User requested push/self-launch. First server results are now user-reported below.
To launch a fresh diagnostic on mbslnx2:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_embodied_mbslnx2.sh
```

The repository launcher uses the dedicated r-elegans Python, starts `nohup`,
and prints the PID and unique log path. You may then disconnect. It now uses
**the exact Mac-tested parameter pair**, bundled in
`r_elegans/assets/mac_rhythm_v1/` (103,829 bytes plus a provenance manifest).
No transfer, server checkpoint discovery or reconstruction is needed. SHA256
checks validate both files before launch; the supervisor copies them into the
new external run directory to preserve exactly what was tested.

`network.npz` includes all 302 cells' channel/fixed parameters, recurrent
connections, release settings, configuration and canonical ordering.
`parent_actor.npz` preserves the corresponding NMJ/muscle readout. This is the
pair used in the 20-s Mac test, with longer persistence now user-reported below;
these parameters are not a new physiological fit.
Use supervisor `--parent-actor` only to explicitly reconstruct a different
candidate, or `--inputs` for another exact pair. No automatic substitution occurs.
The earlier server run below used different parameters; its negative result
remains part of the record. Raw recordings, trajectories and training artifacts
stay external; these two compact parameter assets are included at user request.

Four workers use four disjoint physical CPU cores each, CPU JAX and highest
matmul precision. Nine cases cover three media, +/-1 mV initial perturbations,
CCA1 removal, held voltage, and body/brain timestep refinement. All parameters
are fixed. The queue has a **45-minute wall budget**, plus startup and up to 45 seconds
for cleanup (explicit parent reconstruction has a separate two-minute cap); each case has a 12-minute cap
and 600-s simulation ceiling. It can finish sooner and reports pending cases.
Chunks and state snapshots stay under `/home/carsonc/r-elegans-results/embodied-*`.
The launcher does not modify conda environments.

`progress.json`, per-case logs and `summary.json` distinguish finite execution,
neural persistence, varying muscle output, dynamic shape and drift. Nonzero
displacement alone never sets `locomotion_demonstrated=true`. Snapshots retain
all gates/calcium/body state; the CLI does not yet expose exact checkpoint resume.
User-reported server result, run `embodied-20261107T185913027018Z`:
all nine cases exited 0 and remained finite; queue elapsed 1397.13 s (23.29 min).
Eight reached 600 simulated seconds; brain-step refinement reached 374 s before
its per-case budget. All final 15-s measurement windows had zero voltage,
muscle and joint variation, zero rhythmic RMD cells and zero displacement.
This candidate **did not sustain autonomous activity or movement**. It retained
nonzero, constant muscle activation. Media changes, +/-1 mV initial perturbations
and timestep refinements did not rescue late activity.

Server parent SHA256 `beb069dd051763b6fdc3de977e4a2f77c9db73827cb4e8dfb2f88d899ea27078`
and candidate `95fafcabfce62aa0a2c37dc9c49b475a23ed96264e8d8899640bb9ad30da4882`
differ from the Mac-tested pair. The later run below tests the Mac reference over longer durations.
Full server trajectories/environment/source have not been independently retrieved.
See `experiments/records/E028-server-20261107.json` for the earlier failure.

**2026-11-08 update — Mac reference remains active at ten minutes.**
User-reported run `embodied-20261108T035729978291Z` finished all nine cases with
queue exit 0. Three rhythmic RMD cells remain in the final measurement windows
of all six active 600-s cases (three media, +/-1 mV perturbations, refined body
timestep), with maximum voltage p2p about 21.20 mV and muscle p2p 0.05846.
Active liquid joint p2p is 0.003352 rad; late net displacement is only
2.64e-6 body lengths. Agar displacement is 1.01e-5 body lengths, barely above
the diagnostic drift threshold; neither value establishes useful propulsion.
Both calcium-removal and held-voltage controls have zero rhythmic cells, muscle
variation and body variation. Brain-step refinement retains three rhythmic
cells at 368 s but ends at its case budget, not 600 s.

This supports persistent **unpaced model activity driving small bends**, not a
biologically validated oscillator or coordinated locomotion. Summaries measure
final windows; full trajectories are needed to establish uninterrupted activity
and spatial coordination. The default bundled input identity is expected from
the launch workflow, but this compact output does not include input hashes or
source/environment, so their identity has not been independently verified for
this run. The earlier failure is therefore not evidence that mbslnx2 cannot
reproduce rhythmic activity. Next task: inspect neural/muscle/body phase along
the worm, then develop traveling bends while preserving this reference.
See `experiments/records/E028-server-20261108.json`.

## Conservative fitting policy (E029, 2026-11-07)

User-directed default: **fix stereotypical properties and fit explicit unknowns**.
`brain/fit_constraints.py` now creates and enforces an `unknown-only-v1` policy
for every calcium fit. Earlier E026/E027 all-conductance runs remain historical
experiments; they do not describe the new default.

Fixed: channel gating equations/rates, ionic reversals, calcium handling,
capacitance, leak reversal, anatomical masks, release/reversal priors, indicator
constants/readout and the NMJ/muscle/body contract. LEAK conductance is additionally
fixed to the initial value as a conductance-scale anchor. Established RMD class
profiles and the high-CCA1 exploratory profile stay fixed. Channel priors of
unobserved cells remain fixed in a given recording. These last choices are
conservative model assumptions, **not a declaration that all such values are
known biological constants**. Capacitance/leak need not be identical across cells.

Trainable: other unknown channel densities in observed, unanchored cells, plus
unmeasured chemical/electrical strengths on existing anatomical edges. Contact
counts constrain topology/initialization, not measured conductances. The mask
policy never creates edges or fits new capacitance/reversal/kinetic fields.
The parent-relative penalty/bounds now use the supplied initial checkpoint,
including its nonzero multipliers, rather than pulling those multipliers to zero.

Every run saves `fit_policy.json` and `trainable_masks.npz`, with per-cell reasons,
fixed-property evidence categories and actual trainable counts. Frozen entries
are masked in the forward/gradient path and restored after every optimizer step,
so momentum cannot move them. Summary reports frozen-parameter change counts.
Use `--constraints /external/constraints.json` to freeze additional sourced known
channels/edges. Example (freezes current values; does not assign measurements):

```json
{
  "frozen_channels": [
    {"neuron": "ASEL", "channel": "UNC2", "source": "recording/model identifier", "reason": "hold independently constrained value"}
  ],
  "frozen_edges": [
    {"kind": "gap", "post": "AVAL", "pre": "LUAL", "source": "recording/model identifier", "reason": "hold independently constrained conductance"}
  ]
}
```

Supply a checkpoint containing the intended constrained values; record whether
the source is direct measurement, a published estimate, expression inference,
or a modeling assumption. A gap constraint freezes both directions. Source/reason
and canonical/existing-edge validation are required; unknown schema fields fail.
No universal left/right or motor-neuron equality is imposed without supporting
evidence. Biology, prior assumptions and numerical scale anchors remain distinct.

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

## Autonomous amplitude fitting (E030, 2026-11-08)

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
User-reported server run `embodied-20261108T141954839072Z` retains three RMD
rhythms and 0.02951-rad liquid bends at 600 s. Perturbations/body refinement
retain that amplitude, while both controls stop it. Brain refinement retains
it at 370 s before budget. Full remote input/source hashes remain unretrieved.
See `experiments/records/E030-server-20261108.json`.
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

Long equal-phase drift (BL/s), seeds0..3:
-5.002e-7 / -2.854e-8 / -3.911e-7 / -4.470e-7. Small positive short endpoint
scores reverse or shrink after matching phase in seeds0,2,3; seed1 retains
small short-case positive phase drift but loses it in the long case. Body
refinement changes the tiny residual substantially, sometimes flipping sign.
Seed1 positive-cycle fractions of1 in short cases do not qualify it: speeds
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

## Autonomous body-wave curriculum (E035, 2026-11-08)

The next run develops self-generated body-wide bending before propulsion. It
starts from preserved `autonomous_wiggle_v1`, with all 9,616 unknown neuron
density/chemical/electrical parameters eligible. Injected currents remain zero;
leak, capacitance, kinetics/reversals/calcium, anatomy, release properties, NMJs
and body mechanics stay fixed. No clock, sensory drive, waveform teacher or
body force enters the forward model. Observed spontaneous oscillation supplies
only a frequency/phase reference for the score.

First substage: spread coherent motion into posterior and middle regions. A
continuous log-amplitude coverage score uses both joint bending and filtered
antagonistic muscle drive, including common-frequency power and phase locking.
Once coverage reaches target, reward ordered phase progression across anterior,
middle and posterior regions. Either propagation direction is allowed. Static
posture, a stronger head-only wiggle, synchronous whole-body bending and
opposing standing-wave lobes cannot qualify as a traveling primitive.

Engineering qualification (not animal-calibrated): harmonic p2p >=0.003 rad
for regional joints and >=0.003 for regional drive; common-frequency power
fraction>=0.5, phase locking>=0.8; phase span>=pi/3 in both shape and drive,
consistent-sign spatial increments between0.1 and0.8pi; second shape-mode
variance>=2%. A sustained >=0.01-rad bend must remain. A sustained neuronal
rhythm >=0.05 mV can support it, including graded activity in non-RMD neurons.
Voltages must remain bounded, muscles unsaturated, joints unclipped, and the
late mean body axis must remain well-defined with <0.35-rad turning.

All unknown parameters remain independent. Anatomy only focuses some proposal
blocks on the 103 neurons projecting to middle/posterior muscles and their
existing inputs; global proposals still reach every eligible parameter. New
curriculum bounds are 0.0625–16 times the source density/effective connection
strength, wider than the historical 0.25–4 neighborhood because most non-RMD
channel priors were initially weak transferred estimates. These are search
limits, not measured physiological ranges. All fixed physiological fields
remain anchored. Exact masks, bounds, proposals and accepted coverage are saved.

Run on mbslnx2:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_autonomous_wave_mbslnx2.sh
```

Four disjoint four-core CPU workers, up to four hours including validation,
under nohup. Log/run prefix is `autonomous-wave-*`. The old movement launcher
now delegates to this current curriculum, preventing another endpoint-score
run by mistake; historical `--objective movement` remains available only for
explicit replay from recorded source. The new run does not change earlier
checkpoints or an already-running job.

After a short-window wave target or search budget, validate 300 simulated
seconds, +/-1 mV initial states, half neural/body timesteps and held voltage.
All active cases must qualify; refined amplitudes must agree within25%, phase
increments within0.2 rad and frequency within5%. Held voltage must stop motion.
If any check fails, preserve the candidate but do not promote it. `passed` in
an E035 summary means body-wave qualification, **not forward locomotion**.

Complete-cycle translation is now also measured by integrating centroid
position over full reference cycles, using rising and falling phase origins,
and regressing cycle-mean positions in three late blocks. It no longer rewards
instantaneous endpoint phase. Both shape closure residuals and phase-origin
agreement are reported. Synthetic bounded oscillation has zero progress under
this metric; known translation is recovered.

A later `--objective cycle-movement` (E036) requires explicit `--inputs` from
a passed body-wave `fitted_inputs/` pair. It retains the wave qualification and
requires >=1e-4 BL/s in all cycle-mean blocks, >=5x positive baseline, >=2%
forward/path efficiency, phase-origin agreement and numerical refinement. It
is implemented but has not been launched; no body-wave parent is yet qualified.
No automatic propulsion/food/RL stage starts from this wave run.

Local two-proposal pilot completed all six short validation cases with frozen
fields unchanged and zero injected current. Its posterior recruitment is still
far below target, so it correctly returns `passed=false`. This is pipeline
validation, not a learned body wave. See E035 notebook and experiment record.

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

## Body amplitude only, learned-parameter forks (E037, 2026-11-09)

User directs amplitude first, control later. The current stage rewards only
independent sustained joint-bend amplitudes throughout the body. Standing,
opposing-phase and asynchronous body bends are allowed. There is no phase
progression/locking, shared-frequency, PCA, heading, direction, speed or
efficiency goal. Muscle-drive amplitudes are diagnostics only. Forward
simulation still receives zero injected current, no teacher and no body force.

For each joint, measure the minimum p2p excursion across five late windows.
Require repeated peaks/troughs and persistence rather than a static bend, ramp
or fading transient. Summarize by the median joint amplitude in each contiguous
posterior/middle/anterior third; this prevents one hinge dominating a region
and prevents opposite phases from canceling a regional mean. The engineering
qualification target is >=0.003 rad for each regional median. A small repeated
excursion (>1e-5 rad) and sustained neuronal variation (>=0.05 mV) establish
optimization viability, with finite bounded voltages/no clipping or excessive
saturation. The old >=0.01-rad head requirement is removed. These thresholds
are not animal calibration. Scores differ from E035 coherent-mean coverage.

Continue from the learned E035 seed 2 and seed 3 candidates, not the original
head-only checkpoint. Four fresh random search streams (1000–1003) use those
parents round-robin. A warm-start worker copies the entire completed parent,
including selections/history/source/environment, and verifies saved offsets
reconstruct the parent best model exactly. It retains the original source
anchor and 0.0625–16 bounds, instead of widening another16x around the learned
model. Masks must match; checksums/state/selection records must agree. Parent
files remain untouched. Parent metrics are re-evaluated under amplitude-only
fitness; RNG and proposal schedule start fresh. This is not exact optimizer,
RNG or physical-state resume.

`network_anchor.npz` preserves the original bounds anchor; `network_initial.npz`
is the actual learned starting model. Each output includes `warm_start_source/`,
`warm_start.json`, the complete incoming parent, new histories/checkpoints and
proposal/update coverage. Counters distinguish changes from original anchor
from changes made during this new fork. A copy on the same disk is not an
independent backup. Source/environment records stay enabled.

On mbslnx2:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_body_amplitude_mbslnx2.sh
```

The default parents are pinned to the reviewed cohort
`autonomous-wave-20261109T032054849657Z/seed-2` and `/seed-3`; they are not
silently chosen from an unreviewed latest run. Optional first/second shell
arguments select two other completed worker directories. The launcher audits
parent bounds and prints source paths before starting nohup. Four disjoint
four-core workers have a four-hour cap including setup and validation. New
run/log prefix: `autonomous-body-amplitude-*`. Legacy movement/wave launchers
now delegate here; explicit old objectives/source remain for historical replay.

All 9,616 unknown density/chemical/electrical parameters remain eligible. Leak,
capacitance, kinetics/reversals/calcium, anatomy, release settings, NMJs and
body mechanics stay fixed. Keep the original numerical bounds. Validate 300s,
+/-1mV initial conditions, finer neural/body timesteps and held voltage. Only
amplitude qualification/robustness matters here, without frequency or phase
agreement. A passed E037 checkpoint is an amplitude primitive, **not a wave
or locomotion claim**; no later control stage launches automatically.

The local one-proposal warm-start pilot verifies exact incoming parameters,
original bounds and complete parent copying, zero drive/frozen fields, all six
short validation cases and standardized export. It remains unqualified; this
is pipeline verification using a local parent, not the four-hour server seeds.
Software suite:209 passed. Server results are now reported below.

E037 server report (2026-11-09): four-hour amplitude forks1000–1003 all
completed unqualified (exit2),1,862 proposals total. Head amplitude exceeds
0.003rad in4/4 seeds, middle3/4, posterior0/4. Best posterior median is
2.8316e-5rad (seed1001), about106x below target. All active numerical and
perturbation checks retain this deficit; held-voltage scored excursions zero.
Preserve these independent candidates. Next: read-only posterior
voltage→release→muscle→net-drive→joint audit before another training run.
Local initial-voltage readout has posterior connections and nonzero sensitivity;
this does not establish activity in the learned server checkpoints. See
`experiments/records/E037-server-20261109.json`.

E038 server audit reported (2026-11-09): posterior neural voltage/release
oscillations are weak, and posterior median individual-muscle sustained
excursion is zero in all four workers. Deficit exists before the body-mechanics
stage. Working hypothesis: insufficient posterior neural recruitment; channel,
recurrent and NMJ causes remain unresolved. Next proposed curriculum targets
posterior recruitment/amplitude with zero external drive, preserving independent
head/middle candidates. No new training or skill qualification. Full parsed
reports: `experiments/records/E038-server-20261109.json`.

## Short autonomous tail-amplitude trial (E039, 2026-11-09)

Prepared for user launch on mbslnx2:20-minute cohort, four independent four-core
workers, seeds2000–2003. Fork reviewed E037 workers1001/1003 round-robin, retain
original anchors and16x bounds, copy full lineage, start fresh RNG. Zero external
drive; fixed leak/capacitance/kinetics/release/NMJs/body. All9616 unknown neuron
and recurrent-connection parameters remain eligible.90% sparse proposals focus
on anatomically tail-projecting cells and their existing inputs;10% use the
original full-space proposal. Parent head/middle candidates remain preserved.

Score:95% posterior four-joint median sustained-amplitude coverage,5% capped
posterior neural recruitment. The latter is a search heuristic, not an observed
biological target. Qualification requires sustained posterior median bend
>=0.003rad and validity, not neural reward alone. Body-wide metrics are retained
separately. No phase, control, heading or propulsion requirement.20s fit,60s
long validation, voltage perturbations, finer brain/body steps and held-voltage
control; tail-only refinement tolerance25%. A pass is provisional for these
short windows, not long-duration stability, full-body or locomotion evidence.

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_tail_amplitude_mbslnx2.sh
```

The launcher prints the nohup PID and exact log path. Run/log prefix is
`autonomous-tail-amplitude-*`; setup/search/validation share the20-minute cap
with supervisor termination grace. Read `summary.json`, all validation cases,
`tail_amplitude_qualified`, `posterior_recruitment_score` and regional bend
amplitudes before extending training. No server run has yet been evidenced.

Local E039 pipeline verified:two proposals, no accepted changes; all six short
validation cases completed, zero drive and frozen fields preserved. Exit2 is
expected unqualified status. Full software suite212 passed; server trial remains
user-launched. Local raw artifacts archived externally; compact record in
`experiments/records/E039-tail-amplitude.json`.

E039 server result reported (2026-11-09):0/4 short tail qualifications,151
proposals total. Best fitting tail amplitude gain2.63% (seed2000); others~0.3%.
Best long-validation tail median2.905e-5rad remains103x below0.003rad target;
all active checks unqualified, held-voltage excursions zero. Existing head and
middle amplitude remain above target in these checks. No new skill qualified.
Next proposed diagnostic: posterior-cell autonomous oscillation feasibility
within current bounds before extending unchanged fitting. No new run launched.
Exact parsed results: `experiments/records/E039-server-20261109.json`.

## Posterior excitability diagnostic (E040, 2026-11-09)

Executed locally on the cached E039 smoke lineage (not actual server seed2000):
19 original-bounded channel-density profiles,71 tail-projecting cells; repeat
from initial voltages-65,-50,-35mV. Across5,396 primary isolated cell trials,
no sustained late excursion>=0.05mV qualified. All traces finite/bounded.
Selected strong-inward/weak-outward profile also failed30s confirmation,
+/-1mV initial conditions and finer integration. It can depolarize cells into
plateaus; it did not generate oscillations. This finite screen is not a proof
that no oscillator exists within bounds, or that real motor cells must each
oscillate independently. Recurrent strengths were retained, not searched.

Current full302-cell network:3 of71 tail-projecting cells rhythmic; a diagnostic
subcircuit retaining only their mutual edges:0. Full network with strong-inward
profile:1. Current rhythmic tail-projecting cells were SMDVR/SMBVL/SIBVR;
the39 tail-projecting cells in DA/DB/DD/VA/VB/VD/AS classes all remained below
0.05mV (maximum0.003456mV). Direct-NMJ-defined subcircuit is not a complete
functional posterior circuit; it may omit necessary interneurons. No body was
simulated or motion qualified. Candidate not promoted; working model unchanged.

All biases remain zero; no pacing/current introduced. Leak/C/kinetics/release,
NMJs/body fixed. Only diagnostic graph ablations and bounded unknown densities
changed. Exact profiles, traces and lineage are external. Software214 tests
pass; isolation integrator matches a zero-edge version of the existing network.
See `experiments/records/E040-posterior-excitability.json`.

Repeat on reviewed E039 server seed2000 using the background diagnostic:

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_posterior_excitability_mbslnx2.sh
```

It pins that parent, uses four physical CPU cores, prints exact run/log paths,
and runs main plus alternate-basin screens sequentially under nohup. Completion
code0 means diagnostic finished, not oscillator or behavior success. Results
include full per-cell metrics and source/checkpoint hashes. No server launch
is claimed here. Confirm the actual trained parent before changing the model.

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

## Targeted posterior motor recruitment (E041, 2026-11-09)

Option1 trial prepared for user launch:30-minute nohup cohort on mbslnx2,
four four-core workers,seeds5000–5003 from reviewed E039 seeds2000/2001.
Preserve full parent histories,original anchors and16x bounds;fresh RNG/new-score
baseline. All9616 uncertain non-LEAK densities and existing recurrent strengths
remain eligible. External currents zero;leak,C,kinetics,reversals,Ca handling,
release,NMJs and muscle/body mechanics fixed. No teacher or pacing signal.

Select motor classes DA/DB/DD/VA/VB/VD/AS with>=50% of their normalized
NMJ/joint contribution in posterior four joints;retain95% of eligible tail
contribution. Current frozen readout selects26 cells:VD13,VA12,DA09,DD06,VB11,
DD05,DA08,DD04,DB07,VD12,DA07,DB06,VA09,VB10,VA11,VD09,VB07,VB08,VA08,VD10,
VA10,DB05,VD08,VD11,VB09,AS08. Exact names,weights and fractions saved in policy.
This excludes SMD/SIA/AVK cells that dominated the earlier broad auxiliary.
Selection is an anatomical modeling rule,not measured cell-specific physiology.

Score60% tail bend coverage,35% actual posterior projected dorsal-minus-ventral
muscle excursion,5% contribution-weighted sustained motor release. Both physical
median excursions must reach0.003 to qualify. Neural activity alone,passive
bending and common-mode muscle activation cannot pass. Phase/control/propulsion
remain deferred;body-wide qualification reported separately. Release reward
is a bounded heuristic.20s fits,60s long validation,initial-voltage perturbations,
finer brain/body timesteps and held-voltage control. Both tail bend and net-muscle
amplitudes require25% refinement agreement. Any pass is provisional for these
short windows;no long-duration/locomotion claim.

Proposals:10% original global,45% one/two-hop existing chemical paths from the
current network's own release-rhythmic sources,25% incoming-chemical blocks,
20% motor density/gap blocks. At most16 internally active sources,recomputed
after each accepted rollout,saved in search state. Use2x historical proposal
sigma within unchanged original bounds. Paths can include relevant interneurons;
no new topology edges. No requirement that every motor cell oscillate alone.
The local baseline offers such short paths to12/26 targets;others can still
change via incoming/density/gap/global proposals. This is not a complete search
of all recurrent circuits.

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/launch_posterior_motor_mbslnx2.sh
```

Launcher prints exact PID/log path;prefix`autonomous-posterior-motor-*`.
Budget includes setup/search/validation with supervisor termination grace.
Prior skill checkpoints preserved independently;no head/middle retention gate.
Actual server results are now reported below.

Local E041 six-proposal trial completed all six validations,one accepted
nominal-score change;remains unqualified. Tiny amplitude gains are timestep-
sensitive (fine-step net-muscle and motor-release scores0),so no successful
recruitment claim. Qualification gates reject this result. Target units are
0.003rad bending and0.003 dimensionless net-muscle activation.218 tests pass;
server trial remains user-launched. Raw artifacts external,compact record
`experiments/records/E041-posterior-motor.json`.

E041 server result reported (2026-11-09):0/4 qualifications after295 proposals,
but seed5003 shows strong subthreshold gains:tail bend7.17x to0.0001874rad,
net-muscle excursion32.68x to0.0004799. Gains persist under perturbations and
finer brain/body steps (brain-refined bend+0.008%,muscle+1.54%). Remaining
shortfall16x bend and6.25x muscle. Seed5002 offers a smaller stable gain;5000/5001
improve muscle activity without improving bend. Prefer independent continuation
from5003,with5002 backup;preserve original anchors/bounds and zero input. No new
run launched or skill qualified. Exact record:`experiments/records/E041-server-20261109.json`.

## Three-hour posterior motor continuation (E042 plan, 2026-11-09)

User selected180-minute continuation of unchanged E041 objective. Three workers
(seeds6000–6002) fork reviewed E041 seed5003;one(seed6003) forks5002 from
`autonomous-posterior-motor-20261109T222806260061Z`. Fresh RNG,learned parameters
and original anchor/bounds retained;full copied parent lineage.9616 uncertain
channel/recurrent parameters eligible;zero external drive and fixed fields.
No phase/control/propulsion or retention constraint added;independent parents
preserved. Budget includes setup/search/validation with termination grace;
qualification can stop a worker earlier.

Fit20s unchanged,long validation extended to300s,other perturbation/refinement/
held-voltage cases20s. Same physical muscle AND bend thresholds,25% refinement
agreement. Source/environment and exact parent checksums recorded at launch.
Preflight requires completed posterior-motor parents and verified hashes.
Worker/supervisor algorithm label remains E041;E042 records this continuation.

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/continue_posterior_motor_mbslnx2.sh
```

Runs under nohup,prints PID/exact log. Logs:`autonomous-posterior-motor-continue-*.log`;
cohort folders retain`autonomous-posterior-motor-*`. Shell syntax and command
construction verified with a stub interpreter (no real preflight or training).
Training code unchanged;prior218-test suite applies to that implementation.
No server launch or new qualification claimed. Plan record:
`experiments/records/E042-continuation-plan.json`.

E042 server continuation reported (2026-11-09):2008 proposals,0/4 full passes,
but best6002 increases tail bending7.28x to0.00136388rad and net-muscle activity
7.34x to0.00352342. Net-muscle amplitude exceeds0.003 in all five active checks;
tail bending remains~2.2x below0.003rad target. Long validation0.00139634rad/
0.00361156 muscle;brain refinement changes bend-1.84%,muscle-1.49%. All four
workers improve,held-voltage scored excursions0. Prefer future continuation from
6002,preserve6003 alternate lineage/other backups. Same objective,original bounds,
zero drive;no full skill qualified and no new run launched. Exact record:
`experiments/records/E042-server-20261109.json`.

## Eight-hour posterior motor continuation (E043 plan, 2026-11-09)

User requested another8hours. Current continuation launcher now pins E042
cohort`autonomous-posterior-motor-20261109T232143970960Z`:seeds7000–7002 fork
6002,seed7003 forks6003. Four independent four-core searches,fresh RNG,learned
parameters and original anchor16x bounds retained;full parent copies and histories.
Same E041 objective,all9616 uncertain existing channel/recurrent parameters
eligible,zero external drive,fixed leak/C/kinetics/release/NMJs/body/anatomy.
No phase,heading,RL or propulsion objective and no parent overwrite.

Budget480minutes perworker incl setup/search/validation;supervisor grace can
extend slightly,qualification may end a worker sooner. Fits20s,long validation
300s,other perturbation/refinement/held-voltage checks20s,all unchanged from E042.
Both tail bend>=0.003rad and net-muscle activation excursion>=0.003 required in
active checks;25% refinement agreement,held-voltage quiet. Parent6002 meets the
muscle-amplitude component but remains below bend target;no full pass assumed.

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/continue_posterior_motor_mbslnx2.sh
```

Nohup launcher prints PID/exact log;log prefix remains
`autonomous-posterior-motor-continue-*`,cohort folders`autonomous-posterior-motor-*`.
Same script path now means E043 eight-hour plan;historical E042 three-hour
launcher is preserved in commit`cb99867` and external records. Preflight checks
completed posterior-motor parents and hashes before launch. Algorithm label
remains E041;E043 identifies this operational continuation.

Shell syntax and actual generated argv verified with an interpreter/preflight
shim;no simulation or real parent validation performed locally. Training code
unchanged,no redundant full-suite run. No server launch or new qualification
claimed. Plan:`experiments/records/E043-continuation-plan.json`.

E043 duration amended (2026-11-10):latest user instruction is3hours/180min,
superseding the8-hour preparation and intervening4-hour request. Same launcher,
6002x3+6003x1 parents,seeds7000–7003,20s fit/300s long validation,original bounds
and zero drive. Eight-hour source preserved in commit`30ff375`. No E043 server
launch confirmed;editing launcher does not change an existing process.

## Qualified autonomous tail-amplitude primitive (E043, 2026-11-10)

User-reported server seed7001 passed all predeclared posterior amplitude checks,
exit0/passedTrue. Tail median bend0.00303739rad and net-muscle excursion0.00646743
exceed0.003 targets. Long validation,initial-voltage perturbations and finer
brain/body steps all qualify;held-voltage scored excursions0. Minimum active
bend0.00303260rad (1.09% margin);brain refinement changes bend-0.158%,muscle
+0.036%. One of four candidates qualifies;cohort failure does not invalidate it.

Register this as an autonomous posterior amplitude primitive only. No phase,
propulsion,swimming,whole-body amplitude or natural-animal claim. Same all302-
neuron zero-drive dynamics,frozen readout/body and original bounds. Head/middle
retention was not required and is not shown in the pasted summary. Preserve
winner before future skill work;next inspect full-body amplitudes/trajectories.

Winner remains on mbslnx2 at
`/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z/seed-7001`.
Actual checkpoint/source hashes and raw artifacts have not been independently
retrieved or backed up;no new inference asset bundled in Git. Existing demo/
`autonomous_wiggle_v1` assets are unchanged. Parsed evidence:
`experiments/records/E043-server-20261110.json`. No new training launched.

## Read-only winner full-body assessment (E044, 2026-11-10)

Prepared for E043 winner7001 from
`autonomous-posterior-motor-20261110T140125786528Z`. This reads saved trajectories;
no simulation,training,pacing or parameter/checkpoint edits. It verifies qualified
parent metadata/checksums,records the actual winner hash,copies compact selected
parameters/matching readout/reference metadata externally,and checks source hashes
again after all six saved validation cases. This parameter reference on the same
server is not an independent backup of raw traces or full recursive history.

Report sustained per-joint and filtered muscle-drive amplitudes,regional medians
[tail,middle,head],and each case's recorded body-wide qualification. Reapply the
existing E037 full-body25% refinement tolerance across all three regions against
best-training values plus quiet held-voltage control;passing all per-case regional
targets alone does not establish this separate refinement check. Thresholds and
training acceptance criteria unchanged. Additional descriptive outputs:shape
PCA,spectral phase,equal-reference-joint-phase translation and shape-closure error.
None automatically qualifies locomotion or biological behavior.

Outputs external:report.json,artifact_manifest.json,parameter_reference/,six
scientific trajectory.png figures,cycle samples,and motion.html. The HTML is
standalone,default true-scale12-rod geometry,equal axes,tail/head labels,pause/
scrub/speed controls. Right centered view can explicitly amplify deviations from
late mean posture10/20x;left world view stays true scale. Preview capped600 frames
of final5s,with subsampling noted;all metrics use full arrays. Stroke width is
schematic,not physical anatomy. Position origin is preview-start center.

```bash
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/assess_winner_mbslnx2.sh
```

Launcher pins7001,uses four CPU cores under nohup,prints results/log/HTML/PNG
paths. It checks the dedicated environment has matplotlib. The analysis imports
no JAX/GPU backend;unchanged sustained detector was moved to a pure NumPy/SciPy
module and re-exported at its old path. Source dynamics/qualification math intact.
Completion code0 means analysis completed,not a new behavioral pass. Expected
runtime is a few minutes depending on artifact I/O. User launches on server.

Local pipeline tested using an explicitly allowed UNQUALIFIED cached E041 pilot,
not7001;default rejects such a source. Actual winner head/middle/whole-body
results remain pending until server report. Do not interpret local demo plots
as the trained winner. Compact record:`experiments/records/E044-full-body-assessment.json`.

E044 local verification complete:223 tests pass,all six pilot cases processed,
source hashes unchanged,preview controls executed with a JavaScript DOM/canvas
shim and local scientific PNG inspected. Actual7001 server assessment pending.
Local unqualified pilot results are pipeline evidence only.

E044 server launch reported(2026-11-10):qualified winner7001 preflight verified
SHA256`417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14`.
Output`full-body-assessment-20261110T174954Z-687519`,PID687526,CPUs0–3.
Completion and full-body results not yet reported. No model changes. Exact
paths:`experiments/records/E044-server-launch-20261110.json`.

## Qualified autonomous distributed amplitude (E044, 2026-11-10)

Server assessment of winner7001 reports full_body_amplitude_assay_passTrue.
All five active cases exceed the0.003rad regional bend target and pass stored
full-body checks;existing E037 regional refinement and quiet-control recheck
also passes. Minimum[tail,middle,head]medians:[0.00303260,0.03331643,0.01103545]rad.
Held-voltage excursions zero. Register MOT-17 on the same preserved7001
checkpoint;head/middle amplitude retained within tested conditions. These are
regional medians,not a guarantee for every joint or muscle.

PC1 explains~99.9% of shape variance;near-single-mode dynamics suggest phase
coordination is the next issue,but PCA alone does not establish propagation.
Reported same-reference-phase drift~−1e-5BL/s does not qualify useful propulsion;
full-shape closure is not in pasted output. No coordinated traveling-wave,
locomotion,command control or animal-fidelity claim. No new training/model edits.

Actual motion.html,long/trajectory.png and full report remain at
`/home/carsonc/r-elegans-results/full-body-assessment-20261110T174954Z-687519`.
Only user-console results have been received;actual figures/reference files
not independently downloaded/viewed. Next inspect/export that bundle before
fitting gradual phase progression with amplitude retention. Source checkpoint
hash remains417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14.
Exact parsed report:`experiments/records/E044-server-20261110.json`.

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

## Medium-physics audit status (E045, 2026-11-10)

Current trained winner uses the12-rod SI liquid factory:1mm length,40um
constant radius,water-like viscosity0.001Pa s,Lighthill local RFT. Code/source
comparison finds the expected per-length coefficients0.00329041 parallel and
0.00521525 perpendicular kg/(m s),ratio1.58498. Whole-body coefficients are
those values times0.001m. Coupled torque-driven mechanics impose zero net
force/torque and include external drag in shape dynamics.8 targeted existing
physics tests pass (units,SPD/precision,residual balance,load response,Re).
These are consistency checks,not quantitative experimental calibration.

Status:physically dimensioned leading-order UNBOUNDED NEWTONIAN approximation.
No explicit Petri-dish bottom/wall/free-surface flow,nonlocal hydrodynamic
interactions,body taper or self-contact. Agar mode is a different effective
surface-drag closure,not resolved gel/adhesion/lubrication. Current liquid
winner does not establish physical crawling or a real Petri-dish environment.
Water retains ordinary viscosity;microscale viscous dominance is a scale effect.

Fixed drag wavelength1.5mm is a reference parameter and is not measured from
the learned standing-like shape. Passive viscosity uses a reported upper bound;
active muscle moment scale is a preferred-curvature modeling closure. Neither
is a measured complete actuator/medium calibration. No matched observed
waveform→predicted force/speed benchmark or12→24→48 spatial convergence has
been established. Any such refinement must keep physical curvature and muscle
interpolation consistent,not just increase the segment count. Current neural
branch lacks body-to-brain feedback,so changing drag does not adapt its neural
rhythm through proprioception. That is separate from the force solver.

For scale only,finite differences of copied true-world preview nodes estimate
maximum sampled speed0.194mm/s,Re~0.00777(radius)or0.194(length);these are not
full-resolution fluid velocities or a hydrodynamic-error bound. They support
a viscous-dominated local approximation without certifying all inertial/flow
effects. No new integration or trained parameter changes.

Primary-source context:Boyle etal(2012) uses the same Lighthill convention
(https://pmc.ncbi.nlm.nih.gov/articles/PMC3296079/). Fang-Yen etal(2010) measured
~9.7x transverse drag in a150um glass chamber compared with bulk fluid
(https://www.wenlab.org/wordpress/3.8/wp-content/uploads/2013/12/proc_natl_acad_sci_usa_2010_fang-yen.pdf).
That factor is geometry-specific;do not blindly multiply bulk drag or viscosity.

Recommendation:before quantitative locomotion claims or further medium-specific
training,benchmark body/medium response to a prescribed measured waveform,
check resolution/time convergence and choose bulk versus wall/surface geometry.
Keep7001 as an amplitude primitive under its existing model. Audit record:
`experiments/records/E045-medium-physics-audit.json`.

## Wet-surface environment assessment (E046, 2026-11-10)

The user selects moist compliant substrate locomotion as the primary
natural-behavior target. Standard 1.7% wet NGM agar is the first controlled
proxy. This does not identify the material properties of wild decaying fruit.
The original liquid winner7001 is preserved with the same SHA256 and retains
its liquid-only E043/E044 amplitude qualification.

An independent NumPy float64 reference body now implements published nonlinear
wet-agar towing-force fits (Rabets2014,doi10.1016/j.bpj.2014.09.006). It is
available through `scripts/assess_wet_surface.py`, with a nohup mbslnx2 launcher
`scripts/assess_wet_surface_mbslnx2.sh`. This is a reference assessment, not yet
an environment replacement in the JAX trainer. The constant-drag agar and
liquid factories remain available for comparison/reproducibility.

Local execution:20s initial comparison, then60s comparison of8 cases, both exit0.
The pinned zero-current full network generates endogenous muscle commands once;
identical commands are replayed into each body because this branch has no body
to brain feedback. No driven teacher, fitted force multiplier or new policy.
The liquid replay differs from the original JAX joints by at most1.01e-7rad,
and recovers regional sustained amplitudes~.003038/.033385/.011060rad.

At60s, the nonlinear wet-agar case does not pass repeated/sustained amplitude
checks in any region. Raw late-half regional excursions~.01473/.01235/.01235rad
are dominated by slow posture changes, not established cyclic bending. The
held-voltage control also has slow relaxation; neither its raw motion nor raw
centroid drift should count as a skill. The plotted FFT/phase on such drifting
records is descriptive and must not be interpreted as an oscillation frequency
or traveling gait. Small oscillatory components are visible, so a zero
qualification value does not mean mathematically constant trajectories.

Maximum dimensionless nonlinear force residual<1e-8. Half body timestep changes
raw regional excursions by<.0025%; four-to-eight-point rod quadrature changes
by<.031%. These are numerical checks, not12-to24-rod spatial convergence.
Low-speed regularization matters much more:0.01/0.1/1um/s smoothing changes raw
regional excursions by~15–49%;62% of late default velocity components lie below
0.1um/s. These speeds probe an unmeasured extrapolation of the published fit.
Do not infer natural-substrate accuracy or channel insufficiency from this
single effective-law comparison. All227 tests pass, including4 new reference
physics tests. Exact records and raw trajectories remain external; see E046.

Next:retain7001; use this reference to verify a future fast wet-surface trainer,
address low-speed force uncertainty/settling, then build autonomous surface
amplitude before phase/propulsion. No surface-training run or trained checkpoint
has been produced by this assessment.

## Required environments and action initiation (user clarification, 2026-11-10)

Both water swimming and moist-substrate crawling are required capabilities.
This supersedes wording that made wet-surface crawling the sole primary target;
the E046 historical measurements and liquid-only qualifications are unchanged.
Each medium needs its own physics validation, autonomous amplitude, propagated
wave and sustained useful displacement checks before a locomotion claim.

The user also identifies action initiation, stopping, rest and restarting as
necessary. In the current conductance-network assay, integration begins from
fixed.initial_voltage_mV with initialized gates/calcium; the fitted recurrent
and channel dynamics generate sustained oscillations. There is no immersion
sensor event or biological swim-on command. Altering the medium changes body
loading, not the neural rhythm through sensory/proprioceptive feedback. This
branch has no explicit serotonin/dopamine/FLP-11 signaling state or qualified
neural stop/restart mechanism. Presence of a neuron such as RIS in the anatomical
network does not establish its measured neuromodulatory function in this model.

Biological context:swim initiation and maintenance depend on sensory context
and neuromodulation; serotonin promotes crawl-to-swim transition in tested
assays (Vidal-Gadea et al.2011,doi10.1073/pnas.1108673108). Extended swimming
can alternate with quiescence even in a uniform liquid environment
(Ghosh and Emmons2008,doi10.1242/jeb.023606). RIS activation can suppress motor
oscillations while retaining muscle tone; GABA and FLP-11 contribute differently
to induction and maintenance of stopping (Steuer Costa et al.2019,
doi10.1038/s41467-019-12098-5). These findings do not establish one universal
water sensor or one circuit explaining all resting states.

Next requirement:build a controllable autonomous motor primitive with reproducible
start, sustained movement, pause and restart in continuous neural/body state.
A physiological sensory cue can initiate a bout; a periodic gait teacher is
not required. Subsequent internal/sensory regulation should govern bout duration,
rest and environment adaptation. Measure transition latency, residual shape
velocity, restart reliability and retained movement, without suppressing the
muscle output, freezing voltage or resetting the network to simulate stopping.
Separate checkpoints are acceptable intermediate skills under H1.1; eventual
one-animal adaptation must use one parameter set without manual swaps. No new
start/stop circuit or training has been implemented by this requirements update.

## Neuron roles and sensory-input status (E047, 2026-11-10)

The full model contains the canonical sensory, interneuron and motor-neuron
identities. A new source-linked registry, `r_elegans/neuron_roles.py`, annotates
a selected locomotion/sensory subset without changing simulation parameters.
It records overlapping functional roles and canonical indices. This is not a
complete cell-type atlas:113 neurons are annotated,189 remain unreviewed.
Counts below refer only to the curated subset and are not totals for the worm.

| Selected function | Neurons/classes | Curated count |
| --- | --- | --- |
| Odor sensing | AWA, AWC, AWB bilateral pairs | 6 |
| Soluble chemical sensing | ASE bilateral pair | 2 |
| Touch receptors | ALML/R, PLML/R, AVM, PVM | 6 |
| Forward-associated premotor control | AVBL/R, PVCL/R | 4 |
| Reverse-associated premotor control | AVAL/R, AVDL/R, AVEL/R | 6 |
| Selected sensory integration | AIYL/R, AIBL/R, AIZL/R, RIAL/R | 8 |
| Stop/state-control candidate | RIS | 1 |
| A-type body motor | DA01–09, VA01–12 | 21 |
| B-type body motor | DB01–07, VB01–11 | 18 |
| D-type body motor | DD01–06, VD01–13 | 19 |
| AS body motor | AS01–11 | 11 |
| Selected head motor | RMD and SMD classes | 10 |
| Proprioceptive roles, overlapping other rows | B-type motor neurons, DVA, SMDDL/R | 21 |

Motor versus control is not an exclusive division. B-type motor neurons also
transduce body-bending feedback; DVA is stretch-sensitive; SMDD contributes
proprioceptive steering. Premotor AVB can recruit B-type oscillations in the
cited experiments. The registry lists biological evidence, not measured causal
roles in the fitted worm. Existing direct NMJ edges are reported separately;
a few NMJs from a known interneuron do not justify relabeling it as exclusively
motor or removing its recurrent connections. No neuron is excluded or frozen.

Important implementation distinction:

| Branch | Chemical input into neurons | Physical body-to-brain feedback |
| --- | --- | --- |
| Current autonomous conductance-network/body | None; sensory cells evolve within the recurrent network | None |
| Older normalized full-network actor, chemical-input mode | Engineered concentration/adaptation/derivative features to AWCL/R and ASEL/R | None |
| Older 14-cell steering subcircuit | Engineered chemical features plus sin/cos of synthetic gait phase | None; phase input is not measured bending |

The full normalized actor also supports an explicit applied-current mode for
experiments. It does not establish receptor-level sensing. The older bundled
food-finding demo uses its documented external sensory controller/motor teacher;
that demonstration is not evidence that the autonomous channel brain senses
food. AWC and ASE currently share generic features in the older implementations;
odor-selective and soluble-cue transduction have not been independently fitted.
Transferred AWCON ion-channel priors are not an olfactory receptor mechanism.

A future embodied sensory loop must sample specified chemicals at the sensory
location, transduce them into identified neurons, and feed actual body bending
back into the appropriate motor/proprioceptive cells. Concentration history,
curvature, strain and contact are candidate physical observations; source
coordinates, desired heading and an analytic gait clock are not physiological
sensory inputs. Sensory adaptation, gains, spatial receptive fields and channel
transduction require documented priors or fits, rather than arbitrary signs.

Next causal assays should distinguish sensory responses, AVB/RIS-dependent
state changes, motor rhythms, muscle activation and actual movement. An AVB
perturbation can test recruitment of B-type dynamics; a RIS perturbation tests
stopping only if its relevant mechanisms are represented. These are mechanistic
probes, not normal sensory behavior or a license to clamp outputs. No sensory
loop or neuromodulation was implemented during this read-only annotation audit.

Primary evidence: [odor detection](https://pubmed.ncbi.nlm.nih.gov/8348618/),
[soluble-cue sensing](https://pubmed.ncbi.nlm.nih.gov/1660283/),
[premotor/touch circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565008/),
[AVB recruitment of B-type dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC5948959/),
[B-type proprioception](https://doi.org/10.1016/j.neuron.2012.08.039),
[DVA stretch sensing](https://doi.org/10.1038/nature04538),
[SMDD proprioception](https://doi.org/10.1371/journal.pbio.2004929).

## Embodied sensory connections (E048, 2026-11-10)

The missing world/body inputs are now implemented in a separate opt-in branch.
`body/closed_loop.py` connects the full 302-neuron conductance network, existing
recurrent chemical/electrical edges, anatomical NMJs, 95 muscles and the actual
body. Existing control-to-motor edges are retained. The old `channel_body_chunk`
entry point remains the historical no-input model; original checkpoints are not
rewritten. This supersedes E047's missing-input status only for the new branch.

The causal interval is:

```text
head position -> local chemicals -> adaptive sensory conductances -> neurons
body curvature -> proprioceptive conductances ---------------------> neurons
neurons -> existing recurrent connections/NMJs -> 95 muscles -> body
                                                               |
                                               next interval <-+
```

Only sampled concentrations and local/body-bending measures reach neurons.
No desired gait, phase clock, desired direction, source bearing or distance is
supplied. The world uses meters, concentrations use micromolar, conductances
use nS and currents use pA. Sensory channels contribute `g*(E-V)` inside the
membrane solve, not forced voltage or direct muscle commands. The injected
conductances are held during each body interval while neuronal gates evolve.
Intrinsic bias remains zero in pinned winner7001. Physiological-input proxies
are distinct from the removed artificial periodic pacing/bias currents.

`body/embodied_sensory.py` currently provides:

| Pathway | Observation and routing | Status |
| --- | --- | --- |
| AWA pair | Diacetyl test channel, adapting positive response to increases | Effective input closure; unfitted |
| AWC pair | Benzaldehyde test channel, inhibitory input on increase and excitatory rebound on decrease | Functional sign proxy, not literal TAX-2/4/cGMP kinetics |
| ASEL / ASER | Separate salt channel; increase / decrease respectively | Adaptive ON/OFF proxy, unfitted |
| DB / VB | Actual anterior-weighted dorsal / ventral curvature respectively | Experimental receptive fields derived from anatomical NMJ output territories |
| DVA | Global normalized bending magnitude | Explicit stretch-related proxy, not resolved neurite strain/TRP-4 kinetics |
| SMDDL / SMDDR | Optional anterior dorsal-bend input | Available; default gain zero pending qualification |

All sensors preserve their receptor/adaptation/bending memory across streamed
chunks. Initial chemical memory is equilibrated to the local concentration, so
initialization does not fabricate an odor-on pulse. Paired sensory cells share
one head endpoint; there is no resolved left/right head geometry or fixed
AWC ON/OFF subtype assignment. The model is not an exhaustive sensory system:
AWB, touch, contact/pressure, water detection, gases and temperature transduction
are not added by this implementation. Monoamine and neuropeptide state control,
including physiological RIS-mediated stopping, remains separate work.

The new sensory channels are effective conductances. They do not add direct
calcium influx; voltage-dependent calcium entry remains through the existing
channel library. Receptor biochemistry and calcium responses need comparison
with recordings before physiological claims. Numerical defaults are exposed
in `EmbodiedSensoryConfig`, not calibrated constants:

- Chemical and B-type maximum gains: 0.005 nS; DVA: 0.002 nS; SMDD: 0 nS.
- Receptor, adaptation and bend-filter times: 0.1, 2 and 0.02 seconds.
- Concentration normalization: 1, 1 and 50,000 micromolar for the three channels;
  these are normalization choices, not measured dissociation constants.
- Curvature normalization: 2,000 per meter; anterior offset 0.1 body lengths,
  receptive width 0.15 body lengths. Actual receptor locations remain unknown.
- Effective excitatory/inhibitory reversals: 0 / -80 mV, exploratory choices.

The test world is a prescribed Gaussian concentration field with scheduled
bath exposure/washout. It is not a calibrated bacterial odor plume, finite-dish
chemical transport or a nutrient-consumption model. Its coordinates and stimulus
schedule are world parameters; the brain receives local samples, not the schedule
or source location. Food finding and immersion-triggered swimming are not claimed.

`diagnose_embodied_senses.py` evaluates the fixed, hash-pinned 7001 winner in
water and the nonlinear wet-agar reference. The wet body is integrated between
every neural/sensory update, using `surface_body_step`; command replay after a
neural recording would be invalid with proprioception. The native JAX liquid
path supports the same signals; the wet path is a slower NumPy reference, not a
fast differentiable wet-surface trainer. E046 low-speed drag uncertainty remains.

Primary evidence for signs and routing:
[AWA responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC5045890/),
[AWC odor removal](https://doi.org/10.1038/nature06292),
[ASE asymmetry](https://doi.org/10.1038/nature06927),
[B-type proprioception](https://doi.org/10.1016/j.neuron.2012.08.039),
[DVA stretch sensing](https://doi.org/10.1038/nature04538),
[SMDD feedback](https://doi.org/10.1371/journal.pbio.2004929).
These sources support functional hypotheses, not the numerical defaults above.

E048 local evidence: 236 tests pass. Both the initial 5s and subsequent 20s
fixed-winner diagnostics complete all ten cases with finite states. Compared
with chemical-blocked controls, maximum neural differences are 3.31mV in water
and 3.44mV on wet agar, with smaller but nonzero muscle/body differences.
Blocking proprioception changes muscles by up to 0.0319 / 0.0104 activation
units and joints by 0.0166 / 0.00316rad in water / wet agar respectively.
These are causal routing effects, not behavioral success scores.
Half-timestep endpoint differences are recorded, including stimulus discretization;
no numerical or biological accuracy threshold is inferred after the fact.
No-input liquid qualification stays with its original entry point and checkpoint.
The new model has not been fitted or qualified for food finding, start/stop,
swimming or crawling. Full evidence: E048 in the lab notebook and external archive.

Trace timing is causal: sensory geometry/concentration are sampled from the
interval-start body, receptor/filter state advances over that interval, and the
resulting conductances are held during its neural substeps. Logged input current
uses those conductances and the pre-update voltage. Brain/body samples are at
the interval end. Use the saved states and this convention when comparing
stimulus timing or fitting recordings; do not silently align all fields as
simultaneous endpoint measurements.

### E048 server report and motor-effect qualification (2026-11-10)

The user reports `status=completed` for mbslnx2 run
`/home/carsonc/r-elegans-results/embodied-senses-20261110T220423Z-694360`.
Six reported block comparisons closely match local aggregate effects (largest
relative difference across their reported voltage/muscle/joint maxima: 1.13%).
This is descriptive agreement of summaries; server configuration, source,
versions, hashes and full trajectories have not been independently inspected.

Chemical-block effects: about 3.31mV / 3.44mV maximum neural differences in
water / wet agar, but only 0.000105 / 0.000109 muscle activation differences and
0.0000223 / 0.00000942rad joint differences. Proprioceptive-block effects are
larger: 0.0319 / 0.0104 muscle differences and 0.0166 / 0.00316rad joint differences.
This establishes neither biological importance rankings nor chemical navigation.

Crucial numerical limitation: reported coarse/fine joint differences are
0.001019rad in water and 0.000153rad in wet agar, about 46 / 16 times the
reported chemical-induced joint difference. These sensitivities are not formal
error bars, but they prevent treating the tiny chemical body effect as robust
behavioral evidence without a matched refinement of both active and blocked
runs. Reported coarse/fine neural maxima also differ from the local maxima;
there is no bitwise cross-platform agreement or formal convergence claim.

Decision: retain the wiring as experimental infrastructure. Next work should
check the chemical-to-motor contrast under matched timestep refinements and
probe premotor recruitment/release as a separate start/stop assay. No new
training or behavior qualification follows from this server report. See
`experiments/records/E048-server-20261110.json` and the lab notebook.

### Proprioceptive genetics qualifies the current proxies (2026-11-10)

Knockout/loss-of-function and neuron-manipulation studies motivate using
proprioception as a mechanism-level validation target. trp-4 mutants show
exaggerated bending; trp-1 trp-2 double mutants can move in biased circles;
B-type perturbations can disrupt propagation. Successful motion or larger
amplitude alone is therefore insufficient evidence of a normal motor pattern.
See the source-linked perturbation backlog in TRAINING_PROTOCOLS.md.

A newer mechanistic constraint from [Das et al.2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8448456/)
is particularly relevant: DVA shows directional/compression-versus-stretch
mechanotransduction, opposing TRP-4/TWK-16 effects and residual curvature-related
activity in trp-4 mutants. Our E048 global positive bending-magnitude input into
DVA represents none of that channel specificity or spatial compartmentalization.
It remains an exploratory feedback proxy. Likewise, default SMDD feedback is
zero and has no independent TRP-1/TRP-2 states. A gain deletion is a functional
pathway lesion, not a faithful genetic knockout or whole-neuron ablation.
No parameters or mechanisms were changed by this literature review.

## Unified data-constrained model objective (2026-11-10)

User clarification:the model should reproduce the effects of biological
perturbations and ultimately match all existing compatible evidence together.
The target is one embodied reference worm model across neural activity,
electrophysiology,sensory responses,mechanics,normal behavior and interventions.
The target is not yet achieved. Both water swimming and moist-substrate crawling
remain required. Incremental skills and independent parent checkpoints remain
useful under H1.1;eventual cross-data validation must use shared parameters,
with explicit experimental conditions and mechanistically defined interventions.

`experiments/data_targets_v1.json` now records seven data families,their local
source readiness,missing model capabilities and admission/validation requirements.
It is an initial nonexhaustive inventory,not a joint optimizer or a claim that
all literature has been collected. Known parameters remain fixed by documented
evidence;unknowns require bounds/priors. Measurement and individual variability
set tolerances;contradictory observations remain part of the evidence ledger.

Local read-only source audit reverified all22 artifacts in the core manifest
and all10 in the physiology manifest. The extracted Atanas/Kim candidate
2022-06-14-01 has1600 frames x109 raw fluorescence channels,with paired behavior
arrays including velocity,angular velocity,head angle,reversals and body-angle
representations. At confidence>=4,55 canonical identities are unique and within
trace bounds;this is metadata eligibility,not a final usable-trace count or
proof of whole-brain coverage. Trace quality,conditions,physical units,time
alignment and body-angle mapping remain admission work. Recorded pumping lies
outside the95 body-wall-muscle model. A larger archive exists but was not fully
inventoried or rehashed in this audit;the functional raw-data directory is empty.

This paired recording is a concrete candidate for joint neural/body fitting.
The existing calcium fitter fits selected neural traces;it does not jointly fit
these paired behavior arrays or all data families. No training/validation/test
split has been frozen for the new objective. Prefer holdouts by animal/trial;
a temporal split from a single recording does not establish between-animal
prediction. Observation normalization/calibration must use training data only.

Next:admit paired recordings and additional animals with units/identity/condition
checks,freeze grouped splits,and build shared-model losses and per-family reports.
Use knockout/rescue and untrained conditions as causal validation targets where
the mechanism exists. Do not refit a different network for every experiment or
hide missing biology behind unconstrained time-varying currents. Failure across
compatible data may identify missing mechanisms rather than insufficient run
length. No new model optimization or dataset acquisition occurred in this update.

Server E050 update (user-reported, 2026-11-10): 34 proposals reduced the training
objective from47.930 to47.514 (~0.87%); the reduction persisted on refinement.
Bend fluctuation remains much smaller than recorded. The reported checkpoint
is c435ee69645f27163af21ef5590b05a30151a38df7389a6f2fe70ef4db1ee481;
full artifacts have not been independently retrieved. A completed-search
continuation wrapper now preserves offsets/RNG and original bounds for longer
runs. No new biological qualification or bundled checkpoint promotion.

E075 planned read-only follow-up (2026-11-14):
`sh scripts/launch_control_signal_audit_mbslnx2.sh` analyzes saved E074 cue/sham
trajectories and search histories, with a20min ceiling under nohup. No new worm
simulation, training or model changes. The launcher prints report/log paths.
Report includes neuron voltages/calcium, release ranges, sampled inferred
synaptic drive, offline intended chemical replay, source/target topology and
parameter/history diagnostics. Reconstruction is labeled separately from saved
observations; this audit confers no behavior or causal-control qualification.
Preserve the entire source run and recursive input capsule as dependencies.
See experiments/records/E075-server-summary.json for the reported outcome.
