# Training and experiment schedule

2026-11-24 repository cleanup:46retired one-experiment server shell recipes
were removed; reusable Python runners/reports, tests, assets and all current
E121-E127 tools remain. Some dated launch commands below are historical recipes.
Removed files and recovery revisions are listed in
[the cleanup record](records/repository-cleanup-20261124.json).
Historical reproduction needs the matching recorded source/environment/inputs;
restoring an old wrapper alone does not reproduce an experiment.


Generated from `experiments/records/E*-plan.json` and outcome records by `python -m scripts.build_training_schedule`.
Each row is what was planned (command, budget, parent, trainable fields) and, where recorded, the outcome. A plan is not evidence that
the run happened; outcomes link to the user-reported summaries. Narrative, audits and negative results are in `LAB_NOTEBOOK.md`;
protocol design is in `TRAINING_PROTOCOLS.md`; curricula and policies are the JSON files in `experiments/`.

32 experiments have a plan record. Earlier work (before the plan-record convention) is documented only in the notebook.

## E042 (-)

- Question: -
- Method: -
- Parent: -
- Trainable: -
- Fixed: ['LEAK', 'capacitance', 'kinetics', 'reversals', 'calcium handling', 'release parameters', 'NMJs', 'muscle/body mechanics', 'anatomical masks']
- Command: `sh scripts/continue_posterior_motor_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: -

## E043 (-)

- Question: -
- Method: -
- Parent: -
- Trainable: -
- Fixed: ['LEAK', 'capacitance', 'kinetics', 'reversals', 'calcium handling', 'release parameters', 'NMJs', 'muscle/body mechanics', 'anatomical masks']
- Command: `sh scripts/continue_posterior_motor_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: -

## E063 (-)

- Question: Can the full network sustain repeated post-cue joint bends without periodic pacing?
- Method: -
- Parent: 16146cc539d47277cb2230e1cfced9a00282d9dd8e4e3b18112fde46515a042b
- Trainable: ['unknown non-leak densities', 'existing chemical strengths', 'symmetric electrical pair strengths', 'chemical/B/DVA gain multipliers']
- Fixed: ['anatomical topology', 'NMJ', 'muscle/body mechanics', 'leak/capacitance/reversal potentials', 'channel/reporter/receptor kinetics', 'sensory signs/maps']
- Command: `sh scripts/launch_alternation64_mbslnx2.sh 60`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: [E063-server-summary.json](records/E063-server-summary.json)

## E072 (2026-11-13)

- Question: Sensory stop-rest-restart of qualified water wiggling
- Method: -
- Parent: -
- Trainable: -
- Fixed: -
- Command: `sh scripts/launch_controlled_swimming_mbslnx2.sh`
- Report: `-`
- Budget: 240 minutes total shared E072/E073 including setup, validation and cleanup
- Acceptance: -
- Outcome record: [E072-server-summary.json](records/E072-server-summary.json)

## E073 (2026-11-13)

- Question: Physical forward swimming retaining sensory control
- Method: -
- Parent: -
- Trainable: -
- Fixed: -
- Command: `sh scripts/launch_controlled_swimming_mbslnx2.sh`
- Report: `-`
- Budget: 240 minutes total shared E072/E073 including setup, validation and cleanup
- Acceptance: -
- Outcome record: -

## E074 (-)

- Question: Can extended sensory-control search produce validated stop/rest/restart and then controlled swimming?
- Method: Fresh E072/E073 search with adaptive time allocation; validate best control candidate even on training failure; swimming still fully gated
- Parent: -
- Trainable: Control1318 sensory/control coordinates, swimming9616 full unknown density/connection coordinates
- Fixed: Original topology/leak/capacitance/channel kinetics/reversals/release/NMJ/muscles/water mechanics/sensory routing/kinetics/gains/physical clock
- Command: `sh scripts/launch_controlled_swimming_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: Unchanged E072/E073 exact-checkpoint full control and physical swimming gates; no threshold relaxation
- Outcome record: [E074-server-summary.json](records/E074-server-summary.json)

## E075 (2026-11-14)

- Question: Where does the E074 cue response attenuate between ligand transduction, sensory cells, control interneurons and rhythmic motor output?
- Method: Read saved nominal/no-cue/chemical-blocked/stop-blocked/start-blocked trajectories; no forward neural or mechanical simulation; offline intended chemical replay and sampled synaptic-current reconstruction
- Parent: 2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f
- Trainable: -
- Fixed: -
- Command: `sh scripts/launch_control_signal_audit_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: completed_user_reported ([E075-server-summary.json](records/E075-server-summary.json))

## E076 (2026-11-14)

- Question: Can candidate interneurons reversibly suppress ongoing body motion at the existing model settings?
- Method: 17 diagnostic forward simulations: unchanged sham plus4 interneuron groups x2 strengths x2 reversals; refine strongest baseline-safe suppression candidate on matched neural/body half-step shams
- Parent: -
- Trainable: -
- Fixed: -
- Command: `sh scripts/launch_control_access_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: completed_user_reported_no_qualified_reversible_suppression ([E076-server-summary.json](records/E076-server-summary.json))

## E077 (2026-11-14)

- Question: Can bounded grouped CMA-ES learn gradually stronger sensory-specific slowing while preserving bends, uncued movement and START recovery?
- Method: pycma4.4.1 bounded ask/tell population12 with persistent4-worker evaluation and paired environmental cue episodes
- Parent: 8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a
- Trainable: Expected24 disjoint tied latent log-multiplier groups spanning318 of9616 unknown coordinates; exact schema/bounds persisted at setup
- Fixed: All topology, motor-neuron densities, leak/capacitance/kinetics/reversals/release, sensory gains/routing, NMJs/muscles/water mechanics/physical clocks
- Command: `sh scripts/launch_cma_curriculum_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: completed_no_curriculum_progress_user_reported ([E077-server-summary.json](records/E077-server-summary.json))

## E078 (2026-11-14)

- Question: Can unknown motor/control channels and connections support useful endogenous water-model rhythm that is suppressed by selective AVB inhibition and returns after release?
- Method: CMA-ES4.4.1 population12 with four persistent workers on four physical CPU cores each; full302 network and live body/sensory feedback in float64
- Parent: 8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a
- Trainable: Existing E077 control/sensory unknown groups plus AVB and numbered/head motor nonleak inward/outward densities, AVB-motor chemical/gap strengths and motor-motor strengths; disjoint tied groups persisted at runtime
- Fixed: Anatomical topology, known leak/capacitance/kinetics/reversals/release, sensory gains/routing/kinetics, NMJs, muscle and water mechanics, true physical clocks; other unknown coordinates not in group schema remain fixed
- Command: `sh scripts/launch_avb_training_mbslnx2.sh 480`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: [E078-server-summary.json](records/E078-server-summary.json)

## E079 (2026-11-15)

- Question: Why does confirmed AVB inhibition leave the connected forward motor circuit nearly unchanged in E078?
- Method: Read-only NumPy/SciPy analysis of hash-verified parent and selected ordinary sham/inhibited trajectories and matching model checkpoints; no JAX or forward simulation in audit
- Parent: 8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a
- Trainable: None
- Fixed: All checkpoints, parameters, source trajectories and biological/mechanical clocks remain unchanged
- Command: `sh scripts/launch_avb_audit_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: Input model/trace hashes match E078 references; identical increasing paired clocks and pre-intervention voltage match<=1e-8; finite voltages, positive release slopes, correct matrix shapes and symmetric gaps; produce descriptive cell-resolved report, no automatic behavioral or biological pass threshold
- Outcome record: [E079-server-summary.json](records/E079-server-summary.json)

## E080 (2026-11-15)

- Question: Which NMJ output populations transmit the persistent E078 body rhythm despite AVB inhibition?
- Method: Frozen full302-neuron closed-loop water simulation with selective NMJ release held at each neuron's own baseline mean during30..40s. All recurrent chemical/gap outputs, voltages and proprioceptive sensing remain live. Output release returns to live values at40s.
- Parent: /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855
- Trainable: None
- Fixed: All checkpoint parameters, anatomical topology, known membrane properties, sensory/NMJ/body parameters and water mechanics. Original source files and checkpoints unchanged.
- Command: `sh scripts/launch_nmj_source_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: [E080-server-summary.json](records/E080-server-summary.json)

## E081 (2026-11-15)

- Question: Which RMD/SMD/SMB output subgroup carries the dominant E080 head-motor contribution to the current wiggle?
- Method: Reuse E080 verified frozen full-network water diagnostic. Hold only selected neuron-to-muscle release at own selected-sham20..30s release mean during30..40s; restore live release at40s. Neural recurrence, voltages and proprioception stay live.
- Parent: /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855
- Trainable: None
- Fixed: Every selected-checkpoint parameter, topology, NMJ weights/signs/normalization, muscle/body/water parameters, sensory routes/kinetics and physical clocks
- Command: `sh scripts/launch_head_split_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: [E081-server-summary.json](records/E081-server-summary.json)

## E082 (2026-11-15)

- Question: Can selective SMD membrane inhibition reversibly suppress neural variation and the SMD-transmitted body wiggle, with recovery on release?
- Method: SMD-targeted additional membrane conductance with-80mV reversal; live voltage/gates/recurrence/sensory feedback and original voltage-to-NMJ transform. No voltage freeze, NMJ replacement, muscle gate, external pacer or weight fitting.
- Parent: /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855
- Trainable: None
- Fixed: All302-neuron parameters, topology, NMJ weights/signs/normalization, sensory system and water/body/muscle mechanics, physical clocks. Original checkpoint and reference data unchanged.
- Command: `sh scripts/launch_smd_control_mbslnx2.sh`
- Report: `-`
- Budget: Four concurrent CPU workers x4physical cores, float64 dedicated conda env, nohup.90min re...
- Acceptance: -
- Outcome record: [E082-server-summary.json](records/E082-server-summary.json)

## E083 (2026-11-15)

- Question: What sampled synaptic input and NMJ release modulation accompany persistent ventral SMD voltage variation and residual body commands during E082 inhibition?
- Method: Read-only NumPy/SciPy analysis of saved E082 traces/model/actor and original archived runtime NMJ asset, with no JAX or forward neural/body simulation
- Parent: /home/carsonc/r-elegans-results/smd-control-20261115T191006Z-193279
- Trainable: None
- Fixed: All original weights, channels, topology, NMJ/body/medium parameters, saved time series and physical clocks; no input data modified
- Command: `sh scripts/launch_smd_drive_audit_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: -
- Outcome record: completed ([E083-server-summary.json](records/E083-server-summary.json))

## E084 (2026-11-15)

- Question: Does RIAL causally support the persistent SMD/body rhythm through its existing chemical RIAL-to-SMD route, and is the influence reversible?
- Method: -
- Parent: /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855
- Trainable: None
- Fixed: All302-neuron parameters/topology/kinetics/reversals, NMJ weights/signs/transfer, sensory mappings and body/water mechanics; original checkpoints and input files unchanged
- Command: `sh scripts/launch_rial_control_mbslnx2.sh`
- Report: `-`
- Budget: Four concurrent workers x4physical cores, CPU float64 dedicated conda interpreter, nohup;...
- Acceptance: -
- Outcome record: [E084-server-summary.json](records/E084-server-summary.json)

## E085 (2026-11-15)

- Question: Can unknown sensory/relay/RIA parameters produce small cue-specific body-amplitude attenuation and recovery through existing circuitry while preserving the unpaced water-model gait?
- Method: pycma4.4.1 ask/tell population12, initial sigma0.3, four persistent CPU workers x4physical cores. Squared continuous cycle-amplitude ratio error plus continuous baseline/recovery/useful-gait penalties; no quantized cadence objective or direct movement gate.
- Parent: dd42e3f62bc00befc6052fae60c102bc89aaf9266ae88da1f78f3be30c2766a8
- Trainable: Nonleak inward/outward densities separately per AWAL/AWAR,AIYL/AIYR,AIZL/AIZR,AIBL/AIBR,RIAL/RIAR; disjoint tied multipliers on existing chemical edges from these cells to the specified relays/RIAR or RIAL. Actual groups and full coordinate mapping/bounds persisted before fitting.
- Fixed: All motor-neuron channels, RIAL-to-SMD connection strengths, all gaps, all other coordinates, leak/capacitance/channel kinetics/reversals/synaptic release fields, sensory signs/gains/routing/kinetics, NMJ weights/signs, muscles, body/water mechanics and physical clocks
- Command: `sh scripts/launch_rial_sensory_mbslnx2.sh`
- Report: `-`
- Budget: At least2h reserved for final validation, search lookahead stops generations that will no...
- Acceptance: -
- Outcome record: [E085-server-summary.json](records/E085-server-summary.json)

## E086 (2026-11-16)

- Question: Can chemical sensing reduce the cycle amplitude of RIAL release by1%,2%,5%, while preserving unpaced body gait, baseline RIAL rhythm and post-cue recovery?
- Method: Reuse full302-neuron embodied water model and original chemical schedules. Change fitting objective from body amplitude to RIAL release raw-supported median up/down cycle excursion. Standard deviation/mean remain descriptive; phase shift alone cannot satisfy amplitude gate. Protocols nominal,low-dose,shift,repeat,close-repeat,long-cue unchanged.
- Parent: f8cd5b401d81d297eb40d44ee3e355d5915ef2a07576077e022e16ca6d464b7f
- Trainable: Explicit E085 AWA/AIY/AIZ/AIB/RIA inward/outward nonleak channel density groups and existing source-to-relay/RIAL chemical groups; actual full mapping saved before fitting.
- Fixed: Motor channels, RIAL-to-SMD strengths, all gaps, other parameters, leak/capacitance/kinetics/reversals/release law, sensory gains/signs/kinetics/routing, NMJs, muscles, physical water/body parameters and clocks. No added edges, external oscillatory drive, clamps or direct body-output gates during t...
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_rial_release_mbslnx2.sh`
- Report: `-`
- Budget: 20min reserved for automatic saved-data audit plus final receipt; coordinator ends before...
- Acceptance: -
- Outcome record: [E086-server-summary.json](records/E086-server-summary.json)

## E087 (2026-11-16)

- Question: Can the saturated AIYL relay acquire usable operating range and a cue-specific response/recovery while baseline autonomous body motion is preserved?
- Method: {'optimizer': 'cma==4.4.1', 'seed': 8700, 'population': 12, 'sigma': 0.3, 'workers': 4, 'physical_cores_per_worker': 4, 'precision': 'float64', 'stage_seeds': '8700+stage', 'stage_sigma': 0.15, 'restart_seeds': '9700+restart', 'restart_sigma': 0.2}
- Parent: 5b77ec25266a4849b9e843c4fb1891f243ef31beef18c46fc9fd0d84f989c022
- Trainable: AWAL and AIYL nonleak inward/outward channel-density groups; one existing chemical edge per input source into AIYL; existing AIYL-to-RIAL strength for tonic compensation. Bundled canonical topology inspection lists12 incoming AIYL chemical sources (ASIL,AWAL,AWBL,ASEL,ASER,AFDL,AWCL,AWCR,BAGR,RIH,A...
- Fixed: All other neuron channel densities including RIAL/motor cells, all other chemical strengths including RIAL-to-SMD, all gaps, leak/capacitance/kinetics/reversals/release thresholds/slopes, sensory gains/signs/routing/kinetics, NMJs, muscles, water/body mechanics and clocks.
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_relay_range_mbslnx2.sh`
- Report: `-`
- Budget: Maximum3h total preparation,search,final validation,automatic audit and cleanup.20min res...
- Acceptance: -
- Outcome record: [E087-server-summary.json](records/E087-server-summary.json)

## E088 (2026-11-16)

- Question: Does continuation of the improving E087 operating-range fit from its selected model reach AIYL2% release headroom and2e-5 cue response without weakening baseline gait?
- Method: Explicit --continue-relay requires source experimentE087 and exact pinned selected hash. Fresh CMA at selected full original-anchor coordinates; verify model/reference SHA and full reconstruction. Recenter local+-0.4log intersect original bounds. Not exact CMA/RNG state resume. Old selected, later candidates, optimizer/RNG and recursive dependencies retained.
- Parent: 608587d8cc9267674b5828df02d7891ae8940f46cc6981b62518b7f89068de73
- Trainable: Same focused AWAL/AIYL nonleak density groups, existing inputs into AIYL and existing AIYL-to-RIAL strength.
- Fixed: Same motor/RIAL channel parameters, other chemical/gap weights, leak/capacitance/kinetics/reversals/release fields, sensory gains/signs/kinetics/routing, NMJ/muscle/water/body/clock parameters. No clamps/new edges/drive during fitting.
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_relay_continuation_mbslnx2.sh`
- Report: `-`
- Budget: Same hard TOTAL including prep/search/final validation/automatic audit/cleanup;20min audi...
- Acceptance: All E087 criteria retained verbatim, including8normal paired protocols, blocked control, exact selected checkpoint across cases, neural/body refined response differences, same direction, body preservation and separate AIYL-to-RIAL replay diagnostic. Automatic audit must complete with integrity. Full motor stop/start/rest/restart, RIAL rhythm control/swimming/biological fidelity remain unqualified.
- Outcome record: [E088-server-summary.json](records/E088-server-summary.json)

## E089 (2026-11-16)

- Question: Can sustained8h optimization of the unchanged relay operating-range objective improve headroom/cue responsiveness without losing baseline gait?
- Method: Unchanged E087/E088 focused AWAL/AIYL channel densities and existing AIYL inputs/AIYL-to-RIAL compensation; same range/cue/recovery/body-gait/control/numerical/claim criteria. Seed8900 population12 sigma.3,4CPUworkers x4physicalcores,float64. Stage seeds8900+stage sigma.15, restarts9900+count sigma.2.
- Parent: 1270489a0fb5d625740092dca8d73049faaf52aef568a9d724ffecc27df78dbb
- Trainable: -
- Fixed: All other neuron/motor/RIAL channels, other chemical/gap strengths, leak/capacitance/kinetics/reversals/release fields, sensory gains/signs/routing/kinetics, NMJs,muscles,water/body/physical clocks. No clamp or imposed drive during training.
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_relay_8h_mbslnx2.sh`
- Report: `-`
- Budget: -
- Acceptance: Same relay release q01>=.05,q99<=.98/.95/.90; cue mean differences>=2e-5/5e-5/1e-4, consistent sign, recovery/gait. Same low-dose/shift rung selection and final8normal,blocked and AIYL branch replay. Full RIAL rhythm/motor stop/start/rest/restart/swimming/biological fidelity remain unqualified.
- Outcome record: [E089-server-summary.json](records/E089-server-summary.json)

## E090 (2026-11-17)

- Question: Can another original-bounded local search from E089selected bridge remaining AIYL headroom and cue-response gaps?
- Method: {'optimizer': 'cma==4.4.1', 'seed': 9000, 'population': 12, 'sigma': 0.3, 'workers': 4, 'physical_cores_per_worker': 4, 'precision': 'float64', 'stage_seeds': '9000+stage', 'stage_sigma': 0.15, 'restart_seeds': '10000+restart', 'restart_sigma': 0.2}
- Parent: a84145d65ff503e71b6e6f0bf54ec69d396eed29cc3dfc3010d1c13bb89f4b57
- Trainable: Same E087-E08917focused groups when original topology/positive channels apply: AWAL/AIYL nonleak inward/outward densities, existing AIYL incoming chemical edges and AIYL-to-RIAL compensation. Actual schema recorded at startup.
- Fixed: Same other neuron/motor/RIAL channels, other chemical/gaps, leak/capacitance/kinetics/reversal/release fields, sensory gains/signs/kinetics/routing, NMJ/muscle/body/water/physical clocks. No new edges, clamps or imposed oscillatory drive in fitting.
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_relay_e090_mbslnx2.sh`
- Report: `-`
- Budget: After preparation/baseline, full180min optimization window; validate afterwards up to120m...
- Acceptance: -
- Outcome record: [E090-server-summary.json](records/E090-server-summary.json)

## E091 (2026-11-17)

- Question: Can joint first/close-repeat training close the remaining operating-range and weak-cue gaps while baseline movement/recovery remain preserved?
- Method: {'optimizer': 'cma==4.4.1', 'seed': 9100, 'population': 12, 'sigma': 0.3, 'workers': 4, 'cores_per_worker': 4, 'precision': 'float64', 'stage_seeds': '9100+stage', 'stage_sigma': 0.15, 'restart_seeds': '10100+restart', 'restart_sigma': 0.2}
- Parent: 18eadad77c048a86aa305d59ec37defa3fa2e34ba2da8cf72139d9e6965c00cc
- Trainable: Same focused AWAL/AIYL nonleak densities and existing AIYL inputs/AIYL-to-RIAL compensation; no new edges or signs.
- Fixed: Same other neuron/motor/RIAL channels, other chemical/gap weights, leak/capacitance/kinetics/reversals/release fields, sensory gains/signs/kinetics/routing,NMJ/muscle/water/body/physical clocks. No clamp/clock drive during fitting.
- Command: `cd /home/carsonc/github/r-elegans && git pull --ff-only && sh scripts/launch_relay_e091_mbslnx2.sh`
- Report: `-`
- Budget: Full3h optimization starts after verification and matching training baseline. Then up to2...
- Acceptance: -
- Outcome record: [E091-server-summary.json](records/E091-server-summary.json)

## E092 (2026-11-17)

- Question: Does the self-imposed original AIYR->AIYL effective-strength cap restrict AIYL headroom and repeated sensory responsiveness? Compare isolated changes and matched optimization before more identical fitting.
- Method: Require audited completed E091 with pinned selected model, original bounds, coordinates and exact nominal source trace reproduction. Fresh independent optimizers; not exact mid-run resume. All old dependencies remain external.
- Parent: 0020499c6b900183e954a0e577563584c7ffa19e9cd5264f90c584d8b5b116da
- Trainable: -
- Fixed: All 302 neurons retained. Anatomy/signs/reversals/release law/leak/capacitance/channel kinetics, gaps, other neural parameters, sensory gains/routing, NMJ/muscles/water mechanics and physical clocks unchanged. No imposed drive or clamp in fitting. Final release replay is diagnostic only.
- Command: `sh scripts/launch_relay_experiments_mbslnx2.sh`
- Report: `-`
- Budget: total_minutes=480
- Acceptance: -
- Outcome record: [E092-server-summary.json](records/E092-server-summary.json)

## E093 (2026-11-18)

- Question: Can the audited expanded-prior relay parent support sensory-dependent graded body-motion suppression, sustained cue-on execution/rest and cue-off recovery/restart, preserving baseline wiggle and relay responsiveness?
- Method: Fresh CMA at selected parent with recentered local boxes; not exact optimizer/RNG resume. Retain all original dependencies externally.
- Parent: Unique passing E092 best-single-edge panel; must equal factor1.25 scan model. Select nominal ordinary normal retained source trace. No fitted/random fallback.
- Trainable: Focused AWAL/AIYL nonleak density groups, existing incoming AIYL chemical strengths and AIYL->RIAL, plus RIAL inward/outward nonleak density groups and existing RIAL->SMD chemical output edges. All302 neurons remain modeled; do not fit all weights at this stage.
- Fixed: All other neural channel densities/chemical/gap strengths; topology/signs/reversals/leak/capacitance/channel kinetics/release law; sensory gains/kinetics/routing; all NMJ/muscle/body/water physics and physical clocks. No direct neural forcing/clamp/replay during fitting.
- Command: `sh scripts/launch_sensory_motion_mbslnx2.sh`
- Report: `python -m scripts.report_sensory_motion`
- Budget: total_minutes=240
- Acceptance: -
- Outcome record: [E093-server-summary.json](records/E093-server-summary.json)

## E094 (2026-11-18)

- Question: Does adding the right sensory/relay path (AWAR/AIYR/RIAR, AIYR->RIAR, RIAR->SMD) and a worst-region loss term increase cue-on body-motion suppression beyond E093 while preserving baseline gait, recovery and relay behavior?
- Method: Fresh CMA at parent, seed 9300, population 12, sigma 0.3, same rungs and training margin 1.05; not an exact optimizer/RNG resume.
- Parent: Same audited unique passing E092 isolated 1.25 AIYR->AIYL panel as E093 (matched comparison). Not the E093 selected checkpoint a499c523f31881f3df71fb1d1281ab21130545efa26493a0faa36cb93de0e92a.
- Trainable: E093 groups (AWAL/AIYL densities, incoming AIYL chemical strengths, AIYL->RIAL, RIAL densities, RIAL->SMD) plus AWAR/AIYR/RIAR inward/outward density groups, incoming AIYR chemical strengths, AIYR->RIAR and RIAR->SMD chemical edges. Group list is saved in inputs/cap_groups.json and audited.
- Fixed: All other densities/strengths/gaps, topology/signs, reversals, leak, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks.
- Command: `sh scripts/launch_sensory_motion_bilateral_mbslnx2.sh`
- Report: `python -m scripts.report_sensory_motion --run <E094 run folder>`
- Budget: total_minutes=240
- Acceptance: Unchanged E093 gates: all regions and cases reach the attained rung, baseline/recovery/relay preserved, sensory-blocked control, neural/body half-grid checks, replay dependence.
- Outcome record: [E094-server-summary.json](records/E094-server-summary.json)

## E095 (2026-11-18)

- Question: Can any single cue-to-motor chemical edge strength, scaled within 0.5x-4x of the parent and under an explicit widened engineering prior, produce at least 1% cue-on motion reduction in every body region while preserving baseline gait and recovery?
- Method: No optimizer. One evaluation per (edge, factor): a single chemical edge multiplied by the factor, all else equal to the parent, nominal protocol, target 1%, margin 1. Unit-factor control is the pinned parent reproduction. Then the two cases with the largest weakest-region reduction among gait-and-recovery-preserving cases are re-run with raw traces plus neural-half, body-half and sensory-blocked controls.
- Parent: Same audited unique passing E092 isolated 1.25 AIYR->AIYL panel as E093/E094; hash pinned at runtime in the new input manifest, not invented here.
- Trainable: -
- Fixed: All other densities/strengths/gaps, topology/signs, reversals, leak, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks.
- Command: `sh scripts/launch_edge_scan_mbslnx2.sh`
- Report: `python -m scripts.report_edge_scan --run <E095 run folder>`
- Budget: total_minutes=180, workers=8 x 4 physical cores if >=32 available else 4 x 4
- Acceptance: {'informative_route': 'A finalist with all three regional reductions >= 1%, safe, baseline and recovery preserved, same checkpoint across its four controls, neural/body half-grid differences <= 0.25 x target (0.25%), and sensory-blocked ratios within 0.25% of 1 with relay change <= 5e-6.', 'negative': 'No case reaches 1% in all regions, or finalist controls fail. This would show that none of the scanned single edges, even widened, supplies enough cue gain.', 'scan_cases': 'Metrics-only. The sca...
- Outcome record: [E095-server-summary.json](records/E095-server-summary.json)

## E096 (2026-11-18)

- Question: How much imposed membrane conductance at a single node group is needed to suppress cue-window body motion by at least 1% in every region (tail, middle, head) while preserving baseline gait and recovery, and which node groups can do it at all?
- Method: No optimizer and no weight change. Parent runs the nominal protocol (cue window 40-50 s, response window 41-46 s, baseline 30-40 s, recovery 60-70 s) with no sensory cue. For each of 60 probe cases an imposed membrane conductance is applied to the named neurons during 40-50 s: 10 node groups x {excitatory (reversal 0 mV), inhibitory (reversal -80 mV)} x {0.01, 0.03, 0.1} nS. One baseline run without a probe. Scores use the existing regional joint angular-speed ratio vs the matched no-cue window...
- Parent: Same audited unique passing E092 isolated 1.25 AIYR->AIYL panel as E093-E095; hash pinned at runtime, not invented here.
- Trainable: -
- Fixed: All weights, topology/signs, reversals, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks. The cue bath is off in every case.
- Command: `sh scripts/launch_node_probe_mbslnx2.sh`
- Report: `python -m scripts.report_node_probe --run <E096 run folder>`
- Budget: total_minutes=30, workers=up to 21 x 4 physical cores (84 of 128 server CPUs) if available
- Acceptance: {'per_case_gate': 'A case reaches the target when all three regional ratios are <= 0.99, safe, and baseline/recovery are preserved.', 'primary_readout': 'Minimum tested dose per node group and sign that reaches the gate, and the dose-response curve per region.', 'positive': 'At least one node group reaches the gate at <= 0.1 nS: that dose is the node-level drive budget for the next step.', 'negative': 'No group reaches the gate in every region at <= 0.1 nS: imposed single-node drive within the ...
- Outcome record: [E096-server-summary.json](records/E096-server-summary.json)

## E097 (2026-11-18)

- Question: (1) What are the thresholds, below 0.01 nS, at which imposed single-node conductance suppresses cue-window body motion by >=1% in every region? (2) How does motion reduction scale with the node's own release change? (3) Given the release change the actual cue produces at each node on the same parent, what reduction would the probe curve predict, and how does that compare with the observed 0.03-0....
- Method: No optimizer, no weight change, nominal protocol. (a) Baseline run without cue or probe. (b) Cue run on the same parent (cue bath on, no probe) with all traces retained, giving the node-level drive the real cue delivers. (c) 48 low-dose probes: AIZ, AIY, RIA, SMD, RIB, RIM x excitatory (0 mV) / inhibitory (-80 mV) x 0.0003, 0.001, 0.003, 0.01 nS during 40-50 s, all traces retained. (d) 4 grid controls: AIZ excitation at 0.003 and 0.01 nS on the neural-half and body-half grids. A separate read-o...
- Parent: Same audited unique passing E092 isolated 1.25 AIYR->AIYL panel as E093-E096; hash pinned at runtime, not invented here.
- Trainable: -
- Fixed: All weights, topology/signs, reversals, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks. Probe envelope as E096 (<=0.1 nS, reversal 0 or -80 mV).
- Command: `sh scripts/launch_node_calibration_mbslnx2.sh`
- Report: `python -m scripts.report_node_probe --run <E097 run folder>; python -m scripts.analyze_node_calibration --run <E097 run folder>`
- Budget: total_minutes=60, workers=up to 21 x 4 physical cores (84 of 128 server CPUs)
- Acceptance: {'gate': 'A probe case reaches the gate when all three regional ratios are <= 0.99, safe, and baseline/recovery are preserved (as E096).', 'bracketing': 'A threshold is bracketed for a group and sign when some tested dose fails the gate and the next higher dose passes it; otherwise the threshold is reported as an upper bound only.', 'numerical_check': 'A grid control passes when its regional reductions differ from the ordinary-grid case by at most max(0.25 percentage points, 20% of the ordinary...
- Outcome record: [E097-server-summary.json](records/E097-server-summary.json)

## E098 (2026-11-19)

- Question: Can joint training of the cue-to-RIA-to-motor pathway, under an explicit widened engineering prior and with a RIA node-level target, produce cue-driven whole-body motion suppression of at least 1% in every region with baseline gait, recovery and relay behavior preserved, validated by the full 11-case panel?
- Method: Fresh CMA-ES from the pinned E092 isolated 1.25 parent, population 21 (one wave on 21 workers), sigma 0.3 in the grouped latent space, restart at the incumbent with sigma 0.15 on rung advance, optimizer stop or every 8 generations, seed 9800. Same redundant two-cue protocol, rungs (1, 2, 5, 10, 25, 50, 75, 95 %), training margin 1.05 and final 11-case validation as E093/E094.
- Parent: Same audited unique passing E092 isolated 1.25 AIYR->AIYL panel as E093-E097; hash pinned at runtime, not invented here.
- Trainable: 53 groups: 37 single chemical edges (AWAL/AWAR to AIZ, AIY, ADFR, ASHL, RIFR; those interneurons plus AIBR to RIAL/RIAR; RIAL and RIAR to SMD and RMD targets) and 16 channel-density groups (inward and outward for AIZL, AIZR, RIAL, RIAR and the four SMD cells). The group list is saved in inputs/cap_...
- Fixed: All other densities/strengths/gaps, topology/signs, reversals, leak, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks. The inherited AIYR->AIYL one-edge exception stays frozen.
- Command: `sh scripts/launch_pathway_training_mbslnx2.sh [total minutes, default 420]`
- Report: `python -m scripts.report_sensory_motion --run <E098 run folder>`
- Budget: total_minutes_default=420, workers=21 x 4 physical cores (84 of 128 server CPUs)
- Acceptance: {'success': 'Selected checkpoint passes all 11 validation cases at a rung >= 1% with all regions reduced, safe, baseline/recovery and relay preserved, sensory-block control, neural/body half-grid agreement and both dependence replays, and the automatic audit is clean.', 'partial': 'Reports the attained rung, the weakest-region reduction and the cue-driven RIA release change against the ~0.014 estimate; a checkpoint that improves but misses the gate is a negative result for the question and stay...
- Outcome record: [E098-server-summary.json](records/E098-server-summary.json)

## E099 (2026-11-19)

- Question: Starting from the audited E098 selected checkpoint, can further joint training of the same cue-to-RIA-to-motor pathway, with more edge headroom and a weaker-side (bilateral) RIA node term, raise the weakest-region (tail) cue-driven suppression from 0.53 % to at least 1 % with baseline gait, recovery and relay behavior preserved?
- Method: Fresh CMA-ES (not an exact optimizer or RNG resume) from the pinned E098 selected checkpoint, population 21, sigma 0.3 in the grouped latent space, seed 9900, restarts at the incumbent (sigma 0.15) on rung advance, optimizer stop or every 8 generations. Same redundant two-cue protocol, rungs, training margin 1.05 and 11-case validation as E098.
- Parent: 95097d5be6adddb605515ea6509092fdccf819bde253bafa4738937652e8248b
- Trainable: The same 53 pathway groups as E098 defined relative to the E098 checkpoint: 37 single chemical edges and 16 channel-density groups (inward/outward for AIZL, AIZR, RIAL, RIAR and the four SMD cells). Edges already above the original bound (AWAR->AIZR, AIZR->RIAR, ADFR->RIAR, AIBR->RIAL) are trainabl...
- Fixed: All other densities/strengths/gaps, topology/signs, reversals, leak, kinetics, release law, sensory gains, NMJ/muscle/body/water physics and physical clocks.
- Command: `sh scripts/launch_pathway_continuation_mbslnx2.sh [total minutes, default 180]`
- Report: `python -m scripts.report_sensory_motion --run <E099 run folder>`
- Budget: total_minutes_default=180, workers=21 x 4 physical cores (84 of 128 server CPUs)
- Acceptance: {'success': 'Selected checkpoint passes all 11 validation cases at a rung >= 1 % with all regions reduced, safe, baseline/recovery and relay preserved, sensory-block control, neural/body half-grid agreement and both dependence replays, with a clean automatic audit.', 'partial': 'Report the tail/middle/head reduction against E098 (0.53/1.20/1.43 %), the per-side RIA release changes against 0.02, and whether the weakest region improved; a checkpoint that improves but misses the gate is a negative...
- Outcome record: negative for the 1 percent gate; no qualification of any kind ([E099-server-summary.json](records/E099-server-summary.json))

## E100 (2026-11-20)

- Question: Does the unchanged audited E099 selected checkpoint reproduce the direction and ranking of signal propagation measured in the Randi et al. 2023 optogenetic atlas (110 wild-type animals, DANDI 001075 segmentation files) on held-out animals, when a single-neuron stimulus of one calibrated strength is simulated?
- Method: No training. Load the checkpoint by SHA-256, settle it for 60 s, simulate a 0.5 s added membrane conductance (reversal 0 mV) on each stimulated named neuron, and take the window-mean (1-11 s after offset) change of the calcium indicator relative to an identical unstimulated run. Data responses are relative green/red ratio changes z-scored by per-ROI sham-window spread; only stimulations with a self-response z >= 3 are used. Animals are split by subject (every third held out). The only free quan...
- Parent: -
- Trainable: -
- Fixed: All network parameters. Nothing is trained.
- Command: `sh scripts/launch_atlas_baseline_mbslnx2.sh 180`
- Report: `python -m scripts.report_atlas_baseline --run <run folder>`
- Budget: total_minutes_default=180
- Acceptance: {'evidence_of_agreement': 'On held-out animals at the chosen conductance (phase-averaged): Spearman permutation p < 0.01, AUC for excited pairs > 0.55, model silent on fewer than 50 % of significant pairs, and substep-doubling correlation >= 0.9.', 'negative': "Any criterion fails: the unchanged model's propagation is not detectably related to the atlas at this resolution. A failure of the convergence check makes the other metrics unreliable and is reported as such.", 'inconclusive': 'Fewer tha...
- Outcome record: negative against pre-declared criteria; reference baseline for fitting ([E100-server-summary.json](records/E100-server-summary.json))

