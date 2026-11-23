# Reproducing this project

## E121 execution audit — 2026-11-23

**Finding: the repository contains substantial working simulation and audit code,
but the recently agreed H1.3 workflow is not implemented end to end.** In particular,
CV/all-animal fitting, extrasynaptic integration in the active solver, and a reusable
checked grouped optimizer remain unexecuted plans. Earlier per-run audit passes
validate particular numerical/evaluation procedures; they do not close these gaps.
This distinction should have been made explicit sooner.

Evidence classes: **local** means source inspection/software execution in this
workspace; **reported server** means supplied stdout/committed summaries; **pending**
means raw server evidence is not independently rechecked in this audit yet. No
new simulation, fitting or model promotion was performed by E121 locally.

| Requirement / discussed capability | Actual implementation and active path | Evidence / disposition |
|---|---|---|
|302-cell channel network|Conductance network and topology exist; active in atlas replay.|Local source; E120 reported server checks. Neuron count does not certify physiological completeness.|
|Ion/channel parameters|17 channel templates, fixed kinetics and many prior fields;9538 unknown channel/edge coordinates trained in recent protocols.|`brain/single_compartment.py`, `scripts/stable_atlas_policy.py`; not all measured channels/parameters, not unique recovery.|
|Best candidate|E120 selectedSHAac621eaa129c1c533691f52cce802d6ff259ada754d159bf287a30895c48adf3.|Reported server qualification independently rechecked from saved artifacts in E121; no new forward replication. E115/E112 retained.|
|Dataset fitting|Cached mean stimulus-response z statistics, correlation loss; original43/15/29 split.|`scripts/run_selected_atlas_audit.py`, `stable_atlas_policy.py`; no full time-course/amplitude fit.|
|Fivefold / jackknife model refits|No active fold-aware fitting runner or five independent model fits.|H1.3 is a plan, not an executed capability. New audit emits draft identities only.|
|All87-animal initialization fit|Not implemented/executed in active fitter; existing checkpoint acceptance still uses old split.|New all-data protocol remains to implement; no old result reclassified.|
|Animal identity and genotype audit|Loader can substitute filename for missing ID. Statistics builder counts NWB files as animals and has no genotype filter.|Raw audit confirms110 distinct metadata IDs/allWT/no conflicts; source fallback/no-filter risks remain relevant only for future data.|
|Extrasynaptic signaling|Optional legacy peptide module and269-pair asset; current common-state runner has four state fields and no peptide update.|Implemented separately, inactive in E120. E105 failed variants do not prove the mechanism unimportant.|
|Neuron-specific physiology provenance|Historical polarized baseline used296 transferred AWC templates and6 RMD profiles; later per-cell scales can differ.|Server audit now checks current base/effective rows and exact template match. Not302 independently measured neuron models.|
|Complete extrasynaptic state / paired mechanism comparison|Not integrated into settling/phase states/shams/replays of current runner; no matched new-fold comparison.|Blocking a complete model claim; requires implementation and validation, not another document entry.|
|Reliable reusable optimizer|SPSA/Adam proposals implemented; E119 direction diagnostic and E120 fixed reverse evaluation implemented/executed.|Original SPSA direction measured uphill. No general checked grouped-descent fitter; reverse success is not its implementation.|
|Own neural states and numerics|Implemented for current network-only path; full target-dependent three-level atlas checks and independent state/panel replay.|Reported server E120 pass. Full atlas loss not independently repeated; engineering tolerances not certified gradient accuracy.|
|Body/95-muscle/NMJ/medium|Modules and historical experiments exist; channel-body bridge and coupled sensory code are separate paths.|Not executed by E120. Old motor checkpoints do not qualify changed E120 weights.|
|Sensory/proprioception|Experimental source-routed transduction and closed-loop modules exist; older simplified RL path also exists.|Inactive in current atlas fits. Gains/kinetics not established by E120; old simplified actor differs from conductance brain.|
|Sensory start/sustain/stop/restart and swimming|Not established on the E120 checkpoint.|Remain unqualified; H1.2 gate retained. Rhythmic RMD is not locomotion or learned stopping.|
|Behavior refit from physiological initialization|Discussed future phase, not implemented as an integrated E120+extrasynaptic+CV workflow.|No new RL/behavior launch authorized by audit.|
|Recordkeeping|Source/env/argv, weights, RNG, bounds, controls and per-run audit scripts exist; small summaries in Git.|Much server evidence has only been supplied as compact stdout locally. New recheck writes a separate audit; originals immutable.|

### Server audit outcome received

Run `execution-audit-20261123T202419Z-968700` completed in35.27s, no timeout or
training. Saved E120 qualification and hashes independently recheck without
changing source records. Raw110files correspond to110distinct metadata subjects,
all WT labels; no missing IDs, repeats or genotype conflicts.87animals yield650
accepted trials;23yield none under current filters. Exact rejection-cause breakdown
is not provided. Raw split statistics match the E120 cache at declared1e-12tolerance.
Five fold drafts exist; model-level CV has not run.

E120 base matches bundled reference:two base rows with296transferredAWC priors and
6RMDprofiles;297effective rows after scaling. This confirms shared initialization
provenance, not identical final neurons or neuron-specific biological validation.
No peptide fields in checkpoint. Identity/genotype/data-cache and saved numerical
record risks are resolved for this dataset; the printed old pending-risk labels
were static review statuses. Report reconciliation now derives these labels from
completed components, preserving original status history. CV/extrasynaptic/optimizer/
clean-fold-initialization/body and full-suite-compatibility gaps remain. No need to
rerun the audit merely to change labels. [Server record](experiments/records/E121-server-summary.json).

### Software verification actually run

At audit start source revision951ccb2, worktree clean. Full local suite under
CPU/JAX_ENABLE_X64=true: **620 passed,6 failed**,9 warnings,161.99s. Failures:

- `test_builtin_model_drives_neural_body_without_external_files`: scan phase float32/float64 carry mismatch.
- `test_default_bundle_is_exact_mac_reference_with_complete_readout`: exact float32/reference equality, max difference about4.77e-8.
- `test_membrane_input_matches_passive_conductance_solution`: gates/calcium carry dtype mismatch.
- `test_disabled_gains_recover_legacy_and_streaming_preserves_memory`: voltage/calcium carry dtype mismatch.
- `test_intrinsic_full_network_carry_temporal_gradients_and_legacy_compatibility`: exact-equality difference about5.77e-9.
- `test_isolation_matches_zero_edge_network_with_same_cells_and_initial_state`: carry dtype mismatch.

These six tests all pass in a separate CPU/JAX_ENABLE_X64=false process. That
establishes precision/configuration sensitivity, **not** full float32-suite success
or permission to dismiss the failures. No scientific forward-model code or test
tolerance was changed to suppress them. This is an open mixed-precision/legacy
compatibility issue before integrating new full-body paths. Detailed messages and
XML/log hashes in [E121-local-test-results.json](experiments/records/E121-local-test-results.json).
Local readiness/source hashes in [E121-readiness-local.json](experiments/records/E121-readiness-local.json).

### Executable remaining audit, not another fit

`sh scripts/launch_execution_audit_mbslnx2.sh` runs CPU-only under nohup, maximum60min:

1. Independently recomputes E120 saved-data parameter/selection/signal/control/
   numerical/state/panel/long-RMD checks with artifact/ancestry hashes, in a new
   output directory. Verifies original run records unchanged. This is not a new
   forward numerical or full-atlas-loss replication.
2. Reads/hashes actual segmentation NWB files; reports file count separately from
   unique subjects, missing IDs, repeated subjects/files, genotype labels/conflicts,
   named-neuron/stimulation coverage, accepted trials and files excluded by current
   filters. Reconstructs historical split counts rather than assuming87 usable; independently
   rebuilds train/validation/test pair identities/counts/means from raw data and
   compares them with the actual E120 statistics cache.
3. Generates fixed fivefold **identity-only drafts**, excludes unverified IDs,
   reports overlap/coverage and heldout-only pairs. Drafts are explicitly not
   training-ready or genotype-approved; no CV model trained.
4. Records unimplemented/inactive capabilities and missing evidence, with compact
   report. A completed audit can correctly report the requested workflow unready.

Raw files/E120 weights are not locally mounted. User-reported server E121 results
confirm independent saved-data recheck and raw inventory completed; see outcome
above. No direct local raw-artifact inspection or new forward replication claimed.
No checkpoint, original audit, dataset or active split was changed.

### Execution order after the audit

- Resolve actual identity/genotype/coverage findings and approve a reproducible
  dataset definition; do not mistake draft folds for CV execution.
- Implement extrasynaptic state in the active common-state solver and all state
  snapshots, then verify its forward numerical behavior. Matched network-only and
  extrasynaptic variants need comparable protocols; no automatic use of failed legacy fits.
- Implement/check the low-dimensional grouped optimizer and fold-aware fitting
  runner; demonstrate one fold before running five independent fits. Verify a
  pre-atlas-fit ancestor; E120 cannot be silently used as a clean CV initializer.
- Fit all approved development animals for a labeled initialization after the
  procedure is fixed. Retain numerical/physiological safeguards; modest overfit is
  not a license to claim generalization or biological identification.
- Refit embodied sensory/motor behavior only with actual-checkpoint H1.2 controls
  and independent preservation of prior candidates/skills.

No new long global SPSA run is justified by this audit. Documented plans and
executable acceptance milestones must be tracked separately from now on.

Short version: install the package, fetch and verify the data, run the model, then follow the recorded schedule. What cannot be
reproduced from this repository is stated at the end.

## 1. Environment

- Python >= 3.10. `python -m pip install -e '.[data,fit,env,demo,dev]'` (extras in `pyproject.toml`; training uses `cma==4.4.1`).
- CPU embodied/CMA runs use `export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true`. E101/E102/E104/E105 atlas gradient runs use CUDA/A100s with float64; follow the experiment launcher rather than globally forcing the CPU backend.
- Server runs used a conda environment named `r-elegans` and the launch scripts `scripts/launch_*_mbslnx2.sh`; per-run package
  versions are captured by `scripts/experiment_records.py` in each run folder (not stored here).
- Tests: `JAX_PLATFORMS=cpu python -m pytest` without `JAX_ENABLE_X64` (the bundled checkpoints are float32; under x64 the pinned-bundle test and two
  embodied-sensory tests fail). The full suite currently has 20 failures that also occurred before the polarity work; at least
  `tests/test_petri_dish.py` passes alone, so they look order-dependent (not investigated). Recorded server runs used `JAX_ENABLE_X64=true`.

## 2. Data (never stored in Git)

1. `export R_ELEGANS_DATA_DIR=/path/to/data` (about 8 GB; keep it outside the repository).
2. `sh scripts/fetch_public_data.sh` downloads the public sources that have scripts (OSF, Zenodo, DANDI trace files, Figshare, CeNGEN,
   the authors' GitHub repository) and `python -m scripts.fetch_from_manifest` downloads the core sources listed with URLs and
   prints how to get the rest.
3. Manual downloads (publisher supplements, no stable file URLs): the Ripoll-Sanchez 2023 and Beets 2023 supplementary tables;
   exact DOIs and file names are in `data_manifest/session_sources_manifest.json`. The Atanas archive is extracted into
   `raw/calcium/atanas_kim_2023/processed_h5_E049/`; the original extraction command was not recorded.
4. `python -m scripts.verify_data` compares every file with `data_manifest/*.json` (SHA-256; `--quick` compares sizes only).
   DANDI publishes no SHA-256, so those hashes are ours; they certify what this project used, not what a server serves today.
5. `data_manifest/` lists each source with DOI, license note, files and hashes. Rebuild it with `python -m scripts.build_data_manifest`.
   Raw publisher files are not redistributed; observe each source's license and cite the papers.

## 3. Run the model

- Bundled demo (no external data): `r-elegans-demo` after installation. It uses the engineered sensory controller and the
  bundled runtime model (`r_elegans/assets/`, provenance in `MODEL_ASSET_PROVENANCE.md`).
- `r_elegans/assets/mac_rhythm_v1/` is the pinned 302-neuron conductance network; it produces rhythmic RMD cells and small bends.
  Qualified swimming, sensory start/stop and sensory-controlled rest have NOT been demonstrated (see `CURRENT_MODEL.md`).
- Synapse polarity from data is bundled (`r_elegans/assets/edge_polarity_v1.*`, Fenyves 2020; 2,921 of 3,709 edges signed, 788 undetermined).
  `r_elegans/assets/mac_rhythm_v1_polarized/` is the pinned network with that polarity; `initialize_conductance_network(..., edge_polarity=True)`
  applies it to a new network; `python -m scripts.apply_synapse_polarity --network N.npz --output-dir OUT` polarizes any checkpoint.
  No raw data is needed to use it (`scripts/build_edge_polarity_asset.py` rebuilds the asset from the raw tables). Polarized checkpoints must be requalified.

## 4. Reproduce the analyses

- Model versus data audit: `python -m scripts.audit_model_data_adherence [--network N.npz]` (add `--skip-controls` without atlas data).
- Randi atlas loader coverage: `python -m scripts.summarize_randi_atlas`.
- Peptide candidate set: `python scripts/build_peptide_candidates.py`.
- Atlas baseline of a checkpoint (E100): `python -m scripts.run_atlas_baseline --run RUN --checkpoint-sha256 SHA --data-root $R_ELEGANS_DATA_DIR --output-dir OUT`,
  then `python -m scripts.report_atlas_baseline --run OUT`. It needs the checkpoint (see the end).

## 5. Training schedule

- `experiments/TRAINING_SCHEDULE.md` (generated by `python -m scripts.build_training_schedule`) lists every experiment with a plan record:
  question, method, parent, trainable and fixed fields, launch command, budget, acceptance criteria and outcome record.
- `TRAINING_PROTOCOLS.md` holds the protocol design; `experiments/*.json` hold curricula and policies; `LAB_NOTEBOOK.md` holds the
  dated narrative, audits and negative results. Audit/checkpoint guarantees vary by experiment: earlier embodied/CMA pipelines include a separate automatic saved-data audit. Recent atlas trainers write summaries, checkpoint hashes and numerical checks, but their reports are not complete independent integrity audits; inspect the recorded limitations before continuing.
- The pattern on a server: pull, run the experiment's `scripts/launch_*_mbslnx2.sh <minutes>`, read the report with the experiment's
  `scripts.report_*` command, then record the audit in the notebook before continuing (rule in `AGENTS.md`).
- Embodied motor/control experiments use gradient-free CMA-ES on rollouts. E101/E102/E104/E105 use differentiable Adam fitting to pooled stimulation-atlas targets, with validation selection added in E104/E105. These are separate checkpoint lineages and neither establishes successful swimming or sensory start/stop.

## 6. What cannot be reproduced from the repository alone

- Selected checkpoints of the server experiments (E092-E099 and earlier skills) are not in Git unless a release says so; the repository
  records their SHA-256 only. Results that need them (for example E100) require those files.
- Experiments before the plan-record convention (before E072, except E042/E043/E063) are documented only in the notebook and protocols.
- CMA-ES continuations were fresh optimizer starts, not exact resumes; results also depend on hardware and thread counts.
- Source revision and working-tree state of old runs may be missing (`AGENTS.md`: gaps stay explicit).

## 7. Latest audit and checkpoint distinction (2026-11-21)

E104/E105 accuracy tables report the best-validation checkpoint; convergence, parameter/bound summaries and E105 peptide
values currently describe the final checkpoint. Recheck the selected weights themselves before qualification or continuation.
The E104 start from E102 has validation leakage because E102 trained on those validation animals; its validation-selection
results are not independent. The peptide term currently runs only in atlas stimulation simulations, not the embodied worm.
The full-suite failures noted above are historical reports and were not rerun in this repository review. Current model/run
status and limitations are summarized in `experiments/records/repository-review-20261121.json` and `LAB_NOTEBOOK.md`.
