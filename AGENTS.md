# Scientific recordkeeping

Maintain [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) for scientific work in this repo.

- User instruction 2026-11-25: retain parameters/checkpoints and detailed logs,
  rather than permanent simulation arrays. Keep optimizer/RNG and small scientific
  records. Arrays required for active/incomplete continuation or its recursive
  inputs remain protected; audit before pruning completed-run arrays. Record
  deletions with file hashes and explicitly mark raw evidence unavailable.
  Existing pinned active runners must not be edited in place to change storage.

- User instruction 2026-11-24: a training time allowance ends optimization,
  not final validation/audit. Let validation finish; do not impose an inherited
  total-job timeout unless the user explicitly requests a hard total ceiling.
  Planned stops must occur only after a durable, verified continuation checkpoint.
  Save expensive validation batches independently, reuse completed stages, and
  distinguish batch-boundary continuation from exact in-kernel/optimizer resume.
  Never silently restart completed work or overwrite the source run.

- User instruction 2026-11-16: after every run, perform a complete post-run audit
  before continuing training. Include execution/budget, candidate selection,
  parameter changes/bounds/frozen fields, signal propagation, all behavioral and
  numerical gates, controls and artifact integrity. Record unavailable evidence
  explicitly; an aggregate success flag alone is insufficient. Provide concise
  launch/report commands to the user.

- Before a new experiment, record its question, planned method, parent model,
  fixed/trainable parameters, command, seeds and acceptance criteria. Label
  plans as planned until execution/results are evidenced.
- After execution or reported results, append the outcome, failures, numerical
  checks, interpretation, decision and artifact locations. Distinguish local
  verification, user-reported server stdout and historical reconstruction.
- Keep exact invocation/environment/source records enabled. Preserve parent
  checkpoint hashes, optimizer/RNG files, all selected/final checkpoints and
  recursive reference dependencies outside Git. A fresh optimizer continuation
  is different from exact mid-run state resume.
- Keep raw data, checkpoints, trajectories and full stdout external. Commit
  small summaries, evidence hashes and notebook entries. Do not record secrets.
- Never invent missing run identifiers, timestamps, environments, source
  revisions or backups. Historical gaps remain explicit. Code revision alone
  does not reproduce a dirty working tree.
- Update CURRENT_MODEL.md, README.md and TRAINING_PROTOCOLS.md when outcomes
  supersede a pending status. Preserve earlier negative findings in the notebook
  and date corrections rather than silently rewriting scientific history.
- Tests validate software. Wave imitation, physical movement, and biological
  fidelity are separate claims. Record which was actually demonstrated.
- Current user strategy H1.1 prioritizes skill coverage. Maintain
  SKILL_REPERTOIRE.md and preserve good independent full-network checkpoints.
  Shared-weight interference is recorded but does not block adding standalone
  skills. Defer consolidation; retain physical/numerical criteria per skill.

# Controllable behavior requirement (user-directed H1.2)

- For future behavior training, jointly preserve sensory-controlled initiation,
  sustained execution, stopping/rest and restart. A continuously active motor
  pattern alone is a component milestone, not a controllable behavior.
- Validate repeated and varied cue timing, redundant cues, sensory-blocked/no-cue
  controls, and control-interneuron dependence with baseline movement preserved.
  Freezing all voltages or zeroing muscles is a diagnostic, not learned stopping.
- Gate subsequent behavior training on a validated start/stop parent. Revalidate
  start/stop on the actual changed behavior checkpoint; parent evidence does not
  qualify new weights. Keep qualified independent parents and failed outcomes.
