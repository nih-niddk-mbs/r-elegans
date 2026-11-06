# Scientific recordkeeping

Maintain [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) for scientific work in this repo.

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
