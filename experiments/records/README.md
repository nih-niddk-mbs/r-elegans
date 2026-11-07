# Reported-result metadata

These JSON files are small, derived experiment summaries, not model inputs,
checkpoints, trajectories, or raw scientific datasets. Entry identifiers link
to [LAB_NOTEBOOK.md](../../LAB_NOTEBOOK.md). The evidence index records SHA-256
hashes of the original user-pasted server transcripts, preserved externally
under the OneDrive r-elegans directory.

For E010–E015, `verification` states that these are reported server results. The original
remote manifests, exact execution revision, environment, checkpoint contents,
and backup status have not been independently verified by parsing stdout.
No run identifier or missing setting is fabricated. Gate counts are derived
using the published motor criteria, including the float32 lag-boundary epsilon.

E019 and E021 instead record `evidence_type: locally_verified_artifacts`, with hashes
of inspected local checkpoints/manifests/summaries and external archive paths.
These local records do not establish server execution or outcomes. The evidence
index lists the reported server transcripts; locally verified trial files remain in
their separately checksummed scientific archive.

E020 is truncated reported server stdout. Its aggregate supervisor exit code
is available, but the visible condition's seed and pacing rate are explicitly
null. Do not infer missing per-seed statistics from that excerpt.

For full reconstruction, archive the remote run and recursive dependencies,
then retain the archive on separate storage. See the notebook for the exact
procedure, historical gaps and limitations on exact numerical replication.
