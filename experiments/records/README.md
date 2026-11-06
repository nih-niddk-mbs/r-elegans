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

E019 instead records `evidence_type: locally_verified_artifacts`, with hashes
of inspected local checkpoints/manifests/summaries and external archive paths.
It does not claim a completed remote steering ensemble. The evidence index
lists the reported server transcripts; locally verified trial files remain in
their separately checksummed scientific archive.

For full reconstruction, archive the remote run and recursive dependencies,
then retain the archive on separate storage. See the notebook for the exact
procedure, historical gaps and limitations on exact numerical replication.
