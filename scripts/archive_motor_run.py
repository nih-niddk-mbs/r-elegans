"""Preserve a finished run, checkpoint dependencies, source and checksum inventory.

No JAX import, no network calls, no deletion. Original server paths remain in
the evidence; archive.json maps them to portable copies. A copy on the same
disk is an archive, not an off-machine backup.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

if __package__:
    from .experiment_records import capture_run_record, file_sha256
else:
    from experiment_records import capture_run_record, file_sha256


def run_root(checkpoint):
    """Prefer the whole containing cohort, so its supervisor summary is retained."""
    for parent in checkpoint.parents:
        summary = parent/"summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text())
            if "workers" in data:
                return parent
    return checkpoint.parent


def dependency_roots(root):
    roots, queue, edges = [], [root.resolve()], []
    while queue:
        folder = queue.pop(0)
        if any(folder == known or known in folder.parents for known in roots):
            continue
        roots.append(folder)
        for manifest_path in folder.rglob("manifest.json"):
            # Source snapshots can include example manifests, not live dependencies.
            if "run_records" in manifest_path.relative_to(folder).parts:
                continue
            manifest = json.loads(manifest_path.read_text())
            for field,digest_field in (("initial_actor","initial_actor_sha256"),
                                       ("forward_reference_actor","forward_reference_sha256")):
                value = manifest.get(field)
                if value in (None,"None",""):
                    continue
                checkpoint = Path(value).expanduser()
                if not checkpoint.is_absolute():
                    raise ValueError(f"Relative dependency cannot be resolved unambiguously: {manifest_path}: {value}")
                checkpoint = checkpoint.resolve()
                if not checkpoint.is_file():
                    raise ValueError(f"Missing checkpoint dependency: {checkpoint}")
                actual = file_sha256(checkpoint)
                expected = manifest.get(digest_field)
                if expected and expected != actual:
                    raise ValueError(f"Dependency checksum mismatch: {checkpoint}")
                edges.append(dict(manifest=str(manifest_path),field=field,checkpoint=str(checkpoint),
                    sha256=actual,original_checksum_present=bool(expected)))
                queue.append(run_root(checkpoint))
    return roots,edges


def verify_archive(folder):
    record = json.loads((folder/"archive.json").read_text())
    for name,digest in record["files_sha256"].items():
        path = (folder/name).resolve()
        if folder.resolve() not in path.parents or file_sha256(path) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    return record


def archive_run(root,output,repo):
    root,output,repo = root.resolve(),output.resolve(),repo.resolve()
    if not (root/"summary.json").is_file():
        raise ValueError("Archive only a finished run with summary.json")
    if output == repo or repo in output.parents:
        raise ValueError("Archives must stay outside Git")
    roots,edges = dependency_roots(root)
    if any(output == p or p in output.parents for p in roots):
        raise ValueError("Archive destination must be outside every source run")
    output.mkdir(parents=True,exist_ok=False)
    mapping = []
    for i,source in enumerate(roots):
        destination = Path("runs")/f"{i:03d}-{source.name}"
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError(f"Refusing to follow symlinks in scientific records: {source}")
        shutil.copytree(source,output/destination)
        for path in source.rglob('*'):
            if path.is_file() and file_sha256(path) != file_sha256(output/destination/path.relative_to(source)):
                raise ValueError(f"Source changed or copy mismatch: {path}")
        mapping.append(dict(original=str(source),archived=str(destination)))
    # Copying succeeds before any COMPLETE marker is written. Interrupted copies
    # remain visibly incomplete and never overwrite their source runs.
    record = capture_run_record(output,repo,phase="archive_collection")
    subprocess.run(["git","bundle","create",str(output/"repository.bundle"),"--all"],cwd=repo,check=True,capture_output=True)
    gaps = []
    for source in roots:
        for manifest_path in source.rglob("manifest.json"):
            if "run_records" in manifest_path.relative_to(source).parts:
                continue
            m = json.loads(manifest_path.read_text())
            if not (manifest_path.parent/"run_records").is_dir():
                gaps.append(dict(manifest=str(manifest_path),gap="No invocation/environment/source snapshot at run start; collection-time metadata cannot replace it"))
            revision = m.get("git_revision")
            for name,digest in m.get("source_sha256",{}).items():
                result = subprocess.run(["git","cat-file","blob",f"{revision}:{name}"],cwd=repo,capture_output=True)
                if result.returncode or hashlib.sha256(result.stdout).hexdigest()!=digest:
                    gaps.append(dict(manifest=str(manifest_path),gap="Recorded source hash differs from bundled commit; locate the original source overlay/snapshot",file=name))
    if any(not e["original_checksum_present"] for e in edges):
        gaps.append(dict(gap="Some historical dependencies had no original checksum; collection hashes do not prove their contents at run time"))
    # Inventory covers all copied files, source snapshots and repository bundle.
    inventory = {str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    metadata = dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
        selected_run=str(root),run_mapping=mapping,dependencies=edges,files_sha256=inventory,
        collection_record=str(record.relative_to(output)),provenance_gaps=gaps,
        limitations="Historical gaps are preserved, not reconstructed. Current environment is explicitly collection-time. Hardware/backend differences can affect numerical reproduction.")
    (output/"archive.json").write_text(json.dumps(metadata,indent=2)+"\n")
    verify_archive(output)
    (output/"COMPLETE").write_text("All inventory checksums verified; see archive.json for historical provenance gaps.\n")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",type=Path)
    mode.add_argument("--latest-under",type=Path,help="Latest finished cohort matching --pattern under this directory")
    parser.add_argument("--pattern",default="reverse-*",help="Cohort glob for --latest-under; e.g. steering-*")
    mode.add_argument("--verify",type=Path)
    parser.add_argument("--output-dir",type=Path,help="Parent directory for a fresh archive")
    args=parser.parse_args()
    if args.verify:
        record=verify_archive(args.verify.resolve())
        print(f"Verified {len(record['files_sha256'])} files; {len(record['provenance_gaps'])} recorded provenance gaps")
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required when archiving")
    root=args.run
    if args.latest_under:
        candidates=[p for p in sorted(args.latest_under.glob(args.pattern),reverse=True) if (p/'summary.json').is_file()]
        if not candidates:
            parser.error("No finished matching cohort found")
        root=candidates[0]
    output=args.output_dir.expanduser()/(root.name+'-archive-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    metadata=archive_run(root.expanduser(),output,Path(__file__).resolve().parents[1])
    print(f"Verified archive: {output}\nFiles: {len(metadata['files_sha256'])}; historical provenance gaps: {len(metadata['provenance_gaps'])}")
    print("Copy this archive to separate storage for an off-machine backup.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
