"""Remove recognized completed-run simulation arrays; retain parameters and logs.

Dry run by default. No source/data deletion or /proc access. Active runs and their
recursive input/source runs are protected. A detailed fsynced journal is retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
import zipfile

BASE = Path('/home/carsonc/r-elegans-results')
KEEP_PARTS = {'run_records', 'source', 'inputs', 'optimizer', 'checkpoints', 'weights', 'models', 'parameters'}
KEEP_NAME = re.compile(r'network|model|param|weight|checkpoint|coordinate|origin|optimizer|rng|random|profile|candidate', re.I)
RAW_NAME = re.compile(r'(?:.*trajectory.*|trace-\d+|(?:no_cue|stop_only|stop_start|start_only|active|sham|dose-low)|(?:base-)?(?:starts|response)|final|extension(?:-states|-end)?|group-\d+-target-\d+|batch-\d+|(?:confirm|screen)-.+|(?:current|candidate)-(?:full|posterior-subcircuit))\.npz$')
RAW_KEYS = {'dense_time_s', 'dense_joints', 'time_s', 'voltage_mV', 'means', 'response', 'final_voltage_mV', 'metadata_json', 'provenance_json', 'joints', 'voltage', 't', 'q'}
PARAM_KEYS = re.compile(r'chemical|electrical|conductance|density|densities|capacitance|reversal|neuron_ids|weights|params|parameters|log_|rng|offsets|proposal|covariance', re.I)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def owner_run(path, base):
    try:
        rel = path.resolve().relative_to(base)
    except ValueError:
        return None
    return base/rel.parts[0] if rel.parts else None


def process_runs(base):
    """Read own-user argv through ps; fail closed if process inventory fails."""
    output = subprocess.check_output(['ps', '-ww', '-u', str(os.getuid()), '-o', 'pid=,args='], text=True)
    roots = set()
    for line in output.splitlines():
        try:
            args = shlex.split(line.split(None, 1)[1])
        except (ValueError, IndexError):
            continue
        # Include any result-directory argv, including workers and supervisors.
        for arg in args:
            if arg.startswith(str(base)+'/'):
                root = owner_run(Path(arg), base)
                if root is not None:
                    roots.add(root)
    return roots


def json_files(root):
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('run_records', 'source') and not (Path(folder)/d).is_symlink()]
        for name in files:
            path = Path(folder)/name
            if name.endswith('.json') and not path.is_symlink():
                yield path


def references(value, context, base):
    found = set()
    if isinstance(value, str):
        if value.startswith(str(base)+'/'):
            root = owner_run(Path(value), base)
            if root:
                found.add(root)
    elif isinstance(value, list):
        for item in value:
            found.update(references(item, context, base))
    elif isinstance(value, dict):
        for key, item in value.items():
            # Hash maps carry file references in keys as well as values.
            found.update(references(key, context, base))
            found.update(references(item, context, base))
    return found


def completed(root):
    """Only explicit terminal success receipts/summaries qualify; never guess."""
    for name in ('budget_receipt.json', 'summary.json'):
        path = root/name
        if path.exists():
            row = load(path)
            if row.get('status') == 'running':
                return False
            if name == 'budget_receipt.json':
                return row.get('all_steps_finished') is True and row.get('audit_complete') is True
            return row.get('status') == 'completed' or row.get('all_stages_finished') is True
    return False


def inventory(base, quiet_hours):
    active = process_runs(base)
    protected = set(active)
    eligible = []
    cutoff = time.time()-quiet_hours*3600
    for root in base.iterdir():
        if not root.is_dir() or root.is_symlink() or root.name.startswith('.'):
            continue
        try:
            is_done = completed(root)
        except (OSError, ValueError):
            is_done = False
        # Unknown/incomplete runs are protected, as are their dependencies.
        if not is_done:
            protected.add(root)
        else:
            eligible.append(root)
    queue = list(protected)
    while queue:
        root = queue.pop()
        for path in json_files(root):
            try:
                refs = references(load(path), path.parent, base)
            except (OSError, ValueError):
                # Cannot establish dependencies reliably: refuse deletion.
                raise RuntimeError('Unreadable live/incomplete-run metadata: '+str(path))
            for dependency in refs-protected:
                protected.add(dependency)
                if dependency.is_dir():
                    queue.append(dependency)
    return [r for r in eligible if r not in protected and (r/'budget_receipt.json' if (r/'budget_receipt.json').exists() else r/'summary.json').stat().st_mtime <= cutoff], protected


def recognized_raw(path, root):
    relative = path.relative_to(root)
    if any(part in KEEP_PARTS or 'optimizer' in part.lower() or 'checkpoint' in part.lower() for part in relative.parts[:-1]):
        return False
    if path.is_symlink() or KEEP_NAME.search(path.name) or not RAW_NAME.fullmatch(path.name):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            keys = {Path(name).stem for name in archive.namelist() if name.endswith('.npy')}
    except (OSError, zipfile.BadZipFile):
        return False
    return bool(keys & RAW_KEYS) and not any(PARAM_KEYS.search(k) for k in keys)


def clean(base, apply, quiet_hours, event):
    initial_active = process_runs(base)
    eligible, protected = inventory(base, quiet_hours)
    event('inventory', eligible_runs=len(eligible), protected_runs=sorted(str(p) for p in protected))
    count = total = 0
    cutoff = time.time()-quiet_hours*3600
    for root in sorted(eligible):
        # A completed run is never meant to be resumed during cleanup. Recheck
        # process inventory per run and terminal metadata before each deletion.
        if root in process_runs(base):
            continue
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in KEEP_PARTS and not (Path(folder)/d).is_symlink()]
            for name in files:
                path = Path(folder)/name
                if not recognized_raw(path, root):
                    continue
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or before.st_mtime > cutoff or before.st_nlink != 1:
                    continue
                # Hash before deletion; no trace content or private stdout copied.
                if apply and not process_runs(base).issubset(initial_active):
                    event('stopped_new_job', reason='New result-using process appeared; rerun cleanup to rebuild dependency protection')
                    return count, total
                sha = digest(path)
                details = dict(path=str(path), bytes=before.st_size, sha256=sha,
                               inode=before.st_ino, mtime_ns=before.st_mtime_ns, run=str(root),
                               reason='Completed-run simulation array; parameters and detailed records retained')
                event('delete_planned' if apply else 'would_delete', **details)
                if apply and not process_runs(base).issubset(initial_active):
                    event('stopped_new_job', reason='New result-using process appeared while hashing; nothing further deleted')
                    return count, total
                now = path.lstat()
                if (now.st_ino, now.st_size, now.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns) or not completed(root) or (root/'budget_receipt.json' if (root/'budget_receipt.json').exists() else root/'summary.json').stat().st_mtime > cutoff:
                    event('skipped_changed', path=str(path)); continue
                if apply:
                    path.unlink()
                    event('deleted', **details)
                count += 1; total += before.st_size
    event('finished', apply=apply, files=count, bytes=total,
          evidence_limit='Deleted arrays cannot be independently replayed or re-audited from raw outputs; historical logs/audits remain, with this explicit retention change.')
    return count, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--quiet-hours', type=float, default=1.)
    a = p.parse_args()
    base = a.base.expanduser().resolve()
    if not base.is_dir() or base in (Path('/'), Path.home()) or a.quiet_hours < 1 or not a.quiet_hours < float('inf'):
        p.error('Existing results directory and at least one quiet hour required')
    import fcntl
    with (Path('/tmp')/f'r-elegans-storage-clean-{os.getuid()}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = base/'.cleanup-records'; directory.mkdir(exist_ok=True)
        journal = directory/('storage-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())+'.jsonl')
        def event(kind, **fields):
            row = dict(event=kind, at_utc=datetime.now(timezone.utc).isoformat(), **fields)
            with journal.open('a') as stream:
                stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(row), flush=True)
        event('start', base=str(base), apply=a.apply, program_sha256=digest(Path(__file__)),
              preserve='All parameter/model/optimizer/RNG files, JSON/text/log/source records, active/incomplete runs and recursive dependencies; raw input data outside results untouched')
        try:
            clean(base, a.apply, a.quiet_hours, event)
        except Exception as exc:
            event('failed', error=repr(exc)); raise
        print('Detailed deletion journal:', journal, flush=True)


if __name__ == '__main__':
    main()
