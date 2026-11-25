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
RAW_NAME = re.compile(r'(?:.*trajectory.*|trace-\d+|(?:no_cue|stop_only|stop_start|start_only|active|sham|dose-low)|(?:base-)?(?:starts|response)|final|extension(?:-states|-end)?|group-\d+-target-\d+|batch-\d+|(?:confirm|screen)-.+|validation-[A-Za-z0-9_-]+|(?:current|candidate)-(?:full|posterior-subcircuit))\.npz$')
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


def archived_provenance(path):
    return 'run_records' in path.parts


def json_files(root):
    # Direct file references must obey the same archive exclusion as os.walk.
    if archived_provenance(root):
        return
    if root.is_file():
        if root.suffix == '.json' and not root.is_symlink():
            yield root
        return
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('run_records', 'source') and not (Path(folder)/d).is_symlink()]
        for name in files:
            path = Path(folder)/name
            if name.endswith('.json') and not path.is_symlink():
                yield path


HISTORICAL_CONFIG_DIRS = {'parent_run', 'warm_start_run', 'resume_run', 'precision_reference_assay', 'parent', 'warm_start', 'resume'}


def references(value, context, base, follow_directories=True, role=None):
    found = set()
    if isinstance(value, str):
        candidate = None
        if value.startswith(str(base)+'/'):
            candidate = Path(value)
        elif not value.startswith('/') and '://' not in value and len(value) < 4096:
            if value.endswith(('.npz', '.json')) or value.startswith(('../', './')):
                candidate = context/value
        if candidate is not None:
            path = candidate.resolve()
            try:
                path.relative_to(base)
            except ValueError:
                return found
            if follow_directories or role not in HISTORICAL_CONFIG_DIRS or path.suffix in ('.npz', '.json'):
                found.add(path)
    elif isinstance(value, list):
        for item in value:
            found.update(references(item, context, base, follow_directories, role))
    elif isinstance(value, dict):
        for key, item in value.items():
            # Output destinations are provenance, not inputs. Following them
            # turns an old configuration into protection of all its outputs.
            if key in ('output_dir', 'output', 'log', 'log_path', 'report', 'motion_preview', 'figure'):
                continue
            # Hash maps carry file references in keys as well as values.
            found.update(references(key, context, base, follow_directories))
            found.update(references(item, context, base, follow_directories, key))
    return found


def legacy_complete(root, row):
    """Recognize final legacy reports by schema plus exact saved model identity."""
    if row.get('status') not in (None, 'completed'):
        return False
    assay = (root.name.startswith(('cue-precision-', 'cue-split-', 'cue-resolution-'))
             and isinstance(row.get('all_grids_response_pass'), bool)
             and bool(row.get('cases')) and 'comparisons' in row)
    fit = (root.name.startswith(('sensorimotor-', 'sensorimotor64-', 'alternation64-'))
           and row.get('stop_reason') in ('time_budget', 'iterations')
           and isinstance(row.get('iterations'), int) and bool(row.get('validation')))
    if not (assay or fit):
        return False
    expected = row.get('checkpoint_sha256', '')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        return False
    model = root/('parent_run/network_best.npz' if assay else 'network_best.npz')
    return model.is_file() and not model.is_symlink() and digest(model) == expected


def completed(root):
    """Only explicit terminal success receipts/summaries qualify; never guess."""
    for name in ('budget_receipt.json', 'summary.json'):
        path = root/name
        if path.exists():
            row = load(path)
            if row.get('status') == 'running' or row.get('execution_complete') is False or row.get('all_steps_finished') is False:
                return False
            if name == 'budget_receipt.json':
                return ((row.get('all_steps_finished') is True and row.get('audit_complete') is True)
                        or (row.get('audit_finished') is True and row.get('exit_code') == 0 and not row.get('timed_out')))
            return (row.get('status') == 'completed' or row.get('all_stages_finished') is True
                    or (row.get('audit_complete') is True and row.get('integrity_passed') is True)
                    or legacy_complete(root, row))
    audit = root/'audit.json'
    if audit.exists():
        row = load(audit)
        return (row.get('audit_complete') is True and row.get('integrity_passed') is True
                and row.get('execution_complete') is not False and row.get('all_steps_finished') is not False
                and row.get('full_target_complete') is not False)
    return False


def inventory(base, quiet_hours, event=None):
    event = event or (lambda *a, **k: None)
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
            event('protected_path', path=str(root), reason='Incomplete or unrecognized completion metadata')
        else:
            eligible.append(root)
    queue = list(protected)
    while queue:
        root = queue.pop()
        for path in json_files(root):
            try:
                owner = owner_run(path, base)
                config = path.name in ('configuration.json', 'new_configuration.json', 'old_configuration.json')
                # Copied configurations document old launch directories. Runtime
                # readers use their scalar settings and explicit file inputs,
                # not every trajectory under the historical launch parents.
                snapshot = config and ('inputs' in path.relative_to(base).parts or (owner is not None and completed(owner)))
                refs = references(load(path), path.parent, base, follow_directories=not snapshot)
            except (OSError, ValueError):
                # Cannot establish dependencies reliably: refuse deletion.
                raise RuntimeError('Unreadable live/incomplete-run metadata: '+str(path))
            for dependency in refs-protected:
                protected.add(dependency)
                event('protected_path', path=str(dependency), reason='Required reference', referenced_by=str(path))
                if dependency.is_dir() or dependency.suffix == '.json':
                    queue.append(dependency)
    return [r for r in eligible if r not in protected and terminal_record(r).stat().st_mtime <= cutoff], protected


def terminal_record(root):
    return next(p for p in (root/'budget_receipt.json', root/'summary.json', root/'audit.json') if p.exists())


def retained_folder(parts, index):
    part = parts[index]
    # Old whole-run copies put simulation traces beside model inputs here.
    # Only this specific lineage subtree is traversed, not general data inputs.
    lineage_inputs = part == 'inputs' and index+1 < len(parts) and parts[index+1] == 'parent_lineage'
    return ((part in KEEP_PARTS and not lineage_inputs)
            or 'optimizer' in part.lower() or 'checkpoint' in part.lower())


def recognized_raw(path, root):
    relative = path.relative_to(root)
    if any(retained_folder(relative.parts, i) for i in range(len(relative.parts)-1)):
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
    eligible, protected = inventory(base, quiet_hours, event)
    event('inventory', eligible_runs=len(eligible), protected_paths=sorted(str(p) for p in protected),
          dependency_policy='Whole active/incomplete runs; exact referenced files and directory subtrees, not whole runs for a parameter-file reference')
    count = total = 0
    cutoff = time.time()-quiet_hours*3600
    for root in sorted(eligible):
        # A completed run is never meant to be resumed during cleanup. Recheck
        # process inventory per run and terminal metadata before each deletion.
        if root in process_runs(base):
            continue
        run_count, run_bytes = count, total
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if (d not in KEEP_PARTS or (d == 'inputs' and (Path(folder)/d/'parent_lineage').is_dir())) and not (Path(folder)/d).is_symlink()]
            for name in files:
                path = Path(folder)/name
                resolved = path.resolve()
                if resolved in protected or any(p in resolved.parents for p in protected):
                    continue
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
                if (now.st_ino, now.st_size, now.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns) or not completed(root) or terminal_record(root).stat().st_mtime > cutoff:
                    event('skipped_changed', path=str(path)); continue
                if apply:
                    path.unlink()
                    event('deleted', **details)
                count += 1; total += before.st_size
        event('run_finished', run=str(root), apply=apply, files=count-run_count, bytes=total-run_bytes)
    event('finished', apply=apply, files=count, bytes=total,
          evidence_limit='Deleted arrays cannot be independently replayed or re-audited from raw outputs; historical logs/audits remain, with this explicit retention change.')
    return count, total


def report(base):
    journals = sorted((base/'.cleanup-records').glob('storage-*.jsonl'))
    if not journals:
        print('No cleanup journal found; nothing started.')
        return
    last = None
    runs = []
    blockers = []
    with journals[-1].open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('event') in ('finished', 'failed', 'stopped_new_job'):
                last = row
            if row.get('event') == 'protected_path' and Path(row['path']).parent == base and Path(row['path']).name.startswith(('cue-precision-', 'alternation64-', 'sensorimotor64-')):
                blockers.append(row)
            if row.get('event') == 'run_finished' and row.get('files', 0):
                runs.append(row)
    print('READ-ONLY cleanup report; nothing started.')
    if last is None:
        print('No final event yet; cleanup may still be running.')
    else:
        print(json.dumps({k: last[k] for k in ('event', 'apply', 'files', 'bytes', 'error', 'reason') if k in last}))
    if not runs:
        for row in blockers[:3]:
            print('Protected:', Path(row['path']).name, row['reason'], row.get('referenced_by', ''))
    for row in sorted(runs, key=lambda r: r['bytes'], reverse=True)[:3]:
        print(f"{row['bytes']/1024**3:.2f} GiB", Path(row['run']).name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--report', action='store_true', help='Read-only compact latest cleanup report; starts nothing')
    p.add_argument('--quiet-hours', type=float, default=1.)
    a = p.parse_args()
    base = a.base.expanduser().resolve()
    if not base.is_dir() or base in (Path('/'), Path.home()) or a.quiet_hours < 1 or not a.quiet_hours < float('inf'):
        p.error('Existing results directory and at least one quiet hour required')
    if a.report:
        report(base)
        return
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
