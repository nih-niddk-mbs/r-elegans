"""Vault learned parameters and records from the results directory, then purge it.

User instruction 2026-11-28: keep what was learned (records and trained parameters)
and discard everything else. Stages:

  plan    read-only classification and size report
  build   copy keepers into a new vault directory (never modifies the source)
  verify  re-hash the vault against its manifest
  purge   delete the source directory contents; dry run unless --apply, and only
          after a fresh verification, an unchanged-source check and a typed confirmation

Kept: parameter NPZ/pickle/msgpack files (content-addressed, so copies dedupe) that are
either referenced by hash in repository documents (tier A) or have checkpoint-like
names and parameter-like contents (tier B); every other non-array file, compressed
into one tar.gz per run (gif files excluded). Discarded: simulation arrays,
per-candidate optimizer evaluations (tier C) and any parameter-like file over the size cap.
Active runs (own-user argv naming the source directory) block build and purge.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone

from scripts.clean_results_storage import BASE, PARAM_KEYS, RAW_KEYS, process_runs

REPO = Path(__file__).resolve().parents[1]
TIER_B_NAME = re.compile(r'best|final|selected|winner|parent|origin|initial|baseline|checkpoint|actor|network|optimizer|rng|cma', re.I)
HEX = re.compile(r'(?<![0-9A-Fa-f])([0-9a-f]{8,64})(?![0-9A-Fa-f])')
PARAM_EXT = ('.npz', '.pkl', '.msgpack')
SKIP_RECORD_EXT = ('.gif',)
MANIFEST_EXCLUDED = {'MANIFEST.jsonl', 'MANIFEST.sha256', 'VERIFIED.json', 'PURGE_JOURNAL.jsonl'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def documented_hashes(repo):
    """Hex tokens (8..64 chars) in repository documents, grouped by length."""
    tokens = set()
    for folder, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'node_modules')]
        for name in files:
            if name.endswith(('.md', '.json', '.txt')):
                try:
                    tokens.update(HEX.findall((Path(folder)/name).read_text(errors='ignore')))
                except OSError:
                    continue
    by_length = {}
    for token in tokens:
        by_length.setdefault(len(token), set()).add(token)
    return by_length


def is_documented(sha, by_length):
    return any(sha[:n] in tokens for n, tokens in by_length.items())


def npz_is_simulation(path):
    """Same content rule as the storage cleaner: array keys, not file names."""
    try:
        with zipfile.ZipFile(path) as archive:
            keys = {Path(name).stem for name in archive.namelist() if name.endswith('.npy')}
    except (OSError, zipfile.BadZipFile):
        return False
    return bool(keys & RAW_KEYS) and not any(PARAM_KEYS.search(k) for k in keys)


def classify(base, repo, cap_bytes, log=lambda message: None):
    """Return (entries, summary). Entry kinds: param, record, discard, symlink."""
    by_length = documented_hashes(repo)
    full_documented = set(by_length.get(64, ()))
    entries = []
    seen_full = set()
    count = 0
    hashed = 0
    last_log = time.monotonic()
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs.sort()
        folder = Path(folder)
        for name in sorted(dirs):
            if (folder/name).is_symlink():
                entries.append(dict(rel=str((folder/name).relative_to(base)), kind='symlink', size=0, mtime_ns=0))
        for name in sorted(files):
            path = folder/name
            rel = str(path.relative_to(base))
            count += 1
            if time.monotonic()-last_log >= 10:
                last_log = time.monotonic()
                log(f'{count} files classified, {hashed/2**30:.1f} GiB hashed, at {rel}')
            if path.is_symlink():
                entries.append(dict(rel=rel, kind='symlink', size=0, mtime_ns=0))
                continue
            stat = path.stat()
            entry = dict(rel=rel, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            lower = name.lower()
            if lower.endswith(PARAM_EXT):
                like = bool(TIER_B_NAME.search(name))
                if stat.st_size > cap_bytes:
                    entry.update(kind='discard', why='oversize_param_like' if like else 'array')
                elif lower.endswith('.npz') and npz_is_simulation(path):
                    # Cheap key check first: simulation arrays are never hashed or kept.
                    entry.update(kind='discard', why='simulation')
                else:
                    hashed += stat.st_size
                    sha = digest(path)
                    entry['sha256'] = sha
                    if sha in full_documented:
                        seen_full.add(sha)
                    if is_documented(sha, by_length):
                        entry.update(kind='param', tier='A')
                    elif lower.endswith(('.pkl', '.msgpack')) or (like and not npz_is_simulation(path)):
                        entry.update(kind='param', tier='B')
                    else:
                        entry.update(kind='discard', why='tier_c')
            elif lower.endswith(SKIP_RECORD_EXT):
                entry.update(kind='discard', why='gif')
            else:
                entry.update(kind='record')
            entries.append(entry)
    summary = dict(
        files=len(entries),
        documented_full_hashes=len(full_documented),
        documented_full_hashes_found=len(seen_full),
        documented_full_hashes_missing=sorted(full_documented - seen_full),
    )
    return entries, summary


def top_level(rel):
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else '_toplevel'


def report(entries, base):
    groups = Counter()
    sizes = Counter()
    for entry in entries:
        key = entry['kind'] if entry['kind'] != 'param' else 'param tier ' + entry['tier']
        if entry['kind'] == 'discard':
            key = 'discard ' + entry['why']
        groups[key] += 1
        sizes[key] += entry['size']
    for key in sorted(groups):
        print(f'{sizes[key]/2**30:10.2f} GiB {groups[key]:8d} files  {key}')
    unique = {}
    for entry in entries:
        if entry['kind'] == 'param':
            unique[entry['sha256']] = entry['size']
    print(f'parameter files after content dedup: {len(unique)} unique, {sum(unique.values())/2**30:.3f} GiB')
    for tier in ('A', 'B'):
        sample = [e['rel'] for e in entries if e['kind'] == 'param' and e['tier'] == tier][:6]
        for rel in sample:
            print(f'  sample tier {tier}: {rel}')
    oversize = [e for e in entries if e['kind'] == 'discard' and e['why'] == 'oversize_param_like']
    for entry in sorted(oversize, key=lambda e: -e['size'])[:8]:
        print(f"  oversize parameter-like (discarded): {entry['size']/2**20:.0f} MiB {entry['rel']}")
    symlinks = [e['rel'] for e in entries if e['kind'] == 'symlink']
    if symlinks:
        print(f'{len(symlinks)} symlinks (targets are NOT vaulted), e.g. {symlinks[:3]}')


def write_jsonl(path, rows, compressed=False):
    opener = (lambda p: gzip.open(p, 'wt')) if compressed else (lambda p: open(p, 'w'))
    with opener(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
        stream.flush()


def check_no_active(base):
    active = process_runs(base)
    if active:
        sys.exit('Refusing: active processes reference ' + ', '.join(sorted(str(p) for p in active)))


def build(base, vault, repo, cap_bytes, allow_same_device=False):
    base = base.resolve()
    vault = vault.resolve()
    if base == vault or base in vault.parents or vault in base.parents:
        sys.exit('Vault must be outside the results directory')
    if vault.exists() and any(vault.iterdir()):
        sys.exit(f'Vault {vault} exists and is not empty')
    vault.mkdir(parents=True, exist_ok=True)
    if not allow_same_device and vault.stat().st_dev == base.stat().st_dev:
        sys.exit('Vault is on the same filesystem as the results; pass --allow-same-device to accept this')
    check_no_active(base)
    started = time.time()
    entries, summary = classify(base, repo, cap_bytes, log=lambda m: print(m, file=sys.stderr, flush=True))
    write_jsonl(vault/'inventory.jsonl.gz', entries, compressed=True)

    params = {}
    for entry in entries:
        if entry['kind'] == 'param':
            row = params.setdefault(entry['sha256'], dict(sha256=entry['sha256'], size=entry['size'], tier=entry['tier'],
                                                          ext=Path(entry['rel']).suffix.lower(), paths=[]))
            row['paths'].append(entry['rel'])
            if entry['tier'] == 'A':
                row['tier'] = 'A'
    (vault/'params').mkdir()
    for sha, row in sorted(params.items()):
        source = base/row['paths'][0]
        target = vault/'params'/(sha+row['ext'])
        temporary = target.with_name(target.name+'.tmp')
        shutil.copyfile(source, temporary)
        if digest(temporary) != sha:
            sys.exit(f'Copy of {source} does not match its hash; the file changed during the build')
        os.replace(temporary, target)
    write_jsonl(vault/'params_index.jsonl', params.values())

    groups = {}
    for entry in entries:
        if entry['kind'] == 'record':
            groups.setdefault(top_level(entry['rel']), []).append(entry)
    (vault/'records').mkdir()
    record_counts = {}
    for name, rows in sorted(groups.items()):
        with tarfile.open(vault/'records'/(name+'.tar.gz'), 'w:gz', compresslevel=4) as tar:
            for row in rows:
                tar.add(base/row['rel'], arcname=row['rel'], recursive=False)
        record_counts[name] = len(rows)

    kinds = Counter(e['kind'] for e in entries)
    summary.update(
        built_utc=now(), base=str(base), size_cap_bytes=cap_bytes, repo=str(repo),
        kinds=dict(kinds), unique_param_files=len(params), record_groups=record_counts,
        discarded_bytes=sum(e['size'] for e in entries if e['kind'] == 'discard'),
        discarded_by_reason=dict(Counter(e['why'] for e in entries if e['kind'] == 'discard')),
        build_seconds=round(time.time()-started, 1),
    )
    (vault/'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    (vault/'VAULT_README.txt').write_text(
        'Learned parameters and records vaulted from the results directory (user instruction 2026-11-28).\n'
        'params/<sha256>.<ext>: content-addressed parameter files; params_index.jsonl lists original paths and tiers.\n'
        'records/<run>.tar.gz: every non-array file of that run (gif files excluded).\n'
        'inventory.jsonl.gz: every source file with size, mtime and classification.\n'
        'Simulation arrays and per-candidate optimizer evaluations were not kept; raw evidence from them is unavailable.\n'
        'MANIFEST.jsonl/MANIFEST.sha256 hash every vault file; run `python -m scripts.vault_results verify`.\n')
    write_manifest(vault)
    print(f'Vault built: {vault} ({sum(1 for _ in vault.rglob("*"))} paths); next: verify, copy off-machine, then purge')


def write_manifest(vault):
    rows = []
    for path in sorted(vault.rglob('*')):
        if path.is_file() and path.name not in MANIFEST_EXCLUDED:
            rows.append(dict(path=str(path.relative_to(vault)), size=path.stat().st_size, sha256=digest(path)))
    write_jsonl(vault/'MANIFEST.jsonl', rows)
    (vault/'MANIFEST.sha256').write_text(digest(vault/'MANIFEST.jsonl')+'\n')


def verify(vault):
    vault = vault.resolve()
    manifest = vault/'MANIFEST.jsonl'
    if not manifest.exists() or not (vault/'MANIFEST.sha256').exists():
        sys.exit('No manifest in vault')
    manifest_hash = digest(manifest)
    if manifest_hash != (vault/'MANIFEST.sha256').read_text().strip():
        sys.exit('MANIFEST.jsonl does not match MANIFEST.sha256')
    listed = set()
    problems = []
    total = 0
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        path = vault/row['path']
        listed.add(row['path'])
        if not path.is_file() or path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            problems.append(row['path'])
        total += row['size']
    extra = sorted(str(p.relative_to(vault)) for p in vault.rglob('*')
                   if p.is_file() and p.name not in MANIFEST_EXCLUDED and str(p.relative_to(vault)) not in listed)
    summary = json.loads((vault/'summary.json').read_text())
    for name, count in summary['record_groups'].items():
        try:
            with tarfile.open(vault/'records'/(name+'.tar.gz')) as tar:
                if sum(1 for _ in tar) != count:
                    problems.append(f'records/{name}.tar.gz member count')
        except (OSError, tarfile.TarError):
            problems.append(f'records/{name}.tar.gz unreadable')
    if problems or extra:
        sys.exit(f'Vault verification FAILED: bad={problems[:10]} extra={extra[:10]}')
    (vault/'VERIFIED.json').write_text(json.dumps(dict(verified_utc=now(), manifest_sha256=manifest_hash,
                                                       files=len(listed), bytes=total), indent=1))
    print(f'Vault verified: {len(listed)} files, {total/2**30:.3f} GiB, manifest {manifest_hash[:12]}')
    return manifest_hash


def unchanged_source(base, vault):
    """Return source paths that are new or modified since the build inventory."""
    with gzip.open(vault/'inventory.jsonl.gz', 'rt') as stream:
        known = {row['rel']: (row['size'], row['mtime_ns'], row['kind']) for row in map(json.loads, stream)}
    changed = []
    for folder, dirs, files in os.walk(base, followlinks=False):
        folder = Path(folder)
        for name in list(dirs)+list(files):
            path = folder/name
            rel = str(path.relative_to(base))
            if path.is_symlink():
                if rel not in known or known[rel][2] != 'symlink':
                    changed.append(rel)
            elif path.is_file():
                stat = path.stat()
                if known.get(rel, (None, None))[:2] != (stat.st_size, stat.st_mtime_ns):
                    changed.append(rel)
    return changed


def purge(base, vault, confirm, apply):
    base = base.resolve()
    vault = vault.resolve()
    manifest_hash = verify(vault)
    check_no_active(base)
    changed = unchanged_source(base, vault)
    if changed:
        sys.exit(f'Refusing: {len(changed)} source files are new or changed since the build, e.g. {changed[:10]}')
    children = sorted(base.iterdir())
    with gzip.open(vault/'inventory.jsonl.gz', 'rt') as stream:
        total = sum(e['size'] for e in map(json.loads, stream))
    print(f'{len(children)} top-level entries, {total/2**30:.2f} GiB would be deleted from {base}')
    if not apply:
        print(f'DRY RUN: nothing deleted. To delete: purge --apply --confirm-delete {manifest_hash[:12]}')
        return
    if confirm != manifest_hash[:12]:
        sys.exit(f'--confirm-delete must equal {manifest_hash[:12]}')
    journal = vault/'PURGE_JOURNAL.jsonl'
    with open(journal, 'a') as stream:
        for child in children:
            stream.write(json.dumps(dict(event='start', path=str(child), utc=now()))+'\n'); stream.flush(); os.fsync(stream.fileno())
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
            stream.write(json.dumps(dict(event='deleted', path=str(child), utc=now()))+'\n'); stream.flush(); os.fsync(stream.fileno())
    print(f'Purged {len(children)} entries from {base}; journal {journal}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('plan', 'build', 'verify', 'purge'))
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--vault', type=Path)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--max-param-mb', type=float, default=100.)
    p.add_argument('--allow-same-device', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--confirm-delete')
    a = p.parse_args(argv)
    base = a.base.expanduser().resolve()
    if not base.is_dir() or base in (Path('/'), Path.home()):
        p.error('--base must be an existing results directory')
    if a.stage != 'plan' and a.vault is None:
        p.error('--vault is required')
    cap = int(a.max_param_mb*2**20)
    if a.stage == 'plan':
        entries, summary = classify(base, a.repo, cap, log=lambda m: print(m, file=sys.stderr, flush=True))
        report(entries, base)
        print(f"documented 64-hex hashes: {summary['documented_full_hashes_found']} found in results, "
              f"{len(summary['documented_full_hashes_missing'])} not present")
        for sha in summary['documented_full_hashes_missing'][:20]:
            print('  not found:', sha[:12])
        return
    import fcntl
    with open(f'/tmp/r-elegans-vault-{os.getuid()}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'build':
            build(base, a.vault, a.repo, cap, a.allow_same_device)
        elif a.stage == 'verify':
            verify(a.vault)
        else:
            purge(base, a.vault, a.confirm_delete, a.apply)


if __name__ == '__main__':
    main()
