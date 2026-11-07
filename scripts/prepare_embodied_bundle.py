"""Build a small checksummed E028 input/launcher archive outside the repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


LAUNCHER='''#!/bin/sh
set -eu
worm_repo=/home/carsonc/github/r-elegans
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_inputs=$(CDPATH= cd "$(dirname "$0")" && pwd)
test -x "$worm_python" || { echo "Missing r-elegans conda Python: $worm_python"; exit 2; }
git -C "$worm_repo" pull --ff-only
cd "$worm_repo"
"$worm_python" - "$worm_inputs" "$worm_repo" <<'PY'
import hashlib,json,sys
from pathlib import Path
inputs,repo=map(Path,sys.argv[1:])
m=json.loads((inputs/'inputs.json').read_text())
for name,digest in m['sha256'].items():
    if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('Input checksum mismatch: '+name)
for name,digest in m['source_sha256'].items():
    p=repo/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Source differs from tested bundle: '+name+'; update the checkout/bundle before running.')
import jax,numpy,optax
print('Verified inputs/source; Python:',sys.executable,flush=True)
PY
mkdir -p /home/carsonc/r-elegans-results
worm_stamp=$(date -u +%Y%m%dT%H%M%SZ)
worm_log=/home/carsonc/r-elegans-results/embodied-$worm_stamp-$$.log
nohup "$worm_python" -u -m scripts.run_embodied_diagnostics \
  --inputs "$worm_inputs" --output-dir /home/carsonc/r-elegans-results \
  --minutes 45 --case-minutes 12 --duration 600 --workers 4 --cores-per-worker 4 \
  > "$worm_log" 2>&1 < /dev/null &
worm_pid=$!
echo "Started supervisor PID=$worm_pid"
echo "Log: $worm_log"
echo "You may disconnect. Monitor later with: tail -f $worm_log"
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network',required=True,type=Path)
    parser.add_argument('--parent-actor',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];output=args.output_dir.expanduser().resolve()
    if output==repo or repo in output.parents:parser.error('Keep bundles external')
    output.mkdir(parents=True,exist_ok=False);capture_run_record(output,repo,args)
    stage=output/'E028';stage.mkdir()
    for source,name in [(args.network,'network.npz'),(args.parent_actor,'parent_actor.npz')]:shutil.copy2(source,stage/name)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source_files=list((repo/'r_elegans').rglob('*.py'))+list((repo/'r_elegans/assets').glob('*.json'))
    source_files += [repo/'scripts'/n for n in ('diagnose_channel_body.py','run_embodied_diagnostics.py',
        'benchmark_cpu.py','benchmark_cpu_parallel.py','experiment_records.py')]
    manifest=dict(experiment='E028',sha256={name:sha(stage/name) for name in ('network.npz','parent_actor.npz')},
        source_sha256={str(p.relative_to(repo)):sha(p) for p in sorted(source_files)},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        note='Fixed-parameter diagnostic. Source/data verified before launch. No remote process has been started by bundle creation.')
    (stage/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (stage/'launch.sh').write_text(LAUNCHER)
    subprocess.run(['sh','-n',str(stage/'launch.sh')],check=True)
    archive=output/'r-elegans-E028.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(stage,arcname='E028')
    (output/'archive.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))),flush=True)


if __name__=='__main__':main()
