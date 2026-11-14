#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_archive=${1:-/data/carsonc/r-elegans-results}
# Check destination access before launching into the background.
/home/carsonc/miniconda3/envs/r-elegans/bin/python -c '
import pathlib,sys,tempfile
try:
    target=pathlib.Path(sys.argv[1])
    target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile(dir=target) as stream:
        stream.write(b"destination access check");stream.flush()
    if target.stat().st_dev==pathlib.Path("/home/carsonc/r-elegans-results").stat().st_dev:
        raise RuntimeError("Destination must be on a separate filesystem")
except (OSError,RuntimeError) as error:
    print("Cleanup NOT started: "+str(error),file=sys.stderr)
    print("An administrator must grant carsonc access to the destination; no results were moved or deleted.",file=sys.stderr)
    sys.exit(1)
' "$worm_archive"
worm_log="/tmp/r-elegans-results-cleanup-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.archive_results_mbslnx2 \
    --source /home/carsonc/r-elegans-results --destination "$worm_archive" \
    --apply > "$worm_log" 2>&1 < /dev/null &
echo "Copies inactive results to /data; deletes /home duplicates only after checksum verification"
echo "Active/recent runs skipped; old paths preserved as symlinks. Originals retained on any verification error."
echo "PID: $!"
echo "Log: $worm_log"
