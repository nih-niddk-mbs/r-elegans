"""Standard-library-only invocation, environment and source records."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_run_record(output, repo, arguments=None, *, phase="run_start"):
    """Never overwrite an earlier invocation. Snapshot source, not ignored data.

    Environment variables are an explicit numerical-settings allowlist; secrets
    and arbitrary environment variables are not recorded.
    """
    output, repo = Path(output), Path(repo).resolve()
    timestamp = datetime.now(timezone.utc)
    record = output / "run_records" / (timestamp.strftime("%Y%m%dT%H%M%S%fZ")+"-"+uuid.uuid4().hex[:8])
    record.mkdir(parents=True, exist_ok=False)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)
    packages = sorted({(d.metadata["Name"], d.version) for d in metadata.distributions() if d.metadata["Name"]})
    settings = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
        "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    environment = dict(phase=phase, captured_at_utc=timestamp.isoformat(),
        python_executable=sys.executable, python_version=sys.version,
        python_prefix=sys.prefix, python_path=sys.path,
        platform=platform.platform(), architecture=platform.machine(), hostname=platform.node(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        numerical_environment={k:os.environ[k] for k in settings if k in os.environ},
        packages=dict(packages), package_record_limitations="Versions, not original wheel hashes or a container image")
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"],capture_output=True,text=True,timeout=15)
            environment["gpu_driver_inventory"] = dict(returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired:
            environment["gpu_driver_inventory"] = dict(error="nvidia-smi timed out")
    (record/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    (record/"command.json").write_text(json.dumps(dict(argv=sys.argv,
        interpreter_argv=getattr(sys,"orig_argv",[sys.executable,*sys.argv]), cwd=os.getcwd(),
        parsed_arguments=vars(arguments) if hasattr(arguments,"__dict__") else arguments),indent=2,default=str)+"\n")
    # The project itself is restored from the bundled repository/source snapshot.
    (record/"requirements-recorded.txt").write_text("".join(f"{n}=={v}\n" for n,v in packages if n.lower().replace('_','-') != "r-elegans"))
    files, omitted = {}, []
    tracked = set(git("ls-files","--cached","-z").split(b"\0")) - {b""}
    untracked = set(git("ls-files","--others","--exclude-standard","-z").split(b"\0")) - {b""}
    # Do not sweep arbitrary untracked private configuration/data into records.
    # Python additions are needed to preserve an experimental dirty source tree.
    additions = {p for p in untracked if Path(os.fsdecode(p)).suffix == '.py'}
    for raw in sorted(tracked | additions):
        if not raw:
            continue
        name = os.fsdecode(raw)
        source = repo / name
        if source.is_symlink() or not source.is_file():
            omitted.append(name)
            continue
        destination = record / "source" / name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        files[name] = file_sha256(destination)
    (record/"source_record.json").write_text(json.dumps(dict(
        git_revision=git("rev-parse","HEAD").decode().strip(),
        git_status=git("status","--porcelain").decode(), source_sha256=files,
        omitted_symlinks_or_missing=omitted,
        omitted_untracked_non_python=sorted(os.fsdecode(p) for p in untracked-additions)),indent=2)+"\n")
    return record
