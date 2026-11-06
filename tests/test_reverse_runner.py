"""An unattended reverse run must diagnose both directions and retain all gates."""
import json
import hashlib
from types import SimpleNamespace

import pytest

from scripts import run_motor_curriculum as runner


@pytest.mark.parametrize("failed_stage", [None, "reverse", "forward_retention"])
@pytest.mark.parametrize("refine", [False, True])
def test_reverse_runner_checks_both_directions_even_after_a_failed_gate(tmp_path, monkeypatch, failed_stage, refine):
    source = tmp_path / "wave-source"
    checkpoint = source / "seed-0/wave/actor_best.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"source checkpoint")
    (source / "summary.json").write_text(json.dumps(dict(workers=[dict(seed=0, stages=dict(
        wave=dict(passed=True, exit_code=0, status="completed")))])))
    if refine:
        candidate = source / "seed-0/reverse/actor_best.npz"
        candidate.parent.mkdir()
        candidate.write_bytes(b"candidate")
        (candidate.parent/"manifest.json").write_text(json.dumps(dict(initial_actor=str(checkpoint),
            initial_actor_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
        complete = dict(status="completed", error=None, exit_code=2, passed=False, control_passed=True)
        (source/"summary.json").write_text(json.dumps(dict(workers=[dict(seed=0,
            stages={n:dict(complete) for n in ("reverse","forward_retention","reverse_locomotion")})])))
    output = tmp_path / "results"
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            folder = runner.Path(command[command.index("--output-dir") + 1])
            stage = folder.name
            launched.append((stage, command))
            folder.mkdir()
            passed = stage != failed_stage
            self.returncode = 0 if passed else 2
            self.pid = 123
            (folder / "summary.json").write_text(json.dumps(dict(
                passed=passed, status="completed", error=None)))
            if stage == "reverse":
                (folder / "actor_best.npz").write_bytes(b"candidate")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(runner, "physical_core_cpus", lambda _: ([0], True))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *a: None)
    monkeypatch.setattr(runner, "capture_run_record", lambda *a: None)
    budget = 480 if refine else 180
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--refine-reverse" if refine else "--reverse", "--minutes", str(budget),
        "--workers", "1", "--cores-per-worker", "1", "--resume-run", str(source),
        "--output-dir", str(output)])
    assert runner.main() == (0 if failed_stage is None else 2)
    assert [s for s, _ in launched] == ["reverse", "forward_retention", "reverse_locomotion"]
    training = launched[0][1]
    assert budget-2 < float(training[training.index("--minutes") + 1]) <= budget-1
    if refine:
        assert training[2].endswith("refine_reverse.py")
    assert "--direction" not in launched[1][1]
    assert launched[2][1][-2:] == ["--direction", "reverse"]
