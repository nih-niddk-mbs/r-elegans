from pathlib import Path

import pytest

from scripts.benchmark_cpu import physical_core_cpus
from scripts.benchmark_cpu_parallel import aggregate_reports, worker_cpu_groups


def test_core_selection_stays_inside_affinity_and_distinguishes_sockets(monkeypatch):
    # Only these noncontiguous logical CPUs are available. CPUs 2 and 8 are
    # siblings; the same core ID on another socket is a different core.
    topology = {2: (0, 0), 8: (0, 0), 12: (1, 0), 15: (1, 1)}

    def read_topology(path, *args, **kwargs):
        cpu = int(path.parts[-3].removeprefix("cpu"))
        socket, core = topology[cpu]
        return str(socket if path.name == "physical_package_id" else core)

    monkeypatch.setattr(Path, "read_text", read_topology)
    cpus, known = physical_core_cpus([15, 8, 12, 2])

    assert known
    assert cpus == [2, 12, 15]


def test_unknown_topology_falls_back_to_allowed_logical_cpus(monkeypatch):
    def unavailable(path, *args, **kwargs):
        raise FileNotFoundError("topology unavailable")

    monkeypatch.setattr(Path, "read_text", unavailable)
    cpus, known = physical_core_cpus([8, 2, 15])

    assert not known
    assert cpus == [2, 8, 15]


def test_parallel_groups_are_disjoint_and_refuse_oversubscription():
    groups = worker_cpu_groups([2, 4, 8, 10, 12, 14, 18, 20], workers=4, cores=2)
    assert groups == [[2, 4], [8, 10], [12, 14], [18, 20]]
    with pytest.raises(ValueError, match="Insufficient"):
        worker_cpu_groups([2, 4, 8], workers=2, cores=2)
    with pytest.raises(ValueError, match="unique"):
        worker_cpu_groups([2, 2, 8, 10], workers=2, cores=2)


def test_aggregate_throughput_uses_shared_span_including_start_skew():
    reports = [
        dict(backend="cpu", finite_state=True,
             measurement_start_monotonic_seconds=10.0,
             measurement_end_monotonic_seconds=12.0,
             measured_environment_steps=100, environment_steps_per_second=50),
        dict(backend="cpu", finite_state=True,
             measurement_start_monotonic_seconds=11.0,
             measurement_end_monotonic_seconds=14.0,
             measured_environment_steps=150, environment_steps_per_second=50),
    ]
    result = aggregate_reports(reports)
    # Summing individual rates would incorrectly report 100 steps/s.
    assert result["aggregate_environment_steps_per_second"] == pytest.approx(62.5)
    assert result["measurement_span_seconds"] == 4.0
    reports[0]["finite_state"] = False
    with pytest.raises(ValueError, match="validation"):
        aggregate_reports(reports)
