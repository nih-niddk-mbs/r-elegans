from pathlib import Path

from scripts.benchmark_cpu import physical_core_cpus


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
