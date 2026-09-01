"""Load provenance-aware single-neuron parameters from external storage."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from r_elegans.brain.single_compartment import (
    CHANNEL_NAMES,
    SingleCompartmentParams,
)
from r_elegans.assets import load_asset_document

from .paths import DATA_ROOT_ENV, data_path

DEFAULT_PARAMETER_FILE = "processed/parameters/single_compartment_v1.json"
EVIDENCE_LEVELS = frozenset(
    {
        "direct_measurement",
        "whole_cell_fit",
        "homolog_fit",
        "expression_inference",
        "unknown",
    }
)


@dataclass(frozen=True)
class NeuronParameterRecord:
    """Runnable parameters plus non-numeric provenance retained for auditing."""

    neuron_class: str
    params: SingleCompartmentParams
    evidence: dict[str, str]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ElectrophysiologyTrace:
    """A digitized clamp table with time in seconds and SI response values."""

    time_s: np.ndarray
    responses: np.ndarray
    labels: tuple[str, ...]
    response_unit: str | None
    source_path: Path


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def load_neuron_parameters(
    neuron_class: str,
    *,
    root: str | Path | None = None,
    relative_path: str = DEFAULT_PARAMETER_FILE,
) -> NeuronParameterRecord:
    """Load a bundled neuron class or an external parameter-catalog override."""

    if (
        root is None
        and not os.environ.get(DATA_ROOT_ENV)
        and relative_path == DEFAULT_PARAMETER_FILE
    ):
        document = load_asset_document("single_compartment_v1.json")
    else:
        path = data_path(*Path(relative_path).parts, root=root)
        document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported single-compartment parameter schema")
    try:
        record = document["neurons"][neuron_class.upper()]
    except (KeyError, TypeError) as error:
        raise KeyError(f"No parameters for neuron class {neuron_class!r}") from error

    conductance_map = record.get("conductances_nS", {})
    if set(conductance_map) != set(CHANNEL_NAMES):
        missing = sorted(set(CHANNEL_NAMES) - set(conductance_map))
        extra = sorted(set(conductance_map) - set(CHANNEL_NAMES))
        raise ValueError(f"Conductance channel mismatch; missing={missing}, extra={extra}")
    conductances = jnp.asarray([_number(conductance_map, name) for name in CHANNEL_NAMES])
    if bool(jnp.any(conductances < 0)):
        raise ValueError("Maximal conductances must be nonnegative")

    evidence = record.get("evidence", {})
    if set(evidence) != set(CHANNEL_NAMES):
        raise ValueError("Every conductance requires an evidence label")
    invalid_evidence = set(evidence.values()) - EVIDENCE_LEVELS
    if invalid_evidence:
        raise ValueError(f"Invalid evidence labels: {sorted(invalid_evidence)}")

    params = SingleCompartmentParams(
        capacitance_pF=jnp.asarray(_number(record, "capacitance_pF")),
        conductances_nS=conductances,
        potassium_reversal_mV=jnp.asarray(_number(record, "potassium_reversal_mV")),
        calcium_reversal_mV=jnp.asarray(_number(record, "calcium_reversal_mV")),
        sodium_reversal_mV=jnp.asarray(_number(record, "sodium_reversal_mV")),
        leak_reversal_mV=jnp.asarray(_number(record, "leak_reversal_mV")),
        cell_volume_um3=jnp.asarray(_number(record, "cell_volume_um3")),
        calcium_equilibrium_uM=jnp.asarray(_number(record, "calcium_equilibrium_uM")),
        calcium_removal_ms=jnp.asarray(_number(record, "calcium_removal_ms")),
        free_calcium_fraction=jnp.asarray(_number(record, "free_calcium_fraction")),
    )
    if float(params.capacitance_pF) <= 0 or float(params.cell_volume_um3) <= 0:
        raise ValueError("Capacitance and cell volume must be positive")
    return NeuronParameterRecord(
        neuron_class=neuron_class.upper(),
        params=params,
        evidence=dict(evidence),
        source_ids=tuple(record.get("source_ids", ())),
    )


def load_electrophysiology_trace(
    relative_path: str | Path,
    *,
    root: str | Path | None = None,
) -> ElectrophysiologyTrace:
    """Load an upstream tab-delimited clamp trace without altering its units.

    Several public digitizations prepend zero, one, or three metadata rows.
    The first row whose columns are all numeric marks the data matrix. Labels
    and a response-unit row are retained when present.
    """

    path = data_path(*Path(relative_path).parts, root=root)
    rows: list[list[str]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            cleaned = [value.strip() for value in row]
            if any(cleaned):
                rows.append(cleaned)
    if not rows:
        raise ValueError(f"Electrophysiology file is empty: {path}")

    data_start = None
    for index, row in enumerate(rows):
        try:
            [float(value) for value in row]
        except ValueError:
            continue
        data_start = index
        break
    if data_start is None:
        raise ValueError(f"Electrophysiology file has no numeric rows: {path}")

    width = len(rows[data_start])
    if width < 2:
        raise ValueError("A clamp trace requires time and at least one response")
    numeric_rows = []
    for row in rows[data_start:]:
        if len(row) != width:
            raise ValueError(f"Inconsistent column count in {path.name}")
        try:
            numeric_rows.append([float(value) for value in row])
        except ValueError as error:
            raise ValueError(f"Non-numeric value inside data matrix in {path.name}") from error
    matrix = np.asarray(numeric_rows, dtype=np.float64)

    if data_start:
        header = rows[0]
        labels = tuple(
            (header[index].strip() if index < len(header) else "")
            or f"response_{index}"
            for index in range(1, width)
        )
    else:
        labels = tuple(f"response_{index}" for index in range(1, width))
    response_unit = None
    if data_start > 1 and len(rows[1]) > 1:
        units = {value for value in rows[1][1:width] if value}
        if len(units) == 1:
            response_unit = units.pop()

    return ElectrophysiologyTrace(
        time_s=matrix[:, 0],
        responses=matrix[:, 1:],
        labels=labels,
        response_unit=response_unit,
        source_path=path,
    )
