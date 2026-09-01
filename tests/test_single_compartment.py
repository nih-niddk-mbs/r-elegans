import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain import (
    CHANNEL_INDEX,
    CHANNEL_NAMES,
    GATE_COUNT,
    SingleCompartmentParams,
    initial_single_compartment_state,
    integrate_single_compartment,
    ionic_currents,
    single_compartment_rhs,
)
from r_elegans.data import load_electrophysiology_trace, load_neuron_parameters


def parameters(conductance: float = 0.1) -> SingleCompartmentParams:
    return SingleCompartmentParams(
        capacitance_pF=jnp.asarray(3.1),
        conductances_nS=jnp.full((len(CHANNEL_NAMES),), conductance),
        potassium_reversal_mV=jnp.asarray(-80.0),
        calcium_reversal_mV=jnp.asarray(60.0),
        sodium_reversal_mV=jnp.asarray(30.0),
        leak_reversal_mV=jnp.asarray(-90.0),
        cell_volume_um3=jnp.asarray(31.16),
        calcium_equilibrium_uM=jnp.asarray(0.05),
        calcium_removal_ms=jnp.asarray(33.0),
        free_calcium_fraction=jnp.asarray(0.001),
    )


def test_equilibrium_state_and_currents_are_finite_and_jittable() -> None:
    params = parameters()
    state = initial_single_compartment_state(-70.0, params)

    currents = jax.jit(ionic_currents)(state, params)
    derivative = jax.jit(single_compartment_rhs)(0.0, state, (params, jnp.asarray(0.0)))

    assert state.gates.shape == (GATE_COUNT,)
    assert currents.shape == (len(CHANNEL_NAMES),)
    assert jnp.all((state.gates >= 0.0) & (state.gates <= 1.0))
    assert jnp.all(jnp.isfinite(currents))
    assert jnp.all(jnp.isfinite(derivative.gates))
    assert jnp.isfinite(derivative.voltage_mV)
    assert jnp.isfinite(derivative.calcium_uM)


def test_zero_conductance_has_only_injected_current() -> None:
    params = parameters(conductance=0.0)
    state = initial_single_compartment_state(-70.0, params)
    derivative = single_compartment_rhs(0.0, state, (params, jnp.asarray(6.2)))

    np.testing.assert_allclose(derivative.voltage_mV, 2.0, rtol=1e-6)
    np.testing.assert_allclose(ionic_currents(state, params), 0.0)


def test_short_constant_current_integration_has_expected_voltage_change() -> None:
    params = parameters(conductance=0.0)
    state = initial_single_compartment_state(-70.0, params)

    final = integrate_single_compartment(state, params, 6.2, 1.0)

    np.testing.assert_allclose(final.voltage_mV, -68.0, atol=2e-4)
    np.testing.assert_allclose(final.calcium_uM, 0.05, atol=1e-6)


def test_nca_and_leak_are_ohmic_and_outward_positive() -> None:
    params = parameters(conductance=0.0)
    conductances = params.conductances_nS.at[CHANNEL_INDEX["NCA"]].set(0.2)
    conductances = conductances.at[CHANNEL_INDEX["LEAK"]].set(0.4)
    params = params._replace(conductances_nS=conductances)
    state = initial_single_compartment_state(-70.0, params)
    currents = ionic_currents(state, params)

    np.testing.assert_allclose(currents[CHANNEL_INDEX["NCA"]], -20.0)
    np.testing.assert_allclose(currents[CHANNEL_INDEX["LEAK"]], 8.0)


def test_gradient_flows_through_maximal_conductances() -> None:
    base = parameters()
    state = initial_single_compartment_state(-55.0, base)

    def voltage_slope(conductances: jax.Array) -> jax.Array:
        params = base._replace(conductances_nS=conductances)
        return single_compartment_rhs(0.0, state, (params, jnp.asarray(5.0))).voltage_mV

    gradient = jax.jit(jax.grad(voltage_slope))(base.conductances_nS)

    assert gradient.shape == (len(CHANNEL_NAMES),)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_external_parameter_loader_preserves_evidence(tmp_path) -> None:
    evidence = {name: "whole_cell_fit" for name in CHANNEL_NAMES}
    conductances = {name: 0.0 for name in CHANNEL_NAMES}
    document = {
        "schema_version": 1,
        "neurons": {
            "AWCON": {
                "capacitance_pF": 3.1,
                "conductances_nS": conductances,
                "potassium_reversal_mV": -80.0,
                "calcium_reversal_mV": 60.0,
                "sodium_reversal_mV": 30.0,
                "leak_reversal_mV": -90.0,
                "cell_volume_um3": 31.16,
                "calcium_equilibrium_uM": 0.05,
                "calcium_removal_ms": 33.0,
                "free_calcium_fraction": 0.001,
                "evidence": evidence,
                "source_ids": ["nicoletti2019"],
            }
        },
    }
    target = tmp_path / "processed/parameters"
    target.mkdir(parents=True)
    (target / "single_compartment_v1.json").write_text(json.dumps(document))

    record = load_neuron_parameters("awcon", root=tmp_path)

    assert record.neuron_class == "AWCON"
    assert record.evidence["CCA1"] == "whole_cell_fit"
    assert record.source_ids == ("nicoletti2019",)
    np.testing.assert_allclose(record.params.capacitance_pF, 3.1)


def test_parameter_loader_rejects_missing_channel(tmp_path) -> None:
    target = tmp_path / "processed/parameters"
    target.mkdir(parents=True)
    (target / "single_compartment_v1.json").write_text(
        json.dumps({"schema_version": 1, "neurons": {"AWCON": {"conductances_nS": {}}}})
    )

    with pytest.raises(ValueError, match="channel mismatch"):
        load_neuron_parameters("AWCON", root=tmp_path)


def test_trace_loader_handles_upstream_metadata_rows(tmp_path) -> None:
    target = tmp_path / "raw/physiology"
    target.mkdir(parents=True)
    trace_path = target / "trace.txt"
    trace_path.write_text(
        "S\t0mV\t10mV\n"
        "s\tV\tV\n"
        "processing note\tcolumn one\tcolumn two\n"
        "0.0\t-0.07\t-0.06\n"
        "0.1\t-0.05\t-0.04\n"
    )

    trace = load_electrophysiology_trace("raw/physiology/trace.txt", root=tmp_path)

    assert trace.labels == ("0mV", "10mV")
    assert trace.response_unit == "V"
    np.testing.assert_allclose(trace.time_s, [0.0, 0.1])
    np.testing.assert_allclose(trace.responses, [[-0.07, -0.06], [-0.05, -0.04]])
