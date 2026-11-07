"""Fit conditional physical-unit conductances without replacing trained actors.

Channels: VD05 digitized voltage clamp with held-out voltage steps, explicit
AWCON kinetics transfer and protocol assumptions. Synapses: externally supplied
paired clamp recordings, anatomical edges, fixed release/reversal parameters.
All data/candidates/trajectories stay external. Neither command installs a fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_fitting import (
    voltage_clamp_trace, fit_clamp_conductances, synaptic_design,
    fit_synaptic_conductances,
)
from r_elegans.brain.single_compartment import CHANNEL_NAMES
from r_elegans.data import load_neuron_parameters, load_electrophysiology_trace
from r_elegans.data.paths import get_data_root
from r_elegans.model import load_builtin_model
if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fit_channels(args, output):
    root = get_data_root(args.data_root)
    relative = "raw/physiology/baaiworm2024/VD05_Voltage_Clamp_Trace.txt"
    trace = load_electrophysiology_trace(relative, root=root)
    if trace.labels != ("-10mV", "0mV", "10mV", "20mV", "30mV", "40mV"):
        raise ValueError("VD05 sweep labels disagree with verified protocol")
    if sha(trace.source_path) != "919954fef9dd18e0ee96b586dd52be11a21906da35c8a4f0086e88cca2760656":
        raise ValueError("VD05 trace differs from the audited pinned dataset")
    # Preserve the exact external preset; this is explicitly a cross-class prior.
    record = load_neuron_parameters("AWCON", root=root)
    params = record.params
    shutil.copy2(trace.source_path, output/trace.source_path.name)
    prior_path = root/"processed/parameters/single_compartment_v1.json"
    shutil.copy2(prior_path, output/"source_parameter_catalog.json")
    time = trace.time_s[::5]
    measured = trace.responses[::5]*1e12  # A -> pA, independently audited source units
    dt = float(np.median(np.diff(time)))*1000.
    if not np.allclose(np.diff(time)*1000., dt, rtol=1e-5, atol=1e-7):
        raise ValueError("Uniform clamp sampling required")
    levels = jnp.array([-10.,0.,10.,20.,30.,40.])
    pulse = (time >= -1e-9) & (time < 1.21-1e-9)
    protocols = jnp.where(jnp.asarray(pulse)[None,:], levels[:,None], -60.)
    fit_times = (time >= .005) & (time <= 1.20)
    baseline = np.flatnonzero((time < -.005) & (time > -.025))
    train = np.array([0,2,3,5])
    heldout = np.array([1,4])  # entire 0 and 30 mV sweeps excluded from fitting

    def predict(g, substeps=8):
        p = params._replace(conductances_nS=g)
        predicted = jax.vmap(lambda v: voltage_clamp_trace(p,v,dt,substeps=substeps))(protocols)
        return predicted-jnp.mean(predicted[:,baseline],axis=1,keepdims=True)

    target = measured.T
    prior = np.maximum(np.asarray(params.conductances_nS), .01)
    initial = np.asarray(jax.jit(predict)(jnp.asarray(prior)))
    g, diagnostics = fit_clamp_conductances(
        lambda g: predict(g)[train][:,fit_times], target[train][:,fit_times], prior,
        max_evaluations=args.max_evaluations)
    predicted = np.asarray(jax.jit(predict)(jnp.asarray(g)))
    refined = np.asarray(jax.jit(lambda g: predict(g,16))(jnp.asarray(g)))
    scale = diagnostics["data_scale_pA"]

    def rmse(pred, selection):
        return float(np.sqrt(np.mean((pred[selection][:,fit_times]-target[selection][:,fit_times])**2)))

    validation = rmse(predicted,heldout)
    numerical = float(np.sqrt(np.mean((refined[:,fit_times]-predicted[:,fit_times])**2)))
    report = dict(experiment="E025", kind="voltage_clamp_candidate", neuron="VD05",
        evidence="whole_cell_fit_conditional_on_transferred_kinetics_and_protocol_assumptions",
        source_trace_sha256=sha(trace.source_path), source_parameter_sha256=sha(prior_path),
        source_repository_commit="b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc",
        source_protocol_url="https://github.com/Jessie940611/BAAIWorm/blob/b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc/eworm/model_figure/single_neuron_features.py",
        prior_class="AWCON", fixed_parameters={k:float(v) for k,v in params._asdict().items() if k != "conductances_nS"},
        fixed_kinetics="Existing Nicoletti-derived AWCON/RMD library; not VD-specific kinetics",
        conductances_nS=dict(zip(CHANNEL_NAMES,g.tolist())), optimizer=diagnostics,
        dt_ms=dt, substeps=8, refinement_substeps=16,
        train_steps_mV=levels[train].tolist(), heldout_steps_mV=levels[heldout].tolist(),
        initial_train_rmse_pA=rmse(initial,train), train_rmse_pA=rmse(predicted,train),
        initial_heldout_rmse_pA=rmse(initial,heldout), heldout_rmse_pA=validation,
        numerical_rmse_pA=numerical, train_current_rms_pA=scale,
        assumptions=["Holding -60 mV borrowed from upstream simulation configuration",
            "1.21 s pulse; digitized response in A, converted to pA",
            "Padded zero baselines treated as holding-current-subtracted",
            "Exclude first 5 ms and pulse ending; capacitance/series resistance omitted",
            "Cross-class kinetics and calcium parameters fixed; no expression constraints",
            "No DA/VA/DB/VB channels inferred by this VD5 candidate"],
        gates=dict(optimizer_success=diagnostics["optimizer_success"],
            heldout_adequacy=validation <= .2*scale, numerical_convergence=numerical <= .01*scale),
        installed=False)
    report["passed"] = all(report["gates"].values())
    np.savez_compressed(output/"voltage_clamp_fit.npz", time_s=time,
        protocols_mV=np.asarray(protocols), measured_pA=target, initial_pA=initial,
        fitted_pA=predicted, refined_pA=refined)
    save_json(output/"candidate.json", report)
    return report


def fit_synapses(args, output):
    metadata = json.loads(args.metadata.read_text())
    required = {"voltage_unit":"mV", "current_unit":"pA",
        "current_definition":"outward_intrinsic_and_capacitance_subtracted"}
    if any(metadata.get(k) != v for k,v in required.items()) or not metadata.get("source"):
        raise ValueError("Explicit units, residual-current definition and source required")
    model = load_builtin_model()
    with np.load(args.recordings, allow_pickle=False) as data:
        ids = tuple(data["neuron_ids"].tolist())
        voltage, current = data["voltage_mV"], data["residual_current_pA"]
        reversal, threshold, slope = data["reversal_mV"], data["threshold_mV"], data["slope_mV"]
    if len(ids) != len(set(ids)) or set(ids)-set(model.neuron_ids):
        raise ValueError("Unique canonical neuron names required")
    if len(ids) < 302 and metadata.get("isolated_subcircuit") is not True:
        raise ValueError("Subset recordings must explicitly document isolated_subcircuit=true")
    if not metadata.get("intrinsic_subtraction_source"):
        raise ValueError("Document the source of intrinsic/capacitive current subtraction")
    if (voltage.ndim != 3 or voltage.shape != current.shape or voltage.shape[2] != len(ids)
            or voltage.shape[0] < 2 or not np.isfinite(current).all()):
        raise ValueError("Need matching [trial,time,neuron] arrays with >=2 independent trials")
    idx = np.asarray([model.neuron_ids.index(n) for n in ids])
    chemical = np.asarray(model.connectome.chemical_counts)[np.ix_(idx,idx)] > 0
    gap = np.asarray(model.connectome.gap_counts)[np.ix_(idx,idx)] > 0
    if args.heldout_trial < 0 or args.heldout_trial >= voltage.shape[0]:
        raise ValueError("Heldout trial outside recordings")
    heldout = args.heldout_trial
    train = [i for i in range(voltage.shape[0]) if i != heldout]
    design, edges = synaptic_design(voltage[train].reshape(-1,len(ids)), chemical,gap,reversal,threshold,slope)
    test_design, _ = synaptic_design(voltage[heldout],chemical,gap,reversal,threshold,slope)
    g, diagnostics = fit_synaptic_conductances(design, current[train])
    prediction = test_design@g
    rmse = float(np.sqrt(np.mean((prediction-current[heldout].ravel())**2)))
    report = dict(experiment="E025", kind="conditional_synaptic_fit", metadata=metadata,
        recordings_sha256=sha(args.recordings), metadata_sha256=sha(args.metadata),
        neuron_ids=ids, heldout_trial=heldout, training_trials=train,
        edges=[dict(kind=e[0],post=ids[e[1]],pre=ids[e[2]],conductance_nS=float(x)) for e,x in zip(edges,g)],
        optimizer=diagnostics, heldout_rmse_pA=rmse,
        scope="Paired-clamp effective conductances; known intrinsic subtraction/release/reversal required",
        limitations="No unique-parameter claim; omitted neurons must be electrically isolated in recording",
        installed=False)
    # A large full-connectome matrix can be too expensive to densify just for
    # diagnostics; report that limitation instead of implying identified edges.
    if design.shape[0]*design.shape[1] <= 5_000_000:
        from r_elegans.brain.conductance_fitting import sensitivity_diagnostics
        report["sensitivity"] = sensitivity_diagnostics(design.toarray())
    else:
        report["sensitivity"] = {"status":"not_computed", "reason":"dense diagnostic size limit",
            "unique_conductances_established":False}
    shutil.copy2(args.recordings,output/"source_recordings.npz")
    shutil.copy2(args.metadata,output/"source_metadata.json")
    save_json(output/"candidate.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage",required=True)
    channel = sub.add_parser("channels")
    channel.add_argument("--data-root",type=Path,required=True)
    channel.add_argument("--max-evaluations",type=int,default=100)
    synapse = sub.add_parser("synapses")
    synapse.add_argument("--recordings",type=Path,required=True)
    synapse.add_argument("--metadata",type=Path,required=True)
    synapse.add_argument("--heldout-trial",type=int,default=0)
    for child in (channel,synapse):
        child.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        parser.error("Keep scientific artifacts outside the repository")
    output.mkdir(parents=True,exist_ok=False)
    capture_run_record(output,repo,args)
    report = fit_channels(args,output) if args.stage == "channels" else fit_synapses(args,output)
    print(json.dumps({k:v for k,v in report.items() if k not in ("conductances_nS","fixed_parameters","edges")}),flush=True)
    failed = report.get("passed") is False or report.get("optimizer",{}).get("optimizer_success") is False
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__":
    main()
