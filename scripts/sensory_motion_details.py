"""Read-only E093/E094 bounds, matched progress and sampled signal/current report.

Currents are reconstructed at saved samples. They are not integrated current
balances, physiological measurements or new causal interventions.
"""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.special import expit
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.audit_rial_sensory import frozen_network_fields
from r_elegans.relay_cap_experiments import upper_for
from r_elegans.sensory_motion import score


def edge_summary(v0,v1,network,post,pre):
    g=float(network['chemical'][post,pre]);gap=float(network['gap'][post,pre]);e=float(network['reversal'][pre])
    r0=expit((v0[:,pre]-network['threshold'][pre])/network['slope'][pre])
    r1=expit((v1[:,pre]-network['threshold'][pre])/network['slope'][pre])
    c0=g*r0*(e-v0[:,post]);c1=g*r1*(e-v1[:,post])
    direct_c=g*(r1-r0)*(e-v0[:,post]);direct_g=gap*(v1[:,pre]-v0[:,pre])
    actual_gap=gap*((v1[:,pre]-v1[:,post])-(v0[:,pre]-v0[:,post]))
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    return dict(source=network['ids'][pre],target=network['ids'][post],chemical_nS=g,gap_nS=gap,reversal_mV=e,
        no_cue_release_mean=float(r0.mean()),cue_release_mean=float(r1.mean()),
        no_cue_chemical_mean_inward_pA=float(c0.mean()),actual_chemical_delta_mean_pA=float(np.mean(c1-c0)),
        actual_total_delta_rms_pA=rms(c1-c0+actual_gap),
        release_only_delta_mean_pA=float(direct_c.mean()),release_only_delta_rms_pA=rms(direct_c),
        presynaptic_gap_delta_mean_pA=float(direct_g.mean()),
        presynaptic_total_delta_rms_pA=rms(direct_c+direct_g))


def bound_rows(groups,z,origin,offsets,lower,original_upper,edge):
    upper=upper_for('aiyr_2p5',original_upper,edge);rows=[]
    for value,g in zip(z,groups):
        ix=np.asarray(g['indices'],dtype=int);delta=offsets[ix]-origin[ix]
        rows.append(dict(name=g['name'],coordinates=len(ix),latent=float(value),near_local_bound=bool(abs(value)>=.99),
            local_log_delta_range=[float(delta.min()),float(delta.max())],
            original_upper_exceptions=int(np.sum(offsets[ix]>original_upper[ix]+1e-10)),
            minimum_original_upper_room_log=float(np.min(original_upper[ix]-offsets[ix])),
            minimum_declared_lower_room_log=float(np.min(offsets[ix]-lower[ix])),
            minimum_declared_upper_room_log=float(np.min(upper[ix]-offsets[ix])),
            near_declared_bound_coordinates=int(np.sum(np.minimum(offsets[ix]-lower[ix],upper[ix]-offsets[ix])<=.01*(upper[ix]-lower[ix])))))
    return rows


def analyze(run,out=None):
    run=Path(run).resolve()
    if out is None:out=run.with_name(run.name+'-details-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out=Path(out).resolve();repo=Path(__file__).resolve().parents[1]
    if any(out==r or r in out.parents or out in r.parents for r in (run,repo)):raise ValueError('Separate external analysis output required')
    s=json.loads((run/'summary.json').read_text());audit_folder=run.with_name(run.name+'-audit')
    audit=json.loads((audit_folder/'summary.json').read_text());receipt=json.loads((run/'budget_receipt.json').read_text())
    if s.get('experiment') not in ('E093','E094') or not s.get('execution_valid') or not s.get('validation_complete') or not receipt.get('all_steps_finished') or not audit.get('integrity_passed') or not audit.get('audit_complete'):
        raise ValueError('Completed E093/E094 with clean saved-data audit required')
    if len(s['panels'])!=1:raise ValueError('Unique motion panel required')
    config=json.loads((run/'configuration.json').read_text());worst=bool(config.get('worst_region_term'));bilateral=bool(config.get('bilateral'))
    panel=s['panels'][0];folder=Path(panel['training_output']).resolve()
    if run not in folder.parents:raise ValueError('Selected model outside source run')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,dict(run=str(run),output_dir=str(out),method='saved-data-only'))
    evidence=[]
    def record(path,expected=None):
        sha=file_sha256(path)
        if expected is not None and sha!=expected:raise ValueError('Hash mismatch: '+str(path))
        evidence.append(dict(path=str(path),sha256=sha));return sha
    def load(path):record(path);return json.loads(path.read_text())
    for path in (run/'summary.json',run/'configuration.json',run/'budget_receipt.json',audit_folder/'summary.json'):record(path)
    original_evidence=load(audit_folder/'evidence.json');audited={e['path']:e['sha256'] for e in original_evidence}
    def verify_audited(path,expected=None):
        sha=record(path,expected)
        if audited.get(str(path))!=sha:raise ValueError('File differs from audit evidence: '+str(path))
    model=folder/'network.npz';verify_audited(model,panel['checkpoint_sha256']);verify_audited(folder/'coordinates.npz')
    manifest=load(run/'inputs/manifest.json')
    for name,sha in manifest['sha256'].items():record(run/'inputs'/name,sha)
    with np.load(run/'inputs/cap_prior.npz',allow_pickle=False) as z:
        origin,lower,upper,edge=z['origin'],z['lower'],z['upper'],int(z['edge'])
    with np.load(folder/'coordinates.npz',allow_pickle=False) as z:offsets,point=z['offsets'],z['z']
    groups=load(run/'inputs/cap_groups.json')['aiyr_2p5'];parent=load(run/'parent_features.json')
    initial=load(run/'evaluations/parent-redundant/summary.json');selected=load(folder/'summary.json')
    target=s['target'];baseline_score=score(initial['paired_features'],parent,target,1.,worst)
    selected_score=score(selected['paired_features'],parent,target,1.,worst)
    history=load(run/'generation_history.json') if (run/'generation_history.json').exists() else []
    progress=[]
    for entry in history:
        path=Path(entry['selected_evaluation']).resolve()
        if run not in path.parents:raise ValueError('History model outside source run')
        r=load(path/'summary.json');m=score(r['paired_features'],parent,target,1.,worst)
        progress.append(dict(generation=entry['generation'],checkpoint_sha256=r['checkpoint_sha256'],loss_at_final_target=m['loss'],
            minimum_regional_suppression_percent=float(100*(1-np.max(m['regional_motion_ratios'])))))
    from r_elegans.avb_audit import load_saved_network
    n=load_saved_network(model);nominal=panel['validation'][0];trace_folder=Path(nominal['output']).resolve()
    if run not in trace_folder.parents:raise ValueError('Nominal traces outside source run')
    verify_audited(trace_folder/'network.npz',panel['checkpoint_sha256'])
    traces=[]
    for condition in ('no_cue','cue'):
        path=trace_folder/(condition+'.npz');verify_audited(path,nominal['trajectory_sha256'][condition])
        with np.load(path,allow_pickle=False) as z:traces.append((z['dense_time_s'],z['dense_voltage_mV']))
    (t,v0),(u,v1)=traces
    if not np.array_equal(t,u) or v0.shape!=v1.shape or np.any(np.diff(t)<=0) or not np.isfinite(v0).all() or not np.isfinite(v1).all():raise ValueError('Finite matched voltage clocks required')
    stride=max(1,int(round(.01/float(np.median(np.diff(t))))));signals=[];rial=n['ids'].index('RIAL')
    for category in ('active','response','late'):
        for feature in nominal['paired_features']['no_cue'][category]:
            a,b=feature['window'];ix=np.flatnonzero((t>=a)&(t<b))[::stride]
            if len(ix)<4:raise ValueError('Missing saved signal window')
            a0,b0=v0[ix],v1[ix];cells=[]
            for i,name in enumerate(n['ids']):
                if name not in ('AWAL','AWAR','AIYL','AIYR','RIAL','RIAR') and not name.startswith('SMD'):continue
                ra=expit((a0[:,i]-n['threshold'][i])/n['slope'][i]);rb=expit((b0[:,i]-n['threshold'][i])/n['slope'][i])
                cells.append(dict(neuron=name,no_cue_voltage_mean_mV=float(a0[:,i].mean()),cue_voltage_mean_mV=float(b0[:,i].mean()),
                    voltage_difference_rms_mV=float(np.sqrt(np.mean((a0[:,i]-b0[:,i])**2))),
                    no_cue_release_mean=float(ra.mean()),cue_release_mean=float(rb.mean()),release_mean_change=float(np.mean(rb-ra)),
                    no_cue_release_std=float(ra.std()),cue_release_std=float(rb.std()),
                    no_cue_release_q01_q99=np.quantile(ra,[.01,.99]).tolist()))
            incoming=[edge_summary(a0,b0,n,rial,i) for i in range(len(n['ids'])) if n['chemical'][rial,i]>0 or n['gap'][rial,i]>0]
            incoming.sort(key=lambda e:e['presynaptic_total_delta_rms_pA'],reverse=True)
            ai=n['ids'].index('AIYL')
            output=[edge_summary(a0,b0,n,i,rial) for i,name in enumerate(n['ids']) if name.startswith('SMD') and (n['chemical'][i,rial]>0 or n['gap'][i,rial]>0)]
            signals.append(dict(category=category,window=[a,b],samples=len(ix),sample_stride=stride,cells=cells,
                AIYL_to_RIAL=edge_summary(a0,b0,n,rial,ai),top_RIAL_presynaptic_current_changes=incoming[:6],RIAL_to_SMD=output))
            if bilateral:
                rr=n['ids'].index('RIAR');air=n['ids'].index('AIYR')
                right_in=sorted([edge_summary(a0,b0,n,rr,i) for i in range(len(n['ids'])) if n['chemical'][rr,i]>0 or n['gap'][rr,i]>0],key=lambda e:e['presynaptic_total_delta_rms_pA'],reverse=True)
                signals[-1].update(AIYR_to_RIAR=edge_summary(a0,b0,n,rr,air),top_RIAR_presynaptic_current_changes=right_in[:6],
                    RIAR_to_SMD=[edge_summary(a0,b0,n,i,rr) for i,name in enumerate(n['ids']) if name.startswith('SMD') and (n['chemical'][i,rr]>0 or n['gap'][i,rr]>0)])
    bounds=bound_rows(groups,point,origin,offsets,lower,upper,edge)
    result=dict(run=str(run),checkpoint_sha256=panel['checkpoint_sha256'],target=target,
        baseline_and_selected_protocol='redundant; both re-scored at final target and margin1',
        baseline=baseline_score,selected=selected_score,
        loss_reduction_percent=(100*(baseline_score['loss']-selected_score['loss'])/baseline_score['loss'] if baseline_score['loss']>0 else None),
        completed_generation_progress=progress,changed_coordinates=int(np.sum(offsets!=origin)),
        frozen_fields_unchanged=frozen_network_fields(run/'inputs/motion_parent.npz',model,offsets!=origin),
        groups=bounds,signal_windows=signals,
        scope='Read-only saved-data calculations; release-only current change holds postsynaptic voltage at matched no-cue value. Positive current is inward. Unreported edges and channel currents are not attributed by this report. Not an integrated balance, biological measurement, causal proof, new acceptance gate or new training.')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Detailed saved-data report:',out/'summary.json')
    print(json.dumps({k:result[k] for k in ('checkpoint_sha256','target','loss_reduction_percent','changed_coordinates','frozen_fields_unchanged','completed_generation_progress')},indent=2))
    print('Near-local-bound groups:',[g for g in bounds if g['near_local_bound']])
    for window in signals:
        if window['category']=='response':print(json.dumps(window,indent=2))
    return result
