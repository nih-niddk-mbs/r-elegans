"""One selected-checkpoint forward-only audit; launched on one GPU."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.selected_atlas_audit_checks import write,compare_responses,parameter_audit,selection_audit,saved_arrays


def run_case(case,out):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.extrasynaptic import PeptideParams,coupling_matrices,load_extrasynaptic_asset
    from r_elegans.brain.selected_atlas_replay import atlas_forward,rhythm_forward
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    from r_elegans.atlas_evaluation import pair_metrics,model_pairs
    if not jax.config.x64_enabled:raise ValueError('Audit requires float64')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],case)
    started=time.monotonic()
    if file_sha256(case['manifest'])!=case['manifest_sha256']:raise ValueError('Audit input manifest changed')
    manifest=json.loads(Path(case['manifest']).read_text());source=Path(case['source']);inputs=Path(case['inputs'])
    problems=[];checks={};evidence=[]
    def verify(path,expected):
        got=file_sha256(path);evidence.append(dict(path=str(path),sha256=got,expected=expected))
        if got!=expected:raise ValueError('Input changed: '+str(path))
    for item in manifest['evidence']:verify(Path(item['path']),item['sha256'])
    s=json.loads((inputs/'summary.json').read_text());args=manifest['arguments'];experiment=s['experiment']
    path=inputs/'network_best_val.npz';params,fixed,config=load_network(path)
    cast=lambda a:a.astype(jnp.float64) if jnp.issubdtype(a.dtype,jnp.floating) else a
    params,fixed=jax.tree.map(cast,(params,fixed));network=saved_arrays(path);ids=network['neuron_ids'].tolist()
    integrity=parameter_audit(path,inputs/'parent_network.npz',inputs/'anchor_network.npz',experiment,args)
    masks=integrity.pop('masks',{})
    if (inputs/'trainable_masks.npz').exists():
        recorded=saved_arrays(inputs/'trainable_masks.npz')
        mask_names={'channels':'params_log_channel_scale','chemical':'params_raw_chemical','gap':'params_raw_gap'}
        integrity['recorded_masks_match']=all(np.array_equal(recorded[k],masks[v]) for k,v in mask_names.items())
        if not integrity['recorded_masks_match']:integrity['passed']=False;integrity['errors'].append('Recorded masks differ from declared policy')
    checks['parameters']=integrity;checks['selection']=selection_audit(s)
    checks['source_archive']=dict(passed=bool(manifest['provenance']['original_source_archive_verified'] and all(x['matched'] for x in manifest['provenance']['replay_source_checks'])))
    actual=manifest['statistics']['metadata'];checks['data_split_counts']=dict(passed=all(s['data'].get(k)==v for k,v in actual.items()),rebuilt=actual)
    h=manifest['provenance']['history'];checks['execution_history']=dict(passed=bool(h.get('available') and h.get('steps_match') and h.get('finite')),**h)
    checks['selection']['baseline_resettled_at_selected_step']=bool(s['best_val']['step']%args['resettle_every']==0)
    pep=None;peptide_details=None
    if experiment=='E105':
        asset=load_extrasynaptic_asset();p=saved_arrays(inputs/'network_best_val_peptides.npz')
        if p['raw_exc'].shape!=(len(asset['sender']),) or p['raw_inh'].shape!=p['raw_exc'].shape or p['log_tau_s'].shape!=():raise ValueError('Invalid selected peptide shape')
        if not all(np.isfinite(p[k]).all() for k in ('raw_exc','raw_inh','log_tau_s')):raise ValueError('Nonfinite peptide values')
        if str(p['asset_sha256'])!=manifest['peptide_asset_sha256'] or s['peptide_asset_sha256']!=manifest['peptide_asset_sha256'] or list(asset['neuron_ids'])!=ids:raise ValueError('Selected peptide asset mismatch')
        clipped=bool(np.all((p['raw_exc']>=-12-1e-9)&(p['raw_exc']<=2.95+1e-9)) and np.all((p['raw_inh']>=-12-1e-9)&(p['raw_inh']<=2.95+1e-9)) and 0-1e-9<=float(p['log_tau_s'])<=np.log(30.)+1e-9)
        if not clipped:raise ValueError('Selected peptide parameters exceed declared bounds')
        exc,inh=coupling_matrices(PeptideParams(*(jnp.asarray(p[k]) for k in ('raw_exc','raw_inh','log_tau_s'))),asset['sender'],asset['receiver'])
        pep=(exc,inh,jnp.asarray(p['log_tau_s']))
        ge,gi=np.logaddexp(0,p['raw_exc']),np.logaddexp(0,p['raw_inh']);order=np.argsort(-(ge+gi))[:15]
        peptide_details=dict(bounds_passed=clipped,tau_s=float(np.exp(p['log_tau_s'])),pairs_exc_above_point1=int((ge>.1).sum()),pairs_inh_above_point1=int((gi>.1).sum()),
            top_pairs=[dict(**asset['metadata']['pair_labels'][i],excitatory_gain=float(ge[i]),inhibitory_gain=float(gi[i])) for i in order])
        checks['peptides']=peptide_details
    with np.load(inputs/'responses.npz',allow_pickle=False) as z:
        targets=z['targets'].tolist();saved=z['best_val'];saved_off=z['best_val_peptides_off'] if pep is not None else None
        if not np.isclose(float(z['stimulus_nS']),args['stimulus_nS'],rtol=0,atol=0):raise ValueError('Stimulus metadata mismatch')
    if saved.shape!=(len(targets),len(ids)) or not targets or len(set(targets))!=len(targets):raise ValueError('Invalid stored target responses')
    original_targets=manifest['statistics']['targets']
    if targets!=original_targets[:len(targets)]:raise ValueError('Source targets differ from independently rebuilt atlas targets')
    g=np.zeros((len(targets),len(ids)));g[np.arange(len(targets)),[ids.index(t) for t in targets]]=args['stimulus_nS'];g=jnp.asarray(g)
    steps=args['substeps'];k=min(13,len(targets))
    print('Replaying selected model:',case['label'],'targets=',len(targets),'substeps=',steps,flush=True)
    coarse=atlas_forward(params,fixed,config,g,steps,peptide=pep)
    checks['saved_response_reproduction']=compare_responses(saved,coarse,rtol=1e-6,atol=1e-12,minimum_correlation=.999999)
    fine=atlas_forward(params,fixed,config,g[:k],2*steps,peptide=pep)
    checks['selected_substep_convergence']=compare_responses(coarse[:k],fine)
    checks['selected_substep_convergence']['targets']=targets[:k]
    cache=json.loads(Path(case['statistics']).read_text());stats={}
    for split,x in cache['stats'].items():stats[split]=dict(pairs=[tuple(p) for p in x['pairs']],**{key:np.asarray(x[key]) for key in ('mean_z','n','stouffer')})
    def metrics(response):
        result={}
        for split,st in stats.items():
            m=pair_metrics(model_pairs(response,targets,ids,st),st,seed=args['seed'],permutations=0)
            result[split]={k:v for k,v in m.items() if not k.startswith('permutation')}
        return result
    def agreement(actual,recorded):
        rows=[]
        for split,values in actual.items():
            for key,val in values.items():
                previous=recorded[split].get(key)
                okay=previous is not None and np.isclose(val,previous,rtol=1e-7,atol=1e-9,equal_nan=True)
                rows.append(dict(split=split,metric=key,passed=bool(okay),replayed=val,recorded=previous))
        return dict(passed=all(r['passed'] for r in rows),metrics=rows)
    selected_metrics=metrics(coarse)
    checks['recorded_metric_reproduction']=agreement(selected_metrics,s['best_val_checkpoint'])
    selected_off=None;off=None;fine_off=None
    if pep is not None:
        tiny=PeptideParams(jnp.full(len(p['raw_exc']),-50.,jnp.float64),jnp.full(len(p['raw_inh']),-50.,jnp.float64),pep[2])
        off_c=coupling_matrices(tiny,asset['sender'],asset['receiver']);off=(off_c[0],off_c[1],pep[2])
        off_response=atlas_forward(params,fixed,config,g,steps,peptide=off);fine_off=atlas_forward(params,fixed,config,g[:k],2*steps,peptide=off)
        checks['peptide_off_saved_reproduction']=compare_responses(saved_off,off_response,rtol=1e-6,atol=1e-12,minimum_correlation=.999999)
        checks['peptide_off_substep_convergence']=compare_responses(off_response[:k],fine_off)
        selected_off=metrics(off_response)
        checks['peptide_off_recorded_metric_reproduction']=agreement(selected_off,s['best_val_peptides_off'])
        np.savez_compressed(out/'selected_responses.npz',targets=targets,coarse=coarse,fine=fine,off_coarse=off_response,off_fine=fine_off)
    else:np.savez_compressed(out/'selected_responses.npz',targets=targets,coarse=coarse,fine=fine)
    print('Checking selected rhythm:',case['label'],'peptide_on=',pep is not None,flush=True)
    warmup=max(60.,float(np.ceil(50*np.exp(float(pep[2]))))/10) if pep is not None else 60.
    v=rhythm_forward(params,fixed,config,peptide=pep,warmup_seconds=warmup);vf=rhythm_forward(params,fixed,config,peptide=pep,max_step_ms=.5,warmup_seconds=warmup)
    rmd=[i for i,n in enumerate(ids) if n.startswith('RMD')];cells={}
    reference_path=Path(case['reference']);reference_document=json.loads(reference_path.read_text())
    evidence.append(dict(path=str(reference_path),sha256=file_sha256(reference_path)))
    if reference_document['checkpoint_sha256']!=s['parent_sha256']:raise ValueError('Rhythm reference is not this model parent')
    reference=reference_document['cells']
    for i in rmd:
        m=rhythm_metrics(v[:,i],.002);mf=rhythm_metrics(vf[:,i],.002);ref=reference[ids[i]]
        period=m.get('median_period_s');p0=ref.get('median_period_s')
        amplitude=float(np.mean(m.get('window_amplitudes_mV',[0.])));a0=float(np.mean(ref.get('window_amplitudes_mV',[0.])))
        relative=bool(reference_document.get('finite',True) and ref['passed'] and m['passed'] and mf['passed'] and p0 and period and abs(period/p0-1)<=.1 and a0>0 and abs(amplitude/a0-1)<=.25)
        cells[ids[i]]=dict(coarse=m,fine=mf,relative_reference_criteria_passed=relative,
            coarse_vs_fine=compare_cycles(v[:,i],vf[:,i],.002) if m['passed'] and mf['passed'] else dict(passed=False,reason='Persistence failed'))
    rhythm=dict(finite=bool(np.isfinite(v).all() and np.isfinite(vf).all()),cells=cells,
        component_criteria_passed=bool(all(cells[n]['relative_reference_criteria_passed'] for n in ('RMDL','RMDR'))),
        strict_convergence_passed=bool(all(cells[n]['coarse_vs_fine']['passed'] for n in ('RMDL','RMDR'))),
        voltage_range_mV=[float(v.min()),float(v.max())],all_cell_late_std_mV=np.std(v[len(v)//2:],axis=0),
        peptide_update_period_s=.1 if pep is not None else None,warmup_seconds=warmup,record_seconds=40.,
        reference_checkpoint_sha256=s['parent_sha256'],reference_peptide_term='off (stored parent network)',scope='40s free neural run; no body, sensory start/stop or swimming test. Parent-relative RMD criteria and strict convergence reported separately.')
    np.savez_compressed(out/'selected_rhythm.npz',neuron_ids=[ids[i] for i in rmd],coarse=v[:,rmd],fine=vf[:,rmd],sample_dt_s=.002)
    checks['rhythm']=rhythm
    for item in manifest['evidence']:verify(Path(item['path']),item['sha256'])
    for item in evidence:
        if file_sha256(item['path'])!=item['sha256']:raise ValueError('Input changed during audit: '+item['path'])
    for key,value in checks.items():
        if key=='rhythm':continue
        if not value.get('passed',value.get('bounds_passed',True)):problems.append(key)
    if not rhythm['finite']:problems.append('nonfinite_rhythm')
    verified=not problems
    result=dict(label=case['label'],experiment='E106',source_experiment=experiment,audit_complete=True,integrity_and_atlas_numerics_passed=verified,
        problems=problems,checkpoint_sha256=s['best_val_sha256'],peptide_sha256=s.get('best_val_peptides_sha256'),
        selected_step=s['best_val']['step'],checks=checks,selected_metrics=selected_metrics,selected_peptide_off_metrics=selected_off,
        validation_leakage=manifest.get('validation_leakage'),source_provenance=manifest['provenance'],
        historical_limits=manifest['historical_limits'],elapsed_seconds=time.monotonic()-started,
        model_promoted=False,full_start_stop_qualified=False,swimming_qualified=False)
    write(out/'evidence.json',evidence);write(out/'summary.json',result)
    print(json.dumps(dict(label=case['label'],audit_complete=True,integrity_and_atlas_numerics_passed=verified,problems=problems,rhythm_component_passed=rhythm['component_criteria_passed'])),flush=True)


def run_reference(network,out):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.selected_atlas_replay import rhythm_forward
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    if not jax.config.x64_enabled:raise ValueError('Reference requires float64')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],dict(reference_network=str(network)))
    sha=file_sha256(network);p,f,c=load_network(network)
    p,f=jax.tree.map(lambda x:x.astype(jnp.float64) if jnp.issubdtype(x.dtype,jnp.floating) else x,(p,f))
    v=rhythm_forward(p,f,c,warmup_seconds=60.);vf=rhythm_forward(p,f,c,max_step_ms=.5,warmup_seconds=60.)
    with np.load(network,allow_pickle=False) as z:ids=z['neuron_ids'].tolist()
    ix=[i for i,n in enumerate(ids) if n.startswith('RMD')]
    cells={ids[i]:rhythm_metrics(v[:,i],.002) for i in ix}
    np.savez_compressed(out/'reference_rhythm.npz',coarse=v[:,ix],fine=vf[:,ix],neuron_ids=[ids[i] for i in ix],sample_dt_s=.002)
    if file_sha256(network)!=sha:raise ValueError('Reference changed during audit')
    write(out/'summary.json',dict(checkpoint_sha256=sha,cells=cells,finite=bool(np.isfinite(v).all() and np.isfinite(vf).all()),warmup_seconds=60.,scope='Stored parent network with no peptide term, RMD component only; no body qualification',
        strict_convergence={ids[i]:compare_cycles(v[:,i],vf[:,i],.002) for i in ix if cells[ids[i]]['passed']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path);p.add_argument('--reference-network',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if bool(a.case)==bool(a.reference_network):p.error('Exactly one case or reference network required')
    out=a.output_dir.resolve()
    if a.reference_network:
        run_reference(a.reference_network.resolve(),out);return 0
    case=json.loads(a.case.read_text())
    try:run_case(case,out)
    except Exception as e:
        out.mkdir(parents=True,exist_ok=True);write(out/'summary.json',dict(label=case['label'],audit_complete=False,integrity_and_atlas_numerics_passed=False,problems=[repr(e)],model_promoted=False));print(repr(e),flush=True);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
