"""Independent saved-artifact checks for the selected E104/E105 checkpoints."""
import json,math
from pathlib import Path
import numpy as np


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    return value


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def compare_responses(a,b,*,rtol=.1,atol=0.,minimum_correlation=.9):
    a,b=np.asarray(a),np.asarray(b)
    finite=bool(a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:return dict(passed=False,finite_and_same_shape=False)
    delta=float(np.max(abs(a-b)));scale=float(np.max(abs(a)));relative=delta/max(scale,1e-30)
    correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if min(np.std(a),np.std(b))>0 else None
    # Exact constant equality is reproducible, but not evidence of atlas signal.
    corr_ok=correlation is not None and correlation>=minimum_correlation
    if delta==0: corr_ok=True
    return clean(dict(passed=bool((delta==0 or delta<atol+rtol*scale) and corr_ok),finite_and_same_shape=True,
        correlation=correlation,max_abs_difference=delta,max_abs_reference_response=scale,
        max_abs_difference_over_max_abs_response=relative,relative_tolerance=rtol,absolute_tolerance=atol,
        minimum_correlation=minimum_correlation,constant_response=bool(min(np.std(a),np.std(b))==0)))


def saved_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def parameter_audit(selected_path,parent_path,anchor_path,experiment,args):
    selected,parent,anchor=map(saved_arrays,(selected_path,parent_path,anchor_path))
    errors=[];changes={};counts={};bounds={}
    if set(selected)!=set(parent) or set(selected)!=set(anchor):return dict(passed=False,errors=['Network file fields differ'])
    ids=selected['neuron_ids'].tolist();all_fields=experiment=='E104' and args['parameter_set']=='all'
    frozen_network=experiment=='E105' and args['train_network']=='no'
    reference=parent if frozen_network else anchor
    channels=np.zeros_like(reference['params_log_channel_scale'],bool)
    if all_fields:channels=reference['cells_conductances_nS']>0
    elif not frozen_network:
        channels[:]=True;channels[:,16]=False
        channels[[i for i,n in enumerate(ids) if n.startswith('RMD')]]=False
    masks=dict(params_log_channel_scale=channels,params_raw_chemical=reference['fixed_chemical_mask'].astype(bool),params_raw_gap=reference['fixed_gap_mask'].astype(bool))
    if frozen_network:masks={k:np.zeros_like(v,bool) for k,v in masks.items()}
    allowed_extra={'cells_capacitance_pF','cells_leak_reversal_mV','cells_calcium_removal_ms','cells_free_calcium_fraction','fixed_release_threshold_mV','fixed_release_slope_mV'} if all_fields else set()
    for key,a in selected.items():
        b=reference[key]
        if a.shape!=b.shape:errors.append('Shape changed: '+key);continue
        if a.dtype.kind in 'fc' and not np.isfinite(a).all():errors.append('Nonfinite: '+key)
        if key in masks:
            mask=masks[key];delta=a-b;changes[key]=int(np.count_nonzero(delta));counts[key]=int(mask.sum())
            if not np.array_equal(a[~mask],b[~mask]):errors.append('Frozen parameters changed: '+key)
            limit=math.log(10.) if key=='params_log_channel_scale' else 2.
            bounds[key]=dict(max_abs_change=float(np.max(abs(delta))),at_bound=int(np.sum(mask&(abs(delta)>=limit-1e-6))))
            if np.any(abs(delta[mask])>limit+1e-9):errors.append('Parameter bound exceeded: '+key)
        elif key in allowed_extra:
            # Keep the declared tolerance representable for float32 anchors.
            # Otherwise e.g. float32(20) + 1e-8 rounds back to 20, rejecting
            # harmless float64 log/exp roundoff in saved selected weights.
            x=np.asarray(anchor[key],dtype=np.float64)
            if key=='cells_capacitance_pF':lo,hi=np.minimum(x,1.),np.maximum(x,20.)
            elif key=='cells_leak_reversal_mV':lo,hi=np.minimum(x,-80.),np.maximum(x,-35.)
            elif key=='fixed_release_threshold_mV':lo,hi=np.minimum(x,-65.),np.maximum(x,-20.)
            elif key=='fixed_release_slope_mV':lo,hi=np.minimum(x,2.),np.maximum(x,20.)
            else:lo,hi=x/3.,x*3.
            bounds[key]=dict(changed=int(np.count_nonzero(a-x)),at_lower=int(np.sum(abs(a-lo)<=1e-8)),at_upper=int(np.sum(abs(a-hi)<=1e-8)),minimum=float(a.min()),maximum=float(a.max()),
                maximum_below_lower=float(np.maximum(lo-a,0.).max()),maximum_above_upper=float(np.maximum(a-hi,0.).max()),absolute_tolerance=1e-8,bound_arithmetic_dtype=str(x.dtype))
            if np.any(a<lo-1e-8) or np.any(a>hi+1e-8):errors.append('Per-cell bound exceeded: '+key)
        elif not np.array_equal(a,b):errors.append('Frozen field changed: '+key)
    return dict(passed=not errors,errors=errors,changed_entries=changes,trainable_entries=counts,
        bounds=bounds,reference='parent' if frozen_network else 'anchor',network_frozen=frozen_network,
        per_cell_extras_allowed=sorted(allowed_extra),masks=masks)


def selection_audit(summary):
    evaluations=[e for e in summary.get('evaluations',[]) if isinstance((e.get('val') or {}).get('spearman_mean_z'),(int,float)) and math.isfinite(e['val']['spearman_mean_z'])]
    if not evaluations:return dict(passed=False,reason='No finite recorded validation evaluations')
    chosen=max(evaluations,key=lambda e:e['val']['spearman_mean_z']);recorded=summary['best_val']
    return dict(passed=bool(chosen['step']==recorded['step'] and np.isclose(chosen['val']['spearman_mean_z'],recorded['spearman'],rtol=0,atol=1e-12)),
        expected_step=chosen['step'],recorded_step=recorded['step'],recorded_selection_score=recorded['spearman'],
        note='Matches logged validation maximum, not proof cached settling states equal fresh selected-checkpoint evaluation. No checkpoint reselected using test results.')


def declared_performance(result,control_spearman=None):
    metrics=(result.get('selected_metrics') or {}).get('test',{})
    sp,auc=metrics.get('spearman_mean_z'),metrics.get('auc_excited')
    finite=all(isinstance(x,(int,float)) and math.isfinite(x) for x in (sp,auc))
    exp=result.get('source_experiment');checks={}
    if exp=='E104':
        checks['test_spearman_above_point127']=bool(finite and sp>.127)
        checks['test_auc_at_least_point574']=bool(finite and auc>=.574)
        if result.get('label')!='E104-network_control':
            checks['beats_network_control_by_point02']=bool(finite and control_spearman is not None and sp>control_spearman+.02)
    elif exp=='E105':
        off=((result.get('selected_peptide_off_metrics') or {}).get('test') or {}).get('spearman_mean_z')
        checks['test_spearman_above_point126']=bool(finite and sp>.126)
        checks['test_auc_at_least_point574']=bool(finite and auc>=.574)
        checks['peptide_on_beats_same_checkpoint_off_by_point02']=bool(finite and off is not None and sp-off>=.02)
    return dict(criteria=checks,passed=bool(checks and all(checks.values())),
        numerical_interpretation_allowed=bool(result.get('integrity_and_atlas_numerics_passed')),
        validation_selection_uncontaminated=None if result.get('validation_leakage') is None else not result['validation_leakage'],
        note='Original experiment thresholds; no test-based reselection. Passing arithmetic thresholds without numerics/integrity is not qualification.')
