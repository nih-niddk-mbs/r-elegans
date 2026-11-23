"""Independent identity/signal checks for compact atlas worker artifacts."""
import json
from pathlib import Path
import numpy as np
from scripts.selected_atlas_audit_checks import saved_arrays


def audit_tasks(root, ids, pin):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    results={}
    for path in (root/'tasks').glob('*/task.json'):
        spec=json.loads(pin(path).read_text());pin(spec['model'],spec['model_sha256'])
        r=json.loads(pin(spec['result']).read_text())
        if r['status']!='completed' or r['task']!=spec:raise ValueError('Failed/task identity differs')
        results[spec['name']]=r
        if spec['kind'] in ('autonomy','anchors'):
            a=saved_arrays(pin(r['anchors'],r['anchors_sha256']))
            if str(a['model_sha256'])!=spec['model_sha256']:raise ValueError('Anchor model identity differs')
            for k in ('voltage_mV','gates','calcium_uM','indicator'):
                if not np.isfinite(a['start_'+k]).all():raise ValueError('Nonfinite anchor state')
            if spec['kind']=='autonomy':
                rhythms=[];traces=[]
                for step in (.03125,.015625):
                    a=saved_arrays(pin(Path(spec['output'])/f'rhythm-{step}.npz'));traces.append(a['rmd_mV'])
                    rhythms.append([rhythm_metrics(a['rmd_mV'][:,j],.002) for j in range(2)])
                if rhythms!=r['rhythms']:raise ValueError('Autonomy metrics differ')
                strict=[compare_cycles(traces[0][:,j],traces[1][:,j],.002) for j in range(2)]
                if strict!=r['strict_cycle_checks']:raise ValueError('Cycle diagnostics differ')
            continue
        pin(spec['anchors'],spec['anchors_sha256']);a=saved_arrays(pin(r['artifact'],r['sha256']))
        if str(saved_arrays(spec['anchors'])['model_sha256'])!=spec['model_sha256']:raise ValueError('Replay own-state provenance differs')
        n=len(spec['targets']);count=2*((n+7)//8)+(-n)%8
        labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
        if a['response'].shape!=(4,n,len(ids)) or a['control_means'].shape!=(count,4,len(ids)) or a['reference_sham_rmd_mV'].shape!=(4,12000,2):raise ValueError('Incomplete coverage')
        if a['targets'].tolist()!=spec['targets'] or a['neuron_ids'].tolist()!=ids or a['control_labels'].tolist()!=labels or str(a['anchors_sha256'])!=spec['anchors_sha256'] or float(a['neural_step_ms'])!=spec['step_ms']:raise ValueError('Target/grid/control metadata differs')
        for k in ('response','control_means','reference_sham_rmd_mV'):
            if a[k].dtype!=np.float64 or not np.isfinite(a[k]).all():raise ValueError('Nonfinite/nonfloat64 signal')
        mean=float(abs(a['control_means']-a['control_means'][0]).max());voltage=float(a['maximum_control_voltage_delta_mV'])
        rhythms=[[rhythm_metrics(a['reference_sham_rmd_mV'][p,:,j],.001) for j in range(2)] for p in range(4)]
        if rhythms!=r['rhythms'] or mean!=r['duplicate_mean_error'] or voltage!=r['duplicate_trace_error'] or r['controls_passed']!=(mean<=1e-12 and voltage<=1e-12):raise ValueError('Replay/control metrics differ')
    return results
