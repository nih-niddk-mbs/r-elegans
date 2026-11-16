"""E083 offline E082 rhythmic-drive/release audit; no JAX or forward simulation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.avb_audit import load_saved_network
from r_elegans.smd_drive_audit import load_nmj,muscle_activation,projection,rhythmic_inputs,residual_output,command_summary

PIN='dd42e3f62bc00befc6052fae60c102bc89aaf9266ae88da1f78f3be30c2766a8'


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (repo,run):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Separate external output required')
    s=json.loads((run/'summary.json').read_text())
    if s.get('experiment')!='E082' or not s.get('validation_complete') or s.get('checkpoint_sha256')!=PIN:raise ValueError('Completed pinned E082 required')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    evidence={str(run/'summary.json'):file_sha256(run/'summary.json')}
    from scripts.compare_rhythm_worker import verify_inputs
    manifest=verify_inputs(run/'inputs')
    evidence[str(run/'inputs/manifest.json')]=file_sha256(run/'inputs/manifest.json')
    modelpath=run/'inputs/diagnostic_model.npz'
    if file_sha256(modelpath)!=PIN:raise ValueError('Model hash mismatch')
    network=load_saved_network(modelpath);evidence[str(modelpath)]=PIN
    actorpath=run/'inputs/actor.npz';evidence[str(actorpath)]=manifest['sha256']['actor.npz']
    # Use the runtime topology actually archived during E082, not today's asset.
    roots=[run]+[Path(r['path']) for r in s['screen']+s['confirmation']+s['refinement']]
    rel='r_elegans/assets/runtime_model_v1.json';asset=None;asset_hashes=set()
    for root in roots:
        if root!=run and run not in root.resolve().parents:raise ValueError('Source record outside run')
        for record in root.glob('run_records/*/source_record.json'):
            metadata=json.loads(record.read_text());sha=metadata['source_sha256'][rel]
            path=record.parent/'source'/rel
            if file_sha256(path)!=sha:raise ValueError('Archived topology changed')
            asset_hashes.add(sha);asset=path
            evidence[str(record)]=file_sha256(record);evidence[str(path)]=sha
    if len(asset_hashes)!=1 or asset is None:raise ValueError('Missing or inconsistent E082 archived runtime topology')
    document=json.loads(asset.read_text())
    with np.load(actorpath,allow_pickle=False) as z:
        actor={k:z[k] for k in ('raw_nmj','unknown_polarity','muscle_threshold','raw_muscle_slope')}
    nmj=load_nmj(document,actor,network['ids']);proj=projection()
    rows=s['screen']+s['confirmation']+s['refinement'];reports={};verified={};plot_data=None
    def find(case,variant,grid):
        return next(r for r in rows if r['exit_code']==0 and r['result'] and r['result']['case']['name']==case
                    and r['result']['variant']==variant and r['result']['grid']==grid)
    def load(row,windows):
        folder=Path(row['path']).resolve()
        if run not in folder.parents or row['result']['checkpoint_sha256']!=PIN:raise ValueError('Trial outside pinned run')
        path=folder/'trajectory.npz'
        if path not in verified:
            sha=file_sha256(path)
            if sha!=row['result']['trajectory_sha256']:raise ValueError('Trajectory hash mismatch')
            verified[path]=sha;evidence[str(path)]=sha
        with np.load(path,allow_pickle=False) as z:
            t=z['dense_time_s']
            if t.ndim!=1 or len(t)<16 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid saved clock')
            step=max(1,round(.01/float(np.median(np.diff(t)))))
            indices={name:np.flatnonzero((t>=bounds[0])&(t<bounds[1]))[::step] for name,bounds in windows.items()}
            if any(len(i)<16 for i in indices.values()):raise ValueError('Missing audit window')
            result={name:dict(time=t[i]) for name,i in indices.items()}
            for field in ('dense_voltage_mV','dense_muscles','dense_command','dense_filtered_drive','dense_joints'):
                values=z[field]
                if values.shape[0]!=len(t) or not np.isfinite(values).all():raise ValueError('Invalid saved signal: '+field)
                for name,i in indices.items():result[name][field]=values[i].astype(float)
            return result
    datasets=[('ventral','nominal','ordinary'),('SMD-0.1','nominal','ordinary'),
              ('ventral','long-hold','ordinary'),('ventral','nominal','neural-half')]
    for case,variant,grid in datasets:
        windows=dict(pre=(20,30),off=(85,90) if variant=='long-hold' else (35,40),recovery=(130,150) if variant=='long-hold' else (50,60))
        sham=load(find('sham',variant,grid),windows);inhibited_row=find(case,variant,grid);inhibited=load(inhibited_row,windows)
        label=f'{case}--{variant}--{grid}';report={};max_muscle_error=0.;max_command_error=0.
        for name in windows:
            if not np.array_equal(sham[name]['time'],inhibited[name]['time']):raise ValueError('Paired sampled clocks differ')
            for condition,data in [('sham',sham[name]),('inhibited',inhibited[name])]:
                v=data['dense_voltage_mV'];t=data['time'];release=expit((v-network['threshold'])/network['slope'])
                reconstructed=muscle_activation(release,nmj)
                muscle_error=float(abs(reconstructed-data['dense_muscles']).max())
                command_error=float(abs(reconstructed@proj.T-data['dense_command']).max())
                max_muscle_error=max(max_muscle_error,muscle_error);max_command_error=max(max_command_error,command_error)
                if muscle_error>1e-8 or command_error>1e-6:raise ValueError(f'NMJ/projection reconstruction mismatch: {muscle_error}/{command_error}')
                cells=[]
                for n in ('SMDDL','SMDDR','SMDVL','SMDVR'):
                    post=network['ids'].index(n);cell=rhythmic_inputs(t,v,network,post)
                    g=inhibited_row['result']['case']['conductance_nS'] if condition=='inhibited' and name=='off' and n in inhibited_row['result']['case']['targets'] else 0.
                    imposed=g*(-80.-v[:,post])
                    cell['imposed_inhibitory_current_mean_pA']=float(imposed.mean())
                    cell['imposed_inhibitory_current_std_pA']=float(imposed.std());cells.append(cell)
                key=name+'--'+condition
                report[key]=dict(cells=cells,output=residual_output(v,network,nmj,proj),
                    saved_command=command_summary(data['dense_command']),saved_filtered_drive=command_summary(data['dense_filtered_drive']),
                    saved_joints_rad=command_summary(data['dense_joints']))
        prefix=float(abs(sham['pre']['dense_voltage_mV']-inhibited['pre']['dense_voltage_mV']).max())
        if prefix>1e-8:raise ValueError('Sampled pre-intervention mismatch')
        report['checks']=dict(sampled_pre_voltage_max_difference_mV=prefix,maximum_muscle_reconstruction_error=max_muscle_error,
                              maximum_command_reconstruction_error=max_command_error,command_tolerance=1e-6)
        write(out/(label+'.json'),report);reports[label]=report
        if case=='ventral' and variant=='nominal' and grid=='ordinary':plot_data=(sham['off'],inhibited['off'],report)
        print('Audited:',label,flush=True)
    compact={}
    for label,r in reports.items():
        compact[label]={condition:dict(cells=[{k:c[k] for k in ('neuron','voltage_mean_mV','voltage_std_mV','release_mean','release_std',
            'dominant_voltage_frequency_hz','source_harmonic_sum_pA','top_rhythmic_inputs')} for c in r['off--'+condition]['cells']],
            command=r['off--'+condition]['output']['command'],release_holds=[{k:x[k] for k in ('group','centered_command_rms_ratio','maximum_command_p2p_ratio')}
                for x in r['off--'+condition]['output']['conditional_release_holds']]) for condition in ('sham','inhibited')}
    plot_error=None
    try:plot(out,plot_data,network)
    except ImportError as e:plot_error=str(e)
    write(out/'evidence_hashes.json',evidence)
    write(out/'summary.json',dict(experiment='E083',status='completed',source_run=str(run),checkpoint_sha256=PIN,
        no_training=True,no_new_simulation=True,checks={label:r['checks'] for label,r in reports.items()},results=compact,
        figure=str(out/'signal_chain.png') if plot_error is None else None,plot_error=plot_error,
        limitations='Sampled synaptic reconstruction, not intrinsic-current balance. Source harmonics hold postsynaptic voltage at its own condition mean; recurrent correlations are not causal oscillator proof. Same-window NMJ mean-release counterfactuals omit neural/body feedback and are not additive causal fractions. NumPy float32-template projection is verified only within1e-6 absolute command error.',
        learned_stopping_qualified=False))
    print('Report:',out/'summary.json',flush=True)


def plot(out,data,network):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a,b,report=data;index=network['ids'].index('SMDVL')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for label,d in [('sham',a),('inhibited',b)]:
        t=d['time'];v=d['dense_voltage_mV'][:,index]
        axes[0,0].plot(t,v,label=label)
        axes[0,1].plot(t,expit((v-network['threshold'][index])/network['slope'][index]),label=label)
    j=int(np.argmax(np.ptp(a['dense_command'],axis=0)))
    for label,d in [('sham',a),('inhibited',b)]:axes[1,0].plot(d['time'],d['dense_command'][:,j],label=label)
    c=next(c for c in report['off--inhibited']['cells'] if c['neuron']=='SMDVL')['top_rhythmic_inputs'][:5]
    axes[1,1].bar([r['neuron'] for r in c],[r['source_harmonic_pA'] for r in c])
    for ax,title,y in [(axes[0,0],'SMDVL voltage','mV'),(axes[0,1],'SMDVL NMJ release','release probability'),
                       (axes[1,0],f'Joint command {j}','command'),(axes[1,1],'Candidate rhythmic inputs to inhibited SMDVL','source-varying current harmonic (pA)')]:
        ax.set_title(title);ax.set_ylabel(y)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Time (s)');ax.legend()
    fig.tight_layout();fig.savefig(out/'signal_chain.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
