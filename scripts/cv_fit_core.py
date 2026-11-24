"""E124 grouped physiology fitting, compact forward artifacts and prospective checks."""
import hashlib
import time
from pathlib import Path
import numpy as np
from scripts.run_cv_peptide_pilot import write
from scripts.stable_atlas_policy import Coordinates,loss


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def group_directions(codec):
    groups={}
    for name,columns in [('inward',(6,7,8,14)),('outward',(0,1,2,3,4,5,9,10,11,12,13,15))]:
        x=np.zeros(codec.size);x[:codec.sizes[0]]=np.isin(codec.channel[1],columns);groups[name]=x
    for name,start,end in [('chemical',codec.sizes[0],sum(codec.sizes[:2])),('gap',sum(codec.sizes[:2]),codec.size)]:
        x=np.zeros(codec.size);x[start:end]=1.;groups[name]=x
    return groups


def raw_peptide_gains(balance):
    if not np.isfinite(balance) or abs(balance)>.1+1e-12:raise ValueError('Peptide balance outside bounds')
    base=np.logaddexp(0.,-9.)
    return tuple(float(np.log(np.expm1(base*np.exp(sign*balance)))) for sign in (1.,-1.))


def penalty(x,balance):return float(.001*np.mean((np.asarray(x)/.1)**2)+.001*(balance/.1)**2)


def objective(prediction,targets,ids,fold,x,balance):
    return loss(prediction,targets,ids,fold,'train')+penalty(x,balance)


def choose(current,proposals):
    eligible=[current]+[r for r in proposals if r.get('safe') and r.get('score') is not None and np.isfinite(r['score'])]
    best=min(eligible,key=lambda r:r['score'])
    return best if best['score']<current['score']-1e-10 else current


def response_check(coarse,fine):
    error=float(np.max(abs(coarse-fine)));tol=float(1e-7+.1*np.max(abs(fine)))
    return dict(maximum_difference=error,tolerance=tol,passed=bool(np.isfinite(error) and error<=tol))


def prediction_checks(a,level):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    c=a['controls'];v=a['control_voltage']
    mean_error=float(np.max(abs(c-c[0,:,0][None,:,None,:])))
    voltage_error=float(np.max(abs(v-v[0,:,0][None,:,None,:,:])))
    rhythms=[rhythm_metrics(v[0,i,0,:,cell],.001/2**level) for i in range(4) for cell in range(2)]
    finite=all(np.isfinite(x).all() for x in a.values())
    bounded=bool(a['final_voltage_mV'].min()>=-120 and a['final_voltage_mV'].max()<=80 and a['rmd_min'].min()>=-120 and a['rmd_max'].max()<=80)
    return dict(finite=finite,end_and_rmd_bounded=bounded,duplicate_mean_error=mean_error,
        duplicate_voltage_error_mV=voltage_error,rhythms=rhythms,
        safe=bool(finite and bounded and mean_error<1e-12 and voltage_error<1e-10 and all(r['passed'] for r in rhythms)))


def full_window_checks(low,high,dt=.0005):
    """New60s initialization protocol, independent of original E123 five-cycle gate."""
    from scripts.analyze_convergence_cycles import window_stats,descriptive_comparison
    results=[]
    for start,end in ((30.,60.),(30.,40.),(40.,50.),(50.,60.)):
        for cell,name in enumerate(('RMDL','RMDR')):
            a=window_stats(low[:,cell],dt,start,end);b=window_stats(high[:,cell],dt/2,start,end)
            c=descriptive_comparison(a,b)
            passed=bool(c['available'] and c['whole_train_period_relative_difference']<.01 and
                c['full_window_mean_amplitude_relative_difference']<.02 and c['full_window_cycle_rmse_mV']<.5)
            results.append(dict(neuron=name,window_s=[start,end],comparison=c,passed=passed))
    return results


def admit_probe(now,search_deadline,fold_deadline,pair_seconds,validation_seconds):
    return bool(now+1.3*pair_seconds<search_deadline and now+1.3*pair_seconds+validation_seconds<fold_deadline)


class Evaluator:
    """One persistent GPU; all models and folds share compiled batch16 kernels."""
    def __init__(self,root,arm):
        import jax
        import jax.numpy as jnp
        from r_elegans.brain.conductance_network import load_network
        self.root=Path(root);self.arm=arm;self.enabled=arm=='peptide'
        self.origin=load_arrays(self.root/'parent.npz');self.codec=Coordinates(self.origin)
        self.ids=self.origin['neuron_ids'].tolist();_,fixed,self.config=load_network(self.root/'parent.npz')
        self.fixed=jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.float64),fixed)
        self.asset=load_arrays(self.root/'peptide_asset.npz')
        if self.asset['neuron_ids'].tolist()!=self.ids:raise ValueError('Peptide asset neuron order')
        self.engines={};self.rest_engines={};self.starts={};self.anchor_seconds={};self.batch_seconds={}

    def components(self,x,balance):
        import jax.numpy as jnp
        from r_elegans.brain.conductance_network import ConductanceNetworkParams
        from r_elegans.brain.extrasynaptic import PeptideParams,coupling_matrices
        values=self.codec.apply(x)
        p=ConductanceNetworkParams(**{k:jnp.asarray(values['params_'+k]) for k in ConductanceNetworkParams._fields})
        a,b=raw_peptide_gains(balance);n=len(self.asset['sender'])
        pp=PeptideParams(jnp.full(n,a,dtype=jnp.float64),jnp.full(n,b,dtype=jnp.float64),jnp.asarray(np.log(5.)))
        exc,inh=coupling_matrices(pp,self.asset['sender'],self.asset['receiver'])
        if not self.enabled:exc,inh=exc*0,inh*0
        return values,p,pp,exc,inh

    def anchor(self,x,balance,level):
        import jax.numpy as jnp
        from r_elegans.brain.peptide_state_replay import make_engine
        key=(hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest(),float(balance),level)
        if level not in self.engines:
            self.engines[level]=make_engine(self.config,[self.ids.index('RMDL'),self.ids.index('RMDR')],dt=.001/2**level,step_ms=.03125/2**level)
        if key not in self.starts:
            _,p,pp,exc,inh=self.components(x,balance);before=time.monotonic()
            states,phase=self.engines[level][0](p,self.fixed,exc,inh,pp.log_tau_s)
            self.starts[key]=(states,phase);self.anchor_seconds[level]=time.monotonic()-before
        return self.starts[key]

    def predict(self,x,balance,targets,level,out):
        import jax
        import jax.numpy as jnp
        from scripts.experiment_records import file_sha256
        out=Path(out);out.mkdir(parents=True,exist_ok=False);before=time.monotonic()
        values,p,pp,exc,inh=self.components(x,balance)
        np.savez_compressed(out/'network.npz',**values)
        np.savez_compressed(out/'coordinates.npz',offsets=x,peptide_balance=np.array(balance))
        start,phase=self.anchor(x,balance,level)
        responses=[];controls=[];voltage=[];finals={k:[] for k in start.neural._fields};finals['peptide']=[];mins=[];maxs=[]
        for begin in range(0,len(targets),16):
            names=targets[begin:begin+16];g=np.zeros((18,302));g[np.arange(len(names)),[self.ids.index(n) for n in names]]=.03
            tick=time.monotonic()
            means,final,traces=jax.tree.map(np.asarray,self.engines[level][1](p,self.fixed,exc,inh,pp.log_tau_s,start,jnp.asarray(g)))
            self.batch_seconds[level]=time.monotonic()-tick
            responses.append(means[:,:len(names)]-means[:,-1:]);controls.append(means[:,-2:]);voltage.append(traces[0][:,-2:])
            mins.append(traces[0].min(axis=2));maxs.append(traces[0].max(axis=2))
            for k,v in final.neural._asdict().items():finals[k].append(v)
            finals['peptide'].append(final.peptide)
        data=dict(response=np.concatenate(responses,axis=1),controls=np.stack(controls),control_voltage=np.stack(voltage),
            rmd_min=np.stack(mins),rmd_max=np.stack(maxs),**{'final_'+k:np.stack(v) for k,v in finals.items()},
            **{'start_'+k:np.asarray(v) for k,v in start.neural._asdict().items()},start_peptide=np.asarray(start.peptide))
        np.savez_compressed(out/'prediction.npz',**data)
        row=dict(path=str(out),targets=targets,level=level,phase=phase,elapsed_seconds=time.monotonic()-before,
            checks=prediction_checks(data,level),parameters=self.codec.audit(values,x),balance=float(balance),
            maximum_final_peptide_conductance_nS=float(self.config.synaptic_unit_nS*np.max(np.einsum('ij,...j->...i',np.asarray(exc+inh),data['final_peptide']))),
            sha256={str(out/n):file_sha256(out/n) for n in ('network.npz','coordinates.npz','prediction.npz')})
        write(out/'metrics.json',row)
        print('PREDICTION',self.arm,str(out),len(targets),'targets',round(row['elapsed_seconds'],1),'seconds',flush=True)
        return data,row

    def save_complete_model(self,x,balance,prediction_folder,out):
        from r_elegans.brain.peptide_checkpoint import bundle_model
        from scripts.experiment_records import file_sha256
        prediction_folder=Path(prediction_folder);out=Path(out)
        network=out.with_name(out.stem+'-network.npz');np.savez_compressed(network,**self.codec.apply(x))
        bundle_model(network,prediction_folder/'prediction.npz',out,enabled=self.enabled,asset=self.asset,
            asset_sha256=file_sha256(self.root/'peptide_asset.npz'),network_sha256=file_sha256(network))
        a=load_arrays(out);raw_exc,raw_inh=raw_peptide_gains(balance)
        a['peptide_raw_exc'][:]=raw_exc;a['peptide_raw_inh'][:]=raw_inh
        np.savez_compressed(out,**a)

    def rest(self,x,balance,level,out):
        import jax
        from r_elegans.brain.peptide_checkpoint import make_rest,save_state
        from r_elegans.brain.rhythm_qualification import rhythm_metrics
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        _,p,pp,exc,inh=self.components(x,balance);start,_=self.anchor(x,balance,level)
        initial=jax.tree.map(lambda a:a[0],start);dt=.001/2**level
        if level not in self.rest_engines:
            self.rest_engines[level]=make_rest(self.config,[self.ids.index('RMDL'),self.ids.index('RMDR')],60.,dt=dt,step_ms=.03125/2**level)
        end,trace=jax.tree.map(np.asarray,self.rest_engines[level](p,self.fixed,pp,exc,inh,initial))
        np.savez_compressed(out/'trace.npz',rmd=trace[0],voltage_min=trace[1],voltage_max=trace[2],peptide_mean=trace[3]);save_state(out/'final.npz',end)
        checks=[rhythm_metrics(trace[0][:,i],dt) for i in range(2)]
        row=dict(level=level,dt=dt,seconds=60.,rhythms=checks,
            finite=all(np.isfinite(v).all() for v in jax.tree.leaves((end,trace))),
            all_cell_voltage_bounded=bool(trace[1].min()>=-120 and trace[2].max()<=80))
        write(out/'metrics.json',row);return trace[0],row

    def validation_reserve(self,target_count):
        batches=(target_count+15)//16
        if not all(k in self.anchor_seconds and k in self.batch_seconds for k in (0,1)):raise ValueError('Baseline timings required')
        whole=sum(self.anchor_seconds[k]+batches*self.batch_seconds[k] for k in (0,1))
        quarter=2*(self.anchor_seconds[1]+self.batch_seconds[1])
        long=3*self.anchor_seconds[1]
        return 1.2*(whole+quarter+long)+120.
