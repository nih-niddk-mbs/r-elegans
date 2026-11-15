"""Offline reconstruction of synaptic currents at saved voltage samples.

Positive current is inward. These are sampled model calculations, not measured
currents, a substep-integrated balance, or intrinsic-current reconstruction.
"""
import json
import numpy as np
from scipy.special import expit
from .motor_identity import is_motor


def load_saved_network(path):
    with np.load(path,allow_pickle=False) as z:
        if int(z['schema_version'])!=1:raise ValueError('Unsupported network schema')
        d={k:np.array(z[k],dtype=np.float64) if z[k].dtype.kind=='f' else np.array(z[k]) for k in z.files}
    ids=d['neuron_ids'].tolist();n=len(ids)
    if len(set(ids))!=n:raise ValueError('Duplicate neuron identities')
    unit=json.loads(str(d['config_json']))['synaptic_unit_nS']
    chemical=unit*np.logaddexp(0.,d['params_raw_chemical'])*d['fixed_chemical_mask']
    raw=d['params_raw_gap']
    gap=unit*np.logaddexp(0.,.5*(raw+raw.T))*d['fixed_gap_mask']
    if chemical.shape!=(n,n) or gap.shape!=(n,n):raise ValueError('Network size mismatch')
    if not np.isfinite(chemical).all() or not np.isfinite(gap).all() or np.any(chemical<0) or np.any(gap<0):raise ValueError('Invalid synapses')
    if not np.allclose(gap,gap.T,rtol=1e-12,atol=1e-14):raise ValueError('Asymmetric gap conductances')
    return dict(ids=ids,chemical=chemical,gap=gap,
        reversal=d['fixed_synapse_reversal_mV'],threshold=d['fixed_release_threshold_mV'],slope=d['fixed_release_slope_mV'],
        channel_g=d['cells_conductances_nS']*np.exp(d['params_log_channel_scale']),
        bias=d['fixed_bias_pA'],leak_reversal=d['cells_leak_reversal_mV'])


def incoming_currents(v,release,post,network):
    """[time,pre] currents; runtime topology convention is [post,pre]."""
    chemical=network['chemical'][post]*release*(network['reversal']-v[:,post,None])
    gap=network['gap'][post]*(v-v[:,post,None])
    return chemical,gap


def cell_rows(sham,inhibited,network):
    ids=network['ids'];avb=[ids.index(n) for n in ('AVBL','AVBR')]
    if sham.shape!=inhibited.shape or sham.ndim!=2 or sham.shape[1]!=len(ids):raise ValueError('Matched voltage arrays required')
    if not np.isfinite(sham).all() or not np.isfinite(inhibited).all():raise ValueError('Nonfinite voltages')
    if np.any(network['slope']<=0):raise ValueError('Positive release slopes required')
    ra=expit((sham-network['threshold'])/network['slope']);rb=expit((inhibited-network['threshold'])/network['slope'])
    targets=[i for i,n in enumerate(ids) if n in ('AVBL','AVBR') or is_motor(n)]
    rows=[]
    for post in targets:
        ca,ga=incoming_currents(sham,ra,post,network)
        cb,gb=incoming_currents(inhibited,rb,post,network)
        l1=float(np.mean(np.sum(abs(ca)+abs(ga),axis=1)))
        avb_l1=float(np.mean(np.sum(abs(ca[:,avb])+abs(ga[:,avb]),axis=1)))
        # Change presynaptic AVB alone while holding postsynaptic voltage at sham:
        # distinguishes altered descending signal from changed driving force.
        direct_c=network['chemical'][post,avb]*(rb[:,avb]-ra[:,avb])*(network['reversal'][avb]-sham[:,post,None])
        direct_g=network['gap'][post,avb]*(inhibited[:,avb]-sham[:,avb])
        sensitivity=(network['chemical'][post,avb]*ra[:,avb]*(1-ra[:,avb])/network['slope'][avb]
                     *(network['reversal'][avb]-sham[:,post,None])+network['gap'][post,avb])
        ranks=np.argsort(np.mean(abs(ca)+abs(ga),axis=0))[::-1]
        top=[]
        for pre in ranks:
            magnitude=float(np.mean(abs(ca[:,pre])+abs(ga[:,pre])))
            if magnitude<=0:continue
            top.append(dict(neuron=ids[pre],mean_abs_edge_current_pA=magnitude,
                            chemical_mean_inward_pA=float(ca[:,pre].mean()),gap_mean_inward_pA=float(ga[:,pre].mean())))
            if len(top)==5:break
        std0=float(np.std(sham[:,post]));std1=float(np.std(inhibited[:,post]))
        rows.append(dict(neuron=ids[post],
            sham_mean_mV=float(sham[:,post].mean()),inhibited_mean_mV=float(inhibited[:,post].mean()),
            sham_p2p_mV=float(np.ptp(sham[:,post])),inhibited_p2p_mV=float(np.ptp(inhibited[:,post])),
            voltage_std_ratio=std1/std0 if std0>1e-8 else None,
            maximum_voltage_difference_mV=float(np.max(abs(inhibited[:,post]-sham[:,post]))),
            sham_release_mean=float(ra[:,post].mean()),inhibited_release_mean=float(rb[:,post].mean()),
            sham_release_extreme_fraction=float(np.mean((ra[:,post]<.01)|(ra[:,post]>.99))),
            incoming_edge_L1_mean_pA=l1,avb_edge_L1_mean_pA=avb_l1,avb_L1_fraction=avb_l1/l1 if l1>1e-12 else None,
            avb_chemical_nS=network['chemical'][post,avb].tolist(),avb_gap_nS=network['gap'][post,avb].tolist(),
            avb_direct_chemical_delta_mean_pA=float(direct_c.sum(axis=1).mean()),
            avb_direct_gap_delta_mean_pA=float(direct_g.sum(axis=1).mean()),
            avb_direct_total_delta_rms_pA=float(np.sqrt(np.mean((direct_c+direct_g).sum(axis=1)**2))),
            avb_direct_sensitivity_mean_nS=np.mean(sensitivity,axis=0).tolist(),
            actual_all_synapse_delta_rms_pA=float(np.sqrt(np.mean(((cb+gb)-(ca+ga)).sum(axis=1)**2))),
            sham_leak_mean_inward_pA=float(np.mean(network['channel_g'][post,16]*(network['leak_reversal'][post]-sham[:,post]))),
            fixed_bias_pA=float(network['bias'][post]),top_incoming_sham=top))
    return rows


def analyze_pair(sham,inhibited,network):
    t=np.asarray(sham['dense_time_s']);u=np.asarray(inhibited['dense_time_s'])
    if not np.array_equal(t,u) or t.ndim!=1 or np.any(np.diff(t)<=0):raise ValueError('Identical increasing saved clocks required')
    if t[0]>20 or t[-1]<59.9:raise ValueError('Full60s assay required')
    before=t<30
    prefix=float(np.max(abs(sham['dense_voltage_mV'][before]-inhibited['dense_voltage_mV'][before])))
    if prefix>1e-8:raise ValueError('Pre-intervention voltage mismatch')
    result={}
    for name,(a,b) in dict(pre=(20,30),off=(35,40),recovery=(50,60)).items():
        idx=np.flatnonzero((t>=a)&(t<b))
        if len(idx)<4:raise ValueError('Missing analysis window')
        # At most~100Hz sampling. No new trace files or extrapolated gate states.
        step=max(1,int(round(.01/float(np.median(np.diff(t))))));idx=idx[::step]
        rows=cell_rows(sham['dense_voltage_mV'][idx],inhibited['dense_voltage_mV'][idx],network)
        result[name]=dict(window=[a,b],sample_count=len(idx),sample_stride=step,cells=rows)
    return dict(pre_intervention_max_voltage_difference_mV=prefix,windows=result)


def compare_models(parent,selected):
    if parent['ids']!=selected['ids']:raise ValueError('Identity ordering mismatch')
    ids=parent['ids'];avb=[ids.index(n) for n in ('AVBL','AVBR')]
    rows=[]
    for post,n in enumerate(ids):
        if not n.startswith(('DB','VB','RMD','SMD','SMB')):continue
        rows.append(dict(neuron=n,
            parent_AVB_gap_nS=parent['gap'][post,avb].tolist(),selected_AVB_gap_nS=selected['gap'][post,avb].tolist(),
            parent_AVB_chemical_nS=parent['chemical'][post,avb].tolist(),selected_AVB_chemical_nS=selected['chemical'][post,avb].tolist(),
            parent_total_nonleak_nS=float(parent['channel_g'][post,:16].sum()),selected_total_nonleak_nS=float(selected['channel_g'][post,:16].sum())))
    return rows


def density_group_members(schema,network):
    """Identify actual cells in saved tied density groups, not label guesses."""
    mask=network['channel_g']>0;mask[:,16]=False
    ci=np.nonzero(mask);rows=[]
    for group in schema:
        if not group['name'].startswith('density:'):continue
        indices=np.asarray(group['indices'],dtype=int)
        if np.any(indices<0) or np.any(indices>=len(ci[0])):raise ValueError('Density coordinate outside channel block')
        cells=sorted({network['ids'][i] for i in ci[0][indices]})
        rows.append(dict(group=group['name'],cells=cells,coordinates=len(indices)))
    return rows
