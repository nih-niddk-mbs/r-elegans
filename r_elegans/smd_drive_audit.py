"""E083 sampled current and NMJ transmission audit; no forward neural dynamics."""
import numpy as np
from scipy.special import expit
from .avb_audit import incoming_currents
from .motor_identity import numbered_motor


def peak_frequency(t,x):
    x=np.asarray(x,dtype=float);t=np.asarray(t)
    if len(t)<16 or np.std(x)<1e-8:return None
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-6,atol=1e-10):raise ValueError('Uniform audit sampling required')
    design=np.column_stack((np.ones(len(t)),t-t.mean()))
    detrended=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    power=abs(np.fft.rfft(detrended*np.hanning(len(t))))**2
    f=np.fft.rfftfreq(len(t),dt);use=(f>=.1)&(f<=20.)
    return float(f[np.flatnonzero(use)[np.argmax(power[use])]])


def harmonic(t,x,frequency):
    x=np.asarray(x,dtype=float)
    if frequency is None:return np.zeros(x.shape[1:] or (),dtype=complex)
    phase=2*np.pi*frequency*(t-t[0])
    design=np.column_stack((np.ones(len(t)),t-t.mean(),np.cos(phase),np.sin(phase)))
    coef=np.linalg.lstsq(design,x,rcond=None)[0]
    return coef[2]-1j*coef[3]


def rhythmic_inputs(t,v,network,post):
    release=expit((v-network['threshold'])/network['slope'])
    ca,ga=incoming_currents(v,release,post,network)
    # Hold post voltage constant to remove current variation caused by its own V.
    held=float(v[:,post].mean())
    source_c=network['chemical'][post]*release*(network['reversal']-held)
    source_g=network['gap'][post]*(v-held)
    f=peak_frequency(t,v[:,post]);hc=harmonic(t,source_c,f);hg=harmonic(t,source_g,f)
    total=hc+hg;den=float(np.sum(abs(total)))
    rank=np.argsort(abs(total))[::-1];rows=[]
    for i in rank:
        if abs(total[i])<1e-12:continue
        rows.append(dict(neuron=network['ids'][i],source_harmonic_pA=float(abs(total[i])),
            chemical_harmonic_pA=float(abs(hc[i])),gap_harmonic_pA=float(abs(hg[i])),
            source_harmonic_L1_fraction=float(abs(total[i])/den) if den>0 else None,
            actual_net_edge_current_std_pA=float(np.std(ca[:,i]+ga[:,i])),
            source_only_edge_current_std_pA=float(np.std(source_c[:,i]+source_g[:,i]))))
        if len(rows)==8:break
    groups={}
    for prefix in ('RMD','SMD','SMB','AVB','AVA'):
        use=[i for i,n in enumerate(network['ids']) if n.startswith(prefix)]
        groups[prefix]=dict(harmonic_of_group_sum_pA=float(abs(total[use].sum())),
                           sum_individual_harmonics_pA=float(abs(total[use]).sum()))
    return dict(neuron=network['ids'][post],voltage_mean_mV=held,voltage_std_mV=float(np.std(v[:,post])),
        voltage_p2p_mV=float(np.ptp(v[:,post])),dominant_voltage_frequency_hz=f,
        release_mean=float(release[:,post].mean()),release_std=float(release[:,post].std()),
        release_p2p=float(np.ptp(release[:,post])),release_extreme_fraction=float(np.mean((release[:,post]<.01)|(release[:,post]>.99))),
        mean_release_sensitivity_per_mV=float(np.mean(release[:,post]*(1-release[:,post])/network['slope'][post])),
        source_harmonic_sum_pA=float(abs(total.sum())),sum_edge_harmonics_pA=den,top_rhythmic_inputs=rows,groups=groups,
        actual_total_synaptic_current_std_pA=float(np.std((ca+ga).sum(axis=1))))


def load_nmj(document,actor,ids):
    if list(document['neuron_ids'])!=list(ids):raise ValueError('NMJ identity ordering mismatch')
    actor={k:np.asarray(actor[k],dtype=float) for k in ('raw_nmj','unknown_polarity','muscle_threshold','raw_muscle_slope')}
    sparse=document['nmj'];counts=np.zeros((95,len(ids)));signs=np.zeros_like(counts)
    mi=np.asarray(sparse['muscle_index'],dtype=int);ni=np.asarray(sparse['neuron_index'],dtype=int)
    counts[mi,ni]=sparse['contact_count'];signs[mi,ni]=sparse['sign']
    weights=counts*np.logaddexp(0.,actor['raw_nmj'])
    polarity=np.where(signs!=0,signs,np.tanh(actor['unknown_polarity']))*(counts>0)
    return dict(signed=weights*polarity/np.maximum(weights.sum(axis=1,keepdims=True),1.),
        threshold=actor['muscle_threshold'],slope=.02+np.logaddexp(0.,actor['raw_muscle_slope']))


def muscle_activation(release,nmj):
    return np.clip(expit((release@nmj['signed'].T-nmj['threshold'])/nmj['slope']),1e-5,1.-1e-5)


def projection():
    """NumPy reproduction of fixed11-joint head-at-end projection; verify saved commands."""
    # Original template was formed in float32, then promoted to float64.
    positions=np.asarray(list(range(24))*2+list(range(23))+list(range(24)),dtype=np.float32)/np.float32(23.)
    joints=np.float32(1.)-np.linspace(0.,1.,11,dtype=np.float32)
    local=np.maximum(np.float32(1.)-abs(joints[:,None]-positions[None,:])/np.float32(.1),np.float32(0.))
    dorsal=np.arange(95)<48;ventral=~dorsal
    def normalize(mask):
        w=local*mask[None,:];return w/np.maximum(w.sum(axis=1,keepdims=True),np.float32(1e-8))
    return normalize(dorsal).astype(float)-normalize(ventral).astype(float)


def command_summary(command):
    p2p=np.ptp(command,axis=0)
    return dict(centered_rms=float(np.sqrt(np.mean((command-command.mean(axis=0))**2))),
        maximum_p2p=float(np.max(p2p)),joint_p2p=p2p.tolist(),
        regional_median_p2p=[float(np.median(p2p[g])) for g in np.array_split(np.arange(command.shape[1]),3)])


def residual_output(v,network,nmj,signed_projection):
    ids=network['ids'];r=expit((v-network['threshold'])/network['slope'])
    muscles=muscle_activation(r,nmj);command=muscles@signed_projection.T
    base=command_summary(command);rows=[]
    selectors={
        'ventral-SMD':lambda n:n in ('SMDVL','SMDVR'),
        'dorsal-SMD':lambda n:n in ('SMDDL','SMDDR'),
        'SMD':lambda n:n.startswith('SMD'),
        'RMD':lambda n:n.startswith('RMD'),
        'SMB':lambda n:n.startswith('SMB'),
        'B':lambda n:numbered_motor(n) and n.startswith(('DB','VB')),
        'head':lambda n:n.startswith(('RMD','SMD','SMB')),
        'all':lambda n:True}
    for name,select in selectors.items():
        mask=np.asarray([select(n) for n in ids]);held=r.copy();held[:,mask]=r[:,mask].mean(axis=0)
        cf=command_summary(muscle_activation(held,nmj)@signed_projection.T)
        rows.append(dict(group=name,centered_command_rms_ratio=cf['centered_rms']/base['centered_rms'] if base['centered_rms']>1e-12 else None,
                         maximum_command_p2p_ratio=cf['maximum_p2p']/base['maximum_p2p'] if base['maximum_p2p']>1e-12 else None,
                         command=cf))
    return dict(command=base,mean_muscle_activation=float(muscles.mean()),
        maximum_muscle_p2p=float(np.ptp(muscles,axis=0).max()),conditional_release_holds=rows,
        scope='Offline NMJ transfer counterfactual at fixed saved voltages, holding release to same-window means; no body/neural feedback replay or causal oscillator attribution')
