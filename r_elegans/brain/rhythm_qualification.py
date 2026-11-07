"""Assays for persistent membrane dynamics; not a biological phenotype fit."""
import numpy as np
import jax.numpy as jnp

from .conductance_network import ConductanceNetworkParams,ConductanceNetworkFixed
from .single_compartment import CHANNEL_INDEX
from r_elegans.data.physiology import load_neuron_parameters


def isolated_rmd_grid(leaks,cca1,*,initial_voltage=-35.):
    """Published RMD prior, changing only leak and CCA1; no applied bias."""
    import jax
    base=load_neuron_parameters('RMD').params
    pairs=np.asarray([(leak,ca) for leak in leaks for ca in cca1])
    n=len(pairs)
    cells=jax.tree.map(lambda x:jnp.broadcast_to(x,(n,)+x.shape),base)
    cells=cells._replace(conductances_nS=cells.conductances_nS
        .at[:,CHANNEL_INDEX['LEAK']].set(pairs[:,0])
        .at[:,CHANNEL_INDEX['CCA1']].set(pairs[:,1]))
    params=ConductanceNetworkParams(jnp.zeros((n,17)),jnp.zeros((n,n)),jnp.zeros((n,n)))
    fixed=ConductanceNetworkFixed(cells,jnp.zeros((n,n)),jnp.zeros((n,n)),
        jnp.zeros(n),jnp.full(n,-45.),jnp.full(n,10.),jnp.zeros(n),jnp.full(n,initial_voltage))
    return params,fixed,pairs


def apply_rmd_rhythm_prior(params,fixed,neuron_ids,*,leak_nS=.045,cca1_nS=30.,initial_voltage=-35.):
    """Exploratory RMD profile; NOT an inferred motor-cell channel composition.

    Restores RMD library constants (including capacitance/reversals), changes
    leak/CCA1, zeroes the old compensation bias for those cells only. Other
    neurons and all anatomical weights/masks remain exactly as loaded.
    """
    import jax
    indices=jnp.asarray([i for i,n in enumerate(neuron_ids) if n.startswith('RMD')])
    if not len(indices):raise ValueError('No RMD cells in requested ordering')
    rmd=load_neuron_parameters('RMD').params
    cells=jax.tree.map(lambda values,base:values.at[indices].set(base),fixed.cells,rmd)
    cells=cells._replace(conductances_nS=cells.conductances_nS
        .at[indices,CHANNEL_INDEX['LEAK']].set(leak_nS)
        .at[indices,CHANNEL_INDEX['CCA1']].set(cca1_nS))
    return params._replace(log_channel_scale=params.log_channel_scale.at[indices].set(0.)),fixed._replace(
        cells=cells,bias_pA=fixed.bias_pA.at[indices].set(0.),
        initial_voltage_mV=fixed.initial_voltage_mV.at[indices].set(initial_voltage)),indices


def rhythm_metrics(voltage,dt,*,washout_fraction=.5,minimum_amplitude_mV=2.):
    """Predeclared late-window persistence gates, with cycle and drift measures.

    A plateau, decaying transient or numerical blow-up is not a sustained rhythm.
    Does not establish an experimental action potential classification.
    """
    from scipy.signal import find_peaks
    v=np.asarray(voltage,dtype=float)
    if v.ndim != 1 or len(v)<20 or not np.isfinite(v).all():
        return dict(passed=False,reason='nonfinite_or_short_trace')
    tail=v[int(len(v)*washout_fraction):]
    windows=np.array_split(tail,5)
    amplitude=np.asarray([np.ptp(w) for w in windows])
    peaks,_=find_peaks(tail,prominence=minimum_amplitude_mV/2.,distance=max(1,round(.02/dt)))
    periods=np.diff(peaks)*dt
    retention=float(amplitude[-1]/max(amplitude[0],1e-12))
    gates=dict(bounded=bool(v.min()>-120 and v.max()<80),
        sustained_amplitude=bool(amplitude.min()>=minimum_amplitude_mV),
        amplitude_retained=bool(.8<=retention<=1.2),multiple_cycles=bool(len(peaks)>=5))
    return dict(passed=all(gates.values()),gates=gates,late_peaks=int(len(peaks)),
        window_amplitudes_mV=amplitude.tolist(),amplitude_retention=retention,
        median_period_s=float(np.median(periods)) if len(periods) else None,
        period_cv=float(np.std(periods)/np.mean(periods)) if len(periods) else None,
        voltage_min_mV=float(v.min()),voltage_max_mV=float(v.max()))


def compare_cycles(coarse,fine,dt):
    """Compare late cycles after phase alignment; also retain pointwise error.

    Tiny frequency errors accumulate phase over long records. Compare period,
    amplitude and averaged cycle shape rather than hiding that pointwise drift.
    """
    from scipy.signal import find_peaks
    def cycle(v):
        v=np.asarray(v,dtype=float)[len(v)//2:]
        peaks,_=find_peaks(v,prominence=1.,distance=max(1,round(.02/dt)))
        if len(peaks)<6:return None
        peaks=peaks[-6:]
        waves=[];amplitudes=[]
        for first,last in zip(peaks[:-1],peaks[1:]):
            segment=v[first:last+1]
            waves.append(np.interp(np.linspace(0,1,101),np.linspace(0,1,len(segment)),segment))
            amplitudes.append(float(np.ptp(segment)))
        return np.mean(np.diff(peaks))*dt,float(np.mean(amplitudes)),np.mean(waves,axis=0)
    a,b=cycle(coarse),cycle(fine)
    if a is None or b is None:return dict(passed=False,reason='not_enough_cycles')
    period_error=abs(a[0]-b[0])/b[0]
    amplitude_error=abs(a[1]-b[1])/max(b[1],1e-12)
    waveform_rmse=float(np.sqrt(np.mean((a[2]-b[2])**2)))
    return dict(passed=bool(period_error<.01 and amplitude_error<.02 and waveform_rmse<.5),
        period_relative_error=float(period_error),amplitude_relative_error=float(amplitude_error),
        phase_aligned_waveform_rmse_mV=waveform_rmse,
        coarse_period_s=float(a[0]),fine_period_s=float(b[0]),
        pointwise_rmse_mV=float(np.sqrt(np.mean((np.asarray(coarse)-np.asarray(fine))**2))))
