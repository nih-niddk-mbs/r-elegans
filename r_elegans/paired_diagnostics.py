"""Read-only bound occupancy and sustained dynamic range; no biological pass."""
import numpy as np
from .amplitude_analysis import sustained_excursions


def bound_occupancy(offsets,counts,maximum_factor):
    x=np.asarray(offsets,dtype=float)
    if x.shape!=(sum(counts.values()),) or not np.isfinite(x).all():
        raise ValueError('Finite offset vector matching parameter counts required')
    if not np.isfinite(maximum_factor) or maximum_factor<=1:raise ValueError('Invalid bound')
    limit=np.log(maximum_factor)
    if np.any(np.abs(x)>limit+1e-9):raise ValueError('Offsets exceed anchor bounds')
    result={};start=0
    for name,n in counts.items():
        group=x[start:start+n];start+=n
        result[name]=dict(count=n,nonzero=int(np.count_nonzero(group)),
            near_lower=int(np.sum(group<=-.99*limit)),near_upper=int(np.sum(group>=.99*limit)),
            factor_percentiles=np.percentile(np.exp(group),[0,5,50,95,100]).tolist() if n else [])
    return dict(maximum_factor=maximum_factor,near_bound_definition='Within 1% of log-coordinate bound',categories=result)


def dynamic_range(values,dt,amplitude_floor=0.):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or len(x)<40 or not np.isfinite(x).all() or not np.isfinite(dt) or dt<=0:
        raise ValueError('Finite time-by-channel trace with >=40 samples required')
    late=x[len(x)//2:];t=np.arange(len(late))*dt;t-=t.mean()
    slopes=t@late/max(float(t@t),1e-30)
    detrended=late-late.mean(axis=0)-t[:,None]*slopes
    amplitude,repeated=sustained_excursions(x)
    raw_amplitude=amplitude.copy()
    repeated &= amplitude>=amplitude_floor
    amplitude=np.where(repeated,amplitude,0.)
    return dict(late_mean=late.mean(axis=0).tolist(),late_std=late.std(axis=0).tolist(),
        late_linear_slope_per_second=slopes.tolist(),late_detrended_std=detrended.std(axis=0).tolist(),
        sustained_p2p=amplitude.tolist(),raw_sustained_p2p=raw_amplitude.tolist(),
        reporting_floor=amplitude_floor,repeated_channel_count=int(repeated.sum()),
        repeated_indices=np.flatnonzero(repeated).tolist())


def signal_chain(trace,dt):
    floors=dict(dense_voltage_mV=.1,dense_calcium_uM=.001,dense_indicator=.0001,
                dense_muscles=.0001,dense_command=.0001,dense_filtered_drive=.0001,dense_joints=.0001)
    result={key:dynamic_range(trace[key],dt,floor) for key,floor in floors.items()}
    muscles=np.asarray(trace['dense_muscles']);late=muscles[len(muscles)//2:]
    result['nmj_activation_near_extremes_fraction']=float(np.mean((late<.01)|(late>.99)))
    q=np.asarray(trace['dense_joints']);q=q[len(q)//2:];q=q-q.mean(axis=0)
    singular=np.linalg.svd(q,compute_uv=False);power=singular**2
    result['late_shape_pc1_fraction']=float(power[0]/power.sum()) if power.sum()>1e-20 else 0.
    result['scope']='Descriptive late-half dynamics with explicit numerical reporting floors; not measured biological thresholds or source-amplitude/wave qualification'
    return result
