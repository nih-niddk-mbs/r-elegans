"""Fixed-body replay of recorded commands; no neural simulation."""
from types import SimpleNamespace

import numpy as np

from .body.wet_surface import solve_surface_velocity, wet_agar_drag
from .hybrid_assessment import cycle_excursions


def command_indices(count, dt, speed, source_dt, source_count):
    """Right-end samples of piecewise-held original step commands."""
    progress = np.arange(1, count+1)*dt*speed/source_dt
    nearest = np.rint(progress)
    progress = np.where(np.abs(progress-nearest)<1e-7, nearest, progress)
    indices = np.ceil(progress).astype(int)
    if indices.min() < 1 or indices.max() >= source_count:
        raise ValueError('Playback extends beyond the saved command interval')
    return indices


def replay(segment, body_document, speed, dt, *, held=False, progress=None):
    body = SimpleNamespace(**body_document)
    body.mechanics = SimpleNamespace(**body.mechanics)
    clock = segment['source_time_s']
    source_dt = float(clock[-1]-clock[0])/(len(clock)-1)
    if speed <= 0 or dt <= 0 or not np.isfinite([speed, dt]).all():
        raise ValueError('Finite positive speed and dt required')
    duration = float(clock[-1]-clock[0])/speed
    count = round(duration/dt)
    if speed <= 0 or dt <= 0 or not np.isfinite([speed, dt]).all() or abs(count*dt-duration) > 1e-8:
        raise ValueError('Finite positive speed/dt and integral playback steps required')
    indices = command_indices(count, dt, speed, source_dt, len(clock))
    commands = segment['command'][indices]
    if held:
        commands = np.broadcast_to(segment['command'][1], commands.shape).copy()
    q = segment['joints'][0].copy(); drive = segment['drive'][0].copy()
    position = segment['position'][0].copy(); heading = float(segment['heading'][0])
    n = len(q); length = body.mechanics.segment_length*(n+1)
    trace = dict(time_s=np.arange(count+1)*dt, command=np.vstack((segment['command'][0], commands)),
                 joints=np.empty((count+1,n)), drive=np.empty((count+1,n)),
                 position=np.empty((count+1,2)), heading=np.empty(count+1),
                 active_moment_N_m=np.empty((count,n)), elastic_moment_N_m=np.empty((count,n)),
                 internal_viscous_moment_N_m=np.empty((count,n)),
                 inferred_environmental_moment_N_m=np.empty((count,n)),
                 force_residual=np.empty(count), below_drag_regularization_fraction=np.empty(count))
    trace['joints'][0], trace['drive'][0], trace['position'][0], trace['heading'][0] = q, drive, position, heading
    velocity = None; law = wet_agar_drag()
    for i, command in enumerate(commands):
        drive += -np.expm1(-dt/body.activation_time_constant)*(command-drive)
        active = body.muscle_moment_scale*drive
        velocity, info = solve_surface_velocity(q, active, body, law, dt, initial_velocity=velocity)
        rate = velocity[3:]/length; next_q = q+dt*rate
        if np.max(np.abs(next_q)) >= body.max_joint_angle:
            raise RuntimeError('Diagnostic reached the body joint limit')
        elastic = body.bending_stiffness*next_q
        viscous = body.bending_damping*rate
        c, s = np.cos(heading), np.sin(heading)
        position += dt*np.array(((c,-s),(s,c))) @ velocity[:2]
        heading += dt*velocity[2]/length
        q = next_q
        for key, value in (('active_moment_N_m',active), ('elastic_moment_N_m',elastic),
                           ('internal_viscous_moment_N_m',viscous),
                           ('inferred_environmental_moment_N_m',active-elastic-viscous)):
            trace[key][i] = value
        trace['joints'][i+1], trace['drive'][i+1] = q, drive
        trace['position'][i+1], trace['heading'][i+1] = position, heading
        trace['force_residual'][i] = info['relative_force_residual']
        trace['below_drag_regularization_fraction'][i] = info['fraction_components_below_regularization']
        if progress and ((i+1)%1000 == 0 or i+1 == count):
            progress(i+1,count)
    if not all(np.isfinite(x).all() for x in trace.values()) or np.max(trace['force_residual']) > 1e-8:
        raise RuntimeError('Nonfinite replay or unconverged force balance')
    return trace


def summarize(trace):
    t = trace['time_s']; duration = float(t[-1]); windows = []
    for start, end in ((duration/2,3*duration/4),(3*duration/4,duration)):
        use = (t >= start-1e-10)&(t <= end+1e-10)
        cycles = {name:cycle_excursions(trace[name][use], late_only=False) for name in ('command','drive','joints')}
        windows.append(dict(start_s=start,end_s=end,
            per_joint_cycles={k:v['joint_cycle_excursion_rad'] for k,v in cycles.items()},
            regional_cycles={k:v['regional_cycle_excursion_rad'] for k,v in cycles.items()}))
    return dict(duration_s=duration,windows=windows,
        minimum_joint_cycles=np.min([w['per_joint_cycles']['joints'] for w in windows],axis=0).tolist(),
        minimum_regional_joint_cycles=np.min([w['regional_cycles']['joints'] for w in windows],axis=0).tolist(),
        maximum_force_residual=float(np.max(trace['force_residual'])),
        mean_below_drag_regularization_fraction=float(np.mean(trace['below_drag_regularization_fraction'])),
        torque_p2p_per_joint={k:np.ptp(trace[k][len(trace[k])//2:],axis=0).tolist() for k in
            ('active_moment_N_m','elastic_moment_N_m','internal_viscous_moment_N_m','inferred_environmental_moment_N_m')})


def refinement_check(coarse, fine):
    """Require both raw late waveform and supported cycles to agree."""
    t=coarse['time_s'];use=t>=t[-1]/2
    interpolated=np.column_stack([np.interp(t,fine['time_s'],column) for column in fine['joints'].T])
    error=np.max(np.abs(coarse['joints'][use]-interpolated[use]),axis=0)
    scale=np.ptp(interpolated[use],axis=0)
    x=np.asarray(summarize(coarse)['minimum_joint_cycles']);y=np.asarray(summarize(fine)['minimum_joint_cycles'])
    waveform=bool(np.all(error<=np.maximum(.05*scale,1e-7)))
    cycles=bool(np.all(abs(x-y)<=np.maximum(.05*abs(y),1e-7)))
    return dict(maximum_waveform_difference_rad=float(np.max(error)),waveform_passed=waveform,
                maximum_cycle_difference_rad=float(np.max(abs(x-y))),cycles_passed=cycles,
                supported_cycle_joints=int(np.sum((x>0)&(y>0))),passed=waveform and cycles)
