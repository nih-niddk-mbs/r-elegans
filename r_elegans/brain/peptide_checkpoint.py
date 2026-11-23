"""Explicit complete neural/peptide model bundles and continuation state.

Schema2 intentionally cannot be read as a network-only schema1 checkpoint.
Peptide state is separate from optimizer state; this is not an optimizer resume.
"""
import json
from pathlib import Path
import tempfile
import numpy as np


def bundle_model(network, state, destination, *, enabled, asset, asset_sha256, network_sha256):
    with np.load(network,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    if int(base['schema_version'])!=1:raise ValueError('Schema1 network input required')
    with np.load(state,allow_pickle=False) as z:
        starts={k:z[k] for k in ('start_voltage_mV','start_gates','start_calcium_uM','start_indicator','start_peptide')}
    pairs=len(asset['sender'])
    # Exact E122 priors: zero log gain offsets, raw strengths -9, tau5s.
    values=base|starts|dict(schema_version=np.array(2),network_keys_json=np.array(json.dumps(sorted(base))),
        network_schema_version=base['schema_version'],extrasynaptic_enabled=np.array(bool(enabled)),
        peptide_raw_exc=np.full(pairs,-9.,dtype=np.float64),peptide_raw_inh=np.full(pairs,-9.,dtype=np.float64),
        peptide_log_tau_s=np.array(np.log(5.)),peptide_sender=np.asarray(asset['sender']),
        peptide_receiver=np.asarray(asset['receiver']),peptide_asset_sha256=np.array(asset_sha256),
        source_network_sha256=np.array(network_sha256))
    validate(values)
    np.savez_compressed(destination,**values)


def validate(a):
    if int(a['schema_version'])!=2:raise ValueError('Explicit full-model schema2 required')
    if np.asarray(a['extrasynaptic_enabled']).shape!=() or a['extrasynaptic_enabled'].dtype.kind!='b':
        raise ValueError('Explicit boolean extrasynaptic flag required')
    n=len(a['neuron_ids']);k=len(a['peptide_raw_exc'])
    shapes={'peptide_raw_inh':(k,),'peptide_sender':(k,n),'peptide_receiver':(k,n),
        'start_voltage_mV':(4,n),'start_gates':(4,n,35),'start_calcium_uM':(4,n),'start_indicator':(4,n),'start_peptide':(4,n)}
    if any(a[key].shape!=shape for key,shape in shapes.items()):raise ValueError('Invalid full-state/asset shape')
    for key in shapes:
        if not np.isfinite(a[key]).all():raise ValueError('Nonfinite '+key)
    if not np.isfinite(a['peptide_raw_exc']).all() or any(np.min(a[k]) < -12 or np.max(a[k]) > 2.95 for k in ('peptide_raw_exc','peptide_raw_inh')):
        raise ValueError('Peptide gain bounds')
    if not np.isfinite(a['peptide_log_tau_s']) or not np.log(1.)<=float(a['peptide_log_tau_s'])<=np.log(30.):raise ValueError('Peptide timescale bounds')
    if np.min(a['start_peptide'])<0 or np.max(a['start_peptide'])>1:raise ValueError('Peptide memory bounds')
    if any(np.min(a[k])<0 for k in ('peptide_sender','peptide_receiver')):raise ValueError('Negative expression')


def load_model(path):
    import jax
    import jax.numpy as jnp
    from .conductance_network import load_network,ConductanceNetworkState
    from .extrasynaptic import PeptideParams,coupling_matrices
    from .peptide_state_replay import FullState
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    validate(a)
    base={k:a[k] for k in json.loads(str(a['network_keys_json']))};base['schema_version']=a['network_schema_version']
    with tempfile.TemporaryDirectory(prefix='r-elegans-model-') as temporary:
        network=Path(temporary)/'network.npz';np.savez(network,**base);p,f,c=load_network(network)
    p,f=jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.float64),(p,f))
    pp=PeptideParams(jnp.asarray(a['peptide_raw_exc']),jnp.asarray(a['peptide_raw_inh']),jnp.asarray(a['peptide_log_tau_s']))
    exc,inh=coupling_matrices(pp,a['peptide_sender'],a['peptide_receiver'])
    if not bool(a['extrasynaptic_enabled']):exc,inh=exc*0,inh*0
    start=FullState(ConductanceNetworkState(**{k:jnp.asarray(a['start_'+k]) for k in ConductanceNetworkState._fields}),jnp.asarray(a['start_peptide']))
    return p,f,c,pp,exc,inh,start,a


def save_state(path,state):
    np.savez_compressed(path,**state.neural._asdict(),peptide=np.asarray(state.peptide))


def load_state(path):
    import jax.numpy as jnp
    from .conductance_network import ConductanceNetworkState
    from .peptide_state_replay import FullState
    with np.load(path,allow_pickle=False) as z:
        state=FullState(ConductanceNetworkState(**{k:jnp.asarray(z[k]) for k in ConductanceNetworkState._fields}),jnp.asarray(z['peptide']))
    if not all(np.isfinite(x).all() for x in (*state.neural,state.peptide)):raise ValueError('Nonfinite continuation state')
    return state


def make_rest(config,rmd_indices,seconds,*,dt=.001,step_ms=.03125):
    """No-cue continuation with compact RMD, all-cell bounds and memory traces."""
    import jax
    import jax.numpy as jnp
    from .conductance_network import advance_network
    from .extrasynaptic import peptide_membrane_input,peptide_level_step
    from .peptide_state_replay import FullState
    steps=round(dt*1000/step_ms);frames=round(seconds/dt)
    if min(steps,frames)<1 or not np.isclose(steps*step_ms,dt*1000) or not np.isclose(frames*dt,seconds):raise ValueError('Invalid continuation clocks')
    @jax.jit
    def run(p,f,pp,exc,inh,start):
        def step(state,_):
            membrane=peptide_membrane_input(state.peptide,exc,inh,config.synaptic_unit_nS)
            neural=advance_network(state.neural,p,f,dt,steps,config,membrane_input=membrane)
            state=FullState(neural,peptide_level_step(state.peptide,neural.voltage_mV,f,dt,pp.log_tau_s))
            return state,(neural.voltage_mV[jnp.asarray(rmd_indices)],jnp.min(neural.voltage_mV),jnp.max(neural.voltage_mV),jnp.mean(state.peptide))
        return jax.lax.scan(step,start,None,length=frames)
    return run
