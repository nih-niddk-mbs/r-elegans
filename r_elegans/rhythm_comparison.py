"""Controlled parameter-family substitutions, with no optimization."""
import hashlib

import numpy as np


def assert_fixed_equal(old, new, old_config, new_config):
    if old_config != new_config:
        raise ValueError('Network configurations differ')
    def compare(a, b, path):
        if hasattr(a, '_asdict'):
            if not hasattr(b, '_asdict') or a._fields != b._fields:
                raise ValueError('Fixed structure differs: ' + path)
            for key in a._fields:
                compare(getattr(a, key), getattr(b, key), path + '.' + key)
        elif not np.array_equal(np.asarray(a), np.asarray(b)):
            raise ValueError('Fixed values differ: ' + path)
    compare(old, new, 'fixed')
    if np.any(np.asarray(old.bias_pA) != 0):
        raise ValueError('Zero injected bias required')


def hybrid_parameters(old, new, channel_source, connection_source):
    if channel_source not in ('old', 'new') or connection_source not in ('old', 'new'):
        raise ValueError('Parameter families must be old or new')
    channels = old if channel_source == 'old' else new
    connections = old if connection_source == 'old' else new
    return old._replace(log_channel_scale=channels.log_channel_scale,
                        raw_chemical=connections.raw_chemical, raw_gap=connections.raw_gap)


def comparison_cases():
    cases = []
    def add(name, channels, connections, cue=False, precision='float64', perturbation=0.,
            brain_factor=1., feedback=True):
        cases.append(dict(name=name, channels=channels, connections=connections, cue=cue,
                          precision=precision, perturbation_mV=perturbation,
                          brain_factor=brain_factor, proprioception=feedback))
    for channels in ('old', 'new'):
        for connections in ('old', 'new'):
            for cue in (False, True):
                add(f'{channels}-{connections}-'+('cue' if cue else 'sham'), channels, connections, cue)
    for sign, delta in (('plus', 1.), ('minus', -1.)):
        add('old-old-sham-voltage-'+sign, 'old', 'old', perturbation=delta)
    for source in ('old', 'new'):
        add(f'{source}-{source}-sham-neural-half', source, source, brain_factor=.5)
    add('old-old-sham-feedback-off', 'old', 'old', feedback=False)
    add('old-old-sham-float32', 'old', 'old', precision='float32')
    add('old-old-sham-float32-feedback-off', 'old', 'old', precision='float32', feedback=False)
    return cases


def array_fingerprint(leaves):
    digest = hashlib.sha256()
    for value in leaves:
        x = np.asarray(value)
        digest.update(str((x.dtype.str, x.shape)).encode())
        digest.update(x.tobytes(order='C'))
    return digest.hexdigest()
