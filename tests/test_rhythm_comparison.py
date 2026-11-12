from collections import namedtuple
import json
import subprocess
import sys

import numpy as np
import pytest

from r_elegans.rhythm_comparison import (array_fingerprint, assert_fixed_equal,
                                        comparison_cases, hybrid_parameters)
from scripts.compare_rhythm_worker import verify_inputs


def test_family_substitutions_preserve_sources_and_select_whole_synaptic_family():
    Params = namedtuple('Params', 'log_channel_scale raw_chemical raw_gap')
    old = Params(np.array([1.]), np.array([2.]), np.array([3.]))
    new = Params(np.array([4.]), np.array([5.]), np.array([6.]))
    for ch in ('old', 'new'):
        for syn in ('old', 'new'):
            result = hybrid_parameters(old, new, ch, syn)
            assert result.log_channel_scale is (old if ch == 'old' else new).log_channel_scale
            assert result.raw_chemical is (old if syn == 'old' else new).raw_chemical
            assert result.raw_gap is (old if syn == 'old' else new).raw_gap
    np.testing.assert_array_equal(old.raw_chemical, [2.])
    with pytest.raises(ValueError):
        hybrid_parameters(old, new, 'average', 'old')


def test_incompatible_fixed_fields_abort_instead_of_hybridizing_them():
    Cells = namedtuple('Cells', 'conductances')
    Fixed = namedtuple('Fixed', 'cells mask bias_pA')
    old = Fixed(Cells(np.array([.1], dtype=np.float32)), np.eye(2), np.zeros(2))
    promoted = old._replace(cells=Cells(old.cells.conductances.astype(np.float64)))
    assert_fixed_equal(old, promoted, 'same', 'same')
    with pytest.raises(ValueError, match='cells.conductances'):
        assert_fixed_equal(old, promoted._replace(cells=Cells(np.array([2.]))), 'same', 'same')
    with pytest.raises(ValueError, match='configurations'):
        assert_fixed_equal(old, promoted, 'one', 'other')
    with pytest.raises(ValueError, match='mask'):
        assert_fixed_equal(old, promoted._replace(mask=np.zeros((2, 2))), 'same', 'same')


def test_controls_isolate_precision_feedback_initialization_and_neural_step():
    cases = {c['name']: c for c in comparison_cases()}
    assert len(cases) == 15
    base = cases['old-old-sham']
    for label, delta in [('plus', 1.), ('minus', -1.)]:
        c = cases['old-old-sham-voltage-' + label]
        assert c == dict(base, name=c['name'], perturbation_mV=delta)
    for c in cases.values():
        if c['precision'] == 'float32':
            assert c['channels'] == c['connections'] == 'old'
    for feedback in (False, True):
        assert any(c['precision'] == 'float32' and c['proprioception'] == feedback for c in cases.values())
    assert cases['new-new-sham-neural-half']['brain_factor'] == .5


def test_state_fingerprint_and_input_manifest_detect_changes(tmp_path):
    x = np.array([1., 2.], dtype=np.float32)
    assert array_fingerprint([x]) == array_fingerprint([x.copy()])
    assert array_fingerprint([x]) != array_fingerprint([x.astype(np.float64)])
    assert array_fingerprint([x]) != array_fingerprint([x + 1.])
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(sha256={'old.npz': '0' * 64})))
    (tmp_path / 'old.npz').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_inputs(tmp_path)


def test_supervisor_does_not_initialize_jax():
    subprocess.run([sys.executable, '-c',
                    'import sys; import scripts.run_rhythm_comparison; assert "jax" not in sys.modules'],
                   check=True, capture_output=True)
