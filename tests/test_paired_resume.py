import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.paired_resume import read_resume,MATCH_ARGUMENTS
from scripts.experiment_records import file_sha256


def test_resume_preserves_rng_bounds_and_refuses_changed_objective(tmp_path):
    args={k:1 for k in MATCH_ARGUMENTS};args['maximum_factor']=2
    config={k:'same' for k in ('network_sha256','readout_sha256','prepared_manifest_sha256','split_sha256','network_config','trainable')}
    config['arguments']=args.copy()
    (tmp_path/'configuration.json').write_text(json.dumps(config))
    (tmp_path/'network_best.npz').write_bytes(b'checkpoint')
    (tmp_path/'summary.json').write_text(json.dumps(dict(experiment='E050',status='completed',iterations=34,checkpoint_sha256=file_sha256(tmp_path/'network_best.npz'))))
    np.savez(tmp_path/'search_state.npz',offsets=[.1,.2],iteration=34)
    rng=np.random.default_rng(50);rng.normal(size=20)
    (tmp_path/'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
    offsets,iteration,restored,_=read_resume(tmp_path,config,2)
    assert iteration==34
    np.testing.assert_array_equal(offsets,[.1,.2])
    np.testing.assert_array_equal(restored.normal(size=20),rng.normal(size=20))
    changed={**config,'arguments':{**args,'frames':16}}
    with pytest.raises(ValueError,match='frames'):read_resume(tmp_path,changed,2)
    np.savez(tmp_path/'search_state.npz',offsets=[5.,.2],iteration=34)
    with pytest.raises(ValueError,match='bounds'):read_resume(tmp_path,config,2)
