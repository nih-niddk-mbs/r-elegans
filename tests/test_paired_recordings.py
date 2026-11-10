import h5py
import numpy as np
import pytest

from r_elegans.data.paired_recordings import grouped_splits, load_paired, segment_ids


def test_split_is_order_invariant_and_date_grouped():
    rows = [dict(uid=f'2022-06-{d:02}-0{i}', type='baseline,neuropal')
            for d in range(10,20) for i in (1,2)]
    split = grouped_splits(rows)
    assert split == grouped_splits(list(reversed(rows)))
    assert set(split.values()) == {'train','validation','test'}
    assert split['2022-06-14-01'] == 'train'
    for d in range(10,20):
        assert split[f'2022-06-{d:02}-01'] == split[f'2022-06-{d:02}-02']


def test_segments_honor_gaps_resets_and_reject_overlap():
    np.testing.assert_array_equal(segment_ids([0,1,2,9,10], [[1],[5]]), [0,0,0,1,1])
    np.testing.assert_array_equal(segment_ids([0,1,0,1], [[1,3],[2,4]]), [0,0,1,1])
    with pytest.raises(ValueError, match='Overlapping'):
        segment_ids([0,1,2], [[1,2],[2,3]])
    with pytest.raises(ValueError, match='increase'):
        segment_ids([0,1,0], [[1],[3]])


def recording(path, nonidentity=False):
    with h5py.File(path,'w') as f:
        f['timing/timestamp_confocal'] = [0.,1.,2.,3.]
        f['gcamp/trace_array_original'] = [[1,4,7],[2,5,8],[np.nan,6,9],[4,7,10]]
        f['gcamp/idx_splits'] = [[1],[4]]
        f['gcamp/match_org_to_skip'] = [[1,2,4] if nonidentity else [1,2,3],[1,2,3]]
        for key in ('head_angle','angular_velocity','velocity','body_angle_all','body_angle_absolute','reversal_vec'):
            f['behavior/'+key] = np.arange(4.)


def test_masks_orientation_duplicate_and_uncertain_indexing(tmp_path):
    path = tmp_path/'record.h5'
    recording(path)
    labels = {'1':dict(label='A',confidence=4), '2':dict(label='B',confidence=4),
              '3':dict(label='B',confidence=4)}
    arrays, qc = load_paired(path, labels, ['A','B'], ventral=True)
    assert qc['identified_neurons'] == 1
    assert not arrays['fluorescence_valid'][2,0]
    assert np.isnan(arrays['fluorescence'][2,0])
    np.testing.assert_array_equal(arrays['head_angle'], -np.arange(4.))
    np.testing.assert_array_equal(arrays['velocity'], np.arange(4.))
    recording(path, nonidentity=True)
    arrays, qc = load_paired(path, labels, ['A','B'], ventral=False)
    assert arrays['fluorescence'].shape == (4,0)
    assert not qc['identity_roi_mapping']
    assert 'unverified_nonidentity_roi_index_space' in {r['reason'] for r in qc['excluded_labels']}
