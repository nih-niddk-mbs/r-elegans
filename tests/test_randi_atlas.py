import h5py
import numpy as np

from r_elegans.data.randi_atlas import load_randi_animal, stimulus_responses


def write_nwb(path):
    text = h5py.string_dtype()
    time = np.arange(0., 200., .5)
    green = np.ones((time.size, 4))
    red = np.ones((time.size, 4))
    green[(time >= 101.) & (time < 112.), 1] = 2.  # ROI 1 responds after a 100.0-100.5 s stimulus
    with h5py.File(path, 'w') as f:
        for color, data in (('Green', green), ('Red', red)):
            group = f.create_group(f'processing/ophys/{color}Signals/Interpolated{color}Signal')
            group['timestamps'], group['data'] = time, data
        pump = f.create_group('processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation')
        pump['id'] = np.arange(4)
        pump.create_dataset('neuropal_ids', data=np.array(['0', 'merge', '2', '3'], dtype=object), dtype=text)
        nps = f.create_group('processing/ophys/NeuroPALSegmentations/NeuroPALPlaneSegmentation')
        nps['id'] = np.arange(4)
        nps.create_dataset('labels', data=np.array(['AWAR', 'AVAL', 'AWAR', 'AMso'], dtype=object), dtype=text)
        nps['labels_confidences'] = np.array([1., 1., .5, 1.])
        stim = f.create_group('intervals/OptogeneticStimulusTable')
        stim['start_time'], stim['stop_time'] = np.array([100., 150.]), np.array([100.5, 150.5])
        stim['power'] = np.array([.001, .001])
        stim['target_pumpprobe_id'] = np.array([1., np.nan])
        f.create_group('general/subject').create_dataset('genotype', data='WT', dtype=text)


def test_roi_names_follow_confidence_uniqueness_and_canonical_rules(tmp_path):
    path = tmp_path / 'a.nwb'
    write_nwb(path)
    animal = load_randi_animal(path, ['AWAR', 'AVAL', 'AIYL'])
    # ROI 0 -> AWAR (confident); ROI 1 'merge'; ROI 2 -> AWAR at confidence .5 (rejected); ROI 3 -> not a canonical neuron
    assert animal['names'] == ['AWAR', None, None, None]
    assert animal['status'] == ['named', 'merge', 'not_canonical_or_low_confidence', 'not_canonical_or_low_confidence']
    assert animal['genotype'] == 'WT'


def test_target_columns_use_roi_ids_and_missing_targets_are_negative(tmp_path):
    path = tmp_path / 'a.nwb'
    write_nwb(path)
    animal = load_randi_animal(path, ['AWAR'])
    assert animal['stim_target_column'].tolist() == [1, -1]


def test_stimulus_response_reports_relative_ratio_change_in_the_declared_windows(tmp_path):
    path = tmp_path / 'a.nwb'
    write_nwb(path)
    animal = load_randi_animal(path, ['AWAR'])
    response = stimulus_responses(animal, pre_s=10., post_window=(1., 11.))
    assert np.allclose(response[0], [0., 1., 0., 0.])
