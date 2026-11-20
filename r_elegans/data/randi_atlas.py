"""Read Randi et al. 2023 (DANDI 001075) segmentation NWB files: traces, NeuroPAL labels, optogenetic stimulus table.

Only the small ``desc-segmentation`` files are read; no imaging data. A pump-probe ROI is named only when its
NeuroPAL label is a canonical neuron, its confidence reaches ``min_confidence`` and the name is unique in the animal.
Response amplitudes depend on the chosen windows and signal (documented parameters, not the paper's definition).
"""
from pathlib import Path

import numpy as np

SIGNAL_GROUP = 'processing/ophys/{color}Signals/Interpolated{color}Signal'


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_randi_animal(path, neuron_ids, *, min_confidence=.8):
    import h5py
    neuron_set = set(neuron_ids)
    with h5py.File(path, 'r') as f:
        green_group, red_group = (f[SIGNAL_GROUP.format(color=c)] for c in ('Green', 'Red'))
        time = np.asarray(green_group['timestamps'])
        green, red = np.asarray(green_group['data'], float), np.asarray(red_group['data'], float)
        if green.shape != red.shape or green.shape[0] != time.size:
            raise ValueError(f'{path}: inconsistent signal shapes')
        pump = f['processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation']
        roi_ids = np.asarray(pump['id'])
        neuropal_ref = [_text(v) for v in pump['neuropal_ids'][:]]
        nps = f['processing/ophys/NeuroPALSegmentations/NeuroPALPlaneSegmentation']
        nps_ids = [str(int(v)) for v in np.asarray(nps['id'])]
        labels = [_text(v) for v in nps['labels'][:]]
        confidence = np.asarray(nps['labels_confidences'], float)
        stim = f['intervals/OptogeneticStimulusTable']
        start, stop = np.asarray(stim['start_time'], float), np.asarray(stim['stop_time'], float)
        power = np.asarray(stim['power'], float)
        target_id = np.asarray(stim['target_pumpprobe_id'], float)
        subject = _text(f['general/subject/subject_id'][()]) if 'general/subject/subject_id' in f else Path(path).name
        genotype = _text(f['general/subject/genotype'][()]) if 'general/subject/genotype' in f else ''
    if roi_ids.size != green.shape[1]:
        raise ValueError(f'{path}: ROI id count differs from signal columns')
    lookup = {name: i for i, name in enumerate(nps_ids)}
    names, confidences, status = [], [], []
    for ref in neuropal_ref:
        row = lookup.get(ref)
        if row is None:
            names.append(None); confidences.append(np.nan); status.append('no_neuropal_match' if ref not in ('merge', '') else ref or 'blank')
            continue
        label, score = labels[row], confidence[row]
        confidences.append(score)
        if label in neuron_set and score >= min_confidence:
            names.append(label); status.append('named')
        else:
            names.append(None); status.append('unlabeled' if not label else 'not_canonical_or_low_confidence')
    counts = {n: names.count(n) for n in set(names) if n}
    names = [n if n and counts[n] == 1 else None for n in names]
    column = {int(i): c for c, i in enumerate(roi_ids)}
    target_column = np.array([column.get(int(t), -1) if np.isfinite(t) else -1 for t in target_id])
    return dict(path=str(path), subject=subject, genotype=genotype, time_s=time, green=green, red=red, roi_ids=roi_ids,
                names=names, confidence=np.asarray(confidences), status=status,
                stim_start_s=start, stim_stop_s=stop, stim_power=power, stim_target_id=target_id, stim_target_column=target_column)


def ratio_signal(animal):
    green, red = animal['green'], animal['red']
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(red > 0, green / red, np.nan)


def stimulus_responses(animal, *, pre_s=10., post_window=(1., 11.), signal=None):
    """Relative change (mean post - mean pre)/mean pre of the green/red ratio for each stimulation [stim, roi]."""
    signal = ratio_signal(animal) if signal is None else signal
    time = animal['time_s']
    out = np.full((animal['stim_start_s'].size, signal.shape[1]), np.nan)
    for k, (start, stop) in enumerate(zip(animal['stim_start_s'], animal['stim_stop_s'])):
        before = (time >= start - pre_s) & (time < start)
        after = (time >= stop + post_window[0]) & (time < stop + post_window[1])
        if before.sum() < 3 or after.sum() < 3:
            continue
        base = np.nanmean(signal[before], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[k] = (np.nanmean(signal[after], axis=0) - base) / base
    return out
