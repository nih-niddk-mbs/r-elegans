"""Import identified calcium recordings without guessing neuron identities."""
import bz2
import json
from pathlib import Path

import numpy as np


def load_identified_calcium(h5_path,label_path,neuron_ids,uid,*,frames=24,
                             start_frame=0,min_confidence=4.):
    import h5py
    with h5py.File(h5_path) as f:
        time=np.asarray(f["timing/timestamp_confocal"])
        traces=np.asarray(f["gcamp/trace_array_original"])
    if traces.ndim != 2 or traces.shape[0] != time.size:
        raise ValueError("Expected HDF5 [time,neuron] raw ratiometric traces")
    if frames < 8 or start_frame < 0 or start_frame+frames >= len(time):
        raise ValueError("Need >=8 frames after an initial reference frame")
    raw=Path(label_path).read_bytes()
    labels=json.loads(bz2.decompress(raw) if str(label_path).endswith('.bz2') else raw)
    labels=labels['data'][uid]['idx_neuron-label']
    candidates=[]
    for index,item in labels.items():
        name=item.get('label')
        row=int(index)-1  # source Julia label indices are one-based
        if name in neuron_ids and item.get('confidence',0) >= min_confidence and 0 <= row < traces.shape[1]:
            candidates.append((name,row,float(item['confidence'])))
    names=[c[0] for c in candidates]
    candidates=sorted([c for c in candidates if names.count(c[0]) == 1],key=lambda c:neuron_ids.index(c[0]))
    selection=slice(start_frame+1,start_frame+frames+1)
    reference_time=time[start_frame:start_frame+frames+1]
    dt=np.diff(reference_time)
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Finite strictly increasing timestamps required")
    values=traces[selection]
    train_frames=max(4,int(frames*.75))
    valid=[]
    for name,row,confidence in candidates:
        if np.isfinite(values[:,row]).all() and np.std(values[:train_frames,row]) > 1e-8:
            valid.append((name,row,confidence))
    if not valid:raise ValueError("No unambiguous finite canonical neuron traces")
    indices=np.array([neuron_ids.index(c[0]) for c in valid])
    target=values[:,[c[1] for c in valid]]
    mean=target[:train_frames].mean(axis=0)
    scale=target[:train_frames].std(axis=0)
    return dict(time_s=reference_time[1:]-reference_time[0],intervals_s=dt,
        fluorescence=target,normalized=(target-mean)/scale,training_mean=mean,
        training_std=scale,train_frames=train_frames,indices=indices,
        names=[c[0] for c in valid],rows=[c[1] for c in valid],
        confidence=[c[2] for c in valid],source_uid=uid,
        excluded_labels=len(labels)-len(valid),normalization="Training-window mean/std only")
