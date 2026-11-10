"""Conservative admission of Atanas/Kim paired recordings, without fitting.

Author-processed signals are preserved, including missing values. No resampling,
normalization, neuron imputation, or 62-angle-to-11-joint conversion is performed.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def grouped_splits(rows):
    """Freeze date groups using metadata alone; previously fitted date is train.

    Dates are conservative session groups, not verified unique animal IDs.
    All recording types on a date inherit its fold, including quarantined types.
    """
    dates = sorted({r["uid"][:10] for r in rows})
    exposed = "2022-06-14"
    eligible = {r["uid"][:10] for r in rows if r["type"] == "baseline,neuropal"}
    ordered = sorted(eligible - {exposed}, key=lambda d: hashlib.sha256(
        ("E049-v1:" + d).encode()).hexdigest())
    n = max(1, len(eligible) // 5)
    if len(ordered) < 2 * n + 1:
        raise ValueError("Need independent metadata groups for all three folds")
    folds = {d: "train" for d in dates}
    folds.update({d: "test" for d in ordered[:n]})
    folds.update({d: "validation" for d in ordered[n:2*n]})
    return {r["uid"]: folds[r["uid"][:10]] for r in rows}


def segment_ids(time, source_splits):
    """Honor inclusive one-based Julia ranges and additional acquisition gaps."""
    time = np.asarray(time)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("Need finite timestamp vector")
    ranges = np.asarray(source_splits)
    if ranges.ndim != 2 or ranges.shape[0] != 2:
        raise ValueError("Expected HDF5 split array [2,segments]")
    if not np.equal(ranges, np.floor(ranges)).all():
        raise ValueError("Split indices must be integers")
    ids = np.full(len(time), -1, dtype=int)
    next_id = 0
    for start, end in ranges.T.astype(int):
        if start < 1 or end < start or end > len(time):
            raise ValueError("Invalid segment bounds")
        sl = slice(start-1, end)
        if np.any(ids[sl] >= 0):
            raise ValueError("Overlapping segments")
        dt = np.diff(time[sl])
        if np.any(dt <= 0):
            raise ValueError("Timestamps must increase within each source segment")
        breaks = dt > 1.2 * np.median(dt) if len(dt) else np.array([], dtype=bool)
        ids[sl] = next_id + np.r_[0, np.cumsum(breaks)]
        next_id = int(ids[end-1]) + 1
    if np.any(ids < 0):
        raise ValueError("Source splits do not cover recording")
    return ids


def load_paired(path, labels, neuron_ids, *, ventral, min_confidence=4.):
    """Load aligned signals; quarantine uncertain ROI remappings explicitly.

    The released label generator does not establish a nonidentity skip mapping's
    label index space. Accept only identity mappings pending source verification.
    This restriction is intentionally stricter than the legacy short-window fit.
    """
    import h5py
    if not isinstance(ventral, bool):
        raise ValueError("Explicit dorsal/ventral orientation required")
    with h5py.File(path) as f:
        time = np.asarray(f["timing/timestamp_confocal"], dtype=float)
        traces = np.asarray(f["gcamp/trace_array_original"], dtype=float)
        if traces.ndim != 2 or traces.shape[0] != len(time):
            raise ValueError("Expected time-by-neuron traces")
        segments = segment_ids(time, np.asarray(f["gcamp/idx_splits"]))
        mapping = np.asarray(f["gcamp/match_org_to_skip"])
        identity = (mapping.shape == (2, traces.shape[1]) and
                    np.array_equal(mapping[0], mapping[1]) and
                    set(mapping[1]) == set(range(1, traces.shape[1]+1)))
        names = Counter(x.get("label") for x in labels.values()
                        if isinstance(x, dict) and x.get("confidence", 0) >= min_confidence)
        selected, excluded = [], []
        for index, item in labels.items():
            if not isinstance(item, dict):
                excluded.append({"index": index, "reason": "ambiguous_label_entry"})
                continue
            name = item.get("label")
            row = int(index)-1
            reason = None
            if name not in neuron_ids: reason = "noncanonical_label"
            elif item.get("confidence", 0) < min_confidence: reason = "low_confidence"
            elif names[name] != 1: reason = "duplicate_identity"
            elif not 0 <= row < traces.shape[1]: reason = "out_of_bounds"
            elif not identity: reason = "unverified_nonidentity_roi_index_space"
            if reason:
                excluded.append({"index": index, "label": name, "reason": reason})
            else:
                selected.append((neuron_ids.index(name), row, item["confidence"]))
        selected.sort()
        rows = [s[1] for s in selected]
        values = traces[:, rows]
        arrays = dict(time_s=time, segment_id=segments, fluorescence=values,
                      fluorescence_valid=np.isfinite(values),
                      neuron_indices=np.array([s[0] for s in selected], dtype=int),
                      source_trace_columns=np.array(rows, dtype=int),
                      confidence=np.array([s[2] for s in selected]))
        behavior = {}
        # Only these signs are documented by the publisher's orientation notice.
        for key, unit in (("head_angle", "rad"), ("angular_velocity", "rad/s"),
                          ("velocity", "mm/s"), ("body_angle_all", "rad"),
                          ("body_angle_absolute", "rad"), ("reversal_vec", "binary")):
            value = np.asarray(f["behavior/"+key])
            if value.shape[0] != len(time):
                raise ValueError("Behavior not aligned with confocal timestamps: " + key)
            if key in ("head_angle", "angular_velocity"):
                value = value * (-1 if ventral else 1)
            arrays[key] = value
            arrays[key+"_valid"] = np.isfinite(value)
            behavior[key] = {"unit": unit, "shape": list(value.shape),
                             "finite_fraction": float(np.isfinite(value).mean())}
    return arrays, {"frames": len(time), "segments": int(segments.max()+1),
                    "identified_neurons": len(rows), "identity_roi_mapping": identity,
                    "excluded_labels": excluded, "behavior": behavior}
