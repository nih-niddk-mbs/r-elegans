"""Phase-free paired observation objective for an incremental fitting pilot.

Spatial observations are forward chord bends, not inferred joint angles. Source
nominal segment spacing is 7 binned pixels * 2 / 791 mm/pixel (pinned publisher
notebook, unit.jl and segmentation.jl). Pixel-quantized endpoints and confocal
averaging make the reconstruction approximate. No directional claim is made.
"""
import numpy as np

SOURCE_STEP_M = 7 * 2 / 791 * 1e-3
# Julia source points 1,9,...,49; first point starts one spacing behind nose.
SOURCE_POINTS = np.arange(1, 50, 8)


def chord_bends(points):
    """Unsigned, reflection/translation/rotation invariant adjacent chord bends."""
    points = np.asarray(points, dtype=float)
    delta = np.diff(points, axis=-2)
    a, b = delta[..., :-1, :], delta[..., 1:, :]
    cross = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    angle = np.abs(np.arctan2(cross, np.sum(a*b, axis=-1)))
    valid = (np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
             & (np.linalg.norm(a, axis=-1)>1e-12) & (np.linalg.norm(b, axis=-1)>1e-12))
    return np.where(valid, angle, np.nan)


def source_bends(angles):
    """Reconstruct nominal chords from released head-to-tail tangent angles.

    Positions up to a missing tangent remain usable; positions beyond it do not.
    Physical chord lengths cancel when evaluating bends. Return five observables.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] < SOURCE_POINTS[-1]-1:
        raise ValueError('Insufficient source tangent coverage')
    steps = SOURCE_STEP_M * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    positions = np.concatenate((np.zeros((len(angles),1,2)), np.cumsum(steps,axis=1)),axis=1)
    # positions[0] is Julia point 1 (nominal distance one step from the nose).
    return chord_bends(positions[:, SOURCE_POINTS-1])


def model_source_tangents(joints, segment_length):
    """Observe a 12-rod, tail-to-head model at the same nominal arclengths."""
    q = np.asarray(joints,dtype=float)
    if q.ndim != 2 or q.shape[1] != 11 or not np.isfinite(q).all():
        raise ValueError('Expected finite [time,11] model joints')
    if not np.isfinite(segment_length) or segment_length <= 0:
        raise ValueError('Positive rod length required')
    if SOURCE_POINTS[-1]*SOURCE_STEP_M > 12*segment_length:
        raise ValueError('Source observation extends beyond modeled worm')
    orientation = np.c_[np.zeros(len(q)),np.cumsum(q,axis=1)]
    vectors = segment_length*np.stack((np.cos(orientation),np.sin(orientation)),axis=-1)
    nodes = np.concatenate((np.zeros((len(q),1,2)),np.cumsum(vectors,axis=1)),axis=1)
    # Reverse arclength: the simulator's head is its final endpoint.
    s = 12*segment_length-np.arange(1,50)*SOURCE_STEP_M
    i = np.minimum((s/segment_length).astype(int),11)
    points = nodes[:,i]+(s/segment_length-i)[None,:,None]*vectors[:,i]
    delta = np.diff(points,axis=1)
    return np.arctan2(delta[...,1],delta[...,0])


def model_bends(joints, segment_length):
    return source_bends(model_source_tangents(joints, segment_length))


def fluctuation_statistics(signal, valid=None):
    """Mean, coefficient of variation and adjacent-frame correlation per column.

    These are scale-free descriptive statistics, not calibrated calcium or a
    voltage likelihood. Adjacent-frame correlation is only compared on the exact
    same observation timestamps. No cross-gap pair is allowed by the window loader.
    """
    x = np.asarray(signal,dtype=float)
    if x.ndim != 2:
        raise ValueError('Expected time-by-observable signal')
    mask = np.isfinite(x)
    if valid is not None:
        valid = np.asarray(valid,dtype=bool)
        if valid.shape != x.shape:
            raise ValueError('Mask shape mismatch')
        mask &= valid
    count = mask.sum(axis=0)
    mean = np.where(mask,x,0).sum(axis=0)/np.maximum(count,1)
    centered = np.where(mask,x-mean,0)
    std = np.sqrt((centered**2).sum(axis=0)/np.maximum(count,1))
    cv = std/np.maximum(np.abs(mean),1e-8)
    ac = np.full(x.shape[1],np.nan)
    for j in range(x.shape[1]):
        pairs = mask[:-1,j] & mask[1:,j]
        if pairs.sum() < 3:continue
        a,b = x[:-1,j][pairs],x[1:,j][pairs]
        a,b = a-a.mean(),b-b.mean()
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator > 1e-12:ac[j] = np.dot(a,b)/denominator
    return dict(mean=mean,std=std,cv=cv,lag1=ac,valid=count>=4)


def paired_objective(indicator, bends, target_fluorescence, target_bends, *,
                     fluorescence_valid=None, basal_brightness=.1):
    """Equal-family pilot losses; fixed assumed reporter baseline and scales.

    Relative emission is basal + Hill/lag reporter state from the network. An
    unknown constant optical gain cancels in CV and correlations; additive
    offsets, reference-channel fluctuations and noise remain unmodeled. The
    default scales (.1 CV, .1 rad, .5 correlation) are optimization weights,
    not experimental error bars. No held-out affine readout is fitted.
    """
    if not np.isfinite(basal_brightness) or basal_brightness <= 0:
        raise ValueError('Positive reporter basal brightness required')
    indicator = np.asarray(indicator,dtype=float)
    target = np.asarray(target_fluorescence,dtype=float)
    bends, target_bends = np.asarray(bends),np.asarray(target_bends)
    if indicator.shape != target.shape or bends.shape != target_bends.shape:
        raise ValueError('Predictions and observations must align')
    mask = np.isfinite(target)
    if fluorescence_valid is not None:mask &= np.asarray(fluorescence_valid,dtype=bool)
    measured = fluctuation_statistics(target,mask)
    predicted = fluctuation_statistics(basal_brightness+indicator,mask)
    neural = measured['valid'] & predicted['valid'] & (measured['mean']>1e-8)
    shape_mask = np.isfinite(target_bends)
    actual_shape = fluctuation_statistics(target_bends,shape_mask)
    predicted_shape = fluctuation_statistics(bends,shape_mask)
    shape = actual_shape['valid'] & predicted_shape['valid']
    if not neural.any() or not shape.any():
        raise ValueError('Both data families require usable observations')
    if not np.isfinite(indicator).all() or not np.isfinite(bends).all():
        raise ValueError('Nonfinite model prediction')
    # Missing predicted dynamics cannot make a measured correlation disappear.
    ac_mask = neural & np.isfinite(measured['lag1'])
    ac_pred = np.nan_to_num(predicted['lag1'],nan=0.)
    calcium_cv = float(np.mean(((predicted['cv'][neural]-measured['cv'][neural])/.1)**2))
    calcium_ac = float(np.mean(((ac_pred[ac_mask]-measured['lag1'][ac_mask])/.5)**2)) if ac_mask.any() else 0.
    bend_mean = float(np.mean(((predicted_shape['mean'][shape]-actual_shape['mean'][shape])/.1)**2))
    bend_std = float(np.mean(((predicted_shape['std'][shape]-actual_shape['std'][shape])/.1)**2))
    return dict(total=.5*(calcium_cv+calcium_ac)+.5*(bend_mean+bend_std),
                calcium_cv=calcium_cv,calcium_lag1=calcium_ac,bend_mean=bend_mean,bend_std=bend_std,
                neural_observables=int(neural.sum()),shape_observables=int(shape.sum()),
                predicted_bend_std_rad=predicted_shape['std'].tolist(),
                measured_bend_std_rad=actual_shape['std'].tolist())
