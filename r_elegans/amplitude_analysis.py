"""Pure NumPy/SciPy sustained-amplitude analysis shared by fitting and reports."""
import numpy as np
from scipy.signal import find_peaks


def sustained_excursions(values):
    """Per-channel late-window amplitude; static/ramping/decaying traces fail.

    Repeated peaks and troughs establish wiggling, without a prescribed period
    or shared phase. Amplitude must persist across five late windows.
    """
    x=np.asarray(values,dtype=float);x=x[len(x)//2:]
    if x.ndim!=2 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Invalid amplitude trace')
    amplitudes=np.stack([np.ptp(w,axis=0) for w in np.array_split(x,5)])
    minimum=amplitudes.min(axis=0);maximum=amplitudes.max(axis=0);repeated=np.zeros(x.shape[1],dtype=bool)
    for i in np.flatnonzero(maximum>1e-8):
        prominence=max(1e-8,.1*np.ptp(x[:,i]))
        repeated[i]=(len(find_peaks(x[:,i],prominence=prominence)[0])>=3
            and len(find_peaks(-x[:,i],prominence=prominence)[0])>=3
            and minimum[i]>=.5*maximum[i])
    return np.where(repeated,minimum,0.),repeated


