import numpy as np
from scipy.interpolate import interp1d

def normalize_sequence(seq, n_points=101):
    """
    Interpolates a list of values to a fixed length using linear interpolation.
    """
    if len(seq) < 2:
        return np.full(n_points, seq[0] if seq else 0.0)

    x_orig = np.linspace(0, 1, num=len(seq))
    x_new = np.linspace(0, 1, num=n_points)
    f = interp1d(x_orig, seq, kind='linear')
    return f(x_new)
