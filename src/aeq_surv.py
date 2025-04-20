import numpy as np


def aeq_surv(
    x: np.ndarray, tolerance: float = np.sqrt(np.finfo(float).eps)
) -> np.ndarray:
    """
    Adjust survival times so that very small differences are treated as ties.

    This mirrors R's `aeqSurv` behavior: times within `tolerance` (absolute or
    relative to the mean of the times) are binned to the same value.

    Parameters
    ----------
    x : np.ndarray
        A 2D array of shape (n_samples, 2) for simple survival (time, event)
        or (n_samples, 3) for interval survival (start, stop, event).
    tolerance : float, optional
        Maximum difference to consider two times tied. Defaults to sqrt(machine epsilon).

    Returns
    -------
    np.ndarray
        A new array of the same shape as `x`, with tied times binned to common cut values.

    Raises
    ------
    ValueError
        If `tolerance` is not a positive finite number, or if `x` is not a 2- or 3-column array.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a single positive finite number")
    if tolerance == 0:
        return x.copy()

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape[1] not in (2, 3):
        raise ValueError("x must be a 2D array with 2 or 3 columns")

    times = x_arr[:, :-1].ravel()
    y = np.sort(np.unique(times[np.isfinite(times)]))

    dy = np.diff(y)
    mean_abs_y = np.mean(np.abs(y)) if y.size > 0 else 0.0
    tied = (dy <= tolerance) | ((mean_abs_y > 0) & (dy / mean_abs_y <= tolerance))
    if not np.any(tied):
        return x_arr.copy()

    mask = np.concatenate(([True], ~tied))
    cuts = y[mask]

    def map_to_cut(vals: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(cuts, vals, side="right") - 1
        idx = np.clip(idx, 0, cuts.size - 1)
        return cuts[idx]

    if x_arr.shape[1] == 2:
        binned_times = map_to_cut(x_arr[:, 0])
        events = x_arr[:, 1].astype(int)
        return np.column_stack((binned_times, events))

    starts = map_to_cut(x_arr[:, 0])
    stops = map_to_cut(x_arr[:, 1])
    zero_len = np.where(starts == stops)[0]
    if zero_len.size > 0:
        orig_lengths = x_arr[zero_len, 1] - x_arr[zero_len, 0]
        if np.any(orig_lengths != 0):
            raise ValueError("aeqSurv exception: an interval has effective length 0")
    events = x_arr[:, 2].astype(int)
    return np.column_stack((starts, stops, events))
