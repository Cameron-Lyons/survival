from typing import Sequence, Union
import numpy as np


def aareg_taper(
    taper: Sequence[float], imat: np.ndarray, nevent: Union[float, Sequence[float]]
) -> np.ndarray:
    """
    Compute running averages of an information matrix using taper weights.

    Parameters
    ----------
    taper : Sequence[float]
        A non-empty sequence of positive weights.
    imat : np.ndarray
        A 3D array of shape (n_coef1, n_coef2, n_time) representing the information matrix.
    nevent : Union[float, Sequence[float]]
        Number of events for each time point. Either a scalar or a sequence of length n_time.

    Returns
    -------
    np.ndarray
        A 3D array of the same shape as `imat`, containing the smoothed information matrix.

    Raises
    ------
    ValueError
        If `taper` is empty, contains non-positive values, or if `nevent` is neither a scalar
        nor a sequence of length equal to the time dimension of `imat`.
    """
    if not taper or any((not isinstance(x, (int, float))) for x in taper):
        raise ValueError(
            "Invalid taper vector: must be a non-empty sequence of numbers"
        )
    if any(x <= 0 for x in taper):
        raise ValueError("Invalid taper vector: all values must be positive")
    taper_arr = np.array(taper, dtype=float)
    ntaper = taper_arr.size

    if imat.ndim != 3:
        raise ValueError("imat must be a 3D array")
    p, q, ntime = imat.shape

    if ntaper > ntime:
        taper_arr = taper_arr[:ntime]
        ntaper = ntime

    imat_flat = imat.reshape(p * q, ntime)

    nevent_arr = np.array(nevent, dtype=float)
    if nevent_arr.ndim == 0:
        imat_flat = imat_flat / nevent_arr
    elif nevent_arr.ndim == 1 and nevent_arr.size == ntime:
        imat_flat = imat_flat / nevent_arr
    else:
        raise ValueError(f"nevent must be a scalar or sequence of length {ntime}")

    if ntaper > 1:
        smoother = np.zeros((ntime, ntime), dtype=float)
        tsum = np.cumsum(taper_arr[::-1])

        for i in range(ntaper):
            positions = np.linspace(0, ntaper - 1, num=i + 1, endpoint=True)
            idx = np.round(positions).astype(int)
            weights = taper_arr[idx] / tsum[i]
            smoother[: i + 1, i] = weights

        if ntaper < ntime:
            full_weights = taper_arr / tsum[-1]
            for i in range(ntaper, ntime):
                positions = np.linspace(0, i, num=ntaper, endpoint=True)
                rows = np.round(positions).astype(int)
                smoother[rows, i] = full_weights

        imat_flat = imat_flat @ smoother

    return imat_flat.reshape(p, q, ntime)
