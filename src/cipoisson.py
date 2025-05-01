from typing import Union, Tuple, Literal
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gamma, norm


def cipoisson(
    k: ArrayLike,
    time: ArrayLike = 1,
    p: ArrayLike = 0.95,
    method: Literal["exact", "anscombe"] = "exact",
) -> Union[Tuple[float, float], np.ndarray]:
    """
    Calculate Poisson confidence interval for rates per unit time.

    Parameters
    ----------
    k : int or array-like of int
        Observed counts of events.
    time : float or array-like of float, default 1
        Observation times corresponding to each count.
    p : float or array-like of float, default 0.95
        Confidence level (between 0 and 1).
    method : {'exact', 'anscombe'}, default 'exact'
        Method to use: 'exact' for exact interval using gamma distribution,
        'anscombe' for Anscombe's approximation.

    Returns
    -------
    tuple or ndarray
        If inputs are scalar, returns (lower_rate, upper_rate).
        If inputs are array-like, returns an ndarray of shape (n, 2)
        with columns [lower_rate, upper_rate].
    """
    k_arr = np.atleast_1d(k).astype(float)
    time_arr = np.atleast_1d(time).astype(float)
    p_arr = np.atleast_1d(p).astype(float)

    n = max(k_arr.size, time_arr.size, p_arr.size)
    if k_arr.size != n:
        k_arr = np.resize(k_arr, n)
    if time_arr.size != n:
        time_arr = np.resize(time_arr, n)
    if p_arr.size != n:
        p_arr = np.resize(p_arr, n)

    p_l = (1 - p_arr) / 2
    p_u = 1 - p_l

    method = method.lower()
    if method not in ("exact", "anscombe"):
        raise ValueError(f"Invalid method '{method}'. Choose 'exact' or 'anscombe'.")

    if method == "exact":
        dummy = np.where(k_arr == 0, 1, k_arr)
        lower = np.where(k_arr == 0, 0.0, gamma.ppf(p_l, a=dummy))
        upper = gamma.ppf(p_u, a=k_arr + 1)
    else:
        z = norm.ppf(p_l)
        lower = (np.sqrt(k_arr - 1 / 8) + z / 2) ** 2
        upper = (np.sqrt(k_arr + 7 / 8) - z / 2) ** 2

    mask = time_arr <= 0
    if np.any(mask):
        lower[mask] = np.nan
        upper[mask] = np.nan

    lower_rate = lower / time_arr
    upper_rate = upper / time_arr

    if n == 1:
        return float(lower_rate[0]), float(upper_rate[0])
    return np.vstack([lower_rate, upper_rate]).T
