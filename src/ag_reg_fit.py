from typing import Optional, Sequence, Union, Dict, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings


@dataclass
class Control:
    """
    Control parameters for the Cox aggregation fit.
    """

    iter_max: int
    eps: float
    toler_chol: float
    toler_inf: float


def agreg_fit(
    x: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    strata: Optional[Sequence[Union[int, str]]] = None,
    offset: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    control: Control = Control(iter_max=25, eps=1e-9, toler_chol=1e-12, toler_inf=1e6),
    weights: Optional[np.ndarray] = None,
    method: str = "efron",
    rownames: Optional[List[str]] = None,
    resid: bool = True,
    nocenter: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Fit an aggregated Cox proportional hazards model.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Covariate matrix (n_samples x n_vars).
    y : np.ndarray or pd.DataFrame
        Survival data with three columns: [start, stop, event].
    strata : sequence of int or str, optional
        Stratification factor for each observation. Defaults to no stratification.
    offset : np.ndarray, optional
        Offset term for each observation. Defaults to zeros.
    init : np.ndarray, optional
        Initial coefficient estimates. Defaults to zeros.
    control : Control
        Control parameters (max iterations, tolerances).
    weights : np.ndarray, optional
        Observation weights. Defaults to ones.
    method : {'efron', 'breslow'}
        Method for handling ties. Defaults to 'efron'.
    rownames : list of str, optional
        Names for each observation (used for residuals).
    resid : bool
        Whether to compute martingale residuals. Defaults to True.
    nocenter : sequence of float, optional
        Values for which a covariate column should NOT be centered.

    Returns
    -------
    result : dict
        A dictionary containing fit results. Keys include:
        - coefficients, var, loglik, score, iter, linear_predictors,
          residuals (if resid=True), means, first, info, method, class.
    """
    if isinstance(x, pd.DataFrame):
        feature_names = list(x.columns)
        x = x.values
    else:
        feature_names = [f"x{i}" for i in range(x.shape[1])]
        x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)
    n_obs = y.shape[0]

    event = y[:, 2]
    if np.all(event == 0):
        raise ValueError("Can't fit a Cox model with 0 failures")

    if offset is None:
        offset = np.zeros(n_obs, dtype=float)
    else:
        offset = np.asarray(offset, dtype=float)
    if weights is None:
        weights = np.ones(n_obs, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if np.any(weights <= 0):
            raise ValueError("Invalid weights, must be > 0")

    if not strata:
        y1 = y[:, 0]
        y2 = y[:, 1]
        strata_arr = np.zeros(n_obs, dtype=int)
    else:
        labels, strata_arr = pd.factorize(strata)
        delta = strata_arr * (1 + y[:, 1].max() - y[:, 0].min())
        y1 = y[:, 0] + delta
        y2 = y[:, 1] + delta

    event_mask = y[:, 2] > 0
    dtime = np.sort(np.unique(y2[event_mask]))
    idx1 = np.searchsorted(dtime, y1, side="right") - 1
    idx2 = np.searchsorted(dtime, y2, side="right") - 1
    ignore = idx1 == idx2
    n_used = int(np.sum(~ignore))

    if np.all(strata_arr == 0):
        sort_end = np.lexsort((ignore, -y[:, 1]))
        sort_start = np.lexsort((ignore, -y[:, 0]))
    else:
        sort_end = np.lexsort((ignore, strata_arr, -y[:, 1]))
        sort_start = np.lexsort((ignore, strata_arr, -y[:, 0]))

    n_var = x.shape[1]
    if n_var == 0:
        n_var = 1
        x = np.arange(1, n_obs + 1, dtype=float).reshape(-1, 1)
        maxiter = 0
        null_model = True
        if init is not None and init.size != 0:
            raise ValueError("Wrong length for initial values")
        init = np.array([0.0])
    else:
        null_model = False
        maxiter = control.iter_max
        if init is None:
            init = np.zeros(n_var, dtype=float)
        else:
            init = np.asarray(init, dtype=float)
            if init.size != n_var:
                raise ValueError("Wrong length for initial values")

    if nocenter is None:
        zero_one = np.zeros(x.shape[1], dtype=bool)
    else:
        zero_one = np.all(np.isin(x, list(nocenter)), axis=0)

    x = x.astype(float)
    y = y.astype(float)
    offset = offset.astype(float)
    weights = weights.astype(float)

    agfit = Cagfit4(
        n_used,
        y,
        x,
        strata_arr,
        weights,
        offset,
        init,
        sort_start,
        sort_end,
        int(method == "efron"),
        maxiter,
        control.eps,
        control.toler_chol,
        np.where(zero_one, 0, 1),
    )

    agmeans = np.where(zero_one, 0.0, x.mean(axis=0))

    vmat = agfit["imat"]
    coef = agfit["coef"].astype(float)

    flag = np.asarray(agfit["flag"], dtype=int)
    if flag[0] < n_var:
        which_sing = np.diag(vmat) == 0
    else:
        which_sing = np.zeros(n_var, dtype=bool)

    if maxiter > 1:
        u = np.asarray(agfit["u"], dtype=float)
        infs = np.abs(u @ vmat)
        if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(vmat)):
            raise FloatingPointError("Numeric overflow in optimization.")
        if flag[3] > 0:
            warnings.warn("Ran out of iterations and did not converge")
        else:
            bad = (~np.isfinite(u)) | (infs > control.toler_inf * (1 + np.abs(coef)))
            if np.any(bad):
                warnings.warn(
                    f"Loglik converged before variable {np.where(bad)[0]}; beta may be infinite."
                )

    lp = x.dot(coef) + offset - np.sum(coef * agmeans)

    residuals = None
    if resid:
        xmax = np.log(np.finfo(float).max)
        if np.any(lp > xmax):
            temp = lp + xmax - (1 + lp.max())
            score = np.exp(temp)
        else:
            score = np.exp(lp)
        residuals = Cagmart3(
            n_used,
            y,
            score,
            weights,
            strata_arr,
            sort_start,
            sort_end,
            int(method == "efron"),
        )
        if rownames is not None:
            residuals = pd.Series(residuals, index=rownames)

    if null_model:
        result: Dict[str, Any] = {
            "loglik": agfit["loglik"][1],
            "linear_predictors": offset,
            "method": method,
            "class": ["coxph.null", "coxph"],
        }
        if resid:
            result["residuals"] = residuals
    else:
        coef = pd.Series(coef, index=feature_names)
        if maxiter > 0:
            coef.iloc[which_sing] = np.nan
        info = {
            "rank": flag[0],
            "rescale": flag[1],
            "step_halving": flag[2],
            "convergence": flag[3],
        }
        result = {
            "coefficients": coef,
            "var": vmat,
            "loglik": agfit["loglik"],
            "score": agfit["sctest"],
            "iter": agfit["iter"],
            "linear_predictors": lp,
            "means": agmeans,
            "first": agfit["u"],
            "info": info,
            "method": method,
            "class": "coxph",
        }
        if resid:
            result["residuals"] = residuals

    return result
