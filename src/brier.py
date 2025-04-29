
"""
Brier score calculation for Cox proportional hazards models, translated from R.
"""

from typing import Optional, Any, Dict
import numpy as np
import pandas as pd


class CoxPHModel:
    """
    Placeholder for a fitted Cox proportional hazards model.

    Attributes
    ----------
    coef_ : np.ndarray
        Regression coefficients.
    means_ : np.ndarray
        Mean covariate values from training data.
    method : str
        Fitting method, e.g., 'efron' or 'breslow'.
    call : Any
        Original call information (e.g., data reference).
    is_coxph : bool
        Marker that this is a Cox PH model.
    """
    coef_: np.ndarray
    means_: np.ndarray
    method: str
    call: Any
    is_coxph: bool = True


class SurvFitResult:
    """
    Placeholder for `survfit` output.

    Attributes
    ----------
    time : np.ndarray
        Event times.
    n_event : np.ndarray
        Number of events at each time.
    surv : np.ndarray
        Survival probabilities.
    """
    time: np.ndarray
    n_event: np.ndarray
    surv: np.ndarray


class SummaryResult:
    """
    Placeholder for `summary(survfit)` output.

    Attributes
    ----------
    surv : np.ndarray
        Survival probabilities at requested times.
    """
    surv: np.ndarray



def survfit(
    fit: CoxPHModel,
    newdata: Optional[pd.DataFrame] = None,
    se_fit: bool = True,
    weights: Optional[np.ndarray] = None,
    ctype: int = 1,
    stype: int = 1
) -> SurvFitResult:
    """
    Compute survival fit; must be implemented separately.
    """
    raise NotImplementedError


def summary_survfit(
    sfit: SurvFitResult,
    times: np.ndarray,
    extend: bool = False
) -> SummaryResult:
    """
    Summarize survival fit at specific times; must be implemented separately.
    """
    raise NotImplementedError


def model_frame(fit: CoxPHModel, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Extract model frame; must be implemented separately."""
    raise NotImplementedError


def is_Surv(obj: Any) -> bool:
    """Check if object is a survival response; must be implemented."""
    raise NotImplementedError


def aeq_surv(Y: Any) -> Any:
    """Adjust survival times for ties; must be implemented."""
    raise NotImplementedError


def model_weights(mf: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract case weights; must be implemented."""
    raise NotImplementedError


def model_extract(mf: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    """Extract a variable (e.g., 'id') from model frame; must be implemented."""
    raise NotImplementedError


def survcheck2(Y: Any, id: np.ndarray) -> Any:
    """Check survival data integrity; must be implemented."""
    raise NotImplementedError


def survfit0(sfit: SurvFitResult) -> SurvFitResult:
    """Compute support for Kaplan-Meier on censoring; must be implemented."""
    raise NotImplementedError


def find_interval(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Find interval indices as R's `findInterval`.
    """
    return np.searchsorted(bin_edges, x, side='right') - 1


def brier(
    fit: CoxPHModel,
    times: Optional[np.ndarray] = None,
    newdata: Optional[pd.DataFrame] = None,
    ties: bool = True,
    detail: bool = False,
    timefix: bool = True,
    efron: bool = False
) -> Dict[str, Any]:
    """
    Compute the Brier score for predicted survival from a Cox model.

    Parameters
    ----------
    fit : CoxPHModel
        A fitted Cox proportional hazards model.
    times : np.ndarray, optional
        Times at which to evaluate the Brier score. If None, uses event times
        with observed events.
    newdata : pd.DataFrame, optional
        New data for prediction. If None, uses original training data.
    ties : bool, default=True
        Whether to jitter censored times to the right for ties handling.
    detail : bool, default=False
        If True, return additional components: p0, phat, and effective sample size.
    timefix : bool, default=True
        Whether to apply time-fixing transformation for censoring.
    efron : bool, default=False
        Whether to use Efron's method for baseline hazard when available.

    Returns
    -------
    Dict[str, Any]
        - 'rsquared': R^2 measure based on Brier scores.
        - 'brier'  : Vector of Brier scores for the model.
        - 'times'  : Evaluation times.
        If `detail=True`, also includes:
        - 'p0'     : Baseline event probabilities.
        - 'phat'   : Predicted event probabilities from the model.
        - 'eff_n'  : Effective sample size at each time.

    Raises
    ------
    ValueError
        If `fit` is not a CoxPHModel or delayed entry is present.
    """
    if not getattr(fit, 'is_coxph', False):
        raise ValueError("fit must be a CoxPHModel object")

    mf = model_frame(fit, data=newdata) if newdata is not None else model_frame(fit)
    Y = mf.iloc[:, 0]
    if not is_Surv(Y):
        raise ValueError("response must be a survival object")
    type_ = getattr(Y, 'type', None)
    if type_ not in ('right', 'mright', 'counting', 'mcounting'):
        raise ValueError("response must be right-censored")

    if timefix:
        Y = aeq_surv(Y)

    casewt = model_weights(mf)
    n = len(Y)
    if casewt is None:
        casewt = np.ones(n)
    else:
        casewt = np.asarray(casewt, dtype=float)
        if np.any(casewt < 0) or not np.all(np.isfinite(casewt)):
            raise ValueError("weights must be non-negative and finite")
    casewt = casewt / np.sum(casewt)

    id_arr = model_extract(mf, 'id') or np.arange(n)
    check = survcheck2(Y, id_arr) if id_arr is not None else None
    if check is not None:
        flags = getattr(check, 'flag', None)
        if flags is not None and np.any(flags > 0):
            raise ValueError("one or more flags are >0 in survcheck")
        simple = True
    else:
        simple = True

    if not simple:
        raise NotImplementedError("delayed entry is not implemented")

    if efron and getattr(fit, 'method', '') == 'efron':
        s0 = survfit(Y ~ 1, weights=casewt, se_fit=False, ctype=2, stype=2)  # type: ignore
    else:
        s0 = survfit(Y ~ 1, weights=casewt, se_fit=False, stype=1)  # type: ignore
    if times is None:
        times = s0.time[s0.n_event > 0]
    p0 = 1 - summary_survfit(s0, times, extend=True).surv

    s1 = survfit(fit, newdata=newdata, se_fit=False)
    p1 = 1 - summary_survfit(s1, times, extend=True).surv

    Y_arr = np.asarray(Y)
    dtime = Y_arr[:, -2]
    dstat = Y_arr[:, -1]
    ntime = len(times)

    if ties:
        unique_times = np.unique(dtime)
        mindiff = np.min(np.diff(np.sort(unique_times)))
        dtime = dtime + np.where(dstat == 0, mindiff / 2.0, 0.0)

    censor_fit = survfit(~Surv(dtime, 1 - dstat) ~ 1, weights=casewt)  # type: ignore
    c0 = survfit0(censor_fit)

    n = len(dtime)
    brier_mat = np.zeros((ntime, 2))
    eff_n = np.zeros(ntime)

    for i, t in enumerate(times):
        idx = find_interval(np.minimum(dtime, t), c0.time)
        surv_c0 = c0.surv[idx]
        wt = np.where((dtime < t) & (dstat == 0), 0.0, casewt / surv_c0)
        eff_n[i] = 1.0 / np.sum(wt**2)

        b0 = np.where(dtime > t, p0[i]**2, (dstat - p0[i])**2)
        b1 = np.where(dtime > t, p1[:, i]**2, (dstat - p1[:, i])**2)

        total_wt = np.sum(wt)
        brier_mat[i, 0] = np.sum(wt * b0) / total_wt
        brier_mat[i, 1] = np.sum(wt * b1) / total_wt

    rsq = 1.0 - (brier_mat[:, 1] / brier_mat[:, 0])

    result: Dict[str, Any] = {
        'rsquared': rsq,
        'brier': brier_mat[:, 1],
        'times': times
    }
    if detail:
        result.update({'p0': p0, 'phat': p1, 'eff_n': eff_n})

    return result
