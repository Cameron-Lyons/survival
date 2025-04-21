from typing import Optional, Sequence, Union, Dict, Any
import numpy as np


def ag_exact_fit(
    x: np.ndarray,
    y: np.ndarray,
    strata: Optional[Sequence[Union[int, str]]] = None,
    offset: Optional[Sequence[float]] = None,
    init: Optional[Sequence[float]] = None,
    control: Dict[str, Any] = None,
    weights: Optional[Sequence[float]] = None,
    nocenter: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Fit a Cox proportional‐hazards model using the exact method for ties.

    This mirrors R's `agexact.fit`:
     - Validates inputs
     - Sorts by stop time (and strata, if given), breaking ties by event indicator
     - Calls into a lower‐level routine to compute coefficients, var‐cov matrix,
       log‐likelihood, score test, and (optionally) martingale residuals.

    Parameters
    ----------
    x
        Design matrix of shape (n_samples, n_covariates).
    y
        2D array of shape (n_samples, 2) or (n_samples, 3).
        If 3 columns: (start, stop, event).
        If 2 columns: (time, event), with start=0.
    strata
        Optional stratum labels per observation.  If None or empty, no stratification.
    offset
        Optional offset per observation (length n_samples).  Defaults to zeros.
    init
        Optional initial coefficient vector (length n_covariates).  Defaults to zeros.
    control
        Dict with control parameters:
          - "iter_max": int, maximum iterations
          - "eps": float, convergence threshold
          - "toler_chol": float, tolerance for Cholesky
          - "toler_inf": float, tolerance for infinite estimates
    weights
        Observation weights.  Must be all ones for the exact method.
    method
        Should be "exact" (only exact implemented here).
    rownames
        Optional list of row names (for residual vector indexing).
    resid
        If True, compute martingale residuals.
    nocenter
        Optional sequence of values; any covariate column whose entries are all
        in this set will not be centered.

    Returns
    -------
    results : dict
        - "coefficients": np.ndarray, shape (n_covariates,)
        - "var": np.ndarray, shape (n_covariates, n_covariates)
        - "loglik": np.ndarray, length 2 (log‐lik before/after)
        - "score": float, score test statistic
        - "iter": int, iterations performed
        - "linear_predictors": np.ndarray, shape (n_samples,)
        - "residuals": np.ndarray of martingale residuals (if resid=True)
        - "means": np.ndarray, covariate means used for centering
        - "method": str, "coxph"
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    n, nvar = x.shape

    if weights is not None and any(w != 1 for w in weights):
        raise ValueError("Case weights are not supported for the exact method")

    if y.ndim != 2 or y.shape[0] != n or y.shape[1] not in (2, 3):
        raise ValueError("y must be shape (n,2) or (n,3), matching x rows")
    if y.shape[1] == 3:
        start = y[:, 0].astype(float)
        stop = y[:, 1].astype(float)
        event = y[:, 2].astype(int)
    else:
        start = np.zeros(n, dtype=float)
        stop = y[:, 0].astype(float)
        event = y[:, 1].astype(int)

    if not strata:
        sorted_idx = np.lexsort((-event, stop))
        newstrat = np.zeros(n, dtype=int)
    else:
        strata_arr = np.array(strata)
        sorted_idx = np.lexsort((-event, stop, strata_arr))
        strata_sorted = strata_arr[sorted_idx]
        boundaries = np.diff(strata_sorted) != 0
        newstrat = np.concatenate((boundaries.astype(int), [1]))

    if offset is None:
        offset_arr = np.zeros(n, dtype=float)
    else:
        offset_arr = np.array(offset, dtype=float)
        if offset_arr.size != n:
            raise ValueError("offset must have length n")

    # Reorder data
    sstart = start[sorted_idx]
    sstop = stop[sorted_idx]
    sstat = event[sorted_idx]
    x_sorted = x[sorted_idx, :]
    offset_sorted = offset_arr[sorted_idx]

    # Initial coefficients
    if init is not None:
        if len(init) != nvar:
            raise ValueError("init must have length nvar")
        coef_init = np.array(init, dtype=float)
    else:
        coef_init = np.zeros(nvar, dtype=float)

    # Determine which columns to leave uncentered
    if nocenter is None:
        zero_one = np.zeros(nvar, dtype=bool)
    else:
        nocenter_set = set(nocenter)
        zero_one = np.array(
            [np.all(np.isin(x_sorted[:, j], list(nocenter_set))) for j in range(nvar)]
        )

    # Controls defaults
    if control is None:
        control = {}
    iter_max = int(control.get("iter_max", 0))
    eps = float(control.get("eps", 0.0))
    toler_chol = float(control.get("toler_chol", 0.0))
    toler_inf = float(control.get("toler_inf", 0.0))

    # --- Placeholder for the core exact‐fit routine ---
    # In R this is done via .C("Cagexact", ...)
    # Here you would implement or bind to a C/Fortran routine that solves
    # the exact partial‐likelihood equations, returning:
    #   coef, imat (flat), u, loglik[2], flag, sctest
    #
    # For now, raise to signal it's not implemented:
    raise NotImplementedError("Exact Cox‐fit algorithm (Cagexact) not yet implemented")

    # After implementing the fitting:
    #   var_matrix    = imat.reshape(nvar, nvar)
    #   coef          = ...
    #   loglik        = np.array([...])
    #   score_test    = ...
    #   means         = np.array([...])
    #   linear_pred   = x.dot(coef) + offset_arr - np.sum(coef * means)
    #   resid_vals    = ...  # call martingale residuals routine if resid=True
    #
    # Assemble and return:
    # return {
    #     "coefficients": coef,
    #     "var": var_matrix,
    #     "loglik": loglik,
    #     "score": score_test,
    #     "iter": iter_max,
    #     "linear_predictors": linear_pred,
    #     "residuals": resid_vals if resid else None,
    #     "means": means,
    #     "method": "coxph",
    # }
