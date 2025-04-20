from typing import Optional, Union, List, Tuple
import re

import pandas as pd
import numpy as np
from patsy import dmatrix
from lifelines import AalenAdditiveFitter


def _parse_surv_formula(formula: str) -> Tuple[str, str, str]:
    """
    Parse an R-style Surv() formula of the form
    "Surv(time, event) ~ covariate1 + covariate2 + ...".

    Returns:
        time_col: name of the duration column
        event_col: name of the event indicator column
        covariate_formula: RHS of the formula for covariates
    """
    pattern = r"Surv\(\s*(?P<time>[^,]+)\s*,\s*(?P<event>[^)]+)\s*\)\s*~\s*(?P<cov>.+)"
    m = re.match(pattern, formula)
    if not m:
        raise ValueError("Formula must be of the form Surv(time, event) ~ covariates")
    return (
        m.group("time").strip(),
        m.group("event").strip(),
        m.group("cov").strip(),
    )


def aareg(
    formula: str,
    data: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    subset: Optional[Union[List[int], pd.Series]] = None,
    na_action: Optional[str] = None,
    qrtol: float = 1e-7,
    nmin: Optional[int] = None,
    dfbeta: bool = False,
    taper: int = 1,
    test: str = "aalen",
    cluster: Optional[str] = None,
    model: bool = False,
    x: bool = False,
    y: bool = False,
) -> AalenAdditiveFitter:
    """
    Fit Aalen's additive regression model to survival data.

    Parameters
    ----------
    formula
        An R-style formula string, e.g. "Surv(time, event) ~ age + chol".
    data
        A pandas DataFrame containing all variables referenced in the formula.
    weights
        Optional array of observation weights (same length as `data`).
    subset
        Optional list of row indices or boolean mask to select a subset.
    na_action
        How to handle missing values: e.g. "drop" to drop any NA rows.
    qrtol
        Tolerance for QR decomposition (passed to the fitter).
    nmin
        Minimum number at risk to include a time point (not used directly here).
    dfbeta
        If True, compute dfbeta residuals (not implemented; placeholder).
    taper
        Taper parameter for variance smoothing (not implemented here).
    test
        Which test statistic to compute: "aalen", "variance", or "nrisk"
        (not used directly; placeholder).
    cluster
        Column name in `data` for clustering (not supported; placeholder).
    model
        If True, return the full model frame in the result (not supported).
    x
        If True, include the design matrix in the result (not supported).
    y
        If True, include the response array in the result (not supported).

    Returns
    -------
    AalenAdditiveFitter
        A fitted lifelines AalenAdditiveFitter object.
    """
    # Subset rows if requested
    if subset is not None:
        data = data.loc[subset]

    # Handle missing values
    if na_action == "drop":
        data = data.dropna()

    # Parse the Surv() formula
    time_col, event_col, covariate_formula = _parse_surv_formula(formula)

    # Build design matrix for covariates
    X = dmatrix(covariate_formula, data, return_type="dataframe")

    # Extract durations and event indicators
    durations = data[time_col].to_numpy()
    events = data[event_col].to_numpy().astype(int)

    # Instantiate and fit the model
    fitter = AalenAdditiveFitter(coef_penalizer=0.0, fit_intercept=True, tol=qrtol)
    fitter.fit(
        X,
        duration_col=durations,
        event_col=events,
        weights=weights,
        show_progress=False,
    )

    return fitter
