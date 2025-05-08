from typing import Optional, Union
import re
import warnings

import pandas as pd
from patsy import dmatrices
from lifelines import CoxPHFitter


def clogit(
    formula: str,
    data: pd.DataFrame,
    weights: Optional[Union[str, pd.Series]] = None,
    subset: Optional[Union[pd.Series, list, slice]] = None,
    na_action: str = "na.pass",
    method: str = "breslow",
    **fit_kwargs
) -> CoxPHFitter:
    """
    Fit a conditional logistic regression model via a Cox proportional hazards formulation.

    Parameters
    ----------
    formula : str
        A patsy-style formula for the model. Use `strata(var)` to specify matching strata,
        for example: 'case ~ exposure + strata(matching)'.
    data : pd.DataFrame
        Data containing all variables referenced in `formula`.
    weights : str or pd.Series, optional
        Column name or array of case weights. Ignored if `method='exact'`.
    subset : array-like or boolean mask, optional
        Rows of `data` to include in the fit.
    na_action : {'na.pass', 'na.drop'}, default 'na.pass'
        How to handle missing data. 'na.pass' leaves NAs in place, 'na.drop' removes any row
        with missing values.
    method : {'exact', 'approximate', 'efron', 'breslow'}, default 'breslow'
        Method for handling tied events. 'approximate' maps to 'breslow'.
    **fit_kwargs
        Additional keyword arguments forwarded to `CoxPHFitter.fit`.

    Returns
    -------
    CoxPHFitter
        A fitted Cox proportional hazards model representing the conditional logistic regression.

    Raises
    ------
    ValueError
        If `formula` is missing a '~' or if incompatible options are specified.
    """
    if "~" not in formula:
        raise ValueError("A formula argument is required (must contain '~').")

    df = data.copy()
    if subset is not None:
        df = df.loc[subset]

    if na_action == "na.drop":
        df = df.dropna()

    strata_match = re.search(r"strata\((.*?)\)", formula)
    if strata_match:
        strata_var = strata_match.group(1).strip()
        formula_nost = re.sub(r"\+?\s*strata\([^)]*\)", "", formula)
    else:
        strata_var = None
        formula_nost = formula

    method_map = {
        "exact": "exact",
        "approximate": "breslow",
        "efron": "efron",
        "breslow": "breslow",
    }
    key = method if method in method_map else "breslow"
    ties = method_map[key]

    y, X = dmatrices(formula_nost, data=df, return_type="dataframe")
    event_col = y.columns[-1]  # response variable name

    df_fit = pd.concat([X, df[event_col].rename("event")], axis=1)
    df_fit["duration"] = 1  # constant time for all observations

    cph = CoxPHFitter(ties=ties)

    fit_args = {"duration_col": "duration", "event_col": "event"}
    if strata_var:
        fit_args["strata"] = strata_var
    if weights is not None:
        if key == "exact":
            warnings.warn("weights ignored: not possible for the exact method")
        else:
            fit_args["weights_col"] = weights

    cph.fit(df_fit, **fit_args, **fit_kwargs)
    return cph


def print_clogit(model: CoxPHFitter) -> None:
    """
    Print the summary of a fitted conditional logistic regression model.

    Parameters
    ----------
    model : CoxPHFitter
        A fitted model returned by `clogit`.
    """
    print(model.summary)


def survfit_clogit(*args, **kwargs) -> None:
    """
    Predicted survival curves are not defined for a clogit model.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Predicted survival curves are not defined for a clogit model."
    )
