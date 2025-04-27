"""
Module to compute baseline cumulative hazard for Cox proportional hazards models.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import math


class CoxPHModel:
    """
    Placeholder for a fitted Cox proportional hazards model.
    Attributes:
    -----------
    coef_ : List[float]
        Estimated regression coefficients.
    means_ : List[float]
        Mean values of each covariate in the training data.
    is_multi_state : bool
        Whether this is a multi‐state model.
    is_coxph : bool
        Marker that this is a Cox PH model.
    """

    def __init__(
        self, coef_: List[float], means_: List[float], is_multi_state: bool = False
    ):
        self.coef_ = coef_
        self.means_ = means_
        self.is_multi_state = is_multi_state
        self.is_coxph = True


class SurvFitResult:
    """
    Result of a `survfit` call.
    Attributes:
    -----------
    cumhaz : np.ndarray
        Cumulative hazard estimates.
    time : np.ndarray
        Time points corresponding to the hazards.
    strata : Optional[Dict[str, int]]
        If the model was stratified, a mapping from stratum names to
        the number of time‐points in each stratum.
    """

    def __init__(
        self,
        cumhaz: np.ndarray,
        time: np.ndarray,
        strata: Optional[Dict[str, int]] = None,
    ):
        self.cumhaz = cumhaz
        self.time = time
        self.strata = strata


def survfit(
    fit: CoxPHModel, newdata: Optional[pd.DataFrame] = None, se_fit: bool = True
) -> SurvFitResult:
    """
    Placeholder for the `survfit` function, which must return a SurvFitResult.
    """
    raise NotImplementedError("survfit() must be implemented separately")


def basehaz(
    fit: CoxPHModel, newdata: Optional[pd.DataFrame] = None, centered: bool = True
) -> pd.DataFrame:
    """
    Compute the baseline cumulative hazard for a Cox proportional hazards model.

    This function mirrors the behavior of the R `basehaz()` alias for `survfit()`,
    commonly used to extract the "baseline hazard" from a `coxph` object.

    Parameters
    ----------
    fit : CoxPHModel
        A fitted Cox proportional hazards model (must not be multi‐state).
    newdata : pd.DataFrame, optional
        If provided, should contain exactly one new observation for which to
        compute the cumulative hazard.
    centered : bool, default=True
        When `newdata` is None, if `centered=False`, adjusts the baseline
        cumulative hazard by centering at the mean covariate values.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - "hazard": the cumulative hazard estimates
        - "time": corresponding time points
        - "strata" (optional): stratification labels as a pd.Categorical

    Raises
    ------
    ValueError
        If `fit` is multi‐state or `newdata` has not exactly one row.
    TypeError
        If `fit` is not a CoxPHModel instance.
    """
    if getattr(fit, "is_multi_state", False):
        raise ValueError(
            "the basehaz function is not implemented for multi‐state models"
        )
    if not getattr(fit, "is_coxph", False):
        raise TypeError("must be a CoxPHModel object")

    if newdata is not None:
        if newdata.shape[0] != 1:
            raise ValueError("newdata must have exactly one row")
        sfit = survfit(fit, newdata=newdata, se_fit=False)
        chaz = sfit.cumhaz
    else:
        sfit = survfit(fit, se_fit=False)
        chaz = sfit.cumhaz
        if not centered:
            offset = sum(m * coef for m, coef in zip(fit.means_, fit.coef_))
            chaz = chaz * math.exp(-offset)

    df = pd.DataFrame({"hazard": chaz, "time": sfit.time})

    if sfit.strata is not None:
        levels = list(sfit.strata.keys())
        labels: List[str] = []
        for name, count in sfit.strata.items():
            labels.extend([name] * count)
        df["strata"] = pd.Categorical(labels, categories=levels)

    return df
