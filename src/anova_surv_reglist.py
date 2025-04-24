from typing import List, Any
import numpy as np
import pandas as pd
from scipy import stats


def anova_survreglist(object: List[Any], *args, test: str = "Chisq") -> pd.DataFrame:
    """
    Analysis of deviance for a list of parametric survival regression models.

    This function compares multiple parametric survival regression models and
    generates an analysis of deviance table.

    Parameters
    ----------
    object : List[Any]
        A list of fitted parametric survival regression models.
    *args : Any
        Additional arguments (not used).
    test : str, default="Chisq"
        The type of test to perform. Options are "Chisq" or "none".

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table comparing the models.

    Raises
    ------
    ValueError
        If the first model has a different response from the rest.

    Notes
    -----
    This is a Python port of the R function anova.survreglist.
    """

    def diff_term(term_labels, i):
        """
        Helper function to determine the difference between term labels of two models.

        Parameters
        ----------
        term_labels : List[List[str]]
            List of term labels for two models.
        i : int
            Index for labeling purposes.

        Returns
        -------
        str
            String describing the difference between the models.
        """
        t1 = term_labels[0]
        t2 = term_labels[1]

        m1 = [t in t2 for t in t1]
        m2 = [t in t1 for t in t2]

        if all(m1):
            if all(m2):
                return "="
            else:
                return "+" + "+".join([t for t in t2 if t not in t1])
        else:
            if all(m2):
                return "-" + "-".join([t for t in t1 if t not in t2])
            else:
                return f"{i-1} vs. {i}"

    valid_tests = ["Chisq", "none"]
    if test not in valid_tests:
        raise ValueError(f"test must be one of {valid_tests}")

    rt = len(object)
    if rt == 1:
        return anova_survreg(object[0], test=test)

    forms = []
    for x in object:
        if hasattr(x, "formula"):
            forms.append(str(x.formula).split())
        elif hasattr(x, "formula_"):
            forms.append(str(x.formula_).split())
        else:
            forms.append(["Unknown", "~", "predictors"])

    response = forms[0][0] if len(forms[0]) > 0 else "response"
    subs = [
        forms[i][0] == response if len(forms[i]) > 0 else False
        for i in range(len(forms))
    ]

    if not all(subs):
        print(
            "Warning: Some fit objects deleted because response differs from the first model"
        )

    if sum(subs) == 1:
        raise ValueError("The first model has a different response from the rest")

    filtered_forms = [forms[i] for i in range(len(forms)) if subs[i]]
    filtered_object = [object[i] for i in range(len(object)) if subs[i]]

    dfres = []
    for x in filtered_object:
        if hasattr(x, "df_residual"):
            dfres.append(x.df_residual)
        elif hasattr(x, "df_resid"):
            dfres.append(x.df_resid)
        else:
            dfres.append(np.nan)

    m2loglik = []
    for x in filtered_object:
        if hasattr(x, "loglik"):
            loglik = (
                x.loglik[1]
                if isinstance(x.loglik, (list, tuple, np.ndarray))
                else x.loglik
            )
            m2loglik.append(-2 * loglik)
        else:
            m2loglik.append(np.nan)

    tl = []
    for x in filtered_object:
        if hasattr(x, "terms") and hasattr(x.terms, "term_labels"):
            tl.append(x.terms.term_labels)
        elif hasattr(x, "exog_names"):
            tl.append(x.exog_names)
        else:
            tl.append([])

    rt = len(m2loglik)
    effects = [""] * rt

    for i in range(1, rt):
        effects[i] = diff_term([tl[i - 1], tl[i]], i)

    dm2loglik = -np.diff(m2loglik)
    ddf = -np.diff(dfres)

    heading = ["Analysis of Deviance Table", f"\nResponse: {filtered_forms[0][0]}\n"]

    terms = []
    for form in filtered_forms:
        if len(form) > 2:
            terms.append(form[2])
        else:
            terms.append("~")

    aod = pd.DataFrame(
        {
            "Terms": terms,
            "Resid. Df": dfres,
            "-2*LL": m2loglik,
            "Test": effects,
            "Df": [np.nan] + list(ddf),
            "Deviance": [np.nan] + list(dm2loglik),
        }
    )

    aod.attrs["heading"] = heading

    if test != "none":
        n = (
            len(filtered_object[0].residuals)
            if hasattr(filtered_object[0], "residuals")
            else 0
        )

        o = np.argsort(dfres)

        if test == "Chisq":
            scale = 1
        else:
            residuals = (
                filtered_object[0].residuals
                if hasattr(filtered_object[0], "residuals")
                else []
            )
            scale = (
                np.sum(np.square(residuals)) / dfres[o[0]] if len(residuals) > 0 else 1
            )

        return stat_anova(aod, test, scale, dfres[o[0]], n)

    return aod


def stat_anova(
    aod: pd.DataFrame, test: str, scale: float = 1, df0: float = 0, n: int = 0
) -> pd.DataFrame:
    """
    Add statistical tests to an analysis of deviance table.

    Parameters
    ----------
    aod : pd.DataFrame
        Analysis of deviance data frame.
    test : str
        Type of test to perform.
    scale : float, default=1
        Scale parameter.
    df0 : float, default=0
        Base degrees of freedom.
    n : int, default=0
        Number of observations.

    Returns
    -------
    pd.DataFrame
        Analysis of deviance data frame with additional test statistics.

    Notes
    -----
    This is a Python port of the R function stat.anova.
    """
    if test == "Chisq":
        p_values = [np.nan]
        for i in range(1, len(aod)):
            if not np.isnan(aod["Deviance"].iloc[i]) and not np.isnan(
                aod["Df"].iloc[i]
            ):
                p_values.append(
                    1 - stats.chi2.cdf(aod["Deviance"].iloc[i], aod["Df"].iloc[i])
                )
            else:
                p_values.append(np.nan)

        aod["Pr(>Chi)"] = p_values
    elif test == "F":
        p_values = [np.nan]
        f_values = [np.nan]
        for i in range(1, len(aod)):
            if not np.isnan(aod["Deviance"].iloc[i]) and not np.isnan(
                aod["Df"].iloc[i]
            ):
                f_val = (aod["Deviance"].iloc[i] / aod["Df"].iloc[i]) / scale
                f_values.append(f_val)
                p_values.append(1 - stats.f.cdf(f_val, aod["Df"].iloc[i], df0))
            else:
                f_values.append(np.nan)
                p_values.append(np.nan)

        aod["F"] = f_values
        aod["Pr(>F)"] = p_values

    return aod


def anova_survreg(object: Any, *args, test: str = "Chisq") -> pd.DataFrame:
    """
    Analysis of deviance for parametric survival regression models.

    This function is a placeholder for the anova_survreg function that would be
    implemented separately. It's referenced here for completeness.

    Parameters
    ----------
    object : Any
        A fitted parametric survival regression model.
    *args : Any
        Additional arguments.
    test : str, default="Chisq"
        The type of test to perform.

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table.

    Notes
    -----
    This is just a stub function. The actual implementation would be in a separate file.
    """
    raise NotImplementedError("anova_survreg function is not implemented in this file")
