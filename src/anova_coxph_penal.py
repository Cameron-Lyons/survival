from typing import List, Any
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats


def has_frailty(x):
    return hasattr(x, "pterms") and any(x.pterms == 2)


def anova_coxph_penal(object: Any, *args, test: str = "Chisq") -> pd.DataFrame:
    """
    Analysis of deviance for Cox proportional hazards models with penalized terms.

    This function provides an Analysis of Deviance table for one or more Cox model fits.
    For a single fitted model, it shows the sequential addition of terms; for multiple
    fitted models, it shows the analysis of deviance table comparing the models.

    Parameters
    ----------
    object : CoxPHFitter or similar object
        A fitted Cox proportional hazards model.
    *args : CoxPHFitter objects
        Additional fitted Cox models to be compared with the first model.
    test : str, default='Chisq'
        The type of test to perform. Currently only 'Chisq' is supported.

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table.

    Raises
    ------
    TypeError
        If the object is not a Cox model.
    ValueError
        If the model contains frailty terms or robust variances.

    Notes
    -----
    This is a Python port of the R function anova.coxph.penal.
    """
    if not hasattr(object, "model_type") or object.model_type != "cox":
        raise TypeError("argument must be a cox model")

    dot_args = list(args)

    if len(dot_args) > 0:
        is_cox_model = [
            hasattr(x, "model_type") and x.model_type == "cox" for x in dot_args
        ]
        is_coxme = [
            hasattr(x, "model_type") and x.model_type == "coxme" for x in dot_args
        ]

        if not all([a or b for a, b in zip(is_cox_model, is_coxme)]):
            raise TypeError("All arguments must be Cox models")

        if any(has_frailty(x) for x in dot_args):
            raise ValueError("anova command does not handle frailty terms")

        if any(is_coxme):
            raise NotImplementedError("coxme models are not fully supported yet")
        else:
            return anova_coxphlist([object] + dot_args, test=test)

    if hasattr(object, "rscore") and len(object.rscore) > 0:
        raise ValueError("Can't do anova tables with robust variances")

    if hasattr(object, "pterms") and any(object.pterms == 2):
        raise ValueError("anova command does not handle frailty terms")

    has_strata = hasattr(object, "strata") and object.strata is not None

    mf = object.X  # model frame equivalent
    Y = object.y  # response
    X = object.X  # design matrix

    assign = object.assign if hasattr(object, "assign") else np.arange(X.shape[1])

    if has_strata:
        strata_keep = object.strata
        strats = pd.factorize(strata_keep)[0]

    pname = []
    pindex = []
    if hasattr(object, "pterms") and hasattr(object, "pterms") > 0:
        pname = [
            name for name, val in zip(object.pterms.index, object.pterms) if val > 0
        ]
        term_labels = object.formula.split("+")
        pindex = [term_labels.index(name) for name in pname]

    alevels = sorted(set(assign))
    nmodel = len(alevels)

    df = np.zeros(nmodel + 1, dtype=int)
    loglik = np.zeros(nmodel + 1)

    df[nmodel] = (
        object.df_model
        if hasattr(object, "df_model")
        else np.sum(~np.isnan(object.params))
    )
    loglik[nmodel] = object.log_likelihood_

    df[0] = 0
    loglik[0] = object.log_likelihood_null_

    assign2 = [a for a in assign if a not in pindex]

    for i in range(nmodel - 1):
        j = [a for a in assign2 if a <= alevels[i]]

        X_subset = X.iloc[:, j] if len(j) > 0 else pd.DataFrame(index=X.index)

        for p in [p for p in pindex if p <= i]:
            if pname[p] in X.columns:
                X_subset[pname[p]] = X[pname[p]]

        tfit = CoxPHFitter()
        tfit.fit(pd.concat([X_subset, Y], axis=1), duration_col=Y.name)

        df[i + 1] = (
            tfit.df_model
            if hasattr(tfit, "df_model")
            else np.sum(~np.isnan(tfit.params))
        )
        loglik[i + 1] = tfit.log_likelihood_

    table = pd.DataFrame(
        {
            "loglik": loglik,
            "Chisq": [np.nan] + list(2 * np.diff(loglik)),
            "Df": [np.nan] + list(np.diff(df)),
        }
    )

    if len(test) > 0 and test[0] == "Chisq":
        table["Pr(>|Chi|)"] = [np.nan] + [
            1 - stats.chi2.cdf(c, d)
            for c, d in zip(table["Chisq"][1:], table["Df"][1:])
        ]

    if hasattr(object, "terms") and hasattr(object.terms, "term_labels"):
        row_names = ["NULL"] + object.terms.term_labels
    else:
        row_names = ["NULL"] + [f"term{i}" for i in range(1, nmodel + 1)]

    table.index = row_names[: len(table)]

    if hasattr(object, "terms") and hasattr(object.terms, "response"):
        response_name = object.terms.response
    else:
        response_name = Y.name if hasattr(Y, "name") else "response"

    title = f"Analysis of Deviance Table\n Cox model: response is {response_name}\nTerms added sequentially (first to last)"

    table.attrs["heading"] = title

    return table


def anova_coxphlist(models: List[Any], test: str = "Chisq") -> pd.DataFrame:
    """
    Analysis of deviance for comparing multiple Cox proportional hazards models.

    Parameters
    ----------
    models : List[CoxPHFitter or similar]
        A list of fitted Cox proportional hazards models.
    test : str, default='Chisq'
        The type of test to perform. Currently only 'Chisq' is supported.

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table comparing the models.

    Notes
    -----
    This is a Python port of the R function anova.coxphlist.
    """
    loglik = [model.log_likelihood_ for model in models]
    df = [
        (
            model.df_model
            if hasattr(model, "df_model")
            else np.sum(~np.isnan(model.params))
        )
        for model in models
    ]

    n_models = len(models)
    chisq = np.zeros(n_models)
    df_diff = np.zeros(n_models)

    for i in range(1, n_models):
        chisq[i] = 2 * (loglik[i] - loglik[i - 1])
        df_diff[i] = df[i] - df[i - 1]

    table = pd.DataFrame(
        {"loglik": loglik, "Df": df, "Chisq": chisq, "Df_diff": df_diff}
    )

    if test == "Chisq":
        table["Pr(>|Chi|)"] = [np.nan] + [
            1 - stats.chi2.cdf(c, d) for c, d in zip(chisq[1:], df_diff[1:])
        ]

    table.index = [f"Model {i+1}" for i in range(n_models)]

    title = "Analysis of Deviance Table\nSequential comparison of Cox models"
    table.attrs["heading"] = title

    return table
