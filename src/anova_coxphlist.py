from typing import List, Any
import numpy as np
import pandas as pd
from scipy import stats


def anova_coxphlist(object: List[Any], test: str = "Chisq", **kwargs) -> pd.DataFrame:
    """
    Analysis of deviance for a list of Cox proportional hazards models.

    This function compares multiple Cox proportional hazards models and generates
    an analysis of deviance table. It is typically called from anova_coxph,
    not directly by users.

    Parameters
    ----------
    object : List[CoxPHFitter or similar]
        A list of fitted Cox proportional hazards models.
    test : str, default='Chisq'
        The type of test to perform. Currently only 'Chisq' is supported.
    **kwargs : dict
        Additional arguments (not currently used).

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table comparing the models.

    Raises
    ------
    TypeError
        If the first argument is not a list or if any object in the list is not a Cox model.
    ValueError
        If any model has robust variances, different tie options, different response variables,
        different dataset sizes, or different strata.

    Notes
    -----
    This is a Python port of the R function anova.coxphlist.
    """
    if not isinstance(object, list):
        raise TypeError("First argument must be a list")

    is_coxmodel = [hasattr(x, "model_type") and x.model_type == "cox" for x in object]
    if not all(is_coxmodel):
        raise TypeError("All arguments must be Cox models")

    is_robust = [hasattr(x, "rscore") and x.rscore is not None for x in object]
    if any(is_robust):
        raise ValueError("Can't do anova tables with robust variances")

    ties = [getattr(x, "method", None) for x in object]
    if any(tie != ties[0] for tie in ties):
        raise ValueError("all models must have the same ties option")

    responses = [
        getattr(x, "response_name", None)
        or (x.y.name if hasattr(x, "y") and hasattr(x.y, "name") else "response")
        for x in object
    ]
    sameresp = [resp == responses[0] for resp in responses]

    if not all(sameresp):
        object = [obj for obj, same in zip(object, sameresp) if same]
        non_matching = [resp for resp, same in zip(responses, sameresp) if not same]
        print(
            f"Warning: Models with response {non_matching} removed because response differs from model 1"
        )

    ns = [len(x.residuals) if hasattr(x, "residuals") else len(x.y) for x in object]
    if any(n != ns[0] for n in ns):
        raise ValueError("models were not all fit to the same size of dataset")

    def get_strata_vars(x):
        if hasattr(x, "strata") and x.strata is not None:
            return x.strata
        return []

    strata_vars = [get_strata_vars(x) for x in object]
    has_strata = any(len(vars) > 0 for vars in strata_vars)

    if has_strata:
        strata_match = all(
            (
                (vars == strata_vars[0]).all()
                if hasattr(vars, "all")
                else vars == strata_vars[0]
            )
            for vars in strata_vars
        )
        if not strata_match:
            raise ValueError("models do not have the same strata")

    nmodels = len(object)
    if nmodels == 1:
        from .anova_coxph import anova_coxph  # This would be the equivalent function

        return anova_coxph(object[0], test=test)

    loglik = [
        model.log_likelihood_ if hasattr(model, "log_likelihood_") else model.loglik[-1]
        for model in object
    ]

    df = []
    for x in object:
        if hasattr(x, "df") and x.df is not None:
            df.append(sum(x.df))
        elif not hasattr(x, "params") or x.params is None:
            df.append(0)
        else:
            df.append(sum(~np.isnan(x.params)))

    table = pd.DataFrame(
        {
            "loglik": loglik,
            "Chisq": [np.nan] + list(abs(2 * np.diff(loglik))),
            "Df": [np.nan] + list(abs(np.diff(df))),
        }
    )

    def get_formula_string(x):
        if hasattr(x, "formula"):
            return str(x.formula)
        elif hasattr(x, "formula_"):
            return str(x.formula_)
        else:
            return "formula not available"

    variables = [get_formula_string(x) for x in object]

    table.index = range(1, nmodels + 1)

    title = f"Analysis of Deviance Table\n Cox model: response is {responses[0]}"
    topnote = "\n".join(
        [f" Model {i}: {var}" for i, var in zip(range(1, nmodels + 1), variables)]
    )

    if test is not None:
        table["Pr(>|Chi|)"] = [np.nan] + [
            1 - stats.chi2.cdf(c, d)
            for c, d in zip(table["Chisq"][1:], table["Df"][1:])
        ]

    table.attrs["heading"] = [title, topnote]

    return table
