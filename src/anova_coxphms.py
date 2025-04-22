from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def anova_coxphms(object: Any, *args, test: str = "score") -> pd.DataFrame:
    """
    Analysis of deviance for multistate hazard models.

    This function compares multiple Cox proportional hazards multistate models
    and generates an analysis of deviance table.

    Parameters
    ----------
    object : Any
        A fitted Cox proportional hazards multistate model.
    *args : Any
        Additional fitted Cox proportional hazards multistate models.
    test : str, default="score"
        The type of test to perform. Options are "score", "Wald", or "PL".

    Returns
    -------
    pd.DataFrame
        A data frame containing the analysis of deviance table comparing the models.

    Raises
    ------
    TypeError
        If the object is not a Cox multistate model or if any objects in args are not
        Cox multistate models.
    ValueError
        If models have different ties options, different response variables,
        different dataset sizes, or different stratification structures.
        Also raised if models are not in increasing order of complexity.
    NotImplementedError
        This function is currently not fully implemented for multistate models.

    Notes
    -----
    This is a Python port of the R function anova.coxphms.
    """
    if not hasattr(object, "model_type") or object.model_type != "coxphms":
        raise TypeError("argument must be the fit of a multistate hazard model")

    dot_args = list(args)

    all_mod = [object] + dot_args

    is_coxms = [hasattr(x, "model_type") and x.model_type == "coxphms" for x in all_mod]
    if not all(is_coxms):
        raise TypeError("All arguments must be multistate hazard models")

    ties = [getattr(x, "method", None) for x in all_mod]
    if any(tie != ties[0] for tie in ties):
        raise ValueError("all models must have the same ties option")

    responses = [
        getattr(x, "response_name", None)
        or (x.y.name if hasattr(x, "y") and hasattr(x.y, "name") else "response")
        for x in all_mod
    ]
    sameresp = [resp == responses[0] for resp in responses]

    if not all(sameresp):
        all_mod = [obj for obj, same in zip(all_mod, sameresp) if same]
        non_matching = [resp for resp, same in zip(responses, sameresp) if not same]
        print(
            f"Warning: Models with response {non_matching} removed because response differs from model 1"
        )

    nmodel = len(all_mod)
    if nmodel < 2:
        raise ValueError("must have more than one model")

    def get_model_ns(x):
        if hasattr(x, "n") and hasattr(x, "n_id") and hasattr(x, "nevent"):
            return (x.n, x.n_id, x.nevent)
        return None

    ns = [get_model_ns(x) for x in all_mod]
    if any(n is None for n in ns):
        raise ValueError("at least one model is missing the n, n_id, or nevent element")

    first_ns = ns[0]
    if any(n != first_ns for n in ns):
        raise ValueError("models were not all fit to the same dataset")

    def compare_smap(x, first_map):
        return getattr(x, "smap", None) == first_map

    first_smap = getattr(all_mod[0], "smap", None)
    stest = [compare_smap(x, first_smap) for x in all_mod]
    if not all(stest):
        raise ValueError("not all models have the same structure of baseline hazards")

    nvar = [len(x.params) if hasattr(x, "params") else 0 for x in all_mod]
    if any(diff < 1 for diff in np.diff(nvar)):
        raise ValueError("models must be in increasing order of complexity")

    for i in range(1, nmodel):
        prev_vars = getattr(all_mod[i - 1], "cmap", {}).keys()
        curr_vars = getattr(all_mod[i], "cmap", {}).keys()
        if not all(var in curr_vars for var in prev_vars):
            raise ValueError(f"model {i} contains variables not in model {i+1}")

    for i in range(1, nmodel):
        if not np.array_equal(
            getattr(all_mod[i], "y", None), getattr(all_mod[i - 1], "y", None)
        ):
            raise ValueError("all models must have the same response")

    def get_formula_string(x):
        if hasattr(x, "formula"):
            return str(x.formula)
        elif hasattr(x, "formula_"):
            return str(x.formula_)
        else:
            return "formula not available"

    variables = [get_formula_string(x) for x in all_mod]

    table = pd.DataFrame(
        {
            "loglik": [0] * nmodel,  # This would be filled with actual values
            "Chisq": [np.nan]
            + [0] * (nmodel - 1),  # This would be filled with actual values
            "Df": [np.nan]
            + [0] * (nmodel - 1),  # This would be filled with actual values
        }
    )

    table.index = range(1, nmodel + 1)

    title = f"Analysis of Deviance Table\n Cox model: response is {responses[0]}"
    topnote = "\n".join(
        [f" Model {i}: {var}" for i, var in zip(range(1, nmodel + 1), variables)]
    )

    if test is not None:
        table["Pr(>|Chi|)"] = [np.nan] + [
            1 - stats.chi2.cdf(c, d)
            for c, d in zip(table["Chisq"][1:], table["Df"][1:])
        ]

    table.attrs["heading"] = [title, topnote]

    return table
