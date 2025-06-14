from typing import Callable, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


class CoxZphResult:
    """
    Output object for cox_zph test, mimicking R's behavior but in Python.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        x: np.ndarray,
        time: np.ndarray,
        y: np.ndarray,
        var: np.ndarray,
        strata: Optional[np.ndarray] = None,
        transform: str = "",
        call: Optional[dict] = None,
    ):
        self.table = table
        self.x = x
        self.time = time
        self.y = y
        self.var = var
        self.strata = strata
        self.transform = transform
        self.call = call


def cox_zph(
    fit: Any,
    transform: Union[str, Callable[[np.ndarray], np.ndarray]] = "km",
    terms: bool = True,
    singledf: bool = False,
    global_test: bool = True,
) -> CoxZphResult:
    """
    Test the proportional hazards assumption for a Cox model using scaled Schoenfeld residuals.

    Args:
        fit: A fitted Cox proportional hazards model object from `lifelines` or `statsmodels`.
        transform: A string or callable specifying how to transform survival times ('identity', 'rank', 'log', 'km').
        terms: If True, test proportionality individually for each term.
        singledf: If True, collapse multi-df terms into test with 1 degree of freedom.
        global_test: Include global test of proportionality.

    Returns:
        CoxZphResult: Object containing test results, residuals, times, and variance information.

    Raises:
        ValueError: If the fit object is not from the correct type.

    Notes:
        - The actual implementation of low-level internals, like calculation of Schoenfeld residuals,
          must come from `lifelines`, `statsmodels`, or custom code.
        - Real data handling and advanced cases (frailty, penalization, L2/L1, etc.) need more adaptation.
    """
    if not hasattr(fit, "compute_residuals"):
        raise ValueError(
            "Argument must be a fitted Cox proportional hazards model object."
        )

    times = fit.durations.values
    event = fit.event_observed.values
    X = fit.data.values
    fcoef = fit.params_.values
    varnames = fit.params_.index.tolist()
    nvar = len(varnames)
    n = len(times)

    istrat = np.zeros(n, dtype=int)

    if isinstance(transform, str):
        tname = transform
        if transform == "identity":
            ttimes = times
        elif transform == "rank":
            ttimes = pd.Series(times).rank().values
        elif transform == "log":
            ttimes = np.log(times)
        elif transform == "km":
            from lifelines import KaplanMeierFitter

            kmf = KaplanMeierFitter().fit(times, event_observed=event)
            surv = kmf.survival_function_at_times(times).values
            ttimes = 1 - surv
        else:
            raise ValueError("Unrecognized transform")
    elif callable(transform):
        tname = transform.__name__
        ttimes = transform(times)
    else:
        raise ValueError("transform must be a string or callable")

    gtime = ttimes - np.mean(ttimes[event == 1])

    sresid = fit.compute_residuals("schoenfeld")

    chisq = []
    dfs = []
    pvals = []

    for col in varnames:
        y = sresid[col].values
        t = gtime[sresid.index]
        slope, residuals, _, _, _ = np.linalg.lstsq(
            np.vstack([t, np.ones_like(t)]).T, y, rcond=None
        )
        std_err = np.sqrt(residuals / (len(t) - 2)) / np.std(t)
        test_stat = (slope[0] / std_err) ** 2 if std_err > 0 else 0.0
        pval = 1 - stats.chi2.cdf(test_stat, 1)
        chisq.append(test_stat)
        dfs.append(1)
        pvals.append(pval)

    if global_test:
        global_chisq = sum(chisq)
        global_df = sum(dfs)
        global_pval = 1 - stats.chi2.cdf(global_chisq, global_df)
        chisq.append(global_chisq)
        dfs.append(global_df)
        pvals.append(global_pval)
        row_names = varnames + ["GLOBAL"]
    else:
        row_names = varnames

    tbl = pd.DataFrame({"chisq": chisq, "df": dfs, "p": pvals}, index=row_names)

    return CoxZphResult(
        table=tbl,
        x=gtime[sresid.index],
        time=times[sresid.index],
        y=sresid.values,
        var=None,
        strata=None if np.all(istrat == 0) else istrat[sresid.index],
        transform=tname,
        call=None,
    )
