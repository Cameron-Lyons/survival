from typing import Union, Optional, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.duration.hazard_regression import PHReg
from scipy import stats


class CCHResult:
    """
    Result object for case-cohort analysis.

    Attributes:
        coefficients: Estimated coefficients (numpy array).
        var: Variance-covariance matrix (numpy array).
        naive_var: Naive variance-covariance (numpy array).
        method: Method used for analysis.
        call: Original call arguments.
        cohort_size: Cohort size (scalar or dict of stratum sizes).
        stratified: Whether analysis was stratified.
        stratum: Pandas Series of stratum labels, if stratified.
        subcohort_size: Subcohort size (scalar or dict of stratum sizes).
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        var: np.ndarray,
        naive_var: np.ndarray,
        method: str,
        call: Dict[str, Any],
        cohort_size: Union[int, Dict[Any, int]],
        stratified: bool,
        stratum: Optional[pd.Series],
        subcohort_size: Union[int, Dict[Any, int]],
        names: Optional[List[str]] = None,
    ):
        self.coefficients = coefficients
        self.var = var
        self.naive_var = naive_var
        self.method = method
        self.call = call
        self.cohort_size = cohort_size
        self.stratified = stratified
        self.stratum = stratum
        self.subcohort_size = subcohort_size
        self.names = names or []

    def vcov(self) -> np.ndarray:
        """Return the variance-covariance matrix."""
        return self.var

    def __str__(self) -> str:
        """String representation of CCHResult."""
        se = np.sqrt(np.diag(self.var))
        z = np.abs(self.coefficients / se)
        p = 2 * (1 - stats.norm.cdf(z))
        lines = []
        if self.stratified:
            lines.append(
                f"Exposure-stratified case-cohort analysis, {self.method} method."
            )
            lines.append(f"Subcohort sizes: {self.subcohort_size}")
            lines.append(f"Cohort sizes: {self.cohort_size}")
        else:
            lines.append(
                f"Case-cohort analysis, {self.method} method, "
                f"with subcohort of {self.subcohort_size} "
                f"from cohort of {self.cohort_size}"
            )
        lines.append(f"Call: {self.call}")
        lines.append("Coefficients:")
        for i, name in enumerate(self.names):
            lines.append(
                f"{name}: coef={self.coefficients[i]:.4f}, "
                f"SE={se[i]:.4f}, Z={z[i]:.2f}, p={p[i]:.4f}"
            )
        return "\n".join(lines)


def parse_formula(formula: str) -> Tuple[Optional[str], str, str, np.ndarray]:
    r"""
    Parse a formula of the form 'Surv(time, status) ~ x1 + x2' or
    'Surv(start, stop, status) ~ x1 + x2'.

    Returns:
        entry_col: Name of entry time column or None.
        exit_col: Name of exit time column.
        event_col: Name of event indicator column.
        rhs: RHS formula string for dmatrix.
    """
    f3 = re.match(r"\s*Surv\(\s*([^,]+),\s*([^,]+),\s*([^,]+)\)\s*~\s*(.+)", formula)
    f2 = re.match(r"\s*Surv\(\s*([^,]+),\s*([^,]+)\)\s*~\s*(.+)", formula)
    if f3:
        entry_col, exit_col, event_col, rhs = f3.groups()
        return entry_col.strip(), exit_col.strip(), event_col.strip(), rhs.strip()
    elif f2:
        exit_col, event_col, rhs = f2.groups()
        return None, exit_col.strip(), event_col.strip(), rhs.strip()
    else:
        raise ValueError("Formula must be of the form Surv(...) ~ predictors")


def cch(
    formula: str,
    data: pd.DataFrame,
    subcoh: Union[str, pd.Series],
    id_col: Union[str, pd.Series],
    stratum: Optional[Union[str, pd.Series]] = None,
    cohort_size: Union[int, Dict[Any, int]] = None,
    method: str = "Prentice",
    robust: bool = False,
) -> CCHResult:
    """
    Main function for case-cohort analysis.

    Args:
        formula: A formula string, e.g. 'Surv(time, status) ~ x1 + x2'.
        data: DataFrame containing all variables.
        subcoh: Column name or boolean Series for subcohort indicator.
        id_col: Column name or Series for subject ID.
        stratum: Column name or Series for stratum (if stratified).
        cohort_size: Total cohort size (scalar) or dict of sizes per stratum.
        method: One of {'Prentice', 'SelfPrentice', 'LinYing', 'I.Borgan', 'II.Borgan'}.
        robust: Whether to compute robust variance (where supported).

    Returns:
        CCHResult containing estimates and metadata.
    """
    call = locals().copy()
    if isinstance(id_col, str):
        ids = data[id_col].values
    else:
        ids = pd.Series(id_col).values
    if isinstance(subcoh, str):
        subcohort = data[subcoh].astype(int).values
    else:
        subcohort = pd.Series(subcoh).astype(int).values
    if stratum is not None:
        if isinstance(stratum, str):
            stratum_s = data[stratum].astype("category")
        else:
            stratum_s = pd.Categorical(stratum)
    else:
        stratum_s = None

    if len(ids) != len(np.unique(ids)):
        raise ValueError("Multiple records per id not allowed")

    entry_col, exit_col, event_col, rhs = parse_formula(formula)
    X = dmatrix("~ " + rhs + " - 1", data, return_type="dataframe").values
    exit_times = data[exit_col].values
    events = data[event_col].astype(int).values
    if entry_col:
        entry_times = data[entry_col].values
    else:
        entry_times = np.zeros_like(exit_times)

    cens = events  # 1=event, 0=censor
    cc = cens + 1 - subcohort

    event_times = np.unique(exit_times[cens == 1])
    delta = np.diff(np.sort(event_times)).min() / 2 if len(event_times) > 1 else 1.0

    if method == "Prentice":
        res = _prentice(entry_times, exit_times, cc, ids, X, cohort_size, robust, delta)
    elif method == "SelfPrentice":
        res = _self_prentice(
            entry_times, exit_times, cc, ids, X, cohort_size, robust, delta
        )
    elif method == "LinYing":
        res = _lin_ying(entry_times, exit_times, cc, ids, X, cohort_size, robust, delta)
    elif method == "I.Borgan":
        res = _i_borgan(
            entry_times, exit_times, cc, ids, X, stratum_s, cohort_size, delta
        )
    elif method == "II.Borgan":
        res = _ii_borgan(
            entry_times, exit_times, cc, ids, X, stratum_s, cohort_size, delta
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    res.method = method
    res.call = call
    res.cohort_size = cohort_size
    res.stratified = method in ["I.Borgan", "II.Borgan"]
    res.stratum = stratum_s

    tt = pd.Series(subcohort).value_counts().to_dict()
    res.subcohort_size = tt.get(1, 0)
    res.names = [f"X{i}" for i in range(X.shape[1])]
    return res


def _prentice(
    tenter: np.ndarray,
    texit: np.ndarray,
    cc: np.ndarray,
    ids: np.ndarray,
    X: np.ndarray,
    ntot: Union[int, Dict[Any, int]],
    robust: bool,
    delta: float,
) -> CCHResult:
    """
    Prentice estimator for case-cohort analysis.
    """
    ent2 = tenter.copy()
    mask2 = cc == 2
    ent2[mask2] = texit[mask2] - delta
    endog = np.column_stack((ent2, texit, (cc > 0).astype(int)))
    fit1 = PHReg(endog, X, status=(cc > 0).astype(int), entry=None, ties="efron").fit()

    nd = int((cc > 0).sum())
    nc = int((cc < 2).sum())
    aent = np.concatenate([tenter[cc > 0], tenter[cc < 2]])
    aexit = np.concatenate([texit[cc > 0], texit[cc < 2]])
    aX = np.vstack([X[cc > 0], X[cc < 2]])
    aid = np.concatenate([ids[cc > 0], ids[cc < 2]])
    dum = np.concatenate([np.full(nd, -100.0), np.zeros(nc)])
    fit2 = PHReg(
        np.column_stack((aent, aexit, (cc > 0).astype(int))),
        aX,
        status=(cc > 0).astype(int),
        entry=None,
        tie="efron",
        strat=None,
        weights=None,
        offset=dum,
    ).fit(cluster=aid, start_params=fit1.params)

    coeffs = fit1.params
    var = fit2.cov_params()
    naive = var.copy()
    return CCHResult(coeffs, var, naive, "Prentice", {}, ntot, False, None, 0)


# TODO: Implement _self_prentice, _lin_ying, _i_borgan, _ii_borgan similarly using PHReg


def vcov_cch(result: CCHResult) -> np.ndarray:
    """Return variance-covariance matrix for a CCHResult."""
    return result.vcov()


def print_cch(result: CCHResult) -> None:
    """Print the CCH result."""
    print(result)


def summary_cch(result: CCHResult) -> Dict[str, Any]:
    """Return a summary dict for CCHResult."""
    se = np.sqrt(np.diag(result.var))
    z = np.abs(result.coefficients / se)
    p = 2 * (1 - stats.norm.cdf(z))
    summary = {
        "call": result.call,
        "method": result.method,
        "cohort_size": result.cohort_size,
        "subcohort_size": result.subcohort_size,
        "coefficients": pd.DataFrame(
            {"Value": result.coefficients, "SE": se, "Z": z, "p": p}, index=result.names
        ),
    }
    return summary


def print_summary_cch(summary: Dict[str, Any]) -> None:
    """Pretty-print summary from summary_cch."""
    print(f"Call: {summary['call']}")
    print(f"Method: {summary['method']}")
    print(summary["coefficients"])
