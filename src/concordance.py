from typing import Optional, Sequence, Union, Dict, Tuple, List
import numpy as np
import pandas as pd


class ConcordanceResult:
    """
    Data class to store the result of the concordance computation.
    """

    def __init__(
        self,
        concordance: float,
        n: int,
        count: np.ndarray,
        var: Optional[float] = None,
        cvar: Optional[float] = None,
        dfbeta: Optional[np.ndarray] = None,
        influence: Optional[np.ndarray] = None,
        ranks: Optional[pd.DataFrame] = None,
    ):
        self.concordance = concordance
        self.n = n
        self.count = count
        self.var = var
        self.cvar = cvar
        self.dfbeta = dfbeta
        self.influence = influence
        self.ranks = ranks

    def __str__(self) -> str:
        res = f"n = {self.n}\n"
        res += f"Concordance = {self.concordance:.4f}\n"
        if self.var is not None:
            res += f"Standard error = {np.sqrt(self.var):.4f}\n"
        res += f"Counts (concordant, discordant, tied.x, tied.y, tied.xy):\n"
        res += np.array_str(self.count)
        return res


def concordance(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    weights: Optional[Union[Sequence[float], np.ndarray]] = None,
    cluster: Optional[Union[Sequence[int], np.ndarray]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    influence: int = 0,
    ranks: bool = False,
    reverse: bool = False,
    timewt: str = "n",
    keepstrata: int = 10,
    std_err: bool = True,
) -> ConcordanceResult:
    """
    Compute the concordance index (C-index) between true and predicted values.
    This is a basic adaptation; does not handle time-dependent weights or
    interval-censoring.

    Parameters
    ----------
    y_true : array-like
        True outcome values (can be event/censoring times, or binary/class).
    y_pred : array-like
        Predicted values or risk scores.
    weights : array-like, optional
        Case weights; if None, equal weights are used.
    cluster : array-like, optional
        Optional grouping variable. If None, each observation is its own group.
    ymin : float, optional
        Minimum outcome to consider.
    ymax : float, optional
        Maximum outcome to consider.
    influence : int, optional
        Influence diagnostics, not implemented.
    ranks : bool, optional
        Return ranks information.
    reverse : bool, optional
        If True, swap concordant/discordant counts (e.g. for coxph conventions).
    timewt : str, optional
        Not used in this version.
    keepstrata : int, optional
        Not used in this version.
    std_err : bool, optional
        Whether to compute standard error.

    Returns
    -------
    ConcordanceResult
        Result object with concordance, counts, and (optionally) variance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    weights = np.ones(n) if weights is None else np.asarray(weights)

    mask = np.ones(n, dtype=bool)
    if ymin is not None:
        mask &= y_true >= ymin
    if ymax is not None:
        mask &= y_true <= ymax
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    tied_xy = 0
    pairs = 0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                tied_y += 1
                if y_pred[i] == y_pred[j]:
                    tied_xy += 1
                else:
                    tied_x += 1
                continue
            pairs += 1
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or (
                y_true[j] > y_true[i] and y_pred[j] > y_pred[i]
            ):
                concordant += 1
            elif (y_true[i] > y_true[j] and y_pred[i] < y_pred[j]) or (
                y_true[j] > y_true[i] and y_pred[j] < y_pred[i]
            ):
                discordant += 1
            else:
                tied_x += 1

    total = concordant + discordant + tied_x + tied_y + tied_xy
    if reverse:
        concordant, discordant = discordant, concordant

    concordance_index = (
        (concordant + 0.5 * tied_x) / (concordant + discordant + tied_x)
        if (concordant + discordant + tied_x) > 0
        else np.nan
    )
    count = np.array([concordant, discordant, tied_x, tied_y, tied_xy])
    var = None

    if std_err and pairs > 0:
        c = concordant
        d = discordant
        n_pairs = c + d + tied_x
        if n_pairs > 0:
            var = concordance_index * (1 - concordance_index) / n_pairs
        else:
            var = None

    return ConcordanceResult(
        concordance=concordance_index, n=len(y_true), count=count, var=var
    )


def print_concordance(res: ConcordanceResult, digits: int = 4) -> None:
    """
    Print a formatted summary of ConcordanceResult.
    """
    print(str(res))
