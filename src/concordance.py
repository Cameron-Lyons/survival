from typing import Optional, Union, Literal, Any, List
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ConcordanceResult:
    """Result object for concordance calculations."""

    concordance: Union[float, NDArray[np.float64]]
    count: NDArray[np.float64]
    n: int
    var: Optional[NDArray[np.float64]] = None
    cvar: Optional[NDArray[np.float64]] = None
    dfbeta: Optional[NDArray[np.float64]] = None
    influence: Optional[NDArray[np.float64]] = None
    ranks: Optional[pd.DataFrame] = None
    na_action: Optional[Any] = None
    call: Optional[Any] = None


class ConcordanceBase(ABC):
    """Base class for concordance calculations."""

    @abstractmethod
    def concordance(self, *args, **kwargs) -> ConcordanceResult:
        """Calculate concordance measure."""
        pass


def concordance(
    formula: str,
    data: pd.DataFrame,
    weights: Optional[NDArray[np.float64]] = None,
    subset: Optional[NDArray[np.bool_]] = None,
    na_action: Optional[str] = None,
    cluster: Optional[NDArray[np.int_]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    timewt: Literal["n", "S", "S/G", "n/G2", "I"] = "n",
    influence: int = 0,
    ranks: bool = False,
    reverse: bool = False,
    timefix: bool = True,
    keepstrata: Union[bool, int] = 10,
) -> ConcordanceResult:
    """
    Calculate concordance for survival data.

    Parameters
    ----------
    formula : str
        A model formula specifying the response and predictors.
    data : pd.DataFrame
        The data frame containing the variables in the formula.
    weights : Optional[NDArray[np.float64]], default=None
        Optional case weights.
    subset : Optional[NDArray[np.bool_]], default=None
        Optional logical vector specifying subset of observations to use.
    na_action : Optional[str], default=None
        Function to handle missing values.
    cluster : Optional[NDArray[np.int_]], default=None
        Optional clustering variable.
    ymin : Optional[float], default=None
        Minimum time value to consider.
    ymax : Optional[float], default=None
        Maximum time value to consider.
    timewt : {"n", "S", "S/G", "n/G2", "I"}, default="n"
        Time weighting scheme:
        - "n": No weighting
        - "S": Survival probability weighting
        - "S/G": S/G weighting
        - "n/G2": n/G^2 weighting
        - "I": Inverse risk set weighting
    influence : int, default=0
        Level of influence statistics to compute (0-3).
    ranks : bool, default=False
        Whether to return individual ranks.
    reverse : bool, default=False
        Whether to reverse concordant/discordant classification.
    timefix : bool, default=True
        Whether to apply time fixing for tied event times.
    keepstrata : Union[bool, int], default=10
        Whether to keep strata in output. If int, keep if nstrata <= value.

    Returns
    -------
    ConcordanceResult
        Object containing concordance statistics.

    Raises
    ------
    ValueError
        If formula is missing or invalid.
    TypeError
        If input types are incorrect.
    """
    if not formula:
        raise ValueError("A formula argument is required")

    if ymin is not None and (
        not isinstance(ymin, (int, float)) or not np.isscalar(ymin)
    ):
        raise ValueError("ymin must be a single number")

    if ymax is not None and (
        not isinstance(ymax, (int, float)) or not np.isscalar(ymax)
    ):
        raise ValueError("ymax must be a single number")

    if not isinstance(reverse, bool):
        raise TypeError("The reverse argument must be True/False")

    raise NotImplementedError("Full formula-based concordance not yet implemented")


def concordance_fit(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    strata: Optional[NDArray[np.int_]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    timewt: Literal["n", "S", "S/G", "n/G2", "I"] = "n",
    cluster: Optional[NDArray[np.int_]] = None,
    influence: int = 0,
    ranks: bool = False,
    reverse: bool = False,
    timefix: bool = True,
    keepstrata: Union[bool, int] = 10,
    std_err: bool = True,
) -> ConcordanceResult:
    """
    Core concordance calculation function.

    Parameters
    ----------
    y : NDArray[np.float64]
        Response variable (survival object or numeric).
    x : NDArray[np.float64]
        Predictor variable(s).
    strata : Optional[NDArray[np.int_]], default=None
        Stratification variable.
    weights : Optional[NDArray[np.float64]], default=None
        Case weights.
    ymin : Optional[float], default=None
        Minimum time value.
    ymax : Optional[float], default=None
        Maximum time value.
    timewt : {"n", "S", "S/G", "n/G2", "I"}, default="n"
        Time weighting scheme.
    cluster : Optional[NDArray[np.int_]], default=None
        Clustering variable.
    influence : int, default=0
        Level of influence statistics.
    ranks : bool, default=False
        Whether to compute ranks.
    reverse : bool, default=False
        Whether to reverse concordance.
    timefix : bool, default=True
        Whether to fix tied times.
    keepstrata : Union[bool, int], default=10
        Whether to keep strata.
    std_err : bool, default=True
        Whether to compute standard errors.

    Returns
    -------
    ConcordanceResult
        Concordance statistics.
    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return None

    n = len(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    nvar = x.shape[1]

    if x.shape[0] != n:
        raise ValueError("x and y are not the same length")

    if strata is None:
        strata = np.ones(n, dtype=int)
    if weights is None:
        weights = np.ones(n)

    if len(strata) != n:
        raise ValueError("y and strata are not the same length")
    if len(weights) != n:
        raise ValueError("y and weights are not the same length")

    raise NotImplementedError(
        "Full concordance_fit implementation requires C extensions"
    )


def btree(n: int) -> NDArray[np.int_]:
    """
    Create a balanced binary tree indexing.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    NDArray[np.int_]
        Tree indexing array.
    """

    def tfun(n: int, id: int, power: int) -> List[int]:
        if n == 1:
            return [id]
        elif n == 2:
            return [2 * id + 1, id]
        elif n == 3:
            return [2 * id + 1, id, 2 * id + 2]
        else:
            nleft = power if n == power * 2 else min(power - 1, n - power // 2)
            left = tfun(nleft, 2 * id + 1, power // 2)
            right = tfun(n - (nleft + 1), 2 * id + 2, power // 2)
            return left + [id] + right

    power = 2 ** int(np.floor(np.log2(n - 1)))
    return np.array(tfun(int(n), 0, int(power)))


def print_concordance(result: ConcordanceResult, digits: int = 3) -> None:
    """
    Print concordance results.

    Parameters
    ----------
    result : ConcordanceResult
        Concordance calculation results.
    digits : int, default=3
        Number of digits for display.
    """
    if result.call is not None:
        print("Call:")
        print(result.call)
        print()

    if result.na_action is not None:
        print(f"n={result.n} ({result.na_action})")
    else:
        print(f"n={result.n}")

    if result.var is None:
        print(f"Concordance = {result.concordance:.{digits}f}")
    else:
        if isinstance(result.concordance, np.ndarray) and len(result.concordance) > 1:
            se = np.sqrt(np.diag(result.var))
            df = pd.DataFrame({"concordance": result.concordance, "se": se})
            print(df.round(digits))
        else:
            se = (
                np.sqrt(result.var)
                if np.isscalar(result.var)
                else np.sqrt(result.var[0, 0])
            )
            print(f"Concordance = {result.concordance:.{digits}f} se = {se:.{digits}f}")

    if result.count is not None and (
        result.count.ndim == 1 or result.count.shape[0] < 11
    ):
        print("\nCount statistics:")
        print(np.round(result.count, 2))


class ConcordanceLM:
    """Concordance methods for linear models."""

    def concordance(
        self,
        object: Any,
        *args,
        newdata: Optional[pd.DataFrame] = None,
        cluster: Optional[NDArray[np.int_]] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        influence: int = 0,
        ranks: bool = False,
        timefix: bool = True,
        keepstrata: Union[bool, int] = 10,
        **kwargs,
    ) -> ConcordanceResult:
        """
        Calculate concordance for linear model objects.

        Parameters
        ----------
        object : Any
            Fitted model object.
        *args : Any
            Additional model objects for comparison.
        newdata : Optional[pd.DataFrame], default=None
            New data for prediction.
        cluster : Optional[NDArray[np.int_]], default=None
            Clustering variable.
        ymin : Optional[float], default=None
            Minimum y value.
        ymax : Optional[float], default=None
            Maximum y value.
        influence : int, default=0
            Influence level.
        ranks : bool, default=False
            Whether to compute ranks.
        timefix : bool, default=True
            Time fixing option.
        keepstrata : Union[bool, int], default=10
            Strata keeping option.

        Returns
        -------
        ConcordanceResult
            Concordance statistics.
        """
        raise NotImplementedError("Model-specific concordance not yet implemented")
