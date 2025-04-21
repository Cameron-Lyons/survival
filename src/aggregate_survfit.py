from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


class SurvFit:
    """
    Simple container for survival‐fit results.
    Attributes may include:
      - surv: 2D array, shape (ntime, ndata)
      - pstate: 3D array, shape (ntime, nstate, ndata)
      - cumhaz, std_err, std_cumhaz, lower, upper, conf_int, conf_type, logse, cumhaz: optional arrays
      - newdata: DataFrame or None
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def aggregate_survfit(
    x: SurvFit,
    by: Optional[Union[Sequence[Any], List[Sequence[Any]]]] = None,
    FUN: Callable[[Sequence[float]], float] = np.mean,
    **kwargs,
) -> SurvFit:
    """
    Aggregate a SurvFit object to obtain population averages across groups.

    Parameters
    ----------
    x : SurvFit
        A survfit‐like object with attributes `surv` (ntime×ndata array) and/or
        `pstate` (ntime×nstate×ndata array).
    by : sequence or list of sequences, optional
        Group labels for each of the ndata curves. If None, all curves are
        aggregated into one group.
    FUN : callable, default=np.mean
        Function to apply to each group, e.g., np.mean, np.median, etc.
    **kwargs
        Ignored (for compatibility).

    Returns
    -------
    SurvFit
        A new SurvFit with:
          - `surv` averaged across data dimension (or grouped by `by`),
          - `pstate` likewise aggregated,
          - dropped attributes: std_err, std_cumhaz, lower, upper, conf_int,
            conf_type, logse, cumhaz,
          - `newdata`: DataFrame of group labels (or None if `by` is None).
    """
    if not isinstance(x, SurvFit):
        raise TypeError("x must be a SurvFit object")

    if hasattr(x, "surv") and x.surv is not None:
        ndata = x.surv.shape[1]
    elif hasattr(x, "pstate") and x.pstate is not None:
        ndata = x.pstate.shape[2]
    else:
        raise ValueError("SurvFit object must have `surv` or `pstate`")

    if by is None:
        index = np.ones(ndata, dtype=int)
        group_labels = None
    else:
        if isinstance(by, list):
            by_list = by
        else:
            by_list = [list(by)]
        if any(len(b) != ndata for b in by_list):
            raise ValueError("All `by` sequences must have length = number of curves")
        first = by_list[0]
        codes = pd.Categorical(first).codes  # 0-based
        index = codes + 1  # make it 1-based
        if np.all(index == index[0]):
            index = np.ones(ndata, dtype=int)
            group_labels = None
        else:
            group_labels = by_list

    test = pd.Series(np.arange(ndata)).groupby(index).apply(lambda idx: FUN(idx.values))
    if test.size != index.max() or not np.issubdtype(test.dtype, np.number):
        raise ValueError("FUN must return a single numeric value per group")

    drop_attrs = {
        "std_err",
        "std_cumhaz",
        "lower",
        "upper",
        "conf_int",
        "conf_type",
        "logse",
        "cumhaz",
    }
    newx = SurvFit(**{k: v for k, v in x.__dict__.items() if k not in drop_attrs})

    ngroups = int(index.max())

    if hasattr(x, "surv") and x.surv is not None:
        surv = x.surv  # shape (ntime, ndata)
        ntime = surv.shape[0]
        if group_labels is None:
            newx.surv = (
                surv.mean(axis=1)
                if FUN is np.mean
                else np.apply_along_axis(FUN, 1, surv)
            )
        else:
            agg = np.zeros((ntime, ngroups), dtype=float)
            for t in range(ntime):
                df = pd.DataFrame({"val": surv[t, :], "grp": index})
                grp = df.groupby("grp")["val"].apply(lambda arr: FUN(arr.values))
                agg[t, :] = grp.values
            newx.surv = agg

    if hasattr(x, "pstate") and x.pstate is not None:
        pstate = x.pstate  # shape (ntime, nstate, ndata)
        ntime, nstate, _ = pstate.shape
        if group_labels is None:
            newx.pstate = (
                pstate.mean(axis=2)
                if FUN is np.mean
                else np.apply_along_axis(FUN, 2, pstate)
            )
        else:
            agg3 = np.zeros((ntime, nstate, ngroups), dtype=float)
            for t in range(ntime):
                for s in range(nstate):
                    df = pd.DataFrame({"val": pstate[t, s, :], "grp": index})
                    grp = df.groupby("grp")["val"].apply(lambda arr: FUN(arr.values))
                    agg3[t, s, :] = grp.values
            newx.pstate = agg3

    if group_labels is None:
        newx.newdata = None
    else:
        if len(group_labels) == 1:
            levels = pd.Categorical(group_labels[0]).categories
            newx.newdata = pd.DataFrame({"aggregate": levels})
        else:
            df = pd.DataFrame({f"by{i}": grp for i, grp in enumerate(group_labels)})
            df["group"] = index
            newx.newdata = df.drop_duplicates().reset_index(drop=True)

    return newx
