from typing import Dict, Any
import numpy as np


def agsurv(
    y: np.ndarray,
    x: np.ndarray,
    wt: np.ndarray,
    risk: np.ndarray,
    survtype: int,
    vartype: int,
) -> Dict[str, Any]:
    """
    Compute the components of a Cox model survival curve for one stratum.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples, 2 or 3)
        Survival data.  If 2 columns: [stop, status].
        If 3 columns: [start, stop, status].
    x : np.ndarray, shape (n_samples, n_vars)
        Covariate matrix.
    wt : np.ndarray, shape (n_samples,)
        Case weights.
    risk : np.ndarray, shape (n_samples,)
        Risk score for each observation, typically exp(X @ beta).
    survtype : {1, 2, 3}
        Curve type:
        1 = Kalbfleisch–Prentice (Kaplan–Meier),
        2 = Breslow,
        3 = Efron.
    vartype : {1, 2, 3}
        Variance type:
        1 = Kalbfleisch–Prentice variance,
        2 = Greenwood variance,
        3 = Efron approximation variance.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'n': total number of observations
        - 'time': sorted unique event times
        - 'n_event': weighted number of events at each time
        - 'n_risk': number at risk just before each time
        - 'n_censor': weighted number censored at each time
        - 'hazard': hazard increments λ(t)
        - 'cumhaz': cumulative hazard
        - 'varhaz': variance increments of the cumulative hazard
        - 'ndeath': unweighted count of deaths at each time
        - 'xbar': matrix of covariate means × hazard increments
        - 'surv' (only if survtype==1): survival probabilities
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    wt = np.asarray(wt, dtype=float)
    risk = np.asarray(risk, dtype=float)

    n_obs = y.shape[0]
    nvar = x.shape[1]
    status = y[:, -1].astype(int)
    dtime = y[:, -2]
    death = status == 1

    time = np.unique(np.sort(dtime))
    nevent = np.array([wt[(death) & (dtime == t)].sum() for t in time], dtype=float)
    ncens = np.array([wt[(~death) & (dtime == t)].sum() for t in time], dtype=float)

    wrisk = wt * risk

    def rcumsum(a: np.ndarray) -> np.ndarray:
        """Reverse cumulative sum."""
        return np.cumsum(a[::-1])[::-1]

    wrisk_sum = np.array([wrisk[dtime == t].sum() for t in time], dtype=float)
    wt_sum = np.array([wt[dtime == t].sum() for t in time], dtype=float)
    nrisk = rcumsum(wrisk_sum)
    irisk = rcumsum(wt_sum)

    if y.shape[1] == 2:
        xsum = np.vstack(
            [
                rcumsum(
                    np.array(
                        [(wrisk * x[:, j])[dtime == t].sum() for t in time], dtype=float
                    )
                )
                for j in range(nvar)
            ]
        ).T
    else:
        start = y[:, 0]
        delta = np.min(np.diff(time)) / 2
        entry_times = np.unique(start)
        etime = np.concatenate([entry_times, [entry_times.max() + delta]])
        idx = np.searchsorted(etime, time, side="right") - 1

        esum = rcumsum(
            np.array([wrisk[start == t].sum() for t in entry_times], dtype=float)
        )
        isum = rcumsum(
            np.array([wt[start == t].sum() for t in entry_times], dtype=float)
        )
        arr_esum = np.concatenate([esum, [0.0]])
        arr_isum = np.concatenate([isum, [0.0]])
        nrisk = nrisk - arr_esum[idx]
        irisk = irisk - arr_isum[idx]

        xout = np.vstack(
            [
                rcumsum(
                    np.array(
                        [(wrisk * x[:, j])[start == t].sum() for t in entry_times],
                        dtype=float,
                    )
                )
                for j in range(nvar)
            ]
        ).T
        xin = np.vstack(
            [
                rcumsum(
                    np.array(
                        [(wrisk * x[:, j])[dtime == t].sum() for t in time], dtype=float
                    )
                )
                for j in range(nvar)
            ]
        ).T
        arr_xout = np.vstack([xout, np.zeros((1, nvar))])
        xsum = xin - arr_xout[idx]

    ndeath = np.array([status[dtime == t].sum() for t in time], dtype=int)

    km_inc = None
    if survtype == 1:
        death_idx = np.where(death)[0]
        death_idx = death_idx[np.argsort(dtime[death_idx])]
        km_inc = np.empty_like(time)

    sum1 = sum2 = xbar_efron = None
    if survtype == 3 or vartype == 3:
        xsum2 = np.vstack(
            [
                np.array(
                    [((wrisk * death) * x[:, j])[dtime == t].sum() for t in time],
                    dtype=float,
                )
                for j in range(nvar)
            ]
        ).T
        erisk = np.array([(wrisk * death)[dtime == t].sum() for t in time], dtype=float)
        # tsum = cag_surv5(
        #     len(time), nvar,
        #     ndeath.astype(int),
        #     nrisk,
        #     erisk,
        #     xsum.flatten(),
        #     xsum2.flatten()
        # )
        # Placeholder: replace with actual C call
        tsum = {
            "sum1": np.ones_like(time),
            "sum2": np.ones_like(time),
            "xbar": np.ones((len(time), nvar)),
        }
        sum1 = tsum["sum1"]
        sum2 = tsum["sum2"]
        xbar_efron = tsum["xbar"]

    if survtype in (1, 2):
        hazard = nevent / nrisk
    else:
        hazard = nevent * sum1

    if vartype == 1:
        varhaz = nevent / (nrisk * np.where(nevent >= nrisk, nrisk, nrisk - nevent))
    elif vartype == 2:
        varhaz = nevent / (nrisk**2)
    else:
        varhaz = nevent * sum2

    if vartype in (1, 2):
        xbar = (xsum / nrisk[:, None]) * hazard[:, None]
    else:
        xbar = ndeath[:, None] * xbar_efron

    result: Dict[str, Any] = {
        "n": n_obs,
        "time": time,
        "n_event": nevent,
        "n_risk": irisk,
        "n_censor": ncens,
        "hazard": hazard,
        "cumhaz": np.cumsum(hazard),
        "varhaz": varhaz,
        "ndeath": ndeath,
        "xbar": xbar,
    }
    if survtype == 1:
        result["surv"] = km_inc

    return result
