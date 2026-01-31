import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_agsurv4():
    ndeath = [1, 1, 0, 1, 0]
    risk = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0]
    sn = 5
    denom = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = survival.agsurv4(ndeath, risk, wt, sn, denom)
    assert isinstance(result, list)
    assert len(result) == sn


def test_agsurv5():
    n = 5
    nvar = 2
    dd = [1, 1, 2, 1, 1]
    x1 = [10.0, 9.0, 8.0, 7.0, 6.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0]
    xsum = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    xsum2 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.5, 2.0, 1.5, 1.0, 0.5]

    result = survival.agsurv5(n, nvar, dd, x1, x2, xsum, xsum2)
    assert isinstance(result, dict)
    assert "sum1" in result
    assert "sum2" in result
    assert "xbar" in result


def test_agmart():
    n = 5
    method = 0
    start = [0.0, 0.0, 1.0, 1.0, 2.0]
    stop = [1.0, 2.0, 2.0, 3.0, 3.0]
    event = [1, 0, 1, 0, 1]
    score = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    strata = [1, 0, 0, 0, 0]

    result = survival.agmart(n, method, start, stop, event, score, wt, strata)
    assert isinstance(result, list)
    assert len(result) == n


def test_survfitkm():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    result = survival.survfitkm(
        time=time,
        status=status,
        weights=None,
        entry_times=None,
        position=None,
        reverse=False,
        computation_type=0,
    )
    assert hasattr(result, "time")
    assert hasattr(result, "estimate")
    assert hasattr(result, "std_err")
    assert len(result.time) > 0
    assert len(result.estimate) == len(result.time)


def test_survdiff2():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    result = survival.survdiff2(
        time=time,
        status=status,
        group=group,
        strata=None,
        rho=0.0,
    )
    assert hasattr(result, "observed")
    assert hasattr(result, "expected")
    assert hasattr(result, "chi_squared")
    assert len(result.observed) > 0
