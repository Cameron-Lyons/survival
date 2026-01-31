import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_cipoisson_exact():
    result = survival.cipoisson_exact(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] < result[1]


def test_cipoisson_anscombe():
    result = survival.cipoisson_anscombe(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_cipoisson():
    result_exact = survival.cipoisson(k=5, time=10.0, p=0.95, method="exact")
    result_anscombe = survival.cipoisson(k=5, time=10.0, p=0.95, method="anscombe")
    assert isinstance(result_exact, tuple)
    assert isinstance(result_anscombe, tuple)


def test_norisk():
    time1 = [0.0, 1.0, 2.0, 3.0, 4.0]
    time2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    sort1 = [0, 1, 2, 3, 4]
    sort2 = [0, 1, 2, 3, 4]
    strata = [1, 0, 0, 0, 0]

    result = survival.norisk(time1, time2, status, sort1, sort2, strata)
    assert isinstance(result, list)
    assert len(result) == len(time1)


def test_finegray():
    tstart = [0.0, 0.0, 0.0, 0.0]
    tstop = [1.0, 2.0, 3.0, 4.0]
    ctime = [0.5, 1.5, 2.5, 3.5]
    cprob = [0.1, 0.2, 0.3, 0.4]
    extend = [True, True, False, False]
    keep = [True, True, True, True]

    result = survival.finegray(
        tstart=tstart,
        tstop=tstop,
        ctime=ctime,
        cprob=cprob,
        extend=extend,
        keep=keep,
    )
    assert hasattr(result, "row")
    assert hasattr(result, "start")
    assert hasattr(result, "end")
    assert hasattr(result, "wt")
    assert len(result.row) > 0
