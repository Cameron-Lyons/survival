import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_coxcount1():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    strata = [1, 0, 0, 0, 0, 0, 0, 0]

    result = survival.coxcount1(time, status, strata)
    assert hasattr(result, "time")
    assert hasattr(result, "nrisk")
    assert hasattr(result, "index")
    assert hasattr(result, "status")
    assert len(result.time) > 0


def test_coxcount2():
    time1 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    time2 = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    status = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    sort1 = [0, 2, 4, 1, 3, 5]
    sort2 = [0, 2, 4, 1, 3, 5]
    strata2 = [1, 0, 0, 0, 0, 0]

    result2 = survival.coxcount2(time1, time2, status, sort1, sort2, strata2)
    assert hasattr(result2, "time")
    assert hasattr(result2, "nrisk")
