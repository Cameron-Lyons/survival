import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_collapse():
    y = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 0.0]
    x = [1, 1, 1, 1]
    istate = [0, 0, 0, 0]
    subject_id = [1, 1, 2, 2]
    wt = [1.0, 1.0, 1.0, 1.0]
    order = [0, 1, 2, 3]

    result = survival.collapse(y, x, istate, subject_id, wt, order)
    assert isinstance(result, dict)
    assert "matrix" in result
    assert "dimnames" in result
