"""Complete repetition outputs, warnings and errors from independent stock R calls."""

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/surv_repetition_reference.json").read_text()
)
CASES = [
    pytest.param(case["response"], operation, id=f"{case['name']}-{operation['name']}")
    for case in REFERENCE["cases"]
    for operation in case["results"]
]


def _response(snapshot):
    rows = snapshot["matrix"]
    kind = snapshot["type"]

    def column(index):
        return [math.nan if row[index] is None else row[index] for row in rows]

    return r.Surv._from_normalized(
        time=column(1 if "counting" in kind else 0),
        event=[row[-1] for row in rows],
        start=column(0) if "counting" in kind else None,
        time2=column(1) if kind == "interval" else None,
        surv_type=kind,
        states=snapshot["states"] or (),
        clabel=snapshot["clabel"],
    )


def _argument(value, array):
    if isinstance(value, dict) and "number" in value:
        return math.inf if value["number"] == "Inf" else -math.inf
    return np.asarray(value) if array and isinstance(value, list) else value


@pytest.mark.parametrize(("snapshot", "operation"), CASES)
@pytest.mark.parametrize("array", [False, True], ids=["list", "numpy"])
def test_repetition_matches_stock_rows_metadata_warnings_and_errors(snapshot, operation, array):
    x = _response(snapshot)
    arguments = {name: _argument(value, array) for name, value in dict(operation["args"]).items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in operation:
            with pytest.raises(ValueError, match=re.escape(operation["error"])) as error:
                r.rep_surv(x, **arguments)
            assert str(error.value) == operation["error"]
        else:
            actual = r.rep_surv(x, **arguments)
            expected = operation["response"]
            assert actual.type == expected["type"]
            assert list(actual.states) == (expected["states"] or [])
            assert actual.clabel == expected["clabel"]
            assert actual.ncol == x.ncol
            np.testing.assert_equal(
                np.asarray(actual.as_matrix(), dtype=float),
                np.asarray(expected["matrix"], dtype=float),
            )
    assert [str(warning.message) for warning in caught] == operation["warnings"]


def test_repeated_response_keeps_independent_objects_with_immutable_columns():
    x = r.Surv([1, 3, 5], [0, 1, 0])
    repeated = r.rep_surv(x, 1)
    assert repeated is not x
    assert repeated.equals(x)
    matrix = repeated.as_matrix()
    matrix[0][0] = 100
    assert x.time == repeated.time == (1.0, 3.0, 5.0)


def test_empty_response_repetition_keeps_the_documented_empty_policy():
    x = r.Surv._from_normalized(
        time=[],
        event=[],
        start=[],
        time2=None,
        surv_type="mcounting",
        states=["a", "b"],
        clabel="censor",
    )
    for arguments in ({"times": 2}, {"each": 2}, {"length_out": 10}, {"times": []}):
        repeated = r.rep_surv(x, **arguments)
        assert repeated.equals(x)


def test_nullable_control_missing_values_use_defaults_without_coercion_warnings():
    pd = pytest.importorskip("pandas")
    x = r.Surv([1, 3, 5], [0, 1, 0])
    with warnings.catch_warnings(record=True) as caught:
        assert r.rep_surv(x, each=pd.NA, length_out=pd.NA).equals(x)
    assert caught == []
