"""Current R censor labels survive fitting, curve transformations and serialization."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "curve_labels_reference.json").read_text()
)


def compare_table(table, expected):
    if isinstance(table, r.NamedMatrix):
        rows, columns, values = table.rownames, table.colnames, table.values
    else:
        rows, columns, values = table.from_states, table.to_states, table.counts
    assert rows == expected["rows"]
    assert columns == expected["columns"]
    np.testing.assert_array_equal(values, expected["values"])


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_current_r_curve_labels(case):
    pd = pytest.importorskip("pandas")
    data = {name: case[name] for name in ("time", "start", "id", "event", "group", "x")}
    data["event"] = pd.Categorical(data["event"], categories=case["levels"])
    response = "Surv(start,time,event)" if case["counting"] else "Surv(time,event)"
    curve = r.survfit(f"{response}~group", data, id="id")
    assert curve.clabel == case["label"]
    compare_table(curve.transitions, case["curve"]["transitions"])
    np.testing.assert_allclose(curve.time, case["curve"]["time"])
    np.testing.assert_allclose(curve.pstate, case["curve"]["pstate"], atol=1e-14)
    compare_table(r.survcheck(f"{response}~1", data, id="id").transitions, case["check"])
    cox = r.coxph(f"{response}~x", data, id="id", iter_max=0)
    compare_table(cox.transitions, case["cox"])
    compare_table(r.survfit(cox, newdata={"x": [0.25, 0.75]}).transitions, case["prediction"])
    for converted in (r.survfit0(curve), pickle.loads(pickle.dumps(curve))):  # noqa: S301
        assert converted.clabel == case["label"]
        compare_table(converted.transitions, case["curve"]["transitions"])
    for part in survival.r_api._survfit_strata_curves(curve).values():
        assert part.clabel == case["label"]
        assert part.transitions.colnames[-1] == f"({case['label']})"


def test_numeric_survcheck_uses_current_censor_label():
    check = r.survcheck(r.Surv([1, 2, 3], [0, 1, 0]), id=[1, 2, 3])
    assert check.transitions.to_states == ["event", "(censor)"]


def test_grouped_counting_curves_retain_internal_events():
    # R 3.8-12's grouped time grid accidentally omits these internal events.
    y = r.Surv([0, 2, 0, 3, 0, 4, 0, 5], [2, 5, 3, 6, 4, 7, 5, 8], [0, 1, 1, 0, 0, 1, 1, 0])
    fit = r.survfit(y, group=["a"] * 4 + ["b"] * 4, id=[1, 1, 2, 2, 3, 3, 4, 4])
    assert fit.time == [3, 5, 6, 5, 7, 8]
    assert sum(fit.n_event) == 4
    np.testing.assert_allclose(r.pseudo(fit, times=[3, 5]), [[1, 0.25], [0, 0.25], [1, 1], [1, 0]])
