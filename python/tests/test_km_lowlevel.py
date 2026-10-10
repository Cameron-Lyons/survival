"""Direct Kaplan–Meier components against R, including factor metadata."""

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "km_lowlevel_reference.json").read_text()
)


def factor(case):
    levels = case["levels"]
    codes = [levels.index(value) for value in case["group"]]
    return r.StrataFactor(
        codes, levels, case["group"], [codes.count(i) for i in range(len(levels))]
    )


def response(case):
    data = np.asarray(case["response"])
    return (
        r.Surv(data[:, 0], data[:, 1], data[:, 2], type="counting")
        if data.shape[1] == 3
        else r.Surv(data[:, 0], data[:, 1], type=case["response_type"])
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survfitkm_against_r(case):
    expected = case["expected"]
    if "error" in expected:
        message = "should be one of" if case["name"] == "bad_conf" else expected["error"]
        with pytest.raises((ValueError, TypeError), match=re.escape(message)):
            r.survfitKM(factor(case), response(case), **(case["arguments"] or {}))
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.survfitKM(factor(case), response(case), **(case["arguments"] or {}))
    assert [str(w.message) for w in caught] == case["warnings"]
    for name, target in expected.items():
        actual = getattr(fit, name)
        if target is None:
            assert actual is None, name
        elif name in {"influence_surv", "influence_chaz"}:
            assert len(actual) == len(target)
            for matrix, item in zip(actual, target, strict=True):
                if item is None:
                    assert matrix is None
                else:
                    assert [str(value) for value in matrix.cluster] == item["cluster"]
                    np.testing.assert_allclose(
                        matrix.values, np.array(item["values"], dtype=float), rtol=1e-12, atol=1e-14
                    )
        elif name == "counts":
            fields = [actual.n_risk, actual.n_event, actual.n_censor]
            if actual.n_enter is not None:
                fields.append(actual.n_enter)
            np.testing.assert_allclose(np.asarray(fields).T, target)
        elif name in {"strata", "type", "conf_type", "conf_lower", "logse"}:
            assert actual == target, name
        else:
            np.testing.assert_allclose(
                actual, np.asarray(target, dtype=float), rtol=1e-12, atol=1e-14, err_msg=name
            )


@pytest.mark.parametrize("kind", ["strata", "categorical", "series"])
def test_factor_types_and_retained_output(kind):
    case = next(case for case in REFERENCE["cases"] if case["name"] == "unused_levels")
    x = factor(case)
    if kind != "strata":
        pd = pytest.importorskip("pandas")
        x = pd.Categorical(case["group"], categories=case["levels"])
        if kind == "series":
            x = pd.Series(x)
    fit = r.survfitKM(x, response(case), influence=3)
    assert fit.n == [0, 4, 0, 4, 0]
    assert fit.strata == {"empty1": 0, "b": 4, "empty2": 0, "a": 4, "empty3": 0}
    assert fit.influence_surv[0] is None
    assert fit.influence_surv[2] is None
    assert fit.influence_surv[4] is None
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own test data
    np.testing.assert_allclose(restored.influence_surv[1].values, fit.influence_surv[1].values)
    assert not restored.influence_surv[1].values.flags.writeable
    fit.n[1] = 100
    fit.surv[0] = 100
    assert restored.n == fit.n
    assert restored.surv == fit.surv
    assert not hasattr(fit, "model")
    assert not hasattr(fit._fit, "status")


def test_close_times_are_prepared_not_merged():
    y = r.Surv([1, 1 + 1e-10, 2], [1, 1, 0])
    x = r.strata(["one"] * 3)
    direct = r.survfitKM(x, y)
    full = r.survfit(y)
    assert len(direct.time) == 3
    assert len(full.time) == 2
    assert r.survfit(y, timefix=False).time == direct.time


def test_dotted_keywords_and_ignored_time0():
    x = r.strata(["one"] * 4)
    y = r.Surv([1, 2, 3, 4], [1, 0, 1, 0])
    fit = r.survfitKM(x, y, **{"se.fit": False, "start.time": 2, "time0": object()})
    assert fit.time == [2, 3, 4]
    assert fit.std_err is None
    assert fit.conf_type is None
    assert (
        r.survfitKM(x, y, type="fh2", stype=99, ctype=99).surv
        == r.survfitKM(x, y, stype=2, ctype=2).surv
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": [1] * 4}, "x must be a factor"),
        ({"y": [[1, 1]] * 4}, "y must be a Surv object"),
        ({"x": r.strata(["one"] * 3)}, "different lengths"),
        ({"weights": [1]}, "same length"),
        ({"id": [1]}, "same length"),
        ({"cluster": [1, 2, None, 1]}, "missing"),
        ({"weights": [-1] * 4}, "non-negative"),
        ({"weights": [np.nan] * 4}, "finite"),
        ({"entry": 1}, "TRUE/FALSE"),
        ({"start_time": np.inf}, "all observations removed by start.time"),
        ({"timefix": True}, "unexpected keyword"),
    ],
)
def test_invalid_inputs(kwargs, message):
    args = {"x": r.strata(["one"] * 4), "y": r.Surv([1, 2, 3, 4], [1, 0, 1, 0])}
    with pytest.raises((ValueError, TypeError), match=message):
        r.survfitKM(**{**args, **kwargs})
