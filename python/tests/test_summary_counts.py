"""Requested-time count selection against stock survival 3.8-12."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "summary_counts_reference.json").read_text()
)


@pytest.fixture(scope="module")
def fits():
    data = dict(REFERENCE["data"])
    data["event"] = survival.r_api._r_factor(data["event"], REFERENCE["event_levels"])
    result = {}
    for name, spec in REFERENCE["fits"].items():
        options = {key: spec[key] for key in ("weights", "id", "entry") if key in spec}
        if spec["kind"] in {"cox", "coxms"}:
            options.pop("entry", None)
            model = r.coxph(spec["formula"], data, **options)
            result[name] = r.survfit(model, newdata=spec["newdata"])
        else:
            result[name] = r.survfit(spec["formula"], data, **options)
    return result


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_summary_count_selection_against_stock_r(case, fits):
    options = {"times": case["times"], "extend": case["extend"], "scale": 2, "rmean": "none"}
    if case["dosum"] is not None:
        options["dosum"] = case["dosum"]
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            r.summary_survfit(fits[case["fit"]], **options)
        return
    result = r.model_summary(fits[case["fit"]], **options)
    for field, values in expected.items():
        actual = getattr(result, field, None)
        if values is None:
            assert actual is None or (field == "cumhaz" and len(actual) == 0), field
        else:
            np.testing.assert_allclose(
                actual,
                np.asarray(values, dtype=float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=field,
            )


@pytest.mark.parametrize("kind", ["km", "aj"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_native_summary_accepts_strided_numpy_times(kind, dtype, fits):
    fit = fits[f"{kind}_right_one_unit"]
    times = np.array([0, 100, 2, 100, 4, 100, 7, 100], dtype=dtype)[::2]

    def call(values):
        if kind == "km":
            return survival.surv_analysis.summary_survfit(fit.engine, times=values, dosum=False)
        return fit.engine.summary(times=values, dosum=False)

    actual = call(times)
    expected = call(times.tolist())
    for field in ("time", "n_risk", "n_event", "n_censor"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("kind", ["km", "aj", "cox", "coxms"])
def test_summary_validates_explicit_dosum_only_with_requested_times(kind, fits):
    name = f"{kind}_right_one_unit" if kind in {"km", "aj"} else f"{kind}_right"
    if kind == "cox":
        name += "_one"
    fit = fits[name]
    with pytest.raises(ValueError, match="dosum must be TRUE/FALSE"):
        r.summary_survfit(fit, times=[1, 2], dosum="yes")
    result = r.summary_survfit(fit, dosum="ignored")
    expected = r.summary_survfit(fit)
    np.testing.assert_array_equal(result.n_event, expected.n_event)
