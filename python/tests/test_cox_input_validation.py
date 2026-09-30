"""Native Cox entry points validate owned inputs before fitting or predicting."""

import numpy as np
import pytest

from .helpers import setup_survival_import

core = setup_survival_import()._survival


def _fit(*, penalized=False, stratified=False):
    time = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    status = [1, 1, 0, 1, 1, 0, 1, 0]
    x = [
        [0.2, 1.0],
        [0.8, 0.2],
        [0.4, 0.7],
        [1.1, 1.3],
        [0.7, 0.4],
        [0.3, 1.1],
        [1.3, 0.5],
        [0.5, 0.9],
    ]
    options = {"strata": [17, -3] * 4 if stratified else None}
    if penalized:
        return core.coxpenal_fit(
            time,
            status,
            x,
            penalties=[core.CoxPenalty.ridge(theta=1.0, scale=False)],
            pcols=[[0]],
            assign=[[0], [1]],
            **options,
        )
    return core.coxph_fit(time, status, x, **options)


@pytest.fixture(scope="module")
def fit():
    return _fit()


def _inputs():
    return {
        "newdata": np.array([[0.3, 0.7], [0.5, 0.1]]),
        "new_strata": np.array([0, 0], dtype=np.int32),
        "new_offset": np.array([0.1, -0.2]),
    }


def _call(fit, method, arguments):
    kwargs = dict(arguments)
    if method in {"expected", "survival", "individual"}:
        kwargs.setdefault("new_time", [2.0, 3.0])
        kwargs.setdefault("new_entry", [0.0, 1.0])
    if method == "individual":
        return fit.survfit_individual(id=[1, 2], **kwargs)
    if method == "curves":
        return fit.survfit(**kwargs)
    if method == "at":
        return fit.predict_survival_at([1.0, 2.0], **kwargs)
    if method == "terms":
        return fit.predict_terms(assign=[[0], [1]], se_fit=True, **kwargs)
    if method == "cohort":
        return fit.expected_survival(group=[0, 0], weights=[1.0, 1.0], times=[1.0, 2.0], **kwargs)
    return fit.predict(method, se_fit=True, **kwargs)


_METHODS = ("lp", "risk", "terms", "expected", "survival", "curves", "at", "individual", "cohort")
_BAD_SHARED = [
    (field, values) for field in ("new_strata", "new_offset") for values in ([], [0], [0, 0, 0])
] + [
    ("newdata", np.empty((0, 2))),
    ("newdata", [[1.0], [2.0]]),
    ("newdata", [[1.0, 0.2], [np.inf, 0.4]]),
    ("new_offset", [0.0, -np.inf]),
    ("new_strata", [0, 19]),
]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(("field", "values"), _BAD_SHARED)
def test_prediction_entry_points_reject_bad_shared_inputs(fit, method, field, values):
    inputs = {**_inputs(), field: values}
    before = {name: np.array(value, copy=True) for name, value in inputs.items()}
    with pytest.raises(ValueError, match="newdata|New data"):
        _call(fit, method, inputs)
    for name, snapshot in before.items():
        np.testing.assert_array_equal(inputs[name], snapshot)


@pytest.mark.parametrize("method", ["expected", "survival", "individual"])
@pytest.mark.parametrize("field", ["new_time", "new_entry"])
@pytest.mark.parametrize("values", [[], [0.0], [0.0, 1.0, 2.0], [0.0, np.nan], [0.0, np.inf]])
def test_followup_vectors_are_checked(fit, method, field, values):
    with pytest.raises(ValueError, match=field.replace("_", "data ")):
        _call(fit, method, {**_inputs(), field: values})


@pytest.mark.parametrize("method", ["expected", "survival", "individual"])
@pytest.mark.parametrize("start", [3.0, 4.0])
def test_invalid_counting_intervals_are_rejected(fit, method, start):
    with pytest.raises(ValueError, match="Stop time must be > start time"):
        _call(fit, method, {**_inputs(), "new_entry": [0.0, start]})


@pytest.mark.parametrize("method", ["curves", "individual", "cohort"])
@pytest.mark.parametrize(("field", "values"), _BAD_SHARED)
def test_penalized_curves_share_validation(method, field, values):
    with pytest.raises(ValueError, match="newdata|New data"):
        _call(_fit(penalized=True), method, {**_inputs(), field: values})


@pytest.mark.parametrize("method", ["expected", "survival", "at", "cohort"])
def test_missing_strata_cannot_select_an_arbitrary_baseline(method):
    with pytest.raises(ValueError, match="must carry the strata"):
        _call(_fit(stratified=True), method, {**_inputs(), "new_strata": None})


def test_full_curves_without_strata_return_every_baseline():
    curves = _call(_fit(stratified=True), "curves", {**_inputs(), "new_strata": None})
    assert [curve.stratum for curve in curves] == [-3, 17]
    assert all(np.asarray(curve.surv).shape[1] == 2 for curve in curves)


def test_individual_curves_without_strata_keep_r_first_stratum_default():
    fitted = _fit(stratified=True)
    implicit = _call(fitted, "individual", {**_inputs(), "new_strata": None})
    explicit = _call(fitted, "individual", {**_inputs(), "new_strata": [-3, -3]})
    assert len(implicit) == len(explicit) == 2
    for actual, expected in zip(implicit, explicit, strict=True):
        assert actual.time == expected.time
        np.testing.assert_array_equal(actual.surv, expected.surv)
        np.testing.assert_array_equal(actual.std_err, expected.std_err)


@pytest.mark.parametrize("field", ["newdata", "new_offset"])
def test_partial_predictions_preserve_independent_standard_errors(fit, field):
    inputs = _inputs()
    inputs[field][0] = np.nan
    lp = fit.predict("lp", se_fit=True, **inputs)
    assert np.isnan(lp.fit[0])
    assert np.isfinite(lp.fit[1])
    assert np.isfinite(lp.se_fit[1])
    assert np.isfinite(lp.se_fit[0]) == (field == "new_offset")
    expected = _call(fit, "expected", inputs)
    assert np.isnan(expected.fit[0])
    assert np.isfinite(expected.fit[1])
    for method in ("curves", "individual", "at", "cohort"):
        with pytest.raises(ValueError, match="non-finite"):
            _call(fit, method, inputs)


@pytest.mark.parametrize("function", [core.coxph_fit, core.coxph_fit_raw])
@pytest.mark.parametrize("empty_events", [False, True])
@pytest.mark.parametrize("field", ["status", "entry", "strata", "weights", "offset"])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_fit_entry_points_check_lengths_before_optimization(function, empty_events, field, length):
    inputs = {
        "time": [1.0, 2.0],
        "status": [0, 0] if empty_events else [1, 0],
        "x": [[0.2], [0.4]],
        field: [0] * length,
    }
    with pytest.raises(ValueError, match=field):
        function(**inputs)
