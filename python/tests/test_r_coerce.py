import importlib

import pytest

from .helpers import setup_survival_import
from .r_api_support import _toy_data

survival = setup_survival_import()
r_coerce = importlib.import_module("survival.r._coerce")


def test_row_has_missing_uses_primitive_fast_paths_and_nested_fallbacks():
    cases = [
        (None, True),
        (float("nan"), True),
        (1.25, False),
        (3, False),
        (True, False),
        ("value", False),
        ([1.0, None], True),
        ((1.0, float("nan")), True),
        ([1.0, (2.0, 3.0)], False),
    ]

    for value, expected in cases:
        assert r_coerce._row_has_missing(value) is expected

    class _ListSubclass(list):
        pass

    assert r_coerce._row_has_missing(_ListSubclass([1.0, None])) is True


def test_row_has_missing_preserves_custom_comparison_fallback():
    class _MissingByComparison:
        def __ne__(self, other):
            return True

    class _ComparisonError:
        def __ne__(self, other):
            raise TypeError("comparison unavailable")

    assert r_coerce._row_has_missing(_MissingByComparison()) is True
    assert r_coerce._row_has_missing(_ComparisonError()) is False


def test_row_has_missing_preserves_numpy_and_pandas_sentinels():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    missing = [np.float64("nan"), np.datetime64("NaT"), pd.NA, pd.NaT]
    present = [np.float64(1.0), np.int64(2), pd.Timestamp("2025-01-01")]

    assert all(r_coerce._row_has_missing(value) for value in missing)
    assert not any(r_coerce._row_has_missing(value) for value in present)


def test_r_api_bool_options_accept_numpy_bool_scalars():
    np = pytest.importorskip("numpy")
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    rows = [[0.5, 0.8]]
    scores = [8.0 - value for value in data["time"]]

    direct_reverse = survival.survfit(response, reverse=True)
    numpy_reverse = survival.survfit(response, reverse=np.bool_(True))
    assert numpy_reverse.time == pytest.approx(direct_reverse.time)
    assert numpy_reverse.estimate == pytest.approx(direct_reverse.estimate)

    direct_concordance = survival.concordance(response, scores=scores, reverse=True)
    numpy_concordance = survival.concordance(response, scores=scores, reverse=np.bool_(True))
    numpy_timefix = survival.concordance(response, scores=scores, timefix=np.bool_(False))
    assert numpy_concordance.concordance == pytest.approx(direct_concordance.concordance)
    assert numpy_concordance.reverse is True
    assert numpy_timefix.concordance == pytest.approx(
        survival.concordance(response, scores=scores, timefix=False).concordance
    )

    direct_basehaz = survival.basehaz(fit, centered=False)
    numpy_basehaz = survival.basehaz(fit, centered=np.bool_(False))
    assert numpy_basehaz.cumhaz == pytest.approx(direct_basehaz.cumhaz)
    assert numpy_basehaz.centered is False

    zph = survival.cox_zph(
        fit,
        terms=np.bool_(False),
        singledf=np.bool_(False),
        global_test=np.bool_(False),
    )
    assert zph.variable_names == ["x1", "x2"]
    assert zph.table[-1]["name"] != "GLOBAL"

    prediction = survival.predict(fit, rows, se_fit=np.bool_(True))
    assert isinstance(prediction, survival.r_api.PredictResult)
    assert prediction.fit == pytest.approx(survival.predict(fit, rows))

    uncentered = survival.predict(fit, rows, centered=np.bool_(False))
    assert uncentered == pytest.approx(survival.predict(fit, rows, reference="zero"))

    residual_values = survival.r_api.residuals(fit, weighted=np.bool_(False))
    assert residual_values == pytest.approx(survival.r_api.residuals(fit, weighted=False))

    detail = survival.coxph_detail(fit, riskmat=np.bool_(True))
    assert detail.riskmat is not None


def test_r_api_bool_options_reject_python_truthiness():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    rows = [[0.5, 0.8]]
    scores = [8.0 - value for value in data["time"]]

    with pytest.raises(TypeError, match="reverse"):
        survival.survfit(response, reverse=1)
    with pytest.raises(TypeError, match="reverse"):
        survival.concordance(response, scores=scores, reverse="yes")
    with pytest.raises(TypeError, match="timefix"):
        survival.concordance(response, scores=scores, timefix=1)
    with pytest.raises(TypeError, match="centered"):
        survival.basehaz(fit, centered=1)
    with pytest.raises(TypeError, match="terms"):
        survival.cox_zph(fit, terms=1)
    with pytest.raises(TypeError, match="singledf"):
        survival.cox_zph(fit, singledf="yes")
    with pytest.raises(TypeError, match="global"):
        survival.cox_zph(fit, global_test=1)
    with pytest.raises(ValueError, match="global_test or global"):
        survival.cox_zph(fit, global_test=False, **{"global": True})
    with pytest.raises(TypeError, match="se_fit"):
        survival.predict(fit, rows, se_fit=1)
    with pytest.raises(TypeError, match="centered"):
        survival.predict(fit, rows, centered=1)
    with pytest.raises(TypeError, match="weighted"):
        survival.r_api.residuals(fit, weighted=1)
    with pytest.raises(TypeError, match="riskmat"):
        survival.coxph_detail(fit, riskmat=1)


def test_na_action_accepts_r_style_names_and_rejects_non_strings():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", None, "treated", "treated"],
    }
    direct = survival.survfit(
        survival.Surv([1.0, 3.0, 4.0], [1, 1, 1]),
        group=["control", "treated", "treated"],
    )

    for na_action in ("na.omit", " na.exclude "):
        fit = survival.survfit("Surv(time, status) ~ arm", data=data, na_action=na_action)
        assert list(fit) == list(direct)
        assert fit["control"].estimate == pytest.approx(direct["control"].estimate)
        assert fit["treated"].estimate == pytest.approx(direct["treated"].estimate)

    passthrough = survival.survfit(
        "Surv(time, status) ~ group",
        data=_toy_data(),
        na_action="na.pass",
    )
    default = survival.survfit("Surv(time, status) ~ group", data=_toy_data())
    assert list(passthrough) == list(default)
    assert passthrough["A"].estimate == pytest.approx(default["A"].estimate)
    assert passthrough["B"].estimate == pytest.approx(default["B"].estimate)

    with pytest.raises(TypeError, match="na_action"):
        survival.survfit("Surv(time, status) ~ group", data=_toy_data(), na_action=1)


def test_formula_fit_iteration_counts_accept_integer_valued_floats():
    data = _toy_data()

    cox = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=0.0)
    aft = survival.survreg("Surv(time, status) ~ x1", data=data, max_iter=1.0)

    assert len(cox.coefficients[0]) == 1
    assert len(aft.location_coefficients) == 2
    with pytest.raises(ValueError, match="integer"):
        survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=1.5)
