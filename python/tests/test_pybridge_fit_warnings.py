import importlib
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_call_fit_with_warnings = importlib.import_module("survival.pybridge")._call_fit_with_warnings
_raise_captured_error = importlib.import_module("survival.pybridge")._raise_captured_error


def test_fit_warning_capture_preserves_result_and_records_each_call(capsys):
    expected = object()

    def fit(*, value):
        warnings.warn("fit did not converge", RuntimeWarning, stacklevel=1)
        return value

    for _ in range(2):
        captured = _call_fit_with_warnings(fit, {"value": expected})
        assert captured["result"] is expected
        assert captured["warnings"] == ["fit did not converge"]
    assert capsys.readouterr().err == ""


def test_fit_warning_capture_restores_filters_and_preserves_other_category_filters():
    def fit():
        warnings.warn("hidden user warning", UserWarning, stacklevel=1)
        warnings.warn("visible fit warning", RuntimeWarning, stacklevel=1)
        return 7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        previous_filters = list(warnings.filters)
        captured = _call_fit_with_warnings(fit, {})
        assert captured == {"result": 7, "warnings": ["visible fit warning"]}
        assert warnings.filters == previous_filters
        captured = _call_fit_with_warnings(fit, {}, user_warnings=True)
        assert captured == {"result": 7, "warnings": ["hidden user warning", "visible fit warning"]}
        assert warnings.filters == previous_filters
        with pytest.raises(RuntimeWarning, match="outside fit"):
            warnings.warn("outside fit", RuntimeWarning, stacklevel=1)


def test_fit_warning_capture_preserves_python_exceptions_and_restores_filters():
    expected = ValueError("invalid fit input")

    def fit():
        raise expected

    previous_filters = list(warnings.filters)
    with pytest.raises(ValueError, match="invalid fit input") as caught:
        _call_fit_with_warnings(fit, {})
    assert caught.value is expected
    assert warnings.filters == previous_filters


def test_failed_calls_can_return_warnings_before_resignalling_the_same_exception():
    expected = ValueError("invalid fit input")

    def fit():
        warnings.warn("NaNs produced", UserWarning, stacklevel=1)
        raise expected

    previous_filters = list(warnings.filters)
    captured = _call_fit_with_warnings(fit, {}, user_warnings=True, capture_error=True)
    assert captured["warnings"] == ["NaNs produced"]
    assert captured["result"] is None
    assert captured["error"] is expected
    assert warnings.filters == previous_filters
    with pytest.raises(ValueError, match="invalid fit input") as caught:
        _raise_captured_error(captured["error"])
    assert caught.value is expected


def test_warning_capture_preserves_positional_formula_calls():
    def formula_call(formula, /, *, data):
        warnings.warn("NaNs produced", UserWarning, stacklevel=1)
        return formula, data

    result = _call_fit_with_warnings(
        formula_call, {"data": [1, 2]}, positional=["y ~ log(x)"], user_warnings=True
    )
    assert result == {"result": ("y ~ log(x)", [1, 2]), "warnings": ["NaNs produced"]}
