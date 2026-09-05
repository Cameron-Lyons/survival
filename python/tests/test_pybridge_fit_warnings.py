import importlib
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_call_fit_with_warnings = importlib.import_module("survival.pybridge")._call_fit_with_warnings


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
