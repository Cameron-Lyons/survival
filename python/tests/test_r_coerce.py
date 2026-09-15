"""The shared coercion helpers: R's ``as.character``/``format``/``factor`` and the timefix path."""

import importlib
import math

import pytest

from .helpers import setup_survival_import

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


def test_bool_options_accept_numpy_bools_and_reject_truthiness():
    np = pytest.importorskip("numpy")
    assert r_coerce._normalize_bool_option(np.bool_(True), "flag") is True
    assert r_coerce._normalize_bool_option(False, "flag") is False
    assert r_coerce._normalize_bool_option(None, "flag") is False
    with pytest.raises(TypeError, match="flag must be True or False"):
        r_coerce._normalize_bool_option(1, "flag")


def test_na_action_accepts_r_style_names_and_rejects_non_strings():
    for name in ("na.omit", " na.exclude ", "omit", "exclude"):
        assert r_coerce._normalize_na_action(name) == "omit"
    assert r_coerce._normalize_na_action("na.fail") == "fail"
    assert r_coerce._normalize_na_action(None) == "pass"
    assert r_coerce._normalize_na_action("na.pass") == "pass"
    with pytest.raises(TypeError, match="na_action"):
        r_coerce._normalize_na_action(1)
    with pytest.raises(ValueError, match="na_action must be"):
        r_coerce._normalize_na_action("na.drop")


def test_as_character_renders_numbers_like_r():
    # as.character(c(1, 1.5, 100000, 123456, 1e-4, 1/3, TRUE))
    assert [r_coerce._as_character(v) for v in [1.0, 1.5, 100000.0, 123456.0, 1e-4]] == [
        "1",
        "1.5",
        "1e+05",
        "123456",
        "1e-04",
    ]
    assert r_coerce._as_character(1 / 3) == "0.333333333333333"
    assert r_coerce._as_character(58.7652292950034) == "58.7652292950034"
    assert r_coerce._as_character(True) == "TRUE"
    assert r_coerce._as_character(7) == "7"
    assert r_coerce._as_character(None) == "NA"
    assert r_coerce._as_character("x") == "x"
    # the older helper names are aliases of the same function
    assert r_coerce._strata_value_label is r_coerce._as_character
    assert r_coerce._mstate_event_label is r_coerce._as_character


def test_format_numbers_uses_a_common_layout_like_r():
    # R's format(c(1.5, 2.25, 10)); format(1/3); format(c(0.001, 1e6)); format(100000)
    assert r_coerce._r_format_numbers([1.5, 2.25, 10]) == [" 1.50", " 2.25", "10.00"]
    assert r_coerce._r_format_numbers([1, 2, 3]) == ["1", "2", "3"]
    assert r_coerce._surv_format_number(1 / 3) == "0.3333333"
    assert r_coerce._r_format_numbers([0.001, 1e6]) == ["1e-03", "1e+06"]
    assert r_coerce._r_format_numbers([100000.0]) == ["1e+05"]
    assert r_coerce._r_format_numbers([123456789.0]) == ["123456789"]
    assert r_coerce._r_format_numbers([1 / 3, None, math.inf, -math.inf, math.nan]) == [
        "0.3333333",
        "       NA",
        "      Inf",
        "     -Inf",
        "       NA",
    ]
    assert r_coerce._r_format_numbers([]) == []


def test_factor_levels_follow_r_sort_order_and_declared_categories():
    assert r_coerce._factor_levels([3, 1, 2, None, 1]) == [1, 2, 3]
    assert r_coerce._factor_levels(["b", "a", "c"]) == ["a", "b", "c"]
    assert r_coerce._factor_levels([True, False]) == [False, True]
    assert r_coerce._factor_levels([2.0, 10.0, 1.0]) == [1.0, 2.0, 10.0]
    declared = r_coerce._RFactorVector(["b", "a"], ["z", "b", "a"])
    assert r_coerce._factor_levels(declared) == ["z", "b", "a"]
    codes, labels = r_coerce._factor(declared)
    assert (codes, labels) == ([1, 2], ["z", "b", "a"])
    codes, labels = r_coerce._factor([2, None, 1, 2])
    assert (codes, labels) == ([1, None, 0, 1], ["1", "2"])
    assert r_coerce._r_formula_ordered_levels([2, 1, 2], "x") == (1, 2)
    assert r_coerce._mstate_inferred_levels([2, 0, 1]) == ["0", "1", "2"]
    with pytest.raises(ValueError, match="outside the declared categories"):
        r_coerce._factor(r_coerce._RFactorVector(["q"], ["a"]))


def test_timefix_helpers_route_through_aeq_surv():
    assert r_coerce._aeq_times([1, 1 + 1e-14, 2]) == ([1.0, 1.0, 2.0],)
    start, stop = r_coerce._aeq_times([0, 1e-14, 1], [1, 1 + 1e-14, 2])
    assert (start, stop) == ([0.0, 0.0, 1.0], [1.0, 1.0, 2.0])
    assert r_coerce._survdiff_timefix_values([1, 1 + 1e-14], True) == [1.0, 1.0]
    assert r_coerce._survdiff_timefix_values([1, 1 + 1e-14], False) == [1, 1 + 1e-14]
    assert r_coerce._timefix_vectors([0, 1e-14], [1, 2]) == ([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="effective length 0"):
        r_coerce._aeq_times([0, 1], [1, 1 + 1e-14])
    with pytest.raises(ValueError, match="one or two time columns"):
        r_coerce._aeq_times([1], [2], [3])
