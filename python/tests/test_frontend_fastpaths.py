"""The numpy fast paths of the R front end and its one ``strata()`` builder.

Numeric numpy, pandas and polars columns are coded with vector operations (``Surv``'s
time and status columns, ``is.na``, the ``na.action`` scan, ``factor()`` codes); every
other input keeps the per-element path, and both must give the same result.  The
``strata()`` references come from R 4.5.3 with survival 3.8-12 (``Rscript`` calls quoted
next to the assertions).
"""

from __future__ import annotations

import importlib
import math
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_coerce = importlib.import_module("survival.r._coerce")
r_surv = importlib.import_module("survival.r._surv")

NAN = math.nan
STATUS_CASES = [
    [0, 1, 1, 0, 1],
    [1, 2, 2, 1, 2],
    [0, 1, 3, 0, 1],
    [0.0, 1.0, NAN, 0.0, 1.0],
    [1.0, 2.0, NAN, 1.0, 0.5],
    [NAN, NAN, NAN, NAN, NAN],
    [True, False, True, True, False],
]


def _key(values):
    """Values comparable across NaN (``nan != nan``) and int/float spellings."""

    return [
        "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else value
        for value in values
    ]


def _surv_key(surv):
    return (_key(surv.time), _key(surv.event), surv.type)


def _surv_and_warnings(time, status):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surv = r.Surv(time, status)
    return _surv_key(surv), [str(w.message) for w in caught]


def _inputs(values):
    """The same column as a list, a numpy array and (when installed) pandas/polars."""

    array = np.array(values, dtype=bool if isinstance(values[0], bool) else float)
    yield list(values)
    yield array
    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        yield pd.Series(array)
    try:
        import polars as pl
    except ImportError:
        pass
    else:
        yield pl.Series(array)


@pytest.mark.parametrize("status", STATUS_CASES)
def test_surv_status_coding_is_the_same_for_every_input_kind(status):
    time = [5.0, 3.0, 8.0, 1.0, 2.0]
    expected = _surv_and_warnings(time, list(status))
    for column in _inputs(status):
        assert _surv_and_warnings(np.array(time), column) == expected


def test_surv_status_warning_and_recoding_follow_r():
    # Surv(1:5, c(1, 2, 2, 1, 3)): the maximum is not 2, so 2 and 3 are invalid codes
    with pytest.warns(UserWarning, match="Invalid status value, converted to NA"):
        surv = r.Surv(np.arange(1, 6), np.array([1, 2, 2, 1, 3]))
    assert surv.event == (1, None, None, 1, None)
    # Surv(1:4, c(1, 2, 2, NA)): the 1/2 coding with a missing status
    assert r.Surv(np.arange(1, 5), np.array([1, 2, 2, NAN])).event == (0, 1, 1, None)
    # integer and bool columns
    assert r.Surv(np.array([1, 2]), np.array([2, 1], dtype=np.int32)).event == (1, 0)
    assert r.Surv(np.array([1, 2]), np.array([True, False])).event == (1, 0)
    assert r.Surv(np.array([1, 2], dtype=np.uint8), [1, 0]).time == (1.0, 2.0)


def test_surv_time_rejects_logical_numpy_columns():
    # Surv(c(TRUE, FALSE), c(1, 0)): "Time variable is not numeric"
    with pytest.raises(ValueError, match="Time variable is not numeric"):
        r.Surv(np.array([True, False]), [1, 0])
    with pytest.raises(ValueError, match="Stop time is not numeric"):
        r.Surv([0, 0], np.array([True, True]), [1, 0])


def test_surv_missing_values_match_for_none_nan_and_pandas_na():
    pd = pytest.importorskip("pandas")
    expected = _surv_key(r.Surv([1.0, None, 3.0], [1, 0, None]))
    assert _surv_key(r.Surv(np.array([1.0, NAN, 3.0]), np.array([1, 0, NAN]))) == expected
    time = pd.Series([1, None, 3], dtype="Int64")
    status = pd.Series([1, 0, None], dtype="Int64")
    assert _surv_key(r.Surv(time, status)) == expected
    assert _surv_key(r.Surv(pd.Series([1.0, NAN, 3.0]), [True, False, pd.NA])) == expected


def test_is_na_surv_marks_a_missing_entry_in_any_column():
    # is.na(Surv(c(1, NA, 3, 4), c(1, 1, NA, 0)))
    assert r.is_na_surv(r.Surv(np.array([1, NAN, 3, 4]), np.array([1, 1, NAN, 0]))) == [
        False,
        True,
        True,
        False,
    ]
    # is.na(Surv(c(0, 1, NA), c(1, 2, 3), c(1, 0, 1)))
    counting = r.Surv([0.0, 1.0, None], [1.0, 2.0, 3.0], [1, 0, 1])
    assert r.is_na_surv(counting) == [False, False, True]
    assert r.is_na_surv(r.Surv2([1.0, NAN], [1, 0])) == [False, True]


@pytest.mark.parametrize(
    "values",
    [
        [3.0, 1.0, NAN, 1.0, 2.0],
        [2, 10, 1, 10],
        [0.0, -0.0, 1.5, NAN],
        [True, False, True],
    ],
)
def test_factor_codes_match_the_generic_path(values):
    expected = r_coerce._factor(list(values), "x")
    for column in _inputs(values):
        assert r_coerce._factor(column, "x") == expected
        assert r_coerce._factor_levels(column, "x") == r_coerce._factor_levels(list(values), "x")


def test_factor_levels_put_infinities_in_r_order():
    # levels(factor(c(1, Inf, -Inf, 0))); levels(strata(x)) with x <- c(2, -Inf, Inf, 0)
    assert r_coerce._factor([1.0, math.inf, -math.inf, 0.0])[1] == ["-Inf", "0", "1", "Inf"]
    assert r_coerce._factor(np.array([1.0, np.inf, -np.inf, 0.0]))[1] == ["-Inf", "0", "1", "Inf"]
    expected = ["x=-Inf", "x=0", "x=2", "x=Inf"]
    assert r.strata([2.0, -math.inf, math.inf, 0.0], labels=["x"]).levels == expected
    assert r.strata(np.array([2.0, -np.inf, np.inf, 0.0]), labels=["x"]).levels == expected


def test_factor_of_declared_categories_refuses_other_values():
    codes, labels = r_coerce._factor(r_coerce._RFactorVector(["b", None, "a"], ["b", "a"]))
    assert (codes, labels) == ([0, None, 1], ["b", "a"])
    with pytest.raises(ValueError, match="outside the declared categories"):
        r_coerce._factor(r_coerce._RFactorVector(["q"], ["a"]))


def test_missing_row_scan_matches_the_generic_path():
    columns = {
        "a": [1.0, NAN, 3.0, 4.0],
        "b": [1, 2, 3, 4],
        "c": [True, False, True, True],
        "d": [0.5, 0.5, 0.5, NAN],
    }
    expected = r_coerce._missing_row_indices(list(columns.items()), 4)
    assert expected == {1, 3}
    as_arrays = [(name, np.array(values)) for name, values in columns.items()]
    assert r_coerce._missing_row_indices(as_arrays, 4) == expected
    with pytest.raises(ValueError, match="a must have length 5"):
        r_coerce._missing_row_indices(as_arrays, 5)


def test_formula_fits_are_the_same_for_list_numpy_and_pandas_data():
    pd = pytest.importorskip("pandas")
    lung = {
        name: values for name, values in survival.datasets.load_lung().items() if name[0] != "_"
    }
    frames = [
        lung,
        {name: np.array(values, dtype=float) for name, values in lung.items()},
        pd.DataFrame(lung),
    ]
    fits = [r.survfit("Surv(time, status) ~ sex + strata(ph.ecog)", data) for data in frames]
    for fit in fits[1:]:
        assert fit.strata == fits[0].strata
        assert fit.surv == fits[0].surv
    tests = [r.survdiff("Surv(time, status) ~ sex + strata(ph.ecog)", data) for data in frames]
    for test in tests[1:]:
        assert test.chisq == tests[0].chisq


# --- strata() ----------------------------------------------------------------------


def _padding_data():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "time": [5, 8, 3, 9, 12, 4, 7, 10, 6, 11, 2, 13],
            "status": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            "x": [1, 2] * 6,
            "z": [0.5, 1.2, -0.3, 0.8, 1.9, -1.1, 0.2, 0.4, -0.6, 1.5, 0.9, -0.2],
            "f": pd.Categorical(["a", "bb", "a"] * 4, categories=["a", "bb", "longlevel"]),
        }
    )


# levels(with(d, strata(x, f))) with f <- factor(., levels = c("a", "bb", "longlevel")):
# format() pads to the unused level's width
R_PADDED = ["x=1, f=a        ", "x=1, f=bb       ", "x=2, f=a        ", "x=2, f=bb       "]


def test_multi_variable_strata_pad_to_unused_factor_levels_everywhere():
    d = _padding_data()
    assert r.strata(d["x"], d["f"], labels=["x", "f"]).levels == R_PADDED
    # names(survfit(coxph(Surv(time, status) ~ z + strata(x, f), d))$strata)
    fit = r.coxph("Surv(time, status) ~ z + strata(x, f)", d)
    assert list(fit.strata_levels) == R_PADDED
    assert list(r.survfit(fit).strata) == R_PADDED
    # names(survreg(Surv(time, status) ~ z + strata(x, f), d)$scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(r.survreg("Surv(time, status) ~ z + strata(x, f)", d).strata_levels) == (
            R_PADDED
        )


def test_survreg_strata_keep_the_declared_level_order():
    pd = pytest.importorskip("pandas")
    d = _padding_data().assign(f=pd.Categorical(["z", "a", "z"] * 4, categories=["z", "a"]))
    # sr <- survreg(Surv(time, status) ~ z + strata(f), d); sr$scale
    fit = r.survreg("Surv(time, status) ~ z + strata(f)", d)
    assert fit.strata_levels == ("z", "a")
    assert fit.scale == pytest.approx([0.4701498230, 0.9827083501], rel=1e-8)


def _veteran():
    pd = pytest.importorskip("pandas")
    veteran = pd.DataFrame(
        {
            name: values
            for name, values in survival.datasets.load_veteran().items()
            if name[0] != "_"
        }
    )
    return veteran.assign(
        celltype=pd.Categorical(
            veteran["celltype"], categories=["squamous", "smallcell", "adeno", "large"]
        )
    )


# names(survfit(coxph(Surv(time, status) ~ karno + strata(trt) + strata(celltype), veteran))$strata)
R_VETERAN = [
    f"trt={trt}, {cell}" for trt in (1, 2) for cell in ("squamous", "smallcell", "adeno", "large")
]


def test_several_strata_terms_combine_with_short_labels():
    veteran = _veteran()
    formula = "Surv(time, status) ~ karno + strata(trt) + strata(celltype)"
    fit = r.coxph(formula, veteran)
    assert list(fit.strata_levels) == R_VETERAN
    assert list(r.survfit(fit).strata) == R_VETERAN
    assert fit.coefficients == pytest.approx([-0.03422661868], rel=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(r.survreg(formula, veteran).strata_levels) == R_VETERAN
    # predict(fit, newdata = veteran[1:5, ], type = "expected") codes the newdata strata the
    # same way, so a subset with fewer levels still finds its strata
    expected = r.predict(fit, newdata=veteran.iloc[:5], type="expected")
    assert expected == pytest.approx(
        [0.339696941328158, 2.273391499177269, 1.321029280105599]
        + [0.831148044015886, 0.487480766911646],
        rel=1e-10,
    )


def test_survreg_strata_terms_of_numeric_columns_are_not_padded():
    # names(survreg(Surv(time, status) ~ age + strata(sex) + strata(inst), lung)$scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = r.survreg(
            "Surv(time, status) ~ age + strata(sex) + strata(inst)", survival.datasets.load_lung()
        )
    assert fit.strata_levels[:3] == ("sex=1, inst=1", "sex=1, inst=2", "sex=1, inst=3")
    assert len(fit.strata_levels) == 36


class _CountingStrata:
    """``_core.strata``'s result, counting the reads of its ``levels`` getter (which copies
    every label per read)."""

    def __init__(self, result, reads):
        self._result = result
        self._reads = reads

    @property
    def levels(self):
        self._reads.append(1)
        return self._result.levels

    def __getattr__(self, name):
        return getattr(self._result, name)


def test_strata_labels_read_the_levels_once(monkeypatch):
    reads: list[int] = []
    kernel = r_surv._core.strata

    def counting(*args, **kwargs):
        return _CountingStrata(kernel(*args, **kwargs), reads)

    monkeypatch.setattr(r_surv._core, "strata", counting)
    ids = np.repeat(np.arange(500), 2)
    factor = r.strata(ids)
    assert len(factor.levels) == 500
    assert factor.labels[:3] == ["v1=0", "v1=0", "v1=1"]
    assert len(reads) == 1
    reads.clear()
    data = {
        "time": np.arange(1.0, 1001.0),
        "status": np.tile([1, 1, 0, 1], 250),
        "grp": np.tile([0, 1], 500),
        "gg": ids,
    }
    r.survdiff("Surv(time, status) ~ grp + strata(gg)", data)
    assert len(reads) == 2  # the strata() term and the group factor


def test_grouping_by_a_strata_term_keeps_its_level_order():
    # l <- na.omit(lung[, c("time", "status", "age")]); l$g <- ifelse(l$age > 62, 10, 2)
    # e <- survexp(~ strata(g), data = l, ratetable = coxph(Surv(time, status) ~ age, l))
    # names(e$strata); e$surv[5, ]
    lung = survival.datasets.load_lung()
    data = {name: np.array(lung[name], dtype=float) for name in ("time", "status", "age")}
    data["g"] = np.where(data["age"] > 62, 10.0, 2.0)
    fit = r.coxph("Surv(time, status) ~ age", data)
    expected = r.survexp("~ strata(g)", data=data, ratetable=fit)
    assert expected.strata == ["strata(g)=g=2", "strata(g)=g=10"]
    assert expected.surv[4] == pytest.approx([0.969944763136, 0.960538149447], rel=1e-10)
