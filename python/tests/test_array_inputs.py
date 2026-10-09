import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

HAS_PANDAS = False
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pass

HAS_POLARS = False
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pass


# survfit(Surv(1:5, c(1, 1, 0, 1, 0)) ~ 1)$surv
_KM_TIME = [1.0, 2.0, 3.0, 4.0, 5.0]
_KM_STATUS = [1, 1, 0, 1, 0]
_KM_SURV = [0.8, 0.6, 0.6, 0.3, 0.3]
_KM_WEIGHTS = [1.0, 1.0, 2.0, 1.0, 1.5]
# survfit(..., weights = c(1, 1, 2, 1, 1.5))$surv
_KM_WEIGHTED_SURV = [11 / 13, 9 / 13, 9 / 13, 5.4 / 13, 5.4 / 13]


def test_survfitkm_with_lists():
    result = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS)
    assert result.surv == pytest.approx(_KM_SURV)


def _unaligned_array(values, dtype=np.float64):
    source = np.asarray(values, dtype=dtype)
    array = np.ndarray(source.shape, dtype=dtype, buffer=bytearray(source.nbytes + 1), offset=1)
    array[:] = source
    assert not array.flags.aligned
    return array


@pytest.mark.parametrize("status_dtype", [np.int32, np.int64, np.bool_])
def test_survfitkm_unaligned_vectors_and_noncanonical_boolean_status(status_dtype):
    times = _unaligned_array(_KM_TIME)
    weights = _unaligned_array(_KM_WEIGHTS)
    if status_dtype == np.bool_:
        # NumPy permits any nonzero byte as True, including 2 and 255.
        status = np.frombuffer(bytearray([2, 255, 0, 2, 0]), dtype=np.bool_)
    else:
        status = _unaligned_array(_KM_STATUS, status_dtype)
    for selection in [slice(None), slice(None, None, -1), slice(None, None, 2)]:
        actual = survival.surv_analysis.survfitkm(
            times[selection], status[selection], weights=weights[selection]
        )
        expected = survival.surv_analysis.survfitkm(
            np.array(times[selection]),
            np.array(status[selection], dtype=np.int32),
            weights=np.array(weights[selection]),
        )
        assert actual.time == expected.time
        assert actual.surv == pytest.approx(expected.surv)
        assert actual.std_err == pytest.approx(expected.std_err)


@pytest.mark.parametrize("selection", [slice(None), slice(None, None, -1), slice(None, None, 2)])
def test_unaligned_matrices_for_curve_aggregation_and_yates(selection):
    matrix = _unaligned_array([[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]])[selection]
    actual = survival.surv_analysis.aggregate_survfit(surv=matrix, fun="median")
    expected = survival.surv_analysis.aggregate_survfit(surv=np.array(matrix), fun="median")
    assert actual.surv == expected.surv

    variance = _unaligned_array([[1.0, 0.25], [0.25, 2.0]])
    actual_yates = survival.validation.yates(matrix, [1.0, 0.5], variance)
    expected_yates = survival.validation.yates(np.array(matrix), [1.0, 0.5], np.array(variance))
    assert actual_yates.cmat == expected_yates.cmat
    assert actual_yates.mvar == expected_yates.mvar
    assert [row.pmm for row in actual_yates.estimate] == pytest.approx(
        [row.pmm for row in expected_yates.estimate]
    )
    assert [row.std for row in actual_yates.estimate] == pytest.approx(
        [row.std for row in expected_yates.estimate]
    )


def test_finegray_normalizes_numpy_boolean_storage_and_unaligned_numeric_flags():
    extend = np.frombuffer(bytearray([2, 0, 255]), dtype=np.bool_)
    keep = np.frombuffer(bytearray([2, 0, 255, 2]), dtype=np.bool_)
    for flags in [
        (extend, keep),
        (extend[::-1], keep[::-1]),
        (_unaligned_array(extend, np.int32), _unaligned_array(keep)),
    ]:
        actual = survival.regression.finegray(
            [0, 0, 0], [1, 2, 3], [1, 2, 3, 4], [1, 0.8, 0.6, 0.4], *flags
        )
        expected = survival.regression.finegray(
            [0, 0, 0],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 0.8, 0.6, 0.4],
            *(value.astype(np.int32).astype(bool) for value in flags),
        )
        assert actual.row == expected.row
        assert actual.add == expected.add
        assert actual.start == expected.start
        assert actual.end == expected.end
        assert actual.wt == pytest.approx(expected.wt)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.bool_, ">f8"])
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "strided", "transpose", "broadcast"])
def test_aggregate_survfit_pstate_numpy_layouts_and_dtypes(dtype, layout):
    values = np.arange(60).reshape(4, 5, 3).astype(dtype)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "reverse":
        values = values[::-1, ::-1, ::-1]
    elif layout == "strided":
        values = np.repeat(values, 2, axis=1)[:, ::2, :]
    elif layout == "transpose":
        values = np.ascontiguousarray(values.transpose(2, 1, 0)).transpose(2, 1, 0)
    elif layout == "broadcast":
        values = np.broadcast_to(values[:1], (4, 5, 3))
    by = [survival.surv_analysis.GroupingFactor([0, 1, 0, 1, 0], ["a", "b"])]
    for fun in ["mean", "median", "min", "max"]:
        actual = survival.surv_analysis.aggregate_survfit(pstate=values, by=by, fun=fun)
        reference = survival.surv_analysis.aggregate_survfit(
            pstate=values.astype(float).tolist(), by=by, fun=fun
        )
        np.testing.assert_allclose(actual.pstate, reference.pstate, rtol=0, atol=1e-14)
        assert actual.newdata.labels == reference.newdata.labels


def test_aggregate_survfit_pstate_unaligned_storage_and_array_protocol():
    values = np.ndarray((2, 3, 4), dtype=np.float64, buffer=bytearray(193), offset=1)
    values[:] = np.arange(24).reshape(2, 3, 4)
    assert not values.flags.aligned

    class ArrayProtocol:
        def __array__(self, dtype=None, copy=None):
            return np.array(values, dtype=dtype, copy=copy)

    for input_ in [values, ArrayProtocol()]:
        result = survival.surv_analysis.aggregate_survfit(pstate=input_, fun="median")
        np.testing.assert_allclose(result.pstate, np.median(values, axis=1)[:, None])


@pytest.mark.parametrize(
    ("values", "message", "error_type"),
    [
        (np.ones((2, 3)), "3-dimensional.*2 dimension", TypeError),
        (np.ones((2, 3, 4, 1)), "3-dimensional.*4 dimension", TypeError),
        ([[[1, 2]], [[3, 4], [5, 6]]], "time 1 length mismatch", ValueError),
        ([[[1, 2], [3]]], "data 1 length mismatch", ValueError),
        ([[1, 2]], "3-dimensional.*state row", TypeError),
        ([[[object()]]], "float array value", TypeError),
    ],
)
def test_aggregate_survfit_pstate_rejects_wrong_dimensions_and_ragged_rows(
    values, message, error_type
):
    with pytest.raises(error_type, match=message):
        survival.surv_analysis.aggregate_survfit(pstate=values)


def test_aggregate_survfit_pstate_preserves_empty_sequence_and_array_shapes():
    for values in [[], [[], []], np.empty((2, 0, 3))]:
        with pytest.raises(ValueError, match="data.*margin"):
            survival.surv_analysis.aggregate_survfit(pstate=values)
    empty_times = survival.surv_analysis.aggregate_survfit(pstate=np.empty((0, 3, 2)))
    assert empty_times.pstate == []
    for values in [[[[], []]], np.empty((1, 2, 0))]:
        empty_states = survival.surv_analysis.aggregate_survfit(pstate=values)
        assert empty_states.pstate == [[[]]]


def test_survfitkm_with_numpy():
    result = survival.surv_analysis.survfitkm(np.array(_KM_TIME), np.array(_KM_STATUS))
    assert result.surv == pytest.approx(_KM_SURV)

    strided = survival.surv_analysis.survfitkm(np.array(_KM_TIME)[::2], np.array(_KM_STATUS)[::2])
    assert strided.surv == pytest.approx([2 / 3, 2 / 3, 2 / 3])


@pytest.mark.parametrize("estimator", ["survfitkm", "nelson_aalen"])
def test_survival_curve_status_must_be_integral(estimator):
    fit = getattr(survival.surv_analysis, estimator)
    # Integral float arrays follow the same checked conversion as other
    # native estimators; fractional status values must never be truncated.
    result = fit(np.array(_KM_TIME), np.array(_KM_STATUS, dtype=float))
    assert result.time
    with pytest.raises(TypeError):
        fit(np.array(_KM_TIME), np.array([1.0, 0.5, 0.0, 1.0, 0.0]))
    with pytest.raises(TypeError):
        fit(_KM_TIME, np.array([1, 2**40, 0, 1, 0]))


@pytest.mark.parametrize("estimator", ["survfitkm", "nelson_aalen"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_survival_curve_strided_weighted_arrays(estimator, dtype):
    fit = getattr(survival.surv_analysis, estimator)
    expected = fit(_KM_TIME, _KM_STATUS, weights=_KM_WEIGHTS)
    actual = fit(
        np.array(_KM_TIME, dtype=dtype)[::-1],
        np.array(_KM_STATUS, dtype=bool)[::-1],
        weights=np.array(_KM_WEIGHTS, dtype=dtype)[::-1],
    )
    assert actual.time == expected.time
    if estimator == "survfitkm":
        assert actual.surv == pytest.approx(expected.surv)
        assert actual.std_err == pytest.approx(expected.std_err)
    else:
        assert actual.cumulative_hazard == pytest.approx(expected.cumulative_hazard)
        assert actual.variance == pytest.approx(expected.variance)


@pytest.mark.parametrize("estimator", ["survfitkm", "nelson_aalen"])
def test_survival_curve_rejects_matrix_inputs(estimator):
    with pytest.raises(TypeError, match="dimension"):
        getattr(survival.surv_analysis, estimator)(np.array([_KM_TIME]), _KM_STATUS)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_series():
    df = pd.DataFrame({"time": _KM_TIME, "status": _KM_STATUS})
    result = survival.surv_analysis.survfitkm(df["time"], df["status"])
    assert result.surv == pytest.approx(_KM_SURV)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_values():
    df = pd.DataFrame({"time": _KM_TIME, "status": _KM_STATUS})
    result = survival.surv_analysis.survfitkm(df["time"].values, df["status"].values)
    assert result.surv == pytest.approx(_KM_SURV)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_survfitkm_with_polars():
    df = pl.DataFrame({"time": _KM_TIME, "status": _KM_STATUS})
    result = survival.surv_analysis.survfitkm(df["time"], df["status"])
    assert result.surv == pytest.approx(_KM_SURV)


def test_logrank_with_lists():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    result = survival.validation.logrank_test(time_list, status_list, group_list)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")


def test_logrank_accepts_rho_keyword():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    logrank = survival.validation.logrank_test(time_list, status_list, group_list)
    peto = survival.validation.logrank_test(time_list, status_list, group_list, rho=1.0)

    assert logrank.rho == pytest.approx(0.0)
    assert peto.rho == pytest.approx(1.0)
    assert peto.statistic != pytest.approx(logrank.statistic)
    assert peto.statistic == pytest.approx(
        survival.surv_analysis.survdiff(time_list, status_list, group_list, rho=1.0).chisq
    )


@pytest.mark.parametrize("rho", [0.0, 1.0])
@pytest.mark.parametrize("timefix", [False, True])
@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("stratified", [False, True])
def test_logrank_owned_inputs_preserve_optional_arguments(rho, timefix, counting, stratified):
    time = np.array([1, 2, 2 + 5e-9, 3, 4, 4 + 5e-9, 5, 6, 7, 8, 9, 10])
    status = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int32)
    group = np.array([9, 4, -2, 4, 9, -2, 4, -2, 9, 4, -2, 9], dtype=np.int32)
    start = np.maximum(time - 3.5, 0) if counting else None
    strata = np.arange(len(time), dtype=np.int32) % 2 if stratified else None
    before = [
        None if value is None else value.copy() for value in (time, status, group, start, strata)
    ]
    actual = survival.validation.logrank_test(
        time, status, group, rho=rho, timefix=timefix, entry_times=start, strata=strata
    )
    expected = survival.surv_analysis.survdiff(
        time, status, group, rho=rho, timefix=timefix, start=start, strata=strata
    )
    assert actual.groups == expected.group_codes
    np.testing.assert_array_equal(actual.observed, np.sum(expected.obs, axis=1))
    np.testing.assert_array_equal(actual.expected, np.sum(expected.exp, axis=1))
    np.testing.assert_array_equal(actual.variance, expected.var)
    assert actual.statistic == expected.chisq
    assert actual.p_value == expected.pvalue
    assert actual.df == expected.df
    assert actual.rho == rho
    for value, original in zip((time, status, group, start, strata), before, strict=True):
        if value is not None:
            np.testing.assert_array_equal(value, original)


def test_logrank_with_numpy():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    time_np = np.array(time_list)
    status_np = np.array(status_list, dtype=np.int32)
    group_np = np.array(group_list, dtype=np.int32)
    result = survival.validation.logrank_test(time_np, status_np, group_np)
    assert hasattr(result, "statistic")


def test_logrank_with_numpy_int64():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    time_np = np.array(time_list)
    status_np64 = np.array(status_list, dtype=np.int64)
    group_np64 = np.array(group_list, dtype=np.int64)
    result = survival.validation.logrank_test(time_np, status_np64, group_np64)
    assert hasattr(result, "statistic")


def test_logrank_with_strided_numpy_arrays():
    time = np.array([1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0, 5.0, -1.0])
    status = np.array([1, -1, 1, -1, 0, -1, 1, -1, 0, -1], dtype=np.int32)
    group = np.array([0, -1, 0, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)

    result = survival.validation.logrank_test(time[::2], status[::2], group[::2])

    assert hasattr(result, "statistic")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_logrank_with_pandas():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    df = pd.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.validation.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic")


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_logrank_with_polars():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    df = pl.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.validation.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic")


def test_cv_cox_concordance_with_lists():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    result = survival.validation.cv_cox_concordance(time_list, status_list, covariates, n_folds=2)
    assert hasattr(result, "mean_score")


def test_cv_cox_concordance_with_numpy():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    time_np = np.array(time_list)
    status_np = np.array(status_list, dtype=np.int32)
    result = survival.validation.cv_cox_concordance(time_np, status_np, covariates, n_folds=2)
    assert hasattr(result, "mean_score")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_cv_cox_concordance_with_pandas():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.validation.cv_cox_concordance(df["time"], df["status"], covariates, n_folds=2)
    assert hasattr(result, "mean_score")


def test_crossval_wrappers_validate_inputs_and_survreg_covariates():
    time = [float(i) for i in range(1, 31)]
    status_i32 = [1 if i % 3 != 0 else 0 for i in range(30)]
    status_f64 = [float(value) for value in status_i32]
    covariates = [[i / 30.0] for i in range(30)]

    cox = survival.validation.cv_cox_concordance(time, status_i32, covariates, n_folds=3, seed=42)
    survreg = survival.validation.cv_survreg_loglik(
        time, status_f64, covariates, "weibull", 3, True, 42
    )

    assert len(cox.fold_scores) == 3
    assert len(survreg.fold_scores) == 3
    assert all(len(coefficients) > 0 for coefficients in survreg.fold_coefficients)

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        survival.validation.cv_cox_concordance(time, status_i32, covariates, n_folds=1)

    with pytest.raises(ValueError, match="time and status must have the same non-zero length"):
        survival.validation.cv_cox_concordance(time, status_i32[:-1], covariates, n_folds=3)

    bad_time = list(time)
    bad_time[1] = float("nan")
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.validation.cv_cox_concordance(bad_time, status_i32, covariates, n_folds=3)

    bad_status = list(status_i32)
    bad_status[2] = 2
    with pytest.raises(ValueError, match="status must contain only 0/1 values"):
        survival.validation.cv_cox_concordance(time, bad_status, covariates, n_folds=3)

    with pytest.raises(ValueError, match="covariates length must match time length"):
        survival.validation.cv_survreg_loglik(
            time, status_f64, covariates[:-1], "weibull", 3, True, 42
        )

    ragged_covariates = [row[:] for row in covariates]
    ragged_covariates[3] = [0.1, 0.2]
    with pytest.raises(ValueError, match="covariates row 3 length"):
        survival.validation.cv_survreg_loglik(
            time, status_f64, ragged_covariates, "weibull", 3, True, 42
        )

    bad_weights = [1.0] * 29 + [float("inf")]
    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.validation.cv_cox_concordance(
            time, status_i32, covariates, weights=bad_weights, n_folds=3
        )


def test_survfitkm_with_numpy_weights():
    result = survival.surv_analysis.survfitkm(
        np.array(_KM_TIME), np.array(_KM_STATUS), weights=np.array(_KM_WEIGHTS)
    )
    assert result.surv == pytest.approx(_KM_WEIGHTED_SURV)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_weights():
    df = pd.DataFrame({"time": _KM_TIME, "status": _KM_STATUS, "weights": _KM_WEIGHTS})
    result = survival.surv_analysis.survfitkm(df["time"], df["status"], weights=df["weights"])
    assert result.surv == pytest.approx(_KM_WEIGHTED_SURV)
