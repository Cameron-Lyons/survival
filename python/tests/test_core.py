import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_coxcount1():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    strata = [1, 0, 0, 0, 0, 0, 0, 0]

    result = survival.coxcount1(time, status, strata)
    assert hasattr(result, "time")
    assert hasattr(result, "nrisk")
    assert hasattr(result, "index")
    assert hasattr(result, "status")
    assert len(result.time) > 0


def test_coxcount1_validates_public_inputs():
    with pytest.raises(ValueError, match="status length"):
        survival.coxcount1([1.0], [1.0, 0.0], [1])

    with pytest.raises(ValueError, match="finite"):
        survival.coxcount1([float("nan")], [1.0], [1])

    with pytest.raises(ValueError, match="status values"):
        survival.coxcount1([1.0], [2.0], [1])

    with pytest.raises(ValueError, match="strata values"):
        survival.coxcount1([1.0], [1.0], [2])


def test_coxcount2():
    time1 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    time2 = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    status = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    sort1 = [0, 2, 4, 1, 3, 5]
    sort2 = [0, 2, 4, 1, 3, 5]
    strata2 = [1, 0, 0, 0, 0, 0]

    result2 = survival.coxcount2(time1, time2, status, sort1, sort2, strata2)
    assert hasattr(result2, "time")
    assert hasattr(result2, "nrisk")


def test_coxcount2_validates_public_inputs():
    with pytest.raises(ValueError, match="time2 length"):
        survival.coxcount2([0.0], [], [1.0], [0], [0], [1])

    with pytest.raises(ValueError, match="finite"):
        survival.coxcount2(
            [0.0],
            [float("inf")],
            [1.0],
            [0],
            [0],
            [1],
        )

    with pytest.raises(ValueError, match="status values"):
        survival.coxcount2([0.0], [1.0], [0.5], [0], [0], [1])

    with pytest.raises(ValueError, match="sort1 index out of bounds"):
        survival.coxcount2([0.0], [1.0], [1.0], [1], [0], [1])

    with pytest.raises(ValueError, match="sort1 must be a permutation"):
        survival.coxcount2(
            [0.0, 1.0],
            [1.0, 2.0],
            [1.0, 0.0],
            [0, 0],
            [0, 1],
            [1, 0],
        )


def test_norisk_validates_public_inputs():
    with pytest.raises(ValueError, match="strata values must be strictly increasing"):
        survival.norisk(
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [1, 0, 1],
            [0, 1, 2],
            [0, 1, 2],
            [2, 1],
        )


def test_score_calculation_public_api():
    result = survival.perform_score_calculation(
        time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
        covariates=[0.5, 1.0, 1.5],
        strata=[0, 0, 0],
        score=[1.0, 1.0, 1.0],
        weights=[1.0, 1.0, 1.0],
        method=0,
    )

    assert result["n_observations"] == 3
    assert result["n_variables"] == 1
    assert result["method"] == "breslow"
    assert len(result["residuals"]) == 3
    assert len(result["summary_stats"]) == 2


def test_score_calculation_validates_public_inputs():
    time_data = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0]
    covariates = [0.5, 1.0, 1.5]
    score = [1.0, 1.0, 1.0]
    weights = [1.0, 1.0, 1.0]

    result = survival.perform_score_calculation(
        time_data=time_data,
        covariates=covariates,
        strata=[2, 2, 4],
        score=score,
        weights=weights,
        method=0,
    )
    assert result["n_observations"] == 3

    with pytest.raises(ValueError, match="method must be 0"):
        survival.perform_score_calculation(
            time_data=time_data,
            covariates=covariates,
            strata=[0, 0, 0],
            score=score,
            weights=weights,
            method=2,
        )

    bad_time = time_data.copy()
    bad_time[0] = float("nan")
    with pytest.raises(ValueError, match="time_data contains non-finite"):
        survival.perform_score_calculation(
            time_data=bad_time,
            covariates=covariates,
            strata=[0, 0, 0],
            score=score,
            weights=weights,
            method=0,
        )

    bad_event = time_data.copy()
    bad_event[7] = 0.5
    with pytest.raises(ValueError, match="event values"):
        survival.perform_score_calculation(
            time_data=bad_event,
            covariates=covariates,
            strata=[0, 0, 0],
            score=score,
            weights=weights,
            method=0,
        )

    bad_score = score.copy()
    bad_score[1] = -1.0
    with pytest.raises(ValueError, match="score contains negative value"):
        survival.perform_score_calculation(
            time_data=time_data,
            covariates=covariates,
            strata=[0, 0, 0],
            score=bad_score,
            weights=weights,
            method=0,
        )

    bad_weights = weights.copy()
    bad_weights[2] = -1.0
    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.perform_score_calculation(
            time_data=time_data,
            covariates=covariates,
            strata=[0, 0, 0],
            score=score,
            weights=bad_weights,
            method=0,
        )


def test_agscore3_public_api_and_validation():
    result = survival.perform_agscore3_calculation(
        time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
        covariates=[0.5, 1.0, 1.5],
        strata=[0, 0, 0],
        score=[1.0, 1.0, 1.0],
        weights=[1.0, 1.0, 1.0],
        method=0,
        sort1=[1, 2, 3],
    )

    assert result["method"] == "breslow"
    assert len(result["residuals"]) == 3

    with pytest.raises(RuntimeError, match="Sort1 length does not match observations"):
        survival.perform_agscore3_calculation(
            time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
            covariates=[0.5, 1.0, 1.5],
            strata=[0, 0, 0],
            score=[1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0],
            method=0,
            sort1=[1, 2],
        )

    for bad_sort1 in ([1, 0, 3], [1, 4, 3]):
        with pytest.raises(RuntimeError, match="outside 1..=3"):
            survival.perform_agscore3_calculation(
                time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                covariates=[0.5, 1.0, 1.5],
                strata=[0, 0, 0],
                score=[1.0, 1.0, 1.0],
                weights=[1.0, 1.0, 1.0],
                method=0,
                sort1=bad_sort1,
            )

    with pytest.raises(RuntimeError, match="Sort1 must be a permutation"):
        survival.perform_agscore3_calculation(
            time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
            covariates=[0.5, 1.0, 1.5],
            strata=[0, 0, 0],
            score=[1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0],
            method=0,
            sort1=[1, 1, 3],
        )

    with pytest.raises(ValueError, match="event values"):
        survival.perform_agscore3_calculation(
            time_data=[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.0],
            covariates=[0.5, 1.0, 1.5],
            strata=[0, 0, 0],
            score=[1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0],
            method=0,
            sort1=[1, 2, 3],
        )


def test_cox_score_residuals_public_api():
    breslow = survival.cox_score_residuals(
        y=[1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0],
        strata=[0, 0, 0, 0],
        covar=[1.0, 2.0, 3.0, 4.0],
        score=[1.0, 1.0, 1.0, 1.0],
        weights=[1.0, 1.0, 1.0, 1.0],
        nvar=1,
        method=0,
    )
    efron = survival.cox_score_residuals(
        y=[1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0],
        strata=[0, 0, 0, 0],
        covar=[1.0, 2.0, 3.0, 4.0],
        score=[1.0, 1.0, 1.0, 1.0],
        weights=[1.0, 1.0, 1.0, 1.0],
        nvar=1,
        method=1,
    )

    assert len(breslow) == 4
    assert len(efron) == 4
    assert any(abs(a - b) > 1e-12 for a, b in zip(breslow, efron, strict=True))

    with pytest.raises(ValueError, match="y array must have length >= 2 \\* n"):
        survival.cox_score_residuals(
            y=[1.0, 2.0],
            strata=[0, 0, 0],
            covar=[1.0, 2.0, 3.0],
            score=[1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0],
            nvar=1,
        )


def test_cox_score_residuals_validates_public_inputs():
    y = [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0]
    strata = [2, 2, 4, 4]
    covar = [1.0, 2.0, 3.0, 4.0]
    score = [1.0, 1.0, 1.0, 1.0]
    weights = [1.0, 1.0, 1.0, 1.0]

    assert len(survival.cox_score_residuals(y, strata, covar, score, weights, 1, 0)) == 4

    with pytest.raises(ValueError, match="method must be 0"):
        survival.cox_score_residuals(y, strata, covar, score, weights, 1, 2)

    y_with_nan = y.copy()
    y_with_nan[0] = float("nan")
    with pytest.raises(ValueError, match="y contains non-finite"):
        survival.cox_score_residuals(y_with_nan, strata, covar, score, weights, 1, 0)

    y_with_bad_status = y.copy()
    y_with_bad_status[5] = 0.5
    with pytest.raises(ValueError, match="status values"):
        survival.cox_score_residuals(y_with_bad_status, strata, covar, score, weights, 1, 0)

    score_with_negative = score.copy()
    score_with_negative[1] = -1.0
    with pytest.raises(ValueError, match="score contains negative value"):
        survival.cox_score_residuals(y, strata, covar, score_with_negative, weights, 1, 0)

    weights_with_negative = weights.copy()
    weights_with_negative[2] = -1.0
    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.cox_score_residuals(y, strata, covar, score, weights_with_negative, 1, 0)


def test_schoenfeld_residuals_validates_public_inputs():
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.0, 1.0]
    score = [1.0, 1.0, 1.0, 1.0]
    strata = [0, 0, 0, 0]
    covar = [1.0, 2.0, 3.0, 4.0]

    assert survival.schoenfeld_residuals(y, score, strata, covar, 1, 0) == pytest.approx(
        [-1.5, -1.0, 3.0, 0.0]
    )

    with pytest.raises(ValueError, match="method must be 0"):
        survival.schoenfeld_residuals(y, score, strata, covar, 1, 2)

    y_with_nan = y.copy()
    y_with_nan[1] = float("nan")
    with pytest.raises(ValueError, match="y contains non-finite"):
        survival.schoenfeld_residuals(y_with_nan, score, strata, covar, 1, 0)

    y_with_bad_event = y.copy()
    y_with_bad_event[9] = 0.5
    with pytest.raises(ValueError, match="event values"):
        survival.schoenfeld_residuals(y_with_bad_event, score, strata, covar, 1, 0)

    score_with_negative = score.copy()
    score_with_negative[2] = -1.0
    with pytest.raises(ValueError, match="score contains negative value"):
        survival.schoenfeld_residuals(y, score_with_negative, strata, covar, 1, 0)

    bad_strata = strata.copy()
    bad_strata[0] = 2
    with pytest.raises(ValueError, match="strata values"):
        survival.schoenfeld_residuals(y, score, bad_strata, covar, 1, 0)
