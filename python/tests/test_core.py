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
