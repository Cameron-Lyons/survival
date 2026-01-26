import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_survreg():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    covariates = [
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5],
    ]

    result = survival.survreg(
        time=time,
        status=status,
        covariates=covariates,
        weights=None,
        offsets=None,
        initial_beta=None,
        strata=None,
        distribution="extreme_value",
        max_iter=20,
        eps=1e-5,
        tol_chol=1e-9,
    )
    assert hasattr(result, "coefficients")
    assert hasattr(result, "log_likelihood")
    assert hasattr(result, "iterations")
    assert isinstance(result.coefficients, list)


def test_coxmart():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    result = survival.coxmart(
        time=time,
        status=status,
        score=score,
        weights=None,
        strata=None,
        method=0,
    )
    assert isinstance(result, list)
    assert len(result) == len(time)


def test_coxph_model():
    covariates = [
        [0.5, 1.2],
        [1.8, 0.3],
        [0.2, 2.1],
        [2.5, 0.8],
        [0.8, 1.5],
        [1.5, 0.5],
        [0.3, 1.8],
        [2.2, 1.1],
        [1.0, 0.9],
        [0.7, 1.7],
        [2.0, 0.4],
        [1.2, 1.3],
        [0.9, 2.0],
        [1.6, 0.7],
        [0.4, 1.4],
        [2.1, 1.0],
    ]
    event_times = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    censoring = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]

    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)
    model.fit(n_iters=20)

    assert hasattr(model, "baseline_hazard")
    assert hasattr(model, "risk_scores")

    coefficients = model.coefficients
    assert coefficients is not None

    new_covariates = [[1.0, 2.0], [2.0, 3.0]]
    predictions = model.predict(new_covariates)
    assert isinstance(predictions, list)

    brier = model.brier_score()
    assert isinstance(brier, float)


def test_subject():
    subject = survival.Subject(
        id=1,
        covariates=[1.0, 2.0],
        is_case=True,
        is_subcohort=True,
        stratum=0,
    )
    assert subject.id == 1
    assert subject.is_case is True
