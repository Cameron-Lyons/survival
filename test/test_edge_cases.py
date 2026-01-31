import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_survfitkm_empty_input():
    with pytest.raises(ValueError, match=".*"):
        survival.survfitkm(time=[], status=[])


def test_survfitkm_length_mismatch():
    with pytest.raises(ValueError, match=".*"):
        survival.survfitkm(time=[1.0, 2.0], status=[1.0])


def test_survfitkm_negative_time():
    with pytest.raises(ValueError, match=".*"):
        survival.survfitkm(time=[-1.0, 2.0], status=[1.0, 0.0])


def test_agmart_length_mismatch():
    with pytest.raises(ValueError, match=".*"):
        survival.agmart(
            n=3,
            method=0,
            start=[0.0, 0.0],
            stop=[1.0, 2.0, 3.0],
            event=[1, 0, 1],
            score=[1.0, 1.0, 1.0],
            wt=[1.0, 1.0, 1.0],
            strata=[0, 0, 0],
        )


def test_coxph_model_empty_fit():
    model = survival.CoxPHModel()
    with pytest.raises(ValueError, match=".*"):
        model.fit(n_iters=10)


def test_coxph_model_covariate_dimension_mismatch():
    subject = survival.Subject(
        id=1,
        covariates=[1.0, 2.0, 3.0],
        is_case=True,
        is_subcohort=True,
        stratum=0,
    )

    covariates = [[1.0, 2.0], [2.0, 3.0]]
    event_times = [1.0, 2.0]
    censoring = [1, 0]
    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)

    with pytest.raises(ValueError, match=".*"):
        model.add_subject(subject)


def test_survdiff2_with_strata():
    result = survival.survdiff2(
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        status=[1, 0, 1, 0, 1, 0],
        group=[1, 1, 1, 2, 2, 2],
        strata=[0, 0, 0, 0, 0, 0],
        rho=0.0,
    )
    assert hasattr(result, "chi_squared")


def test_all_censored():
    result = survival.survfitkm(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[0.0, 0.0, 0.0, 0.0],
    )
    assert all(e == 1.0 for e in result.estimate)


def test_single_observation():
    result = survival.survfitkm(
        time=[5.0],
        status=[1.0],
    )
    assert hasattr(result, "time")
    assert hasattr(result, "estimate")
