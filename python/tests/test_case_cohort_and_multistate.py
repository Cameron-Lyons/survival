import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _matched_clogit_dataset():
    dataset = survival.ClogitDataSet()
    for case_status, stratum, covariates in [
        (1, 0, [2.0]),
        (0, 0, [1.0]),
        (1, 1, [3.0]),
        (0, 1, [1.0]),
    ]:
        dataset.add_observation(case_status, stratum, covariates)
    return dataset


def _simple_subject(subject_id, covariate, *, is_case, is_subcohort):
    return survival.Subject(
        id=subject_id,
        covariates=[covariate],
        is_case=is_case,
        is_subcohort=is_subcohort,
        stratum=0,
    )


def _survfitaj_kwargs(*, sefit):
    return {
        "y": [0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0],
        "sort1": [0, 1, 2],
        "sort2": [0, 1, 2],
        "utime": [1.0, 2.0],
        "cstate": [0, 0, 0],
        "wt": [1.0, 1.0, 1.0],
        "grp": [0, 0, 0],
        "ngrp": 1,
        "p0": [1.0, 0.0],
        "i0": [0.0, 0.0],
        "sefit": sefit,
        "entry": False,
        "position": [2, 2, 2],
        "hindx": [[0]],
        "trmat": [[0, 1]],
        "t0": 0.0,
    }


def test_clogit_dataset_tracks_observations_and_covariates():
    dataset = _matched_clogit_dataset()

    assert dataset.get_num_observations() == 4
    assert dataset.get_num_covariates() == 1


def test_conditional_logistic_regression_fit_populates_outputs():
    model = survival.ConditionalLogisticRegression(_matched_clogit_dataset(), max_iter=20, tol=1e-9)
    model.fit()

    assert len(model.coefficients) == 1
    assert 1 <= model.iterations <= 20
    assert isinstance(model.converged, bool)
    assert model.coefficients[0] > 0.0
    assert model.predict([2.0]) > model.predict([1.0])
    assert model.odds_ratios()[0] == pytest.approx(math.exp(model.coefficients[0]))


def test_conditional_logistic_regression_is_invariant_to_row_order():
    forward = survival.ConditionalLogisticRegression(
        _matched_clogit_dataset(), max_iter=20, tol=1e-9
    )

    reversed_dataset = survival.ClogitDataSet()
    for case_status, stratum, covariates in reversed(
        [
            (1, 0, [2.0]),
            (0, 0, [1.0]),
            (1, 1, [3.0]),
            (0, 1, [1.0]),
        ]
    ):
        reversed_dataset.add_observation(case_status, stratum, covariates)
    backward = survival.ConditionalLogisticRegression(reversed_dataset, max_iter=20, tol=1e-9)

    forward.fit()
    backward.fit()

    assert backward.coefficients == pytest.approx(forward.coefficients)


def test_conditional_logistic_regression_uses_strata():
    pooled_dataset = survival.ClogitDataSet()
    for case_status, _stratum, covariates in [
        (1, 0, [5.0]),
        (0, 0, [0.0]),
        (1, 1, [1.0]),
        (0, 1, [-4.0]),
    ]:
        pooled_dataset.add_observation(case_status, 0, covariates)

    matched_dataset = survival.ClogitDataSet()
    for case_status, stratum, covariates in [
        (1, 0, [5.0]),
        (0, 0, [0.0]),
        (1, 1, [1.0]),
        (0, 1, [-4.0]),
    ]:
        matched_dataset.add_observation(case_status, stratum, covariates)

    matched = survival.ConditionalLogisticRegression(matched_dataset, max_iter=20, tol=1e-9)
    pooled = survival.ConditionalLogisticRegression(pooled_dataset, max_iter=20, tol=1e-9)

    matched.fit()
    pooled.fit()

    assert matched.coefficients[0] != pytest.approx(pooled.coefficients[0])


def test_case_cohort_prentice_accepts_public_enum_value():
    cohort = survival.CohortData.new()
    cohort.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=True))
    cohort.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))

    fitted = cohort.fit(survival.CchMethod.Prentice, max_iter=5)

    assert hasattr(fitted, "coefficients")
    assert len(fitted.coefficients) == 1


@pytest.mark.parametrize(
    "method",
    [
        survival.CchMethod.SelfPrentice,
        survival.CchMethod.LinYing,
        survival.CchMethod.IBorgan,
        survival.CchMethod.IIBorgan,
    ],
)
def test_case_cohort_unimplemented_methods_raise(method):
    cohort = survival.CohortData.new()
    cohort.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=True))
    cohort.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))

    with pytest.raises(NotImplementedError, match="only Prentice is currently supported"):
        cohort.fit(method, max_iter=5)


def test_case_cohort_fit_ignores_noncase_outside_subcohort_subjects():
    base = survival.CohortData.new()
    base.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=True))
    base.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))

    with_extra = survival.CohortData.new()
    with_extra.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=True))
    with_extra.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))
    with_extra.add_subject(_simple_subject(3, 10.0, is_case=False, is_subcohort=False))

    base_fit = base.fit(survival.CchMethod.Prentice, max_iter=5)
    extra_fit = with_extra.fit(survival.CchMethod.Prentice, max_iter=5)

    assert extra_fit.coefficients == base_fit.coefficients


def test_case_cohort_fit_includes_cases_outside_subcohort():
    cohort = survival.CohortData.new()
    cohort.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=False))
    cohort.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))

    fitted = cohort.fit(survival.CchMethod.Prentice, max_iter=5)

    assert len(fitted.risk_scores) == 2


def test_cohort_data_returns_added_subject():
    cohort = survival.CohortData.new()
    cohort.add_subject(_simple_subject(7, 1.5, is_case=True, is_subcohort=True))

    subject = cohort.get_subject(0)

    assert subject.id == 7
    assert subject.covariates == [1.5]
    assert subject.is_subcohort is True


def test_survfitaj_basic_outputs_are_consistent():
    result = survival.survfitaj(**_survfitaj_kwargs(sefit=0))

    assert len(result.pstate) == 2
    assert result.n_enter is None
    assert result.std_err is None
    assert result.influence is None
    assert result.pstate[0] == pytest.approx([1.0, 0.0])
    assert result.pstate[1] == pytest.approx([0.5, 0.5])
    assert result.cumhaz[0][0] <= result.cumhaz[1][0]
    assert sum(result.pstate[0]) == pytest.approx(1.0)
    assert sum(result.pstate[1]) == pytest.approx(1.0)
    assert result.n_censor[1][0] == pytest.approx(1.0)
    assert result.n_transition[1] == pytest.approx([1.0, 1.0])


def test_survfitaj_returns_standard_errors_when_requested():
    result = survival.survfitaj(**_survfitaj_kwargs(sefit=1))

    assert result.std_err is not None
    assert result.std_chaz is not None
    assert result.std_auc is not None
    assert result.influence is None
    assert len(result.std_err) == len(result.pstate)
    assert len(result.std_err[0]) == len(result.pstate[0])
    assert len(result.std_chaz[0]) == len(result.cumhaz[0])


def test_survfitaj_rejects_ragged_hindx():
    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["hindx"] = [[0, 1], [2]]

    with pytest.raises(ValueError, match="Invalid hindx array"):
        survival.survfitaj(**kwargs)
