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
        "y": [0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0],
        "sort1": [0, 1, 2],
        "sort2": [0, 1, 2],
        "utime": [1.0, 2.0],
        "cstate": [0, 0, 0],
        "wt": [1.0, 1.0, 1.0],
        "grp": [0, 1, 2],
        "ngrp": 3,
        "p0": [1.0, 0.0],
        "i0": [0.0] * 6,
        "sefit": sefit,
        "entry": False,
        "position": [3, 3, 3],
        "hindx": [[1, 0], [1, 1]],
        "trmat": [[0, 1]],
        "t0": 0.0,
    }


def _competing_survfitaj_kwargs(*, sefit):
    return {
        "y": [
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
            3.0,
            0.0,
            2.0,
            0.0,
            0.0,
            2.0,
            0.0,
        ],
        "sort1": [0, 1, 2, 3],
        "sort2": [0, 1, 2, 3],
        "utime": [1.0, 2.0],
        "cstate": [0, 0, 0, 0],
        "wt": [1.0, 1.0, 1.0, 1.0],
        "grp": [0, 1, 2, 3],
        "ngrp": 4,
        "p0": [1.0, 0.0, 0.0],
        "i0": [0.0] * 12,
        "sefit": sefit,
        "entry": False,
        "position": [3, 3, 3, 3],
        "hindx": [[2, 0, 1], [2, 2, 2], [2, 2, 2]],
        "trmat": [[0, 1], [0, 2]],
        "t0": 0.0,
    }


def test_clogit_dataset_tracks_observations_and_covariates():
    dataset = _matched_clogit_dataset()

    assert dataset.get_num_observations() == 4
    assert dataset.get_num_covariates() == 1
    assert len(dataset) == 4
    assert dataset.is_empty() is False


def test_clogit_dataset_rejects_invalid_observations():
    dataset = survival.ClogitDataSet()
    assert len(dataset) == 0
    assert dataset.is_empty() is True

    with pytest.raises(ValueError, match="0 or 1"):
        dataset.add_observation(2, 0, [1.0])
    with pytest.raises(ValueError, match="finite"):
        dataset.add_observation(1, 0, [math.nan])

    dataset.add_observation(1, 0, [1.0])

    with pytest.raises(ValueError, match="same number of covariates"):
        dataset.add_observation(0, 0, [1.0, 2.0])
    assert len(dataset) == 1


def test_conditional_logistic_regression_rejects_invalid_controls():
    dataset = _matched_clogit_dataset()

    with pytest.raises(ValueError, match="max_iter"):
        survival.ConditionalLogisticRegression(dataset, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        survival.ConditionalLogisticRegression(dataset, tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        survival.ConditionalLogisticRegression(dataset, tol=math.inf)


def test_conditional_logistic_regression_fit_populates_outputs():
    model = survival.ConditionalLogisticRegression(_matched_clogit_dataset(), max_iter=20, tol=1e-9)
    model.fit()

    assert len(model.coefficients) == 1
    assert 1 <= model.iterations <= 20
    assert isinstance(model.converged, bool)
    assert model.coefficients[0] > 0.0
    assert model.predict([2.0]) > model.predict([1.0])
    assert model.odds_ratios()[0] == pytest.approx(math.exp(model.coefficients[0]))


def test_conditional_logistic_regression_predict_validates_rows():
    model = survival.ConditionalLogisticRegression(_matched_clogit_dataset(), max_iter=20, tol=1e-9)

    with pytest.raises(ValueError, match="fit before prediction"):
        model.predict([1.0])

    model.fit()

    with pytest.raises(ValueError, match="expected 1"):
        model.predict([])
    with pytest.raises(ValueError, match="finite"):
        model.predict([math.nan])

    assert math.isfinite(model.predict([2.0]))


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


def test_conditional_logistic_regression_handles_multi_case_strata():
    dataset = survival.ClogitDataSet()
    for case_status, stratum, covariates in [
        (1, 0, [3.0]),
        (1, 0, [2.5]),
        (0, 0, [0.5]),
        (0, 0, [0.0]),
        (1, 1, [4.0]),
        (1, 1, [3.0]),
        (0, 1, [1.0]),
        (0, 1, [0.0]),
    ]:
        dataset.add_observation(case_status, stratum, covariates)

    model = survival.ConditionalLogisticRegression(dataset, max_iter=40, tol=1e-9)
    model.fit()

    assert len(model.coefficients) == 1
    assert model.coefficients[0] > 0.0
    assert model.predict([3.0]) > model.predict([0.0])


def test_case_cohort_prentice_accepts_public_enum_value():
    cohort = survival.CohortData.new()
    assert len(cohort) == 0
    assert cohort.is_empty() is True

    cohort.add_subject(_simple_subject(1, 0.1, is_case=True, is_subcohort=True))
    cohort.add_subject(_simple_subject(2, 0.2, is_case=False, is_subcohort=True))
    assert len(cohort) == 2
    assert cohort.is_empty() is False

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


def test_cohort_data_rejects_out_of_range_subject_index():
    cohort = survival.CohortData.new()
    cohort.add_subject(_simple_subject(7, 1.5, is_case=True, is_subcohort=True))

    with pytest.raises(IndexError, match="subject index 1 out of range"):
        cohort.get_subject(1)


def test_survfitaj_basic_outputs_are_consistent():
    result = survival.survfitaj(**_survfitaj_kwargs(sefit=0))

    assert len(result.pstate) == 2
    assert result.n_enter is None
    assert result.std_err is None
    assert result.influence is None
    assert result.pstate[0] == pytest.approx([2 / 3, 1 / 3])
    assert result.pstate[1] == pytest.approx([1 / 3, 2 / 3])
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
    assert result.std_err[0] == pytest.approx([0.272165526975909] * 2)
    assert result.std_err[1] == pytest.approx([0.272165526975909] * 2)


def test_survfitaj_matches_r_for_simultaneous_competing_transitions():
    result = survival.survfitaj(**_competing_survfitaj_kwargs(sefit=3))

    assert result.n_risk[0] == pytest.approx([4.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    assert result.n_event[0] == pytest.approx([0.0, 1.0, 1.0])
    assert result.pstate[0] == pytest.approx([0.5, 0.25, 0.25])
    assert result.pstate[1] == pytest.approx([0.5, 0.25, 0.25])
    assert result.cumhaz[0] == pytest.approx([0.25, 0.25])
    assert result.n_transition[0] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    tied_error = math.sqrt(3) / 8
    for row in result.std_err:
        assert row == pytest.approx([0.25, tied_error, tied_error])
    for row in result.std_chaz:
        assert row == pytest.approx([tied_error, tied_error])
    assert result.std_auc[0] == pytest.approx([0.0, 0.0, 0.0])
    assert result.std_auc[1] == pytest.approx([0.25, tied_error, tied_error])
    expected_influence = [
        -0.125,
        -0.125,
        0.125,
        0.125,
        0.1875,
        -0.0625,
        -0.0625,
        -0.0625,
        -0.0625,
        0.1875,
        -0.0625,
        -0.0625,
    ]
    for row, expected in zip(result.influence, expected_influence, strict=True):
        assert row == pytest.approx([expected, expected])


def test_survfitaj_matches_r_for_weighted_clustered_influence():
    result = survival.survfitaj(
        y=[0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        sort1=[0, 1, 2],
        sort2=[0, 1, 2],
        utime=[1.0, 2.0],
        cstate=[0, 0, 0],
        wt=[2.0, 1.0, 1.0],
        grp=[0, 0, 1],
        ngrp=2,
        p0=[1.0, 0.0],
        i0=[0.0] * 4,
        sefit=3,
        entry=False,
        position=[3, 3, 3],
        hindx=[[1, 0], [1, 1]],
        trmat=[[0, 1]],
        t0=0.0,
    )

    standard_error = math.sqrt(1 / 32)
    for row in result.pstate:
        assert row == pytest.approx([0.5, 0.5])
    for row in result.std_err:
        assert row == pytest.approx([standard_error, standard_error])
    for row in result.std_chaz:
        assert row == pytest.approx([standard_error])
    assert result.std_auc[1] == pytest.approx([standard_error, standard_error])
    expected_influence = [-0.125, 0.125, 0.125, -0.125]
    for row, expected in zip(result.influence, expected_influence, strict=True):
        assert row == pytest.approx([expected, expected])


def test_survfitaj_is_invariant_to_row_order():
    original = survival.survfitaj(**_competing_survfitaj_kwargs(sefit=3))
    permuted_kwargs = _competing_survfitaj_kwargs(sefit=3)
    rows = [permuted_kwargs["y"][idx : idx + 3] for idx in range(0, 12, 3)]
    permutation = [3, 1, 2, 0]
    permuted_kwargs["y"] = [value for idx in permutation for value in rows[idx]]
    for field in ["cstate", "wt", "grp", "position"]:
        values = permuted_kwargs[field]
        permuted_kwargs[field] = [values[idx] for idx in permutation]
    permuted_kwargs["sort1"] = [0, 1, 2, 3]
    permuted_kwargs["sort2"] = [1, 3, 0, 2]

    permuted = survival.survfitaj(**permuted_kwargs)

    for field in [
        "n_risk",
        "n_event",
        "n_censor",
        "pstate",
        "cumhaz",
        "std_err",
        "std_chaz",
        "std_auc",
        "influence",
        "n_transition",
    ]:
        for actual, expected in zip(
            getattr(permuted, field), getattr(original, field), strict=True
        ):
            assert actual == pytest.approx(expected)
    assert permuted.n_enter is original.n_enter is None


def test_survfitaj_accepts_negative_times_and_zero_hazards():
    result = survival.survfitaj(
        y=[-2.0, -1.0, 0.0, -2.0, 1.0, 0.0, -2.0, 1.0, 0.0],
        sort1=[0, 1, 2],
        sort2=[0, 1, 2],
        utime=[-1.0, 1.0],
        cstate=[0, 0, 1],
        wt=[1.0, 1.0, 1.0],
        grp=[0, 1, 2],
        ngrp=3,
        p0=[2 / 3, 1 / 3],
        i0=[1 / 9, 1 / 9, -2 / 9, -1 / 9, -1 / 9, 2 / 9],
        sefit=3,
        entry=False,
        position=[3, 3, 3],
        hindx=[[0, 0], [0, 0]],
        trmat=[],
        t0=-2.0,
    )

    for row in result.pstate:
        assert row == pytest.approx([2 / 3, 1 / 3])
    assert result.cumhaz == [[], []]
    assert result.n_transition == [[], []]
    standard_error = math.sqrt(2 / 27)
    assert result.std_err[0] == pytest.approx([standard_error] * 2)
    assert result.std_auc[1] == pytest.approx([3 * standard_error] * 2)
    expected_influence = [1 / 9, 1 / 9, -2 / 9, -1 / 9, -1 / 9, 2 / 9]
    for row, expected in zip(result.influence, expected_influence, strict=True):
        assert row == pytest.approx([expected, expected])


def test_survfitaj_rejects_ragged_hindx():
    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["hindx"] = [[0, 1], [2]]

    with pytest.raises(ValueError, match="Invalid hindx array"):
        survival.survfitaj(**kwargs)


def test_survfitaj_validates_public_inputs():
    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["y"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="y length must be a multiple of 3"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["sort1"] = [0, 4, 2]
    with pytest.raises(ValueError, match="sort index 4"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["wt"] = [1.0, float("inf"), 1.0]
    with pytest.raises(ValueError, match="wt contains non-finite"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["cstate"] = [0, 2, 0]
    with pytest.raises(ValueError, match="cstate value 2"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=1)
    kwargs["i0"] = [0.0]
    with pytest.raises(ValueError, match="i0 length must equal ngrp"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["p0"] = [0.6, 0.6]
    with pytest.raises(ValueError, match="p0 probabilities must sum to 1"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["p0"] = [-1.0, 2.0]
    with pytest.raises(ValueError, match="p0 contains negative value"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["trmat"] = [[0, 1, 2]]
    with pytest.raises(ValueError, match="trmat array: matrix must have exactly 2 columns"):
        survival.survfitaj(**kwargs)

    kwargs = _survfitaj_kwargs(sefit=0)
    kwargs["hindx"] = [[2, 0], [1, 1]]
    with pytest.raises(ValueError, match="hindx hazard index 2"):
        survival.survfitaj(**kwargs)


def test_illness_death_public_apis_and_validation():
    model = survival.fit_illness_death(
        entry_time=[0.0, 0.0, 0.0],
        transition_time=[1.0, 0.0, 1.0 + 5e-10],
        exit_time=[2.0, 2.0, 2.0],
        from_state=[0, 0, 0],
        to_state=[1, 2, 1],
        covariates=[[10.0], [20.0], [30.0]],
        config=None,
    )
    prediction = survival.predict_illness_death(
        model,
        current_state=0,
        time_in_state=0.0,
        prediction_times=[0.5, 1.0],
        covariates=[0.1],
    )

    assert len(model.transition_hazards) == 3
    assert model.transition_hazards[0].baseline_times == pytest.approx([1.0])
    assert model.transition_hazards[0].baseline_hazard == pytest.approx([1.0])
    assert model.transition_hazards[0].coefficient == pytest.approx(20.0)
    assert len(prediction.state_probs) == 2
    assert prediction.survival_prob[0] > 0.0

    with pytest.raises(ValueError, match="input vectors must be non-empty"):
        survival.fit_illness_death([], [], [], [], [], None, None)

    with pytest.raises(ValueError, match="exit_time contains non-finite"):
        survival.fit_illness_death([0.0], [0.0], [float("inf")], [0], [0], None, None)

    with pytest.raises(ValueError, match="from_state must contain only 0/1"):
        survival.fit_illness_death([0.0], [0.0], [1.0], [3], [0], None, None)

    with pytest.raises(ValueError, match="transition_time must be between"):
        survival.fit_illness_death([0.0], [3.0], [2.0], [0], [1], None, None)

    with pytest.raises(ValueError, match="covariates row 1 has 2 columns"):
        survival.fit_illness_death(
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [0, 0],
            [1, 2],
            [[1.0], [2.0, 3.0]],
            None,
        )

    with pytest.raises(ValueError, match="current_state must be"):
        survival.predict_illness_death(model, 3, 0.0, [1.0], None)

    with pytest.raises(ValueError, match="time_in_state must be finite"):
        survival.predict_illness_death(model, 0, float("inf"), [1.0], None)

    with pytest.raises(ValueError, match="prediction_times contains negative value"):
        survival.predict_illness_death(model, 0, 0.0, [-1.0], None)


def test_semi_markov_public_apis_and_validation():
    config = survival.SemiMarkovConfig(3)
    model = survival.fit_semi_markov(
        [0.0, 1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 1.5],
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        config,
    )
    prediction = survival.predict_semi_markov(model, 0, 0.5, [0.5, 1.0])

    assert len(model.sojourn_params) == 3
    assert model.get_transition_prob(0, 1) == pytest.approx(1.0)
    assert len(prediction.state_probs) == 2
    assert prediction.time_points == pytest.approx([0.5, 1.0])

    with pytest.raises(ValueError, match="n_states must be positive"):
        survival.SemiMarkovConfig(0)

    with pytest.raises(ValueError, match="state_names length"):
        survival.SemiMarkovConfig(3, ["A"])

    with pytest.raises(ValueError, match="absorbing_states must contain values"):
        survival.SemiMarkovConfig(3, None, None, [3])

    with pytest.raises(ValueError, match="input vectors must be non-empty"):
        survival.fit_semi_markov([], [], [], [], config)

    with pytest.raises(ValueError, match="exit_times contains non-finite"):
        survival.fit_semi_markov([0.0], [float("inf")], [0], [1], config)

    with pytest.raises(ValueError, match="entry_times must be <= exit_times"):
        survival.fit_semi_markov([2.0], [1.0], [0], [1], config)

    with pytest.raises(ValueError, match="from_states must contain values"):
        survival.fit_semi_markov([0.0], [1.0], [-1], [1], config)

    with pytest.raises(ValueError, match="current_state must be"):
        survival.predict_semi_markov(model, 3, 0.0, [1.0])

    with pytest.raises(ValueError, match="time_in_state must be finite"):
        survival.predict_semi_markov(model, 0, float("inf"), [1.0])

    with pytest.raises(ValueError, match="prediction_times contains negative value"):
        survival.predict_semi_markov(model, 0, 0.0, [-1.0])
