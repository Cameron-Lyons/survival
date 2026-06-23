import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_survfitkm_empty_input():
    with pytest.raises(ValueError, match="time cannot be empty"):
        survival.survfitkm(time=[], status=[])


def test_survfitkm_length_mismatch():
    with pytest.raises(ValueError, match="status length mismatch"):
        survival.survfitkm(time=[1.0, 2.0], status=[1.0])


def test_survfitkm_negative_time():
    with pytest.raises(ValueError, match="time contains negative value"):
        survival.survfitkm(time=[-1.0, 2.0], status=[1.0, 0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": [1.0, float("inf")], "status": [1.0, 0.0]}, "time contains non-finite"),
        ({"time": [1.0, 2.0], "status": [1.0, float("inf")]}, "status contains non-finite"),
        (
            {"time": [1.0, 2.0], "status": [1.0, 0.0], "weights": [1.0, float("nan")]},
            "weights contains NaN",
        ),
        (
            {"time": [1.0, 2.0], "status": [1.0, 0.0], "weights": [1.0, float("inf")]},
            "weights contains non-finite",
        ),
        (
            {"time": [1.0, 2.0], "status": [1.0, 0.0], "entry_times": [0.0, float("inf")]},
            "entry_times contains non-finite",
        ),
    ],
)
def test_survfitkm_rejects_non_finite_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        survival.survfitkm(**kwargs)


def test_survfitkm_options_rejects_non_finite_weights():
    options = survival.SurvfitKMOptions(weights=[1.0, float("inf")])

    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.survfitkm_with_options([1.0, 2.0], [1.0, 0.0], options)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: survival.logrank_test([1.0, float("inf")], [1, 0], [0, 1]),
            "time contains non-finite",
        ),
        (
            lambda: survival.logrank_test(
                [1.0, 2.0], [1, 0], [0, 1], entry_times=[0.0, float("inf")]
            ),
            "entry_times contains non-finite",
        ),
        (
            lambda: survival.fleming_harrington_test([1.0, 2.0], [1, 0], [0, 1], rho=float("inf")),
            "rho must be finite",
        ),
        (
            lambda: survival.logrank_trend([1.0, float("inf")], [1, 0], [0, 1]),
            "time contains non-finite",
        ),
        (
            lambda: survival.logrank_trend([1.0, 2.0], [1, 0], [0, 1], scores=[0.0, float("inf")]),
            "scores contains non-finite",
        ),
        (
            lambda: survival.compute_logrank_components(
                [1.0, float("inf")], [1, 0], [1, 2], None, 0.0
            ),
            "time contains non-finite",
        ),
    ],
)
def test_logrank_apis_reject_non_finite_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: survival.logrank_trend([1.0, 2.0], [1], [0, 1]), "status length mismatch"),
        (lambda: survival.logrank_trend([1.0, 2.0], [1, 0], [0]), "group length mismatch"),
        (
            lambda: survival.logrank_trend([1.0, 2.0], [1, 0], [0, 1], scores=[0.0]),
            "scores length mismatch",
        ),
    ],
)
def test_logrank_trend_rejects_invalid_shapes_without_panicking(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: survival.sample_size_survival(float("nan")),
            "hazard_ratio must be finite",
        ),
        (
            lambda: survival.sample_size_survival(1.0),
            "hazard_ratio must be positive and not equal to 1",
        ),
        (
            lambda: survival.sample_size_survival(0.7, power=1.0),
            "power must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.sample_size_survival(0.7, alpha=0.0),
            "alpha must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.sample_size_survival(0.7, allocation_ratio=0.0),
            "allocation_ratio must be positive",
        ),
        (
            lambda: survival.sample_size_survival(0.7, sided=3),
            "sided must be 1 or 2",
        ),
        (
            lambda: survival.sample_size_survival_freedman(0.7, float("inf")),
            "prob_event must be finite",
        ),
        (
            lambda: survival.sample_size_survival_freedman(0.7, 0.0),
            "prob_event must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.power_survival(0, 0.7),
            "n_events must be positive",
        ),
        (
            lambda: survival.power_survival(10, 0.7, allocation_ratio=-1.0),
            "allocation_ratio must be positive",
        ),
    ],
)
def test_power_apis_reject_invalid_parameters(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: survival.expected_events(0, 0.1, 0.7, 12.0, 6.0),
            "n_total must be positive",
        ),
        (
            lambda: survival.expected_events(100, float("nan"), 0.7, 12.0, 6.0),
            "hazard_control must be finite",
        ),
        (
            lambda: survival.expected_events(100, 0.1, 0.0, 12.0, 6.0),
            "hazard_ratio must be positive",
        ),
        (
            lambda: survival.expected_events(100, 0.1, 0.7, -1.0, 6.0),
            "accrual_time must be non-negative",
        ),
        (
            lambda: survival.expected_events(100, 0.1, 0.7, 0.0, 0.0),
            "accrual_time and followup_time cannot both be zero",
        ),
        (
            lambda: survival.expected_events(100, 0.1, 0.7, 12.0, 6.0, allocation_ratio=0.0),
            "allocation_ratio must be positive",
        ),
        (
            lambda: survival.expected_events(100, 0.1, 0.7, 12.0, 6.0, dropout_rate=-0.1),
            "dropout_rate must be non-negative",
        ),
    ],
)
def test_expected_events_rejects_invalid_parameters(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: survival.survobrien([1.0, 2.0], [1], [0.1, 0.2]), "status length mismatch"),
        (
            lambda: survival.survobrien([1.0, 2.0], [1, 0], [0.1]),
            "covariate length mismatch",
        ),
        (
            lambda: survival.survobrien([1.0, 2.0], [1, 0], [0.1, 0.2], [1]),
            "strata length mismatch",
        ),
        (
            lambda: survival.survobrien([1.0, 2.0], [1, 2], [0.1, 0.2]),
            "status must contain only 0/1",
        ),
        (
            lambda: survival.survobrien([1.0, float("inf")], [1, 0], [0.1, 0.2]),
            "time contains non-finite",
        ),
        (
            lambda: survival.survobrien([1.0, 2.0], [1, 0], [0.1, float("nan")]),
            "covariate contains NaN",
        ),
    ],
)
def test_survobrien_rejects_invalid_public_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_compute_logrank_components_all_censored_has_zero_degrees_of_freedom():
    result = survival.compute_logrank_components(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[0, 0, 0, 0],
        group=[1, 1, 2, 2],
        strata=None,
        rho=0.0,
    )

    assert result.chi_squared == 0.0
    assert result.degrees_of_freedom == 0


def test_compute_logrank_components_rejects_invalid_codes():
    with pytest.raises(ValueError, match="status values must be 0 or 1"):
        survival.compute_logrank_components([1.0, 2.0], [1, 2], [1, 2], None, 0.0)

    with pytest.raises(ValueError, match="group must be >= 1"):
        survival.compute_logrank_components([1.0, 2.0], [1, 0], [0, 2], None, 0.0)

    with pytest.raises(ValueError, match="group values must be between 1"):
        survival.compute_logrank_components([1.0, 2.0], [1, 0], [1, 3], None, 0.0)

    with pytest.raises(ValueError, match="strata must be >= 0"):
        survival.compute_logrank_components([1.0, 2.0], [1, 0], [1, 2], [-1, 0], 0.0)

    with pytest.raises(ValueError, match="strata values must be 0 or 1"):
        survival.compute_logrank_components([1.0, 2.0], [1, 0], [1, 2], [0, 2], 0.0)


def test_agmart_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        survival.CountingProcessData(
            start=[0.0, 0.0],
            stop=[1.0, 2.0, 3.0],
            event=[1, 0, 1],
        )


def test_coxph_model_empty_fit():
    model = survival.CoxPHModel()
    with pytest.raises(ValueError, match="no data provided"):
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

    with pytest.raises(ValueError, match="covariate dimension mismatch"):
        model.add_subject(subject)


def test_coxph_model_new_with_data_validates_shapes_and_values():
    with pytest.raises(ValueError, match="event_times length mismatch"):
        survival.CoxPHModel.new_with_data([[1.0]], [1.0, 2.0], [1])

    with pytest.raises(ValueError, match="censoring length mismatch"):
        survival.CoxPHModel.new_with_data([[1.0]], [1.0], [1, 0])

    with pytest.raises(ValueError, match="covariate row 1 has 1 columns"):
        survival.CoxPHModel.new_with_data([[1.0, 2.0], [3.0]], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="event_times contains non-finite"):
        survival.CoxPHModel.new_with_data([[1.0]], [float("inf")], [1])

    with pytest.raises(ValueError, match="censoring must contain only 0/1"):
        survival.CoxPHModel.new_with_data([[1.0]], [1.0], [2])


def test_coxph_model_prediction_validates_covariates():
    model = survival.CoxPHModel.new_with_data([[0.0, 1.0], [1.0, 0.0]], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="covariate row 0 has 1 columns"):
        model.predict([[1.0]])

    with pytest.raises(ValueError, match="covariate row 0 contains NaN"):
        model.survival_curve([[1.0, float("nan")]], None)

    with pytest.raises(ValueError, match="covariate row 0 has 1 columns"):
        model.cumulative_hazard([[1.0]])


def test_compute_logrank_components_with_strata():
    result = survival.compute_logrank_components(
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
