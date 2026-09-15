import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_survfitkm_empty_input():
    with pytest.raises(ValueError, match="time cannot be empty"):
        survival.surv_analysis.survfitkm(time=[], status=[])


def test_survfitkm_length_mismatch():
    with pytest.raises(ValueError, match="status length mismatch"):
        survival.surv_analysis.survfitkm(time=[1.0, 2.0], status=[1])
    with pytest.raises(ValueError, match="start length mismatch"):
        survival.surv_analysis.survfitkm(time=[1.0, 2.0], status=[1, 0], start=[0.0])


def test_survfitkm_accepts_negative_times_like_r():
    result = survival.surv_analysis.survfitkm(time=[-1.0, 2.0], status=[1, 0])
    assert result.time == pytest.approx([-1.0, 2.0])
    assert result.surv == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": [1.0, float("inf")], "status": [1, 0]}, "time contains non-finite"),
        ({"time": [1.0, 2.0], "status": [1, 2]}, "status"),
        (
            {"time": [1.0, 2.0], "status": [1, 0], "weights": [1.0, float("nan")]},
            "weights contains non-finite value NaN",
        ),
        (
            {"time": [1.0, 2.0], "status": [1, 0], "weights": [1.0, float("inf")]},
            "weights contains non-finite",
        ),
        (
            {"time": [1.0, 2.0], "status": [1, 0], "start": [0.0, float("inf")]},
            "start contains non-finite",
        ),
    ],
)
def test_survfitkm_rejects_non_finite_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        survival.surv_analysis.survfitkm(**kwargs)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: survival.validation.logrank_test([1.0, float("inf")], [1, 0], [0, 1]),
            "time contains non-finite",
        ),
        (
            lambda: survival.validation.logrank_test(
                [1.0, 2.0], [1, 0], [0, 1], entry_times=[0.0, float("inf")]
            ),
            "start contains non-finite",
        ),
        (
            lambda: survival.validation.logrank_test([1.0, 2.0], [1, 0], [0, 1], rho=float("inf")),
            "rho must be finite",
        ),
        (
            lambda: survival.surv_analysis.survdiff([1.0, float("inf")], [1, 0], [1, 2]),
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
        (
            lambda: survival.validation.logrank_test([1.0, 2.0], [1], [0, 1]),
            "status length mismatch",
        ),
        (
            lambda: survival.validation.logrank_test([1.0, 2.0], [1, 0], [0]),
            "group length mismatch",
        ),
        (
            lambda: survival.surv_analysis.survdiff([1.0, 2.0], [1, 0], [1, 2], strata=[0]),
            "strata length mismatch",
        ),
    ],
)
def test_logrank_apis_reject_invalid_shapes_without_panicking(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: survival.validation.sample_size_survival(float("nan")),
            "hazard_ratio must be finite",
        ),
        (
            lambda: survival.validation.sample_size_survival(1.0),
            "hazard_ratio must be positive and not equal to 1",
        ),
        (
            lambda: survival.validation.sample_size_survival(0.7, power=1.0),
            "power must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.validation.sample_size_survival(0.7, alpha=0.0),
            "alpha must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.validation.sample_size_survival(0.7, allocation_ratio=0.0),
            "allocation_ratio must be positive",
        ),
        (
            lambda: survival.validation.sample_size_survival(0.7, sided=3),
            "sided must be 1 or 2",
        ),
        (
            lambda: survival.validation.sample_size_survival_freedman(0.7, float("inf")),
            "prob_event must be finite",
        ),
        (
            lambda: survival.validation.sample_size_survival_freedman(0.7, 0.0),
            "prob_event must be greater than 0 and less than 1",
        ),
        (
            lambda: survival.validation.power_survival(0, 0.7),
            "n_events must be positive",
        ),
        (
            lambda: survival.validation.power_survival(10, 0.7, allocation_ratio=-1.0),
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
            lambda: survival.validation.expected_events(0, 0.1, 0.7, 12.0, 6.0),
            "n_total must be positive",
        ),
        (
            lambda: survival.validation.expected_events(100, float("nan"), 0.7, 12.0, 6.0),
            "hazard_control must be finite",
        ),
        (
            lambda: survival.validation.expected_events(100, 0.1, 0.0, 12.0, 6.0),
            "hazard_ratio must be positive",
        ),
        (
            lambda: survival.validation.expected_events(100, 0.1, 0.7, -1.0, 6.0),
            "accrual_time must be non-negative",
        ),
        (
            lambda: survival.validation.expected_events(100, 0.1, 0.7, 0.0, 0.0),
            "accrual_time and followup_time cannot both be zero",
        ),
        (
            lambda: survival.validation.expected_events(
                100, 0.1, 0.7, 12.0, 6.0, allocation_ratio=0.0
            ),
            "allocation_ratio must be positive",
        ),
        (
            lambda: survival.validation.expected_events(
                100, 0.1, 0.7, 12.0, 6.0, dropout_rate=-0.1
            ),
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
        (
            lambda: survival.validation.survobrien([1.0, 2.0], [1], [[0.1, 0.2]]),
            "status length mismatch",
        ),
        (
            lambda: survival.validation.survobrien([1.0, 2.0], [1, 0], [[0.1]]),
            r"continuous\[0\] length mismatch",
        ),
        (
            lambda: survival.validation.survobrien([1.0, 2.0], [1, 0], [[0.1, 0.2]], strata=[1]),
            "strata length mismatch",
        ),
        (
            lambda: survival.validation.survobrien([1.0, 2.0], [1, 2], [[0.1, 0.2]]),
            "status values must be 0 or 1",
        ),
        (
            lambda: survival.validation.survobrien([1.0, float("inf")], [1, 0], [[0.1, 0.2]]),
            "time contains non-finite",
        ),
        (
            lambda: survival.validation.survobrien([1.0, 2.0], [1, 0], [[0.1, float("nan")]]),
            "continuous contains non-finite",
        ),
    ],
)
def test_survobrien_rejects_invalid_public_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_survobrien_expands_one_risk_set_per_event_time():
    expansion = survival.validation.survobrien(
        [1.0, 1.0, 2.0, 3.0], [1, 1, 0, 0], [[10.0, 30.0, 20.0, 40.0]]
    )

    # one risk set (all four rows) at the single event time 1, logit mid-rank transform
    assert expansion.event_times == pytest.approx([1.0])
    assert expansion.row == [0, 1, 2, 3]
    assert expansion.status == [1, 1, 0, 0]
    assert expansion.transformed[0] == pytest.approx(
        [-1.9459101490553135, 0.5108256237659907, -0.5108256237659907, 1.9459101490553132]
    )


def test_survdiff_all_censored_has_zero_degrees_of_freedom():
    result = survival.surv_analysis.survdiff([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [1, 1, 2, 2])

    assert result.chisq == 0.0
    assert result.df == 0
    assert result.pvalue == 1.0
    assert (
        survival.validation.logrank_test([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [1, 1, 2, 2]).df == 0
    )


def test_survdiff_rejects_invalid_codes():
    with pytest.raises(ValueError, match="status values must be 0 or 1"):
        survival.surv_analysis.survdiff([1.0, 2.0], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="length mismatch"):
        survival.surv_analysis.survdiff([1.0, 2.0], [1, 0], [1])


def test_agmart_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        survival.core.CountingProcessData(
            start=[0.0, 0.0],
            stop=[1.0, 2.0, 3.0],
            event=[1, 0, 1],
        )


def test_coxph_fit_validates_shapes_and_values():
    with pytest.raises(ValueError, match="status length mismatch"):
        survival.regression.coxph_fit([1.0, 2.0], [1], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="x has 1 rows"):
        survival.regression.coxph_fit([1.0, 2.0], [1, 0], [[1.0]])
    with pytest.raises(ValueError, match="x must be rectangular"):
        survival.regression.coxph_fit([1.0, 2.0], [1, 0], [[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.coxph_fit([1.0, float("inf")], [1, 0], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="status values must be 0 or 1"):
        survival.regression.coxph_fit([1.0, 2.0], [1, 2], [[1.0], [2.0]])


def test_coxph_fit_prediction_validates_covariates():
    fit = survival.regression.coxph_fit(
        [1.0, 2.0, 3.0], [1, 0, 1], [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    )

    with pytest.raises(ValueError, match="newdata"):
        fit.predict("lp", newdata=[[1.0]])
    with pytest.raises(ValueError, match="newdata"):
        fit.survfit(newdata=[[1.0, float("nan")]])


def test_all_censored():
    result = survival.surv_analysis.survfitkm(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[0, 0, 0, 0],
    )
    assert result.surv == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert result.n_censor == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert result.cumhaz == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_single_observation():
    result = survival.surv_analysis.survfitkm(
        time=[5.0],
        status=[1],
    )
    assert result.time == pytest.approx([5.0])
    assert result.surv == pytest.approx([0.0])
    assert result.n_risk == pytest.approx([1.0])
