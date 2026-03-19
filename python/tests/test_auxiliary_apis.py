import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_brier_helpers():
    assert survival.brier([0.0, 1.0], [0, 1]) == pytest.approx(0.0)
    assert survival.integrated_brier([[0.0, 0.0], [1.0, 1.0]], [0, 1], [1.0, 2.0]) == pytest.approx(
        0.0
    )

    with pytest.raises(ValueError, match="predictions must be between 0 and 1"):
        survival.brier([1.2, 0.5], [1, 0])


def test_statistical_test_helpers():
    lrt = survival.lrt_test(-10.0, -12.0, 1)
    assert lrt.statistic == pytest.approx(4.0)
    assert lrt.df == 1
    assert lrt.p_value == pytest.approx(0.0455, rel=1e-3)

    wald = survival.wald_test_py([1.0, 2.0], [1.0, 2.0])
    assert wald.statistic == pytest.approx(2.0)
    assert wald.df == 2
    assert wald.p_value == pytest.approx(0.3679, rel=1e-3)

    score = survival.score_test_py([1.0], [[2.0]])
    assert score.statistic == pytest.approx(0.5)
    assert score.df == 1
    assert score.p_value == pytest.approx(0.4795, rel=1e-3)

    ph = survival.ph_test(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
        [1.0, 2.0, 3.0, 4.0],
        None,
    )
    assert ph.global_df == 2
    assert len(ph.variable_names) == 2
    assert len(ph.p_values) == 2
    assert all(0.0 <= value <= 1.0 for value in ph.p_values)

    with pytest.raises(ValueError, match="coefficients and std_errors must have the same length"):
        survival.wald_test_py([1.0], [1.0, 2.0])

    with pytest.raises(
        ValueError, match="score_vector length must match information_matrix dimensions"
    ):
        survival.score_test_py([1.0, 2.0], [[1.0]])


def test_bootstrap_ci_helpers_smoke():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    status_i32 = [1, 1, 0, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    cox_covariates = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    survreg_covariates = [[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5], [1.0, 0.6]]

    cox = survival.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=123,
    )
    assert len(cox.coefficients) == 1
    assert len(cox.std_errors) == 1
    assert len(cox.ci_lower) == 1
    assert len(cox.ci_upper) == 1
    assert len(cox.bootstrap_samples) > 0
    assert np.isfinite(cox.coefficients[0])

    survreg = survival.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=123,
    )
    assert len(survreg.coefficients) == 3
    assert len(survreg.std_errors) == 3
    assert len(survreg.ci_lower) == 3
    assert len(survreg.ci_upper) == 3
    assert len(survreg.bootstrap_samples) > 0
    assert np.all(np.isfinite(survreg.coefficients))


def test_bootstrap_ci_helpers_are_deterministic_with_seed():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    status_i32 = [1, 1, 0, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    cox_covariates = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    survreg_covariates = [[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5], [1.0, 0.6]]

    first_cox = survival.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    second_cox = survival.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    assert first_cox.coefficients == pytest.approx(second_cox.coefficients)
    assert first_cox.std_errors == pytest.approx(second_cox.std_errors)
    assert first_cox.ci_lower == pytest.approx(second_cox.ci_lower)
    assert first_cox.ci_upper == pytest.approx(second_cox.ci_upper)
    assert first_cox.bootstrap_samples == second_cox.bootstrap_samples

    first_survreg = survival.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    second_survreg = survival.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    assert first_survreg.coefficients == pytest.approx(second_survreg.coefficients)
    assert first_survreg.std_errors == pytest.approx(second_survreg.std_errors)
    assert first_survreg.ci_lower == pytest.approx(second_survreg.ci_lower)
    assert first_survreg.ci_upper == pytest.approx(second_survreg.ci_upper)
    assert first_survreg.bootstrap_samples == second_survreg.bootstrap_samples


def test_bootstrap_ci_helpers_validate_inputs():
    time = [1.0, 2.0, 3.0, 4.0]
    status_i32 = [1, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    covariates = [[0.1], [0.2], [0.3], [0.4]]

    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        survival.bootstrap_cox_ci(time, status_i32, covariates, n_bootstrap=1)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        survival.bootstrap_survreg_ci(
            time,
            status_f64,
            covariates,
            distribution="weibull",
            n_bootstrap=8,
            confidence_level=1.2,
        )


def test_pystep_helpers():
    simple = survival.perform_pystep_simple_calculation(
        1,
        [0.5],
        [0],
        [2],
        [[0.0, 1.0, 2.0]],
        10.0,
    )
    assert simple["time_step"] == pytest.approx(0.5)
    assert simple["index"] == 0

    step = survival.perform_pystep_calculation(
        1,
        [0.25],
        [0],
        [2],
        [[0.0, 1.0]],
        1.0,
    )
    assert step["time_step"] == pytest.approx(0.75)
    assert step["current_index"] == 1
    assert step["next_index"] == 1
    assert step["weight"] == pytest.approx(1.0)
    assert step["updated_data"] == [1.0]

    with pytest.raises(RuntimeError, match="Data length does not match odim"):
        survival.perform_pystep_simple_calculation(2, [0.5], [0], [2], [[0.0, 1.0, 2.0]], 1.0)


def test_pyears_helper_basic():
    result = survival.perform_pyears_calculation(
        [2.0, 1.0],
        [1.0],
        1,
        [0],
        [1],
        [0.0],
        [0.5],
        [0.0],
        1,
        [0],
        [1],
        [0.0, 5.0],
        1,
        [0.0],
        1,
        2,
    )

    assert result["pyears"] == [2.0]
    assert result["pn"] == [1.0]
    assert result["pcount"] == [1.0]
    assert result["pexpect"] == [1.0]
    assert result["offtable"] == pytest.approx(0.0)


def test_cox_callback_roundtrip():
    def callback(coef, *, which):
        return {
            "coef": [value + which for value in coef],
            "first": [1.0, 2.0],
            "second": [3.0, 4.0],
            "penalty": [5.0, 6.0],
            "flag": [True, False],
        }

    result = survival.cox_callback(
        2,
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0, 0],
        callback,
    )

    assert result == ([3.0, 4.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1, 0])
