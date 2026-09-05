import importlib

import numpy as np
import pytest

from .helpers import setup_survival_import

_surv = setup_survival_import()
sklearn_compat = importlib.import_module("survival.sklearn_compat")
AFTEstimator = sklearn_compat.AFTEstimator
StreamingAFTEstimator = sklearn_compat.StreamingAFTEstimator


class TestSurvreg:
    def test_survreg_weibull_uncensored(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        true_beta = np.array([1.0, 0.5, -0.3])
        log_time = X @ true_beta + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        status = np.ones(n, dtype=np.float64)

        result = _surv.survreg(
            time=time.tolist(),
            status=status.tolist(),
            covariates=X.tolist(),
            distribution="weibull",
            max_iter=100,
        )

        assert len(result.coefficients) == 4
        assert result.log_likelihood < 0
        assert np.isfinite(result.log_likelihood)

    def test_survreg_weibull_censored(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        true_beta = np.array([1.0, 0.5, -0.3])
        log_time = X @ true_beta + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        censor_time = np.random.exponential(3, n)
        observed_time = np.minimum(time, censor_time)
        status = (time <= censor_time).astype(np.float64)

        result = _surv.survreg(
            time=observed_time.tolist(),
            status=status.tolist(),
            covariates=X.tolist(),
            distribution="weibull",
            max_iter=100,
        )

        assert len(result.coefficients) == 4
        assert result.log_likelihood < 0
        assert np.isfinite(result.log_likelihood)

    def test_survreg_lognormal(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        true_beta = np.array([2.0, 0.3, -0.5])
        log_time = X @ true_beta + 0.8 * np.random.randn(n)
        time = np.exp(log_time)
        status = np.ones(n, dtype=np.float64)

        result = _surv.survreg(
            time=time.tolist(),
            status=status.tolist(),
            covariates=X.tolist(),
            distribution="lognormal",
            max_iter=100,
        )

        assert len(result.coefficients) == 4
        assert np.isfinite(result.log_likelihood)

    def test_survreg_loglogistic(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        true_beta = np.array([1.5, 0.4])
        u = np.random.uniform(0, 1, n)
        log_time = X @ true_beta + 0.6 * np.log(u / (1 - u))
        time = np.exp(log_time)
        status = np.ones(n, dtype=np.float64)

        result = _surv.survreg(
            time=time.tolist(),
            status=status.tolist(),
            covariates=X.tolist(),
            distribution="loglogistic",
            max_iter=100,
        )

        assert len(result.coefficients) == 3
        assert np.isfinite(result.log_likelihood)

    def test_survreg_small_sample(self):
        time = [1.0, 2.0, 3.0, 4.0, 5.0]
        status = [1.0, 1.0, 1.0, 1.0, 1.0]
        X = [[1.0], [1.0], [1.0], [1.0], [1.0]]

        result = _surv.survreg(
            time=time,
            status=status,
            covariates=X,
            distribution="weibull",
            max_iter=100,
        )

        assert len(result.coefficients) == 2
        assert result.log_likelihood < 0


class TestAFTEstimator:
    def test_fit_uncensored(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        true_beta = np.array([0.5, -0.3])
        log_time = 1.0 + X @ true_beta + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)

        assert model.n_features_in_ == 2
        assert len(model.coef_) == 2
        assert model.scale_ > 0
        assert hasattr(model, "intercept_")
        assert np.isfinite(model.intercept_)

    def test_fit_censored(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        true_beta = np.array([0.5, -0.3])
        log_time = 1.0 + X @ true_beta + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        censor_time = np.random.exponential(3, n)
        observed_time = np.minimum(time, censor_time)
        status = (time <= censor_time).astype(float)
        y = np.column_stack([observed_time, status])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)

        assert model.n_features_in_ == 2
        assert np.isfinite(model.intercept_)

    def test_predict(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == n
        assert all(predictions > 0)

    def test_predict_median(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)
        median_times = model.predict_median(X)

        assert len(median_times) == n
        assert all(median_times > 0)

    def test_score(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)
        c_index = model.score(X, y)

        assert 0 <= c_index <= 1

    def test_predict_quantile_lognormal(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="lognormal")
        model.fit(X, y)
        quantiles = model.predict_quantile(X, q=0.9)

        assert len(quantiles) == n
        assert all(quantiles > 0)

    @pytest.mark.parametrize("q", [np.nan, np.inf, -np.inf, 0.0, 1.0, -0.1, 1.1])
    def test_predict_quantile_rejects_invalid_q(self, q):
        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64)
        y = np.column_stack([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])

        model = AFTEstimator(distribution="weibull", max_iter=10)
        model.fit(X, y)

        with pytest.raises(ValueError, match="q must be between 0 and 1"):
            model.predict_quantile(X[:2], q=q)

    @pytest.mark.parametrize(
        "distribution",
        ["gaussian", "normal", "logistic", "extreme", "extreme-value", "t", "student-t"],
    )
    def test_identity_distributions_preserve_signed_response_predictions(self, distribution):
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([[-3.0, -2.1, -0.2, 0.0, 0.7, 1.8], np.ones(6)])
        model = AFTEstimator(distribution=distribution).fit(X, y)
        zero_location = -model.intercept_ / model.coef_[0]
        new_x = np.array([[zero_location - 1.0], [zero_location], [zero_location + 1.0]])
        expected = model.intercept_ + new_x @ model.coef_

        predictions = model.predict(new_x)

        np.testing.assert_allclose(predictions, expected, atol=1e-12)
        assert predictions[0] < 0.0
        assert predictions[1] == pytest.approx(0.0, abs=1e-12)
        assert predictions[2] > 0.0
        native_rows = np.column_stack([np.ones(3), new_x]).tolist()
        for q in [0.1, 0.5, 0.9]:
            expected_quantile = np.array(
                model.model_.predict_quantile(covariates=native_rows, quantiles=[q]).predictions
            )[:, 0]
            np.testing.assert_allclose(model.predict_quantile(new_x, q), expected_quantile)
        np.testing.assert_allclose(model.predict_median(new_x), model.predict_quantile(new_x, 0.5))
        if model.model_.distribution == "t":
            assert model.model_.distribution_parameters == [4.0]
            assert np.all(model.predict_quantile(new_x, 0.9) > predictions)

    def test_extreme_value_predictions_match_r_survreg(self):
        # R survival 3.8.11: survreg(Surv(time, status) ~ x, dist="extreme").
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([[-3.0, -2.1, -0.2, 0.0, 0.7, 1.8], np.ones(6)])
        model = AFTEstimator(distribution="extreme", tol=1e-12).fit(X, y)
        new_x = np.array([[-1.0], [0.0], [1.0]])

        np.testing.assert_allclose(
            model.predict(new_x),
            [-1.549681731704994, -0.31856042345594882, 0.91256088479309627],
            atol=2e-7,
        )
        expected = np.array(
            [
                [-2.1809247291644995, -1.6524910427839605, -1.3157300855328296],
                [-0.9498034209154543, -0.42136973453491527, -0.08460877728378435],
                [0.28131788733359075, 0.8097515737141299, 1.1465125309652606],
            ]
        )
        for column, q in enumerate([0.1, 0.5, 0.9]):
            np.testing.assert_allclose(
                model.predict_quantile(new_x, q), expected[:, column], atol=2e-7
            )
        np.testing.assert_allclose(model.predict_median(new_x), expected[:, 1], atol=2e-7)

    @pytest.mark.parametrize(
        "distribution",
        ["weibull", "exponential", "rayleigh", "LOG-GAUSSIAN", "log-normal", "log-logistic"],
    )
    def test_log_time_predictions_preserve_response_convention_and_aliases(self, distribution):
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([np.exp([-3.0, -2.1, -0.2, 0.0, 0.7, 1.8]), np.ones(6)])
        model = AFTEstimator(distribution=distribution).fit(X, y)

        np.testing.assert_allclose(model.predict(X), np.exp(model.intercept_ + X @ model.coef_))
        native_rows = np.column_stack([np.ones(6), X]).tolist()
        native_quantiles = np.array(
            model.model_.predict_quantile(
                covariates=native_rows, quantiles=[0.1, 0.5, 0.9]
            ).predictions
        )
        for column, q in enumerate([0.1, 0.5, 0.9]):
            np.testing.assert_allclose(model.predict_quantile(X, q), native_quantiles[:, column])
        np.testing.assert_allclose(model.predict_median(X), native_quantiles[:, 1])
        if distribution in {"weibull", "exponential", "rayleigh"}:
            np.testing.assert_allclose(
                model.predict_median(X), model.predict(X) * np.log(2.0) ** model.scale_
            )

    def test_predictions_use_fitted_distribution_after_parameter_change(self):
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([np.exp([-3.0, -2.1, -0.2, 0.0, 0.7, 1.8]), np.ones(6)])
        model = AFTEstimator(distribution="weibull").fit(X, y)
        response = model.predict(X)
        median = model.predict_median(X)
        quantile = model.predict_quantile(X, 0.9)

        model.set_params(distribution="gaussian")

        np.testing.assert_array_equal(model.predict(X), response)
        np.testing.assert_array_equal(model.predict_median(X), median)
        np.testing.assert_array_equal(model.predict_quantile(X, 0.9), quantile)

    @pytest.mark.parametrize(("q", "intercept"), [(0.1, 1000.0), (0.9, -1000.0)])
    def test_quantile_combines_location_and_scale_before_response_transform(self, q, intercept):
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([np.exp([-3.0, -2.1, -0.2, 0.0, 0.7, 1.8]), np.ones(6)])
        model = AFTEstimator(distribution="lognormal").fit(X, y)
        model.intercept_ = intercept
        model.coef_ = np.zeros(1)
        model.scale_ = 1000.0
        error_quantile = _surv._survival.survreg_distribution(
            [q], [0.0], [1.0], "gaussian", "quantile"
        )[0]

        with np.errstate(over="raise", under="raise", invalid="raise"):
            prediction = model.predict_quantile(X, q)

        np.testing.assert_allclose(prediction, np.exp(intercept + 1000.0 * error_quantile))
        assert np.all(np.isfinite(prediction) & (prediction > 0.0))

    @pytest.mark.parametrize("method", ["predict", "predict_median", "predict_quantile"])
    def test_prediction_methods_validate_feature_count(self, method):
        X = np.array([[0.1], [0.2], [0.3], [0.4]])
        y = np.column_stack([[1.0, 2.0, 3.0, 4.0], np.ones(4)])
        model = AFTEstimator().fit(X, y)

        with pytest.raises(ValueError, match="X has 2 features, but model expects 1"):
            getattr(model, method)(np.ones((2, 2)))

    def test_score_accepts_signed_responses(self):
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.column_stack([[-3.0, -2.1, -0.2, 0.0, 0.7, 1.8], np.ones(6)])
        model = AFTEstimator(distribution="gaussian").fit(X, y)

        # All 15 pairs are concordant except the pair tied at x=0, which gets half credit.
        assert model.score(X, y) == pytest.approx(29.0 / 30.0)

    def test_acceleration_factors(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = AFTEstimator(distribution="weibull")
        model.fit(X, y)
        af = model.acceleration_factors

        assert len(af) == 2
        assert all(af > 0)

    def test_different_distributions(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        for dist in ["weibull", "lognormal", "loglogistic"]:
            model = AFTEstimator(distribution=dist)
            model.fit(X, y)
            assert model.n_features_in_ == 2
            assert np.isfinite(model.intercept_)

    def test_not_enough_events(self):
        X = np.random.randn(10, 5)
        y = np.column_stack([np.exp(np.random.randn(10)), np.zeros(10)])
        y[0, 1] = 1
        y[1, 1] = 1

        model = AFTEstimator(distribution="weibull")
        with pytest.raises(ValueError, match="Not enough events"):
            model.fit(X, y)


class TestStreamingAFTEstimator:
    def test_streaming_fit(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = StreamingAFTEstimator(distribution="weibull")
        model.fit(X, y)

        assert model.n_features_in_ == 2
        assert np.isfinite(model.intercept_)

    def test_streaming_predict_batched(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        log_time = 1.0 + X @ np.array([0.5, -0.3]) + 0.5 * np.random.randn(n)
        time = np.exp(log_time)
        y = np.column_stack([time, np.ones(n)])

        model = StreamingAFTEstimator(distribution="weibull")
        model.fit(X, y)

        predictions = list(model.predict_batched(X, batch_size=20))
        all_predictions = np.concatenate(predictions)

        assert len(all_predictions) == n
        assert all(all_predictions > 0)
