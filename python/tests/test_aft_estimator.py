import numpy as np
import survival._survival as _surv
from survival.sklearn_compat import AFTEstimator, StreamingAFTEstimator


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
        try:
            model.fit(X, y)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Not enough events" in str(e)


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
