import importlib

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_surv = survival.regression
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

    def test_survreg_small_sample_matches_r(self):
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

        # survreg(Surv(1:5, rep(1, 5)) ~ 1): (Intercept), Log(scale), scale, loglik
        assert isinstance(result, _surv.SurvregFit)
        assert result.coefficients == pytest.approx(
            [1.2220948192663361, np.log(0.43595653120301259)]
        )
        assert result.scale == pytest.approx([0.43595653120301259])
        assert result.log_likelihood == pytest.approx(-8.6710937992775285)
        assert result.intercept_only_log_likelihood == pytest.approx(-8.6710937992775285)
        assert result.converged
        assert result.distribution.name == "Weibull"


# R: d <- data.frame(t = 1:8, s = c(1,1,0,1,1,1,0,1), x = c(.5,.2,.9,.1,.7,.3,.8,.4))
#    survreg(Surv(t, s) ~ x, d, dist = <dist>); predict(newdata = data.frame(x = c(.25, .75)))
_TOY_X = np.array([[0.5], [0.2], [0.9], [0.1], [0.7], [0.3], [0.8], [0.4]], dtype=np.float64)
_TOY_Y = np.column_stack([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 0, 1, 1, 1, 0, 1]])
_TOY_NEW = np.array([[0.25], [0.75]], dtype=np.float64)
_R_SURVREG = {
    "weibull": {
        "coef": (1.1636656987433951, 1.3788867351883287),
        "scale": 0.497638477465639,
        "loglik": -14.307025676163946,
        "response": (4.5194367919992153, 9.0054572090245308),
        "q50": (3.765936055880589, 7.504027064432754),
        "q90": (6.8444220326719512, 13.638236924753228),
    },
    "lognormal": {
        "coef": (0.87233357022867131, 1.3801613779637922),
        "scale": 0.75008271096402956,
        "loglik": -15.235456438042803,
        "response": (3.3783043591269042, 6.7359213695243323),
        "q50": (3.3783043591269042, 6.7359213695243323),
        "q90": (8.8343151503484076, 17.614532582174917),
    },
    "gaussian": {
        "coef": (2.7599598137086114, 5.2840325854668642),
        "scale": 2.4821327719353752,
        "loglik": -14.77042276067333,
        "response": (4.080967960075327, 6.7229842528087591),
        "q50": (4.080967960075327, 6.7229842528087591),
        "q90": (7.2619490998386667, 9.9039653925720987),
    },
    "logistic": {
        "coef": (2.9386494175551521, 5.1242398913335849),
        "scale": 1.4791564979637573,
        "loglik": -14.992074618789822,
        "response": (4.2197093903885481, 6.781829336055341),
        "q50": (4.2197093903885481, 6.781829336055341),
        "q90": (7.4697484014410875, 10.031868347107881),
    },
}


class TestAFTEstimatorAgainstR:
    @pytest.mark.parametrize("distribution", sorted(_R_SURVREG))
    def test_fit_and_predictions_match_r_survreg(self, distribution):
        reference = _R_SURVREG[distribution]
        model = AFTEstimator(distribution=distribution).fit(_TOY_X, _TOY_Y)

        assert model.intercept_ == pytest.approx(reference["coef"][0])
        assert model.coef_ == pytest.approx([reference["coef"][1]])
        assert model.scale_ == pytest.approx(reference["scale"])
        assert model.converged_
        assert model.model_.log_likelihood == pytest.approx(reference["loglik"])

        # type = "response": exp(lp) for the log-transformed distributions, lp otherwise
        assert model.predict(_TOY_NEW) == pytest.approx(reference["response"])
        assert model.predict_median(_TOY_NEW) == pytest.approx(reference["q50"])
        assert model.predict_quantile(_TOY_NEW, q=0.9) == pytest.approx(reference["q90"])
        # concordance(fit): larger predicted time goes with longer survival
        assert model.score(_TOY_X, _TOY_Y) == pytest.approx(0.68181818181818188)

    def test_untransformed_distributions_are_not_exponentiated(self):
        gaussian = AFTEstimator(distribution="gaussian").fit(_TOY_X, _TOY_Y)
        design = np.column_stack([np.ones(2), _TOY_NEW])
        linear = design @ np.array([gaussian.intercept_, *gaussian.coef_])

        assert gaussian.predict(_TOY_NEW) == pytest.approx(linear)
        assert gaussian.predict_median(_TOY_NEW) == pytest.approx(linear)

        weibull = AFTEstimator(distribution="weibull").fit(_TOY_X, _TOY_Y)
        linear = design @ np.array([weibull.intercept_, *weibull.coef_])
        assert weibull.predict(_TOY_NEW) == pytest.approx(np.exp(linear))

    def test_feature_count_is_validated(self):
        model = AFTEstimator().fit(_TOY_X, _TOY_Y)
        with pytest.raises(ValueError, match="expects 1"):
            model.predict(np.array([[0.1, 0.2]], dtype=np.float64))
        with pytest.raises(ValueError, match="expects 1"):
            model.predict_quantile(np.array([[0.1, 0.2]], dtype=np.float64))
        with pytest.raises(ValueError, match="not fitted"):
            AFTEstimator().predict(_TOY_NEW)


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

    def test_predict_quantile_rejects_nonfinite_q(self):
        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64)
        y = np.column_stack([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])

        model = AFTEstimator(distribution="weibull", max_iter=10)
        model.fit(X, y)

        with pytest.raises(ValueError, match="q must be between 0 and 1"):
            model.predict_quantile(X[:2], q=np.nan)

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
