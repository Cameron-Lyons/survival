"""Cox computations agree under stratum shifts that cancel from the likelihood."""

import copy
import pickle

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
native = survival._survival
r = survival.r


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("shift", [1000.0, 1e12, 1e16, 1e308])
def test_stratum_scale_preserves_fits_and_subject_probabilities(ties, counting, shift):
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    offsets = [shift, shift, -shift, -shift]
    options = {"entry": [0.0] * 4} if counting else {}
    arguments = {"method": ties, "strata": [1, 1, 2, 2], "init": [0.4], "iter_max": 0}
    reference = native.coxph_fit([1.0, 2.0, 1.0, 2.0], [1, 0, 1, 0], x, **arguments, **options)
    fit = native.coxph_fit(
        [1.0, 2.0, 1.0, 2.0], [1, 0, 1, 0], x, offset=offsets, **arguments, **options
    )
    for field in ["coefficients", "var", "loglik", "residuals", "first"]:
        np.testing.assert_allclose(getattr(fit, field), getattr(reference, field), rtol=2e-10)
    np.testing.assert_allclose(
        fit.predict_survival_at([0.0, 1.0, 2.0]),
        reference.predict_survival_at([0.0, 1.0, 2.0]),
        rtol=2e-10,
    )
    if ties != "exact":
        np.testing.assert_allclose(fit.score_residuals(), reference.score_residuals(), rtol=2e-10)
        np.testing.assert_allclose(
            fit.schoenfeld_residuals().residuals,
            reference.schoenfeld_residuals().residuals,
            rtol=2e-10,
        )


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("shift", [-1e308, 1e308])
@pytest.mark.parametrize("opposite_strata", [False, True])
def test_large_offsets_keep_covariates_after_copy_and_pickle(protocol, shift, opposite_strata):
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    options = {"strata": [1, 1, 2, 2], "init": [0.4], "iter_max": 0}
    reference = native.coxph_fit([1.0, 2.0, 1.0, 2.0], [1, 0, 1, 0], x, **options)
    offsets = [shift, shift, -shift, -shift] if opposite_strata else [shift] * 4
    fit = native.coxph_fit([1.0, 2.0, 1.0, 2.0], [1, 0, 1, 0], x, offset=offsets, **options)
    fit.basehaz()  # Serialized state must work independently of populated curve caches.
    for restored in [pickle.loads(pickle.dumps(fit, protocol)), copy.copy(fit), copy.deepcopy(fit)]:
        assert restored.linear_predictors == fit.linear_predictors
        np.testing.assert_allclose(
            restored.predict_survival_at([1.0, 2.0]),
            reference.predict_survival_at([1.0, 2.0]),
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            restored.score_residuals(), reference.score_residuals(), rtol=2e-10
        )


def test_formula_initialization_centers_large_finite_offsets_without_overflow():
    data = {
        "time": [1.0, 2.0, 1.0, 2.0],
        "status": [1, 0, 1, 0],
        "x": [0.0, 1.0, 0.0, 1.0],
        "group": [1, 1, 2, 2],
        "off": [-1e308] * 4,
    }
    fit = r.coxph(
        "Surv(time, status) ~ x + strata(group) + offset(off)", data, init=[0.4], iter_max=0
    )
    reference = r.coxph("Surv(time, status) ~ x + strata(group)", data, init=[0.4], iter_max=0)
    np.testing.assert_allclose(r.residuals(fit), r.residuals(reference), rtol=2e-10)
    np.testing.assert_allclose(
        r.predict(fit, type="expected"), r.predict(reference, type="expected"), rtol=2e-10
    )


@pytest.mark.parametrize("coefficient", [-7.0, 0.0, 19.0, 20.0, 100.0])
@pytest.mark.parametrize("deaths", [1, 2])
def test_kalbfleisch_prentice_equal_risks_reduce_to_kaplan_meier(coefficient, deaths):
    x = np.ones((2 * deaths, 1))
    fit = native.coxph_fit(
        [1.0] * deaths + [2.0] * deaths,
        [1] * deaths + [0] * deaths,
        x,
        init=[coefficient],
        iter_max=0,
    )
    curves = fit.survfit(newdata=[[1.0]], stype=1)
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0].surv, 0.5, rtol=2e-12)
    # Retain R's separate vartype=1 formula d/[R*(R-d)], with R²
    # replacing R*(R-d) when R<=d. Unlike the point estimate it depends on scale.
    risk = np.exp(coefficient)
    variance = 1.0 / (2 * deaths * (2 - 1 / risk)) if risk > 0.5 else 1.0 / (4 * deaths)
    np.testing.assert_allclose(np.square(curves[0].std_err), variance, rtol=2e-12)


def test_kalbfleisch_prentice_tied_deaths_allow_unequal_weights_and_survivor_risks():
    fit = native.coxph_fit(
        [1.0, 1.0, 2.0, 2.0],
        [1, 1, 0, 0],
        np.empty((4, 0)),
        weights=[1.0, 3.0, 2.0, 0.5],
        offset=np.log([2.0, 2.0, 1.0, 4.0]),
    )
    curve = fit.survfit(newdata=np.empty((1, 0)), new_offset=[np.log(2.0)], stype=1)[0]
    # Weighted death risk 8 and surviving risk 4 give S=1/(1+8/4).
    np.testing.assert_allclose(curve.surv, 1.0 / 3.0, rtol=2e-12)


def test_terminal_kalbfleisch_prentice_survival_is_zero_at_every_positive_risk():
    fit = native.coxph_fit(
        [1.0, 1.0], [1, 1], np.empty((2, 0)), strata=[1, 2], offset=[1e308, -1e308]
    )
    curves = fit.survfit(newdata=np.empty((1, 0)), new_offset=[-1e308], stype=1)
    assert len(curves) == 2
    for curve in curves:
        np.testing.assert_array_equal(curve.surv, 0.0)


@pytest.mark.parametrize("stype", [1, 2])
def test_event_free_extreme_offsets_have_unit_survival_and_zero_uncertainty(stype):
    x = np.empty((4, 0))
    offsets = [1e308, 1e308, 1e308, -1e308]
    strata = [0, 0, 0, 1]
    fit = native.coxph_fit([1.0, 2.0, 3.0, 4.0], [0] * 4, x, strata=strata, offset=offsets)
    newdata = {"newdata": x, "new_strata": strata, "new_offset": offsets}
    for rows in [{}, newdata]:
        np.testing.assert_array_equal(fit.predict_survival_at([0.0, 5.0], **rows), 1.0)
        expected_rows = {**rows, "new_time": [4.0] * 4} if rows else rows
        expected = fit.predict("expected", se_fit=True, **expected_rows)
        np.testing.assert_array_equal(expected.fit, 0.0)
        np.testing.assert_array_equal(expected.se_fit, 0.0)
    for curve in fit.survfit(**newdata, stype=stype):
        np.testing.assert_array_equal(curve.surv, 1.0)
        np.testing.assert_array_equal(curve.cumhaz, 0.0)
        np.testing.assert_array_equal(curve.std_err, 0.0)


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("prediction", ["expected", "survival"])
@pytest.mark.parametrize("missing", ["offset", "covariate"])
@pytest.mark.parametrize("payload", [0x7FF8000000000000, 0x7FF80000000007A2], ids=["NaN", "R_NA"])
def test_missing_newdata_remains_missing_when_no_events_have_occurred(
    events, prediction, missing, payload
):
    missing_value = np.array([payload], dtype=np.uint64).view(np.float64)[0]
    x = np.empty((2, 0)) if missing == "offset" else np.array([[0.0], [1.0]])
    fit = native.coxph_fit([1.0, 2.0], [int(events), 0], x, init=[0.3] * x.shape[1], iter_max=0)
    result = fit.predict(
        prediction,
        newdata=x if missing == "offset" else [[missing_value], [0.0]],
        new_offset=[missing_value if missing == "offset" else 0.0, 0.0],
        new_time=[0.5, 0.5],
        se_fit=True,
    )
    assert np.isnan(result.fit[0])
    assert np.isnan(result.se_fit[0])
    for value in [result.fit[0], result.se_fit[0]]:
        bits = int(np.array([value], dtype=np.float64).view(np.uint64)[0])
        assert bits & 0xFFFFFFFF == payload & 0xFFFFFFFF
    assert result.fit[1] == (0.0 if prediction == "expected" else 1.0)
    assert result.se_fit[1] == 0.0


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("coefficient", [-1e308, -1e16, 1e16, 1e308])
def test_large_constant_covariate_keeps_small_effects_in_fits_and_predictions(
    ties, counting, coefficient
):
    z = np.array([0.0, 1.0, -0.5, 0.7, 1.2, -0.2])[:, None]
    x = np.column_stack((np.ones(len(z)), z))
    time = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0]
    status = [1, 1, 0, 1, 0, 0]
    options = {"method": ties, "iter_max": 0}
    if counting:
        options["entry"] = [0.0] * len(z)
    reference = native.coxph_fit(time, status, z, init=[0.4], **options)
    fit = native.coxph_fit(time, status, x, init=[coefficient, 0.4], **options)
    for restored in [fit, pickle.loads(pickle.dumps(fit))]:
        np.testing.assert_allclose(restored.loglik, reference.loglik, rtol=2e-12)
        np.testing.assert_allclose(restored.residuals, reference.residuals, rtol=2e-12)
        np.testing.assert_allclose(
            restored.predict("expected").fit, reference.predict("expected").fit, rtol=2e-12
        )
        expected_survival = reference.predict_survival_at([1.0, 2.0, 4.0])
        for newdata in [None, x]:
            np.testing.assert_allclose(
                restored.predict_survival_at([1.0, 2.0, 4.0], newdata=newdata),
                expected_survival,
                rtol=2e-12,
            )


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
@pytest.mark.parametrize("shift", [-1e308, -1e16, 1e16, 1e308])
def test_disjoint_risk_scales_retain_covariates_weights_and_pickle_state(ties, shift):
    x = np.array([0.0, 1.0, -0.5, 0.0, 1.0, -0.5])[:, None]
    entry = [0.0] * 3 + [2.0] * 3
    time = [1.0] * 3 + [3.0] * 3
    status = [1, 1, 0] * 2
    weights = [1.0] * 6 if ties == "exact" else [1.0, 3.0, 2.0] * 2
    offsets = [0.0] * 3 + [shift] * 3
    options = {"method": ties, "init": [0.4], "iter_max": 0, "entry": entry, "weights": weights}
    reference = native.coxph_fit(time, status, x, strata=[0] * 3 + [1] * 3, **options)
    fit = native.coxph_fit(time, status, x, offset=offsets, **options)
    expected = reference.predict("expected").fit
    expected_new = reference.predict(
        "expected",
        newdata=x,
        new_entry=entry,
        new_time=time,
        new_strata=[0] * 3 + [1] * 3,
    ).fit
    for restored in [fit, pickle.loads(pickle.dumps(fit)), copy.deepcopy(fit)]:
        for field in ["loglik", "first", "var", "residuals"]:
            np.testing.assert_allclose(
                getattr(restored, field), getattr(reference, field), rtol=2e-11
            )
        np.testing.assert_allclose(restored.predict("expected").fit, expected, rtol=2e-11)
        np.testing.assert_allclose(
            restored.predict(
                "expected", newdata=x, new_entry=entry, new_time=time, new_offset=offsets
            ).fit,
            expected_new,
            rtol=2e-11,
        )
        curves = restored.survfit_individual(x, entry, time, list(range(6)), new_offset=offsets)
        for row, curve in enumerate(curves):
            np.testing.assert_allclose(np.asarray(curve.cumhaz)[-1], expected_new[row], rtol=2e-11)
            np.testing.assert_allclose(
                np.asarray(curve.surv)[-1], np.exp(-expected_new[row]), rtol=2e-11
            )
