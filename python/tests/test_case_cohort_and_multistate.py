"""Case-cohort (``cch_fit``), conditional logistic (``coxph`` with exact ties) and multi-state
(``survfitaj``) entry points against R survival 3.8.11."""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
regression = survival.regression
surv_analysis = survival.surv_analysis


# ---------------------------------------------------------------------------------------------
# clogit = coxph(Surv(rep(1, n), case) ~ x + strata(stratum), ties = "exact")
# ---------------------------------------------------------------------------------------------


def _clogit(case, x, stratum):
    return regression.coxph_fit(
        [1.0] * len(case), case, [[v] for v in x], strata=stratum, method="exact"
    )


def test_conditional_logistic_regression_is_a_stratified_exact_cox_fit():
    fit = _clogit([1, 0, 1, 0], [2.0, 1.0, 3.0, 1.0], [0, 0, 1, 1])

    # both matched pairs are perfectly separated: the exact partial likelihood diverges
    assert fit.method == regression.TieMethod.Exact
    assert fit.nevent == 2
    assert fit.coefficients[0] > 5.0
    assert fit.loglik[1] > fit.loglik[0]

    balanced = _clogit([1, 0, 0, 1, 0, 0], [2.0, 1.0, 3.0, 1.0, 2.0, 0.5], [0, 0, 0, 1, 1, 1])
    assert math.isfinite(balanced.coefficients[0])
    assert balanced.var[0][0] > 0.0


def test_conditional_logistic_regression_is_invariant_to_row_order():
    case, x, stratum = [1, 0, 0, 1, 0, 0], [2.0, 1.0, 3.0, 1.0, 2.0, 0.5], [0, 0, 0, 1, 1, 1]
    forward = _clogit(case, x, stratum)
    backward = _clogit(case[::-1], x[::-1], stratum[::-1])

    assert backward.coefficients == pytest.approx(forward.coefficients)
    assert backward.loglik == pytest.approx(forward.loglik)
    assert backward.var[0] == pytest.approx(forward.var[0])


def test_conditional_logistic_regression_uses_strata():
    case, x = [1, 0, 0, 1, 0, 0], [2.0, 1.0, 3.0, 1.0, 2.0, 0.5]
    matched = _clogit(case, x, [0, 0, 0, 1, 1, 1])
    pooled = _clogit(case, x, [0] * 6)

    assert matched.coefficients != pytest.approx(pooled.coefficients)
    assert matched.loglik[1] != pytest.approx(pooled.loglik[1])


# ---------------------------------------------------------------------------------------------
# cch on the nwtco case-cohort subset (?cch), age in years as the only covariate
# ---------------------------------------------------------------------------------------------


def _nwtco_case_cohort():
    data = survival.datasets.load_nwtco()
    rel = [int(v) for v in data["rel"]]
    in_subcohort = [bool(v) for v in data["in.subcohort"]]
    rows = [i for i in range(len(rel)) if rel[i] == 1 or in_subcohort[i]]
    return {
        "stop": [float(data["edrel"][i]) for i in rows],
        "status": [rel[i] for i in rows],
        "x": [[float(data["age"][i]) / 12.0] for i in rows],
        "subcohort": [in_subcohort[i] for i in rows],
        "id": [int(data["seqno"][i]) for i in rows],
        "stratum": [0 if int(data["instit"][i]) == 1 else 1 for i in rows],
    }


@pytest.mark.parametrize(
    ("method", "coef", "var"),
    [
        ("Prentice", 0.0794648551679581, 0.000379759657941196),
        ("SelfPrentice", 0.079638780238827866, 0.00037975965794117265),
        ("LinYing", 0.080359785868969108, 0.00036037403539673535),
    ],
)
def test_cch_fit_matches_r(method, coef, var):
    data = _nwtco_case_cohort()
    fit = regression.cch_fit(
        data["stop"], data["status"], data["x"], data["subcohort"], data["id"], 4028, method=method
    )

    # cch(Surv(edrel, rel) ~ age, subcoh = ~subcohort, id = ~seqno, cohort.size = 4028)
    assert isinstance(fit, regression.CchFitResult)
    assert fit.method == method
    assert fit.coefficients == pytest.approx([coef])
    assert fit.var[0] == pytest.approx([var])
    assert fit.naive_var[0] == pytest.approx([var])
    assert fit.cohort_size == [4028]
    assert fit.subcohort_size == [668]
    assert fit.stratified is False
    assert isinstance(fit.fit, regression.CoxPHFit)


@pytest.mark.parametrize(
    ("method", "coef", "var"),
    [
        ("I.Borgan", 0.07966642104375915, 0.0003802667380415622),
        ("II.Borgan", 0.080369599617685111, 0.00036089744345170179),
    ],
)
def test_cch_borgan_fit_matches_r(method, coef, var):
    data = _nwtco_case_cohort()
    fit = regression.cch_borgan_fit(
        data["stop"],
        data["status"],
        data["x"],
        data["subcohort"],
        data["id"],
        data["stratum"],
        [3622, 406],
        method=method,
    )

    # cch(..., stratum = ~stratum, cohort.size = c("1" = 3622, "2" = 406))
    assert fit.method == method
    assert fit.coefficients == pytest.approx([coef])
    assert fit.var[0] == pytest.approx([var])
    assert fit.subcohort_size == [952, 202]
    assert fit.cohort_size == [3622, 406]
    assert fit.stratified is True
    assert fit.stratum[:3] == data["stratum"][:3]


def test_cch_validates_inputs():
    data = _nwtco_case_cohort()
    with pytest.raises(ValueError, match="method"):
        regression.cch_fit(
            data["stop"],
            data["status"],
            data["x"],
            data["subcohort"],
            data["id"],
            4028,
            method="bogus",
        )
    with pytest.raises(ValueError, match="stratum codes must index every value"):
        regression.cch_borgan_fit(
            data["stop"],
            data["status"],
            data["x"],
            data["subcohort"],
            data["id"],
            [s + 1 for s in data["stratum"]],
            [3622, 406],
        )


# ---------------------------------------------------------------------------------------------
# survfitaj = survfit(Surv(time, state) ~ 1) for multi-state responses
# ---------------------------------------------------------------------------------------------

_AJ_TIME = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
_AJ_STATE = [1, 2, 0, 1, 2, 0]  # 0 censored, 1 = "a", 2 = "b"


def test_survfitaj_competing_risks_match_r():
    fit = surv_analysis.survfitaj(_AJ_TIME, _AJ_STATE, ["a", "b"])

    # survfit(Surv(time, factor(state, levels = c("cens", "a", "b"))) ~ 1)
    assert isinstance(fit, surv_analysis.SurvfitAJResult)
    assert fit.states == ["(s0)", "a", "b"]
    assert fit.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert fit.n == [6]
    assert [row[0] for row in fit.n_risk] == pytest.approx([6.0, 5.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(fit.n_event, [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert [row[0] for row in fit.n_censor] == pytest.approx([0.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        fit.pstate,
        [
            [5 / 6, 1 / 6, 0.0],
            [2 / 3, 1 / 6, 1 / 6],
            [0.44444444444444453, 0.38888888888888890, 1 / 6],
            [0.22222222222222227, 0.38888888888888890, 0.38888888888888895],
            [0.22222222222222227, 0.38888888888888890, 0.38888888888888895],
        ],
    )
    np.testing.assert_allclose(
        fit.cumhaz, [[1 / 6, 0.0], [1 / 6, 0.2], [0.5, 0.2], [0.5, 0.7], [0.5, 0.7]]
    )
    np.testing.assert_allclose(
        fit.std_err[:2],
        [
            [0.15214515486254615, 0.15214515486254615, 0.0],
            [0.19245008972987526, 0.15214515486254615, 0.15214515486254618],
        ],
    )
    np.testing.assert_allclose(fit.std_chaz[1], [0.15214515486254615, 0.17888543819998318])
    np.testing.assert_allclose(fit.lower[0], [0.582654795477139609, 0.027849128299274488, 0.0])
    np.testing.assert_allclose(fit.upper[1], [1.0, 0.99743796212470426, 0.99743796212470448])
    assert fit.p0[0] == pytest.approx([1.0, 0.0, 0.0])
    assert fit.transitions == [[0.0, 2.0, 2.0, 2.0], [0.0] * 4, [0.0] * 4]
    assert (fit.hazard_from, fit.hazard_to) == ([0, 0], [1, 2])
    np.testing.assert_allclose(fit.n_transition, [[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]])
    assert fit.influence_pstate is None
    assert fit.strata is None
    assert fit.type == "mright"


def test_survfitaj_weights_cluster_and_influence_match_r():
    fit = surv_analysis.survfitaj(
        _AJ_TIME,
        _AJ_STATE,
        ["a", "b"],
        weights=[2.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        cluster=[1, 1, 2, 2, 3, 3],
        influence=True,
    )

    # survfit(Surv(time, state) ~ 1, weights = w, cluster = cl, influence = TRUE) with the rows
    # in this order: cluster 1 = {(1, a, w=2), (2, b)}, cluster 2 = {(2, censored), (3, a)},
    # cluster 3 = {(4, b, w=2), (5, censored)}.  Which cluster owns the time-2 "b" event is
    # decided by the row order of the two tied rows (see the next test), so every value below
    # is R's for exactly _AJ_STATE.
    assert [row[0] for row in fit.n_risk] == pytest.approx([8.0, 6.0, 4.0, 3.0, 1.0])
    np.testing.assert_allclose(
        fit.pstate,
        [
            [0.75, 0.25, 0.0],
            [0.625, 0.25, 0.125],
            [0.46875, 0.40625, 0.125],
            [0.15625, 0.40625, 0.4375],
            [0.15625, 0.40625, 0.4375],
        ],
    )
    np.testing.assert_allclose(
        fit.std_err,
        [
            [0.1926379375927805, 0.1926379375927805, 0.0],
            [0.28895690638917082, 0.1926379375927805, 0.096318968796390278],
            [0.28752759718090543, 0.21572970736693869, 0.096318968796390278],
            [0.095842532393635146, 0.21572970736693869, 0.12548733128288289],
            [0.095842532393635146, 0.21572970736693869, 0.12548733128288289],
        ],
    )
    (influence,) = fit.influence_pstate
    assert isinstance(influence, surv_analysis.SurvfitAJInfluence)
    # influence.pstate[cluster, time, state]: three clusters x five times x three states
    np.testing.assert_allclose(
        influence.values,
        [
            [
                [-0.15625, 0.15625, 0.0],
                [-0.234375, 0.15625, 0.078125],
                [-0.17578125, 0.09765625, 0.078125],
                [-0.05859375, 0.09765625, -0.0390625],
                [-0.05859375, 0.09765625, -0.0390625],
            ],
            [
                [0.0625, -0.0625, 0.0],
                [0.09375, -0.0625, -0.03125],
                [-0.046875, 0.078125, -0.03125],
                [-0.015625, 0.078125, -0.0625],
                [-0.015625, 0.078125, -0.0625],
            ],
            [
                [0.09375, -0.09375, 0.0],
                [0.140625, -0.09375, -0.046875],
                [0.22265625, -0.17578125, -0.046875],
                [0.07421875, -0.17578125, 0.1015625],
                [0.07421875, -0.17578125, 0.1015625],
            ],
        ],
    )


def test_survfitaj_cluster_influence_follows_the_row_order_of_tied_times_like_r():
    # Rows 2 and 3 are tied at time 2: one is the "b" event, the other is censored.  R (and
    # this port) attribute the event's influence to the cluster of the row that carries it, so
    # transposing the two rows moves the event from cluster 1 to cluster 2 and changes the
    # robust variance from time 2 on.  Both orderings are R 3.8.11's survfit(..., cluster = cl,
    # influence = TRUE) output for the unweighted data.
    cluster = [1, 1, 2, 2, 3, 3]

    fit = surv_analysis.survfitaj(_AJ_TIME, _AJ_STATE, ["a", "b"], cluster=cluster, influence=True)
    np.testing.assert_allclose(
        fit.std_err,
        [
            [0.13608276348795434, 0.13608276348795434, 0.0],
            [0.27216552697590868, 0.13608276348795434, 0.13608276348795437],
            [0.27715980642769938, 0.21436735005167087, 0.13608276348795437],
            [0.13857990321384969, 0.21436735005167087, 0.11415581486979588],
            [0.13857990321384969, 0.21436735005167087, 0.11415581486979588],
        ],
    )
    np.testing.assert_allclose(
        np.asarray(fit.influence_pstate[0].values)[:, 1, :],
        [
            [-0.22222222222222221, 0.1111111111111111, 0.11111111111111113],
            [0.1111111111111111, -0.055555555555555552, -0.055555555555555552],
            [0.1111111111111111, -0.055555555555555552, -0.055555555555555552],
        ],
    )

    swapped_state = [1, 0, 2, 1, 2, 0]  # the "b" event now sits in cluster 2's row
    swapped = surv_analysis.survfitaj(
        _AJ_TIME, swapped_state, ["a", "b"], cluster=cluster, influence=True
    )
    np.testing.assert_allclose(swapped.pstate, fit.pstate)  # the estimate itself is unchanged
    np.testing.assert_allclose(
        swapped.std_err,
        [
            [0.13608276348795434, 0.13608276348795434, 0.0],
            [0.13608276348795434, 0.13608276348795434, 0.13608276348795434],
            [0.29162992125969672, 0.20454372254050485, 0.13608276348795434],
            [0.14581496062984836, 0.20454372254050485, 0.094426287288755281],
            [0.14581496062984836, 0.20454372254050485, 0.094426287288755281],
        ],
    )
    np.testing.assert_allclose(
        np.asarray(swapped.influence_pstate[0].values)[:, 1, :],
        [
            [-0.055555555555555546, 0.1111111111111111, -0.055555555555555552],
            [-0.05555555555555558, -0.055555555555555552, 0.11111111111111112],
            [0.1111111111111111, -0.055555555555555552, -0.055555555555555552],
        ],
    )


def test_survfitaj_counting_process_with_initial_states_matches_r():
    fit = surv_analysis.survfitaj(
        [2.0, 5.0, 4.0, 3.0, 6.0, 4.0],
        [2, 0, 2, 1, 2, 0],
        ["a", "b"],
        start=[0.0, 2.0, 0.0, 0.0, 3.0, 0.0],
        id=[1, 1, 2, 3, 3, 4],
        istate=["a", "b", "a", "a", "a", "a"],
        istate_levels=["a", "b"],
    )

    # survfit(Surv(start, stop, state) ~ 1, id = id, istate = istate)
    assert fit.states == ["a", "b"]
    assert fit.time == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        fit.pstate, [[0.75, 0.25], [0.75, 0.25], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]]
    )
    np.testing.assert_allclose(fit.n_risk, [[4, 0], [3, 1], [3, 1], [1, 1], [1, 0]])
    assert fit.p0[0] == pytest.approx([1.0, 0.0])
    assert fit.transitions == [[1.0, 3.0, 1.0], [0.0, 0.0, 1.0]]


def test_survfitaj_validates_public_inputs():
    with pytest.raises(ValueError, match="state"):
        surv_analysis.survfitaj([1.0, 2.0], [1, 3], ["a", "b"])
    with pytest.raises(ValueError, match="length"):
        surv_analysis.survfitaj([1.0, 2.0], [1, 0], ["a", "b"], weights=[1.0])
    with pytest.raises(ValueError, match="length"):
        surv_analysis.survfitaj([1.0, 2.0], [1, 0], ["a", "b"], start=[0.0])
    with pytest.raises(ValueError, match="conf.type"):
        surv_analysis.survfitaj(_AJ_TIME, _AJ_STATE, ["a", "b"], conf_type="weird")


def test_illness_death_public_apis_and_validation():
    model = survival.surv_analysis.fit_illness_death(
        entry_time=[0.0, 0.0, 0.0],
        transition_time=[1.0, 0.0, 1.0 + 5e-10],
        exit_time=[2.0, 2.0, 2.0],
        from_state=[0, 0, 0],
        to_state=[1, 2, 1],
        covariates=[[10.0], [20.0], [30.0]],
        config=None,
    )
    prediction = survival.surv_analysis.predict_illness_death(
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
        survival.surv_analysis.fit_illness_death([], [], [], [], [], None, None)

    with pytest.raises(ValueError, match="exit_time contains non-finite"):
        survival.surv_analysis.fit_illness_death([0.0], [0.0], [float("inf")], [0], [0], None, None)

    with pytest.raises(ValueError, match="from_state must contain only 0/1"):
        survival.surv_analysis.fit_illness_death([0.0], [0.0], [1.0], [3], [0], None, None)

    with pytest.raises(ValueError, match="transition_time must be between"):
        survival.surv_analysis.fit_illness_death([0.0], [3.0], [2.0], [0], [1], None, None)

    with pytest.raises(ValueError, match="covariates row 1 has 2 columns"):
        survival.surv_analysis.fit_illness_death(
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [0, 0],
            [1, 2],
            [[1.0], [2.0, 3.0]],
            None,
        )

    with pytest.raises(ValueError, match="current_state must be"):
        survival.surv_analysis.predict_illness_death(model, 3, 0.0, [1.0], None)

    with pytest.raises(ValueError, match="time_in_state must be finite"):
        survival.surv_analysis.predict_illness_death(model, 0, float("inf"), [1.0], None)

    with pytest.raises(ValueError, match="prediction_times contains negative value"):
        survival.surv_analysis.predict_illness_death(model, 0, 0.0, [-1.0], None)


def test_semi_markov_public_apis_and_validation():
    config = survival.surv_analysis.SemiMarkovConfig(3)
    model = survival.surv_analysis.fit_semi_markov(
        [0.0, 1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 1.5],
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        config,
    )
    prediction = survival.surv_analysis.predict_semi_markov(model, 0, 0.5, [0.5, 1.0])

    assert len(model.sojourn_params) == 3
    assert model.get_transition_prob(0, 1) == pytest.approx(1.0)
    assert len(prediction.state_probs) == 2
    assert prediction.time_points == pytest.approx([0.5, 1.0])

    with pytest.raises(ValueError, match="n_states must be positive"):
        survival.surv_analysis.SemiMarkovConfig(0)

    with pytest.raises(ValueError, match="state_names length"):
        survival.surv_analysis.SemiMarkovConfig(3, ["A"])

    with pytest.raises(ValueError, match="absorbing_states must contain values"):
        survival.surv_analysis.SemiMarkovConfig(3, None, None, [3])

    with pytest.raises(ValueError, match="input vectors must be non-empty"):
        survival.surv_analysis.fit_semi_markov([], [], [], [], config)

    with pytest.raises(ValueError, match="exit_times contains non-finite"):
        survival.surv_analysis.fit_semi_markov([0.0], [float("inf")], [0], [1], config)

    with pytest.raises(ValueError, match="entry_times must be <= exit_times"):
        survival.surv_analysis.fit_semi_markov([2.0], [1.0], [0], [1], config)

    with pytest.raises(ValueError, match="from_states must contain values"):
        survival.surv_analysis.fit_semi_markov([0.0], [1.0], [-1], [1], config)

    with pytest.raises(ValueError, match="current_state must be"):
        survival.surv_analysis.predict_semi_markov(model, 3, 0.0, [1.0])

    with pytest.raises(ValueError, match="time_in_state must be finite"):
        survival.surv_analysis.predict_semi_markov(model, 0, float("inf"), [1.0])

    with pytest.raises(ValueError, match="prediction_times contains negative value"):
        survival.surv_analysis.predict_semi_markov(model, 0, 0.0, [-1.0])
