import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_agsurv4():
    ndeath = [1, 1, 0, 1, 0]
    risk = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0]
    sn = 5
    denom = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = survival.agsurv4(ndeath, risk, wt, sn, denom)
    assert isinstance(result, list)
    assert len(result) == sn


def test_agsurv5():
    n = 5
    nvar = 2
    dd = [1, 1, 2, 1, 1]
    x1 = [10.0, 9.0, 8.0, 7.0, 6.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0]
    xsum = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    xsum2 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.5, 2.0, 1.5, 1.0, 0.5]

    result = survival.agsurv5(n, nvar, dd, x1, x2, xsum, xsum2)
    assert isinstance(result, dict)
    assert "sum1" in result
    assert "sum2" in result
    assert "xbar" in result


def test_agmart():
    n = 5
    method = 0
    start = [0.0, 0.0, 1.0, 1.0, 2.0]
    stop = [1.0, 2.0, 2.0, 3.0, 3.0]
    event = [1, 0, 1, 0, 1]
    score = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    strata = [1, 0, 0, 0, 0]

    result = survival.agmart(n, method, start, stop, event, score, wt, strata)
    assert isinstance(result, list)
    assert len(result) == n


def test_survfitkm():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    result = survival.survfitkm(
        time=time,
        status=status,
        weights=None,
        entry_times=None,
        position=None,
        reverse=False,
        computation_type=0,
    )
    assert hasattr(result, "time")
    assert hasattr(result, "estimate")
    assert hasattr(result, "std_err")
    assert len(result.time) > 0
    assert len(result.estimate) == len(result.time)


def test_survdiff2():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    result = survival.survdiff2(
        time=time,
        status=status,
        group=group,
        strata=None,
        rho=0.0,
    )
    assert hasattr(result, "observed")
    assert hasattr(result, "expected")
    assert hasattr(result, "chi_squared")
    assert len(result.observed) > 0


def test_aggregate_survfit_public_apis():
    result = survival.aggregate_survfit(
        times=[[1.0, 2.0], [1.5, 2.5]],
        survs=[[0.9, 0.8], [0.95, 0.85]],
        std_errs=[[0.05, 0.06], [0.04, 0.05]],
        weights=[2.0, 1.0],
        conf_level=0.95,
    )
    by_group = survival.aggregate_survfit_by_group(
        times=[[1.0, 2.0], [1.0, 2.0], [1.5, 2.5]],
        survs=[[0.9, 0.8], [0.8, 0.7], [0.95, 0.85]],
        groups=[1, 1, 2],
        weights=[1.0, 2.0, 1.0],
    )
    by_group_sorted = sorted(by_group, key=lambda item: item.n_curves)

    assert result.n_curves == 2
    assert result.time == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert result.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert len(result.surv) == len(result.time)
    assert len(result.std_err) == len(result.time)
    assert all(0.0 <= value <= 1.0 for value in result.lower)
    assert all(0.0 <= value <= 1.0 for value in result.upper)

    assert len(by_group_sorted) == 2
    assert by_group_sorted[0].n_curves == 1
    assert by_group_sorted[0].surv == pytest.approx([0.95, 0.85])
    assert by_group_sorted[1].n_curves == 2
    assert by_group_sorted[1].weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0])

    with pytest.raises(ValueError, match="times and survs must have same length"):
        survival.aggregate_survfit([[1.0]], [[0.9], [0.8]], None, None, None)


def test_survcheck_public_apis_and_validation():
    result = survival.survcheck(
        id=[1, 1, 2],
        time1=[0.0, 2.0, 0.0],
        time2=[2.0, 4.0, 1.0],
        status=[1, 0, 1],
        istate=[0, 1, 0],
    )
    simple = survival.survcheck_simple(
        time=[1.0, -1.0, float("nan")],
        status=[1, 0, 2],
    )

    assert result.n_subjects == 2
    assert result.n_transitions == 3
    assert result.n_problems == 0
    assert result.is_valid is True
    assert result.transitions == {"0 -> 1": 2, "1 -> 0": 1}
    assert result.flags == [0, 0, 0]

    assert simple.n_subjects == 3
    assert simple.n_problems == 2
    assert simple.is_valid is False
    assert simple.invalid_ids == [1, 2]
    assert simple.flags == [0, 4, 4]
    assert any("negative time" in message for message in simple.messages)

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.survcheck([1], [0.0], [1.0, 2.0], [1], None)


def test_royston_public_apis_and_validation():
    result = survival.royston(
        linear_predictor=[0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.9, -0.2],
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        status=[1, 1, 1, 0, 1, 1, 1, 0],
    )
    from_model = survival.royston_from_model(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        coef=[0.5, 1.0],
        n_obs=3,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
    )

    assert result.d > 0.0
    assert result.se > 0.0
    assert 0.0 <= result.r_squared_d <= 1.0
    assert 0.0 <= result.r_squared_ko <= 1.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_events == 6
    assert from_model.n_events == 3

    with pytest.raises(ValueError, match="At least 2 events required"):
        survival.royston([0.1, 0.2], [1.0, 2.0], [1, 0])


def test_yates_public_apis_and_validation():
    result = survival.yates(
        predictions=[1.0, 2.0, 2.0, 4.0],
        factor=["A", "A", "B", "B"],
        weights=[1.0, 2.0, 1.0, 1.0],
        conf_level=0.95,
    )
    pairwise = survival.yates_pairwise(result)
    contrast = survival.yates_contrast(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        coef=[0.5, 1.0],
        n_obs=3,
        n_vars=2,
        factor_col=0,
        factor_levels=[0.0, 1.0],
        predict_type="risk",
    )

    assert result.levels == ["A", "B"]
    assert result.means == pytest.approx([5.0 / 3.0, 3.0])
    assert result.n == [2, 2]
    assert result.predict_type == "linear"
    assert pairwise.level1 == ["A"]
    assert pairwise.level2 == ["B"]
    assert pairwise.difference == pytest.approx([-4.0 / 3.0])
    assert contrast.levels == ["0", "1"]
    assert contrast.predict_type == "risk"
    assert contrast.means[1] > contrast.means[0]

    with pytest.raises(ValueError, match="weights must have same length as predictions"):
        survival.yates([1.0], ["A"], [1.0, 2.0], 0.95)


def test_concordance_metric_public_apis_and_validation():
    uno = survival.uno_c_index(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    comparison = survival.compare_uno_c_indices(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score_1=[0.9, 0.7, 0.4, 0.2],
        risk_score_2=[0.2, 0.4, 0.7, 0.9],
    )
    decomposition = survival.c_index_decomposition(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    gonen = survival.gonen_heller_concordance([0.9, 0.7, 0.4, 0.2])

    assert uno.c_index == pytest.approx(1.0)
    assert uno.comparable_pairs > 0.0
    assert 0.0 <= uno.ci_lower <= uno.ci_upper <= 1.0
    assert comparison.c_index_1 > comparison.c_index_2
    assert comparison.difference > 0.0
    assert 0.0 <= comparison.p_value <= 1.0
    assert decomposition.c_index == pytest.approx(1.0)
    assert 0.0 <= decomposition.alpha <= 1.0
    assert decomposition.n_event_event_pairs > 0
    assert 0.0 <= gonen.cpe <= 1.0
    assert gonen.std_error >= 0.0

    with pytest.raises(ValueError, match="time, status, and risk_score must have the same length"):
        survival.uno_c_index([1.0], [1], [0.1, 0.2])


def test_rcll_public_apis_and_validation():
    result = survival.rcll(
        survival_predictions=[
            [0.95, 0.85, 0.70],
            [0.90, 0.75, 0.55],
            [0.98, 0.92, 0.80],
        ],
        prediction_times=[1.0, 2.0, 3.0],
        event_times=[2.5, 1.5, 3.0],
        status=[1, 1, 0],
        weights=[1.0, 2.0, 1.0],
    )
    single_time = survival.rcll_single_time(
        survival_probs=[0.8, 0.7, 0.9],
        event_times=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        prediction_time=2.0,
        weights=[1.0, 2.0, 1.0],
    )

    assert result.n_events == 2
    assert result.n_censored == 1
    assert result.mean_rcll > 0.0
    assert result.event_contribution > result.censored_contribution
    assert single_time.n_events == 1
    assert single_time.n_censored == 2
    assert single_time.rcll > 0.0

    with pytest.raises(ValueError, match="survival_predictions row 0 has 1 elements, expected 2"):
        survival.rcll([[0.9]], [1.0, 2.0], [1.0], [1], None)


def test_ridge_public_apis_and_validation():
    penalty = survival.RidgePenalty(0.1, False)
    fit = survival.ridge_fit(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        n_obs=3,
        n_vars=2,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        penalty=penalty,
        weights=None,
    )
    best_theta, cv_scores = survival.ridge_cv(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        n_obs=3,
        n_vars=2,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        theta_grid=[0.01, 0.1, 1.0],
        n_folds=2,
    )

    assert penalty.penalty_value([1.0, 2.0]) == pytest.approx(0.25)
    assert penalty.penalty_gradient([1.0, 2.0]) == pytest.approx([0.1, 0.2])
    assert penalty.apply_to_information([1.0, 2.0]) == pytest.approx([1.1, 2.1])
    assert len(fit.coefficients) == 2
    assert len(fit.std_err) == 2
    assert fit.theta == pytest.approx(0.1)
    assert fit.df > 0.0
    assert fit.scale_factors is None
    assert best_theta in [0.01, 0.1, 1.0]
    assert len(cv_scores) == 3

    with pytest.raises(ValueError, match="x length must equal n_obs \\* n_vars"):
        survival.ridge_fit([1.0], 2, 1, [1.0, 2.0], [1, 1], penalty, None)


def test_anova_and_basehaz_public_apis():
    anova = survival.anova_coxph(
        logliks=[-10.0, -8.0, -7.5],
        dfs=[1, 2, 3],
        model_names=["null", "m1", "m2"],
        test="LRT",
    )
    single = survival.anova_coxph_single(-10.0, -8.0, 1, 2)
    times, hazard = survival.basehaz(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 0, 1, 1],
        linear_predictors=[0.0, 0.2, 0.1, -0.1],
        centered=False,
    )

    assert anova.test_type == "LRT"
    assert len(anova.rows) == 3
    assert anova.rows[1].chisq == pytest.approx(4.0)
    assert "Analysis of Deviance Table" in anova.to_table()
    assert len(single.rows) == 2
    assert single.rows[0].model_name == "Null"
    assert single.rows[1].chisq == pytest.approx(4.0)
    assert times == pytest.approx([1.0, 3.0, 4.0])
    assert len(hazard) == 3
    assert hazard[0] < hazard[1] < hazard[2]

    with pytest.raises(ValueError, match="Need at least 2 models for comparison"):
        survival.anova_coxph([-1.0], [1], None, "LRT")

    with pytest.raises(
        ValueError,
        match="time, status, and linear_predictors must have the same length",
    ):
        survival.basehaz([1.0], [1, 0], [0.1], False)
