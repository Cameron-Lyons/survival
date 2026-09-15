import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
population = survival.population


def test_cipoisson_matches_r():
    exact = survival.validation.cipoisson([5], time=[10.0], p=[0.95])
    assert isinstance(exact, survival.validation.CipoissonResult)
    assert exact.lower == pytest.approx([0.1623486], abs=1e-6)
    assert exact.upper == pytest.approx([1.1668332], abs=1e-6)

    ninety = survival.validation.cipoisson([20], time=[4.0], p=[0.90])
    assert (ninety.lower[0], ninety.upper[0]) == pytest.approx((3.313663, 7.265505), abs=1e-6)
    zero = survival.validation.cipoisson([0], time=[10.0])
    assert (zero.lower[0], zero.upper[0]) == pytest.approx((0.0, 0.3688879), abs=1e-6)

    anscombe = survival.validation.cipoisson([5], time=[10.0], p=[0.95], method="anscombe")
    assert (anscombe.lower[0], anscombe.upper[0]) == pytest.approx((0.1507881, 1.1586004), abs=1e-6)
    assert survival.validation.cipoisson([5], method="a").lower == pytest.approx(
        survival.validation.cipoisson([5], time=[1.0], method="anscombe").lower
    )

    # R recycling of k over time
    vector = survival.validation.cipoisson([0, 5, 20], time=[1.0, 2.0, 10.0])
    assert vector.lower == pytest.approx([0.0, 0.81174319505921044, 1.22165195854039443])
    assert vector.upper == pytest.approx(
        [3.68887945411393536, 5.83416603966133351, 3.08883779026745930]
    )

    with pytest.raises(ValueError, match="method"):
        survival.validation.cipoisson([1], time=[1.0], method="")


def test_cipoisson_supports_r_numeric_edge_cases():
    # R: a degenerate confidence level of 0 collapses to the point estimate
    zero_level = survival.validation.cipoisson([1.2], time=[2.0], p=[0.0])
    assert (zero_level.lower[0], zero_level.upper[0]) == pytest.approx(
        (0.443968106737396, 0.938570591679505)
    )
    full_level = survival.validation.cipoisson([1.2], time=[1.0], p=[1.0])
    assert full_level.lower[0] == 0.0
    assert math.isinf(full_level.upper[0])
    # time <= 0 gives NaN as in R
    nonpositive = survival.validation.cipoisson([1.2], time=[0.0])
    assert all(math.isnan(value) for value in (nonpositive.lower[0], nonpositive.upper[0]))


def test_model_selection_public_apis_and_validation():
    criteria = survival.validation.compute_model_selection_criteria(-100.0, 5, 200, 50, None)
    comparison = survival.validation.compare_models(["m1", "m2"], [-100.0, -95.0], [3, 5], 200)
    cv_score = survival.validation.compute_cv_score([0.75, 0.8, 0.7], "c_index")

    assert criteria.aic > 0.0
    assert "Model Selection Criteria" in criteria.summary()
    assert comparison.best_model_aic == "m2"
    assert len(comparison.likelihood_ratio_tests) == 1
    assert cv_score.n_folds == 3
    ci_95 = cv_score.confidence_interval(0.05)
    ci_90 = cv_score.confidence_interval(0.10)
    assert ci_95[0] < cv_score.mean_score < ci_95[1]
    assert ci_95[0] < ci_90[0]
    assert ci_95[1] > ci_90[1]

    with pytest.raises(ValueError, match="log_likelihood must be finite"):
        survival.validation.compute_model_selection_criteria(float("nan"), 5, 200, 50, None)

    with pytest.raises(ValueError, match="n_obs must be greater than 1"):
        survival.validation.compute_model_selection_criteria(-100.0, 5, 1, 1, None)

    with pytest.raises(ValueError, match="n_events cannot exceed n_obs"):
        survival.validation.compute_model_selection_criteria(-100.0, 5, 20, 21, None)

    with pytest.raises(ValueError, match="log_likelihoods contains non-finite"):
        survival.validation.compare_models(["m1"], [float("inf")], [1], 10)

    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        survival.validation.compare_models(["m1"], [-10.0], [1], 0)

    with pytest.raises(ValueError, match="fold_scores contains non-finite"):
        survival.validation.compute_cv_score([0.7, float("nan")], "c_index")

    with pytest.raises(ValueError, match="metric cannot be empty"):
        survival.validation.compute_cv_score([0.7], " ")

    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        cv_score.confidence_interval(0.0)


def test_hyperparameter_public_apis_and_validation():
    risk = [0.9, 0.7, 0.5, 0.3, 0.1, 0.8]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    event = [1, 1, 1, 0, 1, 0]
    param_grid = [("alpha", [0.5, 1.0])]

    search = survival.validation.hyperparameter_search(risk, time, event, param_grid)
    benchmark = survival.validation.benchmark_models(
        [risk, list(reversed(risk))], ["m1", "m2"], time, event
    )
    nested = survival.validation.nested_cross_validation(risk, time, event, param_grid, 3, 2, 42)

    assert len(search.all_scores) > 0
    assert benchmark.best_model in {"m1", "m2"}
    assert len(nested.outer_scores) == 3
    ci_95 = nested.confidence_interval(0.05)
    ci_90 = nested.confidence_interval(0.10)
    assert ci_95[0] <= ci_90[0]
    assert ci_95[1] >= ci_90[1]

    exact_benchmark = survival.validation.benchmark_models(
        [[0.1, 0.9, 0.2], [0.2, 0.8, 0.3]],
        ["m1", "m2"],
        [1.0, 2.0, 3.0],
        [0, 1, 0],
        2.0,
    )
    near_benchmark = survival.validation.benchmark_models(
        [[0.1, 0.9, 0.2], [0.2, 0.8, 0.3]],
        ["m1", "m2"],
        [1.0, 2.0 + 5e-10, 3.0],
        [0, 1, 0],
        2.0,
    )
    assert near_benchmark.brier_scores == pytest.approx(exact_benchmark.brier_scores)

    bad_config = survival.validation.HyperparameterSearchConfig(cv_folds=0)
    with pytest.raises(ValueError, match="cv_folds must be between 2"):
        survival.validation.hyperparameter_search(risk, time, event, param_grid, bad_config)

    bad_risk = list(risk)
    bad_risk[1] = float("nan")
    with pytest.raises(ValueError, match="risk_scores contains non-finite"):
        survival.validation.hyperparameter_search(bad_risk, time, event, param_grid)

    with pytest.raises(ValueError, match="param_grid values for alpha must not be empty"):
        survival.validation.hyperparameter_search(risk, time, event, [("alpha", [])])

    with pytest.raises(ValueError, match=r"model_predictions\[0\] length must match"):
        survival.validation.benchmark_models([[0.9, 0.7]], ["m1"], time, event)

    with pytest.raises(ValueError, match=r"model_predictions\[0\] contains non-finite"):
        survival.validation.benchmark_models(
            [[0.9, float("inf"), 0.5, 0.3, 0.1, 0.8]], ["m1"], time, event
        )

    with pytest.raises(ValueError, match=r"model_names\[0\] must be non-empty"):
        survival.validation.benchmark_models([risk], [" "], time, event)

    with pytest.raises(ValueError, match="inner_folds must be between 2"):
        survival.validation.nested_cross_validation(risk, time, event, param_grid, 3, 5, 42)

    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        nested.confidence_interval(1.0)


def test_net_survival_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    expected_survival = [0.98, 0.96, 0.94, 0.92]
    method = survival.relative.NetSurvivalMethod("pohar_perme")

    net = survival.relative.net_survival(time, status, expected_survival, method, None)
    crude_time, crude_cancer, crude_other = survival.relative.crude_probability_of_death(
        time,
        status,
        expected_survival,
        [1, 2, 1, 2],
        [2.0, 4.0],
    )

    assert net.method == "Pohar-Perme"
    assert len(net.net_survival) == len(time)
    assert all(value >= 0.0 for value in net.net_survival)
    assert crude_time == pytest.approx([2.0, 4.0])
    assert crude_cancer == pytest.approx([0.25, 0.5])
    assert crude_other == pytest.approx([0.0, 0.0])

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative.net_survival([], [], [], method, None)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative.net_survival([1.0], [1, 0], [0.9], method, None)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.relative.net_survival([float("inf")], [1], [0.9], method, None)
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.relative.net_survival([1.0], [2], [0.9], method, None)
    with pytest.raises(ValueError, match="expected_survival"):
        survival.relative.net_survival([1.0], [1], [0.0], method, None)
    with pytest.raises(ValueError, match="weights length mismatch"):
        survival.relative.net_survival([1.0], [1], [0.9], method, [1.0, 1.0])
    with pytest.raises(ValueError, match="at least one positive"):
        survival.relative.net_survival([1.0], [1], [0.9], method, [0.0])

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative.crude_probability_of_death([1.0], [1], [0.9], [], [1.0])
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.relative.crude_probability_of_death([1.0], [2], [0.9], [1], [1.0])
    with pytest.raises(ValueError, match="expected_survival"):
        survival.relative.crude_probability_of_death([1.0], [1], [1.2], [1], [1.0])
    with pytest.raises(ValueError, match="time_points contains non-finite"):
        survival.relative.crude_probability_of_death([1.0], [1], [0.9], [1], [float("nan")])


def test_relative_survival_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    expected_hazard = [0.01, 0.01, 0.02, 0.02]
    age = [60.0, 65.0, 70.0, 75.0]
    x = [0.5, 1.0, 1.5, 2.0]

    relative = survival.relative.relative_survival(time, status, expected_hazard, age, None)
    model = survival.relative.excess_hazard_regression(
        time,
        status,
        x,
        4,
        1,
        expected_hazard,
        10,
        1e-5,
    )

    assert len(relative.relative_survival) == len(time)
    assert len(model.excess_hazard_ratio) == 1
    assert len(model.baseline_excess_hazard) == len(time)

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative.relative_survival([], [], [], [], None)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative.relative_survival([1.0], [1, 0], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.relative.relative_survival([float("nan")], [1], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.relative.relative_survival([1.0], [2], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="expected_hazard contains negative"):
        survival.relative.relative_survival([1.0], [1], [-0.01], [60.0], None)
    with pytest.raises(ValueError, match="age_at_diagnosis contains non-finite"):
        survival.relative.relative_survival([1.0], [1], [0.01], [float("inf")], None)
    with pytest.raises(ValueError, match="follow_up_years must have length"):
        survival.relative.relative_survival([1.0], [1], [0.01], [60.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        survival.relative.excess_hazard_regression([], [], [], 0, 0, [], 10, 1e-5)
    with pytest.raises(ValueError, match="Input arrays must have length n_obs"):
        survival.relative.excess_hazard_regression([1.0], [1, 0], [0.5], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="x length must equal n_obs"):
        survival.relative.excess_hazard_regression([1.0], [1], [], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.relative.excess_hazard_regression(
            [float("inf")], [1], [0.5], 1, 1, [0.01], 10, 1e-5
        )
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.relative.excess_hazard_regression([1.0], [2], [0.5], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="x contains non-finite"):
        survival.relative.excess_hazard_regression(
            [1.0], [1], [float("nan")], 1, 1, [0.01], 10, 1e-5
        )
    with pytest.raises(ValueError, match="expected_hazard contains negative"):
        survival.relative.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [-0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="max_iter must be greater than 0"):
        survival.relative.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [0.01], 0, 1e-5)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        survival.relative.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [0.01], 10, 0.0)


def test_norisk():
    time1 = [0.0, 1.0, 2.0, 3.0, 4.0]
    time2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    sort1 = [0, 1, 2, 3, 4]
    sort2 = [0, 1, 2, 3, 4]
    strata = [1, 0, 0, 0, 0]

    result = survival.surv_analysis.norisk(time1, time2, status, sort1, sort2, strata)
    assert isinstance(result, list)
    assert len(result) == len(time1)


def test_norisk_validates_public_inputs():
    with pytest.raises(ValueError, match="time2 length"):
        survival.surv_analysis.norisk([0.0, 1.0], [1.0], [1, 0], [0, 1], [0, 1], [])

    with pytest.raises(ValueError, match="finite"):
        survival.surv_analysis.norisk([float("nan")], [1.0], [1], [0], [0], [])

    with pytest.raises(ValueError, match="status values"):
        survival.surv_analysis.norisk([0.0], [1.0], [2], [0], [0], [])

    with pytest.raises(ValueError, match="sort1 index out of bounds"):
        survival.surv_analysis.norisk([0.0], [1.0], [1], [-1], [0], [])

    with pytest.raises(ValueError, match="sort1 must be a permutation"):
        survival.surv_analysis.norisk([0.0, 1.0], [1.0, 2.0], [1, 0], [0, 0], [0, 1], [])

    with pytest.raises(ValueError, match="strata values"):
        survival.surv_analysis.norisk([0.0], [1.0], [1], [0], [0], [2])


def test_finegray():
    tstart = [0.0, 0.0, 0.0, 0.0]
    tstop = [1.0, 2.0, 3.0, 4.0]
    ctime = [0.5, 1.5, 2.5, 3.5]
    cprob = [0.1, 0.2, 0.3, 0.4]
    extend = [True, True, False, False]
    keep = [True, True, True, True]

    result = survival.regression.finegray(
        tstart=tstart,
        tstop=tstop,
        ctime=ctime,
        cprob=cprob,
        extend=extend,
        keep=keep,
    )
    assert hasattr(result, "row")
    assert hasattr(result, "start")
    assert hasattr(result, "end")
    assert hasattr(result, "wt")
    assert len(result.row) > 0
    assert isinstance(result, survival.FineGrayOutput)
    assert survival.as_data_frame(result) == {
        "row": result.row,
        "start": pytest.approx(result.start),
        "end": pytest.approx(result.end),
        "wt": pytest.approx(result.wt),
        "add": result.add,
    }


def test_finegray_validates_public_inputs():
    with pytest.raises(ValueError, match="tstop length"):
        survival.regression.finegray([0.0], [], [], [], [True], [])

    with pytest.raises(ValueError, match="exceeds tstop"):
        survival.regression.finegray([2.0], [1.0], [], [], [True], [])

    with pytest.raises(ValueError, match="ctime must be sorted"):
        survival.regression.finegray([0.0], [1.0], [2.0, 1.0], [1.0, 1.0], [True], [True, True])

    with pytest.raises(ValueError, match="cprob must contain values"):
        survival.regression.finegray([0.0], [1.0], [1.0], [-0.1], [True], [True])

    harmless_zero = survival.regression.finegray(
        [0.0, 0.0],
        [1.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.5, 0.0],
        [True, True],
        [True, True, True],
    )
    assert harmless_zero.row == [1, 1, 1, 2]
    assert harmless_zero.wt == pytest.approx([1.0, 0.5, 0.0, 1.0])

    with pytest.raises(ValueError, match="probability is zero"):
        survival.regression.finegray(
            [0.0],
            [2.0],
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [True],
            [True, False, True],
        )


def test_finegray_regression_and_cif_public_api():
    result = survival.regression.finegray_regression(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 2, 0, 1],
        covariates=[[0.0], [1.0], [0.5], [1.5]],
        event_type=1,
        max_iter=5,
        eps=1e-8,
    )
    assert len(result.coefficients) == 1
    assert len(result.hazard_ratio()) == 1
    assert "Fine-Gray" in result.summary()

    cif = survival.regression.competing_risks_cif(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 2, 0, 1],
        event_type=1,
    )
    assert cif.event_type == 1
    assert len(cif.times) == len(cif.cif)


def test_finegray_regression_and_cif_validate_public_inputs():
    with pytest.raises(ValueError, match="time must not be empty"):
        survival.regression.finegray_regression([], [], [], 1)

    with pytest.raises(ValueError, match="covariates contains non-finite"):
        survival.regression.finegray_regression([1.0], [1], [[float("nan")]], 1)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        survival.regression.finegray_regression([1.0], [1], [[0.0]], 1, max_iter=0)

    with pytest.raises(ValueError, match="eps must be"):
        survival.regression.finegray_regression([1.0], [1], [[0.0]], 1, eps=float("inf"))

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.competing_risks_cif([float("inf")], [1], 1)

    with pytest.raises(ValueError, match="status contains negative"):
        survival.regression.competing_risks_cif([1.0], [-1], 1)

    with pytest.raises(ValueError, match="confidence_level"):
        survival.regression.competing_risks_cif([1.0], [1], 1, confidence_level=1.0)


# ---------------------------------------------------------------------------------------------
# Rate tables, survexp, pyears (R survival 3.8.11 references)
# ---------------------------------------------------------------------------------------------


def _lung_like_positions(us):
    """R: match.ratetable of ages 50/60 (in days), sex 1/2 and two dates against survexp.us."""
    return population.match_ratetable(
        us, ["age", "sex", "year"], [[18262.5, 21915.0], [1.0, 2.0], [10957.0, 12949.0]]
    )


def test_reference_ratetables_match_r():
    us = population.survexp_us()

    assert isinstance(us, population.RateTable)
    assert us.dims == [110, 2, 81]
    assert us.dimid == ["age", "sex", "year"]
    assert us.types == [
        population.DimType.Continuous,
        population.DimType.Factor,
        population.DimType.UsYear,
    ]
    assert us.type_codes() == [2, 1, 4]
    assert us.dimnames[1] == ["male", "female"]
    assert us.dimnames[2][:3] == ["1940", "1941", "1942"]
    assert us.cutpoints[0][:3] == pytest.approx([0.0, 365.25, 730.5])
    assert us.cutpoints[1] is None
    # date cutpoints are days since 1970-01-01: 1940-01-01 = -10958
    assert us.cutpoints[2][:2] == pytest.approx([-10958.0, -10592.0])
    assert len(us.rates) == 110 * 2 * 81
    # R: the survexp.us rates for a 50-year-old male in 2000 and a 60-year-old female in 2005
    assert us.rate([50, 0, 60]) == pytest.approx(1.5264926305778617e-05)
    assert us.rate([60, 1, 65]) == pytest.approx(1.9894143715387382e-05)
    assert us.rate([200, 0, 0]) is None
    assert "Rate table with 3 dimensions" in str(us)

    mn = population.survexp_mn()
    usr = population.survexp_usr()
    assert mn.dims == [110, 2, 51]
    assert usr.dims == [110, 2, 2, 81]
    assert usr.dimid == ["age", "sex", "race", "year"]
    assert mn.rate([50, 0, 30]) == pytest.approx(1.1783374031503476e-05)
    assert usr.rate([50, 0, 0, 60]) == pytest.approx(1.3916210970949556e-05)

    check = population.is_ratetable(
        us.dims, us.dimid, us.dimnames, us.cutpoints, us.type_codes(), len(us.rates)
    )
    assert isinstance(check, population.RatetableCheck)
    assert check.valid is True
    assert check.messages == []
    bad = population.is_ratetable(us.dims, us.dimid, us.dimnames, us.cutpoints, us.type_codes(), 5)
    assert bad.valid is False
    assert bad.messages == ["length of the data does not match prod(dim)"]


def test_ratetable_construction_and_lookup():
    table = population.RateTable(
        [2, 2],
        ["age", "sex"],
        [["0", "50"], ["male", "female"]],
        [[0.0, 50 * 365.25], None],
        [2, 1],
        [0.001, 0.002, 0.0005, 0.0015],
    )

    # rates are column-major: rate(age index, sex index)
    assert table.rate([0, 0]) == pytest.approx(0.001)
    assert table.rate([1, 0]) == pytest.approx(0.002)
    assert table.rate([0, 1]) == pytest.approx(0.0005)
    assert table.rate([1, 1]) == pytest.approx(0.0015)
    assert table.rate([2, 0]) is None
    assert "sex has levels of: male female" in str(table)

    # Ederer expected survival over one year: mean of exp(-rate * 365.25) over the two subjects
    expected = population.survexp(table, [[10 * 365.25, 1.0], [60 * 365.25, 2.0]], times=[365.25])
    assert expected.surv[0][0] == pytest.approx(
        (math.exp(-0.001 * 365.25) + math.exp(-0.0015 * 365.25)) / 2
    )

    with pytest.raises(ValueError, match="not a valid ratetable"):
        population.RateTable([2], ["age"], [["0"]], [[0.0, 1.0]], [2], [0.1, 0.2])


def test_ratetable_date_roundtrip_matches_r():
    # ratetableDate(as.Date("2000-02-29")) = 11016 days since 1970-01-01
    assert population.ratetable_date(2000, 2, 29) == pytest.approx(11016.0)
    assert population.ratetable_date(1960) == pytest.approx(-3653.0)

    date = population.days_to_date(11016.0)
    assert isinstance(date, population.CalendarDate)
    assert (date.year, date.month, date.day) == (2000, 2, 29)
    origin = population.days_to_date(-3653.0)
    assert (origin.year, origin.month, origin.day) == (1960, 1, 1)

    with pytest.raises(ValueError, match="day is invalid"):
        population.ratetable_date(2001, 2, 29)


def test_match_ratetable_and_survexp_match_r():
    us = population.survexp_us()
    matched = _lung_like_positions(us)

    assert isinstance(matched, population.MatchRatetableResult)
    # one-based factor subscripts, continuous and date axes as given
    assert matched.r[0] == pytest.approx([18262.5, 1.0, 10957.0])
    assert matched.r[1] == pytest.approx([21915.0, 2.0, 12949.0])
    assert matched.cutpoints[0][:3] == pytest.approx([0.0, 365.25, 730.5])

    times = [365.25, 1826.25]
    ederer = population.survexp(us, matched.r, times=times)
    # R: survexp with rmap on survexp.us at 1 and 5 years (Ederer)
    assert isinstance(ederer, population.SurvExpResult)
    assert ederer.time == pytest.approx(times)
    assert [row[0] for row in ederer.surv] == pytest.approx(
        [0.99356310877379650, 0.96348172772454077]
    )
    assert [row[0] for row in ederer.n_risk] == pytest.approx([2.0, 2.0])
    assert ederer.method == "Ederer"

    follow_up = [365.25, 1826.25]
    hakulinen = population.survexp(us, matched.r, y=follow_up, times=times, method="hakulinen")
    assert [row[0] for row in hakulinen.surv] == pytest.approx(
        [0.99356310877379650, 0.95988097015776463]
    )
    conditional = population.survexp(us, matched.r, y=follow_up, times=times, method="conditional")
    assert [row[0] for row in conditional.surv] == pytest.approx(
        [0.99356278426562006, 0.95988065665052946]
    )
    assert [row[0] for row in conditional.n_risk] == pytest.approx([2.0, 1.0])

    individual_s = population.survexp(us, matched.r, y=follow_up, method="individual.s")
    assert [row[0] for row in individual_s.surv] == pytest.approx(
        [0.99436612720451900, 0.95910517433409759]
    )
    individual_h = population.survexp(us, matched.r, y=follow_up, method="individual.h")
    assert [row[0] for row in individual_h.surv] == pytest.approx(
        [0.0056498029171802994, 0.0417545392736305143]
    )

    with pytest.raises(ValueError, match="method must be"):
        population.survexp(us, matched.r, times=times, method="bad")
    with pytest.raises(ValueError, match="The variable sex is out of range"):
        population.survexp(us, [[109 * 365.25, 5.0, 18262.0]], times=[0.0, 100.0])


def test_pyears_and_summary_match_r():
    us = population.survexp_us()
    matched = _lung_like_positions(us)

    result = population.pyears(
        [365.25, 1826.25],
        event=[1.0, 0.0],
        factors=[1],
        dims=[2],
        cuts=[[]],
        categories_data=[[1.0], [2.0]],
        ratetable=us,
        ratetable_positions=matched.r,
        scale=365.25,
    )
    # R: pyears by sex with the survexp.us expected events, scale = 365.25
    assert isinstance(result, population.PyearsResult)
    assert result.pyears == pytest.approx([1.0, 5.0])
    assert result.n == pytest.approx([1.0, 1.0])
    assert result.event == pytest.approx([1.0, 0.0])
    assert result.expected == pytest.approx([0.0056498029171802976, 0.0417545392736304866])
    assert result.offtable == pytest.approx(0.0)
    assert result.dims == [2]
    assert result.observations == 2

    by_age = population.pyears(
        [365.25, 1826.25],
        event=[1.0, 0.0],
        factors=[0, 1],
        dims=[2, 2],
        cuts=[[0.0, 55 * 365.25, 100 * 365.25], []],
        categories_data=[[18262.5, 1.0], [21915.0, 2.0]],
        scale=365.25,
    )
    # R: pyears by tcut(age) and sex, no rate table; cells are column-major
    assert by_age.pyears == pytest.approx([1.0, 0.0, 0.0, 5.0])
    assert by_age.n == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert by_age.event == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert by_age.expected is None
    assert by_age.dims == [2, 2]

    summary = population.summary_pyears(result, rate=True, ci_r=True, rr=True, ci_rr=True)
    # R: summary of the pyears object with rate, ci.r, rr and ci.rr
    assert isinstance(summary, population.PyearsSummary)
    assert summary.rate == pytest.approx([1.0, 0.0])
    assert summary.ci_r_lower == pytest.approx([0.025317807984289897, 0.0])
    assert summary.ci_r_upper == pytest.approx([5.57164339093889893, 0.73777589082278705])
    assert summary.rr == pytest.approx([176.997324448103, 0.0])
    assert summary.ci_rr_lower == pytest.approx([4.4811842741101318, 0.0])
    assert summary.ci_rr_upper == pytest.approx([986.165972975141131, 88.346788595595811])
    assert summary.total_events == pytest.approx(1.0)
    assert summary.total_pyears == pytest.approx(6.0)

    with pytest.raises(ValueError, match="categories_data must be"):
        population.pyears([1.0, 2.0], factors=[1], dims=[2], cuts=[[]], categories_data=[[1.0]])
    with pytest.raises(ValueError, match="must be given together"):
        population.pyears([1.0, 2.0], categories_data=[[], []], ratetable=us)
