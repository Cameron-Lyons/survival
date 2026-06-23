import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_cipoisson_exact():
    result = survival.cipoisson_exact(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result == pytest.approx((0.1623486, 1.1668332), abs=1e-6)
    assert survival.cipoisson_exact(20, 4.0, 0.90) == pytest.approx(
        (3.313663, 7.265505), abs=1e-6
    )
    assert survival.cipoisson_exact(0, 10.0, 0.95) == pytest.approx(
        (0.0, 0.3688879), abs=1e-6
    )


def test_cipoisson_anscombe():
    result = survival.cipoisson_anscombe(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result == pytest.approx((0.1507881, 1.1586004), abs=1e-6)
    assert survival.cipoisson_anscombe(20, 4.0, 0.90) == pytest.approx(
        (3.304600, 7.266646), abs=1e-6
    )


def test_cipoisson():
    result_exact = survival.cipoisson(k=5, time=10.0, p=0.95, method="exact")
    result_anscombe = survival.cipoisson(k=5, time=10.0, p=0.95, method="anscombe")
    assert isinstance(result_exact, tuple)
    assert isinstance(result_anscombe, tuple)
    assert result_exact == pytest.approx(survival.cipoisson_exact(5, 10.0, 0.95))
    assert result_anscombe == pytest.approx(survival.cipoisson_anscombe(5, 10.0, 0.95))
    assert survival.cipoisson(k=5, method="e") == pytest.approx(survival.cipoisson_exact(5))
    assert survival.cipoisson(k=5, method="a") == pytest.approx(survival.cipoisson_anscombe(5))

    with pytest.raises(ValueError, match="time must be positive and finite"):
        survival.cipoisson_exact(k=1, time=0.0, p=0.95)
    with pytest.raises(ValueError, match="p must be a confidence level between 0 and 1"):
        survival.cipoisson_anscombe(k=1, time=1.0, p=1.0)
    with pytest.raises(ValueError, match="method must uniquely match"):
        survival.cipoisson(k=1, time=1.0, p=0.95, method="")


def test_model_selection_public_apis_and_validation():
    criteria = survival.compute_model_selection_criteria(-100.0, 5, 200, 50, None)
    comparison = survival.compare_models(["m1", "m2"], [-100.0, -95.0], [3, 5], 200)
    cv_score = survival.compute_cv_score([0.75, 0.8, 0.7], "c_index")

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
        survival.compute_model_selection_criteria(float("nan"), 5, 200, 50, None)

    with pytest.raises(ValueError, match="n_obs must be greater than 1"):
        survival.compute_model_selection_criteria(-100.0, 5, 1, 1, None)

    with pytest.raises(ValueError, match="n_events cannot exceed n_obs"):
        survival.compute_model_selection_criteria(-100.0, 5, 20, 21, None)

    with pytest.raises(ValueError, match="log_likelihoods contains non-finite"):
        survival.compare_models(["m1"], [float("inf")], [1], 10)

    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        survival.compare_models(["m1"], [-10.0], [1], 0)

    with pytest.raises(ValueError, match="fold_scores contains non-finite"):
        survival.compute_cv_score([0.7, float("nan")], "c_index")

    with pytest.raises(ValueError, match="metric cannot be empty"):
        survival.compute_cv_score([0.7], " ")

    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        cv_score.confidence_interval(0.0)


def test_hyperparameter_public_apis_and_validation():
    risk = [0.9, 0.7, 0.5, 0.3, 0.1, 0.8]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    event = [1, 1, 1, 0, 1, 0]
    param_grid = [("alpha", [0.5, 1.0])]

    search = survival.hyperparameter_search(risk, time, event, param_grid)
    benchmark = survival.benchmark_models([risk, list(reversed(risk))], ["m1", "m2"], time, event)
    nested = survival.nested_cross_validation(risk, time, event, param_grid, 3, 2, 42)

    assert len(search.all_scores) > 0
    assert benchmark.best_model in {"m1", "m2"}
    assert len(nested.outer_scores) == 3
    ci_95 = nested.confidence_interval(0.05)
    ci_90 = nested.confidence_interval(0.10)
    assert ci_95[0] <= ci_90[0]
    assert ci_95[1] >= ci_90[1]

    exact_benchmark = survival.benchmark_models(
        [[0.1, 0.9, 0.2], [0.2, 0.8, 0.3]],
        ["m1", "m2"],
        [1.0, 2.0, 3.0],
        [0, 1, 0],
        2.0,
    )
    near_benchmark = survival.benchmark_models(
        [[0.1, 0.9, 0.2], [0.2, 0.8, 0.3]],
        ["m1", "m2"],
        [1.0, 2.0 + 5e-10, 3.0],
        [0, 1, 0],
        2.0,
    )
    assert near_benchmark.brier_scores == pytest.approx(exact_benchmark.brier_scores)

    bad_config = survival.HyperparameterSearchConfig(cv_folds=0)
    with pytest.raises(ValueError, match="cv_folds must be between 2"):
        survival.hyperparameter_search(risk, time, event, param_grid, bad_config)

    bad_risk = list(risk)
    bad_risk[1] = float("nan")
    with pytest.raises(ValueError, match="risk_scores contains non-finite"):
        survival.hyperparameter_search(bad_risk, time, event, param_grid)

    with pytest.raises(ValueError, match="param_grid values for alpha must not be empty"):
        survival.hyperparameter_search(risk, time, event, [("alpha", [])])

    with pytest.raises(ValueError, match=r"model_predictions\[0\] length must match"):
        survival.benchmark_models([[0.9, 0.7]], ["m1"], time, event)

    with pytest.raises(ValueError, match=r"model_predictions\[0\] contains non-finite"):
        survival.benchmark_models([[0.9, float("inf"), 0.5, 0.3, 0.1, 0.8]], ["m1"], time, event)

    with pytest.raises(ValueError, match=r"model_names\[0\] must be non-empty"):
        survival.benchmark_models([risk], [" "], time, event)

    with pytest.raises(ValueError, match="inner_folds must be between 2"):
        survival.nested_cross_validation(risk, time, event, param_grid, 3, 5, 42)

    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        nested.confidence_interval(1.0)


def test_net_survival_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    expected_survival = [0.98, 0.96, 0.94, 0.92]
    method = survival.NetSurvivalMethod("pohar_perme")

    net = survival.net_survival(time, status, expected_survival, method, None)
    crude_time, crude_cancer, crude_other = survival.crude_probability_of_death(
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
        survival.net_survival([], [], [], method, None)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.net_survival([1.0], [1, 0], [0.9], method, None)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.net_survival([float("inf")], [1], [0.9], method, None)
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.net_survival([1.0], [2], [0.9], method, None)
    with pytest.raises(ValueError, match="expected_survival"):
        survival.net_survival([1.0], [1], [0.0], method, None)
    with pytest.raises(ValueError, match="weights length mismatch"):
        survival.net_survival([1.0], [1], [0.9], method, [1.0, 1.0])
    with pytest.raises(ValueError, match="at least one positive"):
        survival.net_survival([1.0], [1], [0.9], method, [0.0])

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.crude_probability_of_death([1.0], [1], [0.9], [], [1.0])
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.crude_probability_of_death([1.0], [2], [0.9], [1], [1.0])
    with pytest.raises(ValueError, match="expected_survival"):
        survival.crude_probability_of_death([1.0], [1], [1.2], [1], [1.0])
    with pytest.raises(ValueError, match="time_points contains non-finite"):
        survival.crude_probability_of_death([1.0], [1], [0.9], [1], [float("nan")])


def test_relative_survival_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    expected_hazard = [0.01, 0.01, 0.02, 0.02]
    age = [60.0, 65.0, 70.0, 75.0]
    x = [0.5, 1.0, 1.5, 2.0]

    relative = survival.relative_survival(time, status, expected_hazard, age, None)
    model = survival.excess_hazard_regression(
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
        survival.relative_survival([], [], [], [], None)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.relative_survival([1.0], [1, 0], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.relative_survival([float("nan")], [1], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.relative_survival([1.0], [2], [0.01], [60.0], None)
    with pytest.raises(ValueError, match="expected_hazard contains negative"):
        survival.relative_survival([1.0], [1], [-0.01], [60.0], None)
    with pytest.raises(ValueError, match="age_at_diagnosis contains non-finite"):
        survival.relative_survival([1.0], [1], [0.01], [float("inf")], None)
    with pytest.raises(ValueError, match="follow_up_years must have length"):
        survival.relative_survival([1.0], [1], [0.01], [60.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        survival.excess_hazard_regression([], [], [], 0, 0, [], 10, 1e-5)
    with pytest.raises(ValueError, match="Input arrays must have length n_obs"):
        survival.excess_hazard_regression([1.0], [1, 0], [0.5], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="x length must equal n_obs"):
        survival.excess_hazard_regression([1.0], [1], [], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.excess_hazard_regression(
            [float("inf")], [1], [0.5], 1, 1, [0.01], 10, 1e-5
        )
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.excess_hazard_regression([1.0], [2], [0.5], 1, 1, [0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="x contains non-finite"):
        survival.excess_hazard_regression(
            [1.0], [1], [float("nan")], 1, 1, [0.01], 10, 1e-5
        )
    with pytest.raises(ValueError, match="expected_hazard contains negative"):
        survival.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [-0.01], 10, 1e-5)
    with pytest.raises(ValueError, match="max_iter must be greater than 0"):
        survival.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [0.01], 0, 1e-5)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        survival.excess_hazard_regression([1.0], [1], [0.5], 1, 1, [0.01], 10, 0.0)


def test_norisk():
    time1 = [0.0, 1.0, 2.0, 3.0, 4.0]
    time2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    sort1 = [0, 1, 2, 3, 4]
    sort2 = [0, 1, 2, 3, 4]
    strata = [1, 0, 0, 0, 0]

    result = survival.norisk(time1, time2, status, sort1, sort2, strata)
    assert isinstance(result, list)
    assert len(result) == len(time1)


def test_norisk_validates_public_inputs():
    with pytest.raises(ValueError, match="time2 length"):
        survival.norisk([0.0, 1.0], [1.0], [1, 0], [0, 1], [0, 1], [])

    with pytest.raises(ValueError, match="finite"):
        survival.norisk([float("nan")], [1.0], [1], [0], [0], [])

    with pytest.raises(ValueError, match="status values"):
        survival.norisk([0.0], [1.0], [2], [0], [0], [])

    with pytest.raises(ValueError, match="sort1 index out of bounds"):
        survival.norisk([0.0], [1.0], [1], [-1], [0], [])

    with pytest.raises(ValueError, match="sort1 must be a permutation"):
        survival.norisk([0.0, 1.0], [1.0, 2.0], [1, 0], [0, 0], [0, 1], [])

    with pytest.raises(ValueError, match="strata values"):
        survival.norisk([0.0], [1.0], [1], [0], [0], [2])


def test_finegray():
    tstart = [0.0, 0.0, 0.0, 0.0]
    tstop = [1.0, 2.0, 3.0, 4.0]
    ctime = [0.5, 1.5, 2.5, 3.5]
    cprob = [0.1, 0.2, 0.3, 0.4]
    extend = [True, True, False, False]
    keep = [True, True, True, True]

    result = survival.finegray(
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


def test_finegray_validates_public_inputs():
    with pytest.raises(ValueError, match="tstop length"):
        survival.finegray([0.0], [], [], [], [True], [])

    with pytest.raises(ValueError, match="exceeds tstop"):
        survival.finegray([2.0], [1.0], [], [], [True], [])

    with pytest.raises(ValueError, match="ctime must be sorted"):
        survival.finegray([0.0], [1.0], [2.0, 1.0], [1.0, 1.0], [True], [True, True])

    with pytest.raises(ValueError, match="cprob must contain values"):
        survival.finegray([0.0], [1.0], [1.0], [0.0], [True], [True])


def test_finegray_regression_and_cif_public_api():
    result = survival.finegray_regression(
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

    cif = survival.competing_risks_cif(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 2, 0, 1],
        event_type=1,
    )
    assert cif.event_type == 1
    assert len(cif.times) == len(cif.cif)


def test_finegray_regression_and_cif_validate_public_inputs():
    with pytest.raises(ValueError, match="time must not be empty"):
        survival.finegray_regression([], [], [], 1)

    with pytest.raises(ValueError, match="covariates contains non-finite"):
        survival.finegray_regression([1.0], [1], [[float("nan")]], 1)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        survival.finegray_regression([1.0], [1], [[0.0]], 1, max_iter=0)

    with pytest.raises(ValueError, match="eps must be"):
        survival.finegray_regression([1.0], [1], [[0.0]], 1, eps=float("inf"))

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.competing_risks_cif([float("inf")], [1], 1)

    with pytest.raises(ValueError, match="status contains negative"):
        survival.competing_risks_cif([1.0], [-1], 1)

    with pytest.raises(ValueError, match="confidence_level"):
        survival.competing_risks_cif([1.0], [1], 1, confidence_level=1.0)


def test_ratetable_and_survexp_public_apis():
    ratetable = survival.create_simple_ratetable(
        age_breaks=[0.0, 36500.0, 73000.0],
        year_breaks=[1990.0, 2020.0],
        rates_male=[0.00001, 0.00005],
        rates_female=[0.000008, 0.00004],
    )

    assert ratetable.ndim() == 3
    assert ratetable.dim_names() == ["age", "year", "sex"]
    assert survival.is_ratetable(ratetable.ndim(), True, True) is True
    assert ratetable.lookup({"age": 1000.0, "year": 2000.0, "sex": 0.0}) == pytest.approx(1e-05)

    expected = ratetable.expected_survival(1000.0, 1010.0, 2000.0, 0)
    assert 0.0 < expected <= 1.0

    result = survival.survexp(
        time=[365.0, 730.0],
        age=[18250.0, 21900.0],
        year=[2000.0, 2000.0],
        ratetable=ratetable,
        sex=[0, 1],
        method="conditional",
    )

    assert result.method == "conditional"
    assert result.n == 2
    assert result.time == [365.0, 730.0]
    assert len(result.surv) == 2
    assert result.surv[0] >= result.surv[1]
    assert result.n_risk == [2.0, 1.0]

    individual = survival.survexp_individual(
        time=[365.0, 730.0],
        age=[18250.0, 21900.0],
        year=[2000.0, 2000.0],
        ratetable=ratetable,
        sex=[0, 1],
    )
    assert len(individual) == 2
    assert all(0.0 < value <= 1.0 for value in individual)


def test_survexp_validates_method():
    ratetable = survival.create_simple_ratetable(
        age_breaks=[0.0, 36500.0, 73000.0],
        year_breaks=[1990.0, 2020.0],
        rates_male=[0.00001, 0.00005],
        rates_female=[0.000008, 0.00004],
    )

    with pytest.raises(
        ValueError,
        match="method must be 'hakulinen', 'conditional', or 'individual'",
    ):
        survival.survexp(
            time=[365.0],
            age=[18250.0],
            year=[2000.0],
            ratetable=ratetable,
            method="bad",
        )

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.survexp(
            time=[float("nan")],
            age=[18250.0],
            year=[2000.0],
            ratetable=ratetable,
        )

    with pytest.raises(ValueError, match="age contains negative"):
        survival.survexp(
            time=[365.0],
            age=[-1.0],
            year=[2000.0],
            ratetable=ratetable,
        )

    with pytest.raises(ValueError, match="sex values must be non-negative"):
        survival.survexp(
            time=[365.0],
            age=[18250.0],
            year=[2000.0],
            ratetable=ratetable,
            sex=[-1],
        )

    with pytest.raises(ValueError, match="times must be sorted"):
        survival.survexp(
            time=[365.0, 730.0],
            age=[18250.0, 21900.0],
            year=[2000.0, 2000.0],
            ratetable=ratetable,
            times=[730.0, 365.0],
            method="conditional",
        )

    with pytest.raises(ValueError, match="sex must have same length"):
        survival.survexp_individual(
            time=[365.0],
            age=[18250.0],
            year=[2000.0],
            ratetable=ratetable,
            sex=[0, 1],
        )

    with pytest.raises(ValueError, match="age_breaks"):
        survival.create_simple_ratetable(
            age_breaks=[0.0],
            year_breaks=[1990.0, 2020.0],
            rates_male=[0.00001],
            rates_female=[0.000008],
        )

    with pytest.raises(ValueError, match="rates_male contains non-finite"):
        survival.create_simple_ratetable(
            age_breaks=[0.0, 36500.0],
            year_breaks=[1990.0, 2020.0],
            rates_male=[float("nan")],
            rates_female=[0.000008],
        )

    with pytest.raises(ValueError, match="age coordinate must be finite"):
        ratetable.lookup({"age": float("nan"), "year": 2000.0, "sex": 0.0})

    with pytest.raises(ValueError, match="age_start, age_end, and year_start must be finite"):
        ratetable.expected_survival(0.0, float("inf"), 2000.0, 0)


def test_ratetable_date_roundtrip():
    result = survival.ratetable_date(2000, 2, 29, 1960)

    assert result.days == pytest.approx(14669.0)
    assert result.origin_year == 1960
    assert survival.days_to_date(result.days, result.origin_year) == (2000, 2, 29)

    with pytest.raises(ValueError, match="days must be a finite non-negative value"):
        survival.days_to_date(float("nan"), 1960)

    with pytest.raises(ValueError, match="day is invalid"):
        survival.ratetable_date(2001, 2, 29, 1960)


def test_pyears_summary_and_cells():
    summary = survival.summary_pyears(
        pyears=[2.0, 3.0],
        pn=[1.0, 1.0],
        pcount=[1.0, 2.0],
        pexpect=[0.5, 1.5],
        offtable=0.25,
    )
    cells = survival.pyears_by_cell(
        pyears=[2.0, 3.0],
        pn=[1.0, 1.0],
        pcount=[1.0, 2.0],
        pexpect=[0.5, 1.5],
    )
    smr, lower, upper = survival.pyears_ci(5.0, 2.5, 0.95)
    zero_smr, zero_lower, zero_upper = survival.pyears_ci(0.0, 10.0, 0.95)

    assert summary.total_person_years == pytest.approx(5.0)
    assert summary.total_events == pytest.approx(3.0)
    assert summary.smr == pytest.approx(1.5)
    assert "SMR" in summary.to_table()
    assert len(cells) == 2
    assert cells[0].rate == pytest.approx(0.5)
    assert cells[0].smr == pytest.approx(2.0)
    assert smr == pytest.approx(2.0)
    assert lower == pytest.approx(0.6493946, abs=1e-6)
    assert upper == pytest.approx(4.667333, abs=1e-6)
    assert zero_smr == pytest.approx(0.0)
    assert zero_lower == pytest.approx(0.0)
    assert zero_upper == pytest.approx(0.3688879, abs=1e-6)

    with pytest.raises(ValueError, match="must have the same length"):
        survival.summary_pyears([1.0], [], [1.0], [1.0], 0.0)

    with pytest.raises(ValueError, match="pyears contains non-finite"):
        survival.summary_pyears([float("inf")], [1.0], [1.0], [1.0], 0.0)

    with pytest.raises(ValueError, match="pcount contains negative"):
        survival.pyears_by_cell([1.0], [1.0], [-1.0], [1.0])

    with pytest.raises(ValueError, match="expected must be a finite positive value"):
        survival.pyears_ci(1.0, 0.0, 0.95)

    with pytest.raises(ValueError, match="conf_level"):
        survival.pyears_ci(1.0, 1.0, 1.0)


def test_reference_ratetables_and_expected_survival_helpers():
    us = survival.survexp_us()
    mn = survival.survexp_mn()
    usr = survival.survexp_usr()
    expected = survival.compute_expected_survival(
        age=[365.25 * 50.0, 365.25 * 60.0],
        sex=[0, 1],
        year=[2000.0, 2005.0],
        times=[365.25, 365.25 * 5.0],
    )

    assert us.ndim() == 3
    assert mn.ndim() == 3
    assert usr.ndim() == 3
    assert us.dim_names() == ["year", "age", "sex"]
    assert mn.dim_names() == ["year", "age", "sex"]
    assert usr.dim_names() == us.dim_names()
    assert us.lookup({"year": 2000.0, "age": 365.25 * 50.0, "sex": 0.0}) > 0.0
    assert usr.lookup({"year": 2000.0, "age": 365.25 * 50.0, "sex": 0.0}) == pytest.approx(
        us.lookup({"year": 2000.0, "age": 365.25 * 50.0, "sex": 0.0})
    )

    assert expected.n == 2
    assert expected.time == pytest.approx([365.25, 365.25 * 5.0])
    assert len(expected.expected_survival) == 2
    assert all(0.0 < value <= 1.0 for value in expected.expected_survival)
    assert expected.expected_survival[0] >= expected.expected_survival[1]

    with pytest.raises(ValueError, match="age contains non-finite"):
        survival.compute_expected_survival([float("nan")], [0], [2000.0], [365.25])

    with pytest.raises(ValueError, match="sex values must be non-negative"):
        survival.compute_expected_survival([365.25 * 50.0], [-1], [2000.0], [365.25])

    with pytest.raises(ValueError, match="times contains non-finite"):
        survival.compute_expected_survival([365.25 * 50.0], [0], [2000.0], [float("inf")])


def test_compute_expected_survival_validates_lengths():
    with pytest.raises(ValueError, match="age, sex, and year must have the same length"):
        survival.compute_expected_survival(
            age=[365.25 * 50.0],
            sex=[0, 1],
            year=[2000.0],
            times=[365.25],
        )
