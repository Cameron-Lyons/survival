import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_cipoisson_exact():
    result = survival.cipoisson_exact(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] < result[1]


def test_cipoisson_anscombe():
    result = survival.cipoisson_anscombe(k=5, time=10.0, p=0.95)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_cipoisson():
    result_exact = survival.cipoisson(k=5, time=10.0, p=0.95, method="exact")
    result_anscombe = survival.cipoisson(k=5, time=10.0, p=0.95, method="anscombe")
    assert isinstance(result_exact, tuple)
    assert isinstance(result_anscombe, tuple)


def test_model_selection_public_apis_and_validation():
    criteria = survival.compute_model_selection_criteria(-100.0, 5, 200, 50, None)
    comparison = survival.compare_models(["m1", "m2"], [-100.0, -95.0], [3, 5], 200)
    cv_score = survival.compute_cv_score([0.75, 0.8, 0.7], "c_index")

    assert criteria.aic > 0.0
    assert "Model Selection Criteria" in criteria.summary()
    assert comparison.best_model_aic == "m2"
    assert len(comparison.likelihood_ratio_tests) == 1
    assert cv_score.n_folds == 3
    assert cv_score.confidence_interval(0.05)[0] < cv_score.mean_score

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

    assert summary.total_person_years == pytest.approx(5.0)
    assert summary.total_events == pytest.approx(3.0)
    assert summary.smr == pytest.approx(1.5)
    assert "SMR" in summary.to_table()
    assert len(cells) == 2
    assert cells[0].rate == pytest.approx(0.5)
    assert cells[0].smr == pytest.approx(2.0)
    assert smr == pytest.approx(2.0)
    assert lower < smr < upper

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
