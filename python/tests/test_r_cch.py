import pytest

from .helpers import setup_survival_import
from .r_api_support import _cch_parity_data

survival = setup_survival_import()


@pytest.mark.parametrize(
    ("method", "expected_coefficients", "expected_variance"),
    [
        (
            "Prentice",
            [-0.750094296490168, 0.8328505349093008],
            [[0.5226059635047275, -0.2022114811960434], [-0.2022114811960434, 1.2764980832765467]],
        ),
        (
            "SelfPrentice",
            [-0.7634916900390691, 1.399231426827849],
            [[0.5226059633986218, -0.20221148126102825], [-0.20221148126102825, 1.276498083557965]],
        ),
        (
            "LinYing",
            [-1.3511250601042777, 0.008608309135789173],
            [[0.3500994140596079, 0.06715207958696841], [0.06715207958696841, 0.6314590317131163]],
        ),
    ],
)
def test_cch_formula_matches_r_right_censored_results(
    method: str,
    expected_coefficients: list[float],
    expected_variance: list[list[float]],
):
    fit = survival.cch(
        "Surv(stop, status) ~ x + z",
        _cch_parity_data(),
        subcoh="subcohort",
        id="id",
        cohort_size=80,
        method=method,
        robust=method == "LinYing",
    )

    assert survival.coef(fit) == pytest.approx(expected_coefficients, abs=1e-11)
    for actual, expected in zip(survival.vcov(fit), expected_variance, strict=True):
        assert actual == pytest.approx(expected, abs=1e-11)
    assert survival.coef_names(fit) == ["x", "z"]
    assert fit.method == method
    assert fit.cohort_size == 80
    assert fit.subcohort_size == 14
    assert fit.stratified is False


def test_cch_formula_matches_r_counting_process_results():
    fit = survival.cch(
        "Surv(start, stop, status) ~ x + z",
        _cch_parity_data(),
        subcoh="subcohort",
        id="id",
        cohort_size=80,
        method="LinYing",
        robust=True,
    )

    assert survival.coef(fit) == pytest.approx(
        [-1.1662987644578553, -0.042048877306928675],
        abs=1e-11,
    )
    expected = [
        [0.1917752992327156, -0.16536154052082166],
        [-0.16536154052082166, 0.6718407297652415],
    ]
    for actual, expected_row in zip(fit.var, expected, strict=True):
        assert actual == pytest.approx(expected_row, abs=1e-11)
    assert len(fit.phase2var) == 2
    assert len(fit.martingale_residuals()) == len(fit.status)


@pytest.mark.parametrize(
    ("method", "expected_coefficients", "expected_variance"),
    [
        (
            "I.Borgan",
            [-0.763491690039068, 1.39923142682785],
            [[0.53280614362319, -0.207366276403962], [-0.207366276403962, 1.33942679401654]],
        ),
        (
            "II.Borgan",
            [-1.35112506010428, 0.00860830913578929],
            [[0.282233396842156, 0.00153183282816197], [0.00153183282816197, 0.542554720451637]],
        ),
    ],
)
def test_cch_formula_matches_r_stratified_borgan_results(
    method: str,
    expected_coefficients: list[float],
    expected_variance: list[list[float]],
):
    fit = survival.cch(
        "Surv(stop, status) ~ x + z",
        _cch_parity_data(),
        subcoh="subcohort",
        id="id",
        stratum="group",
        cohort_size={"a": 40, "b": 40},
        method=method,
    )

    assert survival.coef(fit) == pytest.approx(expected_coefficients, abs=1e-11)
    for actual, expected in zip(fit.var, expected_variance, strict=True):
        assert actual == pytest.approx(expected, abs=1e-11)
    assert fit.stratified is True
    assert fit.stratum == _cch_parity_data()["group"]
    assert fit.cohort_sizes == [40, 40]
    assert fit.subcohort_sizes == [7, 7]
    assert len(fit.optimization_fraction) == 2
    assert len(fit.phase2_score_matrix) == 2
    assert len(fit.collapsed_score_rows) == 20


def test_cch_formula_expands_factors_and_validates_sampling_inputs():
    data = _cch_parity_data()
    fit = survival.cch(
        "Surv(stop, status) ~ x * group",
        data,
        subcoh="subcohort",
        id="id",
        cohort_size=80,
        method="Prentice",
    )
    assert survival.coef_names(fit) == ["x", "groupb", "x:groupb"]

    with pytest.raises(ValueError, match="requires stratum"):
        survival.cch(
            "Surv(stop, status) ~ x",
            data,
            subcoh="subcohort",
            id="id",
            cohort_size=[40, 40],
            method="I.Borgan",
        )
    with pytest.raises(ValueError, match="same length"):
        survival.cch(
            "Surv(stop, status) ~ x",
            data,
            subcoh="subcohort",
            id="id",
            stratum="group",
            cohort_size=[80],
            method="II.Borgan",
        )
    with pytest.raises(ValueError, match="multiple records per id"):
        survival.cch(
            "Surv(stop, status) ~ x",
            data,
            subcoh="subcohort",
            id=[1] * 20,
            cohort_size=80,
        )
    invalid_subcohort = list(data["subcohort"])
    invalid_subcohort[1] = 0
    with pytest.raises(ValueError, match="censored observations"):
        survival.cch(
            "Surv(stop, status) ~ x",
            data,
            subcoh=invalid_subcohort,
            id="id",
            cohort_size=80,
        )
