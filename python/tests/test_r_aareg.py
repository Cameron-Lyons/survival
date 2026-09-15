"""``survival.r.aareg`` Aalen additive regression against R survival 3.8.11 (``ovarian``)."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    if isinstance(values, list) and values and isinstance(values[0], list):
        return [pytest.approx(row, rel=rel) for row in values]
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def ovarian():
    return datasets.load_ovarian()


def test_aareg_matches_r(ovarian):
    fit = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", ovarian)
    assert isinstance(fit, r.AaregModelResult)
    assert fit.n == [26, 12, 12]
    assert fit.times[:3] == [59.0, 115.0, 156.0]
    assert fit.nrisk[:3] == [26.0, 25.0, 24.0]
    assert fit.coefficient_names == ["Intercept", "age", "ecog.ps"]
    assert fit.coefficient[:2] == approx(
        [
            [-0.219777759950435, 0.00691897151414902, -0.0891990189962571],
            [-0.32107206341235, 0.00954758742174273, -0.114188198066149],
        ]
    )
    assert fit.test_statistic == approx(
        [-1.36815341992796, 93.8700271586746, 0.0110835307233675], rel=1e-6
    )
    assert fit.test_var == approx(
        [
            [0.379856008822724, -21.8643732141607, -0.220993073503703],
            [-21.8643732141607, 1502.1507065072, -9.8843345227457],
            [-0.220993073503703, -9.8843345227457, 2.42267809834877],
        ],
        rel=1e-6,
    )
    assert fit.tweight[:2] == approx(
        [
            [0.682864296741566, 2507.80872668345, 6.35355140108748],
            [0.65609841103356, 2183.57619516609, 5.97967745307236],
        ]
    )
    assert fit.test == "aalen"
    assert fit.dfbeta is None
    assert fit.test_var2 is None
    assert fit.formula == "Surv(futime, fustat) ~ age + ecog.ps"
    assert fit.weights is None
    assert r.model_formula(fit) == fit.formula
    assert r.model_weights(fit) is None


def test_summary_aareg_matches_r(ovarian):
    fit = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", ovarian)
    summary = r.model_summary(fit)
    assert summary["columns"] == ["slope", "coef", "se(coef)", "z", "p"]
    assert [row["name"] for row in summary["table"]] == ["Intercept", "age", "ecog.ps"]
    assert [row["slope"] for row in summary["table"]] == approx(
        [-0.00717943180642173, 0.000194242827812841, -0.000520067101518314], rel=1e-6
    )
    assert [row["coef"] for row in summary["table"]] == approx(
        [-0.267333077106693, 0.00603271092747513, 0.000202778734584543], rel=1e-6
    )
    assert [row["se"] for row in summary["table"]] == approx(
        [0.120427978765407, 0.00249081989075946, 0.0284768623309702], rel=1e-6
    )
    assert [row["z"] for row in summary["table"]] == approx(
        [-2.21985853991169, 2.42197797996375, 0.00712082434601687], rel=1e-6
    )
    assert [row["p"] for row in summary["table"]] == approx(
        [0.026428371649148, 0.0154362858752888, 0.994318452209057], rel=1e-6
    )
    assert summary["chisq"] == approx(6.03366160741177, rel=1e-6)
    assert summary["df"] == 2
    assert summary["n"] == [26, 12, 12]
    assert r.model_summary(fit, test="nrisk")["chisq"] == approx(6.05609402409755, rel=1e-6)


def test_aareg_options(ovarian):
    dfbeta = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", ovarian, dfbeta=True)
    assert len(dfbeta.dfbeta) == 26
    assert len(dfbeta.dfbeta[0]) == 3
    assert len(dfbeta.dfbeta[0][0]) == 12
    assert [dfbeta.dfbeta[0][k][0] for k in range(3)] == approx(
        [-0.177694097091052, 0.00559410741233639, -0.0721189402673048], rel=1e-6
    )
    assert dfbeta.test_var2 == approx(
        [
            [0.228924088265589, -12.2418252814399, -0.199697281069864],
            [-12.2418252814399, 857.029770525028, -8.07096074047556],
            [-0.199697281069864, -8.07096074047556, 2.07954308537869],
        ],
        rel=1e-6,
    )
    assert "robust_se" in r.model_summary(dfbeta)["table"][0]
    nrisk = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", ovarian, test="nrisk")
    assert nrisk.test == "nrisk"
    assert nrisk.test_statistic == approx(
        [-58.7878245043473, 1.30281264978888, 0.247576955514257], rel=1e-6
    )
    with_model = r.aareg(
        "Surv(futime, fustat) ~ age + ecog.ps", ovarian, model=True, x=True, y=True
    )
    assert with_model.x[0] == [72.3315, 1.0]
    assert with_model.y.type == "right"
    assert "age" in r.model_frame(with_model)
    with pytest.raises(ValueError, match="Strata terms not allowed"):
        r.aareg("Surv(futime, fustat) ~ age + strata(rx)", ovarian)
    with pytest.raises(ValueError, match="test must be one of"):
        r.aareg("Surv(futime, fustat) ~ age", ovarian, test="wald")
    with pytest.raises(ValueError, match="qrtol must be positive"):
        r.aareg("Surv(futime, fustat) ~ age", ovarian, qrtol=0)
