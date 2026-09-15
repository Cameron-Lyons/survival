"""``survival.r.cch`` case-cohort models against R survival 3.8.11 (the ``nwtco`` example)."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def ccoh_data():
    """``?cch``: the relapses plus the subcohort, with factor stage/histol and age in years."""

    nwtco = datasets.load_nwtco()
    keep = [
        i
        for i, (rel, sub) in enumerate(zip(nwtco["rel"], nwtco["in.subcohort"], strict=True))
        if rel == 1 or sub == 1
    ]
    stage_labels = {1: "I", 2: "II", 3: "III", 4: "IV"}
    histol_labels = {1: "FH", 2: "UH"}
    return {
        "seqno": [nwtco["seqno"][i] for i in keep],
        "edrel": [nwtco["edrel"][i] for i in keep],
        "rel": [int(nwtco["rel"][i]) for i in keep],
        "subcohort": [int(nwtco["in.subcohort"][i]) for i in keep],
        "stage": [stage_labels[int(nwtco["stage"][i])] for i in keep],
        "histol": [histol_labels[int(nwtco["histol"][i])] for i in keep],
        "age": [nwtco["age"][i] / 12 for i in keep],
    }


def test_prentice_matches_r(ccoh_data):
    assert len(ccoh_data["seqno"]) == 1154
    fit = r.cch(
        "Surv(edrel, rel) ~ stage + histol + age",
        ccoh_data,
        subcoh="subcohort",
        id="seqno",
        cohort_size=4028,
    )
    assert isinstance(fit, r.CchModelResult)
    assert fit.coef_names == ("stageII", "stageIII", "stageIV", "histolUH", "age")
    assert fit.coefficients == approx(
        [
            0.734570842045653,
            0.597083557948202,
            1.38413196899809,
            1.49806307262597,
            0.0432678728347779,
        ]
    )
    assert [fit.var[i][i] for i in range(5)] == approx(
        [
            0.0283909684857841,
            0.0300852286987785,
            0.0419511598024265,
            0.0255057362305449,
            0.000563153809234572,
        ],
        rel=1e-6,
    )
    assert fit.naive_var == fit.var
    assert fit.phase2var is not None
    assert (fit.method, fit.stratified) == ("Prentice", False)
    assert fit.subcohort_size == (668,)
    assert fit.cohort_size == (4028,)
    assert fit.stratum is None
    assert len(fit.id) == 1154
    assert sum(fit.subcoh) == 668
    assert r.coef(fit) == fit.coefficients
    assert r.coef_names(fit) == list(fit.coef_names)
    assert r.vcov(fit) == fit.var
    summary = r.model_summary(fit)
    assert summary["coefficient_columns"] == ["Value", "SE", "Z", "p"]
    row = summary["coefficients"][0]
    assert [row[k] for k in ("value", "se", "z", "p")] == approx(
        [0.734570842045653, 0.168496197244282, 4.35956926066816, 1.30318703686072e-05], rel=1e-6
    )
    assert summary["coefficients"][3]["p"] == 0.0  # R: 2*(1-pnorm(Z)) underflows to 0
    assert r.confint(fit)[0]["lower"] == approx(
        0.734570842045653 - 1.959963984540054 * 0.168496197244282, rel=1e-6
    )


def test_other_estimators_and_checks(ccoh_data):
    lin_ying = r.cch(
        "Surv(edrel, rel) ~ stage + histol + age",
        ccoh_data,
        subcoh="subcohort",
        id="seqno",
        cohort_size=4028,
        method="LinYing",
        robust=True,
    )
    assert [lin_ying.var[i][i] for i in range(5)] == approx(
        [
            0.0264808338369606,
            0.0282774478991051,
            0.0356806595706353,
            0.0211428256654773,
            0.000529262641888723,
        ],
        rel=1e-6,
    )
    with pytest.warns(RuntimeWarning, match="robust' ignored"):
        r.cch(
            "Surv(edrel, rel) ~ age",
            ccoh_data,
            subcoh="subcohort",
            id="seqno",
            cohort_size=4028,
            robust=True,
        )
    with pytest.warns(RuntimeWarning, match="stratum levels and names"):
        borgan = r.cch(
            "Surv(edrel, rel) ~ histol + age",
            ccoh_data,
            subcoh="subcohort",
            id="seqno",
            stratum="stage",
            cohort_size={"1": 1572, "2": 1052, "3": 944, "4": 460},
            method="I.Borgan",
        )
    assert borgan.stratified
    assert borgan.method == "I.Borgan"
    assert borgan.coefficients == approx([1.50693446701834, 0.0657777731818127])
    assert [borgan.var[i][i] for i in range(2)] == approx(
        [0.0224743236858645, 0.000501403989603528], rel=1e-6
    )
    assert borgan.subcohort_size == (362, 311, 313, 168)
    assert borgan.cohort_size == (1572, 1052, 944, 460)
    assert borgan.stratum[:2] == ("IV", "IV")
    with pytest.raises(ValueError, match="requires 'stratum'"):
        r.cch(
            "Surv(edrel, rel) ~ age",
            ccoh_data,
            subcoh="subcohort",
            id="seqno",
            cohort_size=4028,
            method="II.Borgan",
        )
    with pytest.raises(ValueError, match="cohort size must be a scalar"):
        r.cch(
            "Surv(edrel, rel) ~ age", ccoh_data, subcoh="subcohort", id="seqno", cohort_size=[1, 2]
        )
    with pytest.raises(ValueError, match="Number of records greater than cohort size"):
        r.cch("Surv(edrel, rel) ~ age", ccoh_data, subcoh="subcohort", id="seqno", cohort_size=100)
    with pytest.raises(ValueError, match="Multiple records per id"):
        r.cch(
            "Surv(edrel, rel) ~ age", ccoh_data, subcoh="subcohort", id=[1] * 1154, cohort_size=4028
        )
    with pytest.raises(ValueError, match="Permissible values for subcohort"):
        r.cch("Surv(edrel, rel) ~ age", ccoh_data, subcoh=[2] * 1154, id="seqno", cohort_size=4028)
    with pytest.raises(TypeError, match="cohort.size is required"):
        r.cch("Surv(edrel, rel) ~ age", ccoh_data, subcoh="subcohort", id="seqno")
