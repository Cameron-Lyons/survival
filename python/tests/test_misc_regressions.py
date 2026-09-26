"""Regression tests of ``brier``, ``survcheck``, ``survobrien`` and ``cch`` against R survival
3.8-12: the curves brier reads, survcheck's row numbers after ``na.omit``, survobrien's
``I()`` terms and the id order of cch's Borgan score residuals."""

import importlib
import math

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
r_misc = importlib.import_module("survival.r._misc")


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


def assert_rows_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == approx(expected_row)


# --- brier -------------------------------------------------------------------


def test_brier_reads_the_model_curves_at_the_evaluation_times():
    # R: brier() of coxph(Surv(time, status) ~ age + sex, lung) at times 3, 180 and 365 with
    # detail = TRUE, and at its default times
    fit = r.coxph("Surv(time, status) ~ age + sex", data=datasets.load_lung(), model=True)
    result = r.brier(fit, times=[3, 180, 365], detail=True)
    assert result.brier == approx([0.0, 0.191605092010014, 0.236142139448665])
    assert math.isnan(result.rsquared[0])
    assert result.rsquared[1:] == approx([0.0460864863000336, 0.0232491313563413])
    # every curve is still 1 before the first event time
    assert result.phat[0] == [0.0] * 228
    assert result.phat[1][:4] == approx(
        [0.37766021846661, 0.348294500488361, 0.294579029600006, 0.29879827416173]
    )
    assert [result.phat[2][i] for i in (0, 1, 227)] == approx(
        [0.731243146155405, 0.694623967597056, 0.450503170391115]
    )

    default = r.brier(fit)
    assert len(default.times) == 139
    assert sum(default.brier) == approx(23.8320240782286)
    assert default.brier[-1] == approx(0.0513206230245228)


@pytest.mark.parametrize("formula", ["x", "x + strata(g)"])
def test_brier_reads_each_model_curve_once(monkeypatch, formula):
    # a curve's time and surv getters convert the whole curve (ntimes x n for an unstratified
    # fit), so brier reads each of them once per curve, not once per subject
    data = {
        "time": [2, 3, 3, 5, 6, 8, 9, 11, 12, 14],
        "status": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "x": [0.5, 1.2, -0.3, 0.8, 2.1, -1.0, 0.4, 1.5, -0.7, 0.9],
        "g": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    }
    fit = r.coxph(f"Surv(time, status) ~ {formula}", data=data)
    expected = r.brier(fit, times=[3, 6, 10])
    curves = []

    class Curve:
        def __init__(self, curve):
            self._curve = curve
            self.reads = {"time": 0, "surv": 0}
            curves.append(self)

        @property
        def time(self):
            self.reads["time"] += 1
            return self._curve.time

        @property
        def surv(self):
            self.reads["surv"] += 1
            return self._curve.surv

    class Engine:
        def __init__(self, engine):
            self._engine = engine

        def __getattr__(self, name):
            return getattr(self._engine, name)

        def survfit(self, **kwargs):
            return [Curve(curve) for curve in self._engine.survfit(**kwargs)]

    engine_of = r_misc._coxph_engine
    monkeypatch.setattr(
        r_misc, "_coxph_engine", lambda fit, message: Engine(engine_of(fit, message))
    )
    result = r.brier(fit, times=[3, 6, 10])
    assert result.brier == expected.brier
    assert len(curves) == (1 if formula == "x" else len(data["time"]))
    assert [curve.reads for curve in curves] == [{"time": 1, "surv": 1}] * len(curves)


# --- survcheck ---------------------------------------------------------------


def test_survcheck_numbers_problem_rows_of_the_data_before_na_omit():
    # R: survcheck(Surv(t1, t2, st) ~ x, data = d, id = id), na.omit dropping the rows where x
    # is missing
    gap = r.survcheck(
        "Surv(t1, t2, st) ~ x",
        data={
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "t1": [0, 1, 0, 2, 0, 1, 0, 3],
            "t2": [1, 3, 2, 4, 2, 3, 2, 5],
            "st": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": [1, None, 2, 3, None, 4, 5, 6],
        },
        id="id",
    )
    assert gap.na_action == [2, 5]
    assert (gap.flag.overlap, gap.flag.gap) == (0, 1)
    assert (gap.gap.row, gap.gap.id) == ([8], [4])
    assert gap.overlap is None

    overlap = r.survcheck(
        "Surv(t1, t2, st) ~ x",
        data={
            "id": [1, 1, 2, 2, 3, 3],
            "t1": [0, 1, 0, 1, 0, 3],
            "t2": [1, 3, 2, 4, 2, 5],
            "st": [0, 1, 0, 1, 0, 1],
            "x": [None, 1, 2, 3, None, 4],
        },
        id="id",
    )
    assert overlap.na_action == [1, 5]
    assert (overlap.flag.overlap, overlap.flag.gap) == (1, 0)
    assert (overlap.overlap.row, overlap.overlap.id) == ([4], [2])
    assert overlap.gap is None


# --- survobrien --------------------------------------------------------------

OBRIEN_DATA = {
    "time": [1, 2, 3, 4, 5],
    "status": [1, 0, 1, 1, 1],
    "x": [0.1, 0.4, 0.2, 0.8, 0.5],
    "z": [2, 7, 1, 3, 5],
    "w": [1, 2, 1, 2, 2],
}
# the columns of OBRIEN_DATA over the 11 rows of its four risk sets
OBRIEN_Z = [2, 7, 1, 3, 5, 1, 3, 5, 3, 5, 5]
OBRIEN_W = [1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2]
X_LOGITS = [
    -2.19722457733621912,
    0.0,
    -0.84729786038720356,
    2.19722457733621956,
    0.84729786038720345,
    -1.6094379124341005,
    1.60943791243410073,
    0.0,
    1.09861228866810978,
    -1.09861228866810978,
    0.0,
]
Z_LOGITS = [
    -0.84729786038720356,
    2.19722457733621956,
    -2.19722457733621912,
    0.0,
    0.84729786038720345,
    -1.6094379124341005,
    0.0,
    1.60943791243410073,
    -1.09861228866810978,
    1.09861228866810978,
    0.0,
]


def test_survobrien_leaves_asis_terms_alone():
    # R's ?survobrien example: survobrien(Surv(futime, fustat) ~ age + factor(rx) + I(ecog.ps),
    # data = ovarian)
    frame = r.survobrien(
        "Surv(futime, fustat) ~ age + factor(rx) + I(ecog.ps)", data=datasets.load_ovarian()
    )
    assert list(frame) == ["time", "status", "rx", "ecog.ps", ".id.", "age", ".strata."]
    assert len(frame["time"]) == 230
    assert frame["rx"][:8] == [1, 1, 1, 2, 1, 1, 2, 2]
    assert frame["ecog.ps"][:8] == [1, 1, 2, 1, 1, 2, 2, 2]
    assert frame["ecog.ps"][-8:] == [1, 1, 2, 2, 1, 1, 1, 2]
    assert frame[".id."][:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert frame["age"][:4] == approx(
        [2.2407096892759584, 2.7932080094425165, 1.8607523407150068, -0.72213471743319757]
    )
    assert frame[".strata."][-3:] == [12, 12, 12]

    asis = r.survobrien("Surv(time, status) ~ x + I(z)", data=OBRIEN_DATA)
    assert list(asis) == ["time", "status", "z", ".id.", "x", ".strata."]
    assert asis["z"] == OBRIEN_Z
    assert asis["x"] == approx(X_LOGITS)

    # a kept term keeps every variable it references (R: all.vars), once per term
    expression = r.survobrien("Surv(time, status) ~ x + I(z^2) + I(z * w)", data=OBRIEN_DATA)
    assert list(expression) == ["time", "status", "z", "z.1", "w", ".id.", "x", ".strata."]
    assert expression["z"] == expression["z.1"] == OBRIEN_Z
    assert expression["w"] == OBRIEN_W
    assert expression["x"] == approx(X_LOGITS)
    cut = r.survobrien("Surv(time, status) ~ x + cut(z * w, 3)", data=OBRIEN_DATA)
    assert list(cut) == ["time", "status", "z", "w", ".id.", "x", ".strata."]
    assert (cut["z"], cut["w"]) == (OBRIEN_Z, OBRIEN_W)

    # identity() does not protect a term
    identity = r.survobrien("Surv(time, status) ~ x + identity(z)", data=OBRIEN_DATA)
    assert identity["identity(z)"] == approx(Z_LOGITS)
    with pytest.raises(ValueError, match="No continuous variables to modify"):
        r.survobrien("Surv(time, status) ~ I(z) + factor(w)", data=OBRIEN_DATA)


def test_survobrien_makes_repeated_column_names_unique():
    # R: survobrien(Surv(time, status) ~ z + I(z^2), data = d), whose data.frame() names the
    # transformed z "z.1" beside the raw z of the I() term
    square = r.survobrien("Surv(time, status) ~ z + I(z^2)", data=OBRIEN_DATA)
    assert list(square) == ["time", "status", "z", ".id.", "z.1", ".strata."]
    assert square["z"] == OBRIEN_Z
    assert square["z.1"] == approx(Z_LOGITS)

    # R: the same with the formula factor(w) + w
    factor = r.survobrien("Surv(time, status) ~ factor(w) + w", data=OBRIEN_DATA)
    assert list(factor) == ["time", "status", "w", ".id.", "w.1", ".strata."]
    assert factor["w"] == OBRIEN_W
    assert factor["w.1"] == approx(
        [
            -1.38629436111989057,
            0.84729786038720345,
            -1.38629436111989057,
            0.84729786038720345,
            0.84729786038720345,
            -1.6094379124341005,
            0.69314718055994529,
            0.69314718055994529,
            0.0,
            0.0,
            0.0,
        ]
    )

    # R: ~ z + I(z^2) + z.1 with a data column z.1, a name make.unique then skips
    taken = r.survobrien(
        "Surv(time, status) ~ z + I(z^2) + z.1", data={**OBRIEN_DATA, "z.1": [9, 8, 7, 6, 5]}
    )
    assert list(taken) == ["time", "status", "z", ".id.", "z.2", "z.1", ".strata."]
    assert taken["z"] == OBRIEN_Z
    assert taken["z.2"] == approx(Z_LOGITS)
    assert taken["z.1"] == approx(
        [
            2.19722457733621956,
            0.84729786038720345,
            0.0,
            -0.84729786038720356,
            -2.19722457733621912,
            1.60943791243410073,
            0.0,
            -1.6094379124341005,
            1.09861228866810978,
            -1.09861228866810978,
            0.0,
        ]
    )


# --- cch ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def nwtco_case_cohort():
    """``?cch``'s case-cohort sample of ``nwtco`` with the ids permuted:
    ``d$rid <- ((0:(n - 1)) * 389) %% n + 1`` and ``d$cid <- paste0("p", d$rid)``."""

    nwtco = datasets.load_nwtco()
    keep = [
        i
        for i, (rel, sub) in enumerate(zip(nwtco["rel"], nwtco["in.subcohort"], strict=True))
        if rel == 1 or sub == 1
    ]
    n = len(keep)
    stage_labels = {1: "I", 2: "II", 3: "III", 4: "IV"}
    histol_labels = {1: "FH", 2: "UH"}
    rid = [(i * 389) % n + 1 for i in range(n)]
    return {
        "seqno": [nwtco["seqno"][i] for i in keep],
        "edrel": [nwtco["edrel"][i] for i in keep],
        "rel": [int(nwtco["rel"][i]) for i in keep],
        "subcohort": [int(nwtco["in.subcohort"][i]) for i in keep],
        "stage": [stage_labels[int(nwtco["stage"][i])] for i in keep],
        "histol": [histol_labels[int(nwtco["histol"][i])] for i in keep],
        "age": [nwtco["age"][i] / 12 for i in keep],
        "instit": [int(nwtco["instit"][i]) for i in keep],
        "rid": rid,
        "cid": [f"p{value}" for value in rid],
    }


def _borgan(data, method, ids):
    """R's cch(Surv(edrel, rel) ~ stage + histol + age, subcoh = ~subcohort, id = ids,
    stratum = ~instit, cohort.size = table(nwtco$instit), method = method)."""

    return r.cch(
        "Surv(edrel, rel) ~ stage + histol + age",
        data,
        subcoh="subcohort",
        id=ids,
        stratum="instit",
        cohort_size={"1": 3622, "2": 406},
        method=method,
    )


def test_cch_borgan_score_rows_follow_r_id_order(nwtco_case_cohort):
    expected = {
        "I.Borgan": (
            [
                [
                    0.434508125693249,
                    0.455160419557487,
                    -1.1975296944037,
                    0.500361160937554,
                    2.94647571817348,
                ],
                [
                    -0.282498126382607,
                    0.695542315336831,
                    -0.219370797286702,
                    -0.362620647175176,
                    9.59587870520824,
                ],
                [
                    0.127112961705576,
                    0.133154676656036,
                    0.0854300195617015,
                    0.146377904872864,
                    -0.55424715485416,
                ],
            ],
            [
                0.707731047133707,
                -0.311572305679621,
                -0.19475394326966,
                0.656850908927333,
                -3.95407439623791,
            ],
        ),
        "II.Borgan": (
            [
                [
                    0.374883691848907,
                    0.393913948561484,
                    -1.03535116790586,
                    0.430058324745435,
                    2.54142695721559,
                ],
                [
                    -0.27895267540261,
                    0.687161896070355,
                    -0.223760609505231,
                    -0.376778709231142,
                    9.40784279908694,
                ],
                [
                    0.12943654732043,
                    0.136007147154606,
                    0.0865279755087688,
                    0.148486759791869,
                    -0.565534452075215,
                ],
            ],
            [
                0.582059737144811,
                -0.250141225433122,
                -0.16496839140021,
                0.54319555458312,
                -3.25525952959897,
            ],
        ),
    }
    n = len(nwtco_case_cohort["rid"])
    for method, (head, last) in expected.items():
        fit = _borgan(nwtco_case_cohort, method, "rid")
        assert fit.sc_ids == tuple(range(1, n + 1))
        assert_rows_close(fit.sc[:3], head)
        assert fit.sc[-1] == approx(last)

    numeric = _borgan(nwtco_case_cohort, "I.Borgan", "rid")
    # character ids sort as strings: p1, p10, p100, ...
    character = _borgan(nwtco_case_cohort, "I.Borgan", "cid")
    assert character.sc_ids[:5] == ("p1", "p10", "p100", "p1000", "p1001")
    assert character.sc[1] == approx(numeric.sc[9])
    # factor ids follow their levels: factor(rid, levels = n:1) gives the rows of 1154, 1153, ...
    factor_id = RFactor(nwtco_case_cohort["rid"], range(n, 0, -1))
    factor = _borgan(nwtco_case_cohort, "I.Borgan", factor_id)
    assert factor.sc_ids[:3] == (n, n - 1, n - 2)
    assert_rows_close(factor.sc, numeric.sc[::-1])


def test_cch_prentice_fit_carries_the_point_estimate(nwtco_case_cohort):
    # R's Prentice sets fit$coefficients <- fit1$coefficients on the augmented-data fit
    fit = r.cch(
        "Surv(edrel, rel) ~ stage + histol + age",
        nwtco_case_cohort,
        subcoh="subcohort",
        id="seqno",
        cohort_size=4028,
    )
    assert fit.coefficients == approx(
        [
            0.734570842045653,
            0.597083557948202,
            1.38413196899809,
            1.49806307262597,
            0.0432678728347779,
        ]
    )
    assert list(fit.fit.fit.coefficients) == fit.coefficients
    assert (fit.sc, fit.sc_ids) == (None, None)
