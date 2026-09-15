"""``survival.r.concordance``/``concordancefit`` against R survival 3.8.11 (ovarian, cgd)."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
concordancefit = survival.r._concordance.concordancefit
datasets = survival.datasets


def approx(values, rel=1e-8):
    if isinstance(values, list) and values and isinstance(values[0], list):
        return [pytest.approx(row, rel=rel) for row in values]
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def ovarian():
    return datasets.load_ovarian()


def counts(row):
    return [row[name] for name in ("concordant", "discordant", "tied.x", "tied.y", "tied.xy")]


def test_formula_concordance_matches_r(ovarian):
    result = r.concordance("Surv(futime, fustat) ~ age", ovarian)
    assert isinstance(result, r.ConcordanceResult)
    assert result.concordance == approx(0.215596330275229)
    assert counts(result.count) == [47, 171, 0, 0, 0]
    assert result.n == 26
    assert result.names == ["age"]
    assert result.var == approx(0.00683429490130578, rel=1e-6)
    assert result.cvar == approx(0.00823092893415257, rel=1e-6)
    assert result.std == approx(math.sqrt(0.00683429490130578), rel=1e-6)
    assert result.dfbeta is None
    assert result.influence is None
    assert result.ranks is None
    assert result.formula == "Surv(futime, fustat) ~ age"
    reverse = r.concordance("Surv(futime, fustat) ~ age", ovarian, reverse=True)
    assert reverse.concordance == approx(0.784403669724771)
    assert counts(reverse.count) == [171, 47, 0, 0, 0]
    numeric = r.concordance("futime ~ age", ovarian)  # y ~ x is read as Surv(y) ~ x
    assert numeric.concordance == approx(0.276923076923077)
    assert counts(numeric.count) == [90, 235, 0, 0, 0]
    frame = r.as_data_frame(result)
    assert frame["score"] == ["age"]
    assert frame["concordance"] == approx([0.215596330275229])
    assert frame["var"] == approx([0.00683429490130578], rel=1e-6)


def test_several_predictors_strata_weights_cluster(ovarian):
    two = r.concordance("Surv(futime, fustat) ~ age + ecog.ps", ovarian)
    assert two.names == ["age", "ecog.ps"]
    assert two.concordance == approx([0.215596330275229, 0.479357798165138])
    assert [counts(row) for row in two.count] == [[47, 171, 0, 0, 0], [52, 61, 105, 0, 0]]
    assert two.var == approx(
        [[0.00683429490130578, 0.00151802859630624], [0.00151802859630624, 0.00615280401676528]],
        rel=1e-6,
    )
    assert two.cvar == approx([0.00823092893415257, 0.00616530595067755], rel=1e-6)
    assert two.std == approx(
        [math.sqrt(0.00683429490130578), math.sqrt(0.00615280401676528)], rel=1e-6
    )
    assert r.as_data_frame(two)["score"] == ["age", "ecog.ps"]
    strata = r.concordance("Surv(futime, fustat) ~ age + strata(rx)", ovarian)
    assert strata.concordance == approx(0.19047619047619)
    assert strata.names == ["rx=1", "rx=2"]
    assert [counts(row) for row in strata.count] == [[12, 49, 0, 0, 0], [8, 36, 0, 0, 0]]
    assert strata.var == approx(0.00322663910613376, rel=1e-6)
    weighted = r.concordance("Surv(futime, fustat) ~ age", ovarian, weights=[1, 2] * 13)
    assert weighted.concordance == approx(0.170995670995671)
    assert counts(weighted.count) == [79, 383, 0, 0, 0]
    clustered = r.concordance("Surv(futime, fustat) ~ age + cluster(ecog.ps)", ovarian)
    assert clustered.var == approx(0.000963324571790079, rel=1e-6)
    assert r.concordance("Surv(futime, fustat) ~ age", ovarian, cluster="ecog.ps").var == approx(
        clustered.var, rel=1e-6
    )


@pytest.mark.parametrize(
    ("timewt", "concordance", "var"),
    [
        ("S", 0.222195836301749, 0.00672063049787914),
        ("S/G", 0.229173728381276, 0.00665174171616624),
        ("n/G2", 0.229173728381276, 0.00665174171616624),
        ("I", 0.235915033618783, 0.00687904639774127),
    ],
)
def test_time_weights(ovarian, timewt, concordance, var):
    result = r.concordance("Surv(futime, fustat) ~ age", ovarian, timewt=timewt)
    assert result.concordance == approx(concordance)
    assert result.var == approx(var, rel=1e-6)


def test_influence_ranks_and_bounds(ovarian):
    dfbeta = r.concordance("Surv(futime, fustat) ~ age", ovarian, influence=1)
    assert dfbeta.dfbeta[:3] == approx(
        [-0.0155500378755997, -0.0155500378755997, -0.0201371938389024], rel=1e-6
    )
    assert dfbeta.influence is None
    influence = r.concordance("Surv(futime, fustat) ~ age", ovarian, influence=2)
    assert influence.influence[:2] == [[2.0, 23.0, 0.0, 0.0, 0.0]] * 2
    assert influence.dfbeta is None
    both = r.concordance("Surv(futime, fustat) ~ age", ovarian, influence=3)
    assert both.dfbeta is not None
    assert both.influence is not None
    ranks = r.concordance("Surv(futime, fustat) ~ age", ovarian, ranks=True)
    assert len(ranks.ranks) == 12
    assert ranks.ranks[0] == approx(
        {"time": 59.0, "rank": -0.807692307692308, "timewt": 26.0, "casewt": 1.0}
    )
    assert ranks.ranks[1]["rank"] == approx(-0.88)
    ymax = r.concordance("Surv(futime, fustat) ~ age", ovarian, ymax=500)
    assert ymax.concordance == approx(0.203045685279188)
    assert counts(ymax.count) == [40, 157, 0, 0, 0]
    assert r.concordance("Surv(futime, fustat) ~ age", ovarian, ymin=100).concordance == approx(
        0.215596330275229
    )
    with pytest.raises(TypeError, match="ymin must be a single number"):
        r.concordance("Surv(futime, fustat) ~ age", ovarian, ymin="a")
    with pytest.raises(ValueError, match="timewt must be one of"):
        r.concordance("Surv(futime, fustat) ~ age", ovarian, timewt="x")
    with pytest.raises(ValueError, match="influence must be"):
        r.concordance("Surv(futime, fustat) ~ age", ovarian, influence=4)


def test_counting_process_response():
    result = r.concordance("Surv(tstart, tstop, status) ~ age", datasets.load_cgd())
    assert result.concordance == approx(0.573310499275458)
    assert counts(result.count) == [4233, 3120, 238, 6, 0]
    assert result.var == approx(0.00137101541502595, rel=1e-6)
    with pytest.raises(ValueError, match="not supported for \\(time1, time2\\) data"):
        r.concordance("Surv(tstart, tstop, status) ~ age", datasets.load_cgd(), timewt="S/G")


def test_concordance_of_fits(ovarian):
    fit = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian)
    result = r.concordance(fit)
    assert result.concordance == approx(0.798165137614679)
    assert counts(result.count) == [174, 44, 0, 0, 0]
    assert result.var == approx(0.00582568867555592, rel=1e-6)
    assert result.concordance == approx(fit.concordance["concordance"])
    smaller = r.coxph("Surv(futime, fustat) ~ age", ovarian)
    both = r.concordance(fit, smaller)
    assert both.names == ["fit1", "fit2"]
    assert both.concordance == approx([0.798165137614679, 0.784403669724771])
    assert both.var == approx(
        [[0.00582568867555592, 0.00617202713486753], [0.00617202713486753, 0.00683429490130578]],
        rel=1e-6,
    )
    assert [counts(row) for row in both.count] == [[174, 44, 0, 0, 0], [171, 47, 0, 0, 0]]
    newdata = {key: values[:20] for key, values in ovarian.items() if isinstance(values, list)}
    with_newdata = r.concordance(fit, newdata=newdata)
    assert with_newdata.concordance == approx(0.84070796460177)
    assert with_newdata.n == 20
    stratified = r.concordance(r.coxph("Surv(futime, fustat) ~ age + strata(rx)", ovarian))
    assert stratified.concordance == approx(0.80952380952381)
    assert stratified.names == ["rx=1", "rx=2"]
    assert [counts(row) for row in stratified.count] == [[49, 12, 0, 0, 0], [36, 8, 0, 0, 0]]
    clustered = r.concordance(
        r.coxph("Surv(futime, fustat) ~ age + rx + cluster(ecog.ps)", ovarian)
    )
    assert clustered.var == approx(0.000607106237378652, rel=1e-6)
    with pytest.raises(TypeError, match="not an appropriate fit object"):
        r.concordance(fit, object())
    with pytest.raises(TypeError, match="a formula argument is required"):
        r.concordance(None)


def test_concordancefit_and_deprecated_entry_points(ovarian):
    y = r.Surv(ovarian["futime"], ovarian["fustat"])
    fit = concordancefit(y, ovarian["age"])
    assert fit.concordance == approx(0.215596330275229)
    assert fit.var is not None
    assert concordancefit(y, ovarian["age"], std_err=False).var is None
    assert concordancefit(
        y, [[a, e] for a, e in zip(ovarian["age"], ovarian["ecog.ps"], strict=True)]
    ).concordance == approx([0.215596330275229, 0.479357798165138])
    assert r.concordance(y, scores=ovarian["age"], reverse=True).concordance == approx(
        0.784403669724771
    )
    with pytest.raises(ValueError, match="x and y are not the same length"):
        concordancefit(y, ovarian["age"][:5])
    with pytest.raises(ValueError, match="left or interval censored"):
        concordancefit(r.Surv(ovarian["futime"], ovarian["fustat"], type="left"), ovarian["age"])
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = r.survConcordance("Surv(futime, fustat) ~ age", ovarian)
    assert legacy.concordance == approx(0.784403669724771)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        stats = r.survConcordance_fit(y, ovarian["age"])
    assert (stats["concordant"], stats["discordant"]) == (171.0, 47.0)
    assert stats["std(c-d)"] == approx(2 * math.sqrt(0.00683429490130578) * 218, rel=1e-6)
