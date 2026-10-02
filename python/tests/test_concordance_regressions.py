"""``survival.r.concordance`` regressions against R survival 3.8-12.

Reference values were computed with R 4.5.3 and survival 3.8-12 on the bundled
datasets; the R calls are quoted above each test.
"""

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
concordancefit = survival.r._concordance.concordancefit

COUNT_NAMES = ("concordant", "discordant", "tied.x", "tied.y", "tied.xy")
SURVCONCORDANCE_NAMES = ("concordant", "discordant", "tied.risk", "tied.time", "std(c-d)")


def approx(values, rel=1e-9):
    return pytest.approx(values, rel=rel, abs=1e-15)


def matrix(rows, rel=1e-9):
    return [approx(row, rel) for row in rows]


def counts(rows):
    return [[row[name] for name in COUNT_NAMES] for row in rows]


def _complete(data, *columns):
    """The rows of a bundled dataset with no missing value in ``columns``."""

    n = len(next(iter(data.values())))
    keep = [i for i in range(n) if all(data[name][i] is not None for name in columns)]
    return {name: [values[i] for i in keep] for name, values in data.items()}


@pytest.mark.parametrize(
    ("counting", "timewt"),
    [(False, name) for name in ("n", "S", "S/G", "n/G2", "I")]
    + [(True, name) for name in ("n", "S", "I")],
)
def test_zero_weight_last_event_preserves_concordance_variance(timewt, counting):
    # An event with no weighted observations at risk contributes no variance.
    # Its 0/0 Cox-variance increment used to poison otherwise valid results.
    stop = [1.0, 2.0, 3.0, 4.0]
    start = [0.0, 0.0, 1.5, 1.0]
    status = [1, 1, 1, 1]
    response = r.Surv(start, stop, status) if counting else r.Surv(stop, status)
    retained = (
        r.Surv(start[:-1], stop[:-1], status[:-1]) if counting else r.Surv(stop[:-1], status[:-1])
    )
    actual = concordancefit(
        response, [3.0, 2.0, 4.0, 1.0], weights=[0.5, 1.5, 1.0, 0.0], timewt=timewt
    )
    expected = concordancefit(retained, [3.0, 2.0, 4.0], weights=[0.5, 1.5, 1.0], timewt=timewt)
    assert actual.concordance == approx(expected.concordance)
    assert actual.var == approx(expected.var)
    assert actual.cvar == approx(expected.cvar)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def ovarian():
    return datasets.load_ovarian()


@pytest.fixture(scope="module")
def cox_pair(lung):
    return (
        r.coxph("Surv(time, status) ~ age + strata(sex)", lung),
        r.coxph("Surv(time, status) ~ age", lung),
    )


# --- several fits: R's cord.work --------------------------------------------------------


# f1 <- coxph(Surv(time, status) ~ age + strata(sex), lung)
# f2 <- coxph(Surv(time, status) ~ age, lung)
# c12 <- concordance(f1, f2); c12$concordance; c12$var; c12$count; c12$cvar
def test_several_fits_each_keep_their_own_strata(cox_pair):
    result = r.concordance(*cox_pair)
    assert result.names == ["fit1", "fit2"]
    assert result.concordance == approx([0.545896226415094, 0.550239832117518])
    assert result.var == matrix(
        [
            [0.000669827984401553, 0.000618572261574279],
            [0.000618572261574279, 0.000632125775421889],
        ]
    )
    # the stratified fit's count row is summed over its strata
    assert counts(result.count) == [[5631, 4658, 311, 17, 0], [10717, 8706, 591, 27, 1]]
    assert result.cvar == approx([0.000711293243698327, 0.000677485306439484])
    assert result.dfbeta is None
    assert result.influence is None


# concordance(f1, f2, influence = 1)$dfbeta[1:3, ]
# concordance(f1, f2, influence = 2)$influence[1:2, , ]
# names(concordance(f1, f2, influence = 3)): neither dfbeta nor influence
def test_several_fits_return_dfbeta_or_influence_as_r(cox_pair):
    dfbeta = r.concordance(*cox_pair, influence=1)
    assert len(dfbeta.dfbeta) == 228
    assert dfbeta.dfbeta[:3] == matrix(
        [
            [-0.001552985938056248, -0.000749241944149638],
            [-0.000926788892844429, -0.000973964722146281],
            [0.002675436098255607, 0.001934167467803016],
        ]
    )
    assert dfbeta.influence is None
    influence = r.concordance(*cox_pair, influence=2)
    assert influence.dfbeta is None
    assert [fit[:2] for fit in influence.influence] == [
        [[45, 67, 7, 0, 0], [51, 60, 5, 0, 0]],
        [[85, 96, 8, 0, 0], [75, 96, 8, 0, 0]],
    ]
    both = r.concordance(*cox_pair, influence=3)
    assert both.dfbeta is None
    assert both.influence is None
    with pytest.raises(ValueError, match="influence must be 0, 1, 2 or 3"):
        r.concordance(*cox_pair, influence=4)


# f3 <- coxph(Surv(time, status) ~ sex, lung)
# rk <- concordance(f2, f3, ranks = TRUE)$ranks; dim(rk); rk[c(1, 166, 330), ]
def test_several_fits_stack_their_ranks_with_a_fit_column(lung, cox_pair):
    ranks = r.concordance(cox_pair[1], r.coxph("Surv(time, status) ~ sex", lung), ranks=True).ranks
    assert list(ranks) == ["fit", "time", "rank", "timewt", "casewt"]
    assert len(ranks["time"]) == 330
    rows = [[ranks[name][i] for name in ranks] for i in (0, 165, 329)]
    assert rows == [
        ["fit1", 5.0, approx(0.157894736842105), 228.0, 1.0],
        ["fit2", 5.0, approx(-0.605263157894737), 228.0, 1.0],
        ["fit2", 883.0, approx(0.25), 4.0, 1.0],
    ]


# lung2 <- lung[!is.na(lung$inst), ]
# k1 <- coxph(Surv(time, status) ~ age + cluster(inst), lung2)
# k2 <- coxph(Surv(time, status) ~ sex, lung2)
# R stops for concordance(k1, k2) and for concordance(k2, k1)
# kc <- concordance(k1, k2, cluster = lung2$inst); kc$concordance; kc$var
def test_several_fits_need_identical_clustering(lung):
    lung2 = _complete(lung, "inst")
    k1 = r.coxph("Surv(time, status) ~ age + cluster(inst)", lung2)
    k2 = r.coxph("Surv(time, status) ~ sex", lung2)
    for fits in ((k1, k2), (k2, k1)):
        with pytest.raises(ValueError, match="models must have identical clustering"):
            r.concordance(*fits)
    # an explicit cluster replaces every fit's own
    clustered = r.concordance(k1, k2, cluster=lung2["inst"])
    assert clustered.concordance == approx([0.550741450620397, 0.579113285584586])
    assert clustered.var == matrix(
        [
            [6.24000670706836e-04, -1.38980243694358e-05],
            [-1.38980243694358e-05, 3.01399290492933e-04],
        ]
    )


# lw <- lung; lw$w <- rep(c(1, 2, 0.5, 1.5), length.out = nrow(lw))
# g1 <- coxph(Surv(time, status) ~ age, lw, weights = w)
# g2 <- coxph(Surv(time, status) ~ sex, lw, weights = w)
# crossprod(cbind(concordance(g1, influence = 1)$dfbeta, concordance(g2, influence = 1)$dfbeta))
def test_several_weighted_fits_take_the_crossproduct_of_dfbeta(lung):
    lw = dict(lung)
    lw["w"] = ([1, 2, 0.5, 1.5] * 57)[:228]
    g1 = r.coxph("Surv(time, status) ~ age", lw, weights="w")
    g2 = r.coxph("Surv(time, status) ~ sex", lw, weights="w")
    result = r.concordance(g1, g2)
    assert result.concordance == approx([0.545029578291279, 0.582168968962619])
    # R's cord.work computes t(wt * dfbeta) %*% dfbeta, which weights the already
    # weighted dfbeta twice (var[1, 1] = 0.00133123 there against R's own single-fit
    # 0.000786123); the diagonal here is each fit's own variance.
    assert result.var == matrix(
        [
            [0.000786123356280357, 0.000135742600532029],
            [0.000135742600532029, 0.000513651825852599],
        ]
    )


# s1 <- survreg(Surv(time, status) ~ age, lung); s2 <- survreg(Surv(time, status) ~ age + sex, lung)
# cs <- concordance(s1, s2); cs$concordance; cs$var
def test_several_survreg_fits(lung):
    result = r.concordance(
        r.survreg("Surv(time, status) ~ age", lung),
        r.survreg("Surv(time, status) ~ age + sex", lung),
    )
    assert result.concordance == approx([0.550239832117518, 0.602902967922454])
    assert result.var == matrix(
        [
            [0.000632125775421889, 0.000414229676868845],
            [0.000414229676868845, 0.000649711043764914],
        ]
    )
    assert counts(result.count) == [[10717, 8706, 591, 27, 1], [11911, 7792, 311, 28, 0]]


# --- timewt = "I" with more than ten strata ---------------------------------------------

# Twelve strata g of five rows (uniform times, normal x, set.seed(3)); strata 2, 5, 8
# and 11 have a single event.  concordance(Surv(time, status) ~ x + strata(g), d, timewt = "I")
TWELVE_STRATA = {
    "time": [17.6, 80.9, 39.1, 33.4, 60.6, 60.8, 13.3, 30.2, 58.2, 63.5, 51.7, 51.0, 53.9, 56.2,
             86.9, 83.1, 12.0, 70.7, 89.9, 28.7, 23.6, 2.5, 13.8, 10.2, 24.5, 79.3, 60.4, 91.1,
             56.5, 75.8, 38.5, 38.0, 17.9, 45.9, 26.6, 34.3, 89.1, 21.0, 58.3, 21.6, 28.9, 78.8,
             18.1, 57.5, 42.5, 27.5, 5.7, 11.2, 32.1, 80.3, 23.7, 22.1, 87.8, 99.3, 84.6, 91.1,
             47.7, 23.2, 13.7, 28.7],
    "status": [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,
               0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
               0, 1, 1, 0, 1, 0],
    "x": [0.9, 0.85, 0.73, 0.74, -0.35, 0.71, 1.3, 0.04, -0.98, 0.79, 0.79, -0.31, 1.7, -0.79,
          0.35, -2.27, -0.16, 1.13, -0.46, -0.9, 0.73, -0.81, 0.27, -1.74, -1.41, -0.45, -1.04,
          1.36, 0.92, -0.79, 0.57, 0.92, 0.26, 0.35, 1.17, -0.48, -0.42, 0.96, -1.29, 0.19,
          -0.03, 0.47, 1.02, 0.27, 0.23, 0.75, 1.22, 0.38, -0.99, -0.16, 1.74, -0.35, 0.69,
          1.22, 0.79, -0.01, 0.22, -0.89, 0.44, -0.89],
    "g": [s for s in range(1, 13) for _ in range(5)],
}  # fmt: skip


def test_timewt_i_with_more_than_ten_strata_counts_events_over_all_strata():
    result = r.concordance("Surv(time, status) ~ x + strata(g)", TWELVE_STRATA, timewt="I")
    assert result.concordance == approx(0.420819490586932, rel=1e-12)
    assert result.var == approx(0.0135478638944293, rel=1e-12)
    assert counts([result.count]) == [[approx(6.33333333333333), approx(8.71666666666667), 0, 0, 0]]
    # concordancefit(..., keepstrata = TRUE) does not merge the strata
    kept = r.concordance(
        "Surv(time, status) ~ x + strata(g)", TWELVE_STRATA, timewt="I", keepstrata=True
    )
    assert kept.concordance == approx(0.38200339558573854, rel=1e-12)


# --- row order of clusters and strata ---------------------------------------------------


# cc <- concordance(Surv(time, status) ~ age, lung, cluster = inst, influence = 1)
# length(cc$dfbeta); cc$dfbeta[1:3]; cc$var
def test_clustered_dfbeta_rows_follow_sorted_clusters(lung):
    result = r.concordance(
        "Surv(time, status) ~ age", lung, cluster="inst", influence=1, na_action="omit"
    )
    assert len(result.dfbeta) == 18
    # inst 1, 2, 3
    assert result.dfbeta[:3] == approx(
        [-0.01685417698691620, 0.00473614290794148, -0.00456802793198232]
    )
    assert result.var == approx(0.000624000670706835)


# cl <- c("z", "y", "x")[ovarian$rx]; st <- c("b", "a")[ovarian$rx]
# co <- with(ovarian, concordancefit(Surv(futime, fustat), age, st, cluster = cl, influence = 1))
# co$dfbeta; co$count
def test_character_clusters_and_strata_come_out_in_sorted_order(ovarian):
    cluster = [("z", "y", "x")[int(rx) - 1] for rx in ovarian["rx"]]
    strata = [("b", "a")[int(rx) - 1] for rx in ovarian["rx"]]
    y = r.Surv(ovarian["futime"], ovarian["fustat"])
    result = concordancefit(y, ovarian["age"], strata=strata, cluster=cluster, influence=1)
    assert result.dfbeta == approx([-0.00725623582766440, 0.00725623582766442])  # y, z
    assert result.names == ["a", "b"]
    assert counts(result.count) == [[8, 36, 0, 0, 0], [12, 49, 0, 0, 0]]
    with pytest.raises(ValueError, match="cluster contains missing values"):
        concordancefit(y, ovarian["age"], cluster=[None, *cluster[1:]])


# --- survConcordance and survConcordance.fit ---------------------------------------------


def _row(values):
    return dict(zip(SURVCONCORDANCE_NAMES, values, strict=True))


# s <- survConcordance(Surv(time, status) ~ age, lung); s$concordance; s$stats; s$std.err
# s2 <- survConcordance(Surv(time, status) ~ age + strata(sex), lung)
def test_survconcordance_reports_r_statistics(lung):
    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = r.survConcordance("Surv(time, status) ~ age", lung)
    assert isinstance(result, r.SurvConcordanceResult)
    assert result.concordance == approx(0.550239832117518)
    assert result.n == 228
    assert result.stats == approx(_row([10717, 8706, 591, 28, 1041.8707158463]))
    assert result.std_err == approx(0.0260285479126186)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        stratified = r.survConcordance("Surv(time, status) ~ age + strata(sex)", lung)
    assert stratified.concordance == approx(0.545896226415094)
    assert stratified.stats == {
        "sex=1": approx(_row([4382, 3502, 239, 15, 515.732989747100])),
        "sex=2": approx(_row([1249, 1156, 72, 2, 231.739333593359])),
    }
    assert stratified.std_err == approx(0.0352581284594556)
    with (
        pytest.warns(DeprecationWarning, match="deprecated"),
        pytest.raises(ValueError, match="Only one predictor variable allowed"),
    ):
        r.survConcordance("Surv(time, status) ~ age + sex", lung)


# lw$w <- rep(c(1, 2, 0.5, 1.5), length.out = nrow(lw))
# survConcordance.fit(Surv(lung$time, lung$status), lung$age, lung$sex)
# survConcordance.fit(Surv(lw$time, lw$status), lw$age, weight = lw$w)
# survConcordance.fit(Surv(heart$start, heart$stop, heart$event), heart$age)
# survConcordance(Surv(time, status) ~ age, lw, weights = w)[c("concordance", "std.err")]
@pytest.mark.filterwarnings("ignore:survConcordance:DeprecationWarning")
def test_survconcordance_fit_rows_per_stratum_weights_and_counting_data(lung):
    y = r.Surv(lung["time"], lung["status"])
    weights = ([1, 2, 0.5, 1.5] * 57)[:228]
    stratified = r.survConcordance_fit(y, lung["age"], lung["sex"])
    weighted = r.survConcordance_fit(y, lung["age"], weight=weights)
    heart = datasets.load_heart()
    counting = r.survConcordance_fit(
        r.Surv(heart["start"], heart["stop"], heart["event"]), heart["age"]
    )
    weighted_formula = r.survConcordance(
        "Surv(time, status) ~ age", {**lung, "w": weights}, weights="w"
    )
    assert stratified == {
        "1": approx(_row([4382, 3502, 239, 15, 515.732989747100])),
        "2": approx(_row([1249, 1156, 72, 2, 231.739333593359])),
    }
    assert weighted == approx(_row([16740, 13900.75, 885.75, 50.25, 1464.73677985241]))
    assert counting == approx(_row([2600, 1918, 1, 16, 335.637929153228]))
    assert weighted_formula.concordance == approx(0.545029578291279)
    assert weighted_formula.std_err == approx(0.0232302472499708)
    # R's docount segfaults on a short x; its unstratified path has concordancefit's check
    with pytest.raises(ValueError, match="x and y are not the same length"):
        r.survConcordance_fit(y, lung["age"][:6], lung["sex"])


# --- the fit-object path -----------------------------------------------------------------


# fit <- coxph(Surv(time, status) ~ age, lung)
# concordance(fit, weights = lung$wt.loss)   # and reverse, data, strata, subset, scores, na.action
@pytest.mark.parametrize(
    ("argument", "value", "r_name"),
    [
        ("weights", [1.0] * 228, "weights"),
        ("reverse", True, "reverse"),
        ("data", {}, "data"),
        ("strata", [1] * 228, "strata"),
        ("subset", [1, 2], "subset"),
        ("scores", [1.0] * 228, "scores"),
        ("na_action", "omit", "na.action"),
    ],
)
def test_fit_path_rejects_arguments_r_reads_as_fits(cox_pair, argument, value, r_name):
    with pytest.raises(TypeError, match=f"^{r_name} argument is not an appropriate fit object"):
        r.concordance(cox_pair[1], **{argument: value})


# l100 <- lung[1:100, ]; l100$w <- rep(1:4, 25)
# sfit <- survreg(Surv(time, status) ~ age + sex, data = l100, weights = w)
# cn <- concordance(sfit, newdata = lung[101:228, ]); cn$concordance; cn$var; cn$count
def test_survreg_fit_scored_on_newdata(lung):
    first = {key: values[:100] for key, values in lung.items()}
    first["w"] = [1, 2, 3, 4] * 25
    fit = r.survreg("Surv(time, status) ~ age + sex", first, weights="w")
    result = r.concordance(fit, newdata={key: values[100:] for key, values in lung.items()})
    assert result.n == 128
    assert result.concordance == approx(0.616066323613493)
    assert result.var == approx(0.00131655993065353)
    assert counts([result.count]) == [[3197, 1979, 71, 7, 0]]


# first <- lung[1:100, ]; second <- lung[101:228, ]   (second has 8 missing wt.loss)
# sf <- survreg(Surv(time, status) ~ age + wt.loss + strata(sex), first)
# cf <- coxph(Surv(time, status) ~ age + wt.loss + strata(sex), first)
# for (f in list(sf, cf)) print(concordance(f, newdata = second)[c("concordance", "var", "n")])
# second$sex[5] <- NA
# concordance(coxph(Surv(time, status) ~ age + strata(sex), first), newdata = second)
def test_newdata_rows_with_a_missing_variable_are_omitted(lung):
    first = {key: values[:100] for key, values in lung.items()}
    second = {key: values[100:] for key, values in lung.items()}
    formula = "Surv(time, status) ~ age + wt.loss + strata(sex)"
    survreg_fit = r.survreg(formula, first, na_action="omit")
    scored = r.concordance(survreg_fit, newdata=second)
    assert scored.n == 120
    assert scored.concordance == approx(0.520668425681618)
    assert scored.var == approx(0.0017251965750011)
    assert counts(scored.count) == [[891, 857, 0, 4, 0], [293, 233, 0, 1, 0]]
    cox = r.concordance(r.coxph(formula, first, na_action="omit"), newdata=second)
    assert cox.n == 120
    assert cox.concordance == approx(0.523306948109059)
    assert cox.var == approx(0.00170260624238661)
    missing_stratum = {**second, "sex": [*second["sex"][:4], None, *second["sex"][5:]]}
    stratified = r.concordance(
        r.coxph("Surv(time, status) ~ age + strata(sex)", first), newdata=missing_stratum
    )
    assert stratified.n == 127
    assert stratified.concordance == approx(0.520597127739985)
    assert stratified.var == approx(0.00149110080316185)


# concordance(time ~ age, lung, timewt = "I"): R forces timewt = "n" for a non-Surv response
# lung$lg <- lung$status == 2; concordance(lg ~ age, lung)
# lung$of <- factor(lung$ph.ecog, ordered = TRUE); concordance(of ~ age, lung)
# lung$f2 <- factor(lung$sex, labels = c("m", "f")); concordance(f2 ~ age, lung)
# concordance(factor(ph.ecog) ~ age, lung)   # error
def test_non_surv_responses_follow_concordance_formula(lung):
    numeric = r.concordance("time ~ age", lung, timewt="I")
    assert numeric.concordance == approx(0.468587907408841)
    assert numeric.var == approx(0.000466072978625897)
    logical = r.concordance("lg ~ age", {**lung, "lg": [s == 2 for s in lung["status"]]})
    assert logical.concordance == approx(0.583020683020683)
    assert counts([logical.count]) == [[5902, 4176, 317, 15038, 445]]
    assert logical.var == approx(0.00177228912803445)

    class OrderedFactor(RFactor):
        ordered = True

    ecog = lung["ph.ecog"]
    ordered = r.concordance(
        "of ~ age", {**lung, "of": OrderedFactor(ecog, [0, 1, 2, 3])}, na_action="omit"
    )
    assert ordered.n == 227
    assert ordered.concordance == approx(0.595757200371632)
    assert counts([ordered.count]) == [[9399, 6307, 439, 9193, 313]]
    sex = RFactor(["m" if s == 1 else "f" for s in lung["sex"]], ["m", "f"])
    two_level = r.concordance("f2 ~ age", {**lung, "f2": sex})
    assert two_level.concordance == approx(0.425402576489533)
    assert counts([two_level.count]) == [[5095, 6948, 377, 13073, 385]]
    with pytest.raises(ValueError, match="orderable factor"):
        r.concordance("f3 ~ age", {**lung, "f3": RFactor(ecog, [0, 1, 2, 3])}, na_action="omit")


# concordance(I(status == 2) ~ age, lung); concordance(status == 2 ~ age, lung)
# concordance(I(status == 2) ~ ., lung[, c("status", "age")])   # "." leaves status out
# concordance(factor(sex) ~ age, lung); concordance(factor(ph.ecog) ~ age, lung)   # error
# lung$ch <- ifelse(lung$sex == 1, "a", "b"); concordance(ch ~ age, lung)   # error
def test_response_expressions_follow_concordance_formula(lung):
    for formula in ("I(status == 2) ~ age", "status == 2 ~ age"):
        logical = r.concordance(formula, lung)
        assert logical.concordance == approx(0.583020683020683)
        assert counts([logical.count]) == [[5902, 4176, 317, 15038, 445]]
        assert logical.var == approx(0.00177228912803445)
        assert logical.formula == formula
    dot = r.concordance("I(status == 2) ~ .", {"status": lung["status"], "age": lung["age"]})
    assert dot.concordance == approx(0.583020683020683)
    two_level = r.concordance("factor(sex) ~ age", lung)
    assert two_level.concordance == approx(0.425402576489533)
    assert two_level.var == approx(0.00148742218191791)
    with pytest.raises(ValueError, match="orderable factor"):
        r.concordance("factor(ph.ecog) ~ age", lung, na_action="omit")
    character = {**lung, "ch": ["a" if sex == 1 else "b" for sex in lung["sex"]]}
    with pytest.raises(ValueError, match="orderable factor"):
        r.concordance("ch ~ age", character)
