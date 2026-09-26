"""survfit.coxph regressions against R 4.5.3 with survival 3.8-12.

The old-style ``type`` picks R's curve, ``start.time`` builds the curves from the rows still
at risk, models with an interaction missing its lower-order terms are refused, incomplete
newdata rows are left out (``na.omit``), curves carry R's names (id values, newdata row
names) and the confidence limits of a newdata matrix come from one ``survfit_confint``.
"""

import dataclasses
import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
_coxph = survival.r._coxph


def approx(values, rel=1e-10):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def lung_fit(lung):
    return r.coxph("Surv(time, status) ~ age + sex", lung)


@pytest.fixture(scope="module")
def heart_fit():
    return r.coxph("Surv(start, stop, event) ~ age + transplant", datasets.load_heart())


def _heart_newdata(**changes):
    newdata = {
        "start": [0, 50, 0, 20],
        "stop": [50, 400, 20, 1000],
        "event": [0, 0, 0, 0],
        "age": [-5, -5, 3, 3],
        "transplant": [0, 1, 0, 1],
        "pid": ["b", "b", "a", "a"],
    }
    return {**newdata, **changes}


def _strata_data():
    # set.seed(1); data.frame(time = c(1:10, 21:30), status = rep(c(1, 0, 1, 1, 0), 4),
    #                         g = rep(1:2, each = 10), x = rnorm(20))
    x = [
        -0.62645381074233242,
        0.18364332422208224,
        -0.83562861241004716,
        1.5952808021377916,
        0.32950777181536051,
        -0.82046838411801526,
        0.48742905242848528,
        0.73832470512921733,
        0.57578135165349231,
        -0.30538838715635602,
        1.511781168450848,
        0.38984323641143109,
        -0.62124058054180376,
        -2.2146998871774999,
        1.1249309181431082,
        -0.044933609015230851,
        -0.016190263098946087,
        0.94383621068529922,
        0.82122119509808855,
        0.59390132121750883,
    ]
    return {
        "time": [*range(1, 11), *range(21, 31)],
        "status": [1, 0, 1, 1, 0] * 4,
        "g": [1] * 10 + [2] * 10,
        "x": x,
    }


# ---------------------------------------------------------------------------
# type= and individual=
# ---------------------------------------------------------------------------

_KP = [0.995814008586017, 0.983222465461149, 0.979007485657402]
_BRESLOW = [0.995820708869676, 0.983347601004328, 0.97914552919369]
_EFRON = [0.995820708869676, 0.983268756915752, 0.979067022024147]


@pytest.mark.parametrize(
    ("type_", "surv", "std_err"),
    [
        ("kalbfleisch-prentice", _KP, [0.00419811047562559, 0.00845113015015369]),
        ("kaplan-meier", _KP, [0.00419811047562559, 0.00845113015015369]),
        ("aalen", _BRESLOW, [0.00418931561111944, 0.00840665195863542]),
        ("breslow", _BRESLOW, [0.00418931561111944, 0.00840665195863542]),
        ("tsiatis", _BRESLOW, [0.00418931561111944, 0.00840665195863542]),
        ("efron", _EFRON, [0.00418931561111945, 0.0084465844413705]),
        ("fleming-harrington", _EFRON, [0.00418931561111945, 0.0084465844413705]),
    ],
)
def test_type_picks_rs_curve(lung_fit, type_, surv, std_err):
    # survfit(coxph(Surv(time, status) ~ age + sex, lung), type = type_)
    curve = r.survfit(lung_fit, type=type_)
    assert curve.surv[:3] == approx(surv)
    assert curve.std_err[:2] == approx(std_err)
    assert lung_fit.survfit(type=type_).surv[:3] == approx(surv)


def test_type_with_stype_or_ctype_is_ignored_with_rs_warning(lung_fit):
    with pytest.warns(RuntimeWarning, match="type argument ignored") as caught:
        kp = r.survfit(lung_fit, type="aalen", stype=1)
    assert caught[0].filename == __file__
    assert kp.surv[:3] == approx(_KP)
    with pytest.warns(RuntimeWarning, match="type argument ignored"):
        breslow = r.survfit(lung_fit, type="kalbfleisch-prentice", ctype=1)
    assert breslow.surv[:3] == approx(_BRESLOW)
    with pytest.raises(ValueError, match="'type' should be one of"):
        r.survfit(lung_fit, type="bogus")
    assert r.survfit(lung_fit, type="kap").surv[:3] == approx(_KP)


def test_type_sets_ctype_of_a_breslow_fit(lung):
    fit = r.coxph("Surv(time, status) ~ age + sex", lung, ties="breslow")
    assert r.survfit(fit).surv[:3] == approx(
        [0.995820158292352, 0.98334540329276, 0.979142823072368]
    )
    assert r.survfit(fit, type="efron").surv[:3] == approx(
        [0.995820158292352, 0.983266590080477, 0.979064346688677]
    )


def test_individual_is_accepted_with_rs_warning(lung_fit, heart_fit):
    newdata = {"age": [60], "sex": [1]}
    with pytest.warns(RuntimeWarning, match="the `id' option supersedes `individual'"):
        curve = r.survfit(lung_fit, newdata, individual=False)
    assert curve.surv[:3] == approx([0.995093079780872, 0.980377439641655, 0.975458714919046])
    # survfit(fit, newdata = nd[3:4, ], individual = TRUE): one subject, no names
    subject = {key: values[2:] for key, values in _heart_newdata().items()}
    with pytest.warns(RuntimeWarning, match="supersedes"):
        curve = r.survfit(heart_fit, subject, individual=True)
    assert curve.strata is None
    assert curve.surv[:3] == approx([0.988905694053077, 0.955601757977624, 0.922368969354276])
    with pytest.raises(ValueError, match="individual is only used with a fitted Cox model"):
        r.survfit("Surv(time, status) ~ 1", datasets.load_lung(), individual=True)


# ---------------------------------------------------------------------------
# start.time
# ---------------------------------------------------------------------------


def test_start_time_builds_the_curve_from_the_rows_at_risk(lung_fit):
    # survfit(fit, start.time = 100)
    curve = r.survfit(lung_fit, start_time=100)
    assert curve.start_time == 100.0
    assert curve.n == [196]
    assert len(curve.time) == 164
    assert curve.time[:3] == [105.0, 107.0, 110.0]
    assert curve.n_risk[:3] == [196.0, 194.0, 192.0]
    assert curve.surv[:3] == approx([0.995026715313351, 0.985024763886337, 0.980024944029252])
    assert curve.cumhaz[:3] == approx([0.00498569262283097, 0.0150884971247201, 0.0201772545503708])
    assert curve.std_err[:3] == approx(
        [0.00498652602650291, 0.0087159245458067, 0.0100957264656873]
    )
    assert curve.lower[:3] == approx([0.985349277672496, 0.968340598802159, 0.960823533542582])
    assert curve.upper[:3] == approx([1.0, 1.0, 0.999610081758028])
    assert r.survfit(lung_fit, **{"start.time": 100}, stype=1).surv[:3] == approx(
        [0.995011625582318, 0.984985370091693, 0.979969727221027]
    )
    newdata = r.survfit(lung_fit, {"age": [50, 60], "sex": [1, 2]}, start_time=100)
    assert newdata.surv[0] == approx([0.995074030974004, 0.996501018866574])
    assert newdata.surv[2] == approx([0.980213558640518, 0.985914819509285])
    assert newdata.std_err[2] == approx([0.010350446313988, 0.00729606222312575])
    # the cached curve of the whole fit is untouched
    assert r.survfit(lung_fit).time[0] == 5.0


def test_start_time_reads_the_stop_time_of_counting_data(heart_fit):
    curve = r.survfit(heart_fit, start_time=100)
    assert curve.n == [54]
    assert len(curve.time) == 52
    assert curve.time[:3] == [100.0, 102.0, 109.0]
    assert curve.n_risk[:3] == [50.0, 49.0, 48.0]
    assert curve.surv[:3] == approx([0.979448223850899, 0.958850707206296, 0.958850707206296])
    assert curve.std_err[:3] == approx([0.0214277667125728, 0.0315718279279994, 0.0315718279279994])
    # survfit(fit, newdata = nd, id = nd$pid, start.time = 100), numeric ids 1, 2
    subjects = r.survfit(heart_fit, _heart_newdata(pid=[1, 1, 2, 2]), id="pid", start_time=100)
    assert subjects.strata == {"1": 26, "2": 43}
    assert subjects.n == [54, 54]
    assert subjects.time[:3] == [100.0, 102.0, 109.0]
    assert subjects.surv[:3] == approx([0.981041874972634, 0.962010279902039, 0.962010279902039])
    assert subjects.std_err[:3] == approx(
        [0.0191920562348243, 0.0275374762167643, 0.0275374762167643]
    )


def test_start_time_keeps_a_stratum_it_empties():
    fit = r.coxph("Surv(time, status) ~ x + strata(g)", _strata_data())
    curve = r.survfit(fit, start_time=15)
    assert curve.strata == {"g=1": 0, "g=2": 10}
    assert curve.n == [0, 10]
    assert curve.time == [float(t) for t in range(21, 31)]
    assert curve.surv == approx(
        [0.911703185270467, 0.911703185270467, 0.818916920480482, 0.721769214450701]
        + [0.721769214450701, 0.579122152428905, 0.579122152428905, 0.378441860225577]
        + [0.203946105189285, 0.203946105189285]
    )
    assert curve.std_err[-1] == approx(0.819221646496032)
    assert r.survfit(fit, start_time=15, stype=1).surv[-1] == approx(0.139802086569338)
    uncensored = r.survfit(fit, start_time=15, censor=False)
    assert uncensored.strata == {"g=1": 0, "g=2": 6}
    assert uncensored.time == [21.0, 23.0, 24.0, 26.0, 28.0, 29.0]
    every = r.survfit(fit, {"x": [0.5, -1]}, start_time=15)
    assert every.strata == {"g=1": 0, "g=2": 10}
    assert every.surv[0] == approx([0.921861441044301, 0.859787664575014])
    assert every.surv[-1] == approx([0.246763315487031, 0.0744024345015147])
    # a newdata row in the emptied stratum gets an empty curve, and the row in g=2 its own
    # curve (x = -1, the second column above); R's split() drops the empty stratum here and
    # hands the second row the first row's column (0.921861441044301, ...)
    found = r.survfit(fit, {"x": [0.5, -1], "g": [1, 2]}, start_time=15)
    assert found.strata == {"1": 0, "2": 10}
    assert found.n == [0, 10]
    assert found.surv == [row[1] for row in every.surv]


def test_penalized_fits_take_start_time_and_id():
    fit = r.coxph("Surv(start, stop, event) ~ pspline(age) + transplant", datasets.load_heart())
    assert r.survfit(fit, start_time=100).surv[:3] == approx(
        [0.979207892618949, 0.958406550548767, 0.958406550548767]
    )
    subjects = r.survfit(fit, _heart_newdata(pid=[1, 1, 2, 2]), id="pid")
    assert subjects.strata == {"1": 85, "2": 102}
    assert subjects.surv[:3] == approx([0.992080652630845, 0.968123794410737, 0.94398730070863])


def test_start_time_errors_as_r(lung_fit):
    with pytest.raises(ValueError, match="start.time argument has removed all endpoints"):
        r.survfit(lung_fit, start_time=1000)
    with pytest.raises(ValueError, match="removed all endpoints"):
        r.survfit(r.coxph("Surv(time, status) ~ x + strata(g)", _strata_data()), start_time=30)
    for value in ("a", [1, 2]):
        with pytest.raises(ValueError, match="start.time must be a single numeric value"):
            lung_fit.survfit(start_time=value)


# ---------------------------------------------------------------------------
# interactions without their lower-order terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula",
    [
        "age:sex",
        "age + age:sex",
        "sex + age:sex",
        "sex/age",
        "age:sex + strata(ph.ecog)",
        "age + sex:ph.ecog",
        "age*sex + age:ph.ecog",
        "age + factor(sex) + sex:ph.ecog",
    ],
)
def test_interaction_without_its_lower_order_terms_is_refused(lung, formula):
    fit = r.coxph(f"Surv(time, status) ~ {formula}", lung)
    message = "not able to create a curve for models that contain an interaction without"
    for newdata in (None, {"age": [60], "sex": [1], "ph.ecog": [1]}):
        with pytest.raises(ValueError, match=message):
            r.survfit(fit, newdata)
    # basehaz and survexp(ratetable = fit) build the same curves
    with pytest.raises(ValueError, match=message):
        r.basehaz(fit)
    with pytest.raises(ValueError, match=message):
        r.survexp("~ 1", lung, ratetable=fit)


@pytest.mark.parametrize(
    ("formula", "at_means", "at_newdata"),
    [
        ("age*sex", 0.9958231307313569, 0.9951672901084396),
        ("age + sex + age:sex", 0.9958231307313569, 0.9951672901084396),
        ("age*sex + strata(ph.ecog)", 0.9846822, 0.9901979),
        ("age*sex*ph.ecog", 0.9961871, 0.9953258),
    ],
)
def test_interaction_with_its_margins_gives_curves(lung, formula, at_means, at_newdata):
    fit = r.coxph(f"Surv(time, status) ~ {formula}", lung)
    with pytest.warns(RuntimeWarning, match="the model contains interactions") as caught:
        curve = r.survfit(fit)
    assert caught[0].filename == __file__
    assert curve.surv[0] == pytest.approx(at_means, rel=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        newdata = r.survfit(fit, {"age": [60], "sex": [1], "ph.ecog": [1]})
    assert newdata.surv[0] == pytest.approx(at_newdata, rel=1e-7)


# ---------------------------------------------------------------------------
# newdata with missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, math.nan])
def test_incomplete_newdata_rows_are_left_out(lung_fit, missing):
    # R: survfit(fit, newdata = data.frame(age = c(50, NA, 60), sex = c(1, 2, 1))) drops row 2
    newdata = {"age": [50, missing, 60], "sex": [1, 2, 1]}
    curve = r.survfit(lung_fit, newdata)
    assert curve.newdata is newdata
    assert (len(curve.surv), curve.ncurve) == (186, 2)
    assert curve.surv[0] == approx([0.995860486224548, 0.995093079780872])
    assert curve.std_err[0] == approx([0.00418849710737933, 0.0049280485101332])
    assert curve.surv[99] == approx([0.551371678359325, 0.493621308396549])
    hazard = r.basehaz(lung_fit, newdata)
    assert hazard.hazard[0] == approx([0.00414810528056652, 0.0049189986803758])
    single = r.survfit(lung_fit, {"age": [50, missing], "sex": [1, 2]})
    assert single.surv[:3] == approx([0.995860486224548, 0.983427001357142, 0.979264585784987])
    with pytest.raises(ValueError, match="all rows of newdata have missing values"):
        r.survfit(lung_fit, {"age": [missing, missing], "sex": [1, 2]})


def test_newdata_rows_missing_a_strata_offset_or_transformed_value(lung):
    stratified = r.coxph("Surv(time, status) ~ age + strata(sex)", lung)
    curve = r.survfit(stratified, {"age": [50, 60, 70], "sex": [1, None, 2]})
    assert curve.strata == {"1": 119, "3": 87}
    assert [curve.surv[0], curve.surv[119]] == approx([0.982676086421029, 0.987368363848863])
    every = r.survfit(stratified, {"age": [50, None, 70]})
    assert every.strata == {"sex=1": 119, "sex=2": 87}
    assert every.ncurve == 2
    offset = r.coxph("Surv(time, status) ~ age + offset(wt.loss/100)", lung)
    curve = r.survfit(offset, {"age": [50, 60, 70], "wt.loss": [5, None, -2]})
    assert (len(curve.surv), curve.ncurve) == (179, 2)
    assert curve.surv[0] == approx([0.996718777244269, 0.995255629692632])
    logged = r.coxph("Surv(time, status) ~ log(age) + sex", lung)
    with pytest.warns(UserWarning, match="NaNs produced"):
        curve = r.survfit(logged, {"age": [50, -1, 60], "sex": [1, 2, 1]})
    assert curve.surv[0] == approx([0.995858312428209, 0.995026804966051])


# ---------------------------------------------------------------------------
# curve names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pid", "strata"),
    [
        (["b", "b", "a", "a"], {"b": 85, "a": 102}),
        ([10, 10, 20, 20], {"10": 85, "20": 102}),
        ([10.0, 10.0, 20.0, 20.0], {"10": 85, "20": 102}),
        ([10.5, 10.5, 20, 20], {"10.5": 85, "20": 102}),
        ([1, 1, 1, 1], None),
    ],
)
def test_id_curves_are_named_by_the_id_values(heart_fit, pid, strata):
    curve = r.survfit(heart_fit, _heart_newdata(pid=pid), id="pid")
    assert curve.strata == strata
    assert curve.surv[:3] == approx([0.991314038209925, 0.965110733471733, 0.938764141475753])


@pytest.mark.parametrize(
    ("pid", "strata"),
    [
        ([0.1 + 0.2, 0.1 + 0.2, 0.3, 0.3], {"0.3": 85, "0.3.1": 102}),
        ([1, 1, "1", "1"], {"1": 85, "1.1": 102}),
    ],
)
def test_id_values_that_print_alike_keep_a_curve_each(heart_fit, pid, strata):
    # R: pid = c(0.1 + 0.2, 0.1 + 0.2, 0.3, 0.3) gives two curves, both named "0.3"
    curve = r.survfit(heart_fit, _heart_newdata(pid=pid), id="pid")
    assert curve.strata == strata
    assert curve.n == [172, 172]
    assert sum(curve.strata.values()) == len(curve.time)
    assert [curve.surv[0], curve.surv[85]] == approx([0.991314038209925, 0.988905694053077])


def test_id_curves_leave_out_rows_with_a_missing_value(heart_fit):
    for newdata in (
        _heart_newdata(pid=["b", None, "a", "a"]),
        _heart_newdata(age=[-5, None, 3, 3]),
    ):
        curve = r.survfit(heart_fit, newdata, id="pid")
        assert curve.strata == {"b": 39, "a": 102}
        assert curve.n == [172, 172]
        assert curve.surv[:3] == approx([0.991314038209925, 0.965110733471733, 0.938764141475753])


def test_stratified_newdata_curves_are_named_by_the_row_names(lung):
    pd = pytest.importorskip("pandas")
    fit = r.coxph("Surv(time, status) ~ age + strata(sex)", lung)
    named = pd.DataFrame({"age": [50, 70, 60], "sex": [2, 1, 2]}, index=["x", "y", "z"])
    assert r.survfit(fit, named).strata == {"x": 87, "y": 119, "z": 87}
    automatic = pd.DataFrame({"age": [50, 70, 60], "sex": [2, 1, 2]})
    assert r.survfit(fit, automatic).strata == {"1": 87, "2": 119, "3": 87}
    events = pd.DataFrame({"age": [50, 60], "sex": [1, 2]}, index=["p", "q"])
    assert r.survfit(fit, events, censor=False).strata == {"p": 99, "q": 51}
    subset = pd.DataFrame({"age": [50, None, 60], "sex": [2, 1, 2]}, index=[5, 6, 7])
    assert r.survfit(fit, subset).strata == {"5": 87, "7": 87}


def test_an_index_that_cannot_be_rs_row_names_gives_the_row_numbers(lung):
    # R: rbind(data.frame(age = c(50, 60), sex = 1:2), data.frame(age = c(70, 80), sex = 1:2))
    # numbers the rows 1..4, where pd.concat repeats the index 0, 1; labels that repeat
    # under as.character or are missing cannot be R row names either
    pd = pytest.importorskip("pandas")
    fit = r.coxph("Surv(time, status) ~ age + strata(sex)", lung)
    newdata = pd.concat(
        [
            pd.DataFrame({"age": [50, 60], "sex": [1, 2]}),
            pd.DataFrame({"age": [70, 80], "sex": [1, 2]}),
        ]
    )
    for index in (None, [1, 2, "1", "2"], [0.1 + 0.2, 0.3, 5, 6], ["a", None, "b", "c"]):
        frame = newdata if index is None else newdata.set_axis(pd.Index(index, dtype=object))
        curve = r.survfit(fit, frame)
        assert curve.strata == {"1": 119, "2": 87, "3": 119, "4": 87}
        assert curve.n == [138, 90, 138, 90]
        assert sum(curve.strata.values()) == len(curve.time) == 412
        assert [curve.surv[row] for row in (0, 119, 206, 325)] == approx(
            [0.982676086421029, 0.989248906353565, 0.976119901176607, 0.985161358364436]
        )
        assert r.survfit(fit, frame, censor=False).strata == {"1": 99, "2": 51, "3": 99, "4": 51}


# ---------------------------------------------------------------------------
# confidence limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conf_type", "lower_first", "upper_100", "lower_last"),
    [
        (
            "log",
            [0.987718630476997, 0.991277903285601, 0.98285527624013],
            [0.67432962944107, 0.746457273563999, 0.535477134744255],
            [0.0204669031590677, 0.0658211279880695, 0.00536157735877467],
        ),
        (
            "log-log",
            [0.970430878417979, 0.978998197849288, 0.959248211541189],
            [0.654071289720423, 0.733040590246919, 0.522336861985058],
            [0.0162981271190139, 0.0569855034865478, 0.004298992304298],
        ),
        (
            "logit",
            [0.970684137843333, 0.979126309085245, 0.959723277059395],
            [0.658122263769626, 0.735036668916016, 0.52621810082315],
            [0.0199407753930985, 0.063715156775195, 0.00530614257241901],
        ),
        (
            "arcsin",
            [0.983692988040375, 0.988423852463425, 0.9772959764672],
            [0.660175136397012, 0.737703702241873, 0.52558777946064],
            [0.0125966260892016, 0.0540349489889424, 0.0025673912582926],
        ),
        (
            "plain",
            [0.987685165269576, 0.991261072659632, 0.982790238321204],
            [0.662368267752421, 0.740656745163708, 0.524962302634194],
            [0.0, 0.0383253057670392, 0.0],
        ),
    ],
)
def test_newdata_confidence_limits_match_r(lung_fit, conf_type, lower_first, upper_100, lower_last):
    newdata = {"age": [50, 60, 70], "sex": [1, 2, 1]}
    curve = r.survfit(lung_fit, newdata, conf_type=conf_type)
    assert curve.lower[0] == approx(lower_first)
    assert curve.upper[99] == approx(upper_100)
    assert curve.lower[185] == approx(lower_last)


def test_survfit_reads_the_curves_once_and_limits_them_in_one_call(monkeypatch, lung):
    # the curve getters convert the whole ntime x m block, and R calls survfit_confint once on
    # the whole matrix: one read of each block per curve, one call for all the columns
    fit = r.coxph("Surv(time, status) ~ age + strata(sex)", lung)
    newdata = {"age": [50, 60, 70]}
    expected = r.survfit(fit, newdata)
    reads = []

    class Curve:
        def __init__(self, curve):
            self._curve = curve
            self.reads = {"surv": 0, "cumhaz": 0, "std_err": 0}
            reads.append(self.reads)

        def __getattr__(self, name):
            if name in self.reads:
                self.reads[name] += 1
            return getattr(self._curve, name)

    class Engine:
        def __init__(self, engine):
            self._engine = engine

        def __getattr__(self, name):
            return getattr(self._engine, name)

        def survfit(self, **kwargs):
            return [Curve(curve) for curve in self._engine.survfit(**kwargs)]

    calls = []
    confint = _coxph._core.survfit_confint

    def counted(*args):
        calls.append(len(args[0]))
        return confint(*args)

    monkeypatch.setattr(_coxph._core, "survfit_confint", counted)
    curve = r.survfit(dataclasses.replace(fit, fit=Engine(fit.fit)), newdata)
    assert (curve.lower, curve.upper) == (expected.lower, expected.upper)
    assert reads == [{"surv": 1, "cumhaz": 1, "std_err": 1}] * 2
    assert calls == [len(expected.time) * 3]
