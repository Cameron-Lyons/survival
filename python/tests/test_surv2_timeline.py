"""Timeline data: ``Surv2`` responses in coxph/survfit/survcheck/survSplit and
``fromtimeline``, against R's ``surv2counting`` (R/fromtimeline.R).

Reference values come from R 4.5.3 / survival 3.8-12 (the ``Rscript`` calls are quoted
next to the assertions), which keeps a missing ``Surv2`` status and codes a missing
factor outcome as censored without 3.8-11's level shift.  In R::

    d <- data.frame(id = c(1,1,1,2,2,3,3,3,4,4), t = c(0,4,7,0,5,0,6,9,0,9),
                    s = c(0,1,1,0,0,0,0,1,0,1), x = c(1,NA,2,3,NA,0,NA,1,2,NA))
    dm <- d
    dm$st <- factor(c("none","a","b","none","none","none",NA,"a","none","b"),
                    levels = c("none","a","b"))
    dm$st2 <- factor(c("a","a","b","a","b","b",NA,"a","a","a"),
                     levels = c("none","a","b"))
"""

import importlib
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_coerce = importlib.import_module("survival.r._coerce")

NA = math.nan


def _timeline():
    return {
        "id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "t": [0, 4, 7, 0, 5, 0, 6, 9, 0, 9],
        "s": [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        "x": [1, NA, 2, 3, NA, 0, NA, 1, 2, NA],
    }


def _multistate():
    data = _timeline()
    levels = ["none", "a", "b"]
    data["st"] = r_coerce._r_factor(
        ["none", "a", "b", "none", "none", "none", None, "a", "none", "b"], levels
    )
    data["st2"] = r_coerce._r_factor(["a", "a", "b", "a", "b", "b", None, "a", "a", "a"], levels)
    return data


def _same(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, float) and math.isnan(e):
            assert math.isnan(a)
        else:
            assert a == pytest.approx(e, rel=1e-12, abs=1e-14)


# --- coxph -----------------------------------------------------------------


def test_coxph_converts_surv2_with_last_value_carried_forward():
    # f <- coxph(Surv2(t, s) ~ x, d, id = id)
    fit = r.coxph("Surv2(t, s) ~ x", _timeline(), id="id")
    assert fit.formula == "Surv2(t, s) ~ x"
    assert fit.coefficients == pytest.approx([-0.128265695142221], rel=1e-12)
    assert fit.loglik == pytest.approx([-3.17805383034795, -3.14606337475372], rel=1e-12)
    assert fit.var[0][0] == pytest.approx(0.0667096338468127, rel=1e-12)
    assert fit.n == 6
    # f$y: (0,4] (4,7] (0,5+] (0,6+] (6,9] (0,9]
    assert fit.y.type == "counting"
    assert list(fit.y.start) == [0, 4, 0, 0, 6, 0]
    assert list(fit.y.time) == [4, 7, 5, 6, 9, 9]
    assert list(fit.y.event) == [1, 1, 0, 0, 1, 1]
    # R: residuals(f); then type = "score"; predict(f, type = "expected")
    _same(
        r.residuals(fit),
        [
            0.736162053472385,
            0.668487186614905,
            -0.204139520781859,
            -0.299945493825804,
            -0.504448886608770,
            -0.396115338870856,
        ],
    )
    _same(
        r.residuals(fit, type="score"),
        [
            -0.2505973564571349,
            0.0570064649540151,
            -0.3387877875455851,
            0.4020501153278631,
            0.4560362766213351,
            -0.3257077129004957,
        ],
    )
    _same(
        r.predict(fit, type="expected"),
        [
            0.263837946527615,
            0.331512813385095,
            0.204139520781859,
            0.299945493825804,
            1.504448886608770,
            1.396115338870856,
        ],
    )
    # sf <- survfit(f); sf$time; sf$surv
    curve = r.survfit(fit)
    assert curve.time == [4, 5, 6, 7, 9]
    _same(
        curve.surv,
        [
            0.772396252469229,
            0.772396252469229,
            0.772396252469229,
            0.558355929080741,
            0.130154026294690,
        ],
    )


def test_coxph_model_frame_names_the_surv2_response():
    # names(model.frame(f)); model.frame(f)$x
    # (the plain frame splits the response into its columns)
    fit = r.coxph("Surv2(t, s) ~ x", _timeline(), id="id")
    frame = r.model_frame(fit)
    assert list(frame) == ["start", "stop", "status", "x", "(id)"]
    assert list(frame["x"]) == [1, 1, 3, 0, 0, 2]
    assert list(frame["(id)"]) == [1, 1, 2, 3, 3, 4]
    # names(coxph(Surv2(t, s) ~ x, d, id = id, model = TRUE)$model)
    kept = r.coxph("Surv2(t, s) ~ x", _timeline(), id="id", model=True)
    assert list(kept.model) == ["Surv2(t, s)", "x", "(id)"]


def test_coxph_applies_na_action_after_the_conversion():
    # d$s[3] <- NA: the interval (4, 7] of id 1 has a missing status, and x is
    # carried forward before na.exclude sees it
    # f <- coxph(Surv2(t, s) ~ x, d, id = id, na.action = na.exclude)
    data = _timeline()
    data["s"][2] = None
    fit = r.coxph("Surv2(t, s) ~ x", data, id="id", na_action="na.exclude")
    assert fit.coefficients == pytest.approx([-0.155004981171472], rel=1e-12)
    assert fit.n == 5
    assert fit.na_action.rows == (2,)
    assert fit.na_action.kind == "exclude"
    _same(
        r.residuals(fit),
        [
            0.733866617676929,
            NA,
            -0.195192774627484,
            -0.310754310502019,
            -0.153775385650179,
            -0.074144146897247,
        ],
    )
    # survfit(Surv2(t, s) ~ 1, d, id = id)
    curve = r.survfit("Surv2(t, s) ~ 1", data, id="id")
    assert curve.time == [4, 5, 9]
    _same(curve.surv, [0.75, 0.75, 0.0])


def test_coxph_does_not_carry_its_arguments_forward():
    # the (weights) and (cluster) columns, and a cluster() term, which coxph.R makes its
    # cluster argument, keep a missing value, so na.omit drops that interval
    # d$w <- c(1,NA,1,1,3,1,2,1,1,1); d$z <- c(1,NA,3,1,2,3,1,2,3,1)
    data = _timeline()
    data["w"] = [1, NA, 1, 1, 3, 1, 2, 1, 1, 1]
    data["z"] = [1, NA, 3, 1, 2, 3, 1, 2, 3, 1]
    # coxph(Surv2(t, s) ~ x, d, id = id, weights = w)
    fit = r.coxph("Surv2(t, s) ~ x", data, id="id", weights="w")
    assert fit.coefficients == pytest.approx([-0.132139565424643], rel=1e-12)
    assert fit.var[0][0] == pytest.approx(0.274951131772283, rel=1e-12)
    assert fit.na_action.rows == (2,)
    # coxph(Surv2(t, s) ~ x, d, id = id, cluster = z) and
    # coxph(Surv2(t, s) ~ x + cluster(z), d, id = id)
    for fit in (
        r.coxph("Surv2(t, s) ~ x", data, id="id", cluster="z"),
        r.coxph("Surv2(t, s) ~ x + cluster(z)", data, id="id"),
    ):
        assert fit.coefficients == pytest.approx([-0.155004981171472], rel=1e-12)
        assert fit.var[0][0] == pytest.approx(0.0359344466438995, rel=1e-12)
        assert fit.na_action.rows == (2,)


def test_coxph_one_interval_per_subject_is_right_censored():
    # d1 <- data.frame(id = c(1,1,2,2,3,3), t = c(0,5,0,3,0,4), s = c(0,1,0,0,0,1),
    #                  x = c(1,NA,2,NA,NA,3))
    # coxph(Surv2(t, s) ~ x, d1, id = id)$na.action: 3 (row name 5)
    data = {"id": [1, 1, 2, 2, 3, 3], "t": [0, 5, 0, 3, 0, 4], "s": [0, 1, 0, 0, 0, 1]}
    data["x"] = [1, NA, 2, NA, NA, 3]
    with pytest.warns(RuntimeWarning, match="Ran out of iterations"):
        fit = r.coxph("Surv2(t, s) ~ x", data, id="id")
    assert fit.y.type == "right"
    assert fit.na_action.rows == (3,)
    assert fit.coefficients == [0.0]


def test_surv2_response_arguments_match_by_name():
    # coxph(survival::Surv2(time = t, event = s) ~ x, d, id = id)
    fit = r.coxph("survival::Surv2(event = s, time = t) ~ x", _timeline(), id="id")
    assert fit.coefficients == pytest.approx([-0.128265695142221], rel=1e-12)


def test_surv2_requires_an_id_and_a_timeline_fitter():
    with pytest.raises(ValueError, match="id statement is required"):
        r.coxph("Surv2(t, s) ~ x", _timeline())
    with pytest.raises(ValueError, match="response must be a survival object"):
        r.survreg("Surv2(t, s) ~ x", _timeline())
    data = _timeline()
    data["t"][1] = NA
    with pytest.raises(ValueError, match="id and time cannot be missing"):
        r.coxph("Surv2(t, s) ~ x", data, id="id")
    with pytest.raises(ValueError, match="invalid value for repeated option"):
        r.coxph("Surv2(t, s, repeated = 'often') ~ x", _timeline(), id="id")
    with pytest.raises(ValueError, match="survival object"):
        r.model_frame("Surv2(t, s) ~ x", _timeline())
    with pytest.raises(TypeError, match="'timeline'"):
        r.model_frame("Surv2(t, s) ~ x", _timeline(), timeline=True)


# --- survfit and survcheck ---------------------------------------------------


def test_survfit_converts_surv2():
    # sf <- survfit(Surv2(t, s) ~ 1, d, id = id)
    curve = r.survfit("Surv2(t, s) ~ 1", _timeline(), id="id")
    assert curve.time == [4, 5, 7, 9]
    _same(curve.surv, [0.75, 0.75, 0.5, 0.0])
    assert curve.n_risk == [4, 4, 3, 2]
    assert curve.n_event == [1, 0, 1, 2]


def test_survfit_multistate_timeline():
    # sf <- survfit(Surv2(t, st) ~ 1, dm, id = id): no initial states
    curve = r.survfit("Surv2(t, st) ~ 1", _multistate(), id="id")
    assert curve.states == ["(s0)", "a", "b"]
    assert curve.time == [4, 5, 7, 9]
    assert curve.pstate == [
        [0.75, 0.25, 0.0],
        [0.75, 0.25, 0.0],
        [0.75, 0.0, 0.25],
        [0.0, 0.375, 0.625],
    ]
    # sf2 <- survfit(Surv2(t, st2) ~ 1, dm, id = id): everyone starts in a state,
    # which becomes the istate
    curve = r.survfit("Surv2(t, st2) ~ 1", _multistate(), id="id")
    assert curve.states == ["a", "b"]
    assert curve.time == [5, 7, 9]
    assert curve.pstate == [[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]]
    assert curve.p0 == [[0.75, 0.25]]


def test_survcheck_converts_surv2():
    # sc <- survcheck(Surv2(t, s) ~ 1, d, id = id)
    check = r.survcheck("Surv2(t, s) ~ 1", _timeline(), id="id")
    assert check.transitions.from_states == ["(s0)", "event"]
    assert check.transitions.counts == [[3, 1], [1, 0]]
    assert (check.n_id, check.n_observations, check.n_transitions) == (4, 6, 4)
    flag = check.flag
    assert (flag.overlap, flag.gap, flag.jump, flag.teleport, flag.duplicate) == (0, 0, 0, 0, 0)
    # survcheck(Surv2(t, st2) ~ 1, dm, id = id)$transitions
    check = r.survcheck("Surv2(t, st2) ~ 1", _multistate(), id="id")
    assert check.transitions.from_states == ["a", "b"]
    assert check.transitions.counts == [[0, 2, 1], [1, 0, 0]]


def test_istate_argument_must_agree_with_the_timeline():
    # The intended semantics, not R's: survival 3.8-12's surv2counting compares vectors
    # of different lengths (for a character istate, istate2[match(levels(istate),
    # states)] with as.numeric(istate) at every input row), so R's
    # survfit(Surv2(t, st2) ~ 1, dm, id = id, istate = cur) stops with "istate argument
    # does not agree with initial Surv2 values" for the first cur below and accepts
    # cur <- rep("b", 10) (times 5 7 9).
    data = _multistate()
    # the state each subject is in at the start of every row
    data["cur"] = ["a", "a", "b", "a", "b", "b", "b", "a", "a", "a"]
    curve = r.survfit("Surv2(t, st2) ~ 1", data, id="id", istate="cur")
    assert curve.time == [5, 7, 9]
    data["cur"] = ["b"] * 10
    with pytest.raises(ValueError, match="istate argument does not agree"):
        r.survfit("Surv2(t, st2) ~ 1", data, id="id", istate="cur")


# --- survSplit ---------------------------------------------------------------


def test_survsplit_keeps_the_timeline_form():
    # survSplit(Surv2(t, s) ~ ., d, cut = c(3, 8), id = id)
    split = r.survSplit("Surv2(t, s) ~ .", _timeline(), cut=[3, 8], id="id")
    assert list(split) == ["id", "t", "s", "x"]
    assert split["id"] == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    assert split["t"] == [0, 3, 4, 7, 0, 3, 5, 0, 3, 6, 8, 9, 0, 3, 8, 9]
    assert split["s"] == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    _same(split["x"], [1, 1, NA, 2, 3, 3, NA, 0, 0, NA, NA, 1, 2, 2, 2, NA])
    # survSplit(Surv2(t, s) ~ x, d, cut = c(3, 8), id = id): x (id) t s
    split = r.survSplit("Surv2(t, s) ~ x", _timeline(), cut=[3, 8], id="id")
    assert list(split) == ["x", "(id)", "t", "s"]
    assert split["(id)"] == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    assert split["s"] == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="an id statement is required"):
        r.survSplit("Surv2(t, s) ~ x", _timeline(), cut=[3, 8])


# --- fromtimeline ------------------------------------------------------------


def test_fromtimeline_carries_covariates_forward():
    # fromtimeline(Surv2(t, s) ~ x, d, id = id)
    expected = {
        "x": [1, 1, 3, 0, 0, 2],
        "t1": [0, 4, 0, 0, 6, 0],
        "t2": [4, 7, 5, 6, 9, 9],
        "s": [1, 1, 0, 0, 1, 1],
    }
    assert r.fromtimeline("Surv2(t, s) ~ x", _timeline(), id="id") == expected
    # fromtimeline(Surv(t, s) ~ x, d, id = id)
    assert r.fromtimeline("Surv(t, s) ~ x", _timeline(), id="id") == expected
    # fromtimeline(Surv2(t, s) ~ x, d, id = id, lvcf = FALSE)
    plain = r.fromtimeline("Surv2(t, s) ~ x", _timeline(), id="id", lvcf=False)
    _same(plain["x"], [1, NA, 3, 0, NA, 2])
    # fromtimeline(Surv2(t, s) ~ x, d, id = id, subset = id != 2)
    subset = [value != 2 for value in _timeline()["id"]]
    kept = r.fromtimeline("Surv2(t, s) ~ x", _timeline(), id="id", subset=subset)
    assert kept == {
        "x": [1, 1, 0, 0, 2],
        "t1": [0, 4, 0, 6, 0],
        "t2": [4, 7, 6, 9, 9],
        "s": [1, 1, 0, 1, 1],
    }


def test_fromtimeline_carries_factor_and_character_covariates():
    # d$s[3] <- NA; d$g <- c("a","a","a","b","b","a","a","a","b","b")
    # d$z <- factor(c("u",NA,"v","v",NA,"u",NA,NA,"v","u"))
    # fromtimeline(Surv2(t, s) ~ z + g, d, id = id)
    data = _timeline()
    data["s"][2] = None
    data["g"] = ["a", "a", "a", "b", "b", "a", "a", "a", "b", "b"]
    data["z"] = r_coerce._r_factor(["u", None, "v", "v", None, "u", None, None, "v", "u"], "uv")
    assert r.fromtimeline("Surv2(t, s) ~ z + g", data, id="id") == {
        "z": ["u", "u", "v", "u", "u", "v"],
        "g": ["a", "a", "b", "a", "a", "b"],
        "t1": [0, 4, 0, 0, 6, 0],
        "t2": [4, 7, 5, 6, 9, 9],
        "s": [1, None, 0, 0, 1, 1],
    }


def test_fromtimeline_response_names():
    data = _timeline()
    # fromtimeline(Surv2(t, s) ~ x, d, id = id, yname = c("a", "b", "c"))
    named = r.fromtimeline("Surv2(t, s) ~ x", data, id="id", yname=["a", "b", "c"])
    assert list(named) == ["x", "a", "b", "c"]
    with pytest.raises(ValueError, match="wrong length for yname"):
        r.fromtimeline("Surv2(t, s) ~ x", data, id="id", yname=["a", "b"])
    with pytest.raises(ValueError, match="element of yname conflicts"):
        r.fromtimeline("Surv2(t, s) ~ x", data, id="id", yname=["a", "x", "c"])
    # d2 <- d; d2$t1 <- d2$t; fromtimeline(Surv2(t, s) ~ x + t1, d2, id = id)
    data["t1"] = list(data["t"])
    renamed = r.fromtimeline("Surv2(t, s) ~ x + t1", data, id="id")
    assert list(renamed) == ["x", "t1", "_t1_", "t2", "s"]
    assert renamed["t1"] == [0, 4, 0, 0, 6, 0]
    # one interval per subject: fromtimeline(Surv2(t, s) ~ x, d1, id = id) gives x t s,
    # and yname = c("a", "b") or c("a", "b", "c") names x a b / x a c
    single = {"id": [1, 1, 2, 2, 3, 3], "t": [0, 5, 0, 3, 0, 4], "s": [0, 1, 0, 0, 0, 1]}
    single["x"] = [1, NA, 2, NA, NA, 3]
    result = r.fromtimeline("Surv2(t, s) ~ x", single, id="id")
    assert list(result) == ["x", "t", "s"]
    _same(result["x"], [1, 2, NA])
    assert result["t"] == [5, 3, 4]
    assert result["s"] == [1, 0, 1]
    two = r.fromtimeline("Surv2(t, s) ~ x", single, id="id", yname=["a", "b"])
    assert list(two) == ["x", "a", "b"]
    three = r.fromtimeline("Surv2(t, s) ~ x", single, id="id", yname=["a", "b", "c"])
    assert list(three) == ["x", "a", "c"]


def test_fromtimeline_multistate_and_initial_states():
    data = _multistate()
    # fromtimeline(Surv2(t, st) ~ x, dm, id = id)
    assert r.fromtimeline("Surv2(t, st) ~ x", data, id="id")["st"] == [
        "a",
        "b",
        "censor",
        "censor",
        "a",
        "b",
    ]
    # fromtimeline(Surv2(t, st2) ~ x, dm, id = id): a repeat of the current state is
    # censored, and the initial states give istate
    result = r.fromtimeline("Surv2(t, st2) ~ x", data, id="id")
    assert list(result) == ["x", "istate", "t1", "t2", "st2"]
    assert result["istate"] == ["a", "a", "a", "b", "b", "a"]
    assert result["st2"] == ["censor", "b", "b", "censor", "a", "censor"]
    # fromtimeline(..., repeated = TRUE) counts the repeats; fromtimeline's own
    # repeated argument overrides Surv2(t, st2, repeated = TRUE)
    repeated = r.fromtimeline("Surv2(t, st2) ~ x", data, id="id", repeated=True)
    assert repeated["st2"] == ["a", "b", "b", "censor", "a", "a"]
    own = r.fromtimeline("Surv2(t, st2, repeated = TRUE) ~ x", data, id="id")
    assert own["st2"] == result["st2"]


def test_fromtimeline_checks_its_input():
    data = _timeline()
    with pytest.raises(ValueError, match="the id argument is required"):
        r.fromtimeline("Surv2(t, s) ~ x", data)
    with pytest.raises(ValueError, match="data does not appear to be timeline data"):
        r.fromtimeline("Surv2(t, s) ~ x", data, id=list(range(10)))
    data["t2"] = [value + 1 for value in data["t"]]
    with pytest.raises(ValueError, match="response cannot be of counting process type"):
        r.fromtimeline("Surv(t, t2, s) ~ x", data, id="id")
