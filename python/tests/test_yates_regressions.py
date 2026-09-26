"""Regression tests of ``yates`` against R survival 3.8-12: default (``model = FALSE``) Cox
fits, aliased coefficients and R's estimability check, ``predict = "survival"``, R's
``yates_setup`` errors, and the unused factor levels ``model.frame`` keeps, which give
``coxph``/``survreg``/``concordance`` an aliased column and ``yates`` an NA level."""

import importlib
import math
import warnings

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
_core = importlib.import_module("survival._survival")

CELLTYPES = ["squamous", "smallcell", "adeno", "large"]
SEED = 20240601  # R: set.seed(20240601) before each simulated yates()


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12, nan_ok=True)


def assert_rows_close(actual, expected, rel=1e-8):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == approx(expected_row, rel)


def contrast_rows(result):
    return [(row.name, row.chisq, row.df) for row in result.test]


def _veteran(**extra):
    data = datasets.load_veteran()
    data["celltype"] = RFactor(data["celltype"], CELLTYPES)
    for name, make in extra.items():
        data[name] = make(data)
    return data


# --- model = FALSE -------------------------------------------------------------


def test_yates_rebuilds_the_model_frame_of_a_default_fit():
    # R: yates of celltype on the default (model = FALSE) fit of
    # coxph(Surv(time, status) ~ celltype + karno + trt, veteran)
    fit = r.coxph("Surv(time, status) ~ celltype + karno + trt", _veteran())
    assert fit.model is None
    result = r.yates(fit, "celltype")
    assert result.estimate["pmm"] == approx(
        [0.0, 0.824980187857130, 1.153994413861767, 0.394625463945811]
    )
    assert result.estimate["std"] == approx(
        [0.417980997964406, 0.568033156185915, 0.517272982679515, 0.520452950778976]
    )
    assert contrast_rows(result) == [("global", pytest.approx(17.5010845488993), 3)]
    assert fit.model is None

    # R: the same with a data frame population, karno = c(40, 60, 80) and trt = c(1, 2, 1)
    frame = r.yates(fit, "celltype", population={"karno": [40, 60, 80], "trt": [1, 2, 1]})
    assert frame.estimate["pmm"] == approx(
        [-0.0874072411434579, 0.7375729467136725, 1.0665871727183085, 0.3072182228023532]
    )


# --- aliased coefficients --------------------------------------------------------


def test_yates_drops_aliased_coefficients():
    # R: yates of celltype on coxph(Surv(time, status) ~ celltype + karno + k2, veteran)
    # with k2 = 2 * karno aliased
    fit = r.coxph(
        "Surv(time, status) ~ celltype + karno + k2",
        _veteran(k2=lambda data: [2 * value for value in data["karno"]]),
    )
    assert math.isnan(r.coef(fit)[-1])
    result = r.yates(fit, "celltype")
    assert result.estimate["pmm"] == approx(
        [0.0, 0.715334402378886, 1.157733266785650, 0.325644901681043]
    )
    assert result.estimate["std"] == approx(
        [0.303223819928728, 0.424438625719816, 0.424551925501070, 0.390056779600080]
    )
    assert contrast_rows(result) == [("global", pytest.approx(17.28200970741), 3)]
    assert result.cmat_names == ["celltypesmallcell", "celltypeadeno", "celltypelarge", "karno"]
    assert result.mvar[0] == approx(
        [0.0919446849721694, 0.104121317453997, 0.0931885044374048, 0.0837686945700415]
    )

    # R: the same with predict = "risk" after set.seed(20240601)
    risk = r.yates(fit, "celltype", predict="risk", options={"seed": SEED})
    assert risk.estimate["pmm"] == approx(
        [1.22125757009401, 2.49731342832946, 3.88690958132816, 1.69134830537271]
    )
    assert risk.estimate["std"] == approx(
        [0.0900046102924538, 0.6728611044719064, 1.3030966536844997, 0.5438822386895136]
    )
    assert contrast_rows(risk) == [("celltype", pytest.approx(5.34037071854357), 3)]


def test_yates_gives_na_for_levels_outside_the_row_space_of_the_design():
    # R: coxph(Surv(time, status) ~ celltype + g + karno, veteran) with g the indicator of
    # celltype "large", which aliases g
    fit = r.coxph(
        "Surv(time, status) ~ celltype + g + karno",
        _veteran(g=lambda data: [float(value == "large") for value in data["celltype"]]),
    )
    # yates of celltype: every level is NA (g varies within each level in the data)
    result = r.yates(fit, "celltype")
    assert all(math.isnan(value) for value in result.estimate["pmm"] + result.estimate["std"])
    assert [(row.name, math.isnan(row.chisq), row.df) for row in result.test] == [
        ("global", True, None)
    ]
    assert result.mvar == []
    assert result.cmat == []
    assert result.cmat_names == []
    pairwise = r.yates(fit, "celltype", test="pairwise")
    assert all(math.isnan(row.chisq) and row.df is None for row in pairwise.test)

    # yates of karno at levels 50 and 70
    karno = r.yates(fit, "karno", levels=[50, 70])
    assert karno.estimate["pmm"] == approx([0.809107646088914, 0.187975011482542])
    assert karno.estimate["std"] == approx([0.316047949761309, 0.407669579771154])
    assert contrast_rows(karno) == [("global", pytest.approx(35.9851055623022), 1)]


def test_yates_na_tests_are_those_that_use_a_non_estimable_level():
    # R: coxph(Surv(time, status) ~ celltype * trt + karno) on veteran without the large-cell
    # patients of treatment 2 (trt a factor), which aliases celltypelarge:trt2
    data = _veteran()
    keep = [
        not (cell == "large" and trt == 2)
        for cell, trt in zip(data["celltype"], data["trt"], strict=True)
    ]
    data = {
        name: [value for value, kept in zip(values, keep, strict=True) if kept]
        for name, values in data.items()
    }
    data["celltype"] = RFactor(data["celltype"], CELLTYPES)
    data["trt"] = RFactor([int(value) for value in data["trt"]], [1, 2])
    fit = r.coxph("Surv(time, status) ~ celltype * trt + karno", data)
    assert math.isnan(r.coef(fit)[-1])

    # pairwise yates of celltype
    result = r.yates(fit, "celltype", test="pairwise")
    assert result.estimate["pmm"] == approx(
        [-0.151833090711596, 0.707940261538746, 1.068488110894715, math.nan]
    )
    assert result.estimate["std"] == approx(
        [0.341887720697155, 0.457813009136428, 0.448566350625600, math.nan]
    )
    assert [row.chisq for row in result.test] == approx(
        [10.89743784462409, 15.58538311307057, math.nan, 1.76707005083233, math.nan, math.nan]
    )
    assert [row.df for row in result.test] == [1, 1, None, 1, None, None]
    assert_rows_close(
        result.mvar,
        [
            [0.116887213563496, 0.129323284424246, 0.111274476883248],
            [0.129323284424246, 0.209592751334551, 0.168619692257030],
            [0.111274476883248, 0.168619692257030, 0.201211770913569],
        ],
    )
    assert len(result.cmat) == 4
    assert r.yates(fit, "celltype").test[0].df is None
    trt = r.yates(fit, "trt")
    assert trt.estimate["pmm"] == approx([0.373716617587144, math.nan])

    # the same with predict = "risk" after set.seed(20240601): the non-estimable level keeps
    # its simulated std and mvar entries
    risk = r.yates(fit, "celltype", predict="risk", test="pairwise", options={"seed": SEED})
    assert risk.estimate["pmm"] == approx(
        [1.02852169928166, 2.64992087743419, 3.46085420602894, math.nan]
    )
    assert risk.estimate["std"] == approx(
        [0.205947136237492, 1.035324055566648, 1.406397303534964, 0.601130018165959]
    )
    assert [row.chisq for row in risk.test] == approx(
        [3.00818784899684, 3.43391978379860, math.nan, 0.58579905033240, math.nan, math.nan]
    )
    assert risk.mvar[3] == approx(
        [0.0888443301704846, 0.413179329042424, 0.535740744729067, 0.3613572987402058]
    )


# --- predict = "survival" --------------------------------------------------------


def test_yates_predicts_restricted_mean_survival_as_r():
    # R: yates of celltype with predict = "survival" after set.seed(20240601), on
    # coxph(Surv(time, status) ~ celltype + karno + trt, veteran)
    fit = r.coxph("Surv(time, status) ~ celltype + karno + trt", _veteran())
    result = r.yates(fit, "celltype", predict="survival", options={"seed": SEED})
    assert result.estimate["pmm"] == approx(
        [220.0263455561152, 99.7310793197197, 72.2119953964263, 152.2398949340128]
    )
    assert result.estimate["std"] == approx(
        [9.15867046848566, 29.43161591749605, 21.36650044336478, 44.09350048363265]
    )
    assert contrast_rows(result) == [("celltype", pytest.approx(50.8770049887559), 3)]
    assert result.cmat == []
    assert result.cmat_names == []

    # the summary curves: R's surv rows start at time 0, one row ahead of its time vector;
    # here row i is the curve at time[i] (R's row i + 1)
    summary = result.summary
    assert len(summary.time) == 97
    assert len(summary.surv) == 97
    assert [summary.time[idx] for idx in (0, 1, 49, 96)] == [1, 2, 99, 999]
    assert_rows_close(
        [summary.surv[idx] for idx in (0, 49, 96)],
        [
            [0.9927225474414726, 0.98283260336238942, 0.975948575109591210, 0.98877852173899539],
            [0.5791633733924062, 0.33503456962322054, 0.242861318707611629, 0.46475381246746716],
            [0.0156153763667681, 0.00118177484202206, 0.000223667428353563, 0.00617546647353692],
        ],
    )
    assert_rows_close(
        [summary.std_err[idx] for idx in (0, 96)],
        [
            [0.000546625896422911, 0.00488701936811825, 0.00756820036617423, 0.00356917340612000],
            [0.355969613722677825, 1.66160058911962416, 1.97818333288721004, 1.21424291657361838],
        ],
    )
    assert_rows_close(
        [summary.lower[0], summary.upper[49]],
        [
            [0.991667278552278, 0.973623684380323, 0.961921897607899, 0.981962787368510],
            [0.585839081984027, 0.391490844990018, 0.281202336819963, 0.549502191110555],
        ],
    )
    assert summary.cumhaz[49] == approx([-math.log(value) for value in summary.surv[49]])

    # R: the same with options = list(rmean = 365) and nsim = 50
    restricted = r.yates(
        fit, "celltype", predict="survival", options={"rmean": 365, "seed": SEED}, nsim=50
    )
    assert restricted.estimate["pmm"] == approx(
        [162.8414612572163, 91.9595879234022, 69.7961359196046, 126.9114322862328]
    )
    assert restricted.estimate["std"] == approx(
        [1.04136691739495, 23.40007372662978, 17.12475562480762, 27.65702111443096]
    )
    assert restricted.test[0].chisq == pytest.approx(33.1839742963497)

    # R: the same with options = list(rmean = Inf) and nsim = 50, which equals the default
    unrestricted = r.yates(
        fit, "celltype", predict="survival", options={"rmean": math.inf, "seed": SEED}, nsim=50
    )
    assert unrestricted.estimate["pmm"] == approx(
        [220.0263455561152, 99.7310793197197, 72.2119953964263, 152.2398949340128]
    )
    assert unrestricted.estimate["std"] == approx(
        [8.97519671693424, 32.89450238924809, 20.78022118298346, 48.20696060965924]
    )
    assert unrestricted.test[0].chisq == pytest.approx(63.3011823078696)


def test_yates_setup_errors_follow_r():
    fit = r.coxph("Surv(time, status) ~ celltype + karno + trt", _veteran())
    # R: predict = "survival" on a stratified fit, and predict = "expected"
    stratified = r.coxph("Surv(time, status) ~ celltype + karno + strata(trt)", _veteran())
    with pytest.raises(ValueError, match="stratified models not yet supported"):
        r.yates(stratified, "celltype", predict="survival")
    with pytest.raises(ValueError, match="type expected is not supported"):
        r.yates(fit, "celltype", predict="expected")
    with pytest.raises(ValueError, match="should be one of"):
        r.yates(fit, "celltype", predict="hazard")
    with pytest.raises(ValueError, match="user written prediction functions"):
        r.yates(fit, "celltype", predict=lambda eta: eta)
    with pytest.raises(TypeError, match="unrecognized risk options: rmean"):
        r.yates(fit, "celltype", predict="risk", options={"rmean": 100})
    linear = r.yates(fit, "celltype").estimate
    assert r.yates(fit, "celltype", predict="lp").estimate == linear
    # R: predict = NULL, which match.arg takes as "lp"
    assert r.yates(fit, "celltype", predict=None).estimate == linear


def test_yates_model_uses_the_linear_predictor_for_other_predictions():
    # R's yates passes predict= to yates_setup.default, whose argument is type, so it neither
    # checks nor warns and gives the linear predictor (R: yates(lm(time ~ karno, veteran),
    # "karno", levels = c(40, 60), predict = "risk") is silent)
    data = _veteran()
    model = r.YatesModel("time ~ karno", data, [-60.0, 3.0], [[400.0, -6.0], [-6.0, 0.1]])
    linear = r.yates(model, "karno", levels=[40, 60])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for predict in ("risk", "survival", None):
            other = r.yates(model, "karno", levels=[40, 60], predict=predict)
            assert other.estimate == linear.estimate
    # -60 + 3 * karno, with variance (1, karno) V (1, karno)'
    assert linear.estimate["pmm"] == approx([60.0, 120.0])
    assert linear.estimate["std"] == approx([math.sqrt(80.0), math.sqrt(40.0)])


def test_yates_model_drops_an_aliased_coefficient():
    # R: yates of celltype on lm(time ~ celltype + karno + k2, veteran) with k2 = 2 * karno;
    # the coefficients, vcov(fit, complete = FALSE) and sigma^2 are R's
    data = _veteran(k2=lambda data: [2 * value for value in data["karno"]])
    beta = [
        39.75722409970252,
        -109.24693157458275,
        -128.84929571446227,
        -45.01104968961336,
        2.63638364155418,
        math.nan,
    ]
    variance = [
        [
            1922.6409507258304,
            -716.0480226353174,
            -613.0729063339478,
            -457.8277461957134,
            -22.5355877620018,
        ],
        [
            -716.0480226353174,
            972.9156641939272,
            558.6283115087891,
            539.9667013943634,
            2.708943403706985,
        ],
        [
            -613.0729063339478,
            558.6283115087891,
            1268.486672357035,
            546.9767445228603,
            1.016864027862888,
        ],
        [
            -457.8277461957134,
            539.9667013943634,
            546.9767445228603,
            1272.0499415516533,
            -1.534112781920309,
        ],
        [
            -22.5355877620018,
            2.708943403706985,
            1.016864027862888,
            -1.534112781920309,
            0.370303085291109,
        ],
    ]
    model = r.YatesModel("time ~ celltype + karno + k2", data, beta, variance, 19291.6313423402)
    result = r.yates(model, "celltype")
    assert result.estimate["pmm"] == approx(
        [194.1684820546716, 84.9215504800889, 65.3191863402093, 149.1574323650583]
    )
    assert result.estimate["std"] == approx(
        [23.5186658549885, 20.2797764690940, 26.7316782018240, 27.0151464387539]
    )
    assert result.test[0].chisq == pytest.approx(18.0124373053017)
    assert result.test[0].ss == pytest.approx(347489.300070897)
    assert result.cmat_names == [
        "(Intercept)",
        "celltypesmallcell",
        "celltypeadeno",
        "celltypelarge",
        "karno",
    ]


def test_yates_simulation_validates_its_inputs():
    with pytest.raises(ValueError, match="population matrix columns"):
        _core.yates_risk([[[1.0]]], [0.1, 0.2], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="vmat"):
        _core.yates_risk([[[1.0, 2.0]]], [0.1, 0.2], [[1.0, 0.0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="nsim"):
        _core.yates_risk([[[1.0]]], [0.1], [[1.0]], [0.0], nsim=1)
    with pytest.raises(ValueError, match="estimable"):
        _core.yates_risk([[[1.0]]], [0.1], [[1.0]], [0.0], estimable=[True, False])


# --- unused factor levels ----------------------------------------------------------


def _lung_with_unused_level():
    # R: lung's time and status with g alternating "a", "b", a factor with levels a, b, c
    lung = datasets.load_lung()
    n = len(lung["time"])
    return {
        "time": lung["time"],
        "status": lung["status"],
        "g": RFactor([("a", "b")[idx % 2] for idx in range(n)], ["a", "b", "c"]),
    }


def test_unused_factor_levels_stay_as_aliased_columns():
    data = _lung_with_unused_level()
    # R: coxph of g
    fit = r.coxph("Surv(time, status) ~ g", data)
    assert list(fit.coef_names) == ["gb", "gc"]
    assert r.coef(fit) == approx([0.383302234567895, math.nan])
    assert_rows_close(r.vcov(fit), [[0.024849256452365, 0.0], [0.0, 0.0]])
    assert r.vcov(fit, complete=False) == [[pytest.approx(0.024849256452365)]]
    assert r.predict(fit, newdata={"g": RFactor(["a", "b"], ["a", "b", "c"])}) == approx(
        [0.0, 0.383302234567895]
    )
    # R: yates of g
    result = r.yates(fit, "g")
    assert result.estimate["g"] == ["a", "b", "c"]
    assert result.estimate["pmm"] == approx([0.0, 0.383302234567895, math.nan])
    assert result.estimate["std"] == approx([0.0, 0.157636469296813, math.nan])
    assert result.test[0].df is None
    assert result.cmat == [[0.0], [1.0], [0.0]]

    # R: survreg of g
    reg = r.survreg("Surv(time, status) ~ g", data)
    assert r.coef(reg) == approx([6.160629273095066, -0.273915136202959, math.nan])
    assert_rows_close(
        r.vcov(reg),
        [
            [0.006803439880034439, -0.00680489553126721, 0.0, 0.000028130659140692],
            [-0.006804895531267208, 0.01353797823472198, 0.0, -0.000228308598940690],
            [0.0, 0.0, 0.0, 0.0],
            [0.000028130659140692, -0.000228308598940690, 0.0, 0.003868466061947127],
        ],
    )
    # R: concordance of g
    concordance = r.concordance("Surv(time, status) ~ g", data)
    assert concordance.concordance == approx([0.437518736884181, 0.5])


def test_yates_model_drops_unused_levels_as_lm_does():
    # R: yates of g on lm(time ~ g), whose model.frame drops the unused level c
    # (drop.unused.levels = TRUE); the coefficients, vcov and sigma^2 are R's
    data = _lung_with_unused_level()
    model = r.YatesModel(
        "time ~ g",
        data,
        [347.035087719298, -83.6052631578947],
        [[375.482060170999, -375.482060170999], [-375.482060170999, 750.964120341998]],
        42804.9548594939,
    )
    result = r.yates(model, "g")
    assert result.estimate["g"] == ["a", "b"]
    assert result.estimate["pmm"] == approx([347.035087719298, 263.429824561404])
    assert result.estimate["std"] == approx([19.3773594736486, 19.3773594736486])
    assert contrast_rows(result) == [("global", pytest.approx(9.30782155679765), 1)]
    assert result.test[0].ss == pytest.approx(398420.881578947)
    assert_rows_close(result.mvar, [[375.482060170999, 0.0], [0.0, 375.482060170999]])
    assert result.cmat == [[1.0, 0.0], [1.0, 1.0]]
    assert result.cmat_names == ["(Intercept)", "gb"]
    with pytest.raises(ValueError, match="invalid level for term g"):
        r.yates(model, "g", levels=["a", "c"])


def test_a_subset_keeps_the_levels_it_leaves_unused():
    # R: coxph of e = factor(ph.ecog) on lung with subset = ph.ecog < 3, which leaves
    # level 3 unused
    lung = datasets.load_lung()
    lung["e"] = RFactor(lung["ph.ecog"], [0.0, 1.0, 2.0, 3.0])
    subset = [value is not None and value < 3 for value in lung["ph.ecog"]]
    fit = r.coxph("Surv(time, status) ~ e", lung, subset=subset)
    assert list(fit.coef_names) == ["e1", "e2", "e3"]
    assert r.coef(fit) == approx([0.368640502213039, 0.915301849920346, math.nan])
    assert_rows_close(
        r.vcov(fit),
        [
            [0.0394653413067124, 0.0273465588279593, 0.0],
            [0.0273465588279593, 0.0504267870542204, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    # R: the same survreg fit
    reg = r.survreg("Surv(time, status) ~ e", lung, subset=subset)
    assert r.coef(reg) == approx(
        [6.320348687805493, -0.264727588941228, -0.670775060081751, math.nan]
    )
