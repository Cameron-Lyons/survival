"""Regression tests of ``yates`` against R survival 3.8-12: default (``model = FALSE``) Cox
fits, aliased coefficients and R's estimability check, ``predict = "survival"`` and R's
``yates_setup`` errors."""

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
    lp = r.yates(fit, "celltype", predict="lp")
    assert lp.estimate == r.yates(fit, "celltype").estimate


def test_yates_model_uses_the_linear_predictor_for_other_predictions():
    # R's yates_setup.default: a warning, then the linear predictor
    data = _veteran()
    model = r.YatesModel("time ~ karno", data, [-60.0, 3.0], [[400.0, -6.0], [-6.0, 0.1]])
    linear = r.yates(model, "karno", levels=[40, 60])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        risk = r.yates(model, "karno", levels=[40, 60], predict="risk")
    assert any("linear predictor estimate used by default" in str(w.message) for w in caught)
    assert risk.estimate == linear.estimate
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
