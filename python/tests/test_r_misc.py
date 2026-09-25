"""Tests of ``survival.r._misc`` against values from R survival 3.8.11."""

import math

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r_api

TOY = {
    "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "status": [1, 1, 0, 1, 0, 1, 1, 0],
    "x": [0.2, 0.4, 0.1, 0.8, 1.0, 1.2, 0.6, 1.4],
    "wt": [1.0] * 8,
    "id": list(range(1, 9)),
}


def _veteran():
    data = {k: v for k, v in survival.datasets.load_veteran().items() if not k.startswith("_")}
    data["celltype"] = RFactor(data["celltype"], ["squamous", "smallcell", "adeno", "large"])
    return data


# --- survcheck ---------------------------------------------------------------


def test_survcheck_matches_r_on_multistate_data_with_istate():
    data = {
        "id": ["a", "a", "b", "b", "c"],
        "tstart": [0, 1, 0, 2, 0],
        "tstop": [1, 2, 1, 3, 4],
        "event": RFactor(["B", "C", "B", "C", "censor"], ["censor", "B", "C"]),
        "istate": ["A", "B", "A", "A", "A"],
    }
    formula = 'Surv(tstart, tstop, event, type = "mstate") ~ 1'
    check = r.survcheck(formula, data, id="id", istate="istate")

    assert check.states == ["A", "B", "C"]
    assert check.transitions.from_states == ["A", "B", "C"]
    assert check.transitions.to_states == ["B", "C", "(censored)"]
    assert check.transitions.counts == [[2, 1, 1], [0, 1, 0], [0, 0, 0]]
    assert check.events.states == ["B", "C", "(any)"]
    assert check.events.count == [0, 1, 2]
    assert check.events.subjects == [[1, 2, 0], [1, 2, 0], [1, 0, 2]]
    flag = check.flag
    assert (flag.overlap, flag.gap, flag.jump, flag.teleport, flag.duplicate) == (0, 0, 1, 0, 0)
    assert check.istate == ["A", "B", "A", "B", "A"]
    assert check.n == {"id": 3, "observations": 5, "transitions": 4}
    assert check.jump.row == [4]
    assert check.jump.id == ["b"]
    assert (check.gap, check.overlap, check.teleport) == (None, None, None)
    assert check.y.type == "mcounting"
    assert check.id == data["id"]
    assert check.na_action is None

    without_istate = r.survcheck(formula, data, id="id")
    assert without_istate.states == ["(s0)", "B", "C"]
    assert without_istate.transitions.counts == [[2, 0, 1], [0, 2, 0], [0, 0, 0]]
    assert without_istate.istate == ["(s0)", "B", "(s0)", "B", "(s0)"]


def test_survcheck_reports_problem_rows_of_the_original_data():
    overlap = r.survcheck(
        "Surv(start, stop, status) ~ 1",
        {"id": ["s1", "s1", "s2"], "start": [0, 0.5, 0], "stop": [1, 2, 2], "status": [0, 1, 1]},
        id="id",
    )
    assert overlap.flag.overlap == 1
    assert overlap.overlap.row == [2]
    assert overlap.overlap.id == ["s1"]
    assert overlap.transitions.to_states == ["event"]
    assert overlap.transitions.counts == [[2], [0]]

    # a missing start time is dropped by na.omit; rows keep their original numbers
    gap = r.survcheck(
        "Surv(tstart, tstop, status) ~ 1",
        {
            "id": [1, 1, 2, 2, 3],
            "tstart": [0, 1, 0, 2, None],
            "tstop": [1, 2, 1, 3, 4],
            "status": [1, 0, 0, 1, 1],
        },
        id="id",
    )
    assert gap.n == {"id": 2, "observations": 4, "transitions": 2}
    assert gap.na_action == [5]
    assert gap.gap.row == [4]
    assert gap.gap.id == [2]

    right = r.survcheck(
        r.Surv([1, 2, 3, 4], [1, 0, 1, 0]), id=[1, 2, 3, 3], subset=[True, True, True, True]
    )
    assert right.states == ["(s0)", "event"]
    assert right.flag.overlap == 1
    assert right.overlap.row == [4]
    assert right.events.subjects == [[1, 2]]

    with pytest.raises(ValueError, match="an id argument is required"):
        r.survcheck(r.Surv([1, 2], [1, 0]))
    with pytest.raises(ValueError, match="wrong length for id"):
        r.survcheck(r.Surv([1, 2], [1, 0]), id=[1])
    with pytest.raises(ValueError, match="response must be right censored"):
        r.survcheck(r.Surv([1, 2], [3, 4], type="interval2"), id=[1, 2])
    with pytest.raises(ValueError, match="invalid value for timefix"):
        r.survcheck(r.Surv([1, 2], [1, 0]), id=[1, 2], timefix=1)
    with pytest.raises(ValueError, match="missing values"):
        r.survcheck(r.Surv([1, 2], [1, 0]), id=[1, None], na_action="na.fail")


def test_survcheck_accepts_the_coded_response_the_r_bridge_builds():
    # the multi-state example above with the states coded as the bridge does:
    # state_names = c("A", "B", "C"), status/istate = match(label, state_names)
    codes = r.survcheck(
        id=[1, 1, 2, 2, 3],
        time1=[0, 1, 0, 2, 0],
        time2=[1, 2, 1, 3, 4],
        status=[2, 3, 2, 3, 0],
        istate=[1, 2, 1, 1, 1],
    )
    assert codes.current_states == [1, 2, 1, 2, 1]
    assert codes.jump_rows == [3]
    assert (codes.overlap_rows, codes.gap_rows, codes.teleport_rows) == ([], [], [])
    assert codes.n_transitions == 4

    # without istate the current state codes are 0 for istate0, k for the k-th state
    no_istate = r.survcheck(
        id=[1, 1, 2, 2, 3],
        time1=[0, 1, 0, 2, 0],
        time2=[1, 2, 1, 3, 4],
        status=[1, 2, 1, 2, 0],
    )
    assert no_istate.current_states == [0, 1, 0, 1, 0]
    assert no_istate.n_transitions == 4
    overlap = r.survcheck(id=[1, 1], time1=[0, 0.5], time2=[1, 2], status=[0, 1])
    assert overlap.overlap_rows == [1]

    with pytest.raises(ValueError, match="a formula argument is required"):
        r.survcheck(id=[1, 2], time2=[1, 2])
    with pytest.raises(ValueError, match="only used when no formula"):
        r.survcheck(r.Surv([1, 2], [1, 0]), id=[1, 2], status=[1, 0])


# --- survobrien --------------------------------------------------------------


def test_survobrien_builds_r_data_frames():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "x": [0.1, 0.4, 0.2, 0.8],
        "group": ["a", "a", "b", "b"],
        "id": [10, 11, 12, 13],
        "off": [0.1, 0.2, 0.3, 0.4],
    }
    logits = [
        -1.9459101490553135,
        0.5108256237659907,
        -0.5108256237659907,
        1.9459101490553132,
        -1.0986122886681098,
        1.0986122886681098,
        0.0,
    ]
    frame = r.survobrien("Surv(time, status) ~ x", data=data)
    assert list(frame) == ["time", "status", ".id.", "x", ".strata."]
    assert frame["time"] == [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0]
    assert frame["status"] == [1, 0, 0, 0, 1, 0, 1]
    assert frame[".id."] == [1, 2, 3, 4, 3, 4, 4]
    assert frame["x"] == pytest.approx(logits)
    assert frame[".strata."] == [1, 1, 1, 1, 2, 2, 3]

    assert r.survobrien("Surv(time, status) ~ x + offset(off)", data=data) == frame
    factor = r.survobrien("Surv(time, status) ~ x + group", data=data)
    assert list(factor) == ["time", "status", "group", ".id.", "x", ".strata."]
    assert factor["group"] == ["a", "a", "b", "b", "b", "b", "b"]
    assert r.survobrien("Surv(time, status) ~ x + factor(group)", data=data) == factor

    cluster = r.survobrien("Surv(time, status) ~ x + cluster(id)", data=data)
    assert list(cluster) == ["time", "status", "id", "x", ".strata."]
    assert cluster["id"] == [10, 11, 12, 13, 12, 13, 13]

    strata = r.survobrien("Surv(time, status) ~ x + strata(group)", data=data)
    assert strata == {
        "time": [1.0, 2.0, 3.0, 4.0, 4.0],
        "status": [1, 0, 1, 0, 1],
        "group": ["a", "a", "b", "b", "b"],
        ".id.": [1, 2, 3, 4, 4],
        "x": pytest.approx([-1.0986122886681098, 1.0986122886681098] * 2 + [0.0]),
        ".strata.": [1, 1, 2, 2, 3],
    }

    doubled = r.survobrien(
        "Surv(time, status) ~ x", data=data, transform=lambda v: [2.0 * value for value in v]
    )
    assert doubled["x"] == pytest.approx([0.2, 0.8, 0.4, 1.6, 0.4, 1.6, 1.6])

    counting = r.survobrien(
        "Surv(start, stop, status) ~ x",
        data={
            "start": [0.0, 0.0, 1.0, 2.0],
            "stop": [1.0, 2.0, 3.0, 4.0],
            "status": [1, 0, 1, 1],
            "x": [0.1, 0.4, 0.2, 0.8],
        },
    )
    assert counting["start"] == [0.0, 0.0, 1.0, 2.0, 2.0]
    assert counting["stop"] == [1.0, 2.0, 3.0, 4.0, 4.0]
    assert counting["status"] == [1, 0, 1, 0, 1]
    assert counting["x"] == pytest.approx([-1.0986122886681098, 1.0986122886681098] * 2 + [0.0])


def test_survobrien_applies_subset_and_na_action_before_expanding():
    data = {
        "time": [1, 2, 3, 4, 5],
        "status": [1, 0, 1, 1, 0],
        "x": [0.1, None, 0.2, 0.8, 0.3],
        "z": [2, 3, 4, 5, 6],
    }
    omitted = r.survobrien("Surv(time, status) ~ x + z", data=data)
    assert omitted["time"] == [1.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 4.0, 5.0]
    assert omitted[".id."] == [1, 2, 3, 4, 2, 3, 4, 3, 4]
    assert omitted["z"][4:7] == pytest.approx([-1.6094379124341005, 0.0, 1.6094379124341007])

    subset = r.survobrien(
        "Surv(time, status) ~ x + z", data=data, subset=[t > 1 for t in data["time"]]
    )
    assert subset["time"] == [3.0, 4.0, 5.0, 4.0, 5.0]
    assert subset["x"] == pytest.approx(
        [-1.6094379124341005, 1.6094379124341007, 0.0, 1.0986122886681098, -1.0986122886681098]
    )

    with pytest.raises(ValueError, match="No continuous variables to modify"):
        r.survobrien("Surv(time, status) ~ factor(z)", data=data)
    with pytest.raises(ValueError, match="iteraction terms"):
        r.survobrien("Surv(time, status) ~ x * z", data=data)
    with pytest.raises(ValueError, match="Transform function must be 1 to 1"):
        r.survobrien("Surv(time, status) ~ z", data=data, transform=lambda v: v[:1])
    with pytest.raises(ValueError, match="right censored or"):
        r.survobrien("Surv(time, z, status, type = 'interval') ~ x", data=data)


# --- royston and brier -------------------------------------------------------


def test_royston_matches_r_with_and_without_newdata():
    fit = r.coxph("Surv(time, status) ~ x", data=TOY)
    result = r.royston(fit)
    assert list(result) == ["D", "se(D)", "R.D", "R.KO", "R.N", "C.GH"]
    assert result["D"] == pytest.approx(1.483907190240159)
    assert result["se(D)"] == pytest.approx(1.059615493787745)
    assert result["R.D"] == pytest.approx(0.344556335064422)
    assert result["R.KO"] == pytest.approx(0.375253490543464)
    assert result["R.N"] == pytest.approx(0.339190751422645)
    assert result["C.GH"] == pytest.approx(0.748974097360896)

    newdata = {
        "time": [2, 3, 5, 6, 8, 9],
        "status": [1, 0, 1, 1, 0, 1],
        "x": [0.3, 0.9, 0.5, 1.1, 0.2, 0.7],
    }
    new = r.royston(fit, newdata=newdata)
    assert list(new) == ["D", "se(D)", "R.D", "R.KO", "C.GH"]
    assert new["D"] == pytest.approx(-0.224843328021406)
    assert new["se(D)"] == pytest.approx(0.992767433531538)
    assert new["R.D"] == pytest.approx(0.0119250793252209)
    assert new["R.KO"] == pytest.approx(4.34675376620985e-05)
    assert new["C.GH"] == pytest.approx(0.502626241604716)

    adjusted = r.royston(fit, newdata=newdata, ties=False, adjust=True)
    assert adjusted["D"] == pytest.approx(0.684727829264644)
    assert adjusted["se(D)"] == pytest.approx(0.260795164637866)
    assert adjusted["R.D"] == pytest.approx(-0.225667179503653)

    with pytest.raises(TypeError, match="only for coxph models"):
        r.royston(object())


def test_brier_matches_r_and_checks_the_data_as_r_does():
    fit = r.coxph("Surv(time, status) ~ x", data=TOY, model=True)
    result = r.brier(fit, times=[2.0, 4.0, 6.0], detail=True)
    assert result.times == [2.0, 4.0, 6.0]
    assert result.brier == pytest.approx([0.141118373376419, 0.136829153005458, 0.240950350225449])
    assert result.rsquared == pytest.approx(
        [0.247368675325767, 0.429878529143924, -0.00395979260603663]
    )
    assert result.p0 == pytest.approx([0.25, 0.4, 0.6])
    assert result.eff_n == pytest.approx([8.0, 6.95652173913044, 5.75539568345324])
    assert (result["eff.n"], result["brier"]) == (result.eff_n, result.brier)
    with pytest.raises(KeyError):
        result["call"]
    assert result.phat[0][:3] == pytest.approx(
        [0.443332228197536, 0.318782464012, 0.515010284505113]
    )
    assert r.brier(fit).times == [1.0, 2.0, 4.0, 6.0, 7.0]
    assert r.brier(fit, times=[2.0, 4.0, 6.0]).p0 is None

    same = r.brier(fit, times=[2.0, 4.0, 6.0], newdata=TOY, detail=True)
    assert same.brier == result.brier
    assert same.phat == result.phat

    weighted_fit = r.coxph("Surv(time, status) ~ x", data=TOY, weights="wt", model=True)
    weighted = r.brier(
        weighted_fit,
        times=[2.0, 4.0, 6.0],
        newdata={**TOY, "wt": [8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        detail=True,
    )
    assert weighted.eff_n == pytest.approx([3.1690140845, 3.1163434903, 3.0356177407])
    assert weighted.brier == pytest.approx([0.21987334, 0.09626662, 0.12947811], abs=1e-6)

    counting = {
        "start": [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        "stop": [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        "status": [0, 1, 1, 0, 0, 1],
        "x": [0.2, 0.2, 0.6, 0.6, 1.0, 1.0],
        "id": [1, 1, 2, 2, 3, 3],
    }
    counting_fit = r.coxph("Surv(start, stop, status) ~ x", data=counting, id="id", iter_max=0)
    counting_result = r.brier(counting_fit, times=[3.0, 5.0, 7.0], detail=True)
    assert counting_result.brier == pytest.approx(
        [0.166967022114529, 0.249285544480962, 0.0356739933472524]
    )
    assert counting_result.eff_n == pytest.approx([5.0, 3.94736842105263, 2.52808988764045])
    assert counting_result.phat[0] == pytest.approx([0.283468689426211] * 6)

    gap_fit = r.coxph(
        "Surv(start, stop, status) ~ x",
        data={**counting, "start": [0.0, 3.0, 0.0, 3.0, 0.0, 4.0]},
        id="id",
        iter_max=0,
    )
    with pytest.raises(ValueError, match="flags are >0 in survcheck"):
        r.brier(gap_fit, times=[3.0, 5.0, 7.0])
    staggered_fit = r.coxph(
        "Surv(start, stop, status) ~ x",
        data={
            "start": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "stop": TOY["time"],
            "status": TOY["status"],
            "x": TOY["x"],
            "id": TOY["id"],
        },
        id="id",
    )
    with pytest.raises(ValueError, match="delayed entry is not yet implemented"):
        r.brier(staggered_fit, times=[2.0, 4.0, 6.0])
    with pytest.raises(TypeError, match="fit must be a coxph object"):
        r.brier(object())
    with pytest.raises(ValueError, match="invalid value for timefix"):
        r.brier(fit, timefix="yes")


# --- yates -------------------------------------------------------------------


def test_yates_population_means_match_r():
    fit = r.coxph("Surv(time, status) ~ celltype + karno", _veteran(), model=True)
    result = r.yates(fit, "celltype")
    assert result.estimate["celltype"] == ["squamous", "smallcell", "adeno", "large"]
    assert result.estimate["pmm"] == pytest.approx(
        [0.0, 0.715334402378886, 1.15773326678565, 0.325644901681043]
    )
    assert result.estimate["std"] == pytest.approx(
        [0.303223819928728, 0.424438625719816, 0.42455192550107, 0.39005677960008]
    )
    assert [(row.name, row.df) for row in result.test] == [("global", 3)]
    assert result.cmat_names == ["celltypesmallcell", "celltypeadeno", "celltypelarge", "karno"]
    assert result.cmat[1] == pytest.approx([1.0, 0.0, 0.0, 58.5693430656934])
    assert result.mvar[0][0] == pytest.approx(0.0919446849721694)

    pairwise = r.yates(fit, "celltype", test="pairwise")
    assert [row.name for row in pairwise.test] == [
        "1 vs 2",
        "1 vs 3",
        "1 vs 4",
        "2 vs 3",
        "2 vs 4",
        "3 vs 4",
    ]
    assert [row.chisq for row in pairwise.test] == pytest.approx(
        [8.01412259838, 15.61956492647, 1.38526980233, 2.9988526684, 2.22994736958, 8.04220486484]
    )

    two = r.yates(fit, "celltype", levels=["squamous", "adeno"])
    assert two.estimate["celltype"] == ["squamous", "adeno"]
    assert two.test[0].chisq == pytest.approx(15.6195649265)
    assert two.cmat[0] == pytest.approx([0, 0, 0, 58.5693430657])
    assert two.cmat[1] == pytest.approx([0, 1, 0, 58.5693430657])

    frame = r.yates(fit, "celltype", population={"karno": [40.0, 60.0, 80.0]})
    assert frame.estimate["pmm"] == pytest.approx(
        [-0.0444313855412, 0.6709030168377, 1.1133018812445, 0.2812135161399]
    )
    assert frame.test[0].chisq == pytest.approx(17.2820097074)

    continuous = r.yates(fit, "karno", levels=[40, 60, 80])
    assert continuous.estimate["karno"] == [40, 60, 80]
    assert continuous.estimate["pmm"] == pytest.approx(
        [1.119673963392, 0.498541328786, -0.122591305821]
    )
    assert continuous.test[0].df == 1
    with pytest.raises(ValueError, match="continuous variables require the levels"):
        r.yates(fit, "karno")

    weighted = r.coxph(
        "Surv(time, status) ~ celltype + karno",
        _veteran(),
        weights=[1.0, 2.0] * 68 + [1.0],
        model=True,
    )
    assert r.yates(weighted, "celltype").estimate["pmm"] == pytest.approx(
        [0.0, 0.720748577623, 1.082698939802, 0.361773971452]
    )


def test_yates_populations_and_unsupported_options():
    fit = r.coxph("Surv(time, status) ~ celltype + factor(trt) + karno", _veteran(), model=True)
    sas = r.yates(fit, "celltype", population="sas")
    assert sas.estimate["pmm"] == pytest.approx(
        [0.130872045029, 0.955852232886, 1.284866458891, 0.525497508975]
    )
    assert sas.cmat[1] == pytest.approx([1.0, 0.0, 0.0, 0.5, 58.5693430657])
    with pytest.raises(ValueError, match="population=factorial only applies"):
        r.yates(fit, "celltype", population="factorial")

    factorial_fit = r.coxph("Surv(time, status) ~ celltype + factor(trt)", _veteran(), model=True)
    factorial = r.yates(factorial_fit, "celltype", population="factorial")
    assert factorial.estimate["pmm"] == pytest.approx(
        [0.0989007082481348, 1.19534248088955, 1.26777430267506, 0.39595006164188]
    )
    assert factorial.cmat == [[0, 0, 0, 0.5], [1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5]]
    assert r.yates(factorial_fit, "celltype", population="yates").cmat == factorial.cmat
    trt = r.yates(factorial_fit, "factor(trt)")
    assert trt.estimate["factor(trt)"] == [1.0, 2.0]

    risk = r.yates(fit, "celltype", predict="risk")
    assert all(value > 0 for value in risk.estimate["pmm"])
    assert len(risk.mvar) == 4
    with pytest.raises(NotImplementedError, match="sgtt"):
        r.yates(fit, "celltype", method="sgtt")
    with pytest.raises(ValueError, match="not found in the formula"):
        r.yates(fit, "age")
    with pytest.raises(TypeError, match="data frame or character"):
        r.yates(fit, "celltype", population=3)


# --- cipoisson, bounded links, statefig ----------------------------------------


def test_cipoisson_recycles_like_r():
    assert r.cipoisson(5, time=10.0) == pytest.approx((0.1623486, 1.1668332))
    vector = r.cipoisson([0, 5, 20], time=[1.0, 10.0, 4.0])
    assert [lower for lower, _ in vector] == pytest.approx([0.0, 0.1623486, 3.0541299])
    assert [upper for _, upper in vector] == pytest.approx([3.688879, 1.1668332, 7.722094])
    recycled = r.cipoisson([1, 2], time=[1.0, 2.0, 3.0])
    assert [lower for lower, _ in recycled] == pytest.approx(
        [0.025317808, 0.121104639, 0.008439269]
    )
    assert [upper for _, upper in recycled] == pytest.approx([5.571643, 3.612344, 1.857214])
    assert r.cipoisson(5, time=10.0, method="ansc") == pytest.approx((0.1507881, 1.1586004))
    zero_time = r.cipoisson([1, 2], time=[0.0, 2.0])
    assert [math.isnan(v) for v in zero_time[0]] == [True, True]
    assert zero_time[1] == pytest.approx((0.121104639, 3.612344))
    with pytest.raises(ValueError, match="non-negative"):
        r.cipoisson(-1)
    with pytest.raises(ValueError, match="Invalid method"):
        r.cipoisson(1, method="fancy")


def test_bounded_links_match_r_link_functions():
    x = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    assert r.blogit(x) == pytest.approx(
        [-2.94443898, -2.94443898, -2.94443898, 0.0, 2.94443898, 2.94443898, 2.94443898]
    )
    assert r.bprobit(x) == pytest.approx(
        [-1.64485363, -1.64485363, -1.64485363, 0.0, 1.64485363, 1.64485363, 1.64485363]
    )
    assert r.bcloglog(x) == pytest.approx(
        [-2.97019525, -2.97019525, -2.97019525, -0.36651292, 1.0971887, 1.0971887, 1.0971887]
    )
    assert r.blog(x) == pytest.approx(
        [-2.99573227, -2.99573227, -2.99573227, -0.69314718, -0.05129329, -0.01005034, 0.0]
    )
    assert r.blogit(0.5) == pytest.approx(0.0)
    assert r.blogit([0.0, 0.75, 1.0], edge=0.6) == pytest.approx([0.4054651] * 3)
    missing = r.blogit([0.0, None, 1.0])
    assert missing[0] == pytest.approx(-2.94443898)
    assert math.isnan(missing[1])


def test_statefig_matches_r_layouts():
    connect = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    row = r.statefig([1, 2], connect, states=["A", "B", "C"])
    assert row.states == ["A", "B", "C"]
    assert row.positions == pytest.approx([(0.25, 0.5), (0.75, 0.75), (0.75, 0.25)])
    assert [(arrow.from_state, arrow.to_state) for arrow in row.arrows] == [(0, 1), (0, 2), (1, 2)]

    column = r.statefig([[1], [2]], {"A": [0, 1, 1], "B": [0, 0, 1], "C": [0, 0, 0]})
    assert column.positions == pytest.approx([(0.5, 0.75), (0.25, 0.25), (0.75, 0.25)])

    coordinates = r.statefig([[0.2, 0.7], [0.8, 0.3]], [[0, 1], [0, 0]], states=["a", "b"])
    assert coordinates.positions == [(0.2, 0.7), (0.8, 0.3)]

    with pytest.raises(ValueError, match="number of boxes"):
        r.statefig([1, 2], [[0, 1], [0, 0]], states=["A", "B"])
    with pytest.raises(ValueError, match="square matrix"):
        r.statefig([1], [[0, 1]], states=["A"])
    with pytest.raises(ValueError, match="dimnames"):
        r.statefig([1, 2], connect)
    with pytest.raises(ValueError, match="non-integer number of states"):
        r.statefig([1.5, 1.5], connect, states=["A", "B", "C"])


# --- nsk and pspline ------------------------------------------------------------


def test_nsk_follows_r_boundary_knot_rules():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    basis = r.nsk(x, df=3)
    assert (basis.n_rows, basis.n_cols) == (5, 3)
    assert basis.knots == pytest.approx([2.6666666666666665, 3.333333333333333])
    assert basis.boundary_knots == pytest.approx((1.2, 4.8))
    assert basis.basis[:3] == pytest.approx(
        [-0.30663390663390683, 0.12972972972972977, -0.007507507507507517]
    )
    intercept = r.nsk(x, df=4, intercept=True)
    assert intercept.n_cols == 4
    assert intercept.basis[0] == pytest.approx(1.184411684411685)

    explicit = r.nsk([1.0, 2.0, 3.0, 4.0], knots=[2.0, 3.0], Boundary_knots=[1.0, 4.0])
    assert explicit.knots == pytest.approx([2.0, 3.0])
    assert explicit.basis == pytest.approx(
        [1.0 if row > 0 and row - 1 == col else 0.0 for row in range(4) for col in range(3)]
    )
    assert r.nsk(x, df=3, Boundary_knots=True).boundary_knots == pytest.approx((1.0, 5.0))
    outer = r.nsk(x, knots=[1.0, 2.0, 4.0, 5.0], Boundary_knots=None)
    assert outer.boundary_knots == pytest.approx((1.0, 5.0))
    assert outer.knots == pytest.approx([2.0, 4.0])
    inside = r.nsk(x, knots=[2.0, 4.0], Boundary_knots=[2.5, 3.5])
    assert inside.boundary_knots == pytest.approx((2.0, 4.0))
    assert inside.knots == []

    missing = r.nsk([1.0, math.nan, 2.0, 3.0, 4.0, 5.0], df=3)
    assert missing.n_rows == 6
    assert all(math.isnan(value) for value in missing.basis[3:6])
    assert missing.basis[:3] == pytest.approx(basis.basis[:3])

    with pytest.raises(ValueError, match="wrong length for Boundary.knots"):
        r.nsk(x, knots=[2.0], Boundary_knots=None)
    with pytest.raises(ValueError, match="only finite values"):
        r.nsk([1.0, math.inf], df=3)
    with pytest.raises(ValueError, match="at least one non-missing"):
        r.nsk([math.nan, math.nan], df=3)


def test_pspline_carries_r_attributes():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    basis = r.pspline(x, df=3)
    assert (basis.method, basis.nterm, basis.degree, basis.n_cols) == ("df", 8, 3, 10)
    assert (basis.df, basis.eps, basis.theta) == (3, 0.1, None)
    assert basis.boundary_knots == (1.0, 5.0)
    assert basis.basis[0] == pytest.approx([2.0 / 3.0, 1.0 / 6.0, *([0.0] * 8)])
    assert basis.dmat[0][:3] == pytest.approx([5.0, -4.0, 1.0])
    assert basis.cbase == pytest.approx([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

    outside = r.pspline([0.0, 1.0, 5.0, 6.0], df=3, Boundary_knots=[1.0, 5.0], penalty=False)
    assert outside.penalty is False
    assert outside.basis[0] == pytest.approx([2.0 / 3.0, -5.0 / 6.0, *([0.0] * 8)])
    # predict.pspline in the R bridge passes the stored attributes under the bridge's names
    same = r.pspline(
        [0.0, 1.0, 5.0, 6.0], nterm=8, degree=3, boundary_knots=[1.0, 5.0], penalty=False
    )
    assert same.basis == outside.basis
    with pytest.raises(ValueError, match="only one of"):
        r.pspline(x, Boundary_knots=[1.0, 5.0], boundary_knots=[1.0, 5.0])
    fixed = r.pspline(x, theta=0.5)
    assert (fixed.method, fixed.theta, fixed.n_cols) == ("fixed", 0.5, 12)
    aic = r.pspline(x, df=0)
    assert (aic.method, aic.eps, aic.nterm, aic.n_cols) == ("aic", 1e-5, 15, 17)
    combined = r.pspline(x, df=3, combine=[1] * 10)
    assert (combined.combine, combined.n_cols) == ([1] * 10, 1)
    with_intercept = r.pspline(x, df=3, intercept=True)
    assert (with_intercept.n_cols, len(with_intercept.dmat)) == (11, 11)
    missing = r.pspline([1.0, math.nan, 2.0], df=2)
    assert all(math.isnan(value) for value in missing.basis[1])

    with pytest.raises(ValueError, match="Invalid value for theta"):
        r.pspline(x, theta=1.0)
    with pytest.raises(ValueError, match="Too few degrees"):
        r.pspline(x, df=1)
    with pytest.raises(ValueError, match="nterm' too small"):
        r.pspline(x, df=4, nterm=3)
    with pytest.raises(ValueError, match="Invalid values for Boundary.knots"):
        r.pspline(x, df=3, Boundary_knots=[5.0, 1.0])


def test_frailty_encoding_normalizes_levels_and_sparse_default():
    from survival.r._misc import _frailty_encoding

    encoded = _frailty_encoding(["b", "a", None, "b"], levels=["a", "b"])
    assert encoded == {"codes": [2, 1, None, 2], "levels": ["a", "b"], "nclass": 2, "sparse": False}
    assert _frailty_encoding(list("abcdef"))["sparse"] is True
    with pytest.raises(ValueError, match="outside supplied levels"):
        _frailty_encoding(["c"], levels=["a", "b"])
