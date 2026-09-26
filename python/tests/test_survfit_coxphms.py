"""``survfit`` of a multi-state Cox model (R's ``survfit.coxphms``) and the methods of
its result: summary, survfit0, aggregate, quantile, residuals, subsetting and the
data frame layout.

Reference values are R 4.5.3 / survival 3.8-12 on the data sets of R's tests
coxsurv5.R, coxsurv6.R, mstrata.R, timeline.R and summarydf.R, rebuilt below with R's
row order and factor levels.  Row numbers in the comments are R's (1-based).
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets

from survival.r._coxph import survfit_coxph  # noqa: E402
from survival.r._models import _subset_coxms_curves  # noqa: E402


def approx(values, rel=1e-9):
    """Relative tolerance 1e-9, absolute 1e-12 (for exact zeros), for nested lists too."""

    expected = np.asarray(values, dtype=np.float64) if isinstance(values, list) else values
    return pytest.approx(expected, rel=rel, abs=1e-12)


def cat(values, levels):
    return pd.Categorical(values, categories=levels)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mg():
    m = pd.DataFrame(datasets.load_mgus2())
    m["etime"] = np.where(m.pstat == 1, m.ptime, m.futime)
    event = np.where(m.pstat == 1, "pcm", np.where(m.death == 1, "death", "censor"))
    m["event"] = cat(event, ["censor", "pcm", "death"])
    m["w"] = np.where(m.age > 70, 2, 1)
    m["o"] = 0.01 * (m.dxyr - 1980)
    return m


def _myeloid():
    raw = datasets.load_myeloid()
    base = {name: raw[name] for name in ("id", "trt", "sex", "flt3")}
    merged = r.tmerge(
        base,
        raw,
        "id",
        death=r.event("futime", "death"),
        priortx=r.tdc("txtime"),
        sct=r.event("txtime"),
    )
    data = pd.DataFrame(dict(merged))
    code = (data.sct + 2 * data.death).astype(int)
    data["event"] = cat(np.array(["censor", "sct", "death"])[code], ["censor", "sct", "death"])
    return data


@pytest.fixture(scope="module")
def my():
    return _myeloid()


@pytest.fixture(scope="module")
def my_na():
    data = _myeloid()
    data["sex"] = data["sex"].astype(object)
    data["flt3"] = data["flt3"].astype(object)
    data.loc[data.id.isin([273, 274, 275]), "sex"] = None
    data.loc[data.id.isin([271, 272, 273]), "flt3"] = None
    return data


@pytest.fixture(scope="module")
def lms():
    lung = pd.DataFrame(datasets.load_lung())
    data = lung[lung["ph.ecog"].notna() & (lung["ph.ecog"] < 3)].reset_index(drop=True)
    data["state"] = cat(np.where(data.status == 2, "death", "censor"), ["censor", "death"])
    ecog = data["ph.ecog"].astype(int)
    data["cstate"] = cat(np.array(["ph0", "ph1", "ph2"])[ecog], ["ph0", "ph1", "ph2"])
    data["id"] = np.arange(1, len(data) + 1)
    return data


def _mtest(t1_row3=9.0):
    state = ["a", "b", "a", "b", "c", "a", "c", "censor", "b", "censor"]
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 3, 4, 4, 4, 5, 5],
            "t1": [0, 4, t1_row3, 0, 2, 0, 2, 8, 1, 3],
            "t2": [4, 9, 10, 5, 9, 2, 8, 9, 3, 11],
            "state": cat(state, ["censor", "a", "b", "c"]),
            "x": [0, 0, 0, 1, 1, 0, 0, 0, 2, 2],
        }
    )


def _test2():
    codes = [1, 2, 1, 2, 3, 1, 3, 0, 2, 0, 0, 0, 0, 0]
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9],
            "t1": [0, 8, 18, 0, 4, 0, 4, 16, 2, 6, 0, 0, 7, 8],
            "t2": [8, 18, 20, 10, 18, 4, 16, 18, 6, 22, 5, 10, 10, 15],
            "state": cat(np.array(["censor", "a", "b", "c"])[codes], ["censor", "a", "b", "c"]),
            "x": [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1, 1, 2, 0],
        }
    )


@pytest.fixture(scope="module")
def pbc2():
    seq = pd.DataFrame(datasets.load_pbcseq())
    labels = ["normal", "1-2", "2-4", ">4"]
    seq["bili4"] = pd.cut(seq.bili, [0, 1, 2, 4, 100], labels=labels)
    first = seq[~seq.id.duplicated()]
    base = {name: first[name].tolist() for name in ("id", "age", "sex")}
    data = r.tmerge(
        base,
        {name: first[name].tolist() for name in ("id", "futime")},
        "id",
        death=r.event("futime", (first.status == 2).astype(int).tolist()),
    )
    data = r.tmerge(
        data,
        {"id": seq.id.tolist(), "day": seq.day.tolist(), "bili": seq.bili.tolist()},
        "id",
        bili=r.tdc("day", "bili"),
        bili4=r.tdc("day", (seq.bili4.cat.codes + 1).tolist()),
        bstat=r.event("day", (seq.bili4.cat.codes + 1).tolist()),
    )
    data = pd.DataFrame(dict(data))
    code = data.bili4.astype(int)
    bstat = np.where(data.death == 1, 5, data.bstat).astype(int)
    bstat = np.where(code == bstat, 0, bstat)
    states = ["censor", *labels, "death"]
    data["bstat"] = cat(np.array(states)[bstat], states)
    data["bili4"] = cat(np.array(labels)[code - 1], labels)
    return data


PDATA_STATES = ["censor", "normal", "1-4", "4+", "death"]


@pytest.fixture(scope="module")
def pdata():
    seq = pd.DataFrame(datasets.load_pbcseq())
    visits = seq[["id", "day", "age", "bili", "albumin", "edema", "ast"]].copy()
    visits["bstat"] = pd.cut(visits.bili, [0, 1, 4, 100], labels=False) + 1
    last = seq[~seq.id.duplicated()][["id", "futime", "status"]]
    ends = pd.DataFrame(
        {"id": last.id, "day": last.futime, "bstat": np.where(last.status == 2, 4, 0)}
    )
    data = pd.concat([visits, ends]).sort_values(["id", "day"]).reset_index(drop=True)
    data.loc[data.id == 1, "edema"] = np.nan
    rows = np.flatnonzero(data.id == 2)
    data.loc[rows[(rows + 1) % 2 == 1], "albumin"] = np.nan
    data.loc[np.flatnonzero(data.id == 3)[0], "ast"] = np.nan
    data["bstat"] = cat(np.array(PDATA_STATES)[data.bstat.astype(int)], PDATA_STATES)
    return data


ND_MG = {"age": [60, 80], "sex": ["F", "M"]}
ND_MY = {"trt": ["A", "B"], "sex": ["f", "m"]}


@pytest.fixture(scope="module")
def fa1(mg):
    return r.coxph("Surv(etime, event) ~ age + sex", mg, id="id")


@pytest.fixture(scope="module")
def sf(fa1):
    return r.survfit(fa1, newdata=ND_MG)


@pytest.fixture(scope="module")
def fm(my):
    return r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my, id="id")


@pytest.fixture(scope="module")
def fst(mg):
    return r.coxph("Surv(etime, event) ~ age + mspike + strata(sex)", mg, id="id")


@pytest.fixture(scope="module")
def sfs(fst):
    return r.survfit(fst, newdata={"age": [60, 80], "mspike": [1.2, 1.2]})


def row(curves, number):
    """``curves.pstate[number, , ]`` (R's 1-based row) as nested lists."""

    return curves.pstate[number - 1].tolist()


# ---------------------------------------------------------------------------
# 1-2. competing risks: the curves and their options
# ---------------------------------------------------------------------------


def test_competing_risks_curves(fa1, sf):
    assert isinstance(sf, r.CoxSurvfitMultiStateResult)
    assert len(sf.time) == 268
    assert sf.pstate.shape == (268, 2, 3)
    assert sf.cumhaz.shape == (268, 2, 2)
    assert sf.cumhaz_names == ["1:2", "1:3"]
    assert sf.states == ["(s0)", "pcm", "death"]
    assert sf.p0 == [[1.0, 0.0, 0.0]]
    assert (sf.t0, sf.start_time, sf.n, sf.n_id) == (0.0, 0.0, [1384], [1384])
    assert sf.transitions == fa1.transitions
    assert sf.dim == {"data": 2, "states": 3}
    assert sf.time[:3] == [1.0, 2.0, 3.0]
    assert row(sf, 1) == approx(
        [[0.990483018741919, 0, 0.00951698125808076], [0.949865952291592, 0, 0.0501340477084077]]
    )
    assert sf.cumhaz[0].tolist() == approx([[0, 0.00956255711778373], [0, 0.0514344071944498]])
    assert row(sf, 2) == approx(
        [
            [0.982708987257523, 0.00128601396353815, 0.0160049987789389],
            [0.915343550448308, 0.00153845434405557, 0.0831179952076362],
        ]
    )
    assert sf.time[49] == 50
    assert row(sf, 50) == approx(
        [
            [0.864745100029414, 0.0272954726877502, 0.107959427282836],
            [0.516317741343079, 0.0270101734214449, 0.456672085235476],
        ]
    )
    assert sf.cumhaz[49].tolist() == approx(
        [[0.0293230268192218, 0.115997470705242], [0.0371139181679766, 0.623919007049286]]
    )
    assert sf.n_risk[49] == [948, 0, 0]
    assert row(sf, 268) == approx(
        [
            [0.0451381989553272, 0.274908729652618, 0.679953071392055],
            [2.62611243559767e-06, 0.0640995973533414, 0.935897776534223],
        ]
    )
    assert sf.cumhaz.sum() == approx(801.574652616739)
    assert sf.newdata == ND_MG


def test_curves_emit_no_warning(fa1, recwarn):
    r.survfit(fa1, newdata=ND_MG)
    assert len(recwarn) == 0


def test_curve_options(mg, fa1, sf):
    stype1 = r.survfit(fa1, newdata=ND_MG, stype=1)
    assert row(stype1, 50) == approx(
        [
            [0.864511071784724, 0.0273371336393544, 0.108151794575922],
            [0.513208834104228, 0.0270961811904414, 0.45969498470533],
        ]
    )
    # p (I + A) steps can leave [0, 1], as in R
    assert row(stype1, 268) == approx(
        [
            [0.000457236646148776, 0.287231659939526, 0.712311103414324],
            [-0.000986048753212996, 0.0636597917138248, 0.937326257039388],
        ]
    )
    # coxsurv1 carries the Efron sum over from the later time (R's result)
    ctype2 = r.survfit(fa1, newdata=ND_MG, ctype=2)
    assert row(ctype2, 50) == approx(
        [
            [0.88304667090433, 0.022011915075971, 0.0949414140196994],
            [0.564123794006892, 0.0222190177517526, 0.413657188241355],
        ]
    )
    assert ctype2.cumhaz.sum() == approx(661.88083588846)
    p0 = r.survfit(fa1, newdata=ND_MG, p0=[0.6, 0.4, 0])
    assert row(p0, 50) == approx(
        [
            [0.518847060017648, 0.41637728361265, 0.0647756563697014],
            [0.309790644805848, 0.416206104052867, 0.274003251141285],
        ]
    )
    late = r.survfit(fa1, newdata=ND_MG, start_time=100)
    assert (len(late.time), late.time[0], late.t0) == (168, 101.0, 100.0)
    assert (late.n, late.n_id, late.start_time) == ([540], [540], 100.0)
    assert late.time[49] == 150
    assert row(late, 50) == approx(
        [
            [0.799648143743388, 0.0436584754234696, 0.156693380833142],
            [0.366276678077442, 0.0409795182636799, 0.592743803658878],
        ]
    )
    assert late.cumhaz.sum() == approx(427.321000092015)
    time0 = r.survfit(fa1, newdata=ND_MG, time0=True)
    assert len(time0.time) == 269
    assert time0.time[0] == 0
    assert row(time0, 1) == [[1.0, 0.0, 0.0]] * 2
    assert time0.n_risk[0] == [0, 0, 0]
    assert time0.time[49] == 49
    assert row(time0, 50) == approx(
        [
            [0.867407706062015, 0.0272954726877502, 0.105296821250234],
            [0.524926558611185, 0.0270101734214449, 0.44806326796737],
        ]
    )
    one = r.survfit(fa1, newdata={"age": 70, "sex": "M"})
    assert one.pstate.shape == (268, 1, 3)
    assert row(one, 50) == approx([[0.697829254034201, 0.0273342863706077, 0.274836459595191]])
    assert one.cumhaz.sum() == approx(358.334227277014)
    with pytest.warns(UserWarning, match="se.fit not yet implemented") as caught:
        with_se = r.survfit(fa1, newdata=ND_MG, se_fit=True)
    assert len(caught) == 1
    assert np.array_equal(with_se.pstate, sf.pstate)
    efron = r.survfit(r.coxph("Surv(etime, event) ~ age + sex", mg, id="id", ties="efron"), ND_MG)
    assert efron.ctype == 2
    assert row(efron, 50) == approx(
        [
            [0.883552937468631, 0.022018399663913, 0.0944286628674566],
            [0.563647987795609, 0.0222107973390567, 0.414141214865335],
        ]
    )
    assert efron.cumhaz.sum() == approx(663.138627470627)
    # the old-style type, and the method called directly
    assert np.array_equal(r.survfit(fa1, newdata=ND_MG, type="aalen").pstate, sf.pstate)
    with pytest.warns(RuntimeWarning, match="type argument ignored"):
        r.survfit(fa1, newdata=ND_MG, type="aalen", stype=2)
    assert np.array_equal(survfit_coxph(fa1, ND_MG).pstate, sf.pstate)


def test_weights_and_offsets(mg):
    """Deviation: the offsets are used (R computes and drops them).  The reference is
    R's survfit.coxphms patched to add the data and newdata offsets to the risk scores."""

    fit = r.coxph("Surv(etime, event) ~ age + sex + offset(o)", mg, id="id", weights="w")
    newdata = {"age": [60, 80], "sex": ["F", "M"], "o": [0, 0.5]}
    curves = r.survfit(fit, newdata=newdata)
    assert len(curves.time) == 268
    assert row(curves, 50) == approx(
        [
            [0.8731511625389, 0.0298239788732943, 0.097024858587806],
            [0.36159145257178, 0.0393474505085087, 0.599061096919711],
        ]
    )
    assert curves.cumhaz[49].tolist() == approx(
        [[0.0318194149171623, 0.103827170249096], [0.0623357678279279, 0.954904520714844]]
    )
    assert curves.n_risk[49] == [1404, 0, 0]
    assert row(curves, 268) == approx(
        [
            [0.0304555356942804, 0.26895317356131, 0.700591290744409],
            [1.39105137206697e-11, 0.0630483350775478, 0.936951664908542],
        ]
    )
    assert curves.cumhaz.sum() == approx(1263.20478257167)
    late = r.survfit(fit, newdata=newdata, start_time=100)
    assert len(late.time) == 168
    assert row(late, 50) == approx(
        [
            [0.792426849035069, 0.043882079241139, 0.163691071723792],
            [0.16717999482407, 0.0512912577549286, 0.781528747421001],
        ]
    )
    assert late.cumhaz.sum() == approx(711.811925708033)
    # R's own curves ignore the offsets
    unpatched = [
        [0.868380911799673, 0.0309325404360924, 0.100686547764235],
        [0.526250585212965, 0.0292390077771042, 0.444510407009931],
    ]
    assert row(curves, 50)[0] != approx(unpatched[0])
    assert row(curves, 50)[1] != approx(unpatched[1])


# ---------------------------------------------------------------------------
# 4-5. counting-process data, the Pade path and shared baselines
# ---------------------------------------------------------------------------


def test_counting_process_curves(fm):
    curves = r.survfit(fm, newdata=ND_MY)
    assert (len(curves.time), curves.time[0]) == (693, 4.0)
    assert curves.cumhaz_names == ["1:2", "1:3", "2:3"]
    assert (curves.n, curves.n_id, curves.type) == ([1009], [646], "mcounting")
    for number in (1, 2, 3):
        assert row(curves, number) == [[1.0, 0.0, 0.0]] * 2
    assert curves.time[49] == 79
    assert row(curves, 50) == approx(
        [
            [0.896143072152351, 0.0376821852521201, 0.0661747425955286],
            [0.92122600927329, 0.0326302658267148, 0.0461437248999946],
        ]
    )
    assert curves.cumhaz[49].tolist() == approx(
        [
            [0.0431108845859852, 0.066544315427051, 0.0832624743688602],
            [0.0363622370520703, 0.0456876402599987, 0.0773270164422025],
        ]
    )
    assert curves.n_risk[49] == [566, 20, 0]
    assert row(curves, 693) == approx(
        [
            [0.142186135016751, 0.250605431829478, 0.60720843315377],
            [0.214513602599104, 0.261214562769554, 0.524271834631341],
        ]
    )
    assert curves.cumhaz.sum() == approx(2434.60668251767)
    stype1 = r.survfit(fm, newdata=ND_MY, stype=1)
    assert row(stype1, 50) == approx(
        [
            [0.895933316931601, 0.0379038781486982, 0.0661628049197006],
            [0.921101344224592, 0.0328070776609025, 0.0460915781145047],
        ]
    )
    late = r.survfit(fm, newdata=ND_MY, start_time=50)
    assert (len(late.time), late.t0, late.n, late.n_id) == (662, 50.0, [956], [595])
    assert late.time[49] == 111
    assert row(late, 50) == approx(
        [
            [0.787118868632607, 0.180820241740548, 0.0320608896268447],
            [0.821105517199287, 0.156128047572837, 0.0227664352278754],
        ]
    )
    assert late.cumhaz.sum() == approx(2386.54453412716)


def test_shared_baselines(my):
    newdata = {"trt": ["A", "B"]}
    shared = r.coxph(["Surv(tstart, tstop, event) ~ trt", "1:3 + 2:3 ~ 1 / shared"], my, id="id")
    assert shared.share.scale == approx([1, 1, 1.65575258766549])
    curves = r.survfit(shared, newdata=newdata)
    assert row(curves, 50) == approx(
        [
            [0.894630371692792, 0.0387456604462777, 0.0666239678609302],
            [0.920072902511345, 0.0345431895698894, 0.045383907918766],
        ]
    )
    assert curves.cumhaz[49].tolist() == approx(
        [
            [0.0425716972354451, 0.0687729412987561, 0.113870975516782],
            [0.0369588224395555, 0.0463435477788087, 0.0767334491563618],
        ]
    )
    assert curves.cumhaz.sum() == approx(2489.88312820729)
    common = r.coxph(["Surv(tstart, tstop, event) ~ trt", "1:3 + 2:3 ~ 1 / common"], my, id="id")
    curves = r.survfit(common, newdata=newdata)
    assert row(curves, 50) == approx(
        [
            [0.888625498978777, 0.0386470952569987, 0.0727274057642241],
            [0.925671938419653, 0.0348518188247379, 0.0394762427556084],
        ]
    )
    assert np.array_equal(curves.cumhaz[:, :, 1], curves.cumhaz[:, :, 2])
    assert curves.cumhaz.sum() == approx(2419.79077219628)


def test_shared_baselines_out_of_transition_order(my):
    """Deviation: R's coxsurv2.c reads the transition of each stacked row in data order
    while walking them in sorted order, so a shared baseline of transitions that are
    not contiguous (smap 1 2 1) gets hazard 0 everywhere.  The reference is R with the
    stacked rows sorted by transition first."""

    fit = r.coxph(["Surv(tstart, tstop, event) ~ trt", "1:2 + 2:3 ~ 1 / shared"], my, id="id")
    curves = r.survfit(fit, newdata={"trt": ["A", "B"]})
    assert row(curves, 50) == approx(
        [
            [0.89480656659285, 0.0399105009858224, 0.0652829324213274],
            [0.920328804604818, 0.0349457616693837, 0.0447254337257983],
        ]
    )
    assert curves.cumhaz[49].tolist() == approx(
        [
            [0.044183306094083, 0.0669644046838415, 0.0354266868138494],
            [0.037700309854396, 0.045323966625918, 0.0302285453051607],
        ]
    )
    assert row(curves, 693) == approx(
        [
            [0.117018090758975, 0.299381986694065, 0.58359992254696],
            [0.184948348518944, 0.337235402647487, 0.477816248833571],
        ]
    )
    assert curves.cumhaz.sum() == approx(2627.55700811616)


# ---------------------------------------------------------------------------
# 6. coxsurv5.R hand checks
# ---------------------------------------------------------------------------


def test_zero_coefficients_give_the_nelson_aalen_hazards():
    fit = r.coxph("Surv(t1, t2, state) ~ x", _mtest(), id="id", iter_max=0)
    curves = r.survfit(fit, newdata={"x": [1, 2]})
    assert len(curves.time) == 8
    assert curves.cumhaz_names == ["1:2", "3:2", "1:3", "2:3", "1:4", "2:4"]
    assert curves.time[4] == 8
    for j in range(2):
        assert curves.cumhaz[4, j].tolist() == approx([0.583333333333333, 0, 0.75, 0, 0, 0.5])
        assert curves.pstate[4, j].tolist() == approx(
            [0.263597138115727, 0.238446410027334, 0.343271193750123, 0.154685258106816]
        )
    assert curves.cumhaz.sum() == approx(34.5)
    init = r.coxph(
        "Surv(t1, t2, state) ~ x", _mtest(), id="id", iter_max=0, init=np.log(np.arange(1, 7))
    )
    curves = r.survfit(init, newdata={"x": [0, 1]})
    assert curves.cumhaz[4].tolist() == approx(
        [
            [0.583333333333333, 0, 0.229166666666667, 0, 0, 0.5],
            [0.583333333333333, 0, 0.6875, 0, 0, 3],
        ]
    )
    assert row(curves, 5) == approx(
        [
            [0.44374731008108, 0.259952575396709, 0.127663349489088, 0.168636765033124],
            [0.280597692910797, 0.0201249466265661, 0.315181948544129, 0.384095411918508],
        ]
    )


def test_test2_exact_hazards():
    cox3 = r.coxph("Surv(t1, t2, state) ~ x", _test2(), id="id", iter_max=0)
    curves = r.survfit(cox3, newdata={"x": [0, 1]}, time0=False)
    assert curves.time == [4, 5, 6, 8, 10, 15, 16, 18, 20, 22]
    hazard = np.zeros((10, 6))
    hazard[[0, 3], 0] = [1 / 6, 1 / 5]
    hazard[8, 1] = 1 / 2
    hazard[[2, 4], 2] = 1 / 5
    hazard[7, 3:5] = 1
    hazard[6, 5] = 1 / 2
    for j in range(2):
        assert curves.cumhaz[:, j].tolist() == approx(np.cumsum(hazard, axis=0).tolist())
        assert curves.pstate[-1, j].tolist() == approx(
            [0.170901712801525, 0.20524106765477, 0.220364823908806, 0.403492395634899]
        )
    cox4 = r.coxph(
        "Surv(t1, t2, state) ~ x", _test2(), id="id", iter_max=0, init=np.log(np.arange(1, 7))
    )
    curves = r.survfit(cox4, newdata={"x": [0, 1]}, time0=False)
    assert curves.cumhaz[9].tolist() == approx(
        [
            [0.366666666666667, 0.2, 0.105263157894737, 1, 0.2, 0.5],
            [0.366666666666667, 0.4, 0.315789473684211, 4, 1, 3],
        ]
    )
    assert curves.cumhaz[:, 1].tolist() == approx((curves.cumhaz[:, 0] * np.arange(1, 7)).tolist())
    assert row(curves, 10) == approx(
        [
            [0.510722022250581, 0.101500799635976, 0.157018234382367, 0.230758943731076],
            [0.185916777098732, 0.0741029720312373, 0.150141907168162, 0.589838343701869],
        ]
    )
    stype1 = r.survfit(cox4, newdata={"x": [0, 1]}, stype=1)
    assert row(stype1, 10) == approx(
        [
            [0.478670360110803, 0.047876269621422, 0.191505078485688, 0.281948291782087],
            [0, 0.947737765466297, -1.3415512465374, 1.3938134810711],
        ]
    )


# ---------------------------------------------------------------------------
# 7-8. mstrata.R: proportional baselines and strata
# ---------------------------------------------------------------------------


def test_proportional_baselines_reproduce_single_state_curves(lms):
    single = r.coxph("Surv(time, status) ~ age + sex + factor(ph.ecog)", lms, ties="breslow")
    shared = r.coxph(
        ["Surv(time, state) ~ 1", "1:4 + 2:4 + 3:4 ~ age + sex / common + shared"],
        lms,
        id="id",
        istate="cstate",
        ties="breslow",
    )
    heads = [
        [0.996584521755413, 0.986387793934833, 0.982948109359456, 0.976039474403888],
        [0.994860593296222, 0.979570209430584, 0.974430245517798, 0.96413407695758],
        [0.991614378056953, 0.966828513502418, 0.958551353358232, 0.942054095195622],
    ]
    for state in range(3):
        p0 = [0.0] * 4
        p0[state] = 1.0
        curves = r.survfit(shared, newdata={"age": 65, "sex": 1}, p0=p0)
        cox = r.survfit(single, newdata={"age": [65], "sex": [1], "ph.ecog": [state]})
        assert curves.pstate[:, 0, state].tolist() == approx(cox.surv)
        assert curves.pstate[:4, 0, state].tolist() == approx(heads[state])
    curves = r.survfit(shared, newdata={"age": 65, "sex": 1}, p0=[1, 0, 0, 0])
    assert curves.pstate.shape == (184, 1, 4)
    assert curves.time[49] == 185
    assert curves.cumhaz[49, 0].tolist() == approx(
        [0.277817041770344, 0.418404196527202, 0.683797057865366]
    )
    assert curves.cumhaz.sum() == approx(614.554222306275)
    assert curves.n_risk[49] == [52, 79, 24, 0]


def test_strata(mg, fst, sfs):
    assert len(sfs.time) == 454
    assert sfs.strata == {"F": 227, "M": 227}
    assert (sfs.n, sfs.n_id) == ([627, 746], [627, 746])
    assert sfs.p0 == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert sfs.pstate.shape == (454, 2, 3)
    assert sfs.dim == {"strata": 2, "data": 2, "states": 3}
    assert sfs.time[49] == 51
    assert row(sfs, 50) == approx(
        [
            [0.866818612686145, 0.0286280371800073, 0.104553350133848],
            [0.63650018385509, 0.0342678837812336, 0.329231932363676],
        ]
    )
    assert sfs.cumhaz.sum() == approx(1044.17457124504)
    # the F block is a fit on the women alone with the stratified coefficients
    women = r.coxph(
        "Surv(etime, event) ~ age + mspike",
        mg[mg.sex == "F"],
        id="id",
        init=fst.coefficients,
        iter_max=0,
    )
    alone = r.survfit(women, newdata={"age": [60, 80], "mspike": [1.2, 1.2]})
    assert alone.n == [627]
    assert sfs.time[:227] == alone.time
    for name in ("n_risk", "n_event", "n_censor"):
        assert getattr(sfs, name)[:227] == getattr(alone, name)
    assert sfs.pstate[:227].tolist() == approx(alone.pstate.tolist())
    assert sfs.cumhaz[:227].tolist() == approx(alone.cumhaz.tolist())
    assert row(alone, 227) == approx(
        [
            [0.129533207443701, 0.246902733485289, 0.623564059071009],
            [0.00385602882559278, 0.0927235585618472, 0.90342041261256],
        ]
    )
    assert alone.cumhaz.sum() == approx(436.929254570679)


def test_partially_stratified_model_counts_every_transition_per_stratum(my):
    """As R does: the 1:2 hazard is estimated within each sex, although the fit pools
    its baseline over sex."""

    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt + strata(sex)", "1:2 ~ -strata(sex)"], my, id="id"
    )
    curves = r.survfit(fit, newdata={"trt": ["A", "B"]})
    assert len(curves.time) == 819
    assert curves.strata == {"f": 451, "m": 368}
    assert (curves.n, curves.n_id) == ([564, 445], [361, 285])
    assert curves.time[49] == 98
    assert row(curves, 50) == approx(
        [
            [0.819533479941098, 0.115781839966312, 0.0646846800925896],
            [0.851520568491675, 0.104271613340051, 0.0442078181682742],
        ]
    )
    assert curves.cumhaz[450].tolist() == approx(
        [
            [1.28723892067564, 0.688363489689459, 1.2207257033064],
            [1.11752262174141, 0.466167730562052, 0.901840556754055],
        ]
    )
    assert curves.cumhaz[818].tolist() == approx(
        [
            [1.21694445309212, 0.662610911137554, 1.15278308343854],
            [1.05649614371461, 0.448727785998637, 0.851646307576703],
        ]
    )
    assert curves.cumhaz.sum() == approx(2684.46234263243)


# ---------------------------------------------------------------------------
# 9-10. summary, survfit0 and the data frame layout
# ---------------------------------------------------------------------------


def test_summary_at_times(sf):
    summary = r.summary_survfit(sf, times=[100, 200, 400])
    assert isinstance(summary, r.SummarySurvfitCoxmsResult)
    assert summary.time == [100, 200, 400]
    assert summary.n_risk == [[547, 0, 0], [124, 0, 0], [1, 0, 0]]
    assert summary.n_event == [[0, 73, 634], [0, 34, 197], [0, 8, 28]]
    assert summary.n_censor == [[137, 0, 0], [188, 0, 0], [84, 0, 0]]
    assert summary.pstate.shape == (3, 2, 3)
    # R's column-major order: time fastest, then newdata row, then state
    assert summary.pstate.ravel(order="F").tolist() == approx(
        [
            0.728209802161815,
            0.458713971903348,
            0.122171381149787,
            0.243920455771018,
            0.0322052258324455,
            0.00055617538505226,
            0.0612795642283346,
            0.126037438028164,
            0.274908729652618,
            0.0473810722855863,
            0.0618188184536305,
            0.0640995973533414,
            0.210510633609851,
            0.415248590068488,
            0.602919889197596,
            0.708698471943396,
            0.905975955713924,
            0.935344227261607,
        ]
    )
    assert summary.cumhaz.ravel(order="F").tolist() == approx(
        [
            0.071732248668087,
            0.183847129317913,
            0.927159013954037,
            0.090790927672145,
            0.232693826424823,
            1.17349767419083,
            0.24543383385335,
            0.595481288740056,
            1.17517144252922,
            1.3201221801051,
            3.20293272024194,
            6.32092919852236,
        ]
    )
    table = summary.table
    assert table.rownames == ["1, (s0)", "2, (s0)", "1, pcm", "2, pcm", "1, death", "2, death"]
    assert table.colnames == ["n", "nevent", "rmean"]
    assert table.values == approx(
        [
            [1384, 0, 202.128239575905],
            [1384, 0, 66.6464519075665],
            [1384, 115, 55.3156851393169],
            [1384, 115, 22.5733058784238],
            [1384, 860, 166.556075284778],
            [1384, 860, 334.78024221401],
        ]
    )


def test_summary_at_event_times(sf):
    summary = r.model_summary(sf)
    assert len(summary.time) == 214
    assert summary.time[:5] == [1, 2, 3, 4, 5]
    assert summary.pstate.shape == (214, 2, 3)
    assert summary.rmean_endtime == [424]
    assert r.summary_survfit(sf, rmean="none").rmean_endtime is None


def test_summary_data_frame(sf):
    frame = r.as_data_frame(r.summary_survfit(sf, times=[100, 200]))
    assert list(frame) == ["time", "n.risk", "n.event", "n.censor", "pstate", "state", "age", "sex"]
    assert frame["time"] == [100, 200] * 6
    assert frame["n.risk"] == [547, 124, 547, 124] + [0] * 8
    assert frame["n.event"] == [0, 0, 0, 0, 73, 34, 73, 34, 634, 197, 634, 197]
    assert frame["n.censor"] == [137, 188, 137, 188] + [0] * 8
    assert frame["pstate"] == approx(
        [
            0.728209802161815,
            0.458713971903348,
            0.243920455771018,
            0.0322052258324455,
            0.0612795642283346,
            0.126037438028164,
            0.0473810722855863,
            0.0618188184536305,
            0.210510633609851,
            0.415248590068488,
            0.708698471943396,
            0.905975955713924,
        ]
    )
    assert frame["state"] == ["(s0)"] * 4 + ["pcm"] * 4 + ["death"] * 4
    assert frame["age"] == [60, 60, 80, 80] * 3
    assert frame["sex"] == ["F", "F", "M", "M"] * 3
    # the curves themselves: summary(censored = TRUE, data.frame = TRUE)
    assert len(r.as_data_frame(sf)["time"]) == 268 * 2 * 3


def test_survfit0(fa1, sf):
    zero = r.survfit0(sf)
    time0 = r.survfit(fa1, newdata=ND_MG, time0=True)
    assert zero.time0
    assert zero.time == time0.time
    assert np.array_equal(zero.pstate, time0.pstate)
    assert np.array_equal(zero.cumhaz, time0.cumhaz)
    assert (zero.n_event, zero.n_censor) == (time0.n_event, time0.n_censor)
    assert zero.n_transition == time0.n_transition
    # survfit0 gives its t0 row the first row's number at risk; survfitAJ's time0 row
    # has none
    assert zero.n_risk[0] == [1384, 0, 0]
    assert time0.n_risk[0] == [0, 0, 0]
    assert r.survfit0(time0) is time0


def test_stratified_summaries(sfs):
    """Deviation: survmean2 reports each stratum's own event counts; R's
    ``rep(c(nevent), each = ndata)`` misaligns them."""

    table = r.summary_survfit(sfs).table
    groups = [f"{s}, {i}" for i in (1, 2) for s in ("F", "M")]
    assert table.rownames == [f"{g}, {state}" for state in sfs.states for g in groups]
    values = np.array(table.values)
    assert values[:, 0].tolist() == [627, 746] * 6
    assert values[:, 2].tolist() == approx(
        [
            206.715562029669,
            176.68967431913,
            90.940623011749,
            67.2519253597446,
            47.4340160754926,
            37.5438413153094,
            30.3901909623607,
            20.9329198240534,
            169.850421894838,
            209.766484365561,
            302.66918602589,
            335.815154816202,
        ]
    )
    assert values[:, 1].tolist() == [0] * 4 + [59, 56] * 2 + [368, 486] * 2
    summary = r.summary_survfit(sfs, times=[100, 200])
    assert summary.time == [100, 200, 100, 200]
    assert summary.strata == ["F", "F", "M", "M"]
    assert summary.n_risk == [[282, 0, 0], [66, 0, 0], [263, 0, 0], [58, 0, 0]]
    assert summary.n_event == [[0, 38, 254], [0, 15, 101], [0, 35, 375], [0, 19, 95]]
    assert summary.pstate.shape == (4, 2, 3)
    assert summary.pstate.ravel(order="F").tolist() == approx(
        [
            0.740590997458245,
            0.464531377114512,
            0.649713340218846,
            0.364604602871857,
            0.386725156956234,
            0.0827708036721864,
            0.239172175244892,
            0.0381661499431241,
            0.0557994587233662,
            0.0996615311480805,
            0.0500432025693536,
            0.106795091882037,
            0.0585476276860634,
            0.0812111737430836,
            0.0429055328640814,
            0.0589417779403732,
            0.203609543818389,
            0.435807091737408,
            0.3002434572118,
            0.528600305246105,
            0.554727215357703,
            0.83601802258473,
            0.717922291891027,
            0.902892072116503,
        ]
    )


def test_summarydf_cox4(fst):
    curves = r.survfit(fst, newdata={"age": [70, 80, 90] * 2, "mspike": [0, 0, 0, 1, 1, 1]})
    assert curves.dim == {"strata": 2, "data": 6, "states": 3}
    summary = r.summary_survfit(curves, times=[12, 120, 360])
    frame = r.as_data_frame(summary)
    assert list(frame) == [
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "pstate",
        "strata",
        "state",
        "age",
        "mspike",
    ]
    assert len(frame["time"]) == 108
    expected = {
        1: (12, 559, 0, 2, 0.923351405847513, "F", "(s0)", 70, 0),
        2: (120, 212, 0, 85, 0.519153329942972, "F", "(s0)", 70, 0),
        3: (360, 2, 0, 112, 0.0698378607821909, "F", "(s0)", 70, 0),
        4: (12, 639, 0, 0, 0.873842226822024, "M", "(s0)", 70, 0),
        7: (12, 559, 0, 2, 0.861404841137128, "F", "(s0)", 80, 0),
        36: (360, 1, 0, 114, 1.16462627399401e-05, "M", "(s0)", 90, 1),
        37: (12, 0, 8, 0, 0.00398257459994169, "F", "pcm", 70, 0),
        108: (360, 0, 78, 0, 0.971816661820766, "M", "death", 90, 1),
    }
    for number, values in expected.items():
        got = tuple(frame[name][number - 1] for name in frame)
        assert got[:4] == values[:4]
        assert got[5:] == values[5:]
        assert got[4] == approx(values[4])
    table = summary.table
    assert table.rownames[:12] == [f"{s}, {i}, (s0)" for i in range(1, 7) for s in ("F", "M")]
    assert [row[2] for row in table.values[:12]] == approx(
        [
            147.289428030512,
            112.937270212064,
            90.3941046767168,
            65.4065739060483,
            53.0266786354967,
            35.7415938289469,
            145.18447277883,
            113.724022123018,
            91.2174797683919,
            67.1373752673492,
            54.3657994618182,
            37.3026110531961,
        ]
    )


# ---------------------------------------------------------------------------
# 11. aggregate, quantile, residuals, subsetting
# ---------------------------------------------------------------------------


def test_aggregate(fa1, sf):
    """Deviation: without ``by`` the data axis keeps length 1 (R returns a time x state
    matrix)."""

    pooled = r.aggregate_survfit(sf)
    assert pooled.pstate.shape == (268, 1, 3)
    assert pooled.pstate[49, 0].tolist() == approx(
        [0.690531420686247, 0.0271528230545975, 0.282315756259156]
    )
    assert pooled.cumhaz is None
    assert pooled.newdata is None
    assert pooled.n_transition == sf.n_transition
    sex = ["F", "M", "F", "M"]
    sf4 = r.survfit(fa1, newdata={"age": [60, 70, 80, 90], "sex": sex})
    grouped = r.aggregate_survfit(sf4, by=sex)
    assert grouped.pstate.shape == (268, 2, 3)
    assert grouped.pstate[49].tolist() == approx(
        [
            [0.748070549030124, 0.0288707804896487, 0.223058670480227],
            [0.494771119365984, 0.0255462092519566, 0.479682671382059],
        ]
    )
    assert grouped.newdata == {"aggregate": ["F", "M"]}
    assert grouped.dim == {"data": 2, "states": 3}
    assert r.aggregate_survfit(sf4, by={"sex": sex}).newdata == {"sex": ["F", "M"]}
    largest = r.aggregate_survfit(sf4, by=sex, FUN="max")
    assert largest.pstate[49].tolist() == approx(
        [
            [0.864745100029414, 0.0304460882915472, 0.338157913677619],
            [0.697829254034201, 0.0273342863706077, 0.684528883168927],
        ]
    )
    assert r.summary_survfit(grouped).table.rownames[:2] == ["1, (s0)", "2, (s0)"]


def test_quantile_and_residuals_refuse(sf):
    with pytest.raises(ValueError, match="quantiles are not a well defined quantity"):
        r.quantile_survfit(sf)
    # deviation: R returns Aalen-Johansen residuals that ignore the Cox model
    with pytest.raises(TypeError, match="residuals are not defined for multi-state Cox"):
        r.residuals(sf, times=100)


def test_subsetting(sf, sfs):
    second = _subset_coxms_curves(sf, data=[1])
    assert second.pstate.shape == (268, 1, 3)
    assert second.cumhaz.shape == (268, 1, 2)
    assert second.pstate[49, 0].tolist() == approx(
        [0.516317741343079, 0.0270101734214449, 0.456672085235476]
    )
    assert second.newdata == {"age": [80], "sex": ["M"]}
    assert second.transitions is None
    assert second.dim == {"data": 1, "states": 3}
    assert second.engine is not None
    death = _subset_coxms_curves(sf, states=["death"])
    assert death.pstate.shape == (268, 2, 1)
    assert death.cumhaz is None
    assert death.n_transition is None
    assert death.states == ["death"]
    assert death.oldstate == ("(s0)", "pcm", "death")
    assert np.shape(death.n_risk) == (268, 1)
    assert np.shape(death.n_censor) == (268, 3)
    assert death.p0 == [[0.0]]
    # deviation: n_id stays (R drops it)
    assert death.n_id == [1384]
    with pytest.raises(ValueError, match="summary of a state subset"):
        r.summary_survfit(death)
    with pytest.raises(ValueError, match="survfit0 of a state subset"):
        r.survfit0(death)
    part = _subset_coxms_curves(sf, data=[0], states=["pcm", "death"])
    assert part.pstate[49, 0].tolist() == approx([0.0272954726877502, 0.107959427282836])
    men = _subset_coxms_curves(sfs, strata=["M"])
    assert men.pstate.shape == (227, 2, 3)
    assert (men.strata, men.n, men.p0) == ({"M": 227}, [746], [[1.0, 0.0, 0.0]])
    assert men.pstate[49].tolist() == approx(
        [
            [0.821278698601375, 0.0216163559070182, 0.157104945491607],
            [0.514985376803818, 0.0234239568748938, 0.461590666321289],
        ]
    )
    assert _subset_coxms_curves(sfs, strata=["F"], data=[1]).pstate.shape == (227, 1, 3)


def test_stratum_subset_summary_and_survfit0(sfs):
    """R's ``summary(sfs[2, , ])`` and ``survfit0(sfs[2, , ])``: one stratum keeps its
    label, and survmean2 drops it from the table's row names."""

    men = _subset_coxms_curves(sfs, strata=["M"])
    summary = r.summary_survfit(men, times=[100, 200])
    assert (summary.time, summary.strata) == ([100, 200], ["M", "M"])
    assert summary.n_risk == [[263, 0, 0], [58, 0, 0]]
    assert summary.n_event == [[0, 35, 375], [0, 19, 95]]
    assert summary.pstate.ravel(order="F").tolist() == approx(
        [
            0.6497133402188464,
            0.3646046028718571,
            0.2391721752448915,
            0.0381661499431241,
            0.0500432025693536,
            0.1067950918820372,
            0.0429055328640814,
            0.0589417779403732,
            0.3002434572117996,
            0.5286003052461055,
            0.7179222918910271,
            0.9028920721165029,
        ]
    )
    table = summary.table
    assert table.rownames == [f"{i}, {state}" for state in men.states for i in (1, 2)]
    values = np.array(table.values)
    assert values[:, 0].tolist() == [746] * 6
    assert values[:, 1].tolist() == [0, 0, 56, 56, 486, 486]
    assert values[:, 2].tolist() == approx(
        [
            176.6896743191297,
            67.2519253597446,
            37.5438413153094,
            20.9329198240534,
            209.7664843655609,
            335.8151548162019,
        ]
    )
    everything = r.summary_survfit(men)
    assert everything.pstate.shape == (183, 2, 3)
    assert everything.strata[:3] == ["M", "M", "M"]
    zero = r.survfit0(men)
    assert zero.strata == {"M": 228}
    assert zero.time[:3] == [0, 1, 2]
    assert zero.n_risk[:2] == [[746, 0, 0], [746, 0, 0]]
    assert zero.pstate.shape == (228, 2, 3)
    assert zero.pstate[0].tolist() == [[1, 0, 0], [1, 0, 0]]
    frame = r.as_data_frame(men)
    assert len(frame["time"]) == 227 * 2 * 3
    assert set(frame["strata"]) == {"M"}


# ---------------------------------------------------------------------------
# 12. missing values
# ---------------------------------------------------------------------------


def test_newdata_with_missing_values(fa1, sf):
    """Deviations: ``newdata`` holds the rows used, and na.pass refuses an incomplete
    row (R segfaults)."""

    newdata = {"age": [60, None, 80], "sex": ["F", "M", "M"]}
    for action in (None, "na.exclude"):
        curves = r.survfit(fa1, newdata=newdata, na_action=action)
        assert np.array_equal(curves.pstate, sf.pstate)
        assert curves.newdata == {"age": [60, 80], "sex": ["F", "M"]}
    with pytest.raises(ValueError, match="missing values in object"):
        r.survfit(fa1, newdata=newdata, na_action="na.fail")
    with pytest.raises(ValueError, match="na_action='na.omit'"):
        r.survfit(fa1, newdata=newdata, na_action="na.pass")


def test_formula_list_with_missing_covariates(my_na):
    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", "1:3 + 2:3 ~ sex", "1:2 + 2:3 ~ flt3"],
        my_na,
        id="id",
    )
    curves = r.survfit(fit, newdata={"trt": ["A", "B"], "sex": ["f", "m"], "flt3": ["A", "C"]})
    assert (len(curves.time), curves.n, curves.n_id) == (691, [1006], [645])
    assert curves.time[49] == 79
    assert row(curves, 50) == approx(
        [
            [0.90852715142953, 0.0264647726123451, 0.0650080759581248],
            [0.915663013001918, 0.0366620929980517, 0.0476748940000305],
        ]
    )
    assert curves.cumhaz[49].tolist() == approx(
        [
            [0.0296967772097078, 0.066233728384399, 0.0632450207789335],
            [0.0416742046273351, 0.0464326671045998, 0.104326921140831],
        ]
    )
    assert curves.n_risk[49] == [565, 20, 0]
    assert row(curves, 691) == approx(
        [
            [0.210239801522947, 0.259065494173274, 0.53069470430378],
            [0.178184746096183, 0.222125680646477, 0.59968957325734],
        ]
    )
    assert curves.cumhaz.sum() == approx(2335.16427620597)


def test_after_a_first_level_drop(mg):
    """R fails here ("Failed to reconstruct the original data set"); the reference is
    R on the data without the two rows."""

    data = mg.copy()
    data.loc[[1, 4], "etime"] = np.nan
    fit = r.coxph(["Surv(etime, event) ~ age + sex", "1:3 ~ mspike"], data, id="id")
    curves = r.survfit(fit, newdata={"age": [60, 80], "sex": ["F", "M"], "mspike": [1, 2]})
    assert (len(curves.time), curves.n, curves.n_id) == (268, [1382], [1382])
    assert row(curves, 50) == approx(
        [
            [0.865286611088358, 0.0273270961458592, 0.107386292765782],
            [0.536066061144826, 0.0275118891811591, 0.436422049674015],
        ]
    )
    assert curves.cumhaz.sum() == approx(771.942231627105)


# ---------------------------------------------------------------------------
# 13. pbcseq: coxsurv6.R and timeline.R
# ---------------------------------------------------------------------------


def test_shared_hazard_without_events(pbc2):
    pbc3 = pbc2[pbc2.id < 10].copy()
    pbc3["age"] = pbc3.age.round()
    fit = r.coxph(
        ["Surv(tstart, tstop, bstat) ~ 1", "c(1:4):5 ~ age / common + shared"],
        pbc3,
        id="id",
        istate="bili4",
        init=[0.05, 0.7, 1.5, 3.4],
        iter_max=0,
    )
    curves = r.survfit(fit, newdata={"age": 50}, p0=[0.4, 0.3, 0.2, 0.1, 0])
    assert curves.time == [
        182,
        184,
        199,
        391,
        392,
        400,
        545,
        729,
        768,
        769,
        1012,
        1126,
        1505,
        1559,
        1790,
        1824,
        1925,
        2218,
        2400,
        2466,
        2501,
        2503,
        2515,
        2882,
        3226,
        5169,
    ]
    assert curves.cumhaz_names == [
        "2:1",
        "1:2",
        "3:2",
        "2:3",
        "4:3",
        "2:4",
        "3:4",
        "1:5",
        "2:5",
        "3:5",
        "4:5",
    ]
    assert curves.pstate[:3, 0].tolist() == approx(
        [
            [0.485040606827863, 0.214959393172137, 0.2, 0.1, 0],
            [0.485040606827863, 0.214959393172137, 0.121306131942527, 0.178693868057473, 0],
            [0.485040606827863, 0.291639493084977, 0.044626032029686, 0.178693868057473, 0],
        ]
    )
    assert curves.pstate[-1, 0].tolist() == approx(
        [
            0.129137425527828,
            0.131525146069107,
            0.0194168071124491,
            0.049764137986912,
            0.670156483303704,
        ]
    )
    # 3:5 has no event of its own, yet its shared hazard is positive
    assert curves.cumhaz[-1, 0].tolist() == approx(
        [
            0.666666666666667,
            1.33333333333333,
            1,
            1.16666666666667,
            1,
            0.5,
            3.5,
            0.200726133320012,
            0.404212794433255,
            0.899592117831521,
            6.014577940928,
        ]
    )
    assert curves.cumhaz.sum() == approx(179.767439810033)


def test_timeline_data(pdata):
    fit = r.coxph(["Surv2(day, bstat) ~ 1", "(1:3):4 ~ edema + albumin + ast"], pdata, id="id")
    newdata = {"edema": [0, 1, 0, 1], "albumin": [3] * 4, "ast": [75, 75, 150, 150]}
    curves = r.survfit(fit, newdata=newdata)
    assert len(curves.time) == 542
    assert curves.pstate.shape == (542, 4, 4)
    assert curves.cumhaz_names == ["2:1", "3:1", "1:2", "3:2", "1:3", "2:3", "1:4", "2:4", "3:4"]
    assert curves.cumhaz.sum() == approx(9868.86532707371)
    assert curves.time[49] == 299
    assert row(curves, 50) == approx(
        [
            [0.384159369566762, 0.364319345761874, 0.204899412768364, 0.0466218719029995],
            [0.371427038769773, 0.234779748627947, 0.155957431377278, 0.237835781225002],
            [0.385085412708255, 0.374307429109533, 0.205827997442249, 0.0347791607399618],
            [0.378207424343957, 0.302009656997821, 0.161480793018164, 0.158302125640059],
        ]
    )
    assert row(curves, 542) == approx(
        [
            [0.242152370692434, 0.134759537080291, 0.193269912258668, 0.429818179968607],
            [0.0881500746842112, 0.0272842065332982, 0.0213944574146225, 0.863171261367868],
            [0.257998660678225, 0.148825148975639, 0.206680051540112, 0.386496138806024],
            [0.138352808086619, 0.059074672698249, 0.0370226453490334, 0.765549873866099],
        ]
    )
    counting = pd.DataFrame(dict(r.fromtimeline("Surv2(day, bstat) ~ .", pdata, id="id")))
    counting["bstat"] = cat(counting.bstat, PDATA_STATES)
    fit2 = r.coxph(
        ["Surv(day1, day2, bstat) ~ 1", "(1:3):4 ~ edema + albumin + ast"],
        counting,
        id="id",
        istate="istate",
    )
    same = r.survfit(fit2, newdata=newdata)
    assert same.time == curves.time
    assert same.n_risk == curves.n_risk
    assert same.pstate.tolist() == approx(curves.pstate.tolist())
    assert same.cumhaz.tolist() == approx(curves.cumhaz.tolist())


# ---------------------------------------------------------------------------
# 14-15. errors and pickling
# ---------------------------------------------------------------------------


def test_errors(mg, fa1, fst):
    cases = [
        (lambda: r.survfit(fa1), "multi-state survival requires a newdata argument"),
        (lambda: r.survfit(fa1, newdata=ND_MG, id=[1, 2]), "covariate path is not supported"),
        (lambda: r.survfit(fa1, newdata=ND_MG, individual=True), "covariate path"),
        (
            lambda: r.survfit(fa1, newdata=ND_MG, start_time=[1, 2]),
            "start.time must be a single numeric value",
        ),
        (
            lambda: r.survfit(fa1, newdata=ND_MG, start_time=1e6),
            "start.time has removed all observations",
        ),
        (lambda: r.survfit(fa1, newdata=ND_MG, ctype=3), "ctype must be 1 or 2"),
        (lambda: r.survfit(fa1, newdata=ND_MG, stype=3), "stype must be 1 or 2"),
        (
            lambda: r.survfit(fa1, newdata={"age": [np.nan], "sex": ["F"]}),
            "all rows of newdata have missing values",
        ),
        (
            lambda: r.survfit(fa1, newdata=ND_MG, p0=[0.5, 0.2, 0.2]),
            "p0 must be a numeric vector that adds to 1",
        ),
        (
            lambda: r.survfit(fst, newdata={"age": [60], "mspike": [1], "sex": ["X"]}),
            "factor strata\\(sex\\) has new level X",
        ),
    ]
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()
    # coxph lets a gap through, survfitAJ does not
    gap = r.coxph("Surv(t1, t2, state) ~ x", _mtest(t1_row3=9.5), id="id", iter_max=0)
    with pytest.raises(ValueError, match="one or more flags are >0 in survcheck"):
        r.survfit(gap, newdata={"x": [1]})


def test_pickle(sf):
    restored = pickle.loads(pickle.dumps(sf))  # noqa: S301 - the test's own pickle
    assert restored == sf
    assert np.array_equal(restored.pstate, sf.pstate)
    assert np.array_equal(restored.cumhaz, sf.cumhaz)
    assert restored.engine is not None
    assert r.summary_survfit(restored).table == r.summary_survfit(sf).table
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert r.survfit0(restored).time == r.survfit0(sf).time
