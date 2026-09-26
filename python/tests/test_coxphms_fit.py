"""Multi-state ``coxph`` (R's ``coxphms``): the fit, formula lists, missing values,
``coef``/``vcov(matrix=)`` and ``summary``.

Reference values are R 4.5.3 / survival 3.8-12 on the data sets of R's tests
multi2.R, multi3.R, multistate.R, mstrata.R, residms.R, coxsurv5.R, coxsurv6.R and
timeline.R, rebuilt below with R's row order and factor levels.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


def diag(matrix):
    return [row[i] for i, row in enumerate(matrix)]


def blocks(fit):
    return np.bincount(fit.rmap[:, 1])[1:].tolist()


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
    m["grp"] = m.id % 50
    m["entry"] = "Entry"
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
    states = ["censor", "normal", "1-4", "4+", "death"]
    data["bstat"] = cat(np.array(states)[data.bstat.astype(int)], states)
    return data


@pytest.fixture(scope="module")
def mg_fit(mg):
    return r.coxph("Surv(etime, event) ~ age + sex", mg, id="id")


@pytest.fixture(scope="module")
def common_fit(mg):
    return r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ sex / common"], mg, id="id")


@pytest.fixture(scope="module")
def shared_fit(mg):
    return r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ 1 / shared"], mg, id="id")


# ---------------------------------------------------------------------------
# single-formula fits
# ---------------------------------------------------------------------------


def test_competing_risks_fit(mg_fit):
    fit = mg_fit
    assert isinstance(fit, r.CoxphmsModel)
    assert isinstance(fit, r.CoxphModel)
    assert fit.coef_names == ("age_1:2", "sexM_1:2", "age_1:3", "sexM_1:3")
    assert fit.coefficients == approx(
        [0.0130377951641942, -0.0251369568806912, 0.0645438014943818, 0.391576147058642]
    )
    assert fit.loglik == approx([-6351.08896303735, -6157.72345973042])
    assert fit.score == approx(333.199561019607)
    assert fit.rscore == approx(286.397150075723, rel=1e-6)
    assert fit.wald_test == approx(327.194994021978, rel=1e-6)
    assert (fit.iter, fit.method) == (4, "breslow")
    assert diag(fit.var) == approx(
        [4.45427749400815e-05, 0.0358314862601605, 1.3490345349682e-05, 0.004332719938041],
        rel=1e-6,
    )
    assert diag(fit.naive_var) == approx(
        [6.82128634659839e-05, 0.0355150565516966, 1.30800763318094e-05, 0.00485777217322188]
    )
    assert fit.means == approx([70.4234104046243, 0.0])
    assert (fit.n, fit.nevent, fit.n_id) == (1384, 975, 1384)
    concordance = fit.concordance
    assert [concordance[k] for k in ("concordant", "discordant", "tied.x", "tied.y")] == [
        517190,
        264170,
        9658,
        3182,
    ]
    assert concordance["tied.xy"] == 49
    assert concordance["concordance"] == approx(0.659933149435285)
    assert concordance["std"] == approx(0.00970559511372908, rel=1e-6)
    assert fit.states == ("(s0)", "pcm", "death")
    assert fit.cmap == r.NamedMatrix(["age", "sexM"], ["1:2", "1:3"], [[1, 3], [2, 4]])
    assert fit.smap == r.NamedMatrix(["(Baseline)"], ["1:2", "1:3"], [[1, 2]])
    assert fit.transitions == r.NamedMatrix(
        ["(s0)", "pcm", "death"],
        ["pcm", "death", "(censored)"],
        [[115, 860, 409], [0, 0, 0], [0, 0, 0]],
    )
    assert fit.rmap.shape == (2768, 2)
    assert fit.rmap[:3].tolist() == [[1, 1], [2, 1], [3, 1]]
    assert blocks(fit) == [1384, 1384]
    assert fit.share is None
    assert fit.linear_predictors[:5] == approx(
        [
            -1.58445433621737,
            -1.71483228785931,
            -1.5313645221129,
            -1.87034719638195,
            -1.55837874588898,
        ]
    )
    assert sum(fit.linear_predictors) == approx(275.928710203996)
    assert fit.residuals[:5] == approx(
        [
            -0.0285295525713872,
            -0.021968450238322,
            -0.0445459474894849,
            -0.0725056061684486,
            -0.00703826675277095,
        ]
    )
    assert sum(v * v for v in fit.residuals) == approx(885.082678825288)
    # the unstacked data
    assert len(fit.y) == 1384
    assert fit.y.type == "mright"
    assert len(fit.x) == 1384
    assert fit.x[0] == [88.0, 0.0]
    assert fit.strata is None


def test_efron_and_non_robust(mg):
    efron = r.coxph("Surv(etime, event) ~ age + sex", mg, id="id", ties="efron")
    assert efron.method == "efron"
    assert efron.coefficients == approx(
        [0.0130385698370958, -0.0251377892665754, 0.0648236635628613, 0.393225863731497]
    )
    assert efron.loglik == approx([-6347.63667397438, -6152.89247342735])
    naive = r.coxph("Surv(etime, event) ~ age + sex", mg, id="id", robust=False)
    assert diag(naive.var) == approx(
        [6.82128634659839e-05, 0.0355150565516966, 1.30800763318094e-05, 0.00485777217322188]
    )
    assert naive.rscore is None
    assert naive.wald_test == approx(332.101216138937)
    assert naive.concordance["std"] == approx(0.00970550449064379, rel=1e-6)


def test_weights_and_offset(mg):
    fit = r.coxph("Surv(etime, event) ~ age + sex + offset(o)", mg, id="id", weights="w")
    assert fit.coefficients == approx(
        [0.00861517772833776, 0.000155193446530753, 0.0664726530464528, 0.389430603993082]
    )
    assert fit.loglik == approx([-11192.0325167236, -10905.9604082381])
    assert fit.rscore == approx(300.917145753984, rel=1e-6)
    assert fit.linear_predictors[:3] == approx(
        [-1.99182302475182, -2.2079748020352, -1.94997676493526]
    )


def test_strata_terms(mg):
    fit = r.coxph("Surv(etime, event) ~ age + mspike + strata(sex)", mg, id="id")
    assert fit.na_action == r.NaAction(
        (39, 460, 520, 677, 688, 694, 884, 889, 1169, 1327, 1356), "omit"
    )
    assert (fit.n, fit.nevent) == (1373, 969)
    assert fit.coefficients == approx(
        [0.0164591499397844, 0.8621145181034, 0.0646831789511972, -0.0595613551479498]
    )
    assert fit.loglik == approx([-5640.82255041894, -5439.69599541547])
    assert fit.smap == r.NamedMatrix(
        ["(Baseline)", "strata(sex)"], ["1:2", "1:3"], [[1, 2], [1, 1]]
    )
    assert fit.transitions.values[0] == [115, 854, 404]
    assert fit.strata[:2] == ["F", "F"]
    # the strata row is found by term, wherever an offset() sits (R errors on the second)
    expected = [0.0128649869605635, 0.063998587023079]
    for formula in (
        "Surv(etime, event) ~ age + strata(sex) + offset(o)",
        "Surv(etime, event) ~ offset(o) + age + strata(sex)",
    ):
        fit = r.coxph(formula, mg, id="id")
        assert fit.coefficients == approx(expected)
        assert fit.loglik == approx([-5682.97882733496, -5498.89523085727])


def test_counting_process_illness_death(my):
    fit = r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my, id="id")
    assert fit.coef_names == (
        "trtB_1:2",
        "sexm_1:2",
        "trtB_1:3",
        "sexm_1:3",
        "trtB_2:3",
        "sexm_2:3",
    )
    assert fit.coefficients == approx(
        [
            -0.138994059753283,
            -0.0312506555700437,
            -0.391675484853137,
            0.015635169227796,
            -0.289654524006076,
            0.215699959827038,
        ]
    )
    assert fit.loglik == approx([-3881.64862876869, -3875.4959052967])
    assert fit.iter == 3
    assert fit.rscore == approx(12.0339222575889, rel=1e-6)
    assert fit.wald_test == approx(12.1062888615854, rel=1e-6)
    assert diag(fit.var) == approx(
        [
            0.0110937792554607,
            0.0112121930354871,
            0.0293318336199671,
            0.0291297242448203,
            0.0261062999646504,
            0.0261504680391136,
        ],
        rel=1e-6,
    )
    assert (fit.n, fit.nevent, fit.n_id) == (1009, 684, 646)
    assert fit.transitions.values == [[364, 140, 142], [0, 180, 183], [0, 0, 0]]
    assert blocks(fit) == [646, 646, 363]
    assert fit.concordance["concordance"] == approx(0.532883867065122)
    assert fit.concordance["std"] == approx(0.0114679398123815, rel=1e-6)


def test_multistate_equals_separate_fits(my):
    """multi2.R: the transitions are separate Cox models."""

    fit = r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my, id="id", iter_max=4, robust=False)
    data = my.assign(
        is_sct=(my.event == "sct").astype(int), is_death=(my.event == "death").astype(int)
    )
    before, after = data[data.priortx == 0], data[data.priortx == 1]
    parts = [
        r.coxph(f"Surv(tstart, tstop, {status}) ~ trt + sex", part, iter_max=4, method="breslow")
        for status, part in (("is_sct", before), ("is_death", before), ("is_death", after))
    ]
    assert fit.coefficients == approx([b for part in parts for b in part.coefficients])
    assert fit.loglik == approx([sum(part.loglik[i] for part in parts) for i in (0, 1)])
    assert fit.loglik == approx([-3881.64862876869, -3875.49590529669])
    var = np.array(fit.var)
    for k, part in enumerate(parts):
        assert var[2 * k : 2 * k + 2, 2 * k : 2 * k + 2].tolist() == [
            approx(row) for row in part.var
        ]
    assert var[:2, 2:].tolist() == [[0.0] * 4] * 2
    assert parts[0].var[0] == approx([0.0111026721626867, -0.000857211695370525])


# ---------------------------------------------------------------------------
# formula lists
# ---------------------------------------------------------------------------


def test_formula_list_adds_a_covariate(mg):
    fit = r.coxph(["Surv(etime, event) ~ age + sex", "1:3 ~ mspike"], mg, id="id")
    assert fit.coef_names == ("age_1:2", "sexM_1:2", "age_1:3", "sexM_1:3", "mspike")
    assert fit.coefficients == approx(
        [
            0.0130377951641943,
            -0.0251369568806959,
            0.0649200131710115,
            0.387280242150498,
            -0.0590289738364771,
        ]
    )
    assert fit.loglik == approx([-6305.67503942234, -6111.14964480752])
    assert (fit.nevent, fit.n) == (969, 1384)
    assert blocks(fit) == [1384, 1373]
    assert fit.cmap.values == [[1, 3], [2, 4], [0, 5]]
    assert fit.transitions.values[0] == [115, 860, 409]
    assert fit.na_action is None
    # mspike is missing only on rows that stay in use for 1:2, so nothing is dropped
    failing = r.coxph(
        ["Surv(etime, event) ~ age + sex", "1:3 ~ mspike"], mg, id="id", na_action="na.fail"
    )
    assert failing.coefficients == approx(fit.coefficients)


def test_formula_list_removes_and_shares(mg):
    drop = r.coxph(["Surv(etime, event) ~ age + sex", "1:2 ~ -sex"], mg, id="id")
    assert drop.coef_names == ("age_1:2", "age_1:3", "sexM")
    assert drop.coefficients == approx([0.0131634349599458, 0.0645438014943813, 0.391576147058641])
    assert drop.cmap.values == [[1, 2], [0, 3]]
    for lines in (
        ["Surv(etime, event) ~ age", "0:3 ~ sex"],
        ["Surv(etime, event) ~ age", "1:3 ~ sex / init"],
    ):
        same = r.coxph(lines, mg, id="id")
        assert same.coefficients == approx(drop.coefficients)
    named = r.coxph(["Surv(etime, event) ~ age", '"(s0)":"death" ~ mspike'], mg, id="id")
    assert named.coef_names == ("age_1:2", "age_1:3", "mspike")
    assert named.coefficients == approx(
        [0.0131634349599433, 0.0622968316126558, -0.0575988771445161]
    )


def test_common_coefficient(common_fit):
    assert common_fit.coef_names == ("age_1:2", "sexM", "age_1:3")
    assert common_fit.coefficients == approx(
        [0.0148709083844237, 0.341853933163757, 0.0642069047190597]
    )
    assert common_fit.cmap.values == [[1, 3], [2, 2]]


def test_shared_baselines(mg, shared_fit):
    fit = shared_fit
    assert fit.coef_names == ("age_1:2", "age_1:3", "ph(1:3/1:2)")
    assert fit.coefficients == approx([0.0106778897372408, 0.0624086063114782, -1.66314799096339])
    assert fit.iter == 5
    assert fit.loglik == approx([-7026.90746408329, -6523.30271517694])
    assert fit.cmap == r.NamedMatrix(["age", "ph(1:2)"], ["1:2", "1:3"], [[1, 2], [0, 3]])
    assert fit.smap.values == [[1, 1]]
    assert fit.share.vtype == (0, 2)
    assert fit.share.scale == approx([1.0, 0.189541365446661])
    assert blocks(fit) == [2768]
    common = r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ 1 / common"], mg, id="id")
    assert common.coefficients == approx([0.030261509587681, 0.0587944909946316])
    assert common.smap.values == [[1, 1]]
    assert common.share is None
    both = r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ age / common + shared"], mg, id="id")
    assert both.coef_names == ("age", "ph(1:3/1:2)")
    assert both.coefficients == approx([0.055274884598812, 2.01200026088354])
    assert both.cmap.values == [[1, 1], [0, 2]]
    assert both.share.scale == approx([1.0, 7.47826086955954])


def test_formula_list_missing_values(my_na):
    lines = [
        "Surv(tstart, tstop, event) ~ trt",
        "1:3 + 2:3 ~ sex",
        "1:2 + 2:3 ~ flt3",
    ]
    fit = r.coxph(lines, my_na, id="id", model=True)
    assert fit.na_action == r.NaAction((423, 425, 428), "omit")
    assert (fit.n, fit.n_id, fit.nevent) == (1006, 645, 680)
    assert fit.coef_names == (
        "trtB_1:2",
        "flt3B_1:2",
        "flt3C_1:2",
        "trtB_1:3",
        "sexm_1:3",
        "trtB_2:3",
        "sexm_2:3",
        "flt3B_2:3",
        "flt3C_2:3",
    )
    assert fit.coefficients == approx(
        [
            -0.146708153791885,
            0.444685781322166,
            0.485551968656477,
            -0.393821437935589,
            0.038634856472604,
            -0.309463459131824,
            0.260273817086529,
            0.233813110190132,
            0.549702681668485,
        ]
    )
    assert fit.loglik == approx([-3856.36419031088, -3840.38353386311])
    assert fit.cmap.values == [[1, 4, 6], [0, 5, 7], [2, 0, 8], [3, 0, 9]]
    assert fit.transitions.values == [[363, 138, 142], [0, 179, 183], [0, 0, 0]]
    assert blocks(fit) == [643, 643, 361]
    assert diag(fit.naive_var) == approx(
        [
            0.0110955322339227,
            0.0194525801746213,
            0.0234276492944711,
            0.0294885513441667,
            0.0294386394373801,
            0.0232447681246106,
            0.0231519966491837,
            0.0484400970346877,
            0.0537810519012755,
        ]
    )
    # the clusters stay aligned with the rows after the drop (R's are not)
    assert diag(fit.var) == approx(
        [
            0.0110578456704084,
            0.0194811408616647,
            0.0237951455461563,
            0.029767593487902,
            0.029455811056744,
            0.0269153039507501,
            0.0265064029270663,
            0.0621814882677592,
            0.0691988686089126,
        ],
        rel=1e-6,
    )
    assert fit.rscore == approx(30.3176978576869, rel=1e-6)
    assert fit.wald_test == approx(30.7449097228274, rel=1e-6)
    assert fit.concordance["std"] == approx(0.011656218461181, rel=1e-6)
    # the stored frame is the one fitted
    kept = [row for row in range(len(my_na)) if row + 1 not in (423, 425, 428)]
    for frame in (fit.model, r.model_frame(fit)):
        assert len(frame["trt"]) == 1006
        assert list(frame["trt"]) == my_na.trt.iloc[kept].tolist()
    assert list(r.model_frame(fit)["start"]) == my_na.tstart.iloc[kept].tolist()
    assert list(r.model_frame(fit)["stop"]) == my_na.tstop.iloc[kept].tolist()
    exclude = r.coxph(lines, my_na, id="id", na_action="na.exclude")
    assert exclude.na_action == r.NaAction((423, 425, 428), "exclude")
    with pytest.raises(ValueError, match="missing values in object"):
        r.coxph(lines, my_na, id="id", na_action="na.fail")
    single = r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my_na, id="id")
    assert single.na_action.rows == (425, 426, 427, 428)
    assert (single.n, single.n_id) == (1005, 643)
    assert single.coefficients == approx(
        [
            -0.143484597407023,
            -0.0434725817275016,
            -0.393821437935577,
            0.0386348564726049,
            -0.281006793554858,
            0.224062237715018,
        ]
    )


def test_statedata_and_state_expressions(my):
    statedata = {"state": ["(s0)", "sct", "death"], "tx": [0, 1, None]}
    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", 'tx(0):state("death") ~ sex'],
        my,
        id="id",
        statedata=statedata,
    )
    assert fit.coef_names == ("trtB_1:2", "trtB_1:3", "sexm", "trtB_2:3")
    assert fit.coefficients == approx(
        [-0.141385262465604, -0.391675484853137, 0.015635169227796, -0.26181530792858]
    )
    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", "tx(0):tx(1) ~ sex"],
        my,
        id="id",
        statedata=pd.DataFrame(statedata),
    )
    assert fit.coef_names == ("trtB_1:2", "sexm", "trtB_1:3", "trtB_2:3")
    assert fit.coefficients == approx(
        [-0.138994059753283, -0.0312506555700437, -0.390325247666303, -0.26181530792858]
    )
    expected = [
        -0.141385262465604,
        -0.391675484853137,
        0.015635169227796,
        -0.289654524006076,
        0.215699959827038,
    ]
    for lhs in ('c("(s0)", "sct"):"death"', "(1:2):3", "1:2:3"):
        fit = r.coxph(["Surv(tstart, tstop, event) ~ trt", f"{lhs} ~ sex"], my, id="id")
        assert fit.coefficients == approx(expected)


def test_formula_list_term_rules(my):
    def fit(*lines, **kwargs):
        return r.coxph(["Surv(tstart, tstop, event) ~ trt", *lines], my, id="id", **kwargs)

    assert fit("1:3 ~ sex", "2:3 ~ sex + flt3").coef_names == (
        "trtB_1:2",
        "trtB_1:3",
        "sexm_1:3",
        "trtB_2:3",
        "sexm_2:3",
        "flt3B",
        "flt3C",
    )
    removed = fit("1:3 ~ sex", "1:3 ~ -trt")
    assert removed.coef_names == ("trtB_1:2", "sexm", "trtB_2:3")
    assert removed.coefficients == approx(
        [-0.141385262465604, -0.0181581630364868, -0.26181530792858]
    )
    assert removed.cmap.values == [[1, 0, 3], [0, 2, 0]]
    interaction = r.coxph(["Surv(tstart, tstop, event) ~ trt + sex", "1:3 ~ trt:sex"], my, id="id")
    assert interaction.coef_names == (
        "trtB_1:2",
        "sexm_1:2",
        "trtB_1:3",
        "sexm_1:3",
        "trtB:sexm",
        "trtB_2:3",
        "sexm_2:3",
    )
    star = r.coxph("Surv(tstart, tstop, event) ~ trt * sex", my, id="id")
    assert star.coef_names[:3] == ("trtB_1:2", "sexm_1:2", "trtB:sexm_1:2")
    assert len(star.coef_names) == 9
    common = fit("1:2 + 2:3 ~ sex / common")
    assert common.coef_names == ("trtB_1:2", "sexm", "trtB_1:3", "trtB_2:3")
    assert common.coefficients == approx(
        [-0.145217472998936, 0.0498730854428101, -0.390325247666303, -0.268190887854923]
    )
    assert common.cmap.values == [[1, 3, 4], [2, 0, 2]]
    baseline = fit("1:3 + 2:3 ~ 1 / common")
    assert baseline.smap.values == [[1, 2, 2]]
    assert blocks(baseline) == [646, 1009]
    assert baseline.coefficients == approx(
        [-0.141385262465602, -0.63047196454286, -0.0413423916301038]
    )
    shared = fit("1:3 + 2:3 ~ 1 / shared")
    assert shared.coef_names == ("trtB_1:2", "trtB_1:3", "trtB_2:3", "ph(2:3/1:3)")
    assert shared.coefficients == approx(
        [-0.141385262465602, -0.395520629630838, -0.257694445123988, 0.504255641217598]
    )


def test_strata_in_formula_lists(my):
    dropped = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt + strata(sex)", "1:2 ~ -strata(sex)"], my, id="id"
    )
    assert dropped.smap.values[1] == [0, 1, 1]
    assert dropped.coefficients == approx(
        [-0.141385262465604, -0.389771520261867, -0.302763061567025]
    )
    assert dropped.loglik == approx([-3660.57388399236, -3655.04540086379])
    everywhere = r.coxph("Surv(tstart, tstop, event) ~ trt + strata(sex)", my, id="id")
    assert everywhere.smap.values[1] == [1, 1, 1]
    assert everywhere.coefficients == approx(
        [-0.138059422972667, -0.389771520261867, -0.302763061567025]
    )
    one = r.coxph(["Surv(tstart, tstop, event) ~ trt", "1:3 ~ strata(sex)"], my, id="id")
    assert one.smap.values[1] == [0, 1, 0]
    assert one.coefficients == approx([-0.141385262465604, -0.389771520261867, -0.26181530792858])


def test_cluster_argument(my):
    fit = r.coxph("Surv(tstart, tstop, event) ~ sex", my, id="id", cluster="trt")
    assert fit.rscore == approx(2.0, rel=1e-6)
    assert fit.wald_test == approx(5.44377062386326, rel=1e-6)


def test_lung_shared_hazards_equal_a_factor(lms):
    """mstrata.R: proportional baselines for ph.ecog reproduce the ph.ecog factor."""

    fit = r.coxph(
        ["Surv(time, state) ~ 1", "1:4 + 2:4 + 3:4 ~ age + sex / common + shared"],
        lms,
        id="id",
        istate="cstate",
        ties="breslow",
    )
    assert fit.states == ("ph0", "ph1", "ph2", "death")
    assert fit.coef_names == ("age", "sex", "ph(2:4/1:4)", "ph(3:4/1:4)")
    expected = [0.0107529784064812, -0.544905922709152, 0.409485168736856, 0.900698400667153]
    assert fit.coefficients == approx(expected)
    single = r.coxph("Surv(time, status) ~ age + sex + factor(ph.ecog)", lms, ties="breslow")
    assert single.coefficients == approx(expected)
    assert fit.loglik == approx([-739.278271686054, -724.856750645895])
    assert fit.loglik == approx(single.loglik)
    assert fit.cmap.values == [[1, 1, 1], [2, 2, 2], [0, 3, 4]]
    assert fit.cmap.rownames == ["age", "sex", "ph(1:4)"]
    assert fit.smap.values == [[1, 1, 1]]
    assert fit.share.vtype == (0, 0, 2)
    assert fit.share.scale == approx([1.0, 1.5060422278669, 2.46132149960269])
    assert fit.transitions.values == [[37, 26], [82, 31], [44, 6], [0, 0]]
    assert r.coef(fit, matrix=True).values == [
        approx([0.0107529784064812] * 3),
        approx([-0.544905922709152] * 3),
        approx([0.0, 0.409485168736856, 0.900698400667153]),
    ]


def test_hand_checkable_data():
    """coxsurv5.R's mtest: fixed coefficients, and survcheckallow."""

    init = [math.log(k) for k in range(1, 7)]
    fit = r.coxph("Surv(t1, t2, state) ~ x", _mtest(), id="id", iter_max=0, init=init)
    assert fit.coef_names == ("x_1:2", "x_3:2", "x_1:3", "x_2:3", "x_1:4", "x_2:4")
    assert fit.coefficients == approx(init)
    assert fit.loglik == approx([-6.05600306824555, -6.05600306824555])
    assert fit.iter == 0
    assert fit.rscore == approx(4.51318462303454, rel=1e-6)
    assert blocks(fit) == [5, 2, 5, 2, 5, 2]
    assert fit.rmap[:5].tolist() == [[1, 1], [4, 1], [5, 1], [6, 1], [9, 1]]
    assert fit.transitions == r.NamedMatrix(
        ["(s0)", "a", "b", "c"],
        ["a", "b", "c", "(censored)"],
        [[2, 2, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1]],
    )
    with pytest.raises(ValueError, match="wrong length for init argument"):
        r.coxph("Surv(t1, t2, state) ~ x", _mtest(), id="id", iter_max=0, init=init[:5])
    gap, overlap = _mtest(9.5), _mtest(8.0)
    allowed = r.coxph("Surv(t1, t2, state) ~ x", gap, id="id", iter_max=0)
    assert allowed.loglik == approx([-5.95064255258773] * 2)
    assert allowed.score == approx(5.20610687022901)
    assert allowed.rscore == approx(4.0, rel=1e-6)
    fails = "data set fails survcheck for one or more subjects"
    with pytest.raises(ValueError, match=fails):
        r.coxph("Surv(t1, t2, state) ~ x", overlap, id="id")
    with pytest.raises(ValueError, match=fails):
        r.coxph("Surv(t1, t2, state) ~ x", gap, id="id", control={"survcheckallow": ["overlap"]})
    with pytest.raises(ValueError, match=fails):
        r.coxph("Surv(t1, t2, state) ~ x", gap, id="id", survcheckallow=[])
    # R lets an unknown flag name switch every check off
    with pytest.raises(ValueError, match="survcheckallow must be a subset of"):
        r.coxph("Surv(t1, t2, state) ~ x", overlap, id="id", survcheckallow=["foo"])
    assert r.coxph_control(survcheckallow=["gap", "jump"])["survcheckallow"] == ["gap", "jump"]
    with pytest.raises(ValueError, match="survcheckallow must be a subset of"):
        r.coxph_control(survcheckallow="foo")


def test_errors(mg, my):
    def fails(message, *args, **kwargs):
        with pytest.raises(ValueError, match=message):
            r.coxph(*args, **kwargs)

    fails(
        "formula is a list but the response is not multi-state",
        ["Surv(etime, death) ~ age", "1:2 ~ sex"],
        mg,
        id="id",
    )
    fails("an id statement is required", "Surv(etime, event) ~ age", mg)
    fails(
        "ties='exact' not supported for multistate",
        "Surv(etime, event) ~ age",
        mg,
        id="id",
        ties="exact",
    )
    fails(
        "do not currently support pspline terms", "Surv(etime, event) ~ pspline(age)", mg, id="id"
    )
    fails(
        "do not currently support frailty terms",
        "Surv(etime, event) ~ age + frailty(grp)",
        mg,
        id="id",
    )
    fails(
        "do not currently support ridge penalties", "Surv(etime, event) ~ ridge(age)", mg, id="id"
    )
    fails(r"the tt\(\) transform is not implemented", "Surv(etime, event) ~ tt(age)", mg, id="id")
    base = "Surv(etime, event) ~ age"
    fails("numeric state is out of range", [base, "4:1 ~ sex"], mg, id="id")
    fails("bogus: state not found", [base, '"pcm":"bogus" ~ sex'], mg, id="id")
    fails("term found without a ':' 1", [base, "1 ~ sex"], mg, id="id")
    fails(
        "numeric state is out of range",
        ["Surv(tstart, tstop, event) ~ trt", "c(0, 2):3 ~ sex"],
        my,
        id="id",
    )
    for rhs, bad in (
        ("sex / bogus", "bogus"),
        ("sex / age", "age"),
        ("sex / init(c(1, 2))", "init"),
    ):
        fails(
            f"option not recognized in a covariates formula: {bad}",
            [base, f"1:3 ~ {rhs}"],
            mg,
            id="id",
        )
    statedata = {"state": ["(s0)", "sct", "death"], "tx": [0, 1, None]}
    fails(
        "state variable with no list of values: tx",
        ["Surv(tstart, tstop, event) ~ trt", 'tx:state("death") ~ sex'],
        my,
        id="id",
        statedata=statedata,
    )
    fails("statedata must be a data frame", [base, "1:3 ~ sex"], mg, id="id", statedata=[1, 2])
    fails(
        "statedata data frame must contain a 'state' variable",
        [base, "1:3 ~ sex"],
        mg,
        id="id",
        statedata={"tx": [1, 2, 3]},
    )
    fails("an element of the formula list is not a formula", [base, 3], mg, id="id")
    fails("all formulas must have a left and right side", [base, "~ sex"], mg, id="id")
    fails(r"offset\(\) terms are not supported", [base + " + offset(o)", "1:3 ~ sex"], mg, id="id")
    fails("use strata\\(\\) terms", "Surv(etime, event) ~ age", mg, id="id", strata="sex")
    censored = mg.iloc[:50].assign(event=cat(["censor"] * 50, ["censor", "pcm", "death"]))
    fails("needs at least one event", "Surv(etime, event) ~ age", censored, id="id")
    for formula in (
        "Surv(etime, event) ~ 1",
        ["Surv(etime, event) ~ 1", "1:2 + 1:3 ~ 1 / shared"],
        "Surv(etime, event) ~ strata(sex)",
        "Surv(etime, event) ~ offset(o)",
    ):
        fails("needs at least one covariate", formula, mg, id="id")


def test_missing_istate_in_a_formula_list(my):
    data = my.assign(current=cat(np.where(my.priortx == 1, "sct", "(s0)"), ["(s0)", "sct"]))
    data["current"] = data["current"].astype(object)
    data.loc[4, "current"] = None
    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", "1:2 ~ flt3"], data, id="id", istate="current"
    )
    assert fit.coef_names == ("trtB_1:2", "flt3B", "flt3C", "trtB_1:3", "trtB_2:3")
    assert fit.coefficients == approx(
        [
            -0.147108585105755,
            0.447496585883644,
            0.496490640288565,
            -0.388160212347252,
            -0.261815307928578,
        ]
    )
    assert fit.loglik == approx([-3875.2297164759, -3863.37517571892])
    assert (fit.n, fit.n_id) == (1008, 646)
    assert fit.na_action.rows == (5,)


def test_first_level_drop_resets_the_counts(mg):
    data = mg.copy()
    data.loc[[1, 4], "etime"] = np.nan
    lines = ["Surv(etime, event) ~ age + sex", "1:3 ~ mspike"]
    fit = r.coxph(lines, data, id="id")
    assert (fit.n, fit.n_id, fit.nevent) == (1382, 1382, 967)
    assert fit.na_action.rows == (2, 5)
    assert fit.coefficients == approx(
        [
            0.0130531219406857,
            -0.0256310576497012,
            0.0647994038453437,
            0.391587137201652,
            -0.0615945603716723,
        ]
    )
    assert fit.loglik == approx([-6291.21599136112, -6097.40976390205])
    with pytest.raises(ValueError, match="missing values in object"):
        r.coxph(lines, data, id="id", na_action="na.fail")


def test_cluster_term_keeps_the_list(mg):
    fit = r.coxph(["Surv(etime, event) ~ age + cluster(grp)", "1:3 ~ mspike"], mg, id="id")
    assert fit.coef_names == ("age_1:2", "age_1:3", "mspike")
    assert fit.coefficients == approx([0.0131634349599433, 0.0622968316126558, -0.0575988771445161])
    assert diag(fit.var) == approx(
        [4.86279463410178e-05, 1.83869342150771e-05, 0.00395728868609266], rel=1e-6
    )
    assert fit.rscore == approx(43.8570686981789, rel=1e-6)
    assert fit.wald_test == approx(224.229720522159, rel=1e-6)
    argument = r.coxph(["Surv(etime, event) ~ age", "1:3 ~ mspike"], mg, id="id", cluster="grp")
    assert argument.var == [approx(row) for row in fit.var]


def test_blocks_follow_their_first_transition(my):
    fit = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", "1:2 + 1:3 ~ 1 / shared", "2:3 ~ strata(sex)"],
        my,
        id="id",
    )
    assert fit.smap.values == [[1, 1, 2], [0, 0, 1]]
    assert fit.cmap.values == [[1, 2, 3], [0, 4, 0]]
    assert fit.coef_names == ("trtB_1:2", "trtB_1:3", "trtB_2:3", "ph(1:3/1:2)")
    assert fit.coefficients == approx(
        [-0.149331230493399, -0.369393115117944, -0.302763061567023, -0.847297860386046]
    )
    assert fit.loglik == approx([-4106.12459260022, -4049.17481442626])
    assert diag(fit.var) == approx(
        [0.0116863091706351, 0.0270940732313929, 0.0262590247113391, 0.0190476190484052],
        rel=1e-6,
    )
    noncontiguous = r.coxph(
        ["Surv(tstart, tstop, event) ~ trt", "1:2 + 2:3 ~ 1 / shared"], my, id="id"
    )
    assert noncontiguous.smap.values == [[1, 2, 1]]
    assert noncontiguous.cmap.values == [[1, 2, 3], [0, 0, 4]]
    assert noncontiguous.coef_names == ("trtB_1:2", "trtB_1:3", "trtB_2:3", "ph(2:3/1:2)")
    assert noncontiguous.coefficients == approx(
        [-0.157965320755042, -0.390325247666303, -0.209042401175958, -0.220881626637236]
    )
    assert noncontiguous.loglik == approx([-4151.61579993744, -4143.91775261448])
    assert noncontiguous.iter == 3
    assert noncontiguous.share.vtype == (0, 2)
    assert noncontiguous.share.scale == approx([1.0, 1.0, 0.801811587806775])
    rmap = noncontiguous.rmap
    assert rmap.shape == (1655, 2)
    assert blocks(noncontiguous) == [1009, 646]
    for row, expected in ((1, [1, 1]), (645, [1007, 1]), (646, [1009, 1]), (647, [3, 1])):
        assert rmap[row - 1].tolist() == expected
    for row, expected in ((1009, [1008, 1]), (1010, [1, 2]), (1655, [1009, 2])):
        assert rmap[row - 1].tolist() == expected


def test_multistate_identities(mg):
    """tests/multistate.R."""

    common = r.coxph(["Surv(etime, event) ~ sex", "1:0 ~ age / common"], mg, id="id")
    assert common.coef_names == ("sexM_1:2", "age", "sexM_1:3")
    expected = [0.0905561234948941, 0.0574827668179705, 0.373851590192247]
    assert common.coefficients == approx(expected)
    assert common.loglik == approx([-6351.08896303735, -6172.61726290313])
    for lhs in ('"Entry":"pcm" + "Entry":"death"', '"Entry":state(c("pcm", "death"))'):
        fit = r.coxph(
            ["Surv(etime, event) ~ sex", f"{lhs} ~ age / common"], mg, id="id", istate="entry"
        )
        assert fit.states == ("Entry", "pcm", "death")
        assert fit.coef_names == common.coef_names
        assert fit.coefficients == approx(expected)
        assert fit.loglik == approx(common.loglik)
    names = ("age_1:2", "sexM_1:2", "mspike", "age_1:3", "sexM_1:3")
    values = [
        0.0163469186990396,
        -0.00498770021195383,
        0.883956909836061,
        0.0645438014943819,
        0.391576147058643,
    ]
    for lhs in ('1:state("pcm")', '1:"pcm"', '1:c("pcm")'):
        fit = r.coxph(["Surv(etime, event) ~ age + sex", f"{lhs} ~ mspike"], mg, id="id")
        assert fit.coef_names == names
        assert fit.coefficients == approx(values)
        assert fit.loglik == approx([-6350.61063851234, -6143.46068723565])
    removed = r.coxph(["Surv(etime, event) ~ age + sex + mspike", "1:3 ~ -mspike"], mg, id="id")
    assert removed.coef_names == names
    assert removed.coefficients == approx(values)
    strata = r.coxph(
        ["Surv(etime, event) ~ age + strata(sex)", '1:state("pcm") ~ mspike'],
        mg,
        id="id",
        ties="breslow",
    )
    assert strata.coef_names == ("age_1:2", "mspike", "age_1:3")
    assert strata.coefficients == approx(
        [0.0164591499397845, 0.862114518103395, 0.0642932156003545]
    )
    swapped = r.coxph(
        ["Surv(etime, event) ~ age + sex", "1:2 ~ mspike + strata(sex) - sex"], mg, id="id"
    )
    assert swapped.coef_names == ("age_1:2", "mspike", "age_1:3", "sexM")
    assert swapped.coefficients == approx(
        [0.0164591499397836, 0.862114518103394, 0.0645438014943822, 0.391576147058642]
    )
    assert swapped.smap.values == [[1, 2], [1, 0]]
    assert swapped.cmap.values == [[1, 3], [0, 4], [2, 0]]


def test_pbcseq_shared_hazards(pbc2):
    """multi3.R and coxsurv6.R: bilirubin groups as states."""

    assert len(pbc2) == 1945
    assert pbc2.bstat.value_counts(sort=False).tolist() == [1406, 64, 117, 106, 112, 140]
    lines = ["Surv(tstart, tstop, bstat) ~ 1", "c(1:4):5 ~ age / common + shared"]
    fit = r.coxph(lines, pbc2, id="id", istate="bili4")
    assert fit.states == ("normal", "1-2", "2-4", ">4", "death")
    assert fit.cmap.colnames == [
        f"{source}:{target}" for target in range(1, 6) for source in range(1, 5) if source != target
    ]
    assert fit.coef_names == ("age", "ph(2:5/1:5)", "ph(3:5/1:5)", "ph(4:5/1:5)")
    expected = [0.0481214978977572, 0.744843937488483, 1.51340328010702, 3.42199383994622]
    assert fit.coefficients == approx(expected)
    single = r.coxph("Surv(tstart, tstop, death) ~ age + bili4", pbc2, ties="breslow")
    assert single.coefficients == approx(expected)
    assert fit.loglik == approx([-726.559267793818, -588.569772918283])
    assert fit.iter == 6
    assert (fit.n, fit.nevent, fit.n_id) == (1945, 140, 312)
    assert blocks(fit) == [1945]
    assert fit.share.scale[:13] == approx([1.0] * 13)
    assert fit.share.scale[13:] == approx([2.10611272409041, 4.54216277195281, 30.6304263450471])
    for lhs in ("1:5 + 2:5 + 3:5 + 4:5", "0:5"):
        other = r.coxph([lines[0], f"{lhs} ~ age / common + shared"], pbc2, id="id", istate="bili4")
        assert other.coefficients == approx(expected)
    shared = r.coxph([lines[0], "c(1:4):5 ~ age / shared"], pbc2, id="id", istate="bili4")
    assert shared.coef_names == (
        "age_1:5",
        "age_2:5",
        "age_3:5",
        "age_4:5",
        "ph(2:5/1:5)",
        "ph(3:5/1:5)",
        "ph(4:5/1:5)",
    )
    assert shared.coefficients == approx(
        [
            0.0988230615905067,
            0.115217336101388,
            0.0798805938955006,
            0.0380939891858156,
            -0.225568835086146,
            2.64030321469869,
            6.82288460414275,
        ],
        rel=1e-6,
    )
    assert shared.loglik == approx([-726.559267793818, -584.270186543885])
    separate = r.coxph([lines[0], "0:5 ~ age"], pbc2, id="id", istate="bili4")
    assert separate.coef_names == ("age_1:5", "age_2:5", "age_3:5", "age_4:5")
    assert separate.coefficients == approx(
        [0.093715393492755, 0.104242877310579, 0.0770424345777695, 0.037449585213589]
    )
    assert diag(separate.var) == approx(
        [
            0.00108191420360638,
            0.00203370948451013,
            0.000533148458034974,
            0.000116655432616235,
        ],
        rel=1e-6,
    )
    assert diag(separate.naive_var) == approx(
        [
            0.00128568719939669,
            0.00133049115271884,
            0.000677091548822865,
            7.27806877941574e-05,
        ]
    )
    assert blocks(separate) == [766, 415, 290, 474]
    assert separate.loglik == approx([-505.603046552138, -483.637120926969])


def test_shared_transition_without_events_is_kept(pbc2):
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
    labels = ["2:1", "1:2", "3:2", "2:3", "4:3", "2:4", "3:4", "1:5", "2:5", "3:5", "4:5"]
    assert fit.cmap.colnames == labels
    assert fit.cmap.values == [[0] * 7 + [1] * 4, [0] * 8 + [2, 3, 4]]
    assert fit.smap.values == [[1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]]
    assert fit.loglik == approx([-6.31564400998984] * 2)
    assert fit.share.scale == approx(
        [1.0] * 8 + [2.01375270747048, 4.48168907033806, 29.964100047397]
    )
    assert blocks(fit) == [56]


def test_timeline_formula_list(pdata):
    """timeline.R: a Surv2 response in a formula list."""

    assert len(pdata) == 2257
    fit = r.coxph(["Surv2(day, bstat) ~ 1", "(1:3):4 ~ edema + albumin + ast"], pdata, id="id")
    assert fit.coef_names == (
        "edema_1:4",
        "albumin_1:4",
        "ast_1:4",
        "edema_2:4",
        "albumin_2:4",
        "ast_2:4",
        "edema_3:4",
        "albumin_3:4",
        "ast_3:4",
    )
    assert fit.coefficients == approx(
        [
            2.12644551046452,
            0.905946243207352,
            -0.00672011391924642,
            2.24406512162723,
            -1.24011651414715,
            -0.00996410617699029,
            1.14484032112023,
            -2.18853963388488,
            -0.000115903746268375,
        ],
        rel=1e-6,
    )
    assert fit.loglik == approx([-515.597326840635, -430.352204682389])
    assert (fit.n, fit.nevent, fit.n_id) == (1945, 139, 312)
    assert fit.na_action is None
    assert fit.transitions == r.NamedMatrix(
        ["normal", "1-4", "4+", "death"],
        ["normal", "1-4", "4+", "death", "(censored)"],
        [[0, 91, 3, 9, 77], [63, 0, 109, 21, 60], [1, 31, 0, 110, 35], [0, 0, 0, 0, 0]],
    )
    frame = r.model_frame(fit)
    assert all(len(values) == 1945 for values in frame.values())


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


def test_coef_and_vcov_matrices(mg_fit, common_fit):
    matrix = r.coef(mg_fit, matrix=True)
    assert matrix.rownames == ["age", "sexM"]
    assert matrix.colnames == ["1:2", "1:3"]
    assert matrix.values == [
        approx([0.0130377951641942, 0.0645438014943818]),
        approx([-0.0251369568806912, 0.391576147058642]),
    ]
    assert r.coef(mg_fit) == mg_fit.coefficients
    blocks_ = r.vcov(mg_fit, matrix=True)
    assert list(blocks_) == ["1:2", "1:3"]
    assert blocks_["1:2"].rownames == ["age", "sexM"] == blocks_["1:2"].colnames
    expected = {
        "1:2": [
            [4.45427749400815e-05, 0.000178357162995008],
            [0.000178357162995008, 0.0358314862601605],
        ],
        "1:3": [
            [1.3490345349682e-05, 3.2474525933784e-05],
            [3.2474525933784e-05, 0.004332719938041],
        ],
    }
    for label, values in expected.items():
        assert blocks_[label].values == [approx(row, rel=1e-6) for row in values]
    common = r.vcov(common_fit, matrix=True)
    assert common["1:2"].values == [
        approx(row, rel=1e-6)
        for row in [
            [4.4407995980193e-05, 7.70200371578905e-06],
            [7.70200371578905e-06, 0.00389861431548224],
        ]
    ]
    assert common["1:3"].values == [
        approx(row, rel=1e-6)
        for row in [
            [1.33458762135116e-05, 2.49006003929372e-05],
            [2.49006003929372e-05, 0.00389861431548224],
        ]
    ]
    assert r.vcov(mg_fit) == mg_fit.var
    assert r.coef_names(mg_fit) == list(mg_fit.coef_names)


def test_summary(mg_fit, shared_fit):
    summary = r.model_summary(mg_fit)
    row = summary["coefficients"][2]
    assert row["name"] == "age_1:3"
    assert [row[k] for k in ("coef", "exp_coef", "naive_se", "robust_se", "z")] == approx(
        [
            0.0645438014943818,
            1.06667229906214,
            0.00361663881688639,
            0.00367292054769525,
            17.5728825756612,
        ],
        rel=1e-6,
    )
    assert row["p"] == approx(3.97454244627841e-69, rel=1e-5)
    conf = summary["conf_int"][3]
    assert [conf[k] for k in ("exp(coef)", "exp(-coef)", "lower", "upper")] == approx(
        [1.47931056836788, 0.675990573840959, 1.30026053308818, 1.68301636633348], rel=1e-6
    )
    assert summary["logtest"]["test"] == approx(386.731006613854)
    assert summary["logtest"]["df"] == 4
    assert summary["sctest"]["test"] == approx(333.199561019607)
    assert summary["waldtest"]["test"] == 327.19
    assert summary["robscore"]["test"] == approx(286.397150075723, rel=1e-6)
    assert [summary["rsq"]["rsq"], summary["rsq"]["maxrsq"]] == approx(
        [0.243785277057293, 0.99989670010766]
    )
    assert summary["concordance"]["C"] == approx(0.659933149435285)
    assert summary["concordance"]["se(C)"] == approx(0.00970559511372908, rel=1e-6)
    assert summary["cmap"] == mg_fit.cmap
    assert summary["states"] == ["(s0)", "pcm", "death"]
    shared = r.model_summary(shared_fit, conf_int=0.9)
    row = shared["coefficients"][2]
    assert row["name"] == "ph(1:3/1:2)"
    assert [row[k] for k in ("coef", "exp_coef", "naive_se", "robust_se", "z")] == approx(
        [
            -1.66314799096339,
            0.189541365446661,
            0.580859706190291,
            0.525428653109963,
            -3.16531651085143,
        ],
        rel=1e-6,
    )
    assert row["p"] == approx(0.0015491433527694, rel=1e-5)
    conf = shared["conf_int"][2]
    assert [conf["lower"], conf["upper"]] == approx(
        [0.0798663060433971, 0.449825852667626], rel=1e-6
    )
    assert shared["logtest"]["test"] == approx(1007.2094978127)
    assert shared["logtest"]["df"] == 3


def test_likelihood_generics(mg_fit):
    assert r.loglik(mg_fit) == approx(-6157.72345973042)
    assert r.nobs(mg_fit) == 975
    assert r.degrees_freedom(mg_fit) == 4
    assert r.aic(mg_fit) == approx(12323.4469194608)
    assert r.bic(mg_fit) == approx(12342.9766693448)
    assert r.extract_aic(mg_fit) == approx([4, 12323.4469194608])
    intervals = r.confint(mg_fit)
    assert [entry["name"] for entry in intervals] == list(mg_fit.coef_names)
    assert r.model_formula(mg_fit) == "Surv(etime, event) ~ age + sex"


def test_array_columns_and_pickle(mg, mg_fit):
    columns = {name: mg[name].to_numpy() for name in ("etime", "age", "id")}
    columns["sex"] = mg["sex"].tolist()
    columns["event"] = mg["event"]
    fit = r.coxph("Surv(etime, event) ~ age + sex", columns, id="id")
    assert fit.coefficients == approx(mg_fit.coefficients, rel=1e-12)
    restored = pickle.loads(pickle.dumps(mg_fit))  # noqa: S301 - the test's own pickle
    assert restored.coefficients == mg_fit.coefficients
    assert restored.cmap == mg_fit.cmap
    assert restored.rmap.tolist() == mg_fit.rmap.tolist()
    assert restored.share == mg_fit.share


def test_methods_not_yet_ported_refuse(mg, mg_fit):
    not_yet = [
        lambda fit: r.predict(fit),
        lambda fit: r.predict_terms_constant(fit),
        lambda fit: r.residuals(fit),
        lambda fit: r.survfit(fit),
        lambda fit: r.cox_zph(fit),
        lambda fit: r.coxph_detail(fit),
        lambda fit: r.anova(fit),
        lambda fit: r.fitted(fit),
        lambda fit: r.model_matrix(fit),
        lambda fit: r.model_term_names(fit),
        lambda fit: r.model_weights(fit),
    ]
    for call in not_yet:
        with pytest.raises(NotImplementedError, match="multi-state coxph fits yet"):
            call(mg_fit)
    refused = [
        (lambda fit: r.basehaz(fit), "the basehaz function is not implemented for multi-state"),
        (lambda fit: r.yates(fit, "sex"), "multi-state coxph not yet supported"),
        (lambda fit: r.royston(fit), "not defined for multi-state models"),
        (lambda fit: r.concordance(fit), "concordance is not available for multi-state"),
        (lambda fit: r.brier(fit), "brier is not defined for multi-state coxph fits"),
        (
            lambda fit: r.survexp("~ 1", mg, ratetable=fit, rmap={"age": "age"}),
            "Invalid rate table",
        ),
        (
            lambda fit: r.pyears("Surv(etime, death) ~ 1", mg, ratetable=fit),
            "Invalid rate table",
        ),
    ]
    for call, message in refused:
        with pytest.raises(ValueError, match=message):
            call(mg_fit)


def test_na_pass_design():
    data = {
        "t": [1.0, 2.0, 3.0, 4.0],
        "s": [1, 0, 1, 1],
        "age": [1.0, 2.0, None, 3.0],
        "sex": ["m", None, "f", "m"],
    }
    expected = [[1.0, 1.0], [2.0, math.nan], [math.nan, 0.0], [3.0, 1.0]]
    arrays = {
        **data,
        "age": np.array([1.0, 2.0, np.nan, 3.0]),
        "sex": np.array(["m", None, "f", "m"], dtype=object),
    }
    for columns in (data, arrays):
        frame = r._fit._model_frame("Surv(t, s) ~ age + sex", columns, na_action="pass")
        assert frame.names == ["age", "sexm"]
        np.testing.assert_array_equal(np.array(frame.x), np.array(expected))
