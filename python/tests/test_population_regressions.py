"""Regression tests for the population routines (``pyears``, ``summary.pyears``, ``survexp``,
``cipoisson``) against R 4.5.3 / survival 3.8-12.
"""

from __future__ import annotations

import datetime
import math

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
population = survival.population
datasets = survival.datasets

NAN = math.nan


def approx(values, rel=1e-12):
    return pytest.approx(values, rel=rel, abs=1e-15, nan_ok=True)


def pairs(values, rel=1e-12):
    return [approx(pair, rel) for pair in values]


def _lung_entries():
    lung = datasets.load_lung()
    start = datetime.date(1985, 3, 1)
    return {
        "time": lung["time"],
        "status": lung["status"],
        "sex": lung["sex"],
        "age": [value * 365.25 for value in lung["age"]],
        "entry": [start + datetime.timedelta(days=29 * (i + 1)) for i in range(len(lung["time"]))],
    }


@pytest.mark.parametrize(
    ("method", "male", "female"),
    [
        (
            "hakulinen",
            [
                1.0,
                0.98811931793274022,
                0.97607310564936267,
                0.95300140444874726,
                0.93099104195715754,
            ],
            [1.0, 0.99398818878228878, 0.98767498249199237, 0.97415612281428432, 0.0],
        ),
        (
            "conditional",
            [
                1.0,
                0.98808479216144518,
                0.97592930641106446,
                0.95254022326552523,
                0.92981233627325666,
            ],
            [
                1.0,
                0.99397833237309152,
                0.98763459340096982,
                0.97400798864542881,
                0.96396602085868466,
            ],
        ),
    ],
)
def test_survexp_cohort_methods_follow_each_subject_across_rate_cells(method, male, female):
    result = r.survexp(
        "Surv(time, status) ~ sex",
        _lung_entries(),
        rmap={"age": "age", "sex": "sex", "year": "entry"},
        method=method,
        times=[0, 182, 365, 730, 1000],
    )
    assert [row[0] for row in result.surv] == approx(male, rel=1e-13)
    assert [row[1] for row in result.surv] == approx(female, rel=1e-13)
    assert result.n_risk == [[138, 90], [86, 71], [35, 30], [7, 6], [2, 0]]


TCUT = 'tcut(age, c(0, 65, 120) * 365.25, labels = c("young", "old"))'


def _cohort():
    return {
        "time": [100, 250, 400, 30, 700, 365],
        "status": [1, 0, 1, 0, 1, 0],
        "g": RFactor(["a", "a", "b", "b", "a", "b"], ["a", "b", "c"]),
        "age": [value * 365.25 for value in (60, 65, 70, 55, 80, 75)],
        "sex": ["male", "female", "male", "female", "male", "female"],
        "entry": [
            datetime.date(2000, 1, 1),
            datetime.date(2001, 6, 1),
            datetime.date(1999, 3, 15),
            datetime.date(2002, 2, 2),
            datetime.date(1998, 7, 4),
            datetime.date(2000, 10, 10),
        ],
    }


_COHORT_RMAP = {"age": "age", "sex": "sex", "year": "entry"}


def _by_group(data_frame=False):
    return r.pyears(
        "Surv(time, status) ~ g",
        _cohort(),
        rmap=_COHORT_RMAP,
        ratetable=r.survexp_us(),
        data_frame=data_frame,
    )


def test_cipoisson_gives_a_missing_row_for_a_missing_count():
    limits = r.cipoisson([1, None, 3], 2)
    assert limits[0] == approx((0.012658903992144949, 2.7858216954694495))
    assert all(math.isnan(value) for value in limits[1])
    assert limits[2] == approx((0.30933606144780079, 4.3836365348711617))
    assert all(math.isnan(value) for value in r.cipoisson(0, 0))
    with pytest.raises(ValueError, match="non-negative count"):
        r.cipoisson([-1], 2)


def test_summary_pyears_gives_empty_cells_missing_rates_and_limits():
    result = _by_group()
    assert result.pyears == approx([2.8747433264887063, 2.1765913757700206, 0.0])
    assert result.expected == approx([0.17216054085712138, 0.066775795966720508, 0.0])

    summary = r.summary_pyears(
        result, rate=True, totals=True, scale=1000, **{"ci.r": True, "ci.rr": True}
    )
    assert isinstance(summary, r.SummaryPyearsResult)
    assert summary.dim == [4]
    assert summary.dimnames == {"g": ["a", "b", "c", "Total"]}
    assert summary.n == [3, 3, 0, 6]
    assert summary.event == [2, 1, 0, 3]
    assert summary.pyears == approx(
        [2.8747433264887063, 2.1765913757700206, 0.0, 5.0513347022587265]
    )
    assert summary.expected == approx(
        [0.17216054085712138, 0.066775795966720508, 0.0, 0.23893633682384188]
    )
    assert summary.rate == approx([695.71428571428567, 459.43396226415092, NAN, 593.90243902439033])
    assert summary.ci_r == pairs(
        [
            (84.254227607793538, 2513.1592101296915),
            (11.631860838065263, 2559.8021994219284),
            (NAN, NAN),
            (122.47696091469837, 1735.6349532376066),
        ]
    )
    assert summary.rr == approx([11.617063875628912, 14.975486035364918, NAN, 12.555645741785098])
    assert summary.ci_rr == pairs(
        [
            (1.4068803300576185, 41.964829058708844),
            (0.37914647991478378, 83.43806779503872),
            (NAN, NAN),
            (2.5892760017984355, 36.692924928392451),
        ]
    )
    assert (summary.offtable, summary.observations) == (0.0, 6)

    restored = r.summary_pyears(
        _by_group(data_frame=True), rate=True, ci_r=True, ci_rr=True, totals=True, scale=1000
    )
    assert restored.dimnames == summary.dimnames
    assert restored.rate == approx(summary.rate)
    assert restored.ci_rr == pairs(summary.ci_rr)

    plain = r.summary_pyears(result)
    assert plain.rate is None
    assert plain.ci_r is None
    assert plain.ci_rr is None
    assert plain.rr == approx([11.617063875628912, 14.975486035364918, NAN])


def test_summary_pyears_totals_of_tcut_tables_follow_r():
    cohort = _cohort()
    summary = r.summary_pyears(
        r.pyears(f"Surv(time, status) ~ g + {TCUT}", cohort), rate=True, ci_r=True, totals=True
    )
    assert summary.dim == [4, 3]
    assert list(summary.dimnames.values()) == [["a", "b", "c", "Total"], ["young", "old", "Total"]]
    assert summary.n[0][:2] == [1, 2]
    assert math.isnan(summary.n[0][2])
    assert all(math.isnan(value) for value in summary.n[3])
    assert summary.event == [[1, 1, 2], [0, 1, 1], [0, 0, 0], [1, 2, 3]]
    assert summary.pyears[3] == approx(
        [0.35592060232717315, 4.6954140999315541, 5.0513347022587274]
    )
    assert summary.rate[0] == approx([3.6525000000000003, 0.3844736842105263, 0.69571428571428562])
    assert summary.rate[2] == approx([NAN, NAN, NAN])
    assert summary.ci_r[0][0] == approx((0.09247329366261886, 20.350427485404328))
    assert summary.ci_r[1][0] == approx((0.0, 44.912107353837165))
    assert summary.ci_r[3][2] == approx((0.12247696091469835, 1.7356349532376061))
    assert summary.ci_r[2][1] == approx((NAN, NAN))
    assert summary.rr is None

    three = r.summary_pyears(
        r.pyears(f"Surv(time, status) ~ g + sex + {TCUT}", cohort), totals=True
    )
    assert three.dim == [4, 3, 2]
    assert three.dimnames["sex"] == ["female", "male", "Total"]
    young = [[row[0] for row in block] for block in three.pyears]
    assert young == [
        approx([0.0, 0.27378507871321012, 0.27378507871321012]),
        approx([0.082135523613963035, 0.0, 0.082135523613963035]),
        approx([0.0, 0.0, 0.0]),
        approx([0.082135523613963035, 0.27378507871321012, 0.35592060232717315]),
    ]
    assert three.pyears[3][2][1] == pytest.approx(4.6954140999315532, rel=1e-12)
    assert [[row[1] for row in block] for block in three.event] == [
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 2, 2],
    ]


def test_summary_pyears_rejects_what_r_rejects():
    result = _by_group()
    with pytest.raises(TypeError, match="pyears object"):
        r.summary_pyears(result.pyears)
    with pytest.raises(ValueError, match="conf.level"):
        r.summary_pyears(result, conf_level=1.0)
    with pytest.raises(ValueError, match="scale"):
        r.summary_pyears(result, scale=0)
    with pytest.raises(ValueError, match="at least one term"):
        r.summary_pyears(r.pyears("Surv(time, status) ~ 1", _cohort()), totals=True)
    single = r.summary_pyears(r.pyears("Surv(time, status) ~ 1", _cohort()), rate=True)
    assert single.dim == []
    assert single.n == 6.0
    assert single.rate == pytest.approx(3 / 5.0513347022587274, rel=1e-12)


def test_pyears_factor_terms_take_their_levels_as_model_frame_does():
    cohort = _cohort()
    cohort["h"] = ["x", "y", "x", "y", "z", "x"]
    cohort["w"] = [1, 1, None, None, 1, None]

    def tables(term, **kwargs):
        result = r.pyears(f"Surv(time, status) ~ {term}", cohort, scale=1, **kwargs)
        return result.dimnames[term], result.pyears, result.n

    assert tables("factor(g)") == (["a", "b"], [1050, 795], [3, 3])
    assert tables("as.factor(g)") == (["a", "b", "c"], [1050, 795, 0], [3, 3, 0])
    assert tables("g") == (["a", "b", "c"], [1050, 795, 0], [3, 3, 0])
    without_b = [level != "b" for level in cohort["g"]]
    assert tables("factor(g)", subset=without_b) == (["a", "b"], [1050, 0], [3, 0])
    result = r.pyears("Surv(time, status) ~ factor(g) + w", cohort, scale=1)
    assert result.dimnames["factor(g)"] == ["a", "b"]
    assert result.pyears == [[1050], [0]]
    without_z = [level != "z" for level in cohort["h"]]
    assert tables("factor(h)", subset=without_z) == (["x", "y", "z"], [865, 280, 0], [3, 2, 0])
    assert tables("as.factor(h)", subset=without_z) == (["x", "y", "z"], [865, 280, 0], [3, 2, 0])
    assert tables("h", subset=without_z) == (["x", "y"], [865, 280], [3, 2])
    cohort["o"] = RFactor(cohort["g"], ["b", "c", "a"])
    assert tables("o") == (["b", "c", "a"], [795, 0, 1050], [3, 0, 3])
    assert tables("factor(o)") == (["b", "a"], [795, 1050], [3, 3])
    assert tables("as.factor(o)") == (["b", "c", "a"], [795, 0, 1050], [3, 0, 3])

    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "time": cohort["time"],
            "status": cohort["status"],
            "grp": pd.Categorical(["a", "a", "b", "b", "a", "c"], categories=["a", "b", "c"]),
        }
    )
    sub = frame[frame["grp"] != "c"]
    for term, levels in (("factor(grp)", ["a", "b"]), ("as.factor(grp)", ["a", "b", "c"])):
        result = r.pyears(f"Surv(time, status) ~ {term}", sub, scale=1)
        assert result.dimnames == {term: levels}
        assert result.pyears == [1050, 430, 0][: len(levels)]
        assert result.event == [2, 1, 0][: len(levels)]


def _dated():
    return {
        "time": [100.0, 200.0, 365.0],
        "status": [1, 0, 1],
        "sex": ["male", "female", "male"],
        "birth": [
            datetime.date(1950, 3, 1),
            datetime.date(1960, 7, 15),
            datetime.date(1945, 11, 30),
        ],
        "entry": [datetime.date(2000, 1, 1), datetime.date(2001, 6, 1), datetime.date(1999, 3, 15)],
    }


def test_dates_in_a_non_date_rate_dimension_are_errors():
    data = _dated()
    message = "Data has a date type variable, but the reference ratetable is not a date variable"
    with pytest.raises(ValueError, match=f"{message}: age$"):
        r.survexp("~1", data, rmap={"age": "birth", "sex": "sex", "year": "entry"}, times=[365])
    data["sexd"] = data["birth"]
    with pytest.raises(ValueError, match=f"{message}: age sex$"):
        r.pyears(
            "Surv(time, status) ~ 1",
            data,
            rmap={"age": "birth", "sex": "sexd", "year": "entry"},
            ratetable=r.survexp_us(),
        )


def test_time_differences_are_ages_in_days():
    data = _dated()
    ages = [entry - birth for entry, birth in zip(data["entry"], data["birth"], strict=True)]
    expected = 0.99519210488657173
    for age in (ages, [age.days for age in ages]):
        fit = r.survexp("~1", data, rmap={"age": age, "sex": "sex", "year": "entry"}, times=[365])
        assert fit.surv == approx([expected])
    result = r.pyears(
        "Surv(time, status) ~ 1",
        data,
        rmap={"age": ages, "sex": "sex", "year": "entry"},
        ratetable=r.survexp_us(),
    )
    assert result.expected == pytest.approx(0.0096802978484023857, rel=1e-12)

    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    births = pd.Series(pd.to_datetime(data["birth"]))
    entries = pd.Series(pd.to_datetime(data["entry"]))
    for age in (list(entries - births), list(np.array(ages, dtype="timedelta64[D]"))):
        fit = r.survexp("~1", data, rmap={"age": age, "sex": "sex", "year": "entry"}, times=[365])
        assert fit.surv == approx([expected])
    frame = pd.DataFrame({**data, "birth": births, "entry": entries})
    with pytest.raises(ValueError, match="not a date variable: age"):
        r.survexp("~1", frame, rmap={"age": "birth", "sex": "sex", "year": "entry"}, times=[365])
    dated = r.survexp(
        "~1",
        data,
        rmap={
            "age": ages,
            "sex": "sex",
            "year": list(np.array(data["entry"], dtype="datetime64[D]")),
        },
        times=[365],
    )
    assert dated.surv == approx([expected])


def test_pyears_without_categories_takes_the_default_category_data():
    result = survival.kernels.pyears([1.0, 2.0], event=[1.0, 0.0])
    assert result.pyears == approx([0.0082135523613963042])
    assert result.n == [2.0]
    assert result.event == [1.0]
    assert result.dims == []


def _lung_complete():
    lung = datasets.load_lung()
    names = ("time", "status", "ph.ecog", "age", "sex")
    keep = [i for i, value in enumerate(lung["ph.ecog"]) if value is not None and value == value]
    return {name: [lung[name][i] for i in keep] for name in names}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("conditional", [0.87251720451695414, 0.54191836041458719, 0.29172902012919327]),
        ("hakulinen", [0.87322851549148728, 0.55136764205543476, 0.31200140920479119]),
    ],
)
def test_survexp_coxph_group_with_nobody_at_risk_is_missing(method, expected):
    lung = _lung_complete()
    fit = r.coxph("Surv(time, status) ~ age + sex", lung)
    result = r.survexp(
        "Surv(time, status) ~ ph.ecog", lung, ratetable=fit, method=method, times=[100, 300, 500]
    )
    assert [row[0] for row in result.surv] == approx(expected, rel=1e-9)
    assert [row[3] for row in result.surv] == approx([0.82587377664176709, NAN, NAN], rel=1e-9)


def test_pyears_cut_terms_sort_their_breaks_like_r():
    data = {
        "time": [100, 250, 400, 30, 700, 365],
        "status": [1, 0, 1, 0, 1, 0],
        "age": [value * 365.25 for value in (60, 65, 70, 55, 80, 75)],
    }
    term = "cut(age / 365.25, c(70, 0, 60, 100))"
    result = r.pyears(f"Surv(time, status) ~ {term}", data, scale=1)
    assert result.dimnames == {term: ["(0,60]", "(60,70]", "(70,100]"]}
    assert result.pyears == [130, 650, 1065]
    assert result.n == [2, 2, 2]
    assert result.event == [1, 1, 1]
