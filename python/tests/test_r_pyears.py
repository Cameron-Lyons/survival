"""``pyears``, ``survexp`` and the rate-table helpers against R 4.5 / survival 3.8.11."""

import datetime
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api


def _data():
    return {
        "time": [100, 400, 900, 300],
        "status": [1, 0, 1, 1],
        "age": [60 * 365.25, 70 * 365.25, 65 * 365.25, 80 * 365.25],
        "sex": [1, 2, 1, 2],
        "year": [
            datetime.date(1995, 3, 1),
            datetime.date(1996, 6, 15),
            datetime.date(1997, 1, 1),
            datetime.date(1998, 9, 9),
        ],
        "grp": ["a", "b", "a", "b"],
    }


_RMAP = {"age": "age", "sex": "sex", "year": "year"}


def test_pyears_tabulates_factor_and_tcut_terms_like_r():
    # pyears(Surv(time, status) ~ grp + tcut(age, c(0, 65, 100) * 365.25), d, scale = 365.25)
    result = r.pyears("Surv(time, status) ~ grp + tcut(age, c(0, 65, 100) * 365.25)", _data())
    assert result.dim == [2, 2]
    assert list(result.dimnames) == ["grp", "tcut(age, c(0, 65, 100) * 365.25)"]
    assert result.dimnames["grp"] == ["a", "b"]
    assert result.dimnames["tcut(age, c(0, 65, 100) * 365.25)"] == [
        "    0.00+ thru 23741.25",
        "23741.25+ thru 36525.00",
    ]
    for row, expected in zip(result.pyears, [[0.2737851, 2.464066], [0.0, 1.916496]], strict=True):
        assert row == pytest.approx(expected, rel=1e-6)
    assert result.n == [[1, 1], [0, 2]]
    assert result.event == [[1, 1], [0, 1]]
    assert result.offtable == 0
    assert result.observations == 4
    assert result.tcut is True
    assert result.expected is None
    assert result.group == [
        "a,     0.00+ thru 23741.25",
        "b,     0.00+ thru 23741.25",
        "a, 23741.25+ thru 36525.00",
        "b, 23741.25+ thru 36525.00",
    ]


def test_pyears_data_frame_layout_and_plain_response():
    frame = r.pyears(
        "Surv(time, status) ~ grp + tcut(age, c(0, 65, 100) * 365.25)", _data(), data_frame=True
    ).data
    assert frame["grp"] == ["a", "a", "b"]
    assert frame["tcut(age, c(0, 65, 100) * 365.25)"] == [
        "    0.00+ thru 23741.25",
        "23741.25+ thru 36525.00",
        "23741.25+ thru 36525.00",
    ]
    assert frame["pyears"] == pytest.approx([0.2737851, 2.4640657, 1.9164956], rel=1e-6)
    assert frame["n"] == [1, 1, 2]
    assert frame["event"] == [1, 1, 1]
    # pyears(time ~ 1, d): a plain follow-up, one cell, no events
    plain = r.pyears("time ~ 1", _data())
    assert plain.pyears == pytest.approx(4.654346, rel=1e-6)
    assert plain.n == 4
    assert plain.event is None
    assert plain.dim == []
    assert plain.dimnames == {}
    assert r.as_data_frame(plain)["group"] == ["(all)"]


def test_pyears_expected_events_from_a_rate_table():
    # R's pyears(Surv(time, status) ~ grp, d, ratetable = survexp.us,
    #     rmap = list(age, sex, year), scale = 1)
    result = r.pyears(
        "Surv(time, status) ~ grp", _data(), ratetable=r.survexp_us(), rmap=_RMAP, scale=1
    )
    assert result.expected == pytest.approx([0.05874250, 0.06577112], rel=1e-6)
    assert result.pyears == [1000.0, 700.0]
    assert result.dimnames == {"grp": ["a", "b"]}
    person_years = r.pyears(
        "Surv(time, status) ~ grp",
        _data(),
        ratetable=r.survexp_us(),
        rmap=_RMAP,
        scale=1,
        expect="pyears",
    )
    assert person_years.expected == pytest.approx([976.1598, 689.2017], rel=1e-6)
    # rmap entries may be vectors or constants, and default to same-named columns
    data = {**_data(), "race": ["white"] * 4}
    usr = r.pyears(
        "Surv(time, status) ~ 1", data, ratetable=r.survexp_usr(), rmap={"race": "white"}, scale=1
    )
    assert usr.expected > 0
    with pytest.raises(ValueError, match="Variable not found in the ratetable"):
        r.pyears("Surv(time, status) ~ 1", _data(), ratetable=r.survexp_us(), rmap={"bogus": "age"})
    with pytest.raises(ValueError, match="No rate table specified"):
        r.pyears("Surv(time, status) ~ 1", _data(), rmap=_RMAP)
    with pytest.raises(ValueError, match="Pyears cannot have interaction terms"):
        r.pyears("Surv(time, status) ~ grp:sex", _data())


def test_pyears_vector_call_used_by_the_reticulate_bridge():
    data = _data()
    grouped = r.pyears(r.Surv(data["time"], data["status"]), group=data["grp"], scale=1)
    assert grouped.group == ["a", "b"]
    assert grouped.pyears == [1000.0, 700.0]
    assert grouped.event == [2.0, 1.0]
    counting = r.pyears(None, start=[0, 0], stop=[10, 20], event=[1, 0], group=["x", "x"], scale=1)
    assert counting.pyears == [30.0]
    assert counting.event == [1.0]
    plain = r.pyears(None, time=[10, 20], scale=1)
    assert plain.pyears == 30.0


def test_survexp_ederer_hakulinen_and_individual_match_r():
    # R's survexp(~ 1, d, ratetable = survexp.us, rmap = list(age, sex, year),
    #     times = c(0, 182.5, 365))
    ederer = r.survexp("~ 1", _data(), ratetable=r.survexp_us(), rmap=_RMAP, times=[0, 182.5, 365])
    assert ederer.method == "Ederer"
    assert ederer.time == [0.0, 182.5, 365.0]
    assert ederer.surv == pytest.approx([1.0, 0.9865087, 0.9732566], rel=1e-6)
    assert ederer.n_risk == [4.0, 4.0, 4.0]
    assert ederer.n == 4
    assert ederer.cumhaz[0] == 0.0
    # ... Surv(time, status) ~ grp, method = "hakulinen"
    hakulinen = r.survexp(
        "Surv(time, status) ~ grp",
        _data(),
        ratetable=r.survexp_us(),
        rmap=_RMAP,
        times=[0, 182.5, 365],
        method="hakulinen",
    )
    assert hakulinen.method == "cohort"
    assert hakulinen.strata == ["grp=a", "grp=b"]
    expected_surv = [[1.0, 1.0], [0.9903703, 0.9818916], [0.9799284, 0.9670384]]
    for row, expected in zip(hakulinen.surv, expected_surv, strict=True):
        assert row == pytest.approx(expected, rel=1e-6)
    assert hakulinen.n_risk == [[2.0, 2.0], [1.0, 2.0], [1.0, 1.0]]
    # ... time ~ 1, cohort = FALSE: one expected survival per subject
    individual = r.survexp("time ~ 1", _data(), ratetable=r.survexp_us(), rmap=_RMAP, cohort=False)
    assert individual == pytest.approx([0.9960461, 0.9784402, 0.9466927, 0.9569774], rel=1e-6)
    hazards = r.survexp(
        "time ~ 1", _data(), ratetable=r.survexp_us(), rmap=_RMAP, method="individual.h"
    )
    assert hazards[0] == pytest.approx(-__import__("math").log(0.9960461), rel=1e-4)


def test_survexp_argument_checks_follow_r():
    data = _data()
    with pytest.raises(ValueError, match="either a times argument or a response is needed"):
        r.survexp("~ 1", data, ratetable=r.survexp_us(), rmap=_RMAP)
    with pytest.raises(ValueError, match="Invalid time point requested"):
        r.survexp("~ 1", data, rmap=_RMAP, times=[-1, 2])
    with pytest.raises(ValueError, match="Times must be in increasing order"):
        r.survexp("~ 1", data, rmap=_RMAP, times=[2, 1])
    with pytest.raises(ValueError, match="a response is required in the formula unless"):
        r.survexp("~ 1", data, rmap=_RMAP, times=[1], method="hakulinen")
    with pytest.raises(ValueError, match="Survexp cannot have interaction terms"):
        r.survexp("~ grp:sex", data, rmap=_RMAP, times=[1])
    with pytest.raises(ValueError, match="Can't use tcut variables in expected survival"):
        r.survexp("~ tcut(age, c(0, 100) * 365.25)", data, rmap=_RMAP, times=[1])
    with (
        pytest.raises(ValueError, match="Illegal response value"),
        warnings.catch_warnings(action="ignore"),
    ):
        r.survexp("Surv(time, time, status) ~ 1", data, rmap=_RMAP, times=[1])
    with pytest.warns(UserWarning, match="weights ignored"):
        r.survexp("~ 1", data, rmap=_RMAP, times=[1], weights=[1, 2, 1, 1])
    with pytest.warns(UserWarning, match="se.fit value ignored"):
        r.survexp("~ 1", data, rmap=_RMAP, times=[1], se_fit=True)
    with pytest.raises(NotImplementedError, match="coxph fit as the ratetable"):
        r.survexp("~ 1", data, ratetable=type("Fit", (), {"coefficients": [1.0]})(), times=[1])


def test_survexp_vector_call_used_by_the_reticulate_bridge():
    data = _data()
    days = [r.ratetableDate(value) for value in data["year"]]
    result = r.survexp(
        time=data["time"], age=data["age"], year=days, sex=data["sex"], times=[0, 182.5, 365]
    )
    assert result.method == "cohort"
    assert result.n_risk == [4.0, 3.0, 2.0]
    individual = r.survexp_individual(data["time"], data["age"], days, sex=data["sex"])
    assert individual == pytest.approx([0.9960461, 0.9784402, 0.9466927, 0.9569774], rel=1e-6)


def test_ratetable_helpers_match_r():
    assert r.ratetableDate(datetime.date(2000, 2, 29)) == 11016.0
    assert r.ratetableDate(["1960-01-01", datetime.datetime(1970, 1, 2)]) == [-3653.0, 1.0]
    assert r.ratetableDate([7470, None])[0] == 7470.0
    table = r.survexp_us()
    assert r.is_ratetable(table) is True
    assert r.is_ratetable(1) is False
    assert r.is_ratetable(1, verbose=True) == ["wrong class"]
    assert list(table.dims) == [110, 2, 81]
    assert table.dimid == ["age", "sex", "year"]
    assert table.rate([50, 1, table.dimnames[2].index("2000")]) == pytest.approx(
        8.830104e-06, rel=1e-6
    )
    attributes = {
        "dims": [2, 2],
        "dimid": ["age", "sex"],
        "dimnames": [["0", "1"], ["male", "female"]],
        "cutpoints": [[0.0, 365.25], None],
        "types": [2, 1],
        "rates": [0.1] * 4,
    }
    assert r.is_ratetable(attributes) is True
    assert r.is_ratetable({**attributes, "types": [2, 9]}) is False
    assert isinstance(r.is_ratetable({**attributes, "types": [2, 9]}, verbose=True), list)
    assert list(r.survexp_mn().dims) == [110, 2, 51]
    assert r.survexp_usr().dimid == ["age", "sex", "race", "year"]
