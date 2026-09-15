"""``survdiff`` against R's ``survival`` (3.8.11) on the toy data of ``r_api_support``."""

from __future__ import annotations

import pytest

from .helpers import setup_survival_import
from .r_api_support import _toy_data

survival = setup_survival_import()
r = survival.r_api


def _close(actual, expected, rel=1e-8):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e, rel=rel, abs=1e-12), (actual, expected)


def test_survdiff_formula_matches_r_logrank():
    result = r.survdiff("Surv(time, status) ~ group", _toy_data())

    assert isinstance(result, r.SurvDiffResult)
    assert result.n == [4, 4]
    assert result.groups == ["group=A", "group=B"]
    assert result.strata is None
    _close(result.obs, [3, 2])
    _close(result.exp, [1.12857142857143, 3.87142857142857])
    _close(result.var[0], [0.654897959183673, -0.654897959183673])
    assert result.chisq == pytest.approx(5.347771891555, rel=1e-8)
    assert result.pvalue == pytest.approx(0.0207487690622763, rel=1e-8)
    assert result.df == 1


def test_survdiff_rho_and_surv_input():
    data = _toy_data()
    rho = r.survdiff("Surv(time, status) ~ group", data, rho=1)
    assert rho.chisq == pytest.approx(4.85308056872038, rel=1e-8)
    direct = r.survdiff(r.Surv(data["time"], data["status"]), group=data["group"])
    assert direct.groups == ["A", "B"]
    assert direct.chisq == pytest.approx(5.347771891555, rel=1e-8)


def test_survdiff_strata_terms_give_group_by_stratum_tables():
    data = _toy_data()
    data["s"] = r._r_factor(["y", "x", "y", "x", "y", "x", "y", "x"], ["y", "x"])
    result = r.survdiff("Surv(time, status) ~ group + strata(s)", data)

    assert result.strata == {"y": 4, "x": 4}
    assert result.groups == ["group=A", "group=B"]
    assert len(result.obs) == 2
    assert len(result.obs[0]) == 2
    _close([sum(row) for row in result.obs], [3, 2])
    assert result.df == 1


def test_survdiff_one_sample_test_uses_the_offset():
    data = _toy_data()
    data["expect"] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    result = r.survdiff("Surv(time, status) ~ offset(expect)", data)

    assert result.n == [8]
    assert result.groups == []
    _close(result.obs, [5.0])
    _close(result.exp, [5.6188532638708955])
    assert result.var[0][0] == pytest.approx(5.6188532638708955)
    assert result.chisq == pytest.approx(0.0681597016719068, rel=1e-8)
    assert result.pvalue == pytest.approx(0.7940353408185412, rel=1e-8)
    with pytest.raises(ValueError, match="Cannot have both an offset and groups"):
        r.survdiff("Surv(time, status) ~ group + offset(expect)", data)
    data["expect"][0] = 1.5
    with pytest.raises(ValueError, match="The offset must be a survival probability"):
        r.survdiff("Surv(time, status) ~ offset(expect)", data)


def test_survdiff_argument_errors_match_r():
    data = _toy_data()
    with pytest.raises(ValueError, match="No groups to test"):
        r.survdiff("Surv(time, status) ~ 1", data)
    with pytest.raises(ValueError, match="No groups to test"):
        r.survdiff("Surv(time, status) ~ strata(group)", data)
    with pytest.raises(TypeError, match="The 'formula' argument is not a formula"):
        r.survdiff([1, 2, 3])
    with pytest.raises(ValueError, match="invalid value for timefix option"):
        r.survdiff("Surv(time, status) ~ group", data, timefix="yes")
    with pytest.raises(ValueError, match="survdiff not defined for counting process data"):
        r.survdiff("Surv(x1, time, status) ~ group", data)
    with pytest.raises(ValueError, match="Right censored data only"):
        r.survdiff("Surv(time, status, type = 'left') ~ group", data)
    with pytest.raises(ValueError, match="There is only 1 group"):
        r.survdiff(r.Surv(data["time"], data["status"]), group=["A"] * 8)
    with pytest.raises(TypeError, match="unexpected keyword"):
        r.survdiff("Surv(time, status) ~ group", data, bogus=1)


def test_survdiff_subset_and_na_action():
    data = _toy_data()
    subset = r.survdiff("Surv(time, status) ~ group", data, subset=[0, 1, 2, 4, 5, 6])
    assert subset.n == [3, 3]
    data["status"][2] = None
    omitted = r.survdiff("Surv(time, status) ~ group", data)
    assert omitted.n == [3, 4]
    with pytest.raises(ValueError, match="missing values"):
        r.survdiff("Surv(time, status) ~ group", data, **{"na.action": "na.fail"})
