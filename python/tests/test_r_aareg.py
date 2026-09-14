import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_aareg_formula_matches_weighted_right_censored_reference_values():
    data = {
        "stop": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
        "status": [1, 1, 1, 1, 0, 1],
        "x": [0.0, 1.0, 2.0, 1.0, 3.0, -1.0],
        "z": [1.0, 0.0, 1.0, 2.0, -1.0, 0.0],
        "weight": [1.0, 2.0, 0.5, 1.5, 1.0, 3.0],
    }

    fit = survival.aareg(
        "Surv(stop, status) ~ x + z",
        data=data,
        weights=data["weight"],
        nmin=1,
        model=True,
        x=True,
        y=True,
    )

    assert isinstance(fit, survival.AaregModelResult)
    assert fit.n == [6, 3, 4]
    assert fit.times == [1.0, 2.0, 2.0, 3.0]
    assert fit.nrisk == pytest.approx([9.0, 8.0, 8.0, 5.5])
    assert fit.coefficient_names == ["Intercept", "x", "z"]
    expected_coefficients = [
        [0.0933572710951526, -0.0287253141831239, 0.0825852782764812],
        [0.246498599439776, 0.0560224089635854, -0.0896358543417367],
        [0.0177404295051354, 0.0494864612511671, 0.0541549953314659],
        [0.1, 0.1, 0.4],
    ]
    for actual, expected in zip(fit.coefficient, expected_coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.test_statistic == pytest.approx(
        [2.72165426280414, 2.44529434676382, 2.84852512254203]
    )
    assert fit.model is not None
    assert fit.x is not None
    assert fit.y is not None


def test_aareg_formula_supports_counting_data_and_clustered_influence():
    data = {
        "start": [0.0, 0.0, 1.0, 0.0, 2.0, 1.0],
        "stop": [1.0, 3.0, 3.0, 4.0, 4.0, 2.0],
        "status": [1, 1, 0, 1, 0, 1],
        "x": [0.0, 1.0, 2.0, 1.0, 3.0, -1.0],
        "z": [1.0, 0.0, 1.0, 2.0, -1.0, 0.0],
        "weight": [1.0, 2.0, 0.5, 1.5, 1.0, 3.0],
        "cluster": ["a", "a", "b", "b", "c", "c"],
    }

    fit = survival.aareg(
        "Surv(start, stop, status) ~ x + z + cluster(cluster)",
        data=data,
        weights=data["weight"],
        nmin=1,
    )

    assert fit.n == [6, 3, 4]
    assert fit.times == [1.0, 2.0, 3.0]
    assert fit.nrisk == pytest.approx([4.5, 7.0, 5.0])
    expected_coefficients = [
        [1.0, -1.0, 0.0],
        [0.53030303030303, -0.439393939393939, -0.0151515151515152],
        [1.69411764705882, -0.705882352941176, -0.470588235294118],
    ]
    for actual, expected in zip(fit.coefficient, expected_coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.cluster_levels == ["a", "b", "c"]
    assert fit.dfbeta is not None
    assert len(fit.dfbeta) == 3
    assert fit.robust_test_variance is not None


def test_aareg_formula_validates_model_specific_options():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 1],
        "x": [0.0, 1.0, 0.5, 2.0],
        "group": ["a", "a", "b", "b"],
        "cluster": ["w", "x", "y", "z"],
    }

    with pytest.raises(ValueError, match="strata terms are not allowed"):
        survival.aareg("Surv(time, status) ~ x + strata(group)", data=data, nmin=1)
    with pytest.raises(ValueError, match="taper"):
        survival.aareg("Surv(time, status) ~ x", data=data, nmin=1, taper=0)
    with pytest.raises(ValueError, match="test must be"):
        survival.aareg("Surv(time, status) ~ x", data=data, nmin=1, test="bogus")
    with pytest.raises(ValueError, match="multiple cluster terms"):
        survival.aareg(
            "Surv(time, status) ~ x + cluster(group) + cluster(cluster)",
            data=data,
            nmin=1,
        )
    with pytest.raises(ValueError, match="cluster"):
        survival.aareg("Surv(time, status) ~ x:cluster(group)", data=data, nmin=1)


def test_aareg_explicit_cluster_overrides_formula_cluster_like_r():
    data = {
        "time": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
        "status": [1, 1, 1, 1, 0, 1],
        "x": [0.0, 1.0, 2.0, 1.0, 3.0, -1.0],
        "formula_cluster": ["a", "a", None, "b", "c", "c"],
        "explicit_cluster": ["left", "right", "left", "right", "left", "right"],
    }

    with pytest.warns(RuntimeWarning, match="formula term ignored"):
        fit = survival.aareg(
            "Surv(time, status) ~ x + cluster(formula_cluster)",
            data=data,
            cluster=data["explicit_cluster"],
            nmin=1,
            na_action="omit",
            model=True,
        )

    assert fit.n[0] == len(data["time"])
    assert fit.cluster_levels == ["left", "right"]
    assert fit.dfbeta is not None
    assert len(fit.dfbeta) == 2
    assert fit.model is not None
    assert "formula_cluster" not in fit.model
    assert fit.model["(cluster)"] == data["explicit_cluster"]
