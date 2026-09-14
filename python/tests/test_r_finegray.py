import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_r_style_finegray_matches_right_multistate_fixture():
    data = {
        "id": list(range(1, 9)),
        "time": list(range(1, 9)),
        "event": [
            "target",
            "compete",
            "censor",
            "compete",
            "target",
            "censor",
            "compete",
            "target",
        ],
        "x": [0.2, 0.4, 0.1, 0.8, 0.5, 0.3, 0.7, 0.6],
    }

    result = survival.finegray(
        "Surv(time, event) ~ x",
        data=data,
        etype="target",
        count="dup",
    )

    assert isinstance(result, survival.FineGrayFrame)
    assert result.event == "target"
    assert result["x"] == pytest.approx([0.2, 0.4, 0.4, 0.4, 0.1, 0.8, 0.8, 0.5, 0.3, 0.7, 0.6])
    assert result["fgstart"] == pytest.approx([0, 0, 3, 6, 0, 0, 6, 0, 0, 0, 0])
    assert result["fgstop"] == pytest.approx([1, 3, 6, 8, 3, 6, 8, 5, 6, 8, 8])
    assert result["fgstatus"] == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    assert result["fgwt"] == pytest.approx([1, 1, 5 / 6, 5 / 9, 1, 1, 2 / 3, 1, 1, 1, 1])
    assert result["dup"] == [0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0]
    assert survival.as_data_frame(result) == dict(result)
    copied = result.copy()
    assert isinstance(copied, survival.FineGrayFrame)
    assert copied == result
    assert copied.event == "target"
    assert copied["x"] is result["x"]


def test_r_style_finegray_keeps_model_frame_terms_weights_and_stratum_order():
    data = {
        "time": [1, 2, 3, 1, 2, 3],
        "event": ["target", "censor", "compete"] * 2,
        "x": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "group": ["z", "z", "z", "a", "a", "a"],
        "case_weight": [101.0, 102.0, 103.0, 201.0, 202.0, 203.0],
    }

    result = survival.finegray(
        "Surv(time, event) ~ I(x^2) + x:z + offset(z) + strata(group)",
        data=data,
        weights="case_weight",
        etype="target",
        prefix="cr",
        count="added rows",
    )

    assert list(result) == [
        "I(x^2)",
        "x",
        "z",
        "offset(z)",
        "(weights)",
        "crstart",
        "crstop",
        "crstatus",
        "crwt",
        "added.rows",
    ]
    assert result["x"][:3] == pytest.approx([20.0, 21.0, 22.0])
    assert result["I(x^2)"][:3] == pytest.approx([400.0, 441.0, 484.0])
    assert result["offset(z)"][:3] == pytest.approx([4.0, 5.0, 6.0])
    assert result["(weights)"][:3] == pytest.approx([201.0, 202.0, 203.0])
    assert result["crwt"][:3] == pytest.approx([201.0, 202.0, 203.0])
    assert result["added.rows"] == [0] * 6


def test_r_style_finegray_matches_counting_and_delayed_entry_fixture():
    data = {
        "id": [1, 1, 2, 2, 3, 4],
        "start": [0, 1, 0, 2, 0, 0],
        "stop": [1, 3, 2, 4, 5, 6],
        "event": ["censor", "target", "censor", "compete", "censor", "target"],
        "x": [10, 11, 20, 21, 30, 40],
    }

    result = survival.finegray(
        "Surv(start, stop, event) ~ x",
        data=data,
        id="id",
        etype="target",
        count="dup",
    )

    assert result["x"] == [10, 11, 20, 21, 21, 30, 40]
    assert result["fgstart"] == pytest.approx([0, 1, 0, 2, 5, 0, 0])
    assert result["fgstop"] == pytest.approx([1, 3, 2, 5, 6, 5, 6])
    assert result["fgstatus"] == [0, 1, 0, 0, 0, 0, 1]
    assert result["fgwt"] == pytest.approx([1, 1, 1, 1, 0.5, 1, 1])
    assert result["dup"] == [0, 0, 0, 0, 1, 0, 0]

    delayed = {
        "id": [1, 2, 3, 4, 5],
        "start": [0, 2, 4, 6, 0],
        "stop": [6, 5, 8, 10, 7],
        "event": ["target", "compete", "censor", "target", "censor"],
        "x": [1, 2, 3, 4, 5],
    }
    delayed_result = survival.finegray(
        "Surv(start, stop, event) ~ x",
        data=delayed,
        id="id",
        etype="target",
        count="extra",
    )
    assert delayed_result["x"] == [1, 2, 2, 3, 4, 5]
    assert delayed_result["fgstart"] == pytest.approx([0, 2, 8, 4, 6, 0])
    assert delayed_result["fgstop"] == pytest.approx([6, 6, 10, 8, 10, 7])
    assert delayed_result["fgwt"] == pytest.approx([1, 1, 0.5, 1, 1, 1])
    assert delayed_result["extra"] == [0, 0, 1, 0, 0, 0]


def test_r_style_finegray_preserves_declared_levels_and_timefix():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": pandas.Categorical(
                ["target", "censor", "target", "compete"],
                categories=["censor", "target", "compete"],
                ordered=True,
            ),
            "x": [1, 2, 3, 4],
        }
    )
    subset_result = survival.finegray(
        "Surv(time, event, type='mstate') ~ x",
        data=frame,
        subset=[True, True, True, False],
        etype="target",
    )
    assert subset_result["x"] == [1, 2, 3]
    with pytest.raises(ValueError, match="selected endpoint has no events"):
        survival.finegray(
            "Surv(time, event) ~ x",
            data=frame,
            subset=[True, True, True, False],
            etype="compete",
        )

    plain_levels = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "event": ["censor", "target", "compete", "target"],
        "x": [1, 2, 3, 4],
    }
    plain_subset = survival.finegray(
        "Surv(time, event) ~ x",
        data=plain_levels,
        subset=[True, False, True, True],
    )
    assert plain_subset.event == "target"

    near_ties = {
        "time": [1.0, 1.0 + 1e-10, 2.0, 3.0],
        "event": ["target", "target", "censor", "compete"],
        "x": [1, 2, 3, 4],
    }
    fixed = survival.finegray("Surv(time, event) ~ x", data=near_ties)
    exact = survival.finegray("Surv(time, event) ~ x", data=near_ties, timefix=False)
    assert fixed["fgstop"][:2] == pytest.approx([1.0, 1.0])
    assert exact["fgstop"][:2] == pytest.approx([1.0, 1.0 + 1e-10], abs=0.0, rel=0.0)

    ordered_strata = pandas.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "event": ["target", "censor", "compete"] * 2,
            "x": [10, 11, 12, 20, 21, 22],
            "group": pandas.Categorical(
                ["z", "z", "z", "a", "a", "a"],
                categories=["z", "a"],
                ordered=True,
            ),
        }
    )
    ordered_result = survival.finegray(
        "Surv(time, event) ~ x + strata(group)",
        data=ordered_strata,
    )
    assert ordered_result["x"] == [10, 11, 12, 20, 21, 22]

    two_strata = pandas.DataFrame(
        {
            "time": [1, 2, 3] * 4,
            "event": ["target", "censor", "compete"] * 4,
            "x": [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42],
            "g1": pandas.Categorical(
                ["b"] * 3 + ["a"] * 3 + ["b"] * 3 + ["a"] * 3,
                categories=["b", "a"],
                ordered=True,
            ),
            "g2": pandas.Categorical(
                ["y"] * 6 + ["x"] * 6,
                categories=["y", "x"],
                ordered=True,
            ),
        }
    )
    two_strata_result = survival.finegray(
        "Surv(time, event) ~ x + strata(g1) + strata(g2)",
        data=two_strata,
    )
    assert two_strata_result["x"] == [
        10,
        11,
        12,
        30,
        31,
        32,
        20,
        21,
        22,
        40,
        41,
        42,
    ]

    numpy = pytest.importorskip("numpy")
    array_endpoint = survival.finegray(
        "Surv(time, event) ~ x",
        data=frame,
        etype=numpy.array(["target"]),
    )
    assert isinstance(array_endpoint.event, str)
    assert array_endpoint.event == "target"

    factor_data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "event": [1, 2, 0, 1],
        "label": ["a", "b", "c", "d"],
        "other": ["w", "x", "y", "z"],
    }
    factor_result = survival.finegray(
        "Surv(time, factor(event)) ~ I(label) + identity(other)",
        data=factor_data,
        etype="1",
        count="NA_integer_",
    )
    assert factor_result.event == "1"
    assert factor_result["I(label)"] == ["a", "b", "b", "c", "d"]
    assert factor_result["identity(other)"] == ["w", "x", "x", "y", "z"]
    assert "NA_integer_." in factor_result

    character_factor = {
        "time": [1, 2, 3, 4, 5, 6],
        "event": ["10", "0", "2", "10", "0", "2"],
        "x": [1, 2, 3, 4, 5, 6],
    }
    character_result = survival.finegray(
        "Surv(time, factor(event)) ~ x",
        data=character_factor,
    )
    assert character_result.event == "10"


def test_r_style_finegray_validates_multistate_formula_contract():
    data = {
        "id": [1, 1, 2, 2],
        "start": [0, 1, 0, 1],
        "stop": [1, 2, 1, 2],
        "event": ["censor", "target", "compete", "target"],
        "x": [1, 2, 3, 4],
    }

    with pytest.raises(ValueError, match="requires a subject id"):
        survival.finegray("Surv(start, stop, event) ~ x", data=data)
    with pytest.raises(ValueError, match=r"cluster\(\)"):
        survival.finegray(
            "Surv(stop, event) ~ x + cluster(id)",
            data=data,
        )
    with pytest.raises(ValueError, match="categorical"):
        survival.finegray(
            "Surv(stop, event) ~ x",
            data={**data, "event": [0, 1, 2, 1]},
        )
    with pytest.raises(ValueError, match="not in the data"):
        survival.finegray(
            "Surv(stop, event) ~ x",
            data=data,
            etype="unknown",
        )
    with pytest.raises(ValueError, match="Wrong number of args"):
        survival.finegray(
            "Surv(stop, event, type='counting') ~ x",
            data=data,
        )
    with pytest.raises(ValueError, match="Wrong number of args"):
        survival.finegray(
            "Surv(start, stop, event, type='right') ~ x",
            data=data,
            id="id",
        )

    zero_tail = {
        "time": [1.0, 2.0, 3.0],
        "event": ["target", "censor", "compete"],
        "x": [1, 2, 3],
    }
    assert survival.finegray("Surv(time, event) ~ x", data=zero_tail)["fgstop"] == [
        1.0,
        2.0,
        3.0,
    ]

    zero_before_target = {
        "id": [1, 2, 3, 4],
        "start": [0.0, 0.0, 0.0, 3.0],
        "stop": [1.0, 1.5, 2.0, 5.0],
        "event": ["target", "compete", "censor", "target"],
        "x": [1, 2, 3, 4],
    }
    with pytest.raises(ValueError, match="probability is zero"):
        survival.finegray(
            "Surv(start, stop, event) ~ x",
            data=zero_before_target,
            id="id",
        )

    harmless_zero = {
        "id": [1, 2, 3, 4],
        "start": [0.0, 0.0, 0.0, 3.0],
        "stop": [1.0, 2.0, 2.0, 5.0],
        "event": ["compete", "target", "censor", "compete"],
        "x": [1, 2, 3, 4],
    }
    harmless_result = survival.finegray(
        "Surv(start, stop, event) ~ x",
        data=harmless_zero,
        id="id",
        etype="target",
    )
    assert harmless_result["fgstart"] == pytest.approx([0, 0, 0, 3])
    assert harmless_result["fgstop"] == pytest.approx([2, 2, 2, 5])
    assert harmless_result["fgwt"] == pytest.approx([1, 1, 1, 1])
