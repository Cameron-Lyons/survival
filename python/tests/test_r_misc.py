import importlib
import math
import random

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r_misc = importlib.import_module("survival.r._misc")


def test_survcheck_accepts_formula_direct_and_legacy_inputs():
    counting = survival.Surv([0.0, 1.0, 0.0], [1.0, 2.0, 2.0], [0, 1, 1])
    direct = survival.survcheck(counting, id=[1, 1, 2])
    formula = survival.survcheck(
        "Surv(start, stop, status) ~ 1",
        data={
            "id": [1, 1, 2],
            "start": [0.0, 1.0, 0.0],
            "stop": [1.0, 2.0, 2.0],
            "status": [0, 1, 1],
        },
        id="id",
    )
    overlap = survival.survcheck(
        survival.Surv([0.0, 0.5], [1.0, 2.0], [0, 1]),
        id=["subject-a", "subject-a"],
    )
    legacy = survival.survcheck([1, 1, 2], [0.0, 1.0, 0.0], [1.0, 2.0, 2.0], [0, 1, 1])

    assert direct.n_subjects == 2
    assert direct.n_observations == 3
    assert direct.n_transitions == 2
    assert direct.n_problems == 0
    assert direct.transitions == formula.transitions
    assert direct.flags == formula.flags
    assert overlap.is_valid is False
    assert overlap.overlap_ids == [1]
    assert overlap.overlap_rows == [1]
    assert legacy.n_subjects == 2
    assert legacy.flags == [0, 0, 0]

    class Factor(list):
        def __init__(self, values, categories):
            super().__init__(values)
            self.categories = categories

    multistate = survival.survcheck(
        survival.Surv(
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 2.0, 1.0, 3.0],
            Factor(["B", "C", "B", "C"], ["censor", "B", "C"]),
            type="mstate",
        ),
        id=["a", "a", "b", "b"],
        istate=["A", "B", "A", "A"],
    )
    assert multistate.jump_rows == [3]
    assert multistate.jump_ids == [2]
    assert multistate.current_states == [1, 2, 1, 2]
    assert multistate.transitions == {"1 -> 2": 2, "2 -> 3": 2}

    right = survival.survcheck(survival.Surv([1.0, -1.0, 3.0], [1, 0, 1]))
    assert right.is_valid is False
    assert right.invalid_ids == [1]

    with pytest.raises(ValueError, match="id argument"):
        survival.survcheck(counting)
    with pytest.raises(ValueError, match="not valid for interval2"):
        survival.survcheck(survival.Surv([1.0], [2.0], type="interval2"), id=[1])
    with pytest.raises(ValueError, match="formula requires data"):
        survival.survcheck("Surv(start, stop, status) ~ 1", id=[1])


def test_r_api_statefig_matches_r_coordinate_layouts():
    connect = [[0.0, 1.0], [0.0, 0.0]]

    vector_layout = survival.r_api.statefig([1, 1], connect, states=["a", "b"])
    assert vector_layout["states"] == ["a", "b"]
    for actual_row, expected_row in zip(
        vector_layout["positions"],
        [[0.25, 0.5], [0.75, 0.5]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert vector_layout["edges"] == [[0, 1, 1]]

    column_layout = survival.r_api.statefig([[1], [1]], connect, states=["a", "b"])
    for actual_row, expected_row in zip(
        column_layout["positions"],
        [[0.5, 0.75], [0.5, 0.25]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)

    coordinate_layout = survival.r_api.statefig(
        [[0.2, 0.7], [0.8, 0.3]],
        connect,
        states=["a", "b"],
    )
    for actual_row, expected_row in zip(
        coordinate_layout["positions"],
        [[0.2, 0.7], [0.8, 0.3]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)

    with pytest.raises(ValueError, match="number of boxes"):
        survival.r_api.statefig([1, 2], connect, states=["a", "b"])
    with pytest.raises(ValueError, match="square matrix"):
        survival.r_api.statefig([2], [[0.0, 1.0]], states=["a"])
    with pytest.raises(ValueError, match="one entry per connect row"):
        survival.r_api.statefig([2], connect, states=["a"])


def test_r_api_brier_returns_r_style_cox_model_fields():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "x": [0.2, 0.4, 0.1, 0.8, 1.0, 1.2, 0.6, 1.4],
        "wt": [1.0] * 8,
    }
    fit = survival.coxph("Surv(time, status) ~ x", data=data, model=True, max_iter=50)
    assert fit.coefficients[0] == pytest.approx([-2.1132119551866904], abs=3e-5)

    result = survival.r_api.brier(fit, times=[2.0, 4.0, 6.0], detail=True)

    assert list(result) == ["rsquared", "brier", "times", "p0", "phat", "eff.n"]
    assert result["times"] == pytest.approx([2.0, 4.0, 6.0])
    assert result["p0"] == pytest.approx([0.25, 0.4, 0.6])
    assert result["eff.n"] == pytest.approx([8.0, 6.9565217391304355, 5.755395683453237])
    assert result["brier"] == pytest.approx(
        [0.14111837337641864, 0.13682915300545814, 0.24095035022544881],
        abs=1e-6,
    )
    assert len(result["phat"]) == 3
    assert all(len(row) == len(data["time"]) for row in result["phat"])

    weighted_fit = survival.coxph(
        "Surv(time, status) ~ x",
        data=data,
        weights="wt",
        model=True,
        max_iter=50,
    )
    weighted_newdata = {**data, "wt": [8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    weighted_result = survival.r_api.brier(
        weighted_fit,
        times=[2.0, 4.0, 6.0],
        newdata=weighted_newdata,
        detail=True,
    )
    assert weighted_result["eff.n"] == pytest.approx([3.1690140845, 3.1163434903, 3.0356177407])
    assert weighted_result["brier"] == pytest.approx(
        [0.21987334, 0.09626662, 0.12947811],
        abs=1e-6,
    )
    assert weighted_result["rsquared"] == pytest.approx(
        [0.0838611, 0.5575983, 0.2284805],
        abs=1e-6,
    )

    counting_data = {
        "start": [0.0] * len(data["time"]),
        "stop": data["time"],
        "status": data["status"],
        "x": data["x"],
        "id": list(range(1, len(data["time"]) + 1)),
    }
    counting_fit = survival.coxph(
        "Surv(start, stop, status) ~ x",
        data=counting_data,
        id=counting_data["id"],
        model=True,
        max_iter=50,
    )
    counting_result = survival.r_api.brier(counting_fit, times=[2.0, 4.0, 6.0], detail=True)
    assert counting_result["p0"] == pytest.approx(result["p0"])
    assert counting_result["eff.n"] == pytest.approx(result["eff.n"])
    assert counting_result["brier"] == pytest.approx(result["brier"])
    assert len(counting_result["phat"]) == len(result["phat"])
    for counting_row, right_row in zip(counting_result["phat"], result["phat"], strict=True):
        assert counting_row == pytest.approx(right_row)

    common_start_data = {
        "start": [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        "stop": [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        "status": [0, 1, 1, 0, 0, 1],
        "x": [0.2, 0.2, 0.6, 0.6, 1.0, 1.0],
        "id": [1, 1, 2, 2, 3, 3],
    }
    common_start_fit = survival.coxph(
        "Surv(start, stop, status) ~ x",
        data=common_start_data,
        id=common_start_data["id"],
        model=True,
        max_iter=0,
    )
    common_start_result = survival.r_api.brier(
        common_start_fit,
        times=[3.0, 5.0, 7.0],
        newdata=common_start_data,
        detail=True,
    )
    assert common_start_result["p0"] == pytest.approx([1 / 3, 5 / 9, 1.0])
    assert common_start_result["eff.n"] == pytest.approx([5.0, 3.9473684211, 2.5280898876])
    assert common_start_result["brier"] == pytest.approx([0.1669670221, 0.2492855445, 0.0356739933])
    assert common_start_result["rsquared"][:2] == pytest.approx([0.0608105006, 0.0292245624])
    assert math.isinf(common_start_result["rsquared"][2])
    assert common_start_result["rsquared"][2] < 0.0
    assert common_start_result["phat"][0] == pytest.approx([0.2834686894] * 6)

    gap_data = {**common_start_data, "start": [0.0, 3.0, 0.0, 3.0, 0.0, 4.0]}
    gap_fit = survival.coxph(
        "Surv(start, stop, status) ~ x",
        data=gap_data,
        id=gap_data["id"],
        model=True,
        max_iter=0,
    )
    with pytest.raises(ValueError, match="survcheck"):
        survival.r_api.brier(gap_fit, times=[3.0, 5.0, 7.0])

    custom_id_data = {
        **{name: values for name, values in common_start_data.items() if name != "id"},
        "subject": common_start_data["id"],
    }
    custom_id_fit = survival.coxph(
        "Surv(start, stop, status) ~ x",
        data=custom_id_data,
        id="subject",
        model=True,
        max_iter=0,
    )
    bad_id_newdata = {**custom_id_data, "subject": [1, 2, 2, 1, 3, 3]}
    with pytest.raises(ValueError, match="survcheck"):
        survival.r_api.brier(
            custom_id_fit,
            times=[3.0, 5.0, 7.0],
            newdata=bad_id_newdata,
        )

    staggered_data = {**counting_data, "start": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]}
    staggered_fit = survival.coxph(
        "Surv(start, stop, status) ~ x",
        data=staggered_data,
        id=staggered_data["id"],
        model=True,
        max_iter=50,
    )
    with pytest.raises(NotImplementedError, match="delayed entry"):
        survival.r_api.brier(staggered_fit, times=[2.0, 4.0, 6.0])

    with pytest.raises(TypeError, match="coxph"):
        survival.r_api.brier(object())


def test_r_style_cipoisson_uses_rust_scalar_kernel_with_r_recycling():
    scalar = survival.cipoisson(5, time=10.0)
    vector = survival.cipoisson([0, 5, 20], time=[1.0, 10.0, 4.0])
    recycled = survival.cipoisson([1, 2], time=[1.0, 2.0, 3.0])
    anscombe = survival.cipoisson(5, time=10.0, method="anscombe")
    missing_time = survival.cipoisson([1, 2], time=[0.0, 2.0])
    numeric_edges = survival.cipoisson(
        [1.2, 2.8, float("inf")],
        time=[2.0, 4.0, float("inf")],
        p=[0.0, 1.0, 0.95],
    )
    missing_confidence = survival.cipoisson([0, 1.2], p=[None, None])

    assert scalar == pytest.approx((0.1623486, 1.1668332))
    assert [value for row in vector for value in row] == pytest.approx(
        [0.0, 3.688879, 0.1623486, 1.1668332, 3.0541299, 7.722094]
    )
    assert [value for row in recycled for value in row] == pytest.approx(
        [0.025317808, 5.571643, 0.121104639, 3.612344, 0.008439269, 1.857214]
    )
    assert anscombe == pytest.approx((0.1507881, 1.1586004))
    assert math.isnan(missing_time[0][0])
    assert math.isnan(missing_time[0][1])
    assert missing_time[1] == pytest.approx((0.121104639, 3.612344))
    assert numeric_edges[0] == pytest.approx((0.443968106737396, 0.938570591679505))
    assert numeric_edges[1] == (0.0, float("inf"))
    assert all(math.isnan(value) for value in numeric_edges[2])
    assert missing_confidence[0][0] == 0.0
    assert math.isnan(missing_confidence[0][1])
    assert all(math.isnan(value) for value in missing_confidence[1])
    with pytest.raises(ValueError, match="k must be non-negative"):
        survival.cipoisson(-1)
    with pytest.raises(ValueError, match="method must"):
        survival.cipoisson(1, method="fancy")


def test_r_style_bounded_link_helpers_match_survival_link_functions():
    x = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]

    assert survival.blogit(x) == pytest.approx(
        [-2.94443898, -2.94443898, -2.94443898, 0.0, 2.94443898, 2.94443898, 2.94443898]
    )
    assert survival.bprobit(x) == pytest.approx(
        [-1.64485363, -1.64485363, -1.64485363, 0.0, 1.64485363, 1.64485363, 1.64485363]
    )
    assert survival.bcloglog(x) == pytest.approx(
        [-2.97019525, -2.97019525, -2.97019525, -0.36651292, 1.0971887, 1.0971887, 1.0971887]
    )
    assert survival.blog(x) == pytest.approx(
        [-2.99573227, -2.99573227, -2.99573227, -0.69314718, -0.05129329, -0.01005034, 0.0]
    )
    assert survival.blogit(0.5) == pytest.approx(0.0)
    assert survival.bcloglog(0.5) == pytest.approx(math.log(-math.log(0.5)))
    assert survival.blogit([0.0, 0.75, 1.0], edge=0.6) == pytest.approx(
        [0.4054651, 0.4054651, 0.4054651]
    )
    assert survival.bprobit([0.0, 0.75, 1.0], edge=0.6) == pytest.approx(
        [0.2533471, 0.2533471, 0.2533471]
    )
    assert survival.bcloglog([0.0, 0.75, 1.0], edge=0.6) == pytest.approx(
        [-0.08742157, -0.08742157, -0.08742157]
    )
    missing = survival.blogit([0.0, None, 1.0])
    assert missing[0] == pytest.approx(-2.94443898)
    assert math.isnan(missing[1])
    assert missing[2] == pytest.approx(2.94443898)


def test_r_style_survobrien_uses_direct_vectors():
    result = survival.survobrien(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 1],
        [0.1, 0.4, 0.2, 0.8],
        strata=[1, 1, 2, 2],
    )
    label_result = survival.survobrien(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 1],
        [0.1, 0.4, 0.2, 0.8],
        strata=["a", "a", "b", "b"],
    )

    assert isinstance(result, survival.SurvObrienResult)
    assert result.df == 1
    assert len(result.scores) == 4
    assert math.isfinite(result.statistic)
    assert 0.0 <= result.p_value <= 1.0
    assert label_result.statistic == pytest.approx(result.statistic)
    assert label_result.p_value == pytest.approx(result.p_value)
    assert label_result.scores == pytest.approx(result.scores)
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.survobrien([1.0, 2.0], [1, 2], [0.1, 0.2])

    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "x": [0.1, 0.4, 0.2, 0.8],
        "group": ["a", "a", "b", "b"],
        "id": [10, 11, 12, 13],
        "off": [0.1, 0.2, 0.3, 0.4],
    }
    formula = survival.survobrien("Surv(time, status) ~ x", data=data)
    formula_offset = survival.survobrien("Surv(time, status) ~ x + offset(off)", data=data)
    formula_strata = survival.survobrien(
        "Surv(time, status) ~ x + strata(group)",
        data=data,
    )
    formula_cluster = survival.survobrien(
        "Surv(time, status) ~ x + cluster(id)",
        data=data,
    )
    formula_factor = survival.survobrien(
        "Surv(time, status) ~ x + group",
        data=data,
    )
    formula_factor_wrapper = survival.survobrien(
        "Surv(time, status) ~ x + factor(group)",
        data=data,
    )
    formula_as_factor_wrapper = survival.survobrien(
        "Surv(time, status) ~ x + as.factor(group)",
        data=data,
    )
    transformed = survival.survobrien(
        "Surv(time, status) ~ x",
        data=data,
        transform=lambda values: (
            values[0] * 2.0 if len(values) == 1 else [value * 2.0 for value in values]
        ),
    )
    counting = survival.survobrien(
        "Surv(start, stop, status) ~ x",
        data={
            "start": [0.0, 0.0, 1.0, 2.0],
            "stop": [1.0, 2.0, 3.0, 4.0],
            "status": [1, 0, 1, 1],
            "x": [0.1, 0.4, 0.2, 0.8],
        },
    )
    counting_strata = survival.survobrien(
        "Surv(start, stop, status) ~ x + strata(group)",
        data={
            "start": [0.0, 0.0, 1.0, 2.0, 0.0, 3.0],
            "stop": [1.0, 2.0, 3.0, 4.0, 2.0, 5.0],
            "status": [1, 0, 1, 1, 1, 0],
            "x": [0.1, 0.4, 0.2, 0.8, 0.5, 0.7],
            "group": ["a", "a", "b", "b", "a", "b"],
        },
    )

    assert formula["time"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0])
    assert formula["status"] == [1, 0, 0, 0, 1, 0, 1]
    assert formula[".id."] == [1, 2, 3, 4, 3, 4, 4]
    assert formula_factor["group"] == ["a", "a", "b", "b", "b", "b", "b"]
    assert formula_factor_wrapper == formula_factor
    assert formula_as_factor_wrapper == formula_factor
    assert formula["x"] == pytest.approx(
        [
            -1.9459101490553135,
            0.5108256237659907,
            -0.5108256237659907,
            1.9459101490553132,
            -1.0986122886681098,
            1.0986122886681098,
            0.0,
        ]
    )
    assert formula[".strata."] == [1, 1, 1, 1, 2, 2, 3]
    assert formula_offset == formula
    assert "offset(off)" not in formula_offset
    assert formula_strata["time"] == pytest.approx([1.0])
    assert formula_strata["status"] == [1]
    assert formula_strata[".id."] == [1]
    assert formula_strata["x"] == pytest.approx([0.0])
    assert formula_strata[".strata."] == [1]
    assert list(formula_cluster) == ["time", "status", "x", ".strata."]
    assert formula_cluster["x"] == pytest.approx(formula["x"])
    assert transformed["x"] == pytest.approx([0.2, 0.8, 0.4, 1.6, 0.4, 1.6, 1.6])
    assert counting["start"] == pytest.approx([0.0, 0.0, 1.0, 2.0, 2.0])
    assert counting["stop"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 4.0])
    assert counting["status"] == [1, 0, 1, 0, 1]
    assert counting["x"] == pytest.approx(
        [
            -1.0986122886681098,
            1.0986122886681098,
            -1.0986122886681098,
            1.0986122886681098,
            0.0,
        ]
    )
    assert counting_strata == {
        "start": [1.0],
        "stop": [3.0],
        "status": [0],
        ".id.": [3],
        "x": [0.0],
        ".strata.": [4],
    }


def test_survobrien_group_transform_matches_python_reference():
    from survival.r._misc import _survobrien_default_transform

    rng = random.Random(20260801)  # noqa: S311
    value_pool = [-2.0, -0.0, 0.0, 0.5, 0.5, 3.0]
    for _ in range(200):
        n_rows = rng.randrange(1, 30)
        columns = [
            [rng.choice(value_pool) for _ in range(n_rows)] for _ in range(rng.randrange(1, 5))
        ]
        group_sizes = [rng.randrange(0, 20) for _ in range(rng.randrange(0, 10))]
        row_indices = [rng.randrange(n_rows) for _ in range(sum(group_sizes, start=0))]

        actual = survival._survival.survobrien_transform_groups(
            columns,
            row_indices,
            group_sizes,
        )
        expected = []
        for column in columns:
            transformed = []
            offset = 0
            for group_size in group_sizes:
                group_rows = row_indices[offset : offset + group_size]
                transformed.extend(
                    _survobrien_default_transform([column[row] for row in group_rows])
                )
                offset += group_size
            expected.append(transformed)

        assert len(actual) == len(expected)
        for actual_column, expected_column in zip(actual, expected, strict=True):
            assert actual_column == pytest.approx(expected_column)


def test_r_style_yates_direct_helpers_wrap_rust_kernels():
    result = survival.yates(
        [1.0, 2.0, 4.0, 8.0],
        ["b", "a", "b", "a"],
        weights=[1.0, 2.0, 3.0, 4.0],
        conf_level=0.90,
    )

    assert isinstance(result, survival.YatesResult)
    assert survival.yates is survival.r_api.yates
    assert result.levels == ["a", "b"]
    assert result.means == pytest.approx([6.0, 3.25])
    assert result.se == pytest.approx([math.sqrt(128.0 / 36.0), math.sqrt(10.125 / 16.0)])
    assert result.n == [2, 2]
    assert result.predict_type == "linear"
    assert result.lower[0] == pytest.approx(result.means[0] - 1.6448536269514722 * result.se[0])
    assert result.upper[0] == pytest.approx(result.means[0] + 1.6448536269514722 * result.se[0])

    contrast = survival.yates_contrast(
        [1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        [0.5, 0.25],
        n_obs=3,
        n_vars=2,
        factor_col=1,
        factor_levels=[0.0, 1.0, 2.0],
    )
    assert isinstance(contrast, survival.YatesResult)
    assert contrast.levels == ["0", "1", "2"]
    assert contrast.means == pytest.approx([0.5, 0.75, 1.0])
    assert contrast.se == pytest.approx([0.0, 0.0, 0.0])
    assert contrast.n == [3, 3, 3]

    risk_contrast = survival.yates_contrast(
        [1.0, 0.0, 1.0, 1.0],
        [0.5, 0.25],
        n_obs=2,
        n_vars=2,
        factor_col=1,
        factor_levels=[0.0, 1.0],
        predict_type="risk",
    )
    assert risk_contrast.means == pytest.approx([math.exp(0.5), math.exp(0.75)])

    pairwise = survival.yates_pairwise(contrast)
    assert isinstance(pairwise, survival.YatesPairwiseResult)
    assert pairwise.level1 == ["0", "0", "1"]
    assert pairwise.level2 == ["1", "2", "2"]
    assert pairwise.difference == pytest.approx([-0.25, -0.5, -0.25])
    assert pairwise.se == pytest.approx([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="conf_level must be between 0 and 1"):
        survival.yates([1.0], ["a"], conf_level=1.0)


def test_r_style_nsk_wraps_native_spline_basis():
    basis = survival.nsk([1.0, 2.0, 3.0, 4.0, 5.0], df=3)
    intercept_basis = survival.nsk(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        df=4,
        intercept=True,
    )
    explicit = survival.nsk(
        [1.0, 2.0, 3.0, 4.0],
        knots=[2.0, 3.0],
        **{"Boundary.knots": [1.0, 4.0]},
    )
    range_boundary = survival.nsk(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        df=3,
        Boundary_knots=True,
    )
    boundary_from_knots = survival.nsk(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        knots=[1.0, 2.0, 4.0, 5.0],
        Boundary_knots=None,
    )
    inside_boundary = survival.nsk(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        knots=[2.0, 4.0],
        boundary_knots=[2.5, 3.5],
    )
    missing = survival.nsk([1.0, math.nan, 2.0, 3.0, 4.0, 5.0], df=3)

    assert isinstance(basis, survival._survival.SplineBasisResult)
    assert survival.nsk is survival.r_api.nsk
    assert basis.n_rows == 5
    assert basis.n_cols == 3
    assert basis.knots == pytest.approx([2.6666666666666665, 3.333333333333333])
    assert basis.boundary_knots == pytest.approx((1.2, 4.8))
    assert basis.basis[:3] == pytest.approx(
        [-0.30663390663390683, 0.12972972972972977, -0.007507507507507517]
    )

    assert intercept_basis.n_cols == 4
    assert intercept_basis.basis[:4] == pytest.approx(
        [1.184411684411685, -0.30663390663390683, 0.12972972972972977, -0.007507507507507517]
    )
    assert explicit.boundary_knots == pytest.approx((1.0, 4.0))
    assert explicit.knots == pytest.approx([2.0, 3.0])
    assert explicit.basis == pytest.approx(
        [1.0 if row > 0 and row - 1 == col else 0.0 for row in range(4) for col in range(3)]
    )

    assert range_boundary.boundary_knots == pytest.approx((1.0, 5.0))
    assert range_boundary.knots == pytest.approx([2.333333333333333, 3.6666666666666665])
    assert range_boundary.basis[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)

    assert boundary_from_knots.boundary_knots == pytest.approx((1.0, 5.0))
    assert boundary_from_knots.knots == pytest.approx([2.0, 4.0])
    assert boundary_from_knots.basis[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)

    assert inside_boundary.n_cols == 1
    assert inside_boundary.knots == []
    assert inside_boundary.boundary_knots == pytest.approx((2.0, 4.0))
    assert inside_boundary.basis[0] == pytest.approx(-0.5)

    assert missing.n_rows == 6
    assert missing.n_cols == basis.n_cols
    assert missing.knots == pytest.approx(basis.knots)
    assert missing.boundary_knots == pytest.approx(basis.boundary_knots)
    assert all(math.isnan(value) for value in missing.basis[3:6])
    assert missing.basis[:3] == pytest.approx(basis.basis[:3])
    assert missing.basis[6:] == pytest.approx(basis.basis[3:])

    with pytest.raises(ValueError, match="Boundary.knots"):
        survival.nsk([1.0, 2.0, 3.0], knots=[2.0], Boundary_knots=None)
    with pytest.raises(ValueError, match="x must contain only finite values"):
        survival.nsk([1.0, math.inf], df=3)
    with pytest.raises(ValueError, match="at least one non-missing"):
        survival.nsk([math.nan, math.nan], df=3)


def test_r_style_pspline_builds_survival_basis_contract():
    basis = survival.pspline([1.0, 2.0, 3.0, 4.0, 5.0], df=3)
    outside = survival.pspline(
        [0.0, 1.0, 5.0, 6.0],
        df=3,
        boundary_knots=[1.0, 5.0],
        penalty=False,
    )
    fixed = survival.pspline([1.0, 2.0, 3.0, 4.0, 5.0], theta=0.5)
    aic = survival.pspline([1.0, 2.0, 3.0, 4.0, 5.0], df=0)
    combined = survival.pspline(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        df=3,
        combine=[1] * 10,
    )
    constant = survival.pspline([2.0, 2.0, 2.0], df=2)
    missing = survival.pspline([1.0, float("nan"), 2.0], df=2)

    assert survival.pspline is survival.r_api.pspline
    assert basis["method"] == "df"
    assert basis["nterm"] == 8
    assert basis["degree"] == 3
    assert basis["n_cols"] == 10
    assert basis["boundary_knots"] == pytest.approx([1.0, 5.0])
    assert basis["basis"][0] == pytest.approx([2.0 / 3.0, 1.0 / 6.0, *([0.0] * 8)])
    assert basis["basis"][-1] == pytest.approx([*([0.0] * 7), 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    assert basis["dmat"][0][:3] == pytest.approx([5.0, -4.0, 1.0])
    assert basis["cbase"] == pytest.approx([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

    assert outside["penalty"] is False
    assert outside["basis"][0] == pytest.approx([2.0 / 3.0, -5.0 / 6.0, *([0.0] * 8)])
    assert outside["basis"][-1] == pytest.approx([*([0.0] * 7), -5.0 / 6.0, 2.0 / 3.0, 7.0 / 6.0])
    assert fixed["method"] == "fixed"
    assert fixed["theta"] == pytest.approx(0.5)
    assert fixed["n_cols"] == 12
    assert aic["method"] == "aic"
    assert aic["eps"] == pytest.approx(1e-5)
    assert aic["nterm"] == 15
    assert aic["n_cols"] == 17
    assert combined["combine"] == [1] * 10
    assert combined["n_cols"] == 1
    assert constant["boundary_knots"] == [2.0, 2.0]
    assert constant["basis"][0] == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert constant["basis"] == [constant["basis"][0]] * 3
    assert all(math.isnan(value) for value in missing["basis"][1])

    with pytest.raises(ValueError, match="Invalid value for theta"):
        survival.pspline([1.0, 2.0, 3.0], theta=1.0)
    with pytest.raises(ValueError, match="Too few degrees"):
        survival.pspline([1.0, 2.0, 3.0], df=1)


def test_r_style_frailty_encoding_normalizes_levels_and_sparse_default():
    encoded = r_misc._frailty_encoding(
        ["b", "a", None, "b"],
        levels=["a", "b"],
        sparse=None,
    )
    sparse_encoded = r_misc._frailty_encoding(
        list("abcdef"),
        levels=list("abcdef"),
        sparse=None,
    )

    assert encoded["codes"] == [2, 1, None, 2]
    assert encoded["levels"] == ["a", "b"]
    assert encoded["nclass"] == 2
    assert encoded["sparse"] is False
    assert sparse_encoded["codes"] == [1, 2, 3, 4, 5, 6]
    assert sparse_encoded["sparse"] is True

    with pytest.raises(ValueError, match="outside supplied levels"):
        r_misc._frailty_encoding(["c"], levels=["a", "b"])
