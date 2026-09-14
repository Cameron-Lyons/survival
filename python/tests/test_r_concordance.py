import pytest

from .helpers import setup_survival_import
from .r_api_support import (
    _counting_cox_data,
    _formula_predictor_scores,
    _manual_concordance_bounded_times_and_status,
    _manual_counting_concordance,
    _manual_counting_concordance_counts,
    _manual_counting_time_multipliers,
    _manual_right_concordance_counts,
    _toy_data,
)

survival = setup_survival_import()


def test_concordance_direct_surv_uses_rust_c_index():
    data = _toy_data()
    scores = [8.0 - value for value in data["time"]]
    result = survival.concordance(survival.Surv(data["time"], data["status"]), scores=scores)
    low_level = survival.concordance_index(data["time"], data["status"], scores)
    summary = survival.core.concordance_summary(data["time"], data["status"], scores)

    assert result.concordance == pytest.approx(low_level)
    assert result.c_index == pytest.approx(result.concordance)
    assert result.n == len(data["time"])
    assert result.n_event == sum(data["status"])
    assert result.reverse is False
    assert result.concordant == pytest.approx(summary["concordant"])
    assert result.comparable == pytest.approx(summary["comparable"])
    assert result.conditional_variance == pytest.approx(summary["conditional_variance"])
    assert result.cvar == pytest.approx(result.conditional_variance)

    reversed_result = survival.concordance(
        survival.Surv(data["time"], data["status"]),
        scores=scores,
        reverse=True,
    )
    assert reversed_result.concordance == pytest.approx(
        survival.concordance_index(data["time"], data["status"], [-value for value in scores])
    )
    assert reversed_result.reverse is True


def test_concordance_timefix_false_uses_exact_event_times():
    times = [1.0, 1.0 + 5e-10, 2.0]
    status = [1, 1, 0]
    scores = [0.9, 0.1, 0.5]
    response = survival.Surv(times, status)

    default = survival.concordance(response, scores=scores)
    exact = survival.concordance(response, scores=scores, timefix=False)
    exact_formula_direction = survival.concordance(
        response,
        scores=scores,
        timefix=False,
        reverse=True,
    )
    exact_dotted = survival.concordance(
        "Surv(time, status) ~ score",
        data={"time": times, "status": status, "score": scores},
        **{"time.fix": False},
    )
    fixed_times = [1.0, 1.0, 2.0]
    fixed_concordant, fixed_comparable = _manual_right_concordance_counts(
        fixed_times,
        status,
        scores,
    )
    exact_concordant, exact_comparable = _manual_right_concordance_counts(
        times,
        status,
        scores,
    )

    assert default.concordance == pytest.approx(fixed_concordant / fixed_comparable)
    assert exact.concordance == pytest.approx(exact_concordant / exact_comparable)
    assert exact_dotted.concordance == pytest.approx(exact_formula_direction.concordance)
    assert default.concordant == pytest.approx(fixed_concordant)
    assert default.comparable == pytest.approx(fixed_comparable)
    assert exact.concordant == pytest.approx(exact_concordant)
    assert exact.comparable == pytest.approx(exact_comparable)
    assert default.concordance != pytest.approx(exact.concordance)
    assert default.n_event == exact.n_event == sum(status)


def test_concordance_accepts_r_style_defaults():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
        "score": [0.8, 0.6, 0.2, 0.1],
    }

    default = survival.concordance("Surv(time, status) ~ score", data=data)
    explicit = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights=None,
        subset=None,
        cluster=None,
        ymin=None,
        ymax=None,
        timewt="n",
        influence=0,
        ranks=False,
        reverse=False,
        timefix=True,
        keepstrata=10,
    )
    r_default_timewt = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        timewt=("n", "S", "S/G", "n/G2", "I"),
        influence=None,
        ranks=None,
        keepstrata=None,
    )

    assert explicit.concordance == pytest.approx(default.concordance)
    assert explicit.concordant == pytest.approx(default.concordant)
    assert explicit.comparable == pytest.approx(default.comparable)
    assert r_default_timewt.concordance == pytest.approx(default.concordance)

    tied_risk = survival.concordance(
        survival.Surv([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 1]),
        scores=[0.2, 0.4, 0.4, 1.0],
        reverse=True,
    )
    assert tied_risk.concordance == pytest.approx(0.9)
    assert tied_risk.concordant == pytest.approx(4.5)
    assert tied_risk.comparable == pytest.approx(5.0)
    assert tied_risk.tied_x == pytest.approx(1.0)
    assert tied_risk.tied_y == pytest.approx(0.0)
    assert tied_risk.tied_xy == pytest.approx(0.0)
    assert survival.as_data_frame(tied_risk)["tied.x"] == pytest.approx([1.0])

    same_event_time = survival.concordance(
        survival.Surv([1.0, 2.0, 2.0, 3.0, 4.0], [1, 1, 1, 0, 1]),
        scores=[0.2, 0.4, 0.4, 0.8, 1.0],
        reverse=True,
    )
    assert same_event_time.concordance == pytest.approx(1.0)
    assert same_event_time.concordant == pytest.approx(8.0)
    assert same_event_time.comparable == pytest.approx(8.0)
    assert same_event_time.tied_x == pytest.approx(0.0)
    assert same_event_time.tied_y == pytest.approx(0.0)
    assert same_event_time.tied_xy == pytest.approx(1.0)


def test_survConcordance_deprecated_wrappers_keep_legacy_direction():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
        "score": [0.2, 0.8, 0.4, 0.6],
    }

    with pytest.warns(DeprecationWarning, match="survConcordance"):
        old = survival.survConcordance("Surv(time, status) ~ score", data=data)
    modern = survival.concordance("Surv(time, status) ~ score", data=data)
    with pytest.warns(DeprecationWarning, match="survConcordance.fit"):
        fit_stats = survival.survConcordance_fit(
            survival.Surv(data["time"], data["status"]),
            data["score"],
        )

    assert old.concordance != pytest.approx(modern.concordance)
    assert fit_stats["concordant"] == pytest.approx(old.concordant)
    assert fit_stats["discordant"] == pytest.approx(old.comparable - old.concordant)
    assert list(fit_stats) == ["concordant", "discordant", "tied.risk", "tied.time", "std(c-d)"]


def test_concordance_ranks_return_weighted_event_contributions():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.6, 0.4, 0.1],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])

    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    ranked = survival.concordance(response, scores=data["score"], weights=data["wt"], ranks=True)
    formula_ranked = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        ranks=True,
    )
    reversed_ranked = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ranks=True,
        reverse=True,
    )

    assert default.ranks is None
    assert ranked.ranks is not None
    assert [row["time"] for row in ranked.ranks] == pytest.approx([1.0, 2.0, 3.0])
    assert [row["casewt"] for row in ranked.ranks] == pytest.approx([2.0, 1.0, 3.0])
    assert [row["timewt"] for row in ranked.ranks] == pytest.approx([7.0, 5.0, 4.0])
    assert [row["rank"] for row in ranked.ranks] == pytest.approx([5.0 / 7.0, 4.0 / 5.0, 0.25])
    assert ranked.concordance == pytest.approx(default.concordance)
    assert formula_ranked.ranks is not None
    for formula_row, direct_row in zip(formula_ranked.ranks, reversed_ranked.ranks, strict=True):
        assert formula_row.keys() == direct_row.keys()
        for key in direct_row:
            assert formula_row[key] == pytest.approx(direct_row[key])
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in ranked.ranks) == (
        pytest.approx(2.0 * ranked.concordant - ranked.comparable)
    )
    assert sum(
        row["rank"] * row["timewt"] * row["casewt"] for row in reversed_ranked.ranks
    ) == pytest.approx(2.0 * reversed_ranked.concordant - reversed_ranked.comparable)
    assert reversed_ranked.concordance == pytest.approx(1.0 - ranked.concordance)


def test_concordance_influence_modes_return_r_style_diagnostics():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])

    dfbeta_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=1,
    )
    influence_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=2,
    )
    both = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
    )
    bool_alias = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=True,
    )
    reversed_both = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
        reverse=True,
    )

    assert dfbeta_only.dfbeta is not None
    assert dfbeta_only.influence is None
    assert influence_only.dfbeta is None
    assert influence_only.influence is not None
    assert both.dfbeta == pytest.approx(dfbeta_only.dfbeta)
    assert bool_alias.dfbeta == pytest.approx(dfbeta_only.dfbeta)
    assert both.influence is not None
    assert influence_only.influence is not None
    for both_row, influence_row in zip(both.influence, influence_only.influence, strict=True):
        assert both_row == pytest.approx(influence_row)
    assert both.variance == pytest.approx(sum(value * value for value in both.dfbeta))
    assert both.var == pytest.approx(both.variance)
    assert [sum(row[col] for row in both.influence) for col in range(5)] == pytest.approx(
        [both.concordant, both.comparable - both.concordant, 0.0, 0.0, 0.0]
    )
    assert sum(both.dfbeta) == pytest.approx(0.0)
    assert reversed_both.concordance == pytest.approx(1.0 - both.concordance)
    assert reversed_both.dfbeta == pytest.approx([-value for value in both.dfbeta])
    assert [sum(row[col] for row in reversed_both.influence) for col in range(5)] == pytest.approx(
        [both.comparable - both.concordant, both.concordant, 0.0, 0.0, 0.0]
    )


def test_concordance_cluster_collapses_dfbeta_for_robust_variance():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
        "wt": [2.0, 1.0, 3.0, 1.0],
        "cluster": ["a", "a", "b", "c"],
    }
    response = survival.Surv(data["time"], data["status"])

    unclustered = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
    )
    unclustered_formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
        reverse=True,
    )
    clustered = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        cluster=data["cluster"],
        influence=3,
    )
    formula_clustered = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        cluster="cluster",
        influence=1,
    )
    formula_term_clustered = survival.concordance(
        "Surv(time, status) ~ score + cluster(cluster)",
        data=data,
        weights="wt",
        influence=1,
    )
    variance_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        cluster=data["cluster"],
    )

    expected_dfbeta = [
        unclustered.dfbeta[0] + unclustered.dfbeta[1],
        unclustered.dfbeta[2],
        unclustered.dfbeta[3],
    ]
    expected_formula_dfbeta = [
        unclustered_formula_direction.dfbeta[0] + unclustered_formula_direction.dfbeta[1],
        unclustered_formula_direction.dfbeta[2],
        unclustered_formula_direction.dfbeta[3],
    ]
    assert clustered.dfbeta == pytest.approx(expected_dfbeta)
    assert formula_clustered.dfbeta == pytest.approx(expected_formula_dfbeta)
    assert formula_term_clustered.dfbeta == pytest.approx(expected_formula_dfbeta)
    assert clustered.variance == pytest.approx(sum(value * value for value in expected_dfbeta))
    assert formula_clustered.variance == pytest.approx(clustered.variance)
    assert formula_term_clustered.variance == pytest.approx(clustered.variance)
    assert variance_only.dfbeta is None
    assert variance_only.influence is None
    assert variance_only.variance == pytest.approx(clustered.variance)
    for clustered_row, unclustered_row in zip(
        clustered.influence,
        unclustered.influence,
        strict=True,
    ):
        assert clustered_row == pytest.approx(unclustered_row)


def test_concordance_accepts_case_weights():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.4, 0.9, 0.2, 0.1],
        "wt": [5.0, 1.0, 2.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    expected_concordant, expected_comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
    )

    direct = survival.concordance(response, scores=data["score"], weights=data["wt"])
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        reverse=True,
    )
    formula = survival.concordance("Surv(time, status) ~ score", data=data, weights="wt")
    formula_vector = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights=data["wt"],
    )
    low_level = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
    )

    assert direct.concordance == pytest.approx(expected_concordant / expected_comparable)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert formula_vector.concordance == pytest.approx(formula_direction.concordance)
    assert direct.concordant == pytest.approx(expected_concordant)
    assert direct.comparable == pytest.approx(expected_comparable)
    assert low_level["concordance"] == pytest.approx(direct.concordance)
    assert low_level["concordant"] == pytest.approx(expected_concordant)
    assert low_level["comparable"] == pytest.approx(expected_comparable)
    assert survival.concordance(response, scores=data["score"]).concordance != pytest.approx(
        direct.concordance
    )
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
    ) == pytest.approx(direct.concordance)


def test_concordance_accepts_identity_time_weight():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="I",
    )
    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
        reverse=True,
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        timewt="I",
    )
    summary = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    ) == pytest.approx(direct.concordance)
    assert direct.concordance != pytest.approx(default.concordance)


@pytest.mark.parametrize("timewt", ["S", "S/G", "n/G2"])
def test_concordance_accepts_km_time_weights(timewt):
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 0, 1, 0, 1, 1],
        "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.4],
        "wt": [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
        timewt=timewt,
    )

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt=timewt,
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt=timewt,
        reverse=True,
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        timewt=timewt,
    )
    summary = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt=timewt,
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt=timewt,
    ) == pytest.approx(direct.concordance)


def test_concordance_accepts_time_window_restrictions():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 1, 0],
        "score": [0.2, 0.9, 0.1, 0.8, 0.3, 0.4],
        "wt": [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
    }
    ymin = 2.5
    ymax = 4.0
    bounded_time, bounded_status = _manual_concordance_bounded_times_and_status(
        data["time"],
        data["status"],
        ymin=ymin,
        ymax=ymax,
    )
    concordant, comparable = _manual_right_concordance_counts(
        bounded_time,
        bounded_status,
        data["score"],
        data["wt"],
        timewt="S",
    )
    response = survival.Surv(data["time"], data["status"])

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
        reverse=True,
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert direct.n_event == sum(bounded_status)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert formula.n_event == direct.n_event


def test_concordance_formula_applies_subset_and_na_action():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 0, 1, 0],
        "score": [5.0, None, 3.0, 2.0, 1.0],
    }
    result = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        subset=[0, 1, 2, 3],
        na_action="omit",
    )
    dotted = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        subset=[0, 1, 2, 3],
        **{"na.action": "omit"},
    )

    assert result.concordance == pytest.approx(
        survival.concordance_index([1.0, 3.0, 4.0], [1, 0, 1], [-5.0, -3.0, -2.0])
    )
    assert dotted.concordance == pytest.approx(result.concordance)
    assert result.n == 3
    assert result.n_event == 2
    assert dotted.n == result.n
    assert dotted.n_event == result.n_event


def test_concordance_formula_rejects_offset_terms_like_r():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.2, 0.3, 0.4, 0.5],
        "offset": [1.0, 0.0, 0.0, -1.0],
        "bonus": [0.1, -0.2, 0.3, -0.4],
    }

    with pytest.raises(ValueError, match="Offset terms not allowed"):
        survival.concordance("Surv(time, status) ~ score + offset(offset)", data=data)
    with pytest.raises(ValueError, match="Offset terms not allowed"):
        survival.concordance(
            "Surv(time, status) ~ score + offset(offset + bonus)",
            data=data,
        )


def test_concordance_formula_rejects_offset_only_predictor_like_r():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
        "offset": [1.0, 0.5, 0.2, 0.0],
    }

    with pytest.raises(ValueError, match="Offset terms not allowed"):
        survival.concordance("Surv(time, status) ~ offset(offset)", data=data)


def test_concordance_formula_returns_one_result_per_score_column():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 0, 1, 1],
        "x1": [0.8, 0.6, 0.4, 0.2, 0.1],
        "x2": [0.1, 0.5, 0.2, 0.7, 0.3],
    }
    result = survival.concordance("Surv(time, status) ~ x1 + x2", data=data)
    x1 = survival.core.concordance_summary(
        data["time"],
        data["status"],
        _formula_predictor_scores(data["x1"]),
    )
    x2 = survival.core.concordance_summary(
        data["time"],
        data["status"],
        _formula_predictor_scores(data["x2"]),
    )

    assert result.score_names == ["x1", "x2"]
    assert result.concordance == pytest.approx([x1["concordance"], x2["concordance"]])
    assert result.concordant == pytest.approx([x1["concordant"], x2["concordant"]])
    assert result.comparable == pytest.approx([x1["comparable"], x2["comparable"]])
    assert result.tied_x == pytest.approx([0.0, 0.0])
    assert result.tied_y == pytest.approx([0.0, 0.0])
    assert result.tied_xy == pytest.approx([0.0, 0.0])
    assert result.conditional_variance == pytest.approx(
        [x1["conditional_variance"], x2["conditional_variance"]]
    )
    assert result.n == len(data["time"])
    assert result.n_event == sum(data["status"])


def test_concordance_matrix_scores_return_parallel_cluster_variances():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 1, 0],
        "x1": [0.8, 0.6, 0.4, 0.2, 0.1, 0.3],
        "x2": [0.1, 0.5, 0.2, 0.7, 0.3, 0.9],
        "wt": [1.0, 2.0, 1.0, 0.5, 1.5, 1.0],
        "cluster": ["a", "a", "b", "b", "c", "c"],
    }
    response = survival.Surv(data["time"], data["status"])
    scores = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]

    result = survival.concordance(
        response,
        scores=scores,
        weights=data["wt"],
        cluster=data["cluster"],
    )
    x1 = survival.concordance(
        response,
        scores=data["x1"],
        weights=data["wt"],
        cluster=data["cluster"],
    )
    x2 = survival.concordance(
        response,
        scores=data["x2"],
        weights=data["wt"],
        cluster=data["cluster"],
    )

    assert result.score_names == ["score1", "score2"]
    assert result.concordance == pytest.approx([x1.concordance, x2.concordance])
    assert result.concordant == pytest.approx([x1.concordant, x2.concordant])
    assert result.comparable == pytest.approx([x1.comparable, x2.comparable])
    assert result.tied_x == pytest.approx([x1.tied_x, x2.tied_x])
    assert result.tied_y == pytest.approx([x1.tied_y, x2.tied_y])
    assert result.tied_xy == pytest.approx([x1.tied_xy, x2.tied_xy])
    assert result.variance == pytest.approx([x1.variance, x2.variance])
    assert result.conditional_variance == pytest.approx(
        [x1.conditional_variance, x2.conditional_variance]
    )


def test_concordance_summary_low_level_reports_pair_counts():
    data = _toy_data()
    scores = [8.0 - value for value in data["time"]]
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        scores,
    )

    summary = survival.core.concordance_summary(data["time"], data["status"], scores)

    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert summary["concordance"] == pytest.approx(concordant / comparable)
    assert summary["concordance"] == pytest.approx(
        survival.core.concordance_index(data["time"], data["status"], scores)
    )


def test_concordance_summary_low_level_reports_weighted_tie_counts():
    summary = survival.core.concordance_summary(
        [1.0, 2.0, 2.0, 2.0, 3.0],
        [1, 1, 1, 1, 0],
        [0.1, 0.5, 0.5, 0.9, 0.5],
        weights=[1.0, 2.0, 3.0, 4.0, 5.0],
        timewt="I",
    )

    assert summary["tied_x"] == pytest.approx(25.0 / 14.0)
    assert summary["tied_y"] == pytest.approx(10.0 / 7.0)
    assert summary["tied_xy"] == pytest.approx(3.0 / 7.0)


def test_concordance_formula_accepts_strata_wrapper():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 1],
        "score": [2.0, 1.0, 100.0, 99.0],
        "group": ["A", "A", "B", "B"],
    }

    stratified = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
    )
    collapsed = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=False,
    )
    thresholded = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=0,
    )
    retained = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=True,
    )
    ranked = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        ranks=True,
    )
    influential = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        influence=3,
    )
    unstratified = survival.concordance("Surv(time, status) ~ score", data=data)

    assert stratified.concordance == pytest.approx(0.0)
    assert collapsed.concordance == pytest.approx(stratified.concordance)
    assert thresholded.concordance == pytest.approx(stratified.concordance)
    assert retained.concordance == pytest.approx(stratified.concordance)
    assert unstratified.concordance > stratified.concordance
    assert stratified.n == len(data["time"])
    assert stratified.n_event == sum(data["status"])
    assert ranked.ranks == [
        {"time": 1.0, "rank": -0.5, "timewt": 2.0, "casewt": 1.0},
        {"time": 2.0, "rank": 0.0, "timewt": 1.0, "casewt": 1.0},
        {"time": 3.0, "rank": -0.5, "timewt": 2.0, "casewt": 1.0},
        {"time": 4.0, "rank": 0.0, "timewt": 1.0, "casewt": 1.0},
    ]
    assert influential.influence == [
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
    ]
    assert influential.dfbeta == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert influential.variance == pytest.approx(0.0)


def test_concordance_formula_strata_ranks_support_counting_process_response():
    data = {
        "start": [0.0, 0.0, 0.0, 0.0],
        "stop": [1.0, 2.0, 1.0, 2.0],
        "status": [1, 0, 1, 0],
        "score": [0.9, 0.1, 0.2, 0.8],
        "group": ["A", "A", "B", "B"],
    }

    ranked = survival.concordance(
        "Surv(start, stop, status) ~ score + strata(group)",
        data=data,
        ranks=True,
    )
    influential = survival.concordance(
        "Surv(start, stop, status) ~ score + strata(group)",
        data=data,
        influence=3,
    )

    assert ranked.ranks == [
        {"time": 1.0, "rank": -0.5, "timewt": 2.0, "casewt": 1.0},
        {"time": 1.0, "rank": 0.5, "timewt": 2.0, "casewt": 1.0},
    ]
    assert influential.influence == [
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0],
    ]
    assert influential.dfbeta == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert influential.variance == pytest.approx(0.0)


def test_concordance_counting_process_uses_delayed_entry_risk_sets():
    data = _counting_cox_data()
    scores = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    response = survival.Surv(data["start"], data["stop"], data["status"])

    result = survival.concordance(response, scores=scores)
    expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    reversed_result = survival.concordance(response, scores=scores, reverse=True)
    reversed_expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        [-value for value in scores],
    )

    assert result.concordance == pytest.approx(expected)
    assert result.n == len(data["stop"])
    assert result.n_event == sum(data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    assert result.concordant == pytest.approx(concordant)
    assert result.comparable == pytest.approx(comparable)
    assert reversed_result.concordance == pytest.approx(reversed_expected)
    assert reversed_result.reverse is True


def test_concordance_counting_process_ranks_use_delayed_entry_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.5, 1.5],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
    }
    scores = [0.9, 0.7, 0.4, 0.1]
    response = survival.Surv(data["start"], data["stop"], data["status"])

    ranked = survival.concordance(response, scores=scores, ranks=True)
    exact_ranked = survival.concordance(response, scores=scores, ranks=True, timefix=False)
    formula_direction_ranked = survival.concordance(
        response,
        scores=scores,
        ranks=True,
        reverse=True,
    )
    formula_ranked = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data={**data, "score": scores},
        ranks=True,
    )

    assert ranked.ranks is not None
    assert exact_ranked.ranks is not None
    assert formula_ranked.ranks is not None
    assert [row["time"] for row in ranked.ranks] == pytest.approx(
        sorted(data["stop"][idx] for idx, event in enumerate(data["status"]) if event == 1)
    )
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in ranked.ranks) == (
        pytest.approx(2.0 * ranked.concordant - ranked.comparable)
    )
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in exact_ranked.ranks) == (
        pytest.approx(2.0 * exact_ranked.concordant - exact_ranked.comparable)
    )
    for formula_row, direct_row in zip(
        formula_ranked.ranks,
        formula_direction_ranked.ranks,
        strict=True,
    ):
        assert formula_row.keys() == direct_row.keys()
        for key in direct_row:
            assert formula_row[key] == pytest.approx(direct_row[key])


def test_concordance_counting_process_influence_uses_delayed_entry_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.5, 1.5],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    result = survival.concordance(response, scores=data["score"], influence=3)
    exact = survival.concordance(response, scores=data["score"], influence=3, timefix=False)
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        influence=3,
        reverse=True,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        influence=3,
    )

    assert result.dfbeta is not None
    assert result.influence is not None
    assert exact.dfbeta is not None
    assert exact.influence is not None
    assert formula.dfbeta == pytest.approx(formula_direction.dfbeta)
    assert formula.influence is not None
    assert result.influence is not None
    for formula_row, result_row in zip(formula.influence, formula_direction.influence, strict=True):
        assert formula_row == pytest.approx(result_row)
    assert [sum(row[col] for row in result.influence) for col in range(5)] == pytest.approx(
        [result.concordant, result.comparable - result.concordant, 0.0, 0.0, 0.0]
    )
    assert result.variance == pytest.approx(sum(value * value for value in result.dfbeta))
    assert exact.variance == pytest.approx(sum(value * value for value in exact.dfbeta))


def test_concordance_counting_process_timefix_false_uses_exact_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.0],
        "stop": [1.0, 1.0 + 5e-10, 2.0],
        "status": [1, 0, 0],
        "score": [0.5, 0.9, 0.1],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    default = survival.concordance(response, scores=data["score"])
    exact = survival.concordance(response, scores=data["score"], timefix=False)
    exact_formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        timefix=False,
    )
    fixed_expected = _manual_counting_concordance(
        data["start"],
        [1.0, 1.0, 2.0],
        data["status"],
        data["score"],
    )
    exact_expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
    )
    fixed_low_level = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        timefix=True,
    )
    exact_low_level = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        timefix=False,
    )

    assert default.concordance == pytest.approx(fixed_expected)
    assert exact.concordance == pytest.approx(exact_expected)
    assert exact_formula.concordance == pytest.approx(exact.concordance)
    assert default.concordance == pytest.approx(fixed_low_level["concordance"])
    assert exact.concordance == pytest.approx(exact_low_level["concordance"])
    assert default.concordance != pytest.approx(exact.concordance)


def test_counting_concordance_low_level_matches_manual_risk_sets():
    data = _counting_cox_data()
    scores = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )

    result = survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    expected = concordant / comparable

    assert result == pytest.approx(expected)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert summary["concordance"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="same length"):
        survival.counting_concordance_index([0.0], [1.0, 2.0], [1], [0.5])


def test_counting_concordance_accepts_case_weights():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    data["wt"] = [2.0, 1.0, 3.0, 1.5, 0.5, 4.0]
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
    )
    expected = concordant / comparable

    direct = survival.concordance(response, scores=data["score"], weights=data["wt"])
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        reverse=True,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
    )
    low_level = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
    )

    assert direct.concordance == pytest.approx(expected)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert low_level["concordance"] == pytest.approx(expected)
    assert low_level["concordant"] == pytest.approx(concordant)
    assert low_level["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
    ) == pytest.approx(expected)


def test_counting_concordance_accepts_identity_time_weight():
    data = {
        "start": [0.0, 0.0, 0.0, 1.0],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="I",
    )
    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
        reverse=True,
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        timewt="I",
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    ) == pytest.approx(direct.concordance)
    assert direct.concordance != pytest.approx(default.concordance)


def test_counting_concordance_accepts_survival_time_weight():
    data = {
        "start": [0.0, 0.0, 0.0, 1.0],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="S",
    )

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
        reverse=True,
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        timewt="S",
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="S",
    ) == pytest.approx(direct.concordance)


def test_counting_concordance_duplicate_event_times_share_survival_weight():
    data = {
        "start": [0.0, 0.0, 0.25, 0.0, 1.0],
        "stop": [1.0, 1.0, 2.0, 2.0, 3.0],
        "status": [1, 1, 1, 0, 1],
        "score": [0.9, 0.2, 0.7, 0.1, 0.8],
        "wt": [2.0, 1.0, 3.0, 0.5, 4.0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="S",
    )
    multipliers = _manual_counting_time_multipliers(
        data["start"],
        data["stop"],
        data["status"],
        data["wt"],
        "S",
    )

    result = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
        ranks=True,
        influence=3,
    )

    assert result.concordant == pytest.approx(concordant)
    assert result.comparable == pytest.approx(comparable)
    assert result.concordance == pytest.approx(concordant / comparable)
    assert result.ranks is not None
    assert [row["time"] for row in result.ranks] == pytest.approx([1.0, 1.0, 2.0, 3.0])
    assert [row["rank"] for row in result.ranks] == pytest.approx(
        [9.0 / 13.0, -9.0 / 13.0, -7.0 / 15.0, 0.0]
    )
    assert [row["timewt"] for row in result.ranks] == pytest.approx(
        [
            6.5 * multipliers[1.0],
            6.5 * multipliers[1.0],
            7.5 * multipliers[2.0],
            4.0 * multipliers[3.0],
        ]
    )
    assert result.influence is not None
    column_sums = [sum(row[col] for row in result.influence) for col in range(5)]
    tied_event_weight = data["wt"][0] * data["wt"][1] * multipliers[1.0]
    assert column_sums[0] == pytest.approx(concordant)
    assert column_sums[1] == pytest.approx(comparable - concordant)
    assert column_sums[2] == pytest.approx(0.0)
    assert column_sums[3] == pytest.approx(tied_event_weight)
    assert column_sums[4] == pytest.approx(0.0)
    assert result.variance == pytest.approx(sum(value * value for value in result.dfbeta))


def test_counting_concordance_accepts_time_window_restrictions():
    data = {
        "start": [0.0, 0.0, 0.5, 1.0, 2.5],
        "stop": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0, 0.4],
        "wt": [2.0, 1.0, 3.0, 1.0, 0.5],
    }
    ymin = 1.5
    ymax = 3.0
    bounded_stop, bounded_status = _manual_concordance_bounded_times_and_status(
        data["stop"],
        data["status"],
        ymin=ymin,
        ymax=ymax,
    )
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        bounded_stop,
        bounded_status,
        data["score"],
        data["wt"],
        timewt="S",
    )
    response = survival.Surv(data["start"], data["stop"], data["status"])

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )
    formula_direction = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
        reverse=True,
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert direct.n_event == sum(bounded_status)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(formula_direction.concordance)
    assert formula.n_event == direct.n_event


def test_concordance_formula_strata_supports_counting_process_response():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    data["group"] = ["A", "A", "A", "B", "B", "B"]

    result = survival.concordance(
        "Surv(start, stop, status) ~ score + strata(group)",
        data=data,
    )
    total_concordant = 0.0
    total_comparable = 0.0
    for group in ("A", "B"):
        indices = [idx for idx, value in enumerate(data["group"]) if value == group]
        concordant, comparable = _manual_counting_concordance_counts(
            [data["start"][idx] for idx in indices],
            [data["stop"][idx] for idx in indices],
            [data["status"][idx] for idx in indices],
            [-data["score"][idx] for idx in indices],
        )
        total_concordant += concordant
        total_comparable += comparable

    assert result.concordance == pytest.approx(total_concordant / total_comparable)
    assert result.concordant == pytest.approx(total_concordant)
    assert result.comparable == pytest.approx(total_comparable)
    assert result.n_event == sum(data["status"])


def test_concordance_formula_accepts_counting_process_response():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]

    result = survival.concordance("Surv(start, stop, status) ~ score", data=data)
    expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        _formula_predictor_scores(data["score"]),
    )

    assert result.concordance == pytest.approx(expected)
    assert result.n_event == sum(data["status"])


def test_low_level_concordance_remains_available_from_core_module():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = [1, 2, 1, 2, 1]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    timewt = [1.0, 1.0, 1.0, 1.0, 1.0]
    sortstart = None
    sortstop = [0, 1, 2, 3, 4]

    result = survival.core.concordance(y, x, wt, timewt, sortstart, sortstop)
    assert isinstance(result, dict)
    assert "count" in result
