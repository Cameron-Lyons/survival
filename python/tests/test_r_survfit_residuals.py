import importlib

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r_survfit_residuals = importlib.import_module("survival.r._survfit_residuals")


def test_integrated_step_values_preserve_boundaries_and_preorigin_state():
    result = r_survfit_residuals._integrated_step_values(
        [-1.0, 1.0, 1.0, 3.0],
        [0.25, 0.5, 0.75, 1.0],
        [4.0, 0.0, 0.5, 1.0, 2.0, -1.0],
        start_time=0.0,
        initial_value=0.1,
    )

    assert result == pytest.approx([2.75, 0.0, 0.125, 0.25, 1.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        r_survfit_residuals._integrated_step_values(
            [1.0, 2.0],
            [0.5],
            [2.0],
            start_time=0.0,
            initial_value=0.0,
        )


def test_pseudo_accepts_survfit_results_and_preserves_direct_api():
    response = survival.Surv([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
    fit = survival.survfit(response)
    grouped_fit = survival.survfit(response, group=["A", "A", "B", "B"], model=True)
    formula_fit = survival.survfit(
        "Surv(time, status) ~ group",
        data={
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": [1, 0, 1, 1],
            "group": ["A", "A", "B", "B"],
        },
        model=True,
    )
    matrix = survival.pseudo(fit, times=[1.0, 2.0, 3.0])
    vector = survival.pseudo(fit, times=[2.0])
    frame = survival.pseudo(fit, times=[2.0], data_frame=True)
    uncollapsed = survival.pseudo(fit, times=[1.0, 2.0, 3.0], collapse=False)
    grouped = survival.pseudo(grouped_fit, times=[1.0, 2.0, 3.0])
    grouped_frame = survival.pseudo(grouped_fit, times=[2.0], data_frame=True)
    grouped_uncollapsed = survival.pseudo(
        grouped_fit,
        times=[1.0, 2.0, 3.0],
        collapse=False,
    )
    formula_grouped = survival.pseudo(formula_fit, times=[1.0, 2.0, 3.0])
    direct = survival.pseudo([1.0, 2.0, 3.0], [1, 0, 1], None, "survival")

    for actual_row, expected_row in zip(
        matrix,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, -0.25],
            [1.0, 1.0, 1.25],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(uncollapsed, matrix, strict=True):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(vector, [[0.0], [1.0], [1.0], [1.0]], strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert frame == {
        "id": [1, 2, 3, 4],
        "time": [2.0, 2.0, 2.0, 2.0],
        "pseudo": [0.0, 1.0, 1.0, 1.0],
    }
    assert list(grouped) == ["A", "B"]
    for actual_row, expected_row in zip(
        grouped["A"],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        grouped["B"],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert list(grouped_uncollapsed) == list(grouped)
    for label in grouped:
        for actual_row, expected_row in zip(
            grouped_uncollapsed[label],
            grouped[label],
            strict=True,
        ):
            assert actual_row == pytest.approx(expected_row)
    assert list(formula_grouped) == list(grouped)
    for label in grouped:
        for actual_row, expected_row in zip(formula_grouped[label], grouped[label], strict=True):
            assert actual_row == pytest.approx(expected_row)
    assert grouped_frame == {
        "strata": ["A", "A", "B", "B"],
        "id": [1, 2, 1, 2],
        "time": [2.0, 2.0, 2.0, 2.0],
        "pseudo": [0.0, 1.0, 1.0, 1.0],
    }
    assert type(direct).__name__ == "PseudoResult"
    default_fit_pseudo = survival.pseudo(survival.survfit(response), times=[1.0])
    for actual_row, expected_row in zip(
        default_fit_pseudo,
        [[0.0], [1.0], [1.0], [1.0]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)


def test_pseudo_supports_counting_process_survfit_results():
    response = survival.Surv(
        [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        [0, 1, 1, 0, 0, 1],
    )
    ids = [1, 1, 2, 2, 3, 3]
    fit = survival.survfit(response, id=ids, model=True)
    no_id_fit = survival.survfit(response, model=True)

    survival_values = survival.pseudo(fit, times=[3.0, 5.0, 7.0])
    survival_vector = survival.pseudo(fit, times=[5.0])
    survival_frame = survival.pseudo(fit, times=[5.0], data_frame=True)
    row_level = survival.pseudo(fit, times=[3.0, 5.0, 7.0], collapse=False)
    cumhaz = survival.pseudo(fit, times=[3.0, 5.0, 7.0], type="cumhaz")
    rmst = survival.pseudo(fit, times=[3.0, 5.0, 7.0], type="rmst")
    no_id = survival.pseudo(no_id_fit, times=[3.0, 5.0])

    for actual_row, expected_row in zip(
        survival_values,
        [
            [1.0, 0.2222222, 0.0],
            [0.0, 0.2222222, 0.0],
            [1.0, 0.8888889, 0.0],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        survival_vector,
        [[0.2222222], [0.2222222], [0.8888889]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert survival_frame["id"] == [1, 2, 3]
    assert survival_frame["time"] == [5.0, 5.0, 5.0]
    assert survival_frame["pseudo"] == pytest.approx([0.2222222, 0.2222222, 0.8888889])
    for actual_row, expected_row in zip(
        row_level,
        [
            [0.6666667, 0.4444444, 0.0],
            [1.0, 0.2222222, 0.0],
            [0.0, 0.0, 0.0],
            [0.6666667, 0.6666667, 0.0],
            [1.0, 0.6666667, 0.0],
            [0.6666667, 0.6666667, 0.0],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        cumhaz,
        [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        rmst,
        [[3.0, 5.0, 5.4444444], [3.0, 3.0, 3.4444444], [3.0, 5.0, 6.7777778]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        no_id,
        [
            [0.6666667, 0.4444444],
            [1.3333333, 0.0],
            [-0.6666667, -0.4444444],
            [0.6666667, 0.8888889],
            [1.3333333, 0.8888889],
            [0.6666667, 0.8888889],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    with pytest.raises(TypeError, match="times are required"):
        survival.pseudo(fit)
    grouped_response = survival.Surv(
        [0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0],
        [2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0, 8.0],
        [0, 1, 1, 0, 0, 1, 1, 0],
    )
    grouped_ids = [1, 1, 2, 2, 3, 3, 4, 4]
    grouped_fit = survival.survfit(
        grouped_response,
        group=["A", "A", "A", "A", "B", "B", "B", "B"],
        id=grouped_ids,
        model=True,
    )
    grouped = survival.pseudo(grouped_fit, times=[3.0, 5.0])
    grouped_row_level = survival.pseudo(grouped_fit, times=[3.0, 5.0], collapse=False)
    grouped_cumhaz = survival.pseudo(grouped_fit, times=[3.0, 5.0], type="cumhaz")
    grouped_rmst = survival.pseudo(grouped_fit, times=[3.0, 5.0], type="rmst")

    for label, expected_rows in {
        "A": [[1.0, 0.25], [0.0, 0.25]],
        "B": [[1.0, 1.0], [1.0, 0.0]],
    }.items():
        for actual_row, expected_row in zip(grouped[label], expected_rows, strict=True):
            assert actual_row == pytest.approx(expected_row)
    for label, expected_rows in {
        "A": [[0.5, 0.25], [1.0, 0.25], [0.0, 0.0], [0.5, 0.5]],
        "B": [[1.0, 0.5], [1.0, 1.0], [1.0, 0.0], [1.0, 0.5]],
    }.items():
        for actual_row, expected_row in zip(grouped_row_level[label], expected_rows, strict=True):
            assert actual_row == pytest.approx(expected_row)
    for label, expected_rows in {
        "A": [[0.0, 1.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [0.0, 1.0]],
    }.items():
        for actual_row, expected_row in zip(grouped_cumhaz[label], expected_rows, strict=True):
            assert actual_row == pytest.approx(expected_row)
    for label, expected_rows in {
        "A": [[3.0, 5.0], [3.0, 3.0]],
        "B": [[3.0, 5.0], [3.0, 5.0]],
    }.items():
        for actual_row, expected_row in zip(grouped_rmst[label], expected_rows, strict=True):
            assert actual_row == pytest.approx(expected_row)


def test_multistate_survfit_residuals_and_pseudo_values_use_core_influences():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Factor(
            ["ill", "death", "censor", "death", "ill", "censor"],
            ["censor", "ill", "death"],
        ),
    )
    fit = survival.survfit(response, model=True)
    residuals = survival.survfit_residuals(fit, times=[2.0, 5.0])
    cumhaz = survival.survfit_residuals(
        fit,
        times=[2.0, 5.0],
        type="cumhaz",
    )
    sojourn = survival.survfit_residuals(
        fit,
        times=[2.0, 5.0],
        type="sojourn",
    )
    pseudo = survival.pseudo(fit, times=[2.0, 5.0])

    assert residuals["columns"] == ["(s0)", "ill", "death"]
    assert residuals["column_name"] == "state"
    assert residuals["resid"][0][0] == pytest.approx([-1 / 9, -1 / 27])
    assert residuals["resid"][0][1] == pytest.approx([5 / 36, 11 / 108])
    assert residuals["resid"][5][0] == pytest.approx([1 / 18, 1 / 6])
    assert cumhaz["columns"] == ["1:2", "1:3"]
    assert cumhaz["column_name"] == "transition"
    for actual, expected in zip(
        cumhaz["resid"][0],
        [[5 / 36, 5 / 36], [0.0, 0.0]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        sojourn["resid"][0],
        [[-5 / 36, -47 / 108], [5 / 36, 5 / 9], [0.0, -13 / 108]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        pseudo["pseudo"][0],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert pseudo["pseudo"][5][0] == pytest.approx([1.0, 11 / 9])

    weights = [1.0, 2.0, 1.5, 0.5, 3.0, 1.0]
    weighted_fit = survival.survfit(response, weights=weights, model=True)
    unweighted_residuals = survival.survfit_residuals(
        weighted_fit,
        times=[2.0, 5.0],
    )
    weighted_residuals = survival.survfit_residuals(
        weighted_fit,
        times=[2.0, 5.0],
        weighted=True,
    )
    for row_idx, weight in enumerate(weights):
        for state_idx in range(3):
            assert weighted_residuals["resid"][row_idx][state_idx] == pytest.approx(
                [weight * value for value in unweighted_residuals["resid"][row_idx][state_idx]]
            )

    grouped_fit = survival.survfit(
        response,
        group=["a", "a", "a", "b", "b", "b"],
        model=True,
    )
    grouped = survival.survfit_residuals(grouped_fit, times=[2.0, 5.0])
    grouped_pseudo = survival.pseudo(grouped_fit, times=[2.0, 5.0])
    assert grouped["curve"] == [1, 1, 1, 2, 2, 2]
    assert grouped["resid"][3][0] == pytest.approx([0.0, -1 / 9])
    assert grouped_pseudo["pseudo"][3][2] == pytest.approx([0.0, 1.0])

    counting_response = survival.Surv(
        [0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        Factor(
            ["ill", "death", "ill", "censor", "death", "censor"],
            ["censor", "ill", "death"],
        ),
    )
    counting_fit = survival.survfit(
        counting_response,
        id=[1, 1, 2, 2, 3, 3],
        model=True,
    )
    row_level = survival.survfit_residuals(counting_fit, times=[2.0, 5.0])
    collapsed = survival.survfit_residuals(
        counting_fit,
        times=[2.0, 5.0],
        collapse=True,
        weighted=True,
    )
    assert collapsed["id"] == [1, 2, 3]
    for subject_idx in range(3):
        for state_idx in range(3):
            expected = [
                row_level["resid"][2 * subject_idx][state_idx][time_idx]
                + row_level["resid"][2 * subject_idx + 1][state_idx][time_idx]
                for time_idx in range(2)
            ]
            assert collapsed["resid"][subject_idx][state_idx] == pytest.approx(expected)


def test_survfit_residuals_match_r_right_censored_fixture():
    response = survival.Surv([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0])
    fit = survival.survfit(response)

    survival_residuals = survival.survfit_residuals(
        fit,
        times=[1.0, 2.0, 3.0],
        type="survival",
    )["resid"]
    for actual, expected in zip(
        survival_residuals,
        [
            [-0.1875, -0.1875, -0.09375],
            [0.0625, 0.0625, 0.03125],
            [0.0625, 0.0625, -0.15625],
            [0.0625, 0.0625, 0.21875],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    cumhaz_residuals = survival.survfit_residuals(
        fit,
        times=[1.0, 2.0, 3.0],
        type="cumhaz",
    )["resid"]
    for actual, expected in zip(
        cumhaz_residuals,
        [
            [0.1875, 0.1875, 0.1875],
            [-0.0625, -0.0625, -0.0625],
            [-0.0625, -0.0625, 0.1875],
            [-0.0625, -0.0625, -0.3125],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    auc_residuals = survival.survfit_residuals(
        fit,
        times=[1.0, 2.0, 3.0],
        type="auc",
    )["resid"]
    for actual, expected in zip(
        auc_residuals,
        [
            [0.0, -0.1875, -0.375],
            [0.0, 0.0625, 0.125],
            [0.0, 0.0625, 0.125],
            [0.0, 0.0625, 0.125],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    grouped = survival.survfit(response, group=["a", "a", "b", "b"])
    grouped_residuals = survival.survfit_residuals(
        grouped,
        times=[1.0, 3.0],
        type="survival",
        extra=True,
    )
    for actual, expected in zip(
        grouped_residuals["resid"],
        [
            [-0.25, -0.25],
            [0.25, 0.25],
            [0.0, -0.25],
            [0.0, 0.25],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert grouped_residuals["curve"] == [1, 1, 2, 2]

    collapsed = survival.survfit(
        response,
        id=["a", "a", "b", "b"],
    )
    collapsed_residuals = survival.survfit_residuals(
        collapsed,
        times=[1.0, 3.0],
        type="survival",
        collapse=True,
        weighted=True,
    )
    assert collapsed_residuals["id"] == ["a", "b"]
    for actual, expected in zip(
        collapsed_residuals["resid"],
        [[-0.125, -0.0625], [0.125, 0.0625]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    with pytest.raises(TypeError, match="times argument"):
        survival.survfit_residuals(fit)
    default_fit_residuals = survival.survfit_residuals(
        survival.survfit(response),
        times=[1.0],
    )
    for actual, expected in zip(
        default_fit_residuals["resid"],
        [[-0.1875], [0.0625], [0.0625], [0.0625]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    counting_response = survival.Surv(
        [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        [0, 1, 1, 0, 0, 1],
    )
    counting_fit = survival.survfit(counting_response, id=[1, 1, 2, 2, 3, 3])
    counting_survival = survival.survfit_residuals(
        counting_fit,
        times=[3.0, 5.0, 7.0],
        type="survival",
    )["resid"]
    for actual, expected in zip(
        counting_survival,
        [
            [0.0, 0.0, 0.0],
            [0.1111111, -0.0740741, 0.0],
            [-0.2222222, -0.1481481, 0.0],
            [0.0, 0.0740741, 0.0],
            [0.1111111, 0.0740741, 0.0],
            [0.0, 0.0740741, 0.0],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    counting_cumhaz = survival.survfit_residuals(
        counting_fit,
        times=[3.0, 5.0, 7.0],
        type="cumhaz",
    )["resid"]
    for actual, expected in zip(
        counting_cumhaz,
        [
            [0.0, 0.0, 0.0],
            [-0.1111111, 0.1111111, 0.1111111],
            [0.2222222, 0.2222222, 0.2222222],
            [0.0, -0.1111111, -0.1111111],
            [-0.1111111, -0.1111111, -0.1111111],
            [0.0, -0.1111111, -0.1111111],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    counting_collapsed = survival.survfit_residuals(
        counting_fit,
        times=[3.0, 5.0, 7.0],
        type="survival",
        collapse=True,
        weighted=True,
    )
    assert counting_collapsed["id"] == [1, 2, 3]
    for actual, expected in zip(
        counting_collapsed["resid"],
        [
            [0.1111111, -0.0740741, 0.0],
            [-0.2222222, -0.0740741, 0.0],
            [0.1111111, 0.1481481, 0.0],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    grouped_counting_response = survival.Surv(
        [0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0],
        [2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0, 8.0],
        [0, 1, 1, 0, 0, 1, 1, 0],
    )
    grouped_counting = survival.survfit(
        grouped_counting_response,
        group=["A", "A", "A", "A", "B", "B", "B", "B"],
        id=[1, 1, 2, 2, 3, 3, 4, 4],
    )
    grouped_counting_extra = survival.survfit_residuals(
        grouped_counting,
        times=[3.0, 5.0],
        type="survival",
        extra=True,
    )
    for actual, expected in zip(
        grouped_counting_extra["resid"],
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [-0.25, -0.125],
            [0.0, 0.125],
            [0.0, 0.0],
            [0.0, 0.25],
            [0.0, -0.25],
            [0.0, 0.0],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert grouped_counting_extra["curve"] == [1, 1, 1, 1, 2, 2, 2, 2]
