"""The R bridge must preserve nullable source rows recovered by logical expressions."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("action", ["na.omit", "na.exclude"])
def test_typed_frame_transport_retains_recovered_sources_and_fitted_labels(kind, action):
    from survival.pybridge import _r_data_frame, _r_logical

    data = _r_data_frame(
        {
            "time": [1, 4, 2, 7, 3, 9],
            "status": [1, 0, 1, 1, 0, 1],
            "x": [-1, 0.5, 1, -0.5, 0, 2],
            "a": _r_logical([False, True, None, False, None, True]),
            "b": _r_logical([None, False, True, True, None, False]),
        },
        6,
        ["row A", "row B", "row C", "row D", "row E", "row F"],
    )
    options = {"control": r.coxph_control(iter_max=0)} if kind == "coxph" else {}
    fit = getattr(r, kind)(
        "Surv(time, status) ~ x + offset(a & b)",
        data,
        na_action=action,
        model=True,
        **options,
    )
    result = r.model_frame(fit, _with_metadata=True)
    assert result["row_names"] == ("row A", "row B", "row D", "row F")
    assert result["columns"]["b"] == [None, False, True, False]
    assert result["metadata"]["a"]["kind"] == "logical"
    assert result["metadata"]["b"]["kind"] == "logical"
    assert r.model_frame(fit) == result["columns"]


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_entirely_missing_logical_source_retains_type_without_losing_rows(kind):
    from survival.pybridge import _r_logical

    fit = getattr(r, kind)(
        "Surv(time, status) ~ x + offset(TRUE | a)",
        {
            "time": [1, 4, 2, 7, 3, 9],
            "status": [1, 0, 1, 1, 0, 1],
            "x": [-1, 0.5, 1, -0.5, 0, 2],
            "a": _r_logical([None] * 6),
        },
        model=True,
    )
    result = r.model_frame(fit, _with_metadata=True)
    assert result["columns"]["a"] == [None] * 6
    assert result["metadata"]["a"]["kind"] == "logical"
    assert len(r.model_matrix(fit)["data"]) == 6
