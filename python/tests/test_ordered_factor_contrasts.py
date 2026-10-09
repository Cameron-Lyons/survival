"""Ordered-factor matrices use R's polynomial basis and retain fitted coding."""

import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from survival import r

coerce = importlib.import_module("survival.r._coerce")
contrasts = importlib.import_module("survival.r._contrasts")
bridge = importlib.import_module("survival.pybridge")
expression = importlib.import_module("survival.r._expression")
POLYNOMIAL_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "ordered_factor_reference.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", POLYNOMIAL_REFERENCE, ids=lambda case: f"levels-{case['n']}")
def test_polynomial_basis_matches_stock_r_including_rank_loss(case):
    basis, columns = contrasts._contr_poly(case["n"])
    assert list(columns) == case["columns"]
    # High powers are ill conditioned, so roundoff from different platform
    # BLAS implementations is amplified. Rank-limited R Q differs from a full
    # LAPACK Q by order one; these tolerances still distinguish those bases.
    tolerance = 1e-12 if case["n"] <= 10 else 1e-10 if case["n"] <= 16 else 1e-7
    np.testing.assert_allclose(basis, case["basis"], rtol=1e-9, atol=tolerance)


@pytest.mark.parametrize("n", range(25, 96))
def test_polynomial_basis_remains_finite_and_orthonormal_at_high_level_counts(n):
    basis, columns = contrasts._contr_poly(n)
    matrix = np.asarray(basis)
    assert matrix.shape == (n, n - 1)
    assert len(columns) == n - 1
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(n - 1), rtol=0, atol=3e-14)
    np.testing.assert_allclose(np.sum(matrix, axis=0), 0, rtol=0, atol=3e-14)


def independent_basis(n):
    # Closed-form contrasts from equally spaced monic polynomials, normalized
    # independently of the implementation's QR factorization.
    columns = {
        2: [[-1, 1]],
        3: [[-1, 0, 1], [1, -2, 1]],
        4: [[-3, -1, 1, 3], [1, -1, -1, 1], [-1, 3, -3, 1]],
        5: [
            [-2, -1, 0, 1, 2],
            [2, -1, -2, -1, 2],
            [-1, 2, 0, -2, 1],
            [1, -4, 6, -4, 1],
        ],
    }[n]
    matrix = np.asarray(columns, dtype=float).T
    return matrix / np.linalg.norm(matrix, axis=0)


def data_for(levels: list[Any], source: str) -> Any:
    index = np.arange(1, 81)
    values = [levels[i % len(levels)] for i in index]
    factor = (
        coerce._r_factor(values, levels, ordered=True)
        if source == "r_bridge"
        else pd.Categorical(values, categories=levels, ordered=True)
    )
    if source in {"series", "dataframe"}:
        factor = pd.Series(factor)
    data = {
        "time": (10 + index * 37 % 97 + index / 7).tolist(),
        "status": ((index * 17 + 3) % 11 > 2).astype(int).tolist(),
        "x": np.sin(index * 0.63).tolist(),
        "g": factor,
    }
    return pd.DataFrame(data) if source == "dataframe" else data


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("source", ["pandas", "series", "dataframe", "r_bridge"])
@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_default_polynomial_fit_matches_independent_numeric_design(n, source, kind):
    levels = list("zmacb")[:n]
    data = data_for(levels, source)
    basis = independent_basis(n)
    for column in range(n - 1):
        data[f"p{column}"] = [basis[levels.index(value), column] for value in data["g"]]
    fit = getattr(r, kind)("Surv(time, status) ~ x + g", data, model=True)
    numeric = getattr(r, kind)(
        "Surv(time, status) ~ x + " + " + ".join(f"p{column}" for column in range(n - 1)),
        data,
    )
    suffixes = [".L", ".Q", ".C"][: n - 1] + (["^4"] if n == 5 else [])
    expected_columns = (
        (["(Intercept)"] if kind == "survreg" else []) + ["x"] + ["g" + name for name in suffixes]
    )
    matrix = r.model_matrix(fit, _with_metadata=True)
    assert matrix["columns"] == expected_columns
    assert matrix["contrasts"] == {"g": "contr.poly"}
    np.testing.assert_allclose(matrix["data"], r.model_matrix(numeric)["data"], atol=2e-15)
    np.testing.assert_allclose(r.coef(fit), r.coef(numeric), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(r.vcov(fit), r.vcov(numeric), rtol=2e-10, atol=2e-12)
    new: dict[str, Any] = {"x": [0.0, -1.0, 0.3], "g": [levels[-1], levels[0], levels[-1]]}
    for column in range(n - 1):
        new[f"p{column}"] = [basis[levels.index(value), column] for value in new["g"]]
    np.testing.assert_allclose(
        r.predict(fit, new, type="lp"), r.predict(numeric, new, type="lp"), atol=2e-12
    )


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("wrapper", ["g", "I(g)", "identity(g)", "as.factor(g)", "factor(g)"])
def test_polynomial_levels_follow_constructor_before_subset_and_missing_omission(kind, wrapper):
    levels = ["z", "m", "a", "unused"]
    data = data_for(levels[:3], "pandas")
    data["g"] = pd.Categorical(data["g"], categories=levels, ordered=True)
    data["x"][4] = np.nan
    subset = [7, 2, 7, 0, 4, 1, 9, 3, 10, 5, 8, 12, 15, 14, 22, 31, 40]
    fit = getattr(r, kind)(
        f"Surv(time, status) ~ x + {wrapper}",
        data,
        subset=subset,
        na_action="na.exclude",
        model=True,
    )
    fitted_levels = levels[:3] if wrapper == "factor(g)" else levels
    basis = independent_basis(len(fitted_levels))
    kept = [row for row in subset if row != 4]
    expected = np.asarray([basis[fitted_levels.index(data["g"][row])] for row in kept])
    matrix = r.model_matrix(fit, _with_metadata=True)
    assert matrix["contrasts"] == {wrapper: "contr.poly"}
    np.testing.assert_allclose(
        np.asarray(matrix["data"])[:, -expected.shape[1] :], expected, atol=2e-15
    )
    assert tuple(fit.na_action.rows) == (5,)
    # Plain character new data and changed categorical declarations both use
    # the fitted ordering, including declared levels absent from training rows.
    for source in (
        fitted_levels[::-1],
        pd.Categorical(fitted_levels[::-1], categories=fitted_levels[::-1], ordered=False),
    ):
        new = {"x": [0.0] * len(fitted_levels), "g": source}
        actual = r.model_matrix(fit, new, _with_metadata=True)
        assert actual["contrasts"] == {wrapper: "contr.poly"}
        np.testing.assert_allclose(
            np.asarray(actual["data"])[:, -basis.shape[1] :], basis[::-1], atol=2e-15
        )


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("rhs", ["x * g", "x:g", "g - 1"])
def test_ordered_interactions_and_full_dummy_coding_keep_correct_widths(kind, rhs):
    data = data_for(["z", "m", "a"], "pandas")
    fit = getattr(r, kind)("Surv(time, status) ~ " + rhs, data)
    matrix = r.model_matrix(fit, _with_metadata=True)
    basis = independent_basis(3)
    codes = np.asarray([list(data["g"].categories).index(value) for value in data["g"]])
    categorical = basis[codes]
    start = int(kind == "survreg" and rhs != "g - 1")
    if rhs == "g - 1" and kind == "survreg":
        expected = np.eye(3)[codes]
        assert matrix["columns"] == ["gz", "gm", "ga"]
    elif rhs == "x:g":
        # With no factor main effect, R includes all indicator columns.
        expected = np.asarray(data["x"])[:, None] * np.eye(3)[codes]
    elif rhs == "x * g":
        expected = np.column_stack(
            [data["x"], categorical, np.asarray(data["x"])[:, None] * categorical]
        )
    else:
        expected = categorical
    np.testing.assert_allclose(np.asarray(matrix["data"])[:, start:], expected, atol=2e-15)
    assert matrix["contrasts"] == {"g": "contr.poly"}


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_explicit_factor_contrasts_override_ordered_default_and_survive_pickle(kind):
    data = data_for(["z", "m", "a"], "r_bridge")
    supplied: dict[str, Any] = {
        "data": [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        "columns": ["one", "two"],
    }
    data["g"] = coerce._r_factor(data["g"], ["z", "m", "a"], ordered=True, contrast=supplied)
    fit = getattr(r, kind)("Surv(time, status) ~ x + identity(g)", data, model=True)
    matrix = r.model_matrix(fit, _with_metadata=True)
    expected_metadata = {
        "identity(g)": {
            "data": [list(row) for row in supplied["data"]],
            "rows": ["z", "m", "a"],
            "columns": ["one", "two"],
        }
    }
    assert matrix["contrasts"] == expected_metadata
    assert matrix["columns"][-2:] == ["identity(g)one", "identity(g)two"]
    new: dict[str, Any] = {"x": [0.0, 0.5, 1.0], "g": ["a", "z", "m"]}
    prediction = r.predict(fit, new, type="lp")
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - local test-created object
    supplied["data"][0][0] = 999.0
    supplied["columns"][0] = "changed"
    for model in (fit, restored):
        assert r.model_matrix(model, _with_metadata=True)["contrasts"] == expected_metadata
        np.testing.assert_allclose(r.predict(model, new, type="lp"), prediction, atol=1e-15)
    # factor() constructs a fresh factor, discarding explicit contrast attrs.
    fresh = getattr(r, kind)("Surv(time, status) ~ x + factor(g)", data)
    assert r.model_matrix(fresh, _with_metadata=True)["contrasts"] == {"factor(g)": "contr.poly"}


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_evaluated_matrix_uses_fitted_poly_override_and_explicit_metadata(kind):
    data = data_for(["z", "m", "a"], "pandas")
    fit = getattr(r, kind)("Surv(time, status) ~ g", data)
    source = coerce._r_factor(["a", "z", None], ["z", "m", "a"])
    frame = bridge._r_model_frame({"g": source}, 3)
    matrix = r.model_matrix(fit, frame, _with_metadata=True)
    expected = np.vstack([independent_basis(3)[2], independent_basis(3)[0], [np.nan, np.nan]])
    np.testing.assert_allclose(
        np.asarray(matrix["data"])[:, -2:], expected, atol=2e-15, equal_nan=True
    )
    assert matrix["contrasts"] == {"g": "contr.poly"}


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_evaluated_matrix_retains_named_fitted_contrast_when_metadata_is_absent(kind):
    data = data_for(["z", "m", "a"], "r_bridge")
    supplied = {
        "data": [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        "columns": ["1", "2"],
        "label": "contr.sum",
    }
    data["g"] = coerce._r_factor(data["g"], ["z", "m", "a"], ordered=True, contrast=supplied)
    fit = getattr(r, kind)("Surv(time, status) ~ g", data)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - local test-created object
    for model in (fit, restored):
        for levels in (["z", "m", "a"], ["a", "m", "z"]):
            frame = bridge._r_model_frame(
                {"g": coerce._r_factor(["a", "z", "m"], levels, ordered=True)}, 3
            )
            actual = r.model_matrix(model, frame, _with_metadata=True)
            assert actual["columns"][-2:] == ["g1", "g2"]
            assert actual["contrasts"] == {"g": "contr.sum"}
            basis = np.asarray(supplied["data"])
            expected = basis[[levels.index(value) for value in ["a", "z", "m"]]]
            np.testing.assert_array_equal(np.asarray(actual["data"])[:, -2:], expected)
        frame = bridge._r_model_frame(
            {"g": coerce._r_factor(["a", "z"], ["z", "m", "a", "unused"])}, 2
        )
        actual = r.model_matrix(model, frame)
        assert actual["columns"][-3:] == ["g1", "g2", "g3"]
        np.testing.assert_array_equal(
            np.asarray(actual["data"])[:, -3:], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )


def test_polynomial_default_validates_rs_level_limit():
    with pytest.raises(ValueError, match="orthogonal polynomials.*95 degrees of freedom"):
        r.coxph("Surv(time, status) ~ g", data_for(list(range(96)), "pandas"))


def test_time_transform_factor_result_preserves_its_explicit_contrast_basis():
    data = data_for(["z", "m", "a"], "pandas")
    basis = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]

    def codes(x):
        return [0 if value < -0.3 else 2 if value > 0.3 else 1 for value in x]

    def factor_callback(x, t, riskset, weights):
        return coerce._r_factor(
            ["zma"[code] for code in codes(x)],
            ["z", "m", "a"],
            ordered=True,
            contrast={"data": basis, "columns": ["1", "2"], "label": "contr.sum"},
        )

    def numeric_callback(x, t, riskset, weights):
        return [basis[code] for code in codes(x)]

    fit = r.coxph("Surv(time, status) ~ tt(x)", data, tt=factor_callback)
    explicit = r.coxph("Surv(time, status) ~ tt(x)", data, tt=numeric_callback)
    np.testing.assert_allclose(r.coef(fit), r.coef(explicit), atol=1e-14)
    np.testing.assert_allclose(r.vcov(fit), r.vcov(explicit), atol=1e-14)
    matrix = r.model_matrix(fit, _with_metadata=True)
    assert matrix["columns"] == ["tt(x)1", "tt(x)2"]
    assert matrix["contrasts"] == {"tt(x)": "contr.sum"}
    np.testing.assert_array_equal(matrix["data"], r.model_matrix(explicit)["data"])


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("default", ["contr.sum", "contr.treatment"])
def test_factor_constructor_rebuilds_configured_default_after_dropping_unused_levels(kind, default):
    data = data_for(["z", "m", "a"], "r_bridge")
    configured = {
        "data": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
        if default == "contr.sum"
        else [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "columns": ["1", "2", "3"] if default == "contr.sum" else ["m", "a", "unused"],
        "label": default,
        "default": True,
    }
    data["g"] = coerce._r_factor(
        data["g"], ["z", "m", "a", "unused"], ordered=True, contrast=configured
    )
    fit = getattr(r, kind)("Surv(time, status) ~ factor(g)", data)
    matrix = r.model_matrix(fit, _with_metadata=True)
    assert matrix["contrasts"] == {"factor(g)": default}
    names = ["factor(g)1", "factor(g)2"] if default == "contr.sum" else ["factor(g)m", "factor(g)a"]
    assert matrix["columns"][-2:] == names
    basis = np.asarray(
        [[1, 0], [0, 1], [-1, -1]] if default == "contr.sum" else [[0, 0], [1, 0], [0, 1]]
    )
    expected = basis[[["z", "m", "a"].index(value) for value in data["g"]]]
    np.testing.assert_array_equal(np.asarray(matrix["data"])[:, -2:], expected)


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize(
    ("name", "basis", "column_names"),
    [
        ("contr.sum", [[1, 0], [0, 1], [-1, -1]], ["1", "2"]),
        ("contr.helmert", [[-1, -1], [1, -1], [0, 2]], ["1", "2"]),
        ("contr.SAS", [[1, 0], [0, 1], [0, 0]], ["z", "m"]),
    ],
)
def test_named_fitted_contrasts_override_conflicting_prepared_frame_basis(
    kind, name, basis, column_names
):
    data = data_for(["z", "m", "a"], "r_bridge")
    data["g"] = coerce._r_factor(
        data["g"],
        ["z", "m", "a"],
        ordered=True,
        contrast={"data": basis, "columns": column_names, "label": name},
    )
    fit = getattr(r, kind)("Surv(time, status) ~ g", data)
    source = coerce._r_factor(
        ["a", "z", "m"],
        ["z", "m", "a"],
        ordered=True,
        contrast={"data": independent_basis(3), "columns": [".L", ".Q"], "label": "contr.poly"},
    )
    frame = bridge._r_model_frame({"g": source}, 3)
    actual = r.model_matrix(fit, frame, _with_metadata=True)
    assert actual["columns"][-2:] == ["g" + column for column in column_names]
    assert actual["contrasts"] == {"g": name}
    np.testing.assert_array_equal(np.asarray(actual["data"])[:, -2:], np.asarray(basis)[[2, 0, 1]])


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_factor_constructor_recovers_configured_default_alongside_explicit_basis(kind):
    data = data_for(["z", "m", "a"], "r_bridge")
    explicit = {
        "data": [[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]],
        "columns": ["1", "2", "3"],
        "label": "contr.helmert",
        "default_name": "contr.sum",
    }
    data["g"] = coerce._r_factor(
        data["g"], ["z", "m", "a", "unused"], ordered=True, contrast=explicit
    )
    constructed = getattr(r, kind)("Surv(time, status) ~ factor(g)", data)
    matrix = r.model_matrix(constructed, _with_metadata=True)
    assert matrix["columns"][-2:] == ["factor(g)1", "factor(g)2"]
    assert matrix["contrasts"] == {"factor(g)": "contr.sum"}
    basis = np.asarray([[1, 0], [0, 1], [-1, -1]])
    expected = basis[[["z", "m", "a"].index(value) for value in data["g"]]]
    np.testing.assert_array_equal(np.asarray(matrix["data"])[:, -2:], expected)
