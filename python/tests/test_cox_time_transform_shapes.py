"""Cox time-transform types and expanded fitted data against independent R fits."""

import importlib
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_time_transform_reference.json").read_text()
)


def transform_vector(x, t, riskset, weights):
    return np.asarray(x) * np.log(t)


def transform_root(x, t, riskset, weights):
    return np.asarray(x) * np.sqrt(t)


def transform_named(x, t, riskset, weights):
    return {
        "log": transform_vector(x, t, riskset, weights),
        "root": transform_root(x, t, riskset, weights),
    }


def transform_unnamed(x, t, riskset, weights):
    return np.column_stack(list(transform_named(x, t, riskset, weights).values()))


def transform_one(x, t, riskset, weights):
    return {"log": transform_vector(x, t, riskset, weights)}


def transform_boolean(x, t, riskset, weights):
    return np.asarray(x) > np.median(x)


def transform_factor(x, t, riskset, weights):
    return RFactor(
        ["TRUE" if v else "FALSE" for v in transform_boolean(x, t, riskset, weights)],
        ["TRUE", "FALSE", "unused"],
    )


def transform_ordered(x, t, riskset, weights):
    pd = pytest.importorskip("pandas")
    return pd.Categorical(
        ["low" if v < -0.5 else "high" if v > 0.5 else "middle" for v in x],
        categories=["low", "middle", "high"],
        ordered=True,
    )


def transform_factor_input(x, t, riskset, weights):
    codes = [x.categories.index(v) + 1 for v in x]
    return transform_named(codes, t, riskset, weights)


def transform_constant_input(x, t, riskset, weights):
    return transform_vector([x.categories.index(v) + 1 for v in x], t, riskset, weights)


TRANSFORMS = {
    "vector": transform_vector,
    "named": transform_named,
    "unnamed": transform_unnamed,
    "one": transform_one,
    "boolean": transform_boolean,
    "factor": transform_factor,
    "ordered": transform_ordered,
    "factor_input": transform_factor_input,
    "constant_input": transform_constant_input,
    "root": transform_root,
}


def close(actual, expected):
    if isinstance(actual, dict) and "data" in actual:
        actual = actual["data"]

    def numeric(value):
        if isinstance(value, list):
            return [numeric(v) for v in value]
        return math.nan if value is None else value

    np.testing.assert_allclose(actual, numeric(expected), rtol=3e-7, atol=2e-8, equal_nan=True)


def data_for(case):
    data = dict(REFERENCE["data"])
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    if case.get("missing"):
        data["x"] = list(data["x"])
        for i in (1, 8, 24):
            data["x"][i] = None
    if case.get("counting"):
        data["time"] = [v + (i + 1) / 1000 for i, v in enumerate(data["time"])]
    return data


def fit_case(case, **kwargs):
    names = case["transform"]
    return r.coxph(
        case["formula"],
        data_for(case),
        ties=case["method"],
        tt=None if names is None else [TRANSFORMS[name] for name in names],
        weights="w" if case.get("weighted") else None,
        cluster="id" if case.get("cluster") else None,
        id="id" if case.get("id") else None,
        robust=bool(case.get("cluster") or case.get("id")),
        subset=[i for i in range(60) if (i + 1) % 4] if case.get("subset") else None,
        na_action="na.exclude" if case.get("missing") else "na.omit",
        eps=1e-10,
        iter_max=50,
        **kwargs,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_legacy_expanded_matrix_labels_restore_without_transform_callbacks(case, monkeypatch):
    fit = fit_case(case)
    expected = r.model_matrix(fit, _with_metadata=True)
    legacy = pickle.loads(pickle.dumps(replace(fit, _matrix_rows=None)))  # noqa: S301

    def forbid_callbacks(*args, **kwargs):
        pytest.fail("model_matrix must not rerun time-transform callbacks")

    module = importlib.import_module("survival.r._coxph")
    monkeypatch.setattr(module, "_tt_functions", forbid_callbacks)
    assert r.model_matrix(legacy, _with_metadata=True) == expected


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_transformed_fit_and_expanded_methods_match_r(case):
    fit = fit_case(case)
    assert fit.tt
    assert list(fit.coef_names) == case["names"]
    assert fit.assign == {name: tuple(cols) for name, cols in case["assign"].items()}
    assert (fit.n, fit.nevent) == (case["n"], case["nevent"])
    close(fit.coefficients, case["coefficients"])
    close(fit.var, case["variance"])
    close(fit.loglik, case["loglik"])
    close(fit.means, case["means"])
    close(r.model_matrix(fit), case["x"])
    close(np.column_stack((fit.y.time, fit.y.event)), case["y"])
    close(r.residuals(fit), case["martingale"])
    if case.get("cluster") or case.get("id"):
        close(fit.naive_var, case["naive"])
    detail = r.coxph_detail(fit)
    if case["detail"] is not None:
        for name in ("time", "nrisk", "nevent", "hazard", "means", "score"):
            close(getattr(detail, name), case["detail"][name])
    assert len(detail.x) == len(detail.y) == len(fit.x)


@pytest.mark.parametrize(
    "name", ["named/efron", "factor/efron", "robust/efron", "counting_id/efron"]
)
def test_transformed_models_retain_shapes_after_pickle_and_init(name):
    case = next(c for c in REFERENCE["cases"] if c["name"] == name)
    fit = fit_case(case)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - our own fitted model
    assert restored.coef_names == fit.coef_names
    close(r.model_matrix(restored), fit.x)
    close(r.coxph_detail(restored).score, r.coxph_detail(fit).score)
    close(
        fit_case(case, init=np.nan_to_num(fit.coefficients).tolist()).coefficients, fit.coefficients
    )
    with pytest.raises(ValueError, match="wrong length for init"):
        fit_case(case, init=[0.0] * (fit.nvar + 1))


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "list", "dataframe"])
def test_matrix_return_layouts_keep_columns(layout):
    if layout == "dataframe":
        pd = pytest.importorskip("pandas")
    case = next(c for c in REFERENCE["cases"] if c["name"] == "unnamed/efron")

    def callback(*args):
        matrix = transform_unnamed(*args)
        if layout == "fortran":
            return np.asfortranarray(matrix)
        if layout == "strided":
            storage = np.empty((len(matrix), 4))
            storage[:, ::2] = matrix
            return storage[:, ::2]
        if layout == "list":
            return matrix.tolist()
        if layout == "dataframe":
            return pd.DataFrame(matrix, columns=["1", "2"])
        return matrix

    fit = r.coxph(
        case["formula"], data_for(case), tt=callback, robust=False, eps=1e-10, iter_max=50
    )
    assert list(fit.coef_names) == case["names"]
    close(fit.coefficients, case["coefficients"])
    close(fit.x, case["x"])


def test_callback_contract_uses_formula_order_once_and_one_based_risksets():
    data = dict(REFERENCE["data"])
    calls = []

    def first(x, time, riskset, weights):
        calls.append("x")
        assert min(riskset) == 1
        assert sorted(set(riskset)) == list(range(1, max(riskset) + 1))
        assert weights is None
        return transform_named(x, time, riskset, weights)

    def second(x, time, riskset, weights):
        calls.append("z")
        return transform_root(x, time, riskset, weights)

    fit = r.coxph("Surv(time,status) ~ tt(x):z + tt(z)", data, tt=[first, second], robust=False)
    assert calls == ["x", "z"]
    case = next(c for c in REFERENCE["cases"] if c["name"] == "mixed_order/efron")
    close(fit.coefficients, case["coefficients"])


@pytest.mark.parametrize(
    ("callback", "message"),
    [
        (lambda x, t, r, w: [0.0], "one value per expanded row"),
        (lambda x, t, r, w: [[0, 1]] + [[0]] * (len(x) - 1), "rectangular"),
        (lambda x, t, r, w: [[] for _ in x], "at least one column"),
        (lambda x, t, r, w: [math.inf for _ in x], "infinite predictor"),
        (lambda x, t, r, w: [math.nan for _ in x], "infinite predictor"),
        (lambda x, t, r, w: ["only" for _ in x], "2 or more levels"),
    ],
)
def test_invalid_transform_outputs_fail_before_fitting(callback, message):
    with pytest.raises(ValueError, match=message):
        r.coxph("Surv(time,status) ~ tt(x)", REFERENCE["data"], tt=callback)
