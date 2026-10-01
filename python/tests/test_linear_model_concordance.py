"""External lm/GLM concordance uses the common native scorer."""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/linear_concordance_reference.json").read_text()
)


def model(case, spec, stored=True):
    return r.YatesModel(
        spec["formula"],
        case["data"],
        [np.nan if x is None else x for x in spec["coefficients"]],
        np.eye(len(spec["coefficients"])).tolist(),
        weights=spec["weights"],
        linear_predictors=spec["linear_predictors"] if stored else None,
        family={"linkinv": forbidden} if spec["glm"] else None,
    )


def forbidden(eta):
    pytest.fail("concordance must use linear predictors without inverse links")


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=2e-10,
        atol=2e-12,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("stored", [False, True])
def test_external_model_matches_r(case, stored):
    fits = [model(case, spec, stored) for spec in case["models"]]
    actual = r.concordance(*fits, newdata=case["newdata"], **case["options"])
    expected = case["expected"] if stored else case["reconstructed"]
    for name in ("concordance", "n", "var", "cvar", "dfbeta", "influence"):
        value = getattr(actual, name)
        if expected[name] is None:
            assert value is None
        else:
            close(value, np.squeeze(expected[name]) if np.ndim(value) == 0 else expected[name])
    count = actual.count
    if isinstance(count, dict):
        close(list(count.values()), expected["count"])
    else:
        close([list(row.values()) for row in count], expected["count"])
    assert set(actual.ranks) == set(expected["ranks"])
    for name in actual.ranks:
        if name == "fit":
            assert actual.ranks[name] == expected["ranks"][name]
        else:
            close(actual.ranks[name], expected["ranks"][name])


@pytest.mark.parametrize("container", ["lists", "numpy", "dataframe"])
def test_newdata_omits_response_and_covariates_together(container):
    fit = r.YatesModel(
        "y ~ x + offset(o)",
        {"y": [1, 4, 2, 3], "x": [2, 3, 1, 4], "o": [0, 0.5, -0.5, 1]},
        [1.0, 0.3],
        [[1.0, 0.0], [0.0, 1.0]],
        weights=[1, 2, 3, 4],
        linear_predictors=[30, 20, 10, 0],
    )
    new = {
        "y": [3.0, np.nan, 1.0, 6.0, 2.0, 4.0],
        "x": [3.0, 2.0, np.nan, 1.0, 4.0, 2.0],
        "o": [0.1, 0.0, 0.3, -0.4, np.nan, 0.5],
    }
    if container == "numpy":
        new = {key: np.asarray(value) for key, value in new.items()}
    elif container == "dataframe":
        import pandas as pd

        new = pd.DataFrame(new, index=["b", "b", "a", "x", "z", "k"])
    result = r.concordance(fit, newdata=new, influence=3, cluster=[1, 2, 1])
    expected = r.concordancefit([3.0, 6.0, 4.0], [2.0, 0.9, 2.1], influence=3, cluster=[1, 2, 1])
    for field in ("concordance", "var", "dfbeta", "influence"):
        close(getattr(result, field), getattr(expected, field))
    assert result.count == expected.count
    assert result.n == 3
    with pytest.raises(ValueError, match="cluster"):
        r.concordance(fit, newdata=new, cluster=[1] * 6)


def test_joint_covariance_uses_weighted_cluster_influence_once():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "joint_weights_training")
    fits = [model(case, spec) for spec in case["models"]]
    cluster = [i % 7 for i in range(len(case["data"]["y"]))]
    results = [r.concordance(fit, influence=1, cluster=cluster) for fit in fits]
    expected = np.column_stack([value.dfbeta for value in results])
    combined = r.concordance(*fits, influence=1, cluster=cluster)
    close(combined.dfbeta, expected)
    close(combined.var, expected.T @ expected)
    close(np.diag(combined.var), [value.var for value in results])
    assert not np.allclose(case["raw"]["var"], case["expected"]["var"])


def test_model_comparisons_validate_type_weights_rows_and_response():
    case = REFERENCE["cases"][0]
    fit = model(case, case["models"][0])
    other = replace(fit, weights=[1.0] * len(fit.linear_predictors))
    with pytest.raises(ValueError, match="same weight"):
        r.concordance(fit, other)
    with pytest.raises(TypeError, match="appropriate fit"):
        r.concordance(fit, object())
    aft = r.survreg("Surv(y) ~ x", case["data"], dist="gaussian")
    for models in ((fit, aft), (aft, fit)):
        with pytest.raises(TypeError, match="appropriate fit"):
            r.concordance(*models)
    different = replace(fit, data={**case["data"], "y": np.asarray(case["data"]["y"]) + 1})
    with pytest.warns(RuntimeWarning, match="same response"):
        r.concordance(fit, different)
    small = replace(
        fit,
        data={name: values[:-1] for name, values in case["data"].items()},
        linear_predictors=fit.linear_predictors[:-1],
    )
    with pytest.raises(ValueError, match="same sample size"):
        r.concordance(fit, small)


@pytest.mark.parametrize("response", [None, [True, False, True, False], [[1, 0]] * 4])
def test_external_concordance_requires_numeric_vector_response(response):
    data = {"x": [1, 2, 3, 4]}
    if response is not None:
        data["y"] = response
    fit = r.YatesModel("~ x" if response is None else "y ~ x", data, [1.0, 0.3], np.eye(2))
    with pytest.raises(ValueError, match="numeric vector"):
        r.concordance(fit)


@pytest.mark.parametrize("values", [[], [1.0], [1.0] * 31, [np.nan] * 30, [np.inf] * 30])
def test_stored_predictors_validate_shape_and_finiteness(values):
    case = REFERENCE["cases"][0]
    with pytest.raises(ValueError, match="linear_predictors"):
        replace(model(case, case["models"][0]), linear_predictors=values)


def test_stored_predictors_own_input_and_roundtrip_concurrently():
    case = REFERENCE["cases"][0]
    source = np.asarray(case["models"][0]["linear_predictors"])
    fit = replace(model(case, case["models"][0]), linear_predictors=source)
    expected = r.concordance(fit, influence=3)
    source[:] = np.nan
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own round-trip data
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda _: r.concordance(restored, influence=3), range(3)))
    for actual in results:
        assert actual == expected


def test_stored_training_ties_are_distinct_from_reconstructed_predictions():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "intercept_only_training")
    spec = case["models"][0]
    reconstructed = r.concordance(model(case, spec, stored=False))
    assert reconstructed.concordance == 0.5
    assert reconstructed.count["tied.x"] == 30 * 29 / 2
    assert r.concordance(model(case, spec)).count["tied.x"] < reconstructed.count["tied.x"]


@pytest.mark.parametrize("timefix", [True, False])
def test_numeric_responses_retain_near_ties_in_training(timefix):
    data = {"y": [1.0, 1.0 + 1e-12, 2.0, 3.0], "x": [2.0, 1.0, 3.0, 4.0]}
    fit = r.YatesModel("y ~ x", data, [0.0, 1.0], np.eye(2))
    raw = r.concordancefit(data["y"], data["x"], timefix=timefix)
    training = r.concordance(fit, timefix=timefix)
    assert (
        raw.count
        == training.count
        == {
            "concordant": 5.0,
            "discordant": 1.0,
            "tied.x": 0.0,
            "tied.y": 0.0,
            "tied.xy": 0.0,
        }
    )
    predicted = r.concordance(fit, newdata=data, timefix=timefix)
    assert predicted.count["tied.y"] == int(timefix)
    assert predicted.count["discordant"] == int(not timefix)


@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("numeric", [False, True])
def test_direct_factor_response_uses_declared_order(series, numeric):
    import pandas as pd

    levels = [30, 10, 20] if numeric else ["c", "a", "b"]
    response = pd.Categorical([levels[i] for i in [0, 0, 1, 2]], categories=levels, ordered=True)
    if series:
        response = pd.Series(response)
    actual = r.concordancefit(response, [2, 1, 3, 4], influence=3)
    expected = r.concordancefit([1, 1, 2, 3], [2, 1, 3, 4], influence=3)
    assert actual == expected
    with pytest.raises(ValueError, match="orderable factor"):
        r.concordancefit(pd.Categorical(response, categories=levels, ordered=False), [2, 1, 3, 4])


def test_direct_two_level_factor_is_orderable_and_logical_is_not():
    import pandas as pd

    actual = r.concordancefit(
        pd.Categorical(["b", "a", "b", "a"], categories=["b", "a"]), [2, 1, 3, 4]
    )
    expected = r.concordancefit([1, 2, 1, 2], [2, 1, 3, 4])
    assert actual == expected
    for values in ([True, False, True, False], ["1", "2", "3", "4"]):
        with pytest.raises(ValueError, match="numeric"):
            r.concordancefit(values, [2, 1, 3, 4])
