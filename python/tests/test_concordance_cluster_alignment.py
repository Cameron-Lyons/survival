"""Cluster ordering and joint concordance covariance on retained model rows."""

import math

import numpy as np
import pytest
from survival import r
from survival.r._coerce import _r_factor


def _data():
    rows = range(1, 25)
    return {
        "time": [2 + (i * 13) % 19 for i in rows],
        "status": [int(i % 4 != 0) for i in rows],
        "x": [math.sin(i) for i in rows],
        "z": [math.cos(i / 3) for i in rows],
        "id": [i % 3 for i in range(24)],
    }


@pytest.mark.parametrize("source", ["formula", "column", "vector"])
def test_fitted_categorical_clusters_keep_declared_order(source):
    data = _data()
    data["id"] = _r_factor(data["id"], [2, 0, 1, 7])
    formula = "Surv(time,status)~x"
    kwargs = {}
    if source == "formula":
        formula += "+cluster(id)"
    else:
        kwargs["cluster"] = "id" if source == "column" else data["id"]
    fit = r.coxph(formula, data, **kwargs)
    actual = r.concordance(fit, influence=1)
    expected = r.concordance(fit, cluster=data["id"], influence=1)
    assert actual.dfbeta == pytest.approx(expected.dfbeta)
    assert actual.var == pytest.approx(expected.var)


def test_joint_covariance_is_invariant_to_cluster_names():
    data = _data()
    first = r.coxph("Surv(time,status)~x+cluster(id)", data)
    second = r.coxph("Surv(time,status)~z+cluster(id)", data)
    renamed = r.coxph(
        "Surv(time,status)~z+cluster(id)",
        {**data, "id": [(value + 1) % 3 for value in data["id"]]},
    )
    expected = r.concordance(first, second, influence=1)
    actual = r.concordance(first, renamed, influence=1)
    np.testing.assert_allclose(actual.dfbeta, expected.dfbeta, atol=1e-14)
    np.testing.assert_allclose(actual.var, expected.var, atol=1e-14)


def test_joint_covariance_rejects_different_cluster_partitions():
    data = _data()
    first = r.coxph("Surv(time,status)~x+cluster(id)", data)
    second = r.coxph("Surv(time,status)~z+cluster(id)", {**data, "id": [i // 8 for i in range(24)]})
    with pytest.raises(ValueError, match="identical clustering"):
        r.concordance(first, second)


@pytest.mark.parametrize("layout", ["list", "numpy", "pandas"])
@pytest.mark.parametrize("source", ["formula", "column", "vector"])
def test_categorical_clusters_survive_subset_missingness_and_serialization(layout, source):
    import pickle

    data = _data()
    labels = [str(value) for value in data["id"]]
    levels = ["unused", "2", "0", "1", "last"]
    data["id"] = _r_factor(labels, levels)
    data["x"][3] = None
    data["id"] = _r_factor([*labels[:7], None, *labels[8:]], levels)
    if layout == "numpy":
        data = {key: value if key == "id" else np.asarray(value) for key, value in data.items()}
    elif layout == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({**data, "id": pd.Categorical(data["id"], categories=levels)})
    formula = "Surv(time,status)~x" + ("+cluster(id)" if source == "formula" else "")
    kwargs = {} if source == "formula" else {"cluster": "id" if source == "column" else data["id"]}
    selected = [20, 18, 7, 3, 1, 5, 13, 12, 11, 8]
    retained = [row for row in selected if row not in (3, 7)]
    fit = r.coxph(formula, data, subset=selected, na_action="na.exclude", model=True, **kwargs)
    assert fit.cluster_levels == tuple(levels)
    for model in (fit, pickle.loads(pickle.dumps(fit))):  # noqa: S301 - local round trip
        actual = r.concordance(model, influence=1)
        expected = r.concordance(
            model, cluster=_r_factor([labels[row] for row in retained], levels), influence=1
        )
        assert actual.dfbeta == pytest.approx(expected.dfbeta)
        assert len(actual.dfbeta) == 3
        assert actual.n == len(retained)
        assert model.cluster_levels == tuple(levels)
        assert model.model["(cluster)"] == [labels[row] for row in retained]


@pytest.mark.parametrize("levels", [[2, 0, 1], [0, 2, 1], [9, 1, 2, 0, 8]])
@pytest.mark.parametrize("influence", [0, 1, 2, 3])
def test_joint_factor_order_does_not_change_covariance(levels, influence):
    data = _data()
    first = r.coxph("Surv(time,status)~x+cluster(id)", data)
    reordered = r.coxph(
        "Surv(time,status)~z+cluster(id)", {**data, "id": _r_factor(data["id"], levels)}
    )
    expected = r.concordance(first, reordered, cluster=data["id"], influence=influence, ranks=True)
    actual = r.concordance(first, reordered, influence=influence, ranks=True)
    np.testing.assert_allclose(actual.var, expected.var, atol=1e-14)
    assert actual.dfbeta == expected.dfbeta
    assert actual.influence == expected.influence
    assert actual.ranks == expected.ranks
    # The first model determines the reported group order.
    reverse = r.concordance(reordered, first, influence=influence)
    np.testing.assert_allclose(reverse.var, np.asarray(expected.var)[::-1, ::-1], atol=1e-14)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("counting", [False, True])
def test_joint_covariance_matches_independent_observation_influences(weighted, counting):
    data = _data()
    data["start"] = [value / 100 for value in range(24)]
    data["w"] = [0.5 + i % 4 for i in range(24)]
    response = "Surv(start,time,status)" if counting else "Surv(time,status)"
    kwargs = {"weights": "w"} if weighted else {}
    first = r.coxph(response + "~x+cluster(id)", data, **kwargs)
    renamed = {**data, "id": _r_factor([str((v + 1) % 3) for v in data["id"]], ["2", "1", "0"])}
    second = r.coxph(response + "~z+cluster(id)", renamed, **kwargs)
    observation = np.asarray(
        [r.concordance(fit, cluster=list(range(24)), influence=1).dfbeta for fit in (first, second)]
    ).T
    expected = np.asarray(
        [observation[np.asarray(data["id"]) == level].sum(axis=0) for level in range(3)]
    )
    actual = r.concordance(first, second, influence=1)
    np.testing.assert_allclose(actual.dfbeta, expected, atol=1e-14)
    np.testing.assert_allclose(actual.var, expected.T @ expected, atol=1e-14)


@pytest.mark.parametrize("first_clustered", [False, True])
def test_singleton_clusters_align_with_unclustered_observations(first_clustered):
    data = _data()
    clustered = r.coxph("Surv(time,status)~x", data, cluster=list(reversed(range(24))))
    plain = r.coxph("Surv(time,status)~z", data)
    fits = (clustered, plain) if first_clustered else (plain, clustered)
    cluster = list(reversed(range(24))) if first_clustered else list(range(24))
    expected = r.concordance(*fits, cluster=cluster, influence=1)
    actual = r.concordance(*fits, influence=1)
    np.testing.assert_allclose(actual.var, expected.var, atol=1e-14)
    np.testing.assert_allclose(actual.dfbeta, expected.dfbeta, atol=1e-14)


@pytest.mark.parametrize("source", ["direct", "formula", "fitted"])
@pytest.mark.parametrize("factor", [False, True])
def test_character_labels_and_unused_factor_levels_use_r_order(source, factor):
    data = _data()
    labels = [str([1, 10, 2][value]) for value in data["id"]]
    levels = ["unused", "2", "10", "1", "last"] if factor else ["1", "10", "2"]
    cluster = _r_factor(labels, levels) if factor else labels
    observed = [level for level in levels if level in labels]
    codes = [observed.index(label) for label in labels]
    if source == "direct":
        y = r.Surv(data["time"], data["status"])
        actual = r.concordancefit(y, data["x"], cluster=cluster, influence=1)
        expected = r.concordancefit(y, data["x"], cluster=codes, influence=1)
    elif source == "formula":
        actual = r.concordance("Surv(time,status)~x", data, cluster=cluster, influence=1)
        expected = r.concordance("Surv(time,status)~x", data, cluster=codes, influence=1)
    else:
        fit = r.coxph("Surv(time,status)~x", data, cluster=cluster)
        actual = r.concordance(fit, influence=1)
        expected = r.concordance(fit, cluster=codes, influence=1)
    assert actual.dfbeta == pytest.approx(expected.dfbeta)
    assert len(actual.dfbeta) == len(observed)


@pytest.mark.parametrize("cluster", [[None] * 24, [1] * 23, _r_factor([1] * 24, [2])])
def test_joint_validation_does_not_hide_invalid_cluster_values(cluster):
    data = _data()
    first = r.coxph("Surv(time,status)~x", data)
    second = r.coxph("Surv(time,status)~z", data)
    with pytest.raises((ValueError, TypeError), match="cluster|categor"):
        r.concordance(first, second, cluster=cluster)


def test_fitted_cluster_metadata_is_independent_of_caller_mutations():
    from .r_fixture_support import RFactor

    data = _data()
    cluster = RFactor(data["id"], [2, 0, 1, 7])
    cluster.categories = list(cluster.categories)
    fit = r.coxph("Surv(time,status)~x", data, cluster=cluster)
    before = r.concordance(fit, influence=1)
    cluster[0] = 7
    cluster.categories.reverse()
    after = r.concordance(fit, influence=1)
    assert fit.cluster_levels == (2, 0, 1, 7)
    assert after.dfbeta == before.dfbeta
    assert after.var == before.var


def test_multistate_cox_retains_categorical_cluster_metadata():
    data = _data()
    data["event"] = _r_factor(["censor", "first", "second"] * 8, ["censor", "first", "second"])
    data["subject"] = list(range(24))
    data["id"] = _r_factor(data["id"], [2, 0, 1, 7])
    fit = r.coxph("Surv(time,event)~x+cluster(id)", data, id="subject")
    assert fit.cluster_levels == (2, 0, 1, 7)
    assert list(fit.cluster) == list(data["id"])
    with pytest.raises(ValueError, match="multi-state"):
        r.concordance(fit)
