"""Standalone smoothing searches share the fitters' numerical implementation."""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

regression = setup_survival_import().regression
Controller = regression.PenaltyController
State = regression.PenaltyControlState
CASES = json.loads((Path(__file__).parent / "fixtures/penalty_control_reference.json").read_text())[
    "cases"
]


def compare(actual, expected):
    assert actual.theta == pytest.approx(expected["theta"], abs=1e-9, rel=1e-9)
    assert actual.done == expected["done"]
    np.testing.assert_allclose(actual.history, expected["history"], atol=1e-9, rtol=1e-9)
    if "c_loglik" in expected:
        assert actual.c_loglik == pytest.approx(expected["c_loglik"], abs=1e-9)
        assert actual.half == expected["half"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("arrays", [False, True])
def test_controller_trajectories_match_r(case, arrays):
    options = case["options"]
    if arrays:
        options = {
            key: np.asarray(value) if isinstance(value, list) else value
            for key, value in options.items()
        }
    controller = Controller(case["method"], **options)
    old = controller.initial()
    compare(old, case["initial"])
    for iteration, expected in enumerate(case["path"], 1):
        inputs = expected["input"]
        if arrays:
            inputs = {
                key: np.asarray(value) if isinstance(value, list) else value
                for key, value in inputs.items()
            }
        next_state = controller.step(old, iteration, **inputs)
        compare(next_state, expected)
        compare(pickle.loads(pickle.dumps(next_state)), expected)  # noqa: S301
        old = next_state


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"method": "invalid"}, "method must"),
        ({"method": "fixed"}, "theta is required"),
        ({"method": "gamma", "eps": 0}, "eps"),
        ({"method": "gamma", "eps": float("nan")}, "eps"),
        ({"method": "gamma", "init": []}, "init"),
        ({"method": "gaussian", "init": [1]}, "init"),
        ({"method": "df", "target_df": 1, "thetas": [], "dfs": [], "guess": 1}, "nonzero"),
        ({"method": "df", "target_df": 1, "thetas": [0], "dfs": [0, 1], "guess": 1}, "length"),
        ({"method": "df", "target_df": 1, "thetas": [0], "dfs": [0], "guess": -1}, "guess"),
        ({"method": "aic", "lower": 2, "upper": 1}, "bounds"),
        ({"method": "aic", "theta": 1}, "theta is only"),
        ({"method": "gamma", "target_df": 2}, "only used by df"),
        ({"method": "gamma", "upper": 2}, "only used by aic"),
        ({"method": "fixed", "theta": 1, "gamma_correction": True}, "gamma_correction"),
        ({"method": "fixed", "theta": 1, "init": [1, 2]}, "init is only"),
    ],
)
def test_invalid_configuration_is_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        Controller(**options)


@pytest.mark.parametrize(
    ("state", "iteration", "inputs", "match"),
    [
        (lambda: State(0), 0, {}, "iter must"),
        (lambda: State(0), 2, {}, "row count"),
        (lambda: State(0, history=[[0]]), 2, {}, "column count"),
        (lambda: State(1), 1, {"events_by_group": [1.5]}, "integer counts"),
        (lambda: State(1), 1, {"events_by_group": [float("inf")]}, "integer counts"),
        (lambda: State(1), 1, {"events_by_group": [-1]}, "integer counts"),
        (lambda: State(1), 1, {"events_by_group": [10000001]}, "integer counts"),
        (lambda: State(1), 1, {"loglik": float("nan")}, "loglik"),
    ],
)
def test_malformed_steps_are_rejected(state, iteration, inputs, match):
    with pytest.raises(ValueError, match=match):
        Controller("gamma").step(state(), iteration, **inputs)


def test_state_and_controller_own_their_arrays_and_are_reusable():
    init = np.array([0.1, 1.0])
    controller = Controller("gamma", init=init)
    init[:] = 9
    assert controller.initial().theta == 0.1
    history = np.array([[0.1, -100, -101]])
    old = State(1.0, history=history)
    history[:] = 99
    expected = controller.step(old, 2, loglik=-100, events_by_group=[1, 2, 3])
    copied = pickle.loads(pickle.dumps(controller))  # noqa: S301
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(
                lambda _: copied.step(old, 2, loglik=-100, events_by_group=[1, 2, 3]), range(16)
            )
        )
    assert all(
        value.history == expected.history and value.theta == expected.theta for value in outputs
    )
    assert old.history == [[0.1, -100, -101]]
    old.history[0][0] = 17
    assert old.history == [[0.1, -100, -101]]
    with pytest.raises(AttributeError):
        old.theta = 2


@pytest.mark.parametrize("method", ["df", "aic"])
def test_gamma_correction_can_be_added_to_other_searches(method):
    options = {"target_df": 1.7, "thetas": [0], "dfs": [4], "guess": 0.7} if method == "df" else {}
    controller = Controller(method, gamma_correction=True, **options)
    old = controller.initial()
    actual = controller.step(
        old, 1, df=1.5, neff=10, plik=-90, loglik=-100, events_by_group=[1, 2, 3]
    )
    gamma = Controller("gamma", theta=old.theta)
    expected = gamma.step(gamma.initial(), 1, loglik=-100, events_by_group=[1, 2, 3])
    assert actual.c_loglik == expected.c_loglik
