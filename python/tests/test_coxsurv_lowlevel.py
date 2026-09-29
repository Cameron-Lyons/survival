"""Direct Cox curves against stock R, with ownership and boundary checks."""

import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "coxsurv_reference.json").read_text())


def arguments(case):
    return {
        **{
            key: case[key]
            for key in (
                "ctype",
                "stype",
                "se_fit",
                "unlist",
                "y",
                "x2",
                "risk2",
                "y2",
                "strata2",
                "id2",
            )
        },
        "x": REFERENCE["x"],
        "wt": REFERENCE["weights"],
        "risk": REFERENCE["risk"],
        "varmat": REFERENCE["variance"],
        "strata": None
        if case["strata"] is None
        else r._coerce._r_factor(case["strata"], case["levels"]),
        "rownames": case["rownames"],
    }


def compare(actual, expected, *, kp=False):
    for name, values in expected.items():
        found = getattr(actual, name.replace(".", "_"))
        if name == "strata":
            found = list(found.values())
        if name == "ndeath":
            found = found[:, None]
        reference = np.asarray(values, dtype=float)
        if name == "surv" and kp and np.isnan(reference).any():
            # Stock R can raise a negative roundoff residue to a fractional
            # power after a terminal death. Retain its NaNs in the fixture,
            # but independently verify the zero-survival limit for every run.
            missing = np.isnan(reference)
            rows = missing if missing.ndim == 1 else missing.any(axis=1)
            first = np.flatnonzero(rows & ~np.r_[False, rows[:-1]])
            np.testing.assert_allclose(actual.n_event[first], actual.n_risk[first], atol=1e-12)
            assert np.all(actual.n_event[first] > 0)
            np.testing.assert_allclose(found[missing], 0, atol=2e-12)
            reference[missing] = 0
        np.testing.assert_allclose(found, reference, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_direct_curves_match_r(case):
    fitted = r.coxsurv_fit(**arguments(case))
    if case["unlist"]:
        assert isinstance(fitted, r.CoxSurvFitResult)
        compare(fitted, case["expected"], kp=case["stype"] == 1)
        if fitted.strata is not None:
            assert list(fitted.strata) == case["names"]
        assert fitted.hazard is None
    else:
        assert isinstance(fitted, r.CoxSurvFitList)
        expected = case["expected"]
        assert len(fitted) == len(expected)
        for curve, reference in zip(fitted, expected.values(), strict=True):
            compare(curve, reference, kp=case["stype"] == 1)
        assert fitted.names == (
            tuple(dict.fromkeys(case["id2"])) if case["id2"] else tuple(case["names"])
        )
    if case["unlist"]:
        assert (fitted.std_err is not None) == case["se_fit"]
    else:
        assert all((curve.std_err is not None) == case["se_fit"] for curve in fitted)


def simple(**changes):
    return {
        "y": [[1, 1], [2, 1], [3, 0]],
        "x": [[0.0], [1.0], [2.0]],
        "risk": [1.0] * 3,
        "x2": [[0.5], [1.5]],
        "risk2": [1.0, 2.0],
        "varmat": [[0.2]],
        **changes,
    }


@pytest.mark.parametrize("survtype", [1, 2, 3])
def test_legacy_wrapper_and_unused_arguments(survtype):
    args = simple()
    fit = r.survfitcoxph_fit(
        **{k: v for k, v in args.items() if k != "risk2"},
        newrisk=args["risk2"],
        survtype=survtype,
        vartype=object(),
    )
    modern = r.coxsurv_fit(
        **args,
        stype=1 if survtype == 1 else 2,
        ctype=2 if survtype == 3 else 1,
        cluster=object(),
        position=object(),
        oldid=object(),
    )
    np.testing.assert_array_equal(fit.surv, modern.surv)
    np.testing.assert_array_equal(fit.std_err, modern.std_err)
    with pytest.raises(ValueError, match="risk2 is required"):
        r.survfitcoxph_fit(**{k: v for k, v in args.items() if k != "risk2"})


def test_readonly_views_keep_native_storage_alive_and_pickle_roundtrip():
    fit = r.coxsurv_fit(**simple(unlist=False))
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own round-trip data
    for name in ("time", "surv", "cumhaz", "std_err", "hazard", "varhaz", "ndeath", "xbar"):
        left, right = getattr(fit[0], name), getattr(restored[0], name)
        np.testing.assert_array_equal(left, right)
        assert not left.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            left.setflags(write=True)
    view = fit[0].surv
    expected = view.copy()
    del fit
    gc.collect()
    np.testing.assert_array_equal(view, expected)
    assert np.shares_memory(restored[0].surv, restored[0]._fit.surv)


def test_prepared_responses_strided_matrices_and_zero_covariates():
    args = simple(y=r.Surv([1, 2, 3], [1, 1, 0]), x=np.arange(6.0).reshape(3, 2)[:, ::2] / 2)
    full = r.coxsurv_fit(**args)
    np.testing.assert_array_equal(full.surv, r.coxsurv_fit(**simple()).surv)
    null = r.coxsurv_fit(**simple(x=np.empty((3, 0)), x2=np.empty((2, 0)), varmat=np.empty((0, 0))))
    np.testing.assert_allclose(null.std_err[-1], np.sqrt(1 / 9 + 1 / 4) * np.array([1, 2]))
    no_se = r.coxsurv_fit(**simple(varmat=object(), se_fit=False))
    assert no_se.std_err is None


def test_unused_levels_empty_intervals_and_zero_prediction_risk():
    strata = r._coerce._r_factor(["a"] * 3, ["unused", "a"])
    fitted = r.coxsurv_fit(**simple(strata=strata, risk2=[0.0, 1.0]))
    assert fitted.n == [0, 3]
    assert fitted.strata == {"unused": 0, "a": 3}
    np.testing.assert_array_equal(fitted.surv[:, 0], 1)
    np.testing.assert_array_equal(fitted.cumhaz[:, 0], 0)
    individual = r.coxsurv_fit(
        **simple(strata=strata, y2=[[5, 6], [5, 6]], id2=["empty", "empty"], strata2=[1, 2])
    )
    assert individual.n == [0]
    assert individual.time.shape == individual.surv.shape == individual.std_err.shape == (0,)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"y": [[1, 2], [2, 1], [3, 0]]}, "status"),
        ({"y": [[1, 1, 1], [0, 2, 1], [0, 3, 0]]}, "start"),
        ({"risk": [0, 1, 1]}, "risk"),
        ({"risk2": [1, -1]}, "risk2"),
        ({"risk2": [1, np.inf]}, "risk2"),
        ({"wt": [1, -1, 1]}, "weights"),
        ({"wt": [0, 0, 0]}, "denominator"),
        ({"varmat": [[np.nan]]}, "varmat"),
        ({"varmat": [[1, 0], [0, 1]]}, "shape"),
        ({"x2": [[1, 2]]}, "columns"),
        ({"strata": ["a", None, "a"]}, "non-missing"),
        ({"id2": ["a", None]}, "non-missing"),
        ({"id2": ["a", "a"], "y2": [[1, 1], [0, 2]]}, "start < stop"),
        ({"id2": ["a", "a"], "y2": [[0, 2], [2, 3]], "strata2": [0, 1]}, "strata2"),
        ({"id2": ["a", "a"], "y2": [[0, 2], [2, 3]], "strata2": [1.5, 1]}, "integer"),
        ({"ctype": 3}, "ctype"),
    ],
)
def test_invalid_prepared_inputs_are_rejected(change, message):
    with pytest.raises((ValueError, TypeError), match=message):
        r.coxsurv_fit(**simple(**change))


def test_dotted_keyword_and_public_exports():
    assert r.coxsurv_fit(**simple(), **{"se.fit": False}).std_err is None
    assert survival.r_api.coxsurv_fit is r.coxsurv_fit
    with pytest.raises(TypeError, match="unexpected keyword"):
        r.coxsurv_fit(**simple(), extra=True)
