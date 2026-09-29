"""Plot coordinates captured from the actual R survival graphics methods."""

import json
from pathlib import Path

import numpy as np
import pytest
from survival import plotting, r

REFERENCE = json.loads((Path(__file__).parent / "fixtures/survfit_plot_reference.json").read_text())


def reference_fit(name):
    snapshot = REFERENCE["fits"][name]
    fields = {}
    for field, encoded in snapshot["fields"].items():
        value = None
        if encoded is not None:
            value = np.asarray(encoded["values"], dtype=float)
            if encoded["dim"]:
                value = value.reshape(encoded["dim"], order="F")
            value = value.tolist()
        fields[field.replace(".", "_")] = value
    options = {key: fields[key] for key in ("n", "time", "n_risk", "n_event", "n_censor", "cumhaz")}
    options.update(type=snapshot["type"] or "right", strata=snapshot["strata"])
    kind = snapshot["kind"]
    for field in ("conf_type", "conf_int", "logse"):
        if kind != "coxms":
            options[field] = snapshot[field]
    for field in ("std_err", "std_chaz", "lower", "upper"):
        if kind != "coxms":
            options[field] = fields[field]
    if kind in {"km", "cox"}:
        options["surv"] = fields["surv"]
        if kind == "cox":
            return r.CoxSurvfitResult(**options, colnames=snapshot["colnames"])
        return r.SurvfitResult(**options, t0=snapshot["t0"] or 0)
    options.update(
        pstate=fields["pstate"],
        p0=np.asarray(fields["p0"]).reshape(len(snapshot["strata"] or [""]), -1).tolist(),
        states=snapshot["states"],
        n_transition=[],
        transitions=None,
        n_id=[],
        t0=snapshot["t0"] or 0,
    )
    if kind == "coxms":
        options["pstate"] = np.asarray(options["pstate"])
        options["cumhaz"] = np.asarray(options["cumhaz"])
        return r.CoxSurvfitMultiStateResult(
            **options, cumhaz_names=snapshot["hazard_names"], newdata=None, start_time=None
        )
    return r.SurvfitMultiStateResult(**options, hazard_names=snapshot["hazard_names"])


def options_for(case):
    return {
        key.replace(".", "_"): value
        for key, value in (case["options"] or {}).items()
        if key not in {"pch", "col"}
    }


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=2e-11, atol=2e-13)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_prepared_graphics_match_r(case):
    options = options_for(case)
    bars = options.pop("conf_times", None)
    options.pop("conf_offset", None)
    options.pop("conf_cap", None)
    points = case["method"] == "points"
    censor = options.pop("censor", False)
    if bars is not None:
        options["conf_int"] = True
    if points:
        options["conf_int"] = False
    data = plotting.survfit_plot_data(reference_fit(case["fit"]), **options)
    calls = []
    for curve in data.curves:
        if points:
            keep = np.ones(len(curve.time), dtype=bool) if censor else curve.event
            calls.append({"kind": "point", "x": curve.time[keep], "y": curve.estimate[keep]})
            continue
        if data.plot_estimate:
            xx, yy = curve.step()
            calls.append({"kind": "line", "x": xx, "y": yy})
            if len(curve.censor_time):
                calls.append({"kind": "point", "x": curve.censor_time, "y": curve.censor_value})
            elif options.get("mark_time", False):
                calls.append({"kind": "point", "x": [], "y": []})
        if curve.lower is not None:
            if bars is not None:
                from survival._plot_data import step_at

                query = np.asarray(bars)
                calls.append(
                    {
                        "kind": "segment",
                        "x0": query,
                        "x1": query,
                        "y0": step_at(curve.time, curve.lower, query, right=False),
                        "y1": step_at(curve.time, curve.upper, query, right=False),
                    }
                )
            else:
                for bound in (curve.lower, curve.upper):
                    xx, yy = curve.step(bound)
                    calls.append({"kind": "line", "x": xx, "y": yy})
    expected = case["expected"]
    assert len(calls) == len(expected["calls"])
    for actual, wanted in zip(calls, expected["calls"], strict=True):
        assert actual["kind"] == wanted["kind"]
        if wanted["kind"] == "point":
            # R leaves NA marker coordinates beyond xmax; they are not drawn.
            actual = dict(actual)
            wanted = dict(wanted)
            for pair in (actual, wanted):
                xx, yy = np.asarray(pair["x"], dtype=float), np.asarray(pair["y"], dtype=float)
                keep = np.isfinite(xx) & np.isfinite(yy)
                pair["x"], pair["y"] = xx[keep], yy[keep]
        for key in wanted.keys() - {"kind"}:
            assert_close(actual[key], wanted[key])
    if not points and not options.get("cumprob"):
        assert_close(data.xend, expected["endpoint"]["x"])
        assert_close(data.yend, expected["endpoint"]["y"])
    if case["method"] == "plot":
        assert data.xlog == expected["xlog"]
        # R 3.8-12 sets its ylog variable but fails to update the graphics log
        # argument for these named transforms. Use their documented log axis.
        assert data.ylog == (expected["ylog"] or options.get("fun") in {"log", "logpct"})


def test_data_preparation_preserves_fit_and_does_not_copy_influence_matrices():
    fit = r.survfit(r.Surv([1, 2, 2, 3, 4], [1, 1, 0, 1, 0]), influence=True)
    expected = list(fit.surv)

    def transform(values):
        values *= 2
        return values

    data = plotting.survfit_plot_data(fit, fun=transform, mark_time=True)
    data.curves[0].estimate[:] = -99
    assert fit.surv == expected
    assert fit.influence_surv is not None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fun": "bogus"}, "unrecognized"),
        ({"fun": lambda x: 1}, "one value"),
        ({"log": "z"}, "log must"),
        ({"cumhaz": [2]}, "only applies"),
        ({"cumprob": True}, "multistate"),
        ({"conf_int": 1.2}, "confidence level"),
        ({"conf_int": "wrong"}, "conf_int must"),
        ({"conf_type": "wrong"}, "confidence type"),
        ({"mark_time": [[1, 2]]}, "vector"),
        ({"xmax": -1}, "curve origin"),
        ({"xmax": float("nan")}, "finite"),
    ],
)
def test_plot_data_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        plotting.survfit_plot_data(reference_fit("km"), **options)


def test_plot_data_rejects_unavailable_intervals_and_bad_state_indices():
    with pytest.raises(TypeError, match="survfit"):
        plotting.survfit_plot_data([1, 2])
    with pytest.raises(ValueError, match="standard errors"):
        plotting.survfit_plot_data(reference_fit("no_se"), conf_int=True, conf_type="log")
    for indices in ([0], [9], [1.5], [float("nan")], []):
        with pytest.raises(ValueError, match="one-based indices"):
            plotting.survfit_plot_data(reference_fit("aj"), cumprob=indices)
    with pytest.raises(ValueError, match="not available"):
        plotting.survfit_plot_data(reference_fit("aj"), cumprob=True, conf_int=True)


def test_long_constant_runs_have_constant_size_render_paths():
    fit = r.survfit(r.Surv(np.arange(1, 10001), np.zeros(10000, dtype=int)))
    curve = plotting.survfit_plot_data(fit, conf_int=False).curves[0]
    xx, yy = curve.step()
    np.testing.assert_array_equal(xx, [0, 10000])
    np.testing.assert_array_equal(yy, [1, 1])
