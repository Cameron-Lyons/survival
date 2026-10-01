"""AFT constructors and fitting boundaries reject malformed native inputs."""

import numpy as np
import pytest

from .helpers import setup_survival_import

core = setup_survival_import()._survival
MAX_INDEX = int(np.iinfo(np.uintp).max)
FITTERS = ("survreg_fit", "survreg_fit_raw", "survpenal_fit", "survpenal_fit_raw")


def _arguments():
    return {
        "time": [1.2, 2.5, 0.9, 3.0],
        "status": [1, 0, 1, 1],
        "covariates": [[1.0, 0.2], [1.0, -0.3], [1.0, 0.4], [1.0, 0.1]],
    }


def _fit(name, data, distribution, **kwargs):
    if "survpenal" in name:
        kwargs.update(
            penalties=[core.CoxPenalty.ridge(theta=1, scale=False)], pcols=[[1]], assign=[[0], [1]]
        )
    return getattr(core, name)(data, distribution, **kwargs)


@pytest.mark.parametrize("field", ["status", "time2", "weights", "offset", "strata", "cluster"])
@pytest.mark.parametrize("length", [0, 3, 5])
def test_constructor_checks_all_row_lengths(field, length):
    values = _arguments()
    values[field] = [1] * length
    with pytest.raises(ValueError, match=field):
        core.SurvregData(**values)


@pytest.mark.parametrize("field", ["time", "weights", "offset"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_constructor_checks_finite_response_and_adjustments(field, value):
    with pytest.raises(ValueError, match=field):
        core.SurvregData(**{**_arguments(), field: [value] * 4})


def test_stratum_count_cannot_overflow_but_cluster_labels_are_opaque():
    with pytest.raises(ValueError, match="Invalid strata"):
        core.SurvregData(**_arguments(), strata=[MAX_INDEX] * 4)
    assert core.SurvregData(**_arguments(), cluster=[MAX_INDEX] * 4).cluster == [MAX_INDEX] * 4


@pytest.mark.parametrize("name", FITTERS)
@pytest.mark.parametrize(
    "size", [MAX_INDEX, MAX_INDEX // 2, 2 ** (np.dtype(np.uintp).itemsize * 4)]
)
@pytest.mark.parametrize("custom", [False, True])
def test_unrepresentable_covariance_is_rejected_before_callbacks(name, size, custom):
    def unused(*args):
        raise AssertionError("a callback was invoked before validating the parameter count")

    distribution = (
        core.SurvregDistribution.from_callbacks("unused", unused, unused, unused, unused)
        if custom
        else core.SurvregDistribution("gaussian")
    )
    data = core.SurvregData(**_arguments(), strata=[int(size) - 1] * 4)
    with pytest.raises(ValueError, match="too many AFT parameters"):
        _fit(name, data, distribution)
    data = core.SurvregData(**_arguments())
    with pytest.raises(ValueError, match="too many AFT parameters"):
        _fit(name, data, distribution, nstrat=int(size))


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_validated_data_owns_buffers_and_fit_does_not_mutate_it(order):
    values = _arguments()
    x = np.array(values["covariates"], order="F" if order == "F" else "C")
    if order == "strided":
        backing = np.zeros((4, 4))
        backing[:, ::2] = x
        x = backing[:, ::2]
    time = np.array(values["time"])
    data = core.SurvregData(time, values["status"], x)
    x[:] = np.nan
    time[:] = np.nan
    for name in FITTERS:
        fit = _fit(name, data, core.SurvregDistribution("gaussian"))
        assert np.isfinite(fit.coefficients).all()
        assert data.time == values["time"]
        assert data.covariates == values["covariates"]
