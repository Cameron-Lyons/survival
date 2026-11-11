"""Raw martingale kernels preserve expected counts at extreme risk scales."""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
native = survival._survival


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("score", [1e308, 1e-300, np.nextafter(0.0, 1.0)])
def test_raw_right_martingale_scales(score, ties):
    data = native.CoxMartInput(native.SurvivalData([1.0, 2.0], [1, 0]), [score, score])
    np.testing.assert_allclose(native.coxmart(data, ties=ties), [0.5, -0.5], atol=2e-12, rtol=0)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("scales", [(1e308, 1e-300), (1e-300, 1e308)])
def test_raw_counting_martingale_keeps_disjoint_risk_sets(scales, ties):
    data = native.AndersenGillInput(
        native.CountingProcessData([0.0, 0.0, 3.0, 3.0], [1.0, 2.0, 4.0, 5.0], [1, 0, 1, 0]),
        [scales[0], scales[0], scales[1], scales[1]],
    )
    np.testing.assert_allclose(
        native.agmart(data, ties=ties), [0.5, -0.5, 0.5, -0.5], atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("counting", [False, True])
def test_zero_event_risk_is_a_python_error(counting):
    if counting:
        data = native.AndersenGillInput(
            native.CountingProcessData([0.0, 0.0], [1.0, 2.0], [1, 0]), [0.0, 0.0]
        )
        function = native.agmart
    else:
        data = native.CoxMartInput(native.SurvivalData([1.0, 2.0], [1, 0]), [0.0, 0.0])
        function = native.coxmart
    with pytest.raises(ValueError, match="positive risk-set sum"):
        function(data)
