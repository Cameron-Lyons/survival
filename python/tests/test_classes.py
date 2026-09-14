import math
from statistics import NormalDist

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_link_function_params_match_r_bounded_links():
    link = survival.regression.LinkFunctionParams(edge=0.001)

    for p in (0.1, 0.5, 0.9):
        assert link.blogit(p) == pytest.approx(math.log(p / (1 - p)))
        assert link.bprobit(p) == pytest.approx(NormalDist().inv_cdf(p))
        assert link.bcloglog(p) == pytest.approx(math.log(-math.log(1 - p)))
        assert link.blog(p) == pytest.approx(math.log(p))

    # blogit(x, edge) clamps to [edge, 1 - edge] before the transform
    assert link.blogit(0.0) == pytest.approx(math.log(0.001 / 0.999))
    assert link.blogit(1.0) == pytest.approx(math.log(0.999 / 0.001))
