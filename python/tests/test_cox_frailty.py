import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _fixed_gaussian_fit():
    return survival.r_api._core.coxph_frailty_fit(
        [float(value) for value in range(1, 13)],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [[value] for value in [1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1.0]],
        [0, 1, 2, 3] * 3,
        0.5,
        max_iter=50,
        eps=1e-12,
        toler=1e-13,
        method="breslow",
    )


def test_sparse_gaussian_frailty_native_fit_matches_reference():
    fit = _fixed_gaussian_fit()

    assert fit.coefficients[0] == pytest.approx([-0.561938099930235], abs=1e-12)
    assert fit.frailty == pytest.approx(
        [0.182835600787012, -0.0855299813397072, -0.0416595588809712, -0.0556460605663337],
        abs=1e-12,
    )
    assert fit.frailty_variance == pytest.approx(
        [0.316161474915226, 0.345174733582117, 0.305199707040236, 0.294105320841684],
        abs=1e-12,
    )
    assert fit.covariate_degrees_of_freedom == pytest.approx([0.967518101717635], abs=1e-12)
    assert fit.frailty_degrees_of_freedom == pytest.approx(1.47871752724147, abs=1e-12)
    assert fit.information_matrix[0] == pytest.approx([0.817409122379781], abs=1e-12)
    assert fit.naive_information_matrix[0] == pytest.approx([0.790858122411564], abs=1e-12)
    assert fit.log_likelihood == pytest.approx(
        [-12.6195059232875, -12.2502184876667],
        abs=1e-12,
    )
    assert fit.means == pytest.approx([0.916666666666667], abs=1e-12)
    assert fit.linear_predictors == pytest.approx(
        [
            0.023619805806779,
            0.0362232736451772,
            -0.369456783840275,
            0.347076244383668,
            0.0798136157998024,
            0.204804703624248,
            -0.538038213819345,
            -0.0462804255674963,
            0.416976475757943,
            -0.357133396305987,
            0.304868936076007,
            -0.10247423556052,
        ],
        abs=1e-12,
    )


def test_sparse_gaussian_frailty_native_fit_scales_with_group_indices():
    n = 10_000
    groups = 2_000
    fit = survival.r_api._core.coxph_frailty_fit(
        [float(value) for value in range(1, n + 1)],
        [int(value % 4 == 0) for value in range(n)],
        [[math.sin(value / 97.0)] for value in range(n)],
        [value % groups for value in range(n)],
        0.25,
        max_iter=3,
        eps=1e-7,
        method="breslow",
    )

    assert len(fit.frailty) == groups
    assert len(fit.frailty_variance) == groups
    assert all(math.isfinite(value) for value in fit.coefficients[0])


def test_sparse_gaussian_frailty_native_fit_combines_dense_penalty_and_strata():
    fit = survival.r_api._core.coxph_frailty_fit(
        [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [[value] for value in [1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1.0]],
        [0, 1, 2, 3] * 3,
        0.5,
        strata=[1] * 6 + [2] * 6,
        penalty_matrix=[[0.2]],
        max_iter=50,
        eps=1e-12,
        toler=1e-13,
        method="efron",
    )

    assert fit.coefficients[0] == pytest.approx([0.203649783606771], abs=1e-12)
    assert fit.frailty == pytest.approx(
        [0.199233197374958, -0.0987328693476504, 0.150110660953487, -0.250610988980794],
        abs=1e-12,
    )
    assert fit.frailty_variance == pytest.approx(
        [0.302173996806596, 0.295847015106486, 0.320966626237654, 0.328803061516902],
        abs=1e-12,
    )
    assert fit.covariate_degrees_of_freedom == pytest.approx([0.819597722065366], abs=1e-12)
    assert fit.frailty_degrees_of_freedom == pytest.approx(1.49535374602864, abs=1e-12)
    assert fit.information_matrix[0] == pytest.approx([0.702283308058964], abs=1e-12)
    assert fit.naive_information_matrix[0] == pytest.approx([0.575589799529657], abs=1e-12)
    assert fit.log_likelihood == pytest.approx([-9.16951837745593, -8.71619272199139], abs=1e-12)


@pytest.mark.parametrize(
    ("groups", "theta", "message"),
    [
        ([0, 2], 0.5, "contiguous zero-based"),
        ([0, 0], 0.0, "positive"),
    ],
)
def test_sparse_gaussian_frailty_native_fit_validates_groups_and_theta(groups, theta, message):
    with pytest.raises(ValueError, match=message):
        survival.r_api._core.coxph_frailty_fit(
            [1.0, 2.0],
            [1, 0],
            [[0.0], [1.0]],
            groups,
            theta,
        )
