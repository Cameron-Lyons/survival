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


def test_sparse_gaussian_frailty_native_counting_fit_matches_reference():
    entry_times = [
        0.0,
        0.0,
        0.5,
        1.0,
        0.0,
        2.0,
        3.0,
        1.0,
        4.0,
        2.0,
        6.0,
        5.0,
        7.0,
        8.0,
        6.0,
        10.0,
        11.0,
        9.0,
    ]
    fit = survival.r_api._core.coxph_frailty_fit(
        [float(value) for value in range(1, 19)],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [
            [value]
            for value in [
                1.2,
                0.7,
                1.5,
                0.2,
                1.1,
                0.4,
                1.8,
                0.9,
                0.5,
                1.4,
                0.3,
                1.0,
                0.6,
                1.7,
                0.1,
                1.3,
                0.8,
                1.6,
            ]
        ],
        [0, 1, 2, 3, 4, 5] * 3,
        0.5,
        max_iter=50,
        eps=1e-12,
        toler=1e-13,
        method="breslow",
        entry_times=entry_times,
    )

    assert fit.coefficients[0] == pytest.approx([-0.736602816003447], abs=1e-12)
    assert fit.frailty == pytest.approx(
        [
            -0.00812087501187664,
            0.218837152806033,
            0.0991854342400643,
            -0.173140923804785,
            -0.0974646990423705,
            -0.0392960891870656,
        ],
        abs=1e-12,
    )
    assert fit.log_likelihood == pytest.approx(
        [-18.2340928191392, -17.1661166752336],
        abs=1e-12,
    )
    assert fit.covariate_degrees_of_freedom == pytest.approx([0.888227879539063], abs=1e-12)
    assert fit.frailty_degrees_of_freedom == pytest.approx(2.25629339890398, abs=1e-12)
    assert fit.information_matrix[0] == pytest.approx([0.448731919320106], abs=1e-12)
    assert fit.naive_information_matrix[0] == pytest.approx([0.398576201179192], abs=1e-12)
    assert fit.entry_times == pytest.approx(entry_times)
    assert fit.linear_predictors == pytest.approx(
        [
            -0.192271579012738,
            0.402987856806895,
            -0.305946114561832,
            0.379311188197801,
            -0.207955121442888,
            0.36583545961483,
            -0.634233268614807,
            0.255667293606205,
            0.430656701441615,
            -0.504612191006336,
            0.38132713135987,
            -0.076126229987238,
            0.24969011058933,
            -0.333614959196552,
            0.725297827842994,
            -0.430951909405991,
            0.0130257233581464,
            -0.518087919589306,
        ],
        abs=1e-12,
    )


def test_sparse_gaussian_frailty_native_counting_fit_scales_with_group_indices():
    n = 10_000
    groups = 2_000
    fit = survival.r_api._core.coxph_frailty_fit(
        [float(value) for value in range(1, n + 1)],
        [int(value % 4 == 0) for value in range(n)],
        [[math.sin(value / 97.0)] for value in range(n)],
        [value % groups for value in range(n)],
        0.25,
        entry_times=[max(0.0, float(value - 1_000)) for value in range(1, n + 1)],
        max_iter=3,
        eps=1e-7,
        method="breslow",
    )

    assert len(fit.frailty) == groups
    assert len(fit.frailty_variance) == groups
    assert all(math.isfinite(value) for value in fit.coefficients[0])


def test_sparse_gaussian_frailty_native_counting_fit_combines_penalty_strata_and_ties():
    fit = survival.r_api._core.coxph_frailty_fit(
        [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [[value] for value in [1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1.0]],
        [0, 1, 2, 3] * 3,
        0.5,
        strata=[1] * 6 + [2] * 6,
        penalty_matrix=[[0.2]],
        entry_times=[0.0, 0.0, 0.5, 1.0, 0.0, 2.0, 3.0, 1.0, 4.0, 2.0, 6.0, 5.0],
        max_iter=50,
        eps=1e-12,
        toler=1e-13,
        method="efron",
    )

    assert fit.coefficients[0] == pytest.approx([-0.396637670196876], abs=1e-12)
    assert fit.frailty == pytest.approx(
        [0.130597126706777, -0.0946235856514112, 0.29620046214582, -0.332174003201186],
        abs=1e-12,
    )
    assert fit.frailty_variance == pytest.approx(
        [0.280997264733325, 0.30112511706187, 0.371283991529981, 0.327119681253034],
        abs=1e-12,
    )
    assert fit.covariate_degrees_of_freedom == pytest.approx([0.805320383723899], abs=1e-12)
    assert fit.frailty_degrees_of_freedom == pytest.approx(1.43059887013871, abs=1e-12)
    assert fit.information_matrix[0] == pytest.approx([0.799519783795863], abs=1e-12)
    assert fit.naive_information_matrix[0] == pytest.approx([0.643869579081333], abs=1e-12)
    assert fit.log_likelihood == pytest.approx([-7.74240202181578, -7.08840456550056], abs=1e-12)


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


def test_sparse_gaussian_frailty_native_fit_validates_entry_times():
    arguments = (
        [1.0, 2.0],
        [1, 0],
        [[0.0], [1.0]],
        [0, 0],
        0.5,
    )
    with pytest.raises(ValueError, match="entry_times has 1 rows"):
        survival.r_api._core.coxph_frailty_fit(*arguments, entry_times=[0.0])
    with pytest.raises(ValueError, match=r"entry_times\[0\] must be less"):
        survival.r_api._core.coxph_frailty_fit(*arguments, entry_times=[1.0, 0.0])


def test_sparse_student_t_frailty_native_fit_matches_reference():
    fit = survival.r_api._core.coxph_frailty_fit(
        [float(value) for value in range(2, 20)],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [
            [value]
            for value in [
                -1.2,
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
                1.2,
                -1.0,
                -0.6,
                -0.2,
                0.2,
                0.6,
                1.0,
                1.4,
                -1.4,
                -0.9,
                0.1,
                0.9,
            ]
        ],
        [value // 3 for value in range(18)],
        0.5,
        max_iter=50,
        eps=1e-10,
        toler=1e-13,
        method="breslow",
        distribution="t",
        tdf=5.0,
    )

    assert fit.coefficients[0] == pytest.approx([-0.19743147930125], abs=1e-12)
    assert fit.frailty == pytest.approx(
        [
            0.426144390050408,
            0.298550076830906,
            0.136467056887759,
            -0.00855828519712049,
            -0.24217955197031,
            -0.780582477534266,
        ],
        abs=1e-12,
    )
    assert fit.frailty_variance == pytest.approx(
        [
            0.317599821981425,
            0.249342034909525,
            0.203691040339686,
            0.184566103436754,
            0.18538781841236,
            0.376879269003266,
        ],
        abs=1e-12,
    )
    assert fit.log_likelihood == pytest.approx(
        [-23.6901985559574, -19.9806323372429],
        abs=1e-12,
    )
    assert fit.covariate_degrees_of_freedom == pytest.approx([0.981376565411526], abs=1e-12)
    assert fit.frailty_degrees_of_freedom == pytest.approx(1.63990825488468, abs=1e-12)
    assert fit.distribution == "t"
    assert fit.tdf == pytest.approx(5.0)


@pytest.mark.parametrize("tdf", [2.0, 1.0, float("inf")])
def test_sparse_student_t_frailty_native_fit_validates_tdf(tdf):
    with pytest.raises(ValueError, match="greater than 2"):
        survival.r_api._core.coxph_frailty_fit(
            [1.0, 2.0],
            [1, 0],
            [[0.0], [1.0]],
            [0, 0],
            0.5,
            distribution="t",
            tdf=tdf,
        )

    with pytest.raises(ValueError, match="only valid"):
        survival.r_api._core.coxph_frailty_fit(
            [1.0, 2.0],
            [1, 0],
            [[0.0], [1.0]],
            [0, 0],
            0.5,
            distribution="gaussian",
            tdf=tdf,
        )
