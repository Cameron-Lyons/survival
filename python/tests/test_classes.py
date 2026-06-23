import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_link_function_params():
    link_func = survival.LinkFunctionParams(edge=0.001)

    test_values = [0.1, 0.5, 0.9]
    for val in test_values:
        blogit_result = link_func.blogit(val)
        bprobit_result = link_func.bprobit(val)
        bcloglog_result = link_func.bcloglog(val)
        blog_result = link_func.blog(val)
        assert isinstance(blogit_result, float)
        assert isinstance(bprobit_result, float)
        assert isinstance(bcloglog_result, float)
        assert isinstance(blog_result, float)


def test_pspline_creation():
    x = [float(i) for i in range(1, 21)]

    pspline = survival.PSpline(
        x=x,
        df=3,
        theta=0.1,
        eps=1e-6,
        method="GCV",
        boundary_knots=(0.0, 21.0),
        intercept=False,
        penalty=False,
    )

    assert not pspline.fitted
    assert pspline.coefficients is None
    assert pspline.df == 3
    assert pspline.eps == 1e-6


def test_pspline_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="x contains non-finite"):
        survival.PSpline(
            x=[1.0, float("nan")],
            df=1,
            theta=0.1,
            eps=1e-6,
            method="GCV",
            boundary_knots=(0.0, 5.0),
            intercept=True,
            penalty=False,
        )
    with pytest.raises(ValueError, match="boundary_knots must be finite"):
        survival.PSpline(
            x=[1.0, 2.0],
            df=1,
            theta=0.1,
            eps=1e-6,
            method="GCV",
            boundary_knots=(1.0, 1.0),
            intercept=True,
            penalty=False,
        )
    with pytest.raises(ValueError, match="df must be positive"):
        survival.PSpline(
            x=[1.0, 2.0],
            df=0,
            theta=0.1,
            eps=1e-6,
            method="GCV",
            boundary_knots=(0.0, 5.0),
            intercept=True,
            penalty=False,
        )
    with pytest.raises(ValueError, match="theta must be non-negative"):
        survival.PSpline(
            x=[1.0, 2.0],
            df=1,
            theta=-0.1,
            eps=1e-6,
            method="GCV",
            boundary_knots=(0.0, 5.0),
            intercept=True,
            penalty=False,
        )
    with pytest.raises(ValueError, match="eps must be positive"):
        survival.PSpline(
            x=[1.0, 2.0],
            df=1,
            theta=0.1,
            eps=0.0,
            method="GCV",
            boundary_knots=(0.0, 5.0),
            intercept=True,
            penalty=False,
        )
    with pytest.raises(ValueError, match="x length"):
        survival.PSpline(
            x=[1.0, 2.0],
            df=3,
            theta=0.1,
            eps=1e-6,
            method="GCV",
            boundary_knots=(0.0, 5.0),
            intercept=True,
            penalty=False,
        )


def test_pspline_invalid_method():
    x = [float(i) for i in range(1, 21)]

    pspline_invalid = survival.PSpline(
        x=x,
        df=3,
        theta=0.1,
        eps=1e-6,
        method="INVALID_METHOD",
        boundary_knots=(0.0, 21.0),
        intercept=False,
        penalty=True,
    )
    with pytest.raises(Exception, match="Unsupported penalty method"):
        pspline_invalid.fit()


def test_pspline_predict_without_fit():
    x = [float(i) for i in range(1, 21)]

    pspline_unfitted = survival.PSpline(
        x=x,
        df=3,
        theta=0.1,
        eps=1e-6,
        method="GCV",
        boundary_knots=(0.0, 21.0),
        intercept=False,
        penalty=False,
    )
    with pytest.raises(Exception, match="(?i)not fitted"):
        pspline_unfitted.predict([5.0])


def test_pspline_predict_rejects_non_finite_values():
    pspline = survival.PSpline(
        x=[1.0, 2.0, 3.0, 4.0],
        df=1,
        theta=0.1,
        eps=1e-6,
        method="GCV",
        boundary_knots=(0.0, 5.0),
        intercept=True,
        penalty=False,
    )
    pspline.fit()

    with pytest.raises(ValueError, match="new_x contains non-finite"):
        pspline.predict([1.0, float("inf")])


def test_pspline_singular_fit():
    x = [float(i) for i in range(1, 21)]

    pspline_singular = survival.PSpline(
        x=x,
        df=5,
        theta=1.0,
        eps=1e-6,
        method="GCV",
        boundary_knots=(1.0, 20.0),
        intercept=True,
        penalty=True,
    )
    try:
        pspline_singular.fit()
    except ValueError as e:
        assert "singular" in str(e).lower() or "failed" in str(e).lower()
