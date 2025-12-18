import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing LinkFunctionParams ===")
    link_func = survival.LinkFunctionParams(edge=0.001)
    print(" LinkFunctionParams created successfully")

    test_values: list[float] = [0.1, 0.5, 0.9]
    for val in test_values:
        blogit_result = link_func.blogit(val)
        bprobit_result = link_func.bprobit(val)
        bcloglog_result = link_func.bcloglog(val)
        blog_result = link_func.blog(val)
        print(
            f"   Input {val}: blogit={blogit_result:.4f}, "
            f"bprobit={bprobit_result:.4f}, bcloglog={bcloglog_result:.4f}, "
            f"blog={blog_result:.4f}"
        )
        assert isinstance(blogit_result, float), "blogit should return float"
        assert isinstance(bprobit_result, float), "bprobit should return float"
        assert isinstance(bcloglog_result, float), "bcloglog should return float"
        assert isinstance(blog_result, float), "blog should return float"

    print("\n=== Testing PSpline ===")
    x: list[float] = [float(i) for i in range(1, 21)]

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
    print(" PSpline created successfully")

    assert not pspline.fitted, "Model should not be fitted initially"
    assert pspline.coefficients is None, "Coefficients should be None before fitting"
    print("   Initial state: fitted=False, coefficients=None")

    assert pspline.df == 3, "df getter should return 3"
    assert pspline.eps == 1e-6, "eps getter should return 1e-6"
    print(f"   Getters: df={pspline.df}, eps={pspline.eps}")

    try:
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
        pspline_invalid.fit()
        print("   ERROR: Should have raised an exception for invalid method")
        sys.exit(1)
    except Exception as e:
        assert "Unsupported penalty method" in str(e), f"Wrong error message: {e}"
        print("   Invalid method error handling: OK")

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
    try:
        pspline_unfitted.predict([5.0])
        print("   ERROR: Should have raised an exception for predict without fit")
        sys.exit(1)
    except Exception as e:
        assert "not fitted" in str(e).lower(), f"Wrong error message: {e}"
        print("   Predict without fit error handling: OK")

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
        print("   Linear solve test: fit succeeded (OK)")
    except ValueError as e:
        assert "singular" in str(e).lower() or "failed" in str(e).lower(), f"Wrong error: {e}"
        print("   Linear solve error handling: OK (ValueError raised properly)")

    print("\n All class tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in class tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
