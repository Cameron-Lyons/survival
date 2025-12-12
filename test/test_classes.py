import os
import sys
from typing import Any

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
        df=5,
        theta=1.0,
        eps=1e-6,
        method="GCV",
        boundary_knots=(1.0, 20.0),
        intercept=True,
        penalty=True,
    )
    print(" PSpline created successfully")

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
