#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'wheels'))

try:
    import survival
    print("✅ Successfully imported survival module")
    
    print("\n=== Testing LinkFunctionParams ===")
    link_func = survival.LinkFunctionParams(edge=0.001)
    print(f"✅ LinkFunctionParams created successfully")
    
    test_values = [0.1, 0.5, 0.9]
    for val in test_values:
        blogit_result = link_func.blogit(val)
        bprobit_result = link_func.bprobit(val)
        bcloglog_result = link_func.bcloglog(val)
        blog_result = link_func.blog(val)
        print(f"   Input {val}: blogit={blogit_result:.4f}, bprobit={bprobit_result:.4f}, bcloglog={bcloglog_result:.4f}, blog={blog_result:.4f}")
        assert isinstance(blogit_result, float), "blogit should return float"
        assert isinstance(bprobit_result, float), "bprobit should return float"
        assert isinstance(bcloglog_result, float), "bcloglog should return float"
        assert isinstance(blog_result, float), "blog should return float"
    
    print("\n=== Testing PSpline ===")
    x = [0.1 * i for i in range(20)]
    pspline = survival.PSpline(
        x=x,
        df=10,
        theta=1.0,
        eps=1e-6,
        method="GCV",
        boundary_knots=(0.0, 2.0),
        intercept=True,
        penalty=True,
    )
    print(f"✅ PSpline created successfully")
    pspline.fit()
    print(f"✅ PSpline.fit() executed successfully")
    
    print("\n✅ All class tests passed!")
    
except ImportError as e:
    print(f"❌ Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error in class tests: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

