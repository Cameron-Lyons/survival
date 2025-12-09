#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'wheels'))

try:
    import survival
    print("✅ Successfully imported survival module")
    
    print("\n=== Testing collapse ===")
    # y must have 3*n elements (time1, time2, status)
    # For n=4, we need 12 elements
    y = [1.0, 2.0, 3.0, 4.0,  # time1 for 4 observations
         2.0, 3.0, 4.0, 5.0,  # time2 for 4 observations
         1.0, 0.0, 1.0, 0.0]  # status for 4 observations
    x = [1, 1, 1, 1]
    istate = [0, 0, 0, 0]
    id = [1, 1, 2, 2]
    wt = [1.0, 1.0, 1.0, 1.0]
    order = [0, 1, 2, 3]
    
    result = survival.collapse(y, x, istate, id, wt, order)
    print(f"✅ collapse executed successfully")
    print(f"   Result type: {type(result)}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert 'matrix' in result, "Should have 'matrix' key"
    assert 'dimnames' in result, "Should have 'dimnames' key"
    print(f"   matrix: {result['matrix']}")
    print(f"   dimnames: {result['dimnames']}")
    
    print("\n✅ All utility tests passed!")
    
except ImportError as e:
    print(f"❌ Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error in utility tests: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

