#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'wheels'))

try:
    import survival
    print("✅ Successfully imported survival module")
    
    print("\n=== Testing coxcount1 ===")
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    strata = [1, 0, 0, 0, 0, 0, 0, 0]
    
    result = survival.coxcount1(time, status, strata)
    print(f"✅ coxcount1 executed successfully")
    print(f"   time: {result.time}")
    print(f"   nrisk: {result.nrisk}")
    print(f"   index: {result.index}")
    print(f"   status: {result.status}")
    assert hasattr(result, 'time'), "Missing 'time' attribute"
    assert hasattr(result, 'nrisk'), "Missing 'nrisk' attribute"
    assert hasattr(result, 'index'), "Missing 'index' attribute"
    assert hasattr(result, 'status'), "Missing 'status' attribute"
    
    print("\n=== Testing coxcount2 ===")
    time1 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    time2 = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    status = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    sort1 = [0, 2, 4, 1, 3, 5]
    sort2 = [0, 2, 4, 1, 3, 5]
    strata2 = [1, 0, 0, 0, 0, 0]
    
    result2 = survival.coxcount2(time1, time2, status, sort1, sort2, strata2)
    print(f"✅ coxcount2 executed successfully")
    print(f"   time: {result2.time}")
    print(f"   nrisk: {result2.nrisk}")
    assert hasattr(result2, 'time'), "Missing 'time' attribute"
    assert hasattr(result2, 'nrisk'), "Missing 'nrisk' attribute"
    
    print("\n✅ All core tests passed!")
    
except ImportError as e:
    print(f"❌ Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error in core tests: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

