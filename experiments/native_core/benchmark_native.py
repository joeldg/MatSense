import time
import ctypes
import numpy as np
import sys
import os

# Ensure we can import the original Python module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.analyzer import calculate_fast_kuzushi

# Try loading the compiled Golang library
lib_path = os.path.join(os.path.dirname(__file__), 'libkuzushi.so')
try:
    lib = ctypes.CDLL(lib_path)
    FastKuzushi = lib.FastKuzushi
    FastKuzushi.argtypes = [
        ctypes.POINTER(ctypes.c_double), # kpts array
        ctypes.POINTER(ctypes.c_double), # outDist
        ctypes.POINTER(ctypes.c_double), # outComX
        ctypes.POINTER(ctypes.c_double)  # outComY
    ]
    FastKuzushi.restype = ctypes.c_int
except OSError:
    print(f"❌ Error: Could not find {lib_path}. Did you run 'make'?")
    sys.exit(1)

def go_calculate_kuzushi(kpts):
    """Python wrapper for the CGO exported function"""
    # Flatten array to ctypes format
    # In production, this conversion overhead is avoided by keeping state continuous,
    # but for benchmark fairness, we will accept the flat cast here.
    kpts_flat = kpts.astype(np.float64).flatten()
    kpts_ptr = kpts_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_dist = ctypes.c_double(0)
    out_com_x = ctypes.c_double(0)
    out_com_y = ctypes.c_double(0)
    
    result = FastKuzushi(kpts_ptr, ctypes.byref(out_dist), ctypes.byref(out_com_x), ctypes.byref(out_com_y))
    return bool(result)

def benchmark():
    ITERATIONS = 100_000
    print(f"🚀 Racing Pure Python vs Native Golang (CGO) over {ITERATIONS:,} frames...")

    # Generate mock YOLO keypoints (17 joints, 3 channels)
    kpts_batch = [np.random.rand(17, 3) * 1000.0 for _ in range(ITERATIONS)]
    
    # 1. Race Python
    start_py = time.perf_counter()
    for kpts in kpts_batch:
        _ = calculate_fast_kuzushi(kpts)
    end_py = time.perf_counter()
    py_time = end_py - start_py

    # 2. Race Golang
    start_go = time.perf_counter()
    for kpts in kpts_batch:
        _ = go_calculate_kuzushi(kpts)
    end_go = time.perf_counter()
    go_time = end_go - start_go

    print(f"\n🏁 RESULTS:")
    print(f"   Python savgol/numpy: {py_time:.4f} seconds")
    print(f"   Golang Math Native:  {go_time:.4f} seconds")
    
    if go_time < py_time:
        factor = py_time / go_time
        print(f"   🔥 Native is {factor:.2f}x FASTER!")
    else:
        print(f"   ⚠️ Native is slower (due to ctypes memory array marshalling overhead).")

if __name__ == '__main__':
    benchmark()
