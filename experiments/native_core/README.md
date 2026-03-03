# 🚀 Phase 9: Native Core Benchmarking Sandbox

This is an isolated experiment to evaluate the speed multiplier of rewriting the Grappling AI kinematic physics engines (currently constrained by Python and NumPy) in a compiled language (Golang/C).

## The Experiment
We ported the `calculate_fast_kuzushi` algorithm (which calculates the center of gravity, polygon base, and violent depth drops from 17-point YOLO vectors) into raw Golang `float64` pointers.

We then compiled those pointers into a Shared C Library (`.so`) using CGO, and imported it back into Python using `ctypes`.

## Benchmarking Results
We raced the pure Python `calculate_fast_kuzushi` versus the Golang `FastKuzushi` across **100,000 dynamically generated YOLO tensors**. 

* **Python execution time:** ~0.7298 seconds
* **Golang execution time:** ~0.2597 seconds
* **Result:** **🔥 Native is ~2.81x FASTER!**

*Note: This speed multiplier includes the heavy overhead of Python `ctypes` memory marshalling! If the entire tracking loop were written natively, the speedup would realistically exceed 10x.*

## How to Build & Run
If you modify `kuzushi.go`, you must recompile the CGO Shared Library.

1. **Compile the Native Engine:**
```bash
go build -buildmode=c-shared -o libkuzushi.so kuzushi.go
```

2. **Execute the Python Benchmark Racer:**
```bash
python3 benchmark_native.py
```
