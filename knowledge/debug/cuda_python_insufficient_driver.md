# cuda-python Version Mismatch → cudaErrorInsufficientDriver

## Date: 2026-02-03

## Symptom

All CuTeDSL kernel tests fail with:
```
DSLCudaRuntimeError: cudaErrorInsufficientDriver (error code: 35)
```

The error occurs at `cuda_runtime.cudaGetDeviceCount()` inside
`nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cuda_jit_executor.py`.

Additional error context:
- "CUDA_TOOLKIT_PATH: not set" (or set, doesn't matter)
- "Target SM ARCH: not set"
- "Target SM ARCH unknown is not compatible"
- GPU correctly detected as "Blackwell (sm_100a)"

CPU-only tests pass fine. Both v1 and v3 CuTeDSL kernels fail identically.

## Root Cause

`cuda-python` package version too new for the installed GPU driver.

- **Driver**: 570.172.08 (supports CUDA 12.8)
- **cuda-python installed**: 13.1.1 (bundles CUDA 13.1 runtime stubs)
- The `cuda-python 13.x` runtime requires a driver that supports CUDA 13.x, which driver 570.x does not.

## Fix

```bash
pip install "cuda-python>=12.8,<13"
```

This pins cuda-python to 12.8.x which matches the driver's CUDA 12.8 support.

## Environment Details

- Machine: Docker container with NVIDIA B200 GPU
- Driver: 570.172.08
- CUDA toolkit: 12.8 (V12.8.93)
- nvidia-cutlass-dsl: 4.3.5
- Python: 3.12

## Misleading Error Messages

The error message "Target SM ARCH: not set" and suggestions about `CUTLASS_TARGET_ARCH` are **red herrings**. Setting env vars like `CUDA_HOME`, `CUDA_TOOLKIT_PATH`, or `CUTLASS_TARGET_ARCH=sm_100a` does NOT fix the issue. The real problem is the cuda-python version mismatch — the CUDA runtime initialization fails before any arch detection logic runs.

## Prevention

Added `pip install "cuda-python>=12.8,<13"` to `setup.sh` after the main package install step.
