# FlashInfer trtllm_fp8_block_scale_moe Call Chain

## Overview

```
Python API → PyTorch Custom Op → TVM-FFI C++ Launcher → CUDA Kernels
```

## Detailed Call Chain

### 1. Python API Entry Point
**File:** `flashinfer/fused_moe/core.py:2284`
```python
@flashinfer_api
def trtllm_fp8_block_scale_moe(routing_logits, routing_bias, hidden_states, ...)
```
- Creates output tensor
- Calls `get_trtllm_moe_sm100_module()`

### 2. JIT Module Loading
**File:** `flashinfer/fused_moe/core.py:930`
```python
@functools.cache
def get_trtllm_moe_sm100_module()
```
- Compiles C++ sources via `gen_trtllm_gen_fused_moe_sm100_module()` (`flashinfer/jit/fused_moe.py:212`)
- Caches compiled module

### 3. PyTorch Custom Op
**File:** `flashinfer/fused_moe/core.py:1547`
```python
@register_custom_op("flashinfer::trtllm_fp8_block_scale_moe")
def trtllm_fp8_block_scale_moe_op(...)
```
- Uses AutoTuner to select optimal kernel tactic
- Calls TVM-FFI module

### 4. C++ Launcher
**File:** `csrc/trtllm_fused_moe_kernel_launcher.cu:1638`
```cpp
Tensor trtllm_fp8_block_scale_moe(...)
```
- Creates `Fp8BlockScaleLauncher` (line 708)
- Selects tile_N size from {8, 16, 32, 64, 128}

### 5. Kernel Execution
**File:** `csrc/trtllm_fused_moe_kernel_launcher.cu:942-982` (`Fp8BlockScaleLauncher::run()`)

Executes in sequence:
1. **Routing kernel** — DeepSeek no-aux routing (`csrc/trtllm_fused_moe_routing_deepseek.cu`)
2. **FP8 GEMM1** — hidden → intermediate (uses `wgmma.f32.e4m3.e4m3` tensor core instructions)
3. **SwiGLU activation**
4. **FP8 GEMM2** — intermediate → hidden
5. **Finalization** — weighted accumulation of expert outputs

## Routing Method Types

```python
class RoutingMethodType(IntEnum):
    Default = 0        # Softmax → TopK
    Renormalize = 1    # TopK → Softmax
    DeepSeekV3 = 2     # Sigmoid → Bias → Group top-2 → TopK groups → Global top-K
    Llama4 = 3
```

DeepSeek MoE uses `routing_method_type=2`.

## FP8 Precision Details

- Hardware: FP8 E4M3 tensor core GEMM via PTX `wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3`
- Post-GEMM: `final_accum += scale_a * scale_b * accum` (block scales applied after GEMM)
- Output: FP32 → BF16 conversion via STSM
- Block size: 128 elements

### Key PTX file
**File:** `csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh`
- Line 329-333: WGMMA execution
- Line 343-354: Post-GEMM scale multiplication
- Line 370-375: FP32 → BF16 conversion

### FP8 MMA selector
**File:** `csrc/nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh:920-935`
- Maps BLOCK_N to `SM90_64xNx32_F32E4M3E4M3_SS` structs
