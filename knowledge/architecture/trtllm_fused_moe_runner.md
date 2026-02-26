# trtllm_fused_moe_runner.cu 架构分析

**Date:** 2026-02-02
**Source:** `projects/flashinfer/csrc/trtllm_fused_moe_runner.cu` + `trtllm_fused_moe_kernel_launcher.cu`
**Focus:** FP8 block-scale DeepSeek-V3 MoE (`Fp8BlockScaleLauncher`)

## 文件结构

| 文件 | 作用 |
|------|------|
| `trtllm_fused_moe_kernel_launcher.cu` | Python↔C++ 入口，参数校验，workspace 分配，TVM-FFI 导出 |
| `trtllm_fused_moe_runner.cu` | MoE pipeline runner：Routing、PermuteGemm1、Gemm2、Activation、Finalize |
| `trtllm_fused_moe_dev_kernel.cu` | Device kernel 实现（activation、finalize、convert_sf） |
| `trtllm_fused_moe_routing_deepseek.cu` | DeepSeek-V3 routing kernel |
| `trtllm_batched_gemm_runner.cu` | trtllm cubin GEMM runner（加载并调用预编译 cubin） |

## flashinfer_moe Pipeline 总览

flashinfer_moe **不调用** `group_gemm_fp8_nt_groupwise`。它调用的是 trtllm 的 **fused MoE runner** (`trtllm_fused_moe_runner.cu`)，一个完整的 C++ pipeline：

| Stage | CUDA Kernel | 说明 |
|-------|-------------|------|
| Routing | `routingMainKernel` | Sigmoid + bias + grouped top-k |
| Index | `routingIndicesClusterKernel`/`CoopKernel` | Token→expert mapping |
| GEMM1 | `bmm_E4m3_E4m3E4m3_Fp32_...` (trtllm cubin) | FP8 in → **FP8 out**, gate+up proj |
| SwiGLU | `activationDeepSeekKernel` | Dequant → SwiGLU → FP8 requant |
| GEMM2 | `bmm_Bfloat16_E4m3E4m3_Fp32_...` (trtllm cubin) | FP8 in → **BF16 out** |
| Finalize | `finalizeKernel`/`VecLoad` | Gather back + weighted reduce |

## 完整 Pipeline 调用链

```
Python: trtllm_fp8_block_scale_moe(...)
  ↓ TVM-FFI
C++: trtllm_fp8_block_scale_moe()           [kernel_launcher.cu:1638]
  ↓ 创建 Fp8BlockScaleLauncher
  ↓ launcher->run()                          [kernel_launcher.cu:942]
    ↓
    1. check_routing() + prepare_routing()   → 分配 routing workspace
    2. Routing::Runner::run()                → routingMainKernel + routingIndicesKernel
    3. check_moe() + prepare_moe()           → 分配 GEMM workspace
    4. MoE::Runner::run()                    [runner.cu:568]
       ↓
       4a. PermuteGemm1::Runner::run()       → trtllm cubin GEMM1
       4b. activation::run()                 → activationDeepSeekKernel (dequant→SwiGLU→requant)
       4c. Gemm2::Runner::run()              → trtllm cubin GEMM2
       4d. finalize::run()                   → finalizeKernel
```

## Pipeline 各阶段详解

### Stage 1: Routing

**入口**: `Routing::Runner::run()` [runner.cu:52]

**DeepSeek-V3 routing** (`RoutingMethodType::DeepSeekV3`):
- 调用 `moe::dev::routing::routingDeepSeek::run(routingData, stream)`
- CUDA kernels: `routingMainKernel`, `routingIndicesClusterKernel`/`routingIndicesCoopKernel`

**Routing 输出**:
| Buffer | Shape | 说明 |
|--------|-------|------|
| `routing_expert_indexes` | [T, top_k] | 每 token 选中的 expert ID |
| `expert_weights` | [T, top_k] | BF16 routing weights（sigmoid 输出） |
| `expert_count_histogram` | [E*2] | 每 expert 被路由到的 token 数 |
| `total_num_padded_tokens` | [1] | padded 后总 token 数 |
| `expanded_idx_to_permuted_idx` | [T*top_k] | expanded token idx → permuted idx 映射 |
| `permuted_idx_to_token_idx` | [max_padded] | permuted idx → 原始 token idx |
| `cta_idx_xy_to_batch_idx` | [max_ctas] | CTA → expert batch 映射（供 GEMM dispatch） |
| `cta_idx_xy_to_mn_limit` | [max_ctas] | CTA → M/N 维度限制 |
| `num_non_exiting_ctas` | [1] | 有效 CTA 数 |

**关键设计**: Routing 不仅计算 token→expert 映射，还直接生成 GEMM CTA dispatch table，避免了单独的 scatter/shuffle 步骤。

### Stage 2: PermuteGemm1 (Scatter + GEMM1)

**入口**: `PermuteGemm1::Runner::run()` [runner.cu:283]

**Runner 选项** [runner.cu:224]:
```cpp
TrtllmGenBatchedGemmRunnerOptions options = {
    .dtypeA = dtypeWeights,      // E4m3 (注意 A/B swap)
    .dtypeB = dtypeAct,          // E4m3
    .dtypeC = dtypeAct,          // E4m3 (GEMM1 输出 FP8!)
    .actType = SwiGlu,           // DeepSeek FP8 不在 GEMM1 中 fuse
    .deepSeekFp8 = true,
    .fusedAct = false,           // DeepSeek FP8: 不 fuse activation
    .routeAct = true,            // GEMM 内部做 scatter（route token to expert）
    .transposeMmaOutput = true,  // A/B swap 后需要 transpose
    .tileSize = tileTokensDim,   // 8/16/32/64/128
    .epilogueTileM = 64,         // DeepSeek FP8 用 64（非 FP8 用 128）
};
```

**A/B swap 说明**: `transposeMmaOutput = true` 意味着实际 MMA 计算是 `C' = B^T @ A^T`，然后 transpose 得到 `C = A @ B`。所以 `dtypeA = weights`, `dtypeB = activations`。

**GEMM1 维度** [runner.cu:295]:
```cpp
mRunner.run(
    numTokens,
    intermediateSizeFactor * intermediateSize,  // N = 2 * 2048 = 4096 (gate+up)
    hiddenSize,                                  // K = 7168
    ...
);
```

**输入**:
- `hiddenState`: [T, 7168] FP8_E4M3 (原始 hidden_states，未 scatter)
- `hiddenStateScale`: [56, T] float32 (hidden_states block scale，MN-major)
- `weights`: [32, 4096, 7168] FP8_E4M3 (gemm1_weights)
- `weightsScale`: [32, 32, 56] float32 (gemm1_weights_scale)
- `permutedIdxToTokenIdx`: routing 输出的 permutation 索引

**输出**:
- `gemm1_output`: [max_padded, 4096] FP8_E4M3 (uint8 存储)
- `gemm1_output_scale`: [32, max_padded] float32 (32 = 4096/128)

**关键**: GEMM1 内置 scatter（`routeAct = true`），用 `permutedIdxToTokenIdx` 在 GEMM 内部把 token 路由到对应 expert 的位置。不需要单独的 `shuffleRowsKernel`。

### Stage 3: Activation (SwiGLU + FP8 Requant)

**条件**: 仅在 `mDtypeElt == E4m3 && mUseDeepSeekFp8` 时执行 [runner.cu:596]

**入口**: `moe::dev::activation::run(activationData, stream)` [runner.cu:598]

**Kernel**: `activationDeepSeekKernel`

**Data** [runner.cu:476]:
```cpp
activationData.mDtypeElt = E4m3;
activationData.mUseDeepSeekFp8 = true;
activationData.inPtr = workspace.gemm1_output;           // [padded, 4096] FP8
activationData.outPtr = workspace.activation_output;     // [padded, 2048] FP8
activationData.inDqSfsPtr = workspace.gemm1_output_scale;    // [32, padded] f32
activationData.outDqSfsPtr = workspace.activation_output_scale; // [16, padded] f32
activationData.innerDim = 4096;  // 2 * intermediate_size
```

**计算**:
1. Dequant GEMM1 output: `x_f32 = x_fp8 * scale`
2. Split: `gate = x[:, :2048]`, `up = x[:, 2048:]`
3. SwiGLU: `y = silu(gate) * up`
4. Requant to FP8: `y_fp8 = quantize_fp8(y)`, 生成新的 per-block scale

### Stage 4: GEMM2

**入口**: `Gemm2::Runner::run()` [runner.cu:378]

**Runner 选项** [runner.cu:349]:
```cpp
TrtllmGenBatchedGemmRunnerOptions options = {
    .dtypeA = dtypeWeights,  // E4m3
    .dtypeB = dtypeAct,      // E4m3
    .dtypeC = dtypeOut,      // Bfloat16 (GEMM2 输出 BF16!)
    .deepSeekFp8 = true,
    .fusedAct = false,
    .routeAct = false,       // GEMM2 不做 scatter（已经在 permuted order）
    .transposeMmaOutput = true,
    .epilogueTileM = 64,     // DeepSeek FP8 用 64
};
```

**GEMM2 维度** [runner.cu:387]:
```cpp
mRunner.run(
    numTokens,
    hiddenSize,           // N = 7168
    intermediateSize,     // K = 2048
    ...
);
```

**输入**:
- `permutedHiddenState`: activation_output [padded, 2048] FP8
- `permutedHiddenStateScale`: activation_output_scale [16, padded] f32
- `weights`: [32, 7168, 2048] FP8 (gemm2_weights)
- `weightsScale`: [32, 56, 16] f32 (gemm2_weights_scale)

**输出**:
- `gemm2_output`: [padded, 7168] BF16
- `gemm2_output_scale`: nullptr (BF16 输出不需要 scale)

**关键区别**: GEMM2 的 `routeAct = false`，因为数据已经在 permuted order（按 expert 分组）。

### Stage 5: Finalize (Gather + Weighted Reduce)

**入口**: `moe::dev::finalize::run(finalizeData, stream)` [runner.cu:615]

**Kernel**: `finalizeKernel` / `finalizeKernelVecLoad`

**Data** [runner.cu:492]:
```cpp
finalizeData.mDtypeElt = Bfloat16;       // 输入输出都是 BF16
finalizeData.mDtypeExpW = Bfloat16;      // routing weights 类型
finalizeData.mUseDeepSeekFp8 = false;    // finalize 不涉及 FP8
finalizeData.inPtr = workspace.gemm2_output;    // [padded, 7168] BF16
finalizeData.outPtr = args.output;               // [T, 7168] BF16
finalizeData.expertWeightsPtr = workspace.expert_weights;  // [T, top_k] BF16
finalizeData.expandedIdxToPermutedIdx = ...;     // index mapping
finalizeData.hiddenDim = hidden_size;    // 7168
```

**计算**:
```
output[t, :] = Σ_{k=0}^{top_k-1} expert_weights[t, k] * gemm2_output[permuted_idx[t*top_k + k], :]
```

## Workspace 分配 (Fp8BlockScaleLauncher)

```
prepare_moe() 分配的 workspace:
├── gemm1_output:           [max_padded_gemm1, 4096]  uint8 (FP8)
├── gemm1_output_scale:     [32, max_padded]          float32
├── activation_output:      [max_padded_gemm1, 2048]  uint8 (FP8)
├── activation_output_scale: [16, max_padded_gemm1]   float32
├── gemm2_output:           [max_padded_gemm2, 7168]  bfloat16
├── output:                 [T, 7168]                 bfloat16
├── workspace_fc1:          [fc1_workspace_bytes]     int8 (GEMM scratch)
└── workspace_fc2:          [fc2_workspace_bytes]     int8 (GEMM scratch)
```

## trtllm cubin GEMM 加载机制

cubin 文件从远程 cache 下载到本地 (`TRTLLM_GEN_GEMM_CUBIN_PATH`)：
1. `gen_trtllm_gen_fused_moe_sm100_module()` [jit/fused_moe.py:212] 下载 `flashinferMetaInfo.h` 和 cubin checksums
2. `setup_cubin_loader()` [core.py:933] 初始化 cubin 加载器
3. `TrtllmGenBatchedGemmRunner` 根据 options (dtype, tileSize, etc.) 从 cubin 目录选择并加载对应的预编译 kernel

**可用 tile sizes**: {8, 16, 32, 64, 128}

**Tactic selection**: `MoEConfig{gemm1Config, gemm2Config}` — 每个 tactic 是 GEMM1 和 GEMM2 的 cubin kernel 配置组合。AutoTuner 选择最优组合。

## 与 triton_v2 的架构差异

| 方面 | trtllm fused MoE | triton_v2 |
|------|-------------------|-----------|
| Scatter | Fused into GEMM1 (`routeAct=true`) | 独立 `shuffleRowsKernel` |
| GEMM1 output dtype | FP8_E4M3 | BF16 (sgl_kernel cutlass) |
| SwiGLU | Separate kernel (not fused with GEMM) | Triton kernel (fused SwiGLU+quant) |
| GEMM kernel | trtllm cubin (pre-compiled, SM100 optimized) | cutlass `GemmUniversal<GroupProblemShape>` |
| Gather | `finalizeKernel` (weighted reduce) | `apply_shuffle_mul_sum_kernel` |
| CTA dispatch | Routing 直接生成 CTA table | Python-level expert offsets |
| kernel launches | ~6 kernels | ~10+ kernels |

## Scale Layout 确认

### hidden_states_scale
- Shape: `[K//128, T]` = `[56, T]` (MN-major)
- 校验 [kernel_launcher.cu:848-851]:
  ```cpp
  hidden_states_scale.size(0) == hidden_states.size(1) / 128  // 56
  hidden_states_scale.size(1) == num_tokens                    // T
  ```

### gemm1_weights_scale
- Shape: `[E, 2I//128, K//128]` = `[32, 32, 56]`
- 校验 [kernel_launcher.cu:863-866]:
  ```cpp
  gemm1_weights_scale.size(0) == local_num_experts     // 32
  gemm1_weights_scale.size(1) == 2 * intermediate_size / 128  // 32
  gemm1_weights_scale.size(2) == hidden_size / 128     // 56
  ```

### gemm2_weights_scale
- Shape: `[E, H//128, I//128]` = `[32, 56, 16]`
- 校验 [kernel_launcher.cu:873-876]:
  ```cpp
  gemm2_weights_scale.size(0) == local_num_experts     // 32
  gemm2_weights_scale.size(1) == hidden_size / 128     // 56
  gemm2_weights_scale.size(2) == intermediate_size / 128  // 16
  ```

### GEMM1 输出 scale
- Shape: `[2I//128, max_padded]` = `[32, max_padded]`
- 分配 [kernel_launcher.cu:899-901]:
  ```cpp
  alloc_tensor({2 * intermediate_size / 128, total_max_padded_tokens}, dl_float32, ...)
  ```

### Activation 输出 scale
- Shape: `[I//128, max_padded]` = `[16, max_padded]`
- 分配 [kernel_launcher.cu:905-907]

**所有 scale 都是 MN-major**：第一维是 N/K 的 block 数，第二维是 M（token 数）。

## 对我们 CuTe DSL 实现的启示

1. **不需要单独的 scatter kernel**: trtllm 通过 `routeAct=true` 把 scatter fused 进 GEMM1。如果我们替换 GEMM，需要自己处理 scatter，或者保留原来的 scatter 逻辑。

2. **GEMM1 输出 FP8**: trtllm GEMM1 输出 FP8（不是 BF16），然后 activation kernel 做 dequant→SwiGLU→requant。这减少了内存带宽。

3. **Scale 全部 MN-major float32**: 所有 scale tensor 的 layout 是 `[feature_blocks, token_dim]`，类型 float32。CuTe DSL 目前只支持 FP8/UE8M0 scale with sf_vec_size=32，需要修改。

4. **`transposeMmaOutput = true` + A/B swap**: cubin 内部用 `C' = W^T @ A^T` 计算，然后 transpose 输出。等价于 `C = A @ W`。CuTe DSL 的 NT layout (`grouped_gemm_nt_masked`) 本身就是这种 pattern。

5. **Tile size = batch dimension tiling**: tileTokensDim 控制每个 expert 的 M 维度 tiling。对应 CuTe DSL 的 masked_m 参数。
