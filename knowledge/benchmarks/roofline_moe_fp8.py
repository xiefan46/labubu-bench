"""Roofline model analysis for DeepSeek-V3 FP8 blockwise MoE on B200.

Generates roofline plot and prints theoretical bounds for different token counts.
Run: python knowledge/benchmarks/roofline_moe_fp8.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# B200 hardware specs
# ============================================================
PEAK_FP8_TFLOPS = 4500  # FP8 dense tensor core (TFLOPS), sparse=9000
PEAK_BW_TBS = 8  # HBM3e bandwidth (TB/s)
RIDGE_POINT = PEAK_FP8_TFLOPS / PEAK_BW_TBS  # 562.5 FLOPS/byte

# ============================================================
# MoE geometry (DeepSeek-V3)
# ============================================================
E = 32  # local experts
TOP_K = 8
H = 7168  # hidden size
I = 2048  # intermediate size
BLOCK = 128  # FP8 quant block size

# ============================================================
# FLOPs calculation
# ============================================================
# GEMM1: [T*8, H] x [E, H, 2I] -> [T*8, 2I]
FLOPS_PER_TOKEN_GEMM1 = TOP_K * H * (2 * I) * 2  # multiply-add = 2
# GEMM2: [T*8, I] x [E, I, H] -> [T*8, H]
FLOPS_PER_TOKEN_GEMM2 = TOP_K * I * H * 2
FLOPS_PER_TOKEN = FLOPS_PER_TOKEN_GEMM1 + FLOPS_PER_TOKEN_GEMM2
# SwiGLU FLOPs are negligible vs GEMM

# ============================================================
# Bytes calculation
# ============================================================
# Weights (read once per call, independent of T)
W1_BYTES = E * H * (2 * I) * 1  # FP8
W2_BYTES = E * I * H * 1  # FP8
W1_SCALE = E * (H // BLOCK) * ((2 * I) // BLOCK) * 4  # FP32
W2_SCALE = E * (I // BLOCK) * (H // BLOCK) * 4  # FP32
WEIGHT_BYTES = W1_BYTES + W2_BYTES + W1_SCALE + W2_SCALE

# Activation bytes per token (read + write through pipeline)
# Scatter input: TOP_K * H * 1 (FP8)
# GEMM1 output: TOP_K * 2I * 2 (BF16)
# SwiGLU+Quant output: TOP_K * I * 1 (FP8) + scales
# GEMM2 output: TOP_K * H * 2 (BF16)
# Final output: H * 2 (BF16) -- per original token, not per top_k copy
ACT_BYTES_PER_TOKEN = (
    H * 1  # input hidden_states (FP8)
    + (H // BLOCK) * 4  # input scale
    + TOP_K * H * 1  # scattered input
    + TOP_K * (H // BLOCK) * 4  # scattered scale
    + TOP_K * 2 * I * 2  # GEMM1 output (BF16)
    + TOP_K * I * 1  # SwiGLU+Quant output (FP8)
    + TOP_K * (I // BLOCK) * 4  # quant scale
    + TOP_K * H * 2  # GEMM2 output (BF16)
    + H * 2  # final output (BF16)
)


def total_flops(T):
    return T * FLOPS_PER_TOKEN


def total_bytes(T):
    return WEIGHT_BYTES + T * ACT_BYTES_PER_TOKEN


def arithmetic_intensity(T):
    return total_flops(T) / total_bytes(T)


def theoretical_min_time_us(T):
    """Theoretical minimum time in microseconds."""
    compute_time = total_flops(T) / (PEAK_FP8_TFLOPS * 1e12) * 1e6
    memory_time = total_bytes(T) / (PEAK_BW_TBS * 1e12) * 1e6
    return max(compute_time, memory_time)


def bottleneck(T):
    ai = arithmetic_intensity(T)
    return "compute" if ai > RIDGE_POINT else "bandwidth"


# ============================================================
# Benchmark data (from 2026-02-02 runs)
# ============================================================
# (T, triton_v2_speedup, flashinfer_speedup)
benchmark_data = [
    (1, 19.89, 31.24),
    (7, 30.15, 41.32),
    (14, 31.75, 31.15),
    (15, 34.90, 32.25),
    (16, 26.44, 32.04),
    (32, 31.44, 32.25),
    (52, 30.59, 31.88),
    (53, 31.20, 39.02),
    (54, 22.16, 30.91),
    (55, 26.46, 30.06),
    (56, 30.45, 32.67),
    (57, 26.27, 31.89),
    (58, 30.49, 32.16),
    (59, 28.97, 40.08),
    (62, 28.90, 31.64),
    (80, 24.60, 29.13),
    (901, 25.16, 34.15),
    (11948, 4.74, 15.96),
    (14107, 5.04, 15.66),
]

# ============================================================
# Print table
# ============================================================
print("=" * 110)
print("DeepSeek-V3 FP8 MoE Roofline Analysis — B200 (FP8 dense: 4.5 PFLOPS, HBM: 8 TB/s)")
print("=" * 110)
print(
    f"{'T':>6} {'M_total':>8} {'FLOPs':>10} {'Bytes':>10} {'AI':>8} "
    f"{'Bottleneck':>10} {'Theory min':>11} {'triton_v2':>10} {'flashinfer':>10}"
)
print("-" * 110)

for T, tv2_speedup, fi_speedup in benchmark_data:
    M = T * TOP_K
    flops = total_flops(T)
    nbytes = total_bytes(T)
    ai = arithmetic_intensity(T)
    bn = bottleneck(T)
    tmin = theoretical_min_time_us(T)

    def fmt_flops(f):
        if f >= 1e12:
            return f"{f/1e12:.1f} TFLOP"
        elif f >= 1e9:
            return f"{f/1e9:.1f} GFLOP"
        else:
            return f"{f/1e6:.0f} MFLOP"

    def fmt_bytes(b):
        if b >= 1e9:
            return f"{b/1e9:.2f} GB"
        else:
            return f"{b/1e6:.0f} MB"

    print(
        f"{T:>6} {M:>8} {fmt_flops(flops):>10} {fmt_bytes(nbytes):>10} {ai:>8.1f} "
        f"{bn:>10} {tmin:>9.0f} μs {tv2_speedup:>8.1f}x {fi_speedup:>9.1f}x"
    )

# Crossover point
T_cross = WEIGHT_BYTES * RIDGE_POINT / (FLOPS_PER_TOKEN - ACT_BYTES_PER_TOKEN * RIDGE_POINT)
print(f"\nBandwidth→Compute crossover at T ≈ {T_cross:.0f}")
print(f"Weight bytes: {WEIGHT_BYTES/1e9:.3f} GB")
print(f"Activation bytes per token: {ACT_BYTES_PER_TOKEN/1e3:.1f} KB")
print(f"FLOPs per token: {FLOPS_PER_TOKEN/1e6:.1f} M")
print(f"Ridge point: {RIDGE_POINT:.1f} FLOPS/byte")

# ============================================================
# Plot roofline
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Classic roofline (log-log) ---
ax = axes[0]
ai_range = np.logspace(-1, 4.5, 500)
# Roofline ceiling
perf_bw = ai_range * PEAK_BW_TBS  # TFLOPS (bandwidth-limited)
perf_compute = np.full_like(ai_range, PEAK_FP8_TFLOPS)  # TFLOPS (compute-limited)
perf_roof = np.minimum(perf_bw, perf_compute)

ax.loglog(ai_range, perf_roof, "k-", linewidth=2.5, label="B200 Roofline")
ax.fill_between(
    ai_range, perf_roof * 0.001, perf_roof, alpha=0.05, color="gray"
)
ax.axvline(RIDGE_POINT, color="gray", linestyle=":", alpha=0.5)
ax.text(
    RIDGE_POINT * 1.1,
    PEAK_FP8_TFLOPS * 0.5,
    f"Ridge: {RIDGE_POINT:.0f}",
    fontsize=8,
    color="gray",
)

# Plot our solutions at different T
T_values = [1, 7, 14, 32, 55, 80, 901, 11948, 14107]
colors_tv2 = []
colors_fi = []

for T_val in T_values:
    ai = arithmetic_intensity(T_val)
    # Achieved TFLOPS = FLOPs / actual_time
    # We don't have absolute time, but we can estimate:
    # achieved_perf / peak_perf ≈ speedup / theoretical_max_speedup
    # For bandwidth-bound: achieved = ai * achieved_bw
    # We plot at the arithmetic intensity, and use the roofline to show the gap

    # Find matching benchmark entry
    match = [(t, tv2, fi) for t, tv2, fi in benchmark_data if t == T_val]
    if not match:
        continue
    _, tv2_su, fi_su = match[0]

    # Estimate achieved TFLOPS from the reference relationship:
    # Theory min time for this T
    tmin = theoretical_min_time_us(T_val)
    # If flashinfer speedup is X and it's near optimal, reference time ≈ X * tmin
    # Our time ≈ reference / tv2_speedup = (fi_speedup * tmin) / tv2_speedup
    # But this assumes flashinfer hits the roofline exactly, which it doesn't.
    # Instead, let's estimate: if flashinfer achieves ~80% of peak for bw-bound cases
    # Actually, let's just compute achieved FLOPS/s based on the ratio to roofline

    # Simpler: for the roofline chart, just plot arithmetic intensity vs the roofline
    # and annotate which T it is
    pass

# Plot T values as vertical lines on the roofline
for T_val in T_values:
    ai = arithmetic_intensity(T_val)
    bn = bottleneck(T_val)
    roof_perf = min(ai * PEAK_BW_TBS, PEAK_FP8_TFLOPS)
    ax.plot(ai, roof_perf, "o", markersize=8, color="royalblue", zorder=5)
    # Offset labels to avoid overlap
    offset_y = 1.3 if T_val not in [7, 14, 32] else 0.6
    ax.annotate(
        f"T={T_val}",
        (ai, roof_perf),
        textcoords="offset points",
        xytext=(5, 8 if offset_y > 1 else -15),
        fontsize=7,
        color="royalblue",
    )

ax.set_xlabel("Arithmetic Intensity (FLOPS/byte)", fontsize=11)
ax.set_ylabel("Achievable Performance (TFLOPS)", fontsize=11)
ax.set_title("Roofline Model — DeepSeek-V3 FP8 MoE on B200", fontsize=12)
ax.set_xlim(0.1, 30000)
ax.set_ylim(1, 10000)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

# Add bandwidth-bound / compute-bound labels
ax.text(1, 30, "Bandwidth\nBound", fontsize=9, color="red", alpha=0.7, ha="center")
ax.text(
    5000, 3000, "Compute\nBound", fontsize=9, color="green", alpha=0.7, ha="center"
)

# --- Right: Theoretical min time vs T ---
ax2 = axes[1]
T_range = np.logspace(0, 4.5, 200).astype(int)
T_range = np.unique(T_range)

theory_times = [theoretical_min_time_us(t) for t in T_range]
ax2.loglog(T_range, theory_times, "k-", linewidth=2, label="Theoretical minimum")

# Mark bandwidth-bound vs compute-bound regions
T_bw = [t for t in T_range if bottleneck(t) == "bandwidth"]
T_cp = [t for t in T_range if bottleneck(t) == "compute"]
times_bw = [theoretical_min_time_us(t) for t in T_bw]
times_cp = [theoretical_min_time_us(t) for t in T_cp]

ax2.fill_between(
    T_bw, [0.1] * len(T_bw), times_bw, alpha=0.1, color="red", label="BW-bound zone"
)
if T_cp:
    ax2.fill_between(
        T_cp,
        [0.1] * len(T_cp),
        times_cp,
        alpha=0.1,
        color="green",
        label="Compute-bound zone",
    )

# Plot benchmark data points
T_bench = [d[0] for d in benchmark_data]
tv2_speedups = [d[1] for d in benchmark_data]
fi_speedups = [d[2] for d in benchmark_data]

# Estimate actual times: use flashinfer as ~85% of roofline efficiency for bw-bound
# Reference time ≈ flashinfer_time × flashinfer_speedup
# We estimate flashinfer_time ≈ theoretical_min / 0.85 for bw-bound
# Then triton_v2_time = flashinfer_time × (flashinfer_speedup / triton_v2_speedup)
# Simpler: reference_time ≈ theoretical_min × flashinfer_speedup / efficiency
# Let's just assume reference time is proportional and estimate:

# Better approach: plot theoretical min as reference and overlay speedup-derived times
# If reference impl takes R μs, then solution takes R/speedup μs
# We don't know R, but we can estimate: flashinfer is probably ~70-90% of roofline
# Let's estimate R from flashinfer assuming 80% roofline efficiency:

est_fi_times = []
est_tv2_times = []
for T_val, tv2_su, fi_su in benchmark_data:
    tmin = theoretical_min_time_us(T_val)
    # Estimate: flashinfer achieves ~80% of peak → flashinfer_time ≈ tmin / 0.8
    fi_time = tmin / 0.8
    ref_time = fi_time * fi_su
    tv2_time = ref_time / tv2_su
    est_fi_times.append(fi_time)
    est_tv2_times.append(tv2_time)

ax2.scatter(
    T_bench,
    est_tv2_times,
    marker="s",
    s=50,
    color="darkorange",
    zorder=5,
    label="triton_fused_v2 (est.)",
)
ax2.scatter(
    T_bench,
    est_fi_times,
    marker="^",
    s=50,
    color="forestgreen",
    zorder=5,
    label="flashinfer_moe (est.)",
)

# Crossover line
ax2.axvline(T_cross, color="gray", linestyle="--", alpha=0.5)
ax2.text(T_cross * 1.2, 100, f"T={T_cross:.0f}\n(crossover)", fontsize=8, color="gray")

ax2.set_xlabel("num_tokens (T)", fontsize=11)
ax2.set_ylabel("Time (μs)", fontsize=11)
ax2.set_title("Kernel Time vs Token Count", fontsize=12)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, which="both", alpha=0.3)
ax2.set_xlim(0.8, 20000)
ax2.set_ylim(50, 50000)

plt.tight_layout()

output_path = "knowledge/benchmarks/roofline_moe_fp8_b200.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")
plt.close()
