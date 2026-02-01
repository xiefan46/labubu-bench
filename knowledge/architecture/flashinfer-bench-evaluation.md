# FlashInfer-Bench Evaluation Pipeline

## Core Abstractions

```
Definition → Solution → Workload → Trace
```

- **Definition**: kernel interface spec (inputs, outputs, axes, reference implementation)
- **Solution**: concrete implementation (Python/Triton/CUDA)
- **Workload**: specific input configuration
- **Trace**: execution record with correctness and performance data

## Evaluator Selection

**File:** `flashinfer_bench/bench/evaluators/registry.py`

```python
_EVALUATORS = [SamplingEvaluator, LowBitEvaluator]
_DEFAULT_EVALUATOR = DefaultEvaluator
```

Selection logic (`resolve_evaluator`):
1. Try each evaluator's `can_evaluate(definition)` method
2. If exactly one matches → use it
3. If none match → use `DefaultEvaluator`
4. If multiple match → raise error

### LowBitEvaluator
**File:** `flashinfer_bench/bench/evaluators/lowbit.py`
- Matches: `"moe_fp8_block_scale" in definition.name`
- Default `required_matched_ratio`: 0.95 (hardcoded when cfg is None)

### DefaultEvaluator
- Fallback for all other definitions
- Uses `compute_error_stats` in `utils.py`
- When `required_matched_ratio` is None, defaults to 1.0 (100% match required)

## Tolerance Check Logic

**File:** `flashinfer_bench/bench/utils.py:91-118`

```python
def compute_error_stats(output, reference, cfg):
    abs_error = |x - y|
    rel_error = abs_error / (|y| + 1e-8)
    exceeds_tol_mask = (abs_error > cfg.atol) & (rel_error > cfg.rtol)  # AND logic
    matched_ratio = 1.0 - (exceeds_count / total_elements)
    exceeds_tol = matched_ratio < required_matched_ratio
```

Note: This is different from `torch.isclose` which uses `|a-b| <= atol + rtol*|b|`.

## BenchmarkConfig Defaults

**File:** `flashinfer_bench/bench/config.py`

| Parameter | Default | CLI flag |
|---|---|---|
| atol | 0.01 | `--atol` |
| rtol | 0.01 | `--rtol` |
| required_matched_ratio | None | `--required-matched-ratio` (our addition) |
| timeout_seconds | 300 | `--timeout` |
| warmup_runs | 10 | `--warmup-runs` |
| iterations | 50 | `--iterations` |
| num_trials | 3 | `--num-trials` |

## CLI Entry Point

**File:** `flashinfer_bench/cli/main.py:178-196`

```python
def run(args):
    config = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        ...
        required_matched_ratio=args.required_matched_ratio,  # our addition
    )
```
