# YOCO Fast: FA4 FP8 on B200

This change connects the existing FA4 FP8 forward implementation to vLLM's
quantized-query and paged-KV paths. It is opt-in through `--kv-cache-dtype fp8`;
the default KV-cache precision remains unchanged. Training code is unchanged.

Use the existing YOCO model and serving settings with:

```bash
--fast --dtype bfloat16 --quantization fp8_per_block --kv-cache-dtype fp8 \
--attention-config.backend FLASH_ATTN --attention-config.flash_attn_version 4
```

Q/K/V passed to FA4 use E4M3. QK and PV accumulate in FP32, softmax statistics
remain FP32, and attention outputs remain BF16. The main YOCO residual remains
FP32. This is not an FP8 output-epilogue or whole-pipeline fusion change.

## Runtime fixes

The separately installed FA4 CuTeDSL code needs two compatibility fixes in the
tested runtime. `fa4_compat.py` applies them when loading the forward function:

- Recognize the newer concrete `MmaF8F6F4Op` type while retaining the existing
  handling of BF16 and block-scaled MMA types.
- Bound delayed softmax rescaling for FP8 probabilities. With an exponent
  offset of 8, probabilities are multiplied by 256. The previous 4-bit delay
  permitted a value up to 4096, exceeding E4M3's finite maximum of 448. FP32
  normalization could remain correct while the probability cast saturated.
  Capping the delay at 0.5 bit bounds the peak by `256 * sqrt(2) < 448`.

The vLLM adapter now forwards Q/K/V descales for FP8 inputs, while BF16 calls
retain their existing argument behavior. FP8 capability checking uses the
selected FA version, SM100 capability, and the installed FA4 descale interface.
FA3 remains an SM90/Hopper backend and is not a fallback on these B200 GPUs.

## YOCO cache and dispatch

- FP8 queries stay on FA4 instead of entering the older YOCO Triton decode
  shortcut, which does not apply the query descale.
- Shared-KV readers use their cache writer's K/V scales, including chained
  sharing. Their Q scale remains their own. Different writer/reader cache
  dtypes are rejected.
- `kv_sharing_fast_prefill` requires fixed KV scales. Its self-decoder writes
  shared KV before a cross-layer forward could calculate scales; changing the
  scale after that write would reinterpret cached bytes. Combining it with
  `calculate_kv_scales=True` is rejected. Disable one of those two options.
- The initial model A/B uses the default fixed scale of 1. A calibrated scale
  policy is a separate numerical choice.
- YOCO Align does not enable FP8 KV cache.

## Validation

Validation uses only the existing `bonete01/yoco-fp8-dev` holder with 2 B200.
The Docker-native DeepGEMM files are preserved. Tests cover non-unit descales,
paged GQA with mixed sequence lengths, a short KV tail that reproduced E4M3
saturation, sliding windows, Split-KV, BF16 regression, and CUDA Graph replay.

50 distinct regression tests passed: 40 metadata/configuration checks and 10
actual B200 attention tests. The
full-model TP1 A/B used L3 step28000 with all existing Fast-FP8 projections
(including latent in/out) enabled in both variants. The only precision change
between variants was attention Q/K/V and KV cache. Observed FA4 calls in the
candidate used E4M3 for all three operands; the actual cache tensors were uint8
storage interpreted as E4M3.

| Fixed-text quality probe | BF16 attention | FP8 attention |
| --- | ---: | ---: |
| Mean NLL over 1,024 positions | 1.442959144 | 1.446602719 |

The NLL increase is 0.003643575; `exp(mean NLL)` increases by 0.365%. This is a
fixed WikiText-2 validation prefix sampled every four tokens, not a full-dataset
perplexity or task-accuracy result. The unchanged BF16-attention variant was
also compared against the preceding latent-FP8 validation: all 1,024 log-probs
were identical.

Sequential timing on the same B200 used a warm 512-token prefix, 128 output
tokens, five warmups and seven timed repetitions:

| Batch | BF16 attention (token/s) | FP8 attention (token/s) | Median change |
| --- | ---: | ---: | ---: |
| 1 | 161.25 | 152.68 | -5.31% |
| 8 | 816.16 | 788.15 | -3.43% |

Batch 1 timings were stable: 0.7930–0.7945 s versus 0.8377–0.8387 s.
Batch 8 varied more substantially (1.2486–1.3447 s versus 1.2967–1.4789 s),
so its small median difference is not a robust performance conclusion. These
short-context measurements do not establish a throughput gain. Query
quantization and scale preparation remain separate from their producers.

The first two-GPU timing is retained as preliminary evidence; use
`STEADY_BENCH_SUMMARY.json` for the same-GPU measurements. Tests have finished
with the holder IDLE; the 2 B200 allocation is retained. No changes were
committed, pushed or deployed to the existing inference services.

Evidence directory: `yoco_results/fa4-fp8-20260914-p147q0qb/`. `TASK.patch` and
`MANIFEST.json` isolate this task from pre-existing work; `gpu-tests.xml`,
`MODEL_SUMMARY.json`, `BF16_REGRESSION.json` and the full logs retain the results.
