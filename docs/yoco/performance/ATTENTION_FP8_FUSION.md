# YOCO Fast attention FP8 fusion

This extends the [FA4 FP8 path](FA4_FP8.md) on `fhb-dev-9-18`. It changes vLLM
Fast inference only. The llm-train implementation is unchanged.

## Boundaries

| Stage | Computation and stored output |
| --- | --- |
| QKV / Q projection | Existing FP8 GEMM, BF16 accumulator output |
| Self QK RMSClip + RoPE | FP32 statistics/rotary arithmetic; original BF16 rounding in registers; directly stores E4M3 Q/K and quantizes V in the same launch |
| Cross Q RMSClip | FP32 statistics; directly stores E4M3 using that reader's Q scale |
| Shared NoPE KV | K RMSClip and V quantization in one launch, using the first cross layer's K/V scales |
| Cache write | Scatters prequantized bytes, with no second scaling; supports strided unified KV pools and padded slots |
| FA4 | E4M3 Q/K/V, existing softmax accumulation, BF16 attention output |
| Differential combine | FP32 gates/subtraction, original BF16 rounding in registers, directly stores E4M3 plus per-token group-128 packed UE8M0 scales |
| O projection | Existing DeepGEMM consumes those activations/scales without a standalone quantizer; BF16 output |
| Residual stream | Existing FP32 add/storage |

Attention uses its existing scalar Q/K/V scales; O projection uses dynamic
per-token group-128 power-of-two scales. These are distinct representations.
Keeping BF16 rounding in registers preserves the original precision boundaries
without writing a BF16 tensor between the producer and consumer.

The FP8 query entry explicitly requires BF16 output and fixed layer scales.
Shared readers continue to obtain K/V descales from the cache writer. The
existing rejection of dynamic scale calculation with early shared-KV writes
remains in place.

FP8 stores can change Triton's chosen elements per thread and therefore the RMS
reduction order. The cross-query launch keeps the original eight elements per
lane, including large prefill shapes. Shared-K normalization preserves the
existing eager versus Inductor pre-affine rounding behavior.

## Selection

Enabled by default for supported B200 Fast inference. The input fusion requires
FA4, E4M3 KV cache, fixed scalar scales, BF16 projections, and the implemented
RMSClip/head-dimension-128 geometry. Unsupported input boundaries retain the
existing quantization path.

Output fusion additionally requires the actual online block-FP8 DeepGEMM method,
packed UE8M0 activation scales, differential attention v3, and bias-free TP1.
Ignored or unquantized O projections retain their original method. TP greater
than one retains RowParallelLinear and its reduction semantics.

```bash
--fast --dtype bfloat16 --quantization fp8_per_block \
--kv-cache-dtype fp8 \
--attention-config.backend FLASH_ATTN \
--attention-config.flash_attn_version 4
```

Set `VLLM_YOCO_FP8_ATTENTION_FUSION=0` before starting the engine to reproduce the
unfused FP8 attention baseline. Changing it requires engine/graph reconstruction.

## Validation

The isolated `yoco-fp8-dev` holder retains its two B200 GPUs. Native Docker
DeepGEMM is used without replacement. Production services are untouched.

The 105 targeted checks pass: fused producer bytes/scales versus the existing
operators; zero/tail token counts; cancellation and tiny differential outputs;
compiled shared-key normalization; strided cache pools; masked graph slots;
scale updates during graph replay; query output dtype and bypass validation;
mode, TP and quantization eligibility; and the existing FA4 tests.

Full-model A/B uses the L3 checkpoint `0000-28000-hf`, Fast block-FP8
projections (including latent), FA4 with E4M3 cache and fixed scale 1. The only
variant is the attention-fusion switch. Both variants use the same fixed text
and 1,024 target positions at batch 1 with raw log-probabilities.

| Quality check | Unfused | Fused |
| --- | ---: | ---: |
| Mean NLL | 1.446602718533466 | 1.446602718533466 |
| Maximum absolute per-position log-probability difference | 0 | 0 |
| Changed positions | 0 | 0 |

The unfused variant also exactly reproduces the preceding FA4 FP8 baseline.
All 20 physical attention modules enable both supported fusions in the candidate;
its shared-KV graph buffers are E4M3. This establishes equivalence on the tested
text, rather than a general quality guarantee across workloads.

End-to-end timing is measured separately on one physical B200 after warmup.

## Operator timing

Same B200, CUDA Graph replay, median of three 200 ms measurements. Values are
unfused → fused in microseconds. Self inputs include the cache write.

| Tokens | Self inputs | Cross Q | Differential output |
| ---: | ---: | ---: | ---: |
| 1 | 5.88 → 2.93 | 3.19 → 1.38 | 2.66 → 1.38 |
| 8 | 6.43 → 3.46 | 3.49 → 1.48 | 3.04 → 1.49 |
| 32 | 6.90 → 3.76 | 3.66 → 1.56 | 3.18 → 1.58 |
| 128 | 8.79 → 5.40 | 4.09 → 1.99 | 3.55 → 2.04 |
| 1024 | 33.93 → 22.33 | 9.61 → 6.40 | 6.72 → 6.91 |

The 1024-token differential-output path is slightly slower in this measurement;
this fusion primarily targets decode and small batches. Operator gains do not
by themselves establish an end-to-end throughput improvement.

## End-to-end timing

One physical B200, sequential variants; 512 warm-prefix tokens and 128 generated
tokens, five warmups and seven timed repetitions per batch. Both variants retain
FA4 FP8 and latent FP8. This comparison measures the fusion change.

| Batch | Unfused output tok/s | Fused output tok/s | Observed change |
| ---: | ---: | ---: | ---: |
| 1 | 152.83 | 158.52 | +3.72% |
| 8 | 733.91 | 808.30 | +10.14% |

Batch 1 is stable: unfused runs take 0.8371–0.8381 seconds, fused runs take
0.8067–0.8087 seconds, and generated token sequences match.

Batch 8 fluctuates: unfused 1.2931–1.5409 seconds, fused 1.2640–1.4411 seconds.
The recorded generated token sequences differ. Ordinary offline `generate()`
submits requests individually while the engine runs, so the requested batch size
does not alone fix the actual scheduling groups. Both variants also changed
outputs across repeated ordinary batch-8 runs. The +10.14% figure is an observed
median in this setup, not a stable attribution to fusion.

## Controlled batch-8 precision

The validation script queues all eight requests while the isolated scheduler is
paused (`keep`, without clearing caches), then resumes it. Eight equal-length
streams use 128 teacher-forced steps each: 1,024 fixed targets, one warmup pass
and two scored passes. Each variant reproduces all three passes exactly.

| Check | Result |
| --- | ---: |
| Unfused NLL | 1.497806329604936 |
| Fused NLL | 1.497553201592945 |
| NLL difference | -0.000253128012 |
| Changed log-probability positions | 14 / 1,024 |
| Mean absolute log-probability difference | 0.0004862773 |
| Maximum absolute log-probability difference | 0.1923406 |

The controlled batched scores are repeatable but not bitwise equivalent between
variants. This sample shows no aggregate NLL regression; it does not establish
quality improvement. With the same admission barrier, all seven post-warmup
generation runs are identical within and between variants, across all eight
streams. The barrier test uses both GPUs for numerical validation and does not
supply a replacement same-card batch-8 speed result.

Full measurements, source hashes and validation conditions are in
[ATTENTION_FP8_FUSION.json](ATTENTION_FP8_FUSION.json). Raw logs, per-position
scores, test XML and task-only patches are retained in
`yoco_results/attention-fusion-fp8-20260914-5r1lj1fp` in the development workspace.
