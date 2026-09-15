# YOCO Fast BF16 residual A/B

Changing main residual storage from FP32 to BF16 did not produce a repeatable
end-to-end speedup in this experiment. The switch remains experimental and off
by default. This is inference-only work; llm-train was unchanged.

## End-to-end results

Same physical B200, TP1, current L3 checkpoint, Fast block-FP8 weights and
activations, FA4 FP8 attention/KV, attention fusion enabled. Latent Norm-to-FP8
projection fusion is disabled in both variants. Run order is FP32 A → BF16 A →
BF16 B → FP32 B. Each run uses five decode warmups and seven timed repetitions.
The table pools the 14 timings per precision and reports their median.

| Workload | FP32 residual | BF16 residual | Change |
| --- | ---: | ---: | ---: |
| Batch 1, 512 input + 128 output | 158.38 tok/s | 158.34 tok/s | -0.03% throughput |
| Batch 8, 512 input + 128 output | 812.29 tok/s | 814.23 tok/s | +0.24% throughput |
| 1024 input, cold prefix cache, 1 output | 113.76 ms | 115.84 ms | +1.83% latency |
| 4096 input, cold prefix cache, 1 output | 122.04 ms | 124.70 ms | +2.18% latency |

Batch-8 BF16 throughput varied from 809.16 to 821.30 tok/s between its two
runs, more than the pooled +0.24% difference. Prefill also showed run-to-run
variation; it did not show a benefit. These are offline request measurements,
not production trace throughput or HTTP TTFT. Decode uses warm prompt caching.
Requests are admitted together under a paused scheduler to control actual batch
composition. Both variants produce repeatable outputs across their own A/B
runs, but generated tokens differ between precisions.

## Fused add and Norm microbenchmark

Same B200, repeated CUDA graph execution, hot tensors; two orders per shape.
Times below average the two observations and are not a full-model profile.

| Rows | FP32 residual | BF16 residual | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 1.933 µs | 1.976 µs | -2.21% |
| 8 | 2.113 µs | 2.066 µs | +2.23% |
| 128 | 2.164 µs | 1.926 µs | +11.00% |
| 1024 | 6.753 µs | 4.309 µs | +36.19% |
| 4096 | 26.949 µs | 16.293 µs | +39.54% |

Large row counts benefit from smaller residual reads/writes. Small-batch
latencies remain around 2 µs; FP32 addition/statistics and the reduction are
still present. Even the 4096-row improvement saves only about 10.7 µs per
fused call. Scaling this hot-tensor result across the 60 full-token residual
updates in the self block suggests roughly 0.64 ms, versus about 122 ms for
the complete prefill request. This is an illustrative estimate, not a measured
full-model attribution, and the end-to-end measurements did not show that gain.

## Precision

| Fixed-target set | FP32 mean NLL | BF16 mean NLL | Difference | exp(NLL) change |
| --- | ---: | ---: | ---: | ---: |
| batch1, 1024 targets | 1.446603 | 1.446820 | +0.000217 | +0.0217% |
| batch8, 1024 targets | 1.501194 | 1.456529 | -0.044666 | -4.3683% |

The batch-1 targets cover contexts of 128–4220 tokens. Batch 8 uses eight
controlled streams and 128 teacher-forced steps. Raw logprobs are checked with
a greedy/forced-token anchor, and the prefix cache is reset before evaluation.
The FP32 results exactly reproduce the preceding experiment’s first-pass
baseline, including batch 8.

Near-equal average NLL does not establish numerical equivalence. All 1024
positions changed in each set: maximum absolute target-logprob differences
are 1.2043 for batch 1 and 9.1479 for batch 8. The batch-8 average improvement
is uneven across streams and is not evidence of general quality improvement.
These are correlated samples from one public-text dataset, not broad task
accuracy or long-context validation. No QAT was performed.

## Implementation and checks

Set `VLLM_YOCO_BF16_RESIDUAL=1` before starting Fast to reproduce the candidate;
unset it or set `0` for the unchanged FP32 baseline. Align ignores the switch.

The candidate carries embeddings, residual outputs and cross-block static
buffers in BF16. Each residual addition is performed in FP32, rounded to BF16,
and then used consistently for FP32 Norm statistics and normalization. The
Fast fused op and its fake implementation preserve residual dtype; the Align
op retains FP32 output. Latent norms and the FA4 BF16 output are unchanged.

Runtime checks confirm BF16 input/output residuals for real fused-op calls,
the 42 main norms and the cross-block static buffer, plus unchanged FP8 latent
projections and FA4 FP8 KV. No unreplayed graph-buffer contents are used as
numerical evidence.

The targeted GPU/compatibility run passed 41 checks: residual rounding,
empty/tail/noncontiguous inputs, FP32 baseline, Align, fake metadata and CUDA
graph replay. Applicable pre-commit hooks passed.

## Evidence

Machine-readable results: [BF16_RESIDUAL.json](BF16_RESIDUAL.json).

Raw configurations, timings, target logprobs, scripts, source snapshots and
checks are retained in
`yoco_results/residual-bf16-20260914-em9z0xwy/` (`METHOD.md`, `SUMMARY.json`,
`abba/`, `model-results.tar`, `SOURCE_MANIFEST.json`, and validation logs).
GPU UUID: `464fe39e-1a51-de28-0118-11eac3bc1fbf`.
