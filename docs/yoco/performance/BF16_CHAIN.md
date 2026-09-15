# YOCO Fast BF16 main activation chain

The BF16 main activation and router path improves this same-B200 experiment
by 3.19% at batch 1 and 2.42% at batch 8. Prefill latency falls about 2.5%.
This extends the [BF16 residual-storage experiment](BF16_RESIDUAL.md), adding
BF16 residual boundary arithmetic and BF16 router operands/logits.

**Scope:** main activation tensors at residual/Norm/router boundaries use BF16.
FP8 projections, MoE and FA4 attention/KV retain the existing FP8 configuration.
FP32 kernel-local accumulation/reductions and internal backend workspaces, FP8
scale metadata, selected Top-K probabilities, reference gate weights, RoPE
cache and final sampling logits remain. This is not a globally FP32-free model.
The CUDA BF16 `hrsqrt` intrinsic itself uses an FP32 `rsqrt` instruction.

## Same-card results

One physical B200, TP1, checkpoint `30A3B-180M-L3/0000-28000-hf`, Fast block-FP8,
FA4 FP8 KV, attention fusion enabled, latent Norm fusion disabled. Run order:
FP32 A → BF16-chain A → BF16-chain B → FP32 B. Each precision contributes
14 timings per case; report the pooled median.

| Workload | FP32 baseline | BF16 chain | Change |
| --- | ---: | ---: | ---: |
| Batch 1, 512 input + 128 output | 158.69 tok/s | 163.75 tok/s | +3.19% throughput |
| Batch 8, 512 input + 128 output | 816.02 tok/s | 835.76 tok/s | +2.42% throughput |
| 1024 input, 1 output | 113.50 ms | 110.56 ms | -2.59% latency |
| 4096 input, 1 output | 121.55 ms | 118.51 ms | -2.50% latency |

Decode uses warm prompt caching, five warmups and seven timed repetitions per
run; all batch requests are admitted together under a paused scheduler. Prefill
clears prefix cache for every request, with three warmups and seven timed
repetitions. These are offline request measurements, not HTTP TTFT or production
trace throughput. Both BF16 runs improve on both FP32 runs for the measured
decode cases. Outputs repeat within each precision but differ across precisions,
so the timings include those different autoregressive trajectories.

## Operator timing

Hot-tensor CUDA graph measurements, two orders per shape. Router baseline
includes the full-width BF16-to-FP32 input conversion and TF32-enabled GEMM.

| Rows | Previous router | BF16 router | Previous BF16-storage Norm | Native-BF16-add Norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.756 µs | 3.462 µs | 1.888 µs | 1.985 µs |
| 8 | 8.346 µs | 3.732 µs | 2.069 µs | 2.075 µs |
| 128 | 12.370 µs | 4.127 µs | 1.921 µs | 1.905 µs |
| 1024 | 16.553 µs | 4.477 µs | 4.325 µs | 4.347 µs |
| 4096 | 37.815 µs | 5.389 µs | 16.719 µs | 16.790 µs |

The router boundary/GEMM improvement is substantial; changing the addition
instruction by itself barely changes Norm latency. At batch 1 the router
difference is about 5.29 µs per logical layer, or about 0.21 ms over 40 calls.
That illustrative estimate is consistent with the measured decode gain; it
is not a full-model profiler attribution. PTX confirms `add.rn.bf16` in the
candidate residual kernel.

## Precision

| Fixed-target set | FP32 mean NLL | BF16-chain mean NLL | NLL difference |
| --- | ---: | ---: | ---: |
| batch1, 1024 targets | 1.446603 | 1.444308 | -0.002294 |
| batch8, 1024 targets | 1.501194 | 1.462467 | -0.038727 |

Raw logprobs are validated with a forced/greedy anchor. Batch 1 uses contexts
128–4220; batch 8 uses eight controlled streams and 128 teacher-forced steps.
The prefix cache is reset before evaluation. FP32 exactly reproduces the
previous baseline in both sets.

All sampled target probabilities change. Maximum absolute logprob differences
are 1.2337 at batch 1 and 4.3131 at batch 8. The average NLL improvement on this
correlated public-text sample does not establish general quality improvement
or losslessness. Router weight/logit rounding can change selected experts.
No training or QAT work was performed.

## Implementation

Enable with `VLLM_YOCO_BF16_CHAIN=1` before starting Fast. The switch defaults
to 0 and is independent of the earlier storage-only experiment; enabling it
implies BF16 residual storage even if `VLLM_YOCO_BF16_RESIDUAL=0`. Align ignores
the switch.

- Embeddings, carried residuals, Norm inputs/outputs and cross-block buffers
  use BF16. Materialized residual boundaries add BF16 tensors directly.
- The fused residual/Norm kernel uses native `add.rn.bf16` on SM90+ and keeps
  the same rounded-residual normalization semantics. Older CUDA architectures
  retain register-local FP32 addition with BF16 output.
- Router normalization occurs at cache initialization. Its runtime BF16 weight
  cache refreshes in place on updates. Input, weight and router logits are BF16,
  removing the former full-width `hidden_states.float()` intermediate.
- Fast Top-K reads BF16 logits directly; its selected probabilities remain
  FP32 for the existing MoE backend. The FP32 reference weights and original
  normalized cache are retained for loading/compatibility.

Runtime manifests verify all 42 main Norms, cross-block residual buffers,
20 router caches and real router input/weight/output dtypes, plus unchanged
40 FP8 latent projections and FA4 FP8 attention/KV.

## Validation and evidence

63 kernel and boundary tests passed, covering BF16 addition equivalence to
the prior storage-only mode, empty/tail/noncontiguous inputs, Top-K, no FP32
tensor at the tested residual/router boundaries, cache refresh, fake metadata
and graph replay. An additional 24 existing router/residual/cache compatibility
checks passed. Native BF16 addition was verified in generated PTX. Applicable
pre-commit checks passed.

Machine-readable results: [BF16_CHAIN.json](BF16_CHAIN.json). Raw scripts,
timings, logprobs, runtime manifests, source snapshots and checks are retained
in `yoco_results/bf16-chain-20260914-1iqftvfu/`. `METHOD.md` records exact scope
and settings; `CUDA_BF16_INTRINSIC.json` records the SDK intrinsic source.
GPU UUID: `464fe39e-1a51-de28-0118-11eac3bc1fbf`.
