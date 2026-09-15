# YOCO BF16 reductions and sampling: implementation and backend limits

**The complete request is not implemented.** Native BF16 RMSNorm reductions
and greedy sampling/logprob arithmetic are available as experiments. Current
FA4/DeepGEMM Tensor Core accumulators and scale interfaces cannot simply be
changed to BF16. Other attention/routed-expert scalar kernels retain their
existing precision. Training and native DeepGEMM files are unchanged.

The additional scalar-precision changes produce no meaningful end-to-end gain
in this test and slightly worsen independently scored NLL. Both new flags
remain off by default.

## Backend support, verified on B200

| Requested change | Result |
| --- | --- |
| RMSNorm multiplication and sum reduction | Implemented with native BF16 instructions |
| Greedy sampling logits and logprobs | Implemented in BF16; random/speculative/specific-token-id sampling unsupported |
| FA4 / FP8 GEMM BF16 accumulation | Not supported by current Tensor Core operation types |
| Native DeepGEMM BF16 scale tensors | Rejected by its dtype contract; FP32 or packed integer scales accepted |

The official [PTX instruction descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)
lists F16/F32 matrix-D accumulator types for the relevant MMA kinds; BF16 is
an input type, not a matrix-D accumulation option. Block-scaled MXF8 uses an
FP32 accumulator and E8M0 scales. Current CUTLASS operation construction also
rejects `acc_dtype=BFloat16`, while `Float32` succeeds. The FP8 Triton dot
probe succeeds with FP32 and rejects BF16 accumulation/output.

DeepGEMM accepts the positive-control FP32 scales and produces finite output.
Passing BF16 scales instead raises
`sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt`. Its packed UE8M0 path
stores four 8-bit exponents in an INT32 container; replacing that wire format
with BF16 would not be a drop-in compression. No cast-back workaround is
presented as BF16 backend support.

## Same-GPU end-to-end results

The baseline is the preceding [BF16 main-chain/router](BF16_CHAIN.md), with
FP32 Norm statistics and sampling. Both variants keep FP8 GEMMs, FA4 FP8 KV,
attention fusion and the same checkpoint. Latent Norm-to-FP8 fusion is off.
Run order: baseline A → candidate A → candidate B → baseline B, on one B200.
Each case has 14 measurements per precision; the table reports pooled medians.

| Case | Previous BF16 chain | BF16 Norm + greedy sampling | Change |
| --- | ---: | ---: | ---: |
| Batch 1, 512 input + 128 output | 163.44 tok/s | 163.88 tok/s | +0.27% throughput |
| Batch 8, 512 input + 128 output | 836.98 tok/s | 836.51 tok/s | -0.06% throughput |
| 1024 input + 1 output | 110.51 ms | 111.02 ms | +0.47% latency |
| 4096 input + 1 output | 118.55 ms | 119.32 ms | +0.65% latency |

Decode has five warmups per pass and warm prompt caching, with all requests
admitted together. Prefill has three warmups per pass and a cache reset for
every request. These are offline request times, not HTTP TTFT. Normal decode
timings do not request logprobs. Generated outputs differ between precisions
but repeat within each precision.

## Independent quality audit

Returning lower-precision logprobs changes the metric itself. Therefore the
quality phase additionally computes FP64 log-softmax from unprocessed logits
before masking or penalties, gathers the sampled target, and records it outside
the timed sections. GPU FP64 work here is a validator, not the inference mode.

The asynchronous engine produces discarded sampler calls. Only the unique
contiguous match of all 1024 returned B1 target tokens is used. Both runs match
at audit index 3; unmatched calls are excluded. B8 audit records are not used
without per-request IDs. The public-text target contexts are 128–4220 tokens.

| B1, 1024 targets | Baseline | Candidate |
| --- | ---: | ---: |
| Independently scored mean NLL | 1.444308626 | 1.447189991 |
| Mean NLL from returned logprobs | 1.444308416 | 1.444188148 |

True NLL increases by 0.002881, or +0.2886% in exp(NLL).
The BF16 returned-logprob error averages +0.003002 nats and reaches 0.029988
nats on these targets, masking that small degradation if the returned score
alone is used. This is one correlated text sample, not general task accuracy.

## Scalar implementation and limitations

Set `VLLM_YOCO_BF16_REDUCTIONS=1` for the Fast RMSNorm experiment and
`VLLM_YOCO_BF16_SAMPLING=1` for BF16 greedy logits/logprobs. Use
`VLLM_YOCO_BF16_CHAIN=1` and leave `VLLM_YOCO_FP8_LATENT_NORM_FUSION=0`
to reproduce this experiment. Align ignores the new modes.

Native BF16 add/multiply instructions implement the Norm arithmetic and
reduction tree. A shared 384-KiB BF16 table supplies rsqrt/exp/log values; it
is generated once on the CPU in FP64. Norm and the three logprob kernels were
checked in PTX and contain no FP32 add/subtract/multiply/divide/FMA or elementary
function arithmetic. This changes reduction precision, not only output dtype.

The L3 B200 LM-head kernel now has a BF16-output variant; its GEMM accumulator
still uses FP32. The sampler retains BF16 logits and BF16 raw/processed
logprobs for greedy requests. Min-token masks and logit-bias tensors match the
logit dtype. NumPy export uses a CPU FP64 container for already-rounded BF16
logprobs, preserving values without creating a GPU FP32 logprob tensor.

Random requests, speculative decoding and `logprob_token_ids` are rejected
by this experimental sampler instead of silently running a different precision.
FA4 softmax, routed-expert accumulation and other existing fused kernels are
outside the implemented scalar subset. This is not an all-BF16 inference engine.

## Validation and evidence

97 kernel/boundary tests and 75 existing sampler/LM-head checks passed.
Applicable pre-commit hooks passed. Tests cover error against FP64, masks, tails, full vocabulary
size, native BF16 PTX, fake metadata, graph replay, prior residual/chain modes,
and min-token/logit-bias/export behavior. Existing sampler and LM-head checks
are recorded in the audit logs. Native DeepGEMM paths/hashes and the reserved
holder identity are checked at completion.

Raw evidence: `yoco_results/bf16-internals-20260914-yp2l2wxl/`. This includes
`METHOD.md`, `SUPPORT.json`, downloaded `PTX_ISA.html`, source snapshots,
`v3-abba/`, `model-results.tar`, generated PTX and validation logs. Earlier
failed/aborted integration attempts are retained in the archive and excluded
from the final timing/quality tables.

Machine-readable results: [BF16_INTERNALS.json](BF16_INTERNALS.json).
