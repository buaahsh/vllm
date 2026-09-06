# YOCO L3 Align Batch/Split Invariance Plan

Date: 2026-09-04
Initial scope: YOCO L3, BF16, NVIDIA B200, TP1

## 1. Goal and semantic contract

`--align` must implement a stronger contract than ordinary batch invariance.
For the same model weights and the same logical token prefix, logits and token
generation must not depend on how execution is partitioned.

The target contract covers:

- batch size and request order;
- splitting one batch into multiple micro-batches;
- one-shot prefill versus chunked prefill;
- full-sequence teacher forcing versus per-token KV-cache decode;
- continuous-batching insertion and removal;
- CUDA Graph bucket size, padding, and recompilation context;
- request-scoped sampling with the same seed.

Equivalently, the following value must be a function only of weights, logical
prefix, position, and sampling parameters:

```text
logits(logical_prefix)
```

The first milestone does not include invariance across TP/PP/CP layouts.
Multi-GPU collective reduction order will be handled separately after TP1
passes the full contract.

Both `llm-train` and vLLM must opt into the same numerical kernels. Making only
vLLM batch invariant is insufficient for strict train/inference alignment.
The default training path and vLLM `--fast` path must remain unchanged.

## 2. Current foundation

The current vLLM tree already provides `VLLM_BATCH_INVARIANT` support for:

- fixed-layout BF16/FP32 linear operations;
- deterministic RMSNorm, mean, softmax, and log-softmax;
- Attention configuration constraints;
- deterministic Triton MoE configuration;
- deterministic collective policy.

YOCO cannot enable this facility unchanged. Several YOCO Align paths call
`F.linear` or private custom operations directly, and therefore bypass the
generic dispatch. The existing facility also targets batch composition, not
strict equivalence between prefill, chunked prefill, teacher forcing, and
KV-cache decode.

`--align` should imply process-scoped batch-invariant initialization before
worker and model initialization. It must retain CUDA Graph decode and must not
silently fall back to eager execution.

## 3. Existing kernels that can be reused

Reuse assumes that `llm-train` is changed to call the same numerical operation
where required.

| Operator | Existing implementation | Required integration | Expected cost |
| --- | --- | --- | ---: |
| Embedding lookup | `VocabParallelEmbedding`/table lookup | Verification only | Approximately zero |
| Residual add | Pointwise FP32 add | Preserve operand order | Approximately zero |
| KV-cache write | Existing disjoint slot scatter/update | Verify slot/order invariance | Approximately zero |
| RoPE | `yoco_rotary` | Use in both runtimes | Approximately zero |
| Differential attention combine | `yoco_diff_attention_v3` | Use in both runtimes | Approximately zero |
| Router Top-K | `yoco_fused_topk_routing` | Adopt the same tie and normalization rule in both runtimes | Approximately zero; may improve speed |
| Expert output reduction | `yoco_topk8_sum` | Preserve expert-id-ordered FP32 accumulation | Approximately zero |
| Shared gate and final add | `yoco_fused_shared_gate_moe_output` | Use in both runtimes | Approximately zero |
| BF16/FP32 dense GEMM base | `linear_batch_invariant` | Route YOCO direct linears through it | Shape dependent |
| Probability normalization | Batch-invariant softmax/log-softmax | Connect Router and final probability paths | Small |
| Routed expert base | TritonExperts fixed configuration | Add training adapter and verify exact boundaries | Shape dependent |

The existing request-scoped RNG and greedy/sampling implementations should be
kept, but they require end-to-end invariance tests after logits become exact.

## 4. Kernels that need bounded extension

### 4.1 RMSNorm and fused add-RMSNorm

- Hidden size 3072 already has an Align-specific fixed reduction tree.
- Latent hidden size 1024 must stop falling back to a shape-dependent compiled
  expression.
- Fused add-RMSNorm currently selects different reduction widths for small and
  large token counts. Align must use one fixed reduction tree for every token
  count.
- The FP32 residual materialization and BF16 normalized output boundaries must
  remain identical in training and inference.

Expected end-to-end cost: below 1% if the existing fused kernel is retained.

### 4.2 RMSClip and QK Clip + RoPE

- Small token counts currently retain a compiled RMSClip expression.
- Every head row must use the same 128-element reduction tree at every batch
  size.
- The existing fused weighted QK RMSClip + RoPE kernel can be reused because it
  preserves the intermediate BF16 rounding boundary; training must use the
  same operation.

Expected end-to-end cost: approximately 0-1%.

### 4.3 Dense projections

The following Align projections must use a fixed-K, `SPLIT_K=1` numerical
contract:

- self-attention Q, K, V, and lambda projections;
- cross-attention Q and lambda projections;
- model-level shared K and V projections;
- attention output projection;
- latent input/output projections;
- shared expert gate/up/down projections;
- final LM head.

Q/K/V and lambda remain separate operations unless the shared training kernel
defines a fused operation with identical intermediate rounding semantics.
Direct YOCO `F.linear` calls must not bypass the invariant implementation.

Expected kernel cost versus tuned cuBLAS/CUTLASS: approximately 0-15%, with an
estimated 3-8% end-to-end contribution. These values require a B200 A/B.

### 4.4 Router projection

The current Router temporarily enables TF32 and lets the GEMM implementation
depend on the token-row shape. Align needs one shared rule in both runtimes:

- fixed FP32/IEEE accumulation is the preferred initial implementation;
- fixed TF32 is acceptable only if it proves bitwise invariant across all
  target shapes and fresh compilation contexts;
- normalized Router weights may be cached during inference only if the cached
  tensor is bitwise equal to the training-side fixed normalization.

A naive pad-to-128 implementation previously added about 44-50 microseconds per
Router call for small shapes. It must not be the final implementation because
YOCO invokes the Router 40 times per model step. A fixed per-row kernel should
target an end-to-end cost below 2-5%.

### 4.5 LM head

The existing small-M YOCO LM-head kernel has a fixed hidden-dimension reduction
and the required BF16-store-then-FP32 boundary, but it only covers at most 16
rows. It should be extended to all required logits-row counts or replaced by
one invariant linear implementation for all sizes. Switching algorithms at
M=16 is not allowed unless both algorithms are proven bitwise identical.

Expected cost: small for decode; approximately 5-15% for the LM-head kernel on
large prefill batches. The full-model impact should be measured separately.

## 5. Kernels or integration that require substantial development

### 5.1 Split-invariant attention

This is the primary blocker. Existing FlashAttention batch-invariant settings
do not prove equality across these execution forms:

```text
one-shot prefill
chunked prefill
teacher forcing
KV-cache decode
```

Self/SWA and Cross Attention require a common numerical contract with:

- a fixed query/head mapping;
- a fixed logical key-block traversal order;
- a fixed online-softmax reduction tree;
- identical GQA head mapping;
- identical causal and 513-token SWA window semantics;
- layout adapters for contiguous training K/V and paged inference K/V that do
  not change arithmetic order.

The kernel must be CUDA-Graph-compatible. Recomputing every prefix separately
is not an acceptable implementation because its performance loss would be
prohibitive.

Estimated cost versus the current Fast attention path:

- decode Attention kernel: approximately 0-16%;
- prefill Attention kernel: approximately 10-30%;
- estimated end-to-end contribution: approximately 2-8%.

These are engineering estimates, not measured results for the new kernel.

### 5.2 Train/inference-common MoE

vLLM TritonExperts already exposes a batch-invariant fixed configuration, so a
new W13/W2 kernel is not the first choice. Required work is:

- use the same deterministic dispatch in both runtimes;
- use fixed W13/W2 tile shapes and `SPLIT_K=1`;
- preserve the BF16 and FP32 routing-weight boundary around clamped SwiGLU;
- preserve expert-id-ordered FP32 Top-8 accumulation;
- provide a training adapter and backward implementation;
- verify that adding unrelated tokens to an expert does not change an existing
  token's expert output.

Fast's B200-tuned W13/W2 configuration improves the combined expert kernels by
7.6-26.8% at large token counts and improved the measured 8x1024 prefill
throughput by 3.97%. Fast decode FlashInfer/TRTLLM MoE has also measured up to a
mid-teens throughput advantage over the Triton baseline. The first invariant
implementation is therefore expected to lose approximately 4-10% on prefill
and 10-18% on decode versus the current Fast MoE policy.

## 6. Forward-order implementation plan

1. Make `--align` initialize the batch-invariant runtime before workers start;
   retain CUDA Graph decode and reject unsupported precision/configurations.
2. Validate embedding, residual add, position handling, and KV-cache writes.
3. Fix 3072/1024 RMSNorm and all 128-wide RMSClip shapes.
4. Route every attention/latent/shared/LM-head dense projection through the
   common invariant GEMM contract.
5. Reuse the fixed RoPE and differential-attention kernels.
6. Implement and validate split-invariant Self/SWA Attention, followed by Cross
   Attention.
7. Fix Router normalization, projection, softmax, Top-K, and tie handling.
8. Connect deterministic MoE dispatch, W13, activation, W2, and expert combine
   to both runtimes.
9. Reuse the fused shared gate/shared+routed output operation.
10. Finish with final RMSNorm, LM head, log-softmax, and request-scoped sampling.

Each step must pass its local exactness gate before work moves to the next
forward operator.

## 7. Validation matrix

### Per-operator invariance

Insert the same target row/sequence at different positions among unrelated
inputs and test:

```text
batch = 1, 2, 4, 100, 1024, 2048
```

For each target, compare:

- one batch versus multiple micro-batches;
- eager versus CUDA Graph;
- different graph buckets and padding;
- fresh compilation/cache directories and process restarts;
- different filler-token ordering.

The gate is `torch.equal`, `max_abs_diff == 0`, and exact route IDs where
applicable.

### Per-layer cross-runtime trace

Trace both runtimes at every logical boundary across all 40 YOCO logical layer
calls:

- normalization output;
- Q/K/V and lambda projections;
- normalized/rotated Q/K;
- raw Attention output;
- differential combine and output projection;
- Router logits, probabilities, and expert IDs;
- W13/activation/W2 and routed combine;
- latent and shared-expert outputs;
- residual output.

Stop at and repair the first mismatching boundary rather than relying only on
final KL.

### End-to-end semantic invariance

Required cases:

- ragged mixed-length batches;
- the batch sweep above;
- a 4K prefill in one piece and in multiple chunk sizes;
- teacher forcing versus online KV-cache decode at every logical prefix;
- a long natural rollout;
- temperature-zero greedy generation;
- seeded stochastic generation with full-vocabulary probability capture.

Final acceptance requires:

```text
all traced activations bitwise equal
all Router IDs and probabilities bitwise equal
full-vocabulary logits and log-probabilities bitwise equal
KL = 0
generated token IDs identical
```

## 8. Preliminary performance budget

Existing measurements show that current Fast versus current Align is:

| Workload | Current Fast versus current Align |
| --- | ---: |
| 1K prefill | -1.3% |
| 4K prefill | +2.8% |
| Decode | +9.1% |
| Same-node Mooncake | +10.0% |

The proposed stronger semantic mode is expected to fall in these preliminary
ranges:

| Contract | Versus current Align | Versus current Fast |
| --- | ---: | ---: |
| Batch-composition invariance only | 3-8% slower | 10-18% slower |
| Full batch and execution-split invariance | 8-20% slower | 15-30% slower |
| Worst-case long prefill | 15-25% slower | 20-35% slower |

The estimates are deliberately ranges. The main uncertainty is the common
Attention kernel; Norm, RoPE, Top-K, shared gate, and pointwise operations
should contribute little. Every implementation phase must record B200 kernel
latency and end-to-end throughput against both current Align and Fast before
the estimate is replaced by measured data.

## 9. Isolation and rollout

- Develop in a separate worktree/runtime so an active benchmark continues to
  use its frozen source and image.
- Keep all YOCO-specific kernels and adapters private to YOCO rather than
  changing generic MoE/model behavior for other architectures.
- Keep `--fast` unchanged and add regression gates for its current throughput.
- Introduce the matching llm-train behavior behind an explicit opt-in flag
  until the entire validation matrix passes.
- Do not publish a claim of train/inference semantic invariance until fresh
  process, fresh compilation-cache, chunked-prefill, and online-decode tests
  all pass.

## 2026-09-05 implementation update

The B200 BF16 TP1 inference implementation and its recorded acceptance results
are in [the progress report](YOCO-ALIGN-INVARIANCE-PROGRESS-20260905.md).
It retains CUDA Graph decode, async scheduling and shared-expert overlap.
Training-side integration and end-to-end performance remain separate work.
