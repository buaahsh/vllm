# YOCO Align batch invariance — 2026-09-05

## Result and scope

The inference-side implementation passes the recorded NVIDIA B200, YOCO L3,
BF16, TP1 acceptance cases. CUDA Graph decode, asynchronous scheduling and
shared-expert stream overlap remain enabled. The Fast policy is unchanged.
This is not a claim of train/inference bitwise equivalence, other precisions,
or invariance across distributed layouts.

- Worktree: `/home/lidong1/vllm_test/vllm_yoco_align_invariant`
- Branch: `dev/yoco-align-invariant-20260905`
- Evidence: `/home/lidong1/vllm_test/yoco_results/align-invariant-b200-20260905/`
- Incremental change: `implementation.patch` in that evidence directory.

The worktree includes the initial uncommitted `fhb-dev` snapshot. The original
checkout's tracked diff was verified unchanged; no changes were committed.

## Implementation

- `--align` and its `additional_config` alias initialize process-scoped
  `VLLM_BATCH_INVARIANT` before backend selection and worker startup. BF16
  weights/KV and TP/PP/CP=1 are required. Full decode CUDA Graphs are retained.
- Fixed 1024/3072 RMSNorm and fused FP32 residual-add/RMSNorm, plus one fixed
  128-wide RMSClip path at every token count, including strided inputs.
- Direct YOCO projections and model-owned dense/LM-head methods use invariant
  GEMM. For up to 32 rows, a smaller M=16 tile preserves the K=64 MMA traversal;
  it is checked bitwise against the original M=128 implementation.
- Router projection uses a fixed IEEE FP32 per-row reduction. Softmax, Top-8
  selection and renormalization use a fixed Triton kernel with leftmost ties.
  Selected pairs are sorted by expert ID for the fixed-order expert sum.
- Align uses fixed Triton W13/W2 configuration; shape-dependent DeepGEMM W2 is
  disabled for this mode.
- Align respects the invariant backend's FA2 selection on B200. Pure prefill
  and mixed prefill/decode both use paged KV instead of changing attention
  arithmetic when request arrival changes the batch.

All YOCO launch choices remain private to YOCO. No training source, running
service source, Volcano Job, or Pod was replaced.

## Validation

| Gate | Result | Evidence |
| --- | --- | --- |
| Core operators, batches 1/2/4/100/127/128/129/1024/2048 | 207 cases exact | `fixed-operators-a.json` |
| Fresh process/cache and reversed warmup history (4096 down to 2) | Same 207 cases and fingerprints | `fixed-operators-b.json` |
| Actual 3072-wide Router, normalized/raw weights | Additional 18 cases, repeated with fresh cache/warmup | `fixed-router-l3-{a,b}.json` |
| Real model: submitted batches 1/2/4/100/1024/2048; target at first/middle/last position; early-finishing fillers | 15 cases; all 154,880 log-probabilities and all 8 generated tokens exact | `fixed-large.json` |
| 4K one-shot prefill vs 257-token chunks, followed by 8 decode steps | Tokens and complete log-probabilities exact | `fixed-4k-compare.json` |
| 1024-step sampled trajectory, seed 42, temperature 0.8, top-p 0.95; alone vs 4 requests | Identical tokens and all per-step complete-logit SHA-256 hashes | `fixed-long.json` |
| Focused CUDA regression tests on B200 | 50 passed | `b200-tests-fixed.log` |
| CLI/configuration tests | 81 passed | `arg-tests-final.log` |

Operator checks compare target placement, 17-token microbatches, eager/graph,
and changed filler rows. The model batch sizes above are submission counts;
they do not assert that every scheduler step reached that runtime batch size.
Graphs through bucket 2048 were actually captured. The long test hashes logits
inside the worker to avoid creating millions of Python log-probability objects;
it uses `ignore_eos=True` to keep trajectory lengths equal.

The broader local conversion/config suite retains four failures reproduced in
the original checkout: two incomplete mock configs, an M=1 contiguity assertion,
and a mock MLP without the new loop argument. These are recorded separately in
`baseline-existing-tests.log`; they were not hidden by editing unrelated tests.

## Bugs reproduced

Original weighted RMSClip failed 6 of 108 cases. A 2048-token, 64-head batch
and 17-token microbatches differed by up to 0.125. The fixed kernel removes the
small/large-M reduction switch.

A second failure arose when a request arrived alongside an existing decode:
standalone prefill used contiguous QKV while mixed batches used paged KV.
Embedding, norm, Q/K/V and RoPE were exact; the first traced discrepancy was
layer 0 attention output (616 elements, maximum 0.0078125). Final full-vocabulary
log-probability differences reached 0.811386. Uniform paged-KV dispatch fixes it.

Larger graph-capture configurations exposed a further failure with compiled
Router softmax/Top-K. Replacing that path with fixed Triton routing passed the
same large-bucket configuration and the full matrix above. Tests that disabled
asynchronous scheduling or shared-expert overlap did not establish a necessary
restriction, so both were restored in the final implementation.

## Performance and remaining work

These are CUDA Graph kernel timings, not end-to-end throughput claims. For the
3072-to-8192 Q projection, the M=16 launch reduces the initial invariant kernel
from about 58.7 to 17.8 us at M=1 and 55.3 to 14.6 us at M=8, with identical bits.
The corresponding cuBLAS baseline remains faster. At M=2048, the 3072-wide
IEEE Router measured about 164 us versus 9.7 us for the previous TF32 projection.
Large-M invariant GEMM and Router performance still need work; see `fixed-operator-latency.json` for
measurements including the actual 3072-wide Router. The planning document's
optimistic throughput ranges have not been established by end-to-end tests.

Remaining work includes a training-side opt-in to the same kernels and backward
implementation, complete per-layer/teacher-forcing and cross-runtime logits/KL
validation, other distributed layouts,
and end-to-end performance tuning. Historical same-shape llm-train expression
parity must not be confused with this stronger inference-side execution policy.

## Runtime and reproduction

- Pod: `bonete01/yoco-align-prob-b200-20260901-master-0`
- Model: `/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf`
- Verified runtime: `/data/yoco-align-invariant-20260905/runtime-fixed`
- Python: `/data/yoco-align-invariant-20260905/.venv/bin/python`

Tests used available GPUs 5/6, with short operator checks on GPU 7. Existing
services on the Pod were preserved. The scripts under `tools/yoco_alignment/`
provide operator, model, controlled-insertion, attention-replay and long-logit
checks; use fresh `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` directories for
compilation-context verification. The target supports standard `kubectl exec`
with `KUBECTL_REMOTE_COMMAND_WEBSOCKETS=true` in this environment.
