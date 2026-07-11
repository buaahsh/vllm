# YOCO vLLM / llm-train Precision Alignment Summary

## Scope

This document summarizes the changes that produced measurable precision gains
when aligning:

- vLLM: `/root/code2/vllm`
- llm-train: `/root/code2/llm-train`
- native checkpoint:
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/updates_73750`
- converted HF model:
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/updates_73750_hf_codex_mxfp8_router_runtime_norm`

The target runtime is W8A8 FP8 with DeepGEMM MoE:

```bash
--quantization fp8_per_block
--moe-backend deep_gemm
--attention-backend FLASH_ATTN
--enforce-eager
```

## Final Mixed5 Metrics

Native-to-vLLM full-vocabulary next-token KL:

| Case | KL |
|---|---:|
| `short_hello` | `0.000000e+0` |
| `short_fact` | `0.000000e+0` |
| `medium_english` | `1.125429e-3` |
| `short_zh` | `0.000000e+0` |
| `long_zh` | `9.810246e-3` |
| **Mean** | **`2.187135e-3`** |

Result files:

```text
/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/current_metrics_20260703_032452/native_triton_router_mixed5_20260710/vllm_mixed5.pt
/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/current_metrics_20260703_032452/native_triton_router_mixed5_20260710/compare.json
```

For `long_zh`, block outputs are bit-exact through layer 6. The first block
divergence is now layer 7 (`rel_l2 = 7.25e-4`).

## Required Runtime Environment

```bash
export VLLM_YOCO_NATIVE_FA2_PREFILL=1
export VLLM_DEEPGEMM_MOE_PSUM_LAYOUT=1
export VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS=1
export VLLM_YOCO_COMPILED_TOPK_ROUTING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

Always launch from `/root/code2/vllm`. Do not accidentally import or execute
the copy under `/workspace/shaohanh/vllm`.

## Effective Improvements

### 1. Preserve FP32 residual streams

llm-train converts embedding output to FP32 and keeps residual additions in
FP32. vLLM now follows the same behavior:

- embedding output is converted to FP32;
- attention and MLP outputs are promoted before residual addition;
- the persistent hidden-state workspace uses FP32.

Relevant implementation: `vllm/model_executor/models/yoco.py`.

This removed the large error previously observed between attention output and
the following post-attention RMSNorm. The RMSNorm itself was not the primary
source; feeding it a lower-precision residual stream was.

### 2. Match native YOCO RMSNorm

The YOCO RMSNorm path uses a dedicated Triton implementation with:

- FP32 square accumulation and reciprocal RMS;
- BF16 affine weights, matching native parameter use;
- BF16 output, matching the next quantized operator boundary.

Relevant implementation:
`vllm/model_executor/models/yoco.py::_yoco_rms_norm_kernel`.

The same implementation is used consistently across layers.

### 3. Match native attention numerics

The effective attention changes are:

- force FlashAttention 2 prefill for the alignment path;
- construct and apply RoPE with the native FP32 device semantics;
- compute differential-attention combination using the native operation order;
- retain the same implementation across all universal-loop and cross layers.

Relevant implementation:

```text
vllm/v1/attention/backends/flash_attn.py
vllm/model_executor/models/yoco.py::_yoco_apply_rotary_emb
vllm/model_executor/models/yoco.py::_yoco_diff_attention
```

Same-input replay showed layer 0 attention through `o_proj` at approximately
`1.03e-4` before residual/RMSNorm alignment was fixed.

### 4. Normalize router weights at runtime in FP32

The checkpoint contains FP32 router master weights. llm-train normalizes each
expert row at runtime before the linear projection. vLLM now performs the same
operation:

```python
weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
router_logits = F.linear(hidden_states.float(), weight)
```

Relevant implementation:
`vllm/model_executor/models/yoco.py::_yoco_router_linear`.

The router linear must remain eager. Offline replay showed that eager output
was bit-exact with native logits, while compiling this linear introduced many
one-ULP differences.

### 5. Reproduce native compiled softmax/top-k reductions

This was the final high-impact improvement.

llm-train wraps `topk_routing` in `torch.compile`. Even with bit-exact router
logits, eager PyTorch softmax and selected-weight renormalization differed from
native by a few FP32 ULPs. Those differences occasionally crossed BF16 and FP8
midpoints before W2.

vLLM now uses dedicated Triton kernels for the YOCO alignment path:

- 128-way FP32 softmax using `libdevice.exp` and the native reduction order;
- 8-way selected-weight renormalization using the native reduction order;
- `torch.topk` remains responsible for expert selection.

The persistent layout `BLOCK_ROWS=4, num_warps=4` was selected because it
reproduced native compiled routing weights bit-for-bit on the 110-token
`long_zh` probe. Larger row blocks changed reduction association and produced
different ULPs.

Relevant implementation:

```text
vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
```

This change moved `long_zh` from:

- sign-ULP baseline KL: `1.3562e-2`;
- exact blocks through layer 2;

to:

- native Triton router KL: `9.8102e-3`;
- exact blocks through layer 6.

### 6. Apply routed probabilities before W2 quantization

Native MoE computes:

```text
BF16 W1 output
-> FP32 clamped SwiGLU
-> multiply FP32 routed probability
-> write BF16
-> per-token-group FP8 quantization
-> W2
```

Applying routed probabilities only after W2 does not reproduce native W8A8
payloads. vLLM now fuses row weights into the SwiGLU/FP8 preparation path and
sets final reduction weights to one when the probability was already applied.

Relevant implementation:

```text
vllm/model_executor/models/yoco.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/model_executor/layers/quantization/utils/fp8_utils.py
```

When native-compatible Triton routing is active, the original router weight is
used for both positive and negative activation values. The older negative
`nextafter(..., 0)` compensation is only retained for the eager-routing
fallback; applying it after routing is already aligned becomes a second,
incorrect bias.

### 7. Match native FP8 quantization boundaries

The effective FP8 changes include:

- use `eps=1e-4`, matching llm-train activation quantization;
- use E8M0 power-of-two scales for the DeepGEMM path;
- use multiply-by-reciprocal where required to avoid division-specific ULPs;
- round the fused routed SwiGLU result through BF16 before W2 FP8 quantization;
- use DeepGEMM psum layout and the same grouped-GEMM recipes as native.

These changes made same-input W1 and W2 replays exact except for isolated
payloads caused by router ULP differences. After native routing was reproduced,
the early-layer payload differences disappeared.

### 8. Match shared-expert precision

The shared expert uses the same W8A8 linear and clamped SwiGLU behavior as
native. Its scalar gate follows llm-train's actual definition:

```python
self.shared_gate = MixPrecisionLinear(embed_dim, 1, bias=False)
```

The weight may be stored as an FP32 master parameter, but the actual matmul
casts it to the activation dtype. It must not be forced into an FP32 matmul.

Relevant implementation:
`vllm/model_executor/models/yoco.py::_shared_gate_linear`.

## Diagnostic Infrastructure That Was Useful

### Block and operator dumps

`tools/yoco_alignment/logprob_kl.py` supports native/vLLM hidden-state dumps at
block and operator granularity. These identified whether the first divergence
came from attention, residual/RMSNorm, shared MLP, or routed MoE.

### Dispatch row mapping

Native sorted rows are matched to vLLM grouped rows by `(token, expert)` using
the native dispatch metadata and vLLM `inv_perm`. This made it possible to
compare:

- input hidden state;
- A1 FP8 payload and scale;
- W1 output;
- routed probability;
- A2 FP8 payload and scale;
- W2 output.

### Prepare-stage snapshots

`modular_kernel.py` snapshots inputs before `prepare_async`. This is necessary
because the prepare workspace can overwrite its inputs before the asynchronous
receiver returns. Without the snapshot, debug dumps incorrectly contained
repeated workspace rows.

Useful environment variables:

```bash
YOCO_MOE_DEBUG_DUMP_DIR=/path/to/dump
YOCO_MOE_DEBUG_TOKEN_COUNT=110
YOCO_MOE_DEBUG_CALLS=7,8
YOCO_MOE_DEBUG_LABEL=vllm
```

## Experiments That Did Not Generalize

The following experiments should not be restored without new evidence:

### Custom router/weight overrides

Static or online router overrides changed dispatch/reduction execution paths,
interacted with warmup and graph state, and worsened final KL even when an
offline replay appeared exact.

### Uniform `nextafter` adjustment

Moving every router weight toward zero fixed some isolated FP8 midpoint flips
but created new flips in later layers. A sign-conditional adjustment improved
the earlier baseline but was still compensating for the wrong routing
reduction. Once native routing was reproduced, this adjustment had to be
disabled.

### BF16 midpoint heuristic

Moving exact FP32-to-BF16 midpoint products toward zero repaired one layer 3
payload but broke an already-correct layer 5 midpoint. Product-level code
cannot determine whether the router weight is exact or one ULP high, so this
heuristic is not reliable.

### Compiling both router linear and top-k routing

Compiling the router linear changes logits. The correct combination is:

```text
eager FP32 normalized router linear
+ native-compatible compiled/Triton softmax and top-k renormalization
```

## Reproduction

```bash
cd /root/code2/vllm

export CUDA_VISIBLE_DEVICES=1
export VLLM_YOCO_NATIVE_FA2_PREFILL=1
export VLLM_DEEPGEMM_MOE_PSUM_LAYOUT=1
export VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS=1
export VLLM_YOCO_COMPILED_TOPK_ROUTING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python tools/yoco_alignment/logprob_kl.py vllm \
  --model /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/updates_73750_hf_codex_mxfp8_router_runtime_norm \
  --out /tmp/vllm_mixed5.pt \
  --prompt-suite mixed5 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --quantization fp8_per_block \
  --attention-backend FLASH_ATTN \
  --moe-backend deep_gemm \
  --max-logprobs -1

python tools/yoco_alignment/logprob_kl.py compare \
  --reference /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/current_metrics_20260703_032452/block_debug_mixed5_current/native_mixed5.pt \
  --candidate /tmp/vllm_mixed5.pt \
  --out-json /tmp/compare_mixed5.json
```

## Remaining Precision Work

The current first `long_zh` divergence is at layer 7 and affects one token's
eight routed rows together. That pattern indicates the MoE input hidden state
has already diverged before routing; it is not an isolated router or DeepGEMM
payload error. Further work should compare layer 7 attention, residual addition,
and post-attention RMSNorm for that token before making additional MoE changes.
