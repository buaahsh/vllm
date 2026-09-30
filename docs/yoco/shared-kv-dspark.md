# YOCO Shared-KV DSpark

This experimental V2-runner integration loads the four-layer `SharedKVDraft`
from `dspark-shared-kv-sft20k-20260929/sharedkv-balanced-dp16-8k10k-full10k`.
It uses the **balanced** YOCO target and its frozen embedding/LM head.
The source warm-start teacher is a different checkpoint and must not be used.

## Export and run

Export a committed, independently published training checkpoint. The exporter
verifies the config and model hashes against both publication receipts and
writes a new directory; it never modifies the training checkpoint.

```bash
.venv/bin/python examples/offline_inference/spec_decode/export_yoco_shared_kv.py \
  --checkpoint /path/to/checkpoints/step-003750 \
  --target /mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf \
  --output /path/to/serving/sharedkv-step-003750

VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=0 OMP_NUM_THREADS=4 \
  vllm serve /mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf \
    --spec-model /path/to/serving/sharedkv-step-003750 \
    --spec-method dspark --spec-tokens 8 \
    --dtype bfloat16 --max-model-len 131072 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.75 \
    --no-enable-prefix-caching --enforce-eager
```

The exported config uses the existing Qwen3 DSpark config schema with an
explicit `shared_kv_config` discriminator. The V2 factory dispatches to
`SharedKVSpeculator`; loading it through the ordinary Qwen3 backbone is rejected.
The loader pins the balanced target path and checks the draft weight SHA256.
Keep the target path used at export, or re-export for an equivalent mounted path.

## Inference contract

- Anchor position `a` contains the target's bonus/correction token. The remaining
  block slots contain the trained mask token. Slot `j` predicts position `a+j+1`.
- Current hidden input is raw target `h20[a-1]`, before final RMSNorm. Rejected
  verifier positions are excluded when selecting this row.
- Every query reads exactly target shared K/V positions `[0,a)`. Global Q has
  RMSNorm and **no RoPE**. The cache already contains the actual normalized YOCO
  K/V; the draft performs no K/V projection, cache write, or history copy.
- Local attention is causal within the draft block, with absolute-position RoPE.
  Current-hidden and global-attention gates, slot embeddings, all four layers,
  and the sequential Markov head retain their training checkpoint names.
- FP32 trainable parameters and residuals run under BF16 autocast; frozen
  vocabulary tensors remain BF16. HF-derived RMSNorm weights preserve their
  FP32 promotion semantics. Confidence uses the training FP32 dot product.
- V2's existing DSpark sampler and rejection machinery handle generated tokens.
  Chunked prefill uses the next prefill token as its temporary anchor.

## Validation and limitations

Local validation uses an RTX A6000. It covers CPU rejection boundaries and
block causality, GPU paged attention with both HND and NHD physical layouts,
strict-prefix isolation, no cache mutation, and mixed prefill/decode inputs.
A four-layer small model is also compared against the run's immutable training
implementation with the same HF norm classes. Hidden states and logits are
bitwise equal using the same SDPA path. Switching to paged FlashAttention gives
maximum hidden error 0.006985 and logit error 0.0078125 in that fixture; all eight
argmax tokens agree.

An end-to-end V2 smoke test uses a synthetic 20-layer, three-loop YOCO target
and a four-layer shared-KV draft, three requests, chunked prefill, and 24 generated
tokens per request. All 72 greedy output tokens match a separate ordinary
decode run. These checks establish plumbing and small-model correctness;
they do not establish the real checkpoint's acceptance rate or speedup.

The initial draft runs **eagerly** with TP1/PP1/DP1 and no context parallelism,
quantization, LoRA, or top-k draft approximation. K may be 1 through 8 because
local attention is causal. Target CUDA graphs are independent, but the initial
smoke command above disables them. Full-checkpoint B200 serving, acceptance and
performance validation remain necessary before using this as a benchmark.

The next performance work is to capture the draft step, fuse block-local
projections/norms and Markov sampling, and evaluate shared-cache attention at
matched assistant-turn workloads. Compare accepted draft tokens separately from
the bonus/correction token, and report decode-only latency as well as throughput.
