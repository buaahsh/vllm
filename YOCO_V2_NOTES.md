# YOCO-MoE-30B-A3B-v2 — vLLM serving notes

Status of the v2 architecture changes when serving via this vLLM checkout.

## Summary

| Change          | Status        | Where                                                        |
|-----------------|---------------|-------------------------------------------------------------|
| `qk_rms_clip`   | ✅ Done        | `RMSClip` in `vllm/model_executor/models/yoco.py`           |
| `qk_rms_limit`  | ✅ Done        | Used by `RMSClip` (default 3.0)                              |
| `swiglu_limit` (shared expert) | ✅ Done (exact) | `YOCOSharedExperts` uses `SiluAndMulWithClamp`  |
| `swiglu_limit` (routed experts) | ⚠️ **NOT DONE** | `YOCOMoE.experts` (`FusedMoE`) runs plain SiLU |

## swiglu_limit — what is and isn't done

Training semantics (`llm-train/llm/arch/ffn.py`, `all2all_moe.py`):
`gate = clamp(x_gate, max=L)`, `up = clamp(x_up, -L, L)`, `out = silu(gate) * up`, default `L = 10.0`.

- **Shared (dense) expert** — exact match implemented via
  `SiluAndMulWithClamp(swiglu_limit)` (`vllm/model_executor/layers/activation.py`).
  Verified active: measured shared-expert pre-activation peaked ~18.6 vs limit 10.0.
- **Routed experts** — **still plain SiLU, swiglu_limit NOT applied.**
  The fused-MoE kernels actually used at runtime (FlashInfer TRTLLM / CUTLASS /
  default unquantized) only accept an activation *enum* and expose no hook to
  pass `swiglu_limit` into the fused `gemm1 -> act -> gemm2` kernel. Only certain
  quantized paths (mxfp4 / cutlass / marlin / gpt_oss) honor it
  (`gemm1_clamp_limit`, `swiglu_limit_func` in `fused_moe/utils.py`).
  Decision: shipped as-is because output is coherent (limit=10.0 is loose) and
  the FlashInfer backend is fast. This is an exact-fidelity gap, not a no-op.

### To make the routed path exact (future work)
1. Force the **TRITON** MoE backend (instead of FlashInfer TRTLLM).
2. Pass `swiglu_limit=...` into `FusedMoE(...)` in `YOCOMoE.__init__`.
3. Thread `swiglu_limit` through `apply_moe_activation`
   (`fused_moe/activation.py`) and its callers
   (`fused_moe/fused_moe.py:~1825`, `fused_moe/modular_kernel.py:~886`),
   applying `swiglu_limit_func` (clamp-before-silu) for our SILU activation.

See the `NOTE(swiglu_limit)` comment at `YOCOMoE.experts` in
`vllm/model_executor/models/yoco.py` for the in-code marker.

## config.json / convert_to_hf.py
- `convert_to_hf.py` now emits `qk_norm`, `qk_rms_clip`, `qk_rms_limit` (3.0),
  `swiglu_limit` (10.0). Models converted before this fix need their
  `config.json` patched with those keys (then `--hf-overrides` is unnecessary).

## "Can't stop" / runaway generation
- Symptom: a coherent reasoning block repeats verbatim until `max_tokens`.
- Cause: greedy-ish decoding repetition loop, NOT an architecture bug.
- Fix: recommended sampling defaults `temperature=0.7, top_p=0.95,
  repetition_penalty=1.05` (bake into the model's `generation_config.json`).

## Serve command
```bash
CUDA_VISIBLE_DEVICES=7 vllm serve <model_path> \
  --served-model-name yoco-30A3B-sft --host 0.0.0.0 --port 8000 --trust-remote-code \
  --gpu-memory-utilization 0.9 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
```
Answer text is in `message.content`; chain-of-thought is in `message.reasoning`.
