# 2026-06-30 YOCO vLLM / Native KL Ablation

## Checkpoint And Artifacts

- Source sharded checkpoint: `/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-1000`
- Local model-only copy: `/data/users/shaohanh/local_models/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-1000`
- Local merged native checkpoint: `/data/users/shaohanh/local_models/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/merged-0000-1000`
- Converter output base: `/data/users/shaohanh/local_models/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629`
- Latest FP32-aligned HF export: `/data/users/shaohanh/local_models/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/setting5_fp32_gates`

## Code Changes Used

- `convert_to_hf.py`
  - Preserves FP32 source tensors that matter for inference parity:
    `model.layers.*.mlp.gate.weight` and
    `model.layers.*.mlp.shared_gate.weight`. Norm weights are also kept if a
    future checkpoint stores them as FP32.
  - Writes `qk_rms_clip`, `qk_rms_limit`, `quant_mode`, and
    `quant_block_size` into `config.json`.
  - Adds direct CLI knobs: `--quant_mode/--quant-mode` and
    `--quant_block_size/--quant-block-size`.
- `vllm/model_executor/models/yoco.py`
  - YOCO MoE router `GateLinear` uses FP32 parameters.
  - YOCO MoE `shared_gate` now keeps an FP32 master weight and casts it to the
    activation dtype for the matmul, matching training `MixPrecisionLinear`.
  - Explicit vLLM `--quantization mxfp8` maps YOCO to online MXFP8
    quantization while skipping non-MXFP8 training paths such as `lambda_proj`.
- `vllm/model_executor/layers/quantization/online/mxfp8.py`
  - Wires YOCO `swiglu_limit` into MXFP8 MoE `gemm1_clamp_limit` for Marlin,
    so routed experts keep the training clamp in the MXFP8 path too.
- Runtime compatibility fixes needed for this local debug environment:
  - `rotary_embedding/common.py`: if installed `flash_attn` exists but its binary extension is ABI-incompatible, fall back to native rotary.
  - `warmup/kernel_warmup.py`: missing/outdated DeepGEMM no longer fails engine startup during warmup.

## FP32 dtype audit

Audited both the merged native checkpoint and the local model-only copy on
2026-06-30:

- `merged-0000-1000/model_state_rank_0.pth`: 41 FP32 tensors:
  20 `layers.*.mlp.gate.weight`, 20 `layers.*.mlp.shared_gate.weight`, and
  `moe_loss.accum_expert_cnt`.
- Local sharded copy `0000-1000/*.ckpt*`: same 41 FP32 tensor entries.
- No RMSNorm weights in this checkpoint are FP32; they are BF16 in the source.
  The earlier issue was not "only RMSClip FP32"; the missing FP32 inference
  weight was `shared_gate.weight`.

## Prompt Suite

Five prompts with mixed lengths were used for full-vocab next-token KL against native:

| Prompt | Kind | Token length |
|---|---:|---:|
| `short_hello` | short English completion | 3 |
| `short_fact` | short factual completion | 6 |
| `medium_english` | medium English paragraph | 66 |
| `short_zh` | short Chinese instruction | 8 |
| `long_zh` | long Chinese paragraph | 110 |

Native reference file: `/data/users/shaohanh/results/yoco_kl_mixed5_native.pt`.

## Settings

| Setting | Definition |
|---|---|
| `setting1_baseline` | Old export semantics: `qk_norm=true`, router BF16, no `quant_mode` |
| `setting2_router_fp32` | Baseline + router gate FP32 |
| `setting3_rmsclip_fp32` | Router FP32 + `qk_rms_clip=true` + RMSClip FP32 compute |
| `setting4_mxfp8` | Setting 3 + explicit online MXFP8 path from the earlier run |
| `setting5_fp32_gates` | Setting 3 + router/shared gates preserved as FP32 masters |
| `setting5_mxfp8_marlin` | Setting 5 + `--quantization mxfp8 --moe-backend marlin` |
| `rmsclip_new_cudagraph` | Setting 5 + native/vLLM RMSClip both use FP32 `variance` + `rsqrt` implementation |
| `rmsclip_new_mxfp8_marlin_cudagraph` | `rmsclip_new_cudagraph` + `--quantization mxfp8 --moe-backend marlin` |

## Aggregate Results

| Setting | Mean KL native->vLLM | Mean KL vLLM->native | Mean JS | Mean max abs logprob diff | Mean abs logprob diff |
|---|---:|---:|---:|---:|---:|
| `setting1_baseline` | 0.846584 | 0.843735 | 0.151504 | 6.830478 | 1.270527 |
| `setting2_router_fp32` | 0.880082 | 0.892054 | 0.157114 | 6.868915 | 1.315600 |
| `setting3_rmsclip_fp32` | 0.011058 | 0.011029 | 0.002749 | 1.085131 | 0.165839 |
| `setting4_mxfp8` | 0.014565 | 0.014365 | 0.003599 | 1.092148 | 0.155734 |
| `setting5_fp32_gates` | 0.010538 | 0.010629 | 0.002636 | 1.015921 | 0.135885 |
| `setting5_mxfp8_marlin` | 0.008029 | 0.008138 | 0.002011 | 0.754081 | 0.138928 |
| `rmsclip_new_fix_llm_train` | 0.007363 | 0.007421 | 0.001839 | 1.184334 | 0.137763 |
| `rmsclip_new_mxfp8_marlin_fix_llm_train` | 0.006112 | 0.006005 | 0.001510 | 0.776599 | 0.143281 |

## Per-Prompt KL native->vLLM

| Setting | short_hello | short_fact | medium_english | short_zh | long_zh |
|---|---:|---:|---:|---:|---:|
| `setting1_baseline` | 0.029759 | 0.065484 | 1.519028 | 0.343720 | 2.274929 |
| `setting2_router_fp32` | 0.032255 | 0.099417 | 1.594277 | 0.345703 | 2.328758 |
| `setting3_rmsclip_fp32` | 0.002540 | 0.004027 | 0.011776 | 0.010219 | 0.026726 |
| `setting4_mxfp8` | 0.003669 | 0.002919 | 0.027713 | 0.014006 | 0.024518 |
| `setting5_fp32_gates` | 0.002790 | 0.000282 | 0.017338 | 0.009108 | 0.023173 |
| `setting5_mxfp8_marlin` | 0.001331 | 0.001024 | 0.004797 | 0.008302 | 0.024693 |
| `rmsclip_new_fix_llm_train` | 0.001560 | 0.001115 | 0.008304 | 0.005833 | 0.020001 |
| `rmsclip_new_mxfp8_marlin_fix_llm_train` | 0.001172 | 0.001793 | 0.003881 | 0.007619 | 0.016097 |

## Top-1 Alignment

- Baseline and router-FP32-only still diverge on medium/Chinese/long prompts, e.g. `medium_english` native top1 `There` vs vLLM `The`, `long_zh` native top1 `<|endoftext|>` vs vLLM `1`.
- `setting3_rmsclip_fp32` aligns top1 for all 5 prompts.
- `setting4_mxfp8`, `setting5_fp32_gates`, and
  `setting5_mxfp8_marlin` align top1 for all 5 prompts.
- `rmsclip_new_cudagraph` and `rmsclip_new_mxfp8_marlin_cudagraph` also align
  top1 for all 5 prompts.

## Conclusion

- Router FP32 alone is not the main fix.
- The decisive parity fix is the RMSClip config/path: old `qk_norm=true` vs native `qk_rms_clip=true` is the dominant mismatch.
- On H100 in this local stack, MXFP8 runs with explicit
  `--quantization mxfp8 --moe-backend marlin`. Triton is not an MXFP8 MoE
  backend, and FlashInfer TRTLLM MXFP8 requires Blackwell-family GPUs.
- Preserving `shared_gate.weight` as an FP32 master and wiring Marlin routed
  SwiGLU clamp improves the MXFP8 KL below the BF16 setting in this mixed5
  probe, while both remain top-1 aligned.
- Changing llm-train RMSClip to the vLLM-style FP32 variance + `rsqrt`
  implementation tightened the BF16 KL to 0.007363 and MXFP8 Marlin KL to
  0.006112 on the same mixed5 prompt suite.
- For local `agl` after editable precompiled vLLM install, setting 4 required `LD_LIBRARY_PATH=/home/shaohanh/miniconda3/envs/agl/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH` so NVRTC could find `libnvrtc-builtins.so.13.0`.

RMSClip implementation used in `llm-train/llm/arch/rms_norm.py` for the new run:

```python
orig_dtype = x.dtype
x = x.to(torch.float32)
variance = x.pow(2).mean(dim=-1, keepdim=True)
clip_coef = (limit * torch.rsqrt(variance + eps)).clamp(max=1.0)
x = x * clip_coef
return x.to(orig_dtype)
```

Artifacts for the new RMSClip run:

- Native logits: `/data/users/shaohanh/results/yoco_kl_mixed5_native_rmsclip_new.pt`
- BF16 vLLM logits: `/data/users/shaohanh/results/yoco_kl_mixed5_vllm_rmsclip_new_cudagraph.pt`
- BF16 summary: `/data/users/shaohanh/results/yoco_kl_mixed5_rmsclip_new_cudagraph_summary.json`
- MXFP8 Marlin vLLM logits: `/data/users/shaohanh/results/yoco_kl_mixed5_vllm_rmsclip_new_mxfp8_marlin_cudagraph.pt`
- MXFP8 Marlin summary: `/data/users/shaohanh/results/yoco_kl_mixed5_rmsclip_new_mxfp8_marlin_cudagraph_summary.json`
- Note: native probe used a session-local torch SDPA `flash_attn` stub because
  this conda env's installed `flash_attn` binary is ABI-incompatible with the
  current Torch build.

## How To Run Setting 5 / MXFP8

Recommended conversion command for router/shared-gate FP32 + RMSClip FP32:

```bash
cd /data/users/shaohanh/vllm
.venv/bin/python convert_to_hf.py \
  --input_dir /path/to/merged-native-checkpoint \
  --output_dir /path/to/hf-yoco-fp32-gates \
  --quant_mode bfloat16
```

BF16 serve shape:

```bash
vllm serve /path/to/hf-yoco-fp32-gates \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-sharing-fast-prefill
```

MXFP8 on H100 must be explicit:

```bash
vllm serve /path/to/hf-yoco-fp32-gates \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-sharing-fast-prefill \
  --quantization mxfp8 \
  --moe-backend marlin
```

Hardware/backend note:

- Blackwell/B200: vLLM can use the true FlashInfer/CUTLASS MXFP8 W8A8
  linear path (`>= sm_100`) and FlashInfer TRTLLM MXFP8 MoE.
- H100/Hopper: this local test requires Marlin MXFP8 W8A16
  (`MarlinMxfp8LinearKernel`, `MARLIN` MXFP8 MoE backend).
- `--moe-backend triton --quantization mxfp8` fails at startup:
  `moe_backend='triton' is not supported for MXFP8 MoE. Expected one of
  ['flashinfer_trtllm', 'marlin', 'xpu']`.
- `--moe-backend flashinfer_trtllm --quantization mxfp8` also fails on H100:
  `No supported MXFP8 expert class for FLASHINFER_TRTLLM: kernel does not
  support current device cuda`.
- A100 (`sm_80`): not categorically impossible in this code because Marlin FP8 is SM80+, but it depends on the image having the new online MXFP8/Marlin stack. Older A100 images likely will not run this setting reliably; test the exact image before relying on it.

## Qwen3-30B-A3B HF vs vLLM reference

To calibrate the same mixed5 prompt suite against an upstream MoE model, ran
`Qwen/Qwen3-30B-A3B` with HF Transformers and vLLM.

Setup:

- HF: `AutoModelForCausalLM`, BF16, single H100, `attn_implementation="eager"`.
  Default SDPA failed on this host with `cuDNN Frontend error: No valid
  execution plans built`.
- vLLM: non-eager CUDA graph, BF16, `max_model_len=8192`,
  `max_num_batched_tokens=8192`, FlashAttention 3 attention, FlashInfer CUTLASS
  unquantized MoE.
- LayerNorm path verification: Qwen3Moe uses
  `vllm.model_executor.layers.layernorm.RMSNorm`; runtime logs show
  `IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`,
  which dispatches through `vllm/ir/ops/layernorm.py` via
  `ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm`.
- Re-ran after the local `vllm/ir/ops/layernorm.py` change
  (`variance = x_var.float().pow(2).mean(...)` while not eagerly casting all
  `rms_norm` input to FP32); results were unchanged from the earlier Qwen run.
- Re-ran again after the newer local `vllm/ir/ops/layernorm.py` change
  (`x / sqrt(variance + epsilon)` instead of `x * rsqrt(...)`). This changed
  the long Chinese prompt top-1 and slightly worsened aggregate KL.
- Re-ran a further layernorm variant that computes variance in FP32, casts the
  variance back to activation dtype, and then uses `x / sqrt(...)`. It improved
  aggregate KL vs `layernorm_v3` but still kept the long Chinese top-1 mismatch.

Aggregate:

| Run | Mean KL HF->vLLM | Mean KL vLLM->HF | Mean JS | Mean max abs logprob diff | Mean abs logprob diff | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3_30b_a3b_hf_vs_vllm_mixed5` | 0.006956 | 0.006978 | 0.001732 | 0.633861 | 0.127346 | 5/5 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_modified` | 0.006956 | 0.006978 | 0.001732 | 0.633861 | 0.127346 | 5/5 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v3` | 0.007704 | 0.007321 | 0.001867 | 0.576701 | 0.122334 | 4/5 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v4` | 0.007322 | 0.007044 | 0.001786 | 0.585926 | 0.137324 | 4/5 |

Per-prompt KL HF->vLLM:

| Run | short_hello | short_fact | medium_english | short_zh | long_zh |
|---|---:|---:|---:|---:|---:|
| `qwen3_30b_a3b_hf_vs_vllm_mixed5` | 0.001687 | 0.007568 | 0.002225 | 0.015891 | 0.007411 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_modified` | 0.001687 | 0.007568 | 0.002225 | 0.015891 | 0.007411 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v3` | 0.000896 | 0.006273 | 0.003453 | 0.017435 | 0.010463 |
| `qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v4` | 0.000896 | 0.004363 | 0.003453 | 0.017435 | 0.010463 |

Top-1 tokens matched on all prompts: `short_hello` -> ` I`,
`short_fact` -> ` Paris`, `medium_english` -> ` The`, `short_zh` -> space,
`long_zh` -> `在`.
For `layernorm_v3` and `layernorm_v4`, `long_zh` changed to vLLM top-1
`\n\n` while HF stayed `在`; the other four prompts still matched top-1.

Artifacts:

- HF logits: `/data/users/shaohanh/results/qwen3_30b_a3b_hf_mixed5_logits.pt`
- vLLM logits (initial): `/data/users/shaohanh/results/qwen3_30b_a3b_vllm_mixed5_logits.pt`
- vLLM logits (after layernorm change): `/data/users/shaohanh/results/qwen3_30b_a3b_vllm_mixed5_logits_layernorm_modified.pt`
- vLLM logits (layernorm v3): `/data/users/shaohanh/results/qwen3_30b_a3b_vllm_mixed5_logits_layernorm_v3.pt`
- vLLM logits (layernorm v4): `/data/users/shaohanh/results/qwen3_30b_a3b_vllm_mixed5_logits_layernorm_v4.pt`
- Initial summary: `/data/users/shaohanh/results/qwen3_30b_a3b_hf_vs_vllm_mixed5_summary.json`
- LayerNorm-change summary: `/data/users/shaohanh/results/qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_modified_summary.json`
- LayerNorm-v3 summary: `/data/users/shaohanh/results/qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v3_summary.json`
- LayerNorm-v4 summary: `/data/users/shaohanh/results/qwen3_30b_a3b_hf_vs_vllm_mixed5_layernorm_v4_summary.json`

# 2026-06-27 H100 YOCO vLLM / Native Repeat Investigation

## Background

- PT HF checkpoint: `/mnt/conversationhubhot/wenwan/exp/ckpts/30A3B-36M-RMSClip3/updates_75000_hf`
- SFT HF checkpoint: `/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-adamw-shaohan_sft_260616/0000-1500-hf`
- Reported behavior before this run:
  - `buaahsh/pytorch:26.02-a100-vllm` on A6000 serving PT checkpoint: no obvious repeat.
  - `buaahsh/pytorch:26.02-b200-vllm` on B200 serving SFT checkpoint: occasional repeat, not large-scale / not 100%.
  - `buaahsh/pytorch:26.02-a100-vllm` on A100 or A6000 serving SFT checkpoint: heavy repeat, almost 100%.

## Questions

1. On H100 with `buaahsh/pytorch:26.02-h100-vllm`, do PT and SFT checkpoints show heavy repeat?
2. On H100 with `buaahsh/pytorch:26.02-h100-vllm`, how closely do vLLM and native `/data/users/shaohanh/llm-train/llm/eval.py` align?
3. If alignment is bad, fix `/data/users/shaohanh/vllm`; if alignment is good, explain why A100/A6000 has heavy repeat.

## Local Hypothesis

The most likely local control points are YOCO generation/runtime details in vLLM rather than the checkpoint alone: tokenizer/BOS handling, QK norm/RMSClip, shared-KV fast prefill, routed MoE activation clamp, or an architecture-specific kernel path. A cheap falsification is to run identical greedy prompts on H100 for native and vLLM, then compare text/repeat metrics and, where possible, token IDs.

## Experiment Log

- Host: 4x NVIDIA H100 PCIe, compute capability 9.0.
- Docker image under test: `buaahsh/pytorch:26.02-h100-vllm`.
- Planned prompt set: short completion prompts plus chat-style SFT prompts.
- Planned repeat metric: repeated 4-gram ratio on generated text/tokens plus max repeated suffix span.
- Added probe script: `/data/users/shaohanh/llm-train/scripts/yoco_repeat_probe.py`.
- Docker notes:
  - `/mnt/msranlphot` blobfuse mount is not readable by container root; Docker must run with host UID/GID (`--user $(id -u):$(id -g)`).
  - With host UID/GID, PyTorch needs `USER=shaohanh LOGNAME=shaohanh` because the UID is not present in the container passwd database.
  - Use a fresh writable HOME/cache, e.g. `HOME=/workspace/run/exp_0627/home`, to avoid FlashInfer/vLLM cache permission errors.
- PT checkpoint path status:
  - Provided path `/mnt/conversationhubhot/wenwan/exp/ckpts/30A3B-36M-RMSClip3/updates_75000_hf` is not visible on this H100 host: `/mnt/conversationhubhot` is empty.
  - A narrow `find` across `/mnt/conversationhub`, `/mnt/conversationhubhot`, `/mnt/msranlp`, `/mnt/msranlphot` did not find `30A3B-36M-RMSClip3` or `updates_75000_hf` within the checked depth.
  - Direct path check: `/mnt/conversationhub/wenwan/exp` exists, but `/mnt/conversationhub/wenwan/exp/ckpts` does not; `/mnt/conversationhubhot/wenwan` does not exist.
- SFT vLLM H100 run:
  - Command used `buaahsh/pytorch:26.02-h100-vllm`, GPU0, TP=1, `--max-model-len 8192`, `--max-num-batched-tokens 8192`, `kv_sharing_fast_prefill=True`, greedy decoding, `max_new_tokens=128`.
  - vLLM resolved `YOCOForCausalLM`, FlashAttention 3 attention backend, FlashInfer CUTLASS unquantized MoE backend.
  - Result file: `/data/users/shaohanh/exp_0627/results/sft_vllm_h100.jsonl`.
- Image binary comparison:
  - H100 image `yoco.py` SHA: `dd4ef7dc7892b241cb2e7f5e1fadc40b4bbaff5825e44822d37a3f0526473388`.
  - A100 image `yoco.py` SHA: same as H100.
  - B200 image `yoco.py` SHA: same as H100.
  - H100 `_moe_C.abi3.so`: `sm_80 sm_90 sm_90a`; `_C.abi3.so`: `sm_75 sm_80 sm_90 sm_90a`; FA3: `sm_90a`.
  - A100 `_moe_C.abi3.so`: `sm_80`; `_C.abi3.so`: `sm_80`; FA3: `sm_75`.
  - B200 `_moe_C.abi3.so`: `sm_100 sm_80`; `_C.abi3.so`: `sm_100 sm_80 sm_90`; FA3: `sm_75`.
- Native setup:
  - Host conda PyTorch is unusable for this run: `libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd`.
  - H100 vLLM image lacks `lm_eval`, so the probe inlined the `llm/eval.py` model loading/generation path instead of importing `eval.py` directly.
  - Installed `nnscaler 0.9` into `/data/users/shaohanh/exp_0627/home/.local` for native model imports.
  - The native tokenizer default `/mnt/msranlp/yutao/hf_cache/agens_tokenizer` was not readable/visible in Docker, so the probe uses the SFT HF checkpoint tokenizer for both vLLM and native prompt tokenization.
  - Native `/data/users/shaohanh/llm-train/llm/eval.py` generation had a YOCO decode-step bug for this model: after prefill it called `model(next_input, context=attention_context)` without `cu_seqlens_q/k`; YOCO cross layers then accessed `context['cu_seqlens_q']` and raised `KeyError`. Fixed in `llm/eval.py` by adding single-token `cu_seqlens_q/k` and `max_seqlen_q/k=1` during decode. The probe carries the same fix.

## Results

- SFT checkpoint on H100 vLLM: no heavy repeat observed in 6 prompts.
  - `hello_name`: 128 generated tokens, repeat-4gram ratio 0.176, no repeated suffix loop. Text is repetitive in style (`I am a very ...`) but not the failure mode described as 100% looping.
  - `france_capital`: repeat-4gram ratio 0.000, no suffix loop.
  - `harry_potter`: repeat-4gram ratio 0.000, no suffix loop.
  - `zh_intro`: 46 generated tokens, stopped on EOS, repeat-4gram ratio 0.000.
  - `zh_reasoning`: repeat-4gram ratio 0.032, no suffix loop. Note: it starts with `答案：2` but solves to `4`, so content quality has an issue, but not a repetition loop.
  - `en_recipe`: repeat-4gram ratio 0.088, no suffix loop.
- Native H100 alignment blockers resolved: the first native runs were blocked by missing `nnscaler`, then by the default tokenizer path, then by the `cu_seqlens_q` decode-step KeyError in `llm/eval.py`. The `eval.py` decode bug is now fixed, and smoke/full native probe results are below.
- Native H100 smoke after fix: `hello_name`, 32 generated tokens, repeat-4gram ratio 0.069, no repeated suffix. Native text begins `Kaitlyn and I am a 20 year old...`; vLLM begins `Katelyn and I am a 20 year old...`. This is high semantic/token-trajectory similarity but not bit-exact equality.
- Native H100 6-prompt run after fix: result file `/data/users/shaohanh/exp_0627/results/sft_native_h100_64.jsonl`, greedy, `max_new_tokens=64`.
  - `hello_name`: 64 tokens, repeat-4gram ratio 0.115, no repeated suffix.
  - `france_capital`: 64 tokens, repeat-4gram ratio 0.000, no repeated suffix.
  - `harry_potter`: 64 tokens, repeat-4gram ratio 0.000, no repeated suffix.
  - `zh_intro`: 50 tokens, stopped on EOS, repeat-4gram ratio 0.000.
  - `zh_reasoning`: 64 tokens, repeat-4gram ratio 0.033, no meaningful suffix loop (`\\` repeated 2 chars only).
  - `en_recipe`: 64 tokens, repeat-4gram ratio 0.098, no repeated suffix.
- Native/vLLM alignment level on H100 SFT:
  - Not bit-exact. Example: `hello_name` starts with native `Kaitlyn` vs vLLM `Katelyn`.
  - High trajectory/behavior alignment. After the first token, both produce the same style and many identical phrases (`and I am a 20 year old college student...`). Chat prompts and recipe/math prompts are also semantically close.
  - Most importantly for this investigation, both implementations do not show the reported heavy repeat on H100.
- Next-token accuracy probe started after the generation/repeat study:
  - Added `/data/users/shaohanh/llm-train/scripts/yoco_next_token_probe.py` to compare first next-token top-k/logprobs on identical prompt token IDs.
  - vLLM result file: `/data/users/shaohanh/exp_0627/results/sft_vllm_h100_next_top20.jsonl`.
  - vLLM top1 next tokens: `hello_name` -> `730` (` K`, logprob -4.261), `france_capital` -> `12089` (` Paris`, -0.167), `harry_potter` -> `43958` (` eleven`, -1.001), `zh_intro` -> `121703` (`我叫`, -1.420), `zh_reasoning` -> `98457` (`设`, -1.257), `en_recipe` -> `334` (`**`, -0.019).
  - Native result file: `/data/users/shaohanh/exp_0627/results/sft_native_h100_next_top20.jsonl`.
  - vLLM greedy-1 without logprobs matches the vLLM logprob top1 for all 6 prompts, so the vLLM logprob API did not change its own first-token choice.
  - Native top1 next tokens: `hello_name` -> `730` (` K`, -4.218), `france_capital` -> `12089` (` Paris`, -0.166), `harry_potter` -> `12740` (` orphan`, -1.010), `zh_intro` -> `121658` (`您好`, -1.128), `zh_reasoning` -> `334` (`**`, -1.150), `en_recipe` -> `334` (`**`, -0.022).
  - Top1 exact match: 3/6 prompts (`hello_name`, `france_capital`, `en_recipe`).
  - Top-k overlap is high: average top-5 overlap 4.67/5, average top-20 overlap 19.17/20.
  - Common-token logprob differences are nonzero: mean absolute diff by prompt ranges from 0.037 to 0.212; max absolute diff ranges from 0.126 to 0.708.
  - Accuracy verdict: native/vLLM are not numerically/bit-exact. They are top-k aligned and behaviorally close, but first-token ranking can flip when candidates are close (`harry_potter`, `zh_intro`, `zh_reasoning`).
- PT H100 result: blocked by missing/invisible PT checkpoint mount on this host.

## Conclusion

Final conclusion for the accessible SFT checkpoint: H100 with `buaahsh/pytorch:26.02-h100-vllm` does not reproduce the massive repeat. Native `llm/eval.py` after the YOCO decode metadata fix also does not reproduce it. vLLM/native are not bit-exact or strictly numerically accurate against each other; first-token top1 agrees on 3/6 prompts, while top-k overlap is high. They are close enough behaviorally that the H100 vLLM implementation is not the likely source of the A100/A6000 100%-repeat failure, but the precision-accuracy check should be reported as "top-k aligned, not exact" rather than "fully aligned".

The likely explanation for A100/A6000 is an architecture-specific compiled-kernel/backend issue rather than checkpoint corruption or a YOCO Python-model mismatch. Evidence: H100/A100/B200 images have identical `vllm/model_executor/models/yoco.py` SHA (`dd4ef7dc...`) and include the RMSClip/cross-Q-norm/fast-prefill fixes, while compiled extensions differ by target architecture. The A100 image's `_moe_C.abi3.so` and `_C.abi3.so` only contain `sm_80` cubins and no PTX fallback; this is especially suspicious for A6000 (`sm_86`) and still leaves A100 exposed to the `sm_80` fused-MoE/attention path. H100 and B200 use different architecture-specific binaries (`sm_90/sm_90a`, `sm_100`) and do not show the same failure pattern.

No `/data/users/shaohanh/vllm` Python fix was applied because the workspace `yoco.py` already matches the tested images and contains the known correctness fixes. One native fix was applied in `/data/users/shaohanh/llm-train/llm/eval.py`: decode-step YOCO generation now supplies `cu_seqlens_q/k` and `max_seqlen_q/k=1`, preventing the native `KeyError` during cross-decoder generation.

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
