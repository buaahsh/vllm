We need continue from this exact state. User’s current goal: “提升 KL 对齐到 5e-3级别” for repo `/root/code2/vllm`, comparing vLLM vs llm-train using `vllm/tools/yoco_alignment/logprob_kl.py`. They care about W8A8 with `--moe-backend deep_gemm`; it should not trigger `deep_gemm.fp8_fp4_gemm_nt illegal memory access`. KL target: `*e-3`, specifically now 5e-3 level.

Important workspace context:
- CWD `/root/code2`.
- Actual repo is `/root/code2/vllm`; user explicitly said it should not run `/workspace/shaohanh/vllm`.
- Original checkpoint: `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3`
- Conversion script: `vllm/convert_to_hf.py`
- Model converted under: `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/hf_model`
- Output dir used: `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex`
- User previously corrected:
  `self.shared_gate = MixPrecisionLinear(embed_dim, 1, bias=False)`
  should NOT be:
  `self.shared_gate = MixPrecisionLinear(embed_dim, 1, bias=False, dtype=torch.float32)`
- They also said fp32 residual should help precision.

Current repo edits / diffs:
1. `vllm/model_executor/layers/quantization/quant_utils.py`
   - `ScaledActivationQuantConfig.__init__` now preserves `is_per_tensor=False` instead of forcing per-token:
     ```py
     self.is_per_tensor = is_per_tensor
     ```
   - `act_quant_func` now uses `per_token_group_quant_fp8` for W8A8 MoE hidden states when not per-tensor:
     ```py
     if self.is_per_tensor:
         fp8_act, scale = ops.scaled_fp8_quant(hidden_states, scale=None)
     else:
         fp8_act, scale = per_token_group_quant_fp8(hidden_states, self.group_shape[-1], scale_ue8m0=True)
     return fp8_act, scale
     ```
   - Added imports:
     ```py
     from vllm.model_executor.layers.quantization.fp8_utils import per_token_group_quant_fp8
     from vllm.model_executor.layers.quantization.utils.w8a8_utils import all_close_1d
     ```
   - Comment added about `self.group_shape = group_shape if not is_per_tensor else None`.

2. `vllm/model_executor/layers/fused_moe/layer.py`
   - Added `self.use_fp32_residual = False` in `FusedMoE.__init__`.
   - Passes `use_fp32_residual=self.use_fp32_residual` into `fused_experts(...)`.

3. `vllm/model_executor/layers/fused_moe/fused_moe.py`
   - `fused_experts` signature includes `use_fp32_residual: bool = False`.
   - Calls into backend with `use_fp32_residual=use_fp32_residual`.

4. `vllm/model_executor/layers/fused_moe/backends/deep_gemm.py`
   - `DeepGemmExperts.__init__` accepts `use_fp32_residual: bool = False`, stores it.
   - `forward_impl` now uses `working_hidden_states = hidden_states.float()` when `use_fp32_residual`, otherwise original dtype.
   - Applies activations and reductions on `working_hidden_states`.
   - There was a temporary debug print:
     ```py
     print("[DGDEBUG]", ...)
     ```
     It was removed.
   - Still had some possibly suspicious blank-line / whitespace changes but functional.

5. `vllm/model_executor/models/yoco.py`
   - `YOCOAttention.forward_native` changed residual path to fp32:
     ```py
     if self.post_norm:
         residual = hidden_states.float()
         hidden_states = self.self_attn_layer_norm(hidden_states)
     ...
     if self.post_norm:
         hidden_states = residual + hidden_states.float()
     ```
   - `YOCODecoder.forward_native` now calls `self.self_decoder(..., original_input=input_embeds, ...)` instead of `None`.
   - There was an accidental debug print removal already:
     ```py
     print("[YDEBUG] ...")
     ```
     removed.
   - Current `shared_gate` line should be checked; user insists it must be:
     ```py
     self.shared_gate = MixPrecisionLinear(embed_dim, 1, bias=False)
     ```
     not dtype fp32.

6. `vllm/tools/yoco_alignment/logprob_kl.py`
   - `make_vllm_engine`:
     - `enforce_eager=True`
     - `max_model_len=4096`
     - `gpu_memory_utilization=0.9`
   - `collect_vllm_logprobs`:
     - default `tensor_parallel_size=1`
   - `VLLM_MODEL_DIR` env var support added:
     ```py
     vllm_model = os.environ.get("VLLM_MODEL_DIR", str(checkpoint_dir / "hf_model"))
     ```
   - test cases were reduced/changed. Current likely contains 5 cases only:
     `short_hello`, `short_fact`, `medium_english`, `short_zh`, `long_zh`.
   - `compute_kl` now masks non-finite reference rows before KL:
     ```py
     finite_mask = torch.isfinite(ref)
     row_mask = finite_mask.all(dim=-1)
     ref = ref[row_mask]
     test = test[row_mask]
     ```
   - Also a truncation helper likely exists around 32k rows:
     ```py
     max_rows = int(os.environ.get("YOCO_KL_MAX_ROWS", "32768"))
     ```
   - Need inspect before editing further.

7. `vllm/convert_to_hf.py`
   - `original_input_linear = convert_weight("model.original_input_linear.weight")` changed to `.bfloat16()`.
   - Sets config `tie_word_embeddings=False`.
   - q_scale loading changed to reshape flattened scales:
     ```py
     loaded_scale = (orig_tensor.reshape(loaded_weight.shape[0], -1).mean(dim=1) if orig_tensor.ndim == 1 and orig_tensor.numel() != loaded_weight.shape[0] else orig_tensor)
     loaded_scale = loaded_scale.bfloat16()
     ```
   - `input_scale` now loads bfloat16.
   - `lm_head.weight` now saved separately from `embed_tokens.weight`.
   - There is a syntax error currently from a botched patch: around line ~618 indentation is wrong:
     ```py
     tensor = original_input_linear
     target_weight = f"model.layers.{hf_layer_i}.original_input_linear.weight"
         save_file(tensor, shard_path)
     ```
     Need fix before using conversion.

8. `vllm/tools/yoco_alignment/compare_state_dicts.py`
   - Added alignment for `original_input_linear` and `lm_head.weight`.
   - `tensor_distance` casts integer/bool tensors to float for mean/abs calculations.

9. New/modified helper scripts:
   - `vllm/tools/yoco_alignment/compare_model_outputs.py`
     - Existed untracked/new.
     - Added `VLLM_MODEL_DIR` env support.
   - `vllm/tools/yoco_alignment/export_llm_train_logprobs.py`
     - Created then modified with actual llm-train import path `/mnt/pvc/lidong1/exp/agens/llm-train`.
     - Loads `TransformerDecoderModel`.
     - Has `safe_globals` for custom checkpoint args.
     - Uses prompts mirrored from current `logprob_kl.py`.
     - Uses no autocast after last edit, to match llm-train.
     - Should be considered experimental; did not yet successfully run end-to-end at final state.
   - `.gitignore` updated with local output ignores:
     ```
     kl_codex/
     *.kl.json
     *.kl.pt
     vllm/tools/yoco_alignment/*.pt
     ```

Commands/results already run:
- Earlier KL with wrong fp4/marlin / pre-fixes had terrible KL ~11-12 and illegal memory access for deep_gemm. This improved after per-block activation quant fix.
- A successful deep_gemm vLLM-only logprob export after quant fixes:
  ```bash
  cd /root/code2/vllm
  VLLM_MODEL_DIR=/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/hf_model \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  CUDA_VISIBLE_DEVICES=0 \
  python tools/yoco_alignment/logprob_kl.py \
    /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3 \
    --collect vllm \
    --moe-backend deep_gemm \
    --quantization fp8 \
    --dtype bfloat16 \
    --output-dir /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex \
    --tag mixed3_vllm_fp8_per_block_deep_gemm_flash_attn_tp1
  ```
  This produced:
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/updates_73750_mixed3_vllm_fp8_per_block_deep_gemm_flash_attn_tp1.pt`
  Top1s:
  - short_hello: token 358 ` I` -0.993898
  - short_fact: token 12089 ` Paris` -0.429070
  - medium_english: token 576 ` The` -1.147023
  - short_zh: token 98326 `我` -3.131438
  - long_zh: token 154820 `<|endoftext|>` -2.310829
- Current latest sequential vLLM export after user’s latest `shared_gate` correction / mixed5:
  Command ran successfully from `/root/code2/vllm`:
  ```bash
  VLLM_MODEL_DIR=/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/hf_model \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  CUDA_VISIBLE_DEVICES=0 \
  python tools/yoco_alignment/logprob_kl.py \
    /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3 \
    --collect vllm \
    --moe-backend deep_gemm \
    --quantization fp8 \
    --dtype bfloat16 \
    --output-dir /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex \
    --tag mixed5_vllm_fp8_per_block_deep_gemm_flash_attn_sequential_root_eager_ignore_lm_head
  ```
  Output saved:
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/updates_73750_mixed5_vllm_fp8_per_block_deep_gemm_flash_attn_sequential_root_eager_ignore_lm_head.pt`
  Top1s:
  - short_hello: 358 ` I` -0.994304
  - short_fact: 12089 ` Paris` -0.430578
  - medium_english: 576 ` The` -1.155094
  - short_zh: 98326 `我` -3.137320
  - long_zh: 154820 `<|endoftext|>` -2.311321
  This session is complete now; no running exec session remains.

Latest KL comparisons:
- Using reference:
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/updates_73750_llm_train_reference.pt`
- After modified `compute_kl`, mixed3 comparison against llm-train reference:
  ```bash
  python tools/yoco_alignment/logprob_kl.py ... --compare \
    --reference-logprobs .../updates_73750_llm_train_reference.pt \
    --vllm-logprobs .../updates_73750_mixed3_vllm_fp8_per_block_deep_gemm_flash_attn_tp1.pt
  ```
  Results:
  ```json
  short_hello KL 0.006996, finite_rows 7/7
  short_fact KL 0.004077, finite_rows 9/9
  medium_english KL 0.006318, finite_rows 16/16
  short_zh KL 0.010055, finite_rows 13/13
  long_zh KL 0.003988, finite_rows 20/28
  average_kl 0.006287
  ```
- Latest mixed5 sequential export likely needs compare next. Since it is very close to mixed3 top1/logprobs, expect average around ~0.006-0.007, not yet 5e-3. Next action should compare exact mixed5 file:
  ```bash
  cd /root/code2/vllm
  python tools/yoco_alignment/logprob_kl.py \
    /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3 \
    --compare \
    --reference-logprobs /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/updates_73750_llm_train_reference.pt \
    --vllm-logprobs /mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/kl_codex/updates_73750_mixed5_vllm_fp8_per_block_deep_gemm_flash_attn_sequential_root_eager_ignore_lm_head.pt
  ```

Useful observations:
- After quant fix, illegal memory access gone for W8A8 `--moe-backend deep_gemm`.
- Top logits are now very close:
  llm-train reference top1s from file:
  - short_hello: -0.993651 vs vLLM -0.994304
  - short_fact: -0.428763 vs vLLM -0.430578
  - medium_english: -1.151743 vs vLLM -1.155094
  - short_zh: -3.141055 vs vLLM -3.137320
  - long_zh: -2.311971 vs vLLM -2.311321
  The remaining KL is distribution tail/small broad differences, not argmax.
- Prior `compare_model_outputs.py` was useful to compare FP logits and intermediate summaries, but user wants KL. There was a shape bug comparing kv cache: fixed local if needed? It failed with:
  ```
  RuntimeError: The size of tensor a (160) must match size of tensor b (32) at non-singleton dimension 1
  ```
  Could revisit if needed.

Potential next steps:
1. First inspect current diffs:
   ```bash
   git status --short
   git diff -- vllm/model_executor/layers/quantization/quant_utils.py vllm/model_executor/layers/fused_moe/layer.py vllm/model_executor/layers/fused_moe/fused_moe.py vllm/model_executor/layers/fused_moe/backends/deep_gemm.py vllm/model_executor/models/yoco.py vllm/tools/yoco_alignment/logprob_kl.py vllm/convert_to_hf.py
   ```
2. Fix `convert_to_hf.py` syntax/indentation before any conversion. Search around `original_input_linear` block.
3. Compare latest mixed5 KL.
4. To reach 5e-3:
   - Investigate why short_zh is ~0.010 and medium/short_hello ~0.006.
   - Since top logits are close, likely mismatch in reference / logits truncation / lm_head / embedding tie / output projection precision / activation quant scales.
   - User mentioned fp32 residual: ensure fp32 residuals are active in all relevant paths, not just YOCOAttention and MoE optional flag. Need trace where `FusedMoE.use_fp32_residual` is set from model/config; currently initialized False and likely never enabled, so the deep_gemm fp32 residual path may not actually be used. Search for `use_fp32_residual`.
   - Maybe YOCO model should set `FusedMoE.use_fp32_residual=True` for relevant layers, or backend should always use fp32 residual when applicable for YOCO.
   - Ensure `shared_gate` is exactly without dtype fp32.
   - Consider compare against no-quant or `--moe-backend pytorch` to isolate if KL from quant/MoE or YOCO residual/attention. But user’s main target is W8A8 deep_gemm.
   - Could collect vLLM with `--enable-expert-parallel`? The commands did not use expert parallel. Keep same unless investigating.
   - Could temporarily run `--collect vllm` with `--dtype float32`? vLLM likely not support full fp32 with fp8? Use carefully.
   - Revisit `lm_head`: conversion sets `tie_word_embeddings=False` and saves separate `lm_head.weight`; current script tag says `ignore_lm_head` maybe because model loader ignored? Need inspect yoco HF loader mapping. If lm_head mismatch, KL could persist even with same hidden states. Compare `lm_head.weight` in hf vs original with `compare_state_dicts.py`.

Need be careful:
- Do not run `/workspace/shaohanh/vllm`; always `cd /root/code2/vllm`.
- Use apply_patch for manual edits.
- Use `rg` first.
- There may be dirty user changes; do not revert.
- Final answers should be in Chinese likely, concise, with files and metrics.
- User asked “continue”, now objective is actively improve KL to 5e-3, so implement/verify rather than just propose.