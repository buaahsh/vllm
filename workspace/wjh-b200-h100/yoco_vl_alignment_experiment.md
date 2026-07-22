# YOCO-VL Tensor Alignment Notes

Date: 2026-07-21

## Scope

This note records the current YOCO-VL alignment experiments between:

- Native llm-train inference: `/home/v-jiahaowang/workspace/llm-train/llm/vl_infer.py`
- vLLM one-shot script: `/home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100/run_yoco_vl_once.py`
- Tensor compare script: `/home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100/compare_yoco_vl_tensors.py`

All runs were executed inside Docker container `wjh-b200-h100` with the Docker default Python environment (`/usr/bin/python`), not `.venv`.

## Inputs

- Native checkpoint: `/data/wjh/updates_3000`
- vLLM/HF model: `/data/wjh/updates_3000-hf-vl`
- Tokenizer: `/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622`
- Image: `/home/v-jiahaowang/workspace/llm-train/workspace/dog.jpeg`
- Prompt: `Describe this image in detail.`
- System prompt: `You are a helpful and friendly AI assistant.`
- vLLM dtype: `bfloat16`
- vLLM quantization: `None`
- vLLM MoE backend: `triton`
- Reference quant mode used for comparison: `bfloat16`

## What Is Aligned

- Docker execution environment is aligned: both paths run inside `wjh-b200-h100`.
- Tokenizer path is aligned.
- Image path and input image are aligned.
- Rendered chat prompt is aligned.
- Image preprocessing is aligned:
  - `pixel_values` exact.
  - `grid_hw` exact.
  - image token count exact.
- Special token IDs are aligned by using llm-train's tokenizer:
  - `<sop>` / BOS id: `154824`
  - image start id: `154830`
  - image end id: `154831`
- Token embedding lookup is exact in the direct tensor comparison.
- Final next-token top1 is aligned:
  - token id: `32`
  - decoded text for the one-token test: `A`

## What Is Partially Aligned

The early vision path is close but not bitwise identical.

Baseline tensor diffs with reference vision attention `flash_attention_cute` and default vLLM FlashAttention selection:

| Tensor | Relative RMS |
| --- | ---: |
| `vision_patch_embed` | `1.27983e-06` |
| `vision_block0_attn` | `5.15494e-05` |
| `vision_block0_out` | `1.85693e-04` |
| `vision_encoder` | `1.75065e-02` |
| `projected_image_features` | `1.20487e-02` |
| `combined_input_embeddings` | `1.19146e-02` |

Current interpretation: preprocessing and embedding are aligned; numerical drift starts small in the vision tower and accumulates through the full vision encoder. The projector then carries roughly `1e-2` relative RMS difference into the combined input embeddings.

## What Is Not Fully Aligned

- BOS / prompt length:
  - Native original packed prompt length is `1872`.
  - vLLM string generation prefill length is `1871`.
  - Current final-hidden comparison uses the native no-BOS variant to match vLLM's string prefill length.
  - This aligns the comparison shape, but the original native with-BOS path is still not identical to vLLM's string prompt path.
- Full final hidden matrix still differs substantially:
  - baseline relative Frobenius: `0.366341`
  - baseline global cosine: `0.933160`
  - baseline mean row cosine: `0.932335`
- Next-token hidden still differs:
  - baseline relative RMS: `0.112470`
- Next-token logits are close but not exact:
  - baseline relative RMS: `0.0528689`
  - baseline KL(ref -> vLLM): `0.00762677`
  - baseline JS: `0.00179887`

## Parameter Experiments

No vLLM internal logic was changed for these experiments. Only script/runtime parameters were varied.

| Case | Reference vision attn | vLLM FA version | Vision encoder rel | Projected rel | Final hidden rel | Next hidden rel | Logits rel | KL | JS | Prob L1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | `flash_attention_cute` | default | `0.0175065` | `0.0120487` | `0.366341` | `0.112470` | `0.0528689` | `0.00762677` | `0.00179887` | `0.0670443` |
| vLLM FA4 | `flash_attention_cute` | `4` | `0.0191581` | `0.0126043` | `0.365303` | `0.107001` | `0.0543069` | `0.00908067` | `0.00212319` | `0.0444765` |
| vLLM FA2 | `flash_attention_cute` | `2` | `0.0213659` | `0.0146194` | `0.384461` | `0.0859996` | `0.0474230` | `0.00233142` | `0.00055725` | `0.0293696` |
| ref FA2 + vLLM FA2 | `flash_attention_2` | `2` | `0.0184687` | `0.0122914` | `0.374068` | `0.125253` | `0.0677245` | `0.00383964` | `0.000983514` | `0.0323188` |

Experiment JSON outputs:

- Baseline: `/data/wjh/yoco_vl_tensor_compare_final_divergence.json`
- vLLM FA4: `/data/wjh/yoco_vl_tensor_compare_vllm_fa4.json`
- vLLM FA2: `/data/wjh/yoco_vl_tensor_compare_vllm_fa2.json`
- Reference FA2 + vLLM FA2: `/data/wjh/yoco_vl_tensor_compare_ref_fa2_vllm_fa2.json`

## Current Best Parameter Choice

If the priority is final next-token logits / softmax distribution, use:

```bash
--vllm_flash_attn_version 2
```

This improves final distribution alignment:

- logits relative RMS: `0.0528689 -> 0.0474230`
- KL(ref -> vLLM): `0.00762677 -> 0.00233142`
- JS: `0.00179887 -> 0.00055725`
- probability L1: `0.0670443 -> 0.0293696`

However, this is not a full-chain improvement:

- vision encoder relative RMS gets worse: `0.0175065 -> 0.0213659`
- projected image features get worse: `0.0120487 -> 0.0146194`
- final hidden matrix gets worse: `0.366341 -> 0.384461`

Therefore, `vllm_flash_attn_version=2` is a useful parameter-level choice only if the target metric is the final next-token distribution.

## One-Shot Script Update

`run_yoco_vl_once.py` now exposes:

```bash
--vllm_flash_attn_version {2,3,4}
```

It can also be set through:

```bash
YOCO_VLLM_FLASH_ATTN_VERSION=2
```

Validation command:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 \
  -w /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100 \
  wjh-b200-h100 \
  python run_yoco_vl_once.py \
    --image /home/v-jiahaowang/workspace/llm-train/workspace/dog.jpeg \
    --max_new_tokens 1 \
    --vllm_flash_attn_version 2
```

Observed output:

```text
A
```

## Summary

Preprocessing, tokenizer handling, image token count, and token embedding lookup are aligned. The remaining drift starts in the vision tower and accumulates through the full vision encoder, then enters the language model through projected image features. Final hidden states are still not tightly aligned, but final next-token logits and the top1 token are close. Parameter-only tuning can improve final distribution alignment with `--vllm_flash_attn_version 2`, but it does not improve the whole tensor chain.

## Batch Inference Smoke Test

A vLLM batch script was added at:

```text
/home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100/run_yoco_vl_batch_once.py
```

It mirrors `/home/v-jiahaowang/workspace/llm-train/llm/vl_batch_infer.py` defaults:

- images:
  - `/home/v-jiahaowang/workspace/llm-train/workspace/dog1.jpeg`
  - `/home/v-jiahaowang/workspace/llm-train/workspace/dog2.jpeg`
  - `/home/v-jiahaowang/workspace/llm-train/workspace/dog3.jpeg`
- prompts:
  - `Describe this dog and the surrounding scene in one concise sentence.`
  - `What is the dog doing? Mention its pose and expression.`
  - `List the most noticeable visual details in this image.`
- system prompt: `You are a helpful and friendly AI assistant.`
- tokenizer: `/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622`
- dtype: `bfloat16`
- quantization: `None`
- MoE backend: `triton`
- prefix caching: disabled

Validation commands:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 \
  -w /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100 \
  wjh-b200-h100 \
  python run_yoco_vl_batch_once.py \
    --max_new_tokens 1 \
    --vllm_flash_attn_version 2 \
    --output_json /data/wjh/yoco_vl_batch_vllm_smoke.json
```

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 \
  -w /home/v-jiahaowang/workspace/llm-train \
  wjh-b200-h100 \
  python llm/vl_batch_infer.py \
    --max_new_tokens 1 \
    --output_json /data/wjh/yoco_vl_batch_native_smoke.json
```

One-token batch result:

| Sample | Image tokens | Prompt tokens | Native ids/text | vLLM ids/text |
| ---: | ---: | ---: | --- | --- |
| 0 | `1849` | `1878` | `[32]` / `A` | `[32]` / `A` |
| 1 | `1849` | `1878` | `[785]` / `The` | `[785]` / `The` |
| 2 | `2916` | `2943` | `[785]` / `The` | `[785]` / `The` |

The batch script aligns preprocessing metadata and one-token greedy output with the native batch script under the tested settings.

## Batch KL Result

Batch next-token KL was computed with:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 \
  -w /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100 \
  wjh-b200-h100 \
  python compare_yoco_vl_batch_kl.py \
    --output_json /data/wjh/yoco_vl_batch_kl.json
```

Settings:

- reference quant mode: `bfloat16`
- reference vision attention: `flash_attention_cute`
- vLLM dtype: `bfloat16`
- vLLM quantization: `None`
- vLLM MoE backend: `triton`
- vLLM prefix caching: disabled
- vLLM FlashAttention version: `2`

vLLM scheduled the three samples as two logits calls, `[1, 154880]` and `[2, 154880]`; the script concatenates them in call order before computing divergence. The top1 order matches native, so this ordering is consistent for this run.

| Sample | Native top1 | vLLM top1 | KL ref->vLLM | KL vLLM->ref | JS | Prob L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `32` | `32` | `0.0228355` | `0.0286937` | `0.00627377` | `0.145089` |
| 1 | `785` | `785` | `0.00505332` | `0.00521546` | `0.00126551` | `0.0274208` |
| 2 | `785` | `785` | `0.0312098` | `0.0359915` | `0.00821131` | `0.189127` |

Mean:

- KL ref->vLLM: `0.0196996`
- KL vLLM->ref: `0.0233002`
- JS: `0.00525019`
- probability L1: `0.120546`
- top1 agreement: `1.0`

Logits matrix stats:

- relative Frobenius: `0.0754955`
- global cosine: `0.997146`
- mean abs diff: `0.238900`
- max abs diff: `2.03125`

## Batch KL Reliable Baseline: No BOS

Update date: 2026-07-22

`compare_yoco_vl_batch_kl.py` now exposes two comparison-only controls:

```bash
--reference_prompt_variant {with_bos,no_bos}
--vllm_max_num_seqs N
```

The reliable baseline for comparing native batch logits against the current vLLM path is `no_bos`. vLLM receives rendered string prompts, and this string prompt path does not include llm-train's explicit leading `<sop>` token. Therefore the native reference logits should remove only that leading BOS token before computing KL against vLLM.

This does not change either model's implementation. It only makes the native reference sequence match the actual vLLM string-prefill sequence length.

No-BOS batch KL command:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 \
  -w /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100 \
  wjh-b200-h100 \
  python compare_yoco_vl_batch_kl.py \
    --reference_prompt_variant no_bos \
    --vllm_flash_attn_version 2 \
    --output_json /data/wjh/yoco_vl_batch_kl_nobos.json
```

No-BOS settings:

- native original prompt tokens with BOS: `[1878, 1878, 2943]`
- reference prompt tokens after stripping BOS: `[1877, 1877, 2942]`
- stripped leading BOS: `[true, true, true]`
- vLLM logits calls: `[1, 154880]`, then `[2, 154880]`
- best row permutation by KL and JS: identity `[0, 1, 2]`

No-BOS result:

| Sample | Native top1 | vLLM top1 | KL ref->vLLM | KL vLLM->ref | JS | Prob L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `32` | `32` | `0.00401018` | `0.00406391` | `0.00100410` | `0.0371630` |
| 1 | `785` | `785` | `0.00150866` | `0.00162930` | `0.000391042` | `0.0243541` |
| 2 | `785` | `785` | `0.00814846` | `0.00811508` | `0.00202408` | `0.0946776` |

No-BOS mean:

- KL ref->vLLM: `0.00455577`
- KL vLLM->ref: `0.00460276`
- JS: `0.00113974`
- probability L1: `0.0520649`
- top1 agreement: `1.0`

No-BOS logits matrix stats:

- relative Frobenius: `0.0398478`
- global cosine: `0.999209`
- mean abs diff: `0.124512`
- max abs diff: `1.0`

Compared with the previous with-BOS batch KL (`0.0196996` mean KL, `0.0754955` relative Frobenius), removing the leading BOS from the native reference is the correct comparison setup for the current vLLM string prompt path and the largest parameter/script-level alignment improvement found so far.

Additional parameter checks:

| Case | vLLM logits calls | Top1 agreement | Mean KL ref->vLLM | Mean JS | Mean Prob L1 | Logits rel Frobenius | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `no_bos`, FA2, `max_num_seqs=3` | `[1] + [2]` | `1.0` | `0.00455577` | `0.00113974` | `0.0520649` | `0.0398478` | best current setting |
| `no_bos`, requested FA3 | `[1] + [2]` | `0.666667` | `0.00752165` | `0.00188843` | `0.0743372` | `0.0463918` | current YOCO path logs that FA3 is forced back to FA2; not selected |
| `no_bos`, FA2, `max_num_seqs=1` | `[1] + [1] + [1]` | `1.0` | `0.00943777` | `0.00230064` | `0.0816208` | `0.0491178` | worse; serial vLLM scheduling does not improve alignment |

Current batch conclusion:

- Use no-BOS native reference as the reliable tensor/KL baseline against vLLM string prompts.
- Keep `--vllm_flash_attn_version 2`.
- Keep full batch scheduling (`max_num_seqs=batch_size`) for this three-sample batch.
- Row order is not the cause of the KL gap.
- The remaining difference appears to be numerical/kernel-path drift rather than an obvious prompt, image-token, or row-order mismatch. Further reduction likely requires changing vLLM internals or adding a token-id prompt path that can safely carry the exact native packed prompt.
