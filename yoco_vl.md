# YOCO-VL on vLLM 0.29

This branch adds `YOCOVLForConditionalGeneration` to
`vllm-yoco-version-0.29` (base `28d82e2e464f0464332d742140ad62453f9eddd8`).
It ports the image/video interfaces from
[buaahsh/vllm's YOCO-VL implementation](https://github.com/buaahsh/vllm/blob/wjh/vl_0806/yoco_vl.md)
to the v0.29 multimodal processor API and uses the Kimi K3 vision architecture
already included in this vLLM version.

## Architecture and weights

The composite HF configuration has `model_type: yoco_vl`, a YOCO `text_config`,
and a `KimiK3VisionConfig` under `vision_config`. The vision tower uses 27 layers,
width 1024, 12 attention heads with head dimension 128, bias-free projections,
RMSNorm, 14-pixel patches, bilinear positional interpolation, and 2x2 patch
merging. The patch-merger-v2 projector maps 4096 → 4096 → language hidden width,
with GELU and a final RMSNorm. Vision and language inference use BF16.

The 150A9B checkpoint's trained tower and projector are used; initialization
weights from `Kimi-K3-ViT-init-3072` must not replace the trained weights.
The old K2.5 tower configuration and FP32 projector assumptions do not apply.

`tools/yoco_serving/prepare_yoco_vl_hf.py` attaches the preserved trained vision
weights to an existing YOCO language HF export:

```bash
.venv/bin/python tools/yoco_serving/prepare_yoco_vl_hf.py \
  --source /path/to/0000-2900-hf \
  --output /path/to/0000-2900-hf-vl
```

The source must include `auxiliary/vision_and_mtp.safetensors` with
`vision_backbone.vision_tower.*` and `vision_backbone.vision_projector.*` keys.
The converter checks all 168 vision tensor names, shapes, finite values, and
exact equality after writing. Language shards are hardlinked without changing
values, so the destination must be on the same filesystem. Treat both exports
as immutable. MTP weights remain preserved in the source auxiliary file.

The loader accepts existing language names (`model.*`, `lm_head.*`) and maps
them under `language_model`. It maps trained projector names `proj.0` and
`proj.2` to the existing Kimi projector's `linear_1` and `linear_2`.

## Serving on two B200 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/vllm serve /path/to/0000-2900-hf-vl \
  --served-model-name Agens-150B-52M-2900 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --kv-sharing-fast-prefill \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --chat-template-content-format openai
```

The exported tokenizer includes `template_yoco_vl.jinja`. Use OpenAI chat content
parts with `type: image_url`; URLs and base64 data URLs are accepted. The template
emits `<image>`, which the processor expands into image-start, image embedding
positions, and image-end tokens. Text chat formatting is preserved. To disable
reasoning, pass `chat_template_kwargs: {"enable_thinking": false}`.

Preprocessing matches the training NaViT resize/pad rules, RGB normalization,
and patch ordering. Each image has at most 4096 visual tokens. The 32768-token
context includes images, text, and generated output. Two maximum-size images
consume 8192 visual tokens, plus boundary tokens. Smaller images retain their
smaller token counts. Video preprocessing accepts up to four frames and merges the
frames temporally; the deployment command above enables images only.

YOCO KV sharing and prefill logic inspect the nested text configuration. This
preserves the language model's existing optimized path when wrapped in a
multimodal model.

## Validation

The focused processing/configuration/KV-transfer tests cover pixel ordering,
normalization, image token budgets, image/video ordering, multimodal cache reuse,
and composite configuration loading. Runner tests include both `yoco` and
`yoco_vl` configurations. Run them with the branch's GPU runtime:

```bash
.venv/bin/python -m pytest -q \
  tests/models/multimodal/processing/test_yoco_vl.py \
  tests/model_executor/test_yoco_config.py \
  tests/v1/kv_connector/unit/test_mooncake_yoco_tail.py
.venv/bin/python -m pytest -q tests/v1/worker/test_gpu_model_runner.py -k yoco
```

On the trained 150A9B-52M step-2900 checkpoint, four images (two solid colors,
an OCR image, and a rectangular random image) produced exactly matching pixels
against the training implementation. BF16 vision feature and projector output
comparisons passed cosine > 0.995 and relative L2 < 0.1 for TP=1 and TP=2.
These are tolerance-based numerical checks, not bitwise GPU output equality.
The actual Agens tokenizer also passed 18 image/cache cases and text-template
compatibility checks with reasoning enabled and disabled.
