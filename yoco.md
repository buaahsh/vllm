# YOCO vLLM Serving Notes

## Docker Images

Current 2026-06-28 images:

- B200 / Blackwell: `buaahsh/pytorch:26.02-b200-vllm-0628`
- H100 / Hopper: `buaahsh/pytorch:26.02-h100-vllm-0628`
- A100 / Ampere: `buaahsh/pytorch:26.02-a100-vllm-0628`

## Convert Checkpoint

Use the updated converter. By default it now:

- exports YOCO router gate weights as FP32;
- writes `qk_rms_clip=true` / `qk_norm=false` when the native checkpoint uses RMSClip;
- leaves runtime quantization in BF16 (`quant_mode=bfloat16`).

```bash
cd /data/users/shaohanh/vllm
python convert_to_hf.py \
  --input_dir /path/to/merged-native-checkpoint \
  --output_dir /path/to/hf-yoco
```

No converter flag is needed for runtime FP8. If no `--quantization` flag is
passed, serving stays BF16.

Important naming note: YOCO training's `quant_mode=mxfp8` is DeepGEMM-style
block FP8: activation recipe `(1, 128)` and weight recipe `(128, 128)` when
`quant_block_size=128`, with scales rounded by `ceil_to_ue8m0`. In vLLM,
`--quantization mxfp8` means OCP MXFP8 with block size 32 and E8M0 scales. That
is a different numerical format.

The closest existing vLLM runtime format for YOCO training quantization is
`--quantization fp8_per_block`, because it uses activation blocks `(1, 128)` and
weight blocks `(128, 128)`. It is not bit-exact to YOCO training today because
vLLM's online `fp8_per_block` path uses unrounded FP32 block scales
(`use_ue8m0=False`) while YOCO training rounds scales to `ceil_to_ue8m0`.

TODO: make vLLM online `fp8_per_block` support YOCO training's scale rounding
(`ceil_to_ue8m0`) for both linear and MoE weights/activations, then rerun the
mixed5 KL acceptance with `--quantization fp8_per_block`.

## Alignment / KL Validation

The full-vocab next-token KL validation script is kept in this repo:

```bash
/data/users/shaohanh/vllm/tools/yoco_alignment/logprob_kl.py
```

Example YOCO acceptance flow after converting a checkpoint:

```bash
cd /data/users/shaohanh/vllm

# 1. Native llm-train reference logits from a merged checkpoint.
uv run --active --no-project torchrun --standalone --nproc_per_node=1 \
  tools/yoco_alignment/logprob_kl.py native \
  --model /path/to/hf-yoco \
  --native-checkpoint /path/to/merged-native-checkpoint \
  --out /path/to/results/native.pt \
  --prompt-suite mixed5 \
  --llm-train-dir /data/users/shaohanh/llm-train

# 2. vLLM BF16 logits (YOCO BF16 uses Triton MoE).
uv run --active --no-project python tools/yoco_alignment/logprob_kl.py vllm \
  --model /path/to/hf-yoco \
  --out /path/to/results/vllm-bf16.pt \
  --prompt-suite mixed5 \
  --kv-sharing-fast-prefill \
  --moe-backend triton

# 3. Compare native vs vLLM.
uv run --active --no-project python tools/yoco_alignment/logprob_kl.py compare \
  --reference /path/to/results/native.pt \
  --candidate /path/to/results/vllm-bf16.pt \
  --out-json /path/to/results/bf16-summary.json

# 4. vLLM block-FP8 logits closest to YOCO training quant_mode=mxfp8.
uv run --active --no-project python tools/yoco_alignment/logprob_kl.py vllm \
  --model /path/to/hf-yoco \
  --out /path/to/results/vllm-fp8-per-block.pt \
  --prompt-suite mixed5 \
  --kv-sharing-fast-prefill \
  --quantization fp8_per_block \
  --moe-backend triton
```

The same script also supports HF-vs-vLLM checks, for example:

```bash
uv run --active --no-project python tools/yoco_alignment/logprob_kl.py hf \
  --model Qwen/Qwen3-30B-A3B \
  --out /path/to/results/qwen-hf.pt \
  --prompt-suite mixed5 \
  --attn-implementation eager
```

## Serve BF16

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -p 8001:8001 \
  -v /data/users/shaohanh:/workspace/run \
  -w /workspace/run \
  buaahsh/pytorch:26.02-h100-vllm-0628 \
  vllm serve /workspace/run/path/to/hf-yoco \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name yoco \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --kv-sharing-fast-prefill \
    --moe-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45
```

Use the matching image for the GPU family, for example replace the image with
`buaahsh/pytorch:26.02-b200-vllm-0628` on B200.

## Serve closest training-compatible block FP8

Add `--quantization fp8_per_block` at launch time. This is the closest existing
vLLM online quantization shorthand for YOCO training's `quant_mode=mxfp8`
block-FP8 shape (`recipe_a=(1, 128)`, `recipe_b=(128, 128)`). It is still not
bit-exact unless vLLM uses the same `ceil_to_ue8m0` scale rounding as training.

Do not use `--quantization mxfp8` for training-parity YOCO runs: in vLLM that
name refers to OCP MXFP8 with block size 32 and E8M0 scales, which is not the
format used by YOCO training.

```bash
vllm serve /workspace/run/path/to/hf-yoco \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name yoco \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-sharing-fast-prefill \
  --moe-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --quantization fp8_per_block
```

## Hardware Notes

- B200 / Blackwell: can use OCP MXFP8 kernels if that format is desired, but
  YOCO training-parity serving should still use `fp8_per_block` unless the
  checkpoint/training path switches to OCP MXFP8.
- H100 / Hopper: use `fp8_per_block` for YOCO training-parity runtime FP8.
- A100 / Ampere: use BF16 by default. Do not assume MXFP8 is available in the
  A100 image unless the exact image has the newer online MXFP8/Marlin stack and
  has been tested. Operationally, treat A100 as BF16 for this model.

## Notes

- `--moe-backend triton` is intentional for BF16 YOCO. It avoids relying on the
  FlashInfer unquantized MoE backend for this model family.
- For closest YOCO training-parity runtime FP8, use
  `--quantization fp8_per_block`. `--quantization mxfp8` is only for
  vLLM/OCP-MXFP8 block-32 experiments and is not numerically aligned with YOCO
  training's `quant_mode=mxfp8`.
- `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45`
  should be included for the GLM-style tool/reasoning serving path.
- The converter output is safe to serve as BF16 without any quantization flag;
  runtime quantization is an explicit serving-time choice.