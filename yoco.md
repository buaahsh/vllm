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

No converter flag is needed for MXFP8. Use the vLLM launch-time `--quantization mxfp8`
switch instead. If `--quantization mxfp8` is not passed, serving stays BF16.

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

## Serve MXFP8

Add `--quantization mxfp8` at launch time. For MXFP8, do not use
`--moe-backend triton`: the MXFP8 MoE backend selector accepts `marlin`,
`flashinfer_trtllm`, or `xpu` (or `auto`). On H100 this local test used Marlin.

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
  --moe-backend marlin \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --quantization mxfp8
```

## Hardware Notes

- B200 / Blackwell: recommended for MXFP8. vLLM can use true MXFP8 W8A8 kernels.
- H100 / Hopper: MXFP8 can run through the Marlin MXFP8 fallback path; this was
  validated locally for next-token KL testing.
- A100 / Ampere: use BF16 by default. Do not assume MXFP8 is available in the
  A100 image unless the exact image has the newer online MXFP8/Marlin stack and
  has been tested. Operationally, treat A100 as BF16 for this model.

## Notes

- `--moe-backend triton` is intentional for BF16 YOCO. It avoids relying on the
  FlashInfer unquantized MoE backend for this model family.
- For MXFP8, use `--moe-backend marlin` or omit the backend flag and let `auto`
  pick an MXFP8-compatible backend. `--moe-backend triton` is not valid for the
  current MXFP8 MoE selector.
- `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45`
  should be included for the GLM-style tool/reasoning serving path.
- The converter output is safe to serve as BF16 without any quantization flag;
  runtime quantization is an explicit serving-time choice.