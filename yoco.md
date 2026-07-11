# YOCO vLLM 精度对齐与使用说明

本文记录 YOCO 在 vLLM 与 `llm-train` Native 实现之间已经验证有效的精度改进，
以及 B200 上 W8A8、FA4 CuTe、DeepGEMM、KV-sharing fast prefill 和 CUDA Graph
的推荐配置。

## 验证范围

本轮验证使用：

- vLLM：`/root/code2/vllm`
- llm-train：`/root/code2/llm-train`
- 原始 checkpoint：
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/updates_73750`
- 转换后的 HF 模型：
  `/mnt/pvc/lidong1/exp/agens/30A3B-36M-RMSClip3/updates_73750_hf_codex_mxfp8_router_runtime_norm`
- GPU：NVIDIA B200，SM100
- Runtime quantization：W8A8 `fp8_per_block`
- Attention：FA4 CuTe
- MoE：DeepGEMM

不要使用 `/workspace/shaohanh/vllm`。运行测试和服务前应确认当前目录和
`PYTHONPATH` 指向 `/root/code2/vllm`。

## 最终精度

使用 Native FA4 reference，与开启 FULL CUDA Graph 的 vLLM 比较完整词表的
next-token logprob：

| Case | Native -> vLLM KL | Max logprob diff |
|---|---:|---:|
| `short_hello` | `0` | `0` |
| `short_fact` | `0` | `0` |
| `medium_english` | `0` | `0` |
| `short_zh` | `0` | `0` |
| `long_zh` | `0` | `0` |
| **Mean** | **`0`** | **`0`** |

验证结果表明，当前组合不仅达到 `e-3` 量级，五个 case 的完整词表 logprob
均逐元素一致。

## 有效实现改进

### 1. FP32 embedding 和 residual stream

`llm-train` 会将 embedding output 转成 FP32，并使用 FP32 residual。vLLM 的
YOCO 路径已保持相同行为：

- embedding output 转为 FP32；
- attention/MLP 输出在 residual addition 前提升精度；
- residual hidden state 保持 FP32；
- 各层使用统一实现。

此前 attention `o_proj` same-input replay 的误差约为 `1.03e-4`，但 residual
相加后的 post-attention RMSNorm 误差会放大到约 `2.96e-3`。根因主要是
residual 输入精度，而不是 RMSNorm 公式本身。

### 2. 对齐 YOCO RMSNorm 和 RMSClip

YOCO 使用专用 RMSNorm/RMSClip 路径，对齐 Native 的：

- FP32 square accumulation 和 reciprocal RMS；
- BF16 affine weight 使用方式；
- BF16 operator boundary；
- Q/K RMSClip 配置和执行顺序。

转换脚本会根据 Native checkpoint 写入 `qk_rms_clip`、`qk_norm` 等配置。

### 3. Router gate FP32 与 row-wise L2 normalization

Router gate master weight 保持 FP32。转换脚本默认在 CUDA 上提前执行逐行 L2
normalization，并将结果写入 checkpoint，避免每次 forward 重复 normalization。

这样同时满足：

- 与 Native CUDA reduction 结果对齐；
- 删除 runtime normalization 的额外开销；
- 避免 CPU normalization 与 CUDA reduction 的细小差异。

如果转换机器没有 CUDA，可以使用 `--router-normalization runtime`，但生产模型
优先使用默认的 CUDA offline normalization。

### 4. Native-compatible top-k routing

YOCO routing 路径对齐 `llm-train` 的：

- FP32 softmax；
- `torch.topk(..., sorted=True)` expert selection；
- selected weight renormalization reduction order；
- router weight 应用位置。

这些 FP32 ULP 差异可能在后续 BF16/FP8 quant boundary 被放大，因此 routing
对齐对 `long_zh` 等长输入很重要。

### 5. DeepGEMM routed-row layout

DeepGEMM MoE 路径增加了与 Native 一致的：

- routed token permutation/grouped layout；
- psum row layout；
- router row weight 与 activation quant 融合；
- W2 前 router weight 处理。

这部分由三个运行时开关启用，当前仍需手动设置，详见后文。

### 6. B200 默认选择 FA4 CuTe

B200 的自动选择顺序已经调整为：

```text
FLASH_ATTN -> FLASHINFER -> TRITON_ATTN -> FLEX_ATTENTION
```

SM100 上 `FLASH_ATTN` 会默认选择 FlashAttention version 4，即 FA4 CuTe。
因此 B200 不再要求手动传入：

```bash
--attention-backend FLASH_ATTN
--attention-config.flash_attn_version 4
```

如果需要固定配置以防默认策略变化，仍可显式传入这两个参数。

FA4 same-input replay 已验证 Native CuTe 和 vLLM FA4 的 attention output/LSE
bitwise identical。此前的误差不是 FA4 kernel 本身导致的。

### 7. KV-sharing fast prefill 的短 query 对齐

原始 fast-prefill 会将 cross layers 缩减为 logits-only query rows。对于短输入，
这会改变 W8A8 quantization 和 MoE grouped-row layout，即使 FA4 attention output
本身完全一致，也可能产生最终 KL。

当前 YOCO 策略为：

- `max_query_len <= 8`：cross block 保留完整 token rows；
- `max_query_len > 8`：继续使用 logits-only fast-prefill；
- 其他模型保持原有默认行为。

该策略使 `short_fact` 从约 `1.09e-3` 恢复到 `0`，同时保留长输入的
fast-prefill 优化。

### 8. FULL CUDA Graph 不改变精度

已经验证以下组合：

```json
{"mode": 0, "cudagraph_mode": "FULL"}
```

其含义是：

- 关闭 torch compile/Inductor graph rewrite；
- 保留 eager operator numerics；
- 开启 FULL CUDA Graph replay。

B200 实测捕获 51 个 decode graph，graph pool 实际占用约 `0.47 GiB`。开启后
mixed5 五项 KL 仍全部为 `0`。

## 转换 checkpoint

使用仓库中的 `convert_to_hf.py`：

```bash
cd /root/code2/vllm

python convert_to_hf.py \
  --input_dir /path/to/merged-native-checkpoint \
  --output_dir /path/to/hf-yoco \
  --quant_mode mxfp8 \
  --quant_block_size 128
```

默认行为包括：

- Router gate 以 FP32 导出；
- 在 CUDA 上提前执行 router row-wise L2 normalization；
- 写入 RMSClip/QK norm 配置；
- 写入 Native quantization metadata。

`--quant_mode` 记录 checkpoint 精度元数据。vLLM serving 时是否使用 W8A8，仍由
`--quantization fp8_per_block` 决定。

CPU-only 转换环境使用：

```bash
python convert_to_hf.py \
  --input_dir /path/to/merged-native-checkpoint \
  --output_dir /path/to/hf-yoco \
  --quant_mode mxfp8 \
  --quant_block_size 128 \
  --router-normalization runtime
```

## B200 推荐启动方式

### 必需环境变量

当前三个开关仍通过 `os.getenv()` 控制，默认关闭，因此需要显式设置：

```bash
export VLLM_DEEPGEMM_MOE_PSUM_LAYOUT=1
export VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS=1
export VLLM_YOCO_COMPILED_TOPK_ROUTING=1
```

它们尚未登记到 `vllm/envs.py`，日志可能提示
`Unknown vLLM environment variable`，但代码仍会读取并生效。

必须删除旧的 FA2 prefill 开关：

```bash
unset VLLM_YOCO_NATIVE_FA2_PREFILL
```

`VLLM_YOCO_NATIVE_FA2_PREFILL=1` 会绕过 FA4 pure-prefill 路径并调用 Native
FA2，与当前 B200 默认 FA4 CuTe 的目标冲突。

### 推荐生产命令

```bash
cd /root/code2/vllm

unset VLLM_YOCO_NATIVE_FA2_PREFILL
export VLLM_DEEPGEMM_MOE_PSUM_LAYOUT=1
export VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS=1
export VLLM_YOCO_COMPILED_TOPK_ROUTING=1

vllm serve /path/to/hf-yoco \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name yoco \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --quantization fp8_per_block \
  --kv-sharing-fast-prefill \
  --moe-backend deep_gemm \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'
```

预期日志应包含：

```text
Using FLASH_ATTN attention backend
Using FlashAttention version 4
Using DEEPGEMM Fp8 MoE backend
Capturing CUDA graphs (decode, FULL)
```

### 显式固定 FA4

默认不需要，但需要完全固定 backend 时可以额外添加：

```bash
--attention-backend FLASH_ATTN \
--attention-config.flash_attn_version 4
```

### Eager 排障模式

排查 kernel 或 CUDA Graph 问题时，可以临时移除 `--compilation-config` 并添加：

```bash
--enforce-eager
```

不要同时使用 `--enforce-eager` 和 FULL CUDA Graph 配置。

## KL 精度验证

### 1. 生成 Native FA4 reference

Native 必须真正调用 CuTe。使用 `--native-no-kv-cache`，避免 Native KV-cache
路径绕过 CuTe attention：

```bash
cd /root/code2/vllm

CUDA_VISIBLE_DEVICES=1 \
python tools/yoco_alignment/logprob_kl.py native \
  --model /path/to/hf-yoco \
  --native-checkpoint /path/to/merged-native-checkpoint \
  --llm-train-dir /root/code2/llm-train \
  --native-quant-mode mxfp8 \
  --native-quant-block-size 128 \
  --native-use-cute \
  --native-no-kv-cache \
  --prompt-suite mixed5 \
  --out /tmp/yoco_native_fa4_mixed5.pt
```

日志应显示非零的 CuTe 调用次数，例如：

```text
[native-kl] flash_attn.cute calls=200
```

### 2. 生成 vLLM FULL CUDA Graph 结果

```bash
cd /root/code2/vllm

unset VLLM_YOCO_NATIVE_FA2_PREFILL
export VLLM_DEEPGEMM_MOE_PSUM_LAYOUT=1
export VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS=1
export VLLM_YOCO_COMPILED_TOPK_ROUTING=1

CUDA_VISIBLE_DEVICES=1 \
python tools/yoco_alignment/logprob_kl.py vllm \
  --model /path/to/hf-yoco \
  --out /tmp/yoco_vllm_fa4_cudagraph_mixed5.pt \
  --prompt-suite mixed5 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --quantization fp8_per_block \
  --kv-sharing-fast-prefill \
  --moe-backend deep_gemm \
  --compilation-config-json '{"mode":0,"cudagraph_mode":"FULL"}'
```

### 3. 比较完整词表 KL

```bash
python tools/yoco_alignment/logprob_kl.py compare \
  --reference /tmp/yoco_native_fa4_mixed5.pt \
  --candidate /tmp/yoco_vllm_fa4_cudagraph_mixed5.pt \
  --out-json /tmp/yoco_fa4_cudagraph_compare.json \
  --top-k 20
```

目标是每个 case 至少达到 `e-3` 量级。当前已验证结果为五项 KL 全部为 `0`。

## FA4 same-input replay

需要确认 attention kernel 本身时，使用：

```bash
python tools/yoco_alignment/replay_fa4_same_input.py --help
```

该工具比较：

- Native `flash_attn.cute`；
- vLLM public FA4 interface；
- vLLM vendored FA4 interface；
- split 0/1 和 GQA packing variants；
- output 和 LSE；
- full-query last row 与 single-query/full-KV。

`short_fact` 和 `long_zh` 的 40 次 attention calls 已验证 bitwise identical。

## 常见误区

- 不要设置 `VLLM_YOCO_NATIVE_FA2_PREFILL=1`，否则不会走完整 FA4 prefill。
- 不要遗漏三个 DeepGEMM/routing 环境变量；当前默认值仍是关闭。
- 不要用默认 torch compile 配置做严格 parity；使用
  `mode=0,cudagraph_mode=FULL`。
- 不要仅比较生成文本或 top-1 token；应比较完整词表 next-token logprob。
- `Using FlashInfer for top-p & top-k sampling` 只表示 sampler 使用 FlashInfer，
  不代表 attention backend 是 FlashInfer。
- W8A8 应使用 `--quantization fp8_per_block --moe-backend deep_gemm`，不应进入
  `deep_gemm.fp8_fp4_gemm_nt`。
- 如果短输入在开启 fast-prefill 后单独退化，优先检查 cross-block token row
  layout，而不是先修改 FA4 attention kernel。

## 相关文件

```text
convert_to_hf.py
tools/yoco_alignment/logprob_kl.py
tools/yoco_alignment/replay_fa4_same_input.py
vllm/model_executor/models/yoco.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
vllm/v1/attention/backends/fa_utils.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/utils.py
vllm/v1/worker/gpu_model_runner.py
vllm/platforms/cuda.py
```
