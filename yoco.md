# YOCO vLLM B200 对齐与运行说明

本文记录 YOCO-30B-A3B 在 B200 上与
`/workspace/shaohanh/llm-train` 对齐后的实现、验证结果和推荐运行方式。
对应代码分支为 `shaohanh/yoco-0716`，容器镜像为
`buaahsh/pytorch:26.02-b200-vllm-0716`。

## 已验证模型

- Native checkpoint:
  `/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000`
- nnScaler merged checkpoint:
  `/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged`
- GPU 转换后的 HF checkpoint:
  `/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu`

HF checkpoint 共 21 个文件，大小为 `64,477,415,770` bytes；从本地转换目录
复制到上述路径后，所有文件均通过 SHA256 校验。

旧的 `0000-6000-hf/config.json` 在文件末尾缺少 `}`，不要直接用于验收。

## 对齐原则

KL 只有在 vLLM 和 llm-train Native 使用相同计算条件时才有意义。每次对齐前
必须先固定并记录以下因素：

1. **精度模式一致**：MXFP8 对齐要求两侧都使用 MXFP8；BF16 对齐要求两侧
   都使用 BF16，不能用 MXFP8 vLLM 对比 BF16 Native。
2. **量化配置一致**：MXFP8 两侧都使用 128-element block。Native 使用
   `quant_mode=mxfp8`、`quant_block_size=128`；vLLM 使用
   `--quantization fp8_per_block`。Native 的 torch activation quant fallback
   只是替换不稳定的 Triton 实现，不改变 MXFP8 数值格式。
3. **Attention 一致**：FA2 只能和 FA2 reference 比较；FA4 只能和 FA4
   reference 比较。当前正式验收矩阵使用 FA2，FA4 matched matrix 仍是 TODO。
4. **执行形状一致**：batch size、scheduler forward shape、prompt 顺序、
   chunk 切分位置和 KV-cache 语义必须一致。batch 16 当前使用与 vLLM
   scheduler 一致的 `1 + 15` Native forward shape。
5. **并行和功能开关一致**：TP、EP、KV-sharing fast prefill、chunked
   prefill、CUDA Graph 范围都必须逐项匹配，不能把不同配置的结果混在同一
   个 KL 结论中。

## 当前结论

### 推荐生产配置

以下配置不使用 eager，并同时满足 MXFP8 batch prefill 和稳定 decoding：

- `--max-num-seqs 16`
- `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'`
- `--quantization fp8_per_block`
- `--moe-backend deep_gemm`
- compilation config:

```json
{
  "mode": 0,
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [1, 2, 4, 8, 16]
}
```

prefill 继续使用 FA2。CUDA Graph 单 token decode 在 FA2 backend 内切换到
Triton 2D paged attention，避免 FA2 graph replay 的 KV metadata 错误。

| 验证矩阵 | Native -> vLLM mean KL | 结论 |
| --- | ---: | --- |
| MXFP8，batch 1 | `0` | 完全一致 |
| MXFP8，batch 16 | `0.00758594` | 通过 `< 0.01` |
| BF16，batch 1 | `0.00550893` | 通过 `< 0.01`，但不是 exact |
| BF16，batch 16 | `0.00414718` | 通过 `< 0.01` |
| MXFP8，KV-sharing fast prefill，batch 16 | `0.00735566` | 通过 `< 0.01` |
| MXFP8，TP2 + EP2，batch 16 | `0.00898775` | 通过 `< 0.01` |

BF16 batch 1 当前 mean KL 为 `0.00550893`，尚未达到逐元素一致。主要差异是
Native 使用 DeepGEMM grouped BF16 routed MoE，而 vLLM 使用 Triton BF16 MoE。

MXFP8、BF16、KV-sharing fast prefill 和 TP2/EP2 均已在
`FULL_DECODE_ONLY` 下完成单请求及多请求 greedy decode，没有乱码、连续首
token 重复或单 token collapse。最终容器使用四个中英文请求并发生成 16
tokens，也得到连续、可读的输出。

合并 `shaohanh/yoco-on-0.23` 后重新构建同名镜像并复测，MXFP8 batch 16
仍为 `0.00758594`。启用 Agens parser 后，四个中英文并发 Chat 请求输出
连续可读，Responses API 也可正常返回；parser 合并未改变 YOCO logits。

### 尚未通过的矩阵

- 真实 chunked prefill：110-token 输入按 `64 + 46` 分块，并与同样分块的
  Native KV-cache reference 比较时 KL 为 `0.0279867`；约 4.7K-token 输入
  按 `4096 + remainder` 分块时为 `0.0647879`。因此可以开启
  `--enable-chunked-prefill`，但不能把真正发生切分的长 prompt 标记为已对齐。
- BF16 batch 1：mean KL `0.00550893`，不是 exact zero。
- FA4：仍是低优先级矩阵；FULL graph replay 对 YOCO 不安全。

### Batch invariance

batch 大于 1 时不要求逐元素一致，验收标准是完整词表 aggregate mean KL
小于 `1e-2`，同时 decoding 不重复、不乱码。Native reference 使用和 vLLM
scheduler 一致的 `1 + 15` forward shape。

## TODO 与当前对齐程度

- [ ] **FA4 matched matrix**
    - 当前状态：未验收。FA4 在 YOCO full CUDA Graph 下会导致 self-attention
    和共享 cross-attention replay 错误，生产配置因此使用 FA2。
    - 下一步：确保 vLLM 和 llm-train 同时使用 FA4，在 eager 和 graph
    decode 中分别比较完整词表 KL；不能用 Native FA4 对比 vLLM FA2。
- [ ] **Batch invariance**
    - 当前状态：已达到 aggregate 验收线，但不是 exact。MXFP8 batch 16 mean
    KL 为 `0.00758594`，BF16 batch 16 mean KL 为 `0.00414718`。
    - 下一步：继续降低 scheduler shape 和 packed-row geometry 导致的差异，
    并验证更多 batch size；所有 Native reference 必须复现 vLLM 的实际
    forward shape。
- [ ] **真实 chunked prefill**
    - 当前状态：未达到 `< 0.01`。110-token prompt 在两侧都按 `64 + 46`
    切分时 KL 为 `0.0279867`；约 4.7K-token prompt 在两侧都按
    `4096 + remainder` 切分时 KL 为 `0.0647879`。
    - 下一步：定位 cache-backed prefill 中 attention 与 MoE shape drift；
    在通过前，开启 `--enable-chunked-prefill` 不等于真实切分路径已对齐。
- [ ] **BF16 batch 1 exact**
    - 当前状态：mean KL 为 `0.00550893`，满足 `< 0.01`，但未达到 exact zero。
    - 下一步：实现与 Native grouped DeepGEMM BF16 routed MoE 等价的路径。

### CUDA Graph 状态

已验收配置：

```json
{
  "mode": 0,
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [1, 2, 4, 8, 16]
}
```

已确认不安全的组合：

- FA4 + full CUDA Graph：self-attention 和共享 cross-attention replay 错误。
- 不包含当前 FA2-backend Triton decode 修复的旧代码：FA2
  `FULL_DECODE_ONLY` 会重复首 token。

## 实现要点

### Router

- Router gate 使用 FP32 TF32 GEMM，与 llm-train 一致；
- routing 使用固定 geometry 的 Native 等价 Triton dense graph：
  `softmax -> topk -> renorm -> scatter`，不依赖 Inductor autotune cache；
- routing probabilities 在 W2 activation quantization 前应用。

### RMSNorm

- residual 和 reduction 使用 FP32；
- affine weight 按 BF16 operator boundary 读取；
- token rows 少于 128 时使用 2048 reduction block；
- token rows 至少 128 时使用 4096 reduction block。

### DeepGEMM

- YOCO 自动启用 psum layout 和 W2 前 routed-row weighting；
- eager、非 compile 路径使用真实 active expert row count；
- graph capture 保留静态安全上界。
- EP 下将非本地 expert 的 inverse permutation 初始化为 `-1`，并在
  routed-row weight scatter 时检查 row bounds；这修复了 TP2/EP2 profile 的
  illegal memory 和错误权重写入。

以下旧环境变量不再需要，最终命令不应设置它们：

```text
VLLM_DEEPGEMM_MOE_PSUM_LAYOUT
VLLM_DEEPGEMM_MOE_FUSED_ROW_WEIGHTS
VLLM_YOCO_COMPILED_TOPK_ROUTING
```

## GPU 转换 checkpoint

从 merged checkpoint 转换：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python convert_to_hf.py \
  --input_dir /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged \
  --output_dir /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --quant_mode mxfp8 \
  --quant_block_size 128
```

默认会在 CUDA 上执行 router row-wise L2 normalization，并写入
`qk_rms_clip`、`qk_rms_limit`、`swiglu_limit` 和 quantization metadata。

## 推荐生产启动命令

### 直接运行当前仓库

```bash
vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name yoco \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --quantization fp8_per_block \
  --moe-backend deep_gemm \
  --reasoning-parser agens \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

BF16 使用相同 graph/attention 配置，但去掉 `--quantization`，并改为
`--moe-backend triton`。

### Agens reasoning 与 tool parser

Agens 模型需要同时启用新增的 reasoning parser 和 tool parser：

```text
--reasoning-parser agens
--enable-auto-tool-choice
--tool-call-parser agens
```

`agens` reasoning parser 基于 DeepSeek V3 thinking parser，并将 streaming
reasoning 输出到兼容 CCR 的 `reasoning_content` 字段。`agens` tool parser
基于 GLM-4.7 parser，会合并同一个 tool-call index 在单个 delta 内拆开的
function name 和 arguments。

这些 parser 只处理服务层输出，不参与模型 forward、KV cache、sampling 或
logits 计算，因此不会改变本文件记录的 prefill KL 和 decoding 数值对齐结果。

### 运行发布镜像

```bash
docker run --rm \
  --device nvidia.com/gpu=5 \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8001:8001 \
  -v /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu:/model:ro \
  buaahsh/pytorch:26.02-b200-vllm-0716 \
  vllm serve /model \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name yoco \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --quantization fp8_per_block \
    --moe-backend deep_gemm \
    --reasoning-parser agens \
    --enable-auto-tool-choice \
    --tool-call-parser agens \
    --attention-config \
      '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
    --compilation-config \
      '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

## 完整词表 KL 验证

Native MXFP8 必须使用 torch activation quant fallback；训练侧 Triton
activation quant kernel 在短 prompt 上可能触发 illegal memory。batch 16
reference 使用 `1 + 15` forward shape：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python \
  tools/yoco_alignment/logprob_kl.py native \
  --model /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --native-checkpoint /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged \
  --llm-train-dir /workspace/shaohanh/llm-train \
  --native-quant-mode mxfp8 \
  --native-quant-block-size 128 \
  --native-use-torch-fp8-quant \
  --prompt-suite mixed16 \
  --first-batch-size 1 \
  --batch-size 15 \
  --out /tmp/yoco-native-mxfp8-mixed16.pt
```

非 eager vLLM batch 16：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python \
  tools/yoco_alignment/logprob_kl.py vllm \
  --model /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --max-num-seqs 16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --quantization fp8_per_block \
  --moe-backend deep_gemm \
  --attention-backend FLASH_ATTN \
  --flash-attn-version 2 \
  --compilation-config-json \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --out /tmp/yoco-vllm-mxfp8-mixed16.pt
```

比较：

```bash
.venv/bin/python tools/yoco_alignment/logprob_kl.py compare \
  --reference /tmp/yoco-native-mxfp8-mixed16.pt \
  --candidate /tmp/yoco-vllm-mxfp8-mixed16.pt \
  --model /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --out-json /tmp/yoco-compare-mxfp8-mixed16.json
```

`logprob_kl.py` 只检查 next-token 分布；还必须执行至少 8 tokens 的单/多请求
greedy decoding。

TP2/EP2 验证在 vLLM 命令中追加：

```text
--tensor-parallel-size 2 --enable-expert-parallel
```

KV-sharing fast prefill 验证追加：

```text
--kv-sharing-fast-prefill
```

该配置的 MXFP8 batch 16 mean KL 为 `0.00735566`，graph decode 正常。真实
chunked prefill 尚未通过，不能仅凭 `--enable-chunked-prefill` 启动成功判定
对齐。

## 构建 B200 image

`docker/Dockerfile.b200` clone 固定 commit，并 overlay 当前 YOCO Python 实现和
对齐工具：

```bash
docker build \
  -f docker/Dockerfile.b200 \
  -t buaahsh/pytorch:26.02-b200-vllm-0716 \
  .
```

该 Dockerfile 保留 `donglixp/pytorch:26.02-b200` 中的 Python、PyTorch 和
CUDA 环境，只在固定的 vLLM 基线提交上覆盖本次需要的 Python runtime 文件。

## 相关文件

```text
convert_to_hf.py
tools/yoco_alignment/logprob_kl.py
vllm/model_executor/models/yoco.py
vllm/model_executor/models/config.py
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/v1/attention/backends/flash_attn.py
docker/Dockerfile.b200
```
