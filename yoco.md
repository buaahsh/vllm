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
  reference 比较。对齐 probe 两侧都传入 `--force-fa-num-splits-one`；Native
  FA2 varlen prefill 没有 split API，本身走 non-split path，Native FA4 和 vLLM
  FA2/FA4 则显式固定 `num_splits=1`。
4. **执行形状一致**：batch size、scheduler forward shape、prompt 顺序、
  chunk 切分位置和 KV-cache 语义必须一致。alignment probe 在导入 vLLM 前
  固定 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，使请求全部入队后再调度；batch 16
  在 Native 和 vLLM 两侧都使用一个 16-request forward，不再模拟 `1 + 15`。
  probe 同时默认关闭 prefix caching，避免 `mixed16` 中重复 prompt 被缓存后
  改变实际 token-row geometry。
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

当前正式生产配置的 prefill 继续使用 FA2。CUDA Graph 单 token decode 在
FlashAttention backend 内切换到 Triton 2D paged attention，避免 graph replay
的 KV metadata 错误。`FULL_DECODE_ONLY + flash_attn_version=4` 现在保留 eager
FA4 prefill，并同样只把 single-token graph decode 切到 Triton；该组合仍需重跑
完整验收矩阵。

下表是修改前 V1 multiprocessing 和 `1 + 15` reference shape 的历史 baseline，
不能直接作为新 single-batch/one-split 配置的验收结果：

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
- FA4：`FULL` graph prefill 仍不安全；`FULL_DECODE_ONLY` 的 eager FA4 prefill
  + Triton graph decode 尚待完整验收。

### Batch invariance

batch 大于 1 时不要求逐元素一致，验收标准是完整词表 aggregate mean KL
小于 `1e-2`，同时 decoding 不重复、不乱码。新 probe 关闭 V1 engine-core
multiprocessing，并在 Native/vLLM 两侧使用相同的单个 batch forward。旧的
`1 + 15` 数值只保留为历史 baseline。

2026-07-21 使用同一个 16-request `mixed16` 输入做了 scheduler A/B。为避免
占用正在运行的 30B GPU workload，实验使用 1-layer tiny Llama 和 YOCO tokenizer；
EngineCore、scheduler、request queue 和 model-runner forward loop 与正式模型相同。
两侧都关闭 prefix caching，artifact 均包含 16 个 request、共 582 个 prompt
tokens：

| V1 EngineCore mode | 实际 context forwards |
| --- | --- |
| multiprocessing | `1 request / 3 tokens`，随后 `15 requests / 579 tokens` |
| in-process | 单次 `16 requests / 582 tokens` |

两个 arm 的 16 个完整词表输出逐元素一致，双向 KL、JS、max/mean logprob diff
全部为 `0`。这证明旧 `1 + 15` 是 EngineCore process boundary 下请求边到达边调度
造成的，而关闭 V1 multiprocessing 后，16 个已提交请求进入同一个实际 prefill
forward。该实验只证明 scheduler shape；30B YOCO 的新 one-split/single-batch KL
仍需在 GPU 空闲后重跑。

原始证据保存在 `../logs/yoco_alignment_results/v1_batch_shape_ab_20260721/`：

```text
vllm_batch_shape_mp.log
vllm_batch_shape_mp.pt
vllm_batch_shape_inproc.log
vllm_batch_shape_inproc.pt
vllm_batch_shape_ab_final.json
vllm_batch_shape_logits_ab.json
```

`tools/yoco_alignment/verify_v1_batch_shape.py` 会将 iteration log 与 artifact 中的
request/prompt-token 总数交叉验证；传入新的 Native artifact 时，还会检查其
`model_forwards` 是否只有一个匹配的 16-request/582-token prefill。

## TODO 与当前对齐程度

- [ ] **FA4 matched matrix**
  - 当前状态：未验收。`FULL` graph prefill 仍强制回退到 TRITON_ATTN；
  `FULL_DECODE_ONLY` 保留 eager FA4 prefill，并把 single-token graph decode
  路由到 Triton。
  - 下一步：确保 vLLM 和 llm-train 同时使用 FA4 和 `num_splits=1`，分别比较
  eager prefill 与 decode；不能用 Native FA4 对比 vLLM FA2。
- [ ] **Batch invariance**
  - 当前状态：旧 `1 + 15` baseline 已达到 aggregate 验收线，但新的
  single-batch/one-split matrix 尚未重跑。
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

- FA4 + `FULL` CUDA Graph prefill：self-attention 和共享 cross-attention replay
  错误；`FULL_DECODE_ONLY` 不 graph prefill，不属于该组合。
- 不包含当前 FA2-backend Triton decode 修复的旧代码：FA2
  `FULL_DECODE_ONLY` 会重复首 token。

`FA_CLC=0` 已是 Native/vLLM FA4 默认值。`FA_DISABLE_2CTA=1` 不影响 YOCO
forward：两个 FA4 实现都只在 non-causal、non-local、non-varlen forward 中选择
2CTA，而 YOCO 使用 causal varlen/paged attention，因此不把这两个环境变量作为
对齐条件。

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

### FA2 可复现实验脚本

`run_logprob_kl_prefill_ablation_fa2.sh` 固化 BF16/MXFP8 的四个 prefill arm：

- A：V1 multiprocessing 开、prefix cache 开、heuristic splits、Native `1 + 15`；
- B：V1 multiprocessing 开、prefix cache 关、heuristic splits、Native `1 + 15`；
- C：V1 multiprocessing 关、prefix cache 关、heuristic splits、Native batch 16；
- D：V1 multiprocessing 关、prefix cache 关、`num_splits=1`、Native batch 16。

```bash
CUDA_VISIBLE_DEVICES=5 \
tools/yoco_alignment/run_logprob_kl_prefill_ablation_fa2.sh \
  --model /path/to/0000-6000-hf-gpu \
  --native-checkpoint /path/to/0000-6000-merged \
  --variants bf16,mxfp8 \
  --arms A,B,C,D
```

脚本为每个 arm 保存 Native/vLLM artifact、iteration log 和 comparison JSON；
B/C 同时调用 `verify_v1_batch_shape.py`，将实际 scheduler request/token 数与
artifact 及 Native `model_forwards` 交叉验证。使用 `--stages` 可拆分 Native、
vLLM、compare 和 verify，`--dry-run` 可只打印完整命令。

Native MXFP8 必须使用 torch activation quant fallback；训练侧 Triton
activation quant kernel 在短 prompt 上可能触发 illegal memory。probe 关闭
vLLM V1 engine-core multiprocessing 和 prefix caching，batch 16 两侧都使用
一个 16-request forward：

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
  --batch-size 16 \
  --force-fa-num-splits-one \
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
  --no-v1-multiprocessing \
  --no-enable-prefix-caching \
  --log-iteration-details \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --quantization fp8_per_block \
  --moe-backend deep_gemm \
  --attention-backend FLASH_ATTN \
  --flash-attn-version 2 \
  --force-fa-num-splits-one \
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

### Decode path 与 teacher-forced prefill 对齐

`run_logprob_kl_decode_fa2.sh` 固化 `mixed16`、batch 16、FA2、
`FULL_DECODE_ONLY`、V1 in-process、prefix cache 关闭和 `num_splits=1`。先运行
16-token smoke，再运行 128-token acceptance：

```bash
CUDA_VISIBLE_DEVICES=5 \
tools/yoco_alignment/run_logprob_kl_decode_fa2.sh \
  --model /path/to/0000-6000-hf-gpu \
  --native-checkpoint /path/to/0000-6000-merged \
  --variants bf16,mxfp8 \
  --lengths 16

CUDA_VISIBLE_DEVICES=5 \
tools/yoco_alignment/run_logprob_kl_decode_fa2.sh \
  --model /path/to/0000-6000-hf-gpu \
  --native-checkpoint /path/to/0000-6000-merged \
  --variants bf16,mxfp8 \
  --lengths 128
```

默认输出目录为
`../logs/yoco_alignment_results/fa2_decode_mixed16/`。每个 precision/length
cell 保存 rollout、Native replay、两个 stage log 和 comparison JSON。Native
stage 关闭 KV cache，执行训练式 packed teacher-forced prefill，并强制要求真实
TransformerEngine；可用 `--stages vllm`、`native`、`compare` 分机器或分时段
恢复运行。

#### 128-token FA2 matrix 结果

2026-07-21 已完成 BF16 和 MXFP8 matrix。下表只统计 position 2 及以后，即真实
vLLM KV-cache decode path；position 1 的 prompt-prefill boundary 单独报告。

| metric（true decode only） | BF16 | MXFP8 |
| --- | ---: | ---: |
| compared positions | `1501` | `1651` |
| mean signed logprob diff | `0.000236` | `0.003281` |
| mean absolute diff | `0.020895` | `0.020023` |
| median absolute diff | `0.007037` | `0.002840` |
| p95 absolute diff | `0.084972` | `0.090883` |
| p99 absolute diff | `0.156773` | `0.165311` |
| max absolute diff | `0.350685` | `0.399515` |
| RMS diff | `0.040821` | `0.042353` |
| mean `abs(exp(logdiff) - 1)` | `0.021072` | `0.020368` |
| max `abs(exp(logdiff) - 1)` | `0.420040` | `0.491102` |

selected-token worst case：

- BF16：`short_zh_2`，position `7`，token id `98448`（`三`），
  `vLLM - Native = +0.350685`；
- MXFP8：`short_fact_2`，position `117`，token id `32819`（` vibrant`），
  `vLLM - Native = +0.399515`。

完整词表使用 Native 作为 reference，按 response position 聚合：

| position | execution path | BF16 active / mean KL / mean JS | MXFP8 active / mean KL / mean JS |
| ---: | --- | ---: | ---: |
| 1 | prefill boundary | `16 / 0.007586 / 0.001914` | `16 / 0.011668 / 0.002998` |
| 2 | decode | `13 / 0.003073 / 0.000776` | `13 / 0.003004 / 0.000751` |
| 4 | decode | `13 / 0.001517 / 0.000379` | `13 / 0.001655 / 0.000408` |
| 8 | decode | `13 / 0.003437 / 0.000876` | `13 / 0.009210 / 0.002305` |
| 16 | decode | `13 / 0.001126 / 0.000278` | `13 / 0.000142 / 0.000036` |
| 32 | decode | `13 / 0.001128 / 0.000282` | `13 / 0.004411 / 0.001116` |
| 48 | decode | `13 / 0.001905 / 0.000479` | `13 / 0.001816 / 0.000454` |
| 64 | decode | `13 / 0.001469 / 0.000367` | `13 / 0.000910 / 0.000227` |
| 80 | decode | `10 / 0.000650 / 0.000163` | `13 / 0.001157 / 0.000293` |
| 96 | decode | `10 / 0.002045 / 0.000535` | `13 / 0.001998 / 0.000474` |
| 112 | decode | `10 / 0.000079 / 0.000020` | `13 / 0.003984 / 0.001020` |
| 128 | decode | `10 / 0.000632 / 0.000159` | `13 / 0.001715 / 0.000436` |

所有 true-decode sparse checkpoints 的 mean Native→vLLM KL 都小于 `0.01`，且
没有随 decode length 单调累积。MXFP8 position 1 为 `0.0116682`，超过 `0.01`；
这属于 prompt-prefill boundary，应与 decode path 结论分开。跨全部 sparse
positions（包括 position 1）的 mean Native→vLLM KL 为 BF16 `0.00226508`、
MXFP8 `0.00362708`。

response coverage：

| precision | min / mean / median / max length | reached 128 | EOS/stop early |
| --- | ---: | ---: | ---: |
| BF16 | `1 / 94.8125 / 128 / 128` | `10/16` | `6/16` |
| MXFP8 | `1 / 104.1875 / 128 / 128` | `13/16` | `3/16` |

结果文件：

- BF16：[comparison JSON](../logs/yoco_alignment_results/fa2_decode_mixed16/bf16_decode128_fa2_compare.json)、
  [vLLM rollout](../logs/yoco_alignment_results/fa2_decode_mixed16/bf16_decode128_fa2_vllm.pt)、
  [Native replay](../logs/yoco_alignment_results/fa2_decode_mixed16/bf16_decode128_fa2_native.pt)、
  [vLLM log](../logs/yoco_alignment_results/fa2_decode_mixed16/bf16_decode128_fa2_vllm.log)、
  [Native log](../logs/yoco_alignment_results/fa2_decode_mixed16/bf16_decode128_fa2_native.log)；
- MXFP8：[comparison JSON](../logs/yoco_alignment_results/fa2_decode_mixed16/mxfp8_decode128_fa2_compare.json)、
  [vLLM rollout](../logs/yoco_alignment_results/fa2_decode_mixed16/mxfp8_decode128_fa2_vllm.pt)、
  [Native replay](../logs/yoco_alignment_results/fa2_decode_mixed16/mxfp8_decode128_fa2_native.pt)、
  [vLLM log](../logs/yoco_alignment_results/fa2_decode_mixed16/mxfp8_decode128_fa2_vllm.log)、
  [Native log](../logs/yoco_alignment_results/fa2_decode_mixed16/mxfp8_decode128_fa2_native.log)。

结论：BF16/MXFP8 的 distributional decode alignment 都通过当前 sparse checkpoint
mean-KL `< 0.01` 标准；selected-token tail error 仍然明显，不能只看 aggregate KL。

Native artifact 曾在 forward 完成后因 TransformerEngine status 未从 model loader
返回而触发 `NameError`；`_load_native_model` 现显式返回该 boolean。使用
`--stages native,compare` 可从已有 rollout 恢复，无需重新运行 vLLM generation。

`logprob_kl.py` 只检查 prompt 边界的 next-token 分布。`logprob_kl_decode.py`
先让 vLLM 通过真实 KV-cache decode path 生成 rollout，再让 Native 对完全相同的
token IDs 做一次 packed teacher-forced prefill。vLLM batch size 可配置；生产矩阵
使用 `--batch-size 16 --max-num-seqs 16`：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python \
  tools/yoco_alignment/logprob_kl_decode.py vllm \
  --model /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --out /tmp/yoco-vllm-decode-mxfp8.pt \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --decode-length 64 \
  --vocab-logprob-stride 16 \
  --max-num-seqs 16 \
  --no-enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --quantization fp8_per_block \
  --moe-backend deep_gemm \
  --attention-backend FLASH_ATTN \
  --flash-attn-version 2 \
  --force-fa-num-splits-one \
  --compilation-config-json \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

rollout artifact 保存 prompt/response 文本、精确 token IDs、每个 sampled token 的
raw model logprob、完整 engine/sampling/batch 配置，以及 sparse response
positions 的完整词表 logprobs。FA2 runner 使用 `1,2,4,8` 和之后每 16 tokens
一个 checkpoint，128-token run 因此记录
`1,2,4,8,16,32,48,64,80,96,112,128`。response position 使用 one-based 编号；
position 1 来自 prompt prefill，position 2 及以后来自真实 vLLM decode forward。
EOS 会提前结束单条 response，其余请求继续到 `--decode-length`。

Native replay：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 1 \
  tools/yoco_alignment/logprob_kl_decode.py native \
  --rollout /tmp/yoco-vllm-decode-mxfp8.pt \
  --out /tmp/yoco-native-decode-mxfp8.pt \
  --native-checkpoint /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged \
  --llm-train-dir /workspace/shaohanh/llm-train \
  --native-quant-mode mxfp8 \
  --native-quant-block-size 128 \
  --native-use-torch-fp8-quant \
  --native-local-attention \
  --force-fa-num-splits-one \
  --max-model-len 8192
```

比较：

```bash
.venv/bin/python tools/yoco_alignment/logprob_kl_decode.py compare \
  --vllm /tmp/yoco-vllm-decode-mxfp8.pt \
  --native /tmp/yoco-native-decode-mxfp8.pt \
  --model /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --out-json /tmp/yoco-decode-compare-mxfp8.json
```

JSON 同时报告每个 response position 的
`vLLM selected logprob - Native selected logprob`、mean/RMS/max 和
p50/p95/p99 absolute diff、`|exp(logdiff)-1|` ratio distortion、response-length
coverage，以及固定 position 的完整词表双向 KL/JS。`decode_only_positions` 排除
position 1，避免把 prompt prefill 边界混入 decode path 结论。vLLM 返回的是
sampling transform 之前的 `raw_logprobs`；temperature/top-p/top-k 只决定 rollout
token，不改变与 Native 比较的分布。

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

### 快速迭代

`Dockerfile.b200` 已将耗时的 vLLM 安装放在 Python overlay 之前。同一台机器上
只修改 overlay 清单内的 Python 文件时，直接重复执行上面的 `docker build`
会命中 native extension 和依赖安装缓存，只重新执行末尾的 `COPY` 和镜像导出。
2026-07-16 的 reasoning 修复重建中，所有安装层均为 `CACHED`，没有重新编译
CUDA/C++；约 154 秒主要消耗在导出 30.9 GB 本地镜像。

开发阶段可以完全跳过 build，将单个改动文件 bind mount 到已有镜像。例如在
下文“运行发布镜像”的 `docker run` 命令中额外加入：

```bash
-v "$PWD/vllm/entrypoints/openai/chat_completion/protocol.py:/workspace/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:ro" \
-v "$PWD/vllm/parser/agens_parser.py:/workspace/vllm/vllm/parser/agens_parser.py:ro"
```

容器内使用 editable install，因此重新创建容器后会直接加载挂载的 Python
文件。不要挂载整个本地 `vllm/` 到 `/workspace/vllm/vllm/`，否则会遮住镜像
内已经编译好的 `_C*.so` 等 native extension。修改 C++、CUDA、构建依赖或
Dockerfile 安装步骤时仍必须完整重建。

发布 Python-only 改动时可以让 BuildKit 直接推送，避免先将完整镜像导出到本地
Docker image store：

```bash
docker buildx build \
  --progress=plain \
  --push \
  -f docker/Dockerfile.b200 \
  -t buaahsh/pytorch:26.02-b200-vllm-0716 \
  .
```

如果需要在不同机器或 CI 之间复用编译缓存，使用支持 registry cache 的
`docker-container` builder。第一次仍需完整构建，之后可从 Docker Hub 恢复
缓存：

```bash
docker buildx create \
  --name yoco-b200-builder \
  --driver docker-container \
  --use
docker buildx inspect --bootstrap

docker buildx build \
  --progress=plain \
  --cache-from type=registry,ref=buaahsh/pytorch:26.02-b200-vllm-0716-buildcache \
  --cache-to type=registry,ref=buaahsh/pytorch:26.02-b200-vllm-0716-buildcache,mode=max \
  --push \
  -f docker/Dockerfile.b200 \
  -t buaahsh/pytorch:26.02-b200-vllm-0716 \
  .
```

进一步降低发布延迟时，可以将固定 commit 的完整编译结果发布为不可变
`vllm-base` tag，再用只包含 Python `COPY` 的薄 overlay image 作为最终 tag。
这样 Python-only 发布不会再次经过 vLLM 安装阶段，也不依赖构建机的本地缓存。

## 相关文件

```text
convert_to_hf.py
tools/yoco_alignment/logprob_kl.py
vllm/model_executor/models/yoco.py
vllm/model_executor/models/config.py
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/v1/attention/backends/flash_attn.py
vllm/entrypoints/openai/chat_completion/protocol.py
vllm/parser/agens_parser.py
docker/Dockerfile.b200
```
