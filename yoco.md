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
FA4 prefill，并同样只把 single-token graph decode 切到 Triton。matched prefill
matrix 已完成；exact-image b13 的 BF16 与 MXFP8 + QKV-BF16 `mixed16`
decode16/128 matrix 也已通过。FA2 仍是生产默认，因为 FA4 selective-precision
profile 尚缺 end-to-end throughput、task quality 和真实长上下文验收。

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
- FA4 all-MXFP8 prefill：原 acceptance stacks 的 aggregate mean Native→vLLM
  KL 为 `0.0160540`，exact-image b13 为 `0.0187185`，均未达到 `<0.01`。
  exact-image b13 的 QKV-BF16 policy 将 prefill 降至 `0.00565650`，其
  decode16/128 matrix 已通过；不能用 selective-precision 结果宣称 all-MXFP8
  已通过。

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
forward。该实验只证明 scheduler shape。

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

- [x] **FA4 recommended matched matrix**
  - 当前状态：exact-image b13 BF16 和 MXFP8 + QKV-BF16 的 prefill、decode16
    smoke、decode128 acceptance 均通过。`FULL_DECODE_ONLY` 保留 eager FA4
    prefill，并把 single-token graph decode 路由到 Triton。all-MXFP8 prefill
    仍未通过，不属于已验收 recommendation。
  - 下一步：补 QKV-BF16 end-to-end throughput/task-quality，定位长 prefill 的
    首个 scale/payload/route 分叉，并继续真实 chunked-prefill 验收。
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

vLLM 使用自身 `vllm.vllm_flash_attn.cute` vendored FA4，不安装外部
`flash-attn-4`。当前 precompiled editable wheel 的 vendored source 依赖 CUTLASS
DSL `4.4.2` API（包括 `cute.core.ThrMma`/`make_fragment`），因此
`requirements/cuda.txt` 固定 `nvidia-cutlass-dsl[cu13]==4.4.2`。使用不匹配的
CUTLASS DSL `4.6.0` 会在首次 FA4 forward import 时失败。

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

### 推荐配置统一复现

`run_recommended_configs.sh` 将当前推荐的 FA2/FA4 prefill alignment profiles
统一映射到 `run_logprob_kl_compare.sh`，避免分别手工拼接 precision、attention、
FA4 overlay 和 tuning 参数。默认运行：

- `fa2-bf16`：FA2 BF16 + Triton MoE；
- `fa2-mxfp8`：生产推荐 FA2 MXFP8 + DeepGEMM MoE；
- `fa4-bf16`：exact-image b13 FA4 BF16；
- `fa4-mxfp8-qkv-bf16`：exact-image b13、default exp2、explicit
  `tile_mn=(128,128)`、auto q-stage、attention QKV BF16，其余 MXFP8。

四个 profiles 分配到四张 GPU 并行运行：

```bash
tools/yoco_alignment/run_recommended_configs.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --gpus 0,1,2,3
```

只复现两个 MXFP8 recommendations：

```bash
tools/yoco_alignment/run_recommended_configs.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --profiles fa2-mxfp8,fa4-mxfp8-qkv-bf16 \
  --gpus 0,1
```

profiles 按顺序 round-robin 分配给 `--gpus`；同一 GPU 上的 profiles 串行，不同
GPU workers 并行，因此 `--gpus 0` 可安全地串行执行完整矩阵。若 exact b13 overlay
尚不存在，追加 `--setup-fa4`；已有但不完整的目录不会被自动删除。`--profiles all`
额外运行已知不通过的 `fa4-mxfp8` control。

默认输出为 `../logs/yoco_alignment_results/recommended_configs/<profile>/`，每个
profile 写独立 artifacts 和 `run.log`。全部完成后生成 `summary.json`，列出 observed
KL、本文 recorded KL 和 delta；四个 recommendations 必须通过默认 `<0.01`，control
只报告、不参与 gate。`--dry-run` 会完成 dependency/profile preflight 并打印所有
child commands，不加载模型。该脚本复现的是 matched next-token prefill matrix；
FA2 decode 使用 `run_logprob_kl_decode_fa2.sh`，recommended FA4 decode 使用
`run_recommended_decode_fa4.sh`；chunked-prefill 仍单独验收。

### FA4 matched prefill matrix

2026-07-21 使用 `mixed16`、单个 `16 requests / 582 tokens` forward、V1
in-process、prefix cache 关闭和 `num_splits=1`，比较 Native FA4 与 vLLM vendored
FA4。vLLM 使用 `FULL_DECODE_ONLY`：prefill eager 执行 FA4，只有后续 single-token
decode 才会切换到 Triton attention。本实验只测 next-token prefill，不执行 decode。

matched pairing：

| matrix | Native reference | vLLM candidate |
| --- | --- | --- |
| BF16 FA4 | `quant_mode=bfloat16`、`use_cute=True`（FA4）、no KV cache | BF16、`FLASH_ATTN`、`flash_attn_version=4`、Triton BF16 MoE |
| MXFP8 FA4 | `quant_mode=mxfp8`、block `128`、torch activation quant、`use_cute=True`（FA4）、no KV cache | `quantization=fp8_per_block`、block `128`、`FLASH_ATTN`、`flash_attn_version=4`、DeepGEMM MoE |

两组都使用 `--force-fa-num-splits-one`。因此 MXFP8 结果确实是 **Native MXFP8
FA4 对 vLLM FP8-per-block FA4**，没有与 BF16 或 FA2 reference 混比。

一次运行两个 precision：

```bash
CUDA_VISIBLE_DEVICES=4 tools/yoco_alignment/run_logprob_kl_compare.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --output-dir ../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode \
  --attention-version 4 \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --variants bf16,fp8 \
  --bf16-moe-backend triton \
  --fp8-moe-backend deep_gemm
```

也可以分别复现，避免任一 cell 失败后阻塞另一个：

```bash
# Native BF16 FA4 vs vLLM BF16 FA4
CUDA_VISIBLE_DEVICES=4 tools/yoco_alignment/run_logprob_kl_compare.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --output-dir ../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode \
  --attention-version 4 \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --variants bf16 \
  --moe-backend triton

# Native MXFP8 FA4 vs vLLM FP8-per-block FA4
CUDA_VISIBLE_DEVICES=4 tools/yoco_alignment/run_logprob_kl_compare.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --output-dir ../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode \
  --attention-version 4 \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --variants fp8 \
  --moe-backend deep_gemm
```

runner 默认附加：

```text
Native: --native-use-cute --native-no-kv-cache --force-fa-num-splits-one
vLLM:   --attention-backend FLASH_ATTN --flash-attn-version 4
        --force-fa-num-splits-one --no-v1-multiprocessing
        --no-enable-prefix-caching
        --compilation-config-json
        {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}
```

| precision | Native→vLLM mean KL | vLLM→Native mean KL | mean JS | result |
| --- | ---: | ---: | ---: | --- |
| BF16 | `0.00357287` | `0.00348577` | `0.000880863` | pass |
| MXFP8 / FP8-per-block | `0.0160540` | `0.0156212` | `0.00393370` | fail |

unique prompt Native→vLLM KL：

| prompt | BF16 | MXFP8 |
| --- | ---: | ---: |
| `short_hello` | `0.000338769` | `0.00210331` |
| `short_fact` | `0.00388675` | `0.00608224` |
| `medium_english` | `0.00467639` | `0.0246546` |
| `short_zh` | `0.00526079` | `0.0143746` |
| `long_zh` | `0.00477970` | `0.0377058` |

MXFP8 的主要偏差来自中长 prompt，而不是所有 prompt 同等恶化。结果文件：

- [BF16 comparison](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/native_bf16_fa4_vs_vllm.json)、
  [Native BF16 FA4](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/native_bf16_fa4.pt)、
  [vLLM BF16 FA4](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/vllm_bf16_fa4.pt)；
- [MXFP8 comparison](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)、
  [Native MXFP8 FA4](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/native_mxfp8_fa4.pt)、
  [vLLM FP8-per-block FA4](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode/vllm_fp8_per_block_fa4.pt)。

首次尝试未显式传 compilation config，vLLM 默认进入 `FULL_AND_PIECEWISE`，在
YOCO BF16 MoE 的 `torch.get_float32_matmul_precision()` 处触发 Dynamo graph
break。runner 现默认传 `FULL_DECODE_ONLY`。另一次环境中 CUTLASS DSL `4.6.0`
与 wheel vendored FA4 source 不匹配；恢复官方 vendored path 和 4.4.2 后通过。

#### MXFP8 quantization parity 与 FA2 control

Native/vLLM 使用相同的逻辑 MXFP8 recipe：

| item | Native | vLLM |
| --- | --- | --- |
| weight group | `128 x 128` | `128 x 128` |
| activation group | per-token `1 x 128` | per-token `1 x 128` |
| payload | `float8_e4m3fn` | `float8_e4m3fn` |
| raw scale | `max(abs(x), 1e-4) / 448` | `max(abs(x), 1e-4) / 448` |
| scale rounding | ceil to UE8M0 power of two | ceil to UE8M0 power of two |
| logical scale dtype | FP32 | FP32 |
| DeepGEMM transport | FP32 power-of-two scales | 4 UE8M0 scale bytes packed into `int32` |

在相同 BF16 tensors 上直接比较 quantizer：activation shape `(257, 3072)`、weight
shape `(1280, 3072)`。Native torch reference 与 vLLM 分别得到：

- activation FP8 payload bitwise equal；
- weight FP8 payload bitwise equal；
- activation/weight FP32 UE8M0 scales exact equal，max scale diff `0`；
- vLLM packed activation path 的 FP8 payload 也与 Native bitwise equal。

因此 block size、activation chunk size、FP8 payload、scale rounding 和 logical
scale precision 没有发现 mismatch；vLLM 的 packed `int32` 只是 DeepGEMM scale
transport layout，不改变 quantized values。

为隔离 attention version，另跑相同 `mixed16`、batch 16、single forward、V1
in-process、prefix cache off、`num_splits=1` 和完全相同 MXFP8 recipe 的 FA2
control：

| attention | Native→vLLM mean KL | mean JS |
| --- | ---: | ---: |
| FA2 | `0.00840170` | `0.00211184` |
| FA4 | `0.0160540` | `0.00393370` |

[FA2 control report](../logs/yoco_alignment_results/fa2_prefill_mixed16_split1_inproc_fulldecode_mxfp8/native_mxfp8_fa2_vs_vllm_fp8_per_block.json)
使用与 FA4 完全相同的 quant/backend/batch controls。按 engine 内部比较 attention
version 的最终分布：Native FA2→FA4 mean KL 为 `0.00741357`，vLLM FA2→FA4 为
`0.0129372`。这说明 FA4 数值变化在 vLLM 一侧更大。

最后，对 mixed16 sequence geometry 的同一组 BF16 Q/K/V 分别运行 Native 外部
FA4 `4.0.0b22` 与 vLLM vendored FA4（两者 `num_splits=1`）：local/global causal
attention 均为 max abs diff `0.00390625`、mean abs diff `2.087e-6`、relative L2
`1.154e-4`。该差异很小，但会进入后续 MXFP8 activation quantization，并在 YOCO
universal-loop/MoE 层中反复传播，因此曾是 FA4 MXFP8 额外 drift 的候选来源。下述
common-source A/B 已证明消除该差异不足以改善最终 MXFP8 alignment；下一步应保存
首个 divergence layer 的 pre/post-attention hidden states、quantized activation
payload/scale 和 MoE routing IDs。

Upstream FlashAttention commit
`2409214a03797b168f648ea30df1adbc09ce658a` 修复的是 **FA4 attention 输入本身为
FP8 E4M3FN** 时的 probability saturation：把 E4M3 `max_offset` 从 `8` 降到 `4`。
当前 YOCO MXFP8 matrix 并不走该分支：Native MXFP8 projection 的 DeepGEMM output
显式为 BF16，vLLM `Fp8PerBlock` DeepGEMM kernel 也只支持 BF16 output；两侧传给
FA4 的 Q/K/V 和 `auto` KV cache 都是 BF16。当前旧代码因此在两侧都走
`q_dtype.width == 16`，`max_offset=0`；应用该 commit 后仍为 `0`。

所以该 upstream fix **不太可能解释本次 FA4 MXFP8 drift**，也不建议直接在当前
Native/vLLM acceptance environments 上覆盖安装 latest FA4：这会同时改变 FA4
revision、CUTLASS DSL 和 wrapper ABI，却不会触发被修复的 FP8-QKV branch。若未来
启用 FP8 attention input/KV cache，应在隔离 venv/container 中做 A/B；对当前实验
更干净的检查是只 backport 该 commit 到两侧 source，预期 BF16-QKV kernel/output
不变，并继续做 layer-level hidden/quantized-payload trace。

#### Native/vLLM common-source FA4 A/B

为只改变 Native FA4 source/runtime 而保留 NVIDIA Torch、TransformerEngine 和
DeepGEMM，新增隔离 overlay。Native 使用与 vLLM 完全相同的 vendored CuTe source，
并 pin vLLM 当前 runtime：CUTLASS DSL `4.4.2`、TVM FFI `0.1.9`、Quack `0.4.1`。
source tree SHA-256 为
`e2950f886bcca655de770c2a86d0d5b2fc3c4c5e1f8cda268755fd03f9858b5a`。
该 overlay 仅保留为历史 kernel/source control；canonical matrix runner 不再暴露
overlay options，并始终使用 Native external FA4 与 vLLM vendored FA4，避免把
CUTLASS/TVM/Quack override 带入常规 acceptance environment。

```bash
tools/yoco_alignment/setup_native_vllm_fa4_overlay.sh

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../.native-vllm-fa4-overlay/site-packages:$PWD \
../../.venv-yoco-native/bin/python \
  tools/yoco_alignment/verify_common_fa4.py run \
  --out ../logs/yoco_alignment_results/common_fa4_aligned_20260721/native_overlay.pt

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
../../.venv-yoco-mxfp8/bin/python \
  tools/yoco_alignment/verify_common_fa4.py run \
  --out ../logs/yoco_alignment_results/common_fa4_aligned_20260721/vllm.pt
```

先用相同、可精确表示的 BF16 Q/K/V 分别在 Native NVIDIA Torch 和 vLLM Torch 下
JIT/运行该 vendored source；FA4 output 和 LSE 均 bitwise exact（max abs/relative
L2 均为 `0`）。[common-source kernel report](../logs/yoco_alignment_results/common_fa4_aligned_20260721/compare.json)
同时证明不同 Torch build 本身没有改变该 case 的生成 kernel 结果。

完整模型结果：

| Native FA4 | BF16 Native→vLLM KL | MXFP8 Native→vLLM KL |
| --- | ---: | ---: |
| external `4.0.0b22` | `0.00357287` | `0.0160540` |
| vLLM vendored common source | `0.00341708` | `0.0187185` |

common source 令 BF16 KL 小幅下降 `4.36%`，但 MXFP8 KL 反而上升 `16.60%`。
vLLM old/new BF16、MXFP8 artifacts 各自 bitwise exact；只切换 Native FA4 时，最终
distribution KL 分别为 BF16 `0.00232863`、MXFP8 `0.00510318`。Native common-FA4
MXFP8 同进程 repeat max logprob diff 为 `0`，独立进程完整输出也 exact，排除
run-to-run nondeterminism。

变化具有明确 geometry 选择性：token 长度 `3/6/8` 的 prompts 完全不变；长度
`66/110` 才变化。MXFP8 `medium_english` KL 从 `0.0246546` 降到 `0.0179766`，但
`long_zh` 从 `0.0377058` 升到 `0.0585941`。这是 packed-GQA staged/tiled FA4 path
产生的小扰动被后续离散 MXFP8 quantization/MoE routing 放大的表现，而不是统一方向
的 FA4 bias。

结论：**FA4 revision mismatch 是真实 perturbation，但不是 MXFP8 较大 drift 的
root cause，也不是 alignment fix**。保持 common-source 工具用于控制变量；下一步
优先定位首个 UE8M0 scale exponent、E4M3 payload 或 top-8 expert IDs 分叉的 layer。
结果文件：

- [common-source BF16 report](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_common_fa4/native_bf16_fa4_vs_vllm.json)；
- [common-source MXFP8 report](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_common_fa4/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [Native installed-vs-vendored MXFP8](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_common_fa4/native_installed_vs_vendored_mxfp8.json)；
- [vLLM MXFP8 repeat control](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_common_fa4/vllm_repeat_mxfp8.json)。

#### Attention QKV BF16 selective-precision A/B

`--keep-attention-qkv-bf16` 只作用于 FP8 arm，并保持所有 attention input
projections 为 BF16：self layers 的 Q/K/V、cross layers 的 Q，以及 model-level
shared K/V。`o_proj`、FFN、shared/routed MoE 仍走 MXFP8。Native 将 42 个逻辑
projection module 切到 BF16；vLLM 通过 online quantization ignore patterns 将 fused
`qkv_proj` 整体保留 BF16。

vLLM 的 `QKVParallelLinear` 对 Q/K/V 使用单一 fused quant method，且 quantization
framework 明确禁止 fused shards 使用不同 precision。因此 matched Q/K-only ablation
需要拆开 vLLM projection 或新增 mixed-shard kernel；当前更干净、可维护的第一步是
整个 fused QKV BF16。Native 虽有独立 Q/K/V modules，也同步保留三者 BF16。

```bash
CUDA_VISIBLE_DEVICES=0 \
tools/yoco_alignment/run_logprob_kl_compare.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --attention-version 4 \
  --prompt-suite mixed16 \
  --batch-size 16 \
  --variants fp8 \
  --keep-attention-qkv-bf16 \
  --output-dir ../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed
```

两侧均使用原 acceptance installations：Native external FA4 `4.0.0b22`，vLLM
vendored FA4；其余 batch/process/cache/split/quant recipe controls 与 baseline 相同。

| FA4 MXFP8 policy | Native→vLLM KL | mean JS | mean abs logprob diff |
| --- | ---: | ---: | ---: |
| all projections MXFP8 | `0.0160540` | `0.00393370` | `0.257960` |
| attention QKV BF16 | `0.00947813` | `0.00235316` | `0.111654` |

QKV BF16 将 mean KL 降低 `40.96%`，首次使 FA4 MXFP8 aggregate 低于 `0.01`；16/16
prompts 均改善。unique prompts：

| prompt | all MXFP8 | QKV BF16 |
| --- | ---: | ---: |
| `short_hello` | `0.00210331` | `0.00128999` |
| `short_fact` | `0.00608224` | `0.000555366` |
| `medium_english` | `0.0246546` | `0.00345763` |
| `short_zh` | `0.0143746` | `0.0138509` |
| `long_zh` | `0.0377058` | `0.0309661` |

相对各 engine 自身 BF16 reference，效果并不对称：Native 的 BF16→MXFP8 KL 从
all-MXFP8 `0.00876520` 变为 QKV-BF16 `0.0103209`，而 vLLM 从 `0.0159846` 明显降至
`0.00693412`。因此该实验主要证明 **vLLM QKV quantization 是 cross-engine drift 的
重要非对称来源**，不能解释为 selective BF16 对每个 engine 的最终 logits 都单调
更接近 BF16。

FA2 control 进一步区分了 general QKV effect 与 FA4 interaction：

| attention | all MXFP8 KL | QKV BF16 KL |
| --- | ---: | ---: |
| FA2 | `0.00840170` | `0.00931939` |
| FA4 | `0.0160540` | `0.00947813` |

QKV BF16 令 FA2 KL 小幅上升 `10.92%`，却令 FA4 下降 `40.96%`；切换后 FA2/FA4
只差 `0.000159`。因此它消除的主要是 **FA4 × quantized-QKV interaction**，而不是
普遍降低所有 attention backend 的误差。这也解释了为什么 all-MXFP8 下 FA4 比 FA2
严重，而 QKV BF16 后两者基本收敛到同一 residual floor。

`long_zh` 仍明显超阈值，说明还有其他来源。下一步优先在 `long_zh` trace 中定位
首个 UE8M0 scale/E4M3 payload 或 MoE top-8 route 分叉。该 policy 在 TP1 约保留
`600.8M` weights 为 BF16，相对 FP8 payload 增加约 `0.56 GiB`；是否作为 production
default 还需补吞吐/延迟与 task-quality benchmark。结果文件：

- [QKV BF16 comparison](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/native_mxfp8_qkv_bf16_fa4_vs_vllm_fp8_per_block_qkv_bf16.json)；
- [Native QKV BF16 artifact](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/native_mxfp8_qkv_bf16_fa4.pt)；
- [vLLM QKV BF16 artifact](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/vllm_fp8_per_block_qkv_bf16_fa4.pt)；
- [Native own-BF16 control](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/native_bf16_vs_mxfp8_qkv_bf16.json)；
- [vLLM own-BF16 control](../logs/yoco_alignment_results/fa4_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/vllm_bf16_vs_fp8_qkv_bf16.json)；
- [FA2 QKV BF16 control](../logs/yoco_alignment_results/fa2_prefill_mixed16_split1_inproc_fulldecode_qkv_bf16_installed/native_mxfp8_qkv_bf16_fa2_vs_vllm_fp8_per_block_qkv_bf16.json)。

#### `donglixp/pytorch:26.02-b200` exact FA4 与 exp2 A/B

Docker Hub manifest
`sha256:cdbdc71b773142a98a303d488816d2faf6b9d85d179a4639b92263fca6da4769`
的 package layer 显示，该 image 实际 bake 的不是当前 PyPI beta，
而是以下完整 stack：

| package | image version |
| --- | --- |
| `flash-attn-4` | `4.0.0b13` |
| `nvidia-cutlass-dsl` | `4.5.1` |
| `apache-tvm-ffi` | `0.1.11` |
| `quack-kernels` | `0.4.1` |

迄今 `mixed16` FA4 prefill 实验汇总如下。`original unaligned` 表示 Native external
FA4 `4.0.0b22` 与 vLLM vendored snapshot 各自使用原 runtime；`common b22` 是曾经
误作 image target 的当前 PyPI b22 control，并非 Docker image；最后四行为本节的
exact-image b13 结果。

| FA4 source/runtime relation | exp2 policy | precision policy | Native→vLLM KL | mean JS |
| --- | --- | --- | ---: | ---: |
| original unaligned | default | BF16 | `0.00357287` | `0.000880863` |
| original unaligned | default | all MXFP8 | `0.0160540` | `0.00393370` |
| original unaligned | default | MXFP8 + QKV BF16 | `0.00947813` | `0.00235316` |
| Native b22 vs vLLM vendored | no-exp2 | all MXFP8 | `0.0187185` | `0.00459417` |
| common vLLM vendored snapshot | snapshot default | BF16 | `0.00341708` | `0.000843901` |
| common vLLM vendored snapshot | snapshot default | all MXFP8 | `0.0187185` | `0.00459417` |
| common external b22 (not image) | default | BF16 | `0.00565088` | `0.00143434` |
| common external b22 (not image) | default | all MXFP8 | `0.0111063` | `0.00271762` |
| common external b22 (not image) | default | MXFP8 + QKV BF16 | `0.00662123` | `0.00165084` |
| common external b22 (not image) | no-exp2 | all MXFP8 | `0.0187185` | `0.00459417` |
| exact image b13 | default | BF16 | `0.00341708` | `0.000843901` |
| exact image b13 | default | all MXFP8 | `0.0187185` | `0.00459417` |
| exact image b13 | default | MXFP8 + QKV BF16 | `0.00565650` | `0.00139278` |
| exact image b13 | no-exp2 | all MXFP8 | `0.0187185` | `0.00459417` |

Docker-image alignment 前的 explicit no-exp2 reference 使用 Native b22 no-exp2
artifact 与原 vLLM vendored artifact 重新比较；candidate 与原 baseline vLLM output
bitwise equal。其 KL `0.0187185` 比 original unaligned default `0.0160540` 高
`16.60%`，因此在旧 unaligned stacks 上也没有改善。另一个非-image control 是
common external b22：关闭 exp2 后 KL 从 `0.0111063` 上升到 `0.0187185`
（`+68.54%`）。[explicit unaligned no-exp2 report](../logs/yoco_alignment_results/image_fa4_4.0.0b22/unaligned_no_ex2/native_b22_no_ex2_vs_vllm_vendored.json)

从 image layer 直接抽取的 `flash_attn/cute/*.py` 与隔离 overlay 的 source tree
SHA-256 均为
`edbe1f46fcd2ac531be02900ef6caf7269e449279a0dd23509f8cb47420cf369`。两侧通过
同一 source/runtime profile 运行；vLLM 只增加 private API `(out,lse,p,row_max)`
到 `(out,lse)` 的返回值 adapter，不修改 kernel 参数或实现。

```bash
tools/yoco_alignment/setup_image_fa4_profiles.sh

CUDA_VISIBLE_DEVICES=0 \
tools/yoco_alignment/run_image_fa4_matrix.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged
```

`mixed16`、batch 16、in-process V1、prefix cache off、`num_splits=1` 的结果：

| exact-image FA4 arm | Native→vLLM KL | mean JS | mean abs logprob diff |
| --- | ---: | ---: | ---: |
| BF16 | `0.00341708` | `0.000843901` | `0.111748` |
| all MXFP8 | `0.0187185` | `0.00459417` | `0.198440` |
| MXFP8 + QKV BF16 | `0.00565650` | `0.00139278` | `0.104463` |
| all MXFP8 + no-exp2 | `0.0187185` | `0.00459417` | `0.198440` |

QKV BF16 在 exact-image stack 下将 all-MXFP8 KL 降低 `69.78%`，仍是最有效的
selective-precision control。相比原 acceptance installations，exact-image BF16
略改善，all-MXFP8 反而恶化，而 QKV-BF16 明显改善；FA4 revision 影响仍然
依赖 precision/path，不能用单一方向概括。

no-exp2 profile 只在 b13 source 的 tuning 完成后加入请求的：

```python
self.enable_ex2_emu = False
self.ex2_emu_freq = 0
```

其 source SHA-256 为
`18f2bf4ffe0c4c6122fefe55d31e7ab9433aded50383d1532a0d2e6e9253709b`；除
`flash_fwd_sm100.py` 这两行 override 外，文件集合及内容均与 image profile 相同。

`mixed16` 的 token lengths 只有 `3/6/8/66/110`，均不超过 FA4 `n_block_size=128`。
b13 在 `mask_fn is not None` 时本来就给 `apply_exp2_convert` 传
`ex2_emu_freq=0`；这些 sequence 的唯一 KV block 都是 masked edge block。因此
default/no-exp2 在 `mixed16` 下 Native 和 vLLM artifacts 各自 bitwise exact，
该 A/B 对短 suite 是结构性 no-op。

为实际触发 unmasked interior blocks，另用 `chunk4k` 的 4,682-token repetitive
prompt 做 stress test：

| 4.7k prefill arm | Native→vLLM KL | mean JS | top-20 overlap |
| --- | ---: | ---: | ---: |
| BF16 | `0.0232922` | `0.00586551` | `18/20` |
| all MXFP8 default | `0.395051` | `0.0971621` | `15/20` |
| all MXFP8 no-exp2 | `0.291749` | `0.0669735` | `19/20` |

这些大 KL 是真实 finite/normalized next-token distributions，不是 tokenization、
NaN 或 scheduler token-count 错误：两侧输入 token IDs exact equal，Native 为一个
`1 request / 4682 tokens` forward，vLLM iteration 也为 `1 / 4682`；三组结果 top-1
均为 token `13919`（`" Harry"`）。但它们**不能与 mixed16 的 `~1e-2` 直接比较**：
prompt 长度和数值累积相差约两个数量级，且 BF16 floor 自身已升到 `0.0233`。

更重要的是，相对各 engine 自身 BF16 reference：

| 4.7k MXFP8 profile | Native KL vs own BF16 | vLLM KL vs own BF16 |
| --- | ---: | ---: |
| default | `0.0281959` | `0.326553` |
| no-exp2 | `0.211784` | `0.390367` |

所以 no-exp2 虽将 cross-engine KL 降低 `26.15%`，却使两侧都更远离各自 BF16；
mean abs Native-vLLM logprob diff 也从 `0.55056` 升到 `1.00608`。这不是 accuracy 或
quality improvement，只是两个已显著移动的分布偶然更接近，不建议据此关闭 exp2
emulation。

8-token greedy rollout/replay sanity 显示两 profile 都生成完全相同且连贯的：

```text
 Harry Potter and the Philosopher's Stone
```

token IDs 均为 `[13919, 29218, 323, 279, 7155, 45090, 594, 14292]`；保存完整
distribution 的 positions `1/2/4/8` 上 Native 与 vLLM argmax 都等于生成 token。
large discrepancy 集中在 response position 1（长 prefill boundary），进入真实
KV-cached decode 后 alignment 很紧：

| profile | pos 1 KL | pos 2 KL | pos 4 KL | pos 8 KL | decode-only selected-token mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | `0.297213` | `0.000472728` | `0.0000709748` | `0.000245207` | `0.00122492` |
| no-exp2 | `0.0294102` | `0.000338844` | `0.000124105` | `0.000108733` | `0.00107547` |

因此两 engine 仍能产生有意义且一致的短 continuation；问题是 4.7k all-MXFP8
prefill distribution calibration/accumulation，而不是 decode path 崩坏。当前结论是：
保留 default exp2 tuning；对长 prompt 优先 trace 首个 MXFP8 scale/payload/MoE route
分叉，并扩充多种真实长上下文，而不是从单个重复 prompt 的 cross-engine KL 选择
exp2 policy。结果文件：

- [exact-image BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/default/native_bf16_fa4_vs_vllm.json)；
- [exact-image all-MXFP8](../logs/yoco_alignment_results/image_fa4_4.0.0b13/default/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [exact-image QKV-BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/qkv_bf16/native_mxfp8_qkv_bf16_fa4_vs_vllm_fp8_per_block_qkv_bf16.json)；
- [mixed16 no-exp2](../logs/yoco_alignment_results/image_fa4_4.0.0b13/no_ex2/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [chunk4k BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/chunk4k_bf16/native_bf16_fa4_vs_vllm.json)；
- [chunk4k default MXFP8](../logs/yoco_alignment_results/image_fa4_4.0.0b13/chunk4k_default/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [chunk4k no-exp2](../logs/yoco_alignment_results/image_fa4_4.0.0b13/chunk4k_no_ex2/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [default decode8](../logs/yoco_alignment_results/image_fa4_4.0.0b13/decode8_default/compare.json)；
- [no-exp2 decode8](../logs/yoco_alignment_results/image_fa4_4.0.0b13/decode8_no_ex2/compare.json)。

##### 其他 FA4 knobs

在 exact-image b13、default exp2、all-MXFP8、`mixed16` 上继续测试两个对 YOCO
causal varlen path 实际生效的 knob。`FA_DISABLE_2CTA` 和 `FA_CLC` 未列为实验 arm：
2CTA 要求 non-causal、non-local、non-varlen；CLC 也在 varlen 下关闭，因此对当前
YOCO prefill 是结构性 no-op。

| FA4 config | Native→vLLM KL | mean JS | mean abs logprob diff | vs all-MXFP8 default |
| --- | ---: | ---: | ---: | ---: |
| default | `0.0187185` | `0.00459417` | `0.198440` | baseline |
| explicit `tile_mn=(128,128)` | `0.0187185` | `0.00459417` | `0.198440` | bitwise equal to default |
| `pack_gqa=False` | `0.0187185` | `0.00459417` | `0.198440` | exact no-op |
| `tile_mn=(128,64)` | `0.0151919` | `0.00374743` | `0.183869` | `-18.84%` KL |
| QKV BF16 | `0.00565650` | `0.00139278` | `0.104463` | `-69.78%` KL |
| QKV BF16 + explicit `tile_mn=(128,128)` | `0.00565650` | `0.00139278` | `0.104463` | bitwise equal to QKV-BF16 default |
| QKV BF16 + `tile_mn=(128,64)` | `0.00784983` | `0.00196721` | `0.111309` | `-58.06%` KL |

`pack_gqa=False` 在 Native/vLLM 两侧均明确记录为 override，但完整 logits artifacts
各自与 default bitwise exact，因此无需继续。`tile_n=64` 对 66/110-token prompts
生效：`medium_english` KL `0.0179766 -> 0.0131874`，`long_zh`
`0.0585941 -> 0.0445749`；短 prompts 不变。它是有效的 secondary knob，但相对各
engine default 会同时移动 Native（KL `0.0031690`）和 vLLM（`0.00561434`），且
相对 own-BF16 的 KL 仍为 Native `0.00971392`、vLLM `0.0152713`。

`tile_n=64` 与 QKV-BF16 不正向叠加：组合 KL `0.00784983` 比 QKV-BF16 alone
`0.00565650` 高 `38.78%`。其中 `medium_english` 略改善，但 `long_zh` 从
`0.00915731` 恶化到 `0.0223107`。因此当前优先级为：

1. 保持 default exp2；
2. 若允许 selective precision，优先 QKV BF16；
3. `tile_n=64` 只作为 all-MXFP8 secondary control，不与 QKV-BF16 默认组合。

结果文件：

- [pack-GQA off](../logs/yoco_alignment_results/image_fa4_4.0.0b13/pack_gqa_off/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [tile n64](../logs/yoco_alignment_results/image_fa4_4.0.0b13/tile_n64/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)；
- [tile n64 + QKV BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/tile_n64_qkv_bf16/native_mxfp8_qkv_bf16_fa4_vs_vllm_fp8_per_block_qkv_bf16.json)。

`q_stage` 不是 tile shape 本身，而是每个 CTA 同时 pipeline 的 Q tiles 数。b13 在
SM100 上使用：

```python
seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
q_stage = 2 if seqlen_q_packgqa > tile_m else 1
```

该 YOCO checkpoint 的 differential attention 为 64 Q heads、8 KV heads，故
`qhead_per_kvhead=8`。固定 `tile_m=128` 后 threshold 是
`max_seqlen_q > 16`。`mixed16` 的 batch max 为 110，因此当前所有 layer/call 已
de-facto 使用 `q_stage=2`；单独运行长度 `3/6/8` 的 prompt 才会选择 stage 1。

explicit `tile_mn=(128,128)` 已固定 tile M/N 并与 implicit default bitwise equal，
但它没有固定 `q_stage`。现已用 exact-image b13 的 forced-stage source profiles 完成
stage 1/2 A/B。`mixed16` 的 BF16、all-MXFP8、MXFP8 + QKV-BF16 三种 policy 在
Native 和 vLLM 两侧均满足：

- forced stage 1 vs forced stage 2：mean KL、max/mean absolute logprob diff 全为 `0`；
- auto vs forced stage 2：三种 policy 的 artifact 均 bitwise equal；
- batch-1 auto vs forced stage 1：已比较的 BF16 与 all-MXFP8 artifact 均 bitwise
  equal；直接 stage 1 vs stage 2 比较也未发现数值差异。

因此 q-stage switch 在当前 workload 上数值中性。更重要的是，用相同 short prompt
分别 standalone 和置于 `mixed16` 中比较时，强制 stage 1 与强制 stage 2 的
batch-composition diff 都仍为 nonzero；六个 precision/stage 组合均如此。也就是说，
固定 q-stage **不能**恢复 strict batch invariance，q-stage 不是已观察到 batch
dependence 的根因。完整汇总见
[batch_invariance.json](../logs/yoco_alignment_results/image_fa4_4.0.0b13/qstage_comparisons/batch_invariance.json)。

初步 kernel-only latency microbenchmark（B200，explicit `128x128`，warmup 后 CUDA
event 计时；单次结果，不等同于 end-to-end throughput）：

| geometry | attention | stage 1 (ms) | stage 2 (ms) | stage 2 vs 1 |
| --- | --- | ---: | ---: | ---: |
| mixed16-like, 582 tokens | local `(512,0)` | `0.058249` | `0.057381` | `-1.49%` |
| mixed16-like, 582 tokens | global | `0.056955` | `0.056435` | `-0.91%` |
| standalone, 3 tokens | local `(512,0)` | `0.056222` | `0.057356` | `+2.02%` |
| standalone, 3 tokens | global | `0.055345` | `0.066656` | `+20.44%` |

最终建议：继续显式 pin `tile_mn=(128,128)`，但保留 b13 的 `q_stage=auto`。
auto 在 mixed batch 选择略快的 stage 2，在 tiny prompt 避免 stage 2 的额外开销；
固定任一 stage 都没有数值或 batch-invariance 收益。只有当实验要求 kernel control
flow 也完全不随 shape 改变时才应 pin q-stage，并应把潜在吞吐回退作为代价记录。

##### `yoco-0716` batch-1 starting-point 复测

历史 `shaohanh/yoco-0716` 的 “batch 1” 是 `mixed5` 五个 prompts 分别以一个
one-request forward 执行后聚合，并非只测一个 prompt。历史结果使用 FA2：MXFP8
KL `0`，BF16 KL `0.00550893`。用当前推荐 FA4 条件复测：exact-image b13、default
exp2、explicit `tile_mn=(128,128)`、`num_splits=1`、prefix cache off、V1
in-process、batch size 1：

| batch-1 precision policy | Native→vLLM KL | mean JS | exact-zero prompts |
| --- | ---: | ---: | ---: |
| BF16 | `0.00396589` | `0.000995932` | `0/5` |
| all MXFP8 | `0.00828524` | `0.00207670` | `0/5` |
| MXFP8 + QKV BF16 | `0.00874279` | `0.00216718` | `0/5` |

所以 current FA4 可通过 batch-1 `<0.01`，但未达到 historical FA2 MXFP8 exact
zero。BF16 比历史 `0.00550893` 改善约 `28.0%`；QKV-BF16 在 batch 1 略差于
all-MXFP8，这与 batch-16 中的大幅改善再次说明 selective-precision effect 依赖
batch/sequence geometry。

per-prompt Native→vLLM KL：

| prompt | BF16 | all MXFP8 | QKV BF16 |
| --- | ---: | ---: | ---: |
| `short_hello` | `0.000383975` | `0.00170887` | `0.00221632` |
| `short_fact` | `0.00128207` | `0.000776463` | `0.00278359` |
| `medium_english` | `0.00283869` | `0.00566024` | `0.00391972` |
| `short_zh` | `0.00892018` | `0.0102440` | `0.0159414` |
| `long_zh` | `0.00640453` | `0.0230367` | `0.0188529` |

该结果作为 q-stage A/B 的起点：standalone short prompts 根据长度使用 stage 1，
66/110-token prompts 使用 stage 2；mixed16 batch 则统一 auto-select stage 2。
[batch-1 BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/bsz1_tile_n128/native_bf16_fa4_vs_vllm.json)、
[batch-1 all-MXFP8](../logs/yoco_alignment_results/image_fa4_4.0.0b13/bsz1_tile_n128/native_mxfp8_fa4_vs_vllm_fp8_per_block.json)、
[batch-1 QKV-BF16](../logs/yoco_alignment_results/image_fa4_4.0.0b13/bsz1_tile_n128/native_mxfp8_qkv_bf16_fa4_vs_vllm_fp8_per_block_qkv_bf16.json)。

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

#### 128-token exact-image b13 FA4 recommended matrix

2026-07-24 已完成 BF16 与 MXFP8 + QKV-BF16 两个 profiles。共同条件为
exact-image FA4 `4.0.0b13`、CUTLASS DSL `4.5.1`、TVM FFI `0.1.11`、Quack
`0.4.1`、default exp2、explicit `tile_mn=(128,128)`、auto q-stage、
`mixed16` batch 16、V1 in-process、prefix cache off、`num_splits=1` 和
`FULL_DECODE_ONLY`。position 1 使用 eager FA4 prefill；position 2 以后是 graph
captured Triton KV-cache decode。Native 使用 exact b13 FA4 做一个 packed、no-cache
teacher-forced replay；BF16 为 `16 sequences / 2240 tokens`，QKV-BF16 为
`16 / 2105`。

```bash
tools/yoco_alignment/run_recommended_decode_fa4.sh \
  --model /data/yanqi/model_ckpt/0000-6000-hf \
  --native-checkpoint /data/yanqi/model_ckpt/0000-6000-merged \
  --gpus 0,1 \
  --lengths 16,128
```

launcher 将 BF16 和 MXFP8 + QKV-BF16 分配到不同 GPU 并行执行；同一 profile
内部按 length 串行完成 vLLM rollout、Native replay 和 compare。16-token smoke
的 row-weighted true-decode KL 分别为 `0.00163256`、`0.00460465`；最差 aggregate
checkpoint 分别为 `0.00231970`、`0.00702355`，均通过 `<0.01`。

128-token acceptance 总结。`true-decode weighted` 按 position 2 以后的实际
full-vocab rows 加权，不包含 position 1：

| metric | BF16 | MXFP8 + QKV BF16 |
| --- | ---: | ---: |
| position 1 prefill KL | `0.00386452` | `0.00613731` |
| true-decode sparse rows | `140` | `131` |
| true-decode weighted Native→vLLM KL | `0.00102458` | `0.00244198` |
| true-decode weighted mean JS | `0.000253163` | `0.000602323` |
| all sparse positions Native→vLLM KL | `0.00131586` | `0.00284419` |
| selected-token positions | `1642` | `1507` |
| selected mean / p95 abs diff | `0.015888 / 0.073336` | `0.033587 / 0.126427` |
| selected p99 / max abs diff | `0.120836 / 0.216878` | `0.212744 / 0.401032` |

完整词表按 response position 聚合：

| position | path | BF16 active / KL / JS | MXFP8 + QKV BF16 active / KL / JS |
| ---: | --- | ---: | ---: |
| 1 | prefill | `16 / 0.003865 / 0.000954` | `16 / 0.006137 / 0.001506` |
| 2 | decode | `13 / 0.002531 / 0.000616` | `13 / 0.003725 / 0.000918` |
| 4 | decode | `13 / 0.000550 / 0.000138` | `13 / 0.004783 / 0.001183` |
| 8 | decode | `13 / 0.001157 / 0.000289` | `13 / 0.003279 / 0.000812` |
| 16 | decode | `13 / 0.000344 / 0.000086` | `13 / 0.002214 / 0.000546` |
| 32 | decode | `13 / 0.000309 / 0.000077` | `13 / 0.003628 / 0.000867` |
| 48 | decode | `13 / 0.000180 / 0.000045` | `13 / 0.002320 / 0.000588` |
| 64 | decode | `13 / 0.001171 / 0.000292` | `13 / 0.004174 / 0.001034` |
| 80 | decode | `13 / 0.002655 / 0.000655` | `13 / 0.000227 / 0.000056` |
| 96 | decode | `13 / 0.001103 / 0.000271` | `13 / 0.000116 / 0.000029` |
| 112 | decode | `13 / 0.000142 / 0.000035` | `7 / 0.000124 / 0.000028` |
| 128 | decode | `10 / 0.001160 / 0.000288` | `7 / 0.000138 / 0.000038` |

所有 aggregate true-decode checkpoints 均小于 `0.01`，且没有随 length 单调累积。
逐 prompt 仍有 tail：BF16 最大 prompt/checkpoint KL 是 `short_zh_2` position 2
的 `0.00862323`；QKV-BF16 最大值是同 prompt position 4 的 `0.0155409`，虽然
该 position 的 13-request aggregate 仅 `0.00478343`。selected-token worst case：

- BF16：`short_fact_2` position 116，token `279`（` the`），
  `vLLM - Native = -0.216878`；
- MXFP8 + QKV BF16：`short_zh_2` position 7，token `98448`（`三`），
  `vLLM - Native = -0.401032`。

response coverage：BF16 min/mean/median/max 为 `1/103.625/128/128`，`10/16`
达到 128；QKV-BF16 为 `1/95.1875/104/128`，`7/16` 达到 128。两侧 rollout
均连续可读，没有重复首 token、乱码或单 token collapse；不同 precision 产生不同
trajectory 属正常现象，Native 分别 replay 各自 exact token IDs。

与 FA2 128-token matrix 按同一定义对照：FA4 BF16 的 weighted decode KL、selected
mean/p95/max 分别改善 `36.56%`、`23.96%`、`13.69%`、`38.16%`。QKV-BF16
FA4 相对 FA2 all-MXFP8 的 weighted KL 改善 `10.46%`，但 selected mean/p95
恶化 `67.74%`/`39.11%`，max 基本持平（`+0.38%`）。后者不是纯 attention-version
A/B，因为 precision policy 和 rollout trajectory 均不同；结论是 distributional
decode alignment 通过，但 FP8 selected-token tail 仍需保留为风险项。

结果：BF16 [decode128 comparison](../logs/yoco_alignment_results/fa4_decode_recommended_b13/fa4-bf16/decode128/compare.json)、
QKV-BF16 [decode128 comparison](../logs/yoco_alignment_results/fa4_decode_recommended_b13/fa4-mxfp8-qkv-bf16/decode128/compare.json)、
BF16 [decode16 smoke](../logs/yoco_alignment_results/fa4_decode_recommended_b13/fa4-bf16/decode16/compare.json)、
QKV-BF16 [decode16 smoke](../logs/yoco_alignment_results/fa4_decode_recommended_b13/fa4-mxfp8-qkv-bf16/decode16/compare.json)。

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
tools/yoco_alignment/logprob_kl_decode.py
tools/yoco_alignment/run_recommended_configs.sh
tools/yoco_alignment/run_recommended_decode_fa4.sh
vllm/model_executor/models/yoco.py
vllm/model_executor/models/config.py
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/v1/attention/backends/flash_attn.py
vllm/entrypoints/openai/chat_completion/protocol.py
vllm/parser/agens_parser.py
docker/Dockerfile.b200
```
