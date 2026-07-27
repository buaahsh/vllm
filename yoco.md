# YOCO vLLM B200 对齐与运行说明

本文记录 YOCO-30B-A3B 在 B200 上与
`/workspace/shaohanh/llm-train` 对齐后的实现、验证结果和推荐运行方式。
YOCO-v2 对应已发布代码分支为 `shaohanh/yoco-0716`，容器镜像为
`buaahsh/pytorch:26.02-b200-vllm-0716`。YOCO-v3/L3 开发与验证分支为
`shaohanh/yoco-0725`。

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

YOCO-v3/L3 28k checkpoint：

- Native checkpoint:
  `/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000`
- nnScaler merged checkpoint:
  `/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-merged`
- 转换后的 HF checkpoint:
  `/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf`

v3 HF checkpoint 包含 14 个 safetensor shard、357 个推理权重；转换只跳过
33 个 MTP 专用权重。

## 对齐原则

低层 kernel matched 对齐要求 vLLM 和 llm-train Native 使用相同计算条件；
RL backend A/B 则固定精度、采样和 sampled token sequence，只改变明确记录的
vLLM backend。每次实验前必须先固定并记录以下因素：

1. **精度模式一致**：MXFP8 对齐要求两侧都使用 MXFP8；BF16 对齐要求两侧
   都使用 BF16，不能用 MXFP8 vLLM 对比 BF16 Native。
2. **量化配置一致**：MXFP8 两侧都使用 128-element block。Native 使用
   `quant_mode=mxfp8`、`quant_block_size=128`；vLLM 使用
   `--quantization fp8_per_block`。Native 的 torch activation quant fallback
   只是替换不稳定的 Triton 实现，不改变 MXFP8 数值格式。
3. **区分 matched kernel 对齐和 RL backend A/B**：低层 attention kernel
   对齐要求两侧一致，FA2 只能和 FA2 reference 比较，FA4 只能和 FA4
   reference 比较。RL rollout backend A/B 则固定 llm-train 使用 FA4，
   有意改变 vLLM attention/MoE backend，比较真实 sampled-token logprob。
   当前 Native FA4 固定为 `4.0.0b13`
   (`9bad4bec7326ad28edb5516b8878fd283f8991c0`) 和 CuTeDSL `4.5.1`。
4. **执行形状一致**：batch size、scheduler forward shape、prompt 顺序、
   chunk 切分位置和 KV-cache 语义必须一致。batch 16 当前使用与 vLLM
   scheduler 一致的 `1 + 15` Native forward shape。
5. **并行和功能开关可追踪**：matched 对齐时 TP、EP、KV-sharing fast
   prefill、chunked prefill、CUDA Graph 范围必须一致；backend A/B 中改变的
   项必须单独列出，不能把不同配置混成同一个 KL 结论。

## 当前结论

### YOCO-v3/L3 28k：64-query RL rollout

vLLM 已同时支持旧 YOCO-v2 和新 YOCO-v3。v3 新增语义包括：

- diff-v3：`attn_even * sigmoid(gate_even) -
  attn_odd * sigmoid(gate_odd)`；
- 带可学习 affine weight 的 Q/K RMSClip；
- latent MoE：
  `3072 -> 1024 -> routed experts -> 1024 -> 3072`，两侧 projection 前后
  保持与训练一致的 RMSNorm 顺序；
- `universal_loop=3`。

旧 v2 config 没有 `qk_rms_gamma` 时默认使用 weight-free Q/K clip，不会因
v3 支持而改变旧 checkpoint 的参数结构。

正式矩阵固定：

- 64 个 prompts、`temperature=1`、`top_p=1`、`min_tokens=0`、最多
  256 output tokens；
- 四张 B200，FA4 beta13，非 eager `FULL_DECODE_ONLY`；
- 开启 chunked prefill、KV-sharing fast prefill；
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` 和 `--no-async-scheduling`；
- llm-train BF16/MXFP8 baseline 均使用 FA4，teacher-forcing batch size 16。

| vLLM rollout | output tok/s | 对 BF16 k3 KL | 对 MXFP8 k3 KL |
| --- | ---: | ---: | ---: |
| BF16 TP4 + EP4 | **`2625.99`** | **`0.00294112`** | `0.01082455` |
| MXFP8，4 个 TP1 replica，DeepGEMM MoE | `2390.39` | `0.01137099` | **`0.01054670`** |

BF16 TP4+EP4 是当前精度最优配置，匹配 BF16 baseline 的 k3 KL 明显低于
`5e-3`。MXFP8 DeepGEMM 与 llm-train 原生 DeepGEMM quant baseline 的结果
和 torch activation-quant reference 完全一致；其 sampled KL 为
`0.00938869`，但更保守的非负 k3 KL 为 `0.01054670`，仍略高于严格
`1e-2` 验收线。偏差主要集中在 completion 前 128 tokens；128–256 token
区间 k3 KL 为 `0.00943566`。

Triton MoE 的两 TP2 replica 和四 TP1 replica 静态吞吐分别为 `2908.54`
和 `3084.53 tok/s`。这些数据来自强制最小生成长度的 throughput sweep，
可以用于速度比较，但对应 raw logprob 不能用于 on-policy KL；正式 KL 表只
保留上面的 `min_tokens=0` 结果。

固定每请求 256 tokens 的持续到达结果：

| 配置 | rate | 64-query wall | output tok/s | p50 / p95 | queue max |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 TP4 + EP4 | 8 qps | `12.183s` | `1344.79` | `7.986 / 9.524s` | 0 |
| BF16 TP4 + EP4 | 16 qps | `8.965s` | `1827.52` | `6.834 / 7.820s` | 0 |
| BF16 TP4 + EP4 | 32 qps | `8.238s` | `1988.89` | `7.228 / 7.801s` | 0 |
| BF16 TP4 + EP4 | infinite | `6.033s` | `2715.95` | `6.013 / 6.025s` | 0 |
| MXFP8 4xTP1 Triton | 8 qps | `11.578s` | `1415.13` | `5.217 / 5.531s` | 0 |
| MXFP8 4xTP1 Triton | 16 qps | `8.293s` | `1975.72` | `5.496 / 5.977s` | 0 |
| MXFP8 4xTP1 Triton | 32 qps | `6.661s` | `2459.79` | `5.520 / 6.052s` | 0 |
| MXFP8 4xTP1 Triton | infinite | **`5.118s`** | **`3201.43`** | `5.034 / 5.110s` | 0 |

拓扑限制：

- MXFP8 TP4 不能直接使用：shared expert intermediate size 1280 在 TP4 下
  分成 320，不能被 128-element activation quant block 整除；
- DP4+EP4 在 synchronous V1 + fast-prefill 首批请求中触发 DP/EP
  all-to-all token-size assertion，不作为鲁棒生产候选；
- 两个 TP2 replica 和四个 TP1 replica 都稳定完成静态及 open-loop sweep，
  64/64 请求成功且 queue max 为 0。

因此 v3 四卡推荐：

1. 精度优先：BF16 TP4+EP4；
2. 纯吞吐优先：四个 MXFP8 TP1 + Triton MoE replica，但当前没有通过严格
   on-policy k3 KL 验收；
3. MXFP8 匹配 llm-train backend：四个 TP1 + DeepGEMM MoE，sampled KL
   低于 `1e-2`，但 k3 仍略高于阈值。

### vLLM backend A/B：llm-train 固定 FA4

下面所有结果都固定 llm-train teacher-forcing baseline 使用 FA4，vLLM 侧
有意改变 attention 或 MoE backend。单 B200、非 eager
`FULL_DECODE_ONLY`、64 prompts、Native scoring batch 16，并开启 chunked
prefill 和 KV-sharing fast prefill。

BF16 attention sweep 固定 vLLM Triton MoE，`max_tokens=128`：

| vLLM attention | 64-query wall | output tok/s | 对 BF16+FA4 k3 KL |
| --- | ---: | ---: | ---: |
| FA4 | `12.809s` | `633.95` | **`0.00297739`** |
| FA2 | `12.983s` | `618.48` | `0.00304211` |
| FlashInfer | **`12.568s`** | **`642.66`** | `0.00316659` |

三种 BF16 attention 都通过 `<5e-3`。FlashInfer attention 最快，FA4 的 KL
最低，但两者差异很小。Triton attention 在非 eager CUDA Graph capture 中
触发 illegal memory，不能作为生产候选。

MXFP8 attention sweep 固定 vLLM Triton MoE，`max_tokens=128`：

| vLLM attention | 64-query wall | output tok/s | 对 MXFP8+FA4 k3 KL |
| --- | ---: | ---: | ---: |
| FA4 | **`10.570s`** | **`766.22`** | **`0.01112026`** |
| FA2 | `10.907s` | `736.83` | `0.01264444` |
| FlashInfer | `10.757s` | `751.31` | `0.01245236` |

MXFP8 MoE sweep 固定 vLLM FA4 attention：

| vLLM MoE | 64-query wall | output tok/s | 对 MXFP8+FA4 k3 KL |
| --- | ---: | ---: | ---: |
| Triton | **`10.570s`** | **`766.22`** | **`0.01112026`** |
| DeepGEMM | `13.658s` | `592.02` | `0.01119467` |

因此单卡 MXFP8 当前应使用 **FA4 attention + Triton MoE**：它同时是最快
attention 组合和最快 MoE backend，并且 KL 也是本次 MXFP8 backend sweep
中最低。更长的 warm `max_tokens=256` 正式结果为：

| vLLM 配置 | 64-query wall | output tok/s | matched FA4 k3 KL |
| --- | ---: | ---: | ---: |
| BF16 FlashInfer attention + Triton MoE | `24.903s` | `620.44` | **`0.00294958`** |
| MXFP8 FA4 attention + Triton MoE | **`20.733s`** | **`765.56`** | `0.01024826` |

BF16 明确通过 `<5e-3`。MXFP8 比 max128 更接近阈值，但仍高于 `<1e-2`
约 `2.5e-4`。第一次 MXFP8 max256 请求因 CuTe/FA4 shape JIT 只有
`224.01 tok/s`；同一 server warm repeat 恢复为 `765.56 tok/s`，所以正式
吞吐不包含该一次性编译时间。

MoE backend 兼容性：

- BF16 FlashInfer TRTLLM 不支持 YOCO custom routing method 101；
- BF16 FlashInfer CUTLASS 在加载权重后不再推进；
- MXFP8 vLLM CUTLASS 对当前配置被 oracle 禁用；
- MXFP8 FlashInfer CUTLASS 不支持 128x128 weight block 和 1x128 dynamic
  activation quant；
- MXFP8 FlashInfer TRTLLM 在转换到约 31.27 GiB 后不再推进；
- `--moe-backend auto` 不是安全回退：BF16 自动选择 FlashInfer CUTLASS，
  MXFP8 自动选择 FlashInfer TRTLLM，二者等待 3 分钟均未 ready。

生产启动必须显式传 `--moe-backend triton`，不要依赖 `auto`。

固定每请求 256 tokens 的单卡持续到达结果如下。所有组合均为 64/64 成功、
0 请求错误：

| 配置 | rate | 64-query wall | output tok/s | p50 / p95 | queue max |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 FlashInfer + Triton | 2 qps | `36.204s` | `452.55` | `8.660 / 11.173s` | 12 |
| BF16 FlashInfer + Triton | 4 qps | `30.987s` | `528.74` | `11.664 / 17.292s` | 28 |
| BF16 FlashInfer + Triton | 8 qps | `28.580s` | `573.27` | `14.209 / 21.856s` | 45 |
| BF16 FlashInfer + Triton | 16 qps | `27.246s` | `601.34` | `15.375 / 23.996s` | 48 |
| BF16 FlashInfer + Triton | 32 qps | `26.086s` | `628.08` | `15.638 / 24.556s` | 48 |
| BF16 FlashInfer + Triton | infinite | **`24.201s`** | **`677.00`** | `15.161 / 24.188s` | 48 |
| MXFP8 FA4 + Triton | 2 qps | `33.635s` | `487.11` | `6.962 / 9.185s` | 9 |
| MXFP8 FA4 + Triton | 4 qps | `28.348s` | `577.96` | `10.141 / 14.675s` | 26 |
| MXFP8 FA4 + Triton | 8 qps | `25.465s` | `643.38` | `12.393 / 18.783s` | 40 |
| MXFP8 FA4 + Triton | 16 qps | `25.549s` | `641.29` | `14.500 / 22.429s` | 48 |
| MXFP8 FA4 + Triton | 32 qps | `23.753s` | `689.77` | `14.245 / 22.315s` | 48 |
| MXFP8 FA4 + Triton | infinite | **`20.256s`** | **`808.83`** | `12.712 / 20.244s` | 48 |

单卡 fixed256 的可持续完成率约为 BF16 `2.6 req/s`、MXFP8 `3.2 req/s`；
因此从 4–8 qps 开始持续积压是容量限制，不是请求失败。MXFP8 在 2 qps
保持 queue max 9；更高到达率应增加独立 replica，而不是继续提高单卡
`max_num_seqs`。

### Docker 镜像、KL 定义和单节点启动命令

发布镜像：

```text
buaahsh/pytorch:26.02-b200-vllm-0725
```

以下实测均使用 YOCO-v3/L3 28k：

```text
/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf
```

#### llm-train baseline 和 KL

checkpoint 的 `metadata.json` 记录的训练设置为：

- `quant_mode=mxfp8`、`quant_block_size=128`；
- `use_cute=true`，即 llm-train attention 使用 FA4；
- `max_seq_len=8192`、training dataloader `batch_size=2`；
- diff-v3、`universal_loop=3`、latent MoE dim `1024`；
- 128 routed experts、top-k 8、shared expert dim `1280`；
- Q/K weighted RMSClip limit `3.0`、SwiGLU limit `10.0`。

RL 对齐固定两套 llm-train teacher-forcing baseline：

1. **BF16+FA4**：加载同一个 28k checkpoint，只将
   `modelargs.quant_mode` 覆盖为 `bfloat16`；
2. **MXFP8+FA4**：保持训练时的 `mxfp8` 和 128-element quant block。

两套 baseline 都使用 FA4 beta13、CuTeDSL 4.5.1、Native batch size 16。
对多个 vLLM replica，Native scorer 保留每个 server 收到的 round-robin
request group，不能重新混排 batch。

vLLM 使用 `temperature=1`、`top_p=1`、`top_k=0`、`min_tokens=0` 采样，
保存每个 sampled token 的 raw pre-sampling logprob。llm-train 对完全相同的
prompt 和 completion 做 full causal teacher forcing。对 action
`a ~ p_vllm(.|s)`：

```text
sampled KL = mean(log p_vllm(a|s) - log p_train(a|s))
k3 KL      = mean(exp(log p_train(a|s) - log p_vllm(a|s))
                  - 1
                  - (log p_train(a|s) - log p_vllm(a|s)))
```

文档以非负、低方差的 mean k3 KL 判断 `<1e-2` / `<5e-3`。host 速度是
64 个 query、最多 256 output tokens 的 aggregate output tok/s，包含 HTTP
和 logprob 返回成本，不包含 server 启动及首次 CuTe/Triton JIT。

#### 已实测最佳结果

| GPU | vLLM precision/backend | 64-query wall | output tok/s | matched llm-train k3 KL |
| ---: | --- | ---: | ---: | ---: |
| 1 | BF16, FlashInfer attention, Triton MoE | `24.903s` | `620.44` | **`0.00294958`** |
| 1 | MXFP8, FA4 attention, Triton MoE | **`20.733s`** | **`765.56`** | `0.01024826` |
| 4 | BF16 TP4+EP4, FA4 attention, Triton MoE | `5.964s` | **`2625.99`** | **`0.00294112`** |
| 4 | MXFP8 4xTP1, FA4 attention, DeepGEMM MoE | `6.463s` | `2390.39` | **`0.01054670`** |
| 4 | MXFP8 4xTP1, FA4 attention, Triton MoE, fixed256 load | **`5.118s`** | **`3201.43`** | 不适用 |
| 8 | BF16 2xTP4+EP4 | 未实测 | 未实测 | 未实测 |
| 8 | MXFP8 8xTP1 | 未实测 | 未实测 | 未实测 |

BF16 单卡和四卡都通过 `<5e-3`。MXFP8 目前没有稳定通过 `<1e-2`：
DeepGEMM 是四卡严格 KL 最低配置；Triton 的 `3201.43 tok/s` 来自强制
256-token load test，只能用于吞吐，不能用 raw logprob 计算严格 KL。8 卡
没有填入外推值：本轮构建时 GPU 2–5 被其他 workload 持续占用，无法获得
可信的单节点 8 卡 KL 和速度。

所有命令先设置：

```bash
IMAGE=buaahsh/pytorch:26.02-b200-vllm-0725
MODEL=/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf
COMMON_DOCKER_ARGS=(
  --rm
  --ipc=host
  --network=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v /mnt/pvc/lidong1:/mnt/pvc/lidong1:ro
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0
)
```

**单卡 BF16，当前严格 KL 最优且速度略优于 FA4：**

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --name yoco-bf16-1gpu \
  --gpus '"device=0"' \
  "$IMAGE" vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name yoco-v3 \
  --trust-remote-code \
  --max-model-len 1024 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --no-async-scheduling \
  --attention-backend FLASHINFER \
  --compilation-config \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

**单卡 MXFP8，当前速度和 KL 都优于其他已测 MXFP8 attention/MoE 组合：**

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --name yoco-mxfp8-1gpu \
  --gpus '"device=0"' \
  "$IMAGE" vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name yoco-v3 \
  --trust-remote-code \
  --max-model-len 1024 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --quantization fp8_per_block \
  --moe-backend triton \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --no-async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
  --compilation-config \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

**四卡 BF16，单个 TP4+EP4 endpoint：**

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --name yoco-bf16-4gpu \
  --gpus '"device=0,1,2,3"' \
  "$IMAGE" vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name yoco-v3 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 1024 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --no-async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
  --compilation-config \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

**四卡 MXFP8，四个 TP1 replica；DeepGEMM 为 KL 最低，纯速度可改
`MOE_BACKEND=triton`：**

```bash
MOE_BACKEND=deep_gemm
for gpu in 0 1 2 3; do
  port=$((8100 + gpu))
  docker run -d "${COMMON_DOCKER_ARGS[@]}" \
    --name "yoco-mxfp8-gpu${gpu}" \
    --gpus "device=${gpu}" \
    "$IMAGE" vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$port" \
    --served-model-name yoco-v3 \
    --trust-remote-code \
    --max-model-len 1024 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --quantization fp8_per_block \
    --moe-backend "$MOE_BACKEND" \
    --enable-chunked-prefill \
    --kv-sharing-fast-prefill \
    --no-async-scheduling \
    --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
    --compilation-config \
      '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
done
```

**八卡 BF16 推荐拓扑，两个 TP4+EP4 replica：**

```bash
for spec in "0,1,2,3:8200:0" "4,5,6,7:8201:1"; do
  IFS=: read -r gpus port replica <<<"$spec"
  docker run -d "${COMMON_DOCKER_ARGS[@]}" \
    --name "yoco-bf16-tp4-${replica}" \
    --gpus "\"device=${gpus}\"" \
    "$IMAGE" vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$port" \
    --served-model-name yoco-v3 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 1024 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --moe-backend triton \
    --enable-chunked-prefill \
    --kv-sharing-fast-prefill \
    --no-async-scheduling \
    --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
    --compilation-config \
      '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
done
```

**八卡 MXFP8 推荐拓扑，八个 TP1 replica：**

```bash
MOE_BACKEND=deep_gemm
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((8300 + gpu))
  docker run -d "${COMMON_DOCKER_ARGS[@]}" \
    --name "yoco-mxfp8-gpu${gpu}" \
    --gpus "device=${gpu}" \
    "$IMAGE" vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$port" \
    --served-model-name yoco-v3 \
    --trust-remote-code \
    --max-model-len 1024 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --quantization fp8_per_block \
    --moe-backend "$MOE_BACKEND" \
    --enable-chunked-prefill \
    --kv-sharing-fast-prefill \
    --no-async-scheduling \
    --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
    --compilation-config \
      '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
done
```

多 replica 配置需要由调用端 round-robin 到全部端口；严格 KL scoring 也必须
保留相同 server grouping。不要使用 `--moe-backend auto`。

### FA4 matched reference 配置

以下配置用于两侧 attention/MoE 尽量 matched 的 reference 实验，不是上述
单卡 backend sweep 的速度最优配置。它不使用 eager，并让 prefill 和 CUDA
Graph decode 都使用 FA4：

- `--max-num-seqs 16`
- `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'`
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

| 验证矩阵 | Native -> vLLM mean KL | 结论 |
| --- | ---: | --- |
| BF16，batch 1 | `0.00392071` | 通过 `< 0.01`，但不是 exact |
| BF16，batch 16 | `0.00283000` | 通过 `< 0.01` |
| MXFP8，batch 1，mixed5 | `0.0182856` | 未达到 `< 0.01` |
| MXFP8，batch 16 | `0.0000724250` | 通过 `< 0.01` |
| BF16，TP2 + EP2，batch 16 | `0.00509094` | 通过 `< 0.01` |
| MXFP8，TP2 + EP2，batch 16 | `0.00959670` | 通过 `< 0.01` |

BF16 batch 1 当前 mean KL 为 `0.00392071`，尚未达到逐元素一致。主要差异是
Native 使用 DeepGEMM grouped BF16 routed MoE，而 vLLM 使用 Triton BF16 MoE。

MXFP8 batch 1 的五个 prompt 中，短 prompt KL 为 `0.0011588`，最长中文
prompt 为 `0.0670810`；这部分是 DeepGEMM/activation-quant 小 batch geometry
差异，不是 FA4 graph replay。batch 16 已接近逐元素一致。

`FULL_DECODE_ONLY` 和 `FULL` 均成功 capture `[1, 2, 4, 8, 16]`。BF16 的
四请求、32-token graph 输出与 FA4 eager 逐 token 完全相同；MXFP8 的 `FULL`
与 `FULL_DECODE_ONLY` 也逐 token 完全相同。未出现 graph 导致的乱码、连续
单 token collapse 或首 token 卡死。一个 BF16 中文 greedy completion 会重复
完整 prompt，但 eager 和 graph 完全一致，因此不是 CUDA Graph corruption。

约 4.7K-token 输入按 1024 tokens 分块时，打开 KV-sharing fast prefill 对
同一 vLLM FA4 chunked 输出的增量 KL 为：BF16 `0.000121237`、MXFP8
`0.00400451`，均小于 `1e-2`。

### RL rollout：64 prompts、四卡推理

`tools/yoco_alignment/rl_rollout.py` 模拟标准 RL rollout：

- 64 个唯一中英文、代码、数学和系统设计 chat prompts，prompt 长度
  `28–48` tokens，平均 `37.1`；
- 通过标准 OpenAI completions API 一次提交并发请求；
- `temperature=1`、`top_p=1`、`top_k=0`，保存 sampled token IDs 和 raw
  sampled-token logprobs；
- KL rollout 固定 `min_tokens=0`，默认 `max_tokens=512`；vLLM 返回的是
  stop-token mask 之前的 raw logprob，因此非零 `min_tokens` 只能用于吞吐
  load test，不能用于 sampled KL；
- llm-train 对完全相同的 prompt + sampled completion 做 full causal
  teacher forcing，BF16/MXFP8 两侧均使用真实 FA4；
- 正式结果使用 Native batch 16。多个独立 vLLM server 时，Native scoring
  保留每个 server 实际收到的 round-robin batch composition；
- 吞吐包含 HTTP API 和 logprob 返回开销，不包含 server 启动以及第一次约
  `48–77` 秒的 Triton/FA4 JIT warmup。

这里的 KL 是 RL 实际可获得的 sampled-action estimator，不是每个位置完整词表
的精确 KL：

```text
sampled KL = log p_vllm(a|s) - log p_train(a|s)
k3 KL      = exp(log p_train - log p_vllm) - 1
             - (log p_train - log p_vllm)
```

以非负、低方差的 mean k3 KL 作为 `<1e-2` / `<5e-3` 判断。下表是旧 v2
`min_tokens=64` 的历史 sweep，因此吞吐仍有效，但 KL 列只能作为 logprob
drift 参考，不能作为严格 on-policy KL 验收：

| vLLM setting | 稳态 output tok/s | mean k3 KL | 结论 |
| --- | ---: | ---: | --- |
| BF16 TP1 | `2354` | `0.00391659` | 通过 `<5e-3` |
| BF16 TP2 + EP2 | `3258` | `0.00426475` | 通过 `<5e-3` |
| BF16 TP4 + EP4 | **`4014`** | `0.00418782` | **通过 `<5e-3`，当前最优** |
| BF16 DP4 + EP4 | `3829` | `0.00417150` | 通过 `<5e-3` |
| BF16 4 个独立 TP1 replica | `3453` | `0.00424997` | 通过 `<5e-3` |
| MXFP8 TP1 | `2905` | `0.01051388` | 略高于 `<1e-2` |
| MXFP8 TP2 + EP2 | `3574` | `0.01048570` | 略高于 `<1e-2` |
| MXFP8 DP4 + EP4 | `3028` | `0.01057012` | 未通过；重复吞吐约 `3.0–3.8k` |
| MXFP8 4 个独立 TP1 replica | `3655` | `0.01010610` | 非常接近，但未稳定通过 |

MXFP8 TP1 限制长度时，`max_tokens=256` 为 `2847 tok/s`、
k3 KL=`0.00990026`；`max_tokens=128` 为 `2760 tok/s`、
k3 KL=`0.00903105`。但四卡 DP/replica 下缩短长度没有稳定降低 KL，例如
DP4+EP4 的 max256 k3 KL=`0.01093448`。不同 setting 会 sampled 出不同内容，
因此不能把长度 sweep 当成同一 token 集上的单调曲线。

当前四卡 RL rollout 推荐 **BF16 TP4 + EP4**。它同时比所有已测四卡 MXFP8
setting 更快且 KL 更低；在每卡只有约 16 条 active sequence 时，MXFP8 的
activation quant、scale 处理、MoE routed-row padding 和通信开销没有被足够
大的 GEMM 摊薄。

推荐启动命令：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --host 127.0.0.1 \
  --port 8107 \
  --served-model-name yoco \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --no-async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
  --compilation-config \
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

`rl_rollout.py rollout` 可以重复传入 `--url`，将 prompts round-robin 分给多个
独立 TP1 server，并按所有 server 的共同 wall time 计算 aggregate throughput。

### 尚未通过的矩阵

- 真实 FA4 chunked prefill：约 4.7K-token 输入按 1024 tokens 分块时，
  Native -> vLLM mean KL 为 BF16 `0.285164`、MXFP8 `0.0209809`。因此参数可
  正常运行，KV-sharing 也不会额外破坏结果，但长 prompt 的 matched chunked
  reference 仍未达到 `< 0.01`。
- MXFP8 batch 1 mixed5：mean KL `0.0182856`；长 prompt 尚未通过。
- BF16 batch 1：mean KL `0.00392071`，不是 exact zero。

### Batch invariance

batch 大于 1 时不要求逐元素一致，验收标准是完整词表 aggregate mean KL
小于 `1e-2`，同时 decoding 不重复、不乱码。Native reference 使用和 vLLM
scheduler 一致的 `1 + 15` forward shape。

## TODO 与当前对齐程度

- [x] **FA4 matched matrix**
    - 当前状态：BF16/MXFP8 batch 16、FA4 CUDA Graph decode、TP2/EP2 均通过。
    - vLLM 和 Native 同时使用 FA4 beta13、CuTeDSL 4.5.1，并固定
      `num_splits=1`。
- [x] **RL rollout simulation**
    - 当前状态：64 prompts、标准 `vllm serve`、sampled logprob、Native FA4
      teacher forcing 和四卡 TP/DP/replica sweep 已完成。
    - 四卡推荐 BF16 TP4+EP4：`4014 tok/s`、k3 KL=`0.00418782`。
      四卡 MXFP8 最接近结果为 4 个 TP1 replica：`3655 tok/s`、
      k3 KL=`0.01010610`，尚未稳定通过 `<1e-2`。
- [x] **vLLM backend sweep**
    - 当前状态：llm-train 固定 FA4，已完成 vLLM FA4/FA2/FlashInfer
      attention、Triton/DeepGEMM/FlashInfer/CUTLASS/auto MoE probe、max256
      KL 和 2/4/8/16/32/infinite qps sweep。
    - 单卡推荐 BF16 FlashInfer attention + Triton MoE；MXFP8 使用 FA4
      attention + Triton MoE，但 MXFP8 k3 KL=`0.01024826` 仍略高于阈值。
- [ ] **Batch invariance**
    - 当前状态：batch 16 已通过。MXFP8 mean KL 为 `0.0000724250`，BF16 为
      `0.00283000`；MXFP8 batch 1 mixed5 仍为 `0.0182856`。
    - 下一步：继续降低 scheduler shape 和 packed-row geometry 导致的差异，
    并验证更多 batch size；所有 Native reference 必须复现 vLLM 的实际
    forward shape。
- [ ] **真实 chunked prefill**
    - 当前状态：FA4 cache-backed Native reference 已确认真实调用 CuTeDSL。
      约 4.7K-token prompt 按 1024 tokens 切分时 BF16/MXFP8 仍未通过。
    - 下一步：定位 cache-backed prefill 中 attention 与 MoE shape drift；
    在通过前，开启 `--enable-chunked-prefill` 不等于真实切分路径已对齐。
- [ ] **BF16 batch 1 exact**
    - 当前状态：mean KL 为 `0.00392071`，满足 `< 0.01`，但未达到 exact zero。
    - 下一步：实现与 Native grouped DeepGEMM BF16 routed MoE 等价的路径。
- [x] **Docker build/run**
    - 已构建 `buaahsh/pytorch:26.02-b200-vllm-0725`，镜像约 30.96 GB。
    - 容器内确认 vLLM `0.1.dev1+gf4964c907.d20260726`、FA4 `4.0.0b13`、
      CuTeDSL `4.5.1` 和 B200 CUDA 可用；FA4 non-eager vLLM server 成功
      ready，并通过 OpenAI chat completion 生成连贯中文输出。

### CUDA Graph 状态

已验收配置：

```json
{
  "mode": 0,
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [1, 2, 4, 8, 16]
}
```

FA4 beta13 已确认可用于上述 full CUDA Graph 配置。旧代码中的两层安全
fallback 会把 YOCO FA4 强制切到 Triton；Triton 在 batch 16 graph capture
出现 illegal memory，不能用于判断 FA4 是否安全。

## 实现要点

### FlashAttention 4

- FA2/FA3 native extensions 使用提供 CMake targets 的 vLLM FlashAttention
  commit `bce29425653ec0fbc579d329883030e832d15ada` 构建；
- PT 26.02 预装的 FA4 `4.0.0b13`
  (`9bad4bec7326ad28edb5516b8878fd283f8991c0`) 是 Python/CuTeDSL-only
  package，native build 后复制到 `vllm.vllm_flash_attn.cute` namespace；
- CuTeDSL 固定为 `4.5.1`，QuACK 固定为 `0.4.1`；
- YOCO FA4 full graph 不再在 model config 或 attention backend 内切换到
  Triton；
- YOCO FA4 固定 `num_splits=1`，与 Native beta13 默认执行形状一致。

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
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
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

### YOCO-v2 legacy 0716 镜像

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
      '{"backend":"FLASH_ATTN","flash_attn_version":4}' \
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
  --native-use-cute \
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
  --flash-attn-version 4 \
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

该 FA4 配置的 MXFP8 batch 16 mean KL 为 `0.0000724250`，graph decode
正常。真实 chunked prefill 尚未通过，不能仅凭
`--enable-chunked-prefill` 启动成功判定对齐。

## 构建 B200 image

`docker/Dockerfile.b200` clone 固定 vLLM commit，使用 CMake-compatible
FlashAttention revision 构建 FA2/FA3 native extensions，再从 PT 26.02
overlay FA4 beta13，并固定 CuTeDSL 4.5.1、QuACK 0.4.1 和当前 YOCO Python
实现：

```bash
docker build \
  -f docker/Dockerfile.b200 \
  -t buaahsh/pytorch:26.02-b200-vllm-0725 \
  .
```

该 Dockerfile 保留 `donglixp/pytorch:26.02-b200` 中的 Python、PyTorch 和
CUDA 环境，只在固定的 vLLM 基线提交上覆盖本次需要的 Python runtime 文件。
本轮实际构建镜像约 30.96 GB；容器内已确认 B200、FA4 beta13、CuTeDSL
4.5.1、CUDA Graph 和标准 `vllm serve` 可用。

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
  -t buaahsh/pytorch:26.02-b200-vllm-0725 \
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
  --cache-from type=registry,ref=buaahsh/pytorch:26.02-b200-vllm-0725-buildcache \
  --cache-to type=registry,ref=buaahsh/pytorch:26.02-b200-vllm-0725-buildcache,mode=max \
  --push \
  -f docker/Dockerfile.b200 \
  -t buaahsh/pytorch:26.02-b200-vllm-0725 \
  .
```

进一步降低发布延迟时，可以将固定 commit 的完整编译结果发布为不可变
`vllm-base` tag，再用只包含 Python `COPY` 的薄 overlay image 作为最终 tag。
这样 Python-only 发布不会再次经过 vLLM 安装阶段，也不依赖构建机的本地缓存。

## 相关文件

```text
convert_to_hf.py
tools/yoco_alignment/logprob_kl.py
tools/yoco_alignment/rl_rollout.py
vllm/model_executor/models/yoco.py
vllm/model_executor/models/config.py
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
vllm/v1/attention/backends/flash_attn.py
vllm/entrypoints/openai/chat_completion/protocol.py
vllm/parser/agens_parser.py
docker/Dockerfile.b200
```
