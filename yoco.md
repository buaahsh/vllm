# YOCO B200 最终验收报告

本文只记录本次 YOCO-v2/v3 在 B200 上的最终配置、验收方法和结果，不包含
旧镜像、旧分支或中间调试过程。

## 最终产物

Git 分支：

```text
shaohanh/yoco-serving-final-20260730
```

Docker Hub 镜像：

```text
buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final
registry digest: sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
local image id: sha256:bff890479962a9267862f3903113239f48ac9de982d184239de0a41d17f1b0e6
size: 37,667,878,332 bytes
```

镜像是自包含 runtime，不需要挂载 vLLM 源码。镜像内包含：

- YOCO-v2 DP4 compile key `1a1773b3c5`；
- YOCO-v3 DP4 compile key `b9be5626e8`；
- YOCO-v2 DP8+EP compile key `be47add45b`；
- 对应的 Torch AOT/Inductor 和首请求 Triton JIT cache。

## Prefill-only 后十层裁剪

这是最终分支之后的 follow-up 优化，尚未写入上面的最终镜像，避免与已发布
镜像内容混淆：

```text
baseline branch: shaohanh/yoco-serving-final-20260730
baseline commit: c27db1e189973cea3164ba66b1d00359d4122088
candidate branch: prefill-cut-7-31
candidate feature commit: e9f9fe5c3e48933b12f9eaf57f2cd290b726fabc
```

最终分支的普通 fast prefill 已经让后十个 cross layers 中的后九层只处理
logits token，但第一个 KV-owner cross layer 仍处理完整 prompt。本次改为由
self block 直接把共享 K/V 写入第一个 cross layer 的 cache，再让十个 cross
layers 都只处理 logits token。对于只负责传 KV 的 P 请求，十个 cross layers
全部跳过；D 侧的正常 decode 路径不变。

### P/D token 归属和启用条件

当前约定是 **P token 必须丢弃，D 负责 first token 和之后所有用户可见
token**：

1. Gateway 向 P 发送原始 prompt，并设置 `max_tokens=1`、
   `kv_transfer_params.do_remote_decode=true`；
2. P 为了完成 vLLM 请求仍会产生一个 disposable sampled token，但 Gateway
   必须丢弃 P response 的整个 `choices`，只取 `kv_transfer_params`；
3. Gateway 将原始 prompt 和 P 返回的 KV transfer metadata 一起发送给 D；
4. D 生成第一个以及后续全部用户可见 token。

这不是可选的展示策略，而是本优化的正确性前提。如果 Gateway 拼接或返回
P token，输出不保证正确，因为 P 的 disposable logits 没有经过 cross
layers。端到端测试中 P 的 HTTP `choices[0].text` 实际为一个空格 token，
验证器明确没有把它拼入结果；D 独立生成的首 token 及完整输出与 reference
exact match。因此这里的“丢弃”是 Gateway 丢弃 P 的整个 `choices`，不是要求
P 的原始 HTTP response 必须为空。

裁剪只在以下条件全部满足时启用：

- 模型为 YOCO，并开启 `--kv-sharing-fast-prefill`；
- `data_parallel_size=1`，或者 DP>1 且服务明确配置为专用 P 节点：
  `kv_transfer_config.kv_role=kv_producer`；
- 当前 active batch 中每个请求都是尚未产出 token 的 P 请求；
- 每个请求均为 `max_tokens=1` 且
  `kv_transfer_params.do_remote_decode=true`。

DP>1 时，本地 P-only 判定会打包进已有的 DP coordination all-reduce；只有
所有 rank 都确认当前 batch 为 P-only，才一起跳过后十层。任一 rank 出现
普通请求、D 请求或已经开始生成的请求，所有 rank 都保守回退到原路径，
避免后十层 MoE collective 顺序分叉。`kv_both` 不能证明实例是专用 P 节点，
所以 DP>1 时仍不启用；P 节点应使用：

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
```

### B200 正确性与性能

测试全部在同一台 8×B200 Pod 内完成，本机未运行测试。定向单测覆盖
`tests/model_executor/test_yoco_conversion.py`、
`tests/v1/worker/test_dp_utils.py` 和
`tests/v1/worker/test_gpu_model_runner.py`，结果为：

```text
46 passed, 24 warnings
```

NIXL cross-layer KV alias 专项结果为 `4 passed, 2 skipped`，`ruff check`
通过。YOCO 会给后十层建立指向同一块物理 cache 的 alias；当前 NIXL worker
若直接按 alias 名查原始 `KVCacheConfig`，会报
`KeyError: model.layers.11.self_attn.attn`。本次 baseline/candidate 都叠加了
相同的 27 行 alias 去重修复后再做 A/B，避免测试条件不一致。该修复是当前
NIXL 端到端部署的前置条件，不能只部署 `e9f9fe5c3` 而漏掉它。

端到端 P/D 正确性覆盖 `1,356 / 12,096 / 43,704` prompt tokens。warmed
DP4 的三个 case 中，candidate P/D 最终 D 输出均与单体 reference exact
match；mixed P-only/普通请求 batch 的 4 个普通请求也全部 exact match，且
DP collective 无 hang。首次冷态 P/D 请求曾返回一次空串，未计入通过结果；
生产发布应先完成健康检查和 warm-up，再接收流量。

测试 Pod 为 `bonete01/lidong1-yoco-vllm-pd-0730-master-0`，节点为
`slc01-cl02-hgx-0448`。原始日志和 JSON 结果位于：

```text
/mnt/pvc/lidong1/vllm_pd/prefill-cut-dp-validation-20260730-0448/
```

#### DP1 既有结果

下面是相同 B200 节点、相同 YOCO-v3 模型和服务参数下的 warmed A/B；每个
实例为 DP1，指标是 P 侧 prompt throughput：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `13,558` | `14,884` | `9.8%` |
| 12.1K / c1 | `40,958` | `43,177` | `5.4%` |
| 43.7K / c1 | `46,531` | `49,167` | `5.7%` |
| 12.1K / c4 | `48,601` | `51,006` | `4.9%` |
| 12.1K / c8 | `47,835` | `50,221` | `5.0%` |

长 prompt 和并发场景稳定提升约 `5%`。原因是 baseline 已经把九个 cross
layers 压缩到 logits token，本次主要再移除第一个 KV-owner cross layer
对完整 prompt 的计算，而不是从零裁掉完整十层计算，因此该收益量级符合
预期。

#### 专用 P 节点 DP4

下表使用 `kv_role=kv_producer`、`gpu-memory-utilization=0.85`，指标为
warmed P 侧 prompt throughput。baseline 和 candidate 顺序运行在同一 Pod
的 B200 上：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `10,372` | `9,513` | `-8.3%` |
| 12.1K / c1 | `51,630` | `52,000` | `0.7%` |
| 43.7K / c1 | `67,019` | `70,111` | `4.6%` |
| 12.1K / c4 | `107,174` | `112,501` | `5.0%` |
| 12.1K / c8 | `120,017` | `121,749` | `1.4%` |

短 prompt c1 没有收益；主要收益出现在长 prompt 和 c4。本优化不应宣传成
所有 shape 都更快。

#### 专用 P 节点 DP8

DP8 使用全部八张 B200、`kv_role=kv_producer` 和相同的 `0.85` 显存比例。
下表取三轮 warmed throughput 的中位数：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `9,609` | `9,977` | `3.8%` |
| 12.1K / c1 | `48,079` | `49,332` | `2.6%` |
| 43.7K / c1 | `63,805` | `68,290` | `7.0%` |
| 12.1K / c4 | `101,215` | `116,401` | `15.0%` |
| 12.1K / c8 | `131,174` | `139,720` | `6.5%` |

DP8 mixed P-only/普通请求同样为 4/4 exact 且无 hang。c4 的单轮波动很大：
baseline 为 `34.1K--113.7K`，candidate 为 `56.5K--132.1K tok/s`，所以
`15.0%` 只是三轮中位数，不应视为稳定 SLA；43.7K c1 和 12.1K c8 的提升
更有代表性。

另以 DP4 `kv_role=kv_both` 做了保守回退验收：mixed batch 4/4 exact、无
collective hang；43.7K c1 为 `68,896 tok/s`，接近 baseline 的
`67,019 tok/s`，没有误走专用 producer 才允许的 KV-only shortcut。结合
单测对 role 判定和全 rank agreement 的覆盖，P 节点可以安全使用 DP>1，
但必须明确配置 `kv_producer`；`kv_both` 在 DP>1 会保守回退。

## 验证模型

YOCO-v2 agentic serving：

```text
/mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
```

YOCO-v3/L3：

```text
/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf
```

FA4 Native/vLLM matched alignment：

```text
HF:
/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu

Native merged:
/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged
```

Qwen 对照：

```text
/mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B
```

## 完整启动命令

### YOCO-v2 DP4

```bash
docker run --rm \
  --name yoco-v2-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
  --served-model-name yoco-v2 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

该命令没有 `--enforce-eager`。普通 decode 使用 FULL CUDA Graph；
prefill/decode 混合 step 使用 PIECEWISE graph。

### Qwen3.5-35B-A3B DP4 对照

```bash
docker run --rm \
  --name qwen35-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/pvc:/mnt/pvc:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B \
  --served-model-name qwen35 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Qwen 与 YOCO 的速度对照使用同一个镜像、同一 vLLM commit、BF16、
FlashInfer attention、Triton MoE、DP4、相同 GPU 和相同请求序列。

### YOCO-v3 DP4

```bash
docker run --rm \
  --name yoco-v3-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/pvc:/mnt/pvc:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --served-model-name yoco-v3 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

### FA4 BF16 验收服务

下面的命令直接从最终镜像启动 FA4 beta13、非 eager CUDA Graph 服务：

```bash
docker run --rm \
  --name yoco-fa4-bf16 \
  --network host \
  --ipc host \
  --gpus '"device=0"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --served-model-name yoco-fa4-bf16 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-config.backend FLASH_ATTN \
  --attention-config.flash_attn_version 4 \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

MXFP8 使用相同命令并增加：

```text
--quantization fp8_per_block
```

### YOCO-v2 DP8 + EP

健康的 NVIDIA container runtime 上使用：

```bash
docker run --rm \
  --name yoco-v2-dp8-ep \
  --network host \
  --ipc host \
  --gpus all \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
  --served-model-name yoco-v2 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.68 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

本机 NVIDIA container hook 缺少 GPU2-5 的 `/proc/driver/nvidia/gpus` 条目，
因此本次 DP8 验收使用等价的 `runc` device/driver-library 映射。八个
`/dev/nvidia*`、CUDA 和 NVML 均正常；该问题属于宿主 runtime 元数据。

## Agentic 验证方法

本次性能验收代码已放入：

```text
benchmarks/multi_turn/benchmark_agent_trace.py
```

固定 workload：

- 每条 trajectory 为 40 turns；
- 每轮平均增加 1,800 个 prefill token，并强制生成 200 token；
- 每条 trajectory 合计 72K logical prefill + 8K generation；
- 最终上下文为 80K token；
- prefix 长度按 1,056 token 对齐，稳定触发 prefix cache 和 chunked prefill；
- `min_tokens=max_tokens=200`、`ignore_eos=true`，确保两模型输出 token 数一致；
- 使用 `X-data-parallel-rank` 将同一 trajectory 固定到相同 DP rank；
- 每条 trajectory 使用独立 `cache_salt`；
- 同时采集 vLLM metrics、TTFT、ITL、queue、KV cache、每卡 SM、
  memory bandwidth utilization、显存和功耗。

YOCO c8/c16/c32：

```bash
mkdir -p /tmp/agent-trace/yoco

for concurrency in 8 16 32; do
  python benchmarks/multi_turn/benchmark_agent_trace.py \
    --base-url http://127.0.0.1:8001/v1 \
    --model yoco-v2 \
    --tokenizer \
      /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
    --output /tmp/agent-trace/yoco/c${concurrency}.json \
    --concurrency "${concurrency}" \
    --trajectories "${concurrency}" \
    --turns 40 \
    --prefill-per-turn 1800 \
    --output-per-turn 200 \
    --cache-alignment 1056 \
    --dp-size 4 \
    --gpu-indices 0,1,6,7 \
    --seed 20260729
done
```

Qwen 使用同一命令，只替换：

```text
--model qwen35
--tokenizer /mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B
--output /tmp/agent-trace/qwen/c${concurrency}.json
```

输出文件：

- `<output>`：整场汇总；
- `<output>.turns.jsonl`：逐 trajectory/turn 的 prompt/output token 数、
  TTFT、ITL 和 latency；
- `<output>.runtime.json`：逐秒 vLLM metrics 和每卡 telemetry。

`computed_prefill_tokens_per_service_second` 使用 vLLM
`request_prefill_time_seconds_sum` 作分母；
`generation_tokens_per_service_second` 使用
`request_decode_time_seconds_sum` 作分母。它们用于分开比较 prefill 和 decode，
不能用整场 wall throughput 代替。

## Agentic 性能结果

三档测试分别严格生成 64K、128K、256K token；Qwen 与 YOCO 的 prompt
schedule 和 output token 数完全相同。

| c | wall Q/Y | prefill service tok/s Q/Y | decode service tok/s Q/Y | TTFT p50 Q/Y | ITL p50 Q/Y | avg SM Q/Y |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `85.59 / 109.66s` | `11,576 / 8,649` | `107.16 / 84.24` | `275 / 323ms` | `9.08 / 11.73ms` | `81.2 / 80.5%` |
| 16 | `94.53 / 123.29s` | `9,684 / 7,038` | `99.29 / 74.17` | `360 / 441ms` | `9.91 / 12.97ms` | `83.4 / 81.3%` |
| 32 | `119.42 / 152.65s` | `10,093 / 5,242` | `77.20 / 62.35` | `345 / 629ms` | `12.29 / 15.59ms` | `87.1 / 82.7%` |

其中 `Q/Y` 分别表示 Qwen/YOCO。YOCO 相对 Qwen：

- wall time 慢 `28.1% / 30.4% / 27.8%`；
- prefill service throughput 为 Qwen 的 `74.7% / 72.7% / 51.9%`；
- decode service throughput 为 Qwen 的 `78.6% / 74.7% / 80.8%`。

YOCO 运行状态：

| c | waiting max | KV max | 结论 |
| ---: | ---: | ---: | --- |
| 8 | `0` | `0.57%` | queue/KV 健康 |
| 16 | `0` | `1.24%` | queue/KV 健康 |
| 32 | `5` | `2.70%` | queue/KV 健康 |

c32 没有 eager fallback。此前未覆盖 local batch 8 的 FULL graph 已补齐；
最终 capture sizes 为 `1,2,4,8,16,32`。

Agentic 生产性能使用 FlashInfer attention，而不是 FA4。FA4 用于下面的
matched alignment 验收；在相同 40-turn workload 中，FA4 c32 明显慢于
FlashInfer，因此不作为最终吞吐配置。

## FA4 数值与 decode 验收

matched alignment 固定：

- PyTorch 26.02；
- FlashAttention `4.0.0b13`；
- FA4 commit `9bad4bec7326ad28edb5516b8878fd283f8991c0`；
- CuTeDSL `4.5.1`；
- vLLM 与 llm-train 使用相同 BF16/MXFP8、FA4 和 batch shape；
- MXFP8 两侧均使用 128-element block；
- batch 16 Native reference 使用与 scheduler 一致的 `1 + 15` forward shape；
- vLLM 使用非 eager CUDA Graph。

| 验证矩阵 | Native -> vLLM mean KL | 结论 |
| --- | ---: | --- |
| BF16，batch 1 | `0.00392071` | 通过 `<1e-2`，但不是 exact zero |
| BF16，batch 16 | `0.00283000` | 通过 `<1e-2` |
| MXFP8，batch 1，mixed5 | `0.0182856` | 未通过 `<1e-2` |
| MXFP8，batch 16 | `0.0000724250` | 通过 `<1e-2` |
| BF16，TP2+EP2，batch 16 | `0.00509094` | 通过 `<1e-2` |
| MXFP8，TP2+EP2，batch 16 | `0.00959670` | 通过 `<1e-2` |

非 eager FA4 eager/FULL/FULL_DECODE_ONLY 的 greedy decode 逐 token 一致，
没有 CUDA Graph replay 导致的乱码、首 token 卡死或单 token collapse。
最终 YOCO-v2/v3 serving smoke 中英文生成正常，没有异常重复。

约 4.7K-token 输入按 1,024 token 分块时，打开 KV-sharing fast prefill 对
同一个 vLLM FA4 chunked 输出的增量 KL 为：

| 精度 | fast prefill off -> on KL |
| --- | ---: |
| BF16 | `0.000121237` |
| MXFP8 | `0.00400451` |

因此 KV-sharing fast prefill 本身没有破坏 vLLM 输出。

尚未通过的 matched alignment：

- 约 4.7K-token 的真实 Native -> vLLM chunked prefill：
  BF16 `0.285164`、MXFP8 `0.0209809`；
- MXFP8 batch 1 mixed5：`0.0182856`；
- BF16 batch 1 满足 `<1e-2`，但不是 exact zero。

不能用“服务能开启 `--enable-chunked-prefill`”代替真实 Native-to-vLLM
chunked KL 验收。

## DP fast prefill 与 CUDA Graph

DP 下任一 rank 进入 fast prefill 时，所有 rank 必须统一进入 split
self/cross path；否则 NaiveDPEP MoE collective 的 token vector 会分叉。

最终实现将 fast-prefill metadata 打包进原 DP coordination flag：

- bit 0：ubatch；
- bit 1：fast-prefill active；
- bit 2 起：fast-prefill padded token count。

inactive rank 在其他 rank 开启 fast prefill 时使用主 batch padded count；
所有 rank 都是普通 decode 时返回 `None`，不传播 fast metadata，继续使用
普通 FULL model graph。最终 DP4 mixed prefill/decode、c32 local batch 8、
DP8+EP 均无 collective hang 或 eager fallback。

## DP8 与 YOCO-v3

DP8+EP 验收：

- 78,001-token prompt + 16-token decode：`3.329s`；
- 八张 B200 峰值 SM utilization 均为 `98-99%`；
- GPU5 包含外部约 44GB 占用时峰值显存 `168,889 MiB`；
- FlashInfer、Triton MoE、chunked prefill、fast prefill、EP 和
  `FULL_AND_PIECEWISE` 同时开启；
- 请求后中文生成正常，日志无 eager fallback。

YOCO-v3 验收：

- `diff_v3`、weighted Q/K RMSClip、latent MoE、`universal_loop=3` 可加载；
- 中文和英文直答正常；
- `<|end|>` 被服务层作为 stop string，不返回给用户；
- `get_weather({"city":"Seattle"})` tool call 可正确解析；
- 默认 `enable_thinking=false`，用户显式 chat-template kwargs 仍可覆盖。

## 启动 cache

CUDA Graph 对象不能跨进程持久化；镜像只持久化 Torch
AOT/Inductor/Triton 编译产物。最终镜像实测：

| 模型 | graph compile | graph capture | ready 时间 |
| --- | ---: | ---: | ---: |
| v2 DP4，无新 cache | `67.8s` | `6s` | 约 `5.3 min` |
| v2 DP4，baked cache | `10.5s` | `6s` | 约 `4.7 min` |
| v3 DP4，无新 cache | 约 `125s` | `5s` | 约 `249s` |
| v3 DP4，baked cache | `14.9s` | `5s` | 约 `221s` |

剩余启动时间来自权重读取、KV memory profile、模型 warmup 和 backend
初始化，不是 CUDA Graph capture。

## FLOPs 与 profiler 结论

| 指标 | YOCO-30B-A3B | Qwen3.5-35B-A3B | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| 总参数 | `32.2207B` | `35.9518B` | `0.896x` |
| decode projection + MoE GEMM | `11.943 GF/token` | `4.873 GF/token` | `2.451x` |
| 2K context core compute | `13.117 GF/token` | `5.321 GF/token` | `2.465x` |
| 40K context core compute | `25.554 GF/token` | `11.539 GF/token` | `2.214x` |
| 80K context core compute | `38.661 GF/token` | `18.093 GF/token` | `2.137x` |

YOCO 参数更少但 active FLOPs 更高，主要因为：

- 10 个 self layers 执行 `universal_loop=3`，再执行 10 个 cross layers，
  共 40 次 block execution；
- hidden size 为 3,072，Qwen 为 2,048；
- routed top-8 expert intermediate 为 1,280，Qwen 为 512；
- differential attention 增加 Q 和 combine 工作；
- cross-attention decode work 随上下文增长。

fast prefill 跳过重复 self-decoder 后，80K 理论 active compute 为 YOCO
`11.16 GF/token`、Qwen `11.54 GF/token`。因此 prefill 理论 FLOPs 接近，
剩余差距主要来自 split path、rank imbalance 和小 kernel 效率。

同镜像、同后端的短 torch profile：

| profiled c8 | Qwen | YOCO | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| wall | `6.314s` | `9.035s` | `1.431x` |
| prefill service throughput | `6,195 tok/s` | `5,370 tok/s` | `0.867x` |
| decode service throughput | `96.58 tok/s` | `75.47 tok/s` | `0.781x` |

代表性 rank 的 decode trace：

| 指标 | Qwen | YOCO | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| kernel launches / scheduler step | `1,547` | `1,859` | `1.202x` |
| summed kernel time | `3,793.7ms` | `3,920.8ms` | `1.034x` |
| union GPU busy time | `3,382.8ms` | `3,872.2ms` | `1.145x` |
| 被 overlap 隐藏的 kernel time | `410.9ms` (`10.8%`) | `48.7ms` (`1.2%`) | - |

去掉 collective 和通用 elementwise 后，MoE、GEMM、router、attention 等
主要计算热点合计为 Qwen `1,808ms`、YOCO `2,600ms`，YOCO 为 `1.44x`。
这说明 decode 差距主要是架构 active compute，同时还叠加了：

- YOCO 每 step 多 `20%` kernel launches；
- YOCO 通信/计算 overlap 明显少于 Qwen；
- RMSNorm/RMSClip、gate、differential combine、router TopK/scatter 等
  小 kernel；
- DP fast-prefill rank 不均衡和 collective 等待。

新增的 B200 Triton MoE 配置：

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=320,device_name=NVIDIA_B200.json
```

使 decode batch 8-128 的 MoE microbenchmark 提升约 `9-16%`。

后续优化优先级：

1. 合并 YOCO norm/gate/differential/router 小 kernel；
2. overlap NaiveDPEP AllGather/ReduceScatter 与 expert/shared compute；
3. 改善 DP fast-prefill chunk packing 和 rank 均衡；
4. 继续调优 tiny-token cross-decoder MoE。

## 本次相关代码

```text
benchmarks/multi_turn/benchmark_agent_trace.py
docker/Dockerfile.b200
docker/Dockerfile.b200.runtime
tests/model_executor/test_yoco_config.py
tests/parser/test_agens_parser.py
tests/v1/worker/test_dp_utils.py
vllm/entrypoints/openai/chat_completion/serving.py
vllm/forward_context.py
vllm/model_executor/models/config.py
vllm/model_executor/models/yoco.py
vllm/reasoning/agens_reasoning_parser.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/utils.py
vllm/v1/worker/dp_utils.py
vllm/v1/worker/gpu_model_runner.py
```
