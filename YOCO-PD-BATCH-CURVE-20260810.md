# YOCO NIXL PD batch 容量、吞吐与前向延迟

## 结论

- 测试拓扑是 4 x B200 上的 `P=TP2/DP1, D=TP2/DP1`，代码为
  `fhb-dev@fa8e4eac6a`。
- 当前参数允许并已完成 `max_num_seqs=256` 的短上下文 decode；Prefill 单次
  token batch 上限是 `max_num_batched_tokens=8192`。这两个数字是本轮验证过的
  配置上限，不是继续增加显存压力后的理论极限。
- 实用 decode batch 建议保持 `<=32`。完整 CUDA Graph 只 capture 到 32；实际
  active batch 从 32 增加到 48 后，GPU 同步的 model-forward 中位数从
  `8.08 ms` 跳到 `68.88 ms`，约为 `8.5x`。
- 吞吐最佳点也是 batch 32：D 端 `1,139.27 output tok/s`，PD 端到端
  `1,078.67 output tok/s`、`4.21 req/s`。batch 256 虽能完成，但 D 端只有
  `1,079.91 output tok/s`，端到端 `1,005.90 output tok/s`，同时单请求 p50
  增加到约 `64.66 s`，不适合作为默认值。
- 当前 TP2/DP1 下，1.3K、强制 256 decode、8K、65K 四个 standalone -> PD
  case 均逐 token exact；8K concurrency=4 的 8 个请求也只有一个 token trace。
  这不能推广成所有并行方式均正确：DP2 + CUDA Graph 的并发输出仍有多 trace，
  PCP/DCP 也未在本轮四卡拓扑覆盖。

## 测试口径

```text
Job:                    bonete01/lidong1-yoco-pd-diagnose-g4-0810
Node:                   slc01-cl02-hgx-0297
GPU:                    4 x NVIDIA B200
Topology:               P=TP2/DP1, D=TP2/DP1
max_num_seqs:            256
max_num_batched_tokens:  8192
CUDA Graph capture:      batch 1--32
KV transfer:             NIXL
```

Prefill 吞吐负载为每请求 8,192 个有效 input tokens；Decode 吞吐负载为每请求
1,345-token context + 256 强制输出。前向延迟取 vLLM 日志中 GPU 同步后的
`Batchsize forward time stats` 中位数，不使用 HTTP wall time 替代。

## 正确性覆盖

| Prompt / 输出 | Standalone -> PD token trace | 文本 | 结果 |
| --- | --- | --- | --- |
| 1,356 input，自然停止 6 tokens | exact | exact | 通过 |
| 1,356 input，强制 256 tokens | exact | exact | 通过 |
| 7,999 input，自然停止 21 tokens | exact | exact | 通过 |
| 65,809 input，自然停止 6 tokens | exact | exact | 通过 |
| 7,999 input，concurrency 4、8 requests | 1 个 unique trace | 一致 | 通过 |

batch 1--256 的性能扫描全部成功完成并返回预期数量的 completion tokens，但该扫描
不是逐 batch 的 standalone -> PD token-exact 矩阵，因此不能把“成功完成”解释成
每个 batch 都另做了一轮逐 token 对照。

## 吞吐结果

| 并发 batch | P 端 8K input tok/s | D 端 output tok/s | PD 端到端 output tok/s | E2E req/s | E2E p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59,045.70 | 146.32 | 139.32 | 0.54 | 1.84 s |
| 2 | 63,625.81 | 250.68 | 235.33 | 0.92 | 2.17 s |
| 4 | 66,880.88 | 381.25 | 363.23 | 1.42 | 2.81 s |
| 8 | 70,154.17 | 678.62 | 600.85 | 2.35 | 3.40 s |
| 16 | 73,685.29 | 898.52 | 834.45 | 3.26 | 4.89 s |
| **32** | **74,344.02** | **1,139.27** | **1,078.67** | **4.21** | **7.55 s** |
| 48 | 77,851.85 | 477.06 | 458.14 | 1.79 | 26.75 s |
| 64 | 72,854.28 | 569.28 | 536.29 | 2.09 | 30.43 s |
| 96 | 67,679.70 | 713.64 | 687.65 | 2.69 | 35.60 s |
| 128 | 68,747.18 | 809.82 | 798.47 | 3.12 | 40.86 s |
| 192 | 68,829.52 | 975.41 | 924.55 | 3.61 | 52.80 s |
| 256 | 69,491.64 | 1,079.91 | 1,005.90 | 3.93 | 64.66 s |

Prefill 吞吐在 batch 48 达到本轮峰值 `77.85K input tok/s`，但端到端系统受
Decode 限制；因此综合吞吐和时延的推荐点仍是 batch 32。

## GPU 同步前向延迟

### Decode sequence batch

| 实际 active batch | 样本数 | forward median |
| ---: | ---: | ---: |
| 1 | 989 | 5.85 ms |
| 2 | 125 | 6.40 ms |
| 4 | 131 | 7.56 ms |
| 8 | 78 | 6.28 ms |
| 16 | 119 | 6.91 ms |
| 32 | 70 | 8.08 ms |
| 48 | 42 | 68.88 ms |
| 59 | 44 | 68.56 ms |
| 94 | 58 | 69.80 ms |
| 127 | 40 | 69.99 ms |
| 188 | 33 | 67.94 ms |
| 243 | 44 | 69.05 ms |

64、96、128、192、256 个同时提交的请求在生成过程中会因完成时刻不同变成略小的
actual active batch；上表为每组的主要稳定桶，例如 nominal 256 对应日志中的
actual 243/252，而不是伪造一个不存在的 exact-256 forward 样本。

### Prefill token batch

| token batch | 样本数 | forward median |
| ---: | ---: | ---: |
| 128 | 6 | 53.24 ms |
| 256 | 6 | 52.88 ms |
| 512 | 6 | 52.90 ms |
| 1,024 | 6 | 53.02 ms |
| 2,048 | 6 | 52.79 ms |
| 4,096 | 6 | 53.02 ms |
| 8,192 | 132 | 82.85 ms |

## 证据与资源回收

完整 correctness、throughput、forward stats、客户端记录、服务日志、运行时 source
hash 和镜像信息位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/
  pd-batch-curve-fa8e4eac6a-20260810/
```

`correctness-rc`、吞吐客户端、forward 客户端和解析器返回码均为 0；测试完成时间为
`2026-08-10T10:04:01+00:00`。测试完成后已删除 Volcano Job
`lidong1-yoco-pd-diagnose-g4-0810`，对应 Job 和 Pod 均确认 `NotFound`。

## `fhb-dev@9b9a945c06` 极限吞吐补测

### 结论

在相同的 `P=TP2/DP1, D=TP2/DP1` 四卡拓扑上，扩大 P token budget 和 D CUDA
Graph capture 后，本轮观测到的容量上限为：

| 侧 | 负载 | 峰值点 | 峰值吞吐 | 该点 p50 |
| --- | --- | ---: | ---: | ---: |
| P | 8,192 effective input + 1 output | batch 64 | **84,055 input tok/s** | 3.855 s |
| D | 1,345 context + 512 output | batch 256 | **2,284 output tok/s** | 57.107 s |
| PD | 1,345 input + 256 output | batch 96 | **1,128 output tok/s / 4.407 req/s** | 21.684 s |

这里的“峰值”是当前 `max_num_seqs=256`、固定波次测试中观测到的最大值，不是无穷
队列下的理论硬件上限。D 从 batch 192 到 256 只增加 `0.85%`，已接近平台；同时
单请求 p50 从 `43.25 s` 增至 `57.11 s`，所以 batch 256 只适合作为极限吞吐配置。
实用配置建议 P 保持 batch 48--64，D 保持 batch 64--96；PD batch 64 已达到峰值
端到端吞吐的 `94.6%`，p50 比 batch 96 低约 `6.42 s`。

### 环境和参数

```text
Date:                    2026-08-10 PDT / 2026-08-11 UTC
Job:                     bonete01/lidong1-yoco-pd-diagnose-g4-0810
Node:                    slc01-cl02-hgx-0228
Allocated physical GPU:  P=2,3; D=4,5; 4 x NVIDIA B200 183,359 MiB
Code:                    fhb-dev@9b9a945c065798f2f6ff4180d3f299a81eff9135
Topology:                P=TP2/DP1, D=TP2/DP1
P token budget:          32,768
D token budget:          8,192
max_num_seqs:            256
KV transfer:             NIXL 1.3.2 + UCX 1.21, same-node lo/IPC
```

固定镜像 `yoco-pd-current:4a39087d27` 上只读挂载当前 HEAD 的三个运行时文件，避免
把旧镜像代码冒充当前分支：

```text
5f3009c4...  vllm/model_executor/models/yoco.py
82705781...  vllm/v1/core/sched/scheduler.py
ea5192d4...  vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
```

P 使用 `FULL_AND_PIECEWISE`，最大 token batch 从旧测的 8,192 提高到 32,768。
D 先捕获到 batch 128，再补测捕获到 256：

```json
{"cudagraph_mode":"FULL_AND_PIECEWISE",
 "cudagraph_capture_sizes":[1,2,4,8,16,32,48,64,96,128,160,192,224,256]}
```

graph-128 和 graph-256 的估算显存分别为 `0.61 GiB` 和 `0.71 GiB`/卡；扩到 256
只多约 `0.10 GiB`。可用 KV capacity 从 `14,627,687` 降至 `14,606,442` tokens，
代价约 `0.15%`。

### 正确性门禁

graph-128 和 graph-256 profile 都在性能测试前执行同一门禁，四个主 case 均与
TP2 standalone 逐 token exact：

| Prompt / 输出 | 文本 | token trace |
| --- | --- | --- |
| 1,356 input，自然停止 6 tokens | exact | exact |
| 1,356 input，强制 256 tokens | exact | exact |
| 7,999 input，自然停止 21 tokens | exact | exact |
| 65,809 input，自然停止 6 tokens | exact | exact |

另外 8 个 `7,999 input`、concurrency 4 的重复请求只有一个 unique token trace。
两轮 correctness 和 benchmark 共四个返回码均为 0；服务日志中没有 NIXL transfer
error、failed notification、CUDA OOM 或 traceback。

### P 侧扫描

每请求发送 8,193 prompt tokens；pure-P 按 block boundary 实际计算并发布 8,192
tokens。计时到 producer HTTP response 和 metadata 返回为止，随后由 D 消费 KV
并释放 lease，但清理不计入 P wall time。每个请求使用独立 cache salt。

| Batch | P input tok/s | P request p50 | P request p95 |
| ---: | ---: | ---: | ---: |
| 1 | 58,919 | 0.139 s | 0.140 s |
| 8 | 70,386 | 0.750 s | 0.971 s |
| 16 | 76,601 | 1.243 s | 1.765 s |
| 32 | 78,199 | 2.301 s | 3.222 s |
| 48 | 82,582 | 3.072 s | 4.620 s |
| **64** | **84,055** | **3.855 s** | **5.963 s** |
| 96 | 74,943 | 6.945 s | 10.274 s |
| 128 | 74,444 | 9.308 s | 13.543 s |
| 192 | 75,193 | 13.576 s | 20.167 s |
| 256 | 75,569 | 17.921 s | 26.698 s |

峰值在 batch 64；更大并发只增加排队时延。历史 8K token-budget profile 的 P 峰值
是 `77,852 tok/s`，本轮峰值高 `7.97%`，但它是跨节点时刻、跨 HEAD 的配置补测，
不能替代同一进程内只切 token budget 的严格 A/B。

### D 侧扫描和 CUDA Graph 边界

P 预先生成 NIXL metadata，D 计时包含首次 KV pull、尾段重算和强制生成 512 tokens；
P preparation 不计入 D wall time。graph 只捕获到 128 时，超过边界立即退化：

| D batch | graph-128 output tok/s | graph-256 output tok/s |
| ---: | ---: | ---: |
| 64 | 1,946 | 1,916 |
| 96 | 2,032 | 2,094 |
| 128 | 2,115 | 2,174 |
| 192 | 1,353 | 2,265 |
| 256 | 1,546 | **2,284** |

batch 64 的 `-1.5%` 和 batch 128 的 `+2.8%` 属于独立波次的运行抖动；决定性证据是
192/256 不再从 graph 跌回 eager。graph-256 内部从 batch 128 增至 256，吞吐只再
增加 `5.1%`，而 p50 从 `29.98 s` 增至 `57.11 s`。因此推荐：

- 容量压测或离线吞吐：capture 到 256，允许 active batch 256；
- 在线长输出：capture 到 256 以防突发，但 admission target 设在 64--96；
- 不再使用“只 capture 到 32 且允许 256 并发”的组合，否则 batch 48 开始会出现
  可重复的 eager cliff。

### 端到端两阶段流水线

每个 worker 严格执行 P response -> D request；不同 worker 之间并行，因此 P、NIXL
transfer 和 D 可以跨请求重叠。负载为 1,345 input + 256 强制 output：

| 并发 | Output tok/s | Req/s | Request p50 | Request p95 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 750.60 | 2.932 | 5.226 s | 5.452 s |
| 32 | 976.76 | 3.815 | 8.360 s | 8.368 s |
| 48 | 997.15 | 3.895 | 12.277 s | 12.287 s |
| 64 | 1,067.71 | 4.171 | 15.264 s | 15.278 s |
| **96** | **1,128.27** | **4.407** | **21.684 s** | **21.706 s** |
| 128 | 1,111.09 | 4.340 | 29.339 s | 29.374 s |

端到端在 96 达峰，128 已经回落。该固定波次测试不是生产 Gateway 的稳态开环
arrival-rate 测试，也没有跨节点 RDMA；不能把 `4.407 req/s` 直接外推到多 P、多 D。
一次性触发 256 个 KV pull 时，D metrics 观察到最多 255 个请求短暂处于
`waiting_by_reason=deferred`，随后分波完成且无 transfer error。这说明 Gateway
还需要按 D capacity 对 KV pull 做 admission control，不能只按 HTTP 连接数放行。

### 原始证据

```text
/mnt/pvc/lidong1/vllm_test_artifacts/
  pd-saturation-9b9a945c06-20260810/
    p32k-dgraph128/
    p32k-dgraph256/
```

目录包含 benchmark/correctness JSON、客户端输出、P/D metrics、完整服务日志、
container inspect、GPU dmon、运行时 source hash 和配置。精简 JSON 也保存在测试机：

```text
/home/lidong1/vllm_test/yoco_pd/results/
  pd-saturation-9b9a945c06-20260810/
```

结果落盘后已删除 Volcano Job `lidong1-yoco-pd-diagnose-g4-0810`，Job 和 Pod 均
再次确认 `NotFound`，没有遗留占用 B200。
