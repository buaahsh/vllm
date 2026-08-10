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
