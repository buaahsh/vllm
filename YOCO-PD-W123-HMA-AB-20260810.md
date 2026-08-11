# YOCO PD HMA/SWA W1/W2/W3 A/B（2026-08-10）

## 结论

在同节点 4 x B200、`P=TP2/DP1`、`D=TP2/DP1` 的真实双向 NIXL PD
链路上，比较当前 HMA/SWA window 与
`--disable-hybrid-kv-cache-manager` full-context baseline：

- W1 是 TTFT/流量/容量优化，不是吞吐优化。HMA 把 TTFT 降低
  `84.6%--89.9%`、流量降低 `10.77x`，但本轮输出吞吐回退
  `2.5%--3.1%`；
- W2 同时得到延迟和吞吐收益。batch 1/4 吞吐分别提升 `12.8%/43.4%`，
  TTFT 分别降低 `91.6%/94.9%`，流量降低 `25.11x`；
- W3 是收益最大的生产形态。batch 1/4 吞吐分别提升 `553%/1034%`，
  wall time 分别缩短 `84.7%/91.2%`，双向流量降低 `21.16x`；
- 六个主性能点的 NIXL failed transfer/notification counter 均为 0，
  没有 CUDA OOM；本轮为同节点 `UCX_NET_DEVICES=lo`/CUDA IPC，不代表
  跨节点 RDMA 结果。

因此建议 W2/W3 默认启用 HMA；W1 若主要目标是 TTFT 和可容纳 session 数也应启用，
但不能宣称它提高 65K-output 总吞吐。

## 测试口径

```text
Date:                    2026-08-10 PDT / 2026-08-11 UTC
Job:                     bonete01/lidong1-yoco-pd-w123-g4-0810
Node:                    slc01-cl02-hgx-0063
GPU:                     4 x NVIDIA B200 183,359 MiB
Code:                    fhb-dev@9b9a945c065798f2f6ff4180d3f299a81eff9135
Topology:                P=TP2/DP1, D=TP2/DP1
Precision:               BF16
Attention / MoE:         FlashInfer / Triton
CUDA Graph:              FULL_AND_PIECEWISE, capture through batch 64
P / D token budget:      32,768 / 8,192
max model len / seqs:    131,072 / 64
KV transport:            NIXL 1.3.2 + UCX 1.21, same-node lo/IPC
HMA profile:             default hybrid KV cache manager
Baseline profile:        --disable-hybrid-kv-cache-manager
```

固定镜像上只读覆盖当前 HEAD 的 YOCO model、core scheduler 和 NIXL scheduler；
三个 source SHA256 与此前 saturation 测试相同：

```text
5f3009c4...  vllm/model_executor/models/yoco.py
82705781...  vllm/v1/core/sched/scheduler.py
ea5192d4...  vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
```

HMA 与 baseline 各自冷启动，使用相同模型、镜像、graph、token budget、请求、seed
和预热。唯一服务参数变量是 baseline 关闭 hybrid KV cache manager。每个正式请求
使用独立 cache salt，避免跨样本 prefix cache 命中。

Workload 沿用已有 long-context 定义：

| Workload | 输入/过程 | 输出 |
| --- | --- | --- |
| W1 | 8,192-token single turn | 65,536 tokens |
| W2 | 65,536-token single turn | 16,384 tokens |
| W3 | 40 turns，每轮约 2,925 新 prefill，最终 130K | 325 tokens/turn |

W3 客户端显式保存 D 返回的 KV metadata，并在下一轮送回 P，再把 P metadata 送到
D。因此本轮测到真实 `D -> P -> D` 双向复用；旧 W3 客户端不携带 conversation/KV
metadata，不能用于这个 A/B。

## 性能结果

下表的 tok/s 是 D 侧两卡 TP2 的聚合输出吞吐，不是单卡值。“吞吐变化”和
“wall 缩短”都以 HMA 相对 full-context 计算。

| Workload | Batch | HMA tok/s | Full tok/s | 吞吐变化 | HMA wall | Full wall | wall 缩短 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 1 | 155.07 | 160.10 | -3.14% | 422.63 s | 409.35 s | -3.24% |
| W1 | 4 | 458.92 | 470.63 | -2.49% | 571.22 s | 557.01 s | -2.55% |
| W2 | 1 | 150.89 | 133.78 | +12.79% | 108.59 s | 122.47 s | +11.34% |
| W2 | 4 | 430.44 | 300.24 | +43.37% | 152.25 s | 218.28 s | +30.25% |
| W3 | 1 | 99.99 | 15.31 | +553.02% | 130.01 s | 848.98 s | +84.69% |
| W3 | 4 | 185.22 | 16.34 | +1033.70% | 280.75 s | 3182.82 s | +91.18% |

W3 batch 4 的 wall 即 `4.68 min` 对 `53.05 min`。HMA 从 batch 1 到 batch 4
得到 `1.85x` 吞吐扩展；full-context 只有 `1.07x`，因为多 session 共享相同的
传输带宽并出现 transfer-deferred 串行/轮转。

### TTFT 与 NIXL 流量

NIXL bytes 为 P、D 两端 counter delta 之和；W3 包含 D->P 和 P->D 双向传输。

| Workload | Batch | HMA TTFT | Full TTFT | TTFT 缩短 | HMA bytes | Full bytes | 流量缩减 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 1 | 0.375 s | 2.436 s | 84.59% | 96.4 MB | 1.038 GB | 10.77x |
| W1 | 4 | 0.949 s | 9.418 s | 89.92% | 385.6 MB | 4.153 GB | 10.77x |
| W2 | 1 | 1.621 s | 19.198 s | 91.56% | 331.3 MB | 8.319 GB | 25.11x |
| W2 | 4 | 3.814 s | 75.443 s | 94.94% | 1.325 GB | 33.278 GB | 25.11x |
| W3 | 1 | 1.179 s | 19.242 s | 93.87% | 15.907 GB | 336.663 GB | 21.16x |
| W3 | 4 | 4.194 s | 74.123 s | 94.34% | 63.628 GB | 1.347 TB | 21.16x |

W1 的 65K 输出使持续 decode 占据绝大多数 wall。窗口裁剪显著改善首 token 和
传输量，但不能抵消本轮 HMA layout/scheduler 的约 2.5%--3.1% decode 吞吐差异。
W2/W3 的上下文传输占比更高，因此流量缩减可以转化为显著吞吐收益。

## 正确性与并发非确定性

主性能测试的严格 token-trace 结果：

- W1 batch 1/4：HMA 与 full-context 逐请求 exact；
- W2 batch 1/4：HMA 与 full-context 逐请求 exact；
- W3 batch 1：40/40 turns exact；
- W3 batch 4：四条最终 trace 不完全 exact。首次分叉轮次分别为 1、28、20、19。

不能把最后一项隐藏为“全过”，也不能直接据此判定 HMA 传错。追加的 batch 4、
4-turn、每配置三次复现探针得到：

```text
HMA:          repetition 0 == repetition 1; repetition 2 不同
full-context: repetition 0 == repetition 1 == repetition 2
cross-config: full 的三次都与 HMA repetition 0/1 逐 trajectory、逐 turn exact
```

也就是说，相同 HMA 服务、输入、seed 和 batch 自身会因异步请求到达/实际 batching
组合产生不同 greedy trace；同时 HMA 与 full-context 存在共同且逐 token exact 的
执行路径。主测 W3/b4 跨 profile hash 不同更符合 batch composition 引起的 BF16/
MoE/TP 数值路径变化，而不是已经证明的 KV corruption。

当前正确表述是：传输正确性在 W1/W2、W3/b1 和受控 W3/b4 路径上通过；异步
W3/b4 的“跨任意 batching 都逐 token exact”门禁没有通过。若上线要求 batch-invariant
exact，应继续固定实际 scheduler batch shape，并记录每步 top-k logits/margin 做容差
比较；不能只靠最终 40-turn hash，因为一次临界 argmax 分叉会污染后续全部 prompt。

## 证据与限制

PVC 原始证据：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/
  pd-w123-hma-ab-9b9a945c06-20260810/
```

本地精简 JSON：

```text
/home/lidong1/vllm_test/yoco_pd/results/
  pd-w123-hma-ab-9b9a945c06-20260810/
```

测试工具：

```text
/home/lidong1/vllm_test/yoco_pd/pd_w123_hma_benchmark.py
/home/lidong1/vllm_test/yoco_pd/run_pd_w123_hma_ab.sh
/home/lidong1/vllm_test/yoco_pd/pd_w3_repro_probe.py
/home/lidong1/vllm_test/yoco_pd/run_pd_w3_repro_probe.sh
```

严格 comparison 因 W3/b4 hash 不同按设计返回非零；两个 profile 性能客户端和两个
复现探针均完整返回。性能数据是有效实测，但正确性结论必须保留上述边界。本轮没有
验证跨节点 RDMA、多个 P/D 实例、生产 Gateway 路由或更高 batch。
