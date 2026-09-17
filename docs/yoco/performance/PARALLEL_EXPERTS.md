# Parallel experts：原生 FP8 的专家段拆分

2026-09-17，B200 / TP1 / DP1，无 EP。使用回退分析中的原生 B1/B8 trace，
每档 12 步 × 40 个逻辑层，共 480 次层调用。本轮为离线 trace 与源码分析。

**专家段主要由 routed W13/W2 决定。两者的 kernel 时间合计占该段约
60.9%（B1）和 80.5%（B8）。B1 的激活加量化另占 14.2%，值得单独检查。**

- [B1：在 Perfetto 中放大专家段](http://10.209.224.207:8918/perfetto.html?trace=native-b1&range=experts)。
- [B8：在 Perfetto 中放大专家段](http://10.209.224.207:8918/perfetto.html?trace=native-b8&range=experts)。
- [结构化结果与全部层观测](PARALLEL_EXPERTS.json)。
- [模型 Graph 整体拆分](DECODE_CRITICAL_PATH.md)。

## 这段做什么

这里的 parallel 指同一张 GPU 上 shared/routed 两路重叠。当前配置没有跨卡
all-to-all 或 NCCL；不应把专家段时间解释成 EP 通信开销。

进入此段之前，Latent in/Norm 与 Router GEMM 已完成。Routed 的顺序为：

`TopK → 输入 FP8 量化 → expert layout → W13 → SwiGLU/路由权重/FP8量化 → W2 → TopK sum`

TopK 从 128 个 routed experts 中为每个 token 选 8 个。Routed 使用 latent
维度：每个专家 W13 为 `1024→7680`（gate/up 各 3840），激活输出 3840，
W2 为 `3840→1024`。最终归约 8 个专家的输出。Shared 分支使用原始 3072 维输入，
与上述 routed 工作重叠；两路汇合之后才做 routed 的 Latent out。

## 可相加的 elapsed 分解

下表是每层平均微秒数。各阶段从本阶段第一个 kernel 开始计到下一阶段第一个
kernel 开始，包含阶段间隔；最后一项归约计到结束。因此逐行相加等于专家段
elapsed，不能把这些数字全部称为 GEMM 本体耗时。

| 阶段 | B1 μs/层 | B1 占比 | B8 μs/层 | B8 占比 |
| --- | ---: | ---: | ---: | ---: |
| 分叉到 TopK 开始 | 1.272 | 2.5% | 1.149 | 1.0% |
| TopK / 输入量化 / layout，至 W13 开始 | 7.747 | 15.3% | 10.524 | 8.7% |
| W13，至激活开始 | 16.165 | 32.0% | 62.725 | 52.0% |
| 激活加量化，至 W2 开始 | 7.626 | 15.1% | 8.992 | 7.5% |
| W2，至归约开始 | 15.478 | 30.7% | 34.913 | 29.0% |
| TopK sum | 2.209 | 4.4% | 2.277 | 1.9% |
| Routed 完成后额外等待 Shared | 0 | 0% | 0 | 0% |
| 合计 | **50.498** | **100%** | **120.579** | **100%** |

每步执行 40 次，合计约 **2.020 ms（B1）/4.823 ms（B8）**。
分叉到 TopK 的小间隔是两路首个 kernel 开始时间之差，不是独立测得的
CUDA event 开销。

## 实际 kernel 时长与启动形状

每类 kernel 每层一次，以下为 480 次观测的均值。Grid 是启动的 CTA 数量，
不等于有效工作 CTA 数、活跃专家数或 GPU 利用率。

| Kernel | B1 μs | B8 μs | B1 grid | B8 grid |
| --- | ---: | ---: | --- | --- |
| TopK | 2.349 | 2.339 | 1 | 2 |
| 输入量化 | 2.302 | 3.546 | 1 | 4 |
| Layout | 0.990 | 2.555 | 1 | 1 |
| W13 | **15.665** | **62.386** | 960 | 7680 |
| SwiGLU + 路由权重 + FP8 量化 | **7.189** | **8.601** | 8 × 1 | 8 × 8 |
| W2 | **15.102** | **34.715** | 256 | 1024 |
| TopK sum | 2.209 | 2.277 | 1 | 4 |

Routed 分支本身的无 kernel 时间平均只有 B1 **3.420 μs**、B8 **3.013 μs**；
两路都没有 kernel 运行的区间合计分别约 **0.814 μs**、**0.557 μs**。
这说明本批专家段首先值得调查正在执行的 GPU 工作，不能把整段归因于 CPU
launch 间隔；忙碌时间也不说明 SM/带宽已经充分利用。

B1 的 7 个 routed kernel 未观察到超过 2 ns 的相邻重叠。B8 共 8 对有重叠，
最大约 0.160 μs，平均每层重叠约 0.0012 μs。分析保留这些时间戳，分别计算
kernel 并集和 elapsed，不将全局多 stream 的 kernel 时长直接相加。

Shared 在两档全部 480 次层调用中提前完成，平均提前 B1 **22.935 μs**、
B8 **38.429 μs**。B8 Shared 分支 span 约 82.151 μs，但其中约 59.733 μs
该分支没有 kernel 在执行；另一条 routed stream 此时通常仍在工作。
这不表示 Shared 需要这么多纯计算时间，也不能单凭 trace 断言全部是资源争用。

## 已核对的实现与可验证点

**W13/W2 实际执行的是 Triton `fused_moe_kernel`。** Runtime 名字为
`TritonOrDeepGemmExperts`，但 `M≤16` 的这条 YOCO FP8 路径明确选择 Triton。
Shared/latent 线性层所用的 native DeepGEMM 与 routed GEMM 是不同实现。

启动日志明确使用 default MoE config，并加载 YOCO FP8 W2 调参表；默认选择
逻辑、W2 表及实际 grid 相符：

| 配置 | W13 M/N/K | W2 M/N/K | Warps / stages |
| --- | --- | --- | --- |
| B1 | 16/64/128 | 16/32/128 | 两者均 4 / 4 |
| B8 | 16/64/128 | 16/64/128 | 两者均 4 / 3 |

B1 使用低开销的逐 route 分配，每个被选专家只有一个有效 token 行，但 M tile
仍为 16。B8 先按专家分组，再按 16 行 padding；64 条 route 中重复选中的专家
可以共用一个 M tile。实际执行到的有效 tile 数取决于 expert IDs。
Grid 固定并不意味着每层做了相同数量的有效计算。

本批 B8 W13 的单次时长从 **40.416 到 87.712 μs**，而 grid 始终为 7680。
因此下一步应捕获 expert IDs/有效专家数，与固定 hidden states 和权重一起回放，
区分路由工作量、缓存和 kernel 调度的影响。现有 trace 没有保存这些 tensor 的值。

**B1 激活量化有一个具体的分块候选。** 当前 kernel 使用 `BLOCK_M=8`，一个 CTA
处理 4 个 128-element 量化组。3840 维共有 30 组，故 grid 为
`ceil(30/4) × ceil(8/8) = 8 × 1`。Routed 调用已设 `packed_scales=False`，
仍沿用每 CTA 4 组的组织方式。可以验证更细的行/组分块，增加可并行 CTA，
同时保留 clamp、SwiGLU、路由权重以及每 128 元素的 FP8 scale 规则。
这是待测假设；不能从 8 CTA 直接推出一定能提速或提速倍数。

W13 是 B8 的首要目标；B1 则需同时看 W13/W2 与激活量化。此前 W2 M1/M2/M4
调参仍在使用，不能把它当成未优化的原始 W2。当前没有 DRAM/L2 或实测 stall
计数器，因此还不能断言这些 GEMM 已跑满 HBM。

## 源码与复现

- [低 M 选择 Triton](../../../vllm/model_executor/layers/fused_moe/experts/triton_deep_gemm_moe.py)。
- [YOCO routed W13、激活、W2 调用顺序](../../../vllm/model_executor/layers/yoco_ops/triton_moe.py)。
- [M/N/K 分块、启动网格及无效 tile 退出](../../../vllm/model_executor/layers/fused_moe/fused_moe.py)。
- [专家分组与 padding](../../../vllm/model_executor/layers/fused_moe/moe_align_block_size.py)。
- [激活/路由权重/量化融合及 CTA 分块](../../../vllm/model_executor/layers/yoco_ops/fp8.py)。
- [已有 W2 调参记录](FP8_W2_TUNING.md)。

```bash
.venv/bin/python tools/yoco_alignment/analyze_parallel_experts.py \
  --runs-root ../work/yoco-m1-integration-20260917 \
  --output docs/yoco/performance/PARALLEL_EXPERTS.json
```

审计每图 1,490 个 kernel，每层恰有 7 个 routed、5 个 shared kernel；检查
stage 顺序和 stream、逐层重建已有 parallel_experts 边界。完整结果保留原 trace
SHA256 和每个 step/layer 的观测。12 步为连续 profiler 观测，不作为 12 次独立实验。
