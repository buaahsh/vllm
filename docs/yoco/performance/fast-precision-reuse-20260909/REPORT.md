# Fast BF16 / FP8 公共实现复用

2026-09-09 PDT / 2026-09-10 UTC。基于 `fhb-dev-9-8`、HEAD `511fbed3f75f0fd18a4f93194ca832db2c723f98` 及已有本地 Fast FP8 改动。

本轮先完成[多精度计划](../FAST_PRECISION_PLAN.md)中的公共路径接通。精度下降与 Fast GEMM 分派允许小误差；布局、scale 搬运和缓存地址仍应精确正确。本轮不会用历史 FP8 内部加速比替代 FP8/BF16 精度比较。

后续[Mooncake 2× 同卡评测](../fast-fp8-reuse-mooncake-f2-20260909/REPORT.md)：1401.36→1407.84输出tok/s（+0.46%，基本持平），局部算子收益未转化为明显整模型提速。

## 实现与覆盖

| 部分 | 本轮状态 |
| --- | --- |
| RMSClip、RoPE、残差、attention | 已由 BF16 hidden 路径共用；保留现有实现，运行相关回归 |
| MoE 路由、分组、有效行、合并 | 已使用公共 modular MoE / Triton 框架；继续复用现有有效行量化和布局实现 |
| 模型级 K/V 投影 | 在线 block-128 FP8 接入 BF16 已有的合并投影和 checkpoint 分片加载，共用一次输入量化与 GEMM |
| QKV / Q 与 lambda 投影 | 改为检查各投影实际精度；仅专家量化或相关投影被 ignore 时，BF16 投影仍可使用现有融合 |
| FP8 shared expert | clamped SwiGLU 与输入量化复用 MoE 的 packed UE8M0 kernel，再进入公共 dense GEMM 接口 |
| GEMM / scale | 共用 FP8 dense 的权重读取、GEMM 分派、bias 与输出处理；保留各后端的精度、tile 和 scale 布局 |
| 图与权重缓存 | BF16 router/shared-expert 派生权重与 FP8 MoE scale 共用原地刷新契约，兼容 layerwise reload 暂存 meta 的过程 |

新增 `yoco_fast_linear_fusion()` 按源投影的量化策略判定能否合并。K/V 必须同时未量化，或同时使用完整对齐的 online block-128 FP8；其中一个被 ignore 时保持两个投影。per-tensor 与 MXFP8 的量化域不能直接套用此次 FP8 合并，继续使用各自原路径。

FP8 Q/QKV 与 BF16 lambda 仍分别计算；现有 QKV 本身的合并保持生效。将两种精度放进一个新混合 GEMM 属于后续 kernel 开发，本轮不改动 lambda 的精度来强行合并。

shared-expert 的融合仅在 Fast、TP1、SM100、BF16 hidden/output、DeepGEMM UE8M0、正 clamp limit 和完整 128 列分组时启用。它保留 FP32 clamp/SiLU/multiply、转回 BF16、再量化 FP8 的数值边界。Align 和未满足条件的后端继续走原路径。

`Fp8BlockScaledMMLinearKernel.apply_quantized_weights()` 为普通输入量化和融合量化提供同一个 GEMM 入口。它接受所选后端兼容的量化输入与 scale，不统一不同精度的量化规则或调优参数。

加载测试还发现，编译量化函数直接接收 vLLM Parameter 子类时可能递归进入 `__torch_function__`。在线 block-FP8 加载改为传入底层 `.data`；权重值、block 边界与量化算法保持原样。

## 验证方法

**B200 上 383 项回归通过，零失败、零跳过。**本地 pre-commit（含类型检查）通过。

新增检查涵盖：分层 ignore、仅专家量化、Align/TP 回退、K/V 极不相同的数值幅度、正反加载顺序、独立与合并后的 FP8 权重及 scale 字节、shared-expert 的完整调用、fake tensor 布局、CUDA graph replay、权重变化，以及 layerwise reload 后缓存地址和值。

融合算子和相同量化语义参考链比较相对 L2，采用 1e-3 的初始回归门槛，同时检查有限性。布局和 scale 的等价转换使用精确比较；不把跨精度误差要求成 bitwise。

完整服务保留 `--fast --quantization fp8_per_block`、BF16 hidden/KV、FA4、KV sharing 和 FULL_AND_PIECEWISE 图设置。服务端拒绝 KV-sharing fast-prefill 下的 prompt log-prob 请求，因此使用相同固定前缀，比较基线所选目标 token 在两端的下一步 log-prob。覆盖中英文、代码和数学文本的 15 个固定前缀，另检查 128、8192、79150 输入及 batch8 的生成有限性。这是功能与数值诊断，不是任务质量或 PPL 评测。

首次临时服务完成模型编译后进入 8,505 个形状的全量 DeepGEMM 预热；为功能验证重启该临时服务并设置 `VLLM_DEEP_GEMM_WARMUP=skip`。实际图捕获和测试请求仍执行，涉及的新形状按需编译；不将此配置的启动或请求时间用于吞吐结论。首次停止的 API 进程在宽限期后仍未退出，核对进程身份且确认 engine 已退出后清理了该临时进程组，原 GPU2 服务保持运行。

服务与 kernel 使用原 4 卡 Job 内已分配的 B200 GPU3；已有 GPU2 服务作为数值参考并保持运行。不同物理 GPU 上的服务诊断不用于吞吐比较。局部算子计时如有记录，只比较同一 GPU3 上相同权重与输入的前后调用链。

扩大回归还修正了旧测试中的配置缺项、单行切片连续性判断和 MLP 的 loop 参数。4 个连续性/精确相等失败在修改前源码上复现。旧 Fast weighted RMSClip 相对编译训练表达式的相对 L2 为 0.002759–0.002813，开关梯度均如此；其计算本轮未改，测试改为相对 L2<0.003、逐元素 `rtol=1/64, atol=1e-6`。该门槛仅用于这组既有 Fast 测试；相关 Align 和批次一致性检查保持原精确断言。

## 局部性能

同一 B200 GPU3、相同 FP8 权重和输入，比较 shared-expert 的“clamped SwiGLU→量化→down GEMM”。不包含 gate/up 投影、其他层或服务调度。每点三次交替 CUDA graph 计时，取中位数。

| M | 分离调用 µs | 融合调用 µs | 加速比 |
| --- | ---: | ---: | ---: |
| 1 | 9.32 | 8.77 | 1.063× |
| 2 | 15.53 | 8.97 | 1.731× |
| 4 | 10.76 | 9.06 | 1.187× |
| 8 | 10.77 | 9.11 | 1.183× |
| 16 | 10.78 | 9.29 | 1.160× |
| 32 | 11.36 | 9.31 | 1.220× |
| 128 | 11.96 | 9.82 | 1.218× |
| 512 | 16.56 | 12.05 | 1.374× |
| 2048 | 31.66 | 18.13 | 1.746× |

这 9 个基准输入的前后输出相对 L2 均为 0；更广的幅度/形状测试采用前述误差门槛，不声明普遍 bitwise。这里两端都是 FP8，不能解释为 FP8 相对 BF16 或整模型吞吐的增幅。本轮没有重测 Mooncake。

基准反复使用固定权重，不模拟全模型运行时其他层对缓存的影响。

## 完整模型结果

实际服务快照确认：20 个 shared-expert 的 FP8 激活量化融合均启用，模型级 K/V 为一个 FP8 合并投影；模型加载显存为 32.07 GiB。

| 与修改前 FP8 比较的固定前缀检查 | 结果 |
| --- | ---: |
| 前缀数 / 成功比较数 | 15 / 15 |
| 基线目标 token 未出现在候选 top-20 | 0 |
| top-1 改变 | 0 |
| 同一目标 token 的最大 log-prob 绝对差 | 7.15154×10⁻⁷ |
| 平均 log-prob 绝对差 | 4.76769×10⁻⁸ |

另外，ISL128/C1、ISL8192/C1、ISL79150/C1、ISL128/C8 的 11 个请求均完成指定的 4 个输出 token，log-prob 全部有限。该检查不代表完整任务质量评估，也不提供跨精度或任意输入的 bitwise 保证。

测试后服务队列排空。临时 GPU3 验证服务停止；原 GPU2 服务保持健康，Job 与 GPU allocation 保留。

## 证据

本地执行目录：`/home/lidong1/vllm_test/yoco_results/fast-precision-reuse-20260910T034205Z`。

Pod 内目录：`/data/fast-precision-reuse-20260910T034205Z`。

测试、数值、局部计时、源码清单和备份索引随本报告保存。历史 [FP8 性能表](../FP8.md) 的端到端数值保持原有测量日期与比较条件。

[测试结果](tests-v7.xml) · [服务数值比较](numerical-comparison.json) · [局部计时](shared-bench.json) · [源码清单](SOURCE_MANIFEST.json) · [备份索引](BACKUP.json)
