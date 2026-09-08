# 低并发 W2 持续表

工作负载：固定token形状，65,536输入、16,384输出；BF16，TP1/DP1，同物理B200 GPU2。完整请求输出吞吐包含prefill与调度。此组与Mooncake固定到达率表分开维护。

| 模式 | 并发 | 输出 tok/s | TTFT s | TPOT ms | 测量开始 UTC | 证据 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 旧 Fast 控制组 | 1 | 76.60 | 0.934 | 12.998 | 2026-09-08 05:55:13 | [fast-baseline-b](../fast-low-concurrency-20260908/cases/fast-baseline-b/results.json) |
| Fast（split-KV） | 1 | 154.25 | 0.919 | 6.427 | 2026-09-08 05:46:03 | [fast-split-b](../fast-low-concurrency-20260908/cases/fast-split-b/results.json) |
| Qwen3-30B-A3B | 1 | 163.10 | 1.850 | 6.019 | 2026-09-08 06:07:21 | [qwen-baseline-b](../fast-low-concurrency-20260908/cases/qwen-baseline-b/results.json) |
| 旧 Fast 控制组 | 2 | 142.99 | 1.658 | 13.886 | 2026-09-08 05:58:50 | [fast-baseline-b](../fast-low-concurrency-20260908/cases/fast-baseline-b/results.json) |
| Fast（split-KV） | 2 | 253.09 | 1.656 | 7.801 | 2026-09-08 05:47:51 | [fast-split-b](../fast-low-concurrency-20260908/cases/fast-split-b/results.json) |
| Qwen3-30B-A3B | 2 | 264.08 | 2.828 | 7.401 | 2026-09-08 06:09:06 | [qwen-baseline-b](../fast-low-concurrency-20260908/cases/qwen-baseline-b/results.json) |

当前数据：2026-09-08 UTC，每模式/并发一次完整测量，共享节点diagnostic。详细配置、TTFT、TPOT、全部样本、数值验证和源码见 [报告](../fast-low-concurrency-20260908/REPORT.md)及 [机器可读结果](../fast-low-concurrency-20260908/COMPARISON.json)。旧Fast是本轮同卡控制组，修复Fast恢复FA4 split-KV自动调度。

更新规则：只替换实际重测的模式/并发行，记录对应case的实际时间、源码和工作负载；其他行保留原测量时间。更改输入/输出长度、硬件或调度条件时新增可比组；不以瞬时日志吞吐替代完整完成结果。
