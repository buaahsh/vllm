# Qwen3 与 Align：同卡 B200、同 trace 吞吐对照

日期：2026-09-07。本轮新测 **Qwen3-30B-A3B-Instruct-2507 普通 BF16 模式**；Align 使用刚完成的 GEMM 候选结果。

Qwen3 单卡输出吞吐 **929.63 tok/s**，1P1D **1040.93 tok/s**。相对对应拓扑 Align，分别为 **1.517×** 和 **1.259×**。

## 吞吐与完成情况

| 配置 | 推理 GPU | 完成 / 计划 | 失败 | 输入 tok/s | 输出 tok/s | 输出 tok/s / 推理 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| Align GEMM / 单卡 | 1 | 3638 / 3643 | 5 | 29389.12 | 612.83 | 612.83 |
| Qwen3 / 单卡 | 1 | 3643 / 3643 | 0 | 44096.98 | 929.63 | 929.63 |
| Align GEMM / 1P1D | 2 | 3643 / 3643 | 0 | 39222.23 | 826.76 | 413.38 |
| Qwen3 / 1P1D | 2 | 3643 / 3643 | 0 | 49376.48 | 1040.93 | 520.46 |

以 Qwen3 为分母，Align 单卡输出吞吐低 **34.08%**，1P1D 低 **20.57%**。单卡 Align 的 5 条 600 秒 streaming timeout 仍计入失败；其吞吐和延迟只覆盖成功请求，比较存在幸存者偏差。

固定1×到达时序的计划输出负载约为1072.29 tok/s；这里测量的是该负载及收尾下的实现吞吐，不是系统峰值吞吐。总分配是 4 张 B200；表中每 GPU 效率按实际参与该拓扑的 1 或 2 张卡计算。若按保留的四卡资源预算计算，需要用总吞吐除以 4，JSON 中也保存了该值。

## 延迟与到达时序

每格依次为 P50 / P95 / P99。TTFT/E2E 单位为秒，ITL 单位为毫秒。ITL 为每条请求的平均 token 间隔分布，沿用 AIPerf 原始指标定义；并非单条请求固定的 decode 速率。

| 配置 | TTFT (s) | ITL (ms) | E2E (s) |
| --- | --- | --- | --- |
| Align GEMM / 单卡 | 68.05 / 89.30 / 94.55 | 386.09 / 725.70 / 1010.19 | 84.89 / 282.75 / 367.18 |
| Qwen3 / 单卡 | 41.75 / 77.64 / 82.96 | 80.26 / 183.57 / 475.21 | 50.51 / 104.70 / 129.92 |
| Align GEMM / 1P1D | 63.17 / 108.83 / 112.31 | 54.08 / 59.82 / 61.75 | 73.43 / 124.51 / 140.83 |
| Qwen3 / 1P1D | 1.52 / 3.57 / 4.91 | 36.83 / 46.05 / 46.90 | 3.08 / 25.22 / 37.47 |

| 配置 | 调度 lag P99 (ms) | 最大并发 | 客户端 gate | 服务 gate | 最后计划到达后排空 (s) |
| --- | --- | --- | --- | --- | --- |
| Align GEMM / 单卡 | 295920.39 | 512.0 | 失败 | 通过 | 434.49 |
| Qwen3 / 单卡 | 9533.22 | 512.0 | 失败 | 通过 | 92.06 |
| Align GEMM / 1P1D | 60833.25 | 512.0 | 失败 | 通过 | 178.17 |
| Qwen3 / 1P1D | 5.89 | 102.0 | 通过 | 通过 | 18.07 |

本轮是共享节点、单次测试、无预设延迟 SLO 的诊断。触及并发 512 或迟发时，实际发送时序已经被客户端限流改变；零错误也不能据此认定已通过原始到达速率的容量验收。

## 缓存、抢占与传输

| 配置 / 角色 | 前缀命中率 | 抢占 | 运行峰值 | 等待峰值 |
| --- | --- | --- | --- | --- |
| Align GEMM / 单卡 / standalone | 37.13% | 0.00 | 256.00 | 259.00 |
| Qwen3 / 单卡 / standalone | 9.71% | 62.00 | 129.00 | 447.00 |
| Align GEMM / 1P1D / p | 0.00% | 0.00 | 10.00 | 474.00 |
| Align GEMM / 1P1D / d | 0.23% | 0.00 | 72.00 | 488.00 |
| Qwen3 / 1P1D / p | 38.21% | 0.00 | 19.00 | 21.00 |
| Qwen3 / 1P1D / d | 37.92% | 0.00 | 77.00 | 41.00 |

传输次数、字节数、耗时分位数、失败计数、最终队列及计时外完成通知收尾保存在 `comparison.json`。如需要本地 1-token P 请求消费完成事件，该请求单独记录在 `post-timing-producer-cleanup.json`，不计入 AIPerf 性能时间。

## 相同条件与差异边界

| 项 | 本轮条件 |
| --- | --- |
| Job / Pod | yoco-align-fast-vllm-vllm-train / yoco-align-fast-vllm-vllm-train-master-0 |
| 节点 | slc01-cl02-hgx-0228；同节点其他 GPU 存在外部负载，强度随时间变化 |
| 物理 GPU | 单卡 GPU5；1P1D P GPU4 + D GPU5；GPU2/3 在计时期间空闲 |
| 公共服务配置 | BF16，TP1/DP1，maxlen81920，maxseq256，memory .85，prefix cache + chunked prefill |
| 调度 token budget | 单卡/P32768，D8192 |
| P/D | Mooncake RDMA，16 sender workers，同一代理协议和观测方式 |
| Qwen3 | 普通模式；FA4；FULL_AND_PIECEWISE；Triton MoE；无 YOCO align/profile/KV-sharing 参数 |
| Align | YOCO L3 GEMM 候选；实际 FA2/FULL_DECODE_ONLY；截断 P 当前跳过前缀缓存读取 |
| 客户端 | AIPerf0.12.0，streaming completions，服务端 token counts，默认 BOS，seed42 |
| 到达/上限 | fixed schedule1×，concurrency512，workers32，record-processors1，timeout600s |
| 缓存初态 | 每个 functional/smoke/long case 使用独立非空 cache salt，保留同一 case 内 hash_ids 前缀复用 |

两模型架构、tokenizer、attention/graph 路径和 P 缓存实现存在差异；这是模型与系统整体对照，不能把差值全部归因于 Align GEMM 或 bitwise 要求。服务启动、首次编译、smoke、计时外清理均不计入长测。共享节点的外部负载并非固定，因此同物理 GPU 仍不足以消除跨轮干扰。

## 开源 trace 与 token 核验

使用 Mooncake FAST’25 `toolagent_trace.jsonl` 的 300–900 秒窗口，3,643 条请求、599,999ms 到达跨度，输入 30,518,473 tokens、输出 643,375 tokens；是合成 token/prefix 到达回放，不含真实 agent 提示文本或质量评价。

源 trace 共 23,608 行，maxlen81920 过滤 116 行，保留 23,492 行（99.51%）。冻结窗口 SHA256：`680e526d49545258c0ca5b635d0004a44cce2ce990d533ac32024cefee1f0170`。完整来源、筛选和 smoke SHA 见 `traces/*.manifest.json`。

| 配置 | 成功请求 actual_input − requested_input | 成功请求输出长度不符 |
| --- | --- | --- |
| Align GEMM / 单卡 | {"1": 3628, "2": 10} | 0 |
| Qwen3 / 单卡 | {"0": 3633, "1": 10} | 0 |
| Align GEMM / 1P1D | {"1": 3633, "2": 10} | 0 |
| Qwen3 / 1P1D | {"0": 3633, "1": 10} | 0 |

Qwen 和 YOCO 的 tokenizer.json/tokenizer_config.json SHA 不同，保存在 `INPUT_ENVIRONMENT.json`；输入长度核验采用与 Align 相同的 ±2-token BOS 容差，并明确保存每条实际差值，输出长度必须精确等于 trace。

## 功能与运行审计

两种 Qwen 拓扑均先运行 50 条流式 smoke，完整率、OSL、metrics、队列和传输检查见对应 `cases/*smoke50/COMPLETE.json`。另对 511/512/513/4097 个输入 token 进行单卡与 P/D 的确定性生成对照，完整响应和逐项差值见 `functional-comparison.json`。4/4组生成的32个token均相同，但所选token的log-prob最大绝对差为0.3886835575，并未达到bitwise。长测排空之后，另外保持P/D引擎运行，比较本地P、本地D与代理P/D；该检查位于AIPerf计时之外，结果见 `post-timing-functional/SUMMARY.json`。计时外6组检查（另含513/4097-token自然语言尾部）中，本地P和本地D的所选token log-prob差值均为0；4组已有单卡参照也与本地P/D相同。经过代理P/D后的生成token仍全部相同，但log-prob差异重现，循环输入最大0.3886835575，自然语言输入最大0.1202327609。差异已缩小到P/D执行路径；是否来自增量计算、算术路径或交接元数据仍未定位，不能归因为普通舍入或宣布数值对齐通过。详见 `NUMERICS_OPEN_ISSUE.md`。这些检查只覆盖所选输出token，不能代表全logits或所有输入的数值保证。

复用已冻结 vLLM runtime；测试前后核验全部 3,661 个 vLLM 源码/资产文件。运行中模块路径、包版本和关键源文件 SHA 见 `serving-environment-*.json`，启动实参见 `servers/*/ownership.json`，实际生效配置见服务日志。

启动时观察到 /proc cmdline 在 exec 切换窗口短暂为空。保留原始记录后，使用原 PID、start ticks、PGID 和完整预期 command 逐项校验补全快照；launcher 已修复为等待精确 command 出现。未修改推理代码或性能参数，未用于隐藏服务重启。

## 结果文件

- `comparison.json`：Qwen 与 Align 的完整四组指标和限制。
- `trace-comparison.json/csv`：本轮两组 Qwen 指标。
- `cases/`：客户端记录、HTTP trace、服务指标、GPU 遥测、token 核验与排空。
- `FINAL_AUDIT.json`、`POD_FINAL.json`：源码、GPU、Xid/ECC 可见性、进程清理和 Pod 身份核验。
- `BACKUP.json`：持久 PVC 备份与 SHA 读回校验。

测试后停止本轮服务，保留四卡 Job。没有 commit/push。
