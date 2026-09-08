# YOCO / Qwen3 持续吞吐对照表

只重测本次修改涉及的模式；只更新已实际测量的对应行。其他行保留原值、测量时间和证据。

固定工作负载：Mooncake FAST’25 toolagent，源时间 300–900 秒，1×，3643 请求；AIPerf 0.12.0、concurrency512、workers32、timeout600。BF16、TP1/DP1、maxlen81920、maxseq256；单卡/P预算32768、D预算8192。

Trace SHA256：`680e526d49545258c0ca5b635d0004a44cce2ce990d533ac32024cefee1f0170`。每轮独立 cache_salt。本表使用同一节点上的 GPU5（单卡）、GPU4/5（1P1D）；UUID保存在JSON。

**全部是共享节点、单次重复、无预设延迟SLO的诊断结果。** 这是固定到达负载下的实测吞吐，不是峰值吞吐；接近负载上限时不能据此排序最大能力。各行不是同一时段重测，节点上其他工作的干扰可能变化。

Qwen3与YOCO是不同模型。各模式采用自身执行路径：Align强制的后端/图模式、Fast按角色的MoE策略、Qwen普通执行路径不保证相同；启动命令和配置证据保存在每行manifest。

## 单卡（GPU5）

| 模式 | 输出 tok/s | 每推理GPU tok/s | 完成/计划；错误 | TTFT P95 s | ITL P95 ms | 测量时间 UTC | 状态/证据 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| Qwen3-30B-A3B-Instruct-2507 | 929.63 | 929.63 | 3643/3643；0 | 77.64 | 183.57 | 2026-09-07 10:42:06 | [客户端负载门槛失败](qwen3-compare-b200-20260907/cases/standalone-qwen3-r1-long600s/COMPLETE.json) |
| YOCO Align GEMM | 612.83 | 612.83 | 3638/3643；5 | 89.30 | 725.70 | 2026-09-07 09:13:55 | [请求/长度未全通过；客户端负载门槛失败](align-4gpu-1p1d-20260906/pd-tail-fix/cases/standalone-candidate-fixed-long600s/COMPLETE.json) |
| YOCO Fast | 1037.27 | 1037.27 | 3643/3643；0 | 2.66 | 311.56 | 2026-09-07 12:56:36 | [客户端/服务门槛通过；诊断](fast-compare-b200-20260907/cases/standalone-fast-r1-long600s/COMPLETE.json) |

## 1P1D（P GPU4 / D GPU5）

| 模式 | 输出 tok/s | 每推理GPU tok/s | 完成/计划；错误 | TTFT P95 s | ITL P95 ms | 测量时间 UTC | 状态/证据 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| Qwen3-30B-A3B-Instruct-2507 | 1040.93 | 520.46 | 3643/3643；0 | 3.57 | 46.05 | 2026-09-07 11:00:42 | [客户端/服务门槛通过；P/D log-prob差异待定位](qwen3-compare-b200-20260907/cases/pd-qwen3-r1-long600s/COMPLETE.json) |
| YOCO Align GEMM | 826.76 | 413.38 | 3643/3643；0 | 108.83 | 59.82 | 2026-09-07 08:49:40 | [客户端负载门槛失败](align-4gpu-1p1d-20260906/pd-tail-fix/cases/pd-candidate-fixed-long600s/COMPLETE.json) |
| YOCO Fast | 1045.68 | 522.84 | 3643/3643；0 | 3.90 | 22.58 | 2026-09-07 13:16:45 | [客户端/服务门槛通过；单卡/P-D探针log-prob差0.1322；待定位](fast-compare-b200-20260907/cases/pd-fast-r1-long600s/COMPLETE.json) |

## 解释与更新规则

- 未完成的轮次照实显示；其成功请求吞吐/延迟有幸存者偏差，不能当作完成相同工作量的等价比较。
- Qwen3 1P1D性能门槛通过，但独立探针发现log-prob差异，数值问题仍未定位；Fast不声明bitwise。
- Align引用的是带16项精确M配置的Align GEMM候选；其bitwise结论只覆盖原报告已测条件。
- 主表只显示每个模式/拓扑的最近一次实际测量。旧记录永久保存在[history](throughput/history/)，完整当前字段见[current.json](throughput/current.json)及[current.csv](throughput/current.csv)。
- 单个更新命令只导入一个case、更新一个模式/拓扑；不启动任何测试。失败记录不会被隐藏或改成通过。
- 更换trace、速率、上下文、客户端或调度预算时，应新建可比组，不能覆盖本表。

在本目录中运行（一次导入一个已审计的 case）：

```bash
uv run --python 3.12 update_throughput.py \
  --mode fast --source fast-compare-b200-20260907/trace-comparison.json \
  --case pd-fast-r1-long600s
```

原始报告：[Align](align-4gpu-1p1d-20260906/REPORT.md)、[Qwen3](qwen3-compare-b200-20260907/REPORT.md)、[Fast](fast-compare-b200-20260907/REPORT.md)。

对比图：[吞吐](throughput/figures/throughput.png)、[延迟](throughput/figures/latency.png)、[Fast相对变化](throughput/figures/fast-relative.png)、[到达与完成](throughput/figures/completion.png)。
