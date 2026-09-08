# Mooncake 1.2× 持续表

同一 toolagent 源时间300–900秒，时间戳除以1.2：约500秒到达、3643请求。与[1×持续表](THROUGHPUT.md)分开维护。当前只测Fast；Qwen3和Align的1.2×结果尚未测量。

单卡GPU5；1P1D为P GPU4、D GPU5，B200、TP1、BF16、上下文81920。每case独立cache_salt，AIPerf CLI speedup=1.0、并发上限512。单次、共享节点、未声明SLO，均为diagnostic。

| 模式/拓扑 | 测量开始 UTC | 完成/计划 | 输出tok/s | 输入tok/s | TTFT P95 ms | ITL P95 ms | E2E P95 ms | client/server/drain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [fast/pd](fast-mooncake-f1p2-20260908/REPORT.md) | 2026-09-08T09:43:22+00:00 | 3643/3643 | 1244.77 | 59052.91 | 5274.00 | 24.33 | 16094.88 | PASS/PASS/PASS |
| [fast/standalone](fast-mooncake-f1p2-20260908/REPORT.md) | 2026-09-08T09:23:58+00:00 | 3642/3643 | 1190.07 | 56456.67 | 10222.08 | 338.66 | 98452.09 | FAIL/PASS/PASS |

吞吐按完整完成区间计算，包含到达结束后的排空。client/server/drain通过不代表延迟SLO或容量验收通过。错误、迟发、队列和P99详见每行报告。

Trace SHA256：`5317e2301656c7d5441bbd63e58b7e8b9d3d445e0118729cd7fde0b8717b960c`。

数据：[current.json](throughput-f1p2/current.json) · [CSV](throughput-f1p2/current.csv) · [不可变历史](throughput-f1p2/history/)。

只更新实际测量的拓扑：

```bash
python update_throughput_f1p2.py --source fast-mooncake-f1p2-20260908/load-response.json --topology standalone
python update_throughput_f1p2.py --source fast-mooncake-f1p2-20260908/load-response.json --topology pd
```

脚本校验1.2× trace、请求数、到达跨度和物理GPU，拒绝混入1×或修改已有历史记录。
