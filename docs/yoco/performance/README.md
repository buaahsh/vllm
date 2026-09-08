# YOCO 性能结果入口

日常比较和更新使用 [三模式持续吞吐表](THROUGHPUT.md)。

每次只重测本次修改涉及的模式，并只更新该模式已测的拓扑；其他模式沿用历史结果并保留测量日期。
使用 `update_throughput.py --mode ... --source ... --case ...` 导入一个已审计的case，不会触发其他测试。
固定工作负载和更新规则见表内说明；原始run目录保留失败、原始日志、环境和审计证据。
历史数据在 [throughput/history](throughput/history/)，机器可读当前值在 [throughput/current.json](throughput/current.json)。

## 仓库发布副本

本目录同步到 vLLM 与 llm-train 的 `fhb-dev-9-8` 分支。表中六行保持原测量时间、数值和 source / manifest SHA256；2026-09-07 最后一轮只重测 Fast。此表是推理服务性能，不能用来推断 llm-train 的训练整步吞吐。

仓库保留报告、当前表、不可覆盖的历史记录、图和关键审计 JSON。完整逐请求日志、trace 和 runtime 继续保存在测试归档中：

- Fast：`/mnt/pvc/lidong1/fast-compare-b200-20260907/results.tar`，SHA256 `b7881086ae0b12e0fad329f751705004604daa9530c9abf361ed36f4c54af4d5`。
- Qwen3：归档位置和 SHA256 见 [BACKUP.json](qwen3-compare-b200-20260907/BACKUP.json)。
- Align：归档位置和 SHA256 见 [RESULTS_BACKUP_COMPLETE.json](align-4gpu-1p1d-20260906/RESULTS_BACKUP_COMPLETE.json)。

在本目录中按 [持续表](THROUGHPUT.md) 的命令导入新结果；先把本轮 source、case manifest 与审计证据放到对应的相对路径。`update_throughput.py` 只更新表，不启动压测。

图是本轮测量快照。`plot_throughput.py` 重绘完成曲线还需要归档中的 `cases/*/artifacts/profile_export.jsonl` 与 trace 文件；先按原目录结构还原，再用具有 numpy / matplotlib 的虚拟环境运行。原报告和 JSON 中的绝对路径是测试机器上的来源记录。
