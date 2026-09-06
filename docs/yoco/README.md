# YOCO Align 与 Fast 开发记录

这些文件保存 2026-09-05 的实现说明与测量结果。发布到 `fhb-dev` 时补充了类型标注、
格式和基准脚本 API 整理；推理库与实测版本的可执行 AST 一致（忽略仅供类型检查的标注）。

- [综合报告 PDF](YOCO-Align-Fast-Report-20260905.pdf)：Align 前向对齐、概率归约、训练反向优化与 Fast 性能。
- [Fast 专项报告](fast-optimization-20260905/REPORT.md)：适用条件、开关、固定形状测量、AIPerf 诊断与失败记录。
- [README PDF 快照](README-snapshot-20260905.pdf)：发布前生成的说明快照。
- [短测数据](fast-optimization-20260905/short-comparison.json)与[长 trace 数据](fast-optimization-20260905/trace-comparison.json)。
- [训练侧 fhb-dev](https://github.com/msrallm/llm-train/tree/fhb-dev)：训练接入、grouped MoE backward 及对应验证资料；需要该仓库访问权限。

报告中的源码状态、绝对路径和环境描述对应测量当时的快照。原始 GPU 日志、大张量、
完整 trace 导出和实验 runtime 保留在实验工作区，未全部复制到 Git 仓库；此处发布报告、
图表、汇总数据与 PDF。PDF 中的本地证据链接仍依赖原工作区。

Fast 的主要收益是 Prefill，Decode 与这次固定到达 trace 的总吞吐基本持平。
Fast 不提供 bitwise 保证；Align 的前向一致性结论限于报告明确列出的配置和验证矩阵。
