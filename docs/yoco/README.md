# YOCO Align 与 Fast 开发记录

- [2026-09-08 Fast 低并发 W2 优化](fast-low-concurrency-20260908/REPORT.md)：恢复FA4 split-KV自动调度；完整W2并发1/2吞吐提升2.014×/1.770×，与Qwen差距缩小至5.42%/4.16%。固定形状同卡诊断，Fast非bitwise。见[独立低并发表](performance/LOW_CONCURRENCY.md)。

- [2026-09-08 Align backward 路由越界修复](align-backward-fix-20260908/REPORT.md)：复现并修复Triton3.7.1下的CUDA非法访问；两版本回归、显存检查及NNScaler连续训练通过。旧报告保留历史测量值。

- [2026-09-07 四卡 Align 1P1D 与 llm-train 实测](align-4gpu-1p1d-20260907/REPORT.md) · [PDF](align-4gpu-1p1d-20260907/REPORT.pdf)：四卡B200 Job已完成实测。真实1P1D输出字节105项、与llm-train单条/packed的前向和CE字节72项，原始Align与GEMM候选均通过；同卡1P1D输出吞吐631.87→826.76 tok/s（+30.84%）。ITL P95 55.25→59.82 ms。同GPU单实例对照有5/3643条600秒超时。当前1×trace为共享节点过载诊断，非容量验收；当前P跳过前缀缓存读取，收尾控制请求单独记在计时之外。

- [Align MoE GEMM 本地实验](align-gemm-local-20260906.md)：两端共享配置与 A6000 算子初筛的历史阶段。
- [Align GEMM B200 数值验收](align-gemm-b200-20260906/VALIDATION.md)：725项整模型字节检查，含两端前向及缓存/混合批次；实验默认关闭。
- [Align GEMM B200 完整报告](align-gemm-b200-20260906/REPORT.md)：同卡开源 trace 输出吞吐+5.74%，两端均有5个超时；训练整步基本持平。
- [后续 Align 小批次优化与 1P1D 进度](align-1p1d-b200-20260906/REPORT.md)：补齐2/4/16行配置，开源trace形状诊断；记录申请四卡前的阶段；后续实测见四卡报告。

这些文件保存 2026-09-05 至 2026-09-07 的实现说明与测量结果。发布到 `fhb-dev` 时补充了类型标注、
格式和基准脚本 API 整理；推理库与实测版本的可执行 AST 一致（忽略仅供类型检查的标注）。

- [2026-09-06 Fast decode 优化](fast-decode-optimization-20260906/REPORT.md)：同卡配对测量、固定前缀数值检查与 AIPerf。
- [2026-09-06 Fast decode 报告 PDF](fast-decode-optimization-20260906/REPORT.pdf)。
- [综合报告 PDF](YOCO-Align-Fast-Report-20260905.pdf)：Align 前向对齐、概率归约、训练反向优化与 Fast 性能。
- [Fast 专项报告](fast-optimization-20260905/REPORT.md)：适用条件、开关、固定形状测量、AIPerf 诊断与失败记录。
- [README PDF 快照](README-snapshot-20260905.pdf)：发布前生成的说明快照。
- [短测数据](fast-optimization-20260905/short-comparison.json)与[长 trace 数据](fast-optimization-20260905/trace-comparison.json)。
- [训练侧 fhb-dev](https://github.com/msrallm/llm-train/tree/fhb-dev)：训练接入、grouped MoE backward 及对应验证资料；需要该仓库访问权限。

报告中的源码状态、绝对路径和环境描述对应测量当时的快照。原始 GPU 日志、大张量、
完整 trace 导出和实验 runtime 保留在实验工作区，未全部复制到 Git 仓库；此处发布报告、
图表、汇总数据与 PDF。PDF 中的本地证据链接仍依赖原工作区。

2026-09-05 的主要收益是 Prefill；2026-09-06 的后续工作针对 decode，结果见新报告。
Fast 不提供 bitwise 保证；Align 的前向一致性结论限于报告明确列出的配置和验证矩阵。

## 持续吞吐表

[Fast / Align GEMM / Qwen3 对照表](performance/THROUGHPUT.md)按模式、拓扑逐行更新；保留实际测量日期、失败和证据，不要求每次同时重测三种模式。[报告与维护方法](performance/README.md)。
