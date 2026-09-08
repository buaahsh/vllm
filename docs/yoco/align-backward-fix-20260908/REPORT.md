# Align backward 路由越界修复

日期：2026-09-08（UTC）。原独立审计报告中的 grouped MoE backward CUDA error 已复现并修复。

## 结果

修复后的 Triton 3.7.1 和 3.8.0 各通过 **84 项相关回归 + 5 项概率/CE 检查**。Compute Sanitizer 在普通显存分配器下通过49项，报告0个错误。两个版本的 NNScaler 都完成连续12步 backward/SGD，36个参数梯度始终有限。

生产改动仅为 `llm/kernel/align_moe_backward.py::_count_routes` 的 histogram 有效位置 mask。`llm/arch/align.py` 字节未变，除 `_count_routes` 外的 backward AST 未变；保留 grouped DeepGEMM 路径，没有增加 kernel launch 或主机同步。vLLM 本轮只更新文档。源码核验见 [SOURCE_VERIFICATION.json](SOURCE_VERIFICATION.json)。

## 原因与复现

[2026-09-07 独立审计](prior-audit/REPORT.md)确认了 Align 前向，但 `test_experts_exact_across_batch_and_backward` 在 `_route_map` 报 `CUDA illegal memory access`。该审计的 llm-train `cc700436` 与本轮基线 `a1f1ef07` 使用完全相同的 backward kernel，SHA256 为 `010fe21b48208f1a02d2b496b779805869995882a4dbcf124dd48223b477c54c`。

旧代码每个计数 block 处理1024条路由，把超出实际长度的位置填为专家编号128，却没有给 `tl.histogram` 传 mask。B200/Triton 3.7.1 在本实验中把这些无效值计入0号专家：17个token × Top-8 本应是136条路由，却得到1024条；多出的888条会使后续offset越过 `order` 的实际范围。

| 对照 | 136条路由的计数总和 | 9种长度探针 | 原完整 backward 用例 |
| --- | ---: | --- | --- |
| 旧实现，Triton 3.7.1 | 1024 | 8项错误、1项正确 | `_route_map` CUDA illegal memory access |
| 旧实现，Triton 3.8.0 | 136 | 9项正确 | 本轮未单独重跑该负对照 |
| 修复后，Triton 3.7.1 | 136 | 9项正确 | 通过 |
| 修复后，Triton 3.8.0 | 136 | 9项正确 | 通过 |

这解释了为什么旧测试环境能通过、独立审计却崩溃。原先512/2048-token性能用例的Top-8路由数也恰好是1024的整数倍，不覆盖尾部padding。修复将无效位置填0，并显式使用 `mask=valid` 排除这些位置，不再依赖越界sentinel在不同版本下的处理方式。补丁见 [PATCH.diff](PATCH.diff)。

原始 CUDA 错误日志和新增测试在旧源码下的断言失败均保留：

- [原完整用例的失败](cases/baseline-triton371-original-failure/run.log)：启用 `CUDA_LAUNCH_BLOCKING=1`，定位到 `_route_map`。
- [新增回归对旧实现的负对照](cases/baseline-triton371-regression-negative/run.log)：在计数与CPU oracle比较时失败，尚未进入可能越界的重排。
- [旧3.7.1计数](counts-baseline-triton371.json)、[旧3.8.0计数](counts-baseline-triton380.json)、[修复后3.7.1](counts-fixed-triton371.json)、[修复后3.8.0](counts-fixed-triton380.json)。

## 回归与连续训练

新增48项测试覆盖0/1/8/136/1023/1024/1025/4104条路由、int32/int64索引、混合分布、全部落在首/末专家、空专家和跨block尾部。计数与CPU `bincount` 比较；重排、逆映射与CPU稳定排序比较，并检查两侧保护区及填充位置。

| 验证 | Triton 3.7.1 | Triton 3.8.0 |
| --- | --- | --- |
| 路由、前向、梯度oracle、SGD、共享GEMM配置 | 84 passed | 84 passed |
| 独立进程概率 / CE | 5 passed | 5 passed |
| NNScaler连续训练 | 12步，36个梯度张量有限 | 12步，36个梯度张量有限 |
| eager与compiled初始loss | 6.2097463608，bitwise相同 | 6.2097463608，bitwise相同 |
| 第12步更新后的loss | 0.1069592983 | 0.1069597602 |

测试沿用既有梯度验收：grouped与参考backward的相对RMS误差小于1%，独立dense matmul oracle阈值为2%；没有放宽阈值或更改预期张量。连续训练是两层小YOCO、固定8-token诊断batch，验证可用性与数值有限性，不代表真实数据集收敛或长时间训练稳定性。两个版本的更新轨迹允许不同，完整loss序列保留在 [3.7.1训练记录](cases/fixed-triton371-training-12steps/nnscaler-training.json)和[3.8.0训练记录](cases/fixed-triton380-training-12steps/nnscaler-training.json)。

## 显存检查与未隐藏的诊断失败

首次 Compute Sanitizer 使用 `expandable_segments:True`，49项用例虽全部通过，但 `cuMemCreate` 返回 `CUDA_ERROR_NOT_PERMITTED`，工具退出99。这次工具检查仍按失败记录，见 [日志](cases/fixed-triton371-memcheck/run.log)和[结果](cases/fixed-triton371-memcheck/result.json)。

随后仅为诊断进程改用 `backend:native,expandable_segments:False`，保留原错误检查参数并新建case。49项用例再次全部通过，memcheck报告0错误、工具退出0。普通回归和连续训练继续使用原来的可扩展分配器配置。两次case的完整工具日志均在对应目录中；汇总见 [VALIDATION.json](VALIDATION.json)。

## 环境、前向边界与资料

复用保留的四卡Job `yoco-align-fast-vllm-vllm-train`，只使用已分配的GPU2：NVIDIA B200，UUID `GPU-3c98295b-46bd-92b3-ef14-82b5e35524f7`。Python3.12.3、PyTorch `2.11.0a0+eb65b36914.nv26.02`、CUDA13.1、FlashAttention `2.7.4.post1+nv26.2.44259020`。3.7.1使用独立Triton安装目录，3.8.0使用保留环境。软件/路径记录见 [3.7.1](environment-triton371.json)与[3.8.0](environment-triton380.json)。

本轮vLLM生产代码对应 `6339a81cdf8c7c2d8f8da2cdff11afc6f5580c80`。未重跑真实L3 checkpoint的三端完整词表对照；本轮前向/概率回归以及源码不变核验通过，原三端证据的条件和范围继续沿用。没有扩大到任意输入、CP/EP多卡、长期训练或backward bitwise保证。

本轮没有重测训练整步吞吐或AIPerf，不更新历史性能表数值。测试结束后GPU2显存回到0 MiB，Pod UID、节点和restartCount不变，见 [POD_VERIFICATION.json](POD_VERIFICATION.json)。

原始结果、启动脚本和源码快照同时落盘到本地与PVC，归档读回SHA256一致：

- PVC：`/mnt/pvc/lidong1/align-backward-fix-20260908/results.tar.gz`。
- SHA256：`68525aaa692a65c91bacf7e49c41b83f9d6e0852455448a706a6bdbb4b95b99f`。
- [BACKUP.json](BACKUP.json)记录归档大小、文件数及依赖的旧runtime归档。原失败记录也在归档中。

各case目录保留实际命令、退出码、JUnit和日志。`VALIDATION.json`区分正常通过、预期负对照失败和首次sanitizer诊断失败。
