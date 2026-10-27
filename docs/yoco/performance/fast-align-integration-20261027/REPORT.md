# FAST Mega Gate / Align GEMM 集成验收（2026-10-27）

分支：`fhb-dev-10-27`。本次工作整合已有实现并检查回归；没有重新调优算法、扩大 DP/EP 支持或重测完整模型吞吐。

## 来源与合并方式

| 输入 | 固定版本与内容 |
| --- | --- |
| 共同基线 | `7c3f32c985b5b98f83ee45ce9311d291a90b0348` |
| FAST 工作区 | 基线上的 10 个已跟踪修改文件和 13 个未跟踪文件，完整保存为快照提交 `e4de8344c7959565b49062647730c53faf4303d7` |
| Align | `align-update` 的 `30142cfe6f136ed0bcff3dc6a526d0dc37df00ef`，实现提交为 `e419464d6c753ef72700625f0086003eba4646f4` |

从共同基线创建独立 worktree，提交 FAST 快照，再以保留双方历史的 merge 合入 Align。
`vllm/envs.py` 与 `yoco_moe.py` 自动合并成功，无文本冲突。
原 FAST 工作区的分支、HEAD、Git 状态和全部 23 个修改/新增文件的 SHA256 保持不变；原 Align 工作区保持原分支和提交。
llm-train 继续配套使用已发布的 `align-update`，本次没有修改训练仓库。

## 最终行为

- FAST：保留 Mega Gate v2 的非有限 padding 处理、API v2 检查、编译缓存标识、权重缓存刷新和不兼容配置回退。
  包含 10 月 26 日的纯 Prefill 修复：配置了 `kv_producer` 的引擎用 `max_num_batched_tokens` 判断资格；
  standalone/Decode 保留序列容量规则。默认开关为 1，实际调用阈值为 128 行。
- Align：保留共享 BF16 输入/输出、FP32 累积的 cuBLASLt 前向，算法与 workspace 准入、非对齐输入回退，以及 QKV/shared gate-up 融合。
  默认仍为 legacy；`VLLM_YOCO_ALIGN_GEMM_BACKEND=cublaslt_v1` 与 `VLLM_YOCO_ALIGN_FUSE_PROJECTIONS=1` 显式开启优化。
- Mega Gate 的资格检查在 Align 模式直接拒绝启用。FAST 路由与 Align 的数值契约保持分离。
- 沿用 CUDA worker 的 GPU 本地 CPU NUMA 绑定要求。

Mega Gate 的依赖 patch 和许可证随 FAST 快照一并进入分支，安装方法见[说明](../mega-gate-20261025/README.md)。
Align 通过正式 CMake `_yoco_align_C` target 构建并由正常模块导入，不提交本机二进制文件。

## 本次回归

回归结果汇总见 [validation-summary.json](validation-summary.json)。

| 环境 | 检查 | 结果 |
| --- | --- | --- |
| B200/SM100 | FAST routing 与 Align invariance 原生 kernel 测试 | 170 passed |
| B200/SM100 | YOCO 配置、routing、MoE composition | 121 passed |
| B200/SM100 | 新增 Mega Gate / Align GEMM 同图互操作检查 | 1 passed；B200 合计 292 passed |
| 本地 A6000 | YOCO 配置、routing、MoE composition | 121 passed |
| 本地 A6000 | kernel 测试 | 32 passed、136 skipped、2 failed，见下述基线复现 |

改动范围的 pre-commit 检查和全仓库 mypy（Python 3.12）均通过；合并没有未解决的冲突。

远端测试使用的 5,163 个源码文件与本地逐项 SHA256 一致，GPU7 测试后恢复至 4 MiB / 0% 利用率。

原生测试环境：Torch 2.13.0+cu130、cuBLASLt 130101，加载支持 `yoco_mega_gate_version=2` 的开发 DeepGEMM。
在原开发 Pod 的空闲 GPU7 上分时执行，使用独立源码、编译目录和缓存；未替换已有服务或修改原运行环境。
源码中的 Align C++ SHA256 为 `89add19dec2297cdeb950d3d2ed8266b283b50ec5844c73ca33407d2b0fb58ce`，与已验收实现相同。
本次新构建扩展 SHA256 为 `a44ba0954ede17bd975e8850e19de9d78f0239e92381ee6faf51fff2ab16f0b7`。

新增互操作测试在同一个 CUDA Graph 中运行 Mega Gate 与 Align GEMM，更新输入和两份权重后重放。
检查 FAST 路由结果与其自身 eager 结果逐位相同，Align GEMM 与旧 Align 输出逐位相同。
DeepGEMM 的 scratch 与 stream 绑定，因此预热和 capture 使用同一显式 stream。
首个测试草稿在默认 stream 预热、另一 stream 捕获，触发 scratch 初始化保护；修改测试预热方式后复验，未更改生产实现。

### A6000 失败的归属

`test_align_weighted_clip_does_not_change_at_128_tokens[8/64]` 在 A6000 上失败。
使用原 `align-update` 源码重新运行相同测试，核验实际加载的模块路径后，两项均复现失败。
RMSClip 源文件在原分支与合并版本中相同，SHA256 为 `3c709a0e6c8d2f61990ec31dce7a2a3799a082b505378c1a642b63bafa4ea1d7`。
这两项在本次 B200 原生回归中通过。保留 A6000 的失败记录，不放宽断言，也不声明当前 Align 契约支持 A6000。

### 检查入口

在安装两套配套原生扩展的环境中：

```bash
.venv/bin/python -m pytest --confcutdir=tests/kernels -q \
  tests/kernels/test_yoco_fast_routing.py \
  tests/kernels/test_yoco_align_invariance.py
.venv/bin/python -m pytest --confcutdir=tests/model_executor -q \
  tests/model_executor/test_yoco_config.py \
  tests/model_executor/test_yoco_routing.py \
  tests/model_executor/test_yoco_moe_composition.py
.venv/bin/python -m pre_commit run --files <changed-files>
.venv/bin/python -m pre_commit run mypy-3.12 --all-files --hook-stage manual
```

源码 whitespace 检查排除原样保留的 Matplotlib SVG 和统一 diff `.patch` 文件；后者的空白上下文具有 patch 语义。

## 性能与证据范围

源分支测得的 Align 推理吞吐提升为 C1/C8/C32 的 43.1%/24.4%/14.2%，详见根 README。
FAST Mega Gate 的既有完整模型结果为 C16–64 提升 1.1%–2.8%，同一 maxseq64 引擎的 C1 回退 11.7%；
纯 Prefill 没有证明稳定吞吐收益，且部分输出 token 改变，见[FAST 完整模型结果](../mega-gate-20261025/FULLMODEL_RESULTS.md)
和[Prefill 结果](../mega-gate-20261025/PREFILL_RESULTS.md)。
这些是合并前的历史结果，不能相加或当作新分支重新测速的结果。
本次原生回归与 source hash 核验不替代完整模型重新评分、吞吐测试或任务质量评估。

完整快照、构建日志、JUnit、源文件哈希和保留的失败尝试位于开发任务目录：
`/data/vllm_test/work/yoco-fast-align-merge-20261027/`。
