# fhb-dev-9-18 增量合并

本轮把 `fhb-dev-9-18` 截至 `e8cb46cf64149e5d0386de7b78bfe212b9366b8f`
的增量合入 `vllm-yoco-version-0.29`，继续使用已拆分的模型、配置、算子与权重加载模块。
源码阅读顺序和后续修改入口见[中文代码导读](DEVELOPER_GUIDE.zh-CN.md)。

## 来源与合并范围

- 本轮起点：`fa3d96d0f09d4021075562e0ea5896580cec9822`。
- 第一轮已捕获的完整开发快照：`dd465efd1235376267aee0275019ce1b2fc277d0`。
- 来源分支新增提交：`7f4af5248d`（Fast FP8 推理与 decode benchmark）、
  `e8cb46cf64`（共用计时口径的 BF16 benchmark）。

来源分支的第一个提交包含许多在第一轮已作为未提交开发内容捕获的改动。
因此按完整快照与来源分支的差异识别新增内容，避免把旧接口和大模型文件重新带回。
最终保留真实的合并父提交；原有重构计划继续保留。

| 内容 | 本轮处理 |
| --- | --- |
| 两种精度 decode 入口 | 保留原命令名，公共实现提到 `benchmark_fast_decode.py` |
| 固定 token 输入 | 原样导入 `data/fast_fp8_decode_tokens.json`，保留来源与许可记录 |
| FP8 W2 调参表 | 移除 M16，只保留 M1/M2/M4；W13、数值 kernel 和其他模型默认配置不变 |
| W2 回归 | 继承来源断言，增加实际配置表 M8/M16 回退检查 |
| 历史报告和验证 JSON | 导入 BF16/FP8 decode 与 W2 调参资料，明确属于旧环境测量 |
| 文档入口 | 更新仓库 README、YOCO 与性能索引，新增中文导读和本报告 |

本轮 `vllm/` 下的生产改动只有调参 JSON 撤下 M16。
YOCO 主文件保持 1,040 行，底层算子和数值实现沿用第一轮已验证的版本。

## 脚本怎样适配新版

两个入口分别是
[`benchmark_fast_bf16_decode.py`](../../tools/yoco_alignment/benchmark_fast_bf16_decode.py)和
[`benchmark_fast_fp8_decode.py`](../../tools/yoco_alignment/benchmark_fast_fp8_decode.py)。
它们调用同一个[公共实现](../../tools/yoco_alignment/benchmark_fast_decode.py)，
避免两种精度的计时、输入或输出审计逐渐分叉。

- 模式通过 `additional_config.yoco_execution_mode` 明确设置。
- 移除上游已删除的 `calculate_kv_scales` 参数，记录实际 K/V scale。
- 通过 `model_runner.get_model()` 和当前 `RoutedExperts` 读取专家权重及后端，
  不再把上游的 `FusedMoE` 工厂当成可用于 `isinstance` 的旧类。
- 用公开 `LLM.enqueue()` 和 `wait_for_completion()` 提交、取回请求。
  暂停/恢复仍通过当前 core utility，保持整批入队在计时之外。
- 计时包含 resume 到输出排空，排除暂停、入队和 profiler 启停。
  异常路径也恢复 scheduler、结束已开启的 profiler。
- 源码包中没有 Git 时仍可运行；`source_revision` 留空，结果记录测试端传入的
  源码包 SHA-256、入口/公共实现哈希和输入数据哈希。
- 运行时记录实际加载的 DeepGEMM 模块与原生库路径、库哈希。

配置 preset 沿用来源：单卡 TP1/DP1/EP1、FA4、BF16 residual/router，
FP8 方案启用 block-128 权重与 FP8 KV；BF16 归约/采样实验仍关闭。
默认固定 512-token 输入、128-token 输出，5 次预热、7 次计时，B1/B2/B4/B8/B16。
这是 warm-prefix 离线生成，包含剩余 prefill、采样、调度和输出传输。

## 本轮验证

验证环境沿用第一轮绑定的单卡 B200 holder，Pod UID、节点、镜像及 holder 身份不变。
每份运行源码使用独立快照；Docker 原生 DeepGEMM 保留在
`/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm`，
原生 `_C` SHA-256 为
`73824dc1312e98cf277ae6bc017becf7e963d5b0bfab81e0c6d475d9b769c9e7`。

| 检查 | 结果 |
| --- | --- |
| 本地 W2 配置、计时边界与异常清理 | 32 passed |
| B200 W2、配置、decode scale 与 direct-FP8 回归 | 69 passed |
| BF16 / FP8 完整脚本 | 两种精度各 5 档 batch、5 次预热、7 次计时全部完成 |
| B1 / B8 CUDA Graph profile | 两种精度的 4 份 trace 各 12 个 generation 步骤、12 次 `cudaGraphLaunch` |
| 实际加载与来源 | 各 20 组专家、40 个 latent 投影 dtype 正确；FA4、K/V scale=1、原生 DeepGEMM 哈希核验通过 |
| 代码检查 | 暂存区 pre-commit 与范围内 Python 3.12 mypy 通过 |

2026-09-16，Torch `2.13.0+cu130` / CUDA `13.0`，同一 B200 顺序运行。
BF16 专家后端为 `FlashInferExperts`，FP8 为 `TritonOrDeepGemmExperts`；
KV tensor 分别为 BF16 与保存 FP8 编码的 uint8。所有计时请求实际命中 496/512 输入 token。

| Batch | BF16 总输出 tok/s | FP8 总输出 tok/s | BF16 每输出步 ms | FP8 每输出步 ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 178.70 | 171.95 | 5.596 | 5.816 |
| 2 | 275.18 | 276.11 | 7.268 | 7.243 |
| 4 | 433.87 | 482.76 | 9.219 | 8.286 |
| 8 | 729.69 | 822.15 | 10.964 | 9.731 |
| 16 | 1211.85 | 1522.91 | 13.203 | 10.506 |

表中是各 7 次计时的中位数换算，属于脚本验收和短上下文诊断。
每种精度仅一轮、在共享节点上顺序执行，未做 ABBA 或统计显著性检验；
不据此宣称迁移相对旧环境加速，也不将两种精度差异归因于单个 W2 kernel。
紧凑索引与 profile 哈希见[验证索引](validation/fhb-dev-9-18-merge.json)。

命令和完整原始记录位于工作区 `work/yoco-fhb-dev-918-20260916/`。
核验使用相同 checkpoint、固定输入及计时配置；预热已生成的原生 DeepGEMM
kernel 缓存可复用，本轮显式设置 `VLLM_DEEP_GEMM_WARMUP=skip` 跳过重复全形状预热，
各 batch 的 5 次模型预热和 Graph capture 仍执行。

首个整模型脚本尝试在加载模型前因镜像未安装 Git 而失败；修复的是可选的来源记录，
失败日志保留在 `fhb918-bf16-r1/`，不作为成功的推理测量。

最终通过的源码包为 `fhb918-r2`。两种精度均核对入口、公共实现、固定输入
以及源码包哈希；导出的 trace 按结果 JSON 中的 batch→文件映射审计，
即使 profiler 复用了第一份文件的名称前缀，也分别核对实际 generation batch。

## 如何解读数据

`performance/FAST_BF16_DECODE_VALIDATION.json`、
`FAST_FP8_DECODE_VALIDATION.json` 与 `FP8_W2_TUNING.json`
保持来源分支原始内容；它们对应原来的 Torch/CUDA/vLLM 环境。
本轮结果单独存档，不能把旧报告约 171 token/s 当作新版速度。

W2 kernel 对照覆盖原配置与新配置的输出一致性、两种路由布局、strided 输入、
padding、Graph 更新与独立 FP64 参考。整模型脚本验证两种精度路径、输出长度、
缓存命中和实际图执行，不替代模型质量评估。单卡短上下文测量不覆盖多卡性能、
长上下文容量或 HTTP 服务延迟；第一轮的其他验收范围仍见[迁移报告](migration-v029.md)。

## 保留的调试环境

完成测量并确认测试子进程清理后，沿用原服务配置在 `/workspace/fhb918-r2`
启动 `fhb918-debug-service-r2`。绑定 Pod
`lidong1-yoco-migrate-v029-b200-g1-0915-r2-holder-0` 和单卡 B200 保留，
本地转发恢复为 `127.0.0.1:18888` → Pod `18200`。
恢复后健康检查与 8-token 真实生成均通过。
请求、控制器状态和响应保存在本轮工作区；这些状态描述本次收尾时刻，
后续操作仍应先核验绑定控制器与服务。
