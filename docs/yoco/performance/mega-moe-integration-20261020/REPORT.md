# YOCO MegaMoE 正式接线后的端到端验收

归档日期：2026-10-21（America/Los_Angeles）。开发起点 `7780afb5ac86139269fb31c8dcf330e1feaeabdd`；目标分支 `buaahsh/vllm:fhb-dev-aligndeepinfer-startpoint`。

**结论：启用 Mega 后，短输入 C4/C8 的端到端输出吞吐提升 15.04%/15.76%，4096-token 输入提升 8.86%/9.77%；C1 分别回退 4.90%/6.59%。合入为显式选项，默认 `off`。** 本轮使用仓库中的正式配置接线；观测扩展只记录状态、同步与 Graph 回放，没有替换模型或 expert forward。

A/B 两端共同包含 FP8 target 接入、FP32 proposal cache 和 int64 页地址修复。吞吐增量以**相同 FP8 target + K8 配置下原 routed 后端**为基准；原始分支默认 BF16 serving 配置的总体变化未在本轮测量。

## 吞吐

| 输入 | 并发 C | 当前 output tok/s | Mega output tok/s | 合并两轮变化 | 两轮分别变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| 短输入 | 1 | 419.66 | 399.11 | -4.90% | -4.41% / -5.38% |
| 短输入 | 4 | 885.32 | 1018.52 | +15.04% | +15.50% / +14.59% |
| 短输入 | 8 | 1495.33 | 1730.94 | +15.76% | +16.24% / +15.28% |
| 4096-token 输入 | 1 | 338.13 | 315.86 | -6.59% | -5.66% / -7.49% |
| 4096-token 输入 | 4 | 623.57 | 678.82 | +8.86% | +10.65% / +7.12% |
| 4096-token 输入 | 8 | 882.32 | 968.50 | +9.77% | +12.16% / +7.47% |

同一空闲 B200 GPU 6，顺序 A1(current) → B1(Mega) → B2(Mega) → A2(current)。每请求固定输出 512 token，T=1、top_p=1、K8、忽略 EOS、关闭 APC。每档每轮预热 2 次，短输入正式 5 次，4K 输入正式 3 次；共 **96 次正式测量、48 次预热**。每后端每档短输入 10 个、4K 输入 6 个正式样本。表内吞吐为全部完成 token / 全部计时时长，没有删样本或选最好一轮。

输入为 8 个固定短问题；4K 场景在相同问题前增加合成技术文本填充至 4096 token。固定请求集用于复核计算路径与形状，不代表真实多轮会话、长 trace 或线上容量。

计时包含 target 全部逻辑层、Shared-KV draft、prefill、调度、sampling、接受/拒绝、KV 管理和 GPU drain；权重加载、编译、分词及 HTTP 传输在计时外。四组正式区间均无 Triton JIT 监控告警；原始日志中的启动/数值检查/预热告警保留。

## 延迟和接受率

单位为 ms；下表的 TPOT 是**每请求平均 token 间隔的 P95**，不是逐 token ITL P95。

| 输入 | C | TTFT P95：当前 → Mega | 请求平均 TPOT P95：当前 → Mega | 接受率：当前 → Mega |
| --- | ---: | --- | --- | --- |
| short | 1 | 112.20 → 95.27 | 2.23 → 2.58 | 48.04% → 41.98% |
| short | 4 | 229.02 → 193.78 | 4.21 → 3.79 | 39.80% → 39.06% |
| short | 8 | 229.10 → 193.12 | 5.01 → 4.17 | 37.42% → 38.28% |
| context4096 | 1 | 319.62 → 274.39 | 2.42 → 3.04 | 46.73% → 37.89% |
| context4096 | 4 | 1005.88 → 840.11 | 4.77 → 5.00 | 39.42% → 34.30% |
| context4096 | 8 | 1911.07 → 1552.92 | 7.20 → 6.65 | 35.68% → 32.70% |

4K/C4 的吞吐与 TTFT 改善，但请求平均 TPOT P95 从 4.77 ms 变成 5.00 ms。C1 的接受率降低、生成时间增加，应保留原后端。接受率按 verifier 接受数 / proposal 数统计，包含最终输出上限之外仍然付费计算的验证槽；不将接受率当成吞吐替代指标。全部测量请求返回 512 token，零 preemption，未标记 corrupted。

## 数值和显存边界

- 每组在 4 条固定参考续写上记录 2048 个 token 的原始 logprob；每条续写重复扩展至 512 token，是固定轨迹数值观察，不是独立任务质量集。
- current 两轮平均 NLL 均为 `0.2499422504`；Mega 两轮均为 `0.2513337734`，差 `+0.0013915230`。对应 token 的平均绝对 logprob 差 `0.01409454`、最大 `0.85651493`。
- 相同后端的两轮固定轨迹 logprob 完全相同；正式生成的完整输出 IDs 也分别为 current 24/24、Mega 24/24 相同。
- Mega 保留原 FP8 权重字节与 scale 值，将 routed 激活从 group128 改成 group32；这会改变 target 分布、生成轨迹及接受率，吞吐变化包含这些效应。不能据此宣称质量等价或提升，Align 不使用此路径。
- 20 个物理 MoE 层共享一个 `525678464-byte`（约 501 MiB）工作区，均确认实际选择 Mega。保留旧权重与新布局后，完整测量结束时 PyTorch active allocation 相对 current 平均增加 **19.78 GiB**；这是 active allocation 差，不是峰值 board memory 或最小可用显存保证。

## 代码与运行条件

- `--additional-config '{"yoco_execution_mode":"fast","yoco_mega_moe":"fp8_group32"}'` 选择候选；`off` 保留原分派。参数纳入现有编译缓存标识。
- 只替换 routed core；router、latent input/output、shared expert、独立 shared gate、runner 的 shared overlap 及归约边界保留。所有 M 使用同一个已选后端，没有对路由、请求或异常 case 加特例。
- 初版限 SM100 Fast L3、E128/top8/latent1024/FF3840/clamp10、online block128 FP8、BF16 输入、TP1/PP1/DP1/EP1，禁用 CP/DBO/EPLB/LoRA/sequence parallel/batch invariant。权重冻结，重载明确拒绝并要求新建 engine。
- 同时接入 K8 所需的受约束 FP8 target 支持、FP32 proposal cache，以及物理页地址乘法前的 int64 修复。未混入 OPD 采集 hook。
- target：`/mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf`；draft：`/mnt/pvc/lidong1/opd-fast-rollout-20261016/draft-parent-30000`，配置记录完整提交的 step30000 及权重 SHA256。
- BF16 KV 固定 8 GiB，116559-token 容量；max_model_len=8192，max_num_batched_tokens=2048，max_num_seqs=8。两个后端 target/draft 的最大桶均实际 FULL Graph 回放。
- Torch `2.13.0+cu130`、Triton `3.7.1`、DeepGEMM `2.8.2`。两组共用此前已构建的 DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`，扩展 SHA256 `c0f54cacf14e6024921bd9f34d2119c77b3f5544d8194110e97ed60cf0465d41`，没有在本轮更换该库。
- DeepGEMM 可选 MLOPart localization 不可用，使用普通内存路径；告警保留，未修改共享 MPS。
- Pod `living-cobra-b200g4-dev-f5def941-master-0`，UID `9a026fd8-e732-4fcb-b855-85ea6a6358ed`；节点 `slc01-cl02-hgx-0347`；仅使用 GPU 6 / `GPU-ef7d6ee6-0fce-4e0b-b14c-87bd8a36c5fe`。GPU 6 在运行前空闲，节点 CPU 与其他 GPU 仍有任务。

## 验证与产物

- CPU 定向回归：**71 passed、20 GPU cases skipped**；B200 Shared-KV/scale 定向回归：**28 passed、无 skip**。两套存在重叠，不相加成独立用例数。
- 覆盖配置选择与 Align 拒绝、权重 scale 等价展开、冻结权重重载保护、物理页 65535/65536/65537、SWA 跨页、FP32 proposal cache、既有 prefix/拒绝边界和 Graph 更新。
- 全部适用 pre-commit 通过，包含 Ruff、Markdown、mypy、SPDX、导入规则和 accelerator API 检查。
- 四组均退出 0；最终审计逐文件核对 **7964 个源码文件**，以及复用的原生库和第三方运行时支持文件。冻结源码清单 SHA256：`b5210e2ae973fdb6c6fbe90b34f5a0c1a8d9a63604e474d594851484a1406372`。验收后只补结果文档与数据，执行代码与所测快照一致。
- 所有本任务测试进程已退出，GPU 6 最终显存 **0 MiB**；Pod UID/节点/restart count=0 保持原值，GPU 4/5 的原有 worker PID 保留。

仓库内汇总：[summary.json](summary.json)、[逐次正式计时 CSV](trials.csv)。复现入口为 `tools/yoco_alignment/benchmark_mega_moe.py`，参数见 [使用说明](../MEGA_MOE.md)。

完整本地产物：`/data/vllm_test/work/yoco-mega-integration-20261020/`。`results/` 保留所有逐次计时、请求/输出 IDs、teacher-forced logprob、worker/Graph 状态、四组完整日志、JIT 审计、环境哈希及 GPU 退出状态。早期 pilot 的缺失运行时路径和诊断适配失败日志单独保留，不计入正式结果。所有四组正式运行使用冻结后的统一代码。
