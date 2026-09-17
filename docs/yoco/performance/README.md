# YOCO 性能结果入口

[单卡 Fast BF16 decode 复现](FAST_BF16_DECODE_BENCHMARK.md)与
[Fast FP8 decode 复现](FAST_FP8_DECODE_BENCHMARK.md)共用固定输入和计时实现。
旧分支原始测量保留在各报告；当前版本的适配和复测见[增量合并报告](../fhb-dev-9-18-merge.md)。

新增[Mooncake 1.2×持续表](THROUGHPUT_F1P2.md)和[Fast负载响应报告](fast-mooncake-f1p2-20260908/REPORT.md)。与1×表分开维护。

日常开源trace比较和更新使用 [三模式持续吞吐表](THROUGHPUT.md)。低并发固定形状W2使用独立的 [低并发W2表](LOW_CONCURRENCY.md)；两组工作负载与统计口径不同。

每次只重测本次修改涉及的模式，并只更新该模式已测的拓扑；其他模式沿用历史结果并保留测量日期。
使用 `update_throughput.py --mode ... --source ... --case ...` 导入一个已审计的case，不会触发其他测试。
固定工作负载和更新规则见表内说明；原始run目录保留失败、原始日志、环境和审计证据。
历史数据在 [throughput/history](throughput/history/)，机器可读当前值在 [throughput/current.json](throughput/current.json)。

## 仓库发布副本

原三模式trace表及其证据同步到 vLLM 与 llm-train 的 `fhb-dev-9-8` 分支。2026-09-08只重测Fast单卡与1P1D并更新对应两行；Qwen/Align四行和所有旧history记录保持原数值、时间及SHA256。最新结果见[Fast Mooncake报告](fast-mooncake-20260908/REPORT.md)。此表是推理服务性能，不能用来推断 llm-train 的训练整步吞吐。

仓库保留报告、当前表、不可覆盖的历史记录、图和关键审计 JSON。完整逐请求日志、trace 和 runtime 继续保存在测试归档中：

- 最新Fast：[归档与SHA256](fast-mooncake-20260908/BACKUP.json)。
- 2026-09-07 Fast：`/mnt/pvc/lidong1/fast-compare-b200-20260907/results.tar`，SHA256 `b7881086ae0b12e0fad329f751705004604daa9530c9abf361ed36f4c54af4d5`。
- Qwen3：归档位置和 SHA256 见 [BACKUP.json](qwen3-compare-b200-20260907/BACKUP.json)。
- Align：归档位置和 SHA256 见 [RESULTS_BACKUP_COMPLETE.json](align-4gpu-1p1d-20260906/RESULTS_BACKUP_COMPLETE.json)。

在本目录中按 [持续表](THROUGHPUT.md) 的命令导入新结果；先把本轮 source、case manifest 与审计证据放到对应的相对路径。`update_throughput.py` 只更新表，不启动压测。

图是本轮测量快照。`plot_throughput.py` 重绘完成曲线还需要归档中的 `cases/*/artifacts/profile_export.jsonl` 与 trace 文件；先按原目录结构还原，再用具有 numpy / matplotlib 的虚拟环境运行。原报告和 JSON 中的绝对路径是测试机器上的来源记录。

## 低并发 W2 组

2026-09-08 UTC建立的 [W2表](LOW_CONCURRENCY.md) 对照本轮旧Fast、恢复split-KV后的Fast和Qwen3。工作负载为65,536输入、16,384输出，并发1/2；同物理B200、BF16 TP1，每点一次完整测量，共享节点诊断。Fast代码修改在vLLM仓库；本轮报告、W2表及原始汇总同步保存到vLLM与llm-train的`fhb-dev-9-8`分支。训练仓库同时记录配套vLLM版本。

机器可读当前值在 [low-concurrency/current.json](low-concurrency/current.json)，各行实际测量时间与源码、结果SHA256保存在 [low-concurrency/history](low-concurrency/history/)。更新时只替换实际重测的模式/并发行，保留其他行时间；不能混入Mooncake表。原始证据与PVC归档见 [报告](../fast-low-concurrency-20260908/REPORT.md)。

## Fast block-128 FP8 组

[Fast FP8 W2 独立调参](FP8_W2_TUNING.md)：M1/M2/M4 使用独立配置，M8/M16 保留原选择。

[BF16 内部归约与采样实验](BF16_INTERNALS.md)：实现自有 RMSNorm 和 greedy logprob 的原生 BF16 运算；额外吞吐收益很小，独立 NLL 略升。当前 MMA 累加类型和 DeepGEMM scale 接口不支持全部改成 BF16，实验默认关闭。

[BF16 主干上下游与 router](BF16_CHAIN.md)：残差边界、router 输入/权重/logits 使用 BF16；同卡 B1/B8 吞吐提升 3.19%/2.42%，prefill 延迟降低约 2.5%。保留内部 FP32 累加及控制数据，实验开关默认关闭。

[主残差 BF16 同卡 A/B](BF16_RESIDUAL.md)：decode 基本持平，prefill 未提速；平均 NLL 接近但逐 token 存在变化，实验开关默认关闭。

[latent Norm → FP8 投影实验](LATENT_NORM_FP8.md)：融合实现和验证结果；存在整模型精度漂移，默认关闭。[主残差与 FA4 输出精度分析](RESIDUAL_ATTENTION_PRECISION.md)记录真实激活的局部重放证据。

[FA4 attention 前后 FP8 融合](ATTENTION_FP8_FUSION.md)：自注意力 QKV、跨层 Q、共享 KV 和差分输出的融合量化；保留关键高精度计算。包含 B200 数值回归和同卡计时。

[Fast直接FP8激活与Top-8 logits路由](fast-fp8-route-direct-20260910/REPORT.md)：已接入两项优化并区分shared/Align舍入路径；B200完整回归403项通过，最终定向复查通过。低M routed-expert链路加速约1.3%–4.1%；公开文本1,024个固定位置的新旧FP8平均NLL未观察到退化。局部计时和数值变化见报告，本轮未重测Mooncake。

[训练/推理FP8精度审计](fast-precision-audit-20260910/REPORT.md)：实际checkpoint训练使用mxfp8/block128；推理FP8覆盖与训练基本一致。记录可减少的routed激活转换和路由计算，以及LM head、latent、lambda与KV的降精度候选、实现限制和验证顺序；本轮没有新增性能测量。

[最新 BF16 / FP8 的 Mooncake 2× 同卡对照](fast-bf16-mooncake-f2-20260909/REPORT.md)：Fast BF16 **1238.79**、当前 Fast FP8 **1407.84** 输出 tok/s，FP8/BF16 **1.136×（+13.65%）**。本轮仅补测BF16，FP8保留上一轮正式结果；同一源码、同物理B200、同一2× trace与客户端上限2048，两端3643请求全部成功并排空、客户端/服务端门槛通过。共享节点、每精度一次正式测量、无延迟SLO，按过载诊断解读。见[维护表](FP8.md)。

[公共路径复用前后的 FP8 内部对照](fast-fp8-reuse-mooncake-f2-20260909/REPORT.md)：当前Fast FP8为1407.84输出tok/s，相对本轮修改前基线+0.46%，基本持平；3,643请求全部完成、客户端门槛通过、正式轮无新增JIT。见[维护表](FP8.md)。

[Fast 多精度公共实现复用](fast-precision-reuse-20260909/REPORT.md)记录分层精度判断、K/V 与 shared-expert 融合、缓存刷新及验证；本轮没有更新 Mooncake 或 BF16/FP8 端到端吞吐。后续步骤见[计划](FAST_PRECISION_PLAN.md)。

[最新分派与布局优化 / Mooncake 2×](fast-fp8-dispatch-f2-20260909/REPORT.md)：同卡低并发吞吐提升17.81%–26.32%，2×吞吐1094.75→1407.82 tok/s（+28.60%），TTFT P95降低50.77%。两端客户端上限2048，均3643请求成功并排空、调度门槛PASS；仍属无延迟SLO的共享节点过载诊断。209项测试通过；小batch更换GEMM后端后不保证bitwise，batch8检查出现token变化。见[FP8表](FP8.md)及[结果JSON](fast-fp8-dispatch-f2-20260909/comparison.json)。

[2026-09-09 有效路由行量化优化](fast-fp8-sparse-20260909/REPORT.md)：同卡低并发吞吐提升3.93%–16.07%，Mooncake 1.2×观测提升1.18%；两端客户端调度门槛均未通过，按诊断解读。183项测试通过，详情与最新值见 [FP8表](FP8.md)和[机器可读结果](fast-fp8-sparse-20260909/comparison.json)。

[修改前的 FP8/BF16 同卡对照](fp8-vs-bf16-20260909/REPORT.md)保留原始精度比较条件和数据。

[Fast FP8 适配报告](fast-fp8-compat-20260909/REPORT.md)补齐在线 FP8 默认分派与回退语义；这轮兼容性验证不改写下面的历史吞吐结果。

[FP8表](FP8.md)记录 `--fast --quantization fp8_per_block --moe-backend deep_gemm` 的同卡对照，包含低并发诊断和 Mooncake 1.2×回放。2026-09-08 PDT / 09-09 UTC 这轮缩小小批次 MoE workspace、补齐在线 FP8 启动预热；代码基线为合并后的 `511fbed3f75f0fd18a4f93194ca832db2c723f98`。全量测量结果、失败记录和归档见[报告](fast-fp8-20260908/REPORT.md)。

机器可读结果为 [comparison.json](fast-fp8-20260908/comparison.json)；绘图仅需要同目录的结果和 CSV。该组使用 block-128 W8A8/UE8M0，不能直接与历史 BF16 表或 block-32 MXFP8 结果混合比较，也不代表训练吞吐或前向一致性验证。
