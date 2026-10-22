# Target latent norm → FP8 → Mega 最终输入融合

2026-10-21（America/Los_Angeles）。基线：`3dad2d39f284b1d72fc3a62a841e74f909ce180a`。
本次增量仅为 target routed latent 输入融合；保留当前 B200/TP1/V2/FP8 target/BF16 KV/DSpark K8 配置。

## 实现与选择

现有分流选中 Mega 时，用一个四 warp Triton kernel 完成 FP32 RMSNorm、寄存器内 BF16 舍入、
group32 FP8/E8M0 量化及 top-k 搬运，直接写原共享 workspace，消除 norm 中间 BF16 写回和再读。
通过 YOCO RoutedExperts 子类，将 norm 延后至已有 opaque MoE 边界；latent projection、router、
shared expert 原始3072维输入、输出变换和 overlap 保持原有流程。
小 M 和 prefill 使用单 warp norm-only kernel 及原 routed backend。

按用户确认的数值标准，允许 FP32 归约次序差异；保留 FP32 统计、gamma 精度和 BF16 舍入边界。
未开启近似数学，Mega `fast_math=False`；未调整 output norm→FP8 开关，两端固定0。
现有 YAML auto 仍为 Triton M≤16、Mega decode M≥17、原 prefill 后端。
生产图仍为 **target4 / draft6**，未增加工作负载分流、阈值或图桶。

先前固定单 warp/每lane布局的版本已完成完整对照；随后同设备交错比较四种归约布局。
在 M18/36/72，四 warp 链路约1.74～1.76μs，比保序单 warp约快12%。因此采用四 warp 作为最终候选，
按误差、接受率和实际性能验收，不因少量归约舍入差异否决。

## 局部输入链路

编译 RMSNorm＋原 staging 对比融合 kernel。真实 INT32 router IDs、INT64 最终 IDs；
CUDA Graph 内100次调用、8组交错ABBA、每段20次回放，中位数如下。

| M | 原 norm＋staging（μs） | 融合（μs） | 延迟降低 |
| --- | ---: | ---: | ---: |
| 18 | 2.891 | 1.721 | 40.47% |
| 36 | 2.908 | 1.762 | 39.42% |
| 72 | 2.870 | 1.743 | 39.26% |

这是输入链路收益，不能当作整个 target 或端到端收益。
M9仅为组件检查，当前 C1不选Mega；M128/512/2048保留在summary.json，默认Mega prefill关闭，
不能把这些组件结果当作线上prefill收益。

## 完整 target 与端到端

| 输入 | 并发 | 完整 target 图 ms：基线→融合 | target 延迟降低 | 同引擎吞吐变化 | 独立进程吞吐变化 |
| --- | ---: | --- | ---: | ---: | ---: |
| 短输入 | 1 | 7.4330 → 7.4157 | +0.23% | +0.18% | -0.43% |
| 短输入 | 4 | 9.1192 → 9.0467 | +0.79% | -0.15% | -0.91% |
| 短输入 | 8 | 10.7498 → 10.6241 | +1.17% | +3.73% | +3.20% |
| 8193 | 1 | 7.6865 → 7.6737 | +0.17% | +0.01% | -2.65% |
| 8193 | 4 | 10.1538 → 10.0713 | +0.81% | -0.53% | -4.37% |
| 8193 | 8 | 12.5951 → 12.4610 | +1.06% | +0.18% | -3.77% |

每格2次预热＋3个正式种子、128输出token/request，T=1、top_p=1、忽略EOS；APC与异步调度关闭。
max_model_len12288、max_num_seqs8、batch tokens2048、固定8GiB KV。短输入与8193-token合成上下文，C1/C4/C8。
原生对照在新进程中先运行原始3dad源码，再运行最终融合源码。
同引擎对照在同一权重/KV/工作区上交错切换审计图，基线使用 norm-only＋原staging，保留新norm边界。

完整 target 时间使用实际live batch输入，在未计分audit warmup中测CUDA events，
每格6组交错ABBA、每段20次回放；同时记录同输入hidden、aux与完整logits误差。
审计额外target图池只用于实验，生产图不增加。C1两侧均不选Mega，图计时的微小差异作为低并发控制，不能归因为Mega输入融合收益。
端到端计时包含prefill、target/draft、采样、调度、KV与GPU drain；不含加载、编译、分词、HTTP。

端到端只有小面板点估计，独立进程存在运行顺序漂移；数值差异也会改变采样轨迹和实际执行工作量。
因此分别报告局部、完整target与E2E，不宣称稳定的整体引擎提升，不外推在线/多轮/多卡容量。

## 数值与接受率

六组同输入图的误差如下；relative RMS 为差值RMS/基线RMS，top1变化在整个live验证图的所有位置计数。

| 输入 | 并发 | target hidden relative RMS | logits relative RMS | logits最大绝对差 | logits top1变化位置数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短输入 | 1 | 0 | 0 | 0 | 0 |
| 短输入 | 4 | 0.0609 | 0.0323 | 2.312 | 1 |
| 短输入 | 8 | 0 | 0 | 0 | 0 |
| 8193 | 1 | 0 | 0 | 0 | 0 |
| 8193 | 4 | 0.0317 | 0.0162 | 0.9688 | 0 |
| 8193 | 8 | 0 | 0 | 0 | 0 |

首token的top20 logprobs逐位相等：**True**；256个teacher-forced位置逐位相等：**True**。
这些检查主要走prefill/小M；Mega验证图误差以上表为准，auxiliary hidden误差见summary.json。
同种子完整生成输出相等：原生 **15/30**，同引擎 **13/30**（均含预热）。
不要求生成轨迹逐位相等，也不将轨迹不同本身当作计算错误。

接受率定义为 accepted tokens / draft tokens，以下只计3个正式种子，变化单位为百分点。

| 输入 | 并发 | 同引擎接受率：基线→融合 | 同引擎变化 | 原生变化 |
| --- | ---: | --- | ---: | ---: |
| 短输入 | 1 | 44.940% → 44.940% | +0.000 | +0.000 |
| 短输入 | 4 | 38.678% → 36.832% | -1.846 | -1.846 |
| 短输入 | 8 | 35.463% → 35.326% | -0.137 | -0.137 |
| 8193 | 1 | 39.113% → 39.113% | +0.000 | +0.000 |
| 8193 | 4 | 34.548% → 33.579% | -0.969 | -0.969 |
| 8193 | 8 | 32.453% → 32.535% | +0.082 | +0.082 |

补充检查了此前保存的6组真实入口输入：4组反量化结果完全一致，另外2组分别仅1个元素变化，
最大反量化relative RMS为 **0.0586%**；所有scale组、路由ID和权重搬运一致。
入口误差很稀疏，但本面板C4最终logits的relative RMS已达1.62%～3.23%，最大绝对差2.3125，
其中短输入有1个位置top1变化；不能把整个模型称为近似逐位相等。
本轮未发现输入融合的量化格式/打包错误。按允许归约次序差异并接受局部性能收益的开发标准采用四warp，
同时保留C4接受率与E2E负向结果；没有证明四warp在完整引擎上稳定优于保序版。

本轮检查浮点误差、固定teacher-forced位置、真实采样及投机计数；未做完整下游任务质量评估。
接受率小面板变化同时受生成轨迹影响，不能解释为模型质量增减。

## 回归与复现

B200 **55项kernel/dispatch＋123项模型、prefill、Shared-KV回归通过，零skip**。
覆盖FP8/E8M0格式、实际INT32及备用INT64路由、非连续行、工作区尾部、空输入、极端/非有限数值、
图回放读取实时输入/权重/路由，以及整数合成的BF16舍入边界。
合成边界检查要求反量化relative RMS<1e-3、变化元素比例<1e-3、逐元素至多相邻FP8量级；
路由与权重搬运仍要求精确一致。非Mega norm-only保留其既有单warp布局。

```bash
.venv/bin/python -m pytest --confcutdir=tests/kernels/moe -q tests/kernels/moe/test_yoco_mega_moe.py
.venv/bin/python -m pytest --confcutdir=tests/model_executor -q tests/model_executor/test_yoco_prefill.py tests/model_executor/test_yoco_shared_kv.py tests/model_executor/test_yoco_moe_composition.py tests/model_executor/test_yoco_latent_norm_fp8.py
.venv/bin/python benchmarks/kernels/benchmark_yoco_mega_input.py --output <new-result.json>
```

完整本地产物：`/data/vllm_test/work/yoco-target-mega-input-fusion-20261021/`。
最终冻结源码、运行脚本、配置与全部结果在`attempt5/`；保序版在`attempt4/`，此前失败尝试亦保留。
不混用早期失败/中间版本吞吐；私有checkpoint激活张量不提交。
8059个源码文件、1996个缺失运行时支持文件逐文件验签；执行后只新增/调整说明与结果文档。
运行库沿用Torch2.13/cu130、Triton3.7及DeepGEMM2.8.2（commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`）。
使用既有Pod `living-cobra-b200g4-dev-f5def941-master-0` 的GPU6和已有flock；未更改服务、Job或Pod配置。

[汇总JSON](summary.json) · [六格CSV](summary.csv) · [worker与图验收](worker-receipts.json) · [源码验签](source-audit.json) · [归约布局对比](variant-probe.json) · [真实入口误差](fast-saved-inputs.json)
