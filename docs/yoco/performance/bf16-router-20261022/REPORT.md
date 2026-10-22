# YOCO BF16 router、主链与 DeepInfer 复用评估

2026-10-22，基线 `c616521d8c4d7f5a7182d6315f539f322bd37acf`。

## 采用的修改

Fast 默认启用独立的 `VLLM_YOCO_BF16_ROUTER=1`：router 的输入、计算权重缓存、
GEMM 输出 logits 使用 BF16。checkpoint 主权重与加载时的权重归一化保留 FP32，
BF16 计算缓存不进入 checkpoint，reload 时原地刷新以保留 graph 地址。
当前20层、每层128×3072的BF16 router缓存额外约15 MiB，原FP32主权重继续保留。

实际 target 的 3072 维 Norm 已输出 BF16；原路径会将它上转 FP32 后执行 TF32 gate。
本次直接消费这个 BF16 输出。此前把合成 FP32 输入的转换开销用于实际模型的结论已更正，
原始测量未覆盖或替换，纠正记录保留在前轮评测目录。

独立精度开关注册进 `vllm.envs`，因此进入 AOT compile factors。历史诊断中发现过
自定义开关遗漏 cache key、错误复用 FP32 AOT 的情况；该无效评测已排除。
当前每个引擎都在预热中核对实际 router op，避免只看环境变量判断精度。

`VLLM_YOCO_BF16_CHAIN=0`、`VLLM_YOCO_BF16_REDUCTIONS=0` 仍是默认值。
设置 `VLLM_YOCO_BF16_ROUTER=0 VLLM_YOCO_BF16_CHAIN=0` 可在启动时恢复旧 router；
旧 `BF16_CHAIN=1` 仍隐含启用 BF16 router。Align 忽略 Fast 的 BF16 开关。
没有按并发、M 或接受率动态切换精度；每个引擎保持 target 4 / draft 6 张图。

## 为什么没有全部移除 FP32

本轮已测 BF16 residual/router 主链，以及进一步采用原生 BF16 Norm 归约的组合。
两者在当前 K8 面板都有回退，未证实比独立 router 更稳定，因而未进一步改默认。
完整主链与严格归约组合没有运行本轮 MATH-500；不能沿用 router-only 分数作为其精度结论。

当前 FA4/DeepGEMM Tensor Core 操作没有 BF16 累加器选项；相关 FP8 运算使用 FP32
累加。FP8 scales、softmax/top-k 概率和 K8 随机采样也继续使用当前接口规定的精度。
`VLLM_YOCO_BF16_SAMPLING=1` 仅支持 greedy，不能用于本轮 probabilistic K8。
具体后端探针与 PTX 类型限制见 [BF16 内部实验](../BF16_INTERNALS.md)。
这次没有实现完全消除 FP32 的引擎，也没有为此改成慢速标量 GEMM。

## DeepInfer 可以复用什么

检查的 DeepInfer 为 `9bd9cf`，DeepGEMM 为 `1fc9c669`。
其 `bf16_mega_gate` 融合 GEMM/top-k，但实现 sqrtsoftplus，当前 YOCO 使用 softmax，
因此不能直接替换。其线性层使用的 `deep_gemm.bf16_gemm_nt` 已与 BF16 `F.linear`
对比：本环境选择 cuBLASLt，输出完全一致、GPU 时间基本相同，更换 API 没有额外收益。

额外试过朴素 Triton cast+GEMM 和小 M 向量 kernel；只在少数形状快，C8/prefill 更慢，
未加入新的形状分派。微测使用相同真实权重、随机 BF16 激活和 CUDA Graph replay，
这些结果用于核对计算成本，不直接代表端到端性能。

| gate 输入行数 M | 原 BF16→FP32/TF32 µs | 直接 BF16 µs | DeepGEMM BF16 µs |
| --- | ---: | ---: | ---: |
| 9 | 8.391 | 4.221 | 4.201 |
| 18 | 8.515 | 4.222 | 4.253 |
| 36 | 8.650 | 4.303 | 4.302 |
| 72 | 8.519 | 4.915 | 4.915 |
| 2048 | 21.533 | 5.222 | 5.222 |

9/18/36/72行的原生 BF16 Norm 微测减少约 15%–18% GPU 时间，但输出相对 RMS 差约 0.3%；
完整 K8 引擎没有稳定受益。该实验和独立 router 的采用决定分开。

## 端到端测量口径

共享节点、同物理 B200 GPU 6，TP1、V2、FP8 target、BF16 draft/KV、DSpark K8 probabilistic /
standard rejection、fast-prefill，APC 和 async scheduling 关闭。maxlen139264、maxseq8、
batched tokens2048、KV8GiB，原有 YAML auto Mega 策略保持。target/draft 图数分别为4/6。
计时包括 prefill、target、draft、采样、CPU 调度和 GPU drain，不含加载、编译、预热。
正式计时无 profiler；CPU op audit 在排除的预热中执行。

第一面板使用8条固定提示词的前C条：C1实际上只测试“17×23”一个提示词。
8193输入由同一技术文字前缀补齐。每个case两次预热、三个seed正式测量，
每请求固定128输出token、ignore_eos。
router 对照按 legacy-A / router-A / router-B / legacy-B 顺序执行；legacy-A 来自第一轮冻结源码，
legacy-B 来自最终候选、显式关闭 router flag。两次legacy和两次router各30组预热/正式输出
均逐token一致，两种精度之间各组有token变化。

表中吞吐按**总输出token / 总墙钟秒数**计算；每精度合并两进程、六次正式请求批次。
它包含精度改变引起的生成轨迹与接受率变化，不能当作纯kernel加速比。

| 输入 | 并发 | 旧路径 output tok/s | BF16 router output tok/s | 变化 |
| --- | ---: | ---: | ---: | ---: |
| 短 | 1 | 344.96 | 316.38 | -8.29% |
| 短 | 4 | 912.52 | 984.59 | +7.90% |
| 短 | 8 | 1538.30 | 1549.84 | +0.75% |
| 8193 | 1 | 172.40 | 164.92 | -4.34% |
| 8193 | 4 | 233.42 | 250.44 | +7.29% |
| 8193 | 8 | 260.18 | 270.17 | +3.84% |

短输入C1三个seed合计，旧路径84轮draft、接受302token，BF16为97轮、接受296token；
平均接受长度由4.595降至4.052。相应执行步从87增至100。两次复测接受计数与输出完全重现，
说明该提示词的额外执行轮数抵消了更快gate；不能承诺低并发对所有提示词都获益。

为检查单提示词的局限，补测全部8条提示词：单并发、每题512固定输出、三个seed，
每精度每输入长度24个正式请求；两次预热排除。每精度一个独立引擎，按旧→BF16顺序，
结果仍是固定面板观察值，不能外推全部工作负载。逐提示词数据保存在JSON中。

| 输入 | 旧路径 output tok/s | BF16 router output tok/s | 变化 |
| --- | ---: | ---: | ---: |
| 短 | 382.95 | 392.46 | +2.48% |
| 8193 | 295.23 | 288.27 | -2.36% |

短输入有6/8条提示词提升，8193输入有1/8条提升。BF16 router 的默认选择
包含这些回退；保留启动时恢复旧精度的选项，不增加运行时分派或图集合。

主链进一步降精度的第一轮结果如下。每格只来自三个seed、一个引擎；按总token/总秒数统计，
与历史文档的不同checkpoint、非K8实验不能混合。严格BF16=主链BF16+原生BF16 Norm归约。

| 输入 | 并发 | BF16主链吞吐变化 | 严格BF16吞吐变化 |
| --- | ---: | ---: | ---: |
| 短 | 1 | +6.70% | -0.68% |
| 短 | 4 | +1.86% | +12.06% |
| 短 | 8 | -6.60% | -5.06% |
| 8193 | 1 | -4.66% | -2.94% |
| 8193 | 4 | +3.08% | -0.10% |
| 8193 | 8 | -1.16% | -0.43% |

## MATH-500，128K 输出上限

最终候选重新完成全部500题，每题输出上限131072 token，context139264，temperature1、
top_p1、seed20261022，native Max reasoning模板加step-by-step/boxed指令。自动评分为
BF16 router **490/500（98.0%）**，
对照前轮原FP32/TF32的 **485/500（97.0%）**。
最终500条输出与前轮有效BF16逐token完全一致，评分也完全一致。

评分器使用math-verify0.8，仅在实际`</think>`之后提取最终答案，gold用Latex，prediction用
Expr与boxed优先的Latex，float_rounding5。前轮每侧各一题被提取器误判，复核后为
FP32/TF32 **486/500**、BF16 **491/500**；
复核记录与自动分数分别保存。BF16两题达到131072上限且未完成推理，仍计错，未移出分母。

自动分数配对gain10/loss5，McNemar p=0.302，bootstrap95%差值区间[-0.6,+2.6]个百分点；
本轮未观察到准确率下降，但单轮+1.0pp不构成显著提升。MATH输出长度变化很大，
不能把其运行时间当作隔离后的端到端加速比。

## 验证与证据

最终候选186项模型回归、57项BF16缓存/边界/kernel回归通过；此前主链/严格归约候选
另有185项模型、97项kernel测试通过。图内权重缓存原地刷新、fake metadata、BF16/旧路径
和Align行为由已有套件覆盖；本次新增独立默认值、Norm精度不变与AOT key切换回归。
V2、fast-prefill的auxiliary hidden与decode当前token合同保持现有实现。

冻结源码8070文件、运行支持1996文件SHA256核对通过；最终工作树5431个代码/配置文件
与已测快照一致。所有最终引擎预热都实际调用对应router40次，另一精度0次，
启动与结束的graph数均为4/6。Pod UID、node和restart count核对保持，PID7存在；
本任务释放GPU6后显存为0 MiB、utilization为0%。

本地完整实验目录：`/data/vllm_test/work/yoco-bf16-router-fusion-20261022/`。
前轮FP32/BF16 MATH-500及评分器：`/data/vllm_test/work/yoco-router-math500-20261022/`。
机器可读汇总见 [performance-summary.json](performance-summary.json)、
[quality-summary.json](quality-summary.json) 与 [audit.json](audit.json)。

复现使用已归档的`run_candidate2.sh`（测试、ABBA后半段、完整MATH）与`run_balanced.sh`（扩展C1），以及原`harness-v1/run_profiles.sh`（主链/严格归约）。运行脚本持同一GPU6 advisory flock，检查空闲后顺序执行；未重启或更改Pod、Job或其他GPU服务。完整脚本、配置、逐请求token/计数、日志与评分在实验目录和归档中。

- `candidate2-upload.tar.gz`，SHA256 `5c7f52537de6064a3f88e86a4255b46108d622f3679d20c73a8b7c394fb0a98e`。
- `profiles-upload.tar.gz`，SHA256 `06e10068d55d547e2493d620d2c98adf2a3514499f4ca6e5ab9252c13059ab33`。
- `results-evidence.tar.gz`，SHA256 `2d30dc069d20e6864a78b55b810db283b20fa5b548e8e8ee9e7bef432efb645b`。
