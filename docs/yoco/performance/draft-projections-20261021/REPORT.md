# Draft 共享 RoPE 与 MLP gate/up 合并实测

2026-10-21。两项分别从 `7b525571d0` 实现、测试，**本轮均保留为独立候选，未并入开发基线**。
局部延迟的改善尚未在这个固定面板上形成稳定的完整引擎吞吐收益。
没有与此前未采用的 Markov→采样候选叠加，也没有测试两项组合后的性能。

## 候选代码

| 项目 | 分支 | 代码提交 |
| --- | --- | --- |
| 四层共享 RoPE 准备 | `fhb-dev-draft-shared-rope-20261021` | `04088223dfb93d805caf17e4e8e81bc40aab1177` |
| 合并 MLP gate/up | `fhb-dev-draft-merged-gate-up-20261021` | `8d8bd197c7f4e87c7d4b5c294243967a13c953e9` |

源码位于 `vllm/model_executor/models/yoco_shared_kv.py`，
回归测试在 `tests/model_executor/test_yoco_shared_kv.py`。
同目录报告的数据涵盖两项独立候选；当前分支只包含该分支对应的代码改动。

RoPE 将相同 positions 的频率、角度和 cos/sin 准备移到四层循环外，
保持 FP32/FP64 运算顺序和绝对位置；每次回放重新读取 device positions。
未引入跨请求的持久相位缓存。

MLP 在推理准备时缓存拼接后的 BF16 gate/up 权重，一次 GEMM 后拆成 gate/up，
保留 SiLU、乘法和 down projection。原 FP32 参数和 checkpoint 键名保持；
派生缓存不进入 state_dict，普通及 `assign=True` 重载均原地更新缓存。
每层少一次 GEMM，共少四次；没有增加永久 gate/up 计算权重副本。
当前打包初始化临时物化 FP32 concat，可能增加加载峰值；本轮不宣称显存收益。
不同 GEMM 形状可能改变其他硬件/形状的舍入，下面的逐位一致结论限于实际验收范围。

## 固定条件

B200 单卡，TP1/PP1/DP1/no-CP；V2 runner；FP8 target、BF16 draft GEMM/KV；
K8、完整154880词表、probabilistic draft 与 standard rejection。
Target `/mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf`，
draft `/mnt/pvc/lidong1/opd-fast-rollout-20261016/draft-parent-30000`。

保留 fast-prefill、h20 auxiliary hidden、MoE auto `b200-l3-fp8-tp1-k8`：
M≤16 Triton、M≥17 Mega decode、prefill 原路径。maxlen12288、maxseq8、budget2048、
KV8GiB、APC关闭、async关闭。每请求固定输出128token、ignore_eos=True。

生产图集合仍为 target `[9,18,36,72]` 四张、draft `[8,16,24,32,40,64]` 六张。
两项均无按长度/并发分流的新策略。额外三套 draft 图池只存在于独立审计 harness，
未加入生产实现；独立 native 进程的图集合和实际回放均已核对。

## 同引擎端到端对照

Target 权重、KV 和 target graphs 固定，同一引擎在相邻 generate 调用间切换三种
独立 draft 图。短输入/8193输入 × C1/C4/C8，每格2次预热、7个正式seed，
轮换 variant 顺序。共126个正式batch、546请求、69888输出token。

| 输入 | C | RoPE 吞吐变化 | gate/up 吞吐变化 |
| --- | --- | --- | --- |
| 短输入 | 1 | +0.04% | -0.01% |
| 短输入 | 4 | -0.06% | +0.04% |
| 短输入 | 8 | +0.01% | +0.11% |
| 8193 | 1 | -0.21% | -0.07% |
| 8193 | 4 | +0.38% | +0.16% |
| 8193 | 8 | +0.31% | +0.69% |

84组正式候选/父版配对的完整输出、accepted/proposed/round 计数及逐位置接受计数一致。
接受率没有变化。[完整数据和配对核对](paired-summary.json)，
[吞吐与配对 bootstrap 区间](paired-summary.csv)，[每次计时与输出哈希](paired-trials.csv)。
区间是七个固定面板 seed/计时配对的重采样区间，包含顺序与时间噪声，
不代表文档总体或任务质量的置信区间。

## 独立 native 进程

顺序 parent-a → rope-r → gate-up-g → parent-b。
每格3个正式seed、2次预热；候选各一个进程，父版用前后两个进程的合并吞吐。
共72个正式batch、312请求、39936输出token。

| 输入 | C | 父版 tok/s | RoPE tok/s / 变化 | gate/up tok/s / 变化 |
| --- | --- | --- | --- | --- |
| short | 1 | 334.72 | 336.42 / +0.51% | 335.15 / +0.13% |
| short | 4 | 896.78 | 898.91 / +0.24% | 893.80 / -0.33% |
| short | 8 | 1475.71 | 1479.04 / +0.23% | 1473.76 / -0.13% |
| context8193 | 1 | 178.49 | 178.97 / +0.27% | 178.50 / +0.00% |
| context8193 | 4 | 241.48 | 243.42 / +0.81% | 242.35 / +0.36% |
| context8193 | 8 | 266.58 | 270.20 / +1.36% | 267.81 / +0.46% |

后置父版比前置父版慢约0.31%–3.06%，合并基线后的正向点估计受到漂移影响。
RoPE 相对前置父版为−0.60%～+0.07%，gate/up 为−1.09%～−0.29%。
该进程数不足以单独证明亚百分比改进，应结合上面的同引擎结果判断。
四进程首token top20/logprobs、每进程256个 teacher-forced 位置一致；
所有正式及预热轨迹的输出和投机计数对照一致。
见[native摘要](native-summary.json)、[逐次计时](native-trials.csv)、[图与worker回执](worker-receipts.json)。

## 完整 draft 图事件计时

在独立进程中，于请求 batch 仍完整的第一次匹配图回放处冻结真实 checkpoint、
KV 与输入，再轮换三张审计图计时。每路径9轮，每轮50次回放、3次预热，报告中位数。
包括 draft backbone、LM head、Markov 和采样，不包括 target、prefill、调度。

| 输入 | C | 父版完整 draft ms | 共享 RoPE ms / 延迟降低 | 合并 gate/up ms / 延迟降低 |
| --- | --- | --- | --- | --- |
| 短输入 | 1 | 1.4598 | 1.3953 / +4.42% | 1.4512 / +0.59% |
| 短输入 | 4 | 1.5835 | 1.5323 / +3.23% | 1.5801 / +0.21% |
| 短输入 | 8 | 1.7343 | 1.6767 / +3.32% | 1.7229 / +0.66% |
| 8193 | 1 | 1.5048 | 1.4409 / +4.25% | 1.4981 / +0.44% |
| 8193 | 4 | 1.7027 | 1.6490 / +3.16% | 1.6973 / +0.32% |
| 8193 | 8 | 1.9420 | 1.8853 / +2.92% | 1.9332 / +0.45% |

见[完整输入描述符、事件计时与数值核对](components.json)。
初版 paired harness 在请求结束后取最后一张图做局部计时；核对发现六格都已收缩到 C1。
该初版局部文件保留在外部实验目录 `results/paired/components.json`，
**不作为上表或 C4/C8 结论的依据**。本表均由后续 `components.py` 在匹配完整 batch 时补测，
各格都断言 descriptor.num_reqs 与请求并发相等。
端到端计时从 generate 开始到完成，不受这一局部计时问题影响。
补测 harness 最初只生成32token，长输入下部分请求在完整并发形成前结束，
并发断言中止了这次补测；原文件保留在 `results/components-short-output-attempt/`。
128token 的单 seed 尝试中，长输入 C8 也未形成全前缀完成的完整并发，
保留在 `results/components-single-seed-attempt/`。最终补测沿用端到端的128token和
相同七个seed范围，逐个seed等待匹配batch；确认六格描述符与prefix长度后才计入结果。

单独采集的 eager operator 次数确认 cos/sin 各由4次降为1次，
`aten::mm`由49次降为45次（其中其余投影和采样不变）。
GPU事件计时结束后，另采 eager CPU operator 次数验证减少的工作，
不把 profiler 时间用于性能结果：[operator-counts.json](operator-counts.json)。
局部图连续回放与完整引擎交替执行的缓存、时钟和提交状态不同，不能把局部降幅直接外推。
本轮数据没有进一步证明具体由哪个上游阶段遮蔽了局部收益。

## 正确性和检查

两项分别在本地 A6000 与 B200 跑 model/prefill/Shared-KV 测试，**各92 passed，零跳过**。
RoPE 新回归检查四层共享与逐层准备的 FP32/FP64 精确输出、长绝对位置和动态位置图回放；
MLP 新回归检查真实3072×9216形状下普通/assign重载的 checkpoint、缓存指针、
新输入回放和数值误差。原 cache、causality、SWA8192、分页/拒绝边界和 sampling 回归保留。
两项代码均通过完整 pre-commit 的适用检查。

两轮真实 checkpoint 审计在六格相同输入上比较 hidden、完整 FP32 proposal logits、
draft token 和 confidence 缓冲；两项对父版均逐位一致。
该小面板验证数值轨迹，**不是通用任务质量评测，也不验证其他并行环境**。

[validation.json](validation.json) 保存测试命令与结果；
[source-audit.json](source-audit.json) 核对每版本8046个源码文件及1996个补齐的运行时文件，
[source-receipt.json](source-receipt.json) 保存候选代码与清单哈希。
[结束检查](final-verification.json) 确认 Pod UID/节点/restart 保持、PID7存在、
GPU6释放至0MiB、利用率0%、锁可获取，并核对格式化harness的Python AST与执行版本一致。

## 复现与保留位置

[同引擎 harness](harness/paired.py)、[正确并发的局部 harness](harness/components.py)、
[启动脚本](harness/run.sh)、[局部启动脚本](harness/run-components.sh)、
[LLM 配置](harness/config.json)、[汇总脚本](harness/summarize.py)。
需要已安装的 vLLM CUDA/FlashAttention/Mega 运行时，不能只解包 tracked 源码启动。
仓库内 harness 副本仅补 SPDX、格式规范及退出 trap 变量的显式初始化；执行时的原始文件及其哈希保留在外部实验目录与审计回执。

本地完整原始材料：`/data/vllm_test/work/yoco-draft-projections-20261021/`。
B200不可变源码和原始输出：`/home/fhb/yoco-draft-projections-20261021/`。
开发基线维持 `fhb-dev-aligndeepinfer-startpoint@7b525571d0`。
