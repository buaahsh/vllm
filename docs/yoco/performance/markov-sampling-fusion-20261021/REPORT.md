# Shared-KV Markov → 采样融合

基于 `fhb-dev-aligndeepinfer-startpoint@7b525571d0`，候选分支为
`fhb-dev-markov-sampling-fusion-20261021`，优化完整词表的
Shared-KV 概率 draft 路径。每个槽的 Markov W2 BF16 输出直接进入采样 kernel，
在内部转换为 FP32、与 base logits 按原顺序相加，随后缓存 q、应用温度并进行 Gumbel 抽样。
融合使用独立kernel，普通target的采样实现保持父版原样。

**采用决定：保留为独立候选，未并入开发基线。** 局部链路快18.5%–27.4%，
原生ABBA完整引擎吞吐变化为−0.57%～+0.19%，尚无端到端收益证据。
输出、投机计数和图规模保持已测合同，不把这些替代为性能收益。
见[采用决定](decision.json)。

## 实现和语义

- GEMM、base logits 生成和原始 checkpoint 沿用当前实现。base logits 的一次 FP32
  转换也保留；本次消除每槽的 Markov bias FP32 临时张量和相加后的 logits 临时张量。
- K8 保留八次按前一 token 计算 Markov bias 的顺序依赖；使用完整词表。
  原 seed、position、draft noise salt、温度策略和 FP64 noise 选项保持。
- q-cache 写入 FP32 相加结果，发生在温度与噪声之前。偏置融合接口拒绝 FP16/BF16
  proposal cache，避免提议和拒绝读到不同精度的 q。
- greedy/no-cache、watermark 和接受率估计器需要原路径时，继续物化完整 FP32 logits；
  confidence head 的 Markov embedding 收集与计算保持原顺序。
- 这些选择只取决于引擎配置和消费者，不按长度/并发增加策略分支；没有新增生产图管理器
  或持久缓冲区，继续核对 target 4 / draft 6 的图集合。
- 通用 DSpark 保留原数学路径，通过一个 Markov 采样 hook 接入 Shared-KV 融合。
  通用 Gumbel 源码、接口和 kernel 保持父版原样；专用kernel复用其cache/noise primitive。

实现使用 AI 辅助。改动入口：
`vllm/v1/worker/gpu/sample/gumbel_bias.py`、
`vllm/v1/worker/gpu/spec_decode/dspark/{speculator,shared_kv}.py`。

## 正确性验证

本地模型91、采样40、runner70项通过，4项SM90+用例由B200覆盖。
B200最终 **205 passed，零跳过**：模型/prefill/Shared-KV91、采样40、runner/attention/rejection74。
采样套件另有6项原有的大规模统计测试未选入；本次通过对原实现逐位比较样本和q，
验证改动路径而非用分布容差代替精确合同。

新用例覆盖 BF16/FP32 bias、完整154880词表、词表尾块、非连续stride、不同温度、
FP64 noise、64位position、标量/逐token cache列、请求重排、padding和graph回放。
Shared-KV回放额外与原物化路径比较tokens、q和confidence；拒绝回退与prefill原回归保留。
watermark/接受率估计器用例确认它们读取完整FP32相加结果。

## Markov 与采样局部计时

| C | 原路径 ms | 融合 ms | 延迟降低 |
| --- | ---: | ---: | ---: |
| 1 | 0.2252 | 0.1835 | 18.50% |
| 4 | 0.3264 | 0.2582 | 20.88% |
| 8 | 0.4399 | 0.3194 | 27.39% |

同输入的完整q和样本精确相等。独立eager profiler确认每个K8链路
`aten::_to_copy`由16次降到8次，`aten::add`由8次降到0次；
保留的8次转换来自Markov GEMM输入。即消除8次bias转换及8次logits相加。
局部事件计时先完成，再单独采profile，不将profile开销混入延迟。
[局部完整结果](markov-benchmark.json)。

## 完整引擎端到端验收

| 输入 | C | 父版 tok/s | 融合 tok/s | 合并变化 | 两组对照变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| short | 1 | 335.24 | 333.33 | -0.57% | -0.13% / -1.01% |
| short | 4 | 897.61 | 895.33 | -0.25% | -0.03% / -0.48% |
| short | 8 | 1475.60 | 1472.66 | -0.20% | -0.24% / -0.16% |
| context8193 | 1 | 178.77 | 178.51 | -0.14% | +0.14% / -0.43% |
| context8193 | 4 | 242.49 | 242.95 | +0.19% | +0.32% / +0.06% |
| context8193 | 8 | 268.43 | 266.98 | -0.54% | -0.54% / -0.54% |

四进程共72个正式batch、312请求、39936输出token，另48个预热batch。
A3/C1与A4/C2共36组正式配对的输出和投机计数一致；
以A3为参考的其余三进程共90组核对（含预热）中，输出、accepted/proposed/round及
逐位置接受计数全部一致。四进程首token top-20/logprobs和256个teacher-forced位置一致。
实际图回放已核对，四进程均为target `[9,18,36,72]`、draft `[8,16,24,32,40,64]`，
保持 **target4份、draft6份**；没有把局部审计图装入引擎。

见[summary.json](summary.json)、[原始计时与输出哈希](trials.csv)、
[worker与图回执](worker-receipts.json)。

## 初版回退与隔离验证

初版扩展了通用 Gumbel 接口，局部延迟降低18%–27%，但独立进程ABBA端到端回退
0.26%–1.89%，主要出现在TTFT。之后在同一引擎、相同target权重/KV/图内交替两种Markov
路径，六格吞吐变化为−0.09%～+0.16%，输出一致；没有重现相同幅度的回退。
这不能单独确定初版回退根因，也不能把局部加速当端到端提升。

最终候选将偏置采样放在独立`gumbel_bias.py`，复用原cache/noise primitive；
普通target的`gumbel.py`源码与父版逐字节一致。当前表格只采用此候选的重新测试。
[初版ABBA报告与原始回执](candidate1/REPORT.md)及
[同引擎交替对照](candidate1/paired-control.json)保留，未删除不利结果。

## 复现与边界

B200 运行复用已有隔离 venv，并通过 PYTHONPATH 分别选择父版与当前快照。
两份快照使用相同的预编译运行支持；源码、链接和运行支持文件均进行逐项哈希核对。
完整命令与回执见 [validation.json](validation.json)、[source-audit.json](source-audit.json)。

局部 benchmark 为 `benchmarks/kernels/benchmark_yoco_markov_sampling.py`，使用 synthetic
权重与预计算 base logits，覆盖真实词表 154880、Markov rank 256、K8、C1/C4/C8。
它测八次 Markov GEMM 和采样，不包含 draft backbone、LM head、target 或 prefill；
独立审计图只用于同输入局部对照，未加到生产引擎的图集合中。

端到端使用真实 target 与 step30000 draft、FP8 block128 target、Mega auto、V2、K8、
BF16 KV、fast-prefill，APC/异步调度关闭，固定 128 输出并忽略 EOS，温度 1，
standard rejection。输入为固定短问题及填充到 8193-token 的合成长上下文。
每进程每格两次预热、三个计分 seed；四个独立进程按父版→融合→融合→父版运行。
该固定面板不是任务准确率、HTTP 服务、自然 EOS、真实长会话或所有并发的收益保证。

原始日志、输出、脚本和完整哈希清单：
`/data/vllm_test/work/yoco-markov-sampling-fusion-20261021/`。

实验结束GPU6为0 MiB/0% utilization，锁可重新取得，Pod UID/节点/restart=0/PID7保持；
没有切换线上服务。开发基线保留在`7b525571d0`。
候选源码和本报告的工作树为`/data/vllm_test/work/yoco-markov-sampling-fusion-20261021/source-final`。
