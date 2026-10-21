# Shared-KV Markov → 采样融合

基于 `7b525571d0`，在 `fhb-dev-aligndeepinfer-startpoint` 上优化完整词表的
Shared-KV 概率 draft 路径。每个槽的 Markov W2 BF16 输出直接进入现有采样 kernel，
在内部转换为 FP32、与 base logits 按原顺序相加，随后缓存 q、应用温度并进行 Gumbel 抽样。

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
  没有偏置的通用 Gumbel 调用保留原计算。

实现使用 AI 辅助。改动入口：
`vllm/v1/worker/gpu/sample/gumbel.py`、
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
| 1 | 0.2251 | 0.1838 | 18.35% |
| 4 | 0.3264 | 0.2584 | 20.85% |
| 8 | 0.4399 | 0.3195 | 27.37% |

同输入的完整q和样本精确相等。独立eager profiler确认每个K8链路
`aten::_to_copy`由16次降到8次，`aten::add`由8次降到0次；
保留的8次转换来自Markov GEMM输入。即消除8次bias转换及8次logits相加。
局部事件计时先完成，再单独采profile，不将profile开销混入延迟。
[局部完整结果](markov-benchmark.json)。

## 完整引擎端到端验收

| 输入 | C | 父版 tok/s | 融合 tok/s | 合并变化 | 两组对照变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| short | 1 | 335.36 | 333.12 | -0.67% | -0.79% / -0.55% |
| short | 4 | 895.84 | 892.36 | -0.39% | -0.31% / -0.46% |
| short | 8 | 1473.29 | 1469.41 | -0.26% | -0.16% / -0.36% |
| context8193 | 1 | 179.14 | 176.76 | -1.33% | -1.60% / -1.05% |
| context8193 | 4 | 243.46 | 239.56 | -1.60% | -2.23% / -0.96% |
| context8193 | 8 | 268.94 | 263.87 | -1.89% | -2.49% / -1.27% |

四进程共72个正式batch、312请求、39936输出token，另48个预热batch。
A1/B1与A2/B2共36组正式配对的输出和投机计数一致；
以A1为参考的其余三进程共90组核对（含预热）中，输出、accepted/proposed/round及
逐位置接受计数全部一致。四进程首token top-20/logprobs和256个teacher-forced位置一致。
实际图回放已核对，四进程均为target `[9,18,36,72]`、draft `[8,16,24,32,40,64]`，
保持 **target4份、draft6份**；没有把局部审计图装入引擎。

见[summary.json](summary.json)、[原始计时与输出哈希](trials.csv)、
[worker与图回执](worker-receipts.json)。

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

<!-- ENVIRONMENT -->
