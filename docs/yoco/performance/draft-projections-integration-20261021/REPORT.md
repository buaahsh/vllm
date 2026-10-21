# 采用共享 RoPE 和合并 gate/up：组合验收

2026-10-21。用户确认接受 **draft 侧收益**，本次将两项合入
`fhb-dev-aligndeepinfer-startpoint`。父版为 `7b525571d0`。
此前“独立候选暂缓采用”的决定由本次明确采用决定接替，历史实验数据保留。
不以已证明稳定端到端吞吐收益为本次采用依据，也不将两个独立降幅直接相加。

## 实现

- 四层共享一次 positions→频率/角度/cos/sin 准备，每次回放仍读取新位置，保留FP32/FP64语义。
- frozen MLP 将 BF16 gate/up 计算权重拼接后执行一次 GEMM，保留原FP32参数、checkpoint键名、
  SiLU/down和普通/assign重载的原地缓存更新。永久gate/up计算缓存字节数保持；初始化有临时FP32 concat。
- 保留K8、FP32 proposal cache、保原精度norm/residual、FA4/SWA8192、V2 fast-prefill/h20以及现有MoE YAML auto策略。
  生产仍target四张图 `[9,18,36,72]`、draft六张 `[8,16,24,32,40,64]`，没有新增按长度/并发的策略分支。

采用的独立代码提交为 `04088223dfb93d805caf17e4e8e81bc40aab1177` 与
`8d8bd197c7f4e87c7d4b5c294243967a13c953e9`。本次同时验证两项组合，没有带入 Markov→采样候选。

## 组合性能与精度

B200 TP1/PP1/DP1/no-CP，FP8 target、BF16 draft/KV，maxlen12288/maxseq8/budget2048/KV8GiB，
APC关闭、async关闭；短输入/8193 × C1/C4/C8，temperature1、固定输出128token、ignore_eos。
Target/draft checkpoint、MoE auto profile和提示面板均保持原配置。

| 输入 | C | 父版 draft ms | 组合版 draft ms | draft 延迟降低 | 端到端吞吐点估计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短输入 | 1 | 1.4466 | 1.3973 | 3.41% | +0.09% |
| 短输入 | 4 | 1.5807 | 1.5244 | 3.56% | -0.10% |
| 短输入 | 8 | 1.7245 | 1.6655 | 3.42% | -0.02% |
| 8193 | 1 | 1.4947 | 1.4464 | 3.23% | +0.12% |
| 8193 | 4 | 1.6989 | 1.6438 | 3.24% | +0.02% |
| 8193 | 8 | 1.9344 | 1.8747 | 3.08% | -0.42% |

局部事件计时覆盖完整draft backbone、LM head、Markov与采样；同一真实checkpoint/KV/输入下，
独立审计图池交替6轮ABBA/BAAB、每次50回放、3次预热，取每路径12次计时中位数。
图描述符与完整batch匹配后才计时，六格hidden、完整FP32 proposal logits、draft token和confidence缓冲逐位一致。

端到端点估计来自同引擎固定target权重/KV/target图的18组正式配对（每格3个seed），另12组预热配对；
36个正式batch、156请求、19968输出token。两版全部正式及预热输出、accepted/proposed/round和逐位置计数一致。
GPU事件与同输入检查均在计时循环之前的独立audit warmup执行，不计入上述正式吞吐。
额外baseline图池和计算缓存只存在于审计进程。

这个小面板只报告点估计，不宣称稳定整体吞吐改善。原独立候选的完整draft延迟降低分别为
RoPE 2.92%～4.42%、gate/up 0.21%～0.66%；其同引擎端到端变化分别为−0.21%～+0.38%、
−0.07%～+0.69%。两项独立结果见历史候选分支：
[RoPE报告](https://github.com/buaahsh/vllm/blob/8ee5bf5606d89e93fed22f6fe2aa4633442bc739/docs/yoco/performance/draft-projections-20261021/REPORT.md)、
[gate/up报告](https://github.com/buaahsh/vllm/blob/ba42e65ef79a96c0a3a5995e8fc7e4384b51d591/docs/yoco/performance/draft-projections-20261021/REPORT.md)。

## 原生路径与回归

组合版还通过标准生产代码加载、捕获和运行的独立native进程验证：18个正式batch、78请求、9984输出token，
另12个预热batch。30条轨迹均精确复现归档父版7b的输出与投机计数，
首token top20/logprobs及256个teacher-forced位置逐位一致。归档父版仅作为数值控制，未作为本轮时间对照。
native实际target/draft回放均已核对，生产图集合保持4/6。

本地A6000 **95 passed**；B200 **169 passed，零跳过**：模型/prefill/Shared-KV95、runner/attention/rejection/sampler74。
包含两项新用例与原有动态位置/页表/SWA/因果性、checkpoint/cache重载、拒绝边界和fast-prefill回归。
实际执行命令见[启动脚本](harness/run.sh)，[汇总](summary.json)、[逐格CSV](summary.csv)、
[worker和图回执](worker-receipts.json)、[源码及运行时审计](source-audit.json)。
[结束检查](final-verification.json)确认Pod UID/节点/restart保持、PID7存在、GPU6释放至0MiB/0%、锁可获取。
模型影响仅在此面板做数值轨迹验证，不将其当作通用任务质量评测或其他并行配置的验收。

## 复现和材料

[同引擎与局部harness](harness/paired_merge.py)、[父版模型](harness/baseline_model.py)、
[LLM配置](harness/config.json)、[汇总脚本](harness/summarize.py)。
需对应的CUDA/FlashAttention/Mega运行时和checkpoint；harness脚本按实验根目录布局使用。

完整原始材料：`/data/vllm_test/work/yoco-draft-projections-merge-20261021/`；
B200不可变源码和输出：`/home/fhb/yoco-draft-projections-merge-20261021/`。
源码改动位于 `vllm/model_executor/models/yoco_shared_kv.py`，测试位于
`tests/model_executor/test_yoco_shared_kv.py`。
