# 固定前缀 NLL：必须同时固定 target 的分批

Archived pre-integration experiment. Relative source/data paths below refer to
`/data/vllm_test/work/yoco-dspark-attention-20261021/`.

正式 A/B/B/A 的第一轮 target 采样，4 个请求的 token 和全部 top-20 logprob 逐位一致。但后续 219 个固定参考 token 的 prompt logprob 出现可复现差异：current 为 1.608368039466，FA4 为 1.593811981274。这不能直接解释为 draft attention 改变了 target 质量。

## 原后端就能精确复现两组数值

仅使用 **current target + 原 Triton draft**，关闭 engine multiprocessing，使同一次 `generate` 的请求先全部入队，再按指定组调用。相同 target、相同权重、相同输入，唯一控制变量为分批。`quality_batching.py` 及 `quality-current-batching/` 保存代码、实际 target 输入/hidden 和全部 logprob。

| 四条请求的入队分组 | 219 token 平均 NLL |
| --- | ---: |
| 1 + 3 | 1.608368039466 |
| 2 + 2 | 1.593811981274 |
| 3 + 1 | 1.616179864632 |
| 4 | 1.605085972212 |
| 1 + 1 + 1 + 1 | 1.589452255854 |

1+3 精确复现正式 current 组的 **全部 219 个 logprob**；2+2 精确复现正式 FA4 组的 **全部 219 个 logprob**，不只是平均值相同。这两次都没有执行新的 draft attention。原生异步入口逐条加请求，没有保证每次 forward 分到同一组；本实验也明确设置 `VLLM_BATCH_INVARIANT=0`，当前 target 的结果具有 batch 依赖。

## 相同 target batch 下直接核对 hidden

另对真实 current/FA4 两个引擎，在 target sampling 之前克隆每一步的 GPU input IDs、positions、query offsets 和 target hidden，结束后保存。不用任何日志 hash 代替数值比较。

两组实际批次均为：26、82、74、253、74、253 token。六个 batch 的 input IDs、positions、query offsets 和 **全部 target hidden 均逐位相同**。两次重复的固定前缀 logprob 也逐位相同，NLL 均为 1.608368039466。

证据在 `quality-current-graph/target-trace.pt`、`quality-flash-graph/target-trace.pt` 和对应 `logprobs.json`。这是相同条件下的 target 数值检查，不是综合任务质量评测。真实 draft 固定相同 Markov 前缀的误差和 KL 另见 `draft-results.json`；draft 采样轨迹因 BF16 reduction 舍入而分叉是允许的，继续使用原标准拒绝采样。

## 对报告的使用限制

完整生成吞吐及接受率仍是原生入口的有效观察，包含其实际分批与生成轨迹。原始 teacher NLL 保留在 `fullmodel-results/summary.json`，不能用其 A/B 差值判断新 attention 的质量。此前 Mega 报告中未固定 target batch 的 NLL 同样不能单独作为 FP8 分组配方优劣的证据。
