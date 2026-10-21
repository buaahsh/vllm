# DSpark / Shared-KV attention 复用结果

Archived pre-integration experiment. Relative source/data paths below refer to
`/data/vllm_test/work/yoco-dspark-attention-20261021/`.

> 更新：真实生成过程中的原生 draft 整步补测见 [NATIVE_DRAFT_PROFILE.md](NATIVE_DRAFT_PROFILE.md)。8K 下整步加速 1.23–1.33×；下方 1.26–1.36× 属于独立合成 KV 微基准。

已在独立源码树接入 FA4 paged attention。Shared-KV 本来已复用 DSpark/DFlash 的调度、顺序 Markov 采样和 graph 管理；本次新增的是严格 SWA8192 的 query-head 适配。实现见 `source/vllm/model_executor/models/yoco_shared_kv_swa.py`，三文件增量见 `attention.patch`，数学关系及依赖见 `IMPLEMENTATION.md`。

## 完整 target + draft

同一张 B200，target MoE 固定 current，原 step30000 draft，K=8。A1 → B1 → B2 → A2；每轮每格两次热身、三次正式测量。每格每后端共六个正式样本，所有样本均计入。每请求 256 输出 token，T=1，固定种子。长输入精确 8193 token，短输入 24–30 token；关闭 prefix caching。

吞吐包含 prefill、target、draft、调度与 GPU drain，不包含启动/编译。下表是固定面板的观察值；生成轨迹和接受率会随 BF16 reduction 舍入改变。

| 输入 | 并发 C | 原 Triton tok/s | FA4 tok/s | FA4/原 | 原接受率 | FA4 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短 | 1 | 407.7 | 392.9 | 0.964× | 48.57% | 46.68% |
| 短 | 4 | 834.7 | 874.2 | 1.047× | 37.26% | 38.47% |
| 短 | 8 | 1411.1 | 1382.9 | 0.980× | 36.72% | 37.10% |
| 8193 token | 1 | 145.0 | 149.5 | 1.031× | 14.54% | 13.87% |
| 8193 token | 4 | 296.1 | 319.9 | 1.080× | 15.32% | 17.02% |
| 8193 token | 8 | 401.9 | 408.3 | 1.016× | 16.07% | 15.58% |

接受率 = verifier 接受 draft token 数 / 提议 token 数，保留最后一轮的验证槽。长上下文有 1.6%–8.0% 的端到端吞吐提升，短上下文没有一致收益。没有按上下文或 batch 新增分流。

| 8193-token 输入 | 原每请求平均 decode 时间 | FA4 | 降幅 |
| --- | ---: | ---: | ---: |
| C=1 | 1.3574 s | 1.3075 s | 3.68% |
| C=4 | 2.2223 s | 1.9841 s | 10.72% |
| C=8 | 2.9927 s | 2.9168 s | 2.53% |

decode 时间来自原生 `request_decode_time_seconds` 的每请求均值，区别于整批 wall-clock 吞吐。

## 配对算子及真实四层 draft

固定真实 draft 权重、target embedding/head；合成 current hidden 和 KV。匹配原生 cache 的 8 GiB HND pool `[131072,8,16,256]`，不同请求使用不重叠随机物理页，避免小 cache 全部命中 L2 的偏差。CUDA graph 包含全部适配操作；完整 draft 还包含四层骨干、154880 词表投影和原 DSpark 顺序概率采样。每个函数每个样本图内重复 10 次，五轮正反顺序配对计时，每函数十个样本，表内为中位数。

| 历史 end | C | Attention 原→FA4 μs | 加速 | 完整 draft 原→FA4 μs | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 513 | 1 | 14.82 → 19.58 | 0.76× | 2234.50 → 2246.49 | 0.99× |
| 513 | 4 | 14.92 → 21.11 | 0.71× | 2376.80 → 2392.64 | 0.99× |
| 513 | 8 | 14.97 → 23.32 | 0.64× | 2501.37 → 2527.15 | 0.99× |
| 8193 | 1 | 184.33 → 27.05 | 6.81× | 3121.90 → 2288.77 | 1.36× |
| 8193 | 4 | 240.34 → 49.46 | 4.86× | 3277.69 → 2498.24 | 1.31× |
| 8193 | 8 | 242.24 → 71.41 | 3.39× | 3401.29 → 2709.85 | 1.26× |
| 17003 | 1 | 187.59 → 27.14 | 6.91× | 3120.65 → 2288.69 | 1.36× |
| 17003 | 4 | 240.19 → 49.20 | 4.88× | 3277.37 → 2497.63 | 1.31× |
| 17003 | 8 | 242.41 → 71.94 | 3.37× | 3401.36 → 2713.18 | 1.25× |

九个 case 全部有限。attention 最大相对 L2 0.003100；四层 hidden 最大相对 L2 0.006635；固定同一 Markov 前缀的 logits 最大相对 L2 0.005017，最大平均 KL 0.0002046。采样前缀一旦分叉，后续 bias/logits/confidence 的差异不再是同输入误差。

`draft-results-initial.json` 是早期 67 MiB NHD、小 pool 的探索值，保留追溯；正式算子数字使用最终 8 GiB HND pool 的 `draft-results.json`。FA2 的负面探索保存在 `probe.log`，未用于正式替换。

## 检查及环境

已有 Shared-KV suite：16 passed；改用 accelerator memory API 后，对 4 GiB 地址回绕项再次执行：1 passed。pre-commit Ruff、格式、SPDX、禁止 import 和 CUDA API 检查均通过。四个完整模型进程退出 0，共 72 次正式测量、48 次热身。同后端重复输出逐 token 一致 36/36。target/draft 最大 batch 均捕获并实际重放 FULL graph。每轮七条 Triton JIT 提示均落在数值检查或热身，正式 trial 2–4 没有 JIT 提示。

Pod `living-cobra-b200g4-dev-f5def941-master-0`，物理 GPU 7 / `GPU-4549436e-8247-c0d0-26a8-3b48a91fe43e`。GPU 4/5 既有服务未操作。Torch 2.13.0+cu130，BF16 Q/KV/attention，FP32 概率 logits；target 保留原 block128 FP8 MoE 配方。FA4 Python 组件按当前 vLLM 固定依赖补齐到本任务源码，A/B 两组相同。FA4 graph 等额外分配约 12.81 MiB。

精确配置、输入 token、逐次输出与指标、graph 重放计数在 `fullmodel-results/`。所有 target NLL 相关观察单独见 `QUALITY_BATCHING_NOTE.md`，不把未固定 batch 的 NLL 差当作质量变化。

结束审计：本任务全部模型/benchmark 进程已退出，GPU 7 显存恢复为 0 MiB；Pod UID、节点和 restart count 保持原值。GPU 4/5 既有 worker 未操作。GPU 6 后来被其他任务使用，本任务没有使用或终止它。代码及证据已同步回本地工作区。
