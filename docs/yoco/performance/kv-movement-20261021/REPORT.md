# YOCO KV movement: Mooncake planning and fast-prefill gather

2026-10-21。基于 `5da665ab51215b2204b03d309b26e2506e0b9670`，同时检查
P/D 引擎间传输与单卡 self/cross 边界搬运。本轮采用两项保持数据语义的改动：
Mooncake 在请求内复用 KV group 的连续页分组；fast-prefill 用一个 Triton kernel
将 positions、hidden、K、V 直接 gather 到已有稳定缓冲区。

**真实模型 A/B/B/A 的六个场景中，输出吞吐观测提升 0.76%–1.31%，TTFT 中位数
下降 0.23%–2.33%。收益较小，部分场景接近运行波动。Mooncake 的测量仅涉及 CPU
描述符构造，没有测真实 RDMA/P-D 端到端收益。当前 K8 也不能直接使用这两条 V1
组合。**

## 与 DeepInfer 的对齐结论

对照固定在 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`：

- [README](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md)
  明确完整 KV 系统、VE/P/D 服务与 DP 负载均衡不包含在该开源版本中。
- [部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L157)
  描述异步获取存储 KV、逐层 H2D 与计算重叠，以及用 GDR 处理细粒度 swap-in。
  这是可借鉴的系统设计，不能据此声称已移植或对比了他们完整的传输服务。
- [YOCO boundary](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/yoco.py)
  使用 compact consumer 数据流；其模型、CP 通信和 residual 合同与本分支不同。

本分支已经具备共享 KV alias 去重、存储注册去重、Mooncake 批量 RDMA 和连续页
coalescing。`gpu_model_runner.get_kv_cache_spec()` 跳过共享消费者的 cache spec；
Mooncake `register_kv_caches()` 按 backing storage 注册一次。
因此本轮没有把“去掉重复传输全部共享层”当成新增收益。

Mooncake 的 `wait_for_layer_load()` / `save_kv_layer()` 当前为空操作；已有异步
请求传输并不等同于逐层可用与计算重叠。逐层流水需要重新定义 readiness、buffer
生存期、失败取消和 YOCO KV 源层依赖，不能只替换一个 copy 调用。

NIXL 已有 NumPy 向量化描述符与 prepared transfer handle：见
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` 的
`_build_fa_local()` 和 `register_local_xfer_handler()`。本轮未发现与 Mooncake
相同的逐 region 重复分组热点，因此 NIXL 保持现状，待真实传输时间线定位瓶颈。

## 实现与适用边界

### Mooncake

`_build_transfer_params()` 原来对同一 KV group 的每层/head region 重复扫描相同页表。
现在在完成 prefix trimming 和 logical→kernel block 展开后，按 group 保存分组结果，
只在该 region 确实需要传输时计算。缓存只活在当前请求内。

传输源/目标地址、长度、描述符顺序、协议、异构 TP/HMA 处理和 YOCO 尾块重算规则
均保持原样；不增加跨请求状态或 GPU Graph。

### 单卡 fast-prefill

`yoco_ops/prefill.py` 将原来四次高级索引及 copy 合并，直接写入现有
`fp_positions` / `fp_hidden_states` / `fp_yoco_key` / `fp_yoco_value`。
同宽整数视图保持 FP32/BF16/FP8 的原始位模式，包括 FP8 NaN；支持非连续 stride、
重复 padding、合法负索引和空选择。调用方继续保证索引有效、源目标不重叠。

原有 hidden clone、scatter、最终 norm、KV 写入依赖和 DP metadata 恢复均保留。
CUDA 直接使用融合实现，CPU 保留 Torch 路径；没有增加按 batch/长度选择后端的
阈值、YAML 配置或额外 CUDA Graph 管理器。

纯 P 的 `kv_only_prefill` 原来就会在 gather 前返回，因此这个 gather 改动只加速
实际经过边界的 standalone / 适用的 D-tail 路径。当前兼容性仍是：

- YOCO KV transfer 接入保留的 V1 runner；DSpark Shared-KV K8 要求 V2。
- auxiliary hidden 导出与 fast-prefill 组合仍被 `yoco.py` 明确拒绝。
- V1 fast-prefill 仍拒绝 prompt logprobs；保留这项合同。

本轮真实模型使用 **V1、无投机、Mega 关闭**，不能外推为已合入的 Mega+FA4 K8
组合收益。K8 target/draft 图规格的配置代码未改。

## CPU 描述符规划结果

36 个合成场景：context 512/8192/32768 × region 1/11/88 × 请求 1/8 ×
连续/碎片页。每场景 3 轮 ABBA，每个样本 3 次调用；使用同模块依赖执行冻结的
原始方法与当前 native 方法，36/36 的完整描述符列表相同。

以下为 8K context、一个请求的中位数；region 数是合成布局，不能直接视作层数：

| regions | 页布局 | 原实现 µs | 新实现 µs | 加速比 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 连续 | 138.44 | 136.61 | 1.01× |
| 1 | 碎片 | 1496.00 | 1496.81 | 1.00× |
| 11 | 连续 | 489.09 | 191.74 | 2.55× |
| 11 | 碎片 | 2722.77 | 1758.99 | 1.55× |
| 88 | 连续 | 4125.68 | 331.87 | 12.43× |
| 88 | 碎片 | 21272.62 | 3590.10 | 5.93× |

复用收益随同 group 的重复 region 数增大。单 region 基本不变。**这些数字不是
RDMA 带宽或请求延迟提升**。全部样本在 [planner-benchmark.json](planner-benchmark.json)。
驱动脚本和冻结原方法在原始目录 `pd-planner/`，哈希见
[artifact-hashes.json](artifact-hashes.json)。

## B200 gather 微基准

24 个场景：源行 1K/8K/32K × 选择行 1/4/32/256 × BF16/FP8 KV；
hidden=3072 FP32、KV width=1024，3 轮 ABBA、每个样本 50 次调用。
24/24 逐字节一致，Graph 设备耗时下降 **78.07%–89.28%**，包含 Python 的
eager 墙钟耗时下降 **40.27%–44.88%**。

例如 8K 源行、选 1 行、BF16 KV：设备时间 17.82→1.91 µs，eager
49.72→29.04 µs。这些只覆盖 gather，不包含 attention、clone/scatter/norm 或 TTFT。
原始样本见 [gather-benchmark.json](gather-benchmark.json)。

## 原生真实模型 A/B/B/A

单 B200 GPU6，TP1/PP1/DP1/EP1；同一 L3 checkpoint
`balanced-b040-c035-d025-hf`，online FP8 target、BF16 KV、APC 关闭、
fast-prefill 开启、异步调度关闭、max seqs=4、batch token budget=4096、KV=8 GiB。
使用原生模型，worker extension 只观察状态和 profiler，不替换模型/专家 forward。
四组均确认使用 V1 runner、split blocks 和 fast-prefill；融合 kernel 只出现在 B 组。

依次运行 baseline-a1 / fused-b1 / fused-b2 / baseline-a2。每组 6 个场景，
每场景 2 次热身、3 次正式；共 **72 次正式测量、180 个请求、11520 个输出 token**，
另有 48 次热身。每请求固定生成 64 token、greedy、忽略 EOS。A/B 都使用相同
运行时；baseline 仅恢复两个原生产文件，新增 helper 存在但未被调用。

计时包含 prefill、decode、采样、调度及 GPU drain，排除加载、编译、tokenization
和 HTTP。吞吐为各单元的总输出 token / 总耗时；TTFT 为该单元各请求的中位数，
因此 C4 长 prompt 的排队/调度也包括在内。每个后端每单元只有 6 次正式 batch。

| 输入 token | C | 原 tok/s | 新 tok/s | 吞吐变化 | 原 TTFT ms | 新 TTFT ms | TTFT 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 84.02 | 84.76 | +0.88% | 213.33 | 208.79 | −2.13% |
| 1024 | 4 | 286.80 | 289.02 | +0.77% | 216.64 | 212.49 | −1.91% |
| 8192 | 1 | 72.47 | 73.38 | +1.25% | 325.58 | 317.99 | −2.33% |
| 8192 | 4 | 148.32 | 150.26 | +1.31% | 675.55 | 674.00 | −0.23% |
| 32768 | 1 | 40.53 | 40.84 | +0.76% | 993.80 | 974.55 | −1.94% |
| 32768 | 4 | 55.54 | 55.97 | +0.78% | 2455.02 | 2412.57 | −1.73% |

端到端变化远小于 kernel 变化，而且存在漂移：8K/C4 的两组相邻 A/B 分别
+2.62% / +0.02%；32K/C1 分别 +1.93% / −0.40%。32K/C1 的新实现 TTFT
p95 为 1032.51 ms，原实现 1029.52 ms，尾部没有全面改善。
节点上的其他 GPU 有并行任务，整机没有独占；本轮只使用 GPU6，并用同一 advisory
lock 排他运行。上述结果是此面板观测值，不作为稳定提升或延迟保证。

72/72 正式 batch（连同 48 次热身）均与同输入的基线输出完全一致，无损坏、抢占或
输出 token 数不足。额外 1K/8K 两个输入各生成 8 token、返回 top-5 generated-token
logprobs，四组记录逐值相同。这是数值回归，不是广泛任务质量评估。
初次 pilot 请求了不支持的 prompt logprobs，明确报错后移除该检查；pilot 不计入
正式结果，也未放宽引擎合同。

明细：[model-summary.csv](model-summary.csv)、[model-trials.csv](model-trials.csv)、
[model-request-metrics.csv](model-request-metrics.csv)、
[quality-observations.json](quality-observations.json)、[model-config.json](model-config.json)。
原始 JSON 保留完整 output IDs 和 request metrics，归档哈希见 artifact 索引。

## 验证与可追溯性

- Mooncake connector / HMA / YOCO-tail：原基线 102 项通过，新实现 106 项通过。
  新增测试以 CPU 模拟实际字节复制，覆盖请求隔离、两个 group、多个 region、
  prefix trimming、紧凑/跨 stride 页和 logical block 展开。
- 本机 prefill 58 项通过；B200 同一 suite 58 项通过，无跳过。覆盖任意位模式、
  FP8、非连续 stride、Graph 更新输入、稳定输出地址、padding/clone 与异常时 DP 恢复。
- 正式计时窗口无 JIT 告警；日志中的 JIT 均在 profiler/质量调用或热身阶段。
  既有 MoE 默认配置、OMP4 及进程退出 NCCL 提示均保留在原始日志中。
- scoped pre-commit 通过。贡献由 Codex 辅助完成，本轮未创建上游 PR。

复现入口（使用已安装依赖的隔离 venv）：

```bash
.venv/bin/python -m pytest -q tests/model_executor/test_yoco_prefill.py
.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/test_mooncake_connector.py tests/v1/kv_connector/unit/test_mooncake_connector_hma.py tests/v1/kv_connector/unit/test_mooncake_yoco_tail.py
.venv/bin/python benchmarks/kernels/benchmark_yoco_prefill_gather.py --hidden-width 3072 --selected 1 4 32 256 --output gather.json
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python -m tools.yoco_alignment.benchmark_kv_prefill --variant fused --output fused
```

实际远端完整环境与 ABBA 控制脚本保存在原始目录；直接执行上述工具前需要设置
适合目标机器的 checkpoint、CUDA/runtime 和 GPU 选择。

原始目录：`/data/vllm_test/work/yoco-kv-movement-20261021/`；
远端：`/home/fhb/yoco-kv-movement-20261021/`。
[source-receipt.json](source-receipt.json) 记录被测生产源码哈希；
[final-audit.json](final-audit.json) 记录两套源码及 native/support 全量校验。
候选 7988 个源码文件、基线除两个覆盖文件外 7986 个、两边各 1996 个运行时支持
文件均通过，基线覆盖文件与父提交逐字节相同。最终仅微基准 docstring 去掉过时的
“opt-in”表述，生产代码和测试与被测快照一致。

实验结束 GPU6 回到 0 MiB，锁已释放；Pod UID、节点和 restart count=0 均保持。
未重建 Job/Pod 或重启共享服务。

## 后续值得推进的工作

1. **接通当前 K8：** 先补 V2 KV transfer 与 auxiliary hidden/compact fast-prefill
   的完整合同，再以同 K8 runner 做 A/B。纯搬运微优化不能绕过这些前置条件。
2. **真实 P/D 测量：** 按 cold/local-hit/remote-hit 和整页边界±1，分别记录
   prepare 时间、有效 bytes、descriptor count、RDMA 完成与 D-ready/TTFT；
   比较 Mooncake/NIXL，先区分元数据、网络和等待耗时。
3. **逐层重叠：** 有时间线证据后，再设计 readiness/event、失败回收和 source
   KV 生命周期。DeepInfer 的分层 H2D/GDR 方案需要结合我们的拓扑验证。
4. **compact 返回：** 本轮保留的 full hidden clone/scatter/norm 仍可能是更大的
   边界成本；需要同步 logits/auxiliary 合同，避免改坏 mixed batch 和 DP。
