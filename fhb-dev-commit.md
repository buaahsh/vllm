# `fhb-dev` 逐提交工程记录

本文是 `fhb-dev` 的逐提交工程日志，用于在不依赖 Pull Request 邮件和网页上下文
的情况下，回答以下问题：

- 这个 commit 为什么需要存在；
- 它解决了什么具体问题，哪些问题明确不在范围内；
- 修改了哪些文件，以及每个文件承担什么职责；
- 正确性如何证明，测试覆盖了哪些输入和部署形态；
- 性能指标如何采集，收益能否归因到该 commit；
- 启用条件、已知限制、风险和回滚方式是什么。

更完整的 YOCO 模型、镜像和端到端验收资料仍保存在 `yoco.md`。本文以 commit
为主线，避免不同功能、不同基线或不同测试轮次的结果被混在一起。

## 维护规则

1. `fhb-dev` 以
   `origin/shaohanh/yoco-serving-final-20260730@c27db1e189973cea3164ba66b1d00359d4122088`
   为增量开发起点。
2. 一个功能点对应一个可独立理解和回滚的功能 commit；不把无关功能捆绑到同一
   commit。
3. 每个功能 commit 都必须在本文增加或更新一条记录，至少包括目的、改动文件、
   行为边界、测试和性能。没有性能收益的正确性修复也必须明确写出“为什么预期
   性能为零”，不能借用其他 commit 的指标。
4. 性能数字必须注明硬件、baseline、candidate、输入 shape、并发、warmup 和
   统计口径。组合优化的数据不能冒充单 commit 收益。
5. GitHub 操作使用 `Snow2022jlu`；Git author/committer 使用
   `方涵斌 <2190556589@qq.com>`。AI 辅助通过 commit trailer 明示。
6. 默认将经过本地检查和 GPU 验证的 commit 直接同步到 `fhb-dev`，避免每个小功能
   都创建 PR 触发邮件；确实需要跨团队网页审阅时再单独创建 PR。
7. 从基线到审计水位的每一个可达 commit 都必须出现在“完整 Git 提交审计”中。
   功能、正确性、性能、构建、测试和设计结果展开为正文；只补充已有记录的
   docs-only commit 只进入完整索引，不递归增加一篇重复正文。
8. Git commit 无法在自身内容中预先写入自己的 hash，因此每次日志维护提交审计到
   它的 parent，并在下一次提交中回填真实 hash。当前水位为 `fb8c42b6a0`：相对基线
   共 `45/45` 个可达提交，其中主线 `38/38` 个、合并支线 `7/7` 个，均已列出。

## 功能与工程结果索引

| 序号 | Commit | 类型 | 功能 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | `85eab7b56e` | 性能/PD | KV-only producer 跳过 cross layers | 已进入 `fhb-dev` |
| 2 | `ea4f80d1b4` | 正确性/NIXL | 去重 KV-sharing alias 注册 | 已进入 `fhb-dev` |
| 3 | `d11cc022bd` | 性能/Router | 删除 dense routing materialization | 已进入 `fhb-dev` |
| 4 | `ef60ee0255` | 性能/Norm | 融合 residual add 与 RMSNorm | 已进入 `fhb-dev` |
| 5 | `9988ae737f` | 性能/MoE | Shared Expert 与 Routed MoE 并行 | 已进入 `fhb-dev` |
| 6 | `8eae22948c` | 性能/Activation | FP32 clamped-SwiGLU 单 kernel | 已进入 `fhb-dev` |
| 7 | `1c51cac0d7` | 正确性/PD | Streaming stop 保留 KV metadata | 已进入 `fhb-dev` |
| 8 | `2765c22a1b` | 构建/PD | UCX 1.21 单 runtime 与 SM100 `_C` | 已进入 `fhb-dev` |
| 9 | `7c03e0cb73` | 性能/MoE | B200 YOCO Triton MoE 配置 | 已进入 `fhb-dev` |
| 10 | `8abfd6c6d0` | 性能/RoPE | YOCO BF16 rotary 单 kernel | 已进入 `fhb-dev` |
| 11 | `26614af40f` | 文档/负结果 | 放弃 differential-attention CustomOp | 已进入 `fhb-dev` |
| 12 | `9106abeb3a` | 合并/DP8 | 合入 YOCO B200 multigpu long-context | 已进入 `fhb-dev` |
| 13 | `4364a96501` | 性能/Router | 缓存 FP32 Router 归一化权重 | 已进入 `fhb-dev` |
| 14 | `81df1f21e8` | 正确性/DeepEP | NVSHMEM ABI、RDMA 与 IBGDA 启动保护 | 已进入 `fhb-dev` |
| 15 | `03a0479b67` | 正确性/PD | 对齐 standalone、local cache 与 NIXL PD shape | 已进入 `fhb-dev` |
| 16 | `fa8e4eac6a` | 文档/PD | 当前 PD 部署策略独立报告 | 已进入 `fhb-dev` |
| 17 | `f54af87aec` | 测试/PD | TP2/DP1 batch 容量、吞吐和 forward 曲线 | 已进入 `fhb-dev` |
| 18 | `75d26710b9` | 设计/Cache | LMCache 三级缓存与 Dynamo 适配方案 | 已进入 `fhb-dev` |
| 19 | `e081d38f5a` | 构建/Cache | 固定 LMCache 0.5.3 的 CUDA 13/SM100 runtime | 已进入 `fhb-dev` |
| 20 | `5d296b3958` | 正确性/Cache | LMCache 适配 YOCO 31 份物理 KV | 已进入 `fhb-dev` |
| 21 | `aebb50c6e5` | 正确性/PD | pure-P prefix shape 与 SWA window 传输补充 | 已进入 `fhb-dev` |
| 22 | `e5524539f1` | 测试/PD | PD 极限吞吐与 W1/W2/W3 HMA A/B | 已进入 `fhb-dev` |
| 23 | `本提交` | 性能/Attention | 融合 Q/K RMSClip 与 RoPE | 待提交 |

## 完整 Git 提交审计

审计区间为
`c27db1e189973cea3164ba66b1d00359d4122088..fb8c42b6a0`。
基线本身不计入增量提交数。以下两张表覆盖这个区间内全部 `45` 个可达提交，避免把
docs-only、验证补充或 merge 引入的支线提交藏在正文之外。

### First-parent 主线：38/38

| 序号 | Commit | Subject | 对应记录 |
| ---: | --- | --- | --- |
| 1 | `85eab7b56e` | `perf(yoco): skip cross layers for KV-only producer prefill (#3)` | 第 1 节 |
| 2 | `ea4f80d1b4` | `fix(yoco): deduplicate NIXL KV-sharing aliases (#4)` | 第 2 节 |
| 3 | `d11cc022bd` | `perf(yoco): remove redundant router materialization` | 第 3 节 |
| 4 | `d77fed9d2a` | `docs(yoco): add fhb-dev commit ledger` | 初始化并补记第 1--3 节 |
| 5 | `ef60ee0255` | `perf(yoco): fuse residual add with RMSNorm` | 第 4 节 |
| 6 | `cd1e22c7ff` | `docs(yoco): record fused add-RMSNorm commit` | 补记第 4 节 |
| 7 | `9988ae737f` | `perf(yoco): overlap shared and routed experts` | 第 5 节 |
| 8 | `2e1b8ecc53` | `docs(yoco): record shared expert overlap` | 补记第 5 节 |
| 9 | `8eae22948c` | `perf(yoco): fuse FP32 clamped SwiGLU` | 第 6 节 |
| 10 | `4b0f7d3d42` | `docs(yoco): record FP32 clamped SwiGLU fusion` | 补记第 6 节 |
| 11 | `1c51cac0d7` | `fix(pd): preserve KV metadata on streamed stops` | 第 7 节 |
| 12 | `0f8c9d95b5` | `docs(pd): record streamed-stop KV metadata fix` | 补记第 7 节 |
| 13 | `2765c22a1b` | `build(pd): unify runtime on UCX 1.21` | 第 8 节 |
| 14 | `66a87747fa` | `docs(pd): record UCX 1.21 runtime validation` | 补记第 8 节 |
| 15 | `7c03e0cb73` | `perf(moe): add tuned B200 YOCO config` | 第 9 节 |
| 16 | `84906823de` | `docs(yoco): record B200 MoE tuning results` | 补记第 9 节 |
| 17 | `8abfd6c6d0` | `perf(yoco): fuse BF16 rotary embedding` | 第 10 节 |
| 18 | `625ff8ea06` | `docs(yoco): record BF16 rotary fusion` | 补记第 10 节 |
| 19 | `26614af40f` | `docs(yoco): record rejected diff-attention fusion` | 第 11 节 |
| 20 | `9106abeb3a` | `Merge YOCO B200 multigpu long-context support` | 第 12 节 |
| 21 | `fd6baffea2` | `docs(yoco): record multigpu merge validation` | 第 12 节补充验证 |
| 22 | `0223cd9099` | `docs(yoco): record DP1 and DP4 validation` | 第 12 节补充验证 |
| 23 | `4364a96501` | `perf(yoco): cache normalized router weights` | 第 13 节 |
| 24 | `a9cd5c2072` | `docs(yoco): record Router weight cache results` | 补记第 13 节 |
| 25 | `81df1f21e8` | `fix(yoco): guard DeepEP NVSHMEM runtime` | 第 14 节 |
| 26 | `4a39087d27` | `docs(yoco): record DeepEP NVSHMEM validation` | 补记第 14 节 |
| 27 | `03a0479b67` | `fix(yoco): align fast-prefill shapes across PD` | 第 15 节 |
| 28 | `b63e04909b` | `docs(yoco): record PD shape consistency fix` | 补记第 15 节 |
| 29 | `fa8e4eac6a` | `docs(yoco): add standalone PD strategy report` | 第 16 节 |
| 30 | `f54af87aec` | `docs(yoco): record PD batch capacity and throughput` | 第 17 节 |
| 31 | `75d26710b9` | `docs(yoco): plan LMCache and Dynamo adaptation` | 第 18 节 |
| 32 | `e081d38f5a` | `build(yoco): add pinned LMCache CUDA 13 runtime` | 第 19 节 |
| 33 | `5d296b3958` | `fix(yoco): adapt LMCache to physical KV layout` | 第 20 节 |
| 34 | `53f58e5426` | `docs(yoco): complete fhb-dev commit audit` | 审计到第 20 节 |
| 35 | `aebb50c6e5` | `fix(yoco): preserve pure-P prefix shape with HMA` | 第 21 节 |
| 36 | `9b9a945c06` | `docs(yoco): record HMA and LMCache PD validation` | 补记第 21 节及 LMCache 验证 |
| 37 | `e5524539f1` | `docs(yoco): record PD saturation and HMA workload results` | 第 22 节 |
| 38 | `fb8c42b6a0` | `docs(yoco): audit PD saturation and HMA benchmarks` | 审计到第 22 节 |

### Merge 引入支线：7/7

以下提交由 `9106abeb3a` 引入，都是第 12 节 multigpu long-context 合并的一部分。

| 序号 | Commit | Subject |
| ---: | --- | --- |
| 1 | `6c0b7a35ee` | `perf: tune YOCO B200 long-context serving` |
| 2 | `4a6f4400d4` | `Improve YOCO long-context serving tooling` |
| 3 | `2304668e15` | `Correct YOCO per-engine sequence limits` |
| 4 | `e3974c1c97` | `Document YOCO multigpu long-context results` |
| 5 | `bbd96cacb0` | `Rebuild YOCO B200 presentation probes` |
| 6 | `34f4044acb` | `docs: show direct vllm long-context launches` |
| 7 | `85f7d2ac1b` | `docs: add max-throughput rates to summary` |

---

## 1. KV-only producer 跳过全部 cross layers

### 提交信息

```text
commit: 85eab7b56ec4cb1208364bb0daa74d04f09f440e
subject: perf(yoco): skip cross layers for KV-only producer prefill (#3)
author: 方涵斌 <2190556589@qq.com>
baseline: c27db1e189973cea3164ba66b1d00359d4122088
source PR: https://github.com/buaahsh/vllm/pull/3
diff: 7 files, 503 insertions, 82 deletions
```

### 目的

YOCO 的后十层是共享一份 K/V 的 cross-attention layers。原始 fast-prefill 已经
让后九层只处理需要 logits 的 compact tokens，但第一层同时是共享 KV cache 的
owner，仍然对完整 prompt 执行 attention 和 MoE。

在 P/D 分离的纯 Prefill 节点上，P 的职责只是生成并传输 KV。P 为满足 vLLM 请求
生命周期而采样的 token 会被 Gateway 丢弃，用户可见的 first token 和后续 token
全部由 Decode 节点生成。因此，P 没有必要为 disposable logits 执行十个 cross
layers。本提交的目标是：

1. 保留 self layers 和共享 K/V 投影，保证传给 D 的 KV 完整；
2. 把共享 K/V 直接写入第一个 cross layer 所拥有的 cache；
3. 普通 fast prefill 让十个 cross layers 都只处理 compact logits tokens；
4. 纯 `kv_producer` 请求完全跳过十个 cross layers；
5. DP>1 时保证 active rank 和 idle rank 的 MoE collective 顺序一致，不产生 hang。

### 非目标

- 不改变 Decode 节点的正常 decode 数值路径；
- 不让 P 生成用户可见 token；
- 不支持在专用 `kv_producer` 服务中混入普通非 PD 请求；
- 不改变 NIXL/UCX 的传输协议和内存注册策略；
- 不把 `kv_both` 的动态请求判定无条件扩展到 DP>1。

### 修改文件及职责

#### `vllm/model_executor/layers/attention/attention.py`

- 给 Attention forward 增加 `kv_cache_dummy_dep`，把显式 cache write 作为后续
  attention 的依赖传入，避免编译器或 CUDA Graph 破坏写入与读取顺序。
- 增加 `skip_kv_cache_update`。当 YOCO self block 已经把共享 K/V 写入 owner
  cache 后，第一个 cross layer 可以复用该 cache，不再重复写入。
- direct-call 和 custom-op 两条 attention 路径都遵守同一开关，避免 eager、
  compile 和 CUDA Graph 行为不一致。

#### `vllm/model_executor/models/yoco.py`

- `YOCOSelfBlock` 保留完整 prompt 上的 universal-loop self layers 和共享 K/V
  projection，但不再执行第一个 cross layer。
- 将 `yoco_key`/`yoco_value` reshape 成 owner attention 需要的 layout，通过
  `torch.ops.vllm.unified_kv_cache_update` 直接写入共享 KV cache。
- `YOCOCrossBlock` 从“仅后九层”扩大为“全部十层”；第一层携带 cache-write
  dependency 并设置 `skip_kv_cache_update=True`，所有层都在 compact tokens 上
  运行。
- 给模型 forward 增加 `kv_only_prefill`。为 true 时，共享 KV 写完后直接返回
  self-decoder state 的 norm 结果作为 disposable logits 输入，十层 cross
  attention/MoE 全部跳过。
- 保持普通 fast-prefill 和非 fast-prefill fallback；优化未命中时仍走原有完整
  计算路径。

#### `vllm/v1/worker/gpu_model_runner.py`

- 增加 `is_yoco_dedicated_kv_producer()` 静态角色判断。条件为 YOCO、启用
  `kv_sharing_fast_prefill`、存在 KV transfer config 且角色为 `kv_producer`。
- 专用 `kv_producer` 对 DP1、DP2、DP4、DP8 等任意 DP 数直接启用 KV-only，
  不要求“只支持 DP1”。
- `kv_both` 为兼容旧部署，只在 DP1 下逐请求验证：`max_tokens=1`、尚未生成
  output token、且 `kv_transfer_params.do_remote_decode=true`。
- DP>1 的 `kv_both` 保守回退，因为各 rank 无法仅凭本地 batch 证明整个服务都
  是纯 P 请求。
- 极短 P 请求如果落入普通 FULL graph 会重新包含 cross layers，因此对不超过
  8 tokens 的 KV-only batch 强制 eager，保持裁剪语义正确。
- 将第一个 cross layer owner 加入 fast-prefill eligible layers，使十层共享同一
  compact metadata。

#### `vllm/v1/worker/gpu_worker.py`

- DP>1 的 idle rank 仍需执行 runtime dummy forward，以参加 active rank 发起的
  MoE collectives。
- 当服务是专用 YOCO producer 时，dummy forward 也设置
  `yoco_kv_only_prefill=True`。否则 active rank 已跳过 cross-layer collectives，
  idle rank 却仍进入它们，会产生 collective 数量不一致和 hang。

#### 测试与文档

- `tests/model_executor/test_yoco_conversion.py`：覆盖普通 fast prefill 的十层
  compact-token 执行、KV-only 的十层全跳过，以及显式 KV cache write 的调用
  关系。
- `tests/v1/worker/test_gpu_model_runner.py`：覆盖纯 P batch、专用 producer 静态
  判定、`kv_both` 回退和 idle-rank dummy 行为。
- `yoco.md`：记录 P/D token 归属、部署契约、B200 正确性和 A/B 性能。

### 行为契约

该优化成立的前提不是“P 的 token 最后不展示”，而是 P 的整个 `choices` 必须被
丢弃：

1. Gateway 向 P 发送原始 prompt，设置 `max_tokens=1`，并设置
   `kv_transfer_params.do_remote_decode=true`；
2. P 完成 self layers、共享 KV projection、cache write 和 KV transfer；
3. Gateway 只取 P response 中的 KV transfer metadata，丢弃 P 的 sampled token；
4. Gateway 把原始 prompt 与 transfer metadata 发给 D；
5. D 生成 first token 和之后全部用户可见 token。

专用 producer 应配置：

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
```

如果把普通请求混进 `kv_producer`，或者 Gateway 返回/拼接 P token，P 的 logits
因为没有经过 cross layers 而不保证正确。这是部署契约，不是可选性能开关。

### 正确性验证

定向本机回归为 `3 passed`。隔离 B200 Pod 上两个受影响测试文件结果为：

```text
42 passed, 24 warnings in 86.64s
```

DP4 专用 producer 补充验证中，同一测试集合在 B200 上结果为：

```text
43 passed, 24 warnings in 80.85s
```

端到端验证覆盖 1,356、12,096 和 43,704 prompt tokens：

- baseline P/D、candidate P/D 和单体 reference 的最终 D 输出 exact match；
- DP4 P 产生 KV、独立 D 生成全部可见 token，与单体 reference exact match；
- 三轮连续 D -> P -> D KV 回传中，前一轮输出进入下一轮 prompt，三轮最终输出
  仍与 reference exact match；
- 额外发送 43 个独立 P 请求，覆盖 c1/c4/c8 和最长 43.7K prompt，无 hang、
  collective mismatch 或 KV transfer 失败。

### 性能验证

硬件为 B200，模型、服务参数、UCX 1.21 CUDA modules 和测试专用 NIXL alias
overlay 在 baseline/candidate 间保持一致。每个 shape 交替执行 A/B 五轮，以下为
warmed prompt throughput 中位数。

DP1 producer：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `12,838` | `16,479` | `28.36%` |
| 12.1K / c1 | `39,315` | `40,895` | `4.02%` |
| 43.7K / c1 | `44,910` | `47,899` | `6.66%` |
| 12.1K / c4 | `49,099` | `51,888` | `5.68%` |
| 12.1K / c8 | `48,814` | `51,104` | `4.69%` |

DP4 dedicated producer：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `10,419` | `12,507` | `20.04%` |
| 12.1K / c1 | `48,828` | `54,069` | `10.74%` |
| 43.7K / c1 | `64,757` | `70,462` | `8.81%` |
| 12.1K / c4 | `119,388` | `121,266` | `1.57%` |
| 12.1K / c8 | `152,172` | `157,103` | `3.24%` |

DP1 各组 coefficient of variation 为 `0.30%` 到 `2.07%`；DP4 为 `1.17%`
到 `3.20%`。短 prompt 的比例收益较大；长 prompt 和并发场景通常提升约 3% 到
11%。baseline 已经压缩后九层，本提交主要继续移除第一个 owner cross layer 的
完整 prompt 计算，因此不是“从零裁掉十层完整计算”。

### 风险、观察点与回滚

- 风险最高的是 producer 角色配置错误和 Gateway 错误返回 P token；部署前必须
  验证角色和 token ownership。
- DP>1 应观察所有 rank 的 collective 进度；idle-rank dummy 路径被改动后，
  hang 是最直接的回归信号。
- 应持续用最终 D output 与单体 reference 做 exact match，而不是比较 P logits。
- 该 commit 可整体 revert；回滚后功能仍正确，但第一个 cross layer 恢复完整
  prompt 计算，KV-only producer 也恢复十层 cross 计算。

---

## 2. NIXL KV-sharing alias 注册去重

### 提交信息

```text
commit: ea4f80d1b4882ffbddc9aa7135863bab38ba0fee
subject: fix(yoco): deduplicate NIXL KV-sharing aliases (#4)
author: 方涵斌 <2190556589@qq.com>
baseline: 85eab7b56ec4cb1208364bb0daa74d04f09f440e
source PR: https://github.com/buaahsh/vllm/pull/4
diff: 3 files, 80 insertions, 1 deletion
```

### 目的与根因

YOCO 的 cross layers 共享同一个物理 KV cache。ModelRunner 会在 NIXL worker
已经捕获 `KVCacheConfig` 后，为共享层向 `kv_caches` 增加 alias。alias 与 owner
layer 指向相同 tensor 地址，但旧的 `KVCacheConfig` 中没有 alias 自己的 layer
spec。

旧注册顺序先执行：

```text
layer_spec = self._layer_specs[layer_name]
```

因此遇到 late-added alias 时，会在检查 tensor 地址是否已经注册之前抛出
`KeyError`。这不是 KV 数据错误，而是“同一物理区域具有多个逻辑 layer name”与
NIXL 注册顺序之间的兼容性问题。

本提交的目标是只跳过能够证明为重复物理地址的 alias，同时保持 fail-closed：
真正未知且拥有唯一 tensor 的 layer 仍然报错，不能用 alias 兼容逻辑掩盖错误的
KV cache 配置。

### 修改文件及职责

#### `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`

- 将强制索引改为 `self._layer_specs.get(layer_name)`。
- layer spec 缺失时，将单 tensor 或 tensor list 统一为可检查序列。
- 仅当该逻辑 layer 的所有 tensor `data_ptr()` 都已存在于
  `seen_base_addresses` 时，认定它是 KV-sharing alias 并跳过重复注册。
- alias skip 记录 debug log，便于启动问题诊断。
- 如果任何 tensor 地址尚未注册，抛出带 layer name 的明确 `KeyError`，继续
  fail-closed。
- 对拥有正常 spec 的 layer、MLA DSv32 Indexer、block-first 和分离 K/V layout
  不改变原有分支。

#### `tests/v1/kv_connector/unit/test_nixl_connector.py`

- 在现有参数化 cache-registration 测试中模拟真实时序：connector 已捕获
  specs 后，再加入一个共享 tensor alias，并删除该 alias 的独立 spec。
- 继续覆盖 FlashAttention、TritonAttention、cross-layer blocks 开关、
  block-first layout 和分离 K/V layout。
- 验证 alias 不产生新的 registration entry，而唯一未知 cache 不被静默接受。

#### `yoco.md`

- 记录故障触发顺序、修复的 fail-closed 边界、测试以及为什么稳态性能预期为零。

### 正确性验证

定向 NIXL cache-registration 测试结果：

```text
4 passed, 2 skipped, 55 deselected, 14 warnings in 9.32s
```

参数化断言确认：

- block-first layout 仍生成 2 个 registration entries；
- 分离 K/V layout 仍生成 4 个 registration entries；
- KV-sharing alias 新增 registration entries 为 0；
- 缺少 spec 且具有新物理地址的 cache 仍抛出 `KeyError`。

完全相同的两文件补丁还作为 overlay 进入 B200 DP4 producer 验证：

```text
43 passed, 24 warnings in 80.85s
```

四个 producer worker 均完成 NIXL/UCX 初始化；随后通过 1,356、12,096、43,704
prompt tokens 的 P/D exact-match，并额外完成 43 个 c1/c4/c8 请求，无 hang 或
collective mismatch。

### 性能解释

这是启动期 memory registration 正确性修复，不进入请求稳态热路径。alias 复用
已经注册的物理地址，新增 registration 数为零，因此预期稳态吞吐变化为 0。

本提交没有单独报告 tok/s 提升，也没有复用上一条 KV-only prefill 的性能数字。
它的价值是让共享 KV cache 的 producer 能可靠启动，而不是加速每个请求。

### 风险、观察点与回滚

- 必须保留“所有地址已注册”的条件；如果放宽为只按 layer name 或部分地址判断，
  可能漏注册真实的新内存区域。
- 启动日志中的 alias skip 应只出现在共享 cache 层；未知 layer 的 `KeyError` 不应
  被忽略。
- revert 后不会改变模型数值，但 YOCO KV-sharing + NIXL 会重新在启动注册阶段
  因缺少 alias spec 失败。

---

## 3. Router Top-K 删除 dense materialization

### 提交信息

```text
commit: d11cc022bd987fbdb535c056144c261dcdeacbda
subject: perf(yoco): remove redundant router materialization
author/committer: 方涵斌 <2190556589@qq.com>
baseline: ea4f80d1b4882ffbddc9aa7135863bab38ba0fee
branch: review/yoco-03-router-topk
diff: 3 files, 118 insertions, 20 deletions
```

该 commit 未创建 PR；完成测试后由 `Snow2022jlu` 直接同步到 `fhb-dev`，避免为
单功能提交触发额外 PR 邮件。

### 目的与旧路径开销

YOCO Router 固定为 128 routed experts、每 token 选择 Top-8。旧路径虽然已经先对
FP32 softmax scores 执行一次 `torch.topk`，却又把 8 个权重 scatter 回
`[tokens, 128]` dense tensor，生成同 shape 的 bool routing map，然后对 dense
tensor 执行第二次 `torch.topk` 才返回最终 weights/ids。

旧路径概念上是：

```text
FP32 logits
  -> FP32 softmax [tokens, 128]
  -> Top-8 weights/ids
  -> renormalize + scatter to dense routing_probs [tokens, 128]
  -> routing_map [tokens, 128]
  -> second Top-8
  -> final weights/ids
```

第二次 Top-K 不增加信息；dense scatter 只是为了重新取回刚得到的 Top-8。本提交
删除这段冗余物化，让第一次 Top-K 结果直接成为最终结果。

### 修改文件及职责

#### `vllm/model_executor/models/yoco.py`

- 将 `_yoco_routing_renorm_scatter_kernel` 简化为
  `_yoco_routing_renorm_kernel`。
- kernel 不再读取 expert ids，不再向 128-expert dense output scatter，而是对
  `[tokens, 8]` Top-K weights 原地 post-top-k renormalization。
- `_yoco_topk_routing_impl` 直接返回第一次 `torch.topk` 产生的 weights/ids。
- 删除 `torch.zeros_like(router_logits)`、`routing_probs != 0` 和第二次
  `torch.topk`。
- 保留原 FP32 softmax kernel、`torch.topk` 实现、Top-8 参数和 renorm block
  选择，避免把数值路径变化混入性能优化。

#### `tests/model_executor/test_yoco_conversion.py`

- 对 token 数 1、3、66、110、256 生成固定 seed 的 FP32 random logits。
- reference 明确执行 FP32 softmax、`torch.topk(k=8)` 和 post-top-k renorm。
- expert ids 要求 exact match；weights 使用 `rtol=2e-6, atol=0`。
- 增加全相等 logits case；weights 和 ids 均 exact match，验证 tie order 没有因
  自定义排序或 dense 路径删除而变化。

#### `yoco.md`

- 记录独立 Router 测试和 CUDA Graph microbenchmark；不引用此前
  Router + fused add-RMSNorm 的组合收益。

### 内存变化

每个 token 明确删除两个 dense 临时 tensor：

```text
FP32 routing_probs: 128 * 4 bytes = 512 bytes/token
bool routing_map:   128 * 1 byte  = 128 bytes/token
total:                              640 bytes/token
```

因此 16,384 tokens 时，仅这两个显式 tensor 就少分配 10 MiB。第二次 Top-K 的
workspace 和第一次 Top-K 结果作为临时值产生的额外峰值没有计入，640 bytes/token
是保守且可直接由 shape/dtype 证明的数字。

### 正确性验证

独立 B200 Pod `lidong1-yoco-pr03-router-g1-0804-master-0`，节点
`slc01-cl02-hgx-0346`，定向测试结果：

```text
6 passed, 9 deselected, 19 warnings in 4.05s
```

结果包括五个 random shape 和一个全相等 logits tie case。所有 expert ids 与
baseline 一致；weights 满足上述误差要求，tie case 的 weights/ids exact match。

受影响的三个文件通过：

- `ruff-check`；
- `ruff-format`；
- `markdownlint-cli2`；
- `mypy` hook；
- `git diff --check`；
- 其余适用的 pre-commit hooks。

### 独立性能验证

硬件为同一张 NVIDIA B200。baseline 为本 commit 前的 Router，candidate 为本
commit；两边使用同一组 FP32 logits、相同 softmax kernel 和相同 Top-K 参数。
每个实现分别捕获为 CUDA Graph，warmup 后每个 shape 计时 1,000 次 graph replay，
报告中位 GPU 延迟。

| tokens | baseline (us) | candidate (us) | 加速 | 少分配 dense 临时内存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `24.704` | `16.096` | `1.535x` | 640 B |
| 3 | `28.864` | `15.968` | `1.808x` | 1,920 B |
| 66 | `30.784` | `17.824` | `1.727x` | 42,240 B |
| 110 | `32.288` | `18.336` | `1.761x` | 70,400 B |
| 256 | `34.768` | `20.112` | `1.729x` | 160 KiB |
| 1,024 | `38.912` | `22.048` | `1.765x` | 640 KiB |
| 4,096 | `73.248` | `40.320` | `1.817x` | 2.5 MiB |
| 16,384 | `214.944` | `116.480` | `1.845x` | 10 MiB |

使用 CUDA Graph 是为了匹配实际 serving 形态，并排除逐 kernel Python launch
开销。表中收益来自减少 GPU kernel、dense tensor 读写和 Top-K 工作，不是“忘记
开启 CUDA Graph”造成的假收益。

该表只代表 Router 单算子，不等于整模型吞吐提升。Router 在每个 MoE layer 都会
调用，因此收益会累积，但端到端幅度还受 attention、expert GEMM、collective、
batch shape 和显存带宽占比影响，必须通过后续整模型 A/B 单独证明。

### 风险、观察点与回滚

- 最敏感的兼容点是 `torch.topk` 的 tie order；全相等 logits 测试必须长期保留。
- 当前 fast path 只处理 128 experts、Top-8；其他 expert 数或 Top-K 继续走通用
  PyTorch fallback，不应无验证扩大固定 Triton kernel 的适用范围。
- 需要关注非常大 token batch 下 Top-K workspace 和 graph capture pool；显式
  dense 临时内存已经删除，但不能把 640 bytes/token 解释成完整峰值显存差。
- revert 该 commit 可恢复旧 Router，模型接口不变，不影响前两个 PD/NIXL 提交。

---

## 4. Fused residual add-RMSNorm

### 提交信息

```text
commit: ef60ee02559f788e2ea829a942e40704074c62a6
subject: perf(yoco): fuse residual add with RMSNorm
author/committer: 方涵斌 <2190556589@qq.com>
baseline: d77fed9d2a6e6b84a7a1573b6a0a868667648e95
branch: review/yoco-04-fused-add-rmsnorm
source final state: d934b7411b + cb5b179a44
diff: 3 files, 341 insertions, 7 deletions
```

该功能完成检查和 B200 验证后，由 `Snow2022jlu` 直接同步到 `fhb-dev`，没有创建
PR。旧开发分支的初版实现和 residual-boundary 精度修复在本次迁移时被合并成一个
最终 commit，因此 `fhb-dev` 不包含已知有问题的中间状态。

### 目的

YOCO 每个 token step 执行 40 次 decoder block：10 个 self layers 经过三次
universal loop，再执行 10 个 cross layers。每个 block 的 attention sublayer 后
都依次执行：

```text
attention_output
  -> FP32 residual add
  -> materialize FP32 residual_out
  -> RMSNorm reduction
  -> BF16 normalized output
  -> MoE
```

旧路径的 residual add 与 RMSNorm 是两个独立阶段。RMSNorm 的第一遍 reduction
又需要读取刚写出的完整 FP32 residual，因此存在额外 kernel node 和全量显存读取。

本提交的目标是把 attention output 的 FP32 residual add 放进 RMSNorm reduction
的第一遍：同一次读取中生成 `residual_out` 并累加平方和，第二遍完成 normalize 和
weight multiply。这样仍输出完整 FP32 residual 给 layer 末尾使用，但少一个独立
add 阶段。

### 为什么不能直接使用 vLLM 通用 fused add-RMSNorm

vLLM 已有 `_C.fused_add_rms_norm` 和 `vllm.ir.ops.fused_add_rms_norm`，但当前 CUDA
实现要求 activation、residual 和 weight dtype 与 input dtype 匹配，并原地更新
同 dtype 的 input/residual。

YOCO 的数值契约不同：

```text
attention/MLP sublayer output: BF16 或 FP32
long-lived residual:           FP32
RMSNorm weight:                BF16 checkpoint weight
normalized output:             BF16
```

如果为了复用通用 op 把 residual 降为 BF16，会移动舍入点并在 40 个 block execution
中累积误差；如果让 normalized output 保持 FP32，则又改变后续 projection/MoE 的
输入 dtype 和已验证路径。因此新增 YOCO 专用 mixed-dtype kernel，而不是放宽通用
kernel 的全局 dtype 契约。

### 实现细节

#### 第一遍：add、residual materialization 和 reduction

Triton kernel 每行对应一个 token 的 hidden vector，hidden size 固定为 3072：

1. BF16/FP32 `x` 和 FP32 `residual` 都转为 FP32；
2. 计算 `values = x + residual`；
3. 立即把 `values` 写入 FP32 `residual_out`；
4. 同时以 FP32 累加 `values * values`；
5. reduction 完成后计算 `rsqrt(mean(square) + eps)`。

token 数小于 128 时使用 `REDUCTION_BLOCK=2048`，否则使用 4096；这与原 YOCO
RMSNorm 的选择一致，保留 B200 上已验证的 reduction order。

#### 第二遍：normalize 和 BF16 output

1. 再读 FP32 `residual_out`；
2. 读取转换为 BF16 layout 的 RMSNorm weight，并在 kernel 中转为 FP32；
3. 计算 `values * inv_rms * weight`；
4. 写入 BF16 normalized output，供后续 MoE 使用。

返回值是：

```text
(normalized_bf16, residual_out_fp32)
```

#### fast path 条件与 fallback

只有同时满足以下条件才进入 Triton fast path：

- Triton 可用且运行平台为 CUDA；
- `x` 和 `residual` 都在 CUDA；
- `x` 为 BF16 或 FP32；
- `residual` 为 FP32；
- 两者 shape 完全相同；
- hidden size 为 3072。

其他 device、dtype、shape 或 hidden size 走 PyTorch fallback：先执行
`residual + x.float()`，再调用原 RMSNorm。fallback 不是近似路径，公式和返回
dtype 与融合语义相同。

### 融合边界与被拒绝的旧实验

融合只发生在 decoder layer 内部：

```text
residual_before_attention + attention_output
  -> fused RMSNorm input for MLP
MLP output
  -> residual_out + mlp_output.float()
  -> materialize layer output
```

旧分支初版 `d934b7411b` 还尝试把 MLP output 与 FP32 residual 拆开跨 decoder
layer 携带，让下一层继续融合。这一做法局部算子测试正确，但改变了 layer boundary
的 FP32 materialization/rounding point；长上下文不能保持 bitwise 一致。

后续修复 `cb5b179a44` 撤销跨层 residual carry，只保留 attention 后的层内融合。
本次 `ef60ee0255` 直接移植两者的最终净效果，确保每层末尾仍与旧实现完全相同，
没有把错误中间版本写入 `fhb-dev`。

### 修改文件及职责

#### `vllm/model_executor/models/yoco.py`

- 增加 `_yoco_fused_add_rms_norm_kernel`；
- 增加 CUDA wrapper 和 fake implementation；
- 注册 `torch.ops.vllm.yoco_fused_add_rms_norm`，可被 torch.compile 和 CUDA Graph
  捕获；
- 扩展 YOCO `RMSNorm.forward(x, residual=None)`：无 residual 时完全保留旧接口，
  有 residual 时返回 `(normalized, residual_out)`；
- `YOCODecoderLayer` 在 attention 后调用融合接口；
- layer 末尾继续执行原 MLP residual add，不跨层携带未物化 partial；
- 对无 residual 的 norm call 增加类型断言，明确 compile graph 中返回值仍是单
  tensor。

#### `tests/model_executor/test_yoco_conversion.py`

- CPU hidden-size 4 case 验证 fallback 的 FP32 residual 和 normalized output；
- B200 hidden-size 3072 case 覆盖 1、66、128 tokens；
- CUDA fast path 的 FP32 residual 与 BF16 normalized output 都要求 bitwise match，
  使用 `rtol=0, atol=0`；
- 构造最小 `YOCODecoderLayer` 连续执行两个 loop，与逐语句 legacy forward 对比，
  防止再次移动 layer boundary；
- 原 Router、fast-prefill、KV-only 和权重转换测试继续在同一文件完整运行。

#### `yoco.md`

- 记录独立 baseline、B200 测试、两种 CUDA Graph 计时口径以及边界说明。

### 提交前测试拦截的问题

首次手工迁移时，重载 `forward(x, residual)` 的补丁因上下文匹配过宽，错误落到了
相邻的 `RMSClip` 类，而不是 `RMSNorm`。首轮 B200 定向测试立即得到：

```text
5 failed, 15 deselected
TypeError: RMSNorm.forward() takes 2 positional arguments but 3 were given
```

该失败状态从未 commit 或推送。修正后检查类定义位置，重新同步完整文件，并从零
重跑定向测试、完整测试和 microbenchmark。记录这次拦截的目的，是提醒后续移植
同文件多个 `forward` 方法时必须用 class context 定位，不能只依赖方法签名。

### 最终正确性验证

独立 B200 Pod：

```text
pod:  lidong1-yoco-pr04-add-rmsnorm-g1-0804-master-0
node: slc01-cl02-hgx-0418
GPU:  NVIDIA B200
```

修正后的定向结果：

```text
5 passed, 15 deselected, 19 warnings in 2.52s
```

完整 `tests/model_executor/test_yoco_conversion.py`：

```text
20 passed, 19 warnings in 20.55s
```

验证覆盖：

- CPU fallback；
- CUDA fast path 1/66/128 tokens；
- normalized BF16 output bitwise match；
- FP32 residual output bitwise match；
- 两个连续 decoder loop 的 layer-boundary exact match；
- 原 Router random/tie cases；
- fast-prefill 与 KV-only control flow；
- YOCO v2/v3 state-dict conversion 和配置默认值。

受影响文件还通过 `ruff-check`、`ruff-format`、`mypy`、`markdownlint-cli2`、
`git diff --check` 和其余适用的 pre-commit hooks。

### 性能方法

baseline 与 candidate 都先经过 `torch.compile(fullgraph=True)`：

```text
baseline:
  residual_out = residual + x.float()
  normalized = yoco_rms_norm(residual_out)

candidate:
  normalized, residual_out = yoco_fused_add_rms_norm(x, residual)
```

每个 shape 在计时前先比较两个输出，要求 exact match。随后分别 capture 为 CUDA
Graph，warmup 后执行 1,000 次 graph replay，报告中位数。硬件、输入、weight、
reduction block 和输出 dtype 保持一致。

### 单微算子 CUDA Graph replay

每个 graph 只有一个 add-RMSNorm 操作。约 9 us 的整图固定开销对小 shape 占比很
高，因此该口径更接近“单独发起一个微型 graph”的上限，不适合推断 40 个 layer
中的累计节省：

| tokens | baseline (us) | candidate (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `9.568` | `9.152` | `1.045x` |
| 3 | `9.472` | `9.184` | `1.031x` |
| 66 | `9.536` | `9.472` | `1.007x` |
| 110 | `9.408` | `9.408` | `1.000x` |
| 128 | `9.376` | `9.376` | `1.000x` |
| 256 | `9.472` | `9.472` | `1.000x` |
| 1,024 | `17.184` | `11.136` | `1.543x` |
| 4,096 | `48.032` | `31.136` | `1.543x` |

### 40 个算子节点同图摊销

YOCO 的实际模型 graph 包含 40 次 decoder block execution，而不是为每个 norm
单独 replay 一张 CUDA Graph。第二个 benchmark 把 40 个算子节点 capture 到同一
graph，总延迟除以 40，从而只支付一次整图 replay 固定开销：

| tokens | baseline/op (us) | candidate/op (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `3.217` | `2.162` | `1.488x` |
| 3 | `3.376` | `2.202` | `1.533x` |
| 66 | `3.898` | `2.403` | `1.622x` |
| 110 | `4.505` | `2.456` | `1.834x` |
| 128 | `4.604` | `2.300` | `2.002x` |
| 256 | `6.444` | `2.864` | `2.250x` |
| 1,024 | `18.273` | `6.629` | `2.756x` |
| 4,096 | `68.862` | `26.533` | `2.595x` |

40 个输入在 microbenchmark 中彼此独立，所以该数据衡量的是同图 kernel-node 和
显存流量摊销，不包含真实 attention、MoE 或 collective，也不能直接换算成整模型
tok/s。它的意义是证明即使 serving 已启用 CUDA Graph，减少 graph 内部 kernel
node 和 FP32 tensor 读写仍然有效。

旧分支曾报告 Router + add-RMSNorm 两项合并后的 endpoint 变化；本提交没有把该
组合数据作为单功能收益。后续若报告端到端收益，必须以 `d77fed9d2a` 为 baseline，
只切换本 commit，并交替执行同 workload。

### 风险、观察点与回滚

- 最大正确性风险是再次扩大融合边界、移动 layer-end FP32 materialization；完整
  layer test 和长上下文 exact-match 必须保留。
- fast path 固定 hidden size 3072 和已验证 dtype；其他 shape 必须继续 fallback，
  不应仅为覆盖更多模型而放宽断言。
- reduction block 的 128-token 切换点影响 reduction order；修改 tile 前必须重做
  bitwise test，不能只看 `allclose`。
- custom op 是 inference-only；本提交没有注册 autograd kernel，也不应用于训练。
- revert `ef60ee0255` 可恢复顺序 residual add + RMSNorm；Router、PD 和 NIXL 提交
  不受影响，模型功能仍正确但恢复额外 kernel/显存读取。

---

## 5. Shared Expert 与 Routed MoE 并行

### 提交信息

```text
commit: 9988ae737fc717a9ba1846867667e6ecf441299d
subject: perf(yoco): overlap shared and routed experts
author/committer: 方涵斌 <2190556589@qq.com>
baseline: cd1e22c7ff18883dc2b45b7878c70c370c84cbac
branch: review/yoco-05-shared-expert-overlap
diff: 4 files, 223 insertions, 39 deletions
```

该功能完成本地和 B200 验证后，由 `Snow2022jlu` 直接同步到
`fhb-dev`，不创建 PR，避免小功能提交反复触发邮件。

### 目的与旧路径瓶颈

YOCO-v3 每次 decoder block execution 同时使用 routed experts 和一个 gated
shared expert。旧实现的时间线是：

```text
router
  -> routed latent projection/norm
  -> routed dispatch + expert compute + TP reduction
  -> routed latent norm/projection
  -> shared expert GEMMs + TP reduction
  -> shared sigmoid gate
  -> sum
```

routed 路径运行时，shared expert 所需的本地 GEMM 没有开始；它只能在
routed 路径完全结束后串行执行。vLLM `FusedMoE` 已有一条用于
shared experts 的 auxiliary CUDA stream，但 YOCO 原本在 `FusedMoE` 外部
直接调用 shared module，因而没有利用该 overlap 能力。

本提交的目标是复用现有 stream，让 shared expert GEMM 与 routed
dispatch/expert compute 并行，同时完整保留 YOCO 的 latent transform、
collective 和 shared gate 数值边界。

### 非目标

- 不融合 routed 和 shared 的 TP all-reduce；
- 不修改 Router Top-K、routed scaling 或 expert 权重布局；
- 不增加新 CUDA kernel，也不替换 Triton MoE backend；
- 不改变未显式启用新选项的其他 `FusedMoE` 模型；
- 不把 TP1 端到端数据解读为已完成 TP>1 实机 collective 验收。

### 实现和数值顺序

新时间线是：

```text
routed latent input projection + norm
  -> routed expert compute -------------------+
shared expert compute on auxiliary stream ----+--- synchronize
  -> routed TP reduction
  -> shared TP reduction
  -> routed latent output norm + projection
  -> sigmoid(shared_gate(hidden_states)) * reduced_shared
  -> routed + shared
```

YOCO shared expert 的 row-parallel down projection 改为返回 TP-local output，使
`FusedMoE` 可以在 auxiliary stream 上执行完整本地 shared GEMM。stream
同步后，runner 先 reduction routed output，再 reduction shared output。

两次 collective 不能先求和再做一次 all-reduce。两条路径的 dtype、latent
projection 和舍入点不同，合并 reduction 会改变参考路径的数值顺序。
`reduce_shared_experts_separately=True` 因此是一个显式的 model-specific
contract，而不是默认修改所有 MoE。

routed 输入的 latent down projection + norm 和 routed 输出的 norm + up
projection 被包装为 `FusedMoE` transform hooks。shared 路径继续直接接收原始
hidden states，不会错用 routed latent tensor。

`shared_gate` 是 replicated linear weight，所有 TP rank 的 scale 相同。新路径在
shared reduction 完成后计算 `sigmoid(shared_gate(hidden_states))` 并乘到
reduced shared output，然后才与 routed output 相加。这保持原数学含义，且
避免 gate 在 auxiliary stream 上引入新依赖。

### 修改文件及职责

#### `vllm/model_executor/layers/fused_moe/layer.py`

- 为 `FusedMoE` 构造器增加 `shared_output_transform`；
- 增加 `reduce_shared_experts_separately`，并传递给 runner；
- 开启 separate reduction 但没有 `shared_experts` 时 fail closed，避免在
  serving 中静默跳过必要 collective。

#### `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

- 复用已有 shared expert auxiliary CUDA stream，不新建每 layer stream；
- stream 同步后依次执行 routed reduction 和 shared reduction；
- separate mode 不走原有 late combined reduction，避免重复 all-reduce；
- reduction 后调用 `shared_output_transform`，再与 routed output 相加；
- sequence-parallel 或无 TP/EP reduction 的情况保持对应 fast path。

#### `vllm/model_executor/models/yoco.py`

- 增加 `YOCOLatentInputTransform`，保持 projection -> norm 顺序；
- 增加 `YOCOLatentOutputTransform`，保持 norm -> projection 顺序；
- 增加 `YOCOSharedOutputTransform`，在 reduced shared output 上应用 sigmoid
  gate；
- shared expert 从内部 all-reduce 改为 TP-local output；
- 把 shared experts 和三个 transform 传入 `FusedMoE`，删除外部串行
  shared call 和重复的手工求和。

#### `tests/model_executor/test_yoco_conversion.py`

- 用 BF16 小 tensor exact-match 验证 latent input/output transform 顺序；
- 验证 shared sigmoid gate 接收的是 reduced shared output；
- mock TP2 all-reduce，确认调用顺序是 routed 在前、shared 在后，且
  两个输出没有交换。

### 正确性验证

本地 NVIDIA RTX A6000：

```text
23 passed, 14 warnings in 12.75s
```

独立 B200 Job：

```text
experiment: fhb-moe7-0804
job:        b200
pod:        fhb-moe7-0804-b200-564c6e49-master-0
commit:     9988ae737f
baseline:   cd1e22c7ff
```

B200 相关回归：

```text
68 passed, 24 warnings in 88.70s
```

baseline 和 candidate 的服务都使用 YOCO-v3 BF16、TP1/DP1、FlashInfer
attention、Triton MoE、prefix caching、chunked prefill 和 KV-sharing fast
prefill。两边均为非 eager，启动日志显示 `FULL_AND_PIECEWISE` CUDA Graph。

正确性 A/B 覆盖 1,360、4,096 和 7,000 token target prompt，每次固定
生成 16 tokens，`temperature=0`。两边的生成文本、usage、finish reason、
token id、token logprob 和每步 top-5 logprob 全部 exact match。日志中无
CUDA/NCCL 错误、collective mismatch 或 traceback。

本次端到端实机验收是 TP1，因此真实 multi-TP collective 不在已证明范围。
TP2 顺序由 mock collective 单元测试拦截；若将 YOCO 部署到 TP>1，仍需
补一轮真实多卡 exact-match 和 hang 回归。

所有适用 pre-commit hooks 通过，包括 ruff-check、ruff-format、mypy、
SPDX header、配置检查以及 `git diff --check`。

### 独立端到端性能

baseline 固定为 `cd1e22c7ff`，candidate 固定为 `9988ae737f`，只切换本功能
commit。性能输入为同一 random prompt，实际分词 1,299 input tokens，固定
生成 128 tokens。

| 并发 | 请求数 | baseline tok/s | candidate tok/s | 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | `136.20` | `141.99` | `+4.25%` |
| 4 | 16 | `375.26` | `381.04` | `+1.54%` |
| 8 | 32 | `720.69` | `718.50` | `-0.30%` |

每个并发档位执行 3 轮，表中是三轮整体 output throughput 的中位数。
并发 1 改善 `+4.25%`，并发 4 改善 `+1.54%`；并发 8 为 `-0.30%`，
在本次三轮测量的噪声范围内。因此结论是低、中并发存在可重复收益，
高并发暂未证明收益，不宣称全 shape 提速。

原始结果：

```text
/mnt/pvc/lidong1/vllm_pd/shared-expert-overlap-fhb-dev-0804/
fhb-moe7-0804-b200-564c6e49-master-0
```

### 风险、观察点与回滚

- `shared_output_transform` 在 shared reduction 后执行；如果以后引入需要
  TP-local 数据的 transform，必须增加新的明确契约，不能复用本 hook。
- separate mode 保留两次 collective，所以收益只来自本地 shared/routed
  compute overlap，不包含通信次数优化。
- 需继续观察大 batch 下 auxiliary stream 对 routed kernels 的 SM/显存带宽
  竞争；并发 8 已表明 overlap 不是所有 shape 都必然获益。
- 优化了 stream 时间线但没有减少 CUDA Graph 内的 kernel node 数；后续若继续
  开发单 kernel 融合，必须与本 overlap 分开测量。
- revert `9988ae737f` 可完整恢复串行 shared expert 路径，其他四个
  `fhb-dev` 功能提交不受影响。

---

## 6. Shared Expert FP32 clamped-SwiGLU 单 kernel

### 提交信息

```text
commit: 8eae22948cd627e924d9583f1946eb2b88e8b9eb
subject: perf(yoco): fuse FP32 clamped SwiGLU
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 2e1b8ecc53705574ec38597dcdae81c939b40158
branch: review/yoco-06-fp32-clamped-swiglu
diff: 6 files, 253 insertions, 18 deletions
source validated patch: f265362d2905216c81b8571db20d19f16023a344
```

该功能完成当前分支 A6000 重编译、bitwise/opcheck、YOCO 回归和
microbenchmark 后，由 `Snow2022jlu` 直接同步到 `fhb-dev`；不创建 PR。

### 目的和不能复用旧 kernel 的原因

YOCO-v3 Shared Expert 在 `swiglu_limit > 0` 时必须保持训练侧的 FP32
中间语义：

```python
gate, up = gate_up.float().chunk(2, dim=-1)
gate = gate.clamp(max=limit)
up = up.clamp(min=-limit, max=limit)
output = (silu(gate) * up).to(gate_up.dtype)
```

原实现是一组 PyTorch op，CUDA eager profile 中对应 6 个 GPU kernel。YOCO
每个 token step 有 40 次 decoder block execution，因此即使 serving 已开启 CUDA
Graph，graph 内仍会反复保留这些 kernel node 和 FP32 中间 tensor 读写。

vLLM 已有 `SiluAndMulWithClamp`，但它的 BF16/FP16 路径在 clamp 和
activation 中间会舍入回 input dtype。这是其他模型已建立的数值契约，
直接复用会让 YOCO 离开训练参考路径，全局修改又会破坏其他模型。
因此本提交新增独立 FP32-intermediate op，不改已有 op。

### 数值契约和 kernel 路径

新 kernel 对 BF16、FP16 和 FP32 input 都使用同一公式：

1. input load 后转为 FP32 gate/up；
2. gate 仅设置上界 `limit`；
3. up 设置 `[-limit, limit]` 上下界；
4. 在 FP32 中执行 `gate / (1 + exp(-gate)) * up`；
5. 只在最后 store 时转回 input dtype。

clamp 使用条件比较表达式，而不是 `fminf/fmaxf`。原因是两者的 NaN
选择语义可能不同；条件比较保留 NaN，与 `torch.clamp` 对齐。

launch 路径：

- hidden dim 对齐时使用 `PackedVec` 和 128-bit `ld128/st128`；
- CUDA >= 12.9、SM major >= 10 且 token 数 > 128 时使用 256-bit
  `ld256/st256`；
- 不对齐时使用 scalar fallback；
- zero-token input 直接返回，不 launch 空 grid；
- CUDA stream 使用 PyTorch current stream，可被 torch.compile 和 CUDA Graph
  正常捕获。

### CustomOp 与 YOCO 接入

`SiluAndMulWithClampFP32` 同时提供：

- `forward_native`：CPU、ROCm、XPU 和测试使用的 FP32 PyTorch reference；
- `forward_cuda`：分配 output 并调用新 `_C` op；
- 返回 shape 为 input 最后一维的一半，dtype/device 与 input 相同。

YOCO 用 `enforce_enable=True` 实例化该 CustomOp。这保证即使全局 custom-op
编译配置改变，Shared Expert 仍保留已验证的单 kernel 边界，不在
opaque/compiled 路径里重新展开为多个 PyTorch op。

`swiglu_limit <= 0` 时没有 clamp 契约，YOCO 仍使用原来的 `SiluAndMul`，
不强制经过新 op。Routed experts 的 clamp 路径不在本提交范围。

### 非目标

- 不改变 Routed MoE 的 activation 或 fused expert kernel；
- 不改变 Shared/Routed auxiliary-stream overlap 时间线；
- 不修改其他模型使用的 `SiluAndMulWithClamp`；
- 不把 activation microbenchmark 直接换算成整模型 tok/s；
- 不把旧 source commit 的 B200 endpoint 结果冒充为当前 commit 的新重跑。

### 修改文件及职责

#### `csrc/activation_kernels.cu`

- 增加 scalar 和 packed FP32-intermediate compute helper；
- 增加 `silu_and_mul_clamp_fp32_kernel`；
- 增加 zero-token、scalar、128-bit 和 SM100 256-bit launch dispatch。

#### `csrc/ops.h`

- 声明 `silu_and_mul_clamp_fp32` C++/CUDA 入口。

#### `csrc/torch_bindings.cpp`

- 注册 `_C.silu_and_mul_with_clamp_fp32(Tensor! result, Tensor input,
  float limit) -> ()`；
- 只为 CUDA backend 绑定新 kernel。

#### `vllm/model_executor/layers/activation.py`

- 增加 `SiluAndMulWithClampFP32` CustomOp；
- 定义 FP32 native reference 和 CUDA output buffer 语义；
- ROCm 显式回退 native，CPU/XPU 复用 CustomOp 基类 native fallback。

#### `vllm/model_executor/models/yoco.py`

- 删除多算子 `YOCOClampedSwiGLU` wrapper；
- Shared Expert 在 positive limit 时实例化强制开启的新 CustomOp；
- 保留 Routed Expert 的原 clamp 参数传递和 non-positive fallback。

#### `tests/kernels/core/test_activation.py`

- dtype：FP16、BF16、FP32；
- device：两张 CUDA GPU；
- shape：`d=1279` scalar fallback，`d=1280` 的 1/7/128/129 tokens；
- 新 kernel 与 native FP32 reference 要求 bitwise match；
- 低精度 case 必须与旧 low-precision-intermediate kernel 不同；
- 每个 dtype/shape/device 组合执行 PyTorch opcheck。

### 当前 clean-history 分支的测试

本地硬件和编译环境：

```text
GPU:       NVIDIA RTX A6000, GPU0 and GPU1
PyTorch:   2.11.0+cu130
CUDA:      13.0
NVCC arch: sm_86
extension: current branch _C rebuilt from source
```

activation correctness/opcheck：

```text
30 passed, 468 deselected, 14 warnings in 15.11s
```

YOCO conversion/config 组合回归：

```text
34 passed, 14 warnings in 18.95s
```

当前分支的 6 个功能文件还通过：

- ruff-check 和 ruff-format；
- clang-format；
- mypy；
- typos 和 SPDX header；
- forbidden import / CUDA API / lazy import 检查；
- configuration default/docstring 检查；
- `git diff --check`。

测试环境有两个非代码问题，都在提交前解决：

1. CMake 通过 pip CUDA toolkit 自动配置时未找到已安装的 `libnvrtc.so.13`；
   显式传入 `CUDA_nvrtc_LIBRARY`/`CUDA_NVRTC_LIB` 后，`_C` 完整编译和
   链接通过。
2. repo root 有一个指向不可读外部 mount 的 tokenizer symlink，pytest 在
   收集阶段触发 `EIO`。测试时临时移出该 symlink，shell trap 在测试后原位
   恢复；实际 test case 首次执行即全部通过。

这两个失败都发生在源码测试执行前，不是 kernel/test failure，也没有通过
放宽数值阈值规避。

### 当前分支 A6000 microbenchmark

eager 口径：BF16、`d=1280`，每轮 400 次调用，7 轮取中位数。每个
shape 在计时前要求输出 bitwise match。

| tokens | FP32 多算子 (us) | fused 单 kernel (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `107.771` | `18.611` | `5.79x` |
| 8 | `103.790` | `19.855` | `5.23x` |
| 128 | `110.897` | `18.895` | `5.87x` |

CUDA Graph 口径：两个实现分别 capture；每轮 1,000 次 replay，7 轮取
中位数。capture 后先 replay 并要求输出 bitwise match。

| tokens | FP32 多算子 graph (us) | fused graph (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 129 | `14.402` | `4.833` | `2.98x` |
| 4,096 | `488.818` | `53.897` | `9.07x` |

eager 和 graph 都是 activation 局部数据。graph 结果证明收益来自 graph 内部
node 和中间显存流量减少，不是未开 CUDA Graph 造成的 launch 假收益。

### 原 patch 的 B200 实机验证

当前移植的 6 个功能文件与
`f265362d2905216c81b8571db20d19f16023a344` 的对应 tree 内容完全一致。其旧
parent `21f20666218b6672fccac3230caa7405d29bc7af` 与当前 baseline 在这 6 个
文件上也完全一致。

因此下列数据能证明“相同代码 patch”在 B200/SM100 上的 correctness 和
性能，但记录为 source-patch validation，不宣称是当前 hash 的新 Job。

B200 kernel 验证：

- 两张 B200 使用 CUDA 13.1、PyTorch `2.11.0a0+nv26.02` 为 SM100 重编译
  `_C`；
- 合计 30 correctness + 30 opcheck 全通过；
- 覆盖 FP16/BF16/FP32、scalar、128-bit、256-bit、显式 NaN、CustomOp 和
  16,384-token long batch；
- CUDA Graph capture 后原地更换 static input，replay 仍与新 input reference
  bitwise match；
- profiler 确认 activation 从 6 个 GPU kernel 变为 1 个
  `silu_and_mul_clamp_fp32_kernel`。

B200 endpoint A/B：YOCO-v3/L3 BF16，TP1，FlashInfer attention，Triton MoE，
非 eager `FULL_AND_PIECEWISE` CUDA Graph。baseline/candidate 在 GPU0/1 同时运行并
交换 GPU 布局后重复。

1,360/12,097/43,709-token prompt 各固定生成 64 tokens，在两个 GPU 布局中
文本、token id/string、token logprob 和 top-5 logprob 全部 exact match。

decode A/B 使用 1,360-token prompt、固定生成 128 tokens；每个 GPU 布局对
c1/c4/c8 交替执行 3 轮，合并 6 个 sample 取中位数：

| 并发 | baseline tok/s | candidate tok/s | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | `133.521` | `136.044` | `+1.89%` |
| 4 | `351.494` | `354.833` | `+0.95%` |
| 8 | `749.076` | `756.981` | `+1.06%` |

c1 两个布局分别为 `+2.34% / +1.36%`，c4 为 `+1.31% / +0.71%`，
方向一致。c8 为 `+2.36% / +0.08%`，pooled 约 1% 已接近噪声，不作
SLA 承诺。原始数据：

```text
/mnt/pvc/lidong1/vllm_pd/fp32-clamped-swiglu-0804/
```

### 风险、观察点与回滚

- FP32 intermediate 是 correctness 契约，不是可以为了速度关闭的精度选项；
- packed/scalar 和 128/256-bit 路径要继续保留 NaN、bitwise 和 opcheck
  matrix；
- 256-bit 实际分支只在 SM100+ 且 tokens > 128 时执行，A6000 的
  129-token case 不代替 B200 分支验证；
- `enforce_enable=True` 是确保 graph 中只有一个 activation node 的关键，
  后续修改 CustomOp dispatch 时要重跑 profiler；
- 当前提速局限于 Shared Expert activation，不会减少 Routed MoE kernel；
- revert `8eae22948c` 可恢复多算子 FP32 activation，其他五个
  `fhb-dev` 功能提交不受影响。

---

## 7. Streaming stop 保留 KV transfer metadata

### 提交信息

```text
commit: 1c51cac0d7fc374475c97a932dd3d37b9ab9dfaf
subject: fix(pd): preserve KV metadata on streamed stops
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 4b0f7d3d4249b7d0629619188a2dcc16bda103d9
branch: review/yoco-07-streamed-stop-kv-metadata
source patch: c5a812bf889df2f686bfbb8ce65c5ae852bafacd
diff: 14 files, 207 insertions, 25 deletions
```

该功能通过定向正确性测试后，由 `Snow2022jlu` 直接同步到
`fhb-dev`，不创建 PR。

### 问题根因

vLLM stop 可在两个不同位置被检测：

1. EngineCore/sampler 直接检测 token-id stop、EOS 或 length cap，core output 本身
   已带 finish reason；
2. frontend detokenizer 拼出文本后检测多 token stop string，此时 core 仍
   认为请求 active。

旧实现在第 2 种情况下会把 internal request id 加入 `reqs_to_abort`，然后
AsyncLLM/LLMEngine 调用 abort API。abort 能停止计算和释放 block，但它语义是
客户端取消，不会生成正常 final `EngineCoreOutput`。

P/D connector 在 scheduler `_free_request()` 时可以返回以下 metadata：

```text
remote_engine_id
remote_request_id
remote_block_ids
```

但旧 abort call 丢弃 `_free_request()` 返回值，也没有 final core output 承载它。因此
OpenAI streaming terminal chunk 的 `kv_transfer_params` 为空。Gateway 如果依赖该
metadata 把 KV 交给 Decode，就无法正常继续 P/D 请求。

### 修复后状态机

```text
normal token output
  -> frontend detects stop string
  -> RequestState.pending_stop_reason = matched string
  -> no user-visible finished output yet
  -> reqs_to_stop contains internal request id
  -> sync/async client sends EngineCoreRequestType.STOP
  -> scheduler marks FINISHED_STOPPED
  -> connector free returns kv_transfer_params
  -> EngineCore returns empty-token STOP output to the owning client index
  -> OutputProcessor restores pending stop reason
  -> RequestState finishes exactly once
  -> final chat/completion stream chunk includes kv_transfer_params
```

客户端取消、engine failure 和其他真正 abort 仍走 `ABORT`/`FINISHED_ABORTED`，
不会被误标记为 successful STOP。

### 为什么需要 14 个文件

这个 commit 的文件数高于算子优化，但它横跨的是一条原子 protocol，不是
14 个不相关功能。如果拆分合入，任何中间 commit 都会存在以下至少一个问题：

- STOP request 发不到 MP/DP engine；
- scheduler 已经产生 metadata，但 core 丢弃；
- core 已返回 metadata，但 frontend state 提前删除；
- RequestOutput 有 metadata，但 OpenAI streaming schema 丢弃；
- 用户看到两个 final chunk 或丢失 stop reason。

因此本次保持一个可独立 revert 的 end-to-end correctness commit。

### 修改文件及职责

#### Scheduler

- `vllm/v1/core/sched/interface.py`：新增带 metadata 返回值的 finish 接口。
- `vllm/v1/core/sched/scheduler.py`：抽取 `_finish_requests()`；旧 API 仍返回
  `(request_id, client_index)`，新 API 额外返回 `_free_request()` 生成的
  `kv_transfer_params`。

#### EngineCore 和 client transport

- `vllm/v1/engine/__init__.py`：增加 `EngineCoreRequestType.STOP`，不复用
  `ABORT`。
- `vllm/v1/engine/core.py`：正常 finish request，按 `client_index` 分组，生成
  空 token、`FinishReason.STOP` 且带 metadata 的 final output；MP core 把该 output
  放入对应 output queue。
- `vllm/v1/engine/core_client.py`：覆盖 Inproc、SyncMP、AsyncMP 和 DP-LB async
  client。Inproc 直接返回 final output，MP 通过 queue 异步返回，DP-LB 按
  `reqs_in_flight` 路由到原 engine。

#### Frontend lifecycle

- `vllm/v1/engine/output_processor.py`：把 `reqs_to_abort` 主字段改为
  `reqs_to_stop`，保留 deprecated alias；在 first detection 阶段暂存 stop reason 而不
  finish state；在 final empty output 上恢复 stop reason 并只 finish 一次。
- `vllm/v1/engine/async_llm.py`：async 路径发送 STOP。
- `vllm/v1/engine/llm_engine.py`：sync Inproc 路径立即处理返回的 final output；
  MP 仍等 output queue。

`output_processor.py` 还修正了 final empty-token delta logprobs：Python `x[-0:]`
会返回全部列表，所以 token 列表为空时必须显式返回 `[]`，避免最终
chunk 重复历史 logprobs。

#### OpenAI streaming response

- `vllm/entrypoints/openai/chat_completion/protocol.py`：为 chat stream schema 增加
  optional metadata。
- `vllm/entrypoints/openai/chat_completion/serving.py`：从 `RequestOutput` 传入该字段。
- `vllm/entrypoints/openai/completion/protocol.py`：为 completion stream schema 增加
  optional metadata。
- `vllm/entrypoints/openai/completion/serving.py`：从 `RequestOutput` 传入该字段。

`kv_transfer_params` 默认为 `None`，因此非 P/D 请求使用 `exclude_none`
序列化时不会改变 OpenAI response shape。

#### Tests

- `tests/v1/engine/test_output_processor.py`：把 stop-string case 改为两阶段
  STOP，模拟 final metadata output，验证 stop reason、finished 和 metadata。
- `tests/entrypoints/openai/test_streaming_kv_transfer.py`：验证 chat/completion 两个
  streaming schema 都会序列化非空 metadata。移植时按当前仓库规则补充
  SPDX copyright header。

### 正确性验证

定向 pytest：

```text
6 passed, 30 deselected, 22 warnings in 8.03s
```

该结果包含：

- 4 个 stop-string case：`num_sample_logprobs=None/5` 与
  `include_stop_str_in_output=true/false`；
- 2 个 schema case：chat/completion streaming response。

当前机器无 Hugging Face gated `meta-llama/Llama-3.2-1B` 权限。首次完整
`test_output_processor.py` 收集在 fixture 下载阶段得到 403，未进入测试逻辑。
随后在不修改源码的情况下，于 pytest process 启动前把 fixture 的
`TOKENIZER_NAME` 指向已缓存的 OPT-125M snapshot，动态重建 prompt/generation
vectors。

该完整文件运行得到 `27 passed, 2 skipped`；另有 7 个 `test_stop_token`
case 因测试内部显式要求 tokenizer name 必须等于
`meta-llama/Llama-3.2-1B` 而 fail，不是 output 数值或 stop 逻辑断言失败。
最终门禁只选本 commit 直接修改的 stop-string/schema case，得到上述
`6 passed`。

独立 EngineCore 契约检查使用 fake scheduler，确认：

```text
request status:       FINISHED_STOPPED
output client index:  3
new token ids:        []
finish reason:        STOP
kv_transfer_params:   {"remote_block_ids": [[1, 2]]}
result:               passed
```

功能文件的 pre-commit 全部通过，包括 ruff、mypy、typos、SPDX、
lazy imports、forbidden imports、configuration validation 和 `git diff --check`。

### 性能说明

本 commit 是 correctness fix，预期 GPU kernel 数、常规 decode throughput 和显存峰值
收益为零，因此没有申请 GPU 性能卡，也不复用任何 YOCO 算子数据。

新 work 只在 frontend multi-token stop-string 的 terminal path 发生：一条 STOP
control message，一次 scheduler finish，一个 empty-token final output。普通 token step
不增加 work。终态可能比旧 abort 多一次 IPC/queue round trip，但这是返回
connector metadata 的必要成本；性能优先不能覆盖 P/D 正确性。

### 风险、观察点与回滚

- 最高风险是 frontend state 提前删除或 finish 两次；两阶段 test 必须保留。
- `pending_stop_reason` 不应与 streaming-input chunk lifecycle 混用；当前只在
  `not req_state.streaming_input` 时进入等待路径。
- MP 和 DP-LB 必须按 internal request id 路由 STOP；不能广播到其他 engine。
- empty-token delta logprobs 必须为 `[]`，不能回归 `[-0:]`。
- API 字段是 optional vLLM extension，只在 metadata 非空时出现。
- revert `1c51cac0d7` 可恢复旧 abort path，但会恢复 P/D streaming stop
  metadata 丢失问题；前六个 `fhb-dev` commit 不受影响。

---

## 8. UCX 1.21 单 runtime 与 SM100 native 同步

### 提交信息

```text
commit: 2765c22a1b1572e2a9fc8dc16465c04a6dfd47a7
subject: build(pd): unify runtime on UCX 1.21
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 0f8c9d95b550179b662f67d184b12c0688de8ad0
branch: review/yoco-08-ucx-121-runtime
diff: 2 files, 299 insertions
```

该功能在独立 B200 Pod 完成 correctness/runtime 验证后，由 `Snow2022jlu`
直接同步到 `fhb-dev`，不创建 PR。

### 目的和旧 runtime 风险

基础镜像继承 HPC-X UCX 1.20。若再直接安装带 auditwheel repair 的 NIXL wheel，
wheel 可能在私有目录携带 hash 命名的另一套 `libucs/libucp/libuct/libucm`。
此时同一进程可能出现：

```text
HPC-X / vLLM -> /opt/hpcx/ucx/lib (UCX 1.20)
NIXL plugin   -> nixl_cu13.libs/libuc*-<hash>.so (另一版本)
```

即使两边 API 分别可 import，也不能保证同一进程中的 UCX ABI、global state、
memory registration 和 worker 生命周期兼容。P/D 传 KV 时，这类问题通常表现为
握手卡住、backend 初始化失败或运行一段时间后 transport error，而不是稳定的
Python exception。

本 commit 把契约改成：

1. `/opt/hpcx/ucx` 是镜像内唯一 UCX root；
2. 该 root 原位升级为固定 revision 的 UCX 1.21.0；
3. NIXL UCX plugin 只通过 generic SONAME + canonical RUNPATH 链接该 root；
4. image build、容器启动前检查和可选进程 maps 检查都 fail closed；
5. 当前 YOCO Python 与 `_C` native operator 必须来自同一源码状态。

### 固定依赖和可复现输入

```text
base image:
  registry.hub.docker.com/buaahsh/pytorch@
  sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
base vLLM native revision:
  f4964c907db7ce2d77c2b0ea39e263375b7eba4f
UCX revision / version:
  b6a9d47fccce849c28111f05a7fa8f1c930ff17d / 1.21.0
NIXL revision / version:
  de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72 / 1.3.2
CUDA target:
  TORCH_CUDA_ARCH_LIST=10.0
```

base image 使用 digest 而不是 mutable tag；UCX/NIXL 都按完整 commit SHA shallow
fetch，并在 build 中比较实际 `HEAD`。镜像 labels 记录 UCX/NIXL version 和
revision、vLLM native base revision、CUDA architecture。

### 修改文件及职责

#### `docker/Dockerfile.b200.pd`

Dockerfile 使用同一个 pinned base 的 multi-stage build：

- 删除 inherited UCX 1.20 tree，再把 1.21 安装回 `/opt/hpcx/ucx`；不是在
  第二个 prefix 并排安装；
- 保留 HPC-X 既有 prefix，避免 OpenMPI/HPC-X path 失效；
- source build 启用 shared、MT、CUDA、verbs、mlx5、rdmacm、DM 和 GDRCopy，
  关闭不需要的 static/EFA/Java/KNEM/XPMEM；
- 检查 `libuct_ib.so`、`libuct_ib_mlx5.so`、`libuct_rdmacm.so` 和
  `ucx_info` 版本；
- 从固定 NIXL commit 构建 UCX/POSIX platform wheel，不执行 auditwheel repair，
  因此不会复制私有 UCX library；
- NIXL build 显式关闭 tests/examples、headers 和 `nixl_ep`，CUDA arch 固定 100；
- runtime 卸载所有 inherited `nixl/nixl-cu12/nixl-cu13`，只安装本次生成的
  `nixl==nixl-cu13==1.3.2`；
- metadata wheel 即使关闭 EP 仍会生成 `nixl_ep` import shim，因此把该目录
  重命名为 `.disabled`，防止 vLLM 错选不存在的 `nixl_ep_cu13`；
- overlay 当前完整 Python tree，并用本次 source build 的 `_C.abi3.so` 替换
  base `_C`；其他 native extensions 继续使用 pinned base 版本；
- image build 内执行 single-UCX、NIXL version、`has_nixl_ep()==False`、
  operator registration 和 Python compileall 检查。

Dockerfile 没有写死 `UCX_NET_DEVICES`。不同 GPU/rank 对应的 NUMA-local HCA
不同，网卡选择必须由 P/D launch profile 根据部署拓扑提供。

#### `docker/verify_single_ucx.sh`

脚本默认要求 `/opt/hpcx/ucx` 和 `1.21.x`，也允许通过环境变量覆盖预期值：

```text
EXPECTED_UCX_ROOT
EXPECTED_UCX_VERSION
```

检查顺序：

1. canonical `ucx_info` 存在且版本匹配；
2. `/usr/local/ucx` 如果存在，必须 resolve 到 canonical root；
3. 扫描 `/usr`、`/opt`、`/workspace` 中所有 `libucm/libucp/libucs/libuct`
   文件和 symlink，任何 resolve 到 canonical root 之外的副本都失败；
4. 找到 NIXL `libplugin_UCX.so`；
5. `readelf` 必须看到 canonical UCX RUNPATH/RPATH；
6. `ldd` 中所有 UCX dependency 必须 resolve 到 canonical root；
7. 传入 PID 时，`/proc/<pid>/maps` 必须已加载 UCX，且每条 UCX mapping 都
   位于 canonical root。

脚本使用 `set -euo pipefail`，失败直接返回非零，不把错误降级为 warning。

### 提交前发现并拦截的 native mismatch

第一次构建得到的诊断镜像：

```text
image id: 765b69b6587109660c184431a672148522a2f6a445b8ca57a75777004c7c9209
UCX/NIXL check: passed
new Python activation class: present
torch.ops._C.silu_and_mul_with_clamp_fp32: absent
```

原因是最初 recipe 只 `COPY vllm`，而 base `_C` 来自
`f4964c907db7ce2d77c2b0ea39e263375b7eba4f`。第六项功能已在当前分支修改：

```text
csrc/activation_kernels.cu
csrc/ops.h
csrc/torch_bindings.cpp
```

Python `SiluAndMulWithClampFP32` 会在实例化时读取新 operator；只覆盖 Python
必然在 YOCO Shared Expert 启动时失败。该诊断镜像没有被当作最终结果，也没有
提交一个已知不可启动的 recipe。

修复方式不是复制本机 A6000/SM86 `_C`，而是在 builder 中：

1. 检查 base Git HEAD 等于 pinned native revision；
2. 检查 base 没有额外修改这三个 native 文件；
3. 只覆盖当前分支的三个文件；
4. `TORCH_CUDA_ARCH_LIST=10.0` 配置 CMake；
5. 只构建和安装 `_C` target，避免无关 extension 重编；
6. 使用 CUDA driver stub import 新 `_C` 并断言 operator 已注册。

最终 `_C` 构建了 57 个 object，link/install 成功；`cuobjdump` 可见
`sm_100` cubin。文件 SHA256：

```text
c3faa036746f43a50a10e4013021613b851f84b79e0de5c15bf39c7fdef4b298
```

### 最终镜像和 build-time 验证

```text
tag:      vllm-yoco-pd:ucx121-fhb-dev-0804-r2
image id: sha256:02caed17c8793830bd88748baa8fb203489384b1a00263d3312162afb54c0f47
size:     38,002,453,833 bytes
```

build 内结果：

```text
UCX:       1.21.0
revision:  b6a9d47fccce849c28111f05a7fa8f1c930ff17d
NIXL:      1.3.2
NIXL-cu13: 1.3.2
plugin RUNPATH includes: /opt/hpcx/ucx/lib
has_nixl_ep(): False
new fused op registered: True
Python compileall: passed
```

错误注入验证：

- `EXPECTED_UCX_VERSION=9.99` 被拒绝；
- 复制额外 `/usr/local/lib/libucs.so.duplicate-test` 后被拒绝；
- 两种情况都返回非零，证明 fail-closed 生效。

### A6000 runtime smoke

本机用最终镜像、GPU0 做运行态 smoke：

```text
GPU: NVIDIA RTX A6000, compute capability 8.6
new fused op registered: True
NIXL backends: [UCX]
has_nixl_ep(): False
PID single-UCX maps: passed
```

该 `_C` 只编译 SM100，所以 A6000 只验证 import/registration 和 NIXL/UCX，
没有执行 fused CUDA kernel；数值执行必须在 B200 完成，不能拿 A6000 smoke
替代 SM100 correctness。

### B200 correctness 和 CUDA Graph

独立 B200 资源：

```text
pod:  lidong1-yoco-ucx121-native-g1-0804-master-0
node: slc01-cl02-hgx-0346
GPU:  NVIDIA B200, compute capability 10.0
```

本地临时 image tag 未发布到 cluster registry。为确保测试内容仍与最终镜像一致，
从 image 提取 UCX、NIXL packages/plugin、校验脚本和 `_C` 共 216 MiB，覆盖到
相同 pinned base digest；当前 Git Python tree 用 `git archive` 覆盖。覆盖后比较
`_C` SHA256，与最终 image 完全一致。该方法验证的是最终 image runtime 文件，
不是在 Pod 内重新编译另一份 binary。

数值 matrix：

```text
d:       1280
limit:   10.0
dtypes:  FP16, BF16, FP32
tokens:  1, 7, 128, 129, 4096
cases:   15
oracle:  FP32 clamp -> FP32 SiLU -> FP32 multiply -> output dtype cast
result:  all bitwise match (rtol=0, atol=0)
```

附加验证：

- BF16、129 tokens 的 torch library `opcheck` 通过；
- YOCO `SiluAndMulWithClampFP32(..., enforce_enable=True)` CustomOp bitwise；
- CUDA Graph capture 后原地替换 static input，replay output bitwise；
- NIXL 1.3.2 成功实例化 UCX backend；
- `verify-single-ucx <pid>` 确认进程只加载 canonical UCX；
- `has_nixl_ep() == False`。

最终摘要：

```text
B200_RUNTIME_RESULT={
  "bitwise_cases": 15,
  "capability": [10, 0],
  "cuda_graph": "bitwise",
  "custom_op": "bitwise",
  "gpu": "NVIDIA B200",
  "has_nixl_ep": false,
  "nixl_backends": ["UCX"],
  "opcheck": "passed",
  "ucx_net_devices": "mlx5_0:1"
}
```

测试结束后删除 Volcano Job，未占用遗留 B200 资源。

### RDMA HCA 自动选择问题和部署契约

B200 节点同时暴露：

- 12 个 active、具有效 InfiniBand GID 的 `mlx5_0-4`、`mlx5_7-13` HCA；
- 一个 active Ethernet `mlx5_bond_0`，但其 GID 列表为空。

首次不设置 `UCX_NET_DEVICES` 时，UCX 自动选择无 GID 的 bond，NIXL backend
创建失败：

```text
uct_iface_open(rc_verbs/mlx5_bond_0:1) failed: Address not valid
NIXL_ERR_BACKEND
```

这次失败发生在 15 个数值 case、opcheck、CustomOp 和 CUDA Graph 已完成之后；
它不是 fused operator failure。显式设置：

```text
UCX_NET_DEVICES=mlx5_0:1
```

后，NIXL UCX backend 和 PID maps 立即通过，证明 source-built UCX 1.21 的
verbs/mlx5 transport 可工作。不能在 image 中把 `mlx5_0` 硬编码给所有 rank，
因为多 GPU P/D 节点必须选择各 GPU 对应的 NUMA-local HCA。因此最终约定是：

- image 负责版本、library path 和 ABI 单一性；
- deployment profile 负责按 rank 设置有效的 `UCX_NET_DEVICES`；
- 上线前必须排除 active 但无 GID 的 bond/虚拟 RDMA device；
- 换节点或改 GPU placement 后重新做 agent creation 和 PID maps smoke。

统一到 UCX 1.21 能解决“进程同时加载 1.20/1.21 或 wheel 私有副本”的问题，
但不会自动修复无效网卡选择；两者是独立的稳定性条件。

### 提交前静态检查

通过：

- `docker build --check -f docker/Dockerfile.b200.pd .`，无 warning；
- `bash -n docker/verify_single_ucx.sh`；
- pinned shellcheck 对新脚本单独检查；
- `git diff --check`；
- pre-commit 的 typos、Docker dependency graph、configuration validation、
  attention docs、filename 和 suggestion 等所有适用 hook。

仓库 shellcheck pre-commit hook 即使传入单个新文件仍会扫描全仓，因既有
`.buildkite`、integration 和 utility scripts 的历史告警返回 123。新脚本单独
shellcheck 为 0；没有修改或顺手清理无关旧脚本，功能 commit 仅跳过该全仓 hook。

### 性能说明

这是构建/loader/ABI 稳定性 commit，不改变 YOCO forward、scheduler 或 steady-state
请求算法，预期 tok/s、TTFT 和 kernel 数收益为零。因此：

- 不报告虚构的 endpoint 加速；
- 不把第六项 FP32 activation 的 A6000/B200 性能数据重复算到本 commit；
- B200 测试只报告 bitwise、graph、NIXL backend 和 loader correctness；
- UCX 1.21 多节点带宽、P/D KV transfer throughput 和 tail latency 需要真实
  P/D topology 的独立 A/B，不能用单进程 agent 初始化代替。

build 本身为 SM100 `_C` 编译 57 个 object；这是离线 image build 成本，不进入
请求 latency。

### 风险、观察点与回滚

- `UCX_NET_DEVICES` 必须由部署按 GPU/NIC 拓扑设置；无 GID bond 会让 backend
  初始化失败。
- `_C` 是 SM100-only；不能把本机 SM86 binary 复制进 B200 image，也不能期待
  该 image 在 A6000 执行 CUDA op。
- 当前 native overlay 列表只有三个相对 pinned base 发生变化的文件。后续新增
  C++/CUDA 修改时必须同步扩展列表并重编，不能只 `COPY vllm`。
- NIXL EP 有意关闭；KV transfer 可用不等于 NIXL EP MoE 可用。
- verifier 扫描 `/usr`、`/opt`、`/workspace`。若将来有合法 UCX 安装在另一
  prefix，必须先决定是否仍满足“单 UCX”契约，不能直接放宽检查。
- HPC-X 保留相同 prefix，但 UCX ABI 已升级为 1.21；后续 OpenMPI/HCOLL 集成
  验证应继续使用该 image，不应从宿主 bind-mount 回 1.20。
- revert `2765c22a1b` 可从源码删除 recipe/校验脚本；前七个功能 commit 不受
  影响。已生成的 local Docker image 需要单独按 image lifecycle 清理。

---

## 9. B200 YOCO BF16 Triton MoE 调优配置

### 提交信息

```text
commit: 7c03e0cb730bb11a960eb82a30e73185a743508a
subject: perf(moe): add tuned B200 YOCO config
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 66a87747fafb725f2c3c317adb5acc46e6b928ab
branch: review/yoco-09-b200-moe-config
diff: 1 file, 131 insertions
```

该功能在独立 B200 Pod 完成 correctness/performance A/B 后，由 `Snow2022jlu`
同步到 `fhb-dev`。功能 commit 只有配置文件；本节和 `yoco.md` 的说明由后续
独立 docs commit 追加，不把文档 diff 混入性能功能。

### 目的

YOCO 的一类 BF16 routed MoE 使用 128 experts，单次选择 Top-8，expert
intermediate partition 为 1280。vLLM 没有该精确 B200 shape 的 checked-in
配置时会调用通用 `get_default_config`。默认 tile 要覆盖多种 GPU/shape，无法在
每个 batch-token 点都针对 B200 SM100 的并行度、L2 locality 和 pipeline stage
取最优值。

旧分支 `origin/shaohanh/yoco-0731@70c9edb52b` 已包含一个同名 JSON，但它和
Router、RoPE、differential-attention、benchmark、Dockerfile 以及大段
`yoco.py` 修改混在一个 `884 insertions/deletions` 规模的 commit 中，不能作为
单功能提交审核。本次没有 cherry-pick 旧 commit；只提取配置作为候选，并在
当前 `fhb-dev` runtime 上重新验证。

### 修改文件及职责

#### `vllm/model_executor/layers/fused_moe/configs/E=128,N=1280,device_name=NVIDIA_B200.json`

这是唯一功能文件。它提供 16 个 irregular batch grid point：

```text
1, 2, 4, 8, 16, 24, 32, 48,
64, 80, 84, 96, 128, 256, 512, 1024
```

每个 key 记录：

```text
BLOCK_SIZE_M
BLOCK_SIZE_N
BLOCK_SIZE_K
GROUP_SIZE_M
num_warps
num_stages
```

provenance 字段为 `triton_version=3.7.1`。测试容器实际版本也是 3.7.1；不是
拿其他 Triton 版本生成的配置直接宣称可用。

vLLM 的命中文件名来自：

```text
get_config_file_name(E=128, N=1280, dtype=None, block_shape=None)
-> E=128,N=1280,device_name=NVIDIA_B200.json
```

其中 `N=w2.shape[2]`，是 tensor-parallel partition 的 post-SwiGLU expert
intermediate dimension。限制条件如下：

- 设备名必须规范化为 `NVIDIA_B200`；
- local/global expert layout 传入 loader 后必须得到 `E=128`；
- `N` 必须为 1280；
- BF16/FP16 unquantized path 的 dtype selector 为空；
- FP8、INT8、INT4 或 block-quantized path 会生成不同文件名，不命中本文件；
- batch token 不等于已有 key 时选择绝对距离最近的 key。

该 commit 不修改 FusedMoE kernel、Router、YOCO model、scheduler、NIXL、UCX、
Dockerfile 或 API。

### 为什么没有原样复制旧 JSON

第一次 A/B 使用旧文件逐字节内容，SHA256 为：

```text
3ada46e4a55a501f84f1879d4b9e5b52b9d62696c668722edb5da586363cac9b
```

旧 token=1 tile：

```text
BLOCK_SIZE_M=16
BLOCK_SIZE_N=64
BLOCK_SIZE_K=64
GROUP_SIZE_M=1
num_warps=4
num_stages=5
```

在当前 runtime 上，5 轮交替 A/B 中位数为 baseline `65.83 us`、candidate
`66.17 us`，回退 `0.52%`。decode 小 batch 是重要路径，因此没有用其他 batch
点的收益掩盖这个回退。

最终 token=1 改用当前默认 tile：

```text
BLOCK_SIZE_M=16
BLOCK_SIZE_N=64
BLOCK_SIZE_K=128
GROUP_SIZE_M=1
num_warps=4
num_stages=4
```

修订后 baseline/candidate kernel 配置相同。独立 7 轮中位数为
`65.84/65.92 us`，表面差异 `-0.13%`，属于相同配置的 clock/执行顺序噪声。
其余 15 个旧 grid point 均为稳定正收益或持平，保留原调优值。

### B200 环境和被测内容

```text
Volcano Job: lidong1-yoco-moe-config-g1-0805
Pod:         lidong1-yoco-moe-config-g1-0805-master-0
Node:        slc01-cl02-hgx-0380
GPU:         NVIDIA B200
capability:  10.0
base image:  registry.hub.docker.com/buaahsh/pytorch@
             sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
PyTorch:     2.11.0a0+eb65b36914.nv26.02
Triton:      3.7.1
shape:       E=128, N=1280, K=3072, Top-K=8
dtype:       BF16 weights/input/output, FP32 router logits
```

固定镜像提供 SM100 native extensions 和实际部署依赖；测试前覆盖当前
`fhb-dev` Python tree。loader 输出：

```text
Using configuration from /workspace/vllm/vllm/model_executor/layers/
fused_moe/configs/E=128,N=1280,device_name=NVIDIA_B200.json for MoE layer.
```

这条日志同时证明 device-name normalization、dtype selector、`E/N` 和文件路径
均真实命中，不是只把 JSON 直接传给一个脱离 runtime loader 的 microbenchmark。

### 正确性方法和结果

每个 shape 固定 random seed，baseline/candidate 复用相同：

- BF16 hidden states；
- BF16 `w1=[128,2560,3072]`；
- BF16 `w2=[128,3072,1280]`；
- FP32 router logits；
- `fused_topk(..., topk=8, renormalize=True)` 产生的 weights/ids。

baseline 配置来自当前 `get_default_config`；candidate 来自正常
`get_moe_configs` 文件加载。两者分别通过 `override_config` 运行同一个
`fused_experts` 实现，只改变 Triton tile/meta parameters。

最终修订版结果：

| Tokens | Output shape | Elements | Result |
| ---: | --- | ---: | --- |
| 1 | `[1,3072]` | 3,072 | bitwise |
| 8 | `[8,3072]` | 24,576 | bitwise |
| 32 | `[32,3072]` | 98,304 | bitwise |
| 128 | `[128,3072]` | 393,216 | bitwise |
| 512 | `[512,3072]` | 1,572,864 | bitwise |
| 1024 | `[1024,3072]` | 3,145,728 | bitwise |

总计 `5,237,760` 个元素，`mismatches=0`，每组 `max_abs=0`、`mean_abs=0`。
测试使用 `torch.equal`，没有用 BF16 宽容差掩盖 tile 改变可能带来的数值差异。

### 性能方法

使用当前 `benchmarks/kernels/benchmark_moe.py::benchmark_config`，其计时路径：

1. 构造完整 routed expert weights、input 和 100 组 router logits；
2. 对每个 candidate 做 Triton JIT 和一次同步；
3. capture 一个包含 10 次完整 `fused_topk + fused_experts` 的 CUDA Graph；
4. graph replay warmup 5 次；
5. 每个 sample replay 100 次，即计时覆盖 1,000 次完整 path；
6. baseline/candidate 每轮使用同一 seed，奇偶轮交换先后顺序；
7. 全 16 个 grid point 各 5 轮取中位数；
8. 修订后的 token=1 再做 7 轮独立复测。

计时不包含随机 tensor 初始化、JIT 或 graph capture。它包含 routed FusedMoE
kernel path，但不包含完整 YOCO attention、Shared Expert、跨层 universal loop、
网络请求或 P/D 通信。

### 性能结果

| Tokens | Default median us | Tuned median us | 提升 |
| ---: | ---: | ---: | ---: |
| 1 | 65.84 | 65.92 | -0.13%（同配置噪声） |
| 2 | 100.10 | 95.30 | 5.03% |
| 4 | 152.13 | 142.73 | 6.58% |
| 8 | 226.38 | 220.06 | 2.87% |
| 16 | 331.77 | 326.06 | 1.75% |
| 24 | 398.91 | 393.70 | 1.32% |
| 32 | 437.48 | 432.03 | 1.26% |
| 48 | 512.25 | 468.02 | 9.45% |
| 64 | 532.17 | 483.84 | 9.99% |
| 80 | 556.01 | 488.58 | 13.80% |
| 84 | 557.09 | 490.96 | 13.47% |
| 96 | 560.50 | 493.19 | 13.65% |
| 128 | 550.70 | 498.51 | 10.47% |
| 256 | 566.21 | 554.66 | 2.08% |
| 512 | 587.74 | 587.49 | 0.04% |
| 1024 | 735.99 | 689.78 | 6.70% |

最明显收益集中在 48 到 128 tokens，80/84/96 达到约 13.5% 到 13.8%。
token=512 的 tuned/default tile 实际相同，0.04% 是噪声；它保留为显式 grid key，
避免临近选择落到 256 或 1024 的不同 tile。

### 为什么不报告 endpoint tok/s

测试 Pod 的 PVC 中有 YOCO-v3/L3 checkpoint，但其 TP1 配置为
`moe_ffn_dim=3840`，不会命中 `N=1280` 文件；旧 0731 报告所用的
`E=128,N=1280` checkpoint 位于未挂载的 `/mnt/msranlphot`。用 N=3840 模型跑
endpoint 再把结果归因到本配置是错误的，因此本 commit 只报告真实命中的
isolated CUDA Graph A/B。

上线后必须从启动日志确认该文件命中，再依据实际 batch-token histogram 评估
endpoint 收益。不能把 13.8% kernel best point 直接当成 tok/s 提升；完整服务还
包含 attention、Router、Shared Expert、projection、scheduler 和 P/D transfer。

### 提交前检查

通过：

- `jq empty` JSON 解析；
- B200 loader 命中和 Triton 3.7.1 版本核对；
- 全 16 个 grid point 编译、CUDA Graph capture 和执行；
- 最终配置 tokens `1/8/32/128/512/1024` bitwise correctness；
- token=1 修正后独立 7 轮复测；
- `pre-commit run --files <config>` 的全部适用 hook；
- `git diff --check`。

pre-commit 对该 JSON 没有 Python/C++ formatter work；typos、filename spaces、
Docker dependency graph、configuration validation、attention docs 和 suggestion
均通过。

### 风险、观察点与回滚

- JSON 的 `triton_version` 是 provenance，不是 loader enforcement；升级
  PyTorch/Triton/CUDA 后必须重新调优或至少完整 A/B。
- device name、dtype、`E/N` 任一不匹配都会回退 default config；应监控启动日志。
- irregular grid 使用最近 key，线上 batch 分布可能落在两个 key 中间；需结合
  scheduler 的实际 batched-token histogram 观察。
- 本表只覆盖 BF16 unquantized YOCO shape；不能外推到 FP8、EP local-expert
  shape 或其他 GPU。
- token=1 已主动回退当前默认 tile，避免 decode 最小 batch 付出已知代价。
- revert `7c03e0cb73` 只删除该 JSON，运行时自动恢复 default config；无需重编
  `_C`、重建 UCX/NIXL，也不影响前八项功能。

## 10. YOCO BF16 RoPE 单 kernel

### 提交信息

```text
commit: 8abfd6c6d0
subject: perf(yoco): fuse BF16 rotary embedding
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 84906823de26ee012e2080e03f79e4dbbe4f2f0d
branch: review/yoco-10-rotary-kernel
diff: 2 files, 248 insertions, 5 deletions
```

该功能从旧 `origin/shaohanh/yoco-0731` 中单独重做，不 cherry-pick 混合 commit，
也不迁移旧 Router、differential-attention、Dockerfile、UCX/NIXL 或 scheduler
修改。功能和测试保持在两个文件内；本节与 `yoco.md` 由后续独立 docs commit
追加。

### 目的

原 `YOCORotaryEmbedding` 每次先用 `index_select` 取 position 对应的 FP32
cos/sin，再 `chunk`，随后调用 `torch.compile` 函数分别处理 Q 和 K。即使模型
启用 CUDA Graph，这些 kernel 仍存在于 graph replay 内；CUDA Graph 只消除 CPU
launch 提交开销，不会自动把多 kernel 数据流融合成一个 GPU kernel。

本功能用一次 CustomOp launch 同时完成：

1. 按 position 从共享 FP32 cos/sin cache 读取旋转因子；
2. 从 packed-QKV split 的非连续 row stride 读取 Q/K；
3. 以与当前 Inductor fallback 相同的 FP32 算术顺序计算 RoPE；
4. 直接生成连续 BF16 Q/K 输出。

### 修改文件及职责

#### `vllm/model_executor/models/yoco.py`

新增：

- `_yoco_mul_rn`：显式 PTX `mul.rn.f32`；
- `_yoco_fma_rn`：显式 PTX `fma.rn.f32`；
- `_yoco_rotary_kernel`：Q/K 合并、rotary-pair 扁平化 Triton kernel；
- `_yoco_rotary_cuda`：分配连续输出并以 `BLOCK_SIZE=256`、`num_warps=8` 启动；
- `_yoco_rotary_fake`：为 compile/export 提供与真实输出相同的 shape、dtype、
  device 和连续 stride；
- `torch.ops.vllm.yoco_rotary`：通过 `direct_register_custom_op` 注册；
- `YOCORotaryEmbedding.forward` 快速路径。

快速路径条件为：

```text
HAS_TRITON
CUDA platform and CUDA input
query/key dtype == BF16
query/key last-dimension stride == 1
head_dim == 128
```

其他 dtype、head size、设备或 layout 继续使用原
`_yoco_apply_rotary_emb`。本 commit 不改变 cos/sin cache 的生成方式、最大
position、RoPE base 或 half-rotation 定义。

kernel 允许 query/key row stride 和 head stride 不同，因而可直接消费：

```text
qkv:   [tokens, (Q + 2 * KV) * 128]
query: stride = [packed_width, 128, 1]
key:   stride = [packed_width, 128, 1]
```

输出使用各自 `[tokens, heads, 128]` contiguous layout，和原 `cat` 结果一致。

#### `tests/model_executor/test_yoco_conversion.py`

新增两个测试族：

- `test_yoco_rotary_cuda_matches_compiled_fallback`：两套 head layout、3 套 token
  数、真实 packed QKV split，要求 Q/K `rtol=0, atol=0`；
- `test_yoco_rotary_custom_op_opcheck`：直接用 packed split 的非连续输入运行
  `torch.library.opcheck`，验证 schema、fake/meta、autograd registration 和动态
  compile-facing contract。

### 数值问题和最终算术顺序

旧 0731 kernel 在 position=0 时看起来一致，但 position>0 会出现少量 BF16
mismatch。实测最大一组为：

```text
elements:   1,048,576
mismatches: 19
max_abs:    0.0078125
```

旧测试使用宽松 `atol=4e-3`；本轮不接受用 tolerance 掩盖一个可以保持的现有
bitwise contract。检查当前 PyTorch 26.02 Inductor 输出 PTX 后确认，fallback
不是把四个乘法结果都先 round 后再相加，而是：

```text
second_product = mul.rn.f32(second_input, second_factor)
output = fma.rn.f32(first_input, first_factor, +/-second_product)
```

若直接写普通 Triton `x1 * cos - x2 * sin`，编译器的 contraction/rounding 选择
不保证等同该顺序。最终 kernel 用 inline PTX 明确一个 rounded second product，
再对 first product 与最终 add/sub 使用 rounded FMA，因此在当前 runtime 上逐元素
bitwise 等价。

### kernel 迭代和大 batch 回退处理

没有只凭小 token 点决定保留。迭代过程如下：

1. 旧 head-per-program 版本每个 program 处理一个 Q/K head，小 batch 提升约
   `25%～112%`，但 tokens=512 已回退 `12%～18%`，8K 约回退 40%；
2. 第一版 flat-output kernel 减少 program 粒度，tokens=512 转为
   `+12.51%～+16.59%`，但 48/4 的 1K 和 8K 仍分别为 `-8.98%`、
   `-12.42%`，64/8 的 8K 为 `-11.51%`；
3. 最终按 64 个 rotary pair 扁平化。一个线程只加载一次 `x1/x2` 并同时写两半
   输出，Q/K 合入同一一维 grid，program 数约为 head-per-program 版的四分之一。

最终版本从 1 到 8192 tokens 的全部被测点均为正收益，因此没有在 Python wrapper
中按动态 token shape 切换 kernel/fallback，也没有给 `torch.compile` 引入额外
shape guard。

### B200 正确性验证

```text
Volcano Job: lidong1-yoco-rotary-g1-0805
Pod:         lidong1-yoco-rotary-g1-0805-master-0
Node:        slc01-cl02-hgx-0297
GPU:         NVIDIA B200
PyTorch:     2.11.0a0+eb65b36914.nv26.02
Triton:      3.7.1
dtype:       BF16 Q/K, FP32 cos/sin cache, int64 positions
head_dim:    128
```

correctness matrix：

```text
(query_heads, key_heads) = (48, 4), (64, 8)
tokens                   = 1, 17, 128
positions                = arange(tokens) * 3
input                     = packed QKV split
comparison                = rtol=0, atol=0
```

结果：6 组 Q/K 全部 bitwise，CustomOp packed-stride `opcheck` 通过；rotary
专项 `7 passed, 23 deselected`。随后完整
`tests/model_executor/test_yoco_conversion.py` 为 `30 passed`，覆盖本分支已有
RMSClip、fused add-RMSNorm、Router、Shared Expert、fast prefill 和转换测试。

### B200 CUDA Graph 性能

baseline 为原 `cache.index_select -> chunk -> _yoco_apply_rotary_emb`，其中 apply
已经是 `torch.compile`；candidate 为 `torch.ops.vllm.yoco_rotary`。两者使用相同
input、cache 和 position，并在计时前再次 `torch.equal`。

每个 path 独立 CUDA Graph capture；capture 后 replay warmup 10 次；每个 sample
连续 replay 200 次；每个 shape 做 7 轮，奇偶轮交换 baseline/candidate 顺序，
报告中位数。JIT、graph capture、随机输入和 output 检查不计时，cache position
gather 与 Q/K rotation 均计时。

| Q heads / KV heads | Tokens | Baseline us | Fused us | 提升 |
| --- | ---: | ---: | ---: | ---: |
| 48 / 4 | 1 | 6.1582 | 3.4142 | 80.37% |
| 48 / 4 | 17 | 6.8115 | 3.4042 | 100.09% |
| 48 / 4 | 128 | 8.2082 | 4.1144 | 99.50% |
| 48 / 4 | 512 | 14.3458 | 10.2542 | 39.90% |
| 48 / 4 | 1,024 | 20.5027 | 16.4058 | 24.97% |
| 48 / 4 | 4,096 | 69.6250 | 59.4386 | 17.14% |
| 48 / 4 | 8,192 | 131.0243 | 118.7507 | 10.34% |
| 64 / 8 | 1 | 6.1562 | 3.3333 | 84.69% |
| 64 / 8 | 17 | 8.2056 | 3.1910 | 157.15% |
| 64 / 8 | 128 | 10.2571 | 6.1570 | 66.59% |
| 64 / 8 | 512 | 18.4424 | 12.3091 | 49.83% |
| 64 / 8 | 1,024 | 28.6798 | 22.5331 | 27.28% |
| 64 / 8 | 4,096 | 96.2554 | 81.9664 | 17.43% |
| 64 / 8 | 8,192 | 184.7998 | 163.7840 | 12.83% |

`提升 = baseline / fused - 1`。14 个点全部提升，范围为
`10.34%～157.15%`。这是 isolated RoPE latency，不包含完整 attention、MoE、
collective、scheduler 或 P/D transport，不能把该百分比直接当作 endpoint tok/s。

### 提交前检查

通过：

- `python -m py_compile vllm/model_executor/models/yoco.py`；
- 两个修改文件的完整 `pre-commit run --files ...`；
- `git diff --check`；
- B200 rotary 专项 7 项；
- B200 完整 YOCO conversion 30 项；
- 两套 head layout、14 个 shape 的 CUDA Graph correctness/performance A/B。

### 风险、观察点与回滚

- inline PTX 刻意匹配当前 PyTorch 26.02 Inductor 的 contraction 顺序；升级
  PyTorch/Triton/CUDA 后必须重新核对 PTX 和 bitwise matrix；
- kernel 只覆盖 BF16、head_dim=128、YOCO half-rotation，不外推到 FP16、FP32、
  interleaved RoPE 或其他模型；
- position 必须在 cache 范围内，该约束与原 `index_select` path 相同；
- CUDA Graph 已用于性能测试，结果证明优化不是依赖 eager launch latency；
- revert `8abfd6c6d0` 会恢复原 compiled fallback，不影响此前 Router、MoE、
  SwiGLU、UCX/NIXL 或 P/D 功能。

## 11. Differential-attention CustomOp 负结果

### 提交信息

```text
subject: docs(yoco): record rejected diff-attention fusion
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 625ff8ea06fd0c609aee72580aff12e7ef2b6ea1
branch: review/yoco-11-diff-attention-kernel
functional code diff: none
functional behavior change: none
```

本项按“先实现、正确性和性能验证、再决定是否提交”的流程执行。最终只提交本节
和 `yoco.md` 的实验记录；所有候选 kernel、CustomOp 接入和临时测试均在提交前
撤回。该 docs commit 不应被理解为上线了 differential-attention 新 kernel。

### 复核对象

旧 `origin/shaohanh/yoco-0731@70c9edb52b` 在一个 884 行混合 commit 中新增：

```text
_yoco_diff_attention_kernel
_yoco_diff_attention_cuda / fake
torch.ops.vllm.yoco_diff_attention
self/cross _diff_attention_combine fast path
```

旧测试只覆盖 tokens `1/33/128`、24 head-pairs，并使用
`rtol=1e-2, atol=1e-2`。旧提交没有提供单独的 CUDA Graph 全 shape A/B，不能
直接凭“fused”命名迁移到当前 `fhb-dev`。

### 为什么当前实现已经融合

当前代码为：

```text
diff-v2: attn1 - sigmoid(gate) * attn2
diff-v3: attn1 * sigmoid(gate1) - attn2 * sigmoid(gate2)
```

两个 helper 都有 `@torch.compile`。本轮使用隔离的 TorchInductor cache 和
`TORCH_COMPILE_DEBUG=1` 检查生成代码，v2 的拓扑为：

```text
triton_poi_fused_mul_sigmoid_sub_unsqueeze
```

v3 为：

```text
triton_poi_fused_mul_sigmoid_slice_sub_unsqueeze
```

每个版本都只有一个 pointwise Triton kernel。生成 PTX 中 v2 使用负 sigmoid
与 `attn2` 的 `fma.rn.f32`；v3 先计算第二项乘积，再用第一项乘法与最终减法的
`fma.rn.f32`。`attn[:, 0::2]`、`attn[:, 1::2]` 和 reshape 是 view，不增加
kernel launch。CUDA Graph 会 replay 这个单 kernel，因此不存在“打开 graph
却仍有多个 combine launch”的问题。

### 候选实现

为避免只复测旧代码，本轮实现并比较过四类映射：

1. 接近旧版的 flat output / `BLOCK_SIZE=256, num_warps=4`；
2. `2 pairs/block, 8 warps`，尝试复用每个 head-pair 的 gate/sigmoid；
3. `1 pair/program, 4 warps`，使一个 program 对应 128 维 head；
4. flat output / `BLOCK_SIZE=1024, num_warps=4`，提高每个 CTA 的连续工作量。

所有版本都直接读取交错的 `attn1/attn2`，支持实际 row/head stride，并输出连续
`[tokens, head_pairs, 128]`。数值实现先对照 PTX 使用显式 rounded multiply/FMA，
也测试了让 Triton 3.7 对与 Inductor 相同的表达式自行 contraction。后者同样能
保持 bitwise，且性能更好。

### B200 环境和正确性

```text
Volcano Job: lidong1-yoco-diff-attn-g1-0805
Pod:         lidong1-yoco-diff-attn-g1-0805-master-0
Node:        slc01-cl02-hgx-0206
GPU:         NVIDIA B200
PyTorch:     2.11.0a0+eb65b36914.nv26.02
Triton:      3.7.1
dtype:       BF16 attention/gate/output
head_dim:    128
```

correctness matrix：

```text
diff version: v2, v3
head-pairs:   24, 32
tokens:       1, 17, 128
comparison:   rtol=0, atol=0
```

12 组全部 bitwise；v2/v3 两组 `torch.library.opcheck` 通过，总计临时专项
`14 passed`。这证明放弃原因不是精度做不对，而是无法提供无回退的通用性能。

### 性能方法

baseline 为当前 `@torch.compile` helper，candidate 为实验 CustomOp。每个 shape：

1. 使用相同随机 BF16 attention 和 gate；
2. 计时前 `torch.equal`；
3. baseline/candidate 各自独立 CUDA Graph capture；
4. graph replay warmup 10 次；
5. 每个 sample replay 200 次；
6. 共 7 轮，奇偶轮交换执行顺序；
7. 报告 7 个 sample 的中位数。

计时不包含 JIT、graph capture、随机输入或 correctness 检查。矩阵覆盖 tokens：

```text
1, 17, 128, 512, 1024, 4096, 8192
```

### 接近旧版映射的结果

`2 pairs/block, 8 warps` 在小 shape 偶有约 0%–2% 波动，但从 512 tokens 起
持续回退：

| Head-pairs / 版本 | 512 | 1,024 | 4,096 | 8,192 |
| --- | ---: | ---: | ---: | ---: |
| 24 / v2 | -24.88% | -16.64% | -26.35% | -37.21% |
| 24 / v3 | -24.93% | -27.45% | -33.66% | -43.49% |
| 32 / v2 | -25.08% | -42.81% | -53.73% | -59.57% |
| 32 / v3 | -19.94% | -24.94% | -33.61% | -38.26% |

原因是 B200 Inductor baseline 已经是高效 flat pointwise kernel。按 head pair
切分增加 CTA/layout 开销；所谓 gate 复用不足以抵消内存访问和并行度损失。

### flat-1024 的收益和不能上线的原因

增大 flat block 后，部分大 shape 明显改善：

| Head-pairs / 版本 | 1 | 17 | 128 | 512 | 1,024 | 4,096 | 8,192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 / v2 | -0.78% | -0.53% | -1.93% | +0.05% | +24.82% | +16.64% | +19.65% |
| 24 / v3 | -17.33% | -28.14% | -0.15% | +0.04% | -0.05% | -7.21% | -7.75% |
| 32 / v2 | +0.02% | -0.18% | +0.23% | +0.11% | +32.27% | +30.94% | +35.38% |
| 32 / v3 | +2.34% | +38.74% | +0.10% | +33.28% | +49.84% | +53.13% | +54.17% |

这是强烈的 shape/version-dependent tradeoff，不是通用收益。尤其 24/v3 同时
覆盖真实 YOCO layout，不能因为 32/v3 最好点 `+54.17%` 就忽略它的
`-28.14%` 和大 batch `-7%～-8%`。

可以在 wrapper 中根据 token/head/version 选择 kernel，但这会引入动态 shape
guard、多配置维护和更多 compile/capture 组合；在没有真实线上 shape histogram
及 endpoint A/B 的情况下，不值得为 isolated 局部 best points 增加复杂度。

### 最终决定和后续方向

- 候选 `yoco.py`、测试和 benchmark 脚本均不进入 Git；
- 现有 compiled single-kernel implementation 保持不变；
- 本提交只有 `yoco.md` 与 `fhb-dev-commit.md`，用于避免以后重复做同一迁移；
- 后续若继续，应研究把 combine 与 `lambda_proj` 或 `o_proj` 跨边界融合，或先
  收集生产 token/head/version histogram，再设计经过 endpoint 验证的多配置方案。

回滚本 docs commit 只删除记录，不改变任何运行行为。

---

## 12. YOCO B200 multigpu long-context 合并

### 提交信息

```text
commit: 9106abeb3ad6963b95083688538e53040500e9b8
subject: Merge YOCO B200 multigpu long-context support
first parent: 26614af40fa4e5edfc63143a492ec1777a6ff2ff (fhb-dev)
second parent: 85f7d2ac1b (origin/shaohanh/yoco-b200-longctx-multigpu-20260804)
author/committer: 方涵斌 <2190556589@qq.com>
review branch: integration/fhb-dev-yoco-b200-longctx-multigpu-20260804
```

本次按最终部署方向直接合并 multigpu 分支；此前讨论的单 GPU
`yoco-b200-longctx-0804` 不在合并范围内，也没有作为中间 parent。

### 目的和合并边界

`fhb-dev` 已包含 P/D 裁剪、KV-sharing 修复、Router 简化、fused add-RMSNorm、
Shared/Routed MoE 并行、FP32 clamped-SwiGLU、UCX 1.21/NIXL 以及 BF16 RoPE 等
独立提交。multigpu 分支补充的是 B200 长上下文部署和可复现测试工具。本 merge 的
目标是同时保留两组能力，并使用 multigpu 分支自己的方法完成 DP8 验收。

merge 引入或更新以下 12 个文件：

```text
benchmarks/multi_turn/benchmark_agent_trace.py
docker/Dockerfile.b200.longctx
long_context/README.md
long_context/presentation_tables.md
tools/yoco_serving/benchmark_long_context.sh
tools/yoco_serving/launch_nested_docker.sh
tools/yoco_serving/probe_moe_n1280.py
tools/yoco_serving/serve_long_context.sh
tools/yoco_serving/tune_moe_b200.sh
tools/yoco_serving/warmup_long_context.sh
vllm/model_executor/layers/fused_moe/configs/E=128,N=1280,device_name=NVIDIA_B200.json
yoco.md
```

唯一文本冲突是 N1280 MoE JSON。最终完整采用 multigpu parent 的 hybrid 配置，
并用 `git diff --exit-code 85f7d2ac1b -- <config>` 确认逐字节一致。这样 DP1 的
小 M fallback 与大 M tuned 配置、该分支的表格及复现命令不会被冲突解决悄悄
改变。除该选择外没有手工重写模型或 benchmark 逻辑。

### 提交前静态和单元测试

合并后通过：

- 修改 Python 文件的 `py_compile`；
- 9 个 shell 工具的 `bash -n`，以及 changed-shell 定向 `shellcheck`；
- N1280 JSON parse/结构检查；
- changed files 的 pre-commit；仓库级 shellcheck 仍会报告 `.buildkite/` 等
  未修改历史脚本告警，不归因于本 merge；
- `tests/model_executor/test_yoco_conversion.py`：`30 passed, 14 warnings`。

仓库根目录存在历史坏链接 `agens_tokenizer_0622`，直接 pytest collection 会在
收集阶段失败。本次在 `/tmp/yoco-pytest-9106abeb3a` 使用隔离 `pytest.ini`
运行相同测试文件，避免把无关坏链接当作模型失败。

### B200 镜像和运行环境

```text
Pod: assuring-owl-b200g4-dev-d5aab19e-master-0
Node: slc01-cl02-hgx-0202
GPU: 8 x NVIDIA B200
Model: /data/models/yoco-0000-0800-hf
Baseline image: buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-multigpu-20260804
Candidate source: 9106abeb3ad6963b95083688538e53040500e9b8
Candidate image id: sha256:dbef8b82896fc9257f1eb45acb6b90a2a79eafd440d629a37e162ca3b846738d
Precision: BF16
TP / DP: 1 / 8
Attention / MoE: FlashInfer / Triton
Scheduler / DP sync: async / Gloo
max model len: 131072
max batched prefill tokens: 32768
CUDA Graph: FULL_AND_PIECEWISE
```

候选镜像从已验证 multigpu 镜像覆盖完整合并后 Python tree，并为 SM100 重编译
`_C`。构建后显式验证 `torch.ops._C.silu_and_mul_with_clamp_fp32` 存在，避免
“Python 是 fhb-dev、native 仍是旧镜像”的伪合并。

该嵌套 Docker daemon 的 base 已接近最大父层深度；逐文件 overlay 在最终 commit
layer 时触发 `max depth exceeded`。完整 `_C` 编译和导入检查已经在退出码 0 的
中间容器内完成，因此把该容器 `docker export | docker import` 为单层验证镜像，
并恢复 base 的环境、workdir 和 NVIDIA entrypoint。扁平化后再次导入 native op
通过。最初直接构建 `Dockerfile.b200.pd` 则在拉取 GitHub 依赖前被嵌套 Docker
DNS 阻断，未进入代码编译；不能记成代码失败。

### 端点正确性

baseline 和 candidate 各测试 8,192/65,536 token prompt，每种 shape 做两次
greedy 生成，固定 256 output tokens，检查 `finish_reason=length`、token id 范围、
同 shape repeat hash，以及 baseline/candidate hash。

| Prompt | Baseline SHA256 | Candidate SHA256 | 结果 |
| ---: | --- | --- | --- |
| 8,192 | `9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a` | 同左 | 两次一致，跨镜像一致 |
| 65,536 | `345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983` | 同左 | 两次一致，跨镜像一致 |

四个 candidate 请求和四个 baseline 请求均精确输出 256 token，无失败或提前 EOS。
这比只比较文本或宽松 logit tolerance 更严格，但仍不是整个训练评测集的质量分数。

### multigpu 原方法性能测试

两边都先执行完整 8 轨迹、40 turns、130K final context warmup。candidate 随后按
`tools/yoco_serving/benchmark_long_context.sh` 原样运行 DP8/batch8 的 W1/W2/W3；
single-turn 请求使用独立 cache salt，W3 使用逐 trajectory 的
`X-data-parallel-rank`。

| Workload | 公开 multigpu 基线 | 合并版 | 变化 |
| --- | ---: | ---: | ---: |
| W1 wall time | 692.102 s | 717.54 s | +3.68% |
| W1 output tok/s | 757.53 | 730.67 | -3.55% |
| W1 mean TTFT | 0.533 s | 0.772 s | +44.8% |
| W1 mean TPOT | 10.55 ms | 10.94 ms | +3.70% |
| W2 wall time | 176.693 s | 185.94 s | +5.23% |
| W2 output tok/s | 741.81 | 704.90 | -4.98% |
| W2 mean TTFT | 2.751 s | 2.222 s | -19.2% |
| W2 mean TPOT | 10.61 ms | 11.21 ms | +5.66% |
| W3 wall time | 152.385 s | 138.432 s | -9.16% |
| W3 output tok/s | 682.48 | 751.27 | +10.08% |
| W3 mean TTFT | 0.331 s | 0.325 s | -1.71% |
| W3 mean ITL | 10.73 ms | 9.624 ms | -10.31% |
| W3 prefix-cache hit | 95.58% | 95.58% | 持平 |

W1/W2 都是 8/8 请求成功、0 失败，但原 single-turn harness 不给每个请求设置 DP
rank header。W1 实测只有 7 个 rank 活跃、一个 rank 同时处理 2 个请求；W2 只有
6 个 rank 活跃、两个 rank 各处理 2 个请求。因此 W1/W2 的 `-3.55%/-4.98%`
不能直接归因于代码，既不作为合并回退 gate，也不隐藏在报告之外。后续若要把
single-turn 数字用于 commit 归因，脚本应为 8 个并发请求分别固定 rank。

### 同节点、显式绑 rank 的 W3 A/B

为排除公开基线的节点和时间差异，在同一个 Pod 上删除 candidate 服务、冷启动
原 multigpu 镜像，重新执行完整 40-turn warmup，再以不同 cache salt 运行 W3。
两边均固定 8 个 trajectory 到 8 个 DP rank：

| 指标 | 同节点 baseline | 合并版 | 合并版变化 |
| --- | ---: | ---: | ---: |
| Wall time | 168.010 s | 138.432 s | -17.61% |
| Output tok/s | 619.01 | 751.27 | +21.37% |
| Mean TTFT | 0.245 s | 0.325 s | +32.80%（约 +80 ms） |
| Mean ITL | 12.104 ms | 9.624 ms | -20.49% |
| Prefix-cache hit | 95.58% | 95.58% | 持平 |
| Mean GPU utilization | 91.98%-93.87% | 94.35%-95.41% | 更高且更均衡 |

因此该 merge 的可归因结论是：长 session decode/agentic 吞吐明显提高，ITL 明显
降低，没有出现性能回退；代价是该轮 mean TTFT 增加约 80 ms。不能用 W3 的
`+21.37%` 外推所有 single-turn shape，也不能忽略 TTFT tradeoff。

candidate 冷启动日志还确认 full CUDA Graph 已实际 capture，不是 eager 测试：
engine init 约 160 s、compile 约 42 s、实际 graph pool 约 0.25 GiB；同节点 baseline
约为 195 s、73 s、3.12 GiB。性能差异不是“candidate 没开 CUDA Graph”造成的。

### 原始证据、限制和回滚

原始 JSON、逐 turn JSONL、runtime samples、accuracy JSON 和两边 service log：

```text
/data/fhb-dev-multigpu-results-20260805
```

本轮验证的是单节点 DP8 长上下文 serving，不是 1P2D transport 验收；它不替代
已有 UCX/NIXL P/D 测试，也不证明跨节点网络行为。DeepEP 在两边均因基础镜像
NVSHMEM symbol 不匹配而不可导入，实际都按日志回退到
AllGather+ReduceScatter，故不构成本次 A/B 的变量。

回滚可对 merge commit 使用 `git revert -m 1 9106abeb3a`，这会移除 multigpu
工具、报告和 hybrid N1280 配置，但保留 merge 第一 parent 上全部 `fhb-dev`
优化。不要直接 reset 共享 `fhb-dev`。

### DP1 / DP4 补充验收（2026-08-05）

本轮是 commit 12 合并版的补充实机验证，不增加新的功能 commit，也不改变 Python、
CUDA、Triton、Dockerfile、UCX/NIXL 或 serving 参数。文档提交只修改
`yoco.md` 和 `fhb-dev-commit.md`；按维护规则不把文档提交自身递归加入提交索引。

#### 目的和口径

此前合并后只有 DP8/batch8 的完整数据。本轮用同一个已校验 candidate image
`sha256:dbef8b82896fc9257f1eb45acb6b90a2a79eafd440d629a37e162ca3b846738d`
补测 DP1/batch1 与 DP4/batch4。每种部署都冷启动，先对 8K/65K prompt 各做两次
greedy 256-token 正确性，再执行完整 40-turn/130K warmup，最后依次跑 W1/W2/W3。

```text
runtime source: 9106abeb3ad6963b95083688538e53040500e9b8
Pod: lidong1-yoco-fhb-dev-dp14-g4-0805-master-0
Node: slc01-cl02-hgx-0346
GPU: NVIDIA B200; physical 2 for DP1, physical 2,3,4,5 for DP4
Model: /data/models/yoco-0000-0800-hf
TP: 1
BF16 / FlashInfer / Triton MoE / async scheduling
max model len / max batched tokens: 131072 / 32768
CUDA Graph: FULL_AND_PIECEWISE
```

节点另外四张 GPU 0、1、6、7 被同租作业使用，但被测卡没有进程重叠，结束后
GPU 2--5 均为 `0 MiB / 0%`。这使本轮适合作为同配置回归，不是严格的同节点同时
A/B，性能变化不能单独归因给 commit 12 中任一优化。

#### 正确性

DP1 和 DP4 的 8 个请求全部生成精确 256 tokens、`finish_reason=length`，同 shape
重复一致，并与公开 baseline 哈希相同：

```text
8K:  9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a
65K: 345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983
```

#### 性能结果

| 部署 | Workload | 公开基线 tok/s | 本轮 tok/s | 变化 | 本轮 wall |
| --- | --- | ---: | ---: | ---: | ---: |
| DP1/b1 | W1 | 122.21 | 138.88 | +13.64% | 471.895 s |
| DP1/b1 | W2 | 118.74 | 134.49 | +13.27% | 121.820 s |
| DP1/b1 | W3 | 114.77 | 129.25 | +12.61% | 100.582 s |
| DP4/b4 | W1 | 385.75 | 452.78 | +17.38% | 578.962 s |
| DP4/b4 | W2 | 420.64 | 440.59 | +4.74% | 148.746 s |
| DP4/b4 | W3 | 394.82 | 411.67 | +4.27% | 126.314 s |

延迟与 cache：

| 部署 | Workload | Mean TTFT | Mean TPOT / ITL | Cache hit |
| --- | --- | ---: | ---: | ---: |
| DP1/b1 | W1 | 177.6 ms | TPOT 7.198 ms | N/A |
| DP1/b1 | W2 | 1.159 s | TPOT 7.365 ms | N/A |
| DP1/b1 | W3 | 163.9 ms | ITL 7.249 ms | 95.58% |
| DP4/b4 | W1 | 267.9 ms | TPOT 8.830 ms | N/A |
| DP4/b4 | W2 | 1.631 s | TPOT 8.979 ms | N/A |
| DP4/b4 | W3 | 247.4 ms | ITL 8.975 ms | 95.58% |

所有 W1/W2 请求成功。DP4 single-turn 虽未显式绑 rank，但采样期间 engine 0--3
各有一个 running request、waiting 为 0，generation-token 计数等量推进，没有旧
DP8 测试的负载不均。W3 显式绑 rank；DP1 平均 GPU 利用率 `95.99%`，DP4 四卡
为 `94.83%--95.99%`，两边 waiting 均为 0。

#### 证据和边界

完整 warmup 日志确认覆盖后段 YOCO RMSNorm、fused add-RMSNorm、Router renorm 与
MoE JIT；service log 确认实际 capture CUDA Graph。DeepEP 仍因基础镜像 NVSHMEM
symbol 不匹配回退 AllGather+ReduceScatter，与公开 multigpu 方法一致。

原始 accuracy、W1/W2/W3 detailed JSON、逐 turn JSONL、runtime samples、container
inspect 和 service log 位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/fhb-dev-dp14-20260805/results
```

本轮只验证单节点 DP serving，不替代 1P2D UCX/NIXL transport 验收。回滚不需要
改 runtime；若只撤销本轮记录，revert 对应文档提交即可。

## 13. 缓存 FP32 Router 归一化权重

### 提交信息

```text
commit: 4364a965012e0cbe66b5f247c66f609914048b8c
subject: perf(yoco): cache normalized router weights
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 0223cd9099
branch: review/yoco-12-router-weight-cache
diff: 2 files, 102 insertions, 5 deletions
```

### 目的与旧路径开销

当前 YOCO-v2 checkpoint 没有设置 `router_weights_normalized`，因此运行时按旧格式
处理原始 FP32 gate weight。旧路径每次执行 `YOCOMoE.forward` 都先按 expert row
计算 L2 norm、执行 `clamp_min(1e-6)` 并生成一份归一化权重，然后才计算 Router
linear。

模型有 20 个不同的 MoE module；其中 10 个 self layer 因
`universal_loop=3` 各执行三次，另外 10 个 cross layer 各执行一次，因此每个完整
model step 一共调用 Router 40 次。推理期间 gate weight 不变，旧路径却在每次
调用中重复归一化同一组权重。本提交把这项与 token 无关的计算移到权重加载结束
时，每个 MoE module 只计算并缓存一次。

### 实现与行为边界

`YOCOMoE` 新增 non-persistent buffer `_normalized_gate_weight`。当 checkpoint 的
`router_weights_normalized=false` 或字段缺失时，缓存：

```text
gate.weight / gate.weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
```

forward 随后把缓存权重传给 `yoco_router_linear_tf32`，并关闭该次调用内部的重复
归一化。CPU fallback 同样使用缓存后的权重，因此 CUDA 与 fallback 维持同一个
契约。

- 标准 `load_weights()` 完成后立即初始化全部本地 `YOCOMoE` 缓存；
- 非标准 loader 若绕过 `load_weights()`，第一次 model forward 会兜底初始化；
- 重复调用 `load_weights()` 会先清除初始化标志并重新生成缓存；
- checkpoint 已声明 `router_weights_normalized=true` 时不分配缓存，继续直接使用
  原 gate weight；
- buffer 设置为 `persistent=False`，不会增加或改变 checkpoint/state_dict key；
- 本优化依赖 serving 期间权重不可变。若未来支持运行时原地修改 Router weight，
  修改方必须显式刷新缓存。

每个 gate weight shape 为 `[128, 3072]`、dtype 为 FP32；20 份缓存额外占用
`31,457,280 bytes`，即 `30 MiB` GPU 显存。它用固定的 30 MiB 换取每个 model
step 的重复计算消除，不改变 CUDA Graph pool 或 checkpoint 大小；因为常驻模型
内存增加，自动计算的 KV cache 显存预算可能相应减少约 30 MiB。

### 修改文件及职责

#### `vllm/model_executor/models/yoco.py`

- 注册 non-persistent Router weight cache；
- 增加单 module 初始化方法和 model 级一次性初始化管理；
- 标准 load 完成后 eager 初始化，并为非标准 loader 保留 first-forward fallback；
- CUDA CustomOp 与 CPU fallback 都根据缓存状态选择相同的已归一化权重。

#### `tests/model_executor/test_yoco_conversion.py`

- 验证缓存张量与旧 runtime normalization 逐位一致；
- 验证缓存不进入 `state_dict()`；
- 验证已归一化 checkpoint 不创建冗余缓存；
- 在 CUDA 上覆盖 tokens=`1/8/128`，比较“原权重并在 op 内归一化”和“缓存权重且
  op 内不归一化”，要求 Router logits `rtol=0, atol=0`。

### 正确性和静态检查

本地最终检查：

```text
tests/model_executor/test_yoco_conversion.py: 35 passed, 14 warnings
changed-files pre-commit: passed
  包含 ruff-check、ruff-format、mypy-local、SPDX 和配置检查
git diff --check: passed
```

pytest 使用 `-p no:cacheprovider --confcutdir=tests/model_executor`，用于避开仓库根
目录历史损坏挂载 `agens_tokenizer_0622`；它不跳过本测试文件中的测试。B200
隔离 Job 上完整 YOCO 文件同样为 `35 passed`。

B200 另外覆盖 Router weight `[128, 3072]` 和 tokens=`1/4/8/32/128/1024`。
所有 shape 的 eager 输出和 CUDA Graph capture 输出均逐位一致，统一为：

```text
bitwise_exact: true
max_abs_diff: 0
```

### B200 CUDA Graph 性能

```text
Node: slc01-cl02-hgx-0013
GPU: NVIDIA B200
image: buaahsh/pytorch@sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
weight: FP32 [128, 3072]
hidden states: FP32 [tokens, 3072]
graph: 40 consecutive Router calls
warmup: 100 graph replays
sample: 200 replays per sample, 9 samples, report median
```

baseline graph 每个 Router call 都从原 weight 重新归一化；candidate graph 使用
load-time cache。两条路径分别 capture CUDA Graph，cache 初始化、JIT 和 graph
capture 均不计时。40 次调用对应当前 YOCO 一次完整 model step 的 Router 调用数。

| Tokens | Baseline 40 calls | Cached 40 calls | 节省 | 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 608.57 us | 231.76 us | 376.82 us | 2.63x |
| 4 | 767.87 us | 383.17 us | 384.70 us | 2.00x |
| 8 | 636.24 us | 241.80 us | 394.44 us | 2.63x |
| 32 | 771.76 us | 388.04 us | 383.72 us | 1.99x |
| 128 | 774.90 us | 385.57 us | 389.34 us | 2.01x |
| 1,024 | 777.30 us | 387.47 us | 389.82 us | 2.01x |

被消除的 weight normalization 与 token 数无关，所以绝对节省稳定在约
`0.38--0.39 ms/model step`；tokens 较少时 baseline/candidate 本身的波动会让
比例表现为约 `2.0x--2.6x`。

这是 Router linear 子图数据，只包含 40 次 Router 调用，不包含 attention、
Top-K、routed/shared expert GEMM、collective、scheduler 或 P/D transport。本轮
没有运行同节点完整 serving 的端到端 tok/s A/B，因此不能把表中的 `2.0x--2.6x`
写成模型吞吐提升。上线观察应以端到端 ITL/tok/s 和固定增加的 30 MiB 显存共同
评估。

### 风险、观察点与回滚

- 应在 checkpoint load 后确认 `_router_weight_caches_initialized=true`，避免非标准
  loader 把首次初始化落到 graph capture 内；first-forward fallback 只用于兜底；
- 权重加载后的原地 mutation 会使缓存过期，当前静态 inference weight 契约下不
  会发生；
- Router gate 在 TP rank 间复制，因此每个完整 YOCO model rank 固定增加 30 MiB；
  当前 TP1 的 DP1、DP4、DP8 部署都应按每卡 30 MiB 预算；
- revert `4364a96501` 会恢复每次 forward 的归一化，不影响此前 RoPE、MoE、
  add-RMSNorm、SwiGLU、UCX/NIXL 或 multigpu 功能。

## 14. DeepEP / NVSHMEM ABI、RDMA 与 IBGDA 启动保护

### 提交信息

```text
commit: 81df1f21e8
subject: fix(yoco): guard DeepEP NVSHMEM runtime
author/committer: 方涵斌 <2190556589@qq.com>
baseline: a9cd5c2072
branch: review/yoco-13-deepep-nvshmem
diff: 3 files, 186 insertions, 7 deletions
```

### 目的和旧失败链

multigpu 合并后的 long-context launcher 没有启用 EP，也没有把 RDMA device 传入
nested Docker；同时它用自定义 `LD_LIBRARY_PATH` 覆盖镜像默认值。基础镜像同时
存在 CUDA toolkit NVSHMEM 和 pip `nvidia-nvshmem-cu13` 时，DeepEP extension
可能加载到 ABI/符号不匹配的 host library。此前日志中的
`nvshmem_selected_device_transport` import error 就属于该类 loader 问题。

修正 library 顺序后，DP4 DeepEP LL 的真实下一层问题依次暴露：

1. `MAX_NUM_BATCHED_TOKENS=32768` 产生约 97.5 GiB LL RDMA buffer，触发
   `num_rdma_bytes / sizeof(int4) < INT_MAX`；
2. 改为 8192 后，默认 `NVSHMEM_QP_DEPTH=1024` 小于
   `(8192 + 1) * 2`，DeepEP 初始化断言失败；
3. 使用 QP depth 32768 后进入实际 dispatch，但 NVSHMEM 报
   `init failed for transport: IBGDA`；
4. DeepEP `internode_ll.cu:285` 随后反复触发
   `ibgda_get_state()->num_rc_per_pe >= num_local_experts`，四个 rank 最终以异步
   `CUDA error: unspecified launch failure` 退出，并在宿主记录 Xid 43。

节点并非完全没有 RDMA：Pod 可见 `uverbs/rdma_cm`，`mlx5_ib` 与
`nvidia_peermem` 已加载。决定性条件是
`/proc/driver/nvidia/params` 为 `EnableStreamMemOPs: 0`，且没有 `/dev/gdrdrv`；
这不满足 DeepEP LL 文档要求的两种 IBGDA 启用方式。旧 nested container 的
`HostConfig.Devices` 也只有 NVIDIA GPU device，没有任何 `/dev/infiniband/*`。

### 修改文件及职责

#### `docker/Dockerfile.b200.longctx`

- 将 mutable base tag 固定为已经验证的 registry digest
  `sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b`；
- build 时定位 `deep_ep_cpp`，用 `ldd` 要求 `libnvshmem_host.so.3` 来自 pip
  NVSHMEM 的 `site-packages` 或 `dist-packages` 路径；
- 用 `objdump -T` 检查 `nvshmem_selected_device_transport`；
- 创建临时 CUDA driver stub 后实际 `import deep_ep`，失败则终止 build；
- label 明确该镜像提供 guarded DeepEP/NVSHMEM，而不是声称 LL 在所有宿主可用。

#### `tools/yoco_serving/launch_nested_docker.sh`

- 不再覆盖镜像 `LD_LIBRARY_PATH`，保留 pip NVSHMEM 在 CUDA toolkit copy 之前；
- NVML 改挂到镜像已有搜索目录 `/usr/local/nvidia/lib64`；
- 枚举外层 Pod 实际可见的 `/dev/infiniband` character devices，逐个通过
  `--device` 映射，不使用会额外暴露 GPU 的 `--privileged`；
- 设置 `--ulimit memlock=-1:-1`；
- 透传 EP/backend/buffer/QP/token budget 开关；默认 backend 是本轮 A/B 胜出的
  `allgather_reducescatter`。

#### `tools/yoco_serving/serve_long_context.sh`

- DP>1 的 `auto` 启用 EP，DP1 保持非 EP；允许
  `ENABLE_EXPERT_PARALLEL=0` 显式回滚；
- 仅显式选择 DeepEP backend 时执行 version/import/`ldd` runtime 检查；
- LL `auto` token budget 取 8192，其他 backend 继续使用 32768；
- 按 `(max_tokens + 1) * 2` 计算 QP depth 下限并向上取二次幂；8192 自动得到
  32768；先 `unset` launcher sentinel，避免字符串 `auto` 被 NVSHMEM 当成数值；
- 调用 DeepEP size hint，在启动前拒绝超过 int32 index 上限的 LL buffer；
- LL 要求可见 `uverbs`，并要求 `EnableStreamMemOPs=1` 或 `/dev/gdrdrv`，否则
  code 2 fail closed，不再执行已知会 device assert 的 kernel。

### 构建、静态和参数测试

最终功能镜像从 commit `81df1f21e8` 的精确工作树构建：

```text
tag:  yoco-pr13-deepep-nvshmem-81df1f21e8
id:   sha256:f3920f514a8a164529a7116660dae7f4ee355c14d56fa5e0c21bc4784277d124
size: 37,668,087,694 bytes
```

build 和重新启动的 container 内均确认：

```text
deep_ep: 1.2.1+567632d
nvidia-nvshmem-cu13: 3.6.5
libnvshmem_host.so.3:
  /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3
nvshmem_selected_device_transport: present
import deep_ep with CUDA stub: passed
```

最终 launcher 的 `docker inspect` 显示全部可见 `uverbs/umad/issm/rdma_cm`
已进入 inner container，`memlock` soft/hard 都为 `-1`。参数 smoke 使用 fake vLLM
entrypoint，确认：

- DP1 默认不带 EP 参数，token budget 为 32768；
- DP4 默认带 `--enable-expert-parallel --all2all-backend
  allgather_reducescatter`；
- 显式 `deepep_high_throughput` 会先验证 DeepEP/NVSHMEM，再传给 vLLM；
- 显式 LL 在当前宿主打印 IBGDA 条件错误并返回 code 2。

静态检查：

```text
shellcheck 0.11.0: passed
bash -n: passed
git diff --check: passed
Docker build: passed
```

### B200 DeepEP HT 正确性

```text
Job:  bonete01/lidong1-yoco-pr13-deepep-g4-0805
Node: slc01-cl02-hgx-0201
GPU:  physical 4,5,6,7; NVIDIA B200
Model: /mnt/pvc/lidong1/vllm_test_artifacts/fhb-dev-dp14-20260805/model
DP/EP/TP: 4/4/1
precision / attention / MoE: BF16 / FlashInfer / Triton
```

服务日志确认不是 silent fallback：

```text
Using DeepEPHTAll2AllManager
Using DeepEPHTPrepareAndFinalize
```

8,192 和 65,536 token prompt 各做两次 greedy 256-token 生成。四个请求均输出
精确 256 tokens、`finish_reason=length`，同 shape 重复一致，并与已有 baseline
逐 token hash 相同：

```text
8K:  9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a
65K: 345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983
```

### Pure-P 65K Prefill 严格 A/B

因为目标是 PD 分离后的 pure-P node，性能测试不使用包含 16K/64K 本地 decode 的
W1/W2。两边用同一候选镜像、同一节点和 GPU，DP4/EP4，固定 seed，20 个
`65,536 input + 1 output` 请求，并发 4。harness 在计时前先执行一个单请求 warmup；
两边都是 20/20 成功，总输入 1,310,720 tokens。

| 指标 | AllGather+ReduceScatter | DeepEP HT | DeepEP 变化 |
| --- | ---: | ---: | ---: |
| wall | 11.33 s | 16.79 s | +48.19% |
| total token throughput | 115,676.90 tok/s | 78,058.63 tok/s | -32.52% |
| request throughput | 1.77 req/s | 1.19 req/s | -32.77% |
| mean TTFT | 2.124 s | 3.215 s | +51.35% |
| median TTFT | 1.920 s | 3.006 s | +56.59% |

DeepEP HT 按 vLLM 设计关闭 CUDA Graph；baseline 实际 capture
`FULL_AND_PIECEWISE`。HT 的 engine init 约 188 s、compile 约 72 s，和 baseline
约 189 s/71 s 接近，因此 serving 回退不是把冷启动混入 benchmark 导致的。

这组结果否定了“修好 import 后直接把 DeepEP HT 设为默认”的方案。功能仍保留为
显式实验选项，便于后续结合 E=32 MoE tuning 或新 DeepEP 版本重测；默认保持
AllGather+ReduceScatter。本提交是 correctness/operability 修复，不声称性能收益。

### 边界、风险和回滚

- 当前宿主没有满足 IBGDA 条件，所以只证明 LL 能准确 fail closed，不证明 LL
  dispatch 成功；启用 driver registry 参数需要更新 initramfs 并 reboot，或由节点
  管理员安装/加载 gdrdrv，均不属于容器代码权限；
- `nvidia_peermem` 只解决 GPUDirect RDMA memory registration，不等于已启用
  GPUDirect Async；
- DeepEP 使用 NVSHMEM/IBGDA，UCX 1.21 用于 NIXL/PD KV transfer；两者是独立
  transport 栈，本提交没有修改 `Dockerfile.b200.pd` 或重建 UCX；
- 本轮只测试单节点 DP4/EP4。跨节点 HT、宿主修正后的 LL、Decode 小 batch/CUDA
  Graph 组合需要独立资源验证；
- 当前 E=32,N=1280 没有 B200 专用 Triton MoE config，两边 A/B 都使用相同 default
  config，因此不影响本次归因，但它是后续单独优化点；
- 回滚 `git revert 81df1f21e8` 会恢复旧 launcher 和非 EP 默认，同时也恢复已知的
  NVSHMEM loader/RDMA/IBGDA 风险，不影响此前 Router、RMSNorm、SwiGLU、RoPE、
  UCX/NIXL 或 multigpu merge commit。

全部原始证据保存在：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pr13-deepep-nvshmem-20260805
```

## 本日志初始化

```text
subject: docs(yoco): add fhb-dev commit ledger
scope: 新建 fhb-dev-commit.md，补录前两个 PR 和第三个 Router commit
functional behavior change: none
```

本日志初始化只增加可审计文档，不改变模型、worker、NIXL 或 kernel 行为。其目的
是用单一文件替代分散的 PR 邮件上下文；后续功能 commit 继续按上面的模板追加。

## 15. YOCO fast-prefill 的 standalone、prefix cache 与 NIXL PD shape 对齐

### 提交信息

```text
commit: 03a0479b67
subject: fix(yoco): align fast-prefill shapes across PD
author/committer: 方涵斌 <2190556589@qq.com>
baseline: 4a39087d27
branch: review/yoco-14-pd-rms-shape-consistency
diff: 7 files, 273 insertions, 18 deletions
```

### 问题、根因与修复目标

现有 YOCO fast-prefill 在 1.3K 短请求上出现稳定而可复现的输出分叉，JIT 预热后
仍然存在。相同 prompt 的 standalone fresh、本地 prefix-cache hit 和 NIXL 1P1D
remote-KV hit 可能分别得到不同 key；8K 或 65K 某些候选还会改变停止位置。UCX
传输、RMSNorm、attention backend 和 Triton MoE 都曾被逐项排查，但决定性变量是
模型实际 forward 的 token rows：

- fresh standalone 可能一次执行完整 prompt；
- 本地 prefix cache 只重算最后一个 KV block 的尾段；
- 默认 NIXL remote full hit 只回退并重算最后 1 token；
- PD producer 和 standalone 若又被 scheduler 分成不同 chunk，即使传输 block
  内容完整，P 端产生 KV 的数值路径仍不同。

YOCO Router 使用归一化后的 FP32 weight 和 TF32 GEMM。B200 探针中，12 行与
1,356 行 Router 的 1,536 个 logits 有 1,533 个不同，未归一化权重的最大绝对差
为 `0.0014801`。这些差异足以改变接近边界的 top-k expert，并经 routed MoE 放大
为可见 token 分叉。因此最终修复目标不是强制所有模型使用严格 FP32，而是只在
YOCO fast-prefill 范围内让 fresh/local/remote 三条路径使用相同有效 shape。

最终语义如下：

1. FP32 TF32 Router 的 token rows 小于 128 时补零到 128，再裁回原行数；权重
   normalization 仍只使用原 FP32 weight，大 prefill 继续走原始 GEMM；
2. standalone 和本地 prefix-cache 请求在最后一个 KV block 起点拆分 prompt，
   使两者用相同尾段 shape；
3. NIXL P 节点把 prompt 截到同一 block 边界，只计算并发布完整 prefix block；
4. NIXL D 节点使用同一个公式计算 external token count，只接收该 prefix，尾段
   从未进入 D 的 remote block table，再由 D 本地计算；
5. `_p_side_truncated` 防止通用 scheduler 把已对齐的 P prefix 再拆一次；
6. 少于或等于一个 block 的 YOCO prompt 没有可发布的完整 prefix：P 不被截成
   空 prompt，D 返回 0 external tokens 并完整本地重算；
7. Mamba 检查优先于 YOCO 分支，原有 N−1 producer truncation 和 D receive count
   不变。普通模型及关闭 `kv_sharing_fast_prefill` 的 YOCO 不启用该逻辑。

### 修改文件及职责

#### `vllm/model_executor/models/yoco.py`

- 在 `_yoco_router_linear_tf32_cuda` 中保留 checkpoint 的 TF32 inference 语义；
- 将 `<128` token rows 补零到 128 行后执行 `F.linear`，再裁回有效行；
- 保留并恢复调用前的 CUDA TF32 和 matmul precision 全局状态。

#### `vllm/v1/core/sched/scheduler.py`

- 仅为 `model_type == "yoco"` 且启用 fast-prefill 的配置建立
  `need_yoco_final_prompt_block_split`；
- 在 WAITING 和 RUNNING 两条调度路径用同一个最后 block 起点公式拆分尾段；
- 识别 NIXL 已截断的 P 请求，避免 producer prefix 被二次拆分。

#### `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`

- 将原 Mamba 专用 token-count/truncation helper 泛化，但保留 Mamba 优先语义；
- YOCO P/D 两端共享 block-boundary token-count 公式，消除 remote/local group
  block 数不一致；
- P 端同时更新 prompt token/embedding、`_all_token_ids`、prompt length 和
  `max_tokens`，并通过 `_p_side_truncated` 保证 preemption 后幂等；
- 对 sub-block YOCO prompt 返回安全的 0-transfer 语义，不构造空 prompt。

#### `tests/model_executor/test_yoco_conversion.py`

- 新增 CUDA 参数化测试，覆盖 1/12/127 token rows；
- 逐位验证 CustomOp 输出等于显式 128-row padding 的 TF32 reference。

#### `tests/v1/core/test_scheduler.py`

- 覆盖 YOCO/非 YOCO、fast-prefill 开关、block 边界、已截断 P 请求和实际两步
  schedule；
- 验证 44-token prompt 首步为 32、第二步为 12，而非只修改计数。

#### `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`

- 保留并复跑 Mamba N−1 的 P/D 行为与幂等性；
- 新增 YOCO 44-token P/D 对称 count/truncation case；
- 新增 12-token sub-block case，确认 P 请求有效且 D 不异步拉取 partial block。

#### `tests/v1/kv_connector/unit/utils.py`

- 测试构造器增加 YOCO shape-alignment 开关和固定 block size，不改变生产逻辑。

### run17--run25 排查记录

| Run | 候选 | 关键结果 | 结论 |
| --- | --- | --- | --- |
| 17 | 只做 Router `<128 -> 128` padding | 1.3K standalone=`00340`、PD=`00663`；8K exact；65K 停止不一致 | Router 是放大器，但单独修不完整 |
| 18 | D full remote hit 后只把计算计数回退到 block 起点 | 1.3K exact；8K `04165` 对 `00395`；65K 仍错 | 单边改 D shape 会破坏其他长度 |
| 19 | Router padding + fresh prompt 最后 1 token 独立调度 | 1.3K/8K/65K PD 全 exact；1.3K fresh=`00663`、local hit=`00340` | PD 可对齐，但本地 cache 仍不稳定 |
| 20 | 去掉 Router padding，只保留 final-token split | 1.3K/65K exact，8K 不一致 | Router 固定小 batch shape 仍是必要条件 |
| 21 | fresh/local/remote 都按 block-tail 回退计数 | standalone fresh/local exact；PD 1.3K/65K exact，8K 为 `04165` 对 `02346` | remote 尾块已进 block table，计数回退是伪重算 |
| 22 | P producer 真正截断到 block boundary | standalone fresh/两次 local hit exact；PD 首请求 HTTP 500 | P 发布 84 blocks，D 仍按原 prompt 分配 85 blocks |
| 23 | NIXL per-group block-count 诊断 | 定位 full-attention 第 30 组 `D=85/P=84` | 必须让 NIXL P/D 共享 token-count 公式 |
| 24 | P/D 对称 block prefix transfer | block 传输成功但 1.3K 数值仍不一致 | P 的 1,344 又被 scheduler 拆成 `1,328 + 16` |
| 25 | 加 `_p_side_truncated` shape guard | standalone fresh/local 与 1P1D 三档全部 exact | 最终采用 |

run17--run24 都保留为诊断证据，没有把失败候选或 worker 临时诊断改动带入提交。
尤其没有采用 run21 的“先接收完整 remote block、再只回退
`num_computed_tokens`”方案。

### B200 正确性与时延

最终 run25 的 standalone 首先对 1,356-token prompt 执行 fresh 和两次相同
cache salt 的重复请求。三次文本均为 `" 00663\n"`，本地 cache 命中 1,344
tokens，首 token 及其 logprob 逐位一致。随后串行执行 P 请求、等待 KV ready、再
执行 D 请求；三档均 `all_exact_match: true`：

| Case | Prompt tokens | SHA256 | P time | D time | Total |
| --- | ---: | --- | ---: | ---: | ---: |
| 1.3K warmup | 1,356 | `fde361c254896b017d5496f6e8cd4a128d7e258544675fab97574d01c973c033` | 0.0692 s | 0.2609 s | 0.3301 s |
| 8K | 7,999 | `8b676d6658af7e5d789c559690a8c763683c433e8e857b4146df5054dfa71c01` | 0.1549 s | 0.4190 s | 0.5739 s |
| 65K | 65,809 | `82e7da1423e3c6e149b622b75a9674c5508bf91c1a06099b46e75a22033bfe53` | 1.1278 s | 1.1470 s | 2.2748 s |

上述 Total 是 correctness harness 中串行相加的 wall time，不是在线并发吞吐，也
不能与 standalone 直接解释为性能提升。Router microbenchmark 以未 padding 的
直接 GEMM 为对照，1/12/127 rows 分别约 `+185%/+150%/+160%`，128 rows 约
`+3%`；小 shape 的绝对增加约 `44--50 us/router call`。本提交首先解决
correctness，端到端 QPS、ITL 和高并发 P/D 性能需另做固定并发 A/B。

### 单元、静态与环境测试

最终验证包括：

```text
YOCO conversion/CUDA file: 38 passed
NIXL/Mamba/YOCO focused:   7 passed, 26 deselected
scheduler + NIXL/HMA:      137 passed, 1 external-access failure
ruff check:                passed
ruff format:               passed
git diff --check:          passed
```

唯一未通过项为：

```text
test_fewer_blocks_with_hma[google/gemma-3-1b-it-512]
```

它在加载 HuggingFace `google/gemma-3-1b-it/config.json` 时返回 gated repo HTTP
403；失败发生在模型配置下载阶段，未进入本提交修改的 NIXL scheduler。开发循环
中相关定向集合另有一轮 `145 passed`，最终 B200 端到端不是用 mock connector。

原始 evidence：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-current-4a39087d27-20260805/run25-yoco-nixl-shape-aligned-pd-20260809
```

测试结束后已恢复 Pod 内 PID 7，并释放：

```text
Volcano Job: lidong1-yoco-pd-rms-shape-g2-0809-r2
ConfigMap:   yoco-pd-rms-shape-candidate
```

### 风险、限制与回滚

- 当前只证明 B200 单节点 1P1D、文本 token、三种长度的 greedy exact match；未
  覆盖多 P/D、DP>1、TP>1、高并发、preemption 实压或 prompt embeddings；
- P/D 仍需使用相同 KV block size 和兼容 NIXL metadata；这是既有 connector
  契约，不由本提交增加动态协商；
- 小 batch Router 固定 shape 有明确微基准开销，应在真实 decode 并发下继续观察
  ITL/QPS；大于等于 128 rows 的 prefill 路径不 padding；
- sub-block prompt 会让 P 做一次最终不被 D 使用的短 prefill，换取不构造空请求
  和 D 端完整本地 correctness；该长度应由 router 策略进一步考虑直接走 D；
- Mamba N−1 路径有单测保护，但本轮 B200 服务模型是 YOCO，不是 Mamba；
- 本提交没有修改 CUDA Graph 开关、Docker、UCX 1.21、DeepEP/NVSHMEM、Router
  服务或负载均衡策略；
- `git revert 03a0479b67` 可完整回滚功能提交，不影响此前 DeepEP、Router weight
  cache、RoPE、SwiGLU、add-RMSNorm 或 multigpu merge，但会恢复已知的
  fresh/local/remote shape 分叉。

## 16. 当前 YOCO PD 策略独立报告

```text
commit: fa8e4eac6a30aa14ccd7b0ff8aa5e4ee9309f107
subject: docs(yoco): add standalone PD strategy report
scope: YOCO-PD-STRATEGY.md, fhb-dev-commit.md
functional behavior change: none
```

本提交将散落在 `yoco.md`、功能提交记录、NIXL 文档和实际 run25 harness 中的
PD 约束整理为独立报告，便于部署、Gateway 和审核人员使用。报告明确：

- 推荐 pure `kv_producer` P + independent `kv_consumer` D；
- pure-P 的 DP 不限制为 1，但 DP>1 不应使用混合流量的 `kv_both`；
- P token 必须丢弃，D 负责所有用户可见 token；
- YOCO block-tail、Router fixed-shape 和 NIXL P/D 对称 token count；
- NIXL 1.3.2 + UCX 1.21、side-channel、RDMA HCA 和 single-UCX 契约；
- Gateway 状态机、租约、失败策略、可观测性和上线检查表；
- run25 只证明同节点 1P1D correctness，不能写成跨节点 RDMA 或并发吞吐。

该提交不修改模型、scheduler、NIXL、Docker、launcher 或请求协议，不增加新的
性能归因。回滚本提交只删除独立报告及本条记录，不影响已合入的 PD 功能。

## 17. TP2/DP1 batch 容量、吞吐与 forward 延迟曲线

```text
commit: f54af87aecf78e2db92c15cfa5062fadff1d9509
subject: docs(yoco): record PD batch capacity and throughput
scope: YOCO-PD-BATCH-CURVE-20260810.md, fhb-dev-commit.md
functional behavior change: none
runtime under test: fhb-dev@fa8e4eac6a
```

### 目的与修改文件

本提交把 `fhb-dev@fa8e4eac6a` 的 4 x B200 NIXL PD 容量扫描固化为独立、可审核
报告，回答“当前正确性覆盖到哪里、吞吐是多少、最大 batch 能否运行、为什么
batch 48 性能突然下降”四个问题。只修改两个文档文件：

- `YOCO-PD-BATCH-CURVE-20260810.md`：记录测试口径、正确性、batch 1--256
  吞吐表、Prefill/Decode GPU 同步前向延迟和 PVC 证据；
- `fhb-dev-commit.md`：增加本条提交记录和索引，不修改生产实现。

没有提交 harness、截图、HTML、运行时只读挂载文件或原始大日志；完整证据继续保留
在 PVC，避免扩大 review diff。

### 正确性结果与边界

拓扑为 `P=TP2/DP1, D=TP2/DP1`。同配置 standalone -> PD 覆盖 1,356-token
自然停止、1,356-token 强制 256 decode、7,999-token 和 65,809-token 四档，均为
逐 token 和文本 exact。7,999-token prompt 使用不同 cache salt 执行
concurrency=4、8 requests，8 个请求收敛为一个 token trace。

该结论不能扩大为所有并行方式均正确：先前复核的 DP2 + CUDA Graph 并发仍会在
不同 rank 出现多 trace；PCP/DCP 也未在本轮 4 卡拓扑覆盖。batch 1--256 性能扫描
都返回预期 completion token 数，但没有逐 batch 再运行 standalone -> PD
token-exact 矩阵，报告中对此明确区分。

### 容量、吞吐与性能拐点

服务配置为 `max_num_seqs=256`、`max_num_batched_tokens=8192`，完整 CUDA Graph
只 capture decode batch 1--32。本轮确认短上下文 decode batch 256 和 Prefill
token batch 8192 均可运行，但推荐的综合点是 batch 32：

```text
D service:      1,139.27 output tok/s
PD end-to-end:  1,078.67 output tok/s
PD request QPS: 4.21 req/s
```

GPU 同步 model-forward median 在 actual decode batch 32 为 `8.08 ms`，到 48
变为 `68.88 ms`，增加约 `8.5x`。对应 D 吞吐从 `1,139.27` 降到
`477.06 output tok/s`。batch 256 虽重新达到 `1,079.91 output tok/s`，端到端仅
`1,005.90 output tok/s`，仍低于 batch 32，并且端到端请求 p50 已到
`64.66 s`。因此 256 是已验证的 scheduler 配置上限，不是默认生产目标。

Prefill 在 concurrency 48 达到本轮峰值 `77.85K input tok/s`，但系统端到端受
Decode 限制。Prefill 单 forward 在 token batch 128--4096 约 `52.79--53.24 ms`，
8192 为 `82.85 ms`。

### 测试和证据

```text
Job:       bonete01/lidong1-yoco-pd-diagnose-g4-0810
Node:      slc01-cl02-hgx-0297
GPU:       4 x NVIDIA B200
Transport: NIXL
```

吞吐和 forward 客户端、correctness harness 与 forward stats parser 返回码均为 0。
前向延迟来自 vLLM `Batchsize forward time stats` 的 GPU 同步中位数，不把 HTTP
wall time 当作 model forward。完整结果位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/
  pd-batch-curve-fa8e4eac6a-20260810/
```

测试完成后已删除 Volcano Job，并确认 Job/Pod 均为 `NotFound`。本提交只同步已有
GPU 实测文档，不产生性能行为变化；回滚只需删除独立报告并移除本条记录。

## 18. LMCache 三级缓存与 Dynamo 适配方案

```text
commit: 75d26710b9b0ea3d548d607f2de73557394927ec
subject: docs(yoco): plan LMCache and Dynamo adaptation
scope: YOCO-LMCACHE-DYNAMO-PLAN.md, fhb-dev-commit.md
functional behavior change: none
runtime under review: fhb-dev@f54af87aecf78e2db92c15cfa5062fadff1d9509
```

### 目的与结论

本提交在修改镜像和运行逻辑前，先固定 YOCO 接入 LMCache、GPU/CPU/持久后端三级
KV 缓存和 NVIDIA Dynamo 的实施边界。推荐的第一阶段组合为：

- Prefill 使用 vLLM GPU prefix cache、LMCache 跨请求缓存和 NIXL producer；
- Decode 保持 NIXL consumer，不在第一阶段写 LMCache；
- P 到 D 的实时数据面继续使用已经验证的 NIXL 1.3.2 + UCX 1.21；
- Dynamo 先承担 Frontend、KV-aware 路由和 P/D 编排，不替换 NIXL 数据面；
- LMCache 第一阶段关闭 layerwise、异步加载和 MP sidecar，保持完整 CUDA Graph。

这样可以分别归因 LMCache lookup/load、三级存储、NIXL P/D 和 Dynamo Router 的
正确性及性能，不把多个高风险变化合并为一个不可诊断的上线步骤。

### 修改文件

- `YOCO-LMCACHE-DYNAMO-PLAN.md`：记录架构、固定版本、YOCO 兼容风险、connector
  组合、三级缓存、Dynamo shared-indexer 缺口、测试矩阵、性能口径和回退顺序；
- `fhb-dev-commit.md`：增加本条设计记录及提交索引。

### 源码审阅发现

本轮对当前 vLLM connector、LMCache `v0.5.3@140819c9d57a` 和 Dynamo
`v1.3.1@a49702e4432e` 做了只读审阅，确认：

1. 当前 `docker/Dockerfile.b200.pd` 没有安装 LMCache；LMCache 0.5.3 已有 CUDA 13
   wheel，但仍应从固定提交源码构建，使扩展与镜像内 torch C++ ABI、CUDA 13 和
   SM100 精确对齐；
2. YOCO 后 10 个 cross layers 是 10 个逻辑名字指向同一个物理 KV tensor；当时据此
   推测需要 20 logical -> 11 physical。第 20 节的真实 B200 启动验证进一步发现
   `universal_loop=3`，并将该结论更正为 20 个基础逻辑层 -> 31 份物理 KV；
3. LMCache chunk 和 YOCO/NIXL block-tail 是两层对齐规则。第一阶段固定
   `chunk_size=256`、`save_unfull_chunk=false`、`discard_partial_chunks=true`，不能
   让 LMCache 恢复 D 应本地重算的最后 prompt block；
4. vLLM `MultiConnector` 是“第一个命中负责 load、所有 child 参与 save”，但
   LMCache hit -> P 计算 miss -> NIXL send 的串联语义仍需真实端到端证明；
5. Dynamo 1.3.1 可按 host/disk cache event 给路由打分，但其
   `shared_cache_type` 只有 `none/hicache`，LMCache 共享持久层需要后续独立
   shared-indexer adapter。

### 测试与性能

本提交是设计文档，不修改运行行为，因此没有 GPU 性能收益，也没有申请资源。
执行的检查为：

```text
git diff --check: passed
repository state before edit: fhb-dev == snow2022jlu/fhb-dev
```

方案明确要求后续报告 cold、CPU warm、persistent warm 和 Dynamo routing 四组相对
当前 NIXL-only baseline 的 TTFT、ITL、吞吐、load/store 带宽与各层 hit rate，且
standalone -> LMCache -> NIXL PD 必须逐 token exact。设计稿不能作为兼容性或性能
已经通过的证据。

### 风险与回滚

该提交不修改模型、scheduler、connector、Docker、CUDA Graph、UCX、NIXL、
Gateway 或 Kubernetes profile。回滚只需删除方案文件并移除本条记录，不影响当前
NIXL-only PD 服务。后续实现继续拆成镜像、shared-KV 单测、adapter、P/D 串联、
持久层和 Dynamo 六类独立提交。

## 19. 固定 LMCache 0.5.3 的 CUDA 13/SM100 runtime

```text
commit: e081d38f5aeb9976d9aba8d3fa00d9bf5d3ab7d2
subject: build(yoco): add pinned LMCache CUDA 13 runtime
scope: docker/Dockerfile.b200.pd, fhb-dev-commit.md
functional behavior change: PD 镜像新增 LMCache runtime；默认 NIXL-only 服务行为不变
runtime base: fhb-dev@75d26710b9
```

### 目的与版本契约

本提交只解决“后续 YOCO LMCache adapter 在哪个可复现 runtime 上开发和测试”，
不提前修改 connector、scheduler 或线上启动参数。镜像新增并固定：

```text
LMCache: v0.5.3@140819c9d57a975dbc5678a6459a218e544cb58b
NIXL:    1.3.2@de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72
UCX:     1.21.0@b6a9d47fccce849c28111f05a7fa8f1c930ff17d
CUDA:    13.1 / SM100
torch:   基础镜像自带 2.11.0a0，CXX11 ABI=1
```

LMCache 同时按 tag 和 commit 校验。仅 shallow fetch commit 会让
`setuptools-scm` 看不到 tag 并产出错误的 `0.1.dev1` wheel；现在拉取
`refs/tags/v0.5.3` 后再断言 `HEAD` 等于固定 commit，既保留正确的 `0.5.3`
包版本，也防止 tag 移动后静默改变镜像内容。

### 修改文件

- `docker/Dockerfile.b200.pd`
    - 在 native builder 中使用基础镜像的 torch、CUDA 13.1 和 CXX11 ABI=1 从源码
    构建 LMCache wheel，`TORCH_CUDA_ARCH_LIST=10.0`，不混入 CUDA 12 runtime；
    - runtime 安装固定 wheel 和 `wheel==0.47.0`，后者补齐基础镜像中
    `astunparse` 的既有依赖缺项；
    - 保持 vLLM P 到 D 的传输仍由 NIXL 1.3.2 + UCX 1.21 承担，安装 LMCache 不会
    自动改变 connector；
    - 构建阶段导入 `lmcache.c_ops`，检查 LMCache/NIXL 版本、禁用 `nixl_ep` shim、
    验证唯一 UCX、编译 Python tree，并把 `uv pip check --system` 设为硬门禁；
    - OCI labels 新增 LMCache version/revision，镜像说明明确三组件版本。
- `fhb-dev-commit.md`
    - 记录构建原因、文件范围、验证证据、依赖变化和回退边界。

### 构建与测试结果

完整构建命令：

```text
docker build --progress=plain \
  -f docker/Dockerfile.b200.pd \
  -t vllm-yoco-pd:lmcache053-local .
```

结果：

```text
image:  vllm-yoco-pd:lmcache053-local
digest: sha256:5d96d8cef49b873c615f91fa7176f71e15b1baa16571aa74fce534563ad61e04
status: build exit 0
```

镜像内和成品镜像外分别执行了以下检查：

```text
LMCache package:          0.5.3
LMCache CUDA c_ops:       import passed
LMCache vLLM adapter:     import passed
NIXL packages:            nixl==nixl-cu13==1.3.2
single-UCX verification:  UCX 1.21.0 passed
torch CXX11 ABI:          true
numpy:                    2.1.0（保持基础镜像版本）
transformers:             5.8.1（保持基础镜像版本）
OpenTelemetry:            1.40.0（LMCache 0.5.3 的兼容上限）
uv pip check --system:    all installed packages are compatible
Python compileall:        passed
LMCache c_ops ldd:        no missing shared libraries
git diff --check:         passed
```

依赖解析将基础镜像的 OpenTelemetry 1.44 统一降到 LMCache 约束允许的 1.40，
没有改变 torch、numpy 或 transformers。构建前基础镜像的 `uv pip check` 有 16 个
缺项；LMCache 安装并显式补充 wheel 后，成品镜像为零冲突。

### 性能、限制与回滚

本提交没有启动 LMCache connector，也没有申请 GPU，因此没有可归因的 TTFT、
吞吐或命中率收益。它只提供后续单卡 CPU cold/warm、1P1D LMCache -> NIXL 和三级
缓存测试所需的 runtime；不能把“wheel 可导入”写成 KV 保存/恢复正确。

本节构建时曾按 20 个逻辑 attention layer 对应 11 份物理 KV tensor 估算。第 20 节
通过真实 checkpoint 和 B200 启动把它更正为 `0..10, 20..29, 40..49` 共 31 份；
11--19 仍是 owner 10 的 alias。镜像提交本身尚未启用 connector，因此该估算不影响
镜像行为。回滚本提交只需恢复 `docker/Dockerfile.b200.pd` 并重建镜像，现有
NIXL-only 已发布镜像不受影响。

## 20. YOCO universal-loop-aware LMCache 物理 KV 适配

```text
commit: 5d296b3958c0db1a664dd74dff78e28a4419b588
subject: fix(yoco): adapt LMCache to physical KV layout
scope: lmcache_connector.py, test_lmcache_connector.py,
       YOCO-LMCACHE-DYNAMO-PLAN.md, fhb-dev-commit.md
functional behavior change: YOCO 可安全注册 31 份物理 KV 到 LMCache；
                            非 YOCO connector 行为不变
runtime base: fhb-dev@e081d38f5a
```

### 目的、真实布局与修改文件

第一次 B200 启动使原先的 20 -> 11 假设 fail closed，并暴露 checkpoint 的真实配置：

```text
num_hidden_layers=20
yoco_cross_layers=10
universal_loop=3
physical KV indices=0..10, 20..29, 40..49
```

也就是说，三轮 self-attention 各有 10 份 KV，cross owner 为 layer 10，共 31 份
物理 tensor；逻辑 cross layers 11--19 继续 alias owner 10。本提交围绕这个单一功能
修改四个文件：

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
    - 从 HF config 读取 `universal_loop`，按基础逻辑层号偏移构造物理顺序；
    - 初始化 LMCache 时临时暴露 31 层 metadata 和隔离 namespace，初始化后恢复
    vLLM 原始 20 层配置；
    - 注册前要求物理层集合完整，并严格断言 11--19 与 owner 10 是同一个 tensor
    对象、地址、offset、shape、stride 和 dtype；
    - 只把 31 份唯一物理 tensor 传给 LMCache，避免重复保存 cross alias；
    - YOCO 首版强制非 layerwise、非 async-load、非 blending 和 GPU connector v3，
    不支持的组合直接失败，不静默降级。
- `tests/v1/kv_connector/unit/test_lmcache_connector.py`
    - 覆盖三轮 31 份布局、单轮 11 份布局、alias clone、缺失 alias、31 层 metadata/
    engine 和实际注册集合。
- `YOCO-LMCACHE-DYNAMO-PLAN.md`
    - 把设计阶段的 11 份估算更正为实测 31 份，并把首轮功能配置改为与 vLLM block
    对齐的 16-token chunk。
- `fhb-dev-commit.md`
    - 更正第 18、19 节的历史假设，并记录本次实现和测试证据。

### CPU、镜像与静态检查

本地和目标 LMCache 0.5.3/CUDA 13 镜像内执行同一测试文件：

```text
local pytest:       29 passed
image pytest:       29 passed
ruff check/format:  passed
mypy hook:          passed
all pre-commit:     passed
git diff --check:   passed
```

B200 运行时从 PVC 只读挂载经过测试的 connector，源码 SHA256 为
`6da5612848b13be15e5e199cbc352b00f9d48e816cb82f48d84c9a0a3ccc8b6d`。
节点 Docker 的既有候选镜像已达到最大 layer depth，无法再 `docker commit`；这只影响
临时测试镜像的再封层，不影响第 19 节从干净基础镜像构建的 pinned Dockerfile。

### B200 正确性

```text
Volcano Job: bonete01/lidong1-yoco-lmcache-g1-0810
Pod:         lidong1-yoco-lmcache-g1-0810-master-0
Node:        slc01-cl02-hgx-0297
GPU:         1 x NVIDIA B200
LMCache:     0.5.3-g140819c, LocalCPUBackend, GPU connector v3
Model:       BF16, FlashInfer, Triton MoE, FULL_AND_PIECEWISE CUDA Graph
```

服务日志确认 LMCache 的 `num_layer=31`、`kv_shape=(31, 2, 16, 8, 128)`，以及
`20 logical layers -> 31 physical KV tensors`。对 2,125-token prompt，LMCache cold
保存 2,112 token，warm 和 partial 各真实命中并恢复 2,112 token。

严格门禁分两层：

1. cold LMCache 与无缓存全量重算的 text、token IDs、逐 token logprob 全部 exact；
2. warm/partial 与相同 2,112-token 恢复边界的 vLLM 原生 prefix cache 全部 exact。

原先用 `chunk_size=256` 只恢复 2,048 token，而原生 prefix cache 恢复 2,112 token；
两种路径都成功返回，但与全量重算不逐 token exact。随后证明即使完全不经过
LMCache，vLLM 原生 prefix cache 相对全量重算也有同类差异。因此不能把不同恢复
边界和 prefill kernel 路径的 BF16 数值差异误判为 LMCache 层映射错误；同边界 exact
才是本 connector 的有效正确性证据。

### 性能结果

公平对照先用第一条请求完成 JIT，再比较相同 2,125-token 请求。单次结果如下：

| 场景 | 无 prefix 全量重算 | 原生 GPU prefix | LMCache CPU warm | LMCache / 重算 |
| --- | ---: | ---: | ---: | ---: |
| 相同 prompt | 273.88 ms | 223.96 ms | 239.63 ms | 1.143x |
| 共享前缀、不同尾部 | 277.19 ms | 223.10 ms | 238.03 ms | 1.165x |

LMCache 相对全量重算降低约 12.5%--14.1% 延迟，但比 GPU 原生 prefix cache 慢
约 6.7%--7.0%，符合 CPU tier 需要搬运数据的预期。每次恢复 0.2498 GB，实测读取
26.62--26.74 ms、9.34--9.38 GB/s；cold 保存 56.43 ms、4.43 GB/s。该结果是单卡
短样本功能基准，不外推为并发吞吐或跨节点 persistent tier 收益。

### 已知限制、后续与回滚

LMCache 0.5.3 在 warm retrieve 后打印 `Double unpin` 并把负 pin count 归零。三组结果
仍与原生 prefix cache exact，但这个上游生命周期告警必须在长稳测试前解决，当前
不能把本提交描述为 production ready。以下项目尚未由本提交证明：

- LMCache -> NIXL producer -> independent Decode 的 1P1D 串联；
- local NVMe、共享持久层和跨 Pod cache reuse；
- Dynamo KV-aware 路由、shared indexer、故障回退和并发吞吐；
- TP/DP/EP/PCP 下的 LMCache rank namespace 和缓存一致性。

完整 B200 原始结果位于 PVC：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/lmcache-physical-kv-20260810/results/
```

回滚只需移除 YOCO 专用 physical view；非 YOCO 的 LMCache 和当前 NIXL-only PD
路径没有行为变化。生产 profile 在 double-unpin、P/D 串联和持久层测试完成前继续
默认关闭 LMCache。

## 21. pure-P prefix shape 与 SWA window 传输补充

```text
commit: aebb50c6e5
subject: fix(yoco): preserve pure-P prefix shape with HMA
scope: nixl/scheduler.py, test_nixl_connector_hma.py, YOCO-PD-STRATEGY.md
functional behavior change: YOCO pure-P 请求在本地 cache lookup 前截断，并按请求
                            bypass 本地 prefix read；NIXL HMA 裁剪逻辑不变
runtime base: fhb-dev@53f58e5426
```

### 目的与根因

本轮首先回答“除 full cross-owner 外，PD 是否只传 SWA window”。真实 checkpoint
包含 30 个 512-token self-attention SWA physical groups 和 1 个 full cross-owner
group。既有 NIXL HMA 代码已经逐 group 裁剪，但此前缺少 YOCO 31-group 专项单测、
full-context 对照和 65K payload 证据。

补测 LMCache cold/warm 时又发现一个独立 scheduler edge：1,356-token pure-P 首次请求
生成 1,344-token prefix 后，第二次请求会先在 vLLM 本地命中全部 1,344 tokens；旧
时序随后才把请求截成 1,344，令 `num_new_tokens=0`，触发 scheduler 断言并退出
EngineCore。

最初候选是把命中回退一个 block，重新计算最后 16 tokens。这个候选通过了单测，
但 B200 隔离实验让 P 开 cache、D 关 cache 后，1.3K 和 65K 虽然 NIXL 传输零失败，
输出却不再 exact。原因与第 15 节一致：P 从 1,344-row forward 变成
`1,328 + 16`，Router/MoE 数值路径改变，重新生成的 KV 不适合发送给独立 D。因此
没有提交该候选。

最终实现只对 YOCO pure-P 生效：`on_new_request` 在任何本地 cache lookup 前把请求
截到 NIXL/D 共用的 full-block boundary，并设置 `skip_reading_prefix_cache=true`。
这样服务级 prefix cache 可继续为其他请求启用，但 pure-P 始终用已经验证过的完整
prefix shape；Mamba 和普通模型仍走原 connector query 时序，不扩大行为变化。

### 修改文件

- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`
    - YOCO P 请求提前执行幂等 truncation；
    - 只在 `_p_side_truncated` 成立时按请求 bypass 本地 prefix read；
    - 保持 heartbeat、Mamba 和普通模型路径不变。
- `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`
    - 新增真实 30 SWA + 1 full owner 的 group 裁剪测试；
    - 65,808-token transferable prefix 要求 metadata 为 `30x33+4113`；
    - 新增 1,356 -> 1,344 的 early truncation、cache bypass 与幂等测试。
- `YOCO-PD-STRATEGY.md`
    - 写清 metadata 33 blocks 与实际 payload 32 blocks 的差异；
    - 记录 HMA/full-context A/B、重复请求隔离门禁和原始 PVC 证据。

### 单元、静态与 B200 测试

目标 CUDA 13/LMCache 镜像内相关回归：

```text
pytest NIXL YOCO/Mamba subset: 8 passed, 27 deselected
YOCO 31-group clipping test:  passed
python compileall:             passed
git diff --check:              passed
```

Volcano Job `bonete01/lidong1-yoco-pd-diagnose-g4-0810` 使用 4 x B200，P/D 各
TP2、BF16、FlashInfer、Triton MoE、FULL_AND_PIECEWISE CUDA Graph。HMA window 和
full-context baseline 各三轮，共 18 个计入样本全部 text/token trace exact，NIXL
transfer/notification failure 均为 0：

| Prompt | HMA bytes | Full bytes | 缩减 | HMA D | Full D |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,356 | 68,419,584 | 170,655,744 | 2.49x | 0.212s | 0.436s |
| 7,999 | 95,617,024 | 1,013,776,384 | 10.60x | 0.380s | 2.429s |
| 65,809 | 332,464,128 | 8,356,036,608 | 25.13x | 0.909s | 18.835s |

65K scheduler metadata 为 `30x33+4113=5103` blocks；实际 NIXL bytes 对应
`30x32+4113=5073` blocks，每个全局 block 65,536 bytes。最后又让 P 全局 cache 开、
D cache 关并复用相同 salt，两轮三档共 6/6 exact，证明请求级 bypass 后重复请求
仍真实经过 NIXL 且不再崩溃/分叉。

### LMCache 串联负结果

标准 `MultiConnector(LMCache, NIXL)` 因 LMCache 0.5.3 不支持 HMA，只能退化为一个
full-context group，65K 仍传 8,356,036,608 bytes。未固定 hash seed 时，KV store
成功但 warm hit 恒为 0；固定 `PYTHONHASHSEED=0` 后，1.3K/8K/65K 分别真实命中
1,344/7,984/65,808 tokens，但 `cache_salt` 未隔离 key，且通用 full-hit 只回退
1 token。LMCache -> NIXL -> D 三档均不 exact。因此本提交不启用 LMCache，不宣称
三级缓存收益；完整阻塞项写入 `YOCO-LMCACHE-DYNAMO-PLAN.md`。

### 性能口径、风险与回滚

上述时间来自串行 correctness harness，不是在线并发吞吐。可归因结论仅为 NIXL
payload/D transfer 路径随 prompt 增长显著缩短；P 请求级 cache bypass 会放弃 pure-P
GPU prefix reuse，这是为数值正确性接受的显式代价。

回滚 `aebb50c6e5` 会恢复旧的 P cache lookup 时序，并重新暴露“全 prefix hit 后
`num_new_tokens=0`”崩溃，因此不建议单独回滚。若必须恢复 P prefix reuse，应先设计
不改变 producer KV shape 的专用复用协议并重跑“P cache 开、D cache 关”的三档
exact 门禁，不能恢复已否决的最后一块重算候选。

原始证据：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-swa-transfer-53f58e5426-20260810/
/mnt/pvc/lidong1/vllm_test_artifacts/pd-nixl-hma-p-cache-bypass-53f58e5426-20260810/
/mnt/pvc/lidong1/vllm_test_artifacts/pd-lmcache-nixl-stable-hash-53f58e5426-20260810/
```

## 22. PD 极限吞吐与 W1/W2/W3 HMA A/B 报告

```text
commit: e5524539f1
subject: docs(yoco): record PD saturation and HMA workload results
scope: YOCO-PD-BATCH-CURVE-20260810.md, YOCO-PD-STRATEGY.md,
       YOCO-PD-W123-HMA-AB-20260810.md
functional behavior change: none; this is a measurement and documentation commit
runtime base: fhb-dev@9b9a945c06
```

### 目的

本提交补齐两组此前已经在 4 x B200 上完成、但尚未进入远端提交历史的 PD 实测：

1. 在 `P=TP2/DP1, D=TP2/DP1` 下扩大 P token budget，并把 D CUDA Graph capture
   扩到 batch 256，给出 P、D 和端到端 PD 的 batch--吞吐--延迟曲线；
2. 对 W1、W2、W3 workload 比较当前 HMA/SWA window 传输与 full-context baseline，
   分离 TTFT、总吞吐和 NIXL 流量收益，并如实记录异步 W3/batch-4 的精度边界。

本提交只记录测试证据，没有修改运行时代码、默认参数或服务行为。

### 修改文件

- `YOCO-PD-BATCH-CURVE-20260810.md`
    - 追加 P 侧 batch 1--256、D 侧 graph-128/graph-256 和端到端 batch 16--128 扫描；
    - 记录 1.3K、强制 256 decode、8K、65K 的 standalone 对 PD exact 门禁；
    - 给出在线 admission 与 CUDA Graph capture 的配置建议。
- `YOCO-PD-W123-HMA-AB-20260810.md`
    - 新增完整 W1/W2/W3 HMA 对 full-context 的环境、方法、逐点性能和流量报告；
    - 记录 W3 双向 `D -> P -> D` metadata 复用；
    - 记录 W3/batch-4 主测 hash 差异及三次短复现，避免把并发非确定性写成全量 exact。
- `YOCO-PD-STRATEGY.md`
    - 汇总极限吞吐、推荐 batch、W1/W2/W3 收益和正确性边界；
    - 链接完整报告及 PVC 原始证据。

### 测试环境与结果

两组测试均使用 4 x NVIDIA B200、BF16、FlashInfer、Triton MoE、NIXL 1.3.2 +
UCX 1.21，同节点 `lo`/CUDA IPC，拓扑为 P=TP2、D=TP2。极限吞吐补测的观测峰值：

| 侧 | 负载与峰值点 | 峰值吞吐 |
| --- | --- | ---: |
| P | 8,192 effective input，batch 64 | 84,055 input tok/s |
| D | 1,345 context + 512 output，batch 256 | 2,284 output tok/s |
| PD | 1,345 input + 256 output，batch 96 | 1,128 output tok/s / 4.407 req/s |

D graph capture 从 128 扩到 256 后，batch 192/256 不再跌回 eager；新增图只多约
0.10 GiB/卡，并使 KV capacity 下降约 0.15%。在线建议 P admission 48--64、D
admission 64--96，但 D graph 保留到 256 以覆盖突发。

HMA 对 full-context 的聚合吞吐变化为：

| Workload | batch 1 | batch 4 | NIXL 流量缩减 |
| --- | ---: | ---: | ---: |
| W1 | -3.14% | -2.49% | 10.77x |
| W2 | +12.79% | +43.37% | 25.11x |
| W3 | +553.02% | +1033.70% | 21.16x |

W3/batch-4 wall time 从 3,182.82 s 降至 280.75 s。W1 长 decode 淹没传输收益，
因此只宣称 TTFT、流量和容量改善，不宣称总吞吐提升。

### 正确性、静态检查与限制

- 极限吞吐 profile 的 1.3K、强制 256 decode、8K、65K 均与 TP2 standalone
  逐 token exact，8K concurrency-4 的八个请求只有一个 token trace；
- W1/W2 batch 1/4 与 W3 batch 1 主测逐 token exact；
- 异步 W3/batch-4 主测 hash 不同。短复现证明 HMA 与 full-context 存在逐 turn exact
  的共同路径，但 HMA 会随实际 batching 组合产生不同 greedy trace，因此本提交没有
  把它描述为“任意异步 batching 全量 exact”；
- 六个 W1/W2/W3 性能点的 NIXL failed transfer/notification 均为 0，且无 CUDA OOM；
- `git diff --check` 与 Markdown lint 通过，测试辅助 Python/Shell 脚本语法检查通过；
- 结果只覆盖同节点 CUDA IPC，不能外推为跨节点 UCX RDMA、多 P/D 或 Gateway 稳态性能。

原始证据：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-saturation-9b9a945c06-20260810/
/mnt/pvc/lidong1/vllm_test_artifacts/pd-w123-hma-ab-9b9a945c06-20260810/
```

本轮 Volcano Job `lidong1-yoco-pd-w123-g4-0810` 已释放，并确认对应 Job/Pod
`NotFound`，没有继续占用 B200 资源。

## 23. 融合 self-attention Q/K RMSClip 与 RoPE

```text
commit: 本提交
subject: perf(yoco): fuse QK clip and rotary
scope: yoco.py, test_yoco_conversion.py, fhb-dev-commit.md
functional behavior change: B200/CUDA BF16 的 YOCO self-attention 将
                            Q RMSClip、K RMSClip、RoPE 三个 kernel 合为一个
runtime base: fhb-dev@fb8c42b6a0
```

### 目的与实现

此前每个 YOCO self-attention 层在 QKV projection 后依次执行 Q RMSClip、K
RMSClip 和 RoPE。即使服务开启 full CUDA Graph，GPU 图中仍保留三个 kernel node，
并需要把两份 clip 中间张量写回、再由 RoPE 读回。Decode 的有效 token rows 很小，
这部分主要受固定调度和中间显存流量影响；30 个 self layer 会把开销重复 30 次。

本提交增加一个 Triton CustomOp，同时读取 packed-QKV split 后非连续的 Q/K view，
按 head 在 FP32 中计算 RMS clip coefficient，并立即完成 rotary。kernel 显式执行
`FP32 -> BF16 -> FP32`，保留原始 `RMSClip -> BF16 tensor -> RoPE` 的数值边界，不能
为了少一次转换而改变训练侧舍入语义。Q/K 在同一个 launch 中处理，输出仍是两个
连续 BF16 tensor，attention 接口没有变化。

融合门禁刻意收窄到已经实测的 CUDA、BF16、head_dim=128、无 weight 的
`RMSClip`，并要求 Q/K 的 eps 与 limit 相同。CPU、其他 dtype/head dimension、
weighted RMSClip、普通 qk_norm 或未安装 Triton 时完整走旧实现，不改变通用 YOCO
路径。当前 checkpoint 的 Q/KV heads 为 64/8，测试另覆盖 32/4 和 48/4，避免把
kernel 偶然写死为唯一 head 数。

### 修改文件

- `vllm/model_executor/models/yoco.py`
    - 新增 fused Q/K RMSClip + rotary Triton kernel、CUDA wrapper、fake impl 和
      CustomOp 注册；
    - `YOCOSelfAttention.forward` 在严格门禁满足时调用 fused op；
    - 保留旧 Q/K per-head norm 与 rotary fallback。
- `tests/model_executor/test_yoco_conversion.py`
    - 对 M=1/7/17/128 与 heads=64/8、32/4、48/4 做逐元素 bit-exact 回归；
    - 增加 `torch.library.opcheck`，覆盖 fake tensor、schema 和动态算子注册。
- `fhb-dev-commit.md`
    - 记录实现边界、正确性、单 kernel、standalone 与 Mooncake 1P1D A/B。

### 单元、静态与数值测试

本地 A6000 完整 YOCO 回归为 `51 passed, 14 warnings`；所有 pre-commit hooks、
`py_compile` 和 `git diff --check` 通过。B200 上 `torch.library.opcheck` 通过；
M=1/2/4/8/16/32/64/128 的 Q/K 输出全部逐元素 bit-exact。

真实 TP1/DP1、BF16、FA4、Triton MoE、full CUDA Graph 的 1,360 -> 512 greedy
生成中，baseline/candidate 的 512-token trace、文本和逐 token logprob 全部 exact，
最大与平均 logprob delta 都是 0。

### B200 CUDA Graph 单 kernel 收益

微基准把 30 个 self layer 放入同一个 CUDA Graph 后再摊到每层：

| M | 旧实现 / layer | fused / layer | 30 层节省 |
| ---: | ---: | ---: | ---: |
| 1 | 3.210 us | 1.365 us | 55.37 us |
| 4 | 8.461 us | 1.706 us | 202.66 us |
| 16 | 9.221 us | 1.708 us | 225.40 us |
| 64 | 10.588 us | 2.118 us | 254.10 us |
| 128 | 12.558 us | 2.602 us | 298.70 us |

这说明收益来自 CUDA Graph 内减少 graph node 与中间显存读写，不依赖关闭 CUDA
Graph 后的 Python/CPU launch 开销。

### Standalone serving 压测

同一张 B200、相同 server 配置顺序跑 baseline/candidate，payload 为
1,360 input + 512 forced output：

| C | output tok/s 旧 -> 新 | 吞吐变化 | mean TPOT 改善 |
| ---: | ---: | ---: | ---: |
| 1 | 140.01 -> 142.48 | +1.77% | +1.54% |
| 4 | 173.00 -> 175.89 | +1.67% | +2.31% |
| 8 | 711.72 -> 733.80 | +3.10% | +2.90% |
| 16 | 1,180.86 -> 1,100.51 | -6.80% | +3.19% |
| 32 | 1,838.42 -> 1,879.31 | +2.22% | +2.30% |
| 64 | 2,930.36 -> 3,183.32 | +8.63% | +2.59% |

六个点的 mean TPOT 全部改善。C16 candidate 的 mean TTFT 偶发升至 1,561 ms，
baseline 为 868 ms，令该点总吞吐反向；因此 standalone output-throughput 几何平均
只有 +1.66%，不把 C16 抖动或 C64 的 +8.63% 单点写成稳定 kernel 收益。

### Mooncake 自带 1P1D 评测

继续使用项目已有的 Mooncake `benchmarks/xypd_benchmarks` matrix：同节点两张固定
B200，1P1D、TP1+TP1、Mooncake 0.3.12.post1、RDMA、FA4、Triton MoE、
FULL_AND_PIECEWISE CUDA Graph；payload 为 1,360 random + 50 shared-prefix input、
512 output，每并发 4 folds，且每个 revision 先做 C64 warmup。

| C | output tok/s 旧 -> 新 | 吞吐变化 | mean TPOT 改善 |
| ---: | ---: | ---: | ---: |
| 1 | 139.07 -> 141.46 | +1.72% | +1.50% |
| 4 | 380.46 -> 394.05 | +3.57% | +3.41% |
| 8 | 690.58 -> 703.45 | +1.86% | +1.82% |
| 16 | 1,091.98 -> 1,132.81 | +3.74% | +2.89% |
| 32 | 1,771.66 -> 1,817.69 | +2.60% | +2.67% |
| 64 | 2,799.05 -> 2,866.92 | +2.42% | +2.26% |

六点 output/request throughput 几何平均提升 **2.65%**，mean TPOT 几何平均改善
**2.42%**。baseline/candidate 各三次真实 Mooncake transfer smoke 内部一致，且两版
输出 exact；全部性能请求失败数为 0。日志中的 failed transfer、failed recv 和 KV
expired request 全为 0，P/D 均明确使用 FA4，无 fallback、traceback、CUDA error 或
OOM。

### 证据、边界与回滚

原始结果同时保存在：

```text
/mnt/pvc/lidong1/yoco-qk-clip-rope-20260830/
/mnt/pvc/lidong1/yoco-qk-clip-rope-e2e-20260830/
/mnt/pvc/lidong1/yoco-qk-clip-rope-mooncake-ab-20260830/
/home/lidong1/vllm_test/yoco_results/qk-clip-rope-20260830/
```

本轮证明的是 B200 BF16、当前 64/8 heads 与 30 self-layer checkpoint；没有外推到
其他 dtype/head_dim、weighted qk norm、多节点 RDMA 或 TP/DP>1。门禁外自动 fallback，
回滚本提交即可恢复三个独立算子，不涉及 checkpoint、KV layout、PD 协议或 Mooncake
状态迁移。
