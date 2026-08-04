# `fhb-dev` 功能提交记录

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
7. 本文只逐条记录功能 commit。仅用于补充本日志的文档提交不递归记录自身 hash；
   文档历史可通过 `git log -- fhb-dev-commit.md` 审计。

## 提交索引

| 序号 | Commit | 类型 | 功能 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | `85eab7b56e` | 性能/PD | KV-only producer 跳过 cross layers | 已进入 `fhb-dev` |
| 2 | `ea4f80d1b4` | 正确性/NIXL | 去重 KV-sharing alias 注册 | 已进入 `fhb-dev` |
| 3 | `d11cc022bd` | 性能/Router | 删除 dense routing materialization | 已进入 `fhb-dev` |
| 4 | `ef60ee0255` | 性能/Norm | 融合 residual add 与 RMSNorm | 已进入 `fhb-dev` |

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

## 本日志初始化

```text
subject: docs(yoco): add fhb-dev commit ledger
scope: 新建 fhb-dev-commit.md，补录前两个 PR 和第三个 Router commit
functional behavior change: none
```

本日志初始化只增加可审计文档，不改变模型、worker、NIXL 或 kernel 行为。其目的
是用单一文件替代分散的 PR 邮件上下文；后续功能 commit 继续按上面的模板追加。
