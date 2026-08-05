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
| 5 | `9988ae737f` | 性能/MoE | Shared Expert 与 Routed MoE 并行 | 已进入 `fhb-dev` |
| 6 | `8eae22948c` | 性能/Activation | FP32 clamped-SwiGLU 单 kernel | 已进入 `fhb-dev` |
| 7 | `1c51cac0d7` | 正确性/PD | Streaming stop 保留 KV metadata | 已进入 `fhb-dev` |
| 8 | `2765c22a1b` | 构建/PD | UCX 1.21 单 runtime 与 SM100 `_C` | 已进入 `fhb-dev` |

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

## 本日志初始化

```text
subject: docs(yoco): add fhb-dev commit ledger
scope: 新建 fhb-dev-commit.md，补录前两个 PR 和第三个 Router commit
functional behavior change: none
```

本日志初始化只增加可审计文档，不改变模型、worker、NIXL 或 kernel 行为。其目的
是用单一文件替代分散的 PR 邮件上下文；后续功能 commit 继续按上面的模板追加。
