# YOCO、LMCache 三级 KV 缓存与 Dynamo 适配方案

状态：设计评审稿，尚未修改运行逻辑、构建镜像或执行 GPU 测试。

基线：`fhb-dev@f54af87aecf78e2db92c15cfa5062fadff1d9509`。

## 结论

首版不要让 LMCache 替换现有 NIXL P/D 数据面。推荐组合为：

- Prefill：vLLM GPU prefix cache + LMCache 跨请求缓存 + NIXL producer；
- Decode：NIXL consumer + vLLM GPU KV cache；
- 三级缓存：GPU、Prefill 本机 pinned CPU、持久后端；
- Dynamo：Frontend、KV Router 和 P/D worker 编排；
- P 到 D 的单请求实时 KV 传输继续使用 NIXL 1.3.2 + UCX 1.21；
- LMCache 首版只部署在 P，负责不同请求之间的 prefix KV 复用；
- 保持 LMCache `use_layerwise=false`，不改变当前完整 CUDA Graph 行为。

首版链路如下：

```text
                                      ┌─ L1: P 本机 pinned CPU
                                      │
request -> Dynamo KV Router -> P GPU ─┼─ L2: 持久后端
                               │      │    首选本地 NVMe/GDS 做性能验证
                               │      └─ 共享 PVC/HF3FS/对象存储做跨 Pod 验证
                               │
                               └─ NIXL 1.3.2 / UCX 1.21 -> D GPU -> decode
```

这里把 vLLM GPU cache 记为第一级、LMCache CPU cache 记为第二级、持久后端记为
第三级。本地 NVMe 和共享远端存储是第三级的两种部署形态，第一轮不要同时启用，
以便精确归因延迟和带宽。

## 为什么采用这个拆分

现有 `fhb-dev` 已经证明 NIXL P/D 的 block-tail correctness，并对 UCX 1.21、
NIXL 1.3.2、TP2/DP1 容量和吞吐做过验证。LMCache 与 NIXL 解决的是两个不同问题：

- LMCache：相同或相似 prefix 在不同请求、不同时间之间复用；
- NIXL P/D：把本次 P 已经准备好的 KV 实时交给本次 D。

如果第一步同时替换 P/D 传输、增加三级存储和切换路由器，出现 token 不一致时
无法判断根因来自 YOCO 共享 KV、chunk 对齐、存储回源还是 P/D metadata。因此首版
保持 NIXL 数据面，只在 P 前面增加 LMCache。

## 固定版本与镜像策略

建议固定以下版本，不使用范围依赖：

| 组件 | 版本/提交 | 说明 |
| --- | --- | --- |
| YOCO vLLM | `fhb-dev@f54af87aec` | 当前方案基线 |
| LMCache | `v0.5.3@140819c9d57a` | 当前稳定 tag，需做 YOCO adapter |
| Dynamo | `v1.3.1@a49702e4432e` | 先作为控制面接入 |
| NIXL | `1.3.2@de8115ca97d3` | 与当前 PD 镜像一致 |
| UCX | `1.21.0@b6a9d47fccce` | 与当前 PD 镜像一致 |
| CUDA | 13 / SM100 | 当前 B200 运行环境 |

`docker/Dockerfile.b200.pd` 当前没有安装 LMCache。LMCache 0.5.3 已发布 CUDA 13
wheel，但新镜像仍从上述固定提交源码构建，使扩展与镜像内 torch C++ ABI、CUDA
13 和 SM100 精确对齐，并在镜像构建阶段验证：

1. `torch`、CUDA 和 B200 SM100 可用；
2. `lmcache` 版本和提交固定；
3. LMCache 和 vLLM 只加载同一个 NIXL 1.3.2；
4. NIXL 的 UCX plugin 只解析到 `/opt/hpcx/ucx` 的 UCX 1.21；
5. 不引入第二套 UCX、NIXL CUDA wheel 或 CUDA 12 runtime；
6. LMCache CPU store/load 的最小 smoke test 通过。

第一轮使用进程内 `LMCacheConnectorV1`。暂不使用 `LMCacheMPConnector`，原因是：

- 多进程 sidecar 会同时引入 GPU KV format、共享内存和 sidecar 生命周期变量；
- Dynamo 1.3.1 的相关文档早于 LMCache 0.5.3 发布，仍提示旧 GPU KV format
  兼容问题；0.5.3 已有新 format 实现，但尚未在本 YOCO fork 上验证；
- 进程内模式足以先验证 YOCO shared-KV、三级缓存和 NIXL 串联语义。

MP sidecar 可以在进程内模式正确后单独做一个功能提交和 A/B。

## 已发现的 YOCO 兼容风险

### 1. 共享 cross KV 不能按 10 个普通层保存

YOCO 配置有 10 个 self layers 和 10 个 cross layers，但当前 checkpoint 还设置了
`universal_loop=3`。三个 self-attention pass 各自拥有 10 份 KV，后 10 个 cross
layer name 则指向同一个 owner tensor。当前 LMCache adapter 直接把
`kv_caches.values()` 转为逐层列表，同时按模型基础逻辑层数构造 LMCache KV shape。

直接启用会产生三个风险：

- 同一份 cross KV 被重复存储 10 次；
- 第二、三轮 self-attention KV 被遗漏；
- LMCache 的基础逻辑层数、展开后的物理 tensor 数和恢复位置不一致。

适配时应在 LMCache connector 边界建立稳定的 physical-layer view：

```text
base logical layers:  self 0..9 + cross 10..19 = 20
physical caches:      0..10, 20..29, 40..49    = 31
cross aliases:        11..19 -> owner 10
```

具体约束：

- connector 同时读取 `KVCacheConfig` 和 `universal_loop`，计算稳定物理顺序；
- worker 按同一顺序只向 LMCache 注册唯一 tensor；
- worker 额外断言 cross 11..19 与 owner 的 device、地址、shape、stride、dtype 一致；
- load 只恢复 owner tensor，其他 cross layer 继续通过 vLLM alias 读取；
- cache namespace 加入 physical-layout version，避免旧的 20-layer 数据被误命中；
- 不修改已经验证过的 NIXL alias 去重路径。

这部分应先写纯单元测试，再改 adapter。

### 2. YOCO 最后一个 prompt block 必须继续由 D 重算

现有正确性规则为：

```text
remote_tokens = floor((N - 1) / block_size) * block_size
tail_tokens   = N - remote_tokens
```

LMCache 命中不能把原 prompt 的最后一个 block 偷渡进 NIXL transfer。首版使用：

```yaml
chunk_size: 16
save_unfull_chunk: false
use_layerwise: false
enable_async_loading: false
```

并为 LMCache connector 设置 `discard_partial_chunks=true`。当前 vLLM block size 为
16；B200 单卡验证表明 `chunk_size=16` 可与原生 prefix cache 使用完全相同的恢复
边界，并实现 text、token IDs 和逐 token logprob exact。较大的 64/128/256 chunk
仍可作为纯性能 A/B，但不能把不同恢复边界与全量重算之间的数值差异归因给
connector。

### 3. 首版不启用 layerwise

LMCache layerwise load/store 需要 PIECEWISE CUDA Graph，而且每个 attention layer 都会
推进一次保存/加载状态。YOCO 展开后有 31 份物理 KV，另有 9 个 cross alias，直接
套用基础 20 层状态机不安全，也会改变当前 decode batch 1--32 的完整 graph 性能。

因此第一轮固定 `use_layerwise=false`。如果非 layerwise 已证明收益，再单独研究
YOCO physical-layer-aware layerwise，而不把它混入基础兼容提交。

### 4. cache key 和 namespace 必须防止错误命中

所有 rank 和 Pod 必须固定 `PYTHONHASHSEED`，并使用稳定 hash，例如
`sha256_cbor`。cache namespace 至少应覆盖：

- 模型权重 revision 和 tokenizer revision；
- served model name；
- KV dtype、layout、head shape；
- TP size、TP rank 和 physical-layer layout version；
- vLLM block size 与 LMCache chunk size；
- YOCO 代码/模型配置中影响 KV 数值的 revision；
- 用户传入的 `cache_salt`。

任一字段不一致都应 cold miss，不能尝试加载旧 KV。

## P/D connector 组合

### 第一阶段：不引入 Dynamo

先在现有 Gateway 上验证标准 `MultiConnector`。P 的顶层角色仍保持 dedicated
producer，子 connector 分工如下：

```text
P top-level: kv_producer
  1. LMCacheConnectorV1: load/save 跨请求 prefix
  2. NixlConnector:      把本次完整 P prefix 发送给 D

D:
  NixlConnector: kv_consumer
```

LMCache 必须排在 connector 列表前面，使 scheduler 优先采用 LMCache 命中；NIXL
producer 不参与外部 prefix lookup。一次 P forward 中，LMCache 负责填充已命中块，
模型计算未命中块，NIXL 随后发送两部分合并后的完整 prefix。

现有 `MultiConnector` 的 load 选择和 save 广播看起来支持该流程，但以下语义必须用
端到端测试证明，不能只凭源码推断：

- LMCache 命中请求仍生成唯一一份有效 NIXL transfer metadata；
- 只有 NIXL child 返回 P/D transfer params；
- LMCache load 失败会回退计算，不会把未填充 block 发给 D；
- P 顶层 `kv_producer` 仍触发 YOCO KV-only fast-prefill；
- DP>1 时每个 rank 的 LMCache hit token 数一致。

如果标准 `MultiConnector` 无法同时满足顶层 producer 和 LMCache `kv_both` 子角色，
再增加一个很薄的 `YocoPdCacheConnector` 协调层；不要把特殊判断散落到模型 forward。

### 第二阶段：接入 Dynamo

Dynamo 1.3.1 已提供 vLLM + LMCache + NIXL 的 `PdConnector` 示例；该 connector 是
围绕 `MultiConnector` 的 P/D 协调层。接入时复用上述已经验证过的 child-role 契约，
而不是直接照搬通用模型示例。

职责划分：

- `dynamo.frontend --router-mode kv`：接入、session/prefix-aware 路由；
- P worker：YOCO 自定义 vLLM + LMCache + NIXL；
- D worker：YOCO 自定义 vLLM + NIXL；
- Planner：最后再启用，根据 P queue、D queue、TTFT 和 ITL 扩缩容；
- KV events：向 Router 报告 GPU/CPU/disk cache 位置。

Dynamo 1.3.1 的 Router 已支持不同 cache tier 的权重，默认 host cache credit 为
0.75、disk cache credit 为 0.25。这些值只是初始值，最终应由实测的“加载时间相对
重算时间”决定。

共享远端 cache 需要额外处理。Dynamo 1.3.1 的 `shared_cache_type` 目前只有
`none` 和 `hicache`，没有可直接查询 LMCache 全局目录的类型。因此分两步上线：

1. 首版保持 `shared_cache_type=none`。Router 使用 GPU/CPU/disk worker events；
   LMCache L2 是否命中由请求到达 P 后再查询。这样路由可能不是最优，但正确；
2. 后续在 Dynamo 增加 LMCache shared indexer adapter，通过 LMCache lookup/control
   API 查询共享 L2，并新增 `shared_cache_type=lmcache`。必须处理 eviction、TTL、
   namespace 和 lookup 超时，超时返回 unknown/cold，不能返回乐观命中。

不建议把一次远端命中伪造为所有 P worker 都有本地 CPU 命中，否则 Router 会高估
热度并在缓存已经驱逐后继续错误路由。

## 三级缓存实现顺序

### 第一级：vLLM GPU KV

- 保持现有 prefix caching、YOCO shared-KV 和 CUDA Graph；
- GPU 命中优先于 LMCache；
- D 的 decode hot KV 只留在 D GPU，首版不写 LMCache；
- preemption 和 block free 仍由 vLLM 管理。

### 第二级：LMCache pinned CPU

- 只在 P 启用；
- 容量按节点可用内存配置，不在代码中写死；
- 先同步 load/store，证明正确后再测试 async loading；
- TP/DP rank 使用独立 worker identity，但 namespace 和 token hash 规则一致；
- 记录有效容量、逐出次数、load/store GB/s 和 pinned-memory 对系统的影响。

### 第三级：持久后端

分两种独立 profile 测试：

1. 本地 NVMe：优先测试 NIXL GDS/GDS_MT，失败时以 POSIX 作为对照；
2. 共享存储：先把已有 PVC 以独立目录挂载到 P Pod，做 POSIX correctness、Pod
   重启后复用和多 P 可见性；确认存储能力后再选择 HF3FS、GDS 或对象后端。

PVC 路径必须包含模型和 layout namespace，例如：

```text
/mnt/pvc/lmcache/yoco/<model-revision>/<layout-version>/
```

不要把测试 artifact 目录和在线 KV 目录混用。不同测试 run 使用独立 namespace，
测试完成后通过 TTL/显式清理回收，不按文件名猜测删除。

## 分提交实施计划

每个提交继续追加 `fhb-dev-commit.md`，一次只做一个功能点：

1. `docs(yoco): add LMCache and Dynamo adaptation plan`
   - 仅本方案和提交记录；不改行为。
2. `build(yoco): pin LMCache 0.5.3 in B200 PD image`
   - 只改镜像和镜像验证；不默认启用 LMCache。
3. `test(yoco): cover LMCache physical shared-KV layout`
   - 先增加 shared owner、alias、shape 和 namespace 失败测试。
4. `fix(yoco): deduplicate LMCache shared KV tensors`
   - 只实现 physical-layer view；运行 CPU/mock 和 B200 correctness。
5. `feat(yoco): compose LMCache reuse with NIXL producer`
   - 只增加 P 端 connector profile/launcher；D 不启用 LMCache。
6. `feat(yoco): add LMCache persistent tier profile`
   - 先本地 NVMe，再共享 PVC；二者数据分别报告。
7. `feat(yoco): add Dynamo YOCO deployment profile`
   - Frontend、P/D registration、KV events 和回退；不同时改模型算子。
8. `feat(dynamo): add LMCache shared-cache indexer`
   - 在 Dynamo 仓库独立提交；只有有实测路由收益后才启用。
9. `perf(yoco): tune LMCache chunk and async mode`
   - 纯性能提交；无收益就保留负结果说明并放弃默认启用。

## 正确性测试矩阵

### 输入和命中形态

| 维度 | 必测值 |
| --- | --- |
| Prompt | 1,356；1,356 + 强制 256 decode；7,999；65,809 tokens |
| vLLM block 边界 | `B-1`、`B`、`B+1`，当前 `B=16` |
| LMCache chunk 边界 | 255、256、257，以及多 chunk partial hit |
| 命中 | cold、GPU hit、CPU full hit、CPU partial hit、L2 full/partial hit |
| 生命周期 | eviction、cancel、preemption、P/D timeout、Pod restart |
| Cache key | 相同 revision 命中；不同 revision/dtype/TP/namespace 强制 miss |

每个 case 比较 standalone reference、LMCache standalone 和 LMCache→NIXL→D：

- 逐 token ID exact；
- 首 token logprob exact；
- 最终文本 exact；
- D 只重算一次 YOCO final block tail；
- cross owner 只保存一份，不能出现 10 倍存储；
- load failure 后要么安全重算，要么 fail closed，不能生成错误 token。

### 并行拓扑

按以下顺序扩展，前一层通过后再进入下一层：

1. 单卡 standalone LMCache CPU；
2. 1P1D，P/D 各 TP1；
3. 1P1D，P/D 各 TP2，共 4 x B200；
4. pure P DP2 + D，验证 DP rank hit 一致和 hash 一致；
5. TP2/EP2；
6. CP/DCP 只在基础路径稳定后测试，不能沿用 TP 的结论。

### CUDA Graph

第一轮必须同时记录：

- 当前默认完整 CUDA Graph；
- eager 对照；
- LMCache cold 和 warm；
- 暂不把 layerwise PIECEWISE 结果写成默认路径。

## 性能测试和报告口径

固定同一模型、镜像、GPU、P/D 拓扑、prompt、decode 长度和请求顺序，比较：

| Profile | 用途 |
| --- | --- |
| A. 当前 NIXL PD，无 LMCache | baseline |
| B. LMCache 已启用但 cold | 量化 connector 固定开销 |
| C. LMCache CPU warm | 量化第二级收益 |
| D. LMCache persistent warm | 量化第三级收益 |
| E. Dynamo routing | 量化路由收益和控制面开销 |

每个 profile 至少报告：

- P time、NIXL transfer time、D time、TTFT、ITL；
- input/output tok/s、request QPS、p50/p95/p99；
- GPU、CPU、持久层 hit tokens 和 hit rate；
- LMCache lookup、load、store 延迟与 GB/s；
- CPU pinned memory、NVMe/PVC 带宽和 GPU memory；
- batch/concurrency 1、4、8、16、32；
- 交替 A/B 至少 5 个 warmed samples，中位数和 coefficient of variation。

首版验收标准：

- 所有必测 correctness case exact；
- 不存在 stale/跨 namespace 错误命中；
- LMCache cold 相对 baseline 的固定开销可单独解释；
- warm load 只有在端到端 TTFT/吞吐优于重算时才启用；
- 1.3K 短请求若无收益，Router 直接送 D 或跳过 L2，不强行使用 LMCache；
- 任何性能结论都不能用串行 correctness harness wall time代替在线吞吐。

## 失败回退与上线顺序

建议 feature gate：

```text
YOCO_LMCACHE_ENABLED=0
YOCO_LMCACHE_PERSISTENT_TIER=0
YOCO_DYNAMO_KV_ROUTING=0
```

回退顺序：

1. LMCache lookup/load 超时：按策略回退 P 本地计算；
2. persistent tier 异常：关闭第三级，保留 CPU cache；
3. CPU cache 异常：关闭 LMCache，恢复当前 NIXL-only P/D；
4. Dynamo KV events/indexer 异常：退回 queue/load 路由，不影响 NIXL 数据面；
5. NIXL transfer 失败：维持当前 fail-closed 或 Gateway 重试到 D 本地计算策略。

灰度顺序为单 P、单 P/D、TP2、DP2、两组 P/D，再扩大副本。每一步都保留同配置
NIXL-only baseline，不能原地覆盖唯一可工作的运行 profile。

## 本轮评审后第一步

方案通过后，先做第 2 和第 3 个提交：构建固定 LMCache 0.5.3 的 CUDA 13/B200
镜像，并增加 shared-KV layout 单测。此时仍不申请大规模资源；本地检查通过后申请
2 张 B200 做单卡 CPU tier 和 1P1D correctness。只有 exact 通过，才扩展到 4 张卡、
持久层和 Dynamo。
