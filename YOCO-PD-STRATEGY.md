# YOCO Prefill/Decode 分离策略

## 结论

当前 `fhb-dev` 推荐采用“纯 P 节点 + 纯 D 节点 + Gateway 两阶段请求”的
NIXL P/D 分离：

- P 服务固定为 `kv_producer`，只做 prompt prefill 和 KV 发布；
- D 服务固定为 `kv_consumer`，接收 KV，并生成第一个及之后所有用户可见 token；
- Gateway 必须丢弃 P 返回的 sampled token，只传递 KV metadata；
- YOCO prompt 的最后一个 KV block 固定由 D 重算，使 standalone、本地 prefix
  cache 和远端 NIXL KV 三条路径的计算 shape 一致；
- 传输使用 NIXL 1.3.2 + 单一 UCX 1.21 runtime，生产部署按 GPU/NIC 拓扑选择
  RDMA HCA；
- 纯 P 服务可以使用 DP1、DP2、DP4、DP8，不要求 DP1。DP>1 时必须保持纯 P
  角色，不能向同一服务混入普通 decode 请求。

当前最完整的端到端证明是 B200 同节点 1P1D correctness。DP4 纯 P 也已做过
独立正确性和吞吐测试；多节点 RDMA、1P2D/多 P 多 D 的生产吞吐和故障切换仍需
单独验收。

## 推荐拓扑

```text
                         ┌────────────────────────────┐
                         │          Gateway           │
                         │  session / P-D placement   │
                         └─────────────┬──────────────┘
                                       │ 1. prompt, max_tokens=1,
                                       │    do_remote_decode=true
                                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Pure Prefill service                                                  │
│ kv_role=kv_producer, DP=N, TP=1                                       │
│ YOCO KV-only fast prefill -> publish complete prefix blocks           │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ 2. KVTransferParams
                                │ 3. NIXL / UCX transfers GPU KV
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Decode service                                                        │
│ kv_role=kv_consumer                                                   │
│ receive prefix KV -> recompute final block tail -> visible decode     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ 4. first and all later visible tokens
                                ▼
                              Client
```

推荐先以 1P1D 作为正确性和运维基线，再由 Gateway 扩展到多个 P/D endpoint。
P 和 D 的数量不需要相等：P 按 prompt token 压力扩容，D 按 active sequence、输出
token 和 ITL 压力扩容。扩容前必须先补齐 endpoint 选择、session affinity、容量
反馈、超时和失败隔离，不能仅靠随机轮询。

## 一次请求的完整流程

### 1. Gateway 选择 P 和 D

Gateway 为请求选择一个 P endpoint 和一个 D endpoint，生成唯一 request ID，
并保留原始 prompt、sampling 参数、cache salt 和 session 路由状态。P/D 必须使用
兼容的模型、tokenizer、dtype、KV block size 和 KV layout。

### 2. Gateway 请求 P

P 请求使用原始 prompt，但强制 `max_tokens=1`，并携带：

```json
{
  "do_remote_decode": true,
  "do_remote_prefill": false,
  "remote_engine_id": null,
  "remote_block_ids": null,
  "remote_host": null,
  "remote_port": null
}
```

专用 `kv_producer` 会启用 YOCO KV-only fast-prefill：self block 生成共享 K/V，
十个 cross layers 对纯 P 请求全部跳过。P 为了完成 vLLM 请求仍产生一个 sampled
token，但该 token 没有经过完整 D 路径，只是 disposable token。

### 3. Gateway 丢弃 P token，只保留 metadata

Gateway 必须丢弃 P response 的整个 `choices`，从最终 response 中读取
`kv_transfer_params`。Streaming stop 路径也必须等 EngineCore 正常结束并返回
connector metadata，不能把 stop 当成 abort。

多节点部署时，metadata 中的 `remote_host` 必须是 D 可以访问的 P side-channel
地址，不能使用测试环境的 `127.0.0.1`。Gateway 不应修改 remote engine、request、
block IDs、port 或 TP size。

### 4. Gateway 请求 D

Gateway 把原始 prompt、原始 sampling 参数和 P 返回的完整
`kv_transfer_params` 一起发送给 D。D 使用相同 request ID，异步拉取 prefix KV；
KV ready 后才进入本地尾段计算和 decode。

D 生成第一个以及之后全部用户可见 token。P token 不能拼到 D 输出前面，也不能
作为 speculative token 使用。

### 5. 释放和租约

P 在 transfer 完成通知前保留 KV blocks；heartbeat 会在 D 排队期间延长 lease。
当前验证配置使用：

```json
{
  "kv_lease_duration": 60,
  "bidirectional_kv_xfer": true,
  "decoder_kv_blocks_ttl": 600
}
```

`bidirectional_kv_xfer` 用于多轮场景中复用 D 已有 KV。Gateway 若要使用该能力，
必须保存上一轮 D metadata、保持 session affinity，并正确构造下一轮 P 请求；只
打开配置但丢失 metadata 不会自动获得多轮复用。

## YOCO 的 block-tail 一致性规则

设原 prompt token 数为 `N`，KV block size 为 `B`。YOCO 远端传输的 prefix
token 数为：

```text
remote_tokens = floor((N - 1) / B) * B
tail_tokens   = N - remote_tokens
```

因此至少最后一个 prompt token 所在的完整 block 由 D 本地重算。例如 `B=16`：

| Prompt | P 发布 | D 本地计算 |
| ---: | ---: | ---: |
| 12 | 0 | 12 |
| 16 | 0 | 16 |
| 17 | 16 | 1 |
| 32 | 16 | 16 |
| 44 | 32 | 12 |
| 1,356 | 1,344 | 12 |

这条规则同时应用于三条路径：

- standalone fresh 在最后 block 起点拆分 prompt；
- 本地 prefix-cache hit 从相同边界重算尾段；
- NIXL P 只发布到该边界，D 只接收这些完整 blocks，再重算尾段。

不能先把完整尾块传给 D，再只回退 `num_computed_tokens`。这种做法会让 remote
尾块仍留在 D block table 中，属于“伪重算”，已被 run21 的 8K 回归否定。

NIXL 截断后的 P 请求带 `_p_side_truncated` 标记，通用 scheduler 不再把对齐后的
P prefix 二次拆分。例如 1,344-token P prefix 必须保持一次 1,344-row forward，
不能变成 `1,328 + 16`。

YOCO FP32 TF32 Router 对 token rows 敏感。当前实现把小于 128 行的 Router
输入补零到 128 行再裁回，使小 batch 使用固定 GEMM shape。该规则有明确的小
shape 开销，因此属于 correctness 约束，不应宣称为 kernel 性能优化。

## P 与 D 的服务角色

### P：纯 `kv_producer`

推荐配置：

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backends": ["UCX"],
    "bidirectional_kv_xfer": true,
    "kv_lease_duration": 60,
    "decoder_kv_blocks_ttl": 600
  }
}
```

部署契约：

- 必须开启 `--kv-sharing-fast-prefill`；
- 每个 P 请求必须是 `max_tokens=1`、`do_remote_decode=true`；
- P 服务只接收 P 请求，不提供普通 completions/chat decode；
- DP 可以大于 1。空闲 DP rank 会执行 KV-only dummy forward，保持 MoE
  collective 次数一致；
- TP1 是当前 YOCO 验证配置；改变 TP、KV layout 或 block size 必须重测 P/D
  metadata 和 exact match。

`kv_both` 只保留 DP1 request-level 兼容路径。DP>1 的 `kv_both` 不能证明整个
服务都是 pure-P，因此不会使用专用 producer 的静态 KV-only 优化；生产 P 节点
不推荐使用它。

### D：独立 `kv_consumer`

推荐使用相同 connector 配置，仅将角色改为：

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_consumer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backends": ["UCX"],
    "bidirectional_kv_xfer": true,
    "kv_lease_duration": 60,
    "decoder_kv_blocks_ttl": 600
  }
}
```

D 应保留完整 YOCO cross/decode 路径和 CUDA Graph，不能启用 P 的 KV-only
行为。`kv_load_failure_policy=fail` 是推荐生产策略：传输失败立即失败并由 Gateway
重试/换 endpoint。使用 `recompute` 会把长 prefill 突然压到 latency-sensitive D，
阻塞其他 decode 并放大 tail latency。

## 共同的模型参数

P/D 至少应保持以下参数一致：

```text
model / served model name
tokenizer and revision
dtype = bfloat16
attention backend = FLASHINFER
MoE backend = triton
KV block size and KV cache layout
--enable-prefix-caching
--enable-chunked-prefill
--kv-sharing-fast-prefill
```

当前长上下文配置使用 `FULL_AND_PIECEWISE` CUDA Graph。P 和 D 可分别选择不同的
`max_num_seqs`、GPU memory utilization 和 batch token budget，但每次修改后要
重新检查 chunk shape、显存、TTFT/ITL 和 exact match。run25 correctness profile
使用 `max_num_batched_tokens=8192`；这不是生产吞吐的固定最优值。

## NIXL 与 UCX runtime

当前 PD 镜像由 `docker/Dockerfile.b200.pd` 构建，固定：

```text
UCX:        1.21.0 @ b6a9d47fccce849c28111f05a7fa8f1c930ff17d
NIXL:       1.3.2  @ de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72
CUDA arch:  SM100 / B200
UCX root:   /opt/hpcx/ucx
```

镜像内只能有这一套 UCX。启动后应对 P/D PID 执行：

```bash
verify-single-ucx <pid>
```

### Side channel

每个 worker 需要唯一的 `VLLM_NIXL_SIDE_CHANNEL_PORT`。DP rank 使用
`base_port + dp_rank`；不同主机可以复用相同 base port。多节点时：

```text
VLLM_NIXL_SIDE_CHANNEL_HOST=<P 或 D 对端可访问的本机地址>
VLLM_NIXL_SIDE_CHANNEL_PORT=<该实例唯一 base port>
```

### 数据通道

run25 是同节点 correctness，使用 `UCX_NET_DEVICES=lo` 和 TCP/IPC，不能证明
跨节点 RDMA 带宽。生产环境不得照抄 `lo`：

- 枚举有有效 InfiniBand GID 的 HCA；
- 为每个 GPU/rank 选择 NUMA/GPU-local 的 `mlx5_x:1`；
- 确保 `/dev/infiniband/*`、memlock 和 GPUDirect RDMA 前置条件可见；
- 记录 UCX endpoint、NIXL metrics 和实际 PID library maps；
- 做跨节点带宽、TTFT、ITL、并发和故障注入验收。

UCX/NIXL 是 P/D KV transport；DeepEP 使用 NVSHMEM/IBGDA 做 MoE all-to-all，
两者不是同一条通信栈。修好 UCX 不等于 DeepEP LL 的宿主条件已满足。

## Gateway 必须实现的能力

当前仓库具备 vLLM/NIXL engine 和 response metadata 链路，但没有把完整生产
Gateway 固化为本报告的一部分。Gateway 至少需要：

1. P/D endpoint 注册、健康检查和容量感知选择；
2. 两阶段请求状态机，严格丢弃 P choices；
3. 原始 prompt/sampling 参数在 P/D 两次请求间保持一致；
4. 透传完整 `kv_transfer_params`，并把 P side-channel 地址改为 D 可访问地址；
5. request ID 唯一性、超时、取消和 connector block 释放；
6. session affinity 和可选的 D->P bidirectional metadata 保存；
7. P 失败、D 拉取失败、D decode 失败分别计数，不做无限隐式重试；
8. 对 streaming stop 等终态保留最后一个 metadata-bearing chunk；
9. P token、P/D 时延、KV bytes、external cache hit、lease expiry 和 NIXL error
   的独立可观测性。

对于少于或等于一个 KV block 的 prompt，当前正确性语义是 P 保持有效请求，但 D
不拉取 partial block 并完整本地计算。生产 Gateway 可进一步设置短请求阈值，直接
路由到 D，避免一次无收益的 P 请求；该阈值应根据实际 TTFT A/B 决定。

## 当前验证结果

最终 run25 在同一 B200 节点使用真实 1P1D、NIXL 和两阶段 HTTP 请求，结果：

| Prompt tokens | 结果 | P time | D time | 串行总时延 |
| ---: | --- | ---: | ---: | ---: |
| 1,356 | exact | 0.0692 s | 0.2609 s | 0.3301 s |
| 7,999 | exact | 0.1549 s | 0.4190 s | 0.5739 s |
| 65,809 | exact | 1.1278 s | 1.1470 s | 2.2748 s |

1.3K standalone fresh、两次本地 prefix-cache hit 和 PD 输出均为
`" 00663\n"`；本地命中 1,344 tokens，首 token logprob 逐位一致。原始证据：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-current-4a39087d27-20260805/run25-yoco-nixl-shape-aligned-pd-20260809
```

表中 Total 是先请求 P、再请求 D 的 correctness harness 串行时间，不是并发
吞吐，也不表示相对 standalone 的加速。

较早的 DP4 pure-P 验证覆盖 1,356、12,096、43,704 prompt tokens，D 输出均与
standalone exact match，并证明纯 `kv_producer` 不需要限制 DP1。但这不替代
当前 commit 在多 P/D 或 DP D 端的重新验收。

## 上线前检查表

- [ ] P 使用纯 `kv_producer`，没有普通 decode 流量；
- [ ] D 使用独立 `kv_consumer`，用户只能看到 D token；
- [ ] P/D 镜像、模型、tokenizer、dtype、block size 和 KV layout 兼容；
- [ ] `--kv-sharing-fast-prefill`、prefix cache 和 chunked prefill 已启用；
- [ ] P 请求固定 `max_tokens=1` 且 Gateway 丢弃整个 P choices；
- [ ] Gateway 完整保留 final response 的 KV metadata；
- [ ] side-channel host 对对端可达，所有实例/rank port 唯一；
- [ ] `verify-single-ucx <pid>` 在 P/D 均通过；
- [ ] 多节点使用 GPU-local RDMA HCA，不使用 `lo`；
- [ ] `kv_load_failure_policy=fail`，Gateway 有有界重试和 endpoint 隔离；
- [ ] 1.3K/8K/65K fresh、local-cache、PD exact-match 全通过；
- [ ] 做真实并发 QPS、TTFT、ITL、P/D queue 和 KV transfer 带宽测试；
- [ ] 做 P crash、D crash、transfer timeout、lease expiry 和取消请求测试；
- [ ] 确认多轮 session 的 D->P metadata 保存和 affinity，或明确关闭该能力。

## 尚未完成的生产验收

- 跨节点 UCX RDMA 的带宽、稳定性和 tail latency；
- 1P2D、多 P 多 D 的负载均衡、session affinity 和扩缩容；
- D 端 DP>1、不同 P/D DP 组合及高并发 correctness；
- preemption、KV lease expiry、endpoint 重启和故障恢复；
- Router `<128` padding 后的完整并发 decode QPS/ITL A/B；
- prompt embeddings、多模态输入和异构 KV layout；
- production Gateway 的代码、部署清单和 Dashboard 告警规则。

在这些项目完成前，当前策略可以作为功能正确的开发/验收基线，但不应直接表述
为已经完成多节点生产化。
