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

pure-P 请求即使服务全局开启 prefix caching，也会在 `on_new_request` 阶段先截到
上述边界并设置 `skip_reading_prefix_cache=true`。这是 correctness 约束：实测若让
1,344-token P prefix 命中 1,328 tokens、只重算最后 16 tokens，再把结果传给一个
关闭本地 cache 的独立 D，1.3K 和 65K 都不再逐 token exact。保守实现因此放弃 P
请求级 GPU prefix 复用，保证 fresh 和重复 P 请求都保持同一个完整 prefix shape；
D 的 prefix cache 不受影响。

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
- 服务可以全局开启 prefix caching，但 YOCO pure-P 请求会在 connector 内按请求
  跳过本地 prefix 读取，避免重复请求改变 producer forward shape；
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

### TP2 + TP2 极限吞吐补测

`fhb-dev@9b9a945c06` 在同节点 4 x B200 上把 P token budget 提到 32K，并分别把
D CUDA Graph 捕获到 batch 128 和 256。1.3K、强制 256 decode、8K、65K 均与
TP2 standalone 逐 token exact；当前观测峰值为：

| 口径 | 峰值 | 并发 | 备注 |
| --- | ---: | ---: | --- |
| P，8,192 effective input | 84,055 input tok/s | 64 | 96 后回落 |
| D，1,345 context + 512 output | 2,284 output tok/s | 256 | p50 57.11 s |
| PD，1,345 input + 256 output | 1,128 output tok/s / 4.407 req/s | 96 | p50 21.68 s |

在线建议 P admission target 为 48--64、D 为 64--96；D graph 仍 capture 到 256，
避免突发超过 32 后落回 eager。batch 256 是吞吐极限点而非延迟推荐点。完整参数、
曲线和边界见 `YOCO-PD-BATCH-CURVE-20260810.md`；原始证据位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-saturation-9b9a945c06-20260810/
```

本轮是同节点 `lo/cuda_ipc`，未覆盖跨节点 UCX RDMA；固定波次也不替代生产 Gateway
的开环 arrival-rate、TTFT/ITL SLO 和多 P/D 负载均衡验收。

## SWA window 传输补充与 B200 实测

YOCO checkpoint 的真实物理 KV 布局不是一个 SWA group，而是：

```text
30 x physical self-attention SWA groups: window=512, block_size=16
 1 x shared cross-attention owner group: full attention, owner layer=10
```

NIXL scheduler 对每个 SWA group 在 metadata 中保守保留
`ceil(512 / 16) + 1 = 33` blocks，以覆盖 block 边界重叠；worker 在已对齐的实际
payload 中发送 32 blocks/group。最后一个 cross-owner group 仍发送完整 prefix，
不能裁成窗口。65,809-token 请求因此是：

```text
scheduler metadata: 30 x 33 + 4,113 = 5,103 blocks
actual payload:      30 x 32 + 4,113 = 5,073 blocks
bytes/block:         65,536 bytes（全局 8 KV heads，BF16 K+V）
actual bytes:        332,464,128
```

在 `bonete01/lidong1-yoco-pd-diagnose-g4-0810` 的 4 x B200、P/D 各 TP2 上，
HMA window 与 `--disable-hybrid-kv-cache-manager` full-context baseline 各执行三轮。
18 个计入样本全部 text 和逐 token trace exact，NIXL failed transfer/notification
均为 0：

| Prompt | HMA bytes | Full-context bytes | 流量缩减 | HMA D time | Full-context D time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,356 | 68,419,584 | 170,655,744 | 2.49x | 0.212 s | 0.436 s |
| 7,999 | 95,617,024 | 1,013,776,384 | 10.60x | 0.380 s | 2.429 s |
| 65,809 | 332,464,128 | 8,356,036,608 | 25.13x | 0.909 s | 18.835 s |

随后又做了更严格的重复请求隔离门禁：P 服务全局开启 prefix cache，D 关闭；每档
复用相同 cache salt 两次，强制第二次请求仍真实经过 NIXL。connector 的请求级
bypass 生效后，1.3K/8K/65K 共 6/6 样本逐 token exact；每次仍发生 2 个 TP rank
transfer，SWA metadata 分别为 `30x33+84`、`30x33+499`、`30x33+4113`，NIXL
错误为 0。代表性第二轮结果为：

| Prompt | P time | D time | 串行总时延 | NIXL bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1,356 | 0.0710 s | 0.2134 s | 0.2844 s | 68,419,584 |
| 7,999 | 0.1277 s | 0.3798 s | 0.5076 s | 95,617,024 |
| 65,809 | 0.8772 s | 0.9032 s | 1.7807 s | 332,464,128 |

原始证据目录：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-swa-transfer-53f58e5426-20260810/
/mnt/pvc/lidong1/vllm_test_artifacts/pd-nixl-hma-p-cache-bypass-53f58e5426-20260810/
```

结论是“最后一层”需要更精确地表述为“唯一 full cross-owner group”：30 个 self-KV
只传一个 512-token SWA 窗口，cross-owner 仍传完整上下文。该优化已经实现并在
1.3K--65K 上得到正确性和流量收益验证；它依赖 HMA，任何不支持 HMA 的 child
connector 都会使 MultiConnector 回退到 full-context 布局。

### W1/W2/W3 HMA 对 full-context 实测

在同节点 4 x B200、`P=TP2/DP1`、`D=TP2/DP1` 上进一步执行完整 W1/W2/W3
A/B。HMA 与 `--disable-hybrid-kv-cache-manager` 的主要结果为：

| Workload | Batch | 吞吐变化 | TTFT 缩短 | 流量缩减 |
| --- | ---: | ---: | ---: | ---: |
| W1 | 1 / 4 | -3.14% / -2.49% | 84.59% / 89.92% | 10.77x |
| W2 | 1 / 4 | +12.79% / +43.37% | 91.56% / 94.94% | 25.11x |
| W3 | 1 / 4 | +553.02% / +1033.70% | 93.87% / 94.34% | 21.16x |

W3 batch 4 从 `3182.82 s / 16.34 tok/s` 改善到
`280.75 s / 185.22 tok/s`。W1 的 65K 长 decode 淹没了起始传输收益，所以应表述为
TTFT/容量优化，而不是吞吐优化。

W1/W2 batch 1/4 与 W3 batch 1 主测逐 token exact。异步 W3 batch 4 的主测最终
hash 不同；追加三次短复现显示 HMA 自身会随实际 batching 组合产生不同 greedy
trace，而 full-context 的三次 trace 与 HMA 前两次逐 turn exact。因此没有证据表明
窗口传输损坏 KV，但“任意异步 batching 都逐 token exact”的严格门禁仍未通过。

完整环境、逐点 wall/tok/s/bytes、首次分叉轮次和复现矩阵见
[`YOCO-PD-W123-HMA-AB-20260810.md`](YOCO-PD-W123-HMA-AB-20260810.md)。

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
