# YOCO fast-prefill execution overhead fix

2026-10-21，父提交 `4fd4c9b777c6fa99398b2037339db85bd27595f5`，分支
`fhb-dev-aligndeepinfer-startpoint`。此前 V2 集成的原始回退面板仍保留在
[原报告](../v2-fast-prefill-20261021/REPORT.md)，以下是新的同进程交错对照。

## 最终执行规则

Shared-KV 的纯中间 prefill batch 不会输出 token，也不会使用该步的 draft。
原实现却每个 chunk 都运行 cross 和 K8；现在保留完整 self/KV 写入，跳过这些
无用执行，并清空未使用的 proposal 槽。所有请求都必须尚未完成 prefill 才能走
这条路径；完成 prompt 的 batch、混合 decode、prompt logprobs 等完整行消费者
均保留有效最终 hidden。专用 `kv_producer` 同样跳过无用的 draft。

在已有 `b200-l3-fp8-tp1-k8` 自动配置中，需要输出的 batch 使用完整目标模型。
这样避免 host 主导的 prefill 被拆为两个编译单元、小矩阵 cross、gather/scatter
和完整输出重建，同时使用普通目标模型的 raw h20 返回。完整 prefill metadata
在 attention 构建之前选择，不能拿 compact attention metadata 执行全量 query。

这是“该步是否需要输出”的语义判断，没有新增长度或并发阈值。配置匹配复用现有
MoE 自动 profile：B200、L3、FP8、TP1/PP1/DP1/EP1、Shared-KV K8 和
FULL_DECODE_ONLY。manual/unprofiled 配置及 BF16/Align 保留原 compact 路径；
本轮没有为未测并行配置推断新的选择。

没有 compact metadata 的 eager 尾部复用普通完整目标模型。首次 profiling 仍
预热完整模型与 split blocks，FULL decode 继续复用原图。目标 KV owner、缓存
完成/释放及 P/D handshake 策略不变。实际输出时 h20 仍是最终归一化前的 raw
residual；纯中间 chunk 的辅助占位只满足 runner 返回结构，禁止用于 drafting。

## 验证设计

一个原生 V2 引擎、同一份已加载权重与 tokenizer、同一 KV allocation、同一组
target/draft CUDA Graph。在独立 benchmark worker 中切换三种模式：

- `dense`：关闭模型 fast-prefill 与 helper，执行原全量路径。
- `oldfast`：加载父提交的 prefill wrapper，关闭新加的中间 chunk / 完整行选择
  hook。使用相同 runner 和原数学实现，重现原 compact 路径。
- `newfast`：运行本提交的执行规则。

每个单元、每个模式 2 次预热，随后按六个排列交错运行。短输入和 8193-token
的 C1/C4/C8 各使用 6 个独立 seed；额外 1024/8192-token 的 C1/C4 各 3 个 seed。
每请求 T1、固定 128 输出、忽略 EOS；APC 关闭，batch budget 2048，max seqs 8，
KV 8 GiB，Mega hybrid + FA4，BF16 KV。采用同步 in-process 调度，正式窗口包含
prefill、draft、target、采样、调度和 GPU drain，排除加载/编译、HTTP 和 profiler。

三种模式的图在切换前已捕获，切换不捕获额外图。检查每请求输出预算、corruption
与抢占；保存 token、请求时间戳、原始 metrics counter 和 worker 配置。
同 seed 模式间配对分析，计时重复不冒充独立轨迹。

| 输入 | C | 全量 tok/s | 旧 fast tok/s | 修复后 tok/s | 对旧 fast | 对全量 | 全量/旧 fast/修复后 TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| short | 1 | 334.32 | 306.40 | 334.04 | +9.02% | -0.08% | 94.08 / 102.98 / 94.36 |
| short | 4 | 867.91 | 871.86 | 866.96 | -0.56% | -0.11% | 94.35 / 103.31 / 94.87 |
| short | 8 | 1437.60 | 1427.31 | 1435.97 | +0.61% | -0.11% | 96.29 / 105.65 / 96.63 |
| 8193 | 1 | 162.68 | 153.35 | 174.98 | +14.10% | +7.56% | 468.41 / 514.39 / 412.65 |
| 8193 | 4 | 234.58 | 213.08 | 240.81 | +13.01% | +2.65% | 1073.99 / 1192.36 / 1022.05 |
| 8193 | 8 | 258.39 | 236.05 | 262.25 | +11.10% | +1.49% | 1877.27 / 2071.71 / 1820.70 |
| 1024 | 1 | 280.44 | 287.10 | 280.21 | -2.40% | -0.08% | 92.58 / 103.15 / 93.24 |
| 1024 | 4 | 619.65 | 600.10 | 618.16 | +3.01% | -0.24% | 194.86 / 216.49 / 196.22 |
| 8192 | 1 | 151.09 | 143.45 | 161.85 | +12.83% | +7.12% | 467.84 / 514.73 / 412.98 |
| 8192 | 4 | 230.25 | 211.95 | 235.95 | +11.32% | +2.48% | 1066.63 / 1190.64 / 1012.12 |

正式 144 个 batch、558 个请求、71424 个输出 token，另 60 个预热 batch；无 corruption、抢占或输出预算不足。

短输入 C1/C4/C8 相对全量吞吐变化 −0.08%/−0.11%/−0.11%，首 token 时间恢复到全量水平。8193-token C1/C4/C8 相对旧 fast 分别提升 14.10%/13.01%/11.10%，相对全量提升 7.56%/2.65%/1.49%；TTFT 也降低。1024/8192 两组额外边界对照见表。

48 组配对中，修复后与全量基线的输出 token 和 accepted/draft/request-round 计数全部相同；修复后单个模式共 186 个正式请求、23808 个输出 token。首 token 的四条 top-20/logprob 面板与全量完全一致，三种模式完整行 teacher-forced 的 256 个位置也完全一致。

旧 compact 路径的输出/接受率可以与全量不同，所以对旧 fast 的完整请求变化同时包含执行开销和轨迹恢复。short/C4 与 1024/C1 对旧 fast 分别 −0.56%/−2.40%，不宣称所有场景相对旧数值轨迹都更快；它们与全量分别只差 −0.11%/−0.08%。本面板的精确一致不外推为一般 prompt 分布或任务质量证明。原始计数与输入 SHA 可在 summary、trials、quality 回执中核对。

## 原因核对与实施取舍

修复前的干净 8193/C1 profile 确认 5 次 self、5 次 cross 和 5 次 K8 图启动，
共 200 次目标 MoE 调用。最终 profile 另外在正式窗口外采集，核对实际减少的
执行，以及目标 4 / draft 6 个 FULL 图的真实回放。

曾验证仅对选中行融合 norm/raw-hidden 写回：单元逐位测试通过，但短输入整请求
未显示稳定收益。对 compact cross 单独切换 DeepGEMM 的同进程小面板也没有收益。
最终保留已有数值原语，收敛为上述执行规则，没有合入新的归一化 kernel 或 MoE
后端选择。探索日志和源补丁保留在原始归档，不计为最终实现的性能收益。

## 测试、P/D 与范围

- B200 最终模型/prefill/Shared-KV 79 项、V2 runner/attention/sampler/rejection 74 项，共 **153 项通过，无跳过**；本地对应集合 125 项通过，4 项 SM90+ 用例由 B200 覆盖。
- BF16 greedy + Mega + V2 + K8 + fast-prefill 的 26/8192-token 实模 smoke 通过，确认自动完整输出策略、raw h20 和有限 top-5 logprobs。
- Mooncake RDMA / NIXL UCX：BF16/Align 的 17-token 严格对照均通过，31 组 KV、greedy 续写和 logprobs 全一致。Fast/FP8 + Mega 的 17/1025-token 两种协议均通过传输字节检查，D 确认启用新策略。17-token 的 P/D 与 D 本地重算仍有原有数值差异，1025-token 一致；不把字节传输通过写成所有 Fast 路径等价。
- 第一次 NIXL + Mega 启动在 UCX 配置表解析阶段失败，尚未执行传输。测试机有系统 HPCX 和 NIXL wheel 两套 UCX；对本测试进程设置统一库别名及 `UCX_MODULE_DIR` 后重跑通过。原共享库和 venv 未修改，握手检查保持。无 GPU 的轻量 import 探针没有复现原错误，未以该探针替代端到端验收。失败与成功日志均归档。
- 最终源码 8003 文件及 runtime support 1996 文件逐一审计通过；GPU6=0 MiB、advisory lock 可重新获取，Pod UID/节点/restart=0 与 PID 7 保持。全部暂存文件的 pre-commit 检查通过。

另外用两个新建的独立进程直接运行原生 benchmark CLI，通过实际配置开关比较，不切换模型内部模式。配置唯一差异是 `kv_sharing_fast_prefill`。共 36 个正式 batch、156 请求、19968 输出 token；18 组配对输出一致，首 token/logprobs 和完整行 teacher-forced 也一致。

| 输入 | C | 关闭 tok/s | 开启 tok/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| short | 1 | 329.19 | 329.01 | -0.06% |
| short | 4 | 890.21 | 884.28 | -0.67% |
| short | 8 | 1481.87 | 1467.98 | -0.94% |
| context8193 | 1 | 160.14 | 175.79 | +9.78% |
| context8193 | 4 | 233.40 | 242.22 | +3.78% |
| context8193 | 8 | 261.53 | 269.07 | +2.88% |

该独立进程面板只有一次 off→on 顺序，不把它当作平衡的计时重复。短输入仍观测到 0.06%–0.94% 的回退；8K 提升 2.88%–9.78%。主结论以同进程交错面板为主，独立进程用于确认实际开关行为和数值。

机器可读回执：[summary.json](summary.json)、[native-validation.json](native-validation.json)、[quality.json](quality.json)、[pd-validation.json](pd-validation.json)、[profile-receipt.json](profile-receipt.json)、[validation.json](validation.json)、[source-receipt.json](source-receipt.json)。

这轮是隔离 benchmark 和源码更新，没有切换线上服务、修改 Gateway 或重建
Job/Pod。原始目录为 `/data/vllm_test/work/yoco-fast-prefill-overhead-20261021/`，
远端为 `/home/fhb/yoco-fast-prefill-overhead-20261021/`，包含复现脚本、源 SHA、
完整 token/metrics、失败的启动诊断、逐阶段 profile 及资源回执。传输依赖沿用
前一轮隔离的 `transport-deps`，未修改共享 venv。

实测限制：TP1/PP1/DP1/EP1 的 B200 固定请求集与采样 seed，不是一般任务质量、
跨机传输带宽或生产服务容量结论。Shared-KV 的现有并行支持范围保持。
