# YOCO V2 / Shared-KV fast-prefill integration

2026-10-21。父提交 `0e57d04dd8c8c9ee105fedf6fd687b9d832774ed`，开发分支
`fhb-dev-aligndeepinfer-startpoint`。使用说明见 [V2_FAST_PREFILL.md](../V2_FAST_PREFILL.md)。

本次将 YOCO 的 P/D 专用执行、DP fast-prefill 协调和 BF16 greedy 实验接入 V2，
并允许 Shared-KV K8 在 fast-prefill 下获得最终 raw hidden。沿用前一提交的融合
gather、Mooncake 分组复用，以及已合入的 FP32 proposal cache、Mega 和 paged FA4。
没有移植 DeepInfer 未公开的 KV 服务，也没有新增传输协议或逐层流水。

## 最终实现

- Shared-KV 的辅助配置限定为最终 raw residual 一条流；当前目标为 h20。返回 tensor
  保持完整行索引，但只有 logits/verification 选中行是最终特征。草稿读取的
  `query_end - num_rejected - 1` 始终在选中集合中。clone、padding、KV 写入依赖与
  DP context 恢复保留。
- prompt logprobs 请求运行完整 prompt 行；普通 EAGLE/DSpark 和任意层 extraction
  沿用原有完整执行要求。没有把未执行的 prompt 行当作最终特征返回。
- V2 使用已有通用 connector。注册时排除 shared aliases，只暴露实际 KV owner。
  `YOCOModelState` 承载角色行为：专用 P 执行 self + shared-KV producer，真实和
  dummy rank 都不执行 cross；P 禁用无用途的图捕获。D 保留普通 FULL decode。
- DP 同步跨 rank 的 cross-block token 数，包括 inactive/dummy rank；本轮没有进行
  多 GPU 模型验收。Shared-KV 原有 TP1/PP1/DP1/no-CP 限制不变。
- NIXL 只对“YOCO + DSpark + Shared-KV marker”排除 draft compatibility factors：
  该 draft 不增加传输 KV，h20 在 D 的尾部重算。目标模型、dtype、backend、协议模式
  和 runtime 几何检查仍启用，没有设置 `enforce_handshake_compat=false`。
- BF16 greedy 的 V2 adapter 复用 BF16 logprob primitive，generated/prompt/rejection
  返回分数采用同一策略。保留 greedy-only、SM90+ 和 custom logprob IDs 限制。
  当前常规 T1 K8 测试关闭这个独立精度实验开关。

## 修复一次多余的 prefill 轮次

原调度器只要开启 YOCO fast-prefill，就按最后一个 KV block 拆尾部，包含关闭 APC、
没有 KV transfer 的 cold standalone。观察一个 26-token 请求时，原路径执行
16 + 10 两轮目标模型；每轮有 30 次 self 层和 10 次 cross 层调用，累计 80 次。

最终将此规则限定为启用 APC 或配置 KV transfer 的引擎。它们继续保留 local-hit /
remote-hit / cold 的尾部形状一致性；无缓存 standalone 一轮计算 26 个 token。
这是引擎配置级的调度规则，没有新增 batch/长度后端阈值或 CUDA 图管理器。

首次直接接通的 72 次正式测量出现短输入吞吐 −11.94% 到 −19.11%、TTFT +50% 到
+97%，因此不能用“开关能打开”作为性能验收。调度修正后另跑 72 次，短输入吞吐
约 −0.72% 到 −1.28%、TTFT +1.03% 到 +2.58%；8K/C4 仍有约 −6.41% 的吞吐回退。
这些多进程离线测试保留在原始目录，但四请求 teacher panel 的基线重复出现批次/
数值变化，最终性能面板另外固定请求入队与调度方式。以上不是提速结论。

## 最终性能面板

最终数值由 [summary.json](summary.json) 和 [summary.csv](summary.csv) 生成。
固定同一个 B200 GPU6、TP1/PP1/DP1/EP1、online FP8 target、BF16 KV、K8 probabilistic
draft + standard rejection、Mega hybrid + FA4，APC 关闭、KV=8 GiB、batch budget=2048、
max seqs=8、FULL_DECODE_ONLY。使用 synchronous in-process engine 与关闭 async
scheduling，使所有请求入队后再执行，避免不同入队批次混入数值/性能比较。

顺序为 dense3-a1 / fast3-b1 / fast3-b2 / dense3-a2。短 chat 和 8193-token 输入，
C1/C4/C8，每场景 2 热身 + 3 正式，每请求 128 输出、T1、固定 seeds、忽略 EOS。
四轮共 72 次正式 batch，另 48 次热身。指标包含 prefill、draft、target、采样、
调度和 GPU drain，排除加载、编译、tokenization 与 HTTP。

| 输入 | C | 全量 tok/s | fast tok/s | 吞吐变化 | 全量 TTFT ms | fast TTFT ms | TTFT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 327.70 | 316.46 | -3.43% | 93.65 | 101.42 | +8.30% |
| short | 4 | 888.83 | 876.92 | -1.34% | 95.46 | 100.18 | +4.94% |
| short | 8 | 1476.60 | 1461.14 | -1.05% | 96.71 | 101.49 | +4.95% |
| context8193 | 1 | 158.43 | 153.03 | -3.41% | 476.41 | 502.90 | +5.56% |
| context8193 | 4 | 232.16 | 212.70 | -8.38% | 1069.23 | 1128.68 | +5.56% |
| context8193 | 8 | 260.42 | 245.70 | -5.65% | 1870.85 | 1977.94 | +5.72% |

最终面板没有显示提速：吞吐观测回退 **1.05%–8.38%**，TTFT 中位数增加 **4.94%–8.30%**。功能按要求启用，但不把这次迁移称为性能收益。保留 `--no-fast-prefill` 作为完整目标模型对照。当前 FULL_DECODE_ONLY 下 prefill 仍主要暴露 CPU/kernel submission 开销；减少 cross 的计算行数没有自动转化为墙钟收益。8K/C4 的 draft 接受比例由 34.55% 降到 30.07%，因此整段耗时也混有续写轨迹/接受率变化。

正式窗口完成 312 个请求、39936 个输出 token，无损坏/抢占/输出预算不足。每个 backend、每个单元只有 6 个正式 batch，结果不作为所有请求集的稳定回退幅度或生产容量结论。

target `[9,18,36,72]` 和 draft `[8,16,24,32,40,64]` 的四/六个 FULL 图均实际回放。
本轮没有添加 workload dispatch 图组；不将该图数推广为所有外部 graph 配置的保证。

四次运行的固定续写 teacher-forced 检查中，4 条输入各 64 个位置，共 256 个
logprob 完全一致；该检查请求完整 prompt logprobs，因此走完整行计算。实际 compact
首 token 面板的 78 个共同 top-k 项，平均绝对 logprob 差为 0.1139801，最大为
0.5639524；全量基线两次重复则完全一致。完整行检查通过不能替代 compact 路径的
分布等价证明。本轮没有进行一般任务正确率评估；详见
[quality-summary.json](quality-summary.json) 与 [原始观察](quality-observations.json)。

## P/D 传输与数值分别验收

使用同一 GPU6 上的两个独立原生引擎：P 无 draft，D 带 Shared-KV K8；显式 BF16 KV，
每引擎 512 MiB KV、max seqs=2、max length=4096。Mooncake 使用 RDMA，NIXL 使用
UCX。这是同机同卡功能验证，没有测跨机带宽、跨机 TTFT 或生产容量。

只检查生成文本不足以定位问题，因此 worker 只读观察每组页表对应的物理 KV bytes，
记录 SHA-256、position、seq_len，再比较本地完整重算。未替换模型、专家或 attention
forward。

BF16/Align 的 17-token 边界对照中，两种 connector 都满足：

- 远端命中 16 个 token，P 没有执行 cross，D 返回 h20 并继续 K8。
- P prefix、D 收到的 prefix、D 本地重算的 prefix：31 组 KV 全部逐字节一致。
- 8-token greedy 续写与返回的 top-5 logprobs 均完全一致。
- NIXL 兼容检查保持开启；P/D 的兼容哈希相同。

Fast/FP8 的 17-token 对照中，Mooncake 传输前后 31 组 KV bytes 也完全一致，
remote/local 的 position 均正确；但 P 的 KV 与 D 本地重算有 29 组不同，greedy
续写也不同。该差异不能算作传输损坏，也不能宣称 Fast/FP8 分布或任务质量等价。

| connector | 场景 | token 长度 | 字节/恢复验收 | greedy 与本地重算 |
| --- | --- | --- | --- | --- |
| MooncakeConnector | BF16/Align | 17 | 31 组 KV 全部一致 | 1/1 一致 |
| NixlPullConnector | BF16/Align | 17 | 31 组 KV 全部一致 | 1/1 一致 |
| MooncakeConnector | Fast/FP8 | 15/16/17/31/32/33/1025 | 31 组 KV 全部一致 | 4/7 一致 |
| NixlPullConnector | Fast/FP8 | 15/16/17/31/32/33/1025 | 31 组 KV 全部一致 | 4/7 一致 |

Fast 的七个长度分别命中 0/0/16/16/16/32/1024 token。两种 connector 在 17/31/32-token 输入上均出现续写差异；其余四个用例相同。上述 PASS 针对字节传输，`KV_BYTES_PASS_NUMERICS_DIFFER` 明确区分了续写差异，未把它隐藏成输出等价通过。

完整逐组记录、角色配置、兼容哈希与图模式见 [pd-validation.json](pd-validation.json)。

初始 harness 的空 metrics、bootstrap URL 缺少 scheme，以及 wheel 不支持 `nvlink`
的配置错误都已单独保留，未计入通过结果。实际传输用显式 `rdma`，没有把它称为
NVLink 测量。NIXL 首次连接正确拒绝了原来包含 draft 的不匹配哈希；上述窄范围
Shared-KV 修正后通过，没有关闭兼容校验。

## 测试与环境边界

- 本地 model/runner/attention/DP 定向回归：178 项通过。
- Scheduler YOCO 冷请求/APC/P-D 边界：12 项通过。
- NIXL compatibility/backend/mode 检查：25 项通过。
- B200 最终 sampler/rejection：25 项通过，无跳过；prefill + Shared-KV：77 项通过。
  包含原始位模式、FP8、stride、图回放更新输入、padding、clone、异常 DP 恢复、
  raw/normalized 区分、被拒绝行元数据，以及 BF16 prompt 分块打分。
- 宽范围 scheduler/Mooncake/NIXL-HMA：330 项通过、40 项受环境阻塞。
  39 项由现有 Pixtral/Llava 导入与本机 Transformers 5.17 的符号不匹配造成；
  1 项为 gated Gemma config HTTP 403。独立导入可重现缺失符号，Pixtral 源码与父提交相同。
- 完整 V2/NIXL 定向集合：185 项通过，另 1 个 Qwen timeout smoke 因本地缺少
  `vllm.third_party.flashmla.flash_mla_interface` 未通过；独立重跑复现。未修改这些
  无关依赖或绕过 gated 访问。

- 真实模型普通 YOCO V2 + fast-prefill，BF16 sampling 开关分别为 0/1：
  1024/8192-token 输入均完成，含 prompt-logprob 检查，两组 smoke 均通过。
- 真实模型 V2 + Shared-KV K8 + fast-prefill + BF16 greedy：26/8192-token 输入
  各生成 16 个 token，top-5 logprobs 均有限；worker 确认 V2、
  `YocoBf16GreedySampler`、辅助层 `[20]` 和 FP32 draft logits。
- 最终全部改动文件的 pre-commit（含 mypy）通过。性能采集之后只补充两处
  Sampler proxy 边界的显式 `typing.cast`，计算与分派规则不变；实测版本和最终
  文件的 SHA 分别保留在 [source-receipt.json](source-receipt.json)，未将旧 hash
  冒称最终源码。实测源码 8003 文件、runtime support 1996 文件的远端审计通过。
- 最终 GPU6 为 0 MiB，advisory lock 可重新获取；Pod UID、节点、restart=0 与
  PID 7 保持。smoke、源码/依赖及资源回执见
  [final-validation.json](final-validation.json)。

所有模型实验使用相同隔离 runtime：Torch 2.13.0+cu130、Triton 3.7.1、已固定的
DeepGEMM 与 FA4 CuteDSL。传输包只安装在本次 `transport-deps/`：NIXL/NIXL-cu13
1.4.1、Mooncake CUDA13 0.3.12；没有修改共享 venv。

原始目录：`/data/vllm_test/work/yoco-v2-fast-prefill-20261021/`；远端：
`/home/fhb/yoco-v2-fast-prefill-20261021/`。包含各次源 manifest、失败诊断、完整输出、
输入 token、原始日志、配置、逐页 SHA、重放计数与退出码。本地归档索引保留结果哈希。

实验仅使用 GPU6 的 advisory lock；未重建 Job/Pod、修改 Gateway 或重启共享服务。
贡献由 Codex 辅助完成，没有创建上游 PR。
