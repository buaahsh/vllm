# Align 小批次优化与 1P1D 测试进度

> 2026-09-07 更新：四卡Job及1P1D测试已完成，见[后续实测报告](../align-4gpu-1p1d-20260907/REPORT.md)。以下保留2026-09-06资源申请前的历史记录。

日期：2026-09-06 当地时间（运行日志 UTC 2026-09-07）。

## 已完成与尚未完成

已补齐 B200 Align MoE profile 的 **2/4/16 行**配置，覆盖当前全部 decode graph capture sizes。
原13项配置完全不变，实验仍默认关闭。真实 vLLM 专家入口在分散路由下速度提升
**6.4% / 7.8% / 12.0%**；llm-train 共用同一表，前向算子计时接近。
36组算子初筛、18组两端专家入口检查、**288项整模型输出字节比较全部通过**。
两轮公开 trace smoke 都50/50成功，0错误，token计数相同并完全排空。

**1P1D 尚未实测，没有双GPU系统吞吐、传输开销或每GPU效率结论。**
目标 Pod `yoco-align-prob-b200-20260901-master-0` 实际分配4张B200：GPU2/3/4/5。
GPU2/3/4分别保留8600/8610/8620旧服务；只有GPU5空闲。
NVML能看到的GPU0/1/6/7不属于本Pod分配。原服务未停止、未重启。

## 实现与下一处瓶颈

固定K=32、split-K=1，保留weighted SwiGLU、BF16落地舍入和Top-8顺序累加。
只增加三个精确行数的launch配置，不改kernel算术、训练adapter或backward。
vLLM和llm-train通过同一个`yoco_align_moe.py`读取相同profile，避免两份调参逻辑分叉。

第一次公开trace smoke观测到22种eager MoE行数，640次调用中仅30次命中原profile（4.69%）。
按`行数×调用数`加权，命中约0.0011%；这个权重不是耗时占比。
大prefill实际出现3136/10594/32545/9097/14119/21142等不规则行数，仍走默认配置。
Python计数不包含CUDA Graph replay，不能说全部decode都没有命中。

因此本轮先补稳定的decode图尺寸2/4/16。更大的prefill收益需要在实际1P1D的P端采样后，
针对不规则M验证配置或设计经过验证的分段策略。运行时仍不做最近邻外推。
YOCO dedicated producer已会跳过不需要的cross layers，这属于原有实现，不记为本轮新增优化。

## 算子性能

同一B200 GPU5、L3 BF16形状、分散/集中两种路由，单层完整专家前向，单位μs。
下表来自真实vLLM/训练入口的独立输入验证，使用ABBA顺序的CUDA Graph计时；不是训练整步或系统吞吐。

| 行数 | 路由 | vLLM旧 → 新 μs | 速度比 | train前向旧 → 新 μs | 速度比 |
| --- | --- | --- | --- | --- | --- |
| 2 | 分散 | 120.37 → 113.11 | 1.064× | 120.42 → 113.38 | 1.062× |
| 2 | 集中到8个专家 | 113.07 → 91.36 | 1.238× | 112.88 → 91.20 | 1.238× |
| 4 | 分散 | 174.34 → 161.79 | 1.078× | 174.35 → 161.92 | 1.077× |
| 4 | 集中到8个专家 | 152.30 → 116.95 | 1.302× | 152.69 → 116.74 | 1.308× |
| 16 | 分散 | 403.33 → 360.15 | 1.120× | 403.05 → 359.67 | 1.121× |
| 16 | 集中到8个专家 | 151.34 → 86.20 | 1.756× | 151.25 → 86.48 | 1.749× |

18组入口验证覆盖1/2/3/4/5/8/15/16/17行、分散/集中路由、带梯度前向和单token拆分比较。
没有新增backward或microbatch梯度bitwise要求。此前512/2048-token训练整步没有加速的结论保持不变；
本轮新增的小尺寸算子收益不能代替训练整步测量。

## 整模型字节验证

真实L3权重，BF16、Router FP32、FA2、TP1、16-token pages、single split，
开启prefix cache、chunked prefill、KV-sharing fast-prefill和原有final-block split。

- 原13行profile：缓存/边界51项，实际B2/B4/B16批次66项。
- 新16行profile：缓存/边界及旧profile对照105项，实际B2/B4/B16批次66项。
- 合计288项，hidden、154880维logits、完整log-prob的全部字节相同，最大差0。

覆盖33/129/512/2048-token提示以及15/16/17/31/32的页边界，cold/hit各生成8token；
实际批次检查确认出现对应B的连续解码步。与此前训练验证的衔接限于匹配的首个输出位置，
不能把本轮说成重新跑完训练全模型矩阵，更不能将standalone结论当作PD已验证。

首次分析误把被scheduler丢弃的中间prefill输出当作首个生成位置，cold因此多出一行。
保留原capture及首次失败报告，修正时逐项核对API实际返回的greedy log-prob字节后再映射输出行，
在`standalone-reference-reanalysis`生成新报告；没有更改模型运算或覆盖失败证据。

## 公开 trace smoke

Mooncake FAST'25 `toolagent_trace.jsonl`，源SHA256
`48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71`。
复用既有smoke：context≤81920、output≤128、前50条，保留原ISL/OSL/hash_ids，15秒到达窗口。
smoke SHA256 `bdcbfc5600e3a64cb7de7745d9fee42a8c1b1fa2bec30e037d310cd4560e40c1`。
AIPerf0.12.0、fixed schedule、speedup1.0、独立cache_salt、安全并发上限512、timeout600。
同GPU5、同模型和服务参数：maxlen81920、budget32768、maxseq256、memory.85。
每轮实际输入307502、输出1077 tokens（原trace输入307452，服务端默认BOS每请求+1）；逐请求计数相同。

**以下是两次就绪检查的原始数据，不是受控的端到端性能A/B。**
旧profile的smoke先于额外数值预热，新profile的smoke在数值预热之后，并复用了同一编译cache。
两轮预热状态不同、仅15秒且共享节点；不把吞吐或延迟差值归因于三个新增配置。
原始trace仅有时间、长度与hash关系，不包含真实prompt文本或在线agent状态。

| 指标 | 原13行profile smoke | 新16行profile smoke |
| --- | --- | --- |
| 成功/计划 | 50/50 | 50/50 |
| 输出 tok/s | 66.195 | 69.439 |
| 输入 tok/s | 18899.774 | 19825.900 |
| TTFT P95 ms | 2714.870 | 1552.492 |
| ITL P95 ms | 34.477 | 22.996 |
| E2E P95 ms | 4541.886 | 2448.132 |
| 发送lag P99 ms | 1.976 | 4.144 |
| 最大in-flight | 17.000 | 15.000 |
| 输出长度不匹配 | 0.000 | 0.000 |

两轮客户端门槛通过，server scrape无错误/重置、最大间隔1.16秒，最终running/waiting为0。
未声明latency SLO，结果不构成容量资格；standalone审计中的transfer/proxy检查不适用。
完整P50/P95/P99、时间片、请求记录和原始指标留在cases目录。

## 1P1D 的已备方案

控制器只接受Pod实际分配且空闲的两张不同B200，记录PID/start ticks/PGID，
仅管理本实验创建的P/D/proxy。独立端口8674/8675/8676、bootstrap8997；
P budget32768、D budget8192，其他参数一致，Mooncake RDMA、16 sender workers。
保留Pod原有CUDA13.1兼容库路径，不能直接套用旧PD脚本覆盖LD_LIBRARY_PATH。

`run_pd_suite.py --gpus P_GPU D_GPU`准备按旧Align（profile关闭）和新Align运行：
两端数值检查→smoke50→冻结600秒公开trace→完全排空，采集P/D/proxy与传输计数/字节/时延/队列。
600秒trace SHA256 `680e526d49545258c0ca5b635d0004a44cce2ce990d533ac32024cefee1f0170`，3643请求。
该双GPU流程尚未运行；需先落实第二张卡，再验证启动/传输和长测。
效率需同时报告系统吞吐和系统吞吐÷2，不能把多用一张GPU的总吞吐增长当作等成本效率。

## 版本、文件与清理

新profile SHA256 `7b5f496599475ce9f044af9557cdc482b6d6abe54e7fd67f268a3fe7982672dc`。
两端用`VLLM_YOCO_ALIGN_MOE_CONFIG`指向同一文件，仍按GPU/Torch/Triton/CUDA元数据和精确行数检查。
环境：B200、Torch2.11.0a0+eb65b36914.nv26.02、CUDA13.1、Triton3.8.0。

本地结果：`/home/lidong1/vllm_test/yoco_results/align-1p1d-b200-20260906`；
远端：`/data/yoco-align-1p1d-b200-20260906`，大型原始.pt保留远端。
`RESULTS.json`、`FINAL_AUDIT.json`和各原始报告可复核。
两次启动失败（UUID解析、覆盖兼容库导致unsupported PTX）及首次错误行映射均保留。

临时服务全部停止，GPU5为0MiB；3660个vLLM运行文件和117个训练源文件校验通过，
7个受保护进程身份不变，Pod UID/restartCount不变；六项不可纠正ECC均0，内核日志无GPU错误。
代码和文档仅本地修改，未commit/push。1P1D测试仍待第二张可用GPU。
