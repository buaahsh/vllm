# Shared / latent M1 FP8 分派接入与整模型验证

在 `investigate/yoco-v029-fp8-decode-20260916` 接入已验证的 direct GEMV。
**本轮整模型吞吐基本持平，B1 的样本 NLL 略有上升。** 不将原型的 21.3% 算子
耗时改善表述为整模型提速；原始数据与分派检查见
[结构化结果](SHARED_LATENT_M1_INTEGRATION.json)。

## 默认行为与回退

本分支 `VLLM_YOCO_FP8_SMALL_M` 默认 **1**。YOCO Fast 构造 shared gate/up、shared down、
latent in/out 时，在 SM100、TP1/DP1/PP1、无 EP/EPLB、非 batch-invariant、
BF16 输入/输出、online block-128 FP8 且原后端为 native DeepGEMM UE8M0 时接入。
四个矩阵仍为 `3072→2560`、`1280→3072`、`3072→1024`、`1024→3072`。

实际 GEMM 输入 **M=1** 才执行 direct `N1 / warps4 / stages1`；其他 M 仍调用
原生 DeepGEMM。分派在 opaque custom op 内按实际 tensor shape 判断，避免将
动态 prefill 编译强制限定为 M1。M 指引擎实际执行行数，包括 Graph padding。

初始化复用原量化器；权重转换、packed scales、bias/reshape/output dtype 处理
继承原线性层。Shared down 继续消费现有融合 SwiGLU/量化输出，latent Norm
边界保留。没有创建反量化的整份 BF16/FP32 权重缓存。

回退到原配置：

```bash
VLLM_YOCO_FP8_SMALL_M=0 .venv/bin/python \
  tools/yoco_alignment/benchmark_fast_fp8_decode.py \
  --model /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --output /path/to/new-baseline.json
```

开关在构造 engine / CUDA Graph 前设置，参与编译缓存标识；修改运行中环境变量
不会重建已有 Graph。BF16、Align、其他后端及未覆盖的并行配置保留原实现。

## 验证方法

96 项回归通过，覆盖原有 FP8 latent/shared 运算，以及 M1 实际分派、M2/M8/M32
与原生输出逐位相同、量化器对象复用、Graph 输入更新、开关与 Align/TP/DP/EP
保护、编译身份变化和原有 latent Norm 融合检查。

整模型性能采用同一张 B200、独立 engine A→B→B→A，A 为开关 0，B 为开关 1。
B1/B2/B8 每次输入 512 token、生成 128 token，5 次预热、15 次计时，每配置每档
共 30 个样本。按合并耗时中位数计算总输出吞吐；从整批入队后恢复调度到输出取回。
实际缓存命中 496/512。另采集 B1/B8 各 12 个纯 generation Graph，用真实 kernel
数确认分派，不只检查环境变量或后端名称。

质量检查使用 vLLM V2 原生 `trace_decode_token_ids`。Sampler 在原始 logits
计算完成后指定下一 token，再从原始分布计算该 token 的 raw logprob。
两侧保持相同真实 token 历史，完整执行 autoregressive decode；没有通过遮罩后的
概率计算 NLL。Greedy→replay anchor 核对相同 token 序列的 raw logprob。

B1 为 8 条顺序请求，输入长度 128/256/…/1024，各生成 128 个固定目标，共 1,024
位置；每条请求前清空 prefix cache。B8 为一次同时入队的 8 条请求，每条 512→128，
共 1,024 位置。固定 token 来源与单卡 benchmark 一致。
另做重复公开文本的 4096→16 功能检查；此项不作为长上下文质量评估。

## 同卡 ABBA 结果

| Batch | 原生 tok/s | M1 分派 tok/s | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | 171.22 | 171.66 | +0.25% |
| 2 | 276.41 | 276.75 | +0.12% |
| 8 | 823.83 | 822.28 | −0.19% |

每档每侧 30 个计时样本。差异很小，相对于轮间波动应按基本持平解读。
B1 原型收益没有等比例转化为整个模型的收益；shared/routed 重叠和其余执行链
仍影响单步延迟。此次没有证明显著端到端加速。

两侧均核对到 80 个物理 shared/latent 投影。候选 B1 的每个模型 Graph 实测
**160 次 direct GEMV、81 次其余 dense FP8 GEMM**；原生侧为 0/241。
B8 两侧均为 **0 次 direct、241 次 native dense**。每份 trace 检查 12 次真实
Graph replay、每步 40 次 attention 和 80 次 routed GEMM，回退确实生效。
Graph span 是主模型前向的 elapsed time，LM head/采样在图外，不等同于请求延迟。
B1 最后一轮 greedy 输出在两侧存在差异；B2/B8 对应记录一致。

## 固定 token 的数值结果

| 检查 | 原生 | M1 分派 | 差异 |
| --- | ---: | ---: | ---: |
| B1，1,024 个位置平均 NLL | 1.33018474 | 1.33447083 | +0.00428609 |
| B1，样本 exp(NLL) 相对变化 | — | — | +0.4295% |
| B8，1,024 个位置平均 NLL | 1.46589273 | 1.46589273 | 0 |

B1 有 1,014 个位置的 raw logprob 发生变化，mean/max abs difference 为
0.070594 / 1.032817 nats。小量矩阵输出舍入差异会沿多层和 decode 历史传播；
不能从原型的 16 个不同输出元素推断只有少数 token 受影响。
B8 的所有 raw logprob 逐项相同。

Greedy/replay anchor、全部强制目标 token、有限值以及 4096→16 功能检查均通过。
这些结果覆盖固定公开文本样本，没有建立任务级准确率或长上下文质量等价。
本分支提供回退开关以便继续做任务级比较。

## 代码与原始证据

- [分派与 native fallback](../../../vllm/model_executor/layers/yoco_ops/small_fp8_linear.py)。
- [单卡 benchmark](../../../tools/yoco_alignment/benchmark_fast_fp8_decode.py)：记录实际
  80 个投影的 backend，并保留显式开关用于 A/B。
- [真实 decode 固定 token 检查](../../../tools/yoco_alignment/check_m1_fp8_decode.py)。
- [结果与 Graph 审计脚本](../../../tools/yoco_alignment/analyze_m1_fp8_integration.py)。
- [原型阶段的单算子数据](SHARED_LATENT_SMALL_FP8.md)。

工作区原始目录为 `work/yoco-m1-integration-20260917/`。正式性能记录为
`m1-final-a1/b1/b2/a2`，质量记录为 `m1-quality-native/candidate`，回归记录为
`m1-integration-tests-r1`。首轮隔离目录中的入口符号链接导入了旧统计器，缺少新增
backend 字段；已修正并补跑正式 ABBA，首轮结果不用于最终性能结论。

镜像原生 DeepGEMM `_C` SHA256 保持
`73824dc1312e98cf277ae6bc017becf7e963d5b0bfab81e0c6d475d9b769c9e7`。
性能与质量结果按各自测量范围解释；算子误差小不代表任务级质量完全一致。

保留的调试服务已使用测试后的新版源码、按原 BF16 服务参数恢复；18888 健康、
8-token 真实生成与空队列检查通过。Job/Pod UID、holder、节点及容器 restart count
保持不变。上述 FP8 性能来自独立 benchmark，不将原 BF16 服务冒充 FP8 测速入口。
