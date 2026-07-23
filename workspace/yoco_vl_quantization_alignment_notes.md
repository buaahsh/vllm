# YOCO-VL 量化对齐记录

记录日期：2026-07-22

## 实验环境

- Docker：`wjh-b200`
- Python：容器系统 `/usr/bin/python`
- 两侧推理都在同一个 Docker 环境内执行
- 当前对齐条件：no-BOS、window `(512, 0)`、batch size 3、LLM/Vision 全 FA2、projector FP32
- llm-train checkpoint：`/mnt/nvme/wjh/updates_3000`
- llm-train repo：`/root/workspace/llm-train`

## 已有结果

| 配置 | Mean KL | Top-1 agreement |
|---|---:|---:|
| 双方 BF16 | 0.0036494795 | 100% |
| llm-train block-128 MXFP8 / vLLM 原生 block-32 MXFP8 | 0.0147967767 | 100% |
| 双方 block-128 FP8/UE8M0（boundary 修复前） | 0.0507695191 | 66.7% |
| 双方 block-128 FP8/UE8M0（boundary 修复后） | 0.0518537872 | 66.7% |
| 双方 block-128 FP8/UE8M0（batch 4，更换第 3 条 prompt） | 0.0100627635 | 100% |
| 双方 block-128 FP8/UE8M0（batch 4，第 3 张换为 dog1） | 0.0083984127 | 100% |
| 双方 block-128 FP8/UE8M0（四条分别 batch 1，算术平均） | 0.0088392242 | 100% |

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch_kl_system_all_fa2_projector_fp32_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch_kl_system_all_fa2_projector_fp32_both_mxfp8_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch_kl_system_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch_kl_system_all_fa2_projector_fp32_mxfp8_block128_window512_boundary_fix_run1.json`
- `/mnt/nvme/wjh/yoco_vl_batch_kl_system_all_fa2_projector_fp32_mxfp8_block128_window512_boundary_fix_run2.json`

## Boundary 修复后的 KL 复测

两轮复测均在 `wjh-b200` Docker 的系统 `/usr/bin/python` 中完成。由于原物理 GPU 1 当时显存不足，复测使用同机型的空闲 B200 GPU 6；每一轮中的 llm-train 和 vLLM 都在同一容器、同一物理 GPU 上顺序执行。

设置保持不变：no-BOS、window `(512, 0)`、batch size 3、LLM/Vision 全 FA2、projector FP32、llm-train MXFP8 block-128、vLLM `fp8_per_block` + DeepGEMM E8M0。

两轮的 `distribution` 和 `logits_matrix` 逐字段完全一致：

```text
KL(ref || vLLM): [0.0171546023, 0.0061002364, 0.1323065162]
Mean KL:          0.0518537872
Mean reverse KL:  0.0603422038
Mean JS:          0.0134571111
Top-1 agreement:  66.7%
Reference top-1:  [32, 785, 785]
vLLM top-1:       [32, 785, 32]
```

Logits matrix：

```text
relative Frobenius: 0.0805117562
global cosine:      0.9969063401
mean absolute diff: 0.2456197739
max absolute diff:  2.1875
```

修复前 Mean KL 为 `0.0507695191`，修复后增加 `0.0010842681`。第 2、3 个样本的 KL 完全未变；只有第 1 个样本从 `0.0139017971` 变为 `0.0171546023`。修复的主要收益是消除了非法显存访问并使两次完整推理结果完全稳定，而不是降低两套实现之间的 KL。

## Batch size 4 复测

在原三个样本后追加：

```text
image: dog1.jpeg
query: What colors are most prominent in this image?
```

其余设置与修复后的 batch size 3 实验完全相同。两轮 batch size 4 的 `distribution`、`logits_matrix` 和 `pairwise_distribution` 均逐字段完全一致：

```text
Reference prompt tokens: [1877, 1877, 2942, 1874]
vLLM logits calls:       [1, 3]
KL(ref || vLLM):         [0.0134724360, 0.0030171585, 0.1821898073, 0.0053164153]
Mean KL:                  0.0509989560
Mean reverse KL:          0.0499495044
Mean JS:                  0.0122469142
Top-1 agreement:          75%
Reference top-1:          [32, 785, 785, 785]
vLLM top-1:               [32, 785, 32, 785]
```

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch4_kl_system_all_fa2_projector_fp32_mxfp8_block128_window512_boundary_fix.json`
- `/mnt/nvme/wjh/yoco_vl_batch4_kl_system_all_fa2_projector_fp32_mxfp8_block128_window512_boundary_fix_run2.json`

新增第 4 条自身 KL 较小，但加入该条后原前三条的 reference 分布也发生变化，尤其第 3 条 KL 从 batch size 3 的 `0.1323065162` 变为 `0.1821898073`。两轮 batch size 4 又完全一致，因此这不是随机波动，而是稳定的 batch-shape/batch-composition 数值依赖。后续应分别捕获 llm-train 和 vLLM 在 batch size 3/4 下原前三条的逐层 hidden state 与 expert IDs，确定该依赖首先出现在 attention、普通 projection 还是 MoE routing。

## Batch size 4 更换第 3 条 prompt

保持四张图片和其他三条 prompt 不变，仅将第 3 条 query 更换为：

```text
What animal is shown, and what is visible in the background?
```

推理配置仍为 no-BOS、window `(512, 0)`、LLM/Vision 全 FA2、projector 双方 FP32、llm-train MXFP8 block-128、vLLM `fp8_per_block` + DeepGEMM E8M0。两次完整推理的 `distribution`、`logits_matrix` 和 `pairwise_distribution` 逐字段完全一致：

```text
Reference prompt tokens: [1877, 1877, 2945, 1874]
vLLM logits calls:       [1, 3]
KL(ref || vLLM):         [0.0134724360, 0.0020945326, 0.0166319609, 0.0080521265]
Mean KL:                  0.0100627635
Mean reverse KL:          0.0103201354
Mean JS:                  0.0025288237
Top-1 agreement:          100%
Reference top-1:          [32, 785, 785, 785]
vLLM top-1:               [32, 785, 785, 785]
```

Logits matrix：

```text
relative Frobenius: 0.0535323434
global cosine:      0.9985665679
mean absolute diff: 0.1750501841
max absolute diff:  1.375
```

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch4_kl_third_prompt_changed_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch4_kl_third_prompt_changed_all_fa2_projector_fp32_mxfp8_block128_window512_run2.json`

更换前 batch size 4 的 Mean KL 为 `0.0509989560`，第 3 条 KL 为 `0.1821898073`；更换后分别下降为 `0.0100627635` 和 `0.0166319609`，并且第 3 条的 top-1 从不一致变为一致。由于 prompt token 数和内容都发生了变化，这说明偏差对具体输入非常敏感，不能把下降解释为框架对齐本身得到改善。

## Batch size 4 将第 3 张图片换为 dog1

在上一组“更换第 3 条 prompt”实验的基础上，只将第 3 张图片从 `dog3.jpeg` 换为 `dog1.jpeg`。因此图片序列为 `dog1/dog2/dog1/dog1`，prompt 均保持不变。四张图片此时各有 `1849` 个 image tokens，no-BOS prompt tokens 为 `[1877, 1877, 1878, 1874]`。

两次完整推理的 `distribution`、`logits_matrix` 和 `pairwise_distribution` 逐字段完全一致：

```text
KL(ref || vLLM): [0.0134724360, 0.0030171585, 0.0117876409, 0.0053164153]
Mean KL:          0.0083984127
Mean reverse KL:  0.0088250386
Mean JS:          0.0021376554
Top-1 agreement:  100%
Reference top-1:  [32, 785, 785, 785]
vLLM top-1:       [32, 785, 785, 785]
```

Logits matrix：

```text
relative Frobenius: 0.0547049604
global cosine:      0.9985045195
mean absolute diff: 0.1798724681
max absolute diff:  1.625
```

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch4_kl_third_image_dog1_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch4_kl_third_image_dog1_all_fa2_projector_fp32_mxfp8_block128_window512_run2.json`

与第 3 张使用 `dog3.jpeg`、prompt 相同的上一组相比，Mean KL 从 `0.0100627635` 降至 `0.0083984127`，第 3 条 KL 从 `0.0166319609` 降至 `0.0117876409`。不过 logits relative Frobenius 从 `0.0535323434` 略升到 `0.0547049604`，说明 KL 的下降不代表所有 logits 距离指标都同步改善。

## 四条分别使用 batch size 1

将 `dog1/dog2/dog1/dog1` 四个 image/prompt pair 拆成四次完全独立的 batch size 1 推理。每次任务内部的 llm-train 和 vLLM 均在同一个 `wjh-b200` Docker、同一张 B200 上顺序执行，配置仍为 no-BOS、window `(512, 0)`、LLM/Vision 全 FA2、projector FP32 和双方 block-128 FP8/UE8M0。

| 样本 | batch 4 KL | batch 1 KL | 变化 |
|---:|---:|---:|---:|
| 1 | 0.0134724360 | 0.0113117378 | -0.0021606982 |
| 2 | 0.0030171585 | 0.0049771359 | +0.0019599774 |
| 3 | 0.0117876409 | 0.0066641737 | -0.0051234672 |
| 4 | 0.0053164153 | 0.0124038495 | +0.0070874342 |
| 算术平均 | 0.0083984127 | 0.0088392242 | +0.0004408115 |

四条 batch size 1 的 reference/vLLM top-1 均分别为 `[32]`、`[785]`、`[785]`、`[785]`，全部一致。四条 logits relative Frobenius 分别为 `[0.0513019860, 0.0640134588, 0.0551412404, 0.0539732464]`。

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch1_sample1_dog1_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample2_dog2_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample3_dog1_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample4_dog1_all_fa2_projector_fp32_mxfp8_block128_window512.json`

batch 1 的平均 KL 比 batch 4 高 `5.25%`，但单条变化方向并不一致：第 1、3 条下降，第 2、4 条上升。这确认了执行 shape/batch composition 对当前 MXFP8+MoE 数值偏差有显著且输入相关的影响；拆成 batch 1 并不会统一改善两侧对齐。

## 已确认对齐的项目

对于 llm-train MXFP8 与 vLLM `fp8_per_block`：

- activation group：`1 x 128`
- weight group：`128 x 128`
- 数据类型：FP8 E4M3FN
- scale：`amax.clamp(1e-4) / 448`
- scale rounding：向上取整到 UE8M0 幂次
- B200 上 vLLM 默认启用 DeepGEMM E8M0
- 随机 BF16 权重测试覆盖 `(128, 3072)`、`(1280, 3072)` 和 `(3072, 1280)`，双方量化 payload 与 scale 逐元素完全一致

量化范围也基本一致：

- 量化：attention Q/K/V/O、YOCO shared K/V、shared expert、routed expert W13/W2
- 不量化：embedding、LM head、`lambda_proj`、`shared_gate`
- router 使用 FP32
- ViT 使用 BF16
- projector 双方使用 FP32
- routed MoE 双方都在 W2 前应用 routing probability，并在量化前落到 BF16

llm-train 的 LM head 没有继承模型的 `quant_mode`，因此即使模型设置为 MXFP8，LM head 仍是 BF16。vLLM 的在线量化只处理 `LinearBase` 和 `RoutedExperts`，`ParallelLMHead` 也不会被该配置量化。

## 当前发现的主要偏差和风险

### 1. llm-train activation MXFP8 kernel 缺少尾块边界保护 (已修复)

文件：`/root/workspace/llm-train/llm/kernel/quant.py`

状态：已于 2026-07-22 修复。输入读取使用 M 维 `boundary_check` 和 zero padding，FP8 payload/scale 写入也增加了 M 维 `boundary_check`。修复后的验证覆盖 `M=1/2/3/15/16/17/31/32/33/63/64/65/110/6696`，每种尺寸重复 5 次，均与 vLLM block-128 reference 逐元素一致；CUDA compute-sanitizer memcheck 报告 0 errors。以下保留原问题的原因分析。

`_act_mxfp8_kernel` 的 M 维 grid 使用：

```python
triton.cdiv(m, BLOCK_SIZE_M)
```

但基于 block pointer 的 `tl.load` 和 `tl.store` 没有设置 M 维 boundary check/mask。候选 `BLOCK_SIZE_M` 是 16、32、64；当 M 不能整除所选 block 时，最后一个 block 会越界访问。

复测中，小 M 场景已经观察到：

- scale 在重复运行间变化
- 量化输出偶发变化
- 相邻 CUDA allocation 中的张量被污染

当前三个 reference prompt 的 no-BOS token 总数为：

```text
1877 + 1877 + 2942 = 6696
```

6696 不能整除 16、32 或 64，因此该路径理论上存在尾块越界。大张量测试中有效区域有时仍能与 vLLM 完全一致，这可能是 CUDA allocator 的额外空间暂时吸收了越界写，不能据此认为 kernel 安全。

在修复这个问题前，MXFP8 reference 的最终 KL 不应被视为完全可靠的纯框架对齐结果。

### 2. 融合后的 GEMM 形状不同

即使量化 payload 完全相同，双方的 DeepGEMM 调用形状仍不同：

- llm-train self-attention：共享一次 activation quant，然后分别执行 Q、K、V 三个 GEMM
- vLLM self-attention：将 QKV 权重融合后执行一个 GEMM
- llm-train shared expert：gate/up 分别执行 GEMM
- vLLM shared expert：gate/up 使用融合 GEMM

量化 block 边界可以一致，但不同 N 维形状可能选择不同 kernel tile、pipeline 和累加顺序，产生 BF16 输出偏差。

### 3. MoE permutation、padding 与 reduction 顺序不同

双方虽然都使用 block-128 DeepGEMM，并且都在 W2 前乘 routing probability，但仍可能存在：

- token 按 expert 排序的顺序不同
- 每个 expert 的 padding/alignment 方式不同
- grouped GEMM 的 psum layout 或 tile 不同
- 8 个 expert 输出的 reduction/unpermute 顺序不同
- router softmax、top-k 和 renormalization reduction 顺序不同

router top-k 边界附近的微小数值变化可能改变 expert ID，随后造成明显的逐层分叉。

### 4. 实际 batch 执行形状不同

llm-train reference 将 3 个样本合并为一个 flat batch 执行。当前 vLLM 结果中，`compute_logits` 被分成 `1 + 2` 两次调用。

这意味着至少 LM head 的 M 维 GEMM 形状不同；scheduler 也可能让 backbone prefill 采用不同的请求分组。即使权重和输入逻辑一致，GEMM/attention kernel 的选择与累加顺序仍可能不同。

### 5. 两边都标记为 FA2，但实际 attention 路径仍不同

- llm-train prefill 使用 contiguous varlen FlashAttention 接口
- vLLM 经过 scheduler、KV cache/paged attention 集成路径

因此“全 FA2”只能对齐 attention 实现代际，不能保证调用方式、KV layout 和 kernel 选择完全相同。双方 BF16 的 Mean KL 仍为 `0.0036494795`，说明量化前已经存在残余执行差异；动态量化和 MoE routing 会继续放大这些差异。

### 6. vLLM 原生 MXFP8 不是 llm-train 的 block-128 格式

vLLM 原生 `mxfp8` 使用 OCP block-32：weight 通常按每个输出行的 32 个 K 元素共享 scale。llm-train 使用 activation `1 x 128`、weight `128 x 128`。

因此两者不能视为同一种量化格式。原生 block-32 的 KL 更小，可能主要来自更细的量化粒度，而不是更严格的格式对齐。

## 建议排查顺序

1. 已完成：给 llm-train `_act_mxfp8_kernel` 增加正确的尾块 boundary mask，并通过越界与重复确定性测试。
2. 重新运行 BF16、block-128 MXFP8 KL，至少重复两次确认 reference 完全稳定。
3. 让两侧采用相同的 batch/request 分组；也可先逐样本单独推理，排除 `3` 对 `1+2` 的 GEMM 形状影响。
4. 捕获逐层 hidden state，定位第一个明显分叉的 layer 和 attention/MoE 子层。
5. 同时捕获每层 router logits、top-k expert IDs 和 routing weights，判断是否发生 expert 路由分叉。
6. 对同一组 BF16 input/weight 单独比较 QKV、shared expert gate/up、routed W13/W2 的量化 payload 与 GEMM 输出。
7. 必要时临时让一侧采用相同的 fused/unfused GEMM 形状，区分量化误差与 kernel 累加顺序误差。

## 2026-07-22 模型结构复核补充

本节检查的是当前分支新增的 ViT 和 projector，相对于
`/root/workspace/llm-train` 推理路径是否还有结构或执行方式未对齐。结论是：
对当前 27 层 checkpoint、单张静态图片（`t=1`）和 TP=1，双方的 ViT/projector
数学拓扑已经基本一致，但仍有以下执行差异、脚本盲点和扩展性问题。

### 已确认对齐

- 图片 resize、patchify、归一化、grid 和最终视觉 token 数一致。
- 当前 checkpoint 的 ViT hidden size、层数及其他主要维度一致。
- patch embedding 的最新逐张量比较逐元素完全一致。
- projector 拓扑一致：
  `LayerNorm(1152) -> flatten 4 个相邻 patch 为 4608 -> Linear(4608, 4608) -> GELU -> Linear(4608, 3072)`。
- projector 当前在双方都使用 FP32 输入和计算。

### 7. ViT 的第一个数值分叉位于 block 0 attention

最新 dog3、两边 vision 都使用 FA2 的逐张量结果为：

```text
patch embedding:             exact
block 0 norm0:               exact
block 0 attention rel RMS:   7.43e-4
full vision encoder rel RMS: 2.22e-2
projected features rel RMS:  1.75e-2
```

因此当前证据不支持图片预处理或 patch embedding 有结构错误；首个可见偏差发生在
第 0 层 attention 内部，并在后续 27 层累积。这里尚未定位到 QKV projection、
2D RoPE、FlashAttention 本体还是输出 projection。下一步应分别捕获并比较：

1. Q/K/V projection 输出；
2. 应用 RoPE 后的 Q/K；
3. FlashAttention 返回且尚未经过 `wo` 的输出；
4. `wo` 输出。

### 8. 多图 batch 的 projector GEMM shape 不一致（已对齐）

- vLLM 在 `vllm/model_executor/models/yoco_vl.py` 的 `_process_image_input` 中先
  `torch.cat(image_features)`，对 batch 内所有图片只调用一次 projector，再按长度 split。
- llm-train 在 `/root/workspace/llm-train/llm/vl_batch_infer.py` 的
  `encode_images` 中逐张图片调用 `projector.forward_flat(features.float())`。

两者数学结果等价，但 projector GEMM 的 M 维 shape 不同，可能选择不同 kernel/tile
和累加顺序，产生确定性的 FP32/BF16 rounding drift。这个差异也可能解释部分
batch=4 与四次 batch=1 的样本相关变化。可通过让 vLLM 也逐图执行 projector，或让
llm-train 拼接后一次执行，做一次严格 A/B 验证。

状态：已于 2026-07-22 将 vLLM 改为逐图调用 projector，并完成 batch=4 和四次
batch=1 A/B。所有 logits/KL 指标与修改前逐字段完全一致，因此该差异不是当前 KL
偏差或 batch=1/batch=4 差异的来源。详细数据见文末“Projector batch shape 对齐复测”。

### 9. BOS 对齐尚未由真实 runtime token 验证

- llm-train 的 prompt 构造明确在最前面加入 `tokenizer.bos_id`。
- 实验观察到 vLLM 的实际 prefill 比 llm-train 少 1 个 token，因此当前 KL 使用的是
  llm-train no-BOS variant。
- `workspace/wjh-b200-h100/compare_yoco_vl_tensors.py` 当前把
  `input_ids_exact` 和 `image_mask_exact` 直接硬编码为 `True`，并未比较 vLLM 实际运行时
  的 token IDs 和 image mask。

所以“no-BOS 是正确对齐方式”目前与运行现象一致，但还不是由捕获真实 runtime IDs
严格证明的结论。需要移除这两个硬编码布尔值，捕获 scheduler/model 实际收到的
input IDs、position IDs 和 image mask，再决定统一保留还是统一删除 BOS。

### 10. dummy image 没有覆盖配置允许的最大视觉 token 数

`YOCOVLDummyInputsBuilder` 固定使用 `1024 x 1024` 图片；在当前 resize/patch/merge
配置下约产生 1369 个视觉 token，而 `vision_max_image_tokens` 允许 4096。

这不会影响当前小图片的 KL，但可能让 vLLM profiling 低估最大图片的多模态计算或
显存需求，在实际输入接近 4096 token 时出现显存预算不足/OOM。dummy 尺寸应根据
`vision_max_image_tokens`、patch size、merge size 和单边 patch limit 反推，且测试其
实际处理结果确实达到允许的最大 token 数。

### 11. vLLM 只实现静态图片的 2D MoonViT 路径（已修复）

vLLM 对所有输入使用 `rope_type="rope_2d"`，当前只接受 image。对单张静态图片
`t=1`，这与 llm-train 的 MoonViT3d 路径在数学上可以等价；对于视频或任何 `t>1`
输入，llm-train 还有 temporal position/pooling，而 vLLM 当前没有对应实现。

这是修改前的状态。现已切换为与 llm-train 同源的 `MoonViT3d` 实现，补齐
3D/video processor、时间位置编码、spatial-temporal attention 和 temporal pooling；
单图仍使用 `t=1` 的同一条 3D 路径。

### 12. vision layer 默认值和转换器 fallback 过于宽松

- `YOCOVLConfig` 在没有 `vision_config` 时默认创建 10 层 MoonViT。
- `workspace/wjh-b200-h100/convert_yoco_vl_to_hf.py` 无法从权重推断视觉层数时也回退到 10。
- 当前转换后的 checkpoint 明确写入 27 层，因此当前实验不受这个默认值影响。

风险是 metadata 缺失或权重命名变化时，转换仍能生成错误的 10 层配置，问题可能到
load/inference 阶段才暴露。建议转换器无法可靠推断时直接报错；配置默认值也应避免
静默假设特定层数。

### 13. 缺少 YOCO-VL 专项测试

当前只有 YOCO text config 相关测试，没有覆盖 YOCO-VL processor/model。至少应增加：

- resize/patchify/grid/token-count 与 llm-train fixture 对齐；
- BOS、image start/end/placeholder token 序列；
- 单图与多图 projector shape/输出；
- 27 层 checkpoint config/weight loading；
- dummy image 实际达到最大视觉 token 数；
- 单张图片和视频的端到端 smoke test，以及明确拒绝超过时间位置编码上限的视频 chunk。

### 结构问题建议处理顺序

1. 捕获并比较真实 runtime token IDs、position IDs 和 image mask，修复比较脚本中的硬编码结果，彻底确认 BOS。
2. 已完成：对齐双方 projector 的 batch 执行 shape，并重新计算 batch=1/batch=4 KL；结果与修改前逐字段完全一致。
3. 将 block 0 attention 拆成 QKV、RoPE、FA2 raw output 和 `wo` 四段捕获，定位 ViT 第一个数值分叉。
4. 修正 dummy image，使 profiling 真正覆盖 `vision_max_image_tokens`。
5. 让视觉层数推断失败时 fail closed，并补充 YOCO-VL processor/model tests。
6. 已完成：实现并验证 3D/temporal 路径。

## Projector batch shape 对齐复测

### 代码改动

vLLM 原实现：

```python
lengths = [x.shape[0] for x in image_features]
return self.vision_projector(torch.cat(image_features)).split(lengths)
```

对齐后与 llm-train 一样逐张图片调用 projector：

```python
return tuple(self.vision_projector(features) for features in image_features)
```

每张图片在 pre-norm 前是 `(7396, 1152)`，flatten 后 projector 第一层 Linear 的
M/K shape 是 `(1849, 4608)`。双方 projector 权重与计算仍为 FP32，其他配置不变：
no-BOS、window `(512, 0)`、LLM/Vision 全 FA2、llm-train MXFP8 block-128、vLLM
`fp8_per_block` + DeepGEMM E8M0。每次 reference/vLLM 对比均在同一个
`wjh-b200` Docker 的系统 Python `/usr/bin/python` 中、同一张 B200 上顺序完成。

### Batch size 4

输入仍为 `dog1/dog2/dog1/dog1`，四条 prompt 与上一轮保持一致。

```text
KL(ref || vLLM): [0.0134724360, 0.0030171585, 0.0117876409, 0.0053164153]
Mean KL:          0.0083984127
Mean reverse KL:  0.0088250386
Mean JS:          0.0021376554
Top-1 agreement:  100%
Reference top-1:  [32, 785, 785, 785]
vLLM top-1:       [32, 785, 785, 785]
```

与 projector 对齐前相比，`logits_matrix`、`distribution` 和
`pairwise_distribution` 三部分逐字段完全一致；不仅 KL 相同，logits 的
relative Frobenius `0.0547049604`、global cosine `0.9985045195`、mean absolute
diff `0.1798724681` 和 max absolute diff `1.625` 也都完全相同。

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch4_projector_per_image_all_fa2_projector_fp32_mxfp8_block128_window512.json`

### 四次独立 batch size 1

| 样本 | 对齐前 KL | 对齐后 KL | 是否逐字段一致 |
|---:|---:|---:|:---:|
| 1 | 0.0113117378 | 0.0113117378 | 是 |
| 2 | 0.0049771359 | 0.0049771359 | 是 |
| 3 | 0.0066641737 | 0.0066641737 | 是 |
| 4 | 0.0124038495 | 0.0124038495 | 是 |
| 算术平均 | 0.0088392242 | 0.0088392242 | 是 |

四条 top-1 agreement 均为 100%，logits matrix 和 distribution 也都与各自修改前
结果逐字段一致。

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch1_sample1_projector_per_image_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample2_projector_per_image_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample3_projector_per_image_all_fa2_projector_fp32_mxfp8_block128_window512.json`
- `/mnt/nvme/wjh/yoco_vl_batch1_sample4_projector_per_image_all_fa2_projector_fp32_mxfp8_block128_window512.json`

### 结论

projector 的源码执行方式现在已经结构对齐，但此次 A/B 没有造成任何数值变化。
这说明当前 workload 下 concat 与逐图 projector 并未引入可观察的舍入差异，不能解释
当前 KL，也不能解释 batch=4 与 batch=1 的差异。后续应优先排查真实 runtime
BOS/token/mask，以及 block 0 attention 中 QKV、RoPE、FA2 raw output 和 `wo` 的首个分叉。

## vLLM 视频结构与 llm-train 对齐

vLLM YOCO-VL 已改用 `MoonViT3dPretrainedModel`，其结构与 llm-train 的
`MoonViT3dVisionTower` 同源。对齐项包括：

- `init_pos_emb_time=4`、`pos_emb_type=divided_fixed`；
- `video_attn_type=spatial_temporal`；
- `merge_type=sd2_tpool`，时间维最终池化为同一组空间 token；
- 图片 grid 统一为 `[1, H, W]`，视频 grid 为 `[T, H, W]`；
- 图片和视频复用同一套 begin/pad/end token、ViT 和 FP32 projector；
- 每个视频 chunk 限制为 1–4 帧，长视频需要先采样或切 chunk；
- projector 继续逐媒体项执行，以保持与 llm-train 相同的 GEMM M shape。

vLLM 分片加载不会自动迁移 `persistent=False` 的时间位置编码 buffer。已在
`Learnable2DInterpPosEmbDivided_fixed.forward` 中按 llm-train 实际运行方式，将
`time_weight` 对齐到 2D 位置编码的 device 和 BF16 dtype 后再相加。

验证均在 `wjh-b200` 容器系统 `/usr/bin/python` 中完成：

- 原单图 batch=1 MXFP8/FA2 回归的 KL 仍为 `0.011311737820506096`，所有已记录
  logits 指标与修改前逐字段一致；
- 两帧 `dog1` 视频输入为 `(2, 3, 112, 112)`，预处理得到 128 个 patch、
  `grid_thw=[2,8,8]`，时间池化后得到 16 个视觉 token；
- 视频端到端生成成功，输出开头为
  `The video shows a golden retriever sitting`；
- 新增的 TCHW/THWC、逐帧 patch 顺序、4 帧上限和时间位置 buffer dtype
  测试共 4 项全部通过。

结果文件：

- `/mnt/nvme/wjh/yoco_vl_batch1_video_tower_image_regression.json`
- `/mnt/nvme/wjh/yoco_vl_video_2frame_smoke.json`
