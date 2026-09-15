# fhb-dev-9-18：Fast-FP8 精度分级与实验边界

首次核查：2026-09-13 PDT；2026-09-14 增补 DeepSeek V4 官方策略对照。vLLM 与 llm-train 均已切换到 `fhb-dev-9-18`。vLLM HEAD 为 `511fbed3f7`，还包含未提交的 Fast-FP8 修改；训练 HEAD 为 `0b6978ee`。对象是 YOCO 30A3B-180M-L3 step28000、B200、现有 Fast block-128 FP8 路径。

**结论：优先验证 latent 投影的 W8A8 FP8，以及供 GEMM 使用的融合 FP8 激活；简化 scale 继续作为独立候选。共享全局 NoPE KV 优先列入缓存 FP8 专项。DeepSeek V4 与 llm-train 都保留高精度 LM head，因此将其从上一轮按带宽潜力列出的 P1 调整为 P2 激进实验。Router、门控和统计不列入第一轮 FP8 范围。**

本文的“可以”指适合进入独立实验，不表示已经证明可以默认启用。“本轮保留”是基于当前证据和收益预期的工程选择，不是宣称数学上绝对不能降低精度。现有训练配方提供数值参考，不应自动成为 Fast 推理的精度下限。

## 当前实际精度

| 部分 | 当前 Fast-FP8 |
| --- | --- |
| Q/K/V、attention O、模型级共享 K/V 投影 | FP8 权重与 FP8 激活，输出 BF16 |
| routed W13/W2 | FP8×FP8；小 M Triton / 其余 DeepGEMM；FP32 累加及缩放，输出 BF16 |
| shared expert gate/up/down | FP8×FP8；shared SwiGLU 保留 BF16 舍入边界 |
| latent in/out | BF16；20 个物理 MoE 层各 2 个投影；decode 中因 universal loop 共执行 80 次 |
| lambda / attention gate | BF16；当前 FP8 模式单独保留 |
| shared scalar gate | BF16 操作数；已有融合计算，参数量很小 |
| embedding / LM head | BF16；Fast LM head 先舍入 BF16，再写 FP32 logits |
| Attention Q/K/V、KV cache、attention 输出 | BF16；Q/K/V 投影已经量化不代表 KV cache 是 FP8 |
| Router | FP32 权重、输入和 logits；Fast 已允许 TF32 乘法；Top-8/softmax 保留 FP32 |
| 主残差 | FP32；Fast 在下一次 RMSNorm 中融合残差加法 |
| RMSNorm / RMSClip / RoPE / 差分注意力合并 | 统计或关键中间运算保留 FP32，边界常为 BF16 |
| weight scale | 128×128 block，加载时生成 |
| activation scale | 每 token 每 128 个元素动态生成；DeepGEMM 路径使用 UE8M0 幂次 scale |

FP8 GEMM 直接使用 FP8 操作数，不是先还原整份 BF16 权重后乘法。FP32 累加和 BF16 输出不能被视为“FP8 没生效”。也不能用保留 BF16 的模块数量推断尚未量化的权重比例：最大的 routed experts 已经是 FP8。

## 2026-09-14：DeepSeek V4 官方策略对照

本次以官方 `deepseek-ai/DeepSeek-V4-Flash` 为主，固定 Hugging Face revision `60d8d70770c6776ff598c94bb586a859a38244f1`；对照官方 `inference/model.py`、`inference/kernel.py`、配置和 checkpoint index，再用冻结的 vLLM main `67111973ee` 确认优化实现。官方参考代码会为了实现方便展开部分权重，必须区分 checkpoint 格式、参考计算类型和高性能 kernel。

| 部位 | DeepSeek V4 的证据 | 我们的判断 |
| --- | --- | --- |
| 大型 Q/KV/O 和低秩投影 | `wq_a/wq_b/wkv/wo_a/wo_b` checkpoint 有 FP8 scale；优化 vLLM 使用 FP8 GEMM/FP8 einsum | 现有 QKV/O 已覆盖；YOCO `fc1_latent_proj/fc2_latent_proj` 是新增 FP8 的首选类比候选。DeepSeek 的 attention 低秩投影与 YOCO 的 MoE latent 并非同一个算子，仍需验证 |
| shared expert | 官方共享专家沿用 FP8 Linear；routed experts 则按 checkpoint 选择 FP8/FP4 | 我们 shared/routed 的 W8A8 已生效，无需重复列作新增 FP8 |
| GEMM 输入激活 | 官方 `linear()` 在 FP8/FP4 GEMM 前做动态 block128 `act_quant`；llm-train 同样量化输入 | 可以在 producer 中融合生成供 GEMM 使用的 FP8；FP32 norm 统计保留，若要复现现有数值则在寄存器内保留必要 BF16 舍入；其他 BF16 消费者需要自己的输出 |
| NoPE KV | 官方对 main KV 的 448 个 NoPE 维做 FP8 模拟量化，保留 64 个 RoPE 维 BF16；优化 vLLM 存成相应混合缓存 | 我们 cross-attention 的全局共享 K/V 是 NoPE，适合单独验证 FP8 缓存。先隔离此缓存，后续再评估 self/SWA cache |
| LM head | 官方注明 checkpoint 为 BF16；参考 demo 用 FP32 计算方便生成 logits；优化 vLLM 保持未量化 head。llm-train `self.output` 也走 BF16 默认 | 有性能潜力，但没有“两套参考都用 FP8”的依据；列为 P2 激进实验，不默认纳入新增 FP8 |
| Router | 官方 `Gate.forward` 用 `x.float()` / `weight.float()` 计算，再做评分与 Top-K；优化 vLLM 可用 BF16 输入、FP32 输出路径 | 不推荐 Router 直接 FP8；已有离线数据支持先研究 16-bit 操作数和 FP32 输出 |
| 小型控制投影 | DeepSeek indexer `weights_proj` 保留 BF16；compressor 的投影与聚合保留较高精度 | YOCO lambda/shared scalar gate 没有对应的 FP8 验证先例，不能因其名称含 linear 就一律 FP8 |
| RMSNorm、SwiGLU、softmax、归约 | 官方统计、非线性和 scale 修正累加保留 FP32 | 保留计算精度，优先优化输出位宽、融合与访存 |

需要特别说明：

- 官方 `wo_a` demo 显式构造 BF16 并用 einsum，但紧邻注释明确说明 checkpoint 是 FP8，只是参考实现为简单起见用 BF16。checkpoint index 的 `layers.0.attn.wo_a.scale` 和 vLLM 的 FP8 einsum 都支持这一点，不能将 demo 的存储方式误读为精度要求。
- 官方 KV 参考代码标注 FP8 模拟量化用于匹配 QAT，demo 中缓存张量仍是 BF16；优化后端才落实紧凑存储。我们不能由此推断现有 YOCO checkpoint 未经 KV QAT 也能保持同等质量，尤其一份共享 KV 被多个 cross 层复用。
- 我们 llm-train 的 latent、lambda、shared scalar gate、LM head 没有传入量化模式，因而走 BF16 默认。这个代码事实不证明它们不能量化。latent 的几何形状、归一化边界和 DeepSeek 低秩 FP8 先例提供了最明确的新增候选依据。
- llm-train 即使选择 `mxfp4`，`attention_quant_mode` 仍显式返回 `mxfp8`。这与 DeepSeek 的“专家可更低位宽、大型 attention 投影保持 FP8、控制与统计保留高精度”方向一致。
- DeepSeek V4 与 llm-train 都仍使用动态 block scale。我们的静态 scale 初筛是独立优化证据，不能宣称是两者已经采用的策略。

本轮新增 FP8 候选清单：`model.layers.*.mlp.fc1_latent_proj`、`model.layers.*.mlp.fc2_latent_proj`；以及供既有 FP8 GEMM 使用的融合量化输出。NoPE 全局共享 KV 作为独立后端/缓存实验。`lm_head`、lambda/门控、Router 均需要单独的精度配置与质量结果。

来源：[官方 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)、[官方 kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py)、[官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)。下载源码与 hash 位于工作区 `yoco_results/dsv4-fp8-policy-20260914/`。

## 已经完成、不要重复计算的优化

- Fast 已在原始 logits 上选 Top-8，再做 FP32 的 8 路 softmax；Align 保留其数值路径。
- Routed FP8 已从 FP32 clamped/weighted SwiGLU 结果直接量化 FP8，取消额外 BF16 中间舍入。shared expert 尚未采用这个新舍入策略。
- 已有 shared 激活与量化融合、小 M scale 缓存、布局打包、slot mapping 融合和部分 BF16 专用 kernel。
- 已有验证报告记录 403 项整体回归、最终定向检查、固定前缀和 1,024 个文本位置的 NLL。这些结果只支持对应归档版本和覆盖范围，不能作为以下所有新策略的证明。

## 第一轮优先实验

| 优先级 / 部分 | 候选 | 已有依据 | 尚需确认 |
| --- | --- | --- | --- |
| P0/P1：融合 FP8 激活输出 | 将现有 BF16→FP8 输入量化融合进 norm/其他 producer；有需要时保留 BF16 副输出 | 下游 GEMM 本来就消费 FP8；DeepSeek 和 llm-train 都有量化输入边界 | producer 的舍入、amax、scale 布局和所有消费者；精确重现或明确记录新量化策略 |
| P0：归一化后激活 scale | 动态 block-128 → 每层静态幂次 scale；以 scale=1 作对照 | 同 checkpoint 的历史归一化激活重建误差几乎相同 | 当前 Fast-FP8、prefill/decode、长上下文与任务分布的范围；完整质量；是否省掉归约/调用 |
| P1：latent in/out | BF16 → W8A8 FP8；配合 norm/量化融合 | 3072↔1024 形状适配 block128；每 decode 80 次；BF16 权重 payload 240→120 MiB | 两个方向分别的误差、层间放大、小 M 的量化开销；与新 BF16 kernel 同条件比较 |
| P1/P2：全局共享 NoPE KV | BF16→FP8 缓存；先保持 self/SWA cache 的现有策略 | DeepSeek main KV 的 NoPE FP8 先例；YOCO cross K/V 无 RoPE | 现有后端支持、缓存量化质量、跨层复用、prefix/P-D、长上下文；不能从 DeepSeek QAT 直接外推 |
| P1/P2：固定权重 scale | 每 block → 每矩阵或专家固定 scale | 实际抽样中固定 1/256 的重建误差几乎等于 block scale | 全权重范围扫描；适合的 kernel；加载期少算 amax 本身不带来每步收益 |

scale 的简化不降低 FP8 位宽；它减少范围估计、metadata 或中间计算。不能只把旧 scale tensor 填成 1 而保留原量化权重。量化值和 scale 必须配套更新，且现有 block-scaled kernel 未必自动省掉对应工作。

### 缩放初筛的实际证据

CPU 上从 6 层、42 个真实权重张量抽取 792 个 128×128 block，共 12,976,128 个权重值：

| 方案 | 权重相对 L2 重建误差 | 非零权重变为零 |
| --- | ---: | ---: |
| 当前 block scale | 2.704123% | 0.001233% |
| scale=1 | 2.721896% | 1.149503% |
| 固定 1/256 | 2.704123% | 0.004686% |

历史 BF16 Align 的 input-norm 激活：block scale 为 2.660055%，scale=1 为 2.660584%，固定 1/16 为 2.660055%。幂次 scale 对正常浮点数主要移动指数，不增加尾数位；收益主要在防止溢出/下溢。

这是量化重建初筛，不能替代完整模型 NLL 或任务质量。1/256 是样本上的权重候选，不是通用常数；它对部分历史激活已经产生截断。

完整证据：工作区 `yoco_results/fp8-scale-audit-20260913/`。

## 需要专项验证的候选

| 部分 | 可以尝试什么 | 本轮边界 |
| --- | --- | --- |
| P2：LM head | BF16 → FP8 权重/激活；保留高精度累加和概率计算 | 带宽潜力仍大，payload 907.5→453.75 MiB；DeepSeek 与 llm-train 都未默认采用 FP8 head，必须以独立质量实验支持 |
| Router 投影 | FP16 或 BF16 操作数，FP32 累加/输出；Top-K/softmax 保持 FP32 | 新离线证据支持 16-bit 候选；FP8 明显改变更多专家集合，暂不列第一轮 |
| lambda 投影 | BF16→FP8，并评估与 Q/QKV 的真正融合 | 会影响差分注意力的门控与相减；输出 64 维涉及 block 对齐；只换 dtype 未必提速 |
| KV cache | BF16→FP8 存储，先隔离缓存量化的影响 | 当前 B200+FA4 路径不能直接靠参数启用；需要后端、cache scale、共享 KV、prefix/P-D 等一致适配 |
| routed experts | 独立的 FP4 权重配置；明确 W4A8 / W4A16 / W4A4 与格式 | 已有 30.2B routed 参数；FP8 原始 payload 28.125 GiB，4-bit payload 14.0625 GiB，均未计 scale/padding。需校准、clamp/路由权重兼容 kernel 与任务质量验证 |
| routed W2 activation scale | 动态→静态，按加权 SwiGLU 的真实分布选择 | 归一化激活数据不能代表它；limit=10 可给约 100 的幅度上界，但上界不能证明小值不受损 |
| shared SwiGLU 舍入边界 | 比较 FP32 结果直接量化 FP8 与现有 BF16 舍入后量化 | 可以作为新数值策略实验，不因训练原配方而永久排除；当前已经融合，减少舍入不保证再少一次 launch |
| BF16 logits 存储/融合采样 | 对已舍入 BF16 的 logits 延迟 FP32 展开，或在适用情况下融合 argmax | 保持后续 penalty、温度、log-prob 的计算语义；节省存储不等于降低统计精度；相对 LM head 权重读取，单纯少写 logits 的收益可能很小 |

shared scalar gate、embedding 存储以及 FP32 承载的幂次 scale 可作后续小项，但当前缺少优先于上表的收益证据。

### Router 新增离线敏感性实验

读取真实 checkpoint 的 20 组 FP32 router 权重，在历史 BF16 Align 的 128 个 decode token、40 个逻辑 MoE block 上，共比较 **5,120 次路由输入**。参考为 CPU FP32 linear；稳定排序得到 Top-8，再对选中 logits 做 FP32 softmax。

| 操作数策略，参考累加仍为 FP32 | Top-8 专家集合改变次数 | 改变比例 | logits 相对 L2 |
| --- | ---: | ---: | ---: |
| FP16 输入/权重 | 2 / 5120 | **0.0391%** | 0.00992% |
| BF16 输入/权重 | 21 / 5120 | **0.4102%** | 0.08003% |
| FP8 权重、BF16 输入 | 426 / 5120 | 8.3203% | 1.27771% |
| FP8 权重和激活、block128 scale | 587 / 5120 | **11.4648%** | 1.81314% |

FP16 在这些数值范围内的尾数精度高于 BF16，因此值得列为 Router 首个 16-bit 候选。FP16 的范围更窄，仍需检查当前真实输入范围和输入转换成本。实际 kernel 必须保留 FP32 logits；默认返回 FP16/BF16 的 GEMM 不等同于本实验。

这些百分比是“某次路由的 Top-8 集合是否改变”，不是改变的专家比例、生成 token 改变率或任务准确率下降。重复 token/层事件也不是独立统计样本。

这是操作数舍入的 CPU 敏感性测试，不是 B200 kernel 实测；当前 Fast 已允许 TF32，未将其实际输出作为本轮 A/B 基线。少量路由变化不自动代表质量失败，更多变化也不能单独证明任务质量不可接受。

完整结果与脚本：工作区 `yoco_results/fp8-precision-review-20260913/`；随本文保存的 `FP8_PRECISION_EVIDENCE.json` 含结果摘要与来源。

## 第一轮保留的精度

| 部分 | 本轮选择 | 原因 |
| --- | --- | --- |
| GEMM 累加器、专家结果归约、attention softmax 归约 | FP32 | 位于长求和或归约链；FP8 Tensor Core 的 FP32 累加不是普通 FP32 GEMM 开销；没有已验证的净收益 |
| 主残差 | FP32 | 40 个逻辑 block 的更新逐步积累；小更新可能被 BF16 舍入吞掉，尚无当前模型质量证据支持降级 |
| RMSNorm / RMSClip 的平方和、rsqrt 等统计 | FP32 | 可保留 FP32 统计，同时直接写 FP8 输出以节省转换；不必先降低统计精度 |
| Router 比较/softmax/路由权重，以及 log-prob/NLL 的统计 | FP32；离线必要时 FP64 | Router 投影可单独低精度，离散选择和概率归一化不必同时降低；评估工具本身不能随候选一起降级 |
| RoPE、差分注意力与 clamped SwiGLU 的关键中间计算 | 先保留当前 FP32 | 可融合与调整输出边界；进一步降低涉及长位置相位、相减抵消或非线性误差，暂无优先收益证据 |

“保留”不阻止以后独立实验。当前没有证据支持把这些位置直接整体改成 BF16/FP8，也不应把 dtype 字样作为性能瓶颈证据。

## 两个仓库的配套范围

- vLLM 负责 Fast 候选的 kernel、量化/缓存格式、GEMM 分派与完整服务验证。现有 Align 留作独立数值参考，Fast 不要求 bitwise 或生成 token 完全不变。
- llm-train 负责可切换的前向参考、checkpoint/scale 约定，以及需要时对应的训练实验。
- 训练代码已经有 `mxfp4` 权重、FP8 激活的 W4A8 路径：`llm/arch/linear.py` 与 `llm/kernel/moe_ffn.py` 使用 `recipe_b=(1,32)`，激活仍用 FP8。它可以作为 MXFP4 数值参考，但本次未验证整个训练流程的可用性或训练质量，也没有证明 NVFP4 可直接兼容此格式。
- 训练已有梯度量化、FP32 权重梯度等独立配方；推理静态 scale、Router 16-bit 或 FP4 实验不自动改变 backward、参数更新或优化器状态。
- 冻结模型的静态 scale 不能直接推广到训练：参数和激活分布会随 step 变化，已有量化缓存按参数版本失效；训练需要自己的 scale 更新/校准与收敛验证。

## 验证与采用规则

1. 一次改变一个精度或缩放决策，保留当前 Fast-FP8 和 BF16 对照，明确 weight dtype、activation dtype、accumulator、输出边界及 scale。
2. 静态校准先覆盖当前 Fast-FP8 的代码、数学、普通文本、prefill/decode、不同 M 和长上下文；校准集与质量验证集分开，记录截断与新增零比例。
3. 使用固定前缀 teacher forcing 比较原始 log-prob/NLL；在冻结的代码/数学及长上下文任务上报告结果。小样本 NLL、局部 L2 和 Top-1 不变均不能独立证明可默认启用。
4. 新量化策略报告实际质量/速度取舍，不把非 bitwise、少量路由改变或某个统一 L2 门槛当作所有候选的硬失败。
5. 数值和任务验证后测完整链路与端到端收益；归一化/量化融合、小 M BF16 kernel、FP8 kernel 都可竞争，选择实测效果。
6. payload 与 scale 的数值含义必须一致；不能把已有 FP8 张量的 scale 改为 1 而不重量化。更改 W2 路由权重位置、clamp 或舍入顺序必须被记录为新数值策略，不能当作等价实现。

9 月 13 日完成源码审核、CPU Router 敏感性测试和初始分级；9 月 14 日增补 DeepSeek V4 官方与 llm-train 策略对照，并据此调整候选顺序。未启用任何新精度策略，未修改在线服务，未新增 B200 性能或完整模型质量结论。
