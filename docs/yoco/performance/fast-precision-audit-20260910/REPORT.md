# Fast FP8 与训练 FP8 的精度审计

本文记录实施前的审计快照。后续已完成[routed直接FP8量化与Top-8 logits路由接入](../fast-fp8-route-direct-20260910/REPORT.md)，具体实现与数值/性能结果另见该报告。

**当前推理没有发现成片遗漏的 FP8 投影：权重和 GEMM 操作数的精度覆盖与当前 llm-train 的 FP8 配方基本一致。** 剩余 BF16 的 lambda、latent 投影、shared gate、LM head，以及 FP32 残差/路由，都有训练侧依据。可以继续开发选择性降精度，但应将它作为新的数值策略评估。

更直接的优化机会是：减少 routed SwiGLU 的额外 BF16 舍入和小 M 中间缓冲，简化推理路由的完整 softmax，以及在适用采样条件下减少 FP32 logits 的落地。这些主要减少转换和计算，并非把归一化/累加一律降为 BF16。

本轮为源码、实际 checkpoint 元数据和已有运行快照审计，未新增精度、质量或 GPU 性能测量。候选收益与可接受误差尚未验证。

## 审计对象与证据

- 训练：`llm-train-align-gemm-20260906`，`fhb-dev-9-8`，HEAD `0b6978ee35e95982a4ea10bd8de63d6264bb676d`。
- 推理：`vllm_yoco_align_gemm_20260906`，`fhb-dev-9-8`，HEAD `511fbed3f75f0fd18a4f93194ca832db2c723f98` 加已有本地改动。
- 实际模型：YOCO 30A3B-180M-L3 step28000。`0000-28000-merged/metadata.json` 保存的训练参数是 `quant_mode=mxfp8, quant_block_size=128, use_cute=true`。
- 导出的 `0000-28000-hf/config.json` 写的是 `quant_mode=bfloat16`，描述导出的权重配置，不能据此认定训练未用 FP8。当前推理由启动参数 `--quantization fp8_per_block` 开启在线量化。
- 推理精度依据上一轮 `candidate-before/PASS.json` 内的实际模块快照，和当前本地源码核对。现在保留的在线服务是 BF16；本轮没有切换或加载新服务。

训练这里的 `mxfp8` 是 E4M3 权重/激活、UE8M0 幂次 scale；默认激活按每行128列分组、权重按128×128分块。不能仅凭名字把它当成另一套 block-32 MXFP8 配方。训练和推理的完整内核、分派及舍入边界仍可能不同，未声明 FP8 跨端 bitwise。

所有源文件hash、模型参数、快照引用和模块清单见 [audit.json](audit.json)。本次比较当前训练源码，没有追回原始训练运行的全部编译产物。

## 逐项精度对照

以下区分参数保存格式、GEMM操作数、归约/中间计算和输出格式。Tensor Core 的 FP32 累加器不能直接按普通FP32 GEMM的成本理解。

| 部分 | llm-train FP8 前向 | 当前 Fast FP8 推理 | 判断 |
| --- | --- | --- | --- |
| Q/K/V、attention O、模型级共享 K/V 投影 | W8A8 block128，GEMM输出BF16 | 同等级操作数和输出；已融合QKV/KV | 没有BF16漏量化的大投影 |
| routed experts W13/W2 | W8A8 block128，GEMM输出BF16 | 20组专家均为FP8；小M Triton/其余DeepGEMM | 不能把输出BF16或累加器FP32视作未启用FP8 |
| shared expert up/gate/down | W8A8，层间BF16 | 同样FP8，激活量化已融合 | 已覆盖 |
| lambda / attention gate | 默认BF16 linear | BF16，显式不量化 | 与训练一致；进一步量化是新策略 |
| fc1/fc2 latent 投影 | 默认BF16 linear | 40个BF16投影 | 与训练一致；是可独立实验的降精度对象 |
| shared scalar gate | BF16 GEMM操作数 | BF16操作数 | 与训练一致；训练参数可能保存FP32，forward会转为BF16 |
| embedding / LM head | embedding后转FP32残差；LM head为BF16 GEMM | 相同；Fast LM head先舍入BF16，再写FP32 logits | 不能把FP32 logits误认成FP32 LM head乘法 |
| 主残差 | embedding转FP32；每层 `x + r.float()` | FP32 residual | 训练原本如此，优先保留 |
| RMSNorm / RMSClip | 统计及中间运算保留FP32，输出BF16 | 同等级；具体编译/舍入边界另行验证 | 不宜首先降低统计精度 |
| RoPE | FP32 trig cache与旋转中间计算，输出BF16 | 同等级 | BF16缓存/数学近似需要单独长位置验证 |
| attention Q/K/V 与输出 | 投影量化后恢复BF16，传入attention | BF16 Q/K/V、BF16 KV cache与输出 | 训练并没有把attention操作数全部压成FP8 |
| router logits / softmax | FP32 gate输入/权重和FP32 softmax | FP32数据；Fast router已允许TF32乘法 | 推理并非更保守；继续降低需检查专家选择变化 |
| 专家合并 | BF16专家输出，相关合并实现使用FP32累加 | BF16输出、FP32累加 | 保留较稳妥；不同分布式合并后端需分别核对 |
| scale | UE8M0数值，常用FP32张量承载 | DeepGEMM打包指数；小M缓存展开成FP32 | 展开后的FP32没有增加scale有效位数 |

实际快照共183个带线性/embedding方法的模块：81个FP8、82个BF16、20个FP32 router；此外还有20组FP8 routed experts。82个BF16模块恰为20个lambda、20个shared gate、40个latent投影，以及embedding和LM head。

## 先做的转换与计算优化

### 1. routed SwiGLU 直接量化FP8，减少BF16中间边界

训练 `llm/kernel/quant.py:207` 的融合路径：

```text
BF16 W13输出 → FP32 clamp/SiLU/乘up/乘路由权重 → 直接量化FP8 → W2
```

当前推理 [fp8_utils.py](../../../../vllm/model_executor/layers/quantization/utils/fp8_utils.py) 的 `_silu_mul_quant_fp8_packed_kernel` 在计算量化amax前显式执行 `y.to(bfloat16).to(float32)`。小M Triton分派还先将加权激活写入BF16缓冲，再单独量化。

可开发一个仅用于routed FP8的直接量化入口，同时覆盖DeepGEMM和小M Triton所需的scale布局。对大M融合路径主要节省转换指令；对小M分离路径还可减少缓冲读写和一次launch。**去掉BF16中间舍入并不是降低数值精度，反而更接近当前训练的FP8路径。** FP8结果和scale可能变化，仍需要误差验证。

不能直接全局删除这次舍入：同一个量化kernel也供shared expert使用，训练shared SwiGLU的输出先回到BF16再进入下一层量化。需要按调用方区分数值策略，同时保留BF16/Align路径。

### 2. 重评已有的Top-8 logits候选

历史核对更正：这不是尚未实现的新想法。`4015343fa3`（2026-08-31）已在 [benchmark_yoco_router_topk.py](../../../../benchmarks/kernels/benchmark_yoco_router_topk.py) 中提供 `fused_logits_topk_kernel` / `triton_logits_routing`，实现先Top-8 logits、再8路softmax，并纳入正确性和计时入口。当前服务未选择该候选。本轮未找到足以引用其独立历史收益的测量记录，不推断当时未启用的全部原因。

Fast此前已经删除dense routing_probs/routing_map和第二次Top-K（`d11cc022bd`），将完整softmax、Top-8与重新归一化融合为一个kernel（`4015343fa3`），并直接输出INT32 expert IDs以避免额外cast（`d7dcdd9e83`）。这些优化仍在当前BF16/FP8 Fast公共路由路径中生效。当前kernel注释明确说明保留完整softmax是为了减少与原实现的数值差异。后续工作应重评并按结果接入已有候选，不能重复计算上述已完成收益。

训练需要完整128路概率来计算辅助损失；推理只消费选中的专家和重新归一化后的权重。当前 [yoco.py](../../../../vllm/model_executor/models/yoco.py) 的 `_yoco_fused_topk_routing_kernel` 仍对128路做FP32 softmax，再Top-8并重新归一化。

对有限logits，在实数运算下：

```text
top8(softmax(logits)) 后重新归一化 = softmax(top8(logits))
```

可以保留FP32的比较、指数和归约，只减少指数数量及被抵消的归一化。浮点softmax可能使极近logits变成并列，因此不能保证所有输入下expert IDs或权重bitwise不变。验证应包括边界并列规则、Top-8集合变化率和路由权重误差。

### 3. 延迟或避免FP32 logits落地

当前Fast LM head的输出已经先舍入BF16再转FP32，FP32缓冲未携带额外的logit有效位数。对于没有惩罚项、特殊processor、温度变换且不请求log-prob的纯greedy路径，可研究保留BF16值直接argmax，或者与输出投影/归约融合。

这需要连同 [sampler.py](../../../../vllm/v1/sample/sampler.py) 的统一FP32转换处理；只把LM head输出改为BF16，下一步马上转回FP32，可能增加开销。一般采样、log-prob和归一化统计继续使用FP32。当前已有specific-token log-prob融合，不能把它重复列为新优化。

## 可以继续下压的候选

以下均超过训练当前BF16保留配方。用户已允许小误差，但尚未实测这些候选的质量和速度。

| 优先级 | 候选 | 可影响部分 | 实现与验证重点 |
| --- | --- | --- | --- |
| P1 | LM head BF16→FP8 | 每个输出token均访问大词表权重；原始权重907.5→453.75 MiB | online quantizer目前不处理ParallelLMHead，需要专用适配；保留BF16/FP32输出统计；比较现有小M BF16专用kernel |
| P1 | latent两投影 BF16→FP8 | 40组投影原始权重240→120 MiB；逻辑循环中反复执行 | 1024/3072维符合block128；分别测两投影和整体MoE，量化开销可能抵消小M收益 |
| P2 | lambda BF16→FP8，配合Q/QKV融合 | 全部lambda权重仅7.5→3.75 MiB；主要争取省launch | lambda输出64维，合并需要处理128块边界/padding；仅改dtype不会自动启用现有BF16融合 |
| P2 | KV BF16→FP8 | 长上下文的KV容量和读取带宽 | 当前B200 FA4不可直接启用，需先适配支持的attention/KV存储路径及YOCO KV sharing，并验证长上下文质量 |
| P3 | shared scalar gate BF16→FP8 | 全模型仅120 KiB BF16权重 | 收益空间小，而且已有融合gate计算；不宜增加独立量化/GEMM |
| 暂后置 | residual FP32→BF16、router权重/比较下压、RMS统计下压 | 残差带宽或小算子计算 | 残差误差随逻辑深度积累；router影响离散专家选择。训练本来保留这些精度，缺少先压它们的证据 |

权重大小是按模型形状计算的payload，未计scale、padding、缓存和allocator；不是实测显存或性能增幅。Embedding也是大BF16矩阵，但运行时主要是按token查表，与每步读取全部权重的LM head不同，优先级较低。

当前 [`flash_attn_supports_fp8()`](../../../../vllm/v1/attention/backends/fa_utils.py) 的CUDA条件是FA3且SM90；本模型运行在B200/SM100与FA4。因此FP8 KV不是添加一个参数就能完成的优化，也不能通过只量化Q/K/V投影来实现。

FP32的scale缓存也不是优先数值下压对象：其中值是已经确定的2的整数次幂，改变承载格式不会改善量化本身，且小M展开缓存正是此前实测更快的分派所需。若改为更紧凑格式，应测解包成本，不能只按字节数判断。

## 开发顺序与验证

1. 先做routed直接FP8量化，并重评已有router Top-8 logits候选，分别消融。该路由helper也被Align复用，候选如接入应限定Fast，避免改变Align的数值契约。
2. 再分别尝试LM head和latent投影FP8；lambda量化与真正可用的投影融合一起评估。保留默认策略和可冻结的候选配置。
3. 对长上下文单独开发FP8 KV，明确attention后端变化。暂不优先下压残差、router选择与归一化统计。
4. 数值使用相同输入/相同目标token的log-prob、NLL/PPL；router另看Top-8集合变化。覆盖短输入、8K、长上下文、低并发和批处理。格式与缓存刷新做精确检查，跨精度不要求bitwise。
5. 数值结果通过后，再按既有同卡条件测低并发与Mooncake 2×。Mooncake公开trace只有时序、长度和前缀关系，不能替代真实文本质量验证。

[最新同条件性能基线](../fast-bf16-mooncake-f2-20260909/REPORT.md)：Fast FP8 1407.84、BF16 1238.79输出tok/s，FP8高13.65%；本次审计没有改变这些测量值，也不承诺候选可再提高多少。

## 主要源码定位

训练路径以当前训练仓库为根：

- `llm/arch/linear.py:29,133,218`：FP8和BF16 linear前向、BF16输出及构造默认精度。
- `llm/arch/moe.py:46,54,71`：FP32 router、BF16 latent与shared gate。
- `llm/arch/attention.py:93,118,120`：QKV/O量化，lambda保持默认BF16；后续attention接收BF16张量。
- `llm/arch/model.py:63,68,110,142,261`：FP32残差、BF16 LM head和FP32 logits。
- `llm/arch/rms_norm.py:12,24`、`llm/arch/rotary_embedding.py:15`：归一化与旋转精度。
- `llm/kernel/quant.py:185,207,245`、`llm/kernel/moe_ffn.py:160`：scale规则、直接激活量化及专家GEMM输出。
- `llm/arch/all2all_moe.py:48`、`llm/nnscaler_train.py:238`：训练路由概率和参数/缓存精度策略。

推理路径以当前vLLM仓库为根：

- `vllm/model_executor/models/yoco.py:634,752,1536,1656,1986,3300,4247`：路由、logits舍入、TF32、lambda排除、残差norm、MoE例外与LM head。
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py:240`：额外BF16舍入。
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:395`：小M加权激活的BF16中间缓冲。
- `vllm/model_executor/layers/fused_moe/experts/triton_deep_gemm_moe.py:39`：scale的无损展开缓存。
- `vllm/model_executor/layers/quantization/online/base.py:140`：当前online量化分派范围。
- `vllm/v1/attention/backends/fa_utils.py:194`：FP8 KV硬件/版本限制。
- `vllm/v1/sample/sampler.py:90`：logits统一转FP32。
