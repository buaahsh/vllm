# YOCO 工程化重构计划｜2026-09-15

实施进展与验证记录见 [迁移说明](docs/yoco/migration-v029.md)。下面保留计划制定时的源码审查基线。

建议从“模型结构、数值语义、执行策略、资源生命周期”四个边界整理。完成后，阅读主模型能理解 YOCO 的执行过程；新增一种 kernel 或调整 B200 分派时，能在对应模块内完成修改，并用已有测试验证行为。

本轮只做源码审查和计划，未修改实现、运行测试或操作服务。

## 范围与基线

- 用户指定最新版：`/home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906`，分支 `fhb-dev-9-18`。
- 审查包含已跟踪但未提交的修改及最新未跟踪源码；不能只从 HEAD 创建分支就当作当前版本。
- HEAD：`511fbed3f7`。审查时主文件为 5,437 行，SHA-256：`1c1f288db2af827bb6e424e88519f5ee324158c989e1c69563b5bcde13c92dc7`。
- 源码盘点时间：2026-09-15 08:09 UTC。后续实施前需检查工作区是否继续变化。
- 主要范围是 vLLM 侧 YOCO 实现及其接入边界。训练侧仅检查共享接口兼容性，训练架构重构另行安排。

后续重构目录已按用户要求复制为 `/home/lidong1/vllm_test/vllm-yoco-version-0.29`，使用同名本地分支；`/home/lidong1/vllm_test/vllm-yoco` 是指向该目录的入口链接。副本使用独立 Git 仓库，保留源工作区的已跟踪修改、未跟踪文件和原有符号链接，计划也随副本保存。复制后已进行工作区内容校验；阶段 1 的运行基线与测试尚未执行。后续实现应在此副本内进行，下面的源码链接保留为本次审查的来源。

## 源码里具体的问题

| 观察 | 证据 | 维护上的影响 |
| --- | --- | --- |
| 主文件包含模型、Triton kernel、fake impl、自定义算子注册、策略和调试导出 | [yoco.py][model] 共 5,437 行；约第 377–2063 行集中放 kernel、包装与注册；AST 统计有 19 处 `direct_register_custom_op` 调用 | 阅读模型结构之前，要穿过大量与结构无关的实现；算子变更与模型变更混在一起 |
| 后端选择函数同时修改运行配置 | [`_select_yoco_fast_moe_backend`][backend] 共 130 行，内部会改 `enable_flashinfer_autotune` | 函数名字看起来只是选择，调用却影响后续 warmup；构造顺序成为隐藏依赖 |
| 同一个执行模式分散读取 | `yoco_execution_mode` 在 arg_utils、models/config、model、FlashAttention、GPU runner、GPU worker 六个生产文件中读取 | 默认值、校验和生效时机容易出现分歧 |
| MoE 使用运行时附加属性传递策略 | [模型构造][moe] 设置多项 `experts.yoco_*`；[modular method][modular]、[unquantized method][unquantized]、[fallback][fallback] 再逐项复制 | 加一个选项需要记得改多条通路，漏传会悄悄回到默认值；部分复制发生在 forward 中 |
| 权重装载承担多种职责 | [`load_weights`][weights] 共 258 行，处理别名、融合投影、专家分片、默认 loader、派生缓存刷新 | 新增一个融合投影会继续增加主入口分支；装载完成与缓存可用之间缺少集中说明 |
| 快速 prefill 与运行时状态交织 | [`_fast_prefill_forward`][prefill] 共 142 行，涉及 profile、FULL/PIECEWISE 图、固定地址 buffer、DP metadata、KV-only prefill | 看似局部的整理会改变图生命周期、collective 次序或 KV 写入依赖 |
| 测试也聚集成大文件 | [test_yoco_conversion.py][conversion] 共 2,981 行、68 个 test 函数，包含路由、norm、RoPE、分派、KV-only prefill 和权重转换 | 测试名不能准确表达覆盖范围，找一项行为的验证成本高 |

YOCO 自身确实比普通模型多出 universal loop、共享 cross-KV、diff attention、latent MoE、Align/Fast 等复杂性。同工作区 `qwen3_moe.py` 是 788 行，但不能把这个行数直接作为 YOCO 的目标。合理目标是降低每次修改需要理解的范围。

现有实现已经有值得保留的工程基础：`yoco_fast.py` 的按实际精度决定融合与缓存刷新、`yoco_probabilities.py` 的概率归约、独立 MoE backend 文件，以及针对数值/缓存/尾块边界的测试。后续沿这些边界继续整理。

## 建议的模块归属

以下是目标草图；文件名在实施时可以随现有目录约定微调，职责边界应保持清楚。已经独立的 `yoco_fast.py`、`yoco_probabilities.py`、`yoco_align_moe.py` 和 FP8 算子文件先保留原路径。

```text
vllm/
  config/yoco.py                       # 轻量的规范化选项与策略数据类型
  model_executor/
    models/
      yoco.py                          # 对外入口、模型装配和执行顺序
      yoco_weights.py                  # checkpoint 映射、分片与加载报告
      yoco_prefill.py                  # fast-prefill 运行时编排
    layers/
      yoco_attention.py                # self/cross attention 的模型级组合
      yoco_moe.py                      # router、latent、shared/routed expert 组合
      yoco_ops/
        norm.py                        # RMS/clip/residual 的相关实现
        rotary.py                      # RoPE 与 QK 组合算子
        routing.py                     # router/top-k 与相关归约
        projection.py                  # diff combine、linear/LM-head 等投影算子
      fused_moe/experts/yoco_*.py       # 保持后端实现归属
```

`yoco.py` 最终保留 `YOCOForCausalLM`、`YOCOModel` 及必要的 block/decoder 装配，明确展示 embedding → self loops → 共享 KV producer → cross layers → norm/head。初步希望降至约 600–1,000 行，这只是阅读预算，不能为了行数改变合理边界。

依赖方向：配置数据 → 算子/后端 → attention/MoE 组合 → 模型装配。Scheduler、runner、connector 通过轻量的策略/元数据边界接入；底层 kernel 和公共后端不能反向 import 主模型。

算子按功能成组存放，每组把 reference/fallback、CUDA/Triton 包装、fake impl 和注册放在相邻位置。SelfAttention 与 CrossAttention 继续共享真正相同的投影/归一化组件，保留各自的 KV 语义。沿用现有 vLLM Attention、FusedMoE、量化和 loader 接口，不额外建立一套 YOCO 通用插件框架。

## 六个实施阶段

### 1. 固定当前行为，整理验收入口

把最新版工作区做成独立、可还原的开发快照：记录 HEAD、tracked diff、未跟踪源码、配置与测试；实验大文件只记录索引/校验值。之后从该快照开始重构，避免漏掉尚未提交的 FP8 等工作。

保留现有测试用例，逐步按 config、weights、attention、MoE、prefill、kernel 分类。`test_yoco_conversion.py` 最终主要保留 checkpoint 转换，其他测试随所属模块迁移。新增测试优先覆盖新边界的失败模式，避免大量只检查函数是否被调用的测试。

基线记录需要包含：模式、逐层实际精度、KV dtype、拓扑、backend、图捕获设置、tactic cache 和版本、固定输入、输出及已有预期失败。Fast 与 Align 在独立进程中建立基线。

完成条件：当前源码可复现，现有测试的通过/跳过/失败状态清楚，后续每阶段都有可比较的旧版本。基线自身的缺陷单独登记，不能归入重构收益或回归。

### 2. 先迁出算子，保持注册与调用契约

将主文件中的 norm、rotary、routing、projection 算子按上述功能迁入 `yoco_ops/`，每次迁一组；一起迁移 CUDA 包装、fake impl、reference 和对应测试。此阶段保持运算表达式、cast/rounding、Triton 参数与分派条件。

所有 `torch.ops.vllm.yoco_*` 名字、参数 schema、mutation/alias 声明保持一致。注册沿现有模块导入时序发生，且只有一个归属；不能等到已开始 tracing/capture 时才首次注册，也不能从两个路径重复注册。

保持 `vllm.model_executor.models.yoco.YOCOForCausalLM` 入口。已知内部函数调用者包括本仓库的工具、测试和 benchmark；迁移时同步更新这些调用者，必要的旧路径临时 re-export，并列出具体兼容对象和撤除条件。涉及 monkeypatch 的测试要修正实际引用位置，不能以“旧模块还能 import”作为迁移完成的证据。

完成条件：相关数值测试、fake/opcheck、模型 import/registry 检查通过；eager、compile、CUDA Graph 的执行结果与基线一致，没有新增 graph break。Triton/Inductor 源码路径变化可能导致冷缓存重编译，冷启动成本与预热后的性能分开记录。

### 3. 统一配置解析，让策略选择可解释

引入少量有类型的数据结构，分别表示模型结构参数、用户执行选项和已解析策略。结构参数归一化可由轻量配置模块完成，但保持 HF `YOCOConfig` 对历史字段/别名的兼容。BF16 residual、FP8 fusion、backend tuning 等选项从同一个入口读取并校验，保留现有 CLI、additional_config、环境变量接口。

分清三个时点：

| 时点 | 应处理的内容 |
| --- | --- |
| 引擎配置、worker 启动之前 | 模式及冲突检查、Align 的进程级 invariant/TF32 设置、已知精度和并行限制 |
| 量化配置及设备信息可用、构造具体 layer 时 | 按该层实际精度与 ignore rules 决定融合、backend、autotune 与图策略 |
| forward 时 | 必须依赖当前 batch 的形状/metadata 分派；使用已解析常量和现有可编译分支 |

现有 `YOCOForCausalLMConfig.verify_and_update_config` 在 `quant_config` 构造之前执行，不能把逐层决策全部提前到这个 hook。Align 当前还有 [进程级概率策略][probabilities]，不能通过一个实例对象假装消除了全局状态。

将 `_select_yoco_fast_moe_backend` 改成明确返回“选择结果及配置调整”的决策函数，在已有合法初始化时点统一应用调整。设备探测/tactic cache 读取与纯规则判断分开；最终记录选中后端、适用条件及回退原因。分派阈值和 tactic 值先原样迁移，不顺便重新调参。

完成条件：对已覆盖的模式、精度、角色、图大小、硬件与 backend 组合，新旧决策一致；显式选项的优先级与拒绝条件保持一致；构造顺序和预热行为有测试保护。避免为了策略对象在热路径增加 Python 动态分发或 GPU→CPU 同步。

### 4. 拆模型组件，整理权重和缓存生命周期

迁出 `YOCOSelfAttention`/`YOCOCrossAttention` 和 MoE/shared expert/latent transform 组合，主模型只负责装配与循环执行。最先保持模块属性路径、共享参数关系、compile 边界及 `state_dict` 名称，代码所在文件变化不应改变 checkpoint 参数名。

权重加载拆成明确的流水线：识别来源名称 → 匹配目标及 shard → 调用既有 loader → 汇总加载结果。简单别名/融合投影用映射规则表示，专家 tensor 的拆分仍用专门函数，不强行用一个万能映射表表达所有情况。

单独暴露“加载完成后的派生缓存刷新”步骤，复用 [现有缓存原址刷新契约][fast_helpers]。保持 graph 引用的 storage 地址、non-persistent buffer、layerwise meta 恢复和模型共享引用关系。不能新增一个持有全部模型的 `nn.Module` 管理器，否则可能改变参数路径或形成重复注册。

当前 forward 里有缓存初始化兜底，覆盖不调用模型 `load_weights` 的加载器。先梳理这些入口，在建立统一完成通知前保留兜底。首次加载、重复加载、layerwise reload 分别验证。

当前未知权重会静默跳过，默认 loader 的 `TypeError` 会触发兜底；这是需要审查的边界，尚不能仅凭源码断言已经发生装载错误。迁移先保持兼容；再单独增加允许忽略列表、未加载报告与 loader 适配检查。收紧失败语义应作为独立行为变更评审。

完成条件：checkpoint 名称和分片覆盖一致，重复加载会更新派生缓存，图重放能看到新权重，storage 地址契约成立；同输入模型结果与阶段 1 的基线一致。

### 5. 收口 MoE 后端间的策略传递

将当前多项动态 `yoco_*` 属性整理为有类型、可校验的后端选项。语义上至少区分 routing weight 生效位置、SwiGLU clamp/舍入边界、expert 求和顺序、逐层精度，以及纯性能的 tuning/fallback 设置。

在 layer 创建和后端构造/重建时传入完整策略，由 Triton、FlashInfer、DeepGEMM 适配器消费。fallback 选择后端时传递同一份有效策略，避免每层手工复制一串属性。量化处理或 reload 可能重建后端，要覆盖这些时点；不能只在最初 `__init__` 传一次就认为完成。

先把现有 YOCO 选项收成明确契约，已有公共参数继续复用。确有多个模型需要的语义再提升为公共字段；仅改成没有 `yoco_` 前缀不构成抽象改进。

完成条件：backend/fallback 切换前后所需语义完整、拒绝不兼容组合；Align 的 routed/shared reduction 次序及量化路径的 router-weight 位置保持；公共层变更通过至少一个受影响非 YOCO 模型的回归。

### 6. 最后整理 KV/prefill 边界和实验入口

这是风险最高的阶段，拆成独立提交并逐项验收。先整理 `yoco_prefill.py` 的输入、输出、buffer 所有权和状态，再处理 scheduler/runner/connector 的共享规则。

把“是否为 dedicated KV producer”“本批是否允许省略 cross”“尾块重算起点”定义成明确的策略与元数据契约。共有的无状态尾块计算可以先提取，scheduler 仍负责调度，connector 仍负责传输；不要为了搬代码把这些状态迁进模型类。

保留 self/cross 分别编译、KV 写入依赖、DP metadata 恢复、跨 rank 的执行一致性、FULL decode 路径和固定地址 buffer。把 512 个左侧历史 token → vLLM 窗口 513 的转换集中说明并测试。

P/D 以当前尾块语义为基线：例如 2,048-token prompt、block size 16，P 处理/传递到 2,032，D 重算末尾 16；P 当前跳过相应 prefix-cache 读取的行为也必须保留。恢复 P 端前缀缓存收益是后续功能优化，另立验收。

调试 route dump 从模型热路径中的文件系统判断和全局计数器，迁到显式启用的诊断入口，保持 eager-only 限制。已有 BF16/FP8 实验开关集中记录默认值、验证范围、调用方和退役条件；确认无人依赖且有替代路径后，才删除分支。此次不以文件日期判断功能是否过期。

完成条件：本地缓存、远端 KV、混合批次与重复调度的可观察行为保持；P/D 数值和资源释放回归通过；诊断关闭时没有文件系统轮询，也不改变编译图。

## 验收矩阵与合并门槛

每阶段只运行与修改有关的检查，达到门槛后推进。涉及公共层、图生命周期或 P/D 时扩展到对应集成测试。这里列的是后续验收要求，本轮没有执行这些测试。

| 层次 | 优先复用的验证 | 判定方式 |
| --- | --- | --- |
| 配置与分派 | `test_yoco_config.py`、`test_yoco_fast_standalone_moe.py`、`test_yoco_fast_decode.py`、FP8 配置测试及相关 arg_utils 测试 | 默认值、冲突拒绝、backend、fallback、tactic/图边界与基线一致 |
| 装载与重载 | 现有 conversion、fast precision 测试 | 原名称和融合 shard 不漏载、不重复处理；缓存内容与指针、meta 恢复正确 |
| 算子 | Align invariance、MoE、probabilities、FP8/attention、已有 opcheck | dtype、舍入、mutation/alias、fake tensor 与输入布局保持；比较需要覆盖阈值两侧 |
| 整模型 | 现有 `tools/yoco_alignment/` 工具与固定前缀探针 | Align 在已支持配置下比较 logits、log-prob、CE 字节；覆盖 decode、prefill、chunked/mixed/cache 路径 |
| 图执行 | eager、compile、PIECEWISE、FULL；首次及 reload 后重放 | 无新增 graph break、非法状态修改、capture failure、失效 buffer 引用 |
| P/D 与缓存 | `test_mooncake_yoco_tail.py` 及真实 1P1D 回归 | 尾块、SWA 历史、重复 admission、空传输完成/释放；共享 KV 的值、dtype 和尺度保持 |
| 公共框架 | 被改动的 MoE/attention/runner 测试，加非 YOCO 对照 | 默认公共行为与现有支持范围保持 |
| 性能 | 同卡固定形状测试；里程碑时复用现有长上下文和 trace 负载 | 小 batch、分派阈值附近、长 KV、prefill 均覆盖；分别记录冷启动与预热后结果 |

模式与精度优先选择：Align BF16、Fast BF16、Fast 在线 block-FP8，以及 FP8 模型中部分 layer 被忽略量化的混合情况。FA/TP/DP/PP/CP/EP 只按当前已支持和已验证范围选择；不把“模块拆开后能构造”视为新增支持。迁移实验开关相关代码时，补测相关开关，而非机械穷举全部组合。

性能和数值门槛建议在阶段 1 固定：

- 纯重构时，固定输入、同后端与配置的确定性路径应保持逐位结果。Fast 不承诺与训练 bitwise，不代表可以接受无解释的重构前后漂移；若旧路径本身不确定，先测旧版本自身重复运行的差异。
- Align 的保证沿用已验证范围，不扩张成任意并行、任意量化配置的保证。关键路径改变后需复验与训练侧的共享算子/前向接口。
- 建议以预热后关键固定形状指标退化超过 3% 作为调查线，用至少三次同卡配对测量估计噪声；这不是已经测得的性能，也不是允许每个阶段累计损失 3%。每阶段与直接前序比较，最终还要与最初基线比较。
- trace 记录成功率、请求完整性、客户端是否准时、TTFT/ITL、吞吐和显存。超过负载上限或客户端迟发的数据不能直接作为性能通过证据。显著回退应定位并拆出，不用缩短代码抵消。

## 兼容与回退

实施按上述阶段形成多个可回退提交；算子搬迁、分派重写、装载行为收紧、KV 状态调整分别提交。每个阶段记录所用源码快照及实测结果。出现差异时回退当前阶段，保留最后一个通过验收的版本。

必须维护的接口包括：模型 registry/import、checkpoint 名称与融合 shard、现有配置入口、`torch.ops` schema、KV cache/sharing 名称、graph buffer 生命周期，以及训练侧使用的 `yoco_probabilities` 和 `yoco_align_moe`。训练仓库已有 `align.py`、`cross_entropy.py`、`eval_utils.py` 直接导入这些共享模块，移动时需兼容或协同迁移。

建议首个里程碑完成阶段 1–3：得到可复现基线、独立算子模块和一致的策略解析。第二个里程碑完成模型组件/权重/MoE 后端边界；KV/prefill 的整理单独收尾。最终完成标准是修改有明确归属、行为有自动验证、主模型可直接阅读，行数仅作为辅助信号。

[model]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/models/yoco.py:377
[backend]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/models/yoco.py:245
[moe]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/models/yoco.py:3852
[modular]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py:97
[unquantized]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:313
[fallback]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/layers/fused_moe/experts/fallback.py:164
[weights]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/models/yoco.py:5180
[prefill]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/models/yoco.py:4984
[conversion]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/tests/model_executor/test_yoco_conversion.py:1
[fast_helpers]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/layers/yoco_fast.py:10
[probabilities]: /home/lidong1/vllm_test/vllm_yoco_align_gemm_20260906/vllm/model_executor/layers/yoco_probabilities.py:23
