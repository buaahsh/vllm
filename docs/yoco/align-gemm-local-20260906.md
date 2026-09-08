# Align MoE GEMM 本地优化

本轮为本地实验：RTX A6000（SM86）、Torch 2.11.0+cu130、Triton 3.6.0、CUDA 13.0。
数据为随机 BF16 权重和 hidden states，使用 L3 的真实 MoE 形状：128 experts、Top-8、
hidden=1024、FFN=3840。没有运行 B200、完整模型或 AIPerf；结果不代表端到端吞吐。

## 改动

vLLM 与 llm-train 共用 `vllm/model_executor/layers/yoco_align_moe.py`，分别选择 W13/W2
的 launch 配置。两次 GEMM 共用 M tile 和 expert assignment，允许 N tile、warps、stages
与 program 分组各自配置。保持以下算术边界：

- K tile=32、split-K=1；不引入跨 CTA 的浮点归约。
- W13 输出先存为 BF16；原 weighted SwiGLU 完成 clamp、激活和 router weight 乘法后存为 BF16。
- W2 仍输出 BF16，再按原 Top-8 顺序以 FP32 累加。
- 训练继续保存真实 `pre` / `act`；backward 算术没有改动。

更改 M/N tile 不等于自动保证 bitwise。它可能改变 Triton 的 MMA 布局和生成指令，
因此每个候选都需要逐字节检查。只关闭 split-K 并不是充分条件。

## 选择与本地结果

第一轮较小 M tile 的 72 组检查全部逐字节相同，但没有稳定加速，集中路由还出现回退，
未采用。第二轮大 M/N tile 的 88 组检查也全部逐字节相同，随后为 W13/W2 分别选择配置，
用 A/B/B/A 复测，并用新输入核对两端实际执行入口。

实验 profile 仅列出 M=512/1024/2048；未记录的精确行数保留原配置，没有最近邻外推。
512 使用 M=128、W13 N=64、W2 N=256；1024/2048 使用 M=128、两次 N=256。
实际数值与速度详见本地报告，尤其要区分分散路由和集中路由。

本地回归包含共享配置的拒绝条件、vLLM 实际 `TritonExperts.apply` 的两次 GEMM 与中间值、
训练适配器的保存值和梯度。L3 形状的独立联动检查覆盖配置点及相邻行数，比较 vLLM、
训练带梯度前向和首/中/尾目标的单 token 前向。其范围是 **MoE 层**，不包括 attention、
整模型 logits/log-prob 或完整 NNScaler 训练图。

## 实验开关与两端联动

默认不启用新配置。在两个进程中设置同一个经过验证的 profile：

```bash
export VLLM_YOCO_ALIGN_MOE_CONFIG=/absolute/path/to/validated-profile.json
```

profile 必须匹配 GPU 名称及 Torch/Triton/CUDA 版本。文件格式和 launch 参数被严格校验；
K tile、split-K 或两次 M tile 不符合约束时拒绝加载。显式的调用方 config override 保留。
非 Align、量化、其他权重形状、LoRA、bias 或 expert parallel 路径不采用该配置。
加载结果按进程和设备缓存；修改 profile 后使用新文件路径并重启自己的实验进程，
不要在已有 CUDA Graph 运行期间修改环境或 profile。

本地候选位于：
`vllm/model_executor/layers/fused_moe/experts/yoco_configs/align_moe_NVIDIA_RTX_A6000.experimental.json`。
它是 A6000 算子实验配置；B200 会因环境不匹配而拒绝加载，需要在 B200 上单独筛选、
复验两端和跨 batch 字节结果。完整 Align 模型目前仍要求 B200，未放宽模型运行条件。

llm-train 需使用同步的 `llm-train-align-gemm-20260906` 开发实现；未设置开关时仍兼容
原 `VLLM-ALIGN-REVISION` 指定的默认 Align 版本。候选版本还没有作为新的 B200 配套版本发布。

## 检查与证据

新增测试：

- `tests/model_executor/test_yoco_align_moe_config.py`：配置校验、精确行数、环境与模式约束。
- `tests/kernels/moe/test_yoco_align_gemm.py`：实际专家入口、中间 tensor、单 token 对照。
- llm-train `llm/tests/test_align_gemm_config.py`：共享配置接入、真实保存值、梯度对照。

本地完整日志、所有候选、失败记录、配对时间、联动检查和脚本保存于：
`/home/lidong1/vllm_test/yoco_results/align-gemm-local-20260906/`。
最终数值和适用范围以该目录的 `REPORT.md`、`VALIDATION.json` 为准。

后续在 B200 上需要完成同卡旧/新 Align 的逐字节门禁，再用此前同一份 Mooncake
开源 trace 进行性能验收；本地 kernel 加速不替代这两项。
