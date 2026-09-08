<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## YOCO Align 与 Fast 开发

本分支 `fhb-dev-9-8` 包含 YOCO 的 `--align` 和 `--fast`。Align 的前向一致性结论限于已验证的配置与输入范围；Fast 优先性能，不保证 bitwise。上游发行版不包含本分支的开发改动。

### Fast 低并发 W2 优化（2026-09-08 UTC）

修复了 Fast 沿用 YOCO FA4 强制单 split 的限制：Fast 恢复后端 split-KV 自动调度，Align 保留原规则。同物理 B200、BF16 TP1，完整 W2（65,536 输入 + 16,384 输出）实测：

| 并发 | 旧 Fast tok/s | 修复 Fast tok/s | Qwen3 tok/s | Fast 加速 | 修复 Fast 比 Qwen 低 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.60 | 154.25 | 163.10 | 2.014× | 5.42% |
| 2 | 142.99 | 253.09 | 264.08 | 1.770× | 4.16% |

每个点为一次完整请求测量，包含 prefill 与调度；共享节点、固定形状合成输入，属于诊断结果。36项配置/分派回归通过；48个固定前缀位置的平均/最大KL为0.000841/0.006718，Top-1相同47/48，完整W2的B1生成序列发生变化，Fast不保证bitwise。此轮没有修改llm-train或共享Align kernel。

[完整报告与图表](docs/yoco/fast-low-concurrency-20260908/REPORT.md) · [低并发W2持续表](docs/yoco/performance/LOW_CONCURRENCY.md)。W2组独立记录；随后完成的Mooncake复测见[开源trace报告](docs/yoco/performance/fast-mooncake-20260908/REPORT.md)，两组工作负载不混用。

### Align backward 越界修复（2026-09-08）

训练侧已修复 `_route_map` 的CUDA非法访问：旧padding计数在Triton3.7.1会把136条路由计为1024条；现在给histogram显式传入有效位置mask。两种Triton版本各通过84项相关回归与5项概率/CE检查，显存检查49项通过，NNScaler各完成12步SGD。vLLM前向实现未改，性能表沿用原测量。[报告与失败/修复证据](docs/yoco/align-backward-fix-20260908/REPORT.md)。

### Align GEMM B200 联动验证（2026-09-06）

本地开发为 vLLM / llm-train 增加共享 MoE launch 配置，固定 K=32、split-K=1 和原 BF16 舍入边界。B200 上725项整模型字节比较通过，最大差0，覆盖最长8192-token、B1–256 decode、ragged prefill、带梯度训练的完整 logits/log-prob/CE，以及缓存/chunked/mixed 和实际16K/32K行合批。结论限于已测配置与输入。

同卡 Mooncake FAST’25 开源 trace（600秒、3643请求）中，输出吞吐580.56→613.87 tok/s（+5.74%），ITL P95降低38.77%，E2E P95降低6.51%。两端都只有3638/3643成功，均有5个超时并触及512并发上限；这是过载诊断，不是容量验证通过。512/2048-token训练整步吞吐变化为−0.50%/−0.15%，基本持平。

实验默认关闭。两端用 `VLLM_YOCO_ALIGN_MOE_CONFIG` 指向同一份已验证的 B200 profile；文件在 `vllm/model_executor/layers/fused_moe/experts/yoco_configs/align_moe_NVIDIA_B200.experimental.json`，加载器检查GPU和软件版本。

- [B200 完整报告、开源 trace 与训练结果](docs/yoco/align-gemm-b200-20260906/REPORT.md)
- [逐字节验收范围与中止记录](docs/yoco/align-gemm-b200-20260906/VALIDATION.md)
- [后续小批次优化与 1P1D 进度](docs/yoco/align-1p1d-b200-20260906/REPORT.md)：新增 2/4/16 行配置，真实专家入口在分散路由下加速 6.4%–12.0%，完整模型输出字节检查通过；该阶段的后续实测见下方四卡报告。

### Align 四卡 1P1D 联动实测（2026-09-07）

四卡B200 Job已完成实测。真实1P1D输出字节105项、与llm-train单条/packed的前向和CE字节72项，原始Align与GEMM候选均通过；同卡1P1D输出吞吐631.87→826.76 tok/s（+30.84%）。ITL P95 55.25→59.82 ms。同GPU单实例对照有5/3643条600秒超时。当前1×trace为共享节点过载诊断，非容量验收；当前P跳过前缀缓存读取，收尾控制请求单独记在计时之外。

[报告与原始汇总](docs/yoco/align-4gpu-1p1d-20260907/REPORT.md) · [PDF](docs/yoco/align-4gpu-1p1d-20260907/REPORT.pdf)。

### Fast Mooncake 1.2× 负载复测（2026-09-08 UTC）

保持同一Fast实现和物理B200，将相同3643请求的到达时间从约600秒压缩到500秒。以下为**负载响应**，实现与kernel未修改；1.2×结果独立维护，Qwen3和Align尚未测量该速率。

| 拓扑 | 1× 输出tok/s | 1.2× 输出tok/s | 吞吐变化 | 1.2× TTFT / ITL / E2E P95 ms |
| --- | ---: | ---: | ---: | --- |
| 单卡 | 1038.83 | 1190.07 | +14.56% | 10222.08 / 338.66 / 98452.09 |
| 1P1D | 1046.78 | 1244.77 | +18.91% | 5274.00 / 24.33 / 16094.88 |

单卡 3642/3643完成，1错误，client/server/drain=FAIL/PASS/PASS；1P1D 3643/3643完成，0错误，client/server/drain=PASS/PASS/PASS。单次、共享节点、500秒到达、未声明SLO，属于diagnostic，不能当作容量验收。

[完整报告与图表](docs/yoco/performance/fast-mooncake-f1p2-20260908/REPORT.md) · [1.2×持续表](docs/yoco/performance/THROUGHPUT_F1P2.md)。

### Fast Mooncake 开源 trace 复测（2026-09-08 UTC）

只重测恢复split-KV后的Fast，沿用2026-09-07的同物理B200和完整参数：Mooncake FAST’25 toolagent源时间300–900秒、1×、3643请求；单卡GPU5，1P1D为GPU4/5。Qwen3、Align和W2表保留原测量时间及数值。

| 拓扑 | 旧Fast输出 tok/s | 当前Fast输出 tok/s | 变化 |
| --- | ---: | ---: | ---: |
| 单卡 | 1037.27 | 1038.83 | +0.15% |
| 1P1D | 1045.68 | 1046.78 | +0.10% |

TTFT / ITL / E2E P95变化：单卡 +0.59% / -4.14% / -2.29%；1P1D +66.61% / +0.02% / +9.12%。

两组均3643/3643完成，零错误，客户端/长度/服务/排空门槛通过。这是共享节点、单次测量、无预设SLO的诊断；固定1×输出负载约1072 tok/s，不能据此推断峰值能力或期待W2的加速比例。单卡/P-D四组探针输出token相同，但所选token log-prob最大差0.1599，差异待定位，Fast不保证bitwise。

[完整报告与证据](docs/yoco/performance/fast-mooncake-20260908/REPORT.md) · [三模式持续表](docs/yoco/performance/THROUGHPUT.md)。

### Fast decode 优化（2026-09-06）

在上一轮大 Prefill 优化之上，本轮增加了小 M Triton W13/W2 调参，以及 B200 上预调优的 CUTLASS decode 配置。Triton 只匹配精确测量尺寸；CUTLASS 新路径仅用于 M=64/128/256 的完整、每请求一个 token 的 CUDA Graph。多 token prefill 和混合图保留原 backend 策略。

同一物理 B200、L3 BF16、TP1，128-token 输入、64-token 输出，A/B/B/A，每版每个负载共 6 次，取中位数：

| Batch | 基线 TPOT ms | 优化后 TPOT ms | TPOT 降低 | 端到端吞吐提高 |
| --- | --- | --- | --- | --- |
| 1 | 6.086 | 6.041 | 0.75% | 0.86% |
| 8 | 9.468 | 9.284 | 1.94% | 1.61% |
| 32 | 13.348 | 13.041 | 2.31% | 1.84% |
| 64 | 14.822 | 13.538 | 8.66% | 7.49% |
| 128 | 18.791 | 16.897 | 10.08% | 8.48% |
| 256 | 25.302 | 23.509 | 7.09% | 5.99% |

端到端吞吐含 Prefill 与调度时间。长输入、Prefill 的独立测量及两轮变化见专项报告。

公开 Mooncake FAST’25 trace 的同卡 600 秒回放：完成 3643/3643 与 3643/3643 请求，输出吞吐 1031.65 → 1037.25 tok/s（+0.54%）。TTFT / ITL / E2E P95 分别变化 +0.08% / +6.43% / -3.66%。共享节点、单次前后长测、未声明延迟 SLO，属于 diagnostic，不代表稳定容量或模型质量。

75 项相关回归与 16 项强 clamp 检查通过。固定 token 前缀比较中，本次抽样 top-1 全部一致；部分大 batch 的 logits 及重复结果仍非 bitwise。数值差异和被拒绝的早期候选保留在报告中。

### 启用条件

现有 `--fast` 命令在 B200、L3 BF16、无量化、TP/DP/PP/CP 均为 1、standalone、启用 `--kv-sharing-fast-prefill`，且满足原 FlashInfer 选择条件时自动应用。新增 CUTLASS decode 还要求 graph capture 上限不超过 256，且官方 cache 加载器接受 GPU/CUDA/library 元数据；不匹配时回退。

本次验证的缓存环境：FlashInfer 0.6.8.post1、CUDA 13.1、cuBLAS 13.2.1、cuDNN 91900。启动时不在线 autotune，不增加专家权重副本。`M` 为实际 MoE token 行数，可能包含图 padding，不等于实际请求数。

关闭本轮 CUTLASS decode 选择，保留 Triton 调参与已有 Prefill 策略：

```bash
--additional-config '{"yoco_fast_decode_cutlass": false}'
```

原 `yoco_fast_standalone_flashinfer_moe=false` 开关仍可关闭 standalone FlashInfer 自动选择。

### 报告与历史结果

- [本轮 Fast decode 报告、验证与限制](docs/yoco/fast-decode-optimization-20260906/REPORT.md)
- [本轮报告 PDF](docs/yoco/fast-decode-optimization-20260906/REPORT.pdf)
- [本轮短测数据](docs/yoco/fast-decode-optimization-20260906/short-comparison.json)与[公开 trace 对照](docs/yoco/fast-decode-optimization-20260906/trace-comparison.json)
- [2026-09-05 Prefill 优化报告](docs/yoco/fast-optimization-20260905/REPORT.md)：当轮 Prefill 提升 5.3%–8.9%，Decode 基本持平
- [2026-09-05 综合开发报告 PDF](docs/yoco/YOCO-Align-Fast-Report-20260905.pdf)：Align、概率归约、训练反向与上一轮 Fast 优化
- [2026-09-05 README PDF 快照](docs/yoco/README-snapshot-20260905.pdf)

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
