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

## YOCO PD 分离开发版本

当前开发分支 `shaohanh/yoco-serving-final-20260730-dev` 基于
`origin/shaohanh/yoco-serving-final-20260730`，并已合入
`origin/prefill-cut-7-31`（集成点 `0522e44b11`）。这个 fork 面向 YOCO-v2/v3
在 NVIDIA B200 上的 Prefill/Decode（PD）分离，不是通用 vLLM 发布分支。

更完整的历史配置、模型路径和早期 B200 A/B 数据见 [YOCO 验收报告](yoco.md)。

### 这个版本做了什么

1. **裁掉专用 Prefill 节点不需要的 YOCO cross-layer 计算。** YOCO 的 self
   block 直接把共享 K/V 写入第一个 KV-owner cross layer 的 cache。对于只负责
   传 KV 的请求，十个 cross layers 全部跳过；Decode 路径不变。
2. **把专用 Producer 的 KV-only 模式静态化。** 当服务明确配置
   `kv_role=kv_producer` 时，每个 DP rank 都固定走 KV-only 路径，不再逐 step
   检查请求 metadata，也不再执行额外的 GPU-to-CPU `.item()` 同步。DP 的空闲
   rank 仍执行带相同 KV-only 标记的 dummy forward，避免 MoE collective 次数
   不一致导致 hang。
3. **修复 YOCO 与 NIXL 的 KV cache alias。** YOCO 的多个 cross layers 会指向
   同一块物理 KV cache；NIXL worker 现在会正确处理这些 alias，避免按 layer
   名查原始 `KVCacheConfig` 时触发 `KeyError`。
4. **保留流式 PD 的 KV transfer metadata。** OpenAI-compatible completion 和
   chat completion 的 SSE 最终 chunk 会携带 `kv_transfer_params`，与非流式响应
   一致。
5. **正确处理前端发现的多 token stop string。** 这类请求不再被当作 abort；
   EngineCore 会以正常 `STOP` 完成请求，让 KV connector 有机会释放资源并把
   transfer metadata 放进最终输出。同步、异步、MP、DP load-balancing 路径均
   使用相同语义。
6. **提供单一 UCX 的 B200 PD 镜像。** [Dockerfile.b200.pd](docker/Dockerfile.b200.pd)
   在 `/opt/hpcx/ucx` 构建唯一的 UCX 1.21.0，并针对它构建 NIXL 1.3.2。镜像会
   隐藏不可用的 `nixl_ep` shim，避免 FusedMoE 导入时误选 `nixl_ep_cu13`。
   [verify_single_ucx.sh](docker/verify_single_ucx.sh) 会检查重复 UCX、NIXL plugin
   的 RUNPATH/动态链接，以及可选目标进程实际加载的 UCX。

### PD 请求契约

专用 Producer 必须满足以下条件：

- YOCO 模型启用 `--kv-sharing-fast-prefill`；
- P 服务使用 `NixlConnector` 和 `kv_role=kv_producer`；
- P 请求使用 `max_tokens=1`，并设置
  `kv_transfer_params.do_remote_decode=true`；
- Gateway 丢弃 P response 的整个 `choices`，只把原始 prompt 和 P 返回的
  `kv_transfer_params` 交给 `kv_consumer`；
- D 负责 first token 和之后所有用户可见 token。

示意配置：

```text
P: --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
D: --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer",...}'
```

`kv_producer` 是专用角色，不能混入普通生成或 Decode 请求；这类误路由不会动态
回退，返回的 logits 不保证正确。兼容角色 `kv_both` 在 DP1 仍保留逐请求检查，
DP>1 则保守回退，不应替代生产环境的专用 P 服务。

### B200 合并后验证（2026-08-03）

测试使用同一台机器上的 4 张 B200：静态 `kv_producer`、`kv_consumer`、
standalone reference 和动态 `kv_both` baseline 各占一张卡。运行时为单一
UCX 1.21.0 + NIXL 1.3.2。

- `1,356 / 12,096 / 43,704` prompt tokens 的 P/D 输出均与 standalone
  reference 逐字一致；三轮 D -> P -> D 复用也全部一致。
- 流式 completion 的普通停止和跨 token stop-string 均逐字一致，最终 SSE
  chunk 均包含 `kv_transfer_params`。
- 静态 Producer 相对动态 baseline 的五轮中位 prompt throughput 提升为
  `+0.52%` 到 `+2.24%`，不同 prompt/concurrency shape 均未观察到回退。
- 完整 P -> D 相对 standalone：c1 因两次 HTTP/握手开销吞吐低 `14%` 到
  `21%`；c4 吞吐 `+7.87%`、中位延迟 `-12.10%`；c8 吞吐 `+2.85%`、中位
  延迟 `-5.15%`。
- 四个服务日志中未发现 UCX/NIXL transfer error、HTTP 5xx 或 EngineCore
  error。

原始 JSON 和日志位于共享 PVC：

```text
/mnt/pvc/lidong1/vllm_pd/merged-0803-gpu-validation/
```

### YOCO Router / RMSNorm 优化（2026-08-03）

开发分支 `yoco-router-fusion-0803` 在上述 PD 版本上增加两项保持数值语义的
YOCO 专用优化：

- Router 直接返回 `topk_weights/topk_ids`，删除 dense `routing_probs`、
  `routing_map`、scatter 和第二次 `topk`；softmax、`torch.topk` 和 top-k
  renormalization 的既有计算顺序保持不变。
- 每个 decoder layer 把 attention residual add 和 post-attention RMSNorm 合并为
  一个 Triton kernel。完整 FP32 hidden state 仍在每层结束时物化；不跨层传递
  拆分的 MLP output/residual，以保持原编译图的长上下文数值行为。

B200 验收结果：

- CUDA/模型单测 `20 passed`；fused add-RMSNorm 的输出和 FP32 residual 均与
  顺序实现 bitwise 一致。
- `1,356 / 12,096 / 43,704` prompt tokens 三档的生成文本、token 序列、token
  logprob 和 top-5 logprob 均与 `4ea08bc6d4` baseline 逐项一致。
- fused add-RMSNorm 算子微基准提升约 `1.23x` 到 `1.35x`。standalone endpoint
  的交替三轮中位 prompt throughput 变化为：c1 短 prompt `+4.26%`、c1 12K
  `+0.84%`、c1 43K `+1.79%`、12K c4 `-1.19%`、12K c8 `+1.06%`；整体收益
  较小，主要价值是减少 layer 内的小 kernel launch。

实验过的 Router 单-kernel 版本没有纳入分支：虽然 Router 微基准约快 `2.1x`，
但 top-k weight 最大误差约 `8.94e-8`，并会放大成长上下文 logprob 差异。除非能
恢复 bitwise 计算顺序，否则不应替换当前 Router 路径。

### 构建和当前状态

```bash
docker build -f docker/Dockerfile.b200.pd \
  -t vllm-yoco-pd:ucx121-local .

# 容器启动后检查镜像；传 PID 可继续检查进程实际加载的动态库。
verify-single-ucx [pid]
```

`Dockerfile.b200.pd` 已通过 `docker build --check`。上述 GPU 测试是在 Pod 内
统一 UCX/NIXL 后完成；加入 `nixl_ep` 防护后的镜像仍需实际重建和做一次镜像级
短回归后再发布，不能把当前验证结果视为新镜像已验收。

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
