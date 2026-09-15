# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO Fast online-FP8 defaults and per-layer precision dispatch."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.test_yoco_config import _make_vllm_config
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.models import yoco
from vllm.model_executor.models.config import YOCOForCausalLMConfig


@pytest.mark.parametrize("projection", ["fc1_latent_proj", "fc2_latent_proj"])
@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize(
    "preset,expected_method",
    [
        ("fp8_per_block", "Fp8PerBlockOnlineLinearMethod"),
        ("fp8_per_tensor", "Fp8PerTensorOnlineLinearMethod"),
        ("mxfp8", "Mxfp8OnlineLinearMethod"),
        ("int8_per_channel_weight_only", "UnquantizedLinearMethod"),
    ],
)
def test_latent_linear_precision_and_explicit_ignore(
    monkeypatch, projection, ignored, preset, expected_method
):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.online import fp8

    monkeypatch.setattr(
        fp8,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    args = deepcopy(
        resolve_quantization_config(preset, {} if preset == "mxfp8" else None)
    )
    assert args is not None
    prefix = f"model.layers.0.mlp.{projection}"
    if ignored:
        args.ignore.append(prefix)
    quant = yoco._maybe_build_yoco_quant_config(OnlineQuantizationConfig(args))
    layer = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(layer)
    method = quant.get_quant_method(layer, prefix)
    assert type(method).__name__ == (
        "UnquantizedLinearMethod" if ignored else expected_method
    )


@pytest.mark.parametrize(
    "quantization,overrides,expected",
    [
        (None, None, "triton"),
        ("fp8_per_block", None, "auto"),
        ("fp8_per_tensor", None, "triton"),
        ("mxfp8", None, "triton"),
        ("online", {"moe": {"weight": "fp8_per_block_static"}}, "auto"),
        (None, {"linear": {"weight": "fp8_per_block_static"}}, "triton"),
        ("fp8_per_block", {"moe": {"weight": "fp8_per_tensor_static"}}, "triton"),
    ],
)
@pytest.mark.parametrize("explicit_backend", [None, "triton", "deep_gemm"])
def test_fp8_backend_defaults(quantization, overrides, expected, explicit_backend):
    runtime = _make_vllm_config(
        cudagraph_mode=None, moe_backend=explicit_backend or "auto"
    )
    # This hook runs before runtime.quant_config exists. Exercise the same
    # resolved args as EngineArgs as well as the CLI shorthand itself.
    runtime.model_config = SimpleNamespace(
        quantization=quantization, quantization_config=overrides
    )
    YOCOForCausalLMConfig.verify_and_update_config(runtime)
    assert runtime.kernel_config.moe_backend == (explicit_backend or expected)


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("ignored", [False, True])
def test_fp8_backend_respects_expert_precision(
    monkeypatch, supported, tp_size, ignored
):
    import vllm.utils.deep_gemm as dg

    args = resolve_quantization_config("fp8_per_block", None)
    assert args is not None
    # Do not mutate the global shorthand's ignore list.
    quant = yoco._maybe_build_yoco_quant_config(
        OnlineQuantizationConfig(deepcopy(args))
    )
    if ignored:
        quant.ignored_layers.append(r"re:.*\.layers\.0\.mlp\.experts$")
    monkeypatch.setattr(dg, "is_deep_gemm_supported", lambda: supported)
    backend = yoco._select_yoco_online_fp8_moe_backend(
        quant, "model.layers.0.mlp.experts", tp_size
    )
    assert backend == (
        "deep_gemm" if supported and tp_size == 1 and not ignored else "triton"
    )
    # Ignore rules are local to the expert, not a blanket model-wide gate.
    backend = yoco._select_yoco_online_fp8_moe_backend(
        quant, "model.layers.1.mlp.experts", tp_size
    )
    assert backend == ("deep_gemm" if supported and tp_size == 1 else "triton")


@pytest.mark.parametrize("quantized_head", [False, True])
def test_fp8_model_keeps_only_unquantized_fast_lm_head(
    monkeypatch, quantized_head, default_vllm_config
):
    class DummyModel(torch.nn.Module):
        execution_mode = "fast"
        make_empty_intermediate_tensors = None

        def __init__(self, **kwargs):
            super().__init__()

    real_head = yoco.ParallelLMHead

    def make_head(*args, **kwargs):
        # Use the actual embedding type to exercise online quant dispatch,
        # without allocating a full L3 head or a distributed process group.
        head = real_head.__new__(real_head)
        torch.nn.Module.__init__(head)
        method = kwargs["quant_config"].get_quant_method(head, kwargs["prefix"])
        assert method is None
        head.quant_method = (
            object() if quantized_head else yoco.UnquantizedEmbeddingMethod()
        )
        return head

    monkeypatch.setattr(yoco, "YOCOModel", DummyModel)
    monkeypatch.setattr(yoco, "ParallelLMHead", make_head)
    monkeypatch.setattr(yoco, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(yoco, "_supports_yoco_sm100_lm_head_kernel", lambda mode: True)
    args = resolve_quantization_config("fp8_per_block", None)
    assert args is not None
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=3072, vocab_size=154880)
        ),
        quant_config=OnlineQuantizationConfig(deepcopy(args)),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
    )
    model = yoco.YOCOForCausalLM(vllm_config=runtime)
    assert model.use_sm100_lm_head_kernel == (not quantized_head)


@pytest.mark.parametrize(
    "quantization,overrides,expected",
    [
        (None, None, []),
        ("fp8_per_tensor", None, []),
        ("mxfp8", None, []),
        ("fp8_per_block", None, ["+quant_fp8"]),
        ("online", {"linear": {"weight": "fp8_per_block_static"}}, ["+quant_fp8"]),
        ("online", {"moe": {"weight": "fp8_per_block_static"}}, []),
    ],
)
def test_online_fp8_keeps_platform_quant_in_compiled_model(
    quantization, overrides, expected
):
    runtime = _make_vllm_config(cudagraph_mode=None)
    runtime.model_config = SimpleNamespace(
        quantization=quantization, quantization_config=overrides, enforce_eager=False
    )
    # Check both shorthand and resolved frontend arguments.
    for args in (overrides, resolve_quantization_config(quantization, overrides)):
        runtime.model_config.quantization_config = args
        YOCOForCausalLMConfig.verify_and_update_config(runtime)
        assert runtime.compilation_config.custom_ops == expected


@pytest.mark.parametrize("custom_ops", [["all"], ["none", "+quant_fp8"]])
def test_online_fp8_preserves_explicit_enabled_quant(custom_ops):
    runtime = _make_vllm_config(cudagraph_mode=None)
    runtime.model_config = SimpleNamespace(quantization="fp8_per_block")
    runtime.compilation_config.custom_ops = list(custom_ops)
    YOCOForCausalLMConfig.verify_and_update_config(runtime)
    assert runtime.compilation_config.custom_ops == custom_ops


def test_online_fp8_rejects_disabled_quant_only_for_compiled_model():
    runtime = _make_vllm_config(cudagraph_mode=None)
    runtime.model_config = SimpleNamespace(
        quantization="fp8_per_block", enforce_eager=False
    )
    runtime.compilation_config.custom_ops = ["-quant_fp8"]
    with pytest.raises(ValueError, match="requires quant_fp8"):
        YOCOForCausalLMConfig.verify_and_update_config(runtime)
    runtime.model_config.enforce_eager = True
    YOCOForCausalLMConfig.verify_and_update_config(runtime)
    assert runtime.compilation_config.custom_ops == ["-quant_fp8"]
