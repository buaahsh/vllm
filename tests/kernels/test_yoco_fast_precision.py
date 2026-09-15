# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert FP8 fusion and graph-safe BF16 cache refresh."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.activation import SiluAndMulWithClampFP32
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.model_executor.layers.yoco_fast import refresh_yoco_weight_cache
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, per_block_cast_to_fp8
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(100) or not is_deep_gemm_e8m0_used(),
    reason="requires B200 DeepGEMM UE8M0",
)


@pytest.fixture(scope="module")
def dense_down():
    runtime = VllmConfig()
    runtime.model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="yoco")
    )
    with set_current_vllm_config(runtime):
        config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8Static128BlockSym,
            activation_quant_key=kFp8Dynamic128Sym,
            weight_shape=(3072, 1280),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        kernel = DeepGemmFp8BlockScaledMMKernel(config)
        torch.manual_seed(9220)
        weight = torch.randn(3072, 1280, device="cuda", dtype=torch.bfloat16) / 32
        q, scale = per_block_cast_to_fp8(weight, [128, 128], use_ue8m0=True)
        layer = torch.nn.Module()
        layer.weight_block_size = [128, 128]
        layer.register_parameter("weight", torch.nn.Parameter(q, requires_grad=False))
        layer.register_parameter(
            "weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
        )
        layer.input_scale = None
        kernel.process_weights_after_loading(layer)
        method = Fp8PerBlockOnlineLinearMethod.__new__(Fp8PerBlockOnlineLinearMethod)
        method.input_dtype = method.out_dtype = torch.bfloat16
        method.weight_block_size = [128, 128]
        method.fp8_linear = kernel
        assert method.supports_silu_mul_fusion()
        act = SiluAndMulWithClampFP32(10.0, enforce_enable=True)
        yield layer, method, act


def assert_small_error(actual, expected):
    assert bool(torch.isfinite(actual).all())
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative.item() <= 1e-3, relative.item()


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16, 17, 128, 513])
@pytest.mark.parametrize("magnitude", [1.0, 16.0])
def test_shared_swiglu_quantization_and_dense_gemm(dense_down, rows, magnitude):
    layer, method, act = dense_down
    torch.manual_seed(9221 + rows)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * magnitude
    expected = method.apply(layer, act(x))
    actual = method.apply_silu_mul(layer, x, 10.0)
    assert_small_error(actual, expected)
    # Fake tensor metadata includes the noncontiguous, TMA-aligned scale stride.
    torch.library.opcheck(
        torch.ops.vllm.silu_mul_quant_fp8_packed.default,
        (x, 10.0),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_fused_dense_graph_replay_and_weight_update(dense_down):
    layer, method, act = dense_down
    x = torch.randn(8, 2560, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        method.apply_silu_mul(layer, x, 10.0)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = method.apply_silu_mul(layer, x, 10.0)
    original_weight = layer.weight.clone()
    try:
        for multiplier in [1, -1]:
            layer.weight.copy_(
                (original_weight.float() * multiplier).to(original_weight.dtype)
            )
            x.copy_(torch.randn_like(x) * 16)
            graph.replay()
            assert_small_error(actual, method.apply(layer, act(x)))
    finally:
        layer.weight.copy_(original_weight)


def test_common_cache_refresh_updates_existing_bf16_graph():
    weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    cached = refresh_yoco_weight_cache(None, weight.T.contiguous())
    x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        torch.mm(x, cached)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.mm(x, cached)
    weight.copy_(torch.randn_like(weight))
    assert refresh_yoco_weight_cache(cached, weight.T.contiguous()) is cached
    graph.replay()
    assert torch.equal(actual, torch.mm(x, weight.T.contiguous()))


@pytest.mark.parametrize("reverse_load", [False, True])
@torch.inference_mode()
def test_merged_fp8_kv_load_preserves_block_scales_and_outputs(
    dense_down, reverse_load, monkeypatch
):
    from tests.model_executor.test_yoco_fast_precision import quant_config
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
    )

    # LinearBase's disable_tp avoids communication; parameters still query rank.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    config = quant_config("fp8_per_block")
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        merged = MergedColumnParallelLinear(
            3072,
            [512, 512],
            bias=False,
            quant_config=config,
            prefix="model.yoco_kv_proj",
            disable_tp=True,
        )
        separate = [
            ColumnParallelLinear(
                3072,
                512,
                bias=False,
                quant_config=config,
                prefix=f"model.yoco_{kind}_proj",
                disable_tp=True,
            )
            for kind in ["k", "v"]
        ]
        weights = [torch.randn(512, 3072) / 32, torch.randn(512, 3072) * 8]
        for shard in [1, 0] if reverse_load else [0, 1]:
            merged.weight.weight_loader(merged.weight, weights[shard], shard)
            separate[shard].weight.weight_loader(separate[shard].weight, weights[shard])
        # Deliberately different K/V magnitudes expose accidental scale sharing.
        for shard in [0, 1]:
            assert torch.equal(
                merged.weight[shard * 512 : (shard + 1) * 512].view(torch.uint8),
                separate[shard].weight.view(torch.uint8),
            )
            assert torch.equal(
                merged.weight_scale_inv[shard * 512 : (shard + 1) * 512],
                separate[shard].weight_scale_inv,
            )
        for rows in [1, 8, 128]:
            x = torch.randn(rows, 3072)
            expected = torch.cat([layer(x)[0] for layer in separate], dim=-1)
            assert_small_error(merged(x)[0], expected)


@pytest.mark.parametrize("precision", ["bf16", "fp8", "moe_only"])
@torch.inference_mode()
def test_shared_expert_uses_component_precision(dense_down, monkeypatch, precision):
    from functools import partial

    from tests.model_executor.test_yoco_fast_precision import quant_config
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models import yoco

    monkeypatch.setattr(yoco, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        yoco,
        "MergedColumnParallelLinear",
        partial(MergedColumnParallelLinear, disable_tp=True),
    )
    monkeypatch.setattr(
        yoco, "RowParallelLinear", partial(RowParallelLinear, disable_tp=True)
    )
    config = quant_config(
        {"bf16": None, "fp8": "fp8_per_block", "moe_only": "online"}[precision],
        overrides={"moe": {"weight": "fp8_per_block_static"}}
        if precision == "moe_only"
        else None,
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        shared = yoco.YOCOSharedExperts(
            3072,
            1280,
            config,
            False,
            "model.layers.0.mlp.shared_experts",
            execution_mode="fast",
        )
        for layer, shape in [
            (shared.gate_up_proj, (2560, 3072)),
            (shared.down_proj, (3072, 1280)),
        ]:
            layer.weight.weight_loader(layer.weight, torch.randn(*shape) / 32)
        shared.initialize_fast_weight_cache()
        assert shared.use_fast_fp8_swiglu == (precision == "fp8")
        assert shared.use_fast_down_transpose == (precision != "fp8")
        for rows in [1, 8]:
            x = torch.randn(rows, 3072)
            expected = shared.down_proj(shared.act_fn(shared.gate_up_proj(x)[0]))[0]
            assert_small_error(shared(x), expected)
