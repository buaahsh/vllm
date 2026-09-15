# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real latent shapes: online FP8 loading, GEMM operands, and graph replay."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.quantization import resolve_quantization_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.models.yoco import _maybe_build_yoco_quant_config
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(100) or not is_deep_gemm_e8m0_used(),
    reason="requires B200 DeepGEMM UE8M0",
)


@pytest.fixture(scope="module", autouse=True)
def single_rank_parallel_group(tmp_path_factory):
    store = tmp_path_factory.mktemp("latent-fp8-tp") / "init"
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{store}",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            yield
        finally:
            cleanup_dist_env_and_memory()


def _quantized_reference(x, weight):
    groups = x.float().reshape(x.shape[0], -1, 128)
    a_scale = torch.exp2(
        torch.ceil(
            torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    a = (groups / a_scale).to(torch.float8_e4m3fn).float() * a_scale
    n, k = weight.shape
    blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
    b_scale = torch.exp2(
        torch.ceil(
            torch.log2(blocks.abs().amax((1, 3), keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    b = (blocks / b_scale).to(torch.float8_e4m3fn).float() * b_scale
    return F.linear(a.reshape_as(x).double(), b.reshape_as(weight).double()).to(
        torch.bfloat16
    )


def _assert_gemm_matches(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative.item() < 1e-3, relative.item()


@pytest.fixture(
    params=[(3072, 1024, "fc1_latent_proj"), (1024, 3072, "fc2_latent_proj")]
)
def latent_linear(request):
    k, n, name = request.param
    runtime = VllmConfig()
    runtime.model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="yoco")
    )
    runtime.compilation_config.custom_ops = ["all"]
    args = deepcopy(resolve_quantization_config("fp8_per_block", None))
    assert args is not None
    quant = _maybe_build_yoco_quant_config(OnlineQuantizationConfig(args))
    with (
        torch.inference_mode(),
        set_current_vllm_config(runtime),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        torch.manual_seed(918)
        layer = ReplicatedLinear(
            k,
            n,
            bias=False,
            quant_config=quant,
            prefix=f"model.layers.0.mlp.{name}",
            return_bias=False,
        )
        assert isinstance(layer.quant_method, Fp8PerBlockOnlineLinearMethod)
        weight = torch.randn(n, k) / k**0.5
        layer.weight.weight_loader(layer.weight, weight)
        assert layer.weight.dtype == torch.float8_e4m3fn
        yield layer, weight


@pytest.mark.parametrize("rows", [1, 8, 32, 129])
@torch.inference_mode()
def test_latent_fp8_operands_and_numerics(latent_linear, monkeypatch, rows):
    layer, weight = latent_linear
    kernel = layer.quant_method.fp8_linear
    original = kernel.apply_block_scaled_mm
    calls = []

    def checked(A, B, As, Bs):
        calls.append((A.dtype, B.dtype))
        return original(A, B, As, Bs)

    monkeypatch.setattr(kernel, "apply_block_scaled_mm", checked)
    x = torch.randn(rows, weight.shape[1], device="cuda", dtype=torch.bfloat16)
    actual = layer(x)
    expected = _quantized_reference(x, weight)
    _assert_gemm_matches(actual, expected)
    assert calls == [(torch.float8_e4m3fn, torch.float8_e4m3fn)]


@torch.inference_mode()
def test_latent_fp8_graph_replays_new_inputs(latent_linear):
    layer, weight = latent_linear
    x = torch.randn(8, weight.shape[1], device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        layer(x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(x)
    for magnitude in (0.001, 1.0, 16.0):
        x.copy_(torch.randn_like(x) * magnitude)
        graph.replay()
        _assert_gemm_matches(actual, _quantized_reference(x, weight))
