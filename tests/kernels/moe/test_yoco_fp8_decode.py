# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO's small FP8 decode path must consume the final weight scales."""

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.yoco import YocoMoEPolicy
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    transform_sf_into_required_layout,
)

pytestmark = pytest.mark.skipif(
    not is_deep_gemm_e8m0_used() or not current_platform.is_device_capability(100),
    reason="requires SM100 DeepGEMM UE8M0",
)


@pytest.fixture(scope="module")
def scales():
    torch.manual_seed(9217)
    s1 = torch.exp2(torch.randint(-20, 10, (128, 60, 8), device="cuda").float())
    s2 = torch.exp2(torch.randint(-20, 10, (128, 8, 30), device="cuda").float())
    packed = [
        transform_sf_into_required_layout(
            sf=sf, mn=mn, k=k, recipe=(1, 128, 128), num_groups=128, is_sfa=False
        )
        for sf, mn, k in [(s1, 7680, 1024), (s2, 1024, 3840)]
    ]
    return s1, s2, *packed


def make_experts(scales):
    _, _, s1, s2 = scales
    config = make_dummy_moe_config(
        num_experts=128, experts_per_token=8, hidden_dim=1024, intermediate_size=3840
    )
    config.yoco = YocoMoEPolicy.for_mode("fast")
    config.apply_router_weight_before_w2 = True
    quant = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=[128, 128])
    return TritonOrDeepGemmExperts(config, quant)


def test_decode_scale_cache_and_dispatch(scales):
    s1, s2, packed1, packed2 = scales
    experts = make_experts(scales)
    assert experts.configure_yoco_fp8_decode()
    fallback = experts.fallback_experts
    assert fallback.moe_config.yoco.fp8_decode_aligned
    assert fallback.moe_config.yoco.direct_fp8_activation
    assert experts.experts.moe_config.yoco.direct_fp8_activation
    assert torch.equal(fallback.w1_scale, s1)
    assert torch.equal(fallback.w2_scale, s2)
    assert experts.experts.w1_scale is packed1
    assert experts.experts.w2_scale is packed2
    w1 = torch.empty(128, 7680, 1024, dtype=torch.float8_e4m3fn, device="meta")
    w2 = torch.empty(128, 1024, 3840, dtype=torch.float8_e4m3fn, device="meta")
    for m in [1, 2, 4, 8, 16, 17, 32, 64, 128]:
        x = torch.empty(m, 1024, dtype=torch.float8_e4m3fn, device="meta")
        selected = experts._select_experts_impl(x, w1, w2)
        assert selected is (fallback if m <= 16 else experts.experts)
    assert not experts.configure_yoco_fp8_decode(max_tokens=0)
    assert experts._select_experts_impl(x[:1], w1, w2) is experts.experts


@pytest.mark.parametrize(
    "feature", ["tp_size", "dp_size", "ep_size", "enable_eplb", "in_dtype"]
)
def test_decode_cache_rejects_unvalidated_parallel_modes(scales, feature):
    experts = make_experts(scales)
    if feature == "in_dtype":
        experts.moe_config.in_dtype = torch.float16
    else:
        setattr(
            experts.moe_config.moe_parallel_config,
            feature,
            True if feature == "enable_eplb" else 2,
        )
    assert not experts.configure_yoco_fp8_decode()
    assert experts._yoco_fp8_decode_limit == 0


def test_decode_cache_refreshes_after_scale_change(scales):
    s1, s2, packed1, packed2 = scales
    experts = make_experts(scales)
    assert experts.configure_yoco_fp8_decode()
    old = experts.fallback_experts
    assert old.w1_scale is not None
    # A pre-existing graph must observe refreshed scales, not stale storage.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            old.w1_scale.sum()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_sum = old.w1_scale.sum()
    # Each packed byte gains one exponent. A reload gets fresh plain scales.
    experts.quant_config._w1.scale = packed1 + 0x01010101
    experts.quant_config._w2.scale = packed2 + 0x01010101
    assert experts.configure_yoco_fp8_decode()
    assert experts.fallback_experts is not old
    assert experts.fallback_experts.w1_scale is old.w1_scale
    assert experts.fallback_experts.w2_scale is old.w2_scale
    assert torch.equal(experts.fallback_experts.w1_scale, 2 * s1)
    assert torch.equal(experts.fallback_experts.w2_scale, 2 * s2)
    graph.replay()
    assert torch.equal(captured_sum, (2 * s1).sum())
    replacement = make_experts(scales)
    assert replacement.configure_yoco_fp8_decode(
        cached_scales=experts._yoco_fp8_scale_cache
    )
    assert replacement.fallback_experts.w1_scale is old.w1_scale
    graph.replay()
    assert torch.equal(captured_sum, s1.sum())
