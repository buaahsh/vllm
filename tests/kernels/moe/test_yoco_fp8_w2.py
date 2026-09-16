# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate FP8 W2 tiles with both route layouts and CUDA Graph replay."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
    try_get_yoco_fp8_w2_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(100), reason="requires B200"
)

BASE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(91519)
    w = torch.randn(128, 1024, 3840, device="cuda", dtype=torch.bfloat16)
    w = w.to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-10, -1, (128, 8, 30), device="cuda").float())
    return w, scales


def make_case(tokens, grouped, strided=False):
    torch.manual_seed(91520 + tokens)
    rows = tokens * 8
    a_storage = torch.randn(rows, 3968 if strided else 3840, device="cuda").to(
        torch.float8_e4m3fn
    )
    a = a_storage[:, :3840]
    scales = torch.exp2(torch.randint(-4, 3, (rows, 30), device="cuda").float())
    ids = (torch.arange(rows, device="cuda").view(tokens, 8) % 128).int()
    if grouped:
        sorted_ids, expert_ids, padded = moe_align_block_size(ids, 16, 128)
    else:
        sorted_ids, expert_ids = None, ids.view(-1)
        padded = torch.tensor([rows * 16], device="cuda", dtype=torch.int32)
    storage = torch.full((tokens, 8, 1040), 19.0, device="cuda", dtype=torch.bfloat16)
    return a, scales, ids, sorted_ids, expert_ids, padded, storage


def run(a, scale, w, weight_scale, output, sorted_ids, expert_ids, padded, config):
    invoke_fused_moe_triton_kernel(
        a,
        w,
        output,
        scale,
        weight_scale,
        None,
        sorted_ids,
        expert_ids,
        padded,
        False,
        1,
        config,
        tl.bfloat16,
        True,
        False,
        False,
        False,
        False,
        [128, 128],
    )


@pytest.mark.parametrize("tokens", [1, 2, 4, 16])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_fp8_w2_matches_original_and_preserves_padding(
    weights, tokens, grouped, strided
):
    a, scale, _, sorted_ids, expert_ids, padded, storage = make_case(
        tokens, grouped, strided
    )
    expected = torch.empty(tokens, 8, 1024, device="cuda", dtype=torch.bfloat16)
    config = try_get_yoco_fp8_w2_config(tokens, 128, 1024, 3840, BASE)
    assert (config != BASE) == (tokens in (1, 2, 4))
    run(
        a, scale, weights[0], weights[1], expected, sorted_ids, expert_ids, padded, BASE
    )
    run(
        a,
        scale,
        weights[0],
        weights[1],
        storage[:, :, :1024],
        sorted_ids,
        expert_ids,
        padded,
        config,
    )
    assert torch.equal(storage[:, :, :1024], expected)
    assert torch.isfinite(expected).all()
    assert (storage[:, :, 1024:] == 19).all()
    assert BASE["BLOCK_SIZE_N"] == 128


@pytest.mark.parametrize("tokens,grouped", [(1, False), (4, True)])
def test_fp8_w2_graph_replays_new_routes_inputs_and_scales(weights, tokens, grouped):
    a, scale, ids, sorted_ids, expert_ids, padded, storage = make_case(tokens, grouped)
    output = storage[:, :, :1024]
    expected = torch.empty_like(output)
    config = try_get_yoco_fp8_w2_config(tokens, 128, 1024, 3840, BASE)
    fn = lambda: run(
        a, scale, weights[0], weights[1], output, sorted_ids, expert_ids, padded, config
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for shift in [9, 47, 91]:
        a.copy_(torch.randn_like(a.float()).to(a.dtype))
        scale.mul_(2)
        ids.add_(shift).remainder_(128)
        if grouped:
            new_sorted, new_expert, new_padded = moe_align_block_size(ids, 16, 128)
            sorted_ids.copy_(new_sorted)
            expert_ids.copy_(new_expert)
            padded.copy_(new_padded)
        run(
            a,
            scale,
            weights[0],
            weights[1],
            expected,
            sorted_ids,
            expert_ids,
            padded,
            BASE,
        )
        graph.replay()
        assert torch.equal(output, expected)
        assert (storage[:, :, 1024:] == 19).all()


def test_fp8_w2_agrees_with_independent_block_scaled_fp64_reference(weights):
    a, scale, ids, sorted_ids, expert_ids, padded, storage = make_case(1, False)
    config = try_get_yoco_fp8_w2_config(1, 128, 1024, 3840, BASE)
    output = storage[:, :, :1024]
    run(
        a, scale, weights[0], weights[1], output, sorted_ids, expert_ids, padded, config
    )
    references = []
    for row, expert in enumerate(ids.flatten().tolist()):
        x = (a[row].double().reshape(30, 128) * scale[row].double()[:, None]).flatten()
        w = weights[0][expert].double().reshape(8, 128, 30, 128)
        s = weights[1][expert].double().reshape(8, 1, 30, 1)
        references.append(torch.mv((w * s).reshape(1024, 3840), x))
    ref = torch.stack(references).reshape(output.shape)
    assert (output.double() - ref).norm() / ref.norm() < 0.004
