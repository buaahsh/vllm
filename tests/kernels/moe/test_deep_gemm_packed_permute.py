# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fused expert metadata and scale packing against independent references."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
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


def check_result(x, scales, ids, expert_map, result, experts):
    q, packed, layout, inverse = result
    local_ids = ids.long().clone()
    if expert_map is not None:
        local_ids = torch.where(
            ids >= 0, expert_map[ids.clamp_min(0).long()], -1
        ).long()
    valid = local_ids >= 0
    counts = torch.bincount(local_ids[valid], minlength=experts)
    ends = (((counts + 127) // 128) * 128).cumsum(0).int()
    assert torch.equal(layout, ends)
    assert bool((inverse[~valid] == -1).all())
    rows = inverse[valid].long()
    assert len(rows.unique()) == len(rows)
    expected = x.view(torch.uint8)[:, None, :].expand(-1, ids.shape[1], -1)[valid]
    assert torch.equal(q.view(torch.uint8)[rows], expected)
    dense_scales = torch.ones(q.shape[0], scales.shape[1], device="cuda")
    dense_scales[rows] = scales[:, None, :].expand(-1, ids.shape[1], -1)[valid]
    reference = transform_sf_into_required_layout(
        sf=dense_scales.unsqueeze(0),
        mn=q.shape[0],
        k=x.shape[1],
        recipe=(1, 128, 128),
        num_groups=1,
        is_sfa=True,
    ).squeeze(0)
    assert packed.dtype == torch.int32 and packed.stride(0) == 1
    assert torch.equal(packed[rows], reference[rows])


@pytest.mark.parametrize("tokens", [1, 7, 129])
@pytest.mark.parametrize("hidden", [128, 1024, 3840])
@pytest.mark.parametrize("nonlocal_experts", [False, True])
def test_fused_prefix_and_packed_scatter(tokens, hidden, nonlocal_experts):
    torch.manual_seed(9214)
    experts, topk = 17, 4
    x = torch.randint(-16, 16, (tokens, hidden), device="cuda").to(torch.float8_e4m3fn)
    # Noncontiguous scales cover both strides. Include non-power-of-two
    # positive scales to check packing against DeepGEMM's own converter.
    scales = torch.exp2(torch.randn(hidden // 128, tokens, device="cuda") * 4 - 8).T
    ids = torch.rand(tokens, experts, device="cuda").topk(topk).indices.int()
    expert_map = None
    if nonlocal_experts:
        expert_map = torch.arange(experts, device="cuda", dtype=torch.int32)
        expert_map[1::2] = -1
        ids[:, 0] = -1
    rows = compute_aligned_M(tokens, topk, experts, 128, None)
    storage = torch.full(
        (rows + 2, hidden), 32, device="cuda", dtype=torch.float8_e4m3fn
    )
    result = deepgemm_moe_permute(
        x,
        scales,
        ids,
        experts,
        expert_map,
        None,
        aq_out=storage[:rows],
        use_psum_layout=True,
        pack_scales=True,
    )
    check_result(x, scales, ids, expert_map, result, experts)
    assert bool((storage[rows:].float() == 32).all())


def test_packed_permute_graph_replay_changes_expert_distribution():
    torch.manual_seed(9215)
    tokens, experts, topk, hidden = 33, 17, 4, 1024
    x = torch.randn(tokens, hidden, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randn(tokens, hidden // 128, device="cuda") - 8)
    ids = torch.rand(tokens, experts, device="cuda").topk(topk).indices.int()
    expert_map = torch.arange(experts, device="cuda", dtype=torch.int32)
    expert_map[1::2] = -1

    def run():
        return deepgemm_moe_permute(
            x,
            scales,
            ids,
            experts,
            expert_map,
            None,
            use_psum_layout=True,
            pack_scales=True,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for invalid in [False, True, False]:
        ids.copy_(torch.rand(tokens, experts, device="cuda").topk(topk).indices.int())
        if invalid:
            ids.fill_(-1)
        scales.copy_(torch.exp2(torch.randn_like(scales) - 8))
        graph.replay()
        check_result(x, scales, ids, expert_map, result, experts)
