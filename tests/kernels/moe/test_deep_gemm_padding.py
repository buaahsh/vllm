# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing-independent bounds for graph-captured DeepGEMM workspaces."""

from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.yoco_ops.fp8_permute import (
    compute_aligned_M,
    deepgemm_moe_permute,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported


@pytest.mark.parametrize(
    "tokens,expected",
    [(1, 1024), (2, 2048), (4, 4096), (8, 8192), (16, 16384), (32, 16512)],
)
def test_decode_padding_only_reserves_reachable_experts(tokens, expected):
    assert compute_aligned_M(tokens, 8, 128, 128, None) == expected


@pytest.mark.parametrize("alignment", [4, 8, 128])
def test_workspace_bounds_every_small_expert_distribution(alignment):
    # Includes empty experts, skewed routing and counts straddling an
    # alignment boundary. Compute required rows from counts, independently
    # of the allocation formula.
    for counts in product(range(10), repeat=4):
        routed_tokens = sum(counts)
        allocated = compute_aligned_M(routed_tokens, 1, 4, alignment, None)
        required = sum(
            ((count + alignment - 1) // alignment) * alignment for count in counts
        )
        old_bound = (
            (routed_tokens + 4 * (alignment - 1) + alignment - 1) // alignment
        ) * alignment
        assert required <= allocated <= old_bound, counts
        assert allocated % alignment == 0


def test_explicit_cpu_counts_still_determine_workspace():
    counts = torch.tensor([0, 1, 127, 128, 129], dtype=torch.int32)
    metadata = SimpleNamespace(expert_num_tokens_cpu=counts)
    assert compute_aligned_M(512, 8, 5, 128, metadata) == 640


def test_empty_input_preserves_existing_workspace():
    assert compute_aligned_M(0, 8, 128, 128, None) == 16256


@pytest.mark.skipif(
    not is_deep_gemm_supported(), reason="requires the DeepGEMM CUDA backend"
)
@pytest.mark.parametrize("tokens", [1, 2, 7, 15, 16, 17, 64])
@pytest.mark.parametrize("routing", ["spread", "concentrated", "nonlocal"])
@pytest.mark.parametrize("psum", [False, True])
def test_permutation_fits_workspace_with_guard_rows(tokens, routing, psum):
    device = "cuda"
    experts, topk, hidden, alignment = 128, 8, 128, 128
    values = torch.arange(tokens * hidden, device=device).reshape(tokens, hidden)
    inputs = (values % 16).to(torch.float8_e4m3fn)
    scales = torch.ones(tokens, 1, device=device)
    ids = (
        torch.arange(tokens * topk, device=device).reshape(tokens, topk) % experts
    ).to(torch.int32)
    expert_map = None
    if routing == "concentrated":
        ids = torch.arange(topk, device=device, dtype=torch.int32).repeat(tokens, 1)
    elif routing == "nonlocal":
        expert_map = torch.arange(experts, device=device, dtype=torch.int32)
        expert_map[1::2] = -1
        ids[:, 0] = -1
    rows = compute_aligned_M(tokens, topk, experts, alignment, None)
    storage = torch.full(
        (rows + 4, hidden), 32.0, device=device, dtype=torch.float8_e4m3fn
    )
    quantized, permuted_scales, layout, inverse = deepgemm_moe_permute(
        inputs,
        scales,
        ids,
        experts,
        expert_map,
        None,
        aq_out=storage[:rows],
        use_psum_layout=psum,
    )
    local = ids >= 0
    if expert_map is not None:
        local &= expert_map[ids.clamp_min(0).long()] >= 0
    valid_rows = inverse[local].long()
    assert bool(((valid_rows >= 0) & (valid_rows < rows)).all())
    assert bool((inverse[~local] == -1).all())
    expected = inputs.float()[:, None, :].expand(-1, topk, -1)[local]
    torch.testing.assert_close(quantized.float()[valid_rows], expected, rtol=0, atol=0)
    assert bool((permuted_scales[valid_rows] == 1).all())
    assert bool((storage[rows:].float() == 32).all())
    if psum:
        assert int(layout[-1]) <= rows
