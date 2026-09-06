# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.models.yoco import RMSClip, RMSNorm, _yoco_align_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("hidden", [1024, 1280, 3072])
@pytest.mark.parametrize("rows", [1, 8, 17, 32, 33, 128])
@torch.inference_mode()
def test_align_small_gemm_preserves_large_tile_results(hidden, rows):
    torch.manual_seed(31)
    # Both operands are strided; bias must follow the BF16 GEMM store.
    x = torch.randn(rows, hidden * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    w = torch.randn(512, hidden * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        _yoco_align_linear(x, w, bias), linear_batch_invariant(x, w, bias)
    )


@pytest.mark.parametrize("heads", [8, 64])
@torch.inference_mode()
def test_align_weighted_clip_does_not_change_at_128_tokens(heads):
    torch.manual_seed(20260905)
    module = RMSClip(128, has_weight=True, execution_mode="align").cuda().bfloat16()
    module.weight.data.uniform_(-2, 2)
    # This used to mix Inductor's small-M reduction with a different Triton
    # reduction at M>=128, producing BF16 differences at rounding boundaries.
    storage = 4 * torch.randn(2048, heads + 1, 128, device="cuda", dtype=torch.bfloat16)
    x = storage[:, :heads]
    whole = module(x)
    split = torch.cat([module(part) for part in x.split(17)])
    assert torch.equal(whole, split)
    perm = torch.randperm(x.shape[0], device="cuda")
    assert torch.equal(module(x[perm]), whole[perm])


@pytest.mark.parametrize("hidden", [1024, 3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_align_fused_residual_norm_matches_unfused_and_graph(hidden, dtype):
    torch.manual_seed(19)
    module = RMSNorm(hidden, execution_mode="align").cuda()
    module.weight.data.uniform_(-2, 2)
    x = torch.randn(129, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x, dtype=torch.float32)
    normalized, residual_out = module(x, residual)
    expected_residual = residual + x.float()
    assert torch.equal(residual_out, expected_residual)
    assert torch.equal(normalized, module(expected_residual))
    split = [module(a, b) for a, b in zip(x.split(17), residual.split(17))]
    assert torch.equal(torch.cat([part[0] for part in split]), normalized)
    assert torch.equal(torch.cat([part[1] for part in split]), residual_out)
    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        graph_output = module(x, residual)
    graph.replay()
    assert torch.equal(graph_output[0], normalized)
    assert torch.equal(graph_output[1], residual_out)
