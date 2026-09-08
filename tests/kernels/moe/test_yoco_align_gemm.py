# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual modular expert path with independent W13/W2 tiles."""

import json

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm.model_executor.layers.yoco_align_moe import envs
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def assert_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.fixture(scope="module")
def weights():
    if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
        pytest.skip("requires CUDA BF16 tensor cores")
    torch.manual_seed(71717)
    return (
        torch.randn(128, 7680, 1024, device="cuda", dtype=torch.bfloat16) / 32,
        torch.randn(128, 1024, 3840, device="cuda", dtype=torch.bfloat16) / 3840**0.5,
    )


@pytest.mark.parametrize(
    "rows,skew",
    [(1, True), (17, False), (129, True), (512, False), (1024, False), (2048, True)],
)
@torch.no_grad()
def test_modular_align_gemm_stages_and_single_token(
    weights, rows, skew, tmp_path, monkeypatch
):
    w13, w2 = weights
    torch.manual_seed(rows + 1921)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16) * (
        8 if skew else 1
    )
    ids = (
        torch.rand(rows, 8 if skew else 128, device="cuda")
        .argsort(-1)[:, :8]
        .int()
        .contiguous()
    )
    probs = torch.rand(rows, 8, device="cuda")
    probs /= probs.sum(-1, keepdim=True)
    expert = triton_moe.TritonExperts(
        make_dummy_moe_config(128, 8, 1024, 3840), FUSED_MOE_UNQUANTIZED_CONFIG
    )
    expert.yoco_align_weighted_swiglu = True
    expert.yoco_align_moe_sum = True
    expert.swiglu_limit = 0.5 if skew else 10.0
    cfg = dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        SPLIT_K=1,
        num_warps=8,
        num_stages=3,
    )
    profile = dict(
        schema_version=1,
        environment=dict(
            device_name=current_platform.get_device_name(),
            torch_version=str(torch.__version__),
            triton_version=triton.__version__,
            cuda_version=torch.version.cuda,
        ),
        w13_shape=list(w13.shape),
        w2_shape=list(w2.shape),
        top_k=8,
        configs={str(rows): dict(w13=cfg, w2=dict(cfg, BLOCK_SIZE_N=256))},
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", None)
    snapshots = []
    original = triton_moe.invoke_fused_moe_triton_kernel

    def traced(*args, **kwargs):
        original(*args, **kwargs)
        snapshots.append((args[0].clone(), args[2].clone(), args[11].copy()))

    monkeypatch.setattr(triton_moe, "invoke_fused_moe_triton_kernel", traced)

    def run(inputs, route_ids, routing_weights):
        m = len(inputs)
        shapes = expert.workspace_shapes(
            m, 7680, 1024, 8, 128, 128, None, MoEActivation.SILU
        )
        workspace13 = inputs.new_empty(shapes[0])
        workspace2 = inputs.new_empty(shapes[1])
        output = torch.empty_like(inputs)
        expert.apply(
            output,
            inputs,
            w13,
            w2,
            routing_weights,
            route_ids,
            MoEActivation.SILU,
            128,
            None,
            None,
            None,
            workspace13,
            workspace2,
            None,
            False,
        )
        return output

    baseline = run(x, ids, probs)
    before = snapshots[:]
    snapshots.clear()
    monkeypatch.setattr(envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", str(path))
    candidate = run(x, ids, probs)
    after = snapshots[:]
    assert len(before) == len(after) == 2
    assert after[0][2] == cfg
    assert after[1][2] == dict(cfg, BLOCK_SIZE_N=256)
    for old, new in zip(before, after):
        assert_bytes(old[0], new[0])  # input to W2 is the rounded weighted activation
        assert_bytes(old[1], new[1])
    assert_bytes(baseline, candidate)
    # Compare the same target in the first, middle and last physical row with
    # the one-row launch. A different route assignment cannot change its value.
    for index in sorted({0, rows // 2, rows - 1}):
        snapshots.clear()
        single = run(
            x[index : index + 1], ids[index : index + 1], probs[index : index + 1]
        )
        assert_bytes(candidate[index : index + 1], single)
