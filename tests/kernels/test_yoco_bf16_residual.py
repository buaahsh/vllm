# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast BF16 residual storage with FP32 add and norm statistics."""

import pytest
import torch

from vllm.model_executor.models.yoco import RMSNorm


@pytest.mark.parametrize("mode", ["align", "fast"])
@pytest.mark.parametrize("enabled", [False, True])
def test_residual_mode_and_cpu_fallback(monkeypatch, mode, enabled):
    monkeypatch.setenv("VLLM_YOCO_BF16_RESIDUAL", str(int(enabled)))
    dtype = torch.bfloat16 if mode == "fast" and enabled else torch.float32
    norm = RMSNorm(4, execution_mode=mode)
    assert norm.residual_dtype == dtype
    r = torch.tensor([[1.0, -1.0, 128.0, -128.0]], dtype=dtype)
    x = torch.tensor([[0.001, -0.001, 0.125, -0.125]], dtype=torch.bfloat16)
    expected_r = (r.float() + x.float()).to(dtype)
    with torch.no_grad():
        actual, actual_r = norm(x, r)
        expected = norm(expected_r)
    assert torch.equal(actual_r, expected_r)
    assert actual_r.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 3, 8, 127, 128, 513])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_fused_round_then_norm(monkeypatch, enabled, rows, strided):
    monkeypatch.setenv("VLLM_YOCO_BF16_RESIDUAL", str(int(enabled)))
    norm = RMSNorm(3072, dtype=torch.bfloat16).cuda()
    dtype = norm.residual_dtype
    torch.manual_seed(918 + rows)
    stride = 2 if strided else 1
    r = (torch.randn(rows, 3072 * stride, device="cuda") * 32).to(dtype)[:, ::stride]
    x = torch.randn(rows, 3072 * stride, device="cuda", dtype=torch.bfloat16)
    x = x[:, ::stride]
    norm.weight.uniform_(-1, 1)
    expected_r = (r.float() + x.float()).to(dtype)
    expected = norm(expected_r)
    actual, actual_r = norm(x, r)
    assert actual_r.dtype == dtype
    assert torch.equal(actual_r, expected_r)
    # Same stored sum and FP32 reduction tree as the separate Norm.
    assert torch.equal(actual, expected)
    if rows:
        ref = expected_r.float()
        ref = (
            ref
            * torch.rsqrt(ref.square().mean(-1, keepdim=True) + norm.eps)
            * norm.weight.float()
        ).bfloat16()
        torch.testing.assert_close(actual, ref, rtol=0.008, atol=0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_residual_graph_replay_and_fake(monkeypatch, enabled):
    monkeypatch.setenv("VLLM_YOCO_BF16_RESIDUAL", str(int(enabled)))
    norm = RMSNorm(3072, dtype=torch.bfloat16).cuda()
    x = torch.randn(8, 3072, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x, dtype=norm.residual_dtype)
    for _ in range(3):
        norm(x, r)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_r = norm(x, r)
    for scale in [0.001, 1.0, 100.0]:
        x.copy_(torch.randn_like(x) * scale)
        r.copy_(torch.randn_like(r) * 32)
        graph.replay()
        expected_r = (r.float() + x.float()).to(r.dtype)
        assert torch.equal(actual_r, expected_r)
        assert torch.equal(actual, norm(expected_r))
    torch.library.opcheck(
        torch.ops.vllm.yoco_fused_add_rms_norm.default,
        (x, r, norm.weight, norm.eps),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_align_fused_keeps_fp32_with_experiment_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_YOCO_BF16_RESIDUAL", "1")
    norm = RMSNorm(3072, dtype=torch.bfloat16, execution_mode="align").cuda()
    x = torch.randn(3, 3072, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x, dtype=torch.float32)
    actual, actual_r = norm(x, r)
    assert actual_r.dtype == torch.float32
    assert torch.equal(actual_r, r + x.float())
    assert torch.equal(actual, norm(actual_r))
    torch.library.opcheck(
        torch.ops.vllm.yoco_align_fused_add_rms_norm.default,
        (x, r, norm.weight, norm.eps),
        test_utils=("test_schema", "test_faketensor"),
    )
