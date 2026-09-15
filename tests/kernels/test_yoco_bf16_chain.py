# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 activation boundaries, native residual addition and BF16 router."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.models import yoco


@pytest.fixture(autouse=True)
def chain_environment(monkeypatch):
    monkeypatch.setenv("VLLM_YOCO_BF16_CHAIN", "1")
    monkeypatch.setenv("VLLM_YOCO_BF16_RESIDUAL", "0")


class TensorDtypes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.outputs = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        tensors = output if isinstance(output, (tuple, list)) else [output]
        self.outputs.extend(x.dtype for x in tensors if isinstance(x, torch.Tensor))
        return output


def test_materialized_residual_boundary_has_no_fp32_tensor():
    x = torch.randn(3, 3072, dtype=torch.bfloat16)
    r = torch.randn_like(x)
    with TensorDtypes() as observed:
        actual = yoco._yoco_add_residual(x, r, torch.bfloat16, True)
    assert actual.dtype == torch.bfloat16
    assert torch.float32 not in observed.outputs
    assert torch.equal(actual, (x.float() + r.float()).bfloat16())


def test_align_ignores_bf16_chain():
    norm = yoco.RMSNorm(4, execution_mode="align")
    assert not norm.bf16_chain
    assert norm.residual_dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [0, 1, 3, 8, 127, 128, 513])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_native_bf16_add_norm_matches_previous_storage_mode(rows, strided):
    norm = yoco.RMSNorm(3072, dtype=torch.bfloat16).cuda()
    stride = 2 if strided else 1
    torch.manual_seed(918 + rows)
    x = torch.randn(rows, 3072 * stride, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x) * 32
    x, r = x[:, ::stride], r[:, ::stride]
    norm.weight.uniform_(-1, 1)
    actual, actual_r = norm(x, r)
    expected, expected_r = torch.ops.vllm.yoco_fused_add_rms_norm(
        x, r, norm.weight, norm.eps
    )
    assert torch.equal(actual_r, expected_r)
    assert torch.equal(actual, expected)
    assert actual.dtype == actual_r.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_bf16_chain_graph_replay_and_fake():
    x = torch.randn(8, 3072, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    norm = yoco.RMSNorm(3072, dtype=torch.bfloat16).cuda()
    weight = torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 32

    def run():
        h, rr = norm(x, r)
        logits = torch.ops.vllm.yoco_router_linear_bf16(h, weight)
        return h, rr, logits

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for scale in [0.001, 1.0, 32.0]:
        x.copy_(torch.randn_like(x) * scale)
        weight.copy_(torch.randn_like(weight) / 32)
        graph.replay()
        for a, b in zip(actual, run()):
            assert torch.equal(a, b)
            assert a.dtype == torch.bfloat16
    for op, args in [
        (torch.ops.vllm.yoco_bf16_add_rms_norm.default, (x, r, norm.weight, norm.eps)),
        (torch.ops.vllm.yoco_router_linear_bf16.default, (x, weight)),
    ]:
        torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 8, 128])
@torch.inference_mode()
def test_bf16_router_topk_and_tensor_boundaries(rows):
    x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 32
    with TensorDtypes() as observed:
        logits = torch.ops.vllm.yoco_router_linear_bf16(x, weight)
    assert logits.dtype == torch.bfloat16
    assert torch.float32 not in observed.outputs
    assert torch.equal(logits, torch.nn.functional.linear(x, weight))
    probs, ids = yoco._yoco_topk_routing(x, logits, 8, True)
    expected_ids = logits.float().argsort(dim=-1, descending=True, stable=True)[:, :8]
    expected_probs = logits.float().gather(1, expected_ids).softmax(-1)
    assert torch.equal(ids.long(), expected_ids)
    torch.testing.assert_close(probs, expected_probs, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("normalized", [False, True])
@torch.inference_mode()
def test_bf16_router_cache_refresh(normalized):
    module = yoco.YOCOMoE.__new__(yoco.YOCOMoE)
    torch.nn.Module.__init__(module)
    module.execution_mode = "fast"
    module.bf16_chain = True
    module.router_weights_normalized = normalized
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.register_buffer("_normalized_gate_weight", None, persistent=False)
    module.register_buffer("_bf16_gate_weight", None, persistent=False)
    module.initialize_router_weight_cache()
    pointer = module._bf16_gate_weight.data_ptr()
    module.gate.weight.copy_(torch.randn_like(module.gate.weight))
    module.initialize_router_weight_cache()
    weight = module.gate.weight
    if not normalized:
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    assert module._bf16_gate_weight.data_ptr() == pointer
    assert torch.equal(module._bf16_gate_weight, weight.bfloat16())
