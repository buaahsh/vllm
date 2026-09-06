# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for YOCO-specific behavior in the modular Triton MoE path."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tests.kernels.moe.utils import make_dummy_moe_config, modular_triton_fused_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    make_unquantized_swiglu_params,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm import (
    yoco_deep_gemm_w2,
    yoco_deep_gemm_w2_workspace_rows,
)
from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
    select_yoco_decode_config,
    select_yoco_w2_config,
    select_yoco_w13_config,
    yoco_swapped_clamped_swiglu,
    yoco_topk8_sum,
)
from vllm.model_executor.layers.fused_moe.experts.yoco_trtllm_bf16 import (
    YocoTrtLlmBf16Experts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def test_yoco_flashinfer_autotune_uses_persistent_cache(monkeypatch, tmp_path) -> None:
    import vllm.model_executor.warmup.kernel_warmup as warmup
    import vllm.utils.flashinfer as flashinfer_utils

    cache = tmp_path / "autotune.json"
    monkeypatch.setenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE", str(cache))
    recorded = {}

    @contextlib.contextmanager
    def fake_autotune(*, cache=None):
        recorded["cache"] = cache
        yield

    monkeypatch.setattr(flashinfer_utils, "autotune", fake_autotune)

    def dummy_run(num_tokens, **kwargs):
        recorded["num_tokens"] = num_tokens
        recorded["kwargs"] = kwargs
        assert flashinfer_utils._is_fi_autotuning

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        _dummy_run=dummy_run,
    )
    warmup.flashinfer_autotune(runner)

    assert recorded["cache"] == str(cache)
    assert recorded["num_tokens"] == 8192
    assert recorded["kwargs"] == {"skip_eplb": True, "is_profile": True}
    assert not flashinfer_utils._is_fi_autotuning


def test_yoco_private_trtllm_backend_mapping() -> None:
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
        backend_to_kernel_cls,
        map_unquantized_backend,
    )

    backend = map_unquantized_backend("yoco_flashinfer_trtllm")
    assert backend == UnquantizedMoeBackend.YOCO_FLASHINFER_TRTLLM
    assert backend_to_kernel_cls(backend) is YocoTrtLlmBf16Experts


def test_yoco_private_trtllm_passes_clamp_and_output(monkeypatch) -> None:
    import flashinfer.fused_moe

    recorded = {}

    def fake_trtllm_bf16_routed_moe(**kwargs):
        recorded.update(kwargs)
        kwargs["output"].fill_(3)
        return kwargs["output"]

    monkeypatch.setattr(
        flashinfer.fused_moe,
        "trtllm_bf16_routed_moe",
        fake_trtllm_bf16_routed_moe,
    )
    experts = YocoTrtLlmBf16Experts.__new__(YocoTrtLlmBf16Experts)
    experts.num_experts = 4
    experts.intermediate_size = 8
    experts.max_capture_size = 64
    experts._swiglu_limit = None
    experts._swiglu_alpha = None
    experts._swiglu_beta = None
    experts._swiglu_limit_tensor = None
    experts.swiglu_limit = 10.0

    output = torch.empty(2, 8, dtype=torch.bfloat16)
    experts.apply(
        output=output,
        hidden_states=torch.zeros(2, 8, dtype=torch.bfloat16),
        w1=torch.zeros(4, 1, 16, 1, dtype=torch.bfloat16),
        w2=torch.zeros(4, 1, 8, 1, dtype=torch.bfloat16),
        topk_weights=torch.full((2, 2), 0.5, dtype=torch.float32),
        topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert torch.equal(output, torch.full_like(output, 3))
    assert recorded["output"] is output
    assert recorded["topk_ids"][0].dtype == torch.int32
    assert torch.equal(recorded["gemm1_alpha"], torch.ones(4))
    assert torch.equal(recorded["gemm1_beta"], torch.zeros(4))
    assert torch.equal(recorded["gemm1_clamp_limit"], torch.full((4,), 10.0))
    assert recorded["tune_max_num_tokens"] == 64


def test_yoco_private_trtllm_supports_only_tp1() -> None:
    base = dict(tp_size=1, pcp_size=1, dp_size=1, ep_size=1, enable_eplb=False)
    assert YocoTrtLlmBf16Experts._supports_parallel_config(SimpleNamespace(**base))
    for field in ("tp_size", "pcp_size", "dp_size", "ep_size"):
        parallel = dict(base)
        parallel[field] = 2
        assert not YocoTrtLlmBf16Experts._supports_parallel_config(
            SimpleNamespace(**parallel)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_weights_are_applied_before_w2(monkeypatch, workspace_init) -> None:
    import vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm as yoco_dg

    set_random_seed(8300)
    num_tokens, num_experts, topk = 7, 4, 2
    hidden_size, intermediate_size = 64, 128
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 3
    ).contiguous()
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 5
    ).contiguous()
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 5
    ).contiguous()
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    moe_config = make_dummy_moe_config(
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
    )
    moe = modular_triton_fused_moe(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    moe.fused_experts.swiglu_limit = 10.0
    moe.fused_experts.yoco_align_weighted_swiglu = True
    moe.fused_experts.yoco_align_deep_gemm_w2 = True

    original_yoco_w2 = yoco_dg.yoco_deep_gemm_w2

    def fake_deep_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        d: torch.Tensor,
        grouped_layout: torch.Tensor,
        **kwargs,
    ) -> None:
        assert kwargs["use_psum_layout"]
        start = 0
        for expert, end in enumerate(grouped_layout.cpu().tolist()):
            if end > start:
                d[start:end].copy_(F.linear(a[start:end], b[expert]))
            start = end

    def injected_yoco_w2(*args, **kwargs) -> None:
        original_yoco_w2(*args, **kwargs, deep_gemm_impl=fake_deep_gemm)

    monkeypatch.setattr(yoco_dg, "supports_yoco_deep_gemm_w2", lambda: True)
    monkeypatch.setattr(yoco_dg, "yoco_deep_gemm_w2", injected_yoco_w2)
    actual = moe.apply(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    reference_rows = []
    for token_idx in range(num_tokens):
        expert_outputs = []
        for route_idx in range(topk):
            expert_idx = int(topk_ids[token_idx, route_idx])
            w13_out = F.linear(
                hidden_states[token_idx : token_idx + 1], w13[expert_idx]
            )
            gate, up = w13_out.float().chunk(2, dim=-1)
            activation = (
                F.silu(gate.clamp(max=10.0))
                * up.clamp(min=-10.0, max=10.0)
                * topk_weights[token_idx, route_idx]
            ).to(torch.bfloat16)
            expert_outputs.append(F.linear(activation, w2[expert_idx]))
        reference_rows.append(torch.stack(expert_outputs).sum(dim=0))
    expected = torch.cat(reference_rows)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    moe.fused_experts.yoco_align_weighted_swiglu = False
    post_w2_weighted = moe.apply(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    assert not torch.equal(actual, post_w2_weighted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 17])
def test_yoco_deep_gemm_w2_pack_layout_and_unpack(num_tokens: int) -> None:
    set_random_seed(8400 + num_tokens)
    num_experts, topk = 4, 2
    intermediate_size, hidden_size = 64, 96
    num_assignments = num_tokens * topk
    activation = torch.randn(
        num_assignments,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # Repeated routes and empty experts exercise boundaries independently of
    # the usual top-k uniqueness guarantee.
    topk_ids = torch.tensor(
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        device="cuda",
        dtype=torch.int64,
    ).repeat((num_tokens + 3) // 4, 1)[:num_tokens]
    output = torch.empty(
        num_assignments, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    packed_rows = yoco_deep_gemm_w2_workspace_rows(num_tokens, topk, num_experts)
    packed_input_workspace = torch.empty(
        packed_rows * intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    packed_output_workspace = torch.empty(
        packed_rows * hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    call: dict[str, object] = {}

    def fake_deep_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        d: torch.Tensor,
        grouped_layout: torch.Tensor,
        **kwargs,
    ) -> None:
        call["layout"] = grouped_layout.clone()
        call["kwargs"] = kwargs
        start = 0
        for expert, end in enumerate(grouped_layout.cpu().tolist()):
            if end > start:
                d[start:end].copy_(F.linear(a[start:end], b[expert]))
            start = end

    yoco_deep_gemm_w2(
        output,
        activation,
        w2,
        topk_ids,
        packed_input_workspace,
        packed_output_workspace,
        deep_gemm_impl=fake_deep_gemm,
    )

    routed_experts = topk_ids.flatten().long()
    expected = torch.cat(
        [
            F.linear(activation[row : row + 1], w2[int(expert)])
            for row, expert in enumerate(routed_experts)
        ]
    )
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    layout = call["layout"]
    assert isinstance(layout, torch.Tensor)
    assert layout.dtype == torch.int32
    assert int(layout[-1]) == packed_rows
    assert call["kwargs"] == {
        "use_psum_layout": True,
        "expected_m_for_psum_layout": packed_rows,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_deep_gemm_w2_is_bitwise_training_exact_on_sm100() -> None:
    from vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm import (
        supports_yoco_deep_gemm_w2,
    )
    from vllm.utils.deep_gemm import _import_deep_gemm

    if not current_platform.is_device_capability_family(100):
        pytest.skip("B200/SM100 validation")
    if not supports_yoco_deep_gemm_w2():
        pytest.skip("BF16 DeepGEMM is unavailable")

    set_random_seed(8450)
    num_tokens, num_experts, topk = 7, 4, 2
    intermediate_size, hidden_size = 64, 128
    activation = torch.randn(
        num_tokens * topk,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor(
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        device="cuda",
        dtype=torch.int64,
    ).repeat(2, 1)[:num_tokens]

    packed_rows = yoco_deep_gemm_w2_workspace_rows(num_tokens, topk, num_experts)
    actual = torch.empty(
        num_tokens * topk, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    yoco_deep_gemm_w2(
        actual,
        activation,
        w2,
        topk_ids,
        torch.empty(
            packed_rows * intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.empty(
            packed_rows * hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )

    sorted_ids, _, post_pad = moe_align_block_size(
        topk_ids, 128, num_experts, pad_sorted_ids=True
    )
    training_rows = int(post_pad.item())
    training_ids = sorted_ids[:training_rows].long()
    valid = training_ids < activation.shape[0]
    training_input = torch.zeros(
        training_rows,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    training_input[valid] = activation[training_ids[valid]]
    counts = count_expert_num_tokens(topk_ids, num_experts, None)
    padded_counts = ((counts + 127) // 128) * 128
    training_layout = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    training_output = torch.empty(
        training_rows, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    deep_gemm = _import_deep_gemm()
    assert deep_gemm is not None
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        training_input,
        w2,
        training_output,
        training_layout,
        use_psum_layout=True,
        expected_m_for_psum_layout=training_rows,
    )
    expected = torch.empty_like(actual)
    expected[training_ids[valid]] = training_output[valid]
    assert torch.equal(actual, expected)


def test_yoco_w2_config_reuses_w13_dispatch() -> None:
    w13 = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 3,
    }
    valid = {**w13, "BLOCK_SIZE_N": 64, "num_warps": 4}
    invalid = {**valid, "BLOCK_SIZE_M": 64}

    assert select_yoco_w2_config({128: valid}, 120, w13) == valid
    assert select_yoco_w2_config({128: invalid}, 120, w13) is w13


def test_yoco_w13_config_selects_nearest_token_bucket() -> None:
    small = {"BLOCK_SIZE_M": 16}
    large = {"BLOCK_SIZE_M": 256}
    configs = {8: small, 7168: large}

    assert select_yoco_w13_config(configs, 7) is None
    assert select_yoco_w13_config(configs, 8) is small
    assert select_yoco_w13_config(configs, 7000) is large


def test_yoco_decode_config_is_bounded() -> None:
    small = {"BLOCK_SIZE_M": 16}
    large = {"BLOCK_SIZE_M": 32}
    configs = {1: small, 256: large}
    assert select_yoco_decode_config(configs, 0) is None
    assert select_yoco_decode_config(configs, 1) is small
    assert select_yoco_decode_config(configs, 128) is None
    assert select_yoco_decode_config(configs, 256) is large
    assert select_yoco_decode_config(configs, 257) is None
    assert select_yoco_decode_config(None, 128) is None


def test_yoco_decode_configs_preserve_prefill_and_align(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe.experts import yoco_triton as module

    decode = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128}
    prefill = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256}
    fallback = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128}
    monkeypatch.setattr(module.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(get_device_name=lambda: "NVIDIA B200"),
    )
    monkeypatch.setattr(module, "_load_yoco_w13_configs", lambda *args: {2048: prefill})
    monkeypatch.setattr(module, "_load_yoco_w2_configs", lambda *args: {2048: prefill})
    monkeypatch.setattr(
        module,
        "_load_yoco_decode_configs",
        lambda *args: {1: decode, 128: decode, 256: decode},
    )
    assert module.try_get_yoco_w13_config(128, 128, 3840, 1024) is decode
    assert module.try_get_yoco_w13_config(257, 128, 3840, 1024) is None
    assert module.try_get_yoco_w13_config(2048, 128, 3840, 1024) is prefill
    assert module.try_get_yoco_w2_config(128, 128, 1024, 3840, decode) is decode
    assert module.try_get_yoco_w2_config(128, 128, 1024, 3840, fallback) is fallback
    assert module.try_get_yoco_w2_config(257, 128, 1024, 3840, fallback) is fallback
    assert module.try_get_yoco_w2_config(2048, 128, 1024, 3840, prefill) is prefill
    monkeypatch.setattr(module.envs, "VLLM_BATCH_INVARIANT", True)
    assert module.try_get_yoco_w13_config(128, 128, 3840, 1024) is None
    assert module.try_get_yoco_w2_config(128, 128, 1024, 3840, fallback) is fallback


def test_yoco_flashinfer_clamped_swiglu_parameters() -> None:
    alpha, beta, limit = make_unquantized_swiglu_params(4, "cpu", 10.0)

    assert torch.equal(alpha, torch.ones(4))
    assert torch.equal(beta, torch.zeros(4))
    assert torch.equal(limit, torch.full((4,), 10.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_swapped_clamped_swiglu_is_bitwise_exact() -> None:
    set_random_seed(8701)
    gate = torch.randn(8, 3840, device="cuda", dtype=torch.bfloat16) * 8
    up = torch.randn_like(gate) * 8
    swapped = torch.cat((up, gate), dim=-1)
    actual = torch.empty_like(gate)
    yoco_swapped_clamped_swiglu(actual, swapped, 10.0)

    expected = (
        F.silu(gate.float().clamp(max=10.0)) * up.float().clamp(min=-10.0, max=10.0)
    ).to(torch.bfloat16)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 37, 2048])
def test_yoco_topk8_sum_is_bitwise_training_exact(num_tokens: int) -> None:
    set_random_seed(8500 + num_tokens)
    expert_output = torch.randn(
        num_tokens,
        8,
        1024,
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = torch.empty(
        num_tokens,
        1024,
        device="cuda",
        dtype=torch.bfloat16,
    )

    yoco_topk8_sum(expert_output, actual)

    accumulator = torch.zeros_like(actual, dtype=torch.float32)
    for route_idx in range(8):
        accumulator += expert_output[:, route_idx].float()
    expected = accumulator.to(torch.bfloat16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("align", "fast", "num_tokens", "expect_private"),
    [
        (False, False, 4096, False),
        (False, True, 2047, False),
        (False, True, 2048, True),
        (True, False, 1, True),
    ],
)
def test_yoco_moe_sum_dispatch(
    monkeypatch,
    align: bool,
    fast: bool,
    num_tokens: int,
    expect_private: bool,
) -> None:
    import vllm.model_executor.layers.fused_moe.experts.yoco_triton as yoco_triton

    calls: list[str] = []

    class FakeExperts:
        yoco_align_moe_sum = align
        yoco_fast_moe_sum = fast

        def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
            calls.append("common")

    def fake_yoco_sum(input: torch.Tensor, output: torch.Tensor) -> None:
        calls.append("private")

    monkeypatch.setattr(yoco_triton, "yoco_topk8_sum", fake_yoco_sum)
    input = torch.empty(num_tokens, 8, 1)
    output = torch.empty(num_tokens, 1)
    TritonExperts._yoco_moe_sum(FakeExperts(), input, output)

    assert calls == (["private"] if expect_private else ["common"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_deep_gemm_w2_falls_back_when_unsupported(
    monkeypatch, workspace_init
) -> None:
    import vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm as yoco_dg

    monkeypatch.setattr(yoco_dg, "supports_yoco_deep_gemm_w2", lambda: False)
    num_tokens, num_experts, topk = 3, 4, 2
    hidden_size, intermediate_size = 64, 128
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w13 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda", dtype=torch.int64)
    topk_weights = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    moe_config = make_dummy_moe_config(
        num_experts, topk, hidden_size, intermediate_size
    )
    moe = modular_triton_fused_moe(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    moe.fused_experts.swiglu_limit = 10.0
    moe.fused_experts.yoco_align_weighted_swiglu = True
    moe.fused_experts.yoco_align_deep_gemm_w2 = True

    actual = moe.apply(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    assert torch.isfinite(actual).all()
