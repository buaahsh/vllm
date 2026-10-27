# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.yoco_ops.norm import RMSClip, RMSNorm
from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear

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


@pytest.fixture
def scoped_cublaslt(monkeypatch):
    from vllm import envs
    from vllm.model_executor.layers.yoco_align_linear import align_gemm_backend

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("YOCO scoped cuBLASLt policy is validated on SM100")
    monkeypatch.setattr(envs, "VLLM_YOCO_ALIGN_GEMM_BACKEND", "cublaslt_v1")
    align_gemm_backend.cache_clear()
    assert align_gemm_backend() == "cublaslt_v1"
    yield
    align_gemm_backend.cache_clear()


@pytest.mark.parametrize(
    "n,k",
    [
        (8192, 3072),
        (1024, 3072),
        (3072, 4096),
        (64, 3072),
        (1280, 3072),
        (3072, 1280),
        (3072, 1024),
    ],
)
@pytest.mark.parametrize("rows", [1, 17, 129])
@torch.inference_mode()
def test_scoped_gemm_preserves_bytes_and_replays_changed_input(
    scoped_cublaslt, n, k, rows
):
    """Guard arithmetic, live Graph input and isolation from backward settings."""
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    torch.manual_seed(rows + n + k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    matmul = torch.backends.cuda.matmul
    before = (
        matmul.allow_tf32,
        matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.preferred_blas_library(),
    )
    expected = _yoco_align_linear_legacy(x, w, bias)
    actual = _yoco_align_linear(x, w, bias)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _yoco_align_linear(x, w, bias)
    x.add_(0.125)
    graph.replay()
    expected = _yoco_align_linear_legacy(x, w, bias)
    assert torch.isfinite(output).all()
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    after = (
        matmul.allow_tf32,
        matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.preferred_blas_library(),
    )
    assert after == before


@pytest.mark.parametrize(
    "n,k,strided", [(1, 3072, False), (96, 64, False), (1024, 3072, True)]
)
@torch.inference_mode()
def test_scoped_gemm_retains_legacy_for_unadmitted_shapes(
    scoped_cublaslt, n, k, strided
):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    step = 2 if strided else 1
    x = torch.randn(17, k * step, device="cuda", dtype=torch.bfloat16)[:, ::step]
    w = torch.randn(n, k * step, device="cuda", dtype=torch.bfloat16)[:, ::step]
    assert torch.equal(
        _yoco_align_linear(x, w).view(torch.int16),
        _yoco_align_linear_legacy(x, w).view(torch.int16),
    )


@torch.inference_mode()
def test_scoped_gemm_compile_preserves_operator_boundary(scoped_cublaslt):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    compiled = torch.compile(_yoco_align_linear, fullgraph=True, dynamic=True)
    weight = torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    for rows in [1, 17, 129]:
        x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
        actual = compiled(x, weight)
        expected = _yoco_align_linear_legacy(x, weight)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Storage admission happens inside the opaque runtime operation, where
    # actual pointer alignment is available even for a compiled caller.
    x = torch.randn(17, 4097, device="cuda", dtype=torch.bfloat16)[:, 1:]
    weight = torch.randn(3072, 4097, device="cuda", dtype=torch.bfloat16)[:, 1:]
    assert torch.equal(
        compiled(x, weight).view(torch.int16),
        _yoco_align_linear_legacy(x, weight).view(torch.int16),
    )


@pytest.mark.parametrize(
    "n,k",
    [
        (8192, 3072),
        (1024, 3072),
        (3072, 4096),
        (64, 3072),
        (1280, 3072),
        (3072, 1280),
        (3072, 1024),
    ],
)
@pytest.mark.parametrize("rows", [1, 8, 33, 129])
@pytest.mark.parametrize("pattern", ["range", "cancellation", "signed_zero"])
@torch.inference_mode()
def test_scoped_gemm_adversarial_values(scoped_cublaslt, n, k, rows, pattern):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    torch.manual_seed(9000 + rows + n)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if pattern == "range":
        x = torch.ldexp(
            x.float(), torch.randint(-10, 11, x.shape, device="cuda")
        ).bfloat16()
        w = torch.ldexp(
            w.float(), torch.randint(-10, 11, w.shape, device="cuda")
        ).bfloat16()
    elif pattern == "cancellation":
        x[:, 1::2] = -x[:, ::2]
        w[:, 1::2] = w[:, ::2]
        x[:, -1] += 0.125
    else:
        x.zero_()
        x[:, 1::2] = -0.0
    expected = _yoco_align_linear_legacy(x, w)
    actual = _yoco_align_linear(x, w)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("offset", [0, 1, 8])
@pytest.mark.parametrize("n,k", [(1024, 3072), (3072, 4096)])
@torch.inference_mode()
def test_scoped_gemm_row_stride_and_storage_alignment(
    scoped_cublaslt, monkeypatch, offset, n, k
):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    calls = []
    native = torch.ops._C.yoco_align_gemm

    def observe(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(torch.ops._C, "yoco_align_gemm", observe)
    torch.manual_seed(857)
    x = torch.randn(17, k + 32, device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    w = torch.randn(n, k + 32, device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    actual = _yoco_align_linear(x, w)
    expected = _yoco_align_linear_legacy(x, w)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert calls == ([] if offset == 1 else [True])


@torch.inference_mode()
def test_scoped_gemm_concurrent_streams(scoped_cublaslt):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pairs = [
        (
            torch.randn(129, 3072, device="cuda", dtype=torch.bfloat16),
            torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16),
        )
        for _ in streams
    ]
    torch.accelerator.synchronize()
    outputs = []
    for stream, (x, w) in zip(streams, pairs):
        with torch.cuda.stream(stream):
            for _ in range(10):
                result = _yoco_align_linear(x, w)
            outputs.append(result)
    for stream in streams:
        stream.synchronize()
    for result, (x, w) in zip(outputs, pairs):
        expected = _yoco_align_linear_legacy(x, w)
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 17, 129, 2048])
@torch.inference_mode()
def test_scoped_qkv_fusion_preserves_norm_boundary(scoped_cublaslt, monkeypatch, rows):
    from vllm import envs
    from vllm.model_executor.layers.yoco_align_linear import _projection_fusion_enabled
    from vllm.model_executor.layers.yoco_ops.projection import (
        _yoco_align_linear_legacy,
        _yoco_align_qkv_linear,
    )

    monkeypatch.setattr(envs, "VLLM_YOCO_ALIGN_FUSE_PROJECTIONS", True)
    _projection_fusion_enabled.cache_clear()
    try:
        torch.manual_seed(843)
        x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn(10240, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
        output = _yoco_align_qkv_linear(x, w, 8192, 1024)
        expected = [
            _yoco_align_linear_legacy(x, t) for t in w.split([8192, 1024, 1024])
        ]
        norm = RMSClip(128, has_weight=True, execution_mode="align").cuda().bfloat16()
        for actual, reference in zip(output, expected):
            assert torch.equal(
                actual.contiguous().view(torch.int16), reference.view(torch.int16)
            )
            a = norm(actual.view(rows, -1, 128))
            b = norm(reference.view(rows, -1, 128))
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    finally:
        _projection_fusion_enabled.cache_clear()


@pytest.mark.parametrize("rows", [0, 1, 32768, 32769])
def test_qkv_fusion_keeps_separate_projections_outside_admitted_rows(monkeypatch, rows):
    from vllm.model_executor.layers import yoco_align_linear
    from vllm.model_executor.layers.yoco_ops import projection

    monkeypatch.setattr(yoco_align_linear, "_projection_fusion_enabled", lambda: True)
    widths = []

    def observe(x, weight):
        widths.append(weight.shape[0])
        return x.new_empty((x.shape[0], weight.shape[0]))

    monkeypatch.setattr(projection, "_yoco_align_linear", observe)
    x = torch.empty(rows, 3072, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(10240, 3072, dtype=torch.bfloat16, device="meta")
    projection._yoco_align_qkv_linear(x, weight, 8192, 1024)
    assert widths == ([10240] if 0 < rows <= 32768 else [8192, 1024, 1024])


@pytest.mark.parametrize("rows", [1, 17, 129, 1024])
@torch.inference_mode()
def test_scoped_lm_head_matches_legacy_store(scoped_cublaslt, rows):
    from vllm.model_executor.layers.yoco_ops.projection import _yoco_align_linear_legacy

    torch.manual_seed(533 + rows)
    x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(154880, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    actual = _yoco_align_linear(x, w)
    expected = _yoco_align_linear_legacy(x, w)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
