# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers import yoco_probabilities as probs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("rows", [1, 8, 17, 128, 1024, 2048])
def test_all_interfaces_and_batch_invariance(rows):
    torch.manual_seed(95)
    x = torch.randn(rows, 154880, device="cuda", dtype=torch.bfloat16)
    lp = probs.log_softmax(x)
    same(lp, probs.log_softmax(x.float()))
    labels = torch.arange(rows, device="cuda") % x.shape[1]
    selected = probs.token_logprobs(x, labels)
    same(selected, lp.gather(1, labels[:, None]).squeeze(1))
    loss, _, _ = probs.cross_entropy_forward(x.float(), labels)
    same(loss, -selected)
    for index in sorted({0, rows // 2, rows - 1}):
        same(lp[index : index + 1], probs.log_softmax(x[index : index + 1]))
    if rows == 17:
        expected = x.double() - x.double().logsumexp(-1, keepdim=True)
        torch.testing.assert_close(lp.double(), expected, rtol=0, atol=6e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_tiles_signed_zero_and_ignore(dtype):
    x = torch.full((3, 32769), float("-inf"), device="cuda", dtype=dtype)
    x[:, -1] = 1000
    lp = probs.log_softmax(x)
    assert torch.isneginf(lp[:, :-1]).all()
    assert (lp[:, -1] == 0).all() and torch.signbit(lp[:, -1]).all()
    labels = torch.tensor([32768, -100, 32768], device="cuda")
    loss, _, z = probs.cross_entropy_forward(x, labels)
    assert (loss == 0).all() and not torch.signbit(loss).any()
    assert z[1] == 0
    ids = torch.tensor([[0, 32768]] * 3, device="cuda")
    same(probs.token_logprobs(x, ids), lp.gather(1, ids))


def test_layout_and_empty_rows():
    x = torch.randn(3, 257, 4, device="cuda")
    lp = probs.log_softmax(x, dim=1)
    same(lp, probs.log_softmax(x.transpose(1, 2), -1).transpose(1, 2))
    empty = torch.empty(0, 256, device="cuda")
    assert probs.log_softmax(empty).shape == empty.shape
    assert probs.token_logprobs(
        empty, torch.empty(0, device="cuda", dtype=torch.int64)
    ).shape == (0,)


def test_aten_autograd_and_all_sampler_paths():
    # A fresh test process starts with the default CUDA operator untouched.
    x = torch.randn(3, 257, device="cuda", requires_grad=True)
    probs.enable(register_aten=True)
    lp = x.log_softmax(-1)
    same(lp, probs.log_softmax(x.detach()))
    assert lp.grad_fn is not None
    dy = torch.randn_like(lp)
    (dx,) = torch.autograd.grad(lp, (x,), dy)
    xd = x.detach().double()
    p = (xd - xd.logsumexp(-1, keepdim=True)).exp()
    expected = dy.double() - p * dy.double().sum(-1, keepdim=True)
    torch.testing.assert_close(dx.double(), expected, rtol=1e-4, atol=2e-6)
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    same(Sampler.compute_logprobs(x.detach()), lp)
    ids = torch.tensor([[0, 1, 256]] * 3, device="cuda")
    same(compute_token_logprobs(x.detach(), ids), lp.gather(1, ids))
    # The processed path uses ATen after top-k/top-p masking.
    masked = x.detach().clone()
    masked[:, :250] = float("-inf")
    same(masked.log_softmax(-1), probs.log_softmax(masked))
