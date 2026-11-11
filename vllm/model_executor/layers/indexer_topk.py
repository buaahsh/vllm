# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k kernels for the DSA sparse attention indexer."""

import functools

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.workspace import current_workspace_manager

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# ---------------------------------------------------------------------------
# DeepSelect (vllm._deepselect_C)
# ---------------------------------------------------------------------------

# Matches the -1 fill convention used for topk_indices_buffer elsewhere.
IDX_OOB_FILL_VALUE = -1

try:
    import vllm._deepselect_C  # noqa: F401  (registers torch.ops.deep_select)
except ImportError as e:
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "Failed to import the DeepSelect extension (vllm._deepselect_C): %s", e
    )


@functools.lru_cache(maxsize=1)
def get_deep_select_stride_requirement() -> tuple[int, int]:
    """Stride alignment requirement (input, output) in bytes."""
    return torch.ops.deep_select.get_alignment_requirement()


def is_deep_select_supported(input: torch.Tensor, topk: int) -> bool:
    """Whether the kernel accepts this input (dtype/stride/topk constraints)."""
    return (
        topk <= 4096
        and input.shape[1] < 2**23
        and input.dtype in (torch.float32, torch.bfloat16)
        and input.stride(1) == 1
        and input.stride(0)
        * input.element_size()
        % get_deep_select_stride_requirement()[0]
        == 0
    )


def _get_empty_and_aligned_tensor(
    dim0: int, dim1: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Tensor with shape (dim0, dim1) whose stride(0) is 32B-aligned."""
    output_stride_requirement = get_deep_select_stride_requirement()[1] // (
        dtype.itemsize
    )
    assert output_stride_requirement > 0
    dim1_rounded = (
        (dim1 + output_stride_requirement - 1) // output_stride_requirement
    ) * output_stride_requirement
    return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]


def deep_select_topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Select the top-k indices per row of `input` with DeepSelect.

    Args:
        input: (num_rows, vocab_size), bf16 or fp32. stride(1) must be 1 and
            stride(0) must be 1024B-aligned.
        topk: Number of elements to select per row; must be <= 4096.
        end: Optional (num_rows,) int32 tensor with the exclusive right
            boundary of each row. Rows with `end[i] < topk` get their
            remaining indices filled with -1.
        output_idx: Optional preallocated (num_rows, topk) output tensor whose
            stride(0) is 32B-aligned (e.g. a slice of a wider buffer).
        indices_dtype: Output dtype when `output_idx` is not provided.

    Returns:
        The (num_rows, topk) indices tensor.
    """
    assert input.dim() == 2 and input.stride(1) == 1
    assert (
        input.stride(0) * input.element_size() % get_deep_select_stride_requirement()[0]
        == 0
    )

    num_rows = input.shape[0]
    if output_idx is None:
        output_idx = _get_empty_and_aligned_tensor(
            num_rows, topk, input.device, indices_dtype
        )
    else:
        assert output_idx.dtype == indices_dtype
        assert output_idx.shape[0] >= num_rows and output_idx.shape[1] >= topk

    torch.ops.deep_select.topk(
        input,
        topk,
        None,  # begin is not supported
        end,
        False,  # sorted_value
        False,  # sorted_index
        None,  # output_value
        output_idx,
        None,  # output_idx_offset
        IDX_OOB_FILL_VALUE,
        float("-inf"),  # value_oob_fill_value
        False,  # return_value
        False,  # abort_when_nan_found: dummy/capture passes can feed NaN
        # logits from uninitialized KV cache; the reference topk kernels do
        # not trap on NaN, and trapping would abort CUDA graph capture.
    )
    return output_idx


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deterministic top-k (teacher-migrate T-F16, env VLLM_TM_DET_TOPK=1, default off).
# Ported from vLLM PR #66088 (top_k_per_row_batch_invariant / _bi_topk_chunk):
# select by score, ties by the smaller column, write ascending indices with -1
# padding. Pure PyTorch; the radix-select kernels assign output slots with
# atomicAdd, so their index order (and occasionally the selected set) varies
# run to run, and the sparse MLA kernels accumulate in index order.
# ---------------------------------------------------------------------------
import os as _tm_os  # noqa: E402

TM_DET_TOPK = _tm_os.environ.get("VLLM_TM_DET_TOPK", "0") == "1"
_BI_TOPK_MAX_KEY_ELEMS = 64 * 1024 * 1024  # 512 MiB of int64 keys per chunk


def top_k_per_row_batch_invariant(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Select by score then smaller column; write ascending indices and -1 padding.

    NaNs rank above +inf. Unique int64 keys provide deterministic selection;
    chunks bound temporary memory independently of the batch size.
    logits: [num_rows, width] float32 (row-contiguous); row_ends: [num_rows]
    exclusive bounds; topk_indices: [num_rows, topk_tokens] int32 output.
    """
    assert logits.dtype == torch.float32 and logits.dim() == 2
    assert logits.stride(1) == 1
    num_rows, width = logits.shape
    if topk_tokens > width:
        topk_indices[:, width:topk_tokens].fill_(-1)
    if num_rows == 0 or width == 0 or topk_tokens == 0:
        return
    max_rows = max(1, _BI_TOPK_MAX_KEY_ELEMS // width)
    for start in range(0, num_rows, max_rows):
        _bi_topk_chunk(
            logits[start : start + max_rows],
            row_ends[start : start + max_rows],
            topk_indices[start : start + max_rows],
            topk_tokens,
        )


def _bi_topk_chunk(logits, row_ends, topk_indices, topk_tokens) -> None:
    num_rows, width = logits.shape
    k = min(topk_tokens, width)
    if num_rows == 0 or k == 0:
        return
    device = logits.device
    # Signed integer order follows float order. Canonicalize numerical ties.
    bits = logits.view(torch.int32)
    key32 = bits.bitwise_right_shift(31)
    key32.bitwise_and_(0x7FFFFFFF).bitwise_xor_(bits)
    key32.masked_fill_(logits == 0, 0)
    key32.masked_fill_(torch.isnan(logits), 0x7FFFFFFF)
    key = key32.to(torch.int64)
    del key32
    key.bitwise_left_shift_(32)
    key.bitwise_or_((width - 1) - torch.arange(width, device=device))
    # Exclude garbage columns with a key below every valid score, including -inf.
    key.masked_fill_(
        torch.arange(width, device=device).unsqueeze(0)
        >= row_ends.to(torch.int64).unsqueeze(1),
        torch.iinfo(torch.int64).min,
    )
    cols = torch.topk(key, k, dim=-1).indices
    del key
    keyed = torch.sort(
        torch.where(cols < row_ends.to(torch.int64).unsqueeze(1), cols, width), dim=-1
    ).values
    out = torch.where(keyed < width, keyed.to(torch.int32), -1)
    topk_indices[:, :k].copy_(out)


def tm_det_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Drop-in for ops.top_k_per_row_prefill with ABSOLUTE [row_starts, row_ends) column bounds.

    Output indices are relative to row_starts (the kernel's convention, stride1 == 1 path).
    Rows are re-based with a gather (bounded by _BI_TOPK_MAX_KEY_ELEMS) before the
    deterministic selection.
    """
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    starts = row_starts[:num_rows].to(torch.int64)
    lens = (row_ends[:num_rows].to(torch.int64) - starts).clamp_min(0)
    width = int(lens.max().item()) if num_rows else 0
    if width == 0:
        topk_indices[:num_rows].fill_(-1)
        return
    lg = logits.float() if logits.dtype != torch.float32 else logits
    max_rows = max(1, _BI_TOPK_MAX_KEY_ELEMS // width)
    cols = torch.arange(width, device=logits.device, dtype=torch.int64)
    for s0 in range(0, num_rows, max_rows):
        s1 = min(num_rows, s0 + max_rows)
        idx = (starts[s0:s1].unsqueeze(1) + cols.unsqueeze(0)).clamp_max(lg.shape[1] - 1)
        rel = torch.gather(lg[s0:s1], 1, idx).contiguous()
        top_k_per_row_batch_invariant(rel, lens[s0:s1], topk_indices[s0:s1], topk_tokens)


@functools.cache
def get_indexer_topk(backend: str) -> "SparseIndexerTopk":
    return SparseIndexerTopk(backend)


class SparseIndexerTopk(torch.nn.Module):
    """The sparse indexer's decode top-k stage.

    Selects among the available top-k kernels (see
    kernel_config.sparse_indexer_topk_backend) and runs the chosen one.
    """

    def __init__(self, backend: str | None = None) -> None:
        super().__init__()
        if backend is None:
            backend = (
                get_current_vllm_config().kernel_config.sparse_indexer_topk_backend
            )
        self._backend = backend
        self._is_cuda = current_platform.is_cuda()
        self._has_deep_select = self._is_cuda and (
            current_platform.is_device_capability_family(100)
        )
        self._has_flashinfer_topk = has_flashinfer()
        self._cooperative_capable = self._is_cuda and (
            current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )

    def resolve_backend(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """Resolve the decode top-k implementation from the configured
        backend ("auto" = the pre-existing chain, or a validated explicit
        value)."""
        if self._backend == "auto":
            return self._resolve_auto(logits, topk_tokens, num_rows)

        failures: list[str] = []
        if self._backend == "cooperative":
            failures = self._cooperative_constraints(logits, topk_tokens, num_rows)
        elif self._backend == "persistent":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if topk_tokens not in (512, 1024, 2048):
                failures.append(
                    f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
                )
        elif self._backend == "deep_select":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            elif not self._has_deep_select:
                failures.append("requires SM100a/SM103a (10.x device family)")
            elif not is_deep_select_supported(logits, topk_tokens):
                failures.append(
                    f"inputs violate DeepSelect's constraints: dtype={logits.dtype},"
                    f" stride={logits.stride()}, topk={topk_tokens}"
                )
        elif self._backend == "flashinfer":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if not self._has_flashinfer_topk:
                failures.append(
                    "flashinfer.topk.top_k_ragged_transform is not importable"
                )
            if logits.dtype != torch.float32 or logits.stride(1) != 1:
                failures.append(
                    f"requires fp32 logits with stride(1) == 1, got "
                    f"dtype={logits.dtype}, stride={logits.stride()}"
                )
        if failures:
            raise RuntimeError(
                f"sparse_indexer_topk_backend='{self._backend}' was requested, but: "
                + "; ".join(failures)
            )
        return self._backend

    def _resolve_auto(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """The pre-existing priority chain: cooperative -> persistent ->
        per_row. deep_select/flashinfer/torch are opt-in only."""
        if not self._cooperative_constraints(logits, topk_tokens, num_rows):
            return "cooperative"
        if self._is_cuda and topk_tokens in (512, 1024, 2048):
            return "persistent"
        return "per_row"

    def _cooperative_constraints(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> list[str]:
        """Unmet constraints of cooperative_topk (empty when applicable)."""
        failures = []
        if not self._is_cuda:
            failures.append("requires a CUDA platform")
        if topk_tokens not in (512, 1024, 2048):
            failures.append(
                f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
            )
        if num_rows > 64:
            failures.append(f"num_rows must be <= 64, got {num_rows}")
        if logits.stride(0) % 4 != 0:
            failures.append(
                f"logits.stride(0) must be divisible by 4, got {logits.stride(0)}"
            )
        if self._is_cuda and not self._cooperative_capable:
            failures.append("requires SM90+ and is not supported on the SM12x family")
        return failures

    @staticmethod
    def _row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
        """Per-row exclusive end offsets (int32, (num_rows,)) for top-k
        kernels that take ragged lengths (DeepSelect, FlashInfer, torch
        reference).

        seq_lens is (B, next_n) per-row effective lens for native spec decode
        and (B, 1) otherwise, in which case per-row lens are derived the same
        way as the other decode top-k kernels.
        """
        if seq_lens.numel() == num_rows:
            row_ends = seq_lens.reshape(-1)
        else:
            next_n_offsets = torch.arange(
                next_n, dtype=torch.int32, device=seq_lens.device
            )
            row_ends = (
                (seq_lens.reshape(-1, 1) - next_n + 1 + next_n_offsets)
                .clamp_(min=0)
                .reshape(-1)
            )
        assert row_ends.dtype == torch.int32
        return row_ends

    def forward(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk_indices: torch.Tensor,
        topk_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Run the resolved decode top-k implementation, writing into
        topk_indices (int32, -1 fill for rows shorter than topk_tokens)."""
        backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])
        if backend == "deep_select":
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            deep_select_topk(logits, topk_tokens, end=row_ends, output_idx=topk_indices)
        elif backend == "cooperative":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                max_seq_len,
            )
        elif backend == "persistent":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        elif backend == "flashinfer":
            # Deferred: importing flashinfer initializes CUDA at import time.
            from flashinfer.topk import top_k_ragged_transform

            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            offsets = torch.zeros(
                logits.shape[0], dtype=torch.int32, device=logits.device
            )
            # top_k_ragged_transform selects within [0, row_ends[i]) per
            # row and -1-fills past the row length.
            indices = top_k_ragged_transform(logits, offsets, row_ends, topk_tokens)
            topk_indices.copy_(indices)
        elif backend == "torch":
            # Debug reference: mask everything past each row's end, then topk.
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            cols = torch.arange(logits.shape[1], device=logits.device)
            masked = logits.masked_fill(
                cols.unsqueeze(0) >= row_ends.unsqueeze(1), float("-inf")
            )
            indices = masked.topk(topk_tokens, dim=-1).indices
            in_range = torch.arange(topk_tokens, device=logits.device).unsqueeze(
                0
            ) < row_ends.unsqueeze(1)
            indices = torch.where(in_range, indices, -1)
            topk_indices.copy_(indices)
        else:
            assert backend == "per_row", f"unknown topk backend: {backend}"
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
