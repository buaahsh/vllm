# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark YOCO L3's diff-v3 combine followed by the BF16 o_proj.

The default shape is the TP1 cross-attention path:

* attention: ``[M, 64, 128]``
* logical o_proj input: ``[M, 4096]``
* o_proj weight: ``[3072, 4096]``

Only PyTorch and Triton are imported so this can run on the standalone B200
tuning pod without a vLLM installation.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


def _training_diff_o_proj(
    attention: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = attention * torch.sigmoid(gate).unsqueeze(-1)
    output = output[:, 0::2] - output[:, 1::2]
    return F.linear(output.flatten(1), weight)


def _current_diff_o_proj(
    attention: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    first_gate = gate[:, 0::2]
    second_gate = gate[:, 1::2]
    output = attention[:, 0::2] * torch.sigmoid(first_gate).unsqueeze(-1)
    output -= attention[:, 1::2] * torch.sigmoid(second_gate).unsqueeze(-1)
    return F.linear(output.flatten(1), weight)


@triton.jit
def _diff_v3_kernel(
    attention_ptr,
    gate_ptr,
    output_ptr,
    gate_token_stride,
    HEAD_GROUP: tl.constexpr,
    NUM_HEAD_PAIRS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    group = tl.program_id(0)
    groups_per_token: tl.constexpr = NUM_HEAD_PAIRS // HEAD_GROUP
    token = group // groups_per_token
    pair_in_token = (group % groups_per_token) * HEAD_GROUP + tl.arange(0, HEAD_GROUP)[
        :, None
    ]
    dims = tl.arange(0, HEAD_DIM)[None, :]
    first_head_in_token = 2 * pair_in_token
    attention_base = token * 2 * NUM_HEAD_PAIRS * HEAD_DIM
    first_offset = attention_base + first_head_in_token * HEAD_DIM + dims
    gate_offset = token * gate_token_stride + first_head_in_token

    first_gate = tl.load(gate_ptr + gate_offset).to(tl.float32)
    second_gate = tl.load(gate_ptr + gate_offset + 1).to(tl.float32)
    first = tl.load(attention_ptr + first_offset).to(tl.float32)
    second = tl.load(attention_ptr + first_offset + HEAD_DIM).to(tl.float32)
    result = first * tl.sigmoid(first_gate) - second * tl.sigmoid(second_gate)
    output_offset = token * NUM_HEAD_PAIRS * HEAD_DIM + pair_in_token * HEAD_DIM + dims
    tl.store(output_ptr + output_offset, result)


def _fast_diff_v3(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    num_tokens = attention.shape[0]
    if num_tokens < 64:
        head_group, num_warps = 4, 4
    elif num_tokens < 512:
        head_group, num_warps = 8, 4
    else:
        head_group, num_warps = 16, 8
    output = torch.empty(
        (num_tokens, 32, 128),
        dtype=attention.dtype,
        device=attention.device,
    )
    _diff_v3_kernel[(num_tokens * 32 // head_group,)](
        attention,
        gate,
        output,
        gate.stride(0),
        HEAD_GROUP=head_group,
        NUM_HEAD_PAIRS=32,
        HEAD_DIM=128,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _fast_diff_o_proj(
    attention: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return F.linear(_fast_diff_v3(attention, gate).flatten(1), weight)


@triton.jit
def _fused_diff_o_proj_kernel(
    attention_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    attention_token_stride,
    gate_token_stride,
    NUM_HEAD_PAIRS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    program = tl.program_id(0)
    num_programs_m = tl.cdiv(num_tokens, BLOCK_M)
    num_programs_n = tl.cdiv(OUTPUT_DIM, BLOCK_N)
    programs_per_group = GROUP_M * num_programs_n
    group = program // programs_per_group
    first_program_m = group * GROUP_M
    group_size_m = tl.minimum(num_programs_m - first_program_m, GROUP_M)
    program_in_group = program % programs_per_group
    program_m = first_program_m + program_in_group % group_size_m
    program_n = program_in_group // group_size_m

    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_cols = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < num_tokens
    col_mask = output_cols < OUTPUT_DIM
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for pair in range(NUM_HEAD_PAIRS):
        first_head = 2 * pair
        first_gate = tl.load(
            gate_ptr + rows[:, None] * gate_token_stride + first_head,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        second_gate = tl.load(
            gate_ptr + rows[:, None] * gate_token_stride + first_head + 1,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        first = tl.load(
            attention_ptr
            + rows[:, None] * attention_token_stride
            + first_head * HEAD_DIM
            + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            attention_ptr
            + rows[:, None] * attention_token_stride
            + (first_head + 1) * HEAD_DIM
            + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # Match the standalone diff-v3 output's BF16 boundary before o_proj.
        logical_input = (
            first * tl.sigmoid(first_gate) - second * tl.sigmoid(second_gate)
        ).to(tl.bfloat16)
        weights = tl.load(
            weight_ptr
            + output_cols[None, :] * (NUM_HEAD_PAIRS * HEAD_DIM)
            + pair * HEAD_DIM
            + dims[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(logical_input, weights)

    tl.store(
        output_ptr + rows[:, None] * OUTPUT_DIM + output_cols[None, :],
        accumulator,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _fused_diff_o_proj(
    attention: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    num_tokens = attention.shape[0]
    output = torch.empty(
        (num_tokens, 3072), dtype=attention.dtype, device=attention.device
    )
    grid = (triton.cdiv(num_tokens, block_m) * triton.cdiv(3072, block_n),)
    _fused_diff_o_proj_kernel[grid](
        attention,
        gate,
        weight,
        output,
        num_tokens,
        attention.stride(0),
        gate.stride(0),
        NUM_HEAD_PAIRS=32,
        HEAD_DIM=128,
        OUTPUT_DIM=3072,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _capture(
    fn: Callable[[], torch.Tensor], graph_nodes: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        result = fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            result = fn()
    return graph, result


def _time_graph(
    graph: torch.cuda.CUDAGraph, iterations: int, graph_nodes: int
) -> float:
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * graph_nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128, 256, 512, 1024, 1410, 2048],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph-nodes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--gate-token-stride", type=int, default=8256)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    device = torch.device("cuda")
    weight = torch.randn(3072, 4096, device=device, dtype=torch.bfloat16) * 0.02
    weight_t = weight.t().contiguous()
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    configs = (
        (16, 64, 4, 2),
        (16, 128, 8, 2),
        (32, 64, 4, 2),
        (32, 128, 8, 2),
        (64, 64, 4, 2),
        (64, 128, 8, 2),
    )
    config_names = [
        f"m{block_m}n{block_n}w{warps}s{stages}"
        for block_m, block_n, warps, stages in configs
    ]
    print(
        "batch  linear_us  mm_t_us  training_us  current_us  fast_us  "
        + "  ".join(config_names)
        + "  base_exact  fused_max_abs"
    )

    for batch in args.batches:
        attention = torch.randn(batch, 64, 128, device=device, dtype=torch.bfloat16)
        gate_storage = torch.randn(
            batch,
            args.gate_token_stride,
            device=device,
            dtype=torch.bfloat16,
        )
        gate = gate_storage[:, :64]
        diff_input = (
            attention[:, 0::2] * torch.sigmoid(gate[:, 0::2]).unsqueeze(-1)
            - attention[:, 1::2] * torch.sigmoid(gate[:, 1::2]).unsqueeze(-1)
        ).flatten(1)

        training_fn = torch.compile(_training_diff_o_proj, fullgraph=True)
        current_fn = torch.compile(_current_diff_o_proj, fullgraph=True)
        fns = (
            lambda diff_input=diff_input: F.linear(diff_input, weight),
            lambda diff_input=diff_input: torch.mm(diff_input, weight_t),
            lambda training_fn=training_fn, attention=attention, gate=gate: training_fn(
                attention, gate, weight
            ),
            lambda current_fn=current_fn, attention=attention, gate=gate: current_fn(
                attention, gate, weight
            ),
            lambda attention=attention, gate=gate: _fast_diff_o_proj(
                attention, gate, weight
            ),
            *(
                lambda block_m=block_m,
                block_n=block_n,
                warps=warps,
                stages=stages,
                attention=attention,
                gate=gate: _fused_diff_o_proj(
                    attention,
                    gate,
                    weight,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=warps,
                    num_stages=stages,
                )
                for block_m, block_n, warps, stages in configs
            ),
        )
        captures = [_capture(fn, args.graph_nodes) for fn in fns]
        graphs = [capture[0] for capture in captures]
        results = [capture[1] for capture in captures]
        torch.accelerator.synchronize()
        base_exact = all(torch.equal(result, results[2]) for result in results[:5])
        fused_max_abs = max(
            (result.float() - results[2].float()).abs().max().item()
            for result in results[5:]
        )

        samples: list[list[float]] = [[] for _ in graphs]
        for round_idx in range(args.rounds):
            order = list(range(len(graphs)))
            if round_idx % 2:
                order.reverse()
            order = order[round_idx % len(order) :] + order[: round_idx % len(order)]
            for graph_idx in order:
                samples[graph_idx].append(
                    _time_graph(graphs[graph_idx], args.iterations, args.graph_nodes)
                )
        times = [statistics.median(values) for values in samples]
        print(
            f"{batch:5d} "
            + " ".join(f"{elapsed:11.3f}" for elapsed in times)
            + f" {base_exact} {fused_max_abs:.6f}"
        )


if __name__ == "__main__":
    main()
