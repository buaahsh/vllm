# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 TP4 benchmark for YOCO's post-o_proj BF16 all-reduce.

Run with, for example::

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
        benchmark_yoco_tp4_all_reduce.py --backend nccl

The benchmark intentionally keeps row 0 of every rank's input identical
across token counts.  This lets it check batch invariance in addition to
latency.  When ``--vllm-library`` is supplied, it also benchmarks vLLM's
IPC custom all-reduce kernel without importing the vLLM Python package.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.platforms import current_platform

HIDDEN_SIZE = 3072
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 224, 256, 512, 1024, 2048, 4096)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("nccl", "custom", "symm"),
        default="nccl",
    )
    parser.add_argument(
        "--batch-invariant-nccl",
        action="store_true",
        help="Apply the NCCL environment used by VLLM_BATCH_INVARIANT.",
    )
    parser.add_argument(
        "--vllm-library",
        help="Path to vLLM's _C.abi3.so; required by --backend custom.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(TOKEN_COUNTS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--terse", action="store_true")
    parser.add_argument(
        "--graph-nodes",
        type=int,
        default=20,
        help="All-reduce nodes captured in one CUDA graph.",
    )
    return parser.parse_args()


def _configure_batch_invariant_nccl() -> None:
    """Match vLLM's VLLM_BATCH_INVARIANT NCCL policy before PG init."""
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"


def _init_distributed() -> tuple[int, int, dist.ProcessGroup]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"This benchmark requires TP4, got world_size={world_size}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    return rank, local_rank, cpu_group


def _make_rank_inputs(local_rank: int, max_tokens: int) -> torch.Tensor:
    generator = torch.Generator(device=f"cuda:{local_rank}")
    generator.manual_seed(0x5A17 + local_rank)
    return torch.randn(
        (max_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=f"cuda:{local_rank}",
        generator=generator,
    )


def _bf16_order_references(rank_values: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    refs: dict[str, torch.Tensor] = {}
    refs["fp32_then_bf16"] = sum(value.float() for value in rank_values).to(
        torch.bfloat16
    )
    for start in range(4):
        order = [(start + offset) % 4 for offset in range(4)]
        value = rank_values[order[0]].clone()
        for index in order[1:]:
            value = value + rank_values[index]
        refs["bf16_seq_" + "".join(str(index) for index in order)] = value
    refs["bf16_pair_01_23"] = (rank_values[0] + rank_values[1]) + (
        rank_values[2] + rank_values[3]
    )
    refs["bf16_pair_02_13"] = (rank_values[0] + rank_values[2]) + (
        rank_values[1] + rank_values[3]
    )
    return refs


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "bitwise": bool(torch.equal(actual, expected)),
        "different": int(
            torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16))
        ),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


@dataclass
class CustomAllReduce:
    ptr: int
    meta_ptrs: list[int]
    buffer_ptrs: list[int]
    rank_data: torch.Tensor
    max_size: int
    rank: int
    cpu_group: dist.ProcessGroup
    ops: object

    @classmethod
    def create(
        cls,
        library: str,
        rank: int,
        local_rank: int,
        cpu_group: dist.ProcessGroup,
        max_size: int,
    ) -> CustomAllReduce:
        torch.ops.load_library(library)
        # The standalone benchmark extension registers this namespace.  A
        # production vLLM build uses the otherwise identical _C_custom_ar
        # namespace; accepting either keeps the harness useful in both cases.
        try:
            ops = torch.ops.yoco_custom_ar
            ops.meta_size()
        except (AttributeError, RuntimeError):
            ops = torch.ops._C_custom_ar

        def shared_buffer(size: int) -> list[int]:
            pointer, handle = ops.allocate_shared_buffer_and_handle(size)
            handles: list[object | None] = [None] * 4
            dist.all_gather_object(handles, handle, group=cpu_group)
            pointers: list[int] = []
            for peer_rank, peer_handle in enumerate(handles):
                if peer_rank == rank:
                    pointers.append(pointer)
                else:
                    pointers.append(ops.open_mem_handle(peer_handle))
            return pointers

        meta_ptrs = shared_buffer(ops.meta_size() + max_size)
        buffer_ptrs = shared_buffer(max_size)
        rank_data = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=f"cuda:{local_rank}",
        )
        ptr = ops.init_custom_ar(meta_ptrs, rank_data, rank, True)
        ops.register_buffer(ptr, buffer_ptrs)
        return cls(
            ptr,
            meta_ptrs,
            buffer_ptrs,
            rank_data,
            max_size,
            rank,
            cpu_group,
            ops,
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.nbytes >= self.max_size:
            raise RuntimeError(
                f"custom all-reduce input {value.nbytes} exceeds {self.max_size}"
            )
        output = torch.empty_like(value)
        self.ops.all_reduce(
            self.ptr,
            value,
            output,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return output

    def graph_latency_us(
        self,
        value: torch.Tensor,
        graph_nodes: int,
        warmup: int,
        repeats: int,
    ) -> float:
        output = torch.empty_like(value)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(graph_nodes):
                # A zero registered-buffer pointer is the CUDA-graph path: the
                # C++ communicator records the input address for IPC mapping.
                self.ops.all_reduce(self.ptr, value, output, 0, 0)

        handles_raw, offsets = self.ops.get_graph_buffer_ipc_meta(self.ptr)
        local_data = [list(handles_raw), list(offsets)]
        all_data: list[list[list[int]] | None] = [None] * 4
        dist.all_gather_object(all_data, local_data, group=self.cpu_group)
        handles = [entry[0] for entry in all_data if entry is not None]
        all_offsets = [entry[1] for entry in all_data if entry is not None]
        self.ops.register_graph_buffers(self.ptr, handles, all_offsets)

        for _ in range(warmup):
            graph.replay()
        torch.accelerator.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / (repeats * graph_nodes)

    def close(self) -> None:
        self.ops.dispose(self.ptr)
        self.ops.free_shared_buffer(self.meta_ptrs[self.rank])
        self.ops.free_shared_buffer(self.buffer_ptrs[self.rank])


@dataclass
class SymmAllReduce:
    buffer: torch.Tensor
    group_name: str
    max_size: int

    @classmethod
    def create(
        cls,
        local_rank: int,
        cpu_group: dist.ProcessGroup,
        max_size: int,
    ) -> SymmAllReduce:
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(
            max_size // torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
            device=f"cuda:{local_rank}",
        )
        group_name = cpu_group.group_name
        handle = symm_mem.rendezvous(buffer, group_name)
        if handle.multicast_ptr == 0:
            raise RuntimeError("B200 symmetric-memory multicast mapping is unavailable")
        return cls(buffer, group_name, max_size)

    def run_to(self, value: torch.Tensor, output: torch.Tensor) -> None:
        if value.nbytes >= self.max_size:
            raise RuntimeError(
                f"symmetric all-reduce input {value.nbytes} exceeds {self.max_size}"
            )
        workspace = self.buffer[: value.numel()]
        workspace.copy_(value.view(-1))
        # SM100 TP4 is not in vLLM's multimem world-size table, so this is
        # exactly the production SymmMemCommunicator algorithm.
        torch.ops.symm_mem.two_shot_all_reduce_(
            workspace,
            "sum",
            self.group_name,
        )
        output.copy_(workspace.view_as(output))

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(value)
        self.run_to(value, output)
        return output

    def graph_latency_us(
        self,
        value: torch.Tensor,
        graph_nodes: int,
        warmup: int,
        repeats: int,
    ) -> float:
        output = torch.empty_like(value)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(graph_nodes):
                self.run_to(value, output)
        for _ in range(warmup):
            graph.replay()
        torch.accelerator.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / (repeats * graph_nodes)


def _nccl_all_reduce(value: torch.Tensor) -> torch.Tensor:
    output = value.clone()
    dist.all_reduce(output)
    return output


def _time_eager(
    operation: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        operation(value)
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        operation(value)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _time_nccl_cuda_graph(
    value: torch.Tensor,
    graph_nodes: int,
    warmup: int,
    repeats: int,
) -> float:
    work = value.clone()
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            dist.all_reduce(work)
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (repeats * graph_nodes)


def main() -> None:
    args = _parse_args()
    if args.batch_invariant_nccl:
        _configure_batch_invariant_nccl()
    rank, local_rank, cpu_group = _init_distributed()
    max_tokens = max(args.tokens)
    rank_inputs = _make_rank_inputs(local_rank, max_tokens)

    custom: CustomAllReduce | None = None
    symm: SymmAllReduce | None = None
    if args.backend == "custom":
        if not args.vllm_library:
            raise ValueError("--backend custom requires --vllm-library")
        # Production SM100 TP4 caps custom AR at 2 MiB when symmetric-memory
        # fallback is available.  Allocate slightly more for the strict '<'
        # threshold and reject larger benchmark shapes below.
        custom = CustomAllReduce.create(
            args.vllm_library,
            rank,
            local_rank,
            cpu_group,
            max_size=2 * 1024 * 1024,
        )
        operation = custom
    elif args.backend == "symm":
        symm = SymmAllReduce.create(
            local_rank,
            cpu_group,
            max_size=32 * 1024 * 1024,
        )
        operation = symm
    else:
        operation = _nccl_all_reduce

    base_row: torch.Tensor | None = None
    results: list[dict[str, object]] = []
    for tokens in args.tokens:
        value = rank_inputs[:tokens]
        if custom is not None and value.nbytes >= custom.max_size:
            continue
        if symm is not None and value.nbytes >= symm.max_size:
            continue

        output = operation(value)
        torch.accelerator.synchronize()

        rank_outputs = [torch.empty_like(output) for _ in range(4)]
        dist.all_gather(rank_outputs, output)
        same_across_ranks = all(torch.equal(output, peer) for peer in rank_outputs)

        local_row = value[0].clone()
        rank_rows = [torch.empty_like(local_row) for _ in range(4)]
        dist.all_gather(rank_rows, local_row)
        refs = _bf16_order_references(rank_rows)

        row = output[0].detach().cpu()
        if base_row is None:
            base_row = row.clone()
        batch_invariant = torch.equal(row, base_row)
        batch_diff = (row.float() - base_row.float()).abs()

        latency_us = _time_eager(
            operation,
            value,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if custom is not None:
            graph_latency_us = custom.graph_latency_us(
                value,
                graph_nodes=args.graph_nodes,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        elif symm is not None:
            graph_latency_us = symm.graph_latency_us(
                value,
                graph_nodes=args.graph_nodes,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:
            graph_latency_us = _time_nccl_cuda_graph(
                value,
                graph_nodes=args.graph_nodes,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        if rank == 0:
            result: dict[str, object] = {
                "backend": args.backend,
                "tokens": tokens,
                "bytes": value.nbytes,
                "latency_us": latency_us,
                "graph_latency_us": graph_latency_us,
                "same_across_ranks": same_across_ranks,
                "batch_invariant_row0": batch_invariant,
                "batch_row0_different": int(
                    torch.count_nonzero(
                        row.view(torch.int16) != base_row.view(torch.int16)
                    )
                ),
                "batch_row0_max_abs": float(batch_diff.max().item()),
                "references": {
                    name: _compare(row, reference.cpu())
                    for name, reference in refs.items()
                },
            }
            results.append(result)
            display = result
            if args.terse:
                display = {
                    key: result[key]
                    for key in (
                        "backend",
                        "tokens",
                        "bytes",
                        "latency_us",
                        "graph_latency_us",
                        "same_across_ranks",
                        "batch_invariant_row0",
                        "batch_row0_different",
                        "batch_row0_max_abs",
                    )
                }
            print(json.dumps(display, sort_keys=True), flush=True)
        dist.barrier()

    if rank == 0:
        summary = {
            "backend": args.backend,
            "torch": torch.__version__,
            "device": current_platform.get_device_name(local_rank),
            "results": (
                results
                if not args.terse
                else [
                    {
                        "tokens": result["tokens"],
                        "graph_latency_us": result["graph_latency_us"],
                    }
                    for result in results
                ]
            ),
        }
        print("SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)

    if custom is not None:
        custom.close()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
