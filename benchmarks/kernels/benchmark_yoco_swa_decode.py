# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark YOCO L3 TP=4 sliding-window decode attention.

This compares the paged-cache FlashAttention 2/4 paths with vLLM's Triton
unified attention kernel.  The default shape is the per-rank shape used by the
30B-A3B L3 model: 16 query heads, 2 KV heads, head size 128, and a 513-token
sliding window (``window_size=(512, 0)``).
"""

import argparse
import gc
import statistics
from collections.abc import Callable

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.vllm_flash_attn import flash_attn_varlen_func

Kernel = Callable[[], None]


def _event_us(fn: Kernel, warmup: int, iterations: int, samples: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(timings)


def _graph_us(fn: Kernel, warmup: int, iterations: int, samples: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(timings)


@torch.inference_mode()
def benchmark_case(
    batch_size: int,
    context_len: int,
    kernel_names: list[str],
    warmup: int,
    iterations: int,
    samples: int,
) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    num_query_heads = 16
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    window_size = [512, 0]
    scale = head_size**-0.5

    blocks_per_seq = (context_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    generator = torch.Generator(device=device).manual_seed(
        batch_size * 100_000 + context_len
    )
    query = torch.randn(
        batch_size,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    value_cache = torch.randn(
        key_cache.shape, dtype=dtype, device=device, generator=generator
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        batch_size, blocks_per_seq
    )
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)

    outputs = {
        "fa2": torch.empty_like(query),
        "fa4": torch.empty_like(query),
        "triton": torch.empty_like(query),
    }

    def fa(version: int, num_splits: int, output: torch.Tensor) -> None:
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=context_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_table,
            softcap=0.0,
            fa_version=version,
            num_splits=num_splits,
        )

    def triton() -> None:
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=outputs["triton"],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=context_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

    all_kernels: dict[str, Kernel] = {
        "fa2": lambda: fa(2, 0, outputs["fa2"]),
        "fa4": lambda: fa(4, 1, outputs["fa4"]),
        "triton": triton,
    }
    kernels = {name: all_kernels[name] for name in kernel_names}
    successful: list[str] = []
    for name, fn in kernels.items():
        try:
            fn()
            torch.accelerator.synchronize()
            successful.append(name)
        except Exception as exc:
            print(
                f"ERROR,batch={batch_size},context={context_len},kernel={name},"
                f"error={exc!r}"
            )

    reference_name = "fa4" if "fa4" in successful else successful[0]
    reference = outputs[reference_name].float()
    for name in successful:
        error = (outputs[name].float() - reference).abs()
        print(
            f"ACCURACY,batch={batch_size},context={context_len},kernel={name},"
            f"reference={reference_name},max_abs={error.max().item():.8g},"
            f"mean_abs={error.mean().item():.8g}"
        )

    for name in successful:
        fn = kernels[name]
        try:
            event_us = _event_us(fn, warmup, iterations, samples)
            graph_us = _graph_us(fn, warmup, iterations, samples)
            print(
                f"RESULT,batch={batch_size},context={context_len},kernel={name},"
                f"event_us={event_us:.4f},graph_us={graph_us:.4f}"
            )
        except Exception as exc:
            print(
                f"ERROR,batch={batch_size},context={context_len},kernel={name},"
                f"phase=benchmark,error={exc!r}"
            )

    gc.collect()
    torch.accelerator.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[128, 513, 1024, 4096]
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["fa2", "fa4", "triton"],
        default=["fa4", "triton"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    capability = current_platform.get_device_capability()
    assert capability is not None
    print(
        f"DEVICE,name={current_platform.get_device_name()},"
        f"capability={capability.major}.{capability.minor},"
        f"torch={torch.__version__},cuda={torch.version.cuda}"
    )
    for context_len in args.contexts:
        for batch_size in args.batches:
            benchmark_case(
                batch_size,
                context_len,
                args.kernels,
                args.warmup,
                args.iterations,
                args.samples,
            )


if __name__ == "__main__":
    main()
