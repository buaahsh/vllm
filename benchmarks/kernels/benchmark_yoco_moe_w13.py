# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune YOCO L3 BF16 expert GEMMs on B200.

The checked-in fused-MoE configuration is shared by W13 and W2.  This
benchmark can optimize W13, W2, or their sum while reporting both kernels.
For the W2-only objective, BLOCK_SIZE_M is fixed to W13's selected value so
the runtime can reuse one token dispatch. W2 includes the router-weight
multiply fused by the real inference path. Defaults reproduce TP4
(``K=1024,N=960``); pass ``--intermediate-size 3840`` for TP1.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

Config = dict[str, int]


def _load_vllm_fused_moe_kernel(source: Path):
    """Load the two Triton functions directly from vLLM's source file."""
    tree = ast.parse(source.read_text(), filename=str(source))
    wanted = {"write_zeros_to_output", "fused_moe_kernel"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise RuntimeError(f"Could not find {sorted(wanted)} in {source}")
    module = ast.Module(body=functions, type_ignores=[])
    namespace: dict[str, Any] = {
        "__file__": str(source),
        "__name__": "yoco_moe_kernel_standalone",
        "triton": triton,
        "tl": tl,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["fused_moe_kernel"]


def _canonical(config: Config) -> Config:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def _config_key(config: Config) -> tuple[int, ...]:
    config = _canonical(config)
    return tuple(config[key] for key in config)


def _load_seed_configs(paths: list[Path]) -> list[Config]:
    configs: dict[tuple[int, ...], Config] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for name, config in data.items():
            if name == "triton_version":
                continue
            canonical = _canonical(config)
            configs[_config_key(canonical)] = canonical
    return list(configs.values())


def _load_token_configs(path: Path) -> dict[int, Config]:
    data = json.loads(path.read_text())
    data.pop("triton_version", None)
    return {int(tokens): _canonical(config) for tokens, config in data.items()}


def _nearest_config(configs: dict[int, Config], tokens: int) -> Config:
    nearest = min(configs, key=lambda candidate: abs(candidate - tokens))
    return configs[nearest]


def _objective_value(objective: str, w13_us: float, w2_us: float) -> float:
    if objective == "w13":
        return w13_us
    if objective == "w2":
        return w2_us
    return w13_us + w2_us


def _base_search_space(seed_configs: list[Config]) -> list[Config]:
    configs = {_config_key(config): config for config in seed_configs}
    ranges = {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [64, 128, 256],
        "BLOCK_SIZE_K": [64, 128],
        "GROUP_SIZE_M": [1],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4],
    }
    keys = list(ranges)
    for values in product(*(ranges[key] for key in keys)):
        config = dict(zip(keys, values))
        configs[_config_key(config)] = config
    return list(configs.values())


def _refine_configs(best: list[Config]) -> list[Config]:
    configs: dict[tuple[int, ...], Config] = {}
    for seed in best:
        for group_m, stages, warps in product([1, 8, 16, 32, 64], [2, 3, 4, 5], [4, 8]):
            config = {
                **seed,
                "GROUP_SIZE_M": group_m,
                "num_stages": stages,
                "num_warps": warps,
            }
            configs[_config_key(config)] = config
    return list(configs.values())


@dataclass
class Assignment:
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    post_pad: torch.Tensor


def _make_assignment(topk_ids: torch.Tensor, block_m: int, experts: int) -> Assignment:
    flat = topk_ids.flatten().cpu()
    assignments = flat.numel()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)
    padded_counts = (
        torch.div(counts + block_m - 1, block_m, rounding_mode="floor") * block_m
    )
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), padded_counts.cumsum(dim=0)]
    )

    max_padded = assignments + experts * (block_m - 1)
    if assignments < experts:
        max_padded = min(assignments * block_m, max_padded)
    sorted_ids = torch.full((max_padded,), assignments, dtype=torch.int32)
    expert_ids = torch.full(
        ((max_padded + block_m - 1) // block_m,), -1, dtype=torch.int32
    )
    for expert in range(experts):
        count = int(counts[expert])
        start = int(offsets[expert])
        if count:
            source_start = int(counts[:expert].sum())
            sorted_ids[start : start + count] = order[
                source_start : source_start + count
            ].to(torch.int32)
            blocks = int(padded_counts[expert]) // block_m
            expert_ids[start // block_m : start // block_m + blocks] = expert

    return Assignment(
        sorted_ids=sorted_ids.cuda(),
        expert_ids=expert_ids.cuda(),
        post_pad=torch.tensor([int(offsets[-1])], dtype=torch.int32, device="cuda"),
    )


def _launch(
    kernel,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    routed_weights: torch.Tensor | None,
    assignment: Assignment,
    topk: int,
    config: Config,
) -> None:
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    em = assignment.sorted_ids.numel()
    grid = (triton.cdiv(em, block_m) * triton.cdiv(b.shape[1], block_n),)
    kernel[grid](
        a,
        b,
        c,
        None,
        None,
        None,
        routed_weights,
        assignment.sorted_ids,
        assignment.expert_ids,
        assignment.post_pad,
        b.shape[1],
        b.shape[2],
        em,
        a.shape[0] * topk,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        naive_block_assignment=False,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=routed_weights is not None,
        top_k=topk,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        HAS_BIAS=False,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def _time_kernel(
    kernel,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    routed_weights: torch.Tensor | None,
    assignment: Assignment,
    topk: int,
    config: Config,
    *,
    graph_nodes: int,
    repeats: int,
    rounds: int,
) -> float:
    for _ in range(3):
        _launch(kernel, a, b, c, routed_weights, assignment, topk, config)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            _launch(kernel, a, b, c, routed_weights, assignment, topk, config)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (repeats * graph_nodes))
    graph.reset()
    return statistics.median(samples)


def _benchmark_config(
    kernel,
    a1: torch.Tensor,
    w1: torch.Tensor,
    c1: torch.Tensor,
    a2: torch.Tensor,
    w2: torch.Tensor,
    c2: torch.Tensor,
    routed_weights: torch.Tensor,
    assignment: Assignment,
    config: Config,
    *,
    graph_nodes: int,
    repeats: int,
    rounds: int,
) -> tuple[float, float]:
    w13_us = _time_kernel(
        kernel,
        a1,
        w1,
        c1,
        None,
        assignment,
        8,
        config,
        graph_nodes=graph_nodes,
        repeats=repeats,
        rounds=rounds,
    )
    w2_us = _time_kernel(
        kernel,
        a2,
        w2,
        c2,
        routed_weights,
        assignment,
        1,
        config,
        graph_nodes=graph_nodes,
        repeats=repeats,
        rounds=rounds,
    )
    return w13_us, w2_us


def _benchmark_objective(
    objective: str,
    kernel,
    a1: torch.Tensor,
    w1: torch.Tensor,
    c1: torch.Tensor,
    a2: torch.Tensor,
    w2: torch.Tensor,
    c2: torch.Tensor,
    routed_weights: torch.Tensor,
    assignment: Assignment,
    config: Config,
    *,
    graph_nodes: int,
    repeats: int,
    rounds: int,
) -> tuple[float, float]:
    """Time only the requested kernel during the broad config screen."""
    if objective == "w13":
        return (
            _time_kernel(
                kernel,
                a1,
                w1,
                c1,
                None,
                assignment,
                8,
                config,
                graph_nodes=graph_nodes,
                repeats=repeats,
                rounds=rounds,
            ),
            0.0,
        )
    if objective == "w2":
        return (
            0.0,
            _time_kernel(
                kernel,
                a2,
                w2,
                c2,
                routed_weights,
                assignment,
                1,
                config,
                graph_nodes=graph_nodes,
                repeats=repeats,
                rounds=rounds,
            ),
        )
    return _benchmark_config(
        kernel,
        a1,
        w1,
        c1,
        a2,
        w2,
        c2,
        routed_weights,
        assignment,
        config,
        graph_nodes=graph_nodes,
        repeats=repeats,
        rounds=rounds,
    )


def _default_config(tokens: int) -> Config:
    if tokens <= 32:
        block_m = 16
    elif tokens <= 96:
        block_m = 32
    elif tokens <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if tokens <= 64 else 128,
        "BLOCK_SIZE_K": 128 if tokens <= 64 else 64,
        "GROUP_SIZE_M": 16 if tokens // 128 > 128 else 1,
        "num_warps": 4 if tokens <= 128 else 8,
        "num_stages": 4 if tokens <= 32 else 3,
    }


def _make_problem(tokens: int, experts: int, n: int, k: int):
    random_order = torch.rand(tokens, experts, device="cuda").argsort(dim=1)
    topk_ids = random_order[:, :8].to(torch.int32).contiguous()
    routed_weights = torch.rand((tokens, 8), dtype=torch.float32, device="cuda")
    routed_weights /= routed_weights.sum(dim=-1, keepdim=True)
    a1 = torch.randn((tokens, k), dtype=torch.bfloat16, device="cuda")
    a2 = torch.randn((tokens * 8, n), dtype=torch.bfloat16, device="cuda")
    c1 = torch.empty((tokens, 8, 2 * n), dtype=torch.bfloat16, device="cuda")
    c2 = torch.empty((tokens, 8, k), dtype=torch.bfloat16, device="cuda")
    return topk_ids, routed_weights, a1, a2, c1, c2


def _validate_kernel(
    kernel,
    w1: torch.Tensor,
    w2: torch.Tensor,
    experts: int,
    n: int,
    k: int,
) -> None:
    problem = _make_problem(1, experts, n, k)
    topk_ids, routed_weights, a1, a2, c1, c2 = problem
    config = _default_config(1)
    assignment = _make_assignment(topk_ids, config["BLOCK_SIZE_M"], experts)
    _launch(kernel, a1, w1, c1, None, assignment, 8, config)
    _launch(kernel, a2, w2, c2, routed_weights, assignment, 1, config)
    torch.accelerator.synchronize()

    routed_experts = topk_ids.flatten().to(torch.int64)
    token_rows = torch.zeros(8, dtype=torch.int64, device="cuda")
    w13_reference = torch.bmm(
        w1[routed_experts].float(), a1[token_rows].float().unsqueeze(-1)
    ).squeeze(-1)
    w2_reference = torch.bmm(
        w2[routed_experts].float(), a2.float().unsqueeze(-1)
    ).squeeze(-1) * routed_weights.flatten().float().unsqueeze(-1)
    w13_actual = c1.view(8, 2 * n).float()
    w2_actual = c2.view(8, k).float()
    torch.testing.assert_close(w13_actual, w13_reference, rtol=0.02, atol=1.0)
    torch.testing.assert_close(w2_actual, w2_reference, rtol=0.02, atol=1.0)
    print(
        "correctness=pass "
        f"w13_max_abs={(w13_actual - w13_reference).abs().max().item():.6f} "
        f"w2_max_abs={(w2_actual - w2_reference).abs().max().item():.6f}"
    )


def _screen(
    kernel,
    configs: list[Config],
    problem,
    assignments: dict[int, Assignment],
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    keep: int,
    objective: str,
) -> list[tuple[float, float, float, Config]]:
    _, routed_weights, a1, a2, c1, c2 = problem
    results = []
    for index, config in enumerate(configs, 1):
        try:
            w13_us, w2_us = _benchmark_objective(
                objective,
                kernel,
                a1,
                w1,
                c1,
                a2,
                w2,
                c2,
                routed_weights,
                assignments[config["BLOCK_SIZE_M"]],
                config,
                graph_nodes=5,
                repeats=3,
                rounds=1,
            )
        except Exception as error:
            print(f"skip config={config}: {type(error).__name__}: {error}")
            continue
        score = _objective_value(objective, w13_us, w2_us)
        results.append((score, w13_us, w2_us, config))
        if index % 25 == 0:
            best_us = min(results, key=lambda item: item[0])[0]
            print(f"screened={index}/{len(configs)} best_us={best_us:.3f}")
    results.sort(key=lambda item: item[0])
    return results[:keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--seed-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=960,
        help="Per-TP-rank expert intermediate width N.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Latent routed-expert input/output width K.",
    )
    parser.add_argument("--base-limit", type=int)
    parser.add_argument(
        "--objective",
        choices=("total", "w13", "w2"),
        default="total",
        help="Kernel time used to select the winning configuration.",
    )
    parser.add_argument(
        "--w13-config",
        type=Path,
        help=(
            "Shared/W13 config map whose BLOCK_SIZE_M must be reused when "
            "--objective=w2."
        ),
    )
    parser.add_argument(
        "--benchmark-only-config",
        type=Path,
        help="Skip tuning and remeasure the nearest config from this JSON.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            8192,
            16384,
            32768,
        ],
    )
    args = parser.parse_args()

    torch.manual_seed(20260831)
    kernel = _load_vllm_fused_moe_kernel(args.kernel_source)
    experts = args.experts
    n = args.intermediate_size
    k = args.hidden_size
    w1 = torch.randn((experts, 2 * n, k), dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn((experts, k, n), dtype=torch.bfloat16, device="cuda")
    _validate_kernel(kernel, w1, w2, experts, n, k)
    seeds = _load_seed_configs(args.seed_config)
    base_configs = _base_search_space(seeds)
    if args.base_limit is not None:
        base_configs = base_configs[: args.base_limit]
    if args.objective == "w2" and args.w13_config is None:
        parser.error("--objective=w2 requires --w13-config")
    w13_configs = (
        _load_token_configs(args.w13_config) if args.w13_config is not None else None
    )

    print(
        f"device={current_platform.get_device_name()} triton={triton.__version__} "
        f"E={experts} N={n} K={k} topk=8 objective={args.objective} "
        f"base_configs={len(base_configs)}"
    )

    benchmark_only_configs = None
    if args.benchmark_only_config is not None:
        benchmark_only_configs = json.loads(args.benchmark_only_config.read_text())
        benchmark_only_configs.pop("triton_version", None)
        benchmark_only_configs = {
            int(tokens): _canonical(config)
            for tokens, config in benchmark_only_configs.items()
        }

    output: dict[str, Any] = {"triton_version": triton.__version__}
    for tokens in args.tokens:
        problem = _make_problem(tokens, experts, n, k)
        topk_ids = problem[0]
        default = (
            _nearest_config(w13_configs, tokens)
            if args.objective == "w2" and w13_configs is not None
            else _default_config(tokens)
        )
        token_base_configs = base_configs
        if args.objective == "w2":
            token_base_configs = [
                config
                for config in base_configs
                if config["BLOCK_SIZE_M"] == default["BLOCK_SIZE_M"]
            ]
        if benchmark_only_configs is not None:
            nearest = min(
                benchmark_only_configs,
                key=lambda candidate: abs(candidate - tokens),
            )
            tuned = benchmark_only_configs[nearest]
            if (
                args.objective == "w2"
                and tuned["BLOCK_SIZE_M"] != default["BLOCK_SIZE_M"]
            ):
                raise ValueError(
                    "W2 config changes BLOCK_SIZE_M and cannot reuse W13 dispatch: "
                    f"tokens={tokens}, W13={default}, W2={tuned}"
                )
            assignments = {
                block_m: _make_assignment(topk_ids, block_m, experts)
                for block_m in {
                    tuned["BLOCK_SIZE_M"],
                    default["BLOCK_SIZE_M"],
                }
            }
            _, routed_weights, a1, a2, c1, c2 = problem
            tuned_w13, tuned_w2 = _benchmark_config(
                kernel,
                a1,
                w1,
                c1,
                a2,
                w2,
                c2,
                routed_weights,
                assignments[tuned["BLOCK_SIZE_M"]],
                tuned,
                graph_nodes=20,
                repeats=50,
                rounds=7,
            )
            default_w13, default_w2 = _benchmark_config(
                kernel,
                a1,
                w1,
                c1,
                a2,
                w2,
                c2,
                routed_weights,
                assignments[default["BLOCK_SIZE_M"]],
                default,
                graph_nodes=20,
                repeats=50,
                rounds=7,
            )
            tuned_total = tuned_w13 + tuned_w2
            default_total = default_w13 + default_w2
            tuned_score = _objective_value(args.objective, tuned_w13, tuned_w2)
            default_score = _objective_value(args.objective, default_w13, default_w2)
            output[str(tokens)] = tuned
            print(
                f"tokens={tokens} nearest={nearest} "
                f"baseline={default_total:.3f}us "
                f"(w13={default_w13:.3f},w2={default_w2:.3f}) "
                f"tuned={tuned_total:.3f}us "
                f"(w13={tuned_w13:.3f},w2={tuned_w2:.3f}) "
                f"objective_speedup={default_score / tuned_score:.3f} "
                f"config={tuned}"
            )
            args.output.write_text(json.dumps(output, indent=4) + "\n")
            del problem, assignments
            torch.accelerator.empty_cache()
            continue

        assignments = {
            block_m: _make_assignment(topk_ids, block_m, experts)
            for block_m in {
                config["BLOCK_SIZE_M"] for config in token_base_configs + [default]
            }
        }
        first_pass = _screen(
            kernel,
            token_base_configs,
            problem,
            assignments,
            w1,
            w2,
            keep=5,
            objective=args.objective,
        )
        refinement = _refine_configs([item[3] for item in first_pass])
        second_pass = _screen(
            kernel,
            refinement,
            problem,
            assignments,
            w1,
            w2,
            keep=5,
            objective=args.objective,
        )

        finalists = {_config_key(item[3]): item[3] for item in first_pass + second_pass}
        finalists[_config_key(default)] = default
        final_results = []
        _, routed_weights, a1, a2, c1, c2 = problem
        for config in finalists.values():
            w13_us, w2_us = _benchmark_config(
                kernel,
                a1,
                w1,
                c1,
                a2,
                w2,
                c2,
                routed_weights,
                assignments[config["BLOCK_SIZE_M"]],
                config,
                graph_nodes=20,
                repeats=20,
                rounds=5,
            )
            score = _objective_value(args.objective, w13_us, w2_us)
            final_results.append((score, w13_us, w2_us, config))
        final_results.sort(key=lambda item: item[0])
        best_score, best_w13, best_w2, best_config = final_results[0]
        base_score, base_w13, base_w2, _ = next(
            item
            for item in final_results
            if _config_key(item[3]) == _config_key(default)
        )
        output[str(tokens)] = best_config
        best_total = best_w13 + best_w2
        base_total = base_w13 + base_w2
        print(
            f"tokens={tokens} baseline={base_total:.3f}us "
            f"(w13={base_w13:.3f},w2={base_w2:.3f}) "
            f"best={best_total:.3f}us (w13={best_w13:.3f},w2={best_w2:.3f}) "
            f"objective_speedup={base_score / best_score:.3f} "
            f"config={best_config}"
        )
        args.output.write_text(json.dumps(output, indent=4) + "\n")
        del problem, assignments
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
