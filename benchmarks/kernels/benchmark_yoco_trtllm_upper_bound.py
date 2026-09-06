# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare YOCO's clamped CUTLASS and TRTLLM BF16 MoE implementations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    ActivationType,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    convert_to_block_layout,
    get_w2_permute_indices_with_cache,
)


@torch.compile
def _shared_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float().clamp(max=10.0)
    up = up.float().clamp(min=-10.0, max=10.0)
    return (F.silu(gate) * up).to(gate_up.dtype)


def _capture(fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _convert_to_trtllm_layout(
    w13: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cache: dict[torch.Size, torch.Tensor] = {}
    converted_w13 = []
    converted_w2 = []
    for expert in range(w13.shape[0]):
        w13_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13[expert].view(torch.uint8), 128
        )
        w2_indices = get_w2_permute_indices_with_cache(
            cache, w2[expert].view(torch.uint8), 128
        )
        expert_w13 = (
            w13[expert]
            .clone()
            .view(torch.uint8)[w13_indices.to(w13.device)]
            .contiguous()
        )
        expert_w2 = (
            w2[expert].clone().view(torch.uint8)[w2_indices.to(w2.device)].contiguous()
        )
        converted_w13.append(
            convert_to_block_layout(expert_w13.view(torch.uint8), 128).view(
                torch.bfloat16
            )
        )
        converted_w2.append(
            convert_to_block_layout(expert_w2.view(torch.uint8), 128).view(
                torch.bfloat16
            )
        )
    return torch.stack(converted_w13).contiguous(), torch.stack(
        converted_w2
    ).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--tune-max-num-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cache")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Load cached tactics without profiling missing shapes.",
    )
    parser.add_argument(
        "--overlap-shared",
        action="store_true",
        help="Time TRTLLM concurrently with YOCO's MLP shared expert.",
    )
    parser.add_argument(
        "--sweep-tactics",
        action="store_true",
        help="Benchmark every legal TRTLLM tactic at the largest token count.",
    )
    parser.add_argument("--tactic-repeats", type=int, default=20)
    parser.add_argument("--tactic-rounds", type=int, default=3)
    parser.add_argument("--print-all-tactics", action="store_true")
    parser.add_argument("--reference-tactic", type=int, nargs=2)
    parser.add_argument(
        "--tactic-candidates",
        nargs="+",
        help="Optional TILE:CONFIG identities to benchmark instead of all tactics.",
    )
    parser.add_argument("--routing-capture", type=Path)
    parser.add_argument(
        "--compare-routing-mode",
        action="store_true",
        help=(
            "Compare YOCO's fused external Top-K plus pre-routed TRTLLM "
            "against TRTLLM's monolithic logits-routing entry point."
        ),
    )
    parser.add_argument(
        "--route-samples",
        type=int,
        default=32,
        help="Number of layer/step routing matrices to replay per graph.",
    )
    args = parser.parse_args()

    experts = 128
    topk = 8
    hidden = 1024
    intermediate = 3840
    max_tokens = max(args.tokens)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260903)
    inputs = torch.randn(
        max_tokens,
        hidden,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    routing = torch.randn(
        max_tokens,
        experts,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_logits, topk_ids = torch.topk(routing, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    captured_topk_ids: torch.Tensor | None = None
    captured_topk_weights: torch.Tensor | None = None
    if args.routing_capture is not None:
        capture = json.loads(args.routing_capture.read_text())
        case = capture["cases"][str(max_tokens)]
        request_routes = [record["routed_experts"] for record in case["requests"]]
        routes = torch.tensor(request_routes, dtype=torch.int32)
        if (
            routes.ndim != 4
            or routes.shape[0] != max_tokens
            or routes.shape[-1] != topk
        ):
            raise ValueError(
                f"invalid captured route shape {tuple(routes.shape)} for M={max_tokens}"
            )
        # [request, step, layer, topk] -> [step * layer, request, topk].
        routes = routes.permute(1, 2, 0, 3).reshape(-1, max_tokens, topk)
        sample_count = min(args.route_samples, routes.shape[0])
        sample_indices = torch.linspace(
            0, routes.shape[0] - 1, sample_count, dtype=torch.long
        )
        captured_topk_ids = routes.index_select(0, sample_indices).to(device)
        captured_topk_weights = torch.full(
            captured_topk_ids.shape,
            1.0 / topk,
            dtype=torch.float32,
            device=device,
        )
        topk_ids = captured_topk_ids[0]
        topk_weights = captured_topk_weights[0]
    # Force a row well beyond limit=10 so this is a real clamp test rather
    # than merely comparing two backends on the inactive-clamp region.
    inputs[0].fill_(10.0)

    # Start in the [up, gate] order required by both FlashInfer backends.
    w13 = torch.empty(
        experts,
        2 * intermediate,
        hidden,
        dtype=dtype,
        device=device,
    ).fill_(0.01)
    w2 = torch.empty(
        experts,
        hidden,
        intermediate,
        dtype=dtype,
        device=device,
    ).fill_(0.001)
    trtllm_w13, trtllm_w2 = _convert_to_trtllm_layout(w13, w2)
    alpha = torch.ones(experts, dtype=torch.float32, device=device)
    beta = torch.zeros_like(alpha)
    limit = torch.full_like(alpha, 10.0)
    trtllm_outputs = {
        pdl: torch.empty(max_tokens, hidden, dtype=dtype, device=device)
        for pdl in (False, True)
    }
    shared_inputs = shared_gate_up = shared_down = None
    shared_stream = None
    if args.overlap_shared:
        shared_inputs = torch.randn(
            max_tokens,
            3072,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        shared_gate_up = torch.randn(
            2560,
            3072,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        shared_down = torch.randn(
            3072,
            1280,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        shared_stream = torch.cuda.Stream()

    def make_trtllm(num_tokens: int, enable_pdl: bool) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            result = trtllm_bf16_routed_moe(
                topk_ids=(topk_ids[:num_tokens], topk_weights[:num_tokens]),
                hidden_states=inputs[:num_tokens],
                gemm1_weights=trtllm_w13,
                gemm2_weights=trtllm_w2,
                num_experts=experts,
                top_k=topk,
                n_group=None,
                topk_group=None,
                intermediate_size=intermediate,
                local_expert_offset=0,
                local_num_experts=experts,
                # Match YOCO's precomputed, renormalized Top-K route.
                routing_method_type=1,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=args.tune_max_num_tokens,
                activation_type=ActivationType.Swiglu.value,
                gemm1_alpha=alpha,
                gemm1_beta=beta,
                gemm1_clamp_limit=limit,
                output=trtllm_outputs[enable_pdl][:num_tokens],
            )
            return result[0] if isinstance(result, list) else result

        return run

    def wrap_shared(
        routed: Callable[[], torch.Tensor], num_tokens: int
    ) -> Callable[[], torch.Tensor]:
        if not args.overlap_shared:
            return routed
        assert shared_inputs is not None
        assert shared_gate_up is not None
        assert shared_down is not None
        assert shared_stream is not None

        def run() -> torch.Tensor:
            current_stream = torch.cuda.current_stream()
            shared_stream.wait_stream(current_stream)
            output = routed()
            with torch.cuda.stream(shared_stream):
                gate_up = F.linear(shared_inputs[:num_tokens], shared_gate_up)
                activated = _shared_swiglu(gate_up)
                F.linear(activated, shared_down)
            current_stream.wait_stream(shared_stream)
            return output

        return run

    def make_timed(num_tokens: int, enable_pdl: bool) -> Callable[[], torch.Tensor]:
        return wrap_shared(make_trtllm(num_tokens, enable_pdl), num_tokens)

    if args.compare_routing_mode:
        if captured_topk_ids is not None:
            raise ValueError(
                "--compare-routing-mode requires logits, not --routing-capture"
            )

        from vllm.model_executor.models.yoco import _yoco_topk_routing_impl

        routed_output = torch.empty_like(trtllm_outputs[True])
        logits_output = torch.empty_like(trtllm_outputs[True])
        routing_replay = torch.empty(
            max_tokens,
            topk,
            dtype=torch.int16,
            device=device,
        )

        def make_external_routing(num_tokens: int) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:
                weights, ids = _yoco_topk_routing_impl(
                    routing[:num_tokens],
                    topk,
                )
                result = trtllm_bf16_routed_moe(
                    topk_ids=(ids, weights),
                    hidden_states=inputs[:num_tokens],
                    gemm1_weights=trtllm_w13,
                    gemm2_weights=trtllm_w2,
                    num_experts=experts,
                    top_k=topk,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=intermediate,
                    local_expert_offset=0,
                    local_num_experts=experts,
                    routing_method_type=1,
                    enable_pdl=True,
                    tune_max_num_tokens=args.tune_max_num_tokens,
                    activation_type=ActivationType.Swiglu.value,
                    gemm1_alpha=alpha,
                    gemm1_beta=beta,
                    gemm1_clamp_limit=limit,
                    output=routed_output[:num_tokens],
                )
                return result[0] if isinstance(result, list) else result

            return run

        def make_logits_routing(num_tokens: int) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:
                result = trtllm_bf16_moe(
                    routing_logits=routing[:num_tokens],
                    routing_bias=None,
                    hidden_states=inputs[:num_tokens],
                    gemm1_weights=trtllm_w13,
                    gemm2_weights=trtllm_w2,
                    num_experts=experts,
                    top_k=topk,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=intermediate,
                    local_expert_offset=0,
                    local_num_experts=experts,
                    routing_method_type=1,
                    norm_topk_prob=True,
                    enable_pdl=True,
                    tune_max_num_tokens=args.tune_max_num_tokens,
                    activation_type=ActivationType.Swiglu.value,
                    routing_replay_out=routing_replay[:num_tokens],
                    gemm1_alpha=alpha,
                    gemm1_beta=beta,
                    gemm1_clamp_limit=limit,
                    output=logits_output[:num_tokens],
                )
                return result[0] if isinstance(result, list) else result

            return run

        AutoTuner.get().clear_cache()
        print(
            "tokens external_us monolithic_us monolithic_gain_pct "
            "route_ids_exact output_exact output_max_abs output_mean_abs"
        )
        with autotune(not args.cache_only, cache=args.cache):
            make_external_routing(max_tokens)()
            make_logits_routing(max_tokens)()

        for num_tokens in args.tokens:
            functions = (
                make_external_routing(num_tokens),
                make_logits_routing(num_tokens),
            )
            graphs = tuple(_capture(function) for function in functions)
            samples: tuple[list[float], list[float]] = ([], [])
            for round_index in range(args.rounds):
                order = (0, 1) if round_index % 2 == 0 else (1, 0)
                for graph_index in order:
                    samples[graph_index].append(
                        _time_graph(graphs[graph_index], args.repeats)
                    )
            times = tuple(statistics.median(sample) for sample in samples)
            outputs = tuple(function().clone() for function in functions)
            _, expected_ids = _yoco_topk_routing_impl(
                routing[:num_tokens],
                topk,
            )
            difference = (outputs[0].float() - outputs[1].float()).abs()
            routes_equal = torch.equal(
                expected_ids.to(torch.int16), routing_replay[:num_tokens]
            )
            print(
                f"{num_tokens:6d} {times[0]:11.3f} {times[1]:14.3f} "
                f"{100.0 * (times[0] / times[1] - 1.0):19.3f} "
                f"{str(routes_equal):>15} "
                f"{str(torch.equal(outputs[0], outputs[1])):>12} "
                f"{difference.max().item():14.8f} "
                f"{difference.mean().item():15.8f}"
            )
        return

    if args.sweep_tactics:
        from flashinfer.fused_moe import WeightLayout
        from flashinfer.fused_moe.core import (
            DtypeTrtllmGen,
            Fp8QuantizationType,
            MoeRunnerInputs,
            RoutingInputMode,
            get_trtllm_moe_sm100_module,
        )

        direct_output = trtllm_outputs[True][:max_tokens]
        runner = get_trtllm_moe_sm100_module().MoERunner(
            top_k=topk,
            num_local_experts=experts,
            dtype_act=DtypeTrtllmGen.Bfloat16,
            dtype_weights=DtypeTrtllmGen.Bfloat16,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation_type=ActivationType.Swiglu.value,
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            num_experts=experts,
        )
        routing_samples = (
            captured_topk_ids
            if captured_topk_ids is not None
            else topk_ids[:max_tokens].unsqueeze(0)
        )
        weight_samples = (
            captured_topk_weights
            if captured_topk_weights is not None
            else topk_weights[:max_tokens].unsqueeze(0)
        )
        direct_inputs = [
            MoeRunnerInputs(
                output=direct_output,
                routing_logits=None,
                topk_ids=sample_ids,
                expert_weights=sample_weights,
                hidden_states=inputs[:max_tokens],
                hidden_states_scale=None,
                gemm1_lora_delta=None,
                per_token_scale=None,
            ).to_list()
            for sample_ids, sample_weights in zip(routing_samples, weight_samples)
        ]
        runner_kwargs = {
            "routing_bias": None,
            "routing_input_mode": RoutingInputMode.UnpackedPrecomputed,
            "gemm1_weights": trtllm_w13,
            "gemm2_weights": trtllm_w2,
            "gemm1_alpha": alpha,
            "gemm1_beta": beta,
            "gemm1_clamp_limit": limit,
            "num_experts": experts,
            "n_group": None,
            "topk_group": None,
            "local_expert_offset": 0,
            "local_num_experts": experts,
            "routed_scaling_factor": None,
            "routing_method_type": 1,
            "use_shuffled_weight": True,
            "weight_layout": int(WeightLayout.BlockMajorK),
            "do_finalize": True,
            "enable_pdl": True,
            "activation_type": ActivationType.Swiglu.value,
            "norm_topk_prob": True,
            "routing_replay_out": None,
        }
        tactics = runner.get_valid_tactics(direct_inputs[0], None)
        if args.tactic_candidates:
            selected = {
                tuple(int(value) for value in identity.split(":"))
                for identity in args.tactic_candidates
            }
            tactics = [
                tactic
                for tactic in tactics
                if tuple(int(value) for value in tactic) in selected
            ]
            missing = selected - {
                tuple(int(value) for value in tactic) for tactic in tactics
            }
            if missing:
                raise ValueError(f"invalid tactic candidates: {sorted(missing)}")
        print(f"valid_tactics={len(tactics)} route_samples={len(direct_inputs)}")
        measurements: list[tuple[float, object]] = []
        for tactic in tactics:

            def run(tactic=tactic) -> torch.Tensor:
                for sample_inputs in direct_inputs:
                    if args.overlap_shared:
                        assert shared_inputs is not None
                        assert shared_gate_up is not None
                        assert shared_down is not None
                        assert shared_stream is not None
                        current_stream = torch.cuda.current_stream()
                        shared_stream.wait_stream(current_stream)
                    runner.forward(sample_inputs, tactic=tactic, **runner_kwargs)
                    if args.overlap_shared:
                        with torch.cuda.stream(shared_stream):
                            gate_up = F.linear(
                                shared_inputs[:max_tokens], shared_gate_up
                            )
                            activated = _shared_swiglu(gate_up)
                            F.linear(activated, shared_down)
                        current_stream.wait_stream(shared_stream)
                return direct_output

            function = run
            try:
                graph = _capture(function)
                elapsed = statistics.median(
                    _time_graph(graph, args.tactic_repeats)
                    for _ in range(args.tactic_rounds)
                ) / len(direct_inputs)
            except RuntimeError as error:
                print(f"tactic={tactic} error={error}")
                continue
            identity = tuple(int(value) for value in tactic)
            measurements.append((elapsed, identity))
            if args.print_all_tactics:
                print(f"tactic={identity} elapsed_us={elapsed:.3f}")
        if args.reference_tactic is not None:
            reference_identity = tuple(args.reference_tactic)
            reference_elapsed = next(
                elapsed
                for elapsed, identity in measurements
                if identity == reference_identity
            )
            print(
                f"reference tactic={reference_identity} "
                f"elapsed_us={reference_elapsed:.3f}"
            )
        print("fastest tactics:")
        for elapsed, tactic in sorted(measurements, key=lambda item: item[0])[:20]:
            print(f"{elapsed:.3f} us tactic={tactic}")
        return

    AutoTuner.get().clear_cache()
    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} tune_max={args.tune_max_num_tokens} "
        f"overlap_shared={args.overlap_shared}"
    )
    print(
        "tokens pdl_off_us pdl_on_us pdl_gain_pct pdl_exact pdl_max_abs "
        "ref_exact ref_max_abs unclamped_max_abs first_actual first_ref"
    )
    with autotune(not args.cache_only, cache=args.cache):
        make_timed(max_tokens, False)()
        make_timed(max_tokens, True)()

    for num_tokens in args.tokens:
        functions = (
            make_timed(num_tokens, False),
            make_timed(num_tokens, True),
        )
        graphs = tuple(_capture(function) for function in functions)
        samples: tuple[list[float], list[float]] = ([], [])
        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for graph_index in order:
                samples[graph_index].append(
                    _time_graph(graphs[graph_index], args.repeats)
                )
        times = tuple(statistics.median(sample) for sample in samples)
        outputs = tuple(function().clone() for function in functions)
        actual = outputs[1][0:1]
        projected = F.linear(inputs[0:1], w13[0])
        up, gate = projected.chunk(2, dim=-1)
        clamped = up.float().clamp(-10.0, 10.0) * F.silu(gate.float().clamp(max=10.0))
        unclamped = up.float() * F.silu(gate.float())
        reference = F.linear(clamped.to(dtype), w2[0])
        unclamped_reference = F.linear(unclamped.to(dtype), w2[0])
        ref_max_abs = (actual.float() - reference.float()).abs().max().item()
        unclamped_max_abs = (
            (actual.float() - unclamped_reference.float()).abs().max().item()
        )
        pdl_difference = (outputs[0].float() - outputs[1].float()).abs()
        print(
            f"{num_tokens:6d} {times[0]:10.3f} {times[1]:9.3f} "
            f"{100.0 * (times[0] / times[1] - 1.0):12.3f} "
            f"{str(torch.equal(outputs[0], outputs[1])):>9} "
            f"{pdl_difference.max().item():.8f} "
            f"{str(torch.equal(actual, reference)):>9} {ref_max_abs:.8f} "
            f"{unclamped_max_abs:.8f} {actual[0, 0].item():.6f} "
            f"{reference[0, 0].item():.6f}"
        )


if __name__ == "__main__":
    main()
