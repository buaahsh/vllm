# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native same-GPU YOCO V2 P/D compatibility check; not a network benchmark."""

import argparse
import hashlib
import json
import multiprocessing as mp
import socket
import traceback
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from tools.yoco_alignment.benchmark_mega_moe import DRAFT, TARGET


class PDValidationWorker:
    def pd_snapshot(self, prefix):
        import torch

        from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers

        runner = self.model_runner
        aliases = get_shared_kv_cache_layers(self.vllm_config)
        layers = self.vllm_config.compilation_config.static_forward_context
        hashes = {}
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            block = runner.kernel_block_sizes[gid]
            for name in group.layer_names:
                if name in aliases:
                    continue
                spec = group.kv_cache_spec
                if hasattr(spec, "kv_cache_specs"):
                    spec = spec.kv_cache_specs[name]
                window = getattr(spec, "sliding_window", None)
                start = max(0, (prefix - window) // block) if window else 0
                end = prefix // block
                ids = runner.block_tables.input_block_tables[gid][0, start:end].long()
                cache = layers[name].kv_cache
                if cache.shape[0] == 2 and cache.ndim >= 3:
                    cache = cache.movedim(1, 0)
                data = cache.index_select(0, ids).contiguous().view(torch.uint8).cpu()
                hashes[name] = dict(
                    sha256=hashlib.sha256(data.numpy().tobytes()).hexdigest(),
                    nonzero=int(data.count_nonzero()),
                    blocks=end - start,
                )
        return dict(
            hashes=hashes,
            positions=runner.input_buffers.positions[:18].cpu().tolist(),
            seq_lens=runner.input_buffers.seq_lens[:2].cpu().tolist(),
        )

    def pd_status(self):
        runner = self.model_runner
        backend = runner.kv_connector.kv_connector
        worker = backend.connector_worker
        return dict(
            runner=type(runner).__module__,
            engine_id=self.vllm_config.kv_transfer_config.engine_id,
            model_state=type(runner.model_state).__name__,
            force_eager=runner.model_state.force_eager,
            fast_prefill=runner.model.fast_prefill_enabled,
            full_prefill_on_output=getattr(
                runner.model_state, "full_prefill_on_output", False
            ),
            aux_layers=list(runner.model.model.aux_hidden_state_layers),
            registered_layers=getattr(worker, "registered_layer_names", []),
            connector=type(backend).__name__,
            compatibility_hash=getattr(worker, "compat_hash", None),
            cg_mode=str(runner.cudagraph_manager.cudagraph_mode),
            backend_name=getattr(worker, "backend_name", None),
            cache_dtype=str(self.vllm_config.cache_config.cache_dtype),
            cache_layout=str(
                self.vllm_config.cache_config.get_resolved_kv_cache_layout()
            ),
        )


def engine_process(
    pipe,
    role,
    connector,
    bootstrap_port,
    side_port,
    draft,
    execution_mode,
    quantization,
    mega_moe,
):
    import os

    os.environ["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap_port)
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_port)
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    if execution_mode == "align":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    llm = None
    try:
        from vllm import LLM, SamplingParams

        config = dict(
            model=TARGET,
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            quantization=None if quantization == "none" else quantization,
            attention_config=(
                {"backend": "FLASH_ATTN", "flash_attn_version": 4}
                if execution_mode == "align"
                else {"backend": "FLASHINFER"}
            ),
            max_model_len=4096,
            block_size=16,
            max_num_seqs=2,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.35,
            kv_cache_memory_bytes=512 * 1024**2,
            enable_prefix_caching=False,
            kv_sharing_fast_prefill=True,
            async_scheduling=False,
            disable_log_stats=False,
            additional_config={
                "yoco_execution_mode": execution_mode,
                "yoco_mega_moe": "fp8_group32" if mega_moe else "off",
            },
            worker_extension_cls=(
                "tools.yoco_alignment.benchmark_v2_pd.PDValidationWorker"
            ),
            compilation_config=dict(
                cudagraph_mode="FULL_DECODE_ONLY",
                cudagraph_capture_sizes=[1, 2, 4, 8, 9, 16, 18],
                max_cudagraph_capture_size=18,
            ),
            kv_transfer_config=dict(
                kv_connector=connector,
                kv_role="kv_producer" if role == "p" else "kv_consumer",
                engine_id=f"yoco-v2-{connector.lower()}-{role}",
                kv_connector_extra_config={
                    "mooncake_protocol": "rdma",
                    "num_workers": 2,
                    "backends": ["UCX"],
                    "kv_lease_duration": 120,
                },
            ),
        )
        if role == "d":
            config["speculative_config"] = dict(
                model=draft,
                method="dspark",
                num_speculative_tokens=8,
                draft_sample_method="probabilistic",
                rejection_sample_method="standard",
            )
        llm = LLM(**config)
        status = llm.collective_rpc("pd_status")[0]
        assert status["runner"] == "vllm.v1.worker.gpu.model_runner", status
        assert status["force_eager"] == (role == "p"), status
        if role == "p":
            assert status["cg_mode"] == "NONE", status
        pipe.send(dict(status="ready", worker=status, config=config))
        while True:
            command = pipe.recv()
            if command["op"] == "shutdown":
                return
            if command["op"] == "snapshot":
                pipe.send(
                    dict(
                        status="snapshot",
                        data=llm.collective_rpc(
                            "pd_snapshot", args=(command["prefix"],)
                        )[0],
                    )
                )
                continue
            ids = command["ids"]
            params = SamplingParams(
                temperature=0,
                max_tokens=1 if role == "p" else 8,
                ignore_eos=True,
                logprobs=None if role == "p" else 5,
                extra_args={"kv_transfer_params": command["transfer"]},
            )
            output = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[
                0
            ]
            pipe.send(
                dict(
                    status="output",
                    prompt_length=len(output.prompt_token_ids),
                    output_ids=list(output.outputs[0].token_ids),
                    logprobs=[
                        {str(k): asdict(v) for k, v in row.items()}
                        for row in (output.outputs[0].logprobs or [])
                    ],
                    num_cached_tokens=output.num_cached_tokens,
                    transfer=output.kv_transfer_params,
                    metrics=asdict(output.metrics)
                    if output.metrics is not None
                    else None,
                )
            )
    except BaseException:
        pipe.send(dict(status="error", traceback=traceback.format_exc()))
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def receive(pipe, timeout=600):
    if not pipe.poll(timeout):
        raise TimeoutError("P/D validation worker did not respond")
    message = pipe.recv()
    assert message["status"] != "error", message
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connector", choices=["MooncakeConnector", "NixlConnector"], required=True
    )
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--execution-mode", choices=["fast", "align"], default="fast")
    parser.add_argument(
        "--quantization", choices=["fp8_per_block", "none"], default="fp8_per_block"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mega-moe", action="store_true")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[15, 16, 17, 31, 32, 33, 1025]
    )
    args = parser.parse_args()
    if args.mega_moe and (
        args.execution_mode != "fast" or args.quantization != "fp8_per_block"
    ):
        parser.error("--mega-moe requires Fast FP8 execution")
    args.output.mkdir(parents=True, exist_ok=False)
    processes, pipes = [], {}
    result = dict(connector=args.connector, scope=__doc__, workers={}, cases=[])

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    bootstrap = free_port()
    try:
        for role in ["d", "p"]:
            parent, child = mp.get_context("spawn").Pipe()
            process = mp.get_context("spawn").Process(
                target=engine_process,
                args=(
                    child,
                    role,
                    args.connector,
                    bootstrap,
                    free_port(),
                    args.draft,
                    args.execution_mode,
                    args.quantization,
                    args.mega_moe,
                ),
            )
            process.start()
            processes.append(process)
            pipes[role] = parent
            result["workers"][role] = receive(parent)
            save()
            print("READY", role, flush=True)
        if args.connector == "NixlConnector":
            producer_hash = result["workers"]["p"]["worker"]["compatibility_hash"]
            decoder_hash = result["workers"]["d"]["worker"]["compatibility_hash"]
            assert producer_hash == decoder_hash, result["workers"]
        for length in args.lengths:
            ids = ([1234, 2345, 3456, 4567] * (length // 4 + 1))[:length]
            transfer_id = f"yoco-v2-{args.connector}-{length}"
            pipes["p"].send(
                dict(
                    op="generate",
                    ids=ids,
                    transfer=dict(
                        do_remote_decode=True,
                        do_remote_prefill=False,
                        transfer_id=transfer_id,
                        remote_engine_id=None,
                        remote_block_ids=None,
                        remote_host=None,
                        remote_port=None,
                    ),
                )
            )
            producer = receive(pipes["p"])
            expected_cached = max(0, (length - 1) // 16 * 16)
            pipes["p"].send(dict(op="snapshot", prefix=expected_cached))
            producer_snapshot = receive(pipes["p"])
            if args.connector == "MooncakeConnector":
                transfer = dict(
                    remote_engine_id=result["workers"]["p"]["worker"]["engine_id"],
                    remote_bootstrap_addr=f"http://127.0.0.1:{bootstrap}",
                    transfer_id=transfer_id,
                )
            else:
                assert producer["transfer"], producer
                transfer = producer["transfer"]
            transfer.update(do_remote_prefill=True, do_remote_decode=False)
            pipes["d"].send(dict(op="generate", ids=ids, transfer=transfer))
            remote = receive(pipes["d"])
            pipes["d"].send(dict(op="snapshot", prefix=expected_cached))
            remote_snapshot = receive(pipes["d"])
            pipes["d"].send(dict(op="generate", ids=ids, transfer=None))
            local = receive(pipes["d"])
            pipes["d"].send(dict(op="snapshot", prefix=expected_cached))
            local_snapshot = receive(pipes["d"])
            result["cases"].append(
                dict(
                    length=length,
                    producer=producer,
                    remote=remote,
                    local=local,
                    producer_snapshot=producer_snapshot,
                    remote_snapshot=remote_snapshot,
                    local_snapshot=local_snapshot,
                )
            )
            save()
            assert remote["num_cached_tokens"] == expected_cached, remote
            producer_hashes = producer_snapshot["data"]["hashes"]
            remote_hashes = remote_snapshot["data"]["hashes"]
            local_hashes = local_snapshot["data"]["hashes"]
            assert producer_hashes == remote_hashes, (
                "Transferred KV bytes differ",
                length,
                producer_snapshot,
                remote_snapshot,
            )
            if args.execution_mode == "align":
                assert producer_hashes == local_hashes, (
                    "Align P and D prefix states differ",
                    length,
                    producer_snapshot,
                    local_snapshot,
                )
            result["cases"][-1]["same_output_as_local"] = (
                remote["output_ids"] == local["output_ids"]
            )
            if args.execution_mode == "align":
                assert remote["output_ids"] == local["output_ids"], (
                    length,
                    remote,
                    local,
                )
            assert len(remote["output_ids"]) == 8
            assert not remote["metrics"]["is_corrupted"]
            assert not remote["metrics"]["num_preemptions"]
            print("PASS", length, "cached", expected_cached, flush=True)
        all_equal = all(x["same_output_as_local"] for x in result["cases"])
        result["kv_transfer_status"] = "PASS"
        result["generation_comparison_status"] = (
            "IDENTICAL" if all_equal else "DIFFERENT"
        )
        result["status"] = "PASS" if all_equal else "KV_BYTES_PASS_NUMERICS_DIFFER"
        save()
    finally:
        for pipe in pipes.values():
            with suppress(BrokenPipeError, EOFError):
                pipe.send(dict(op="shutdown"))
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join(30)


if __name__ == "__main__":
    main()
