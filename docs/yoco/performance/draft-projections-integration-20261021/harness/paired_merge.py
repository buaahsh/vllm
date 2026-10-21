# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit baseline and combined draft paths on fixed target weights and graphs."""

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import baseline_model
import torch

from tools.yoco_alignment.benchmark_mega_moe import PROMPTS, MegaMoEBenchmarkWorker
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id

ROOT = Path(__file__).resolve().parent
MODES = ["baseline", "combined"]


class MergeWorker(MegaMoEBenchmarkWorker):
    def _set_forwards(self, mode):
        model = self.model_runner.speculator.model
        model.forward = self._model_forwards[mode]
        for i, layer in enumerate(model.layers):
            layer.forward = self._layer_forwards[mode][i]
            layer.mlp = self._mlps[mode][i]

    @torch.inference_mode()
    def merge_init(self):
        spec = self.model_runner.speculator
        model = spec.model
        self.ab_bind_graphs()
        self._model_forwards = {
            "combined": model.forward,
            "baseline": MethodType(baseline_model.SharedKVDraft.forward, model),
        }
        self._layer_forwards = {
            "combined": [layer.forward for layer in model.layers],
            "baseline": [
                MethodType(baseline_model.SharedKVLayer.forward, layer)
                for layer in model.layers
            ],
        }
        original_mlps = []
        for layer in model.layers:
            packed = layer.mlp
            with torch.device("meta"):
                original = baseline_model.SharedKVMLP(model.c)
            original.gate_proj = baseline_model._SharedKVInferenceLinear(
                packed.gate_proj
            )
            original.up_proj = baseline_model._SharedKVInferenceLinear(packed.up_proj)
            original.down_proj = packed.down_proj
            original.eval().requires_grad_(False)
            torch.testing.assert_close(
                torch.cat(
                    (original.gate_proj._bf16_weight, original.up_proj._bf16_weight)
                ),
                packed._bf16_gate_up_weight,
                rtol=0,
                atol=0,
            )
            original_mlps.append(original)
        self._mlps = {
            "baseline": original_mlps,
            "combined": [layer.mlp for layer in model.layers],
        }
        self._managers = {"combined": spec.shared_kv_cudagraph_manager}
        self._set_forwards("baseline")
        spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
        manager = spec.shared_kv_cudagraph_manager
        manager.pool = torch.cuda.graph_pool_handle()
        spec.capture()
        assert set(manager.graphs) == set(self._managers["combined"].graphs)
        self._managers["baseline"] = manager
        self._checks = []
        self._pending = None
        self._replays = {mode: {} for mode in MODES}
        for mode, manager in self._managers.items():
            original = manager.run_fullgraph

            def replay(desc, original=original, mode=mode):
                counts = self._replays[mode]
                key = str(desc.num_tokens)
                counts[key] = counts.get(key, 0) + 1
                result = original(desc)
                pending = self._pending
                if pending and mode == "baseline" and desc.num_reqs == pending["batch"]:
                    ends = spec._prefix_lens[: desc.num_reqs].cpu().tolist()
                    if min(ends) >= pending["minimum"]:
                        self._pending = None
                        self._compare(desc, ends)
                return result

            manager.run_fullgraph = replay
        self.merge_set("combined")
        return self.merge_status()

    def merge_set(self, mode):
        torch.accelerator.synchronize()
        self._set_forwards(mode)
        manager = self._managers[mode]
        self.model_runner.speculator.shared_kv_cudagraph_manager = manager
        set_graph_pool_id(manager.pool)

    def merge_arm(self, batch, minimum):
        self._pending = {"batch": batch, "minimum": minimum}

    def _time_graphs(self, desc):
        samples = {mode: [] for mode in MODES}
        for trial in range(6):
            order = MODES + MODES[::-1]
            if trial % 2:
                order = MODES[::-1] + MODES
            for mode in order:
                graph = self._managers[mode]
                for _ in range(3):
                    graph.run_fullgraph(desc)
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(50):
                    graph.run_fullgraph(desc)
                end.record()
                end.synchronize()
                samples[mode].append(begin.elapsed_time(end) / 50)
        median = {mode: statistics.median(values) for mode, values in samples.items()}
        return {
            "samples_ms": samples,
            "median_ms": median,
            "latency_reduction_pct": (1 - median["combined"] / median["baseline"])
            * 100,
        }

    @torch.inference_mode()
    def _compare(self, desc, ends):
        spec = self.model_runner.speculator
        names = ["draft_tokens", "draft_logits", "draft_token_confidence_probs"]
        saved = {name: getattr(spec, name).clone() for name in names}
        hidden = []
        hook = spec.model.register_forward_hook(
            lambda module, args, output: hidden.append(output.clone())
        )
        eager = {}
        try:
            for mode in MODES:
                self._set_forwards(mode)
                spec._run_shared_kv(
                    desc.num_reqs,
                    spec._cache_owner.kv_cache,
                    spec._prefix_block_table[: desc.num_reqs],
                    spec.max_model_len,
                )
                torch.accelerator.synchronize()
                eager[mode] = {name: getattr(spec, name).clone() for name in names}
            for name in names:
                torch.testing.assert_close(
                    eager["baseline"][name], saved[name], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    eager["combined"][name], saved[name], rtol=0, atol=0
                )
            torch.testing.assert_close(hidden[0], hidden[1], rtol=0, atol=0)
            timing = self._time_graphs(desc)
            self._checks.append(
                {
                    "batch": desc.num_reqs,
                    "prefix_lens": ends,
                    "descriptor": str(desc),
                    "exact": ["hidden", *names],
                    "timing": timing,
                }
            )
        finally:
            hook.remove()
            self._set_forwards("baseline")
            for name, value in saved.items():
                getattr(spec, name).copy_(value)

    def merge_status(self):
        return {
            "worker": self.ab_status(),
            "checks": self._checks,
            "pending": self._pending,
            "audit_graphs": {
                mode: {
                    "tokens": manager.captured_token_counts(),
                    "replays": self._replays[mode],
                }
                for mode, manager in self._managers.items()
            },
            "scope": "The second draft pool and baseline caches are audit-only.",
        }


def main():
    from vllm import LLM, SamplingParams

    out = ROOT / "results/paired"
    out.mkdir(exist_ok=False)
    config = json.loads((ROOT / "config.json").read_text())
    config["worker_extension_cls"] = "paired_merge.MergeWorker"

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    save("manifest.json", {"config": config, "scope": __doc__, "modes": MODES})
    llm = LLM(**config)
    save("worker.json", llm.collective_rpc("merge_init"))
    tokenizer = llm.get_tokenizer()
    short = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for prompt in PROMPTS
    ]
    filler = tokenizer.encode(
        "This technical note describes cache reuse, model execution, "
        "request scheduling, and numerical accuracy. ",
        add_special_tokens=False,
    )
    rows = []
    for length in [0, 8193]:
        prompts = []
        for ids in short:
            count = length - len(ids)
            if length:
                ids = (filler * (count // len(filler) + 1))[:count] + ids
            prompts.append({"prompt_token_ids": ids})
        for c in [1, 4, 8]:
            llm.collective_rpc("merge_arm", args=(c, length or 1))
            llm.collective_rpc("merge_set", args=("baseline",))
            for audit_seed in range(20261020, 20261027):
                llm.generate(
                    prompts[:c],
                    SamplingParams(
                        temperature=1, max_tokens=128, ignore_eos=True, seed=audit_seed
                    ),
                    use_tqdm=False,
                )
                audit_status = llm.collective_rpc("merge_status")
                save(
                    "audit-warmup.json",
                    {
                        "length": length,
                        "concurrency": c,
                        "seed": audit_seed,
                        "status": audit_status,
                    },
                )
                if audit_status[0]["pending"] is None:
                    break
            assert audit_status[0]["pending"] is None
            for trial in range(-2, 3):
                order = MODES if trial % 2 == 0 else MODES[::-1]
                for mode in order:
                    llm.collective_rpc("merge_set", args=(mode,))
                    params = SamplingParams(
                        temperature=1,
                        max_tokens=128,
                        ignore_eos=True,
                        seed=20261020 + max(trial, 0),
                    )
                    before = [asdict(metric) for metric in llm.get_metrics()]
                    llm.collective_rpc("ab_synchronize")
                    begin = time.perf_counter()
                    outputs = llm.generate(prompts[:c], params, use_tqdm=False)
                    llm.collective_rpc("ab_synchronize")
                    elapsed = time.perf_counter() - begin
                    row = {
                        "length": length,
                        "concurrency": c,
                        "mode": mode,
                        "trial": trial,
                        "seed": params.seed,
                        "warmup": trial < 0,
                        "elapsed_s": elapsed,
                        "tokens": c * 128,
                        "metrics_before": before,
                        "metrics_after": [
                            asdict(metric) for metric in llm.get_metrics()
                        ],
                        "output_ids": [
                            list(output.outputs[0].token_ids) for output in outputs
                        ],
                        "request_metrics": [
                            asdict(output.metrics) for output in outputs
                        ],
                    }
                    assert all(len(ids) == 128 for ids in row["output_ids"])
                    assert all(
                        not metric["is_corrupted"] and not metric["num_preemptions"]
                        for metric in row["request_metrics"]
                    )
                    rows.append(row)
                    save("trials.json", rows)
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in [
                                    "length",
                                    "concurrency",
                                    "mode",
                                    "trial",
                                    "elapsed_s",
                                ]
                            }
                        ),
                        flush=True,
                    )
            status = llm.collective_rpc("merge_status")
            assert status[0]["pending"] is None
            save("status.json", status)
    llm.collective_rpc("merge_set", args=("combined",))
    status = llm.collective_rpc("merge_status")
    assert len(status[0]["checks"]) == 6
    save("complete.json", {"status": "PASS", "worker": status})
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
