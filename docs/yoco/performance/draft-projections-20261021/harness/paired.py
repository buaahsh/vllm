# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit-only same-engine baseline / shared RoPE / packed gate-up comparisons."""

import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import torch

from tools.yoco_alignment.benchmark_mega_moe import PROMPTS, MegaMoEBenchmarkWorker
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id

ROOT = Path(__file__).resolve().parent
MODES = ["baseline", "rope", "gate-up"]


def load_model(version):
    name = "audit_" + version.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        name, ROOT / version / "vllm/model_executor/models/yoco_shared_kv.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def diff(actual, expected):
    value = (actual.float() - expected.float()).abs()
    return {
        "equal": bool(torch.equal(actual, expected)),
        "max_abs": value.max().item(),
        "mean_abs": value.mean().item(),
        "different": int((actual != expected).sum().item()),
        "elements": actual.numel(),
        "dtype": str(actual.dtype),
    }


class ProjectionWorker(MegaMoEBenchmarkWorker):
    def _set_forwards(self, mode):
        spec = self.model_runner.speculator
        model = spec.model
        model.forward = self._model_forwards[mode]
        for i, layer in enumerate(model.layers):
            layer.forward = self._layer_forwards[mode][i]
            layer.mlp = self._mlps[mode][i]

    @torch.inference_mode()
    def paired_init(self):
        spec = self.model_runner.speculator
        model = spec.model
        rope, packed = load_model("rope"), load_model("gate-up")
        self._model_forwards = {mode: model.forward for mode in MODES}
        self._model_forwards["rope"] = MethodType(rope.SharedKVDraft.forward, model)
        self._layer_forwards = {
            mode: [layer.forward for layer in model.layers] for mode in MODES
        }
        self._layer_forwards["rope"] = [
            MethodType(rope.SharedKVLayer.forward, layer) for layer in model.layers
        ]
        self._mlps = {mode: [layer.mlp for layer in model.layers] for mode in MODES}
        self._mlps["gate-up"] = [
            packed._SharedKVInferenceMLP(layer.mlp) for layer in model.layers
        ]
        self._managers = {"baseline": spec.shared_kv_cudagraph_manager}
        for mode in MODES[1:]:
            self._set_forwards(mode)
            spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
            manager = spec.shared_kv_cudagraph_manager
            manager.pool = torch.cuda.graph_pool_handle()
            spec.capture()
            assert set(manager.graphs) == set(self._managers["baseline"].graphs)
            self._managers[mode] = manager
        self._pending = None
        self._checks = []
        self._last_desc = None
        self._replays = {mode: {} for mode in MODES}
        for mode, manager in self._managers.items():
            original = manager.run_fullgraph

            def replay(desc, original=original, mode=mode):
                key = str(desc.num_tokens)
                self._replays[mode][key] = self._replays[mode].get(key, 0) + 1
                result = original(desc)
                self._last_desc = desc
                pending = self._pending
                if pending and mode == "baseline" and desc.num_reqs == pending["batch"]:
                    ends = spec._prefix_lens[: desc.num_reqs].cpu().tolist()
                    if min(ends) >= pending["minimum"]:
                        self._pending = None
                        self._compare(desc, ends)
                return result

            manager.run_fullgraph = replay
        self.paired_set("baseline")
        return self.paired_status()

    def paired_set(self, mode):
        torch.accelerator.synchronize()
        self._set_forwards(mode)
        manager = self._managers[mode]
        self.model_runner.speculator.shared_kv_cudagraph_manager = manager
        set_graph_pool_id(manager.pool)
        self._active = mode

    def paired_arm(self, batch, minimum):
        self._pending = {"batch": batch, "minimum": minimum}

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
            row = {"batch": desc.num_reqs, "prefix_lens": ends, "comparisons": {}}
            for i, mode in enumerate(MODES[1:], 1):
                result = {"hidden": diff(hidden[i], hidden[0])}
                result.update(
                    {
                        name: diff(eager[mode][name], eager["baseline"][name])
                        for name in names
                    }
                )
                if mode == "rope":
                    assert all(item["equal"] for item in result.values())
                row["comparisons"][mode] = result
            self._checks.append(row)
        finally:
            hook.remove()
            self._set_forwards("baseline")
            for name, value in saved.items():
                getattr(spec, name).copy_(value)

    def component_timing(self):
        """Repeat the latest full draft graph on fixed actual checkpoint inputs."""
        desc = self._last_desc
        assert desc is not None
        self.paired_set("baseline")
        samples = {mode: [] for mode in MODES}
        for trial in range(9):
            order = MODES[trial % 3 :] + MODES[: trial % 3]
            if trial % 2:
                order = order[::-1]
            for mode in order:
                graph = self._managers[mode]
                for _ in range(3):
                    graph.run_fullgraph(desc)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(50):
                    graph.run_fullgraph(desc)
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) / 50)
        self.paired_set("baseline")
        medians = {mode: statistics.median(values) for mode, values in samples.items()}
        return {
            "descriptor": str(desc),
            "prefix_lens": self.model_runner.speculator._prefix_lens[: desc.num_reqs]
            .cpu()
            .tolist(),
            "samples_ms": samples,
            "median_ms": medians,
            "latency_reduction_pct": {
                mode: (1 - medians[mode] / medians["baseline"]) * 100
                for mode in MODES[1:]
            },
        }

    def paired_status(self):
        return {
            "worker": self.ab_status(),
            "audit_graphs": {
                mode: {
                    "tokens": manager.captured_token_counts(),
                    "replays": self._replays[mode],
                }
                for mode, manager in self._managers.items()
            },
            "same_input": self._checks,
            "pending": self._pending,
            "scope": (
                "Three draft pools are audit-only. Production keeps target 4 / "
                "draft 6. Same target weights, KV and target graphs."
            ),
        }


def main():
    from vllm import LLM, SamplingParams

    out = ROOT / "results/paired"
    out.mkdir(exist_ok=False)
    config = json.loads((ROOT / "config.json").read_text())
    config["worker_extension_cls"] = "paired.ProjectionWorker"

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    save(
        "manifest.json",
        {
            "config": config,
            "scope": (
                "Same engine rotating three isolated draft paths; no Markov fusion."
            ),
            "modes": MODES,
        },
    )
    llm = LLM(**config)
    save("worker.json", llm.collective_rpc("paired_init"))
    tokenizer = llm.get_tokenizer()
    short = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for p in PROMPTS
    ]
    filler = tokenizer.encode(
        (
            "This technical note describes cache reuse, model execution, "
            "request scheduling, and numerical accuracy. "
        ),
        add_special_tokens=False,
    )
    rows, components = [], []
    for length in [0, 8193]:
        prompts = []
        for ids in short:
            count = length - len(ids)
            ids = (
                ((filler * (count // len(filler) + 1))[:count] + ids) if length else ids
            )
            prompts.append({"prompt_token_ids": ids})
        for c in [1, 4, 8]:
            llm.collective_rpc("paired_arm", args=(c, length or 1))
            for trial in range(-2, 7):
                shift = max(trial, 0) % 3
                order = MODES[shift:] + MODES[:shift]
                if trial % 2:
                    order = order[::-1]
                for mode in order:
                    llm.collective_rpc("paired_set", args=(mode,))
                    params = SamplingParams(
                        temperature=1,
                        max_tokens=128,
                        ignore_eos=True,
                        seed=20261020 + max(trial, 0),
                    )
                    before = [asdict(m) for m in llm.get_metrics()]
                    llm.collective_rpc("ab_synchronize")
                    start = time.perf_counter()
                    outputs = llm.generate(prompts[:c], params, use_tqdm=False)
                    llm.collective_rpc("ab_synchronize")
                    elapsed = time.perf_counter() - start
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
                        "metrics_after": [asdict(m) for m in llm.get_metrics()],
                        "output_ids": [list(o.outputs[0].token_ids) for o in outputs],
                        "request_metrics": [asdict(o.metrics) for o in outputs],
                    }
                    assert all(len(ids) == 128 for ids in row["output_ids"])
                    assert all(
                        not m["is_corrupted"] and not m["num_preemptions"]
                        for m in row["request_metrics"]
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
            components.append(
                {
                    "length": length,
                    "concurrency": c,
                    "timing": llm.collective_rpc("component_timing"),
                }
            )
            save("components.json", components)
            save("status.json", llm.collective_rpc("paired_status"))
    save(
        "complete.json",
        {"status": "PASS", "worker": llm.collective_rpc("paired_status")},
    )
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
