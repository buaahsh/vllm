# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component timing at the first matching live batch, before requests drain."""

import json
from pathlib import Path

import torch
from paired import MODES, PROMPTS, ProjectionWorker

ROOT = Path(__file__).resolve().parent


class ComponentWorker(ProjectionWorker):
    @torch.inference_mode()
    def paired_init(self):
        self._component_rows = []
        return super().paired_init()

    @torch.inference_mode()
    def _compare(self, desc, ends):
        super()._compare(desc, ends)
        spec = self.model_runner.speculator
        names = ["draft_tokens", "draft_logits", "draft_token_confidence_probs"]
        saved = {name: getattr(spec, name).clone() for name in names}
        self._last_desc = desc
        try:
            result = self.component_timing()
            assert len(result["prefix_lens"]) == desc.num_reqs == len(ends)
            self._component_rows.append(
                {"batch": desc.num_reqs, "prefix_lens": ends, "timing": result}
            )
        finally:
            for name, value in saved.items():
                getattr(spec, name).copy_(value)

    @torch.inference_mode()
    def component_profiles(self):
        spec = self.model_runner.speculator
        desc = self._last_desc
        names = ["draft_tokens", "draft_logits", "draft_token_confidence_probs"]
        saved = {name: getattr(spec, name).clone() for name in names}
        profiles = {}
        try:
            for mode in MODES:
                self._set_forwards(mode)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    spec._run_shared_kv(
                        desc.num_reqs,
                        spec._cache_owner.kv_cache,
                        spec._prefix_block_table[: desc.num_reqs],
                        spec.max_model_len,
                    )
                    torch.accelerator.synchronize()
                profiles[mode] = {
                    event.key: event.count
                    for event in profile.key_averages()
                    if event.key
                    in {
                        "aten::linear",
                        "aten::mm",
                        "aten::cos",
                        "aten::sin",
                        "aten::arange",
                        "aten::pow",
                    }
                }
        finally:
            self._set_forwards("baseline")
            for name, value in saved.items():
                getattr(spec, name).copy_(value)
        return {
            "descriptor": str(desc),
            "counts": profiles,
            "scope": (
                "CPU operator counts in a separate eager replay after all event"
                " timings; not latency data."
            ),
        }

    def component_results(self):
        return {"rows": self._component_rows, "status": self.paired_status()}


def main():
    from vllm import LLM, SamplingParams

    out = ROOT / "results/components"
    out.mkdir(exist_ok=False)
    config = json.loads((ROOT / "config.json").read_text())
    config["worker_extension_cls"] = "components.ComponentWorker"

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    save("manifest.json", {"config": config, "scope": __doc__, "modes": MODES})
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
    for length in [0, 8193]:
        prompts = []
        for ids in short:
            count = length - len(ids)
            if length:
                ids = (filler * (count // len(filler) + 1))[:count] + ids
            prompts.append({"prompt_token_ids": ids})
        for c in [1, 4, 8]:
            llm.collective_rpc("paired_set", args=("baseline",))
            llm.collective_rpc("paired_arm", args=(c, length or 1))
            for attempt in range(7):
                seed = 20261020 + attempt
                llm.generate(
                    prompts[:c],
                    SamplingParams(
                        temperature=1, max_tokens=128, ignore_eos=True, seed=seed
                    ),
                    use_tqdm=False,
                )
                result = llm.collective_rpc("component_results")
                save("results.json", result)
                if result[0]["status"]["pending"] is None:
                    break
            assert result[0]["status"]["pending"] is None
            assert result[0]["rows"][-1]["batch"] == c
            print(
                json.dumps(
                    {
                        "length": length,
                        "concurrency": c,
                        "attempt": attempt,
                        "seed": seed,
                        "result": result[0]["rows"][-1],
                    }
                ),
                flush=True,
            )
    result = llm.collective_rpc("component_results")
    assert len(result[0]["rows"]) == 6 and result[0]["status"]["pending"] is None
    save("operator-counts.json", llm.collective_rpc("component_profiles"))
    save("complete.json", {"status": "PASS", "results": result})
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
