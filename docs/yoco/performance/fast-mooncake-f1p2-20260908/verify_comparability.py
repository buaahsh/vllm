# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Require the same source/runtime, model, physical GPUs and replay settings."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "fast-mooncake-20260908"


def load(root, name):
    return json.loads((root / name).read_text())


def normalize(value):
    if isinstance(value, str):
        return (
            value.replace(ROOT.name, OLD.name)
            .replace("long500s", "long600s")
            .replace("t300-900s-f1p2", "t300-900s-f1")
        )
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def main():
    checks = {
        "runtime_sources_equal": load(ROOT, "GIT_SOURCE.json")["files"]
        == load(OLD, "GIT_SOURCE.json")["files"],
        "compiled_runtime_equal": load(ROOT, "ALIGN_RUNTIME_SHA256.json")
        == load(OLD, "ALIGN_RUNTIME_SHA256.json"),
        "aiperf_source_equal": load(ROOT, "AIPERF_SOURCE_BEFORE.json")["files"]
        == load(OLD, "AIPERF_SOURCE_BEFORE.json")["files"],
        "model_metadata_equal": load(ROOT, "MODEL_FILES.json")
        == load(OLD, "MODEL_FILES.json"),
        "tokenizer_config_equal": load(ROOT, "INPUT_ENVIRONMENT.json")["files"]
        == load(OLD, "INPUT_ENVIRONMENT.json")["files"],
        "timestamps_only_scaled": load(ROOT, "TRACE_VERIFICATION.json")[
            "only_timestamps_changed"
        ],
    }
    cases = []
    for topology in ["standalone", "pd"]:
        before = load(OLD, f"cases/{topology}-fast-r1-long600s/manifest.json")
        after = load(ROOT, f"cases/{topology}-fast-r1-long500s/manifest.json")
        same_commands = normalize(after["command"]) == before["command"]
        same_devices = {k: v["uuid"] for k, v in after["state"]["devices"].items()} == {
            k: v["uuid"] for k, v in before["state"]["devices"].items()
        }
        roles = {}
        for role, previous in before["state"]["processes"].items():
            current = after["state"]["processes"][role]
            roles[role] = dict(
                command_equal=normalize(current["command"]) == previous["command"],
                environment_equal=normalize(current["environment"])
                == previous["environment"],
                align_profile_equal=current["align_profile"]
                == previous["align_profile"],
            )
        cases.append(
            dict(
                topology=topology,
                same_aiperf_parameters_except_rate_paths_and_salt=same_commands,
                same_physical_gpus=same_devices,
                unique_nonempty_salt=bool(after["cache_salt"])
                and after["cache_salt"] != before["cache_salt"],
                roles=roles,
            )
        )
    passed = all(checks.values()) and all(
        c["same_aiperf_parameters_except_rate_paths_and_salt"]
        and c["same_physical_gpus"]
        and c["unique_nonempty_salt"]
        and all(all(r.values()) for r in c["roles"].values())
        for c in cases
    )
    proof = dict(
        passed=passed,
        checks=checks,
        cases=cases,
        comparison=(
            "Fast load response at different explicit rates; not a same-load kernel A/B"
        ),
        shared_node=True,
    )
    proof["workload_scope"] = (
        "Trace ISL/OSL/order/hash_ids are preserved; realized synthetic "
        "prompt tokens are not proven identical across rate files. "
        "See TOKEN_REALIZATION.json for the one-token input-count deltas."
    )
    (ROOT / "COMPARABILITY.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))
    assert passed


if __name__ == "__main__":
    main()
