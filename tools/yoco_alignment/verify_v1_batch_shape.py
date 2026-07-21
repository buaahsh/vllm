#!/usr/bin/env python3
"""Verify vLLM prefill forward shapes from scheduler iteration logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


ITERATION_PATTERN = re.compile(
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiprocessing-log", required=True)
    parser.add_argument("--multiprocessing-artifact", required=True)
    parser.add_argument("--in-process-log", required=True)
    parser.add_argument("--in-process-artifact", required=True)
    parser.add_argument(
        "--native-artifact",
        help="Optional Native artifact whose model_forwards must be one matched batch.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--allow-no-legacy-split",
        action="store_true",
        help=(
            "Do not fail if the multiprocessing arm happens to schedule one "
            "batch. The in-process arm is still required to be one batch."
        ),
    )
    return parser.parse_args()


def _iteration_rows(path: str) -> list[dict[str, int]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in ITERATION_PATTERN.finditer(text)
    ]
    if not rows:
        raise ValueError(f"No vLLM iteration-detail records found in {path}")
    return rows


def _artifact_expectation(path: str, expected_multiprocessing: bool) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actual_multiprocessing = payload.get("v1_multiprocessing_enabled")
    if actual_multiprocessing is not expected_multiprocessing:
        raise ValueError(
            f"Unexpected multiprocessing metadata in {path}: "
            f"{actual_multiprocessing!r}"
        )
    if payload.get("prefix_caching_enabled") is not False:
        raise ValueError(
            f"Prefix caching must be disabled for matched token rows: {path}"
        )
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No results found in {path}")
    return {
        "num_requests": len(results),
        "num_prompt_tokens": sum(
            int(result["prompt"]["prompt_len"]) for result in results
        ),
        "prompt_names": [result["prompt"]["name"] for result in results],
    }


def _summarize(
    log_path: str,
    artifact_path: str,
    expected_multiprocessing: bool,
) -> dict[str, Any]:
    expectation = _artifact_expectation(artifact_path, expected_multiprocessing)
    rows = _iteration_rows(log_path)
    context_forwards = [
        {
            "iteration": row["iteration"],
            "num_requests": row["context_requests"],
            "num_tokens": row["context_tokens"],
        }
        for row in rows
        if row["context_requests"]
    ]
    return {
        "multiprocessing_enabled": expected_multiprocessing,
        "log": str(Path(log_path).resolve()),
        "artifact": str(Path(artifact_path).resolve()),
        "expected": expectation,
        "context_forwards": context_forwards,
        "observed_total_context_requests": sum(
            row["num_requests"] for row in context_forwards
        ),
        "observed_total_context_tokens": sum(
            row["num_tokens"] for row in context_forwards
        ),
    }


def _is_one_matched_prefill(summary: dict[str, Any]) -> bool:
    expected = summary["expected"]
    forwards = summary["context_forwards"]
    if len(forwards) != 1:
        return False
    return (
        forwards[0]["num_requests"] == expected["num_requests"]
        and forwards[0]["num_tokens"] == expected["num_prompt_tokens"]
    )


def _native_summary(path: str, expected: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    results = payload.get("results", [])
    prompt_names = [result["prompt"]["name"] for result in results]
    if prompt_names != expected["prompt_names"]:
        raise AssertionError("Native and vLLM artifacts contain different prompts")
    model_forwards = payload.get("model_forwards")
    if not isinstance(model_forwards, list):
        raise ValueError(
            f"Native artifact does not contain model_forwards metadata: {path}"
        )
    matching_forwards = [
        forward
        for forward in model_forwards
        if forward.get("phase") == "prefill"
    ]
    verified = len(matching_forwards) == 1 and (
        matching_forwards[0].get("num_requests") == expected["num_requests"]
        and matching_forwards[0].get("num_tokens")
        == expected["num_prompt_tokens"]
        and matching_forwards[0].get("prompt_names") == expected["prompt_names"]
    )
    if not verified:
        raise AssertionError(
            f"Native did not execute one matched prefill: {matching_forwards}"
        )
    return {
        "verified": True,
        "artifact": str(Path(path).resolve()),
        "model_forwards": model_forwards,
    }


def main() -> None:
    args = parse_args()
    multiprocessing = _summarize(
        args.multiprocessing_log,
        args.multiprocessing_artifact,
        True,
    )
    in_process = _summarize(
        args.in_process_log,
        args.in_process_artifact,
        False,
    )
    if multiprocessing["expected"] != in_process["expected"]:
        raise AssertionError("A/B artifacts do not contain identical prompts")

    in_process_verified = _is_one_matched_prefill(in_process)
    legacy_split_observed = not _is_one_matched_prefill(multiprocessing)
    if not in_process_verified:
        raise AssertionError(
            "In-process vLLM did not execute one matched prefill: "
            f"{in_process['context_forwards']}"
        )
    if not legacy_split_observed and not args.allow_no_legacy_split:
        raise AssertionError(
            "The multiprocessing arm did not exhibit a split in this trial; "
            "repeat the A/B run or pass --allow-no-legacy-split."
        )

    native = (
        _native_summary(args.native_artifact, in_process["expected"])
        if args.native_artifact
        else None
    )

    result = {
        "verified": True,
        "in_process_one_matched_prefill": in_process_verified,
        "legacy_split_observed": legacy_split_observed,
        "multiprocessing": multiprocessing,
        "in_process": in_process,
        "native": native,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()