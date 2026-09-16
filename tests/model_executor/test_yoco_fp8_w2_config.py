# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W2-only tuning must preserve W13 dispatch and unmeasured paths."""

from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.experts import yoco_triton as yt

BASE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}


def test_selection_does_not_mutate_w13_or_cached_table():
    base = BASE.copy()
    candidate = {**BASE, "BLOCK_SIZE_N": 32, "num_stages": 4}
    selected = yt.select_yoco_fp8_w2_config({1: candidate}, 1, base)
    assert selected["BLOCK_SIZE_N"] == 32
    assert selected["BLOCK_SIZE_K"] == base["BLOCK_SIZE_K"]
    assert selected["BLOCK_SIZE_M"] == base["BLOCK_SIZE_M"]
    selected["BLOCK_SIZE_N"] = 64
    assert base == BASE
    assert candidate["BLOCK_SIZE_N"] == 32


@pytest.mark.parametrize("tokens", [0, 3, 5, 7, 9, 15, 17, 32, 128])
def test_no_extrapolation_to_unmeasured_rows(tokens):
    assert (
        yt.select_yoco_fp8_w2_config({1: BASE, 2: BASE, 4: BASE}, tokens, BASE) is BASE
    )


@pytest.mark.parametrize(
    "key,value", [("BLOCK_SIZE_M", 32), ("BLOCK_SIZE_K", 64), ("GROUP_SIZE_M", 8)]
)
def test_dispatch_and_partial_sum_boundaries_are_preserved(key, value):
    candidate = {**BASE, key: value, "BLOCK_SIZE_N": 32}
    assert yt.select_yoco_fp8_w2_config({1: candidate}, 1, BASE) is BASE


@pytest.fixture
def supported(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_YOCO_FP8_W2_TUNING", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(yt.current_platform, "is_device_capability", lambda *a: True)
    monkeypatch.setattr(yt.current_platform, "get_device_name", lambda: "NVIDIA B200")
    monkeypatch.setattr(
        yt,
        "_load_yoco_decode_configs",
        lambda name: {1: {**BASE, "BLOCK_SIZE_N": 32, "num_stages": 4}},
    )


def test_eligible_fp8_w2_uses_private_table(supported):
    actual = yt.try_get_yoco_fp8_w2_config(1, 128, 1024, 3840, BASE)
    assert actual["BLOCK_SIZE_N"] == 32
    assert BASE["BLOCK_SIZE_N"] == 128


@pytest.mark.parametrize(
    "tokens,experts,output,input_size",
    [
        (0, 128, 1024, 3840),
        (17, 128, 1024, 3840),
        (1, 64, 1024, 3840),
        (1, 128, 512, 3840),
        (1, 128, 1024, 4096),
    ],
)
def test_other_shapes_keep_original_config(
    supported, tokens, experts, output, input_size
):
    assert (
        yt.try_get_yoco_fp8_w2_config(tokens, experts, output, input_size, BASE) is BASE
    )


@pytest.mark.parametrize("flag", ["disabled", "batch_invariant", "other_device"])
def test_mode_and_device_fallbacks(supported, monkeypatch, flag):
    if flag == "disabled":
        monkeypatch.setattr(envs, "VLLM_YOCO_FP8_W2_TUNING", False)
    elif flag == "batch_invariant":
        monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    else:
        monkeypatch.setattr(
            yt.current_platform, "is_device_capability", lambda *a: False
        )
    assert yt.try_get_yoco_fp8_w2_config(1, 128, 1024, 3840, BASE) is BASE


def test_explicit_config_override_is_respected(supported):
    with override_config({**BASE, "BLOCK_SIZE_N": 64}):
        assert yt.try_get_yoco_fp8_w2_config(1, 128, 1024, 3840, BASE) is BASE


def test_missing_table_keeps_original_config(supported, monkeypatch):
    monkeypatch.setattr(yt, "_load_yoco_decode_configs", lambda name: None)
    assert yt.try_get_yoco_fp8_w2_config(1, 128, 1024, 3840, BASE) is BASE


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16])
def test_shipped_table_excludes_m16_after_full_model_regression(
    supported, monkeypatch, tokens
):
    table = Path(yt.__file__).with_name("yoco_configs") / (
        "decode_fp8_w2_E=128,N=1024,K=3840,device_name=NVIDIA_B200.json"
    )
    monkeypatch.setattr(
        yt, "_load_yoco_decode_configs", lambda name: yt._parse_yoco_configs(table)
    )
    actual = yt.try_get_yoco_fp8_w2_config(tokens, 128, 1024, 3840, BASE)
    assert (actual != BASE) == (tokens in (1, 2, 4))
    assert BASE["BLOCK_SIZE_N"] == 128
