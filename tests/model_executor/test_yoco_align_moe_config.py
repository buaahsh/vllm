# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import yoco_align_moe as shared
from vllm.triton_utils import triton


@pytest.fixture
def profile():
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 3,
    }
    return {
        "schema_version": 1,
        "environment": {
            "device_name": "test device",
            "torch_version": str(torch.__version__),
            "triton_version": triton.__version__,
            "cuda_version": torch.version.cuda or "test CUDA",
        },
        "w13_shape": [128, 7680, 1024],
        "w2_shape": [128, 1024, 3840],
        "top_k": 8,
        "configs": {"512": {"w13": config, "w2": dict(config, BLOCK_SIZE_N=64)}},
    }


def write_profile(tmp_path, profile):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return str(path)


@pytest.fixture
def runtime(monkeypatch):
    calls = []

    def name(index):
        calls.append(index)
        return "test device"

    monkeypatch.setattr(
        shared,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, get_device_name=name),
    )
    monkeypatch.setattr(shared.envs, "VLLM_BATCH_INVARIANT", True)
    if torch.version.cuda is None:
        monkeypatch.setattr(torch.version, "cuda", "test CUDA")
    yield calls
    shared.load_yoco_align_moe_profile.cache_clear()
    shared._verified_profile.cache_clear()


def select(rows=512, **kwargs):
    args = dict(
        num_tokens=rows,
        w13_shape=(128, 7680, 1024),
        w2_shape=(128, 1024, 3840),
        top_k=8,
        base_config=shared._BASE.copy(),
    )
    args.update(kwargs)
    return shared.get_yoco_align_moe_configs(**args)


def test_exact_rows_shared_assignment_and_copy(profile, tmp_path, monkeypatch, runtime):
    monkeypatch.setattr(
        shared.envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", write_profile(tmp_path, profile)
    )
    a, b = select(device_index=2)
    assert a["BLOCK_SIZE_M"] == b["BLOCK_SIZE_M"] == 128
    assert a["BLOCK_SIZE_N"] == 128 and b["BLOCK_SIZE_N"] == 64
    assert a["BLOCK_SIZE_K"] == b["BLOCK_SIZE_K"] == 32
    a["BLOCK_SIZE_K"] = 128
    assert select(device_index=2)[0]["BLOCK_SIZE_K"] == 32
    assert runtime == [2]
    for rows in (0, 1, 511, 513, 1024):
        assert select(rows, device_index=2) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("BLOCK_SIZE_K", 64),
        ("SPLIT_K", 2),
        ("num_warps", 16),
        ("BLOCK_SIZE_M", 48),
        ("num_stages", True),
        ("enable_fp_fusion", False),
    ],
)
def test_rejects_unreviewed_arithmetic_and_launch_keys(profile, tmp_path, key, value):
    profile["configs"]["512"]["w13"][key] = value
    with pytest.raises(ValueError):
        shared.load_yoco_align_moe_profile(write_profile(tmp_path, profile))


def test_rejects_mismatched_assignment(profile, tmp_path):
    profile["configs"]["512"]["w2"]["BLOCK_SIZE_M"] = 64
    with pytest.raises(ValueError, match="share the M tile"):
        shared.load_yoco_align_moe_profile(write_profile(tmp_path, profile))


@pytest.mark.parametrize(
    "key", ["device_name", "torch_version", "triton_version", "cuda_version"]
)
def test_dependency_mismatch_is_explicit(profile, tmp_path, monkeypatch, runtime, key):
    profile["environment"][key] = "different"
    monkeypatch.setattr(
        shared.envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", write_profile(tmp_path, profile)
    )
    with pytest.raises(ValueError, match="environment mismatch"):
        select()


def test_default_fast_and_other_shapes_do_not_load_profile(monkeypatch, runtime):
    monkeypatch.setattr(shared.envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", None)
    assert select() is None
    monkeypatch.setattr(
        shared.envs, "VLLM_YOCO_ALIGN_MOE_CONFIG", "/does/not/exist.json"
    )
    monkeypatch.setattr(shared.envs, "VLLM_BATCH_INVARIANT", False)
    assert select() is None
    monkeypatch.setattr(shared.envs, "VLLM_BATCH_INVARIANT", True)
    assert select(top_k=4) is None
    assert select(w13_shape=(128, 1920, 1024)) is None
    assert select(w2_shape=(64, 1024, 3840)) is None


@pytest.mark.parametrize(
    "override",
    [
        {"BLOCK_SIZE_M": 16},
        {"BLOCK_SIZE_K": 64},
        {"num_warps": 8},
        {"unknown": 1},
    ],
)
def test_explicit_config_overrides_are_preserved(profile, override):
    base = {**shared._BASE, **override}
    frozen = copy.deepcopy(base)
    assert shared.select_yoco_align_moe_configs(profile, 512, base) is None
    assert base == frozen


@pytest.mark.parametrize("rows", ["0", "0512", "512.0", "-1"])
def test_invalid_token_row_keys(profile, tmp_path, rows):
    profile["configs"][rows] = profile["configs"].pop("512")
    with pytest.raises(ValueError, match="token-row key"):
        shared.load_yoco_align_moe_profile(write_profile(tmp_path, profile))
