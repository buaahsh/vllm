# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO conversion regression tests."""

import pytest
import torch

from convert_to_hf import convert_state_dict, create_hf_config


def test_mega_rejects_weight_reload_before_consuming_or_modifying_weights():
    from vllm.model_executor.models.yoco_weights import load_weights

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    model._yoco_mega_moe = object()

    def replacement():
        raise AssertionError("The frozen model must reject before consuming weights")
        yield

    with pytest.raises(RuntimeError, match="frozen weights"):
        load_weights(model, replacement())
    torch.testing.assert_close(model.weight, torch.ones(1))


def test_yoco_embedding_conversion_preserves_lookup_exactly() -> None:
    weight = torch.arange(35, dtype=torch.bfloat16).view(7, 5)
    tokens = torch.tensor([6, 0, 3, 3, 1])

    converted = convert_state_dict({"tok_embeddings.weight": weight})

    converted_weight = converted["model.embed_tokens.weight"]
    assert torch.equal(converted_weight, weight)
    assert torch.equal(
        torch.nn.functional.embedding(tokens, converted_weight),
        torch.nn.functional.embedding(tokens, weight),
    )


def test_convert_yoco_v3_attention_and_latent_moe_weights() -> None:
    gate = torch.randn(64, 16)
    fc1 = torch.randn(8, 16)
    fc2 = torch.randn(16, 8)
    norm = torch.randn(8)
    state = {
        "layers.0.self_attn.gate_proj.weight": gate,
        "layers.0.mlp.fc1_latent_proj.weight": fc1,
        "layers.0.mlp.fc2_latent_proj.weight": fc2,
        "layers.0.mlp.fc1_latent_norm.weight": norm,
        "layers.0.mlp.fc2_latent_norm.weight": norm.clone(),
    }

    converted = convert_state_dict(state)

    assert torch.equal(converted["model.layers.0.self_attn.lambda_proj.weight"], gate)
    assert torch.equal(converted["model.layers.0.mlp.fc1_latent_proj.weight"], fc1)
    assert torch.equal(converted["model.layers.0.mlp.fc2_latent_proj.weight"], fc2)
    assert torch.equal(converted["model.layers.0.mlp.fc1_latent_norm.weight"], norm)
    assert torch.equal(converted["model.layers.0.mlp.fc2_latent_norm.weight"], norm)


def test_convert_legacy_diff_v3_interleaves_gate_rows() -> None:
    legacy_gate = torch.arange(8).reshape(4, 2)

    converted = convert_state_dict(
        {"layers.0.self_attn.lambda_proj.weight": legacy_gate},
        legacy_diff_v3=True,
    )

    expected = legacy_gate[[0, 2, 1, 3]]
    assert torch.equal(
        converted["model.layers.0.self_attn.lambda_proj.weight"],
        expected,
    )


def test_create_yoco_v3_latent_config(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_norm": False,
            "qk_rms_clip": True,
            "qk_rms_gamma": True,
            "diff_v2": False,
            "diff_v3": True,
            "yoco_cross_layers": 10,
            "yoco_window_size": 512,
            "universal_loop": 1,
            "moe": True,
            "moe_expert_num": 128,
            "moe_top_k": 8,
            "moe_ffn_dim": 3840,
            "moe_latent_dim": 1024,
            "moe_latent_norm": True,
            "d_shared_expert": 1280,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v3"]
    assert not config["diff_v2"]
    assert not config["diff_attention"]
    assert config["qk_rms_gamma"]
    assert config["moe_latent_dim"] == 1024
    assert config["moe_latent_norm"]


def test_create_yoco_v2_defaults_to_weight_free_qk_clip(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_rms_clip": True,
            "diff_v2": True,
            "diff_v3": False,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v2"]
    assert not config["diff_v3"]
    assert not config["qk_rms_gamma"]


def test_create_legacy_diff_both_lamb_as_v3(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "diff_attention": True,
            "diff_both_lamb": True,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v3"]
    assert not config["diff_v2"]
    assert not config["diff_attention"]


@pytest.mark.parametrize("num_redundant", [0, 2, 6])
def test_yoco_expert_checkpoint_aliases_reload_without_replacing_storage(num_redundant):
    from types import SimpleNamespace

    import pytest

    from vllm.model_executor.models.yoco import YOCOForCausalLM

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.config = SimpleNamespace(
        num_hidden_layers=1,
        yoco_cross_layers=0,
        moe_intermediate_size=2,
        num_experts=2,
    )
    causal_lm.num_redundant_experts = num_redundant
    causal_lm.model = torch.nn.Module()
    causal_lm.model.embed_tokens = torch.nn.Embedding(1, 3)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    routed = torch.nn.Module()
    layer.mlp.experts.routed_experts = routed
    causal_lm.model.layers = torch.nn.ModuleList([layer])
    causal_lm._rotary_caches_initialized = False
    causal_lm._router_weight_caches_initialized = False
    num_physical = 2 + num_redundant
    routed.w13_weight = torch.nn.Parameter(
        torch.empty(num_physical, 4, 3), requires_grad=False
    )
    routed.w2_weight = torch.nn.Parameter(
        torch.empty(num_physical, 3, 2), requires_grad=False
    )

    def expert_loader(param, weight, name, shard, expert_id):
        target = param[expert_id]
        if shard in ("w1", "w3"):
            target = target.narrow(0, 0 if shard == "w1" else 2, 2)
        target.copy_(weight)

    routed.w13_weight.weight_loader = expert_loader
    routed.w2_weight.weight_loader = expert_loader
    prefix = "model.layers.0.mlp.experts."
    addresses = [routed.w13_weight.data_ptr(), routed.w2_weight.data_ptr()]
    for offset in (0, 100):
        w13 = torch.arange(24).reshape(8, 3).float() + offset
        w2 = torch.arange(12).reshape(6, 2).float() + offset
        loaded = causal_lm.load_weights(
            [
                (prefix + "w13_weight", w13),
                (prefix + "w2_weight", w2),
                ("unused.quant_scale", torch.ones(1)),
            ]
        )
        assert loaded == {
            prefix + "routed_experts." + key for key in ("w13_weight", "w2_weight")
        }
        for physical_id in range(num_physical):
            logical_id = physical_id % 2
            assert torch.equal(
                routed.w13_weight[physical_id], w13.view(2, 4, 3)[logical_id]
            )
            assert torch.equal(
                routed.w2_weight[physical_id], w2.view(2, 3, 2)[logical_id]
            )
        assert [routed.w13_weight.data_ptr(), routed.w2_weight.data_ptr()] == addresses
        assert causal_lm._yoco_weight_load_report.ignored == ("unused.quant_scale",)

    # Two different parameters must not claim the same legacy checkpoint key.
    layer.mlp.experts.w13_weight = torch.nn.Parameter(
        torch.zeros_like(routed.w13_weight)
    )
    with pytest.raises(ValueError, match="Ambiguous YOCO checkpoint parameter alias"):
        causal_lm.load_weights([])


def _mtp_training_tensors():
    from tools.yoco_serving.prepare_yoco_mtp_hf import PREFIX
    from vllm.transformers_utils.configs.yoco_mtp import expected_mtp_shapes

    config = dict(hidden_size=8, cross_head=2, head_dim=4, d_ffn=4, diff_v3=True)
    tensors = {}
    for name, shape in expected_mtp_shapes(config).items():
        value = torch.arange(torch.tensor(shape).prod()).reshape(shape).bfloat16()
        if name == "norm.weight":
            tensors[PREFIX + "mtp_norm.weight"] = value
            continue
        suffix = name.removeprefix("mtp.")
        if suffix == "ffn.gate_up_proj.weight":
            tensors[PREFIX + "mtp_block.ffn.gate_proj.weight"] = value[:4]
            tensors[PREFIX + "mtp_block.ffn.up_proj.weight"] = value[4:]
            continue
        if suffix == "self_attn.q_proj.weight":
            value = value.reshape(4, 4, 8)
        suffix = suffix.replace("self_attn.lambda_proj.", "self_attn.gate_proj.")
        tensors[PREFIX + "mtp_block." + suffix] = value
    return config, tensors


def test_mtp_conversion_preserves_head_order_and_gate_up_values():
    from tools.yoco_serving.prepare_yoco_mtp_hf import PREFIX, convert_mtp_weights

    config, tensors = _mtp_training_tensors()
    actual = convert_mtp_weights(tensors, config)
    assert len(actual) == 12
    assert torch.equal(
        actual["mtp.self_attn.q_proj.weight"].reshape(4, 4, 8),
        tensors[PREFIX + "mtp_block.self_attn.q_proj.weight"],
    )
    gate, up = actual["mtp.ffn.gate_up_proj.weight"].chunk(2)
    assert torch.equal(gate, tensors[PREFIX + "mtp_block.ffn.gate_proj.weight"])
    assert torch.equal(up, tensors[PREFIX + "mtp_block.ffn.up_proj.weight"])
    assert torch.equal(
        actual["mtp.self_attn.lambda_proj.weight"],
        tensors[PREFIX + "mtp_block.self_attn.gate_proj.weight"],
    )


def test_mtp_conversion_rejects_missing_q_norm_and_nonfinite_weight():
    import pytest

    from tools.yoco_serving.prepare_yoco_mtp_hf import PREFIX, convert_mtp_weights

    config, tensors = _mtp_training_tensors()
    tensors.pop(PREFIX + "mtp_block.self_attn.q_norm.weight")
    with pytest.raises(ValueError, match="inventory"):
        convert_mtp_weights(tensors, config)
    config, tensors = _mtp_training_tensors()
    tensors[PREFIX + "mtp_block.proj.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        convert_mtp_weights(tensors, config)
