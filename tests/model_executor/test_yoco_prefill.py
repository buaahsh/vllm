# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO prefill regression tests."""

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.models.yoco import YOCOCrossBlock, YOCOForCausalLM, YOCOModel


@pytest.mark.parametrize("mode", ["align", "fast"])
def test_aux_features_use_final_encoder_pass_and_pre_final_norm(mode):
    class Layer(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, positions, hidden, *args):
            return hidden + self.increment

        def forward_with_residual(self, positions, hidden, *args, input_residual=None):
            residual = hidden if input_residual is None else hidden + input_residual
            return torch.full_like(hidden, self.increment), residual

    class Norm(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, hidden, residual=None):
            if residual is None:
                return (hidden * self.scale).to(torch.bfloat16)
            value = hidden + residual
            return (value * self.scale).to(torch.bfloat16), value

    model = YOCOModel.__new__(YOCOModel)
    torch.nn.Module.__init__(model)
    model.do_not_compile = True
    model.execution_mode = mode
    model.residual_dtype = torch.float32
    model.universal_loop = 2
    model.first_cross_layer_idx = 3
    model.num_hidden_layers = 6
    model.yoco_cross_layers = 3
    model.layers = torch.nn.ModuleList(Layer(i + 1) for i in range(6))
    model.yoco_norm = Norm(1)
    model.norm = Norm(2)
    model.project_yoco_kv = lambda h: (h, h)
    model.normalize_yoco_kv = lambda k, v: (k, v)
    model.aux_hidden_state_layers = (3, 5, 6)
    result, features = model(None, torch.arange(2), inputs_embeds=torch.ones(2, 4))
    assert len(features) == 3
    for actual, value in zip(features, [13, 22, 28], strict=True):
        torch.testing.assert_close(
            actual, torch.full((2, 4), float(value), dtype=torch.bfloat16)
        )
    torch.testing.assert_close(result, torch.full((2, 4), 56.0, dtype=torch.bfloat16))
    model._fourval_export_postnorm = True
    postnorm_result, postnorm_features = model(
        None, torch.arange(2), inputs_embeds=torch.ones(2, 4)
    )
    assert len(postnorm_features) == 4
    torch.testing.assert_close(postnorm_features[-1], postnorm_result)
    for before, after in zip(features, postnorm_features[:-1], strict=True):
        torch.testing.assert_close(before, after)
    model._fourval_export_postnorm = False
    model.aux_hidden_state_layers = ()
    torch.testing.assert_close(
        model(None, torch.arange(2), inputs_embeds=torch.ones(2, 4)), result
    )


def test_aux_extraction_rejects_decoder_skipping_prefill():
    model = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(model)
    model.fast_prefill_enabled = True
    with pytest.raises(ValueError, match="full decoder execution"):
        model.set_aux_hidden_state_layers((14, 21, 27, 28))


def test_fast_prefill_runs_all_cross_layers_on_compact_tokens() -> None:
    calls = []

    class RecordingCrossLayer(torch.nn.Module):
        def forward(
            self,
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=None,
            skip_kv_cache_update=False,
        ):
            calls.append(
                {
                    "num_tokens": hidden_states.shape[0],
                    "loop_idx": loop_idx,
                    "has_cache_dependency": kv_cache_dummy_dep is not None,
                    "skip_kv_cache_update": skip_kv_cache_update,
                }
            )
            return hidden_states + 1

    block = YOCOCrossBlock.__new__(YOCOCrossBlock)
    torch.nn.Module.__init__(block)
    block._cross_layers = [RecordingCrossLayer() for _ in range(10)]

    num_logits_tokens = 3
    hidden_states = torch.zeros(num_logits_tokens, 8)
    output = YOCOCrossBlock.forward(
        block,
        torch.arange(num_logits_tokens),
        hidden_states,
        torch.zeros(num_logits_tokens, 2),
        torch.zeros(num_logits_tokens, 2),
        torch.empty(0),
    )

    assert len(calls) == 10
    assert all(call["num_tokens"] == num_logits_tokens for call in calls)
    assert all(call["loop_idx"] == 0 for call in calls)
    assert calls[0]["has_cache_dependency"]
    assert calls[0]["skip_kv_cache_update"]
    assert not any(call["has_cache_dependency"] for call in calls[1:])
    assert not any(call["skip_kv_cache_update"] for call in calls[1:])
    torch.testing.assert_close(output, hidden_states + 10)


def test_fast_cross_block_carries_residual_between_layers() -> None:
    class ResidualCrossLayer(torch.nn.Module):
        def __init__(self, output_value: float) -> None:
            super().__init__()
            self.output_value = output_value

        def forward_with_residual(
            self,
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=None,
            skip_kv_cache_update=False,
            input_residual=None,
        ):
            del (
                positions,
                loop_idx,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep,
                skip_kv_cache_update,
            )
            residual = (
                hidden_states
                if input_residual is None
                else input_residual + hidden_states.float()
            )
            output = torch.full_like(
                hidden_states,
                self.output_value,
                dtype=torch.bfloat16,
            )
            return output, residual

    block = YOCOCrossBlock.__new__(YOCOCrossBlock)
    torch.nn.Module.__init__(block)
    block._cross_layers = [
        ResidualCrossLayer(1.0),
        ResidualCrossLayer(2.0),
        ResidualCrossLayer(3.0),
    ]
    block.execution_mode = "fast"

    hidden_states = torch.zeros(2, 8)
    output = YOCOCrossBlock.forward(
        block,
        torch.arange(2),
        hidden_states,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.empty(0),
    )

    # Layer inputs materialize as 0, 1, and 3; the final pending output is 3.
    torch.testing.assert_close(output, torch.full_like(output, 6.0))


def test_kv_only_prefill_skips_every_cross_layer() -> None:
    class SelfBlock(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None):
            hidden_states = torch.full((positions.numel(), 8), 2.0)
            kv = torch.zeros(positions.numel(), 2)
            return hidden_states, kv, kv, torch.empty(0)

    class CrossBlock(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("KV-only prefill must not execute cross layers")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_block = SelfBlock()
            self.cross_block = CrossBlock()
            self.norm = torch.nn.Identity()
            self.full_model_warmed = True

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.model = Model()

    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,  # type: ignore[arg-type]
        slot_mapping={},
    )
    with override_forward_context(context):
        output = YOCOForCausalLM._fast_prefill_forward(
            causal_lm,
            input_ids=torch.arange(4),
            positions=torch.arange(4),
            kv_only_prefill=True,
        )

    torch.testing.assert_close(output, torch.full((4, 8), 2.0))
