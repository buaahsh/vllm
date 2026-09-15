# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO prefill regression tests."""

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.models.yoco import YOCOCrossBlock, YOCOForCausalLM


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
