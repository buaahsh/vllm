# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO prefill regression tests."""

from types import SimpleNamespace

import pytest
import torch

from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
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


def test_fast_prefill_aux_accepts_only_shared_kv_final_residual():
    model = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(model)
    model.fast_prefill_enabled = True
    model.fast_prefill_shared_kv_aux = True
    model.model = SimpleNamespace(
        num_hidden_layers=20,
        _set_aux_hidden_state_layers=lambda layers: setattr(
            model.model, "layers", layers
        ),
    )
    model.set_aux_hidden_state_layers((20,))
    assert model.model.layers == (20,)
    for layers in [(19,), (19, 20), (20, 21)]:
        with pytest.raises(ValueError, match="final raw residual"):
            model.set_aux_hidden_state_layers(layers)


def test_fast_prefill_aux_preserves_raw_selected_rows_and_rejection_anchors():
    """Only retained rows are final hidden; every possible rejection anchor is kept."""
    from vllm.model_executor.models.yoco_prefill import _fast_prefill_forward

    hidden = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    # A chunked prefill row, followed by all verification rows of another request.
    indices = torch.tensor([7, 8, 9, 10, 11, 0])
    core = SimpleNamespace(
        self_block=lambda *args: (hidden, hidden, hidden, torch.empty(0)),
        cross_block=lambda p, h, k, v, dep: h + 100,
        full_model_warmed=True,
        fuse_fp8_shared_kv=False,
        aux_hidden_state_layers=(20,),
        fp_positions=torch.empty(12, dtype=torch.int64),
        fp_hidden_states=torch.empty_like(hidden),
        fp_yoco_key=torch.empty_like(hidden),
        fp_yoco_value=torch.empty_like(hidden),
        norm=lambda h: (h * 2).to(torch.bfloat16),
    )
    model = SimpleNamespace(
        model=core, _get_fast_prefill_indices=lambda: (indices, 5, None)
    )
    context = ForwardContext(no_compile_layers={}, attn_metadata=None, slot_mapping={})
    with override_forward_context(context):
        normalized, (raw,) = _fast_prefill_forward(model, None, torch.arange(12))
    assert raw.shape == normalized.shape == hidden.shape
    assert raw.dtype == normalized.dtype == torch.bfloat16
    for anchor in [7, 11, 10, 9, 8]:
        torch.testing.assert_close(raw[anchor], (hidden[anchor] + 100).to(raw.dtype))
        torch.testing.assert_close(normalized[anchor], raw[anchor] * 2)
    # The padding row must not replace the real full-batch row at index zero.
    torch.testing.assert_close(raw[0], hidden[0].to(raw.dtype))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "kv_dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("num_selected", [0, 1, 7, 257])
def test_fused_prefill_gather_preserves_bits_and_unused_storage(
    hidden_dtype, kv_dtype, strided, num_selected
) -> None:
    """Copy arbitrary bits, repeated/negative indices and tails without conversion."""
    from vllm.model_executor.layers.yoco_ops.prefill import gather_prefill_inputs

    torch.manual_seed(20261020)
    step = 2 if strided else 1

    def storage(rows, width, dtype):
        result = torch.empty((rows * step, width * step), device="cuda", dtype=dtype)
        raw = result.view(torch.uint8)
        raw.copy_(torch.randint(256, raw.shape, dtype=torch.uint8, device="cuda"))
        return result

    source_storage = [
        storage(263, width, dtype)
        for width, dtype in [(513, hidden_dtype), (129, kv_dtype), (67, kv_dtype)]
    ]
    output_storage = [
        storage(num_selected + 2, width, dtype)
        for width, dtype in [(513, hidden_dtype), (129, kv_dtype), (67, kv_dtype)]
    ]
    expected = [x.clone() for x in output_storage]
    sources = [x[::step, ::step] for x in source_storage]
    outputs = [x[::step, ::step] for x in output_storage]
    indices = torch.zeros(num_selected * step, dtype=torch.int64, device="cuda")[::step]
    indices[::2] = -1
    positions = torch.arange(263 * step, device="cuda")[::step]
    out_positions = torch.full(((num_selected + 2) * step,), -9, device="cuda")
    expected_positions = out_positions.clone()
    expected_positions[::step][:num_selected] = positions[indices]
    bit_dtypes = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    for source, target in zip(sources, expected):
        bits = bit_dtypes[source.element_size()]
        target.view(bits)[::step, ::step][:num_selected] = source.view(bits)[indices]

    gather_prefill_inputs(indices, positions, *sources, out_positions[::step], *outputs)

    assert torch.equal(out_positions, expected_positions)
    for actual, reference in zip(output_storage, expected):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_prefill_gather_graph_reads_updated_inputs(kv_dtype) -> None:
    """Replay must retain buffer addresses while reading new row IDs and source bits."""
    from vllm.model_executor.layers.yoco_ops.prefill import gather_prefill_inputs

    indices = torch.tensor([4, 1, 1], device="cuda")
    sources = [
        torch.arange(8, device="cuda"),
        torch.randn(8, 513, device="cuda"),
        torch.zeros(8, 129, device="cuda", dtype=kv_dtype),
        torch.zeros(8, 129, device="cuda", dtype=kv_dtype),
    ]
    outputs = [torch.empty_like(x[:3]) for x in sources]
    addresses = [x.data_ptr() for x in outputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gather_prefill_inputs(indices, *sources, *outputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_prefill_inputs(indices, *sources, *outputs)
    for row_ids in ([7, 0, 3], [2, 2, 6]):
        indices.copy_(torch.tensor(row_ids, device="cuda"))
        for source in sources:
            raw = source.view(torch.uint8)
            raw.copy_(torch.randint(256, raw.shape, device="cuda", dtype=torch.uint8))
        graph.replay()
        assert [x.data_ptr() for x in outputs] == addresses
        for source, output in zip(sources, outputs):
            assert torch.equal(
                output.view(torch.uint8).reshape(3, -1),
                source.view(torch.uint8).reshape(8, -1)[indices],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fail_cross", [False, True])
def test_prefill_gather_preserves_clone_padding_and_dp_context(fail_cross) -> None:
    """Cross-block reuse cannot corrupt untouched rows or leak compact DP metadata."""
    from vllm.model_executor.models.yoco_prefill import _fast_prefill_forward

    hidden = torch.arange(40, device="cuda", dtype=torch.float32).reshape(5, 8)
    original = hidden.clone()
    key = torch.arange(20, device="cuda", dtype=torch.float32).reshape(5, 4)
    value = key + 20
    positions = torch.arange(5, device="cuda")
    indices = torch.tensor([1, 4, 0], device="cuda")
    compact_dp = torch.tensor([3, 3])
    original_dp = DPMetadata(torch.tensor([5, 5]))
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,  # type: ignore[arg-type]
        slot_mapping={},
        dp_metadata=original_dp,
    )
    dependency = torch.empty(0, device="cuda")
    model = SimpleNamespace(
        self_block=lambda *args: (hidden, key, value, dependency),
        fuse_fp8_shared_kv=False,
        fp_positions=torch.empty_like(positions),
        fp_hidden_states=torch.empty_like(hidden),
        fp_yoco_key=torch.empty_like(key),
        fp_yoco_value=torch.empty_like(value),
        norm=lambda x: x * 0.5,
    )

    def cross(p, h, k, v, dep):
        assert torch.equal(context.dp_metadata.num_tokens_across_dp_cpu, compact_dp)
        assert dep is dependency
        for actual, source, buffer in zip(
            (p, h, k, v),
            (positions, original, key, value),
            (
                model.fp_positions,
                model.fp_hidden_states,
                model.fp_yoco_key,
                model.fp_yoco_value,
            ),
        ):
            assert actual.data_ptr() == buffer.data_ptr()
            assert torch.equal(actual, source[indices])
        hidden.fill_(-123)
        if fail_cross:
            raise RuntimeError("cross failed")
        return h + 10

    model.cross_block = cross
    causal_lm = SimpleNamespace(
        model=model,
        _get_fast_prefill_indices=lambda: (indices, 2, compact_dp),
    )
    with override_forward_context(context):
        if fail_cross:
            with pytest.raises(RuntimeError, match="cross failed"):
                _fast_prefill_forward(causal_lm, None, positions)
        else:
            output = _fast_prefill_forward(causal_lm, None, positions)
            expected = original.clone()
            expected[indices[:2]] += 10
            assert torch.equal(output, expected * 0.5)
        assert context.dp_metadata is original_dp
