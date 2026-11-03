# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO's EPLB registration, weight ownership and dynamic routing contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.yoco_moe import YOCOMoE
from vllm.model_executor.layers.yoco_ops.routing import _yoco_topk_routing
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.model_executor.models.yoco import YOCOForCausalLM


def _model_with_moe_layers(num_layers: int) -> YOCOForCausalLM:
    model = YOCOForCausalLM.__new__(YOCOForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Identity()])
    for _ in range(num_layers):
        routed = RoutedExperts.__new__(RoutedExperts)
        nn.Module.__init__(routed)
        routed.local_num_experts = 3
        routed.w13_weight = nn.Parameter(torch.randn(3, 8, 4), requires_grad=False)
        routed.w2_weight = nn.Parameter(torch.randn(3, 4, 4), requires_grad=False)
        # Quantization scales must migrate alongside their expert's weights.
        routed.w13_weight_scale_inv = nn.Parameter(
            torch.ones(3, 2, 1), requires_grad=False
        )
        runner = MoERunner.__new__(MoERunner)
        nn.Module.__init__(runner)
        runner.moe_config = SimpleNamespace(
            num_logical_experts=4, num_experts=6, num_local_experts=3
        )
        runner.routed_experts = routed
        runner.router = CustomRoutingRouter(2, 6, _yoco_topk_routing, EplbLayerState())
        moe = YOCOMoE.__new__(YOCOMoE)
        nn.Module.__init__(moe)
        moe.experts = runner
        # Shared and latent parameters are not expert-indexed and must stay put.
        moe.shared_gate = nn.Linear(4, 1)
        moe.fc1_latent_proj = nn.Linear(4, 4)
        layer = nn.Module()
        layer.mlp = moe
        model.model.layers.append(layer)
    model._init_moe_metadata()
    return model


@pytest.mark.parametrize("num_layers", [0, 2])
def test_yoco_registers_only_weight_owning_moe_layers(num_layers):
    model = _model_with_moe_layers(num_layers)
    # The same layers are also reused by compiled self/cross blocks.
    model.model.repeated_block = model.model.layers
    model._init_moe_metadata()
    assert model.num_moe_layers == num_layers
    if num_layers == 0:
        assert get_mixture_of_experts_model(model) is None
        return
    assert get_mixture_of_experts_model(model) is model
    assert (model.num_logical_experts, model.num_physical_experts) == (4, 6)
    assert model.num_local_physical_experts == 3
    assert model.num_expert_groups == 1
    assert model.num_shared_experts == 1
    loads = torch.zeros(num_layers, 6, dtype=torch.int32)
    mapping = torch.full((num_layers, 4, 6), -1, dtype=torch.int64)
    counts = torch.ones(num_layers, 4, dtype=torch.int64)
    model.set_eplb_state(loads, mapping, counts)
    for i, layer in enumerate(model.moe_layers):
        assert (
            layer.eplb_state.logical_to_physical_map.data_ptr() == mapping[i].data_ptr()
        )
        assert layer.eplb_state.expert_load_view.data_ptr() == loads[i].data_ptr()
        registered = {w.data_ptr() for w in model.expert_weights[i]}
        assert registered == {p.data_ptr() for p in layer.routed_experts.parameters()}
    mapping[1, 0, 0] = 5
    assert model.moe_layers[1].eplb_state.logical_to_physical_map[0, 0] == 5


def test_yoco_rejects_checkpoint_reload_after_eplb_registration():
    model = _model_with_moe_layers(1)
    model.set_eplb_state(torch.zeros(1, 6), torch.zeros(1, 4, 6), torch.ones(1, 4))
    before = [w.clone() for w in model.expert_weights[0]]

    def weights():
        raise AssertionError("Must reject before consuming the checkpoint")
        yield

    with pytest.raises(RuntimeError, match="reload after EPLB"):
        model.load_weights(weights())
    for actual, expected in zip(model.expert_weights[0], before):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_graph_observes_new_replica_map():
    device = torch.device("cuda")
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]] * 8, device=device)
    hidden = torch.zeros(8, 4, device=device)
    mapping = torch.tensor([[0, 4], [1, 5], [2, -1], [3, -1]], device=device)
    counts = torch.tensor([2, 2, 1, 1], device=device)
    loads = torch.zeros(6, dtype=torch.int32, device=device)
    state = EplbLayerState()
    state.expert_load_view = loads
    state.logical_to_physical_map = mapping
    state.logical_replica_count = counts
    state.should_record_tensor = torch.tensor(True, device=device)
    state.num_unpadded_tokens_tensors = [torch.tensor(5, device=device)]
    router = CustomRoutingRouter(2, 6, _yoco_topk_routing, state)
    reference_weights, logical_ids = _yoco_topk_routing(hidden, logits, 2, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            router.select_experts(hidden, logits)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, physical_ids = router.select_experts(hidden, logits)
    for new_map, physical_to_logical in [
        ([[0, 4], [1, 5], [2, -1], [3, -1]], [0, 1, 2, 3, 0, 1]),
        ([[2, 5], [0, 3], [1, -1], [4, -1]], [1, 2, 0, 1, 3, 0]),
    ]:
        mapping.copy_(torch.tensor(new_map, device=device))
        loads.zero_()
        graph.replay()
        torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
        inverse = torch.tensor(physical_to_logical, device=device)
        assert torch.equal(inverse[physical_ids.long()], logical_ids.long())
        assert physical_ids[:, 0].unique().numel() == 2
        expected_load = torch.bincount(physical_ids[:5].flatten().long(), minlength=6)
        torch.testing.assert_close(loads.long(), expected_load)
    state.should_record_tensor.fill_(False)
    loads.zero_()
    graph.replay()
    assert loads.sum().item() == 0
