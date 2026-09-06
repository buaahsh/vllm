# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic worker that records first-layer tensors inside CUDA Graphs."""

from pathlib import Path

import torch

from vllm.v1.worker.gpu_worker import Worker


class GraphTraceWorker(Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        root = self.get_model()
        self.invariant_trace_buffers = {}
        counts = {}

        def hook(name):
            def capture(module, inputs, output):
                if name == "embed_tokens":
                    counts.clear()
                count = counts.get(name, 0)
                counts[name] = count + 1
                for boundary, values in (("inputs", inputs), ("output", output)):
                    if isinstance(values, torch.Tensor):
                        values = (values,)
                    if not isinstance(values, (tuple, list)):
                        continue
                    for index, value in enumerate(values):
                        if (
                            not isinstance(value, torch.Tensor)
                            or not value.is_cuda
                            or value.ndim == 0
                            or value.shape[0] > 128
                        ):
                            continue
                        key = (name, count, boundary, index, tuple(value.shape))
                        if key not in self.invariant_trace_buffers:
                            self.invariant_trace_buffers[key] = torch.empty_like(value)
                        self.invariant_trace_buffers[key].copy_(value.detach())

            return capture

        for name, module in root.model.named_modules():
            if (
                name == "embed_tokens"
                or name == "layers.0"
                or name.startswith("layers.0.")
            ):
                module.register_forward_hook(hook(name))
        return result

    def save_graph_trace(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.invariant_trace_buffers.items()}, path)
