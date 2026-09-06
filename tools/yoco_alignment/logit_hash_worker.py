# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hash complete logits without constructing per-vocabulary Python logprob objects."""

import hashlib

import torch

from vllm.v1.worker.gpu_worker import Worker


class LogitHashWorker(Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self.logit_hashes = {}

        def record(module, inputs, output):
            request_ids = self.model_runner.input_batch.req_ids
            if not request_ids or len(request_ids) != output.shape[0]:
                return  # Startup profiling has no live requests.
            logits = output.detach().contiguous().view(torch.uint8).cpu()
            for index, request_id in enumerate(request_ids):
                digest = hashlib.sha256(logits[index].numpy().tobytes()).hexdigest()
                self.logit_hashes.setdefault(request_id, []).append(digest)

        self.get_model().logits_processor.register_forward_hook(record)
        return result

    def reset_logit_hashes(self):
        self.logit_hashes = {}

    def get_logit_hashes(self):
        return self.logit_hashes
