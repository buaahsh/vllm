# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

import vllm.envs as envs
from vllm.config.yoco import get_yoco_execution_mode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class YOCOModelState(DefaultModelState):
    fast_prefill_min_query_len = 8
    coordinate_fast_prefill = True

    def custom_sampler(self, sampler):
        if (
            envs.VLLM_YOCO_BF16_SAMPLING
            and get_yoco_execution_mode(self.vllm_config.additional_config) == "fast"
        ):
            from vllm.v1.worker.gpu.sample.yoco_bf16 import YocoBf16GreedySampler

            wrapped = YocoBf16GreedySampler(sampler)
            rejection = None
            if self.vllm_config.speculative_config is not None:
                from vllm.v1.worker.gpu.sample.sampler import Sampler
                from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
                    RejectionSampler,
                )

                rejection = RejectionSampler(
                    cast(Sampler, wrapped),
                    self.vllm_config.speculative_config,
                    self.device,
                )
            return wrapped, rejection
        return None

    @property
    def force_eager(self) -> bool:
        config = self.vllm_config
        transfer = config.kv_transfer_config
        return bool(
            config.cache_config.kv_sharing_fast_prefill
            and transfer is not None
            and transfer.kv_role == "kv_producer"
        )

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        if self.force_eager:
            # Dedicated P never executes the cross decoder, including on idle
            # DP ranks. Do not capture dense FULL graphs that it cannot replay.
            return AttentionCGSupport.NEVER, "YOCO KV-only prefill"
        return super().get_additional_cg_support()

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        if self.force_eager:
            inputs["kv_only_prefill"] = True
        return inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.force_eager:
            inputs["kv_only_prefill"] = True
        return inputs
