# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal HF config for the YOCO (You Only Cache Once) checkpoint format.

The reference YOCO HF checkpoint stores all architecture knobs as flat
attributes on ``config.json`` with ``model_type = "yoco"``. The native
vLLM implementation at
``vllm/model_executor/models/yoco.py`` consumes these attributes via
generic ``getattr(config, name, None)`` lookups (see ``_cfg_int``), so the
config class itself only needs to advertise ``model_type = "yoco"`` and
accept arbitrary ``**kwargs``. The base ``PretrainedConfig.__init__`` then
copies every entry from ``config.json`` onto the instance as an attribute.

Registered under ``model_type = "yoco"`` via
``vllm.transformers_utils.config._CONFIG_REGISTRY``.
"""
from transformers import PretrainedConfig


class YOCOConfig(PretrainedConfig):
    model_type = "yoco"

    # Expose HF-canonical attribute names that the rest of vLLM probes for
    # (e.g. `num_experts` in `get_num_experts`) as aliases of the YOCO-native
    # fields actually stored in ``config.json``. Lookups via
    # ``getattr(config, canonical_name)`` are routed to the YOCO name by
    # ``PretrainedConfig``.
    attribute_map = {
        "num_experts": "moe_expert_num",
        "num_local_experts": "moe_expert_num",
        "num_experts_per_tok": "moe_top_k",
    }

    def __init__(self, **kwargs):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["YOCOForCausalLM"]
        super().__init__(**kwargs)
