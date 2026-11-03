# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO language backbone with the Kimi-K3 vision tower and projector."""

from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.kimi_k3 import KimiK3VisionConfig
from vllm.transformers_utils.configs.yoco import YOCOConfig


class YOCOVLConfig(PretrainedConfig):
    model_type = "yoco_vl"
    sub_configs = {"text_config": YOCOConfig, "vision_config": KimiK3VisionConfig}

    def __init__(
        self,
        vision_config: dict | KimiK3VisionConfig | None = None,
        text_config: dict | YOCOConfig | None = None,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        image_placeholder_token_id: int = 0,
        image_placeholder: str = "<image>",
        video_placeholder: str = "<video>",
        vision_max_image_tokens: int = 4096,
        vision_align_mode: str = "resize",
        vision_patch_limit_on_one_side: int = 512,
        vision_image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        vision_image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        pad_token_id: int = 154856,
        **kwargs,
    ):
        self.vision_config = (
            KimiK3VisionConfig(**(vision_config or {}))
            if isinstance(vision_config, dict) or vision_config is None
            else vision_config
        )
        self.text_config = (
            YOCOConfig(**(text_config or {}))
            if isinstance(text_config, dict) or text_config is None
            else text_config
        )
        if hasattr(self.text_config, "hidden_size"):
            self.vision_config.text_hidden_size = self.text_config.hidden_size
        if tuple(self.vision_config.merge_kernel_size) != (2, 2):
            raise ValueError("YOCO-VL requires a 2x2 patch merge kernel")
        if vision_max_image_tokens <= 0:
            raise ValueError("vision_max_image_tokens must be positive")
        if vision_align_mode not in ("resize", "pad"):
            raise ValueError("vision_align_mode must be resize or pad")
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_placeholder_token_id = image_placeholder_token_id
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.vision_max_image_tokens = vision_max_image_tokens
        self.vision_align_mode = vision_align_mode
        self.vision_patch_limit_on_one_side = vision_patch_limit_on_one_side
        self.vision_image_mean = vision_image_mean
        self.vision_image_std = vision_image_std
        kwargs.setdefault("architectures", ["YOCOVLForConditionalGeneration"])
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
