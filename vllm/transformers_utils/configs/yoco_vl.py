# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for YOCO vision-language checkpoints."""

from vllm.transformers_utils.configs.moonvit import MoonViTConfig
from vllm.transformers_utils.configs.yoco import YOCOConfig

from transformers.configuration_utils import PretrainedConfig


class YOCOVLConfig(PretrainedConfig):
    model_type = "yoco_vl"

    def __init__(
        self,
        vision_config: dict | MoonViTConfig | None = None,
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
        if vision_config is None:
            vision_config = MoonViTConfig(num_hidden_layers=10)
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = YOCOConfig()
        elif isinstance(text_config, dict):
            text_config = YOCOConfig(**text_config)
        self.text_config = text_config

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

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["YOCOVLForConditionalGeneration"]
        super().__init__(pad_token_id=pad_token_id, **kwargs)
