# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO vision-language model support."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import BatchFeature
from transformers.activations import GELUActivation

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.moonvit import MoonVitPretrainedModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.moonvit import MoonViTConfig
from vllm.transformers_utils.configs.yoco_vl import YOCOVLConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .vision import is_vit_use_data_parallel, run_dp_sharded_mrope_vision_model


@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024


def _navit_resize_image(
    width: int,
    height: int,
    *,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    max_image_tokens: int | None,
    align_mode: str,
) -> dict[str, int]:
    if align_mode not in ("pad", "resize"):
        raise ValueError(f"Unsupported vision align_mode={align_mode!r}")

    effective_patch_limit = in_patch_limit
    if max_image_tokens is not None:
        effective_patch_limit = min(
            in_patch_limit, max_image_tokens * merge_kernel_size**2
        )

    scale_by_area = math.sqrt(
        effective_patch_limit
        / (
            max(1.0, width // patch_size)
            * max(1.0, height // patch_size)
        )
    )
    scale_by_width = patch_limit_on_one_side * patch_size / width
    scale_by_height = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, scale_by_area, scale_by_width, scale_by_height)

    new_width = min(
        max(1, int(width * scale)), patch_limit_on_one_side * patch_size
    )
    new_height = min(
        max(1, int(height * scale)), patch_limit_on_one_side * patch_size
    )

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor
    pad_width = (factor - new_width % factor) % factor
    if align_mode == "resize":
        new_height += pad_height
        new_width += pad_width
        pad_height = 0
        pad_width = 0

    token_height = (new_height + pad_height) // factor
    token_width = (new_width + pad_width) // factor
    return {
        "new_width": new_width,
        "new_height": new_height,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "num_tokens": token_height * token_width,
    }


def _to_pil_image(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D image array, got shape {array.shape}")
        if array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                max_value = float(np.nanmax(array)) if array.size else 1.0
                if max_value <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _patchify(pixel_values: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    _, height, width, channels = pixel_values.shape
    assert channels == 3
    patches = pixel_values.reshape(
        1,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, channels, patch_size, patch_size)
    grid_hw = np.array([height // patch_size, width // patch_size], dtype=np.int64)
    return patches, grid_hw


def _process_image(
    image: object,
    config: YOCOVLConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    image = _to_pil_image(image).convert("RGB")
    width, height = image.size
    vision_config = config.vision_config
    merge_kernel_size = vision_config.merge_kernel_size[0]

    resize_info = _navit_resize_image(
        width,
        height,
        patch_size=vision_config.patch_size,
        merge_kernel_size=merge_kernel_size,
        in_patch_limit=16384,
        patch_limit_on_one_side=config.vision_patch_limit_on_one_side,
        max_image_tokens=config.vision_max_image_tokens,
        align_mode=config.vision_align_mode,
    )

    image = image.resize(
        (resize_info["new_width"], resize_info["new_height"]),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image)
    if resize_info["pad_height"] or resize_info["pad_width"]:
        array = np.pad(
            array,
            (
                (0, resize_info["pad_height"]),
                (0, resize_info["pad_width"]),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

    patches, grid_hw = _patchify(array[np.newaxis, ...], vision_config.patch_size)
    patches = patches.astype(np.float32, copy=False)
    patches /= np.float32(255.0)

    mean = np.asarray(config.vision_image_mean, dtype=np.float32)
    std = np.asarray(config.vision_image_std, dtype=np.float32)
    patches -= mean.reshape(1, 3, 1, 1)
    patches /= std.reshape(1, 3, 1, 1)
    return patches, grid_hw, resize_info["num_tokens"]


class YOCOVLMultiModalProjector(nn.Module):
    def __init__(
        self,
        config: YOCOVLConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel()
        vision_config = config.vision_config

        self.hidden_size = (
            vision_config.hidden_size
            * vision_config.merge_kernel_size[0]
            * vision_config.merge_kernel_size[1]
        )

        self.pre_norm = nn.LayerNorm(
            vision_config.hidden_size,
            eps=config.text_config.norm_eps,
        )
        self.proj = nn.ModuleList(
            [
                ReplicatedLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    prefix=maybe_prefix(prefix, "proj.0"),
                ),
                GELUActivation(),
                ReplicatedLinear(
                    self.hidden_size,
                    config.text_config.hidden_size,
                    bias=True,
                    prefix=maybe_prefix(prefix, "proj.2"),
                ),
            ]
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.to(self.pre_norm.weight.dtype)
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states, _ = self.proj[0](hidden_states)
        hidden_states = self.proj[1](hidden_states)
        hidden_states, _ = self.proj[2](hidden_states)
        return hidden_states


class YOCOVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    image_grid_hws: Annotated[torch.Tensor, TensorShape("ni", 2)]


YOCOVLImageInputs = YOCOVLImagePixelInputs


class YOCOVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(YOCOVLConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        config = self.get_hf_config()
        vision_config = config.vision_config
        resize_info = _navit_resize_image(
            image_width,
            image_height,
            patch_size=vision_config.patch_size,
            merge_kernel_size=vision_config.merge_kernel_size[0],
            in_patch_limit=16384,
            patch_limit_on_one_side=config.vision_patch_limit_on_one_side,
            max_image_tokens=config.vision_max_image_tokens,
            align_mode=config.vision_align_mode,
        )
        return resize_info["num_tokens"]

    @property
    def image_placeholder_token_id(self) -> int:
        return self.get_hf_config().image_placeholder_token_id

    @property
    def image_start_token_id(self) -> int:
        return self.get_hf_config().image_start_token_id

    @property
    def image_end_token_id(self) -> int:
        return self.get_hf_config().image_end_token_id

    @property
    def image_placeholder(self) -> str:
        return self.get_hf_config().image_placeholder


class YOCOVLDummyInputsBuilder(BaseDummyInputsBuilder[YOCOVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.image_placeholder * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")
        return {
            "image": self._get_dummy_images(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class YOCOVLMultiModalProcessor(BaseMultiModalProcessor[YOCOVLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get("image_grid_hws", torch.empty((0, 2)))
        image_grid_sizes = image_grid_hws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_grid_hws=MultiModalFieldConfig.batched("image"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        images = mm_data.get("images")
        add_special_tokens = bool(tok_kwargs.get("add_special_tokens", True))

        if not images:
            input_ids = tokenizer.encode(
                prompt,
                add_special_tokens=add_special_tokens,
            )
            return BatchFeature({"input_ids": [input_ids]})

        config = self.info.get_hf_config()
        pixel_values = []
        image_grid_hws = []
        num_image_tokens = []
        for image in images:
            patches, grid_hw, num_tokens = _process_image(image, config)
            pixel_values.append(torch.from_numpy(patches))
            image_grid_hws.append(torch.from_numpy(grid_hw))
            num_image_tokens.append(num_tokens)

        parts = prompt.split(self.info.image_placeholder)
        if len(parts) - 1 != len(num_image_tokens):
            raise ValueError(
                "The prompt must contain one <image> placeholder per image; "
                f"got {len(parts) - 1} placeholders and "
                f"{len(num_image_tokens)} images"
            )

        input_ids: list[int] = []
        bos_token = getattr(tokenizer, "bos_token", None)
        if add_special_tokens and not (
            isinstance(bos_token, str) and prompt.lstrip().startswith(bos_token)
        ):
            input_ids.append(config.bos_token_id)

        for index, part in enumerate(parts):
            input_ids.extend(tokenizer.encode(part, add_special_tokens=False))
            if index < len(num_image_tokens):
                input_ids.append(config.image_start_token_id)
                input_ids.extend(
                    [config.image_placeholder_token_id] * num_image_tokens[index]
                )
                input_ids.append(config.image_end_token_id)

        output: dict[str, object] = {"input_ids": [input_ids]}
        output["pixel_values"] = torch.cat(pixel_values, dim=0)
        output["image_grid_hws"] = torch.stack(image_grid_hws, dim=0)
        return BatchFeature(output)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        if mm_items.get_count("image", strict=False) == 0:
            return False
        images = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
        return isinstance(images, ImageProcessorItems)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_start_token_id = self.info.image_start_token_id
        image_end_token_id = self.info.image_end_token_id
        image_placeholder_token_id = self.info.image_placeholder_token_id
        tokenizer = self.info.get_tokenizer()

        def get_num_image_tokens(item_idx: int) -> int:
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                return images.get_feature_size(item_idx)

            image_size = images.get_image_size(item_idx)
            return self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )

        def get_replacement(
            item_idx: int,
            suffix_token_ids: list[int] | None = None,
        ) -> PromptUpdateDetails[list[int]]:
            num_image_tokens = get_num_image_tokens(item_idx)
            token_ids = (
                [image_start_token_id]
                + [image_placeholder_token_id] * num_image_tokens
                + [image_end_token_id]
            )
            if suffix_token_ids:
                token_ids.extend(suffix_token_ids)
            return PromptUpdateDetails.select_token_id(
                token_ids, image_placeholder_token_id
            )

        def get_expanded_target(item_idx: int) -> list[int]:
            return get_replacement(item_idx).full

        updates: list[PromptUpdate] = [
            PromptReplacement(
                modality="image",
                target=get_expanded_target,
                replacement=get_replacement,
            ),
        ]

        seen_targets: set[tuple[int, ...]] = set()
        for prefix in ("", " "):
            for suffix in ("", "\n", " "):
                target = tokenizer.encode(
                    prefix + self.info.image_placeholder + suffix,
                    add_special_tokens=False,
                )
                target_key = tuple(target)
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)

                prefix_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
                suffix_token_ids = tokenizer.encode(suffix, add_special_tokens=False)

                def replacement_with_affixes(
                    item_idx: int,
                    prefix_token_ids: list[int] = prefix_token_ids,
                    suffix_token_ids: list[int] = suffix_token_ids,
                ) -> PromptUpdateDetails[list[int]]:
                    replacement = get_replacement(item_idx)
                    return replace(
                        replacement,
                        full=prefix_token_ids + replacement.full + suffix_token_ids,
                    )

                updates.append(
                    PromptReplacement(
                        modality="image",
                        target=target,
                        replacement=replacement_with_affixes,
                    )
                )

        return updates


@MULTIMODAL_REGISTRY.register_processor(
    YOCOVLMultiModalProcessor,
    info=YOCOVLProcessingInfo,
    dummy_inputs=YOCOVLDummyInputsBuilder,
)
class YOCOVLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: YOCOVLConfig = model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        assert isinstance(config.vision_config, MoonViTConfig)
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonVitPretrainedModel(
                config.vision_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.vision_projector = YOCOVLMultiModalProjector(
                config=config,
                prefix=maybe_prefix(prefix, "vision_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["YOCOForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> YOCOVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_hws = kwargs.pop("image_grid_hws", None)

        if pixel_values is None:
            return None

        return YOCOVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )

    @torch.inference_mode()
    def _process_image_pixels(self, inputs: YOCOVLImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        image_grid_hws = inputs["image_grid_hws"]
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                image_grid_hws.tolist(),
                rope_type="rope_2d",
            )
        return self.vision_tower(pixel_values, image_grid_hws)

    def _process_image_input(self, image_input: YOCOVLImageInputs) -> NestedTensors:
        assert image_input["type"] == "pixel_values"
        image_features = self._process_image_pixels(image_input)
        assert isinstance(image_features, (list, tuple))
        lengths = [x.shape[0] for x in image_features]
        return self.vision_projector(torch.cat(image_features)).split(lengths)

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
