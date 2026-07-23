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
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel
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
    VideoEmbeddingItems,
    VideoProcessorItems,
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
    num_frames, height, width, channels = pixel_values.shape
    assert channels == 3
    patches = pixel_values.reshape(
        num_frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, channels, patch_size, patch_size)
    grid_thw = np.array(
        [num_frames, height // patch_size, width // patch_size], dtype=np.int64
    )
    return patches, grid_thw


def _resize_media_frames(
    frames: Sequence[Image.Image],
    config: YOCOVLConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    if not frames:
        raise ValueError("A video must contain at least one frame")

    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("All video frames must have the same width and height")

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

    arrays = []
    for frame in frames:
        frame = frame.resize(
            (resize_info["new_width"], resize_info["new_height"]),
            resample=Image.Resampling.BICUBIC,
        )
        array = np.asarray(frame)
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
        arrays.append(array)

    return np.stack(arrays, axis=0), resize_info


def _normalize_patches(patches: np.ndarray, config: YOCOVLConfig) -> np.ndarray:
    patches = patches.astype(np.float32, copy=False)
    patches /= np.float32(255.0)

    mean = np.asarray(config.vision_image_mean, dtype=np.float32)
    std = np.asarray(config.vision_image_std, dtype=np.float32)
    patches -= mean.reshape(1, 3, 1, 1)
    patches /= std.reshape(1, 3, 1, 1)
    return patches


def _process_image(
    image: object,
    config: YOCOVLConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    image = _to_pil_image(image).convert("RGB")
    arrays, resize_info = _resize_media_frames([image], config)
    patches, grid_thw = _patchify(arrays, config.vision_config.patch_size)
    patches = _normalize_patches(patches, config)
    return patches, grid_thw[1:], resize_info["num_tokens"]


def _process_video(
    video: object,
    config: YOCOVLConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()

    if isinstance(video, np.ndarray):
        if video.ndim != 4:
            raise ValueError(f"Expected a 4D video array, got shape {video.shape}")
        raw_frames: Sequence[object] = list(video)
    elif isinstance(video, Sequence) and not isinstance(video, (str, bytes)):
        raw_frames = video
    else:
        raise TypeError(f"Unsupported video type: {type(video)!r}")

    frames = [_to_pil_image(frame).convert("RGB") for frame in raw_frames]
    max_frames = config.vision_config.init_pos_emb_time
    if len(frames) > max_frames:
        raise ValueError(
            f"YOCO-VL supports at most {max_frames} frames per video chunk, "
            f"but received {len(frames)}. Sample or chunk the video first."
        )

    arrays, resize_info = _resize_media_frames(frames, config)
    patches, grid_thw = _patchify(arrays, config.vision_config.patch_size)
    patches = _normalize_patches(patches, config)
    return patches, grid_thw, resize_info["num_tokens"]


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
                nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                ),
                GELUActivation(),
                nn.Linear(
                    self.hidden_size,
                    config.text_config.hidden_size,
                    bias=True,
                ),
            ]
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.to(self.pre_norm.weight.dtype)
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.proj[0](hidden_states)
        hidden_states = self.proj[1](hidden_states)
        hidden_states = self.proj[2](hidden_states)
        return hidden_states


class YOCOVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    image_grid_thws: Annotated[torch.Tensor, TensorShape("ni", 3)]


YOCOVLImageInputs = YOCOVLImagePixelInputs


class YOCOVLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"] = "pixel_values_videos"

    pixel_values_videos: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    video_grid_thws: Annotated[torch.Tensor, TensorShape("nv", 3)]


YOCOVLVideoInputs = YOCOVLVideoPixelInputs


class YOCOVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(YOCOVLConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

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

    def get_num_video_tokens(
        self,
        *,
        video_width: int,
        video_height: int,
        num_frames: int,
    ) -> int:
        max_frames = self.get_hf_config().vision_config.init_pos_emb_time
        if not 1 <= num_frames <= max_frames:
            raise ValueError(
                f"YOCO-VL video chunks require 1..{max_frames} frames, "
                f"got {num_frames}"
            )
        return self.get_num_image_tokens(
            image_width=video_width,
            image_height=video_height,
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # Both images and temporally pooled video chunks produce the same
        # number of output tokens for a given H/W grid. Reserve two positions
        # for begin/end markers when the configured maximum exceeds seq_len.
        max_tokens = min(
            self.get_hf_config().vision_max_image_tokens,
            max(0, seq_len - 2),
        )
        return {modality: max_tokens for modality in ("image", "video")}

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

    @property
    def video_placeholder(self) -> str:
        return self.get_hf_config().video_placeholder

    @property
    def max_video_frames(self) -> int:
        return self.get_hf_config().vision_config.init_pos_emb_time


class YOCOVLDummyInputsBuilder(BaseDummyInputsBuilder[YOCOVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        placeholders = [self.info.image_placeholder] * mm_counts.get("image", 0)
        placeholders.extend(
            [self.info.video_placeholder] * mm_counts.get("video", 0)
        )
        return " ".join(placeholders)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")
        return {
            "image": self._get_dummy_images(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_frames=self.info.max_video_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }


class YOCOVLMultiModalProcessor(BaseMultiModalProcessor[YOCOVLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thws = hf_inputs.get("image_grid_thws", torch.empty((0, 3)))
        video_grid_thws = hf_inputs.get("video_grid_thws", torch.empty((0, 3)))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thws.prod(-1)
            ),
            image_grid_thws=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_thws.prod(-1)
            ),
            video_grid_thws=MultiModalFieldConfig.batched("video"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        images = mm_data.get("images", [])
        videos = mm_data.get("videos", [])
        add_special_tokens = bool(tok_kwargs.get("add_special_tokens", True))

        if not images and not videos:
            input_ids = tokenizer.encode(
                prompt,
                add_special_tokens=add_special_tokens,
            )
            return BatchFeature({"input_ids": [input_ids]})

        config = self.info.get_hf_config()
        image_pixel_values = []
        image_grid_thws = []
        num_image_tokens = []
        for image in images:
            patches, grid_hw, num_tokens = _process_image(image, config)
            image_pixel_values.append(torch.from_numpy(patches))
            image_grid_thws.append(
                torch.from_numpy(np.concatenate(([1], grid_hw)))
            )
            num_image_tokens.append(num_tokens)

        video_pixel_values = []
        video_grid_thws = []
        num_video_tokens = []
        for video in videos:
            patches, grid_thw, num_tokens = _process_video(video, config)
            video_pixel_values.append(torch.from_numpy(patches))
            video_grid_thws.append(torch.from_numpy(grid_thw))
            num_video_tokens.append(num_tokens)

        num_image_placeholders = prompt.count(self.info.image_placeholder)
        num_video_placeholders = prompt.count(self.info.video_placeholder)
        if num_image_placeholders != len(num_image_tokens):
            raise ValueError(
                "The prompt must contain one <image> placeholder per image; "
                f"got {num_image_placeholders} placeholders and "
                f"{len(num_image_tokens)} images"
            )
        if num_video_placeholders != len(num_video_tokens):
            raise ValueError(
                "The prompt must contain one <video> placeholder per video; "
                f"got {num_video_placeholders} placeholders and "
                f"{len(num_video_tokens)} videos"
            )

        # Tokenize the original placeholders and let vLLM apply the modality-
        # aware prompt replacements. Image and video intentionally use the
        # same begin/pad/end token IDs, so applying replacements here would
        # make same-sized image/video blocks indistinguishable to the registry.
        input_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        output: dict[str, object] = {"input_ids": [input_ids]}
        if image_pixel_values:
            output["pixel_values"] = torch.cat(image_pixel_values, dim=0)
            output["image_grid_thws"] = torch.stack(image_grid_thws, dim=0)
        if video_pixel_values:
            output["pixel_values_videos"] = torch.cat(video_pixel_values, dim=0)
            output["video_grid_thws"] = torch.stack(video_grid_thws, dim=0)
        return BatchFeature(output)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        start_token_id = self.info.image_start_token_id
        end_token_id = self.info.image_end_token_id
        placeholder_token_id = self.info.image_placeholder_token_id

        def get_num_tokens(modality: str, item_idx: int) -> int:
            if modality == "image":
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

            videos = mm_items.get_items(
                "video", (VideoEmbeddingItems, VideoProcessorItems)
            )
            if isinstance(videos, VideoEmbeddingItems):
                return videos.get_feature_size(item_idx)
            video = videos.get(item_idx)
            if video is None or len(video) == 0:
                raise ValueError(f"Cannot get size of video at {item_idx}")
            # VideoProcessorItems assumes tensor/array frames are CHW when
            # reporting their size. YOCO-VL accepts both TCHW and THWC, so use
            # the same layout-aware conversion as the actual preprocessing.
            frame_size = _to_pil_image(video[0]).size
            return self.info.get_num_video_tokens(
                video_width=frame_size[0],
                video_height=frame_size[1],
                num_frames=len(video),
            )

        def build_updates(modality: str, placeholder: str) -> list[PromptUpdate]:
            def get_replacement(
                item_idx: int,
                suffix_token_ids: list[int] | None = None,
            ) -> PromptUpdateDetails[list[int]]:
                token_ids = (
                    [start_token_id]
                    + [placeholder_token_id] * get_num_tokens(modality, item_idx)
                    + [end_token_id]
                )
                if suffix_token_ids:
                    token_ids.extend(suffix_token_ids)
                return PromptUpdateDetails.select_token_id(
                    token_ids, placeholder_token_id
                )

            def get_expanded_target(item_idx: int) -> list[int]:
                return get_replacement(item_idx).full

            modality_updates: list[PromptUpdate] = [
                PromptReplacement(
                    modality=modality,
                    target=get_expanded_target,
                    replacement=get_replacement,
                )
            ]

            seen_targets: set[tuple[int, ...]] = set()
            for prefix in ("", " "):
                for suffix in ("", "\n", " "):
                    target = tokenizer.encode(
                        prefix + placeholder + suffix,
                        add_special_tokens=False,
                    )
                    target_key = tuple(target)
                    if target_key in seen_targets:
                        continue
                    seen_targets.add(target_key)

                    prefix_token_ids = tokenizer.encode(
                        prefix, add_special_tokens=False
                    )
                    suffix_token_ids = tokenizer.encode(
                        suffix, add_special_tokens=False
                    )

                    def replacement_with_affixes(
                        item_idx: int,
                        prefix_token_ids: list[int] = prefix_token_ids,
                        suffix_token_ids: list[int] = suffix_token_ids,
                    ) -> PromptUpdateDetails[list[int]]:
                        replacement = get_replacement(item_idx)
                        return replace(
                            replacement,
                            full=(
                                prefix_token_ids
                                + replacement.full
                                + suffix_token_ids
                            ),
                        )

                    modality_updates.append(
                        PromptReplacement(
                            modality=modality,
                            target=target,
                            replacement=replacement_with_affixes,
                        )
                    )
            return modality_updates

        updates: list[PromptUpdate] = []
        if mm_items.get_count("image", strict=False):
            updates.extend(build_updates("image", self.info.image_placeholder))
        if mm_items.get_count("video", strict=False):
            updates.extend(build_updates("video", self.info.video_placeholder))
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
        if modality.startswith("video"):
            return "<video>"
        raise ValueError("Only image and video modalities are supported")

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

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.vision_projector = YOCOVLMultiModalProjector(
                config=config,
                prefix=maybe_prefix(prefix, "vision_projector"),
            ).to(dtype=torch.float32)

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
        image_grid_thws = kwargs.pop("image_grid_thws", None)

        if pixel_values is None:
            return None

        return YOCOVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thws=image_grid_thws,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> YOCOVLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thws = kwargs.pop("video_grid_thws", None)

        if pixel_values_videos is None:
            return None

        return YOCOVLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thws=video_grid_thws,
        )

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, YOCOVLImageInputs | YOCOVLVideoInputs]:
        inputs: dict[str, YOCOVLImageInputs | YOCOVLVideoInputs] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in inputs:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    inputs["image"] = image_input
            elif input_key == "pixel_values_videos" and "video" not in inputs:
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    inputs["video"] = video_input
        return inputs

    def _process_media_pixels(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        grid_thws: torch.Tensor,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thws with shape [N, 3], got {grid_thws.shape}"
            )
        expected_patches = int(
            grid_thws.to(torch.int64).prod(dim=-1).sum().item()
        )
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                f"Pixel/grid mismatch: got {pixel_values.shape[0]} patches but "
                f"grid_thws describes {expected_patches}"
            )

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thws.tolist(),
                rope_type="rope_2d",
            )
        return self.vision_tower(pixel_values, grid_thws)

    @torch.inference_mode()
    def _process_image_pixels(
        self, inputs: YOCOVLImagePixelInputs
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        return self._process_media_pixels(
            inputs["pixel_values"], inputs["image_grid_thws"]
        )

    @torch.inference_mode()
    def _process_video_pixels(
        self, inputs: YOCOVLVideoPixelInputs
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        return self._process_media_pixels(
            inputs["pixel_values_videos"], inputs["video_grid_thws"]
        )

    def _project_media_features(
        self, media_features: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.vision_projector(features) for features in media_features)

    def _process_image_input(self, image_input: YOCOVLImageInputs) -> NestedTensors:
        assert image_input["type"] == "pixel_values"
        image_features = self._process_image_pixels(image_input)
        # Keep the projector GEMM M-shapes identical to llm-train, which runs
        # PatchMergerMLP independently for each image in a multimodal batch.
        return self._project_media_features(image_features)

    def _process_video_input(self, video_input: YOCOVLVideoInputs) -> NestedTensors:
        assert video_input["type"] == "pixel_values_videos"
        return self._project_media_features(self._process_video_pixels(video_input))

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        mm_inputs = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_inputs:
            return None

        embeddings: tuple[torch.Tensor, ...] = ()
        for modality, media_input in mm_inputs.items():
            if modality == "image":
                embeddings += tuple(self._process_image_input(media_input))
            else:
                embeddings += tuple(self._process_video_input(media_input))
        return embeddings

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
