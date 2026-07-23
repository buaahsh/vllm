# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
)
from vllm.model_executor.models.yoco_vl import _process_image, _process_video
from vllm.transformers_utils.configs.yoco_vl import YOCOVLConfig


@pytest.fixture
def yoco_vl_config() -> YOCOVLConfig:
    return YOCOVLConfig(
        vision_config={
            "patch_size": 14,
            "merge_kernel_size": [2, 2],
            "init_pos_emb_time": 4,
        },
        vision_max_image_tokens=16,
    )


def test_process_video_matches_per_frame_image_patches(
    yoco_vl_config: YOCOVLConfig,
) -> None:
    first = np.zeros((28, 28, 3), dtype=np.uint8)
    second = np.full((28, 28, 3), 255, dtype=np.uint8)

    first_patches, first_grid, image_tokens = _process_image(
        Image.fromarray(first), yoco_vl_config
    )
    second_patches, second_grid, _ = _process_image(
        Image.fromarray(second), yoco_vl_config
    )
    video_patches, video_grid, video_tokens = _process_video(
        np.stack([first, second]), yoco_vl_config
    )

    np.testing.assert_array_equal(video_grid, [2, *first_grid])
    np.testing.assert_array_equal(first_grid, second_grid)
    np.testing.assert_array_equal(
        video_patches, np.concatenate([first_patches, second_patches])
    )
    assert video_tokens == image_tokens == 1


def test_process_video_accepts_tchw_and_thwc(
    yoco_vl_config: YOCOVLConfig,
) -> None:
    video_thwc = np.arange(2 * 28 * 28 * 3, dtype=np.uint8).reshape(2, 28, 28, 3)
    video_tchw = np.moveaxis(video_thwc, -1, 1)

    thwc_result = _process_video(video_thwc, yoco_vl_config)
    tchw_result = _process_video(video_tchw, yoco_vl_config)

    for thwc_value, tchw_value in zip(thwc_result, tchw_result, strict=True):
        np.testing.assert_array_equal(thwc_value, tchw_value)


def test_process_video_enforces_temporal_embedding_limit(
    yoco_vl_config: YOCOVLConfig,
) -> None:
    video = np.zeros((5, 28, 28, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="at most 4 frames"):
        _process_video(video, yoco_vl_config)


def test_temporal_position_buffer_follows_loaded_weight_dtype() -> None:
    pos_emb = Learnable2DInterpPosEmbDivided_fixed(
        height=2,
        width=2,
        num_frames=4,
        dim=8,
    )
    pos_emb.weight.data = pos_emb.weight.data.to(torch.bfloat16)
    assert pos_emb.time_weight.dtype == torch.float32

    output = pos_emb(
        torch.zeros((8, 8), dtype=torch.bfloat16),
        torch.tensor([[2, 2, 2]]),
    )

    assert output.dtype == torch.bfloat16
