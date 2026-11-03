# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pixel, placeholder, and cache contracts for YOCO with the Kimi-K3 tower."""

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
)
from vllm.model_executor.models.yoco_vl import _process_image, _process_video
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.transformers_utils.configs.yoco_vl import YOCOVLConfig


@pytest.fixture
def config():
    return YOCOVLConfig(
        text_config=dict(
            hidden_size=32,
            d_model=32,
            num_hidden_layers=2,
            n_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            max_position_embeddings=256,
            universal_loop=1,
            yoco_cross_layers=1,
            moe=False,
        ),
        vision_max_image_tokens=16,
        image_placeholder_token_id=7,
        image_start_token_id=8,
        image_end_token_id=9,
        pad_token_id=10,
    )


@pytest.fixture
def processor(tmp_path, config):
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "<image>": 1,
                "<video>": 2,
                "hello": 3,
                "what": 4,
                "first": 5,
                "second": 6,
                "[IMG]": 7,
                "[BEGIN]": 8,
                "[END]": 9,
                "[PAD]": 10,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<image>", "<video>", "[IMG]", "[BEGIN]", "[END]"],
    )
    hf_tokenizer.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        max_model_len=256,
        limit_mm_per_prompt={"image": 2, "video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(
        model,
        cache=MultiModalProcessorOnlyCache(model),
    )


@pytest.mark.parametrize(
    "size", [(1, 1), (31, 49), (1501, 33), (33, 1501), (1024, 1024)]
)
@pytest.mark.parametrize("align", ["resize", "pad"])
def test_aligned_pixels_never_exceed_image_budget(config, size, align):
    config.vision_align_mode = align
    pixels, grid, count = _process_image(Image.new("RGB", size, "white"), config)
    assert 0 < count <= 16
    assert pixels.shape == (int(grid.prod()), 3, 14, 14)
    assert count == int(grid.prod()) // 4
    assert np.all(grid % 2 == 0)
    assert pixels.dtype == np.float32
    assert pixels.min() >= -1 and pixels.max() <= 1


def test_normalization_and_patch_order_match_rgb_pixels(config):
    image = np.zeros((28, 56, 3), dtype=np.uint8)
    image[:, :28, 0] = 255
    image[:, 28:, 2] = 255
    patches, grid, count = _process_image(Image.fromarray(image), config)
    np.testing.assert_array_equal(grid, [2, 4])
    assert count == 2
    np.testing.assert_array_equal(patches[[0, 1, 4, 5], 0], 1)
    np.testing.assert_array_equal(patches[[2, 3, 6, 7], 2], 1)
    np.testing.assert_array_equal(patches[:, 1], -1)


def test_video_matches_per_frame_images_and_layouts(config):
    frames = np.stack(
        [np.zeros((28, 28, 3), np.uint8), np.full((28, 28, 3), 255, np.uint8)]
    )
    result = _process_video(frames, config)
    chw_result = _process_video(np.moveaxis(frames, -1, 1), config)
    for actual, expected in zip(result, chw_result, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        result[0],
        np.concatenate(
            [_process_image(Image.fromarray(frame), config)[0] for frame in frames]
        ),
    )
    np.testing.assert_array_equal(result[1], [2, 2, 2])
    assert result[2] == 1
    with pytest.raises(ValueError, match="at most 4 frames"):
        _process_video(np.zeros((5, 28, 28, 3), np.uint8), config)


def test_temporal_positions_do_not_promote_bf16_activations():
    position = Learnable2DInterpPosEmbDivided_fixed(2, 2, 4, 8)
    position.weight.data = position.weight.data.bfloat16()
    assert position.time_weight.dtype == torch.float32
    result = position(torch.zeros(8, 8, dtype=torch.bfloat16), [[2, 2, 2]])
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("video_first", [False, True])
def test_same_sized_image_and_video_keep_prompt_order_with_cache(
    processor, video_first
):
    prompt = "<video> hello <image>" if video_first else "<image> hello <video>"
    media = {
        "image": [Image.new("RGB", (28, 28), "red")],
        "video": [np.zeros((2, 28, 28, 3), np.uint8)],
    }
    outputs = [
        processor(prompt, mm_items=processor.info.parse_mm_data(media))
        for _ in range(2)
    ]
    for output in outputs:
        locations = output["mm_placeholders"]
        assert (
            locations["video"][0].offset < locations["image"][0].offset
        ) == video_first
        assert locations["video"][0].length == locations["image"][0].length == 3
        assert output["prompt_token_ids"].count(7) == 2
    assert outputs[0]["prompt_token_ids"] == outputs[1]["prompt_token_ids"]
    assert outputs[0]["mm_hashes"] == outputs[1]["mm_hashes"]


def test_image_tokens_match_feature_counts_and_text_only_is_unchanged(processor):
    media = {"image": [Image.new("RGB", (28, 28)), Image.new("RGB", (56, 28))]}
    output = processor(
        "<image> first <image> second", mm_items=processor.info.parse_mm_data(media)
    )
    assert output["prompt_token_ids"].count(7) == 3
    assert [p.length for p in output["mm_placeholders"]["image"]] == [3, 4]
    plain = processor("hello", mm_items=processor.info.parse_mm_data({}))
    assert plain["prompt_token_ids"] == processor.info.get_tokenizer().encode("hello")
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        processor("<image>", mm_items=processor.info.parse_mm_data(media))


def test_config_round_trip_preserves_k3_projector_width(tmp_path, config):
    config.save_pretrained(tmp_path)
    reloaded = YOCOVLConfig.from_pretrained(tmp_path)
    assert reloaded.vision_config.text_hidden_size == 32
    assert reloaded.vision_config.hidden_size == 1024
    assert reloaded.vision_config.qkv_hidden_size == 1536
    assert reloaded.vision_config.num_attention_heads == 12
    assert reloaded.vision_config.norm_type == "rmsnorm"
    assert reloaded.get_text_config().model_type == "yoco"


@pytest.mark.parametrize("prompt", ["<image><image>", "(<image>),<image>!"])
def test_bpe_marker_boundaries_preserve_all_embedding_tokens(tmp_path, config, prompt):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["<image><image> (<image>),<image>! !!!!"] * 10,
        trainers.BpeTrainer(
            vocab_size=100,
            special_tokens=["[UNK]", "[BEGIN]", "[END]", "[PAD]"],
        ),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    config.image_placeholder_token_id = hf_tokenizer.convert_tokens_to_ids("!")
    config.image_start_token_id = hf_tokenizer.convert_tokens_to_ids("[BEGIN]")
    config.image_end_token_id = hf_tokenizer.convert_tokens_to_ids("[END]")
    config.pad_token_id = hf_tokenizer.pad_token_id
    config.text_config.vocab_size = len(hf_tokenizer)
    assert hf_tokenizer.encode("!!!!") != [config.image_placeholder_token_id] * 4
    hf_tokenizer.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    model = ModelConfig(model=str(tmp_path), max_model_len=256)
    processor = MULTIMODAL_REGISTRY.create_processor(
        model, cache=MultiModalProcessorOnlyCache(model)
    )
    media = {"image": [Image.new("RGB", (56, 28)), Image.new("RGB", (56, 28))]}
    for _ in range(2):
        output = processor(prompt, mm_items=processor.info.parse_mm_data(media))
        locations = output["mm_placeholders"]["image"]
        assert len(locations) == 2
        assert all(int(location.is_embed.sum()) == 2 for location in locations)
        assert locations[0].offset < locations[1].offset
