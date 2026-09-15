# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO operators; register shared numerical dependencies before use."""

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON and current_platform.is_cuda():
    from vllm.model_executor.layers import (
        yoco_attention_fp8 as yoco_attention_fp8,
    )
    from vllm.model_executor.layers import (
        yoco_bf16_math as yoco_bf16_math,
    )
    from vllm.model_executor.layers import (
        yoco_norm_fp8 as yoco_norm_fp8,
    )
