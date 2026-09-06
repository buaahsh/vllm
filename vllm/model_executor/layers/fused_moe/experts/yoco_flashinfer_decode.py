# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment-checked CUTLASS decode tactics for YOCO L3 BF16 on B200."""

import functools
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)
YOCO_CUTLASS_DECODE_ROWS = (64, 128, 256)


@functools.cache
def load_yoco_decode_cutlass_cache() -> bool:
    """Load only measured tactics; FlashInfer validates the dependency metadata."""
    path = Path(__file__).with_name("yoco_configs") / "cutlass_decode_b200.json"
    if not path.is_file():
        return False
    try:
        from flashinfer.autotuner import AutoTuner

        loader = getattr(AutoTuner.get(), "load_configs", None)
        if loader is None or not loader(str(path)):
            return False
    except (ImportError, OSError, ValueError) as exc:
        logger.debug("YOCO decode tactic cache is unavailable: %s", exc)
        return False
    logger.info_once("Loaded environment-matched YOCO CUTLASS decode tactics")
    return True
