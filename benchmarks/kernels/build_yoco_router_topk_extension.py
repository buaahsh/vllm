# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the minimal CUDA extension used by benchmark_yoco_router_topk.py."""

from pathlib import Path

from torch.utils.cpp_extension import load

benchmark_dir = Path(__file__).resolve().parent
repo_root = benchmark_dir.parents[1]
build_dir = benchmark_dir / "build"
build_dir.mkdir(exist_ok=True)
library = load(
    name="yoco_topk_bench_ext",
    sources=[
        str(benchmark_dir / "yoco_topk_softmax_bindings.cpp"),
        str(repo_root / "csrc/moe/topk_softmax_kernels.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=str(build_dir),
    is_python_module=False,
    verbose=True,
)
print(library)
