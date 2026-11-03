# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor contract for the YOCO single-depth, dense MTP head."""


def expected_mtp_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    width = config["hidden_size"]
    heads = config["cross_head"]
    head_dim = config["head_dim"]
    intermediate = config["d_ffn"]
    gate_heads = heads * (2 if config.get("diff_v3", False) else 1)
    return {
        "mtp.embedding_norm.weight": (width,),
        "mtp.hidden_norm.weight": (width,),
        "mtp.proj.weight": (width, width),
        "mtp.input_norm.weight": (width,),
        "mtp.self_attn.q_proj.weight": (2 * heads * head_dim, width),
        "mtp.self_attn.q_norm.weight": (head_dim,),
        "mtp.self_attn.o_proj.weight": (width, heads * head_dim),
        "mtp.self_attn.lambda_proj.weight": (gate_heads, width),
        "mtp.post_attn_norm.weight": (width,),
        "mtp.ffn.gate_up_proj.weight": (2 * intermediate, width),
        "mtp.ffn.down_proj.weight": (width, intermediate),
        "norm.weight": (width,),
    }
