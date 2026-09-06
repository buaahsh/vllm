# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

if TYPE_CHECKING:
    from vllm.v1.worker.block_table import BlockTable


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _compute_yoco_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mapping_ptrs,
    total_cp_rank,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    slot_mapping_ptr = _load_ptr(slot_mapping_ptrs + group_idx, tl.int64)

    if req_idx == tl.num_programs(1) - 1:
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_idx, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_idx)
    block_size = tl.load(block_sizes + group_idx)
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == total_cp_rank
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )
        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_yoco_decode_slot_mappings_kernel(
    num_reqs,
    max_num_tokens,
    positions_ptr,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mapping_ptrs,
    total_cp_rank,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PAD_BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    phase = tl.program_id(1)
    slot_mapping_ptr = _load_ptr(slot_mapping_ptrs + group_idx, tl.int64)

    if phase == 1:
        for i in range(num_reqs, max_num_tokens, PAD_BLOCK_SIZE):
            offsets = i + tl.arange(0, PAD_BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    rows = tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_reqs
    block_table_ptr = _load_ptr(block_table_ptrs + group_idx, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_idx)
    block_size = tl.load(block_sizes + group_idx)
    positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    block_indices = positions // virtual_block_size
    block_numbers = tl.load(
        block_table_ptr + rows * block_table_stride + block_indices,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    virtual_block_offsets = positions - block_indices * virtual_block_size
    is_local = (
        virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % TOTAL_CP_WORLD_SIZE == total_cp_rank
    local_block_offsets = (
        virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
    )
    slot_ids = block_numbers * block_size + local_block_offsets
    slot_ids = tl.where(is_local, slot_ids, PAD_ID)
    tl.store(slot_mapping_ptr + rows, slot_ids, mask=row_mask)


class YOCOMultiGroupSlotMapping:
    """Fuse YOCO's 31 KV-group slot mappings into one GPU launch."""

    def __init__(self, block_tables: list["BlockTable"]) -> None:
        if len(block_tables) <= 1:
            raise ValueError("YOCO fused slot mapping requires multiple groups")
        first = block_tables[0]
        if first.device.type != "cuda":
            raise ValueError("YOCO fused slot mapping requires CUDA")

        for block_table in block_tables[1:]:
            assert block_table.device == first.device
            assert block_table.max_num_batched_tokens == first.max_num_batched_tokens
            assert block_table.pcp_world_size == first.pcp_world_size
            assert block_table.pcp_rank == first.pcp_rank
            assert block_table.dcp_world_size == first.dcp_world_size
            assert block_table.dcp_rank == first.dcp_rank
            assert (
                block_table.cp_kv_cache_interleave_size
                == first.cp_kv_cache_interleave_size
            )

        self.num_groups = len(block_tables)
        self.max_num_batched_tokens = first.max_num_batched_tokens
        self.total_cp_world_size = first.pcp_world_size * first.dcp_world_size
        self.total_cp_rank = first.pcp_rank * first.dcp_world_size + first.dcp_rank
        self.cp_kv_cache_interleave_size = first.cp_kv_cache_interleave_size
        self.block_table_ptrs = torch.tensor(
            [table.block_table.gpu.data_ptr() for table in block_tables],
            dtype=torch.uint64,
            device=first.device,
        )
        self.block_table_strides = torch.tensor(
            [table.block_table.gpu.stride(0) for table in block_tables],
            dtype=torch.int64,
            device=first.device,
        )
        self.block_sizes = torch.tensor(
            [table.block_size for table in block_tables],
            dtype=torch.int32,
            device=first.device,
        )
        self.slot_mapping_ptrs = torch.tensor(
            [table.slot_mapping.gpu.data_ptr() for table in block_tables],
            dtype=torch.uint64,
            device=first.device,
        )

    def compute(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # One token per request is the CUDA-Graph decode case. Keep very large
        # batches on the general kernel instead of creating an oversized CTA.
        if 0 < num_reqs <= 1024 and positions.shape[0] == num_reqs:
            _compute_yoco_decode_slot_mappings_kernel[(self.num_groups, 2)](
                num_reqs,
                self.max_num_batched_tokens,
                positions,
                self.block_table_ptrs,
                self.block_table_strides,
                self.block_sizes,
                self.slot_mapping_ptrs,
                self.total_cp_rank,
                TOTAL_CP_WORLD_SIZE=self.total_cp_world_size,
                CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
                PAD_ID=PAD_SLOT_ID,
                BLOCK_ROWS=triton.next_power_of_2(num_reqs),
                PAD_BLOCK_SIZE=1024,
                num_warps=4,
            )
            return
        _compute_yoco_slot_mappings_kernel[(self.num_groups, num_reqs + 1)](
            positions.shape[0],
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes,
            self.slot_mapping_ptrs,
            self.total_cp_rank,
            TOTAL_CP_WORLD_SIZE=self.total_cp_world_size,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
