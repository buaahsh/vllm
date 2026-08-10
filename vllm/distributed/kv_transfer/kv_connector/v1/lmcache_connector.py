# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

YOCO_LMCACHE_LAYOUT_VERSION = "yoco-physical-kv-v1"


@dataclass(frozen=True)
class _YocoLMCachePhysicalKVLayout:
    """Stable LMCache view of YOCO's shared cross-attention KV cache."""

    logical_num_layers: int
    first_cross_layer: int
    universal_loop: int
    physical_layer_names: tuple[str, ...]
    namespace: str = YOCO_LMCACHE_LAYOUT_VERSION

    @property
    def physical_num_layers(self) -> int:
        return len(self.physical_layer_names)

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: "KVCacheConfig",
        logical_num_layers: int,
        num_cross_layers: int,
        universal_loop: int,
    ) -> "_YocoLMCachePhysicalKVLayout":
        if num_cross_layers <= 0 or num_cross_layers >= logical_num_layers:
            raise ValueError(
                "YOCO LMCache requires cross layers in the range "
                f"[1, {logical_num_layers - 1}], got {num_cross_layers}"
            )
        if universal_loop < 1:
            raise ValueError(
                f"YOCO LMCache requires universal_loop >= 1, got {universal_loop}"
            )

        first_cross_layer = logical_num_layers - num_cross_layers
        names_by_index: dict[int, str] = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                layer_index = extract_layer_index(layer_name)
                previous = names_by_index.setdefault(layer_index, layer_name)
                if previous != layer_name:
                    raise ValueError(
                        "YOCO LMCache found multiple attention caches for logical "
                        f"layer {layer_index}: {previous!r} and {layer_name!r}"
                    )

        # YOCO gives each repeated self-attention pass a physical layer index
        # offset by the base model's logical layer count. Cross-attention
        # layers share the first cross layer's KV tensor across every pass.
        expected_indices = list(range(first_cross_layer + 1))
        for loop_index in range(1, universal_loop):
            expected_indices.extend(
                loop_index * logical_num_layers + layer_index
                for layer_index in range(first_cross_layer)
            )
        physical_indices = sorted(names_by_index)
        if physical_indices != expected_indices:
            raise ValueError(
                "YOCO LMCache physical KV layout must contain self layers and "
                f"cross owner 0..{first_cross_layer}; got {physical_indices}"
            )

        return cls(
            logical_num_layers=logical_num_layers,
            first_cross_layer=first_cross_layer,
            universal_loop=universal_loop,
            physical_layer_names=tuple(
                names_by_index[index] for index in expected_indices
            ),
        )

    @staticmethod
    def _tensor_layout(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tensor.device,
            tensor.data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    def physical_view(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = set(self.physical_layer_names).difference(kv_caches)
        if missing:
            raise ValueError(
                "YOCO LMCache is missing physical KV caches: "
                + ", ".join(sorted(missing))
            )

        names_by_index: dict[int, str] = {}
        for layer_name in kv_caches:
            layer_index = extract_layer_index(layer_name)
            previous = names_by_index.setdefault(layer_index, layer_name)
            if previous != layer_name:
                raise ValueError(
                    "YOCO LMCache registration has multiple attention caches for "
                    f"logical layer {layer_index}: {previous!r} and {layer_name!r}"
                )

        physical_indices = {
            extract_layer_index(name) for name in self.physical_layer_names
        }
        cross_alias_indices = set(
            range(self.first_cross_layer + 1, self.logical_num_layers)
        )
        expected_registration_indices = physical_indices | cross_alias_indices
        if set(names_by_index) != expected_registration_indices:
            raise ValueError(
                "YOCO LMCache registration must contain every physical KV "
                "layer and shared cross alias; expected "
                f"{sorted(expected_registration_indices)}, got "
                f"{sorted(names_by_index)}"
            )

        owner_name = names_by_index[self.first_cross_layer]
        owner = kv_caches[owner_name]
        owner_layout = self._tensor_layout(owner)
        for layer_index in range(self.first_cross_layer + 1, self.logical_num_layers):
            alias_name = names_by_index[layer_index]
            alias = kv_caches[alias_name]
            if alias is not owner or self._tensor_layout(alias) != owner_layout:
                raise ValueError(
                    f"YOCO cross layer {layer_index} KV cache must alias owner "
                    f"layer {self.first_cross_layer} exactly"
                )

        return {name: kv_caches[name] for name in self.physical_layer_names}


@contextmanager
def _use_yoco_lmcache_physical_model_config(
    vllm_config: "VllmConfig", layout: _YocoLMCachePhysicalKVLayout
) -> Iterator[None]:
    """Expose YOCO's physical KV layout while LMCache builds its services."""
    model_config = vllm_config.model_config
    model_arch_config = model_config.model_arch_config
    original_num_layers = model_arch_config.total_num_hidden_layers
    original_model_name = model_config.model

    if original_num_layers != layout.logical_num_layers:
        raise ValueError(
            "YOCO model layer count changed while initializing LMCache: "
            f"{original_num_layers} != {layout.logical_num_layers}"
        )

    model_arch_config.total_num_hidden_layers = layout.physical_num_layers
    model_config.model = f"{original_model_name}::{layout.namespace}"
    try:
        yield
    finally:
        model_config.model = original_model_name
        model_arch_config.total_num_hidden_layers = original_num_layers


def _validate_yoco_lmcache_impl(
    impl: Any, layout: _YocoLMCachePhysicalKVLayout
) -> None:
    config = impl.config
    unsupported = {
        "use_layerwise": config.use_layerwise,
        "enable_async_loading": config.enable_async_loading,
        "enable_blending": config.enable_blending,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(
            "YOCO LMCache physical KV layout does not yet support: "
            + ", ".join(enabled)
        )
    if not config.use_gpu_connector_v3:
        raise ValueError("YOCO LMCache requires use_gpu_connector_v3=true")

    metadata = impl.lmcache_engine_metadata
    if metadata is None:
        raise RuntimeError("LMCache did not create metadata for the YOCO connector")
    if metadata.kv_shape[0] != layout.physical_num_layers:
        raise ValueError(
            "LMCache physical layer count does not match YOCO: "
            f"{metadata.kv_shape[0]} != {layout.physical_num_layers}"
        )

    namespace_suffix = f"::{layout.namespace}"
    if not metadata.model_name.endswith(namespace_suffix):
        raise ValueError(
            "LMCache YOCO cache namespace is missing from model_name: "
            f"{metadata.model_name!r}"
        )
    if impl.num_layers != layout.physical_num_layers:
        raise ValueError(
            "LMCache adapter layer count does not match YOCO: "
            f"{impl.num_layers} != {layout.physical_num_layers}"
        )

    engine = impl.lmcache_engine
    if engine is not None:
        if engine.num_layers != layout.physical_num_layers:
            raise ValueError(
                "LMCache engine layer count does not match YOCO: "
                f"{engine.num_layers} != {layout.physical_num_layers}"
            )
        if engine.metadata.kv_shape != metadata.kv_shape:
            raise ValueError("LMCache engine and manager metadata disagree")

    logger.info(
        "Configured YOCO LMCache layout %s: %d logical layers -> %d physical "
        "KV tensors (cross owner layer %d)",
        layout.namespace,
        layout.logical_num_layers,
        layout.physical_num_layers,
        layout.first_cross_layer,
    )


class LMCacheKVEvents(KVConnectorKVEvents):
    """
    Concrete implementation of KVConnectorKVEvents using KVEventAggregator.
    """

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "LMCacheKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<LMCacheKVEvents events={self.get_all_events()}>"


class LMCacheConnectorV1(KVConnectorBase_V1):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        LMCache requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled. The wait_for_layer_load and save_kv_layer
        methods perform actual async synchronization that cannot be
        captured in CUDA graphs.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        self._yoco_physical_kv_layout: _YocoLMCachePhysicalKVLayout | None = None
        if vllm_config.model_config.hf_config.model_type == "yoco":
            logical_num_layers = vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config
            )
            num_cross_layers = int(
                getattr(vllm_config.model_config.hf_config, "yoco_cross_layers", 0)
            )
            universal_loop = int(
                getattr(vllm_config.model_config.hf_config, "universal_loop", 1)
            )
            self._yoco_physical_kv_layout = (
                _YocoLMCachePhysicalKVLayout.from_kv_cache_config(
                    kv_cache_config,
                    logical_num_layers=logical_num_layers,
                    num_cross_layers=num_cross_layers,
                    universal_loop=universal_loop,
                )
            )

            extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
            disabled_features = (
                "lmcache.use_layerwise",
                "lmcache.enable_async_loading",
                "lmcache.enable_blending",
            )
            explicitly_enabled = [
                key for key in disabled_features if extra_config.get(key) is True
            ]
            if explicitly_enabled:
                raise ValueError(
                    "YOCO LMCache physical KV layout does not yet support: "
                    + ", ".join(explicitly_enabled)
                )
            for key in disabled_features:
                extra_config.setdefault(key, False)
            extra_config.setdefault("lmcache.use_gpu_connector_v3", True)

        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", False
        )
        if self._yoco_physical_kv_layout is not None and use_native:
            raise ValueError(
                "YOCO LMCache 0.5.3 physical KV layout requires the packaged "
                "adapter; set use_native=false"
            )
        if use_native:
            logger.info("Initializing native LMCache connector")
            # lazy import
            from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

            _adapter = lmcache_integration.vllm_v1_adapter

            cls = _adapter.LMCacheConnectorV1Impl
        else:
            logger.info("Initializing latest dev LMCache connector")
            # lazy import
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
            )

            cls = LMCacheConnectorLatestImpl

        if self._yoco_physical_kv_layout is not None:
            with _use_yoco_lmcache_physical_model_config(
                vllm_config, self._yoco_physical_kv_layout
            ):
                self._lmcache_engine = cls(vllm_config, role, self)
            _validate_yoco_lmcache_impl(
                self._lmcache_engine, self._yoco_physical_kv_layout
            )
        else:
            self._lmcache_engine = cls(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        if self._yoco_physical_kv_layout is not None:
            kv_caches = self._yoco_physical_kv_layout.physical_view(kv_caches)

        if hasattr(self._lmcache_engine, "register_kv_caches"):
            self._lmcache_engine.register_kv_caches(kv_caches)
        else:
            logger.warning(
                "LMCache engine does not support register_kv_caches, "
                "please check and use the latest version"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        method = getattr(self._lmcache_engine, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()

        # Fallback for older versions that don't support this method
        return set()

    def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """

        events = self._lmcache_engine.get_kv_events()  # type: ignore [attr-defined]
        if not events:
            return None

        blocks: list[BlockStored] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
                lora_name=getattr(e, "lora_name", None),
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None
