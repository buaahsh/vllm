# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        Fp8BlockScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
    )

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.kernel import MoEBackend
from vllm.config.yoco import YocoExecutionMode, YocoMoEPolicy
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClampFP32
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.yoco_fast import refresh_yoco_module_cache
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm as RMSNorm
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_HIDDEN_SIZE as _YOCO_L3_HIDDEN_SIZE,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_align_linear as _yoco_align_linear,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_router_linear as _yoco_align_router_linear,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_topk_routing as _yoco_align_topk_routing,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_normalized_router_linear as _yoco_normalized_router_linear,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_topk_routing as _yoco_topk_routing,
)
from vllm.model_executor.models.yoco_config import (
    _cfg_int,
    _select_yoco_online_fp8_moe_backend,
    _swiglu_limit,
)
from vllm.model_executor.models.yoco_diagnostics import create_yoco_route_dumper
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


@torch.compile
def _yoco_align_shared_expert_swiglu(
    up: torch.Tensor,
    gate: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Mirror llm-train's compiled shared-expert SwiGLU expression."""
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return up * F.silu(gate)


@torch.compile
def _yoco_align_shared_expert_swiglu_unclamped(
    up: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return up * F.silu(gate)


class YOCOSharedExperts(nn.Module):
    """Shared-expert MLP for YOCO MoE blocks (SwiGLU)."""

    _fast_down_weight_t: torch.Tensor | None
    act_fn: SiluAndMul | SiluAndMulWithClampFP32

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
        prefix: str,
        swiglu_limit: float = 10.0,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        # llm-train evaluates two independent BF16 projections in this order:
        # ``up_proj(x)``, then ``gate_proj(x)``.  A merged-N cuBLAS GEMM can
        # select a different reduction kernel and is therefore not a strict
        # numerical substitute.  Align reuses contiguous views of the packed
        # checkpoint parameter but restores the two original GEMM boundaries.
        tp_size = get_tensor_model_parallel_world_size()
        self.use_separate_projection = (
            execution_mode == "align" and quant_config is None and tp_size == 1
        )
        self.use_fast_down_transpose = (
            execution_mode == "fast"
            and tp_size == 1
            and hidden_size == _YOCO_L3_HIDDEN_SIZE
            and intermediate_size == 1280
        )
        self.register_buffer("_fast_down_weight_t", None, persistent=False)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.use_fast_down_transpose = self.use_fast_down_transpose and isinstance(
            self.down_proj.quant_method, UnquantizedLinearMethod
        )
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerBlockOnlineLinearMethod,
        )

        self.use_fast_fp8_swiglu = (
            execution_mode == "fast"
            and tp_size == 1
            and swiglu_limit > 0
            and intermediate_size % 128 == 0
            and isinstance(self.down_proj.quant_method, Fp8PerBlockOnlineLinearMethod)
            and self.down_proj.quant_method.supports_silu_mul_fusion()
        )
        # Clamped SwiGLU to match training (``swiglu_limit``).  When the limit is
        # non-positive, fall back to the plain (unclamped) activation.
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit > 0:
            self.act_fn = SiluAndMulWithClampFP32(
                self.swiglu_limit,
                enforce_enable=True,
            )
        else:
            self.act_fn = SiluAndMul()

    def initialize_fast_weight_cache(self) -> None:
        """Cache B200's faster M=1 down-projection operand layout."""
        if not self.use_fast_down_transpose:
            return
        weight = cast(torch.Tensor, self.down_proj.weight)
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            return
        capability = torch.cuda.get_device_capability(weight.device)
        if capability[0] != 10:
            return
        with torch.no_grad():
            self._fast_down_weight_t = refresh_yoco_module_cache(
                self, "_fast_down_weight_t", weight.t().contiguous()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_separate_projection:
            weight = cast(torch.Tensor, self.gate_up_proj.weight)
            gate_weight = weight.narrow(0, 0, self.intermediate_size)
            up_weight = weight.narrow(
                0,
                self.intermediate_size,
                self.intermediate_size,
            )
            # Python evaluates llm-train's ``swiglu(up_proj(x), gate_proj(x))``
            # arguments left-to-right. Keep that order as well as each BF16
            # GEMM store before the compiled FP32 activation expression.
            up = _yoco_align_linear(x, up_weight)
            gate = _yoco_align_linear(x, gate_weight)
            if self.swiglu_limit > 0:
                x = _yoco_align_shared_expert_swiglu(
                    up,
                    gate,
                    self.swiglu_limit,
                )
            else:
                x = _yoco_align_shared_expert_swiglu_unclamped(up, gate)
        else:
            gate_up, _ = self.gate_up_proj(x)
            if self.use_fast_fp8_swiglu and gate_up.dtype == torch.bfloat16:
                return cast(
                    "Fp8PerBlockOnlineLinearMethod", self.down_proj.quant_method
                ).apply_silu_mul(
                    self.down_proj, gate_up.contiguous(), self.swiglu_limit
                )
            x = self.act_fn(gate_up)
        if self._fast_down_weight_t is not None and x.shape[0] == 1:
            x = torch.mm(x, self._fast_down_weight_t)
        else:
            x, _ = self.down_proj(x)
        return x


class YOCOLatentInputTransform(nn.Module):
    """Project and normalize only the routed-expert input."""

    def __init__(self, proj: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.proj = proj
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        if isinstance(projected, tuple):
            projected = projected[0]
        return self.norm(projected)


class YOCOLatentOutputTransform(nn.Module):
    """Normalize and project only the routed-expert output."""

    def __init__(self, norm: nn.Module, proj: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.proj = proj
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerBlockOnlineLinearMethod,
        )

        self.fuse_fp8_norm = (
            envs.VLLM_YOCO_FP8_LATENT_NORM_FUSION
            and isinstance(norm, RMSNorm)
            and norm.execution_mode == "fast"
            and norm.dim == 1024
            and norm.weight.dtype == torch.bfloat16
            and isinstance(proj, ReplicatedLinear)
            and proj.bias is None
            and isinstance(proj.quant_method, Fp8PerBlockOnlineLinearMethod)
            and proj.quant_method.supports_silu_mul_fusion()
        )
        if self.fuse_fp8_norm:
            quantizer = cast(
                "Fp8PerBlockOnlineLinearMethod",
                cast(ReplicatedLinear, proj).quant_method,
            ).fp8_linear.quant_fp8
            self.fuse_fp8_norm = quantizer._enforce_enable or quantizer.enabled()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_fp8_norm and x.dtype == torch.bfloat16:
            quantized, scales = torch.ops.vllm.yoco_latent_rms_norm_fp8(
                x,
                self.norm.weight,
                self.norm.eps,
            )
            # ReplicatedLinear has no TP collective to bypass. The selected
            # backend and its weight/scale lookup are shared with normal apply.
            method = cast(
                "Fp8PerBlockOnlineLinearMethod",
                cast(ReplicatedLinear, self.proj).quant_method,
            )
            kernel = cast("Fp8BlockScaledMMLinearKernel", method.fp8_linear)
            return kernel.apply_quantized_weights(self.proj, quantized, scales)
        projected = self.proj(self.norm(x))
        return projected[0] if isinstance(projected, tuple) else projected


class YOCOCombinedOutputTransform(nn.Module):
    """Apply YOCO's shared gate and combine reduced shared+routed outputs."""

    def __init__(
        self,
        shared_gate: ReplicatedLinear,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.shared_gate = shared_gate
        self.execution_mode = execution_mode

    def forward(
        self,
        shared_output: torch.Tensor,
        routed_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.execution_mode == "fast"
            and HAS_TRITON
            and current_platform.is_cuda()
            and shared_output.is_cuda
            and routed_output.is_cuda
            and hidden_states.is_cuda
            and shared_output.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
            and hidden_states.dtype == torch.bfloat16
            and shared_output.shape == routed_output.shape == hidden_states.shape
            and shared_output.shape[-1] == 3072
            and shared_output.is_contiguous()
            and routed_output.is_contiguous()
            and hidden_states.is_contiguous()
        ):
            return torch.ops.vllm.yoco_fused_shared_gate_moe_output(
                shared_output,
                routed_output,
                hidden_states,
                cast(torch.Tensor, self.shared_gate.weight),
            )

        # llm-train builds shared_gate with default MixPrecisionLinear settings:
        # no MXFP8 path and the parameter follows the module default dtype.
        linear = _yoco_align_linear if self.execution_mode == "align" else F.linear
        scale = linear(
            hidden_states,
            cast(torch.Tensor, self.shared_gate.weight).to(hidden_states.dtype),
        )
        gated_shared = torch.sigmoid(scale) * shared_output
        # Keep the same operand order as llm-train's
        # ``final_hidden_states + shared_gate_score * self.shared(x)``.
        return routed_output + gated_shared


class YOCOMoE(nn.Module):
    """YOCO MoE block: routed top-k + gated shared expert + final reduce."""

    _normalized_gate_weight: torch.Tensor | None
    _bf16_gate_weight: torch.Tensor | None
    fc1_latent_proj: ReplicatedLinear | None
    fc2_latent_proj: ReplicatedLinear | None
    fc1_latent_norm: RMSNorm | nn.Identity | None
    fc2_latent_norm: RMSNorm | nn.Identity | None

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
        moe_backend_override: MoEBackend | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.execution_mode = execution_mode
        self.bf16_chain = execution_mode == "fast" and envs.VLLM_YOCO_BF16_CHAIN
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.num_experts = _cfg_int(config, "num_experts", "moe_expert_num")
        self.top_k = _cfg_int(config, "num_experts_per_tok", "moe_top_k", "top_k")
        self.moe_intermediate_size = _cfg_int(
            config, "moe_intermediate_size", "moe_ffn_dim"
        )
        self.moe_latent_dim = _cfg_int(config, "moe_latent_dim", default=0)
        self.moe_latent_norm = bool(getattr(config, "moe_latent_norm", False))
        self.shared_intermediate_size = _cfg_int(
            config, "shared_expert_intermediate_size", "d_shared_expert"
        )
        self.swiglu_limit = _swiglu_limit(config)
        self.router_weights_normalized = bool(
            getattr(config, "router_weights_normalized", False)
        )
        num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        self._yoco_logical_route_info = (
            (
                layer_idx,
                num_hidden_layers - yoco_cross_layers,
                _cfg_int(config, "universal_loop", default=1),
            )
            if layer_idx is not None
            else None
        )
        # Older YOCO checkpoints keep the raw router weights and normalize
        # them at inference time. The weights are immutable after loading, so
        # cache that normalized FP32 tensor once instead of rebuilding it in
        # every decoder-block execution. Keep it non-persistent so checkpoint
        # names and serialization remain unchanged.
        self._route_dumper = create_yoco_route_dumper()
        self.register_buffer("_normalized_gate_weight", None, persistent=False)
        self.register_buffer("_bf16_gate_weight", None, persistent=False)

        # Keep FP32 checkpoint weights; the BF16 chain uses a separate cache.
        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.set_out_dtype(torch.float32)

        # Keep the shared output local so FusedMoE can execute these GEMMs on its
        # auxiliary stream. The runner still performs a separate TP all-reduce
        # after the routed reduction to preserve YOCO's numerical boundaries.
        self.shared_experts = YOCOSharedExperts(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
            swiglu_limit=self.swiglu_limit,
            execution_mode=execution_mode,
        )

        # Scalar shared-expert sigmoid gate.  Replicated across TP — every
        # rank computes the same per-token scaling factor.
        self.shared_gate = ReplicatedLinear(
            input_size=self.hidden_size,
            output_size=1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_gate",
        )

        expert_hidden_size = self.moe_latent_dim or self.hidden_size
        if self.moe_latent_dim:
            rms_eps = float(
                getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
            )
            # Fast latent GEMMs follow the configured linear precision, including
            # online FP8 and explicit per-layer ignore rules. Align keeps BF16.
            latent_quant_config = quant_config if execution_mode == "fast" else None
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.moe_latent_dim,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.fc1_latent_proj",
                return_bias=False,
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_latent_dim,
                output_size=self.hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.fc2_latent_proj",
                return_bias=False,
            )
            self.fc1_latent_norm = (
                RMSNorm(
                    self.moe_latent_dim,
                    eps=rms_eps,
                    execution_mode=execution_mode,
                )
                if self.moe_latent_norm
                else nn.Identity()
            )
            self.fc2_latent_norm = (
                RMSNorm(
                    self.moe_latent_dim,
                    eps=rms_eps,
                    execution_mode=execution_mode,
                )
                if self.moe_latent_norm
                else nn.Identity()
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None
            self.fc1_latent_norm = None
            self.fc2_latent_norm = None

        routed_input_transform = (
            YOCOLatentInputTransform(self.fc1_latent_proj, self.fc1_latent_norm)
            if self.fc1_latent_proj is not None and self.fc1_latent_norm is not None
            else None
        )
        routed_output_transform = (
            YOCOLatentOutputTransform(self.fc2_latent_norm, self.fc2_latent_proj)
            if self.fc2_latent_proj is not None and self.fc2_latent_norm is not None
            else None
        )
        combined_output_transform = YOCOCombinedOutputTransform(
            self.shared_gate,
            execution_mode=execution_mode,
        )

        # NOTE(swiglu_limit): Both the shared expert (above, via the fused FP32
        # clamped SwiGLU op) and the ROUTED experts (below) apply the training
        # ``swiglu_limit`` clamp (clamp-before-silu), for exact train/inference
        # parity. In align mode, the routed clamp and FP32 routing probability
        # are fused before the BF16 store and W2 GEMM, matching llm-train's
        # ``fused_silu`` rounding boundary. Fast BF16 keeps the cheaper
        # mathematically equivalent W2-epilogue weighting. The routed clamp is
        # threaded through FusedMoE into the modular experts. Align and the
        # W8A8 Triton fallback use ``yoco_weighted_swiglu``. Both W8A8 backends
        # apply routing probabilities before W2 input quantization. CAUTION:
        # the loose limit=10.0
        # can make some checkpoints (observed: adamw-3000) degenerate under pure
        # greedy decoding; use temperature>0 in production. Kept on per owner's
        # request for training fidelity.
        routing_function = (
            _yoco_align_topk_routing
            if execution_mode == "align"
            else _yoco_topk_routing
        )
        if (
            execution_mode == "fast"
            and quant_config is not None
            and moe_backend_override is None
            and get_current_vllm_config().kernel_config.moe_backend == "auto"
        ):
            moe_backend_override = _select_yoco_online_fp8_moe_backend(
                quant_config,
                f"{prefix}.experts",
                get_tensor_model_parallel_world_size(),
            )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=expert_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func="softmax",
            custom_routing_function=routing_function,
            swiglu_limit=self.swiglu_limit,
            apply_router_weight_before_w2=True,
            yoco_policy=YocoMoEPolicy.for_mode(cast(YocoExecutionMode, execution_mode)),
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            combined_output_transform=combined_output_transform,
            reduce_shared_experts_separately=True,
            use_tuned_config=execution_mode == "fast",
            moe_backend_override=moe_backend_override,
            prefix=f"{prefix}.experts",
        )

    def initialize_router_weight_cache(self) -> None:
        if self.execution_mode == "align" or self.router_weights_normalized:
            self._normalized_gate_weight = None
        else:
            with torch.no_grad():
                weight = cast(torch.Tensor, self.gate.weight)
                normalized = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
                self._normalized_gate_weight = refresh_yoco_module_cache(
                    self, "_normalized_gate_weight", normalized
                )
        if getattr(self, "bf16_chain", False):
            weight = cast(torch.Tensor, self.gate.weight)
            if self._normalized_gate_weight is not None:
                weight = self._normalized_gate_weight
            self._bf16_gate_weight = refresh_yoco_module_cache(
                self, "_bf16_gate_weight", weight.detach().to(torch.bfloat16)
            )
        else:
            self._bf16_gate_weight = None

    def forward(self, hidden_states: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        gate_weight = cast(torch.Tensor, self.gate.weight)
        normalize_gate_weight = not self.router_weights_normalized
        if self._normalized_gate_weight is not None:
            gate_weight = self._normalized_gate_weight
            normalize_gate_weight = False

        if getattr(self, "bf16_chain", False):
            assert self._bf16_gate_weight is not None
            # Both GEMM operands and logits are BF16. The small selected
            # Top-K probabilities retain the FP32 backend interface.
            router_input = hidden_states.to(torch.bfloat16)
            if hidden_states.is_cuda and current_platform.is_cuda():
                router_logits = torch.ops.vllm.yoco_router_linear_bf16(
                    router_input, self._bf16_gate_weight
                )
            else:
                router_logits = F.linear(router_input, self._bf16_gate_weight)
        elif self.execution_mode == "align":
            router_logits = _yoco_align_router_linear(
                hidden_states.float(), gate_weight, normalize_gate_weight
            )
        elif hidden_states.is_cuda and current_platform.is_cuda():
            router_logits = torch.ops.vllm.yoco_router_linear_tf32(
                hidden_states.float(),
                gate_weight,
                normalize_gate_weight,
            )
        else:
            if not normalize_gate_weight:
                router_logits = F.linear(hidden_states.float(), gate_weight)
            else:
                router_logits = _yoco_normalized_router_linear(
                    hidden_states.float(), gate_weight
                )
        dumper = getattr(self, "_route_dumper", None)
        if dumper is not None:
            dumper(
                hidden_states,
                router_logits,
                self.top_k,
                self._yoco_logical_route_info,
                loop_idx,
                execution_mode=self.execution_mode,
            )
        # FusedMoE overlaps the local shared-expert GEMMs with routed dispatch
        # and expert compute. It then preserves YOCO's original order: routed
        # TP reduction, shared TP reduction, latent output transform, sigmoid
        # shared gate, and finally the sum.
        final = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return final.view(num_tokens, hidden_dim)
