# YOCO expert parallel load balancing

YOCO can use vLLM's existing EPLB to collect expert loads, replicate busy
experts, and migrate their weights within a fixed expert-parallel group.
The model retains its logical top-k routing and shared-expert computation.
EPLB maps the selected logical experts to physical replicas after routing.

For example, append these options to an otherwise compatible YOCO launch:

```bash
--tensor-parallel-size 4 \
--enable-expert-parallel \
--enable-eplb \
--eplb-config '{"num_redundant_experts":8,"window_size":1000,"step_interval":3000,"use_async":true}'
```

`num_redundant_experts` is the extra number **per MoE layer across the whole
EP group**, not per GPU. The logical count plus the redundant count must be
divisible by the EP size. For 128 logical experts and 8 extra replicas on
4 GPUs, each GPU stores 34 physical experts. Zero extra replicas is also
valid: EPLB then only changes placement. These slots are allocated during
initialization; changing the replica budget requires a new engine.

YOCO uses unrestricted top-k, so its EPLB expert-group count is one. Repeated
universal-loop execution accumulates into the same weight-owning layer's load
statistics. Shared experts, router weights and latent projections are excluded
from expert weight migration. Quantized expert scales are registered with
their corresponding weights through the existing vLLM interface.

The packed YOCO checkpoint loader populates every physical slot using the
same initial physical-to-logical mapping as EPLB. Registration happens after
weight loading and quantization. Checkpoint reload after EPLB registration
is rejected before consuming weights: reusing the initial mapping after a
rebalance would corrupt the association between weights and routes.

## Execution and compatibility

- Synchronous EPLB moves weights between inference steps. Asynchronous EPLB
  prepares weights in the background and installs them at the existing vLLM
  synchronization points. Neither path requires restarting the engine.
  This does not imply zero latency overhead.
- The EP group has a fixed size. YOCO rejects elastic EP during model
  initialization; growing or shrinking the live GPU group is not supported.
- Mega Gate already excludes EP. The frozen-weight MegaMoE path and the
  YOCO-private TRTLLM BF16 backend do not support EPLB; their existing
  compatibility checks remain in effect. Use a supported modular MoE backend,
  such as Triton for BF16.
- Online FP8 uses the existing vLLM quantization and EPLB weight/scale interfaces.
  B200 FP8 execution, real checkpoint quality, throughput and tail latency
  still require validation on the target deployment. This adaptation is not
  a claim of bitwise train/inference equivalence for Align execution.

## Validation

The focused tests cover initial redundant loading (including more replicas
than logical experts), dense-only model detection, unique MoE registration,
weight/scale ownership, rejection of reload after registration, and CUDA Graph
replay with an updated map and padded-token load accounting.

The distributed test constructs a real small BF16 YOCO MoE block, including
latent transforms and a shared expert, on two CUDA GPUs. It exercises both
synchronous and asynchronous EPLB, checks that a new layout is installed,
compares outputs before and after migration, verifies migrated weights against
the original logical checkpoint, and checks that unrelated parameters and
expert storage addresses remain unchanged.

```bash
.venv/bin/python -m pytest --confcutdir=tests/model_executor -q \
  tests/model_executor/test_yoco_eplb.py \
  tests/model_executor/test_yoco_conversion.py \
  tests/model_executor/test_yoco_moe_composition.py

.venv/bin/python -m pytest --confcutdir=tests/distributed -q \
  tests/distributed/test_yoco_eplb.py
```
