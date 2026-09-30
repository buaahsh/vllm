# YOCO DSpark serving branch

This branch packages the vLLM source changes used by the Dense2 DSpark serving
experiment. It is based on commit
`28d82e2e464f0464332d742140ad62453f9eddd8` and includes the runtime's optional
post-final-norm feature export.

## Included changes

- Keep DSpark draft KV-cache layer names separate from looped YOCO target layers.
- Export auxiliary features from the final encoder pass and before the target's
  final normalization, in the target compute dtype.
- Support the optional `num_hidden_layers + 1` auxiliary feature identifier for
  the post-final-norm state used by target validation.
- Reject decoder-skipping prefill when auxiliary features are required.
- Support dense YOCO feed-forward layers and attention implementations needed by
  the target/draft configuration.
- Avoid exporting partial hidden states for cancelled or failed requests.

## Running

Use a compatible vLLM build and CUDA environment, the matching Speculators
package, and a YOCO target checkpoint paired with a DSpark draft checkpoint.
This source branch does not contain model weights, native shared libraries, or
an embedded copy of Speculators. Build/install dependencies using the repository's
normal environment instructions; preserve the deployment image's native DeepGEMM.

The validated deployment uses a single B200, BF16, TP1, the V2 model runner,
Fast execution, and FlashInfer. The target's configuration must select the
intended execution mode. For example, after installing this branch:

```bash
export TARGET_MODEL=/path/to/yoco-target
export DRAFT_MODEL=/path/to/dspark-draft

VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=0 OMP_NUM_THREADS=4 \
  vllm serve "$TARGET_MODEL" \
    --spec-model "$DRAFT_MODEL" \
    --spec-method dspark \
    --spec-tokens 8 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.75 \
    --no-enable-prefix-caching \
    --max-cudagraph-capture-size 1152
```

For K8 at 128 decoding requests, verification needs `128 * (8 + 1) = 1152`
tokens. A graph capture ceiling of 1024 does not cover that shape. The explicit
ceiling above includes it. K6 can be selected with `--spec-tokens 6`.

The layer IDs in the draft checkpoint must match the target's exported
features. A postnorm identifier is an export marker, not an additional parameter
layer or a replacement for the configured draft input features.

## Validation and limits

The source was copied from the serving runtime without changing its model
implementation. The publication checkout retains the local cache-name regression
test and adds coverage for the existing postnorm export. The following CPU tests
passed with GPU visibility disabled: 27 passed.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/model_executor/test_yoco_prefill.py \
  tests/model_executor/test_yoco_attention.py::test_dspark_cache_names_do_not_collide_with_looped_yoco \
  tests/v1/kv_connector/unit/test_hidden_states_connector.py
```

Existing GPU runs establish that this target/draft setup starts and performs
speculative decoding. They do not establish quality equivalence across Fast
batch sizes or K values. Earlier fixed-token-prefix performance runs are not a
validated assistant-turn workload, so this branch makes no production speedup
claim based on those numbers. No new GPU workload is started by publishing this
branch.

For assistant-response evaluation, finish complete user/tool messages, supply
the assistant generation prefix, and honor the checkpoint's EOS/role termination
IDs. Keep input panels and output limits consistent across compared concurrency
levels; report accepted draft tokens separately from the correction/bonus token.
