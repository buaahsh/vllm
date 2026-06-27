# Serving strategy — 4 YOCO-30A3B checkpoints on 4× B200, one port

Host **four different** `30A3B-A3B` SFT checkpoints simultaneously, **one model
per GPU** (GPUs 0–3), all reachable through a **single OpenAI-compatible
endpoint on port 8000**, selected by the request's `model` field.

Set up 2026-06-27.

## Why this layout (not DP / TP)

Each checkpoint is ~30B-total / ~3B-active MoE and fits comfortably on one
180 GB B200. We want four *distinct* models live at once, so the right split is
**1 full replica per GPU** — no tensor/data parallelism. vLLM serves exactly one
model per process, so we run **4 independent single-GPU servers** on internal
ports 8001–8004 and put a tiny **router** on 8000 that dispatches by model name.

Single-GPU (TP=1) keeps the routed-expert intermediate at the full 1280, so the
FlashInfer TRTLLM MoE backend does **not crash** — but it still produces
**incorrect / garbled output** for this model (see the root-cause section
below). **Always serve with `--moe-backend triton`** on every backend, single-
or multi-GPU. The launcher already does this.

## Model → GPU → port → name

| Served name  | GPU | Internal port | Checkpoint (`…/exp/sft/…`)                                   |
|--------------|-----|---------------|-------------------------------------------------------------|
| `muon-3000`  | 0   | 8001          | `30A3B-73k-sft-65k-muon-bsz1M-…/0000-3000-hf`               |
| `muon-1500`  | 1   | 8002          | `30A3B-73k-sft-65k-muon-bsz1M-…/0000-1500-hf`               |
| `adamw-3000` | 2   | 8003          | `30A3B-73k-sft-65k-adamw-…/0000-3000-hf`                    |
| `adamw-1500` | 3   | 8004          | `30A3B-73k-sft-65k-adamw-…/0000-1500-hf`                    |

Client picks the model by name against **port 8000**:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"adamw-3000","messages":[{"role":"user","content":"hi"}],
       "max_tokens":256,"temperature":0.7,"top_p":0.95}'
```

`GET /v1/models` on 8000 lists all four; `GET /health` reports router status.
Answer text is in `message.content`; chain-of-thought in `message.reasoning`.

## The router

`/workspace/shaohanh/vllm_model_router.py` — a ~120-line aiohttp reverse proxy:
reads the `model` field from each request body, forwards (streaming-safe, SSE)
to the matching backend, aggregates `/v1/models` and `/health`. Unknown model →
HTTP 404 listing valid names.

## Gotcha: vLLM startup hangs on a heavy `eval_outputs/` subdir

On startup vLLM runs a **recursive `rglob`** over the model directory (to detect
a Mistral repo). One checkpoint (`adamw/0000-3000-hf`) contains an
`eval_outputs/` subdirectory holding large eval dumps on the FUSE mount;
recursing into it **hangs the server forever**, frozen right after printing the
launch args, before `Initializing a V1 LLM engine` (main thread stuck in
`request_wait_answer`; no engine-core child ever spawns).

Stack: `create_model_config → get_config → is_mistral_model_repo →
any_pattern_in_repo_files → list_repo_files → pathlib.rglob → scandir` (hang).

**Fix:** serve from a *clean directory of symlinks* to only the top-level model
files (config, tokenizer, `*.safetensors`, index), excluding all subdirs. These
live under `/workspace/shaohanh/served_models/<name>-hf/`. The launch script
rebuilds them automatically. (Alternatively, point `--served-model-name` at a
copy without `eval_outputs/`, or delete that subdir from the checkpoint.)

## Start / stop / status

```bash
bash /workspace/shaohanh/serve_4models_8000.sh start    # build clean dirs, launch 4 backends + router
bash /workspace/shaohanh/serve_4models_8000.sh status   # list models + GPU usage
bash /workspace/shaohanh/serve_4models_8000.sh stop      # kill router + 4 backends
```

All processes are launched with `setsid` (detached, PPID=1) so they survive the
launching shell. Logs:
`/workspace/shaohanh/vllm_<name>_<port>.log` and `vllm_router_8000.log`.

## Operational notes

- Launching all four backends **simultaneously** loads 4×60 GB from the same
  FUSE mount at once and can stall one of them; the script waits on each
  `/health` and the router only needs backends up before it routes to them. If a
  backend lags, it still joins once healthy.
- `--gpu-memory-utilization 0.9` ≈ 164 GB/GPU; ~60 GB weights leaves ample KV
  cache for `--max-model-len 65536`.
- Recommended sampling to avoid runaway repetition (esp. `muon-3000`):
  `temperature=0.7, top_p=0.95, repetition_penalty=1.05`.
- Use idle GPUs only; on this shared box GPUs 4/5/7 are often busy. This setup
  pins 0–3 via `CUDA_VISIBLE_DEVICES`.

## 乱码 (garbled output) root cause & fix — 2026-06-27

> **DEFINITIVE ROOT CAUSE & FIX (read this first — supersedes the analysis
> below).** The garbling was caused by the **FlashInfer TRTLLM Unquantized MoE
> backend**, which vLLM auto-selects for this model. YOCO-30A3B has
> `moe_ffn_dim = 1280`, which is **not 128-aligned**. On multi-GPU this backend
> **crashes** (`intermediate_size % 128 == 0` check, 64 vs 0); on single-GPU it
> does **not** crash but silently produces **incorrect results** → single-token
> repetition / 乱码.
>
> **Fix:** serve every backend with **`--moe-backend triton`**. The Triton MoE
> kernel computes correctly and fixes ALL four checkpoints completely —
> verified across English, Chinese, short and long generations, with **no
> sampling penalty of any kind**. The launcher `serve_4models_8000.sh` now
> always passes `--moe-backend triton`.
>
> Backend selection appears in the startup log:
> `unquantized.py:212 Using TRITON Unquantized MoE backend` (good) vs.
> `Using FlashInfer TRTLLM Unquantized MoE backend` (broken — the old default).
>
> The `qk_rms_clip` config additions and the `<sop>` template removal below were
> partial / red-herring fixes (the config params are still correct to keep, as
> they match training and `adamw-1500`, but they were NOT the primary cause).
> The router's `frequency_penalty` injection is **no longer needed** and has
> been removed — Triton output is stable without it.

### Routed-expert SwiGLU clamp — tried and REJECTED (2026-06-27)

Training applied a SwiGLU clamp (`swiglu_limit=10.0`, clamp-before-silu) to the
**routed** experts (training kernel `llm-train/llm/kernel/moe_ffn.py:
_fused_silu_kernel`), but vLLM's fused-MoE routed path runs plain SiLU and
ignores it (only the *shared* expert clamps, via `SiluAndMulWithClamp`). Routed
activations exceed the limit (~18.6 vs 10.0), so it looked like a real
train/inference parity gap worth closing.

We **implemented and tested** the exact training-faithful routed clamp
(threaded `swiglu_limit` from `FusedMoE(...)` → `UnquantizedFusedMoEMethod.
forward_native` → `FusedMoEExpertsModular.activation` → `swiglu_limit_func`;
gate/up ordering verified identical to the working `silu_and_mul`).

**Result: REJECTED.** With the clamp on, `adamw-3000` **degenerates under greedy
decoding** (endless `*` repetition), while the **unclamped** path is clean for
all four checkpoints. The loose limit=10.0 introduces a hard nonlinearity right
where the routed activation peaks sit, and in bf16 inference that tips greedy
decoding into a degenerate attractor for at least one checkpoint. So even though
the clamp is "theoretically" faithful to training, the **unclamped Triton path
is the more robust choice** and is what we ship.

Takeaway: the real 乱码 fix was the **TRITON MoE backend** (`--moe-backend
triton`), NOT any activation clamp. The routed-expert clamp code was fully
reverted; the model intentionally serves routed experts unclamped.

### Earlier (partial) investigation — kept for history

**Symptom:** 3 of the 4 checkpoints (`muon-3000`, `muon-1500`, `adamw-3000`)
produced total garbage — endless single-token repetition like `江江江…` /
`用用用…` — even with greedy decoding. Only `adamw-1500` was clean.

**Investigation:** the user suspected the chat template. There *was* a template
difference (the 3 bad ones had a literal `<sop>` as the first line, which
double-encodes the GLM start-of-prompt token), and that was fixed — but it was
**not** the cause of the garble.

**Real root cause:** `config.json` was missing the YOCO numerical-stability
clipping params that the model was trained with. The working `adamw-1500` dir
had them (hand-added, with `*.orig` backups); the other 3 still had the
unmodified original config. Without these, activation outliers blow up →
degenerate repetition. vLLM only enables the clamps when present
(`yoco.py`: `if getattr(config, "qk_rms_clip", False)`).

Added to all 3 `config.json` (matching `adamw-1500`):
```json
"qk_rms_clip": true,
"qk_rms_limit": 3.0,
"swiglu_limit": 10.0
```
This eliminated the total collapse — all models now start coherent.

**Residual repetition (separate, milder issue):** the later/muon checkpoints
still fall into repetition loops on longer generations (`adamw-1500` does not —
it is the only fully-stable checkpoint). A **`frequency_penalty`** fixes this;
note vLLM does **not** read `frequency_penalty` from `generation_config.json`
(only `repetition_penalty/temperature/top_p/top_k/min_p` — see
`vllm/config/model.py: get_diff_sampling_param`). So the router
(`vllm_model_router.py`) injects `frequency_penalty: 0.7` by default for
`muon-3000/muon-1500/adamw-3000` when the client omits it.

`muon-1500` is the weakest checkpoint — it can still drift off-topic / loop on
long outputs; this is model quality, not a serving-config problem.

**Edits applied (all on the shared FUSE source dirs, backups kept):**
- `config.json`  → added the 3 clip params (`*.cfgbak-*` backup)
- `generation_config.json` → added `temperature/top_p/repetition_penalty`
  defaults matching `adamw-1500` (`*.genbak-*` backup)
- `chat_template.jinja` + `tokenizer_config.json` → removed leading `<sop>`
  line to match `adamw-1500` (`*.sop-bak-*` backups)

Restart the affected backends after editing any of these (vLLM reads them at
startup only).

**Recommended client params for the 3 unstable checkpoints:**
`temperature=0.7, top_p=0.95, frequency_penalty=0.7` (the router adds the last
one automatically).

### Confirmed: the cause was an OLD convert_to_hf.py (2026-06-27)

Diffing the convert script the user actually used against the repo's current
`vllm/convert_to_hf.py` confirmed the root cause is a **stale converter**:

- Old script's `create_hf_config` did **not** emit `qk_rms_clip / qk_rms_limit /
  swiglu_limit` → converted `config.json` lacked the training clamps → garble.
- Old script also force-prepended `<sop>` to the chat template
  (`ensure_chat_template_has_bos`); the current script copies the template as-is.

The current repo `convert_to_hf.py` already fixes both (reads
`ma.get("qk_rms_clip"...)` from metadata, copies template unchanged). Training
`metadata.json` for these runs carries `qk_rms_clip=True, qk_rms_limit=3.0,
swiglu_limit=10.0`, so re-converting with the current script yields a correct
config automatically.

**Weights do NOT need re-conversion:** the weight-conversion logic
(`convert_state_dict`, gate row-norm, `save_sharded`) is byte-identical between
the two script versions; only config/template generation differed. The in-place
`config.json` patch applied to the 3 dirs is therefore equivalent to a re-convert.
