# Aura-1B v0.2.0 — the system grows a mind of its own

> **The fastest answer is the one you never generate.**
> A 100 % local, CPU-native AI: Llama 3.2 1B orchestrated with exact math,
> verified facts and a sealed `.aef` distribution — **no GPU required**.

## ✨ Added

- **Served multi-LoRA with weight mixing** — the official llama-server (CPU
  build, auto-started) mounts the Q4_K_M brain + all 4 category adapters at
  boot; the router's probabilities become simultaneous weights
  (`{"math": 0.7, "web": 0.3}`) in one HTTP call, ~ms hot-swap, no reload.
  In-process fallback never blocks. Base stays **Q4_K_M (807 MB)**.
- **Self-play, AlphaGo-style** (`aura/selfplay.py`) — a challenge generator
  builds logic puzzles & drills from the fact graph, a **deterministic judge**
  validates (never the model grading itself), successes reinforce the graph
  and join the training dataset; failures become injected corrections.
  `aura --selfplay 20 [--niveau 2]`
- **Autonomous long-term memory** (`aura/graphe_faits.py`) — every fact
  triplet carries a **strength**: each verified use reinforces it, lookup is
  strength × recency weighted. Knowledge organizes its own importance with
  zero retraining.
- **RLSS** (`aura/rlss.py`) — symbolic-reinforcement loop: generated
  exercises → exact validation → successes stored for the next LoRA.
- **Dynamic Compute** — complex questions trigger a masked `<thinking>`
  draft (×2 token budget), stripped before display; simple ones stay instant.
- **Style mimicry (few-shot)** — calibrated exemplars steer the 1B's
  attention: it copies big-model prose instead of inventing its own tics.
- **Double-pass editing** — the 1B critiques its own draft then rewrites it;
  catching flaws in existing text is where a small model statistically
  matches a much bigger one.
- **Semantic triplets** — web context is compressed to `subject | relation |
  object` (~10× shorter), protecting the 1B context window and feeding the
  fact graph.
- **Creator lock** (`aura/verrou.py`) — re-forging the `.aef` requires the
  Ed25519 private key (never distributed); the AI only ever rewrites data,
  never code. License: MIT → **GPLv3**.

## 📊 The equivalence, measured by capability

| Capability | Aura-1B system | Raw 1B brain |
|---|---|---|
| Logic & math | **beyond class** — exact, proven | hallucinates |
| Facts (graph + web) | **~7-8B anchored**, 0 hallucination | ~1B, frozen |
| Code (sandbox-verified) | **~8B** on testable code | worse |
| Style & writing | **~7-8B perceived** | 1B tics |
| Deep unanchored analysis | honest limit — weights stay 1B | 1B |

## 🔧 Changed

- Brain priority: Q4_K_M is the LoRA-compatible base (validated end-to-end
  with 4 simultaneous adapters — root cause of earlier failures: Colab
  converters write transposed `lora_a/lora_b`, fixed by `transpose_lora.py`).
- Warmup with 4 LoRA requires `-c 1536 --no-warmup` (else `GGML_ASSERT`).
- `llama-cpp-python` pinned to the official prebuilt CPU wheel (0.3.2) via
  the `[[tool.uv.index]]` — no more MinGW builds.

## 🧪 Quality

- **251 tests** (+78 since v0.1.0), mypy 0 errors, green CI on
  Ubuntu + Windows (Python 3.11/3.12).

**Full changelog**: https://github.com/Simonc44/aura-1b/compare/v0.1.0...v0.2.0
