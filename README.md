# 🧬 Aura-1B — Hybrid Neuro-Symbolic AI

**Aura-1B** is an autonomous, 100% local AI architecture that splits language, exact math and factual memory into specialized modules — instead of asking a single small model to do everything (and hallucinate when it can't).

## Architecture

```text
              [ QUESTION ]
                    │
                    ▼
NIVEAU 0 — INSTANTANÉ (< 5 ms)          ← CPU-native core
  ├─ Exact math (safe AST eval)         « square of 12 » → 144 in 0.3 ms
  └─ Semantic cache (cosine ≥ 0.85)     repeated question → 1.3 ms
     │ (otherwise, ~60-70% of the time)
     ▼
NIVEAU 1 — EXPERTS
  ├─ 📐 PGS (genetic programming)       exact formula, zero error
  └─ 🌐 DuckDuckGo (real-time facts)    the internet = external hard drive
     │
     ▼
NIVEAU 2 — LLAMA 3.2 1B (Q4_K_M)        ← only what remains
  instruction-tuned FR+EN, 9-14 tok/s on CPU
```

### The pillars

| Pillar | Tech | Why |
|---|---|---|
| **Level 0** | Exact math via safe AST evaluator (never `eval()`) + semantic answer cache (TF-IDF char n-grams, cosine ≥ 0.85) | **The fastest answer is the one you never generate.** Measured: ×3700 on direct math, ×7500 on repeats. Contextual questions ("my name...") are never cached. |
| **Router** | TF-IDF + Logistic Regression (150 examples) + cosine prototypes | Classifies any question (typos included) — no hardcoded if/else. |
| **Brain** | **Llama 3.2 1B Instruct (Q4_K_M, 807 Mo)** via llama.cpp | Good French AND English out of the box. Chosen over Q6_K by benchmark (3/3 vs 1/3). |
| **Symbolic expert** | Genetic programming (gplearn), protected `pow` | Finds the **exact** law (`mul(mul(X0,X0),X0)` for X³) — zero hallucination, verifiable. |
| **Web memory** | DuckDuckGo RAG (`ddgs`), no API key | Model parameters stay free for language & logic; facts stay current. |

Plus **auto-improvement**: wrong answers recorded via `enregistrer_correction()` are injected into future prompts — the same mistake is never made twice.

### Rich mode (writing quality)

Open questions (≥ 9 words or *explique/analyse/compare...*) trigger a 3-step pipeline:
1. **Enriched search** — dual query (facts + analysis) + domain **lexicon** injected as keywords.
2. **Masked Chain-of-Thought** — planning inside `<thinking>` tags, stripped by regex; the user only sees the polished answer.
3. **Multi-pass generation** — pass 1: detailed plan + 5 logical connectors; pass 2: styled writing following that plan.

Cost: ~2× latency — reserved for questions that deserve it. Measured on this machine: fact 0.3 s, math 0.0 s, rich essay 66 s.

## Hardware profile (auto-tuning)

Aura inspects the machine it runs on (CPU model, cores, RAM, AVX2, disk) and derives optimal settings — no manual tuning:

```bash
python -m aura.profil_materiel
```

| Machine detected | Settings derived |
|---|---|
| RAM ≥ 6 Go | ctx 1536, batch 768, threads = all logical cores |
| RAM < 6 Go | ctx 1024, batch 512 (guaranteed fit) |

The forge bakes these into the `.aef`; the kernel applies them at boot, and refuses to load a model that cannot fit in RAM (OOM guard *before* the crash).

### GPU support

`n_gpu_layers` is derived automatically at load time:

- llama.cpp build with CUDA/Vulkan + discrete GPU (NVIDIA/AMD/Intel Arc) detected → **full offload** (`-1`), big speedup.
- CPU-only build or integrated graphics (like the reference machine's Intel UHD) → **0** (pure CPU, which is optimal here: 13.4 tok/s measured).
- Force CPU: set `AURA_GPU=0`.

## The `.aef` format + Aura.exe (encrypted distribution)

The whole system ships as **one sealed binary file**:

```text
aura_system.aef (770 Mo, monolithic)
├── HEADER (148 B)  magic "AURA" · version · build · CPU config · 3× SHA-256
├── CONFIG (LZMA)   auto-tuned settings compiled for the target machine
├── CODE (ZIP)      the entire aura/ python package (verified by hash)
└── WEIGHTS (raw)   the GGUF byte-for-byte (loaded directly, zero-copy)
```

**Integrity**: one altered byte anywhere → boot refused (SHA-256 over config, code and weights — tested).

**Confidential publication** — distribute the system without exposing sources or weights:

```bash
uv run python scripts/construire_exe.py            # -> dist/Aura.exe (~8.6 Mo)
uv run python scripts/forger_aef.py --chiffrer     # -> aura_system.aef.enc (AES-256-GCM)
```

On the user machine (ask them for the secret, or `set AURA_SECRET=...`):

```bash
Aura.exe aura_system.aef.enc "your question"       # decrypts, verifies, extracts, answers
Aura.exe --cache "your question"                    # next runs: instant start
```

Protection layers:
- **AES-256-GCM** (AEAD): without the secret, the payload is indistinguishable from random noise; any tampering breaks decryption.
- Key derived via **PBKDF2-HMAC-SHA256, 600 000 iterations** — never stored in the file.
- Aura.exe embeds **compiled bytecode only** and delegates execution to the local python after cryptographic verification (shared cache via `AURA_CACHE`).
- Honest note: no binary is unbreakable — this blocks passive copying, not a determined reverse-engineer with the secret.

## Install

```bash
uv sync                                        # core deps (llama.cpp included)
python scripts/telecharger_llama.py            # downloads the 807 Mo GGUF
```

## Run

```bash
# Built-in demo: discover Y = X³ + fetch physics news
uv run python -m aura --demo

# Free-form question
uv run python -m aura "Quelle est la capitale de la France ?"

# Numeric data (JSON) → exact formula
uv run python -m aura "model this" --x "[[1],[2],[3],[4]]" --y "[1,8,27,64]"

# Interactive chat with conversation memory
uv run python -m aura --chat
```

### From the sealed file

```bash
python -m aura.kernel aura_system.aef "question"    # verify + extract + answer
python -m aura.kernel --cache "question"             # cache already filled
```

## Python API

```python
from aura import Aura1B

ia = Aura1B()
report = ia.executer_detaille(
    "Quelles sont les dernieres decouvertes en physique ?",
    X=[[0.0], [0.25], [0.5], [0.75], [1.0]],
    y=[0.0, 0.015625, 0.125, 0.421875, 1.0],
)
print(report["formule"])       # mul(mul(X0, X0), X0)
print(report["erreur_pgs"])    # 0.0
print(report["reponse"])       # grounded synthesis (Llama 3.2)
```

Without the GGUF the pipeline degrades honestly (web facts + exact formula, clearly labeled) — it never fabricates a "generated" answer.

## Benchmark (CPU-only reference machine)

| Brain | Speed | French | Load |
|---|---|---|---|
| **Llama 3.2 1B Q4_K_M** (current) | **13.4 tok/s** | ✅ instruction-tuned | 3.4 s |
| Qwen 2.5 1.5B (removed) | 12.1 tok/s | ✅ | — |
| Mamba-790M / RWKV-430M (removed) | 1.5-1.8 tok/s | ❌ base models | 30-400 s |

System-level quiz vs Qwen 2.5 1.5B: **Aura 5/5 vs 4/5** (exact math, real-time facts, memory) — the organization beats the bigger brain on verifiable questions.

## Tests

```bash
uv run pytest -q        # 50 tests: level-0, lexicon, masked CoT, multi-pass, routing, .aef crypto, encryption
```

## License

MIT
