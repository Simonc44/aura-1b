# 🧬 Aura-1B — Hybrid Neuro-Symbolic AI

[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)

**Aura-1B** is an autonomous, 100% local AI architecture that splits language, exact math and factual memory into specialized modules — instead of asking a single small model to do everything (and hallucinate when it can't).

## Architecture

```text
              [ QUESTION ]
                    │
                    ▼
NIVEAU 0 — INSTANTANÉ (< 5 ms)          ← CPU-native core
  ├─ Exact math (safe AST eval)         « square of 12 » → 144 in 0.3 ms
  ├─ PAL (dates · units · %)            « 5 miles en km » · « dans 45 jours »
  └─ Semantic cache (cosine ≥ 0.85)     repeated question → 1.3 ms
     │ (otherwise, ~60-70% of the time)
     ▼
NIVEAU 1 — EXPERTS
  ├─ 🧩 Program-of-Thoughts (puzzles)    1B writes steps, AST guarantees them
  ├─ ⚖️ Logic solver (mini-SAT)          knights/knaves, attributions — pure Python
  ├─ 💻 PoT-code (sandboxed)             1B writes code + asserts → verified or refused
  ├─ 📐 PGS (genetic programming)        exact formula, zero error
  ├─ 📚 Fact graph (MiniRAG-lite)        triplets fed by verified answers only
  └─ 🌐 DuckDuckGo (real-time facts)     the internet = external hard drive
     │
     ▼
NIVEAU 2 — LLAMA 3.2 1B (Q4_K_M)        ← only what remains
  instruction-tuned FR+EN, 9-14 tok/s on CPU
  └─ 🔍 CRITIC check                     short factual answers re-anchored on web proof
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

### The verified-intelligence layer (why a 1B stops hallucinating)

Every mechanism below shares one rule: **the small model proposes, the program proves** — anything unproven is either verified or refused, never invented.

| Mechanism | What it fixes | How |
|---|---|---|
| **CRITIC check** | factual LLM answers from memory (the classic 1B slip) | short factual answers are re-generated anchored on web proof before delivery; the verified answer **replaces** the old one in the semantic cache (reconsolidation) |
| **Fact graph** (`graphe_faits.py`) | offline long-tail knowledge | triplets `(subject, relation, object)` in JSONL, multi-word coverage lookup (natural multi-hop). Fed **only** from web-verified answers + your `knowledge.jsonl` (223 entries) — it can never contain a hallucination |
| **PAL** (`pal.py`) | dates, units, compound percentages | « 15% de 200 plus 30% de 100 » → 60, « 100 f en c », « combien de jours jusqu'au 25 décembre » — deterministic, < 1 ms |
| **Program-of-Thoughts** (`raisonneur.py`) | word puzzles with numbers | the 1B writes `ETAPE 1/ETAPE 2` lines, the safe AST evaluates each step; a puzzle like « Léo a 4 ans, Marie le double, Paul 3 de plus » → **11, every step verified** |
| **Logic solver** (`logique.py`) | number-free logic (knights & knaves, attributions) | the 1B formalizes `ENTITES/DOMAINE/CONDITION`, a pure-Python mini-SAT deduces exactly; invented constraints are detected → graceful fallback |
| **PoT-code** (`potcode.py`) | broken code generation | the 1B writes a function + asserts, a sandbox (restricted builtins + instruction budget via `settrace`) executes everything; a failing assert = the code is never delivered |
| **StoryWriter-lite** | long-form depth | rich mode writes each section of the plan separately, with memory of previous sections — short generations stay inside the 1B's comfort zone |

### Rich mode (writing quality)

Open questions (≥ 9 words or *explique/analyse/compare...*) trigger a 3-step pipeline:
1. **Enriched search** — dual query (facts + analysis) + domain **lexicon** injected as keywords.
2. **Masked Chain-of-Thought** — planning inside `<thinking>` tags, stripped by regex; the user only sees the polished answer.
3. **Multi-pass generation** — pass 1: detailed plan + 5 logical connectors; pass 2: styled writing following that plan.

Cost: ~2× latency — reserved for questions that deserve it. Measured on this machine: fact 0.3 s, math 0.0 s, rich essay 66 s.

## Teaching your AI (it learns from you)

Aura improves from feedback with no retraining. Three levers, from easiest to deepest:

1. **Correct a wrong answer** (auto-improvement):

   ```python
   from aura import autoamelioration
   autoamelioration.enregistrer_correction(
       "quel est le carre de 5", "20",   # its wrong answer
       "25",                             # the right one
       "math")
   ```

   The correction is stored (`.cache_corrections.jsonl`) and injected into future prompts: the same mistake is never made twice.

2. **Feed the fact graph** — anything web-verified is learned automatically, and you can add curated facts:

   ```python
   from aura import graphe_faits
   graphe_faits.ajouter("Canberra", "est la capitale de", "l'Australie",
                        source="manuel")
   ```

   Or in bulk from a knowledge file: `python scripts/nourrir_graphe.py knowledge.jsonl`
   (skips identity entries, tags health/law as *general information*).

3. **Rehearse** (spaced repetition): the semantic cache remembers verified answers —
   ask again tomorrow and the answer comes back in ~1 ms. Wrong entries are replaced
   by reconsolidation (`mettre_a_jour`), never duplicated.

Roadmap: LoRA fine-tuning (reasoning) + MEMIT fact editing on Colab — the two
levers that raise the *weights* themselves while keeping the 807 Mo size and speed.

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

### Windows — one command, then `aura` in PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/installer.ps1
```

The installer copies the package to `%LOCALAPPDATA%\Aura`, creates the venv,
installs the dependencies, downloads the brain (807 MB, once), and registers a
`aura` function in your PowerShell profile. Then, in any **new** PowerShell:

```powershell
aura "quel est le carre de 7"        # -> 49
aura                                  # interactive chat
aura --verifier "question"            # force full crypto verification
```

It prefers the sealed `aura_system.aef.enc` (AES-256-GCM + Ed25519 signature),
falls back to the plain `.aef`, and reads `secret_aef.txt` automatically
(provide it at install time via the `AURA_SECRET_INSTALL` environment variable).

### Development install

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
uv run pytest -q        # 173 tests
uv run mypy aura/       # 0 error (strict-ish config in pyproject.toml)
```

CI (GitHub Actions) runs the full suite on Ubuntu + Windows (Python 3.11/3.12,
no GGUF needed — the brain tests skip themselves), re-validates the whole
`.aef` chain (forge → Ed25519 signature → boot → tamper detection) on Linux,
and syntax-checks the PowerShell installer.

## License

MIT
