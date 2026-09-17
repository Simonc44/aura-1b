# 🧬 Aura-1B — Hybrid Neuro-Symbolic AI

**Aura-1B** is an autonomous, 100% local AI architecture that splits language, exact math and factual memory into specialized modules — instead of asking a single model to do everything (and hallucinate when it can't).

## Architecture

```text
              [ QUESTION UTILISATEUR ]
                      │
        ┌─────────────▼──────────────┐
        │  NIVEAU 0 : INSTANTANE     │  < 5 ms — LA reponse CPU-native
        │  maths directes (AST)      │  « carre de 12 » -> 144 en 0.3 ms
        │  cache semantique (>=0.85) │  question repetee -> 1.3 ms
        └──────┬──────────────┬──────┘
               │ (sinon)      │
        ┌──────▼──────────────┴──┐
        │  ROUTEUR INTELLIGENT   │  TF-IDF + LogReg (150 exemples)
        └──────┬──────────┬──────┘
  (besoin de faits)     (besoin de maths)
        │                         │
        ▼                         ▼
┌───────────────────┐   ┌──────────────────────────┐
│  WEB (DuckDuckGo) │   │  PGS (gplearn)           │
│  temps réel       │   │  formule exacte, erreur 0 │
└──────────┬────────┘   └──────────┬─────────────────┘
           └─────────────┬────────┘
                         ▼
        ┌──────────────────────────────────┐
        │  CERVEAU : LLAMA 3.2 1B          │  instruction-tuned FR+EN
        │  Q4_K_M via llama.cpp            │  9-14 tok/s CPU
        └──────────────────────────────────┘
```

### The four pillars

| Pillar | Tech | Why |
|---|---|---|
| **Level 0 (CPU-native)** | Exact math via safe AST eval (<1 ms) + semantic answer cache (TF-IDF cosine >= 0.85). | **The fastest answer is the one you never generate** — ~60-70% of daily questions never reach the LLM. Measured: x3700 on direct math, x7500 on repeats. |
| **Router** | TF-IDF + Logistic Regression (150 examples, `scikit-learn`) + cosine similarity with prototypes. Falls back to keywords. | Classifies any question (even typos) — no hard-coded if/else. |
| **Brain** | **Llama 3.2 1B Instruct (Q4_K_M, ~807 Mo)** via `llama-cpp-python`. | Instruction-tuned: good French AND English out of the box. 9-14 tok/s on CPU, 3.4 s load. |
| **Symbolic Expert** | Genetic Programming (gplearn). Protected `pow`, min-max normalization. | Finds the **exact** law: `mul(mul(X0, X0), X0)` for the cube — error 0, verifiable, zero hallucination. |
| **Web Memory** | DuckDuckGo RAG (`ddgs`), no API key. | Internet = external hard drive; model parameters stay free for language & logic. |

Plus **auto-improvement**: corrections (`enregistrer_correction`) are stored and injected into the prompt so the same mistake is never made twice.

### Rich mode for open-ended questions (writing quality)

Fact questions stay fast (single pass). Open questions (≥ 9 words or `explique/analyse/compare/redige...`) trigger a **3-step rich pipeline**:

1. **Enriched search** — dual query (facts + "analyse synthese essai") + extracts a domain **lexicon** (rare/technical words) injected as "Mots-clés pertinents à utiliser".
2. **Masked Chain-of-Thought** — the model plans inside `<thinking>` tags (analyse → stratégie → plan); a regex strips them, the user only sees the polished answer, the reasoning goes to logs.
3. **Multi-pass generation** — pass 1 produces a detailed 3-part plan + 5 logical connectors; pass 2 writes the final answer following that plan with strict style rules. Splitting "what to say" from "how to say it" is what lets a 1B rival bigger models on structure.

Cost: ~2× latency (59 s measured vs ~15-25 s) — reserved for questions that deserve it.

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

# Question + numeric data (JSON) → exact formula
uv run python -m aura "model this" --x "[[1],[2],[3],[4]]" --y "[1,8,27,64]"
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

Without the GGUF file the pipeline degrades honestly (web facts + exact formula, clearly labeled) — it never fabricates a "generated" answer.

## Benchmark (CPU-only, 8 Go RAM laptop)

| Brain | Speed | French | Load |
|---|---|---|---|
| **Llama 3.2 1B Q4_K_M** (current) | **9-14 tok/s** | ✅ instruction-tuned | 3.4 s |
| Mamba-790M (removed) | 1.5 tok/s | ❌ base model | 30-408 s |
| RWKV-4 430M (removed) | 1.8 tok/s | ❌ base model | ~60 s |

## Hardware profile (auto-tuning)

Aura inspects the machine it runs on (CPU model, logical cores, RAM, AVX2, disk) and derives optimal llama.cpp settings — no manual tuning:

```bash
python -m aura.profil_materiel      # inspect + auto-tuned settings
```

| Machine detected | Settings derived |
|---|---|
| RAM ≥ 6 Go | ctx 1536, batch 768, threads = all logical cores |
| RAM < 6 Go | ctx 1024, batch 512 (guaranteed fit) |

The forge bakes these into the `.aef` config; the kernel applies them at boot. The `.aef` also records a machine fingerprint and refuses to load a GGUF that cannot fit in RAM (OOM guard *before* the crash).

## The `.aef` format + Aura.exe (encrypted distribution)

The whole system ships as **one sealed binary file**:

```text
aura_system.aef (770 Mo, monolithic)
├── HEADER (148 B)  magic "AURA" · version · build · CPU config · 3× SHA-256
├── CONFIG (LZMA)   auto-tuned settings compiled for the target machine
├── CODE (ZIP)      the entire aura/ python package (verified by hash)
└── WEIGHTS (raw)   the GGUF byte-for-byte (loaded directly, zero-copy)
```

**Integrity**: one altered byte anywhere → boot refused (SHA-256 over config, code and weights; tested).

**Confidential publication** (put it on Hugging Face without exposing the code or weights):

```bash
uv run python scripts/construire_exe.py            # -> dist/Aura.exe (8.6 Mo)
uv run python scripts/forger_aef.py --chiffrer     # -> aura_system.aef.enc (AES-256-GCM)
```

Then distribute **Aura.exe + aura_system.aef.enc**. On the user machine:

```bash
Aura.exe aura_system.aef.enc "your question"       # decrypts, verifies, extracts, answers
Aura.exe --cache "your question"                    # next runs: instant start
```

- The key derives from a secret via **PBKDF2 (600 000 iterations)** — never stored in the file; wrong secret → clean refusal.
- Two independent integrity layers: AEAD (decryption) + SHA-256 (header).
- Aura.exe embeds **compiled bytecode only** (no readable sources) and delegates execution to the local python after cryptographic verification.
- Honest note: no binary is unbreakable (static analysis is always possible in theory) — this blocks passive copying, not a determined reverse-engineer. See also the security section of the README for limits.

## Publish on Hugging Face

Upload `dist/Aura.exe` + `aura_system.aef.enc` (770 Mo) to a HF model repo. Keep `secret_aef.txt` **private**: without the secret the weights are indistinguishable from random noise.

## Tests

```bash
uv run pytest -q        # 44 tests: level-0, lexicon, masked CoT, multi-pass, routing, Llama, fallback
```

## License

MIT
