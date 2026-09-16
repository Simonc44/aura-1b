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

## Tests

```bash
uv run pytest -q        # 44 tests: level-0, lexicon, masked CoT, multi-pass, routing, Llama, fallback
```

## License

MIT
