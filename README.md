# 🧬 Aura-1B — Hybrid Neuro-Symbolic AI

**Aura-1B** is an asymmetric AI architecture that splits language, exact math and factual memory into three specialized modules — instead of asking a single Transformer to do everything (and hallucinate when it can't).

## Architecture

```text
              [ QUESTION UTILISATEUR ]
                      │
        ┌─────────────▼──────────────┐
        │  ROUTEUR INTELLIGENT       │  TF-IDF + LogReg (150 exemples)
        │  classifie l'intention     │  + cosinus prototypique
        └──────┬──────────────┬──────┘
  (besoin de faits)     (besoin de maths)
        │                         │
        ▼                         ▼
┌───────────────────┐   ┌──────────────────────────┐
│  WEB (DuckDuckGo) │   │  PGS (gplearn)           │
│  0 hallucination  │   │  formule exacte, erreur 0 │
└──────────┬────────┘   └──────────┬─────────────────┘
           └─────────────┬────────┘
                         ▼
        ┌──────────────────────────────────┐
        │  SYNTHESE STRUCTUREE             │  Prompt 3 sections claires
        │  (Mamba-130M ou Ollama 1.5B)     │  web + formule + question
        └──────────────────────────────────┘
```

### The four pillars

| Pillar | Tech | Why |
|---|---|---|
| **Router** | TF-IDF + Logistic Regression (150 examples, `scikit-learn`) + cosine similarity with prototype sentences. Falls back to keywords if sklearn is missing. | Classifies any question (even typos) — no hard-coded if/else. |
| **Orchestrator** | Ollama 1.5B instruct (default) or Mamba-130M (`--cerveau mamba`). | Understands, routes, writes the final answer. |
| **Symbolic Expert** | Genetic Programming (gplearn). Protected `pow`, min-max normalization. | Finds the **exact** law: `mul(mul(X0, X0), X0)` for the cube — error 0, verifiable, zero hallucination. |
| **Web Memory** | DuckDuckGo RAG (`ddgs`), no API key. | Internet = external hard drive; model parameters stay free for language & logic. |

## Orchestrator brains

```bash
uv run python -m aura --demo                      # Ollama (default, fastest on CPU)
uv run python -m aura --demo --cerveau mamba     # real Mamba-1.4B SSM
```

**Honest benchmark (CPU-only, laptop 8 Go RAM):**

| Cerveau | Vitesse | Qualité | Besoin |
|---|---|---|---|
| Ollama 1.5B instruct | ~25 tok/s | bon en français | rien |
| Mamba-130M (réel, CPU) | **~12 tok/s** | correct, léger | `uv sync --extra mamba` |
| Mamba-1.4B | ❌ pagefile | bon | 16 Go RAM |

Les kernels CUDA (`mamba_ssm`, `causal_conv1d`) accéléreraient Mamba x3-x5 — mais nécessitent une GPU NVIDIA. Sur CPU, Mamba shine pour sa mémoire fixe (inputs de 1000 tokens = même RAM que 10 tokens).

Fallback : torch absent, OOM ou modèle inaccessible → Ollama → template honnête (aucune hallucination).

## Install

```bash
uv sync                # core (gplearn, ddgs, requests)
uv sync --extra mamba  # + torch/transformers for the Mamba brain (~2 GB)
```

## Run

```bash
# Built-in demo: discover Y = X³ + fetch physics news (works even without Ollama)
uv run python -m aura --demo

# Free-form question with numeric data (JSON)
uv run python -m aura "model this" --x "[[1],[2],[3],[4]]" --y "[1,8,27,64]"

# Choose your local brain
uv run python -m aura --demo --modele qwen2.5:1.5b-instruct
```

## Python API

```python
from aura import Aura1B

ia = Aura1B()  # Ollama defaults
report = ia.executer_detaille(
    "Quelles sont les dernieres decouvertes en physique ?",
    X=[[0.0], [0.25], [0.5], [0.75], [1.0]],
    y=[0.0, 0.015625, 0.125, 0.421875, 1.0],
)
print(report["formule"])       # mul(mul(X0, X0), X0)
print(report["erreur_pgs"])    # 0.0
print(report["reponse"])       # grounded synthesis
```

Without a running LLM the pipeline degrades honestly (web facts + exact formula, clearly labeled) — it never fabricates a "generated" answer.

## Tests

```bash
uv run pytest -q        # 8 tests: law discovery, protected pow, routing, fallback
```

## Roadmap

- [ ] Swap the orchestrator slot to a real **Mamba-1B** SSM (same interface)
- [ ] Cache layer for web results (respectful rate limiting)
- [ ] Multi-variable laws (X already supports n columns)
- [ ] Unit-aware symbolic regression (quantities, not just floats)

## License

MIT
