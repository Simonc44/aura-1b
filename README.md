# 🧬 Aura-1B — Hybrid Neuro-Symbolic AI

**Aura-1B** is an asymmetric AI architecture that splits language, exact math and factual memory into three specialized modules — instead of asking a single Transformer to do everything (and hallucinate when it can't).

## Architecture

```text
              [ USER QUESTION ]
                      │
        ┌─────────────▼──────────────┐
        │  1. ORCHESTRATOR           │  Local LLM (Mamba slot)
        │     understand · route     │
        └──────┬──────────────┬──────┘
 (needs facts)               (needs exact math)
        │                            │
        ▼                            ▼
┌───────────────────────┐   ┌──────────────────────────┐
│  2. WEB MEMORY (RAG)  │   │  3. SYMBOLIC EXPERT (PGS)│
│  DuckDuckGo, no key   │   │  gplearn: add sub mul    │
│  always up to date    │   │  div pow → exact formula │
└───────────┬───────────┘   └────────────┬─────────────┘
            └──────────────┬─────────────┘
                           ▼
        ┌──────────────────────────────────┐
        │  4. FINAL ANSWER (Orchestrator)  │
        │  grounded synthesis, FR/EN       │
        └──────────────────────────────────┘
```

### The three pillars

| Pillar | Tech | Why |
|---|---|---|
| **Orchestrator** | Any local OpenAI-compatible LLM (Ollama by default). *Mamba brain:* the real **`state-spaces/mamba-1.4b-hf`** SSM (fixed-size memory) is integrated with automatic fallback to Ollama — `--cerveau mamba`. | Understands, routes, writes. |
| **Symbolic Expert (~50M eq.)** | Genetic Programming (gplearn) evolving equation trees with `add/sub/mul/div/pow`. Protected `pow` (clipped exponent, no inf/nan). Min-max normalization makes laws like Y = X³ representable. | Finds the **exact** law: `mul(mul(X0, X0), X0)` for the cube — error 0, verifiable, zero hallucination. |
| **Web Memory** | DuckDuckGo RAG (`ddgs`), no API key. Triggered only when the question needs external facts. | Internet = external hard drive; model parameters stay free for language & logic. |

## Orchestrator brains

```bash
uv run python -m aura --demo                      # Ollama (default, fastest on CPU)
uv run python -m aura --demo --cerveau mamba     # real Mamba-1.4B SSM
```

**Honest benchmark (CPU-only, RTX-less laptop):** Mamba-1.4B answers correctly but runs at ~2 tok/s in pure PyTorch (the optimized `mamba_ssm`/`causal_conv1d` kernels are CUDA-only). Ollama's 1.5B instruct stays ~10x faster. Mamba also needs the `mamba` extra (`uv sync --extra mamba`, ~2 GB torch). Fallback is automatic: torch missing, model unreachable or OOM → Ollama → grounded template. Mamba becomes the right choice on CUDA hardware or for very long inputs (fixed-size memory).

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
