<div align="center">

<img src="assets/aura.jpg" alt="Aura-1B" width="420"/>

**The fastest answer is the one you never generate.**

A 100% local, CPU-native AI system that splits language, exact math and factual
memory into specialized modules — instead of asking one small model to do
everything (and hallucinate when it can't).

[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Static Badge](https://img.shields.io/badge/mypy-checked-brightgreen)
![GitHub last commit](https://img.shields.io/github/last-commit/Simonc44/aura-1b/main?label=last%20commit)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Simonc44/aura-1b/pulls)
![GitHub stars](https://img.shields.io/github/stars/Simonc44/aura-1b?style=social)

![GPU](https://img.shields.io/badge/GPU-not%20required-success)
![Brain](https://img.shields.io/badge/brain-807%20MB%20Q4__K__M-orange)
![Engine](https://img.shields.io/badge/powered%20by-llama.cpp-2CA5A0)
![Format](https://img.shields.io/badge/sealed%20format-.aef-blueviolet)

[Why](#-why-aura-1b) •
[Quickstart](#-quickstart) •
[Architecture](#-architecture) •
[Teach it](#-teaching-your-ai) •
[.aef format](#-the-aef-format--encrypted-distribution) •
[Benchmarks](#-benchmarks) •
[Roadmap](#-roadmap)

</div>

---

## 💡 Why Aura-1B?

On a machine without GPU, a 1B LLM is slow (~14 tok/s) and hallucinates.
Aura turns that weakness into a system design rule:

> **The small model proposes. The program proves.**
> Anything unproven is verified against a source or refused — never invented.

The result: instant answers (0–5 ms) on 60–70 % of everyday questions, exact
math and dates, real-time facts, and a 1B brain used only where it matters.

| | Aura-1B | Typical 8B on the same PC |
|---|---|---|
| Everyday question | **0.0–2.3 s** | 30–60 s |
| Math / dates / units | **exact, verifiable** | hallucinates off-stats |
| Facts | **live web + proof check** | frozen at training cutoff |
| RAM footprint | **~1 GB** | ~4.5 GB |
| Long-tail knowledge & deep analysis | ⚠️ system's limit | better — see [honest limits](#-honest-limits) |

## 🚀 Quickstart

### One command (Windows, PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/installer.ps1
```

Then, in a **new** PowerShell window:

```powershell
aura "quel est le carre de 7"     # -> 49
aura                              # interactive chat
```

The installer copies the package to `%LOCALAPPDATA%\Aura`, creates the venv,
downloads the brain (807 MB, once) and registers the `aura` command.

### Development install

```bash
uv sync                                        # deps (llama.cpp included)
python scripts/telecharger_llama.py            # 807 MB GGUF, once
uv run python -m aura --demo                   # demo: X³ formula + physics news
uv run python -m aura "quelle est la capitale de la France ?"
uv run python -m aura --chat                   # chat with memory
```

## 🏗 Architecture

<img src="assets/architecture.png" alt="Aura-1B architecture diagram" width="100%"/>

### The pillars

| Pillar | Tech | Why |
|---|---|---|
| **Level 0** | Exact math via safe AST evaluator (never `eval()`) + semantic cache (TF-IDF char n-grams, cosine ≥ 0.85) | **The fastest answer is the one you never generate.** Measured: ×3700 on direct math, ×7500 on repeats. Contextual questions ("my name…") are never cached. |
| **Router** | TF-IDF + Logistic Regression (150 examples) + cosine prototypes + confidence threshold | Classifies any question (typos included) — no hardcoded if/else. Low-confidence routing refuses to guess. |
| **Brain** | **Llama 3.2 1B Instruct (Q4_K_M, 807 MB)** via llama.cpp | Good French AND English out of the box. Brain-swappable: `AURA_GGUF=MINICPM` loads MiniCPM5-1B (OpenBMB) — measured **6/7 @ 3.0 s vs 3/7 @ 10.2 s** on the same quiz, so Llama stays the default. |
| **Symbolic expert** | Genetic programming (gplearn), protected `pow` | Finds the **exact** law (`mul(mul(X0,X0),X0)` for X³) — zero hallucination, verifiable. |
| **Web memory** | DuckDuckGo RAG (`ddgs`), no API key | Model parameters stay free for language & logic; facts stay current. |

### The verified-intelligence layer (why a 1B stops hallucinating)

| Mechanism | What it fixes | How |
|---|---|---|
| **CRITIC check** | factual LLM answers from memory | short factual answers are re-generated anchored on web proof before delivery; the verified answer **replaces** the old one in the cache (reconsolidation) |
| **Fact graph** (`graphe_faits.py`) | offline long-tail knowledge | triplets `(subject, relation, object)` in JSONL, multi-word coverage lookup (natural multi-hop). Fed **only** from web-verified answers + your `knowledge.jsonl` (223 entries) — it can never contain a hallucination |
| **PAL** (`pal.py`) | dates, units, compound percentages | « 15% de 200 plus 30% de 100 » → 60, « 100 f en c », « combien de jours jusqu'au 25 décembre » — deterministic, < 1 ms |
| **Program-of-Thoughts** (`raisonneur.py`) | word puzzles with numbers | the 1B writes `ETAPE 1/ETAPE 2` lines, the safe AST evaluates each step: « Léo a 4 ans, Marie le double, Paul 3 de plus » → **11, every step verified** |
| **Logic solver** (`logique.py`) | number-free logic (knights & knaves, attributions) | the 1B formalizes `ENTITES/DOMAINE/CONDITION`, a pure-Python mini-SAT deduces exactly; invented constraints are detected → graceful fallback |
| **PoT-code** (`potcode.py`) | broken code generation | the 1B writes a function + asserts, a sandbox (restricted builtins + instruction budget via `settrace`) executes everything; a failing assert = the code is never delivered |
| **Auto-improvement** | repeated mistakes | wrong answers recorded via `enregistrer_correction()` are injected into future prompts — the same mistake is never made twice |

### Rich mode (writing quality)

Open questions (≥ 9 words or *explique/analyse/compare…*) trigger a 3-step pipeline:
1. **Enriched search** — dual query (facts + analysis) + domain **lexicon** injected as keywords.
2. **Masked Chain-of-Thought** — planning inside `<thinking>` tags, stripped by regex; the user only sees the polished answer.
3. **Sectioned generation** — each part of the GBNF-constrained plan is written separately, with memory of previous sections (StoryWriter-lite).

Cost: ~2× latency — reserved for questions that deserve it. Measured on the
reference machine: fact 0.3 s, math 0.0 s, rich essay 66 s.

## 🎓 Teaching your AI

Aura improves from feedback with no retraining. Three levers, from easiest to deepest:

**1. Correct a wrong answer** — stored and injected into future prompts:

```python
from aura import autoamelioration
autoamelioration.enregistrer_correction(
    "quel est le carre de 5", "20",   # its wrong answer
    "25",                             # the right one
    "math")
```

**2. Feed the fact graph** — anything web-verified is learned automatically; add curated facts:

```python
from aura import graphe_faits
graphe_faits.ajouter("Canberra", "est la capitale de", "l'Australie",
                     source="manuel")
```

Or in bulk: `python scripts/nourrir_graphe.py knowledge.jsonl`
(skips identity entries, tags health/law as *general information*).

**3. Let it rehearse** — verified answers join the semantic cache: ask again
tomorrow, the answer comes back in ~1 ms. Wrong entries are replaced by
reconsolidation (`mettre_a_jour`), never duplicated.

> [!TIP]
> The fact graph only learns from **web-verified** content — by design, it
> cannot memorize a hallucination.

## 🔒 The `.aef` format + encrypted distribution

The whole system ships as **one sealed binary file**:

```text
aura_system.aef (807 MB, monolithic)
├── HEADER (148 B)  magic "AURA" · version · build · CPU config · 3× SHA-256
├── CONFIG (LZMA)   auto-tuned settings compiled for the target machine
├── CODE (ZIP)      the entire aura/ python package (verified by hash)
└── WEIGHTS (raw)   the GGUF byte-for-byte (loaded directly, zero-copy)
```

**Integrity**: one altered byte anywhere → boot refused (SHA-256 over config,
code and weights — tested). Optional **Ed25519 signature** of the file.

```bash
uv run python scripts/construire_exe.py            # -> dist/Aura.exe (~8.6 MB)
uv run python scripts/forger_aef.py --chiffrer     # -> aura_system.aef.enc (AES-256-GCM)
```

On the user machine (secret delivered separately, never in the file):

```powershell
Aura.exe aura_system.aef.enc "your question"       # decrypt, verify, answer
Aura.exe --cache "your question"                    # next runs: instant start
```

> [!IMPORTANT]
> AES-256-GCM + PBKDF2 (600 000 iterations): without the secret the payload
> is indistinguishable from noise, and any tampering breaks decryption.
> Honest note: this blocks passive copying — not a determined reverse-engineer.

## ⚙️ Hardware auto-tuning & GPU

Aura inspects the machine (CPU, cores, RAM, AVX2) and derives optimal settings:

```bash
python -m aura.profil_materiel
```

| Machine | Derived settings |
|---|---|
| RAM ≥ 6 GB | ctx 1536, batch 768, threads = all logical cores |
| RAM < 6 GB | ctx 1024, batch 512 (guaranteed fit) |

GPU support: CUDA/Vulkan build + discrete GPU → full offload; CPU-only → 0
(optimal here: 13.4 tok/s measured). Force CPU with `AURA_GPU=0`.

## 📊 Benchmarks

| Brain | Speed | French | Load |
|---|---|---|---|
| **Llama 3.2 1B Q4_K_M** (current) | **13.4 tok/s** | ✅ instruction-tuned | 3.4 s |
| MiniCPM5-1B Q4_K_M (optional, `AURA_GGUF=MINICPM`) | ~11 tok/s + thinking tokens | ✅ | 3.2 s |
| Qwen 2.5 1.5B (removed) | 12.1 tok/s | ✅ | — |
| Mamba-790M / RWKV-430M (removed) | 1.5-1.8 tok/s | ❌ base models | 30-400 s |

Same-quiz head-to-head (`scripts/comparer_cerveaux.py`): **Llama 6/7 @ 3.0 s/answer** vs **MiniCPM5 3/7 @ 10.2 s/answer** — MiniCPM5 is a *thinking-first* brain: without its native `<think>` phase it underperforms, with it (`AURA_THINK=1`) it is 3× slower. Aura's orchestration already reasons through experts, so a fast direct-answer brain wins here.

System-level quiz vs Qwen 2.5 1.5B: **Aura 5/5 vs 4/5** (exact math, real-time
facts, memory) — the organization beats the bigger brain on verifiable questions.

### Honest limits

> [!NOTE]
> On **verifiable** questions (math, facts, dates, format), Aura-1B beats
> bigger models by construction — they guess, it proves. On **deep analysis,
> long-tail knowledge and abstract logic**, a fine-tuned 8B still wins: those
> skills live in the weights, and that is what the roadmap targets.

## 🗺 Roadmap

- [x] Level 0: exact math, semantic cache, PAL
- [x] Verified-intelligence layer: CRITIC, fact graph, PoT, logic solver, PoT-code
- [x] `.aef` sealed format + Ed25519 + encrypted distribution
- [x] Green CI (4 OS/py matrices + `.aef` chain + installer syntax)
- [ ] **LoRA fine-tune** on Colab — reasoning depth, keeps the 807 MB size
- [ ] **MEMIT fact editing** — fix long-tail facts directly in the weights
- [ ] **lm-evaluation-harness** — public, comparable brain scores
- [ ] Hugging Face release (`.aef` + `Aura.exe` + model card)

## 🧪 Tests & quality

```bash
uv run pytest -q        # 173 tests (CI: 167 + 6 skipped — no GGUF in CI)
uv run mypy aura/       # 0 error
```

CI (GitHub Actions) runs on **Ubuntu + Windows** (Python 3.11/3.12), re-validates
the whole `.aef` chain (forge → Ed25519 signature → boot → tamper detection)
on Linux, and syntax-checks the PowerShell installer.

## 📄 License

MIT — see [LICENSE](LICENSE).
