<div align="center">

<img src="assets/aura.png" alt="Aura-1B" width="420"/>

**The fastest answer is the one you never generate.**

A 100% local, CPU-native AI system that splits language, exact math and factual
memory into specialized modules — instead of asking one small model to do
everything (and hallucinate when it can't).

Seven cognitive agents, exact symbolic experts and self-play turn a 1B brain
into a system that behaves like a **7-8B where it counts** — at 1B speed.

[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg?style=flat-square)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-5bc0de.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-5bc0de)
![mypy](https://img.shields.io/badge/mypy-checked-5bc0de)
![Last commit](https://img.shields.io/github/last-commit/Simonc44/aura-1b/main?label=last%20commit&color=5bc0de)
![PRs](https://img.shields.io/badge/PRs-welcome-5bc0de)
<a href="https://github.com/Simonc44/aura-1b/stargazers"><img src="https://img.shields.io/github/stars/Simonc44/aura-1b?style=social" alt="stars - aura-1b" /></a>

![GPU](https://img.shields.io/badge/GPU-not%20required-5bc0de)
![Brain](https://img.shields.io/badge/brain-807%20MB%20Q4__K__M-5bc0de)
![Engine](https://img.shields.io/badge/powered%20by-llama.cpp-5bc0de)
![Format](https://img.shields.io/badge/sealed%20format-.aef-5bc0de)

**Other languages:** [Français](README.fr.md)

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

### The equivalence, measured by capability

| Capability | Aura-1B system | Raw 1B brain |
|---|---|---|
| Logic & math | **beyond class** — exact, proven | hallucinates |
| Facts (graph + web) | **~7-8B anchored**, 0 hallucination | ~1B, frozen |
| Code (sandbox-verified) | **~8B** on testable code | worse |
| Style & writing | **~7-8B perceived** — few-shot mimicry + double-pass editing | 1B tics |
| Deep unanchored analysis | honest limit — weights stay 1B | 1B |

The mechanism is the one that makes a 70B *feel* smarter than its parameter
count: organization, verification and style — not raw size. On verifiable
tasks Aura no longer plays in the 3B league: it behaves like a **7-8B** while
keeping 1B speed and footprint.

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
| **Cognitive agents** (`agents.py`) | small-model brittleness | 7 lightweight agents orchestrate the 1B: task planner (complex asks split into 2-3 steps before writing), RAG compressor (only the 3 useful web sentences reach the LLM), style cleaner (removes 1B language tics), recursive debugger (failed code is retried with the exact sandbox error, max 3), persona router (system prompt adapted per category), style mimicry (few-shot exemplars calibrate the answer to big-model prose), double-pass editor (the 1B critiques its own draft then rewrites — catching flaws in existing text is statistically easier than writing perfectly first-try) |
| **LoRA hot-swap** (`adaptateurs.py`) | one brain, one specialty | dynamic adapter swapping via the llama.cpp C-API: the base GGUF loads once, category adapters (~10-40 MB, trained on Colab with PEFT) mount/unmount without reloading. LRU in memory (default 1 adapter = near-zero footprint). `AURA_ADAPTATEURS=1` to enable |
| **Served multi-LoRA** (`serveur_lora.py`) | one brain, several simultaneous specialties | the official llama-server (CPU build, auto-started by Aura) mounts the Q4_K_M brain + all 4 category adapters at boot; the router's probabilities become simultaneous weights `{"math": 0.7, "web": 0.3}` in one HTTP call — no reload, ~ms hot-swap. In-process fallback never blocks |
| **Dynamic Compute** | one speed for everything | complex questions trigger a masked `<thinking>` draft (×2 token budget) stripped before display; simple ones stay instant |
| **Long-term memory** (`graphe_faits.py`) | equal-weight knowledge | every triplet carries a strength; each **verified** use reinforces it, lookup is strength × recency weighted — knowledge organizes its own importance, zero retraining |
| **Self-play** (`selfplay.py`) | improvement needs a user | a challenge generator builds logic puzzles & drills from the fact graph, a deterministic judge validates, successes reinforce the graph and join the training dataset — AlphaGo-style closed loop. `aura --selfplay 20` |
| **RLSS** (`rlss.py`) | unvalidated self-training | generated exercises → brain answers → exact validation → successes stored for the next LoRA, failures become injected corrections |
| **Auto-improvement** | repeated mistakes | wrong answers recorded via `enregistrer_correction()` are injected into future prompts — the same mistake is never made twice |

### Rich mode (writing quality)

Open questions (≥ 9 words or *explique/analyse/compare…*) trigger a 3-step pipeline:
1. **Enriched search** — dual query (facts + analysis) + domain **lexicon** injected as keywords.
2. **Masked Chain-of-Thought** — planning inside `<thinking>` tags, stripped by regex; the user only sees the polished answer.
3. **Sectioned generation** — each part of the GBNF-constrained plan is written separately, with memory of previous sections (StoryWriter-lite).
4. **Few-shot mimicry + double pass** — style exemplars calibrate the draft, then a critique pass lists the flaws and the model rewrites: editing existing text is where a 1B statistically matches a much bigger one.

Cost: ~2-3× latency — reserved for questions that deserve it. Measured on the
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
> bigger models by construction — they guess, it proves. Logic, code, style
> and knowledge are now **system-compensated** (perceived level: 7-8B); only
> **deep unanchored abstract analysis** stays 70B territory — the weights are
> still 1B, and that honesty is part of the design.

## 🗺 Roadmap

- [x] Level 0: exact math, semantic cache, PAL
- [x] Verified-intelligence layer: CRITIC, fact graph, PoT, logic solver, PoT-code
- [x] Cognitive agents: planner, RAG compressor, style cleaner, recursive code debugger, persona router
- [x] Dynamic LoRA hot-swapping: category adapters mounted on the live context (C-API), LRU memory registry
- [x] `.aef` sealed format + Ed25519 + encrypted distribution
- [x] Green CI (4 OS/py matrices + `.aef` chain + installer syntax)
- [x] Served multi-LoRA (weights mix) + Dynamic Compute + RLSS + self-play + long-term memory
- [x] Style mimicry (few-shot) + double-pass editing
- [ ] **LoRA fine-tune** on Colab — reasoning depth, keeps the 807 MB size
- [ ] **MEMIT fact editing** — fix long-tail facts directly in the weights
- [ ] **lm-evaluation-harness** — public, comparable brain scores
- [ ] Hugging Face release (`.aef` + `Aura.exe` + model card)

## 🧪 Tests & quality

```bash
uv run pytest -q        # 251 tests (CI: 245 + 6 skipped — no GGUF in CI)
uv run mypy aura/       # 0 error
```

CI (GitHub Actions) runs on **Ubuntu + Windows** (Python 3.11/3.12), re-validates
the whole `.aef` chain (forge → Ed25519 signature → boot → tamper detection)
on Linux, and syntax-checks the PowerShell installer.

## 📄 License

GPLv3 — see [LICENSE](LICENSE). Copyleft: any derivative of Aura's multi-agent architecture must remain open-source under the same terms.
