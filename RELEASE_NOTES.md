# Aura-1B v0.1.0 — first public release

> **The fastest answer is the one you never generate.**
> A 100 % local, CPU-native AI: Llama 3.2 1B orchestrated with exact math,
> verified facts and a sealed `.aef` distribution — **no GPU required**.

## ✨ Added

- **Level 0 — instant answers (< 5 ms)**: exact math via safe AST evaluator,
  PAL (dates, units, compound percentages), semantic answer cache with
  reconsolidation.
- **Verified-intelligence layer**: CRITIC web re-anchoring of short factual
  answers, fact graph (MiniRAG-lite) fed exclusively by verified content
  (223 curated triplets), Program-of-Thoughts puzzles with per-step AST
  verification, pure-Python mini-SAT logic solver, sandboxed PoT-code expert
  (restricted builtins + instruction budget), genetic symbolic regression
  (exact formulas, zero error).
- **Rich mode**: dual-query search with domain lexicon, masked
  chain-of-thought, sectioned generation with GBNF-constrained planning.
- **Auto-improvement**: correction injection, confidence-calibrated routing
  (TF-IDF + Logistic Regression), single-cycle decision (fan-out + abandon).
- **`.aef` sealed format v0.7**: monolithic file (header + config + code +
  807 MB weights), 3× SHA-256 integrity, AES-256-GCM encryption
  (PBKDF2, 600 000 iterations), optional Ed25519 signature, instant `--cache`
  boot.
- **Hardware auto-tuning**: CPU/RAM/AVX2 detection → derived ctx/batch/threads;
  OOM guard before load; automatic GPU offload when a discrete GPU exists.
- **Installer**: one PowerShell command → `aura` available in the terminal.

## 🔧 Changed

- Brain migrated to **Llama 3.2 1B Instruct Q4_K_M** (chosen over Qwen 2.5
  1.5B by benchmark: better French, 13.4 tok/s on CPU).
- Rich-mode generation now writes **section by section** with memory of
  previous sections (StoryWriter-lite pattern).

## 🐛 Fixed

- Compound dictated math evaluated as a whole expression
  (« 15 divise par 3 plus 4 puissance 2 » → 21, not 16).
- Reconsolidation on an empty cache no longer fails (first-run / CI case).
- GBNF availability and GPU-layer detection skip cleanly on builds without
  llama.cpp.

## 🔒 Security

- Sealed distribution: any altered byte → boot refused (tested in CI,
  including a deliberate tamper-detection check on Linux).
- PoT-code sandbox: no `import`/`open`/`eval`/`exec`, deterministic
  kill-switch for infinite loops.

## 📦 Installation

### Windows — one command

```powershell
powershell -ExecutionPolicy Bypass -File scripts/installer.ps1
```

Then in a new PowerShell window:

```powershell
aura "quel est le carre de 7"     # -> 49
aura                              # interactive chat
```

### From source

```bash
uv sync
python scripts/telecharger_llama.py     # 807 MB brain, once
uv run python -m aura --demo
```

## 📊 Quality gates at release time

| Gate | Status |
|---|---|
| Tests | 173 passed (CI: 167 + 6 skipped, no GGUF in CI) |
| mypy | 0 error on 19 source files |
| CI | Ubuntu + Windows × Python 3.11/3.12, `.aef` chain, installer syntax |
| Footprint | 807 MB brain + ~1 MB logic, runs in ~1 GB RAM |

**Full changelog**: see the auto-generated notes below this message.
