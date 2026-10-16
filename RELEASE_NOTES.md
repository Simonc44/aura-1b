# RontoGPT v0.3.0 — measured, not claimed

> **The fastest answer is the one you never generate.**
> A 100 % local, CPU-native AI: a 1B brain routed through exact solvers, verified
> facts and a sealed `.aef` distribution — **no GPU required**.

## ✨ Added

- **Escalation to Gemini as a measured decision** (`aura/aiguilleur.py` +
  `aura/gemini.py`) — when the local answer is judged insufficient (on the task
  *and* on the response), the question goes out to Google's API. The HTTP call
  lives in one module, the key is read from the environment or `.env` and never
  written to the URL, and `python -m aura.gemini` diagnoses key, model and
  network end to end. Set `AURA_GEMINI=0` to cut it, or simply leave without a
  key: the local path never blocks on it.
- **A terminal UI in TypeScript** (`tui/`) with its own smoke harness, next to
  the existing GUI and the HTTP API on `:8760` — one port of entry, decoupled
  from the engine.
- **A declarative expert registry** (`aura/registre_experts.py`). The cascade
  order is no longer a chain of `if`: every expert declares its kind, cost,
  source and *reason*, the invariant ("no approximate before an exact") is
  carried by the kind, and a test compares the registry with the order
  **actually executed**. It caught a missing expert by itself.
- **Latency contracts** (`aura/contrats.py`). A "rich" answer promises 900
  characters inside a 45 s budget; the response now carries the budget verdict
  instead of silently shipping something shorter.
- **`scripts/audit_dependances.py`** — the security audit now runs locally,
  Windows included. `pip-audit --no-deps -r` cannot finish there (two distinct
  causes, both documented), and auditing a `.venv` that had drifted from the
  lock once reported three advisories the lock did not have.

## 🔧 Changed

- **Python 3.11 works again.** The project did not compile under the version it
  advertises — an f-string only 3.12 accepts. A `compileall` gate with the real
  3.11 interpreter now refuses that class of mistake.
- **The biggest module lost its output bus** (`aura/flux_cerveau.py`): phases,
  subscribers, the `<think>`-aware dispatcher and the turn's reflection memory
  moved out, block by block, with the public surface unchanged.
- **The security job can no longer pass green while verifying nothing.** The
  audit used to run on a path that could be missing; it is now computed by the
  venv's own interpreter and checked before use.
- **A single bad header read no longer lasts forever.** `famille()` cached an
  empty result as if it were a success: one transient read hiccup (a few KB)
  renamed the brain to "unknown" **for the whole process** and replaced its
  end-of-turn tags with generic ones. Only a non-empty result is cached now, and
  a regression test proven by restoring the old code.
- **Every dependency comes from the project metadata** (`uv sync --frozen`), on
  a random test order, nothing hand-copied.

## 🧪 Quality

- **1159 tests, 0 skipped on a complete machine** (+908 since v0.2.0). The three
  tests that skipped **everywhere** — two needed an optional `lm_eval`, one
  asserted a disk state that had moved — are gone: the first two now measure
  against a synthetic task tree (groups, `include:` chains, duplicate names), the
  third was retired because its invariant is locked elsewhere.
- **Coverage is measured, not claimed: 72 %** (branches included, 9535
  statements). Getting that number found a real bug: `aura/potcode.py` undid its
  own tracer with `sys.settrace(None)`, erasing the host's tracer — coverage
  first reported 20 %.
- **`mypy aura/` 0 errors over 74 files**, untyped function bodies included;
  **ruff 0**; the TUI typechecks (`tsc --noEmit`); **8 CI jobs** — Ubuntu and
  Windows × Python 3.11/3.12, the whole `.aef` chain (forge → Ed25519 → boot →
  tamper detection), typing, lint, the 3.11 gate, the security audit and the
  installer.
- New in this release: a **logit-lens probe published with its own caveats**
  (`resultats/NOTE_sonde_gemma4b.md`) and an **honest self-assessment out of
  100 000** (`docs/AUTOEVALUATION_2026-10-16.md`) where every point lost is tied
  to a measurement.

## ⚠️ Honest limits

- **On GSM8K the pipeline does not rescue the brain**: 0.16 / 0.20 bare (100
  questions), 0.12 through the API. The failure is the reasoning, not the format.
- **Long-form writing remains the weak spot** (4/10 in the latest full run).
- **No defensible ranking against 3B/8B models** — no shared benchmark exists.
  A 4B probe shows the fact forming mid-network while the last layer starts a
  sentence instead, and no digit of `17 × 23` ever reaches the top 3.
- **Coverage is 72 % on average and very uneven** (`aura/tui.py` sits at 24 %).
- **One advisory is accepted and written down**: `diskcache` (pulled in by
  `llama-cpp-python`) has no published fix.
- The `.aef` **does not compress the weights**: the sealed file is ~808 MB.

## 📦 Downloads

| File | What it is |
|---|---|
| `RontoGPT-v0.3.0-windows.zip` | `RontoGPT.exe` (bytecode-only kernel, ~12-15 MB) + README + LICENSE |
| `RontoGPT-Setup-0.3.0.exe` | Windows installer (Inno Setup), user-level, no admin needed |
| `checksums-sha256.txt` | SHA-256 of the ZIP |

The first launch builds the environment and fetches the 807 MB brain
(`scripts/installer.ps1`), then boots from the sealed `.aef` — integrity is
verified once, and `--cache` starts instantly afterwards.

**Full changelog**: https://github.com/Simonc44/rontogpt/compare/v0.2.0...v0.3.0
