# Contributing to RontoGPT

First off, **thank you** — RontoGPT is a young project and every contribution
matters, from a typo fix to a whole new reasoning expert.

The golden rule of this codebase, which every contribution should respect:

> **The small model proposes. The program proves.**
> Anything unproven is verified against a source or refused — never invented.

## 📜 Code of Conduct

By participating, you are expected to uphold our
[Code of Conduct](CODE_OF_CONDUCT.md). Report unacceptable behavior to the
address in that file.

## 🐛 How to report a bug

1. **Search first** — check [open issues](https://github.com/Simonc44/aura-1b/issues)
   to avoid a duplicate.
2. Open a **Bug report** issue (a template guides you) and fill in:
   - your OS, Python version, whether the GGUF is downloaded;
   - the **exact command** you ran;
   - the full console output.
3. Label the severity honestly: does it produce a **wrong answer silently**
   (highest priority — RontoGPT's whole point is trust) or a crash?

> [!IMPORTANT]
> A silently wrong answer is treated as a security-adjacent bug. If RontoGPT
> answers something unverifiable with confidence, that is a reportable defect.

## 💡 How to suggest a feature

Open a **Feature request** issue and describe:
- the **problem** you are trying to solve, not just the solution you imagine;
- which of RontoGPT's design rules it touches (router? level 0? an expert? the
  fact graph?) — the architecture doc in the README is the map;
- whether it fits the "proposes / proves" rule. A feature that makes RontoGPT
  *guess more* needs a very strong case.

## 🧩 Good first contributions

| Difficulty | Ideas |
|---|---|
| 🟢 Easy | docs typos, README improvements, new PAL units/dates formats (+ tests) |
| 🟡 Medium | new expert triggers (router examples), fact-graph patterns, benchmark questions |
| 🔴 Hard | a new expert (following `potcode.py` pattern: propose → verify → refuse), GBNF grammars, confidence calibration |

## 🛠 Development workflow

```bash
# 1. Fork & clone your fork
git clone git@github.com:<you>/rontogpt.git
cd rontogpt

# 2. Environment (uv is the reference, plain venv works too)
uv sync
python scripts/telecharger_llama.py        # optional: 807 MB brain

# 3. Quality gates — all four must pass before you push
uv run pytest -q        # 1152 tests
uv run mypy aura/       # 0 error (74 files, bodies of untyped functions included)
uv run ruff check .     # 0 violation (cible py311)
uv run pytest -q --cov  # coverage (source + branch come from pyproject.toml)
```

### Branch naming & commits

- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `bench/<name>`
- Commits: one logical change per commit, imperative mood
  (`fix: reconsolidation on empty cache`, not `fix bug`).
- Tests are **part of the change**: a fix ships with the test that would
  have caught it; a new expert ships with its own test file.

### Pull requests

- Fill the PR template (it checks the three gates for you).
- Keep PRs focused — one feature or one fix; link the issue it closes.
- CI runs the full matrix (Ubuntu + Windows × Python 3.11/3.12) plus the
  `.aef` chain, the typing invariant (`mypy`), ruff and the TUI typecheck;
  a red CI is fine, we debug together.

### Code style

- Python 3.11+, fully type-annotated (mypy-clean is enforced by config).
- **3.11 is the floor and ruff enforces it** (`target-version = "py311"`).
  A 3.12-only syntax is refused even if your machine runs 3.12 — that is how
  `aura/llama_cerveau.py` stayed unparseable for every 3.11 user.
- Ruff (default rules: E4/E7/E9/F) is enforced by CI; it flags errors, not
  style. Add `# noqa: <rule>` with a reason only for a real false positive.
- French is the language of the codebase (docstrings, comments, log
  messages, without accents for portability); English for the README and
  user-facing docs.
- New experts follow the pattern of `aura/potcode.py`: the model proposes,
  a deterministic component verifies, and failure means *refusal with
  fallback*, never a best-effort wrong answer.
- No new heavyweight dependency without an issue discussion first — the
  whole point of RontoGPT is a small footprint.

## 🧪 Testing your change

- `pytest -q` locally (with the GGUF if you have it: 1152 tests; without:
  the big-brain tests skip themselves — CI covers that path).
- The suite runs in a **random order** (`pytest-randomly`): a test that only
  passes because another ran before it is a bug in the test, not bad luck.
- **Coverage: `uv run pytest -q --cov`.** Source and branch are read from
  `[tool.coverage]` in `pyproject.toml`, so the bare flag is enough. There is
  intentionally **no `--cov-fail-under`**: at today's level a floor would pass
  for years without guaranteeing anything, and a self-skipping test would
  turn the CI red with no line changed. The number is printed on every CI run
  — read it, and treat a **drop** as a question to answer.
- **Never call `sys.settrace(None)`** to undo your own trace function: it
  erases the host's tracer (debugger, profiler, coverage). Save it first
  (`avant = sys.gettrace()`) and put it back in the `finally`. `aura/potcode.py`
  did this and made coverage under-report by 38 points on the full suite.
- For routing changes: run a small live quiz and paste the before/after in
  the PR. Measurements beat opinions.

## 📄 License

By contributing, you agree that your contributions will be licensed under
the [GPL-3.0-or-later License](LICENSE) that covers the project. (The repo
used to ship a contradictory `LICENSE.mit.txt`; it has been removed.)
