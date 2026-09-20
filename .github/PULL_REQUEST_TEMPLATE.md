## What does this PR do?

<!-- One sentence: the intention, not the diff. -->

## Related issue

<!-- Closes #123 (or "N/A") -->

## Type of change

- [ ] 🐛 Bug fix (non-breaking, ships with the test that would have caught it)
- [ ] ✨ New feature / new expert
- [ ] 📝 Docs
- [ ] 🔧 Refactor (no behavior change)
- [ ] 📊 Benchmark / routing examples

## Quality gates (all three must pass)

- [ ] `uv run pytest -q` — all tests pass (with GGUF: 173; without, big-brain tests skip)
- [ ] `uv run mypy aura/` — 0 error
- [ ] Fits the golden rule: **the small model proposes, the program proves** —
      anything unproven is verified or refused, never invented

## If this adds an expert or changes routing

- [ ] Follows the `potcode.py` pattern: propose → deterministic verify → refuse with fallback
- [ ] Live quiz before/after pasted below (measurements beat opinions)

```text
# before:
# after:
```

## Checklist

- [ ] Commits are one logical change each, imperative mood
- [ ] New/changed user-facing behavior reflected in the README
- [ ] No new heavyweight dependency (or discussed in an issue first)
