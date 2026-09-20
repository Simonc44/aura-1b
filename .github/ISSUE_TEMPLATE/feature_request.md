---
name: ✨ Feature request
about: Suggest an idea for Aura-1B
title: "[feature] "
labels: enhancement
assignees: ""
---

**Problem to solve**
What are you trying to do that Aura can't (or can't do well)? Describe the
*problem*, not only the solution you have in mind.

**Proposed solution**
What should happen instead? If you have an implementation idea, describe it
here.

**Which part of the architecture does it touch?**
- [ ] Level 0 (instant answers: math, PAL dates/units/%, semantic cache)
- [ ] Router (classification, confidence)
- [ ] An expert (PoT, logic solver, PoT-code, PGS, fact graph, web memory)
- [ ] CRITIC verification / reconsolidation
- [ ] Rich mode (writing quality)
- [ ] `.aef` format / kernel / distribution
- [ ] Docs / benchmarks / other

**Does it fit the golden rule?**
Aura's design rule: *the small model proposes, the program proves* — anything
unproven is verified or refused. Does your feature:
- [ ] add something **verifiable** (great fit)
- [ ] improve **routing / speed** (great fit)
- [ ] make the model guess more (needs a strong justification)

**Alternatives considered**
Any workaround or alternative solution you have tried.

**Additional context**
Links, papers, prior art — anything that helps evaluate the idea.
