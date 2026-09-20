# Security Policy

Aura-1B is a local AI runtime whose **core promise is trust**: exact math,
verified facts, sealed binaries. Security reports are treated as top
priority — a system that can be silently tampered with loses its reason
to exist.

## Supported versions

| Version | Supported |
|---|---|
| `main` branch | ✅ |
| latest `v0.x` tag | ✅ |
| older tags | ❌ |

Aura moves fast; please verify your issue against the latest `main` before reporting.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use **GitHub's private vulnerability reporting**:
[Security → Report a vulnerability](https://github.com/Simonc44/aura-1b/security/advisories/new)

Include:
- the component concerned (`kernel.py`, `chiffrement.py`, `signature.py`,
  `potcode.py` sandbox, `filtre_instantane.py` AST evaluator, …);
- a minimal proof of concept (a tampered `.aef`, a sandbox escape payload,
  an AST bypass expression…);
- your assessment of severity.

You will get an acknowledgment within **72 hours**, and we will keep you
informed of the fix progress. Credit is given in the release notes unless
you prefer to stay anonymous.

## Security model — what Aura guarantees

| Layer | Guarantee |
|---|---|
| **`.aef` integrity** | 3× SHA-256 (config, code, weights): one altered byte → boot refused. Tested in CI (tamper detection). |
| **`.aef` confidentiality** | AES-256-GCM (AEAD) + PBKDF2-HMAC-SHA256, 600 000 iterations. The secret is never stored in the file. |
| **Authenticity** | optional Ed25519 signature of the sealed file. |
| **PoT-code sandbox** | restricted builtins (no `import`, `open`, `eval`, `exec`, `compile`, `getattr`), instruction budget enforced by `settrace` — infinite loops are killed deterministically. |
| **AST evaluator** | whitelisted node types only (never `eval()`), power/exponent bounds. |
| **Hallucination control** | the fact graph only learns from web-verified answers; short factual answers are re-anchored on web proof (CRITIC) before delivery. |

## Scope — honest limits

> [!NOTE]
> The sealed format **blocks passive copying**, not a determined
> reverse-engineer holding the secret. This is stated in the README: the
> `.aef` is an integrity + distribution mechanism, not DRM.

Out of scope for the sealed format: memory-forensics extraction on a machine
where the system is legitimately running. In scope and very welcome: any
**sandbox escape** of `potcode.py`, any **AST evaluator bypass**, any way to
make Aura **deliver an unverified answer as verified**.

## Known design trade-offs

- `Aura.exe` embeds compiled bytecode only and delegates execution to the
  local Python after cryptographic verification — a conscious choice for
  transparency over obfuscation.
- The fact graph is only as good as its sources: it is fed exclusively from
  web-verified content, but the web itself can be wrong; contested facts are
  re-consolidated when contradicted by stronger proof.
