---
license: mit
language:
  - fr
  - en
base_model:
  - meta-llama/Llama-3.2-1B-Instruct
tags:
  - llm
  - cpu
  - local-ai
  - llama-cpp
  - gguf
  - rag
  - neuro-symbolic
  - program-of-thoughts
  - on-device-ai
  - sealed-distribution
  - llama-3.2
  - minicpm
library_name: llama.cpp
pipeline_tag: text-generation
---

<div align="center">

<img src="https://huggingface.co/Simonc44/aura-1b/resolve/main/assets/aura.png" alt="Aura-1B" width="420"/>

**La réponse la plus rapide est celle qu'on n'a pas besoin de générer.**

*The fastest answer is the one you never generate.*

[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)

**English below — [repo complet / full repo](https://github.com/Simonc44/aura-1b)**

</div>

---

# 🇫🇷 Aura-1B — le système IA scellé, sans GPU

Aura-1B n'est pas un modèle de plus : c'est un **système orchestré** qui
sépare le langage, les maths exactes et la mémoire factuelle en modules
spécialisés. Règle d'or : *le petit modèle propose, le programme prouve* —
ce qui n'est pas prouvé est vérifié ou refusé, jamais inventé.

| | Aura-1B | Un 8B sur le même PC |
|---|---|---|
| Question quotidienne | **0,0–2,3 s** | 30–60 s |
| Maths / dates / unités | **exact** (AST, PAL, PGS) | hallucine hors statistiques |
| Faits | **web en direct + preuve CRITIC** | figés à la date d'entraînement |
| RAM | **~1 Go** | ~4,5 Go |
| GPU | **aucun requis** | recommandé |

## 📦 Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `aura_system.aef.enc` | **Le système complet scellé et chiffré** (AES-256-GCM) : config auto-tunée + code + cerveau 807 Mo en un seul fichier |
| `aura_system.aef.sig` | Signature **Ed25519** du fichier scellé (authenticité vérifiable) |
| `Aura.exe` (dans `Aura-v0.1.0-windows.zip` du [repo GitHub](https://github.com/Simonc44/aura-1b/releases)) | Le noyau léger (~8 Mo) : vérifie, déchiffre, exécute |
| — | ⚠️ **Le secret de déchiffrement n'est JAMAIS publié ici** — il se obtient auprès de l'auteur |

## 🚀 Utilisation

```powershell
# 1. le secret (donné a part) :
set AURA_SECRET=<le secret>

# 2. premier lancement : verifie les 3 SHA-256 + la signature, puis repond
Aura.exe aura_system.aef.enc "quel est le carre de 7"
# -> 49

# 3. runs suivants : demarrage instantane (cache deja rempli)
Aura.exe --cache "15% de 200 plus 30% de 100"
# -> 60
```

Vous préférez le code ouvert ? Le système complet (orchestrateur, experts,
tests, installeur) est **open source** :
[github.com/Simonc44/aura-1b](https://github.com/Simonc44/aura-1b)

## 🔒 Sécurité

- **Intégrité** : 3× SHA-256 (config, code, poids) — un octet altéré = boot refusé
- **Confidentialité** : AES-256-GCM + PBKDF2 (600 000 itérations), secret jamais stocké dans le fichier
- **Authenticité** : signature Ed25519 du fichier scellé
- Note honnête : cela bloque la copie passive, pas un ingénieur inverse déterminé.

## ⚠️ Limites honnêtes

Sur les questions **vérifiables** (maths, faits, dates, format), Aura-1B bat
des modèles plus grands par construction — eux devinent, il prouve. Sur
**l'analyse profonde, le savoir de niche et la logique abstraite**, un 8B
fine-tuné gagne encore : ces compétences vivent dans les poids (feuille de
route : LoRA + MEMIT). Détail complet dans le README du repo.

---

# 🇬🇧 Aura-1B — the sealed AI system, GPU-free

Aura-1B is not another model: it is an **orchestrated system** that splits
language, exact math and factual memory into specialized modules. Golden rule:
*the small model proposes, the program proves* — anything unproven is verified
or refused, never invented.

| | Aura-1B | Typical 8B on the same PC |
|---|---|---|
| Everyday question | **0.0–2.3 s** | 30–60 s |
| Math / dates / units | **exact** (AST, PAL, PGS) | hallucinates off-stats |
| Facts | **live web + CRITIC proof check** | frozen at cutoff |
| RAM | **~1 GB** | ~4.5 GB |
| GPU | **none required** | recommended |

## 📦 Files

| File | Role |
|---|---|
| `aura_system.aef.enc` | **The whole sealed, encrypted system** (AES-256-GCM): auto-tuned config + code + 807 MB brain in a single file |
| `aura_system.aef.sig` | **Ed25519 signature** of the sealed file |
| `Aura.exe` (in the GitHub release zip) | The lightweight kernel (~8 MB): verify, decrypt, run |
| — | ⚠️ **The decryption secret is NEVER published here** — ask the author |

## 🚀 Usage

```powershell
set AURA_SECRET=<the secret>
Aura.exe aura_system.aef.enc "square of 7"        # -> 49
Aura.exe --cache "15% of 200 plus 30% of 100"     # -> 60, instant start
```

Full open-source system: [github.com/Simonc44/aura-1b](https://github.com/Simonc44/aura-1b)

## 🔒 Security

3× SHA-256 integrity (one altered byte → boot refused) · AES-256-GCM +
PBKDF2-600k · Ed25519 signature. Honest note: blocks passive copying, not a
determined reverse-engineer.

## ⚠️ Honest limits

On **verifiable** questions (math, facts, dates, format), Aura-1B beats larger
models by construction — they guess, it proves. On **deep analysis, long-tail
knowledge and abstract logic**, a fine-tuned 8B still wins (roadmap: LoRA +
MEMIT).

## 📄 Licence / License

MIT — see [LICENSE](https://github.com/Simonc44/aura-1b/blob/main/LICENSE).

## 📚 Citation

```bibtex
@software{aura1b2026,
  author  = {Simon [Simonc44]},
  title   = {Aura-1B: Hybrid Neuro-Symbolic AI System},
  year    = {2026},
  url     = {https://github.com/Simonc44/aura-1b}
}
```
