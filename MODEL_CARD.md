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

<img src="https://huggingface.co/Simonc-44/aura-1b/resolve/main/assets/aura.png" alt="Aura-1B" width="420"/>

**La réponse la plus rapide est celle qu'on n'a pas besoin de générer.**
*The fastest answer is the one you never generate.*

<a href="https://github.com/Simonc44/aura-1b/stargazers"><img src="https://img.shields.io/github/stars/Simonc44/aura-1b?style=social" alt="stars - aura-1b"/></a>
[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-5bc0de.svg)](https://github.com/Simonc44/aura-1b/blob/main/LICENSE)
![GPU](https://img.shields.io/badge/GPU-not%20required-5bc0de)

[🌐 Code open source](https://github.com/Simonc44/aura-1b) · [📦 Release v0.1.0](https://github.com/Simonc44/aura-1b/releases) · [📖 README complet](https://github.com/Simonc44/aura-1b#readme)

</div>

---

Aura-1B n'est pas un modèle de plus : c'est un **système orchestré** qui
sépare le langage, les maths exactes et la mémoire factuelle en modules
spécialisés. Règle d'or : *le petit modèle propose, le programme prouve* —
ce qui n'est pas prouvé est vérifié ou refusé, jamais inventé.

<div align="center">

<img src="https://huggingface.co/Simonc-44/aura-1b/resolve/main/assets/architecture.png" alt="Architecture d'Aura-1B" width="820"/>

</div>

| | Aura-1B | Un 8B sur le même PC |
|---|---|---|
| Question quotidienne | **0,0–2,3 s** | 30–60 s |
| Maths / dates / unités | **exact** (AST, PAL, PGS) | hallucine hors statistiques |
| Faits | **web en direct + preuve CRITIC** | figés à la date d'entraînement |
| RAM | **~1 Go** | ~4,5 Go |
| GPU | **aucun requis** | recommandé |

## 📦 Téléchargements

| Fichier | Taille | Rôle |
|---|---|---|
| [`aura_system.aef.enc`](https://huggingface.co/Simonc-44/aura-1b/resolve/main/aura_system.aef.enc) | 779 Mo | **Le système complet scellé et chiffré** (AES-256-GCM) : config auto-tunée + code + cerveau 807 Mo en un seul fichier |
| [`aura_system.aef.sig`](https://huggingface.co/Simonc-44/aura-1b/resolve/main/aura_system.aef.sig) | 110 o | Signature **Ed25519** (authenticité vérifiable) |
| [`Aura.exe`](https://huggingface.co/Simonc-44/aura-1b/resolve/main/Aura.exe) | 8,2 Mo | Le noyau léger : vérifie, déchiffre, exécute |

> [!WARNING]
> **Le secret de déchiffrement n'est JAMAIS publié ici** — il s'obtient
> auprès de l'auteur. Le `.aef` garantit l'*intégrité* (un octet altéré =
> boot refusé), pas un DRM absolu.

## 🚀 Utilisation

```powershell
# 1. le secret (donné à part) :
set AURA_SECRET=<le secret>

# 2. premier lancement : vérifie 3× SHA-256 + signature, puis répond
Aura.exe aura_system.aef.enc "quel est le carre de 7"
# -> 49

# 3. runs suivants : démarrage instantané (cache déjà rempli)
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

## 📚 Citation

```bibtex
@software{aura1b2026,
  author  = {Simon [Simonc44]},
  title   = {Aura-1B: Hybrid Neuro-Symbolic AI System},
  year    = {2026},
  url     = {https://github.com/Simonc44/aura-1b}
}
```

## 📄 Licence

MIT — see [LICENSE](https://github.com/Simonc44/aura-1b/blob/main/LICENSE).
