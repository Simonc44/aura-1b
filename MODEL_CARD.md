---
license: mit
language:
  - fr
  - en
base_model:
  - google/gemma-3-1b-it
  - Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF
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
  - gemma-3
  - gemma3
  - reasoning
  - minicpm
library_name: llama.cpp
pipeline_tag: text-generation
---

<div align="center">

<img src="https://huggingface.co/Simonc-44/aura-1b/resolve/main/assets/aura.png" alt="RontoGPT" width="420"/>

**La réponse la plus rapide est celle qu'on n'a pas besoin de générer.**
*The fastest answer is the one you never generate.*

<a href="https://github.com/Simonc44/aura-1b/stargazers"><img src="https://img.shields.io/github/stars/Simonc44/aura-1b?style=social" alt="stars - aura-1b"/></a>
[![Tests](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml/badge.svg)](https://github.com/Simonc44/aura-1b/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-5bc0de.svg)](https://github.com/Simonc44/aura-1b/blob/main/LICENSE)
![GPU](https://img.shields.io/badge/GPU-not%20required-5bc0de)

[🌐 Code open source](https://github.com/Simonc44/aura-1b) · [📦 Release v0.1.0](https://github.com/Simonc44/aura-1b/releases) · [📖 README complet](https://github.com/Simonc44/aura-1b#readme)

</div>

---

RontoGPT n'est pas un modèle de plus : c'est un **système orchestré** qui
sépare le langage, les maths exactes et la mémoire factuelle en modules
spécialisés. Règle d'or : *le petit modèle propose, le programme prouve* —
ce qui n'est pas prouvé est vérifié ou refusé, jamais inventé.

<div align="center">

<img src="https://huggingface.co/Simonc-44/aura-1b/resolve/main/assets/architecture.png" alt="Architecture d'RontoGPT" width="820"/>

</div>

| | RontoGPT | Un 8B sur le même PC |
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
| [`RontoGPT.exe`](https://huggingface.co/Simonc-44/aura-1b/resolve/main/RontoGPT.exe) | 8,2 Mo | Le noyau léger : vérifie, déchiffre, exécute |

> [!WARNING]
> **Le secret de déchiffrement n'est JAMAIS publié ici** — il s'obtient
> auprès de l'auteur. Le `.aef` garantit l'*intégrité* (un octet altéré =
> boot refusé), pas un DRM absolu.

## 🚀 Utilisation

```powershell
# 1. le secret (donné à part) :
set AURA_SECRET=<le secret>

# 2. premier lancement : vérifie 3× SHA-256 + signature, puis répond
RontoGPT.exe aura_system.aef.enc "quel est le carre de 7"
# -> 49

# 3. runs suivants : démarrage instantané (cache déjà rempli)
RontoGPT.exe --cache "15% de 200 plus 30% de 100"
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

Sur les questions **vérifiables** (maths, faits, dates, format), RontoGPT bat
des modèles plus grands par construction — eux devinent, il prouve. Sur
**l'analyse profonde, le savoir de niche et la logique abstraite**, un 8B
fine-tuné gagne encore : ces compétences vivent dans les poids (feuille de
route : LoRA + MEMIT). Détail complet dans le README du repo.

**Chiffré, pas supposé** (10 oct. 2026, `lm-eval` 0.4.13, cerveau nu sans le pipeline) :
GSM8K 5-shot = **0,16** (strict-match) / **0,20** (flexible-extract) sur 100 questions.
Le format de réponse est bon (le modèle termine bien par `#### N`), c'est le raisonnement qui
casse — donc le score est réel et non un artefact de mesure. Les tâches à choix multiple
(ARC, HellaSwag, MMLU…) restent **non mesurables** ici : llama.cpp n'expose pas les logprobs du
prompt, et les scores qu'on en tirerait seraient du hasard déguisé. Le pont
(`scripts/lm_eval_aura.py`) refuse ces tâches explicitement. Détail dans `ARCHITECTURE.md` § 3quinquies.

Et **à travers RontoGPT elle-même** (mêmes 100 questions, via l'API `/v1/ask`, 10 oct. 2026) :
0,12 en flexible-extract, 0,00 en strict-match. Ce dernier chiffre ne mesure rien — RontoGPT
répond en français et ne produit jamais le marqueur `####` qu'attend ce filtre. La comparaison
honnête est donc 0,12 contre 0,20 : **sur GSM8K, le pipeline ne rattrape pas le cerveau**, il
ne couvre pas ces énoncés. Cette mesure a aussi mis au jour un défaut réel dans la couche
« exacte » (`_calcul_direct` répond à toute phrase contenant `a/b`). Détail dans
`ARCHITECTURE.md` § 3sexies.

**Sonde d'interprétabilité sur un 4B** (16 oct. 2026, `scripts/sonde_interp.py` — « logit lens » couche par
couche sur `unsloth/gemma-3-4b-it`, 5 prompts, mesures brutes dans
`resultats/sonde_gemma4b.json`) : le fait est **déjà formé en interne** bien avant la sortie —
« Canberra » sort à **0,9706 dès la couche 28**, six couches avant la fin, « Oui » à 0,9381 et
« Je » à 0,9341 en couche 33 — mais la **dernière couche ne le livre pas** : elle est quasi
certaine (p ≥ 0,9997) et prédit l'**amorce d'une phrase** (`' Cette'`, `' C'`, `' É'`).
*Savoir n'est pas répondre*, et ici cela se mesure. Et **sur `17 * 23`, aucun chiffre n'entre
jamais dans le top 3, sur aucune des 35 positions** (le mieux est `'1'` à 0,0006, ~1 600 fois
sous l'espace) : il n'y a pas de produit « déjà calculé » à extraire — l'argument chiffré du
solveur déterministe, plutôt que d'un modèle plus gros. Mesures, réserves et reproduction
(prompt sondé **sans** le gabarit de conversation ; 4B et non le 1B livré) :
[`resultats/NOTE_sonde_gemma4b.md`](resultats/NOTE_sonde_gemma4b.md).

## 📚 Citation

```bibtex
@software{aura1b2026,
  author  = {Simon [Simonc44]},
  title   = {RontoGPT: Hybrid Neuro-Symbolic AI System},
  year    = {2026},
  url     = {https://github.com/Simonc44/aura-1b}
}
```

## 📄 Licence

MIT — see [LICENSE](https://github.com/Simonc44/aura-1b/blob/main/LICENSE).
