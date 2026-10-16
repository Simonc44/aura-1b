# Architecture d'RontoGPT — schéma, plus/moins, évolutions

> Relevé sur la machine réelle (HP ProBook 440 G10), 6 octobre 2026.
> Chiffres **mesurés** sauf mention « estimation » ou « mesure antérieure ».

---

## 1. Schéma d'ensemble

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ MATÉRIEL                                                                     │
│  Intel U300 : 1 cœur P (4,4 GHz) + 4 cœurs E (3,3 GHz) = 6 threads, 12-15 W  │
│  16 Go DDR4-3200 MONO-CANAL (1 barrette)      → 1 emplacement libre, max 64  │
│  Intel UHD Graphics (iGPU, RAM partagée)      → pas de NPU (GNA seul)        │
│  NVMe 238 Go (~1 084 Mo/s)                    → 30 Go libres (86 % plein)    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│ MOTEUR  ── llama-server VULKAN 0.4.1-dev (build 11111, serveur/vulkan/) ──── │
│   -ngl 99            tout sur l'iGPU         prefill 216-227 tok/s (×3,1/CPU)│
│   -fa on             flash attention                                         │
│   --cache-type-k/v q8_0   cache KV en 8 bits                                 │
│   -c 4096            fenêtre de contexte                                     │
│   -t 6               tous les threads logiques                               │
│   --spec-type ngram-mod     brouillon n-gram, vérifié par la cible           │
│                       decode 20,4 tok/s · ×2,7 sur contenu répétitif         │
│   --lora ×4          multi-LoRA servi, mix de poids à chaud                  │
│   modèle : Gemma-3-1B-it Heretic-Thinking Q4_K_M (806 Mo) — gemma3, 26 couches │
│   (l'ancien Llama-3.2-1B-Instruct reste dispo : AURA_GGUF=LLAMA)              │
│   repli : chemin in-process (llama-cpp-python) si le serveur tombe           │
│   autre build present : b7400 CPU seul (serveur/b7400/, non utilise)         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ HTTP 127.0.0.1:8757
┌──────────────────────────────────────────────────────────────────────────────┐
│ INTELLIGENCE  ── cascade de executer_detaille (orchestrateur.py + 5 mixins) ─│
│                                                                              │
│  ① routeur (SVM/TF-IDF)  classifie la question                               │
│  ② niveau0 ── filtre_instantane  [SOUMIS EN PARALLÈLE, consommé le 1er]      │
│        ├ calcul direct    27/100   < 1 ms                                    │
│        ├ pal               2/100   < 1 ms                                    │
│        └ cache sémantique MiniLM (seuil 0,70) — « answer without generating »│
│  ③ fast-lang             0/100  ⚠️ NE SERT JAMAIS sur le banc                │
│  ④ calcul-achat          6/100   budget − prix×quantité                      │
│  ⑤ calcul-verbal        11/100   ajout / retrait / moitié                    │
│  ⑥ calcul-taux           5/100   taux×durée, total÷durée, partage            │
│  ⑥bis chaines-numeriques    4/100  ancrage + ecarts chiffres                 │
│  ⑥ter problemes-arithmetiques 3/100  fractions, perimetre / aire             │
│  ⑦ experts spéciaux                                                          │
│        ├ tri-transitif   1/8 logique  (relations transitives)                │
│        ├ logique         formalisation 1B → solveur SAT exact                │
│        └ potcode         1B écrit une fonction → sandbox + asserts           │
│  ⑧ rag-binaire           9/100   index local 259 docs (verifie), extrait     │
│  ⑨ raisonneur            PoT + AST + vérification des calculs                │
│  ⑨bis PORTE DE CLARIFICATION  question trop ouverte → 1-3 questions          │
│        (aura/clarification.py) — s'efface si la conversation a un sujet      │
│  ⑩ GÉNÉRATION (seulement si tout le reste a échoué)                          │
│        └ serveur_lora.chat → repli llama_cerveau.generer                     │
│           (mode riche supprime : plus de plan ni de sections)                │
│           flux typé : PHASE_PENSEE (le  thinking) puis PHASE_REPONSE         │
│  ⑪ filtre_instantane.enregistrer  écrit la réponse au cache                  │
│  ⑫ AIGUILLEUR (opt-in, GOOGLE_API_KEY) — aura/aiguilleur.py                  │
│        lit la TÂCHE demandée (dissertation, traduction) ET la RÉPONSE        │
│        rendue (vide, refus, boucle, langue, fait deviné sans source) et      │
│        décide : garder le 1B, ou passer la main à Gemini (aura/gemini.py     │
│        ne garde que l'appel HTTP). Un tier exact/vérifié n'est jamais rejugé │
│                                                                              │
│  transversal : budget/échéance 45 s · disjoncteur anti-boucle · 4 LoRA       │
│  entrées : api.py :8760 · GUI · Coucou/MacTheme                              │
│  mesure : scripts/banc_100.py (100 questions, 8 catégories)                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Mesure clé** : **86/100** questions du banc se résolvent **sans générer** (0,5 ms à 0,4 s au lieu
de 7–13 s mesurés). Mesuré en ~2,5 s par `scripts/sous_banc.py` (73/100 au 6 oct. 2026, 66/100
avant). **14 générations** sont encore tentées, contre 27 avant le 7 oct. 2026 — et
`JUSTE` = 86/100, c'est-à-dire la même valeur que la couverture : **tout ce qui est couvert est
exact**.

La **porte ⑨bis** ne propose jamais de réponse : elle peut *arrêter* la cascade pour
demander une précision (cf. §3ter). Sa position est déclarée elle aussi
(`registre_experts.PORTES` → `apres-experts`) : une question que les étages ② à ⑨
savent trancher ne peut pas recevoir « précise ta demande ».

⚠️ Les étages ⑥ à ⑨ **ne sont plus une suite de `if`** : leur ordre est déclaré dans
`aura/registre_experts.py` (avec le genre — exact / vérifié / flou / génératif — et la
raison de chaque rang) et vérifié par `tests/test_registre_experts.py`, qui compare l'ordre
**réellement exécuté** au registre. La liste ci-dessus est l'illustration de ce registre,
pas sa source : le lire pour connaître l'ordre exact.

---

## 2. Plus / moins / évolution, choix par choix

### Matériel

| Choix | ✅ Plus | ⚠️ Moins | 🔼 Évolution |
|---|---|---|---|
| **Intel U300** (1 P + 4 E, 15 W) | Très sobre ; gen récente (AVX2/VNNI/FMA) ; pas de throttle vu | **1 seul cœur performant** : parallélisme mesuré **1,6×** sur 6 threads | Rien à faire — soudé. Contourner par le symbolique (levier 2) |
| **16 Go mono-canal** | Suffisant pour un 1B (807 Mo) | **Goulot n°1** : decode *memory-bound*, mesuré 20,4 tok/s vs borne ≈ 31 (estimation) | **2ᵉ barrette 16 Go** → double canal, ~30-35 tok/s (estimation). Emplacement libre vérifié |
| **iGPU UHD** | Vulkan ×3,1 sur le prefill, gratuit | Pas de VRAM → **partage la bande passante** du CPU | Déjà exploité. Ne rien attendre de plus |
| **NVMe 238 Go** | ~1 Go/s, large | **30 Go libres (14 %)** : compression/swap et builds risqués | Terminer le nettoyage admin (script livré) → ~45 Go |
| **GNA, pas de NPU** | — | Aucune accélération IA matérielle exploitable | Aucune : le GNA est pour l'audio/wake-word, pas les LLM |

### Moteur d'inférence

| Choix | ✅ Plus | ⚠️ Moins | 🔼 Évolution |
|---|---|---|---|
| **llama.cpp** | GGUF + multi-LoRA + GBNF + Vulkan, tout dispo | Deux chemins (serveur / in-process) → **bug de parité réel** (le serveur n'envoyait aucune pénalité) | Unifier l'échantillonnage (dataclass + 2 backends) |
| **Vulkan `-ngl 99`** | prefill **×3,1** (66→220 tok/s) | decode seulement **+9 %** (limité par la RAM) | Rien. Déjà optimal ici |
| **Q4_K_M 807 Mo** | 3/3 en raisonnement, **10 % plus rapide que Q6_K** (mesure antérieure) | Qualité < Q6/Q8 | `imatrix` (outils déjà présents) : qualité à taille égale |
| **multi-LoRA `--lora ×4`** | 4 spécialisations sans recharger, hot-swap ms | Adaptateurs liés à l'archi 1B → **bloque le passage à 3B** | Plus de données d'entraînement par catégorie |
| **`--spec-type ngram-mod`** (n-min 8) | **×2,7** sur texte **répété** (acceptation 100 %) et **coût nul ailleurs** : il ne propose alors **aucun** token brouillon, le calcul est identique à `off` | Ne sert à rien sur du texte neuf (mais ne coûte rien) | Tester n-min 16/24 — **non mesuré** |
| ~~`ngram-cache`~~ **abandonné** | ×1,35 sur texte répété | **−16 à −42 %** sur plan / liste / code : il brouillonne (60/24/88 tokens) et l'acceptation est de **1,7 % / 0 % / 0 %** — du coût pur | Retiré le 6 oct. 2026, remplacé par `ngram-mod`. Mesure : `scripts/mesure_lookup.py` |
| **KV `q8_0` + `-fa on`** | Cache KV divisé par 2 | Quantification KV = légère perte de fidélité | `-c 4096` : contexte borné, adapté aux prompts usuels (~1500 tok) |
| **`-c 4096`** | Marge pour les prompts et réponses habituels | Les très longues conversations peuvent élaguer les tours anciens | Résumer les tours anciens au lieu de les garder verbatim |
| **Cache de prompt** | — | **`--cache-reuse` jamais utilisé** | À **mesurer** : le serveur réutilise déjà les slots par LCP (`f_keep 0,158`), gain incertain |
| **Chemin in-process (repli)** | Jamais bloquant | Duplication : 1486 lignes, deux échelles de relance | Fusionner avec le client serveur |

### Intelligence / orchestration

| Choix | ✅ Plus | ⚠️ Moins | 🔼 Évolution |
|---|---|---|---|
| **Cascade impérative** | Simple à lire, latence nulle | **L'ordre est porteur de sens** et n'existe que dans l'ordre des `if` ; 2 questions du banc en dépendent | **Registre d'experts** à priorité déclarée |
| **niveau0 en 1er** | Le plus gros contributeur (27/100) ; < 1 ms | **Une mémoire floue prime sur un solveur exact** (observé : le cache a répondu 240 avant `calcul-taux`) | Remonter les solveurs exacts **avant** le cache |
| **Cache sémantique MiniLM 0,70** | Répond sans générer | Peut servir une réponse *proche mais fausse* ; pollue les mesures si non isolé | Idem ci-dessus + garder l'isolation `AURA_FILTRE_CACHE` |
| **Solveurs `calcul-*`** | **22/100**, déterministes, 0–5 ms | Couverture partielle (pas de conversion d'unités) | **Levier n°2** : chaque question gagnée vaut ~10 000× |
| **`fast-lang`** | Zéro latence | **0/100 — ne sert jamais** sur le banc | Supprimer, ou mesurer son apport hors banc avant de trancher |
| **`tri-transitif`** | Exact | **1/8** en logique | Étendre aux écarts chiffrés (KO connu `logique-05`) |
| **`logique` + formalisation 1B** | Solveur exact derrière une entrée en langage naturel | Dépend de la formalisation → **source de variance** | Améliorer le prompt/parseur, pas le solveur |
| **`potcode` + sandbox** | Le code faux ne sort jamais (asserts) | Deux appels LLM → lent | Réserver aux cas où le symbolique échoue (c'est déjà le cas) |
| **`rag-binaire`** | 9/100, ~5 ms, zéro hallucination | 259 documents seulement (compte vérifié) | Ajouter des faits (c'est le levier qualité le moins cher) |
| **Routeur TF-IDF + régression** | Aucun appel LLM (~ms) ; sous 300 exemples c'est une **LogisticRegression**, donc des **probas calibrées** — réutilisées comme poids du mix multi-LoRA | À partir de 300 exemples ça bascule en **LinearSVC** : marges plus larges mais **plus de probas calibrées** → poids LoRA moins fiables | Garder une tête calibrée (Platt/isotonic) sur la marge du SVC |
| **`raisonneur` (PoT + AST)** | Chaque calcul est **vérifié** par un AST, jamais cru sur parole | Plusieurs passes LLM → lent ; n'arrive qu'après les solveurs | Étendre l'extraction d'étapes à la famille « taux » (aujourd'hui couverte par `calcul-taux`) |
| **Porte de clarification** (`aura/clarification.py`) | Répond « je ne sais pas de quoi tu parles » sans générer ; supprime la classe « réponse hors-sujet sur question vide » (« Explique », « Écris un poème ») | N'ajoute **aucune** connaissance : elle évite une réponse fausse, elle n'en produit pas ; un faux positif coûte un aller-retour à l'utilisateur | Étendre les déclencheurs sur des cas réels manqués — **toujours par règles**, jamais par sonde LLM (cf. §3ter) |
| **Mémoire de conversation** | Suit ~4 tours, permet les questions contextuelles | Mange la fenêtre (`-c 4096`) et **gonfle le prefill** | Résumer les tours anciens au lieu de les garder verbatim |
| **Embeddings MiniLM ONNX INT8 (384-d)** | ~23 Mo, 100 % local, alimente le cache sémantique | Un vecteur de 384-d ne sépare pas deux questions proches → il a fallu un filtre `_hit_douteux` | Modèle d'embeddings plus fin, ou seuil **par catégorie** au lieu d'un 0,70 global |
| **API HTTP :8760 + GUI + Coucou** | Une seule porte d'entrée, découplée du moteur (l'île Coucou ne parle qu'à l'API) | Un process de plus à superviser (le superviseur du thème relance tout ensemble) | Exposer le **tier de provenance** dans la réponse API : le banc et le debug en ont besoin |
| **Budget 45 s / échéance** | Cadre la latence | **« riche » fait 37–40 s / 250 mots** → les KO « court » viennent de là | Arbitrage produit explicite : longueur vs latence, par catégorie |
| **Disjoncteur anti-boucle** | Coupe une boucle pendant le flux | N'existe que sur le chemin in-process (corrigé côté serveur par DRY) | Unifier |
| **`banc_100`** | 8 catégories, contrôle strict par nombre/mots | **13,7 h**, variance ±1-2, latences corrompues (`21675 s`), attribution manuelle | Sous-banc déterministe (~60 q, secondes) en garde-fou + banc long manuel |
| **`pytest` 1146 + `mypy`** | Filet réel, **73 modules typés, 0 erreur** [`§3septies`](#3septies-orchestrateur-éclaté-aiguilleur-nommé-et-ci-qui-vérifie-16-oct-2026) | `mypy` passe mais **ne vérifie pas les corps non typés** (`annotation-unchecked`) | Activer `--check-untyped-defs` sur le cœur |
| **Ordre des tests aléatoire** | Un état partagé entre tests ne peut plus se cacher dans l'ordre alphabétique des fichiers | Latence de suite plus variable ; le seed n'est pas figé (on ne rejoue pas le même ordre) | — |
| **`.aef` scellé** | Distribution en un fichier, intègre config, code et GGUF | Le GGUF reste aussi extrait dans `.cache_aef/cerveau.gguf` : le `.aef` et le cache occupent chacun l'espace des poids | Le format ne compresse pas les poids ; choisir le cerveau et quantifier le GGUF déterminent la taille |
| **Python 3.12 + uv** | Dépendances propres, venv rapide | — | Aucun : les solveurs sont à 0–5 ms, Rust n'apporterait rien |

---

## 3. Priorisation des évolutions

| # | Action | Gain attendu | Coût | Risque |
|---|---|---|---|---|
| 1 | **`--spec-type ngram-cache` → `ngram-mod` n-min 8** | supprime une perte de 16-42 % sur le texte structuré, garde le ×2,7 sur le répété | **fait** | Nul |
| 2 | **2ᵉ barrette 16 Go** | decode ~20 → ~30-35 tok/s (estimation) | ~30-50 € | Nul |
| 3 | **Registre d'experts + priorité déclarée** | ordre testable, ajout d'expert trivial | **fait** | Faible |
| 4 | **Solveurs exacts AVANT le cache flou** | supprime la classe « mémoire > raisonnement » | **fait** | Faible |
| 5 | **Sous-banc de provenance en garde-fou** | 100 questions en ~5 s au lieu de 13,7 h | **fait** | Nul |
| 6 | **Plus de couverture déterministe** | 66 → **73/100** (tri-transitif 1/8 → 4/8, chaînes numériques +4) | **fait** | Nul |
| 7 | **Unifier l'échantillonnage serveur/in-process** | supprime une classe de bugs (déjà 1 vu) + relance anti-boucle côté serveur | **fait** | Faible |
| 8 | **Routeur calibré au-delà de 300 exemples** | supprime « deux comportements pour un seuil » | **fait** | Faible |
| 9 | **Contrat de sortie par mode** | la contradiction 45 s / 900 caractères est calculée et affichée | **fait** | Nul |
| 10 | **Arbitrage produit `riche` : longueur vs latence** | 39,8 s requis pour 45 s de budget, 11,8 s de déficit au débit dégradé | **décision à prendre** | — |
| 11 | **Fusionner le cache de prompt** | inconnu | à mesurer | — |
| 12 | **`imatrix` sur le 1B** | qualité à taille égale | 1 passe de calibration | Faible |
| 13 | Régler `--spec-ngram-mod-n-min` (8 / 16 / 24 / 48) | inconnu | 1 mesure | Nul |
| 14 | **Sondes de capacité** (fast-lang répond mais le banc ne le teste pas) | verrouille les capacités hors banc | 10 lignes | Nul |
| 15 | **Porte de clarification** (question trop ouverte → 1 à 3 questions avant la réponse) | supprime la classe « le 1B invente un sujet » ; 0 interception sur les 100 questions du banc | **fait** | Faible (un faux positif = un aller-retour) |
| 16 | **Couverture déterministe, 2ᵉ vague** : `cube de N`, familles verbales (écart, groupes, chacun, même unité), `problemes-arithmetiques` (fractions, périmètre/aire) | **73 → 86/100**, générations **27 → 14** ; mesuré : 7,8 s (réponse FAUSSE) → **13 ms** (juste) | **fait** | Nul |
| 17 | **Injecter les 3 faits manquants dans l'index RAG** (année bissextile, côtés de l'hexagone, besoins des plantes) | +3 questions sans générer | à faire | Faible, mais touche `rag.bin` (artefact scellé) |

---

## 3bis. Les huit problèmes d'architecture : ce qui a été corrigé (6 oct. 2026)

| # | Problème | Traitement | Preuve |
|---|---|---|---|
| 1 | **L'ordre de la cascade était une suite de `if`** — on ne pouvait ni le relire, ni le tester, ni ajouter un expert sans risque | **Registre déclaratif** : `aura/registre_experts.py` (nom, genre, coût, source, *raison*). Le genre porte l'invariant : *aucun approximatif avant un exact*. La cascade **itère** sur le registre | `tests/test_registre_experts.py` — 7 tests, dont **l'ordre réellement exécuté comparé au registre** (il a détecté tout seul l'ajout de `chaines-numeriques`) |
| 2 | **Le flou passait avant l'exact** — le cache sémantique (seuil 0,70) répondait avant les solveurs | Étage séparé : `repondre` = exact, `repondre_cache` = flou. Le cache est consulté **après tous les solveurs** et après le savoir vérifié (rag) | `tests/test_ordre_exact_avant_flou.py` — **cache empoisonné** (réponse fausse sur la question exacte) : le solveur gagne. Et l'inverse aussi : cache muet ⇒ il sert toujours |
| 3 | **Le budget de latence n'était pas négocié** — 7 des 12 KO `riche` = réponses COURTES, silencieusement | **Contrat déclaré** (`aura/contrats.py`) : 900 car → 237 tok + 280 tok de plan = **39,8 s pour 45 s de budget**, et **11,8 s de déficit** au débit dégradé (9 tok/s). Journalisé + attaché à la réponse | `tests/test_contrats.py` ; le diagnostic s'affiche désormais sur les 6 questions `riche` du sous-banc (il était muet) |
| 4 | **Deux chemins d'inférence divergents** | **Source unique** : `aura/echantillonnage.py` (profils NORMAL / RELANCE / SECTION). Le DRY reste serveur-only, mais il est *déclaré* comme tel. La **relance anti-boucle**, qui n'existait que côté in-process, est ajoutée au serveur | `tests/test_echantillonnage.py` — dont un contrôle **statique** : plus aucune valeur de pénalité écrite en dur hors du module |
| 5 | **Le banc n'était pas un garde-fou** (13,7 h, variance ±1-2, latences corrompues) | **Sous-banc de provenance** : `scripts/sous_banc.py` — 100 questions en **~5 s**, génération et réseau coupés, mesure *qui* a répondu et compare à une référence | 0 question instable sur deux runs ; détecte `PERDU` / `GAGNE` / `CHANGE` / `VALEUR` |
| 6 | **Le routeur changeait de nature à 300 exemples** (probas calibrées → softmax de marges, qui servent de **poids du mix multi-LoRA**) | **Calibration de Platt** autour du `LinearSVC` : les deux régimes rendent une *probabilité* | `tests/test_routeur_calibration.py` — les deux régimes somment à 1 et répondent la même chose |
| 7 | **La capacité était couplée à la taille du modèle** (4 LoRA liés à Llama-3.2-1B) | **Ce n'est pas corrigeable** sans re-entraîner : c'est un choix d'architecture. Ce qui l'était : le couplage était **invisible** — rien n'empêchait de charger un 3B, et llama.cpp aurait produit du bruit *sans erreur*. Déclaration + refus au boot. **9 oct. 2026 — le cas s'est produit** : la bascule vers Gemma-3-1B (26 couches / dim 1152 / 4-1 têtes) a rendu les 4 adaptateurs Llama inutilisables, et c'est la **mesure** de l'en-tête GGUF (`aura/gguf_meta.py`) qui a prononcé le refus, pas une liste de noms. Ré-entraînement : `scripts/colab_lora_gemma.ipynb`. **Résolution 9 oct. 2026** : le verdict lit désormais l'architecture **de l'adaptateur lui-même** (son `general.architecture` GGUF) et la compare au cerveau chargé — plus de constante à resynchroniser. Déposer les `.gguf` Gemma dans `adaptateurs/` réactive le montage sans toucher au code, et les Llama restent refusés | `tests/test_couplage_architecture.py` |
| 8 | **Du code mort qui fait croire à des capacités** | **Correction de mon propre diagnostic** : `fast-lang` n'est PAS mort — il répond juste et en < 0,2 ms (« conjugue parler au présent »). Le **banc** ne pose aucune de ces questions. Le vrai problème est l'inverse : une capacité qui marche et que rien ne teste. `tri-transitif` : **1/8 → 4/8** | `chaines_numeriques` + `tri_transitif` : **4/4** de la famille numérique, **0 contamination** sur les 96 autres questions |

---

## 3ter. La porte de clarification — demander au lieu d'inventer (7 oct. 2026)

Un 1B ne dit jamais « ta question est trop vague ». Face à « Explique » ou « Écris un
poème », il **invente un sujet** et rédige 300 mots à côté. La réponse est fausse non parce
que le cerveau se trompe, mais parce que la demande ne portait aucun contenu — et un 1B n'a
ni le contexte ni la prudence qu'il faudrait pour deviner ou pour demander.

`aura/clarification.py` ajoute une **porte** : quand la question est trop ouverte, RontoGPT pose
**1 à 3 questions** et **ne répond pas**. Le tour suivant apporte la précision ; la question
d'origine et la précision sont **recollées** et le pipeline repart — cette fois avec un sujet.

| Décision | Pourquoi |
|---|---|
| **Juge déterministe, pas le LLM** | Le juge d'un petit modèle est mauvais (c'est exactement ce qui a produit `_CAPACITE`) ; sonder le 1B coûterait ~0,5 s **à chaque question non résolue** et ajouterait du bruit à une décision qui doit être relisible |
| **Position `apres-experts`** (déclarée dans `registre_experts.PORTES`) | Une question que le niveau 0 ou `calcul_taux` savent trancher ne doit **jamais** recevoir « précise ta demande » — c'est le bug du cache sémantique (6 oct. 2026), à l'envers |
| **Un seul tour** (`Suivi`) | Une porte qui se repose indéfiniment est un questionnaire, pas un assistant. Après la relance, on répond avec ce qu'on a |
| **Objets génériques et renvois déclenchés seulement sans sujet en conversation** | Après « parlons des volcans », « Écris un poème » désigne évidemment les volcans. Prendre la suite d'un échange pour une question vide serait le pire faux positif : celui qui fait perdre du temps à quelqu'un **qui vient de parler** |
| **Recollage par concaténation** | Sur « Convertis 100 » + « en km », une fusion enrichie (« Précision : en km ») rendrait la phrase **incalculable** par `pal.py` : le calcul redeviendrait une génération |
| **Jamais écrite au cache** | Une relance n'est pas une réponse : elle n'est pas enregistrée, sinon le cache la resservirait un jour comme réponse |

**Déclencheurs** (agressivité « moyenne ») : `verbe-nu` (« Explique », « Aide-moi »),
`objet-generique` (« Écris un poème », « Fais un plan »), `renvoi-sans-sujet` (« Explique ça »,
« Et ensuite ? »), `conversion-incomplete` (« Convertis 100 » sans unité cible) et
`mot-de-passe` (« Génère un mot de passe » sans longueur ni critères).

**Ce qui ne déclenche jamais** : « raconte moi une blague » (le sujet n'ajoute rien),
« Écris un poème sur la mer », « Résume la Révolution française » — et les 100 questions de
`banc_100.py`. Coupure : `AURA_CLARIFICATION=0`.

La réponse porte `cerveau_choisi = "clarification"`, `en_attente = true` et un bloc
`clarification` (`famille`, `manque`, `questions`), exposés tels quels par `/v1/ask` et la
GUI web : un front peut marquer l'état « question en attente » au lieu de présenter une
relance comme une réponse.

**Preuve** : `tests/test_clarification.py` — dont **les 100 questions du banc, en 2
contextes, ne sont interceptées 0 fois** ; et `tests/test_registre_experts.py` — **l'ordre
réellement exécuté** : un expert qui sait répondre gagne, même sur « Explique ».

---

## 3quater. Deuxième vague symbolique — et un chiffre faux (7 oct. 2026)

Objectif donné : **utiliser au maximum le moteur symbolique, moins le LLM**. La méthode n'a pas
changé : on part du **sous-banc de provenance**, qui liste les questions tombant encore sur le
LLM. Il en restait 27 ; parmi elles, **13 étaient du calcul pur** que le lexique ne couvrait pas.

| Famille | Exemple | Où |
|---|---|---|
| `cube de N` | « Quel est le cube de 7 ? » → 343 | `filtre_instantane` (le « carré de N » voisin était déjà exact) |
| écart | « 14 ans et 9 ans, quel est l'écart ? » → 5 | `calcul_verbal` |
| groupes | « 6 rangées de 7 chaises » → 42 ; « 3 fois 8 » → 24 | `calcul_verbal` (réduction **avant** le balayage : « 5 boîtes de 12 œufs, j'en casse 7 » → 53) |
| chacun | « 9 élèves et chacun a 4 cahiers » → 36 | `calcul_verbal` |
| même unité | « 18 km puis 9 km, combien en tout ? » → 27 | `calcul_verbal` |
| `j'en distribue 23` | retrait | `calcul_verbal` (le pronom « en » est **exigé** : « distribue 12 bonbons entre 4 » est une division) |
| fractions en mots | « 60 litres aux deux tiers » → 40 | `problemes_arithmetiques` (`Fraction` exact, sinon 60×(2/3) = 39,999…) |
| géométrie | « 7 cm de long et 4 cm de large » → périmètre 22 / aire 28 | `problemes_arithmetiques` |

**Résultat** : 73 → **86/100** sans générer, générations **27 → 14**. `JUSTE` = 86/100 : rien
n'est couvert sans être exact.

Mesure directe sur le moteur de production (même question, solveurs neutralisés puis actifs) :

| Question | Avant (LLM) | Après (symbolique) |
|---|---|---|
| « Quel est le cube de 7 ? » | 7,1 s → 343 | **2 ms** → 343 |
| « 18 km puis 9 km, combien en tout ? » | 8,2 s → réponse verbeuse | **0,44 s** → 27 |
| « 60 litres aux deux tiers » | 7,8 s → « environ **40,67** litres » (**faux**) | **13 ms** → 40 |

La troisième ligne est le point important : le LLM avait **la bonne valeur intermédiaire** et
concluait quand même à 40,67. Le symbolique ne gagne pas seulement du temps, il supprime une
réponse fausse.

### Le chiffre faux, et pourquoi il faut le raconter

La première référence écrite après ces ajouts annonçait **97/100**. C'était un artefact : pour
supprimer une duplication, `sous_banc` avait été amené à **dériver** la liste des tiers exacts
du registre (`_tiers_exacts()`), mais l'appel était placé **au niveau module** — donc importé
**avant** que `main()` ne pose `AURA_FILTRE_CACHE`. Le banc figeait `filtre_instantane._FICHIER`
sur le **vrai cache de production** et se répondait à lui-même : 11 questions servies par la
mémoire du projet, plus vite et sans rien démontrer.

Deux corrections :
1. la dérivation est **paresseuse** (appelée dans `main()`, après l'isolation), verrouillée par
   `tests/test_registre_experts.py::TestCoherenceDuSousBanc` ;
2. le banc **n'écrit plus** dans le cache pendant un run (`enregistrer` neutralisé) : `--cache`
   sert ses *hits*, jamais les réponses que le banc vient de fabriquer. C'est ce qui avait fait
   dire à `riche-02`..`riche-10` qu'ils étaient « couverts » (ils se répondaient la sentinelle
   `[GENERE]` en 1 ms, cosinus ≥ 0,70).

Leçon générale, valable pour tout ce dossier : **une liste recopiée est une source de vérité
cachée ; une liste dérivée est un effet de bord d'import.** Les deux se paient.

---

## 3quinquies. Qualité du cerveau brut, mesurée pour de vrai (10 oct. 2026)

Question posée : « le cerveau vaut quoi **tout seul** ? » — c'est-à-dire sans le pipeline
symbolique, sur les compétences qui vivent dans les poids. Réponse par `lm-eval` 0.4.13 via
`scripts/lm_eval_aura.py`, contre le serveur llama.cpp b7400 sur `127.0.0.1:8757`.

**Ce qui est mesurable** (tâches génératives, `generate_until`) :

| Tâche | n-shot | Filtre | Score | Stderr | Échantillon |
|---|---|---|---|---|---|
| gsm8k | 5 | strict-match | **0,16** | ± 0,0368 | 100 (seed 0) |
| gsm8k | 5 | flexible-extract | **0,20** | ± 0,0402 | 100 (seed 0) |

Run : `local-chat-completions`, `--apply_chat_template`, `tokenizer_backend=None`,
`num_concurrent=1`, **2543 s** pour 100 questions sur 1319. Les deux essais à `--limit 10`
donnaient 0,2 puis 0,3 : même ordre, c'est l'échantillon de 100 qui compte.

**Pourquoi ce n'est pas un bug du banc.** Reproduit à la main avec **exactement** le format que
le banc envoie (5 exemples en tours user/assistant alternés, `temperature=0`) : le modèle
**produit bien** un `#### N` final — 5 réponses sur 5 bien formées — et se trompe sur le
raisonnement :

| Question | Attendu | Réponse du modèle |
|---|---|---|
| robe, 2 m de fibre bleue + moitié de blanche | 3 | « 2 bolts of blue fiber = 2*2=4 … Total = 4+4 = **8** » |
| maison 80 000 $, +150 %, 50 000 $ de travaux | 70 000 | « 80,000 * 1.50 = 120,000 … profit is 120,000 - 80,000 = **40 000** » (travaux oubliés) |

Le format est bon, l'arithmétique est fausse. Le score est bas **pour de vrai**, il ne vient pas
d'une extraction ratée (même 0,2 sur les 5 cas refaits à la main). Ce n'est donc pas `parse_logprobs`
ni la regex des filtres qu'il faut aller corriger.

**Ce que ça veut dire.** 0,16–0,20 en 5-shot est loin des **62,8 %** de GSM8K 8-shot CoT publiés
pour Gemma 3 1B *officiel* (recoupement Embedl, mars 2025). L'écart ne s'explique pas seulement
par le protocole : le poids embarqué est un **Q4_K_M** d'un **fine-tune abliteré**
(« Heretic-Uncensored-Thinking »), pas le modèle d'origine. Conclusion à garder telle quelle :
**le cerveau brut n'est pas le raisonneur du système — le pipeline l'est.** C'est enfin chiffré.

**Ce qui n'est PAS mesurable, et pourquoi on refuse le chiffre.** Toutes les tâches à choix
multiple (`arc_easy`, `hellaswag`, `mmlu`, `winogrande`, `piqa`, `boolq`) passent par des
**logprobs du prompt**, que llama.cpp n'expose pas : `echo:true, logprobs:N` ne renvoie que les
tokens **générés** (`choices[0].logprobs.content`), là où lm-eval attend
`logprobs["token_logprobs"]`. Cause amont : issue **ggml-org/llama.cpp#27174** (16 août 2026,
*closed as not planned*). Les scores obtenus par cette voie sont du **hasard silencieux**
(arc_challenge 0,2433 pour 0,25 de chance ; winogrande 0,500 ; piqa 0,4933) : les publier
serait inventer. D'où le choix du pont : **refus explicite** (`exit=2`, message citant l'issue)
plutôt qu'un chiffre faux — et une échappatoire assumée, `--loglikelihood`, qui force le run en
prévenant que les chiffres **ne sont pas valides**.

**Résolu le 10 oct. 2026.** `mmlu` est un **GROUPE** (`_mmlu.yaml` porte `group: mmlu`), pas
un fichier de tâche : chercher `mmlu.yaml` seul le rendait invisible, donc il passait le refus.
La résolution suit maintenant le nom **DÉCLARÉ** (`group:` / `task:`) avant le nom de fichier,
puis `include:` vers le template, puis les sous-tâches, puis les étiquettes (`tag:`). Le nom
déclaré doit primer : **trois** fichiers s'appellent `_mmlu_stem.yaml` (variantes llama3,
llama3_cot, et le vrai mmlu) et le tri par chemin tombait sur une variante qui note en
génération, faisant passer un groupe à choix multiple pour mesurable.

Corollaire : `winogrande` et `boolq` sont désormais refusés eux aussi — ils ne l'étaient pas,
et la liste ci-dessus l'annonçait à tort. État vérifié : refusés = `mmlu`, `mmlu_stem`,
`mmlu_humanities`, `mmlu_abstract_algebra`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`,
`winogrande`, `boolq`, `lambada_openai` ; mesurable = `gsm8k` ; inconnue = laissée passer.
Résolution en **0,5 s** (index bâti une fois) contre 264 s sans les caches.

---

## 3sexies. RontoGPT complet sur GSM8K — le pipeline n'aide pas, et une bêtise est trouvée (10 oct. 2026)

Question posée : le pipeline **rattrape-t-il** le cerveau ? Meme banc que § 3quinquies, mais
par la porte d'entree reelle — `POST /v1/ask` sur l'API du projet — avec
`scripts/mesure_gsm8k_aura.py`. Les 100 memes questions, les deux memes filtres d'extraction.

| Mesure | Cerveau nu (`lm_eval`) | RontoGPT complet (`/v1/ask`) |
|---|---|---|
| strict-match | 0,16 | **0,00** |
| flexible-extract | 0,20 | **0,12** |
| servi sans generation | 0 % | **9 %** (7 instantane + 2 clarification) |

Latence : mediane **17,7 s**, p90 27,0 s, max 36,5 s. 100/100 sans erreur d'API.
**Ce chiffre etait gonfle** — il a ete pris pendant que d'autres commandes tournaient sur la
machine. Re-mesure au repos (20 questions, rien d'autre en cours, meme API isolee) : mediane
**9,1 s**, p90 11,2 s, plus un demarrage a froid a 88,6 s sur la premiere question. Les deux
echantillons different (20 contre 100 questions), donc l'ecart n'est pas une mesure de la
contention seule — mais il va dans ce sens, et **9,1 s est le chiffre a retenir**.

### Le 0,00 strict-match ne veut rien dire — et il faut le dire avant de s'en servir

**0 reponse sur 91 contient `####`.** RontoGPT est un assistant conversationnel francais : elle
reformule l'enonce en prose et conclut par une phrase, jamais par le marqueur de completion
GSM8K. Le filtre `strict-match` cherche `#### (\-?[0-9\.\,]+)` : il ne peut pas trouver.
**Seul `flexible-extract` est comparable**, et il donne 0,12 contre 0,20.

### Donc RontoGPT fait moins bien que son cerveau — mais l'ecart n'a pas la cause qu'on croit

91/100 sont partis en generation : RontoGPT presente au modele un contrat francais d'assistant,
la ou `lm-eval` lui donnait 5 exemples anglais de GSM8K. **Le modele est le meme, la consigne
non.** Les 9 restants se repartissent en 7 reponses symboliques et 2 demandes de clarification
(la porte de clarification a juge l'enonce trop ouvert : ce n'est pas une reponse fausse, c'est
un refus — mais le banc le compte faux, comme il doit).

Conclusion utilisable, et elle est modeste : **sur GSM8K, le pipeline ne rattrape rien.** Il
ne couvre pas ces enonces, et l'interface conversationnelle ne rend pas le cerveau meilleur.
Cela ne dit rien des taches ou le pipeline EST la reponse (§ 3quater : 86/100 sur le banc
maison) — cela delimite son domaine.

### La betise trouvee : `_calcul_direct()` repond a des phrases qui ne sont pas des calculs

Les 7 reponses symboliques sont **toutes fausses**, et toutes pour la meme raison. Les 7
enonces contiennent une fraction ecrite `a/b`, et la derniere regle de
`filtre_instantane._calcul_direct` cherche une expression arithmetique **n'importe ou dans la
phrase**, sans exiger que la question soit un calcul :

| Enonce (extrait) | Attendu | RontoGPT a repondu |
|---|---|---|
| « Each floor contains 8 units, and **3/4** of the building is occupied » | 30 | **0,75** |
| « his comic books weigh **1/4** pound each » | 15 | **0,25** |
| « Two girls each got **1/6** of the 24 liters » | 10 | **0,166667** |
| « **2/5** are boys and **2/3** of the girls are in the girl scout » | 40 | **0,4** |

Preuve minimale, reproductible sans serveur ni LLM :

```python
_calcul_direct("I paid 3/4 of the price yesterday.")   # -> '0.75'
_calcul_direct("The recipe needs 1/2 cup of sugar and 2 eggs.")  # -> '0.5'
```

Le motif fautif est le dernier du fichier — celui commente « expression arithmetique brute :
« calcule 45*12+3 » ». L'intention etait un calcul DICTE ; le motif matche en realite toute
suite `nombre op nombre` rencontree dans une phrase. **C'est grave au-dela de GSM8K** : la
couche « exacte » est censee ne jamais se tromper, et ici elle repond faux, en 0,02 s, avec
l'aplomb d'un resultat prouve — pendant que le projet affirme (§ 3quater) que le symbolique
« supprime une reponse fausse ». **Corrige le 10 oct. 2026.** La regle exige desormais que la question **SOIT**
l'expression : il ne doit rester aucune lettre une fois l'expression retiree. Verifie : les
13 cas du contrat de `tests/test_critic_maths.py` passent, les 7 questions GSM8K ne sont plus
capturees (`filtre_instantane.repondre` renvoie None sur les 7), et « 2+2 » / « 12 * 4 »
restent calcules. Piege attrape au passage : `combien font` (pluriel) manquait dans
`_DEMANDE_MATH`, qui ne declarait que le singulier « combien fait ». Poser la garde seule
aurait donc casse « combien font 2+2 ? » — les deux corrections vont ensemble, et un test de
non-regression couvre chacun des deux sens.

### Un contre-controle qui a echoue, et pourquoi il est ecrit quand meme

`/metrics` renvoie **501** sur ce build : impossible de compter les generations cote serveur
par ce chemin. J'ai donc pris le compteur `id_task` de `/slots`, lu avant et apres le run.
Resultat annonce : **28 625** generations pour 100 questions — absurde. Verifie par une mesure
controlee (une seule requete, delta des slots) : **1 requete -> 39 increments** de `id_task`.
Ce n'est pas un compteur de requetes, c'est un identifiant interne de tache. Le
contre-controle etait **invalide**, pas les donnees : la part generee repose uniquement sur
l'etiquette `cerveau`. Lecon a garder : un temoin exterieur ne vaut que si sa semantique est
verifiee sur un cas connu avant le run.

---

## 3septies. Orchestrateur éclaté, aiguilleur nommé et CI qui vérifie (16 oct. 2026)

Le dépôt partait d'un `main` **rouge** : 12 tests échoués et un badge « mypy : 0 »
contredit par **57 erreurs**. Les deux sont réparés à la cause, et l'invariant est
désormais **vérifié par la CI** au lieu d'être annoncé.

### L'orchestrateur est éclaté en étages, avec un contrat typé

`aura/orchestrateur.py` (178 lignes) est maintenant une **façade** :

| Module | Rôle |
|---|---|
| `pipeline_rontogpt.py` | `executer` / `executer_detaille`, cache de réponses, branchement de l'aiguilleur |
| `experts_rontogpt.py` | validation avant exécution, fan-out parallèle, routage des experts |
| `raisonnement_rontogpt.py` | logique (mini-SAT + vote), PoT/arbre de réflexion, code |
| `synthese_rontogpt.py` | prompt système, garde-fous de contamination, `_generer` |
| `verification_rontogpt.py` | vérification outillée (pattern CRITIC) |

Ces mixins s'appellent entre eux et partagent l'état porté par
`RontoGPT.__init__` : chacun déclare ce contrat sous `if TYPE_CHECKING` (état de
l'instance et méthodes des mixins frères, en `Callable`). Le bloc **disparaît à
l'exécution** — aucun import circulaire, aucune dépendance au runtime — et c'est
ce qui ramène `mypy aura/` à **0 erreur sur 73 modules**.

### L'aiguilleur est un module, pas un `if` dans l'appel HTTP

La décision d'escalade vivait dans `aura/gemini.py`, mélangée à l'appel réseau.
Elle est maintenant dans **`aura/aiguilleur.py`** : un composant qui ne parle à
personne et rend une `Decision` (`escalader`, `raison`, `moment`, `fournisseur`).
Il lit, dans cet ordre : la **tâche demandée** (rédaction longue, traduction —
compétences que le 1B n'a pas, donc escalade même si sa réponse est longue et
bien formée), puis la **réponse rendue** (vide, erreur, refus, aveu, boucle,
écho, trop court, mauvaise langue, problème, fait deviné sans source, hors
domaine). Un **tier exact ou vérifié n'est jamais rejugé** : une preuve ne se
remplace pas par une réponse probable. `gemini.py` ne garde que l'appel, plus
un `disponible()` (clé présente, pas en pause) que l'aiguilleur consulte — sans
cible disponible, **rien ne part**.

### Les 12 échecs : 4 bugs, 8 fantômes

* **8 échecs n'existaient que dans la suite complète.** Cause unique :
  `cache_reponses._CACHE_REPONSES`, un dict **de module**, resservait au test
  suivant la réponse mémorisée par le précédent dès que deux tests posaient la
  même question. Une fixture autouse le vide (et l'isolation de
  `filtre_instantane` était déjà en place) : *le pipeline n'était pas en cause*.
* **4 échecs réels** : le compresseur RAG n'appliquait son plafond de morceaux
  qu'aux sources **longues** (8 lignes courtes tenaient encore 8 extraits dans
  la fenêtre du 1B) ; et trois tests contredisaient le contrat **courant** du
  code (le solveur rédige la solution logique, le 1B ne fait que formaliser ;
  un calcul n'est plus chargé depuis le cache) — ce sont les tests qui ont été
  réalignés, pas le pipeline.
* **Ordre aléatoire** (`pytest-randomly`, dans le groupe `dev` et donc en CI) :
  la suite est verte sur plusieurs seeds, et cette classe de régression ne peut
  plus revenir en silence.

### La CI installe les MÉTADONNÉES, pas une liste recopiée

Quatre jobs : `tests` (Ubuntu + Windows × Python 3.11/3.12), `chaine-aef`
(forge → signature Ed25519 → boot → altération), **`typage`** (`mypy aura/`) et
`installateur` (syntaxe PowerShell). Les dépendances viennent de
`[project].dependencies` via **`uv sync --frozen`**, et `uv lock --check`
refuse un `pyproject.toml` désynchronisé du verrou. La liste écrite à la main
qui vivait dans le workflow est supprimée : c'est elle qui avait laissé
**`google-adk`** (dépendance déclarée du projet) hors de la CI, où
`tests/test_agents_adk.py` échouait à l'import. `uv.lock` est versionné, donc
les versions sont reproductibles.

**Le fichier de workflow ne se lisait même pas.** Le libellé d'une étape
contenait un « `: ` » dans un scalaire non quoté (`Tests (ordre ALEATOIRE : …)`) :
en YAML, un deux-points suivi d'une espace termine une clé, donc le fichier
était **invalide** et GitHub rejetait le workflow ENTIER — les quatre jobs
faisaient semblant d'exister. Réparé (libellé quoté) et vérifié en le
**chargeant** (`yaml.safe_load`), pas en le relisant : la syntaxe d'un YAML ne
se juge pas à l'œil.

### Le paquet livré contient enfin ses données

La wheel ne déclarait que `aura` + `rontogpt` : ni `concepts.toml`, ni
`adaptateurs/`, ni `tui/`, alors que le code les lit **par le parent du paquet**
(`Path(__file__).parent.parent`). `pip install` livrait donc un RontoGPT
dégradé. `only-include` + `exclude` les embarquent à la racine de la wheel
(vérifié : la wheel extraite résout les trois chemins), en écartant
`tui/node_modules` et les adaptateurs Llama archivés.

### Un seul point d'entrée produit

Le superviseur vivait dans un `main.py` **hors paquet**, sans test ni
référence : il est fusionné dans la CLI (`python -m aura --superviser`,
`--superviser --api` pour l'API seule) et couvert par
tests/test_superviseur.py. `relancer.ps1` reste ce qu'il est — une commodité de
**développement Windows** (libère le port, vérifie le design, relance), et il
est enfin **documenté** au lieu d'être un fichier que personne ne cite.

### Et le plus gros module a perdu son post-traitement

`llama_cerveau.py` (1 885 lignes) mêlait le **chargement du modèle** à deux
garde-fous de **texte** : le disjoncteur anti-répétition et la canalisation
(le 1B récitait son prompt). Ces ~290 lignes sont **pures** (`texte -> texte`
ou `texte -> bool`, aucun état de module) : elles vivent dans
`aura/post_traitement.py` et sont **ré-exportées** par `llama_cerveau`, dont
l'espace de noms reste l'API historique — `llama_cerveau._en_boucle(...)` et
les monkeypatches des tests continuent de fonctionner. Le module de cerveau
est passé de 1 885 à 1 612 lignes, sans une ligne réécrite (le bloc déplacé
est **verbatim**, ce qui est vérifiable dans le diff).

---

## 4. Ce qu'il ne faut PAS faire

| Piste | Pourquoi |
|---|---|
| **ONNX Runtime / OpenVINO** | Pas de NPU (`ComputeAccelerator = 0`). Perdrait GGUF + LoRA + `.aef` pour un gain non démontré |
| **Réécrire l'orchestrateur en Rust** | Les solveurs sont à **0–5 ms** ; le coût est la génération. Gain non mesurable |
| **Passer à 3B** | 5,3 tok/s (mesure antérieure) → ~2 min pour 250 mots, contre 45 s de budget. **Casse les 4 LoRA** |
| **Fallback cloud** | Contredit « 100 % local » |
| **Remplacer llama.cpp** | C'est le bon moteur ici : Vulkan + multi-LoRA + GBNF, déjà au bout du matériel |
| **Forcer un brouillon spéculatif par modèle** | Mesuré **négatif** sur cette machine (17,0 → 14,3 tok/s) |

---

## 5. Le modèle symbolique peut-il « calculer les mots probables » plus vite qu'un LLM ?

C'est la question « et si on remplaçait la génération par du comptage de n-grammes ». Mesuré,
pas supposé. Le banc est `scripts/mesure_lookup.py` (résultats : `serveur/mesure_lookup_*.json`),
dérive contrôlée en A-B-C-A (`off` reproduit à ±1-2 %).

### 5.1 Comme GÉNÉRATEUR : non, et c'est structurel

Modèles n-grammes entraînés sur le corpus réel d'RontoGPT (8 910 mots, 1 649 types), génération
gloutonne argmax (= « le mot le plus probable, calculé par les maths ») :

| Ordre | Sortie | Constat |
|---|---|---|
| n = 2 | `... tous les A sont des B et aucun B n'est un C, il est également possible de conclure que tous les ...` | **39 mots distincts sur 200 générés**, le mot le plus fréquent revient **20×** → cycle de 5 mots |
| n = 3 | `... Mia est plus lente que Tom, donc Mia est plus lente que Tom, donc ...` | **26 distincts sur 200**, répétition **26×** → cycle de 9 mots |
| n = 4 | `... Analysons la situation : * Mia est plus lente que Tom ...` | **Recopie verbatim** d'une phrase du corpus (mémorisation, pas génération) |

Un modèle de Markov d'ordre fini ne peut pas représenter une contrainte **longue portée**
(répondre à *cette* question). Il fait l'un ou l'autre : boucler, ou recracher son corpus.
Ce n'est pas un manque de données, c'est la forme du modèle.

### 5.2 Comme BROUILLON vérifié : oui, mais uniquement sur le texte répété

Le n-gramme peut proposer, le LLM **vérifie** : le résultat a exactement la distribution du
LLM (qualité préservée), seul le temps change. Le taux d'acceptation mesure exactement
« la fréquence à laquelle le mot calculé par comptage est celui que le LLM valide » :

| Régime | Acceptation mesurée | Effet sur le débit |
|---|---|---|
| Texte **nouveau** (dissertations du banc) | **0 à 45 %** | **aucun gain** (−18 % en moyenne pour `ngram-cache` sur 4 questions `riche`) |
| Texte **structuré** (plan de passe 1, listes, code) | **0 à 2 %** | **−16 à −42 %** |
| Texte **répété** (mêmes phrases, listes identiques) | **100 %** | **×1,35 à ×2,7** |

Conclusion honnête : le comptage **ne peut pas** dépasser le LLM sur du texte neuf — l'acceptation
est trop basse, et ce plafond est mathématique, pas matériel. Il gagne uniquement là où la sortie
est déjà dans le contexte.

**Décision appliquée** : `--spec-type ngram-cache` → `ngram-mod --spec-ngram-mod-n-min 8`.
Sur plan / liste / code, `ngram-mod` ne propose **aucun** token brouillon (donc calcul identique
à `off`, perte nulle par construction) ; sur le texte répété il accepte 100 % et fait ×2,7.
C'est donc un gain **sans risque**, là où `ngram-cache` était une perte réelle.

### 5.3 Le vrai « calculer au lieu de générer »

Le comptage de mots rapporte **×2,7 sur la partie répétitive**. La couche déterministe, elle,
rapporte **×10 000** : 5 ms au lieu de 37 s, et sur **59 %** des questions du banc. Le « calcul
par les maths » qui fait gagner n'est donc pas la statistique des mots, c'est la **solution exacte
du problème** (arithmétique, logique, SAT, extraction RAG) — voir §3, lignes 3 et 5.

---

## 6. En une phrase

Le moteur et le runtime sont **déjà au plafond du matériel**, et le brouillon statistique **ne peut
pas** dépasser le LLM sur du texte neuf (acceptation 0-45 % : plafond mathématique, pas matériel).
L'ordre de la cascade n'est plus une suite de `if` mais un **registre déclaré et vérifié**, et
**86/100** des questions se résolvent sans générer. Le levier qui rapporte le plus est de déplacer
des questions de *générer* vers *résoudre* (7,8 s → 13 ms, et la réponse devient **juste**) ; la
**2ᵉ barrette de RAM** relève le plafond de débit.

**Une décision reste ouverte, et elle n'est pas technique** : le mode `riche` promet 900 caractères
dans un budget de 45 s, alors que les deux passes coûtent **39,8 s au débit nominal** et
**51,8 s au débit dégradé** (mesuré sous contention). Il faut soit allonger le budget, soit
raccourcir le contrat — `aura/contrats.py` chiffre les deux et le dit à chaque dissertation,
au lieu de livrer une réponse courte sans prévenir.
