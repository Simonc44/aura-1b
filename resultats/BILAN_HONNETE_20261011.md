# Bilan honnete Aura-1B — 11 octobre 2026

Tous les chiffres de cette note viennent des fichiers de `resultats/` et des
`.json` de banc a la racine. Quand un chiffre est douteux, c'est ecrit.
Quand un chiffre ne mesure rien, c'est ecrit aussi.

---

## 1. Le chiffre a ne PAS citer : le 100/100 du 9 octobre

`banc_100.gemma.json` annonce **100/100** en **83,9 s** (0,84 s/question).
C'est faux comme mesure : 7 des 10 questions `riche` (les seules qui exigent
une generation longue) ont ete **servies par le cache semantique en 0 s**.
Preuve, champ `cerveau` question par question :

```text
01:27s/gemma3-1b  02:0s/cache-sem  03:0s/cache-sem  04:0s/cache-sem
05:0s/cache-sem   06:27s/gemma3-1b 07:0s/cache-sem  08:27s/gemma3-1b
09:0s/cache-sem   10:0s/cache-sem
```

Ce run n'a pas non plus de cle `provenance` (ajoutee apres).
**Il ne faut plus le publier.** Tous les runs citables ont ete faits cache neuf.

## 2. Banc maison FR (100 questions) — l'etat reel

| Run | date | score | verifiable | forme | duree | riche | cerveau LLM | cache |
|---|---|---|---|---|---|---|---|---|
| `banc_100.gemma.json` | 09/10 22:59 | 100/100 | 85/85 | 15/15 | 83,9 s | 10/10 | gemma | **7/10 en cache** |
| `banc_100.gemma.ab.json` | 10/10 01:32 | 96/100 | 85/85 | 11/15 | 291,1 s | 6/10 | gemma | non |
| `banc_100.llama.ab.json` | 10/10 01:40 | 97/100 | 85/85 | 12/15 | 443,2 s | 7/10 | llama | non |
| `banc_100.gemma.postfix.json` | 11/10 02:02 | 92/100 | 85/85 | 7/15 | 120,8 s | 2/10 | gemma 3/11 | non |
| **`banc_100.gemma.redaction.json`** | **11/10 10:42** | **98/100** | **85/85** | **13/15** | **216,8 s** | **8/10** | **gemma 9/11** | **non** |
| `resultats/diagnostic_banc_pollue_..` | 11/10 15:34 | 89/100 | 84/85 | 5/15 | 116,6 s | 0/10 | gemma 0/11 | non |
| `resultats/baseline_aura_.._isole` | 11/10 15:44 | 92/100 | 84/85 | 8/15 | 412,8 s | 3/10 | gemma 3/11 | non |
| `resultats/aura_apres_correctifs_..` | 11/10 15:59 | 94/100 | 85/85 | 9/15 | 347,6 s | 4/10 | gemma 4/10 | non |

**Le meilleur chiffre defendable est donc 98/100** (11/10 10:42, 216,8 s,
cache neuf), pas 100/100.

### 2.1 Mais 85 des 100 points ne viennent PAS du modele

Provenance du run a 98/100 :

```text
niveau0-instantane 31/31      calcul-verbal 16/16     rag-binaire 9/9
calcul-achat 6/6              calcul-taux 5/5         constante 5/5
chaines-numeriques 4/4        tri-transitif 4/4       micro-faits 4/4
problemes-arithmetiques 3/3   constante-capacite 2/2
gemma3-1b 9/11   <-- le LLM
```

Le score se decompose en **85 points verifiables** (couches deterministes :
filtre instantane, calcul verbal, calculs d'achat/taux, RAG) et **15 points de
"forme"** (5 format + 10 riche). Le LLM ne porte que **11 questions sur 100**.

Autrement dit : **"98/100" mesure surtout l'orchestration, pas l'intelligence
du modele.** C'est une force du produit (0 token, 0 s sur 85 % des questions)
et une faiblesse de la mesure (le cerveau est presque invisible).

### 2.2 Les 2 echecs restants sont des LONGUEURS, pas du non-sens

`riche-01` (709 car) et `riche-03` (527 car), jugees `KO:court`.
La regle est `_defauts_riche` (`scripts/bench_qualite.py`) :
`if len(reponse) <= 900: defauts.append("court")`, parce que la consigne
demande 3 parties de 60-110 mots (~1100-2000 car).

Le contenu, lui, est correct et structure (`riche-03` a bien ses trois
parties). **Le budget n'est pas en cause** : mesure instrumentee de
`budget_restant(deadline, 320)` -> renvoie **320**, aucune coupe.

## 3. GSM8K — le seul banc standard disponible

### 3.1 Cerveau nu (lm-eval, `resultats/*/results_*.json`)

| Protocole | n | strict | flexible | erreur std |
|---|---|---|---|---|
| gsm8k 5-shot | 100 | **0,16** | **0,20** | +-0,037 / +-0,040 |
| gsm8k 5-shot | 10 | 0,20 puis 0,30 | idem | +-0,13 / +-0,15 (bruit) |
| gsm8k_cot 8-shot | 50 | **0,12** | **0,14** | +-0,046 / +-0,050 |

Pour memoire : le **8-shot CoT est PIRE que le 5-shot simple** (0,12 vs 0,16).

References externes : la carte officielle de `gemma-3-1b-it` annonce **62,8**
en GSM8K 0-shot ; le harness d'EleutherAI (issue #3258, ouverte) mesure
**22,2 flexible / 2,5 strict** sur ce genre de modele.

Reserve honnete : le fichier de comparaison nomme le cerveau charge
`Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q4_k_m.gguf`,
qui **n'est pas** le `gemma-3-1b-it` officiel de la carte. Une part de l'ecart
peut venir du checkpoint, pas seulement du protocole.

### 3.2 Via le pipeline `/v1/ask` (dialogue FR)

`resultats/gsm8k_aura/gsm8k_aura.jsonl`, 100 enonces :

- **strict = 0/100 = 0,00** — mais ce 0 ne mesure rien : **0/100 reponses
  contiennent `####`**. Le filtre strict est structurellement insatisfiable.
- **flexible = 12/100 = 0,12**
- latence mediane **17,7 s**/enonce (moyenne 17,1 s, max 36,5 s)
- routage : `gemma3-1b` 91, `niveau0-instantane` 7, `clarification` 2
- 0 reponse vide

### 3.3 L'experience CoT qui a echoue (et a ete annulee)

Forcer un budget de raisonnement (320 tokens) sur le 1B :

| | avant | apres patch |
|---|---|---|
| flexible | 0,12 | **0,03** |
| reponses vides | 0 | **37/100** |
| latence mediane | 17,7 s | 13,65 s |

Le 1B derive et echoie le prompt systeme. **Tout a ete annule** (fichiers
restaures). Traces : `resultats/gsm8k_aura/gsm8k_aura_patch.jsonl`
(flexible 2/200, 128 vides) et `NOTE_experience_budget_reverte.md`.
Cela recoupe Wei et al. 2022 (arXiv:2201.11903) : le CoT est **emergent**,
il **degrade** les petits modeles (chaines "fluent but illogical").

## 4. Le cerveau est-il le meilleur ? (gemma vs llama vs R1)

### 4.1 gemma-1B vs llama-1B — meme pipeline, 100 questions, cache neuf

| | gemma | llama |
|---|---|---|
| score | 96/100 | 97/100 |
| verifiable | **85/85** | **85/85** |
| forme | 11/15 | 12/15 |
| riche | 6/10 | 7/10 |
| duree totale | **291,1 s** | 443,2 s |
| latence max | 83,2 s | 64,9 s |

Les **12 couches hors-LLM sont byte-identiques**. Un seul desaccord sur
100 questions (`riche-01`). Verdict : llama achete **+1 point pour +52 % de
temps**. Aucune superiorite de fond, aucune raison de changer.

### 4.2 vs DeepSeek-R1-Distill-Qwen-1.5B (le "plus intelligent" sur le papier)

Meme 7 questions FR (`resultats/comparaison_r1_20261011.txt`) :

| | gemma-1B | R1-Distill-1.5B |
|---|---|---|
| score | **5/7** | 4/7 |
| temps / reponse | **2,7 s** | 72,1 s (**x27**) |
| pic RSS | **1 165,6 Mo** | 1 870,5 Mo (+60 %) |
| qualite FR | correcte | **fuite d'anglais** |

Exemples de reponses R1 :

- « Quelle est la capitale de la France ? » -> `The capital of France is "La France."`
- « capitale de l'Australie » -> `Sydney` (faux)
- « photosynthese » -> `La photosynthese est l'processus permettant a des plantes de convertir le light`

Sur les 10 redactions, R1 fait **6/10** (256,7 s).

Verdict : le modele au meilleur pedigree mathematique (AIME 28,9 / MATH-500 83,9)
est **le moins bon ici** — il n'est pas francophone et il est 27x trop lent.
**On ne l'active pas par defaut.**

## 5. Reponse directe : est-ce que c'est le meilleur ?

- **Meilleur assemblage, oui.** Aura (pipeline complet) gagne sur la
  conversation FR : meme justesse que llama, 1,5x plus rapide ; face a R1,
  +1 point de score, 27x plus rapide, 60 % de RAM en moins.
- **Meilleur modele, non.** Le cerveau est le maillon faible : il ne porte
  que 11 des 100 questions du banc et il plafonne a 0,16-0,20 sur GSM8K.
  Ce qui fait la qualite percue d'Aura, ce sont les couches deterministes.

## 6. Faiblesses mesurees (par ordre de gravite)

1. **Chiffres historiques non comparables — la cause est corrigee dans
   l'outil.** Le meme code a donne 89, 92, 94, 96, 97, 98 ou 100/100 selon le
   cache du moment. Preuve : 89/100 en 116,6 s (`diagnostic_banc_pollue`)
   contre 92/100 en 412,8 s (`baseline_aura_.._isole`), meme code. Un seul
   fichier est franchement pollue : `banc_100.gemma.json` (100/100), dont
   7 des 10 questions riches ont ete servies par `cache-sem` en 0 s.
   **La version courante de `scripts/banc_100.py` corrige la cause** : elle
   cree un cache *vide et unique par run* (`.banc100_cache_<pid>_<ns>.jsonl`),
   neutralise son propre `enregistrer` et isole la memoire personnelle. Un
   nouveau run ne peut donc plus servir de hit. Pour trier un ancien fichier :
   regarder le champ `cerveau` (`cache-sem` = pollue) et la presence de
   `provenance` (absente avant le 10/10).
2. **Raisonnement / arithmetique : 0,12-0,20 GSM8K.** Loin des 62,8 annonces.
   Le CoT n'aide pas a cette taille, il nuit. C'est une limite de capacite,
   pas un bug d'orchestration.
3. **Redaction longue : la seule categorie qui n'est jamais parfaite**
   (2/10 a 10/10 selon les runs, 8/10 au meilleur). Modes d'echec observes :
   - arret trop tot (709 et 527 car au lieu de >900) ;
   - **echo de consigne** (« Voici une reponse a la question posee en gardant
     le ton demande : precis, fluide... ») ;
   - detournement par l'expert web (« Aucune source ne definit directement... »)
     -> **bug de routage, corrige** (extraction-web 0/2 disparait de la
     provenance ; gemma passe de 6/8 a 9/11).
4. **Latence.** En in-process, une redaction coute 15-83 s. Le serveur Vulkan
   (~3x plus rapide) n'a pas ete mesure sur ces questions.
5. **Methode du banc.** Fait maison, en francais, correction heuristique
   (boucle / balise / copie / >900 car). Il **ne peut pas juger la verite**
   d'une dissertation — c'est ecrit noir sur blanc dans `scripts/banc_100.py`.
   Il n'est **pas comparable** a GSM8K/MMLU.
6. **Empreinte.** 1 165,6 Mo de pic RSS (gemma), 1 870,5 Mo (R1). Le seuil
   vise (<2-3 Go sur un vieux PC) est tenu, mais sans marge avec R1.

## 7. Ce qu'on peut defendre aujourd'hui, et rien de plus

- **98/100 sur son propre banc FR, cache neuf** — en precisant que
  85 des 85 points verifiables viennent de couches deterministes et que le
  LLM ne porte que 11 questions (9/11).
- **12/100 en GSM8K flexible via le pipeline** ; le strict (0/100) est
  ininterpretable car Aura n'emet jamais `####`.
- **1,5x plus rapide que llama pour 1 point de moins ; 27x plus rapide que
  R1 pour 1 point de plus.**
- **Le CoT sur 1B est un piege mesure**, pas une opinion.

Tout le reste (100/100, "aussi fort qu'un 8B", GSM8K 62,8) n'est pas soutenu
par les fichiers de ce depot.

## 8. Comment rejouer une mesure comparable

```bash
.venv/Scripts/python.exe scripts/banc_100.py --json resultats/banc_<date>.json
```

Le cache est cree vide et unique a chaque run : deux runs successifs donnent
des chiffres comparables entre eux (ce qui n'etait pas le cas des fichiers
d'octobre). Verifier apres coup que `provenance` est presente et qu'aucun
`cerveau` ne vaut `cache-sem`.
