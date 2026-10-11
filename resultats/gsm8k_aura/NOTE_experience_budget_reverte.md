# Experience « budget de raisonnement » — REVERTEE

Statut : **correctifs annules**, le systeme est revenu a l'etat d'origine.

## Hypothese testee
Le score `flexible-extract` de 0,12 sur GSM8K venait d'un budget de tokens
trop court : `_max_tokens_adaptatif` renvoyait 120 tokens pour tout enonce,
ce qui tronquait la reponse avant le resultat.

## Correctifs essayes
1. `_max_tokens_adaptatif` : 120 -> 320 tokens pour un enonce a etapes
   (detection `_est_raisonnement` : chiffre + mot de quantite/operateur).
2. `_avec_raisonnement` : consigne de decomposition DANS LA LANGUE de la
   question, appliquee aussi sur le chemin serveur.
3. `routeur._EXEMPLES_MATH` : ajout d'exemples de problemes-mots.

## Mesure (gsm8k_aura_patch.jsonl, run100_patch.log)
| | base (origine) | patch (320 + CoT) |
|---|---|---|
| flexible-extract | **0,12** (100 q) | **0,03** (71 q valides) |
| reponses vides | 0 | **37** |
| latence mediane | 9-18 s | 13,7 s |

## Pourquoi c'est pire
- Le 1B `gemma3-1b` n'utilise pas le budget supplementaire : il derive. A
  320 tokens il produit des reponses formulaiques (« Reponse : X. (C'est le
  calcul simple de ...) ») ou echoie des blocs du prompt systeme.
- 37 reponses vides sur 100.
- Le routage « math » est de toute facon REJETE par `_valider_experts`
  (« math sans donnees numeriques -> ignore »), donc l'ajout d'exemples
  d'entrainement du routeur est neutralise.

## Conclusion
Le plafond de 120 tokens n'etait PAS la cause du 0,12. La limite est la
**capacite du cerveau 1B**. Elargir le budget ne cree pas de raisonnement :
il cree du remplissage.
