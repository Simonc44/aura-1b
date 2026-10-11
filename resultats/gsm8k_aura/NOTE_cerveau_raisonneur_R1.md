# Cerveau raisonneur distille (R1) — evalue, PAS active par defaut

Date : 11 oct. 2026.

## Ce qui a ete fait
- Telecharge `modeles/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (1,12 Go, qwen2).
- Declare dans `llama_cerveau._CHEMINS["R1_DISTILL"]`.
- Budget de tokens DECOUPLE du cerveau (`_cerveau_raisonneur()` + `_est_raisonnement()`).

## Ce qui est actif par defaut
- Le cerveau reste **Gemma** (seul compatible avec les 4 adaptateurs LoRA du disque).
- Le **decouplage du budget est actif** : Gemma garde 120 tokens (inchange),
  un cerveau raisonneur recoit 512 sur un enonce a etapes. Aucun changement
  pour Gemma (mesure : elargir son budget la faisait DERIVER, 0,12 -> 0,03).

## Mesure du R1 (contexte : AURA_GGUF=R1_DISTILL, in-process)
- Enonce des oeufs de GSM8K : **Gemma repond 56 (faux)**, R1 repond **18 (juste)**
  avec une chaine ` thinking` correcte.
- Reponse avec `\boxed{3}` sur le 2e enonce.
- Latence : **155 s** avec le profil `qualite` complet ; **51-66 s** avec
  `AURA_AGENTS=0 AURA_VERIF_WEB=0 AURA_TOKENS_RAISONNEMENT=512`.

## Pourquoi ce n'est PAS le defaut
1. **Adaptateurs orphelins** : les 4 LoRA du disque sont Gemma ; R1 (qwen2) les
   refuse (cf. `tests/test_couplage_architecture.py`).
2. **Lenteur** : 51-66 s par enonce dur, incompatible avec la contrainte
   « rapide / vieux PC ».

## Pour l'activer
    AURA_GGUF=R1_DISTILL AURA_AGENTS=0 AURA_VERIF_WEB=0
(et accepter la perte des adaptateurs LoRA)
