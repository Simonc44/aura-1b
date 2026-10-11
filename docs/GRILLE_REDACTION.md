# Evaluation des redactions RontoGPT

## Pourquoi ce banc existe

Le banc historique classe une redaction comme reussie surtout si elle depasse
900 caracteres et ne contient pas certaines boucles ou consignes recopiees.
Ce controle de forme ne mesure ni la pertinence, ni la justesse, ni la couverture
du sujet. Ce jeu conserve les reponses pour permettre une evaluation humaine
reproductible, question par question.

## Grille humaine

Noter chaque critere de 0 a 2 :

- **Pertinence** : 0 hors sujet, 1 partiellement sur le sujet, 2 repond directement.
- **Couverture** : 0 manque la demande principale, 1 couvre une partie, 2 couvre
  tous les points listes dans `attendus`.
- **Exactitude** : 0 erreurs importantes, 1 quelques affirmations douteuses,
  2 affirmations plausibles et convenablement nuancees.
- **Structure** : 0 incoherente, 1 organisation faible, 2 progression claire.
- **Clarte** : 0 difficile a lire ou repetitif, 1 lisible avec des defauts,
  2 clair, naturel et sans consignes internes recopiees.

Le total va de 0 a 10. L'evaluateur doit noter la reponse sans savoir quelle
version du systeme l'a produite. Pour les comparaisons, utiliser les memes
questions, au moins deux generations par version, le meme PC, le meme modele,
les memes reglages, et publier la latence avec les scores. Garder les notes par
question : une moyenne seule masque les regressions.

Les points `attendus` sont une aide a la couverture, pas une liste de mots a
recopier. Une formulation differente peut satisfaire un point. Toute affirmation
factuelle specifique doit etre verifiee avant de donner le score d'exactitude 2.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\evaluer_redaction.py --sortie resultats\redaction_baseline.jsonl
```

`--repetitions 2` lance deux generations par question. Les reponses sont ecrites
au fur et a mesure; l'historique de chaque generation et le cache du banc sont
isoles de la memoire utilisateur. Les champs de note humaine restent vides tant
qu'une personne n'a pas note les reponses.
