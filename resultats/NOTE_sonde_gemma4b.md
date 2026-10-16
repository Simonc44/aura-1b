# Sonde « logit lens » sur Gemma-3-4B : ou la reponse se forme, et ou elle disparait

Date : 16 oct. 2026. Modele : `unsloth/gemma-3-4b-it` (bf16, CPU, 34 couches ->
35 positions mesurees), par `scripts/sonde_interp.py`. Mesures brutes :
`resultats/sonde_gemma4b.json`.

## Ce qui a ete mesure

Pour chaque couche, la sonde projette l'etat cache du DERNIER token sur le
vocabulaire (norme finale + `lm_head`), et releve les 3 tokens les plus
probables. 5 prompts, 35 positions chacun. Ce n'est pas un score : c'est une
reponse a « a quel etage le modele decide ».

## Ce que ca montre

| Prompt | L28 | L32 | L33 | L34 (derniere) |
|---|---|---|---|---|
| `capitale de l'Australie ?` | `' Canberra'` **0,9706** (Sydney 0,0293) | `' Canberra'` 0,5186 (Sydney 0,4039) | `' Can'` 0,8356 | `' Cette'` **0,9999** |
| `Qui es-tu ?` | `' I'` 0,9663 *(anglais)* | `' Je'` 0,6845 | `' Je'` 0,9341 | `' C'` **0,9998** |
| `Terre tourne autour du Soleil ?` | `' Yes'` 0,7997 *(anglais)* | `' Oui'` 0,5600 | `' Oui'` 0,9381 | `' É'` **0,9997** |
| `Combien font 17 * 23 ?` | `' '` 1,0 | `' '` 0,9995 | `' '` 0,9990 | `' '` 1,0 |
| `17 * 23 =` (sans espace final) | `' ('` 0,5622 | `' '` 0,8350 | `' '` 0,9465 | `' '` 1,0 |

1. **La reponse est formee AVANT la derniere couche, et avec une forte
   confiance** : « Canberra » sort a **0,9706 des la couche 28** (six couches
   avant la fin), « Oui » a 0,9381 et « Je » a 0,9341 en couche 33. Le fait est
   donc *deja la*, en interne, bien avant la sortie.
2. **La derniere couche ne la livre pas.** Elle est quasi certaine
   (p >= 0,9997) et predit l'**amorce d'une phrase** — `' Cette'`, `' C'`,
   `' É'` — la ou deux couches plus bas se trouvaient la capitale et le « oui ».
   Autrement dit : *savoir n'est pas repondre*, et « le modele sait-il X ? » ne
   se lit PAS sur sa derniere couche, ni sur sa reponse.
3. **Sur l'arithmetique, il n'y a rien a extraire.** Sur le prompt `17 * 23`,
   **aucun chiffre n'entre dans le top 3 d'AUCUNE des 35 positions** : le
   meilleur candidat est `'1'` a **0,0006** (couches 29 a 33), soit ~1 600 fois
   moins que l'espace (0,9995 en couche 32). Il n'y a donc pas de produit « deja
   calcule » que le modele cacherait : le solveur deterministe n'est pas une
   optimisation, c'est la seule voie — ce que dit aussi le GSM8K mesure
   (0,16-0,20, `MODEL_CARD.md`).
4. **La langue de la reponse se decide tard, et elle part de l'anglais** :
   `' I'` (0,9663) et `' Yes'` (0,7997) en couche 28, contre `' Je'` / `' Oui'`
   a partir de la couche 30. A garder en tete pour un projet francophone bati
   sur un modele anglais — sous la reserve du point ci-dessous.
5. **Controle de cablage** : la couche 0 redonne le dernier caractere du prompt
   (le `:` du « R: ») a p = 1,0 — la sonde est donc branchee correctement — et
   les couches 1 a 8 ne veulent **rien dire** (fragments tamil, bengali,
   cyrillique autour de p = 0,5). Le flux residuel n'y est pas encore aligne
   sur le vocabulaire de sortie, et les couches intermediaires restent plates
   (p = 0,0039 en couche 17) : le signal ne devient exploitable que dans les
   dernieres couches.

## Ce que ca ne prouve PAS (lu avant de citer un chiffre)

- **Le gabarit de conversation n'a PAS ete applique** : le prompt part brut,
  alors que le snapshot en livre un (`chat_template.jinja`). Le modele est donc
  sonde hors de son format d'entrainement, et les amorces de phrase de la
  derniere couche peuvent n'etre que ca. La prochaine mesure utile est la meme
  sonde avec `<start_of_turn>user ... <end_of_turn>` : tant qu'elle n'est pas
  faite, le point 2 est une mesure d'interpretabilite, pas une propriete du
  modele en usage.
- **C'est un 4B, pas le cerveau livre** (Gemma-3-1B, 26 couches). Rien ici ne
  dit ce que le 1B sait : ce fichier repond a « un plus gros modele aide-t-il,
  et ou », pas a « faut-il en changer ».
- **Une seule passe, 5 prompts, top 3, bf16, CPU.** Les couches 1 a 28 sur
  l'arithmetique ne sont pas lisibles, donc on ne peut pas affirmer que le
  modele ne calcule *nulle part* : on affirme que rien n'apparait la ou la
  sonde est confiante (L28-L34).
- **Le fichier ne portait pas sa provenance** (ni modele, ni date, ni `top`) :
  le modele a ete retrouve par le nombre de positions (35 = 34 couches +
  l'embedding, la config du cache) et par le nom du fichier. Les mesures sont
  intactes ; seule la provenance a ete ajoutee, et `scripts/sonde_interp.py`
  l'ecrit desormais lui-meme (`_methode`).

## Reproduire

`torch` et `transformers` ne sont pas des dependances du projet (`uv.lock`) :
la sonde tourne dans un environnement qui les a.

    uv run --with torch --with transformers python scripts/sonde_interp.py \
        --modele unsloth/gemma-3-4b-it --json resultats/sonde_gemma4b.json
